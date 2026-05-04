#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import random
import time

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

import train_equi_df as train_impl


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    def log(self, payload: dict):
        assert self._file is not None
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()


class H5VisionSequenceDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        spans: list[train_impl.EpisodeSpan],
        n_obs_steps: int,
        horizon: int,
        state_center: np.ndarray,
        state_scale: np.ndarray,
        action_center: np.ndarray,
        action_scale: np.ndarray,
        rotation_repr: str,
        image_normalization: str,
        use_ft: bool = False,
        image_crop_size: int | None = None,
    ):
        self.h5_path = h5_path
        self.spans = spans
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.state_center = state_center.astype(np.float32)
        self.state_scale = state_scale.astype(np.float32)
        self.action_center = action_center.astype(np.float32)
        self.action_scale = action_scale.astype(np.float32)
        self.rotation_repr = rotation_repr
        self.image_normalization = image_normalization
        self.use_ft = use_ft
        self.image_crop_size = image_crop_size
        self._file = None
        self.samples: list[tuple[int, int]] = []
        for episode_idx, span in enumerate(self.spans):
            for step in range(span.length):
                self.samples.append((episode_idx, step))

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        self._ensure_open()
        episode_idx, step_idx = self.samples[idx]
        span = self.spans[episode_idx]

        obs_offsets = np.arange(step_idx - self.n_obs_steps + 1, step_idx + 1)
        obs_offsets = np.clip(obs_offsets, 0, span.length - 1)
        obs_rows = span.start + obs_offsets

        action_offsets = np.arange(step_idx, step_idx + self.horizon)
        action_offsets = np.clip(action_offsets, 0, span.length - 1)
        action_rows = span.start + action_offsets

        wrist = np.asarray(train_impl._read_h5_rows(self._file["wrist_rgb"], obs_rows), dtype=np.float32)
        wrist = train_impl._center_crop_numpy(wrist, self.image_crop_size) / 255.0
        if self.image_normalization == "minus_one_one":
            wrist = wrist * 2.0 - 1.0
        elif self.image_normalization == "imagenet":
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            wrist = (wrist - mean) / std
        elif self.image_normalization != "zero_one":
            raise ValueError(f"Unsupported image_normalization: {self.image_normalization}")

        state = train_impl._state_from_h5(self._file, obs_rows, use_ft=self.use_ft)
        state = (state - self.state_center) / self.state_scale

        raw_action = np.asarray(train_impl._read_h5_rows(self._file["action"], action_rows), dtype=np.float32)
        action = train_impl._convert_action_repr_numpy(raw_action, self.rotation_repr)
        action = (action - self.action_center) / self.action_scale

        return {
            "obs": {
                "wrist": torch.from_numpy(wrist.astype(np.float32)),
                "state": torch.from_numpy(state.astype(np.float32)),
            },
            "action": torch.from_numpy(action.astype(np.float32)),
        }


def build_datasets(cfg):
    h5_path = cfg.task.dataset_path
    spans = train_impl._load_episode_spans(h5_path)
    max_train = int(cfg.task.dataset.max_train_episodes or 0)
    max_val = int(cfg.task.dataset.max_val_episodes or 0)
    train_spans, val_spans = train_impl._split_episodes(
        spans,
        float(cfg.task.dataset.val_ratio),
        int(cfg.training.seed),
        max_train,
        max_val,
    )
    if not train_spans:
        raise ValueError("Training split is empty.")

    stats = train_impl._compute_stats(
        h5_path,
        train_spans,
        cfg.policy.rotation_repr,
        cfg.task.dataset.state_action_normalization,
        bool(cfg.task.dataset.ft),
    )
    state_center = np.asarray(stats.state_center, dtype=np.float32)
    state_scale = np.asarray(stats.state_scale, dtype=np.float32)
    action_center = np.asarray(stats.action_center, dtype=np.float32)
    action_scale = np.asarray(stats.action_scale, dtype=np.float32)

    common_kwargs = dict(
        h5_path=h5_path,
        n_obs_steps=int(cfg.n_obs_steps),
        horizon=int(cfg.horizon),
        state_center=state_center,
        state_scale=state_scale,
        action_center=action_center,
        action_scale=action_scale,
        rotation_repr=cfg.policy.rotation_repr,
        image_normalization=cfg.task.dataset.image_normalization,
        use_ft=bool(cfg.task.dataset.ft),
        image_crop_size=cfg.task.dataset.image_crop_size,
    )
    train_dataset = H5VisionSequenceDataset(spans=train_spans, **common_kwargs)
    val_dataset = H5VisionSequenceDataset(spans=val_spans, **common_kwargs)
    return train_dataset, val_dataset, stats, train_spans, val_spans


class TrainDiffusionUnetImageWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        self.cfg = cfg
        self.output_dir = str(output_dir or cfg.output_dir)
        self.global_step = 0
        self.epoch = 0

        seed = int(cfg.training.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_model(self, state_dim: int, action_dim: int):
        return train_impl.EquiDiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            n_obs_steps=int(self.cfg.n_obs_steps),
            image_embed_dim=int(self.cfg.policy.image_embed_dim),
            obs_feature_dim=int(self.cfg.policy.obs_feature_dim),
            vision_encoder=self.cfg.policy.vision_encoder,
            diffusion_step_embed_dim=int(self.cfg.policy.diffusion_step_embed_dim),
            down_dims=[int(x) for x in self.cfg.policy.down_dims],
            kernel_size=int(self.cfg.policy.kernel_size),
            n_groups=int(self.cfg.policy.n_groups),
            horizon=int(self.cfg.horizon),
            n_action_steps=int(self.cfg.n_action_steps),
            num_inference_steps=int(self.cfg.policy.num_inference_steps),
            cond_predict_scale=bool(self.cfg.policy.cond_predict_scale),
            rotation_repr=self.cfg.policy.rotation_repr,
        )

    def _save_checkpoint(self, path: Path, model, ema, optimizer, lr_scheduler, history, best_val, stats):
        payload = {
            "model": model.state_dict(),
            "ema_model": ema.shadow.state_dict() if ema is not None else None,
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "history": history,
            "epoch": self.epoch,
            "best_val": best_val,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "stats": asdict(stats),
            "global_step": self.global_step,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def _load_checkpoint(self, path: Path, model, ema, optimizer, lr_scheduler):
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if ema is not None and checkpoint.get("ema_model") is not None:
            ema.shadow.load_state_dict(checkpoint["ema_model"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.optimization_step = checkpoint["ema_state"].get("optimization_step", ema.optimization_step)
        self.epoch = int(checkpoint.get("epoch", 0)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        return checkpoint.get("history", []), float(checkpoint.get("best_val", float("inf")))

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_dataset, val_dataset, stats, train_spans, val_spans = build_datasets(cfg)
        train_loader = DataLoader(train_dataset, **OmegaConf.to_container(cfg.dataloader, resolve=True))
        val_loader = DataLoader(val_dataset, **OmegaConf.to_container(cfg.val_dataloader, resolve=True))

        state_dim = len(stats.state_center)
        action_dim = len(stats.action_center)
        device = torch.device(cfg.training.device)

        model = self._build_model(state_dim, action_dim).to(device)
        ema_model = self._build_model(state_dim, action_dim).to(device) if cfg.training.use_ema else None
        train_impl._initialize_policy_modules(model, train_dataset, action_dim, int(cfg.horizon), device)
        if ema_model is not None:
            train_impl._initialize_policy_modules(ema_model, train_dataset, action_dim, int(cfg.horizon), device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optimizer.lr),
            betas=tuple(cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
        diffusion_scheduler = train_impl._build_diffusion_scheduler(cfg.policy)
        batches_per_epoch = len(train_loader) if cfg.training.max_train_steps is None else min(len(train_loader), int(cfg.training.max_train_steps))
        optimizer_steps_per_epoch = max(math.ceil(batches_per_epoch / max(int(cfg.training.gradient_accumulate_every), 1)), 1)
        total_train_steps = max(optimizer_steps_per_epoch * int(cfg.training.num_epochs), 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=train_impl._make_lr_lambda(total_train_steps, int(cfg.training.lr_warmup_steps), cfg.training.lr_scheduler),
        )
        ema = (
            train_impl.ExponentialMovingAverage(
                model,
                ema_model,
                update_after_step=int(cfg.ema.update_after_step),
                inv_gamma=float(cfg.ema.inv_gamma),
                power=float(cfg.ema.power),
                min_value=float(cfg.ema.min_value),
                max_value=float(cfg.ema.max_value),
            )
            if cfg.training.use_ema
            else None
        )

        action_center = np.asarray(stats.action_center, dtype=np.float32)
        action_scale = np.asarray(stats.action_scale, dtype=np.float32)
        model.set_normalizer(action_center, action_scale)
        if ema_model is not None:
            ema_model.set_normalizer(action_center, action_scale)

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_dict.update(
            {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "num_train_episodes": len(train_spans),
                "num_val_episodes": len(val_spans),
                "resolved_total_train_steps": total_train_steps,
            }
        )
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
        with (output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(stats), f, indent=2)

        history = []
        best_val = float("inf")
        latest_path = checkpoint_dir / "latest.pt"
        if cfg.training.resume and latest_path.exists():
            history, best_val = self._load_checkpoint(latest_path, model, ema, optimizer, lr_scheduler)

        if wandb is not None and cfg.logging.mode != "disabled":
            wandb_run = wandb.init(
                dir=str(output_dir),
                config=config_dict,
                project=cfg.logging.project,
                mode=cfg.logging.mode,
                name=cfg.logging.name,
                tags=list(cfg.logging.tags),
            )
        else:
            wandb_run = None

        train_sampling_batch = None
        run_start = time.time()
        log_path = output_dir / "logs.jsonl"
        with JsonlLogger(log_path) as json_logger:
            for epoch in range(self.epoch, int(cfg.training.num_epochs)):
                self.epoch = epoch
                model.train()
                epoch_start = time.time()
                train_losses = []
                optimizer.zero_grad(set_to_none=True)
                effective_train_steps = len(train_loader) if cfg.training.max_train_steps is None else min(len(train_loader), int(cfg.training.max_train_steps))
                for batch_idx, batch in enumerate(train_loader, start=1):
                    wrist = batch["obs"]["wrist"].to(device, non_blocking=True)
                    state = batch["obs"]["state"].to(device, non_blocking=True)
                    action = batch["action"].to(device, non_blocking=True)
                    if train_sampling_batch is None:
                        train_sampling_batch = {
                            "obs": {"wrist": wrist.detach().clone(), "state": state.detach().clone()},
                            "action": action.detach().clone(),
                        }

                    raw_loss = train_impl._compute_diffusion_loss(model, diffusion_scheduler, wrist, state, action)
                    loss = raw_loss / max(int(cfg.training.gradient_accumulate_every), 1)
                    loss.backward()

                    is_accum_boundary = (
                        batch_idx % max(int(cfg.training.gradient_accumulate_every), 1) == 0
                        or batch_idx == effective_train_steps
                    )
                    if is_accum_boundary:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()
                        if ema is not None:
                            ema.update(model)
                        self.global_step += 1

                    train_losses.append(raw_loss.item())
                    if cfg.training.max_train_steps is not None and batch_idx >= int(cfg.training.max_train_steps):
                        break

                train_loss = float(np.mean(train_losses))
                eval_model = ema.shadow if ema is not None else model
                val_loss = train_loss
                if epoch % int(cfg.training.val_every) == 0 and len(val_dataset) > 0:
                    val_loss = train_impl.evaluate(
                        eval_model,
                        val_loader,
                        diffusion_scheduler,
                        device,
                        max_val_steps=cfg.training.max_val_steps,
                    )

                step_log = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "global_step": self.global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_time": time.time() - epoch_start,
                    "run_elapsed": time.time() - run_start,
                }
                if epoch % int(cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                    eval_model.eval()
                    with torch.inference_mode():
                        result = eval_model.predict_action(train_sampling_batch["obs"], diffusion_scheduler)
                        target_action = eval_model._repr_to_raw_action(
                            eval_model._denormalize_action(train_sampling_batch["action"])
                        )
                        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
                            result["action_pred"], target_action
                        ).item()

                history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **({"train_action_mse_error": step_log["train_action_mse_error"]} if "train_action_mse_error" in step_log else {})})
                with (output_dir / "history.json").open("w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)

                self._save_checkpoint(latest_path, model, ema, optimizer, lr_scheduler, history, best_val, stats)
                epoch_ckpt = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
                if epoch % int(cfg.training.checkpoint_every) == 0:
                    self._save_checkpoint(epoch_ckpt, model, ema, optimizer, lr_scheduler, history, best_val, stats)
                if val_loss <= best_val:
                    best_val = val_loss
                    self._save_checkpoint(checkpoint_dir / "best.pt", model, ema, optimizer, lr_scheduler, history, best_val, stats)

                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                print(
                    f"[INFO] Epoch {epoch + 1:03d}: train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"epoch_time={train_impl._format_duration(step_log['epoch_time'])} "
                    f"run_elapsed={train_impl._format_duration(step_log['run_elapsed'])}"
                )

        if wandb_run is not None:
            wandb_run.finish()
