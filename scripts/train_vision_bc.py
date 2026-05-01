#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline behavior cloning trainer for wrist RGB + proprio -> teacher action."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train an offline vision behavior cloning policy from HDF5 rollouts.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset from scripts/rsl_rl/vision_log.py.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and config.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--val_split", type=float, default=0.1, help="Episode-level validation split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Optional cap on training samples. 0 means all.")
    parser.add_argument("--max_val_samples", type=int, default=0, help="Optional cap on validation samples. 0 means all.")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "r3m_resnet18"],
        help="Visual encoder backend.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze the visual encoder and train only the projection/head.",
    )
    parser.add_argument("--image_embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--log_interval", type=int, default=100, help="Print training progress every N batches. 0 disables.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class DatasetStats:
    proprio_mean: list[float]
    proprio_std: list[float]
    action_mean: list[float]
    action_std: list[float]


class H5VisionActionDataset(Dataset):
    """Lazy HDF5 dataset backed by row indices."""

    def __init__(self, h5_path: str, indices: np.ndarray, proprio_mean: np.ndarray, proprio_std: np.ndarray):
        self.h5_path = h5_path
        self.indices = indices.astype(np.int64)
        self.proprio_mean = proprio_mean.astype(np.float32)
        self.proprio_std = proprio_std.astype(np.float32)
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        self._ensure_open()
        row = int(self.indices[idx])
        wrist_rgb = self._file["wrist_rgb"][row].astype(np.float32) / 255.0
        joint_pos = self._file["joint_pos"][row].astype(np.float32)
        gripper_pos = self._file["gripper_pos"][row].astype(np.float32)
        prev_action = self._file["prev_action"][row].astype(np.float32)
        action = self._file["action"][row].astype(np.float32)

        proprio = np.concatenate((joint_pos, gripper_pos, prev_action), axis=0)
        proprio = (proprio - self.proprio_mean) / self.proprio_std

        return {
            "image": torch.from_numpy(wrist_rgb),
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
        }


class VisionBCPolicy(nn.Module):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        image_embed_dim: int,
        hidden_dims: list[int],
        encoder_type: str = "custom_cnn",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.freeze_encoder = freeze_encoder

        if encoder_type == "custom_cnn":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            encoder_output_dim = 64
        elif encoder_type == "r3m_resnet18":
            try:
                from r3m import load_r3m
            except ImportError as exc:
                raise ImportError(
                    "install r3m packages inside the current environment"
                    "Install in current environment before using --encoder_type r3m_resnet18."
                ) from exc
            self.encoder = load_r3m("resnet18")
            encoder_output_dim = 512
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.image_proj = nn.Sequential(
            nn.Linear(encoder_output_dim, image_embed_dim),
            nn.ReLU(),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        layers: list[nn.Module] = []
        mlp_input_dim = image_embed_dim + proprio_dim
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(prev_dim, hidden_dim), nn.ELU()))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_head = nn.Sequential(*layers)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(0, 3, 1, 2).contiguous()

        if self.encoder_type == "custom_cnn":
            features = self.encoder(image)
        elif self.encoder_type == "r3m_resnet18":
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
            mean = torch.tensor(self.IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
            std = torch.tensor(self.IMAGENET_STD, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
            image = (image - mean) / std
            if self.freeze_encoder:
                with torch.no_grad():
                    features = self.encoder(image)
            else:
                features = self.encoder(image)
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        return self.image_proj(features)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        image_embedding = self.encode_image(image)
        return self.policy_head(torch.cat((image_embedding, proprio), dim=-1))


def _episode_keys(h5_file: h5py.File) -> np.ndarray:
    env_id = np.asarray(h5_file["env_id"], dtype=np.int64)
    if "episode_id" in h5_file:
        episode_id = np.asarray(h5_file["episode_id"], dtype=np.int64)
        return np.stack((env_id, episode_id), axis=1)
    return env_id[:, None]


def build_split_indices(
    h5_path: str,
    val_split: float,
    seed: int,
    max_train_samples: int,
    max_val_samples: int,
):
    print(f"[INFO] Opening dataset: {h5_path}")
    with h5py.File(h5_path, "r") as h5_file:
        required = ["wrist_rgb", "joint_pos", "gripper_pos", "prev_action", "action"]
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise KeyError(f"Dataset is missing required keys: {missing}")
        print(f"[INFO] Found keys: {sorted(h5_file.keys())}")
        print(f"[INFO] Total samples in H5: {len(h5_file['action'])}")

        episode_keys = _episode_keys(h5_file)
        unique_keys, inverse = np.unique(episode_keys, axis=0, return_inverse=True)
        episode_ids = np.arange(len(unique_keys))
        rng = np.random.default_rng(seed)
        rng.shuffle(episode_ids)
        print(f"[INFO] Unique episode keys: {len(unique_keys)}")

        num_val_episodes = max(1, int(round(len(episode_ids) * val_split)))
        val_episode_ids = set(episode_ids[:num_val_episodes].tolist())

        train_indices = np.flatnonzero(~np.isin(inverse, list(val_episode_ids)))
        val_indices = np.flatnonzero(np.isin(inverse, list(val_episode_ids)))

        if max_train_samples > 0 and len(train_indices) > max_train_samples:
            train_indices = rng.choice(train_indices, size=max_train_samples, replace=False)
        if max_val_samples > 0 and len(val_indices) > max_val_samples:
            val_indices = rng.choice(val_indices, size=max_val_samples, replace=False)
        # h5py fancy indexing requires strictly increasing indices.
        train_indices = np.sort(train_indices)
        val_indices = np.sort(val_indices)
        print(f"[INFO] Train samples after cap: {len(train_indices)}")
        print(f"[INFO] Val samples after cap: {len(val_indices)}")
        print("[INFO] Computing dataset normalization statistics from train split...")

        joint_pos = np.asarray(h5_file["joint_pos"][train_indices], dtype=np.float32)
        gripper_pos = np.asarray(h5_file["gripper_pos"][train_indices], dtype=np.float32)
        prev_action = np.asarray(h5_file["prev_action"][train_indices], dtype=np.float32)
        action = np.asarray(h5_file["action"][train_indices], dtype=np.float32)

        proprio = np.concatenate((joint_pos, gripper_pos, prev_action), axis=1)
        proprio_mean = proprio.mean(axis=0)
        proprio_std = proprio.std(axis=0)
        proprio_std = np.maximum(proprio_std, 1.0e-6)

        action_mean = action.mean(axis=0)
        action_std = action.std(axis=0)
        action_std = np.maximum(action_std, 1.0e-6)

        stats = DatasetStats(
            proprio_mean=proprio_mean.tolist(),
            proprio_std=proprio_std.tolist(),
            action_mean=action_mean.tolist(),
            action_std=action_std.tolist(),
        )

        sample_proprio_dim = proprio.shape[1]
        sample_action_dim = action.shape[1]

    return train_indices, val_indices, stats, sample_proprio_dim, sample_action_dim


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            proprio = batch["proprio"].to(device)
            action = batch["action"].to(device)
            pred = model(image, proprio)
            loss = loss_fn(pred, action)
            total_loss += loss.item() * image.shape[0]
            total_count += image.shape[0]
    return total_loss / max(total_count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    run_start_time = time.time()
    print(f"[INFO] Starting offline BC training with seed={args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    train_indices, val_indices, stats, proprio_dim, action_dim = build_split_indices(
        args.dataset,
        args.val_split,
        args.seed,
        args.max_train_samples,
        args.max_val_samples,
    )
    print(f"[INFO] Proprio dim: {proprio_dim}")
    print(f"[INFO] Action dim: {action_dim}")

    train_dataset = H5VisionActionDataset(
        args.dataset,
        train_indices,
        np.asarray(stats.proprio_mean, dtype=np.float32),
        np.asarray(stats.proprio_std, dtype=np.float32),
    )
    val_dataset = H5VisionActionDataset(
        args.dataset,
        val_indices,
        np.asarray(stats.proprio_mean, dtype=np.float32),
        np.asarray(stats.proprio_std, dtype=np.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[INFO] Train batches per epoch: {len(train_loader)}")
    print(f"[INFO] Val batches per epoch: {len(val_loader)}")

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    model = VisionBCPolicy(
        proprio_dim,
        action_dim,
        args.image_embed_dim,
        args.hidden_dims,
        encoder_type=args.encoder_type,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    num_trainable_params = sum(param.numel() for param in trainable_params)
    print(
        f"[INFO] Model and optimizer initialized. "
        f"encoder_type={args.encoder_type}, freeze_encoder={args.freeze_encoder}, "
        f"trainable_params={num_trainable_params}"
    )

    best_val_loss = float("inf")
    history = []

    with open(output_dir / "config.json", "w", encoding="ascii") as f:
        json.dump(vars(args), f, indent=2)
    with open(output_dir / "stats.json", "w", encoding="ascii") as f:
        json.dump(asdict(stats), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start_time = time.time()
        print(f"[INFO] Epoch {epoch}/{args.epochs} started.")

        for batch_idx, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device)
            proprio = batch["proprio"].to(device)
            action = batch["action"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(image, proprio)
            loss = loss_fn(pred, action)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image.shape[0]
            seen += image.shape[0]

            if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)):
                epoch_elapsed = time.time() - epoch_start_time
                avg_batch_time = epoch_elapsed / batch_idx
                remaining_batches = len(train_loader) - batch_idx
                epoch_eta = avg_batch_time * remaining_batches
                epoch_pct = 100.0 * batch_idx / max(len(train_loader), 1)
                print(
                    f"[INFO] Epoch {epoch}/{args.epochs} | "
                    f"batch {batch_idx}/{len(train_loader)} ({epoch_pct:.1f}%) | "
                    f"train_loss={loss.item():.6f} | "
                    f"elapsed={_format_duration(epoch_elapsed)} | "
                    f"eta={_format_duration(epoch_eta)}"
                )

        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - run_start_time
        avg_epoch_time = total_elapsed / epoch
        total_eta = avg_epoch_time * (args.epochs - epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"epoch_time={_format_duration(epoch_elapsed)} "
            f"total_elapsed={_format_duration(total_elapsed)} "
            f"eta={_format_duration(total_eta)}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(args),
            "stats": asdict(stats),
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

        with open(output_dir / "history.json", "w", encoding="ascii") as f:
            json.dump(history, f, indent=2)

    print(f"[INFO] Finished training. Best val loss: {best_val_loss:.6f}")
    print(f"[INFO] Saved checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
