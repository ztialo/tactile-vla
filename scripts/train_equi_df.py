#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline EquiDF-style visuomotor policy training from HDF5 demonstrations."""

from __future__ import annotations

import argparse
import functools
import json
import math
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
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models

try:
    from escnn import nn as escnn_nn
except ImportError:
    escnn_nn = None

DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs").joinpath("train_equi_df.yaml")


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return data


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config for EquiDF training.",
    )
    bootstrap_args, remaining_argv = bootstrap.parse_known_args()
    config = _load_yaml_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Train an EquiDF-style visuomotor diffusion policy from HDF5 demos.")
    parser.add_argument(
        "--config",
        type=str,
        default=bootstrap_args.config,
        help="Path to YAML config for EquiDF training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.get("dataset"),
        help="Path to HDF5 dataset from scripts/rsl_rl/vision_log.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.get("output_dir"),
        help="Directory to save checkpoints and config.",
    )
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 100))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=config.get("lr", 1.0e-4))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 1.0e-5))
    parser.add_argument(
        "--state_action_normalization",
        type=str,
        default=config.get("state_action_normalization", "zscore"),
        choices=["zscore", "limits"],
        help="Normalization for low-dimensional state and action.",
    )
    parser.add_argument(
        "--adam_betas",
        type=float,
        nargs=2,
        default=config.get("adam_betas", [0.95, 0.999]),
        help="AdamW betas.",
    )
    parser.add_argument("--adam_eps", type=float, default=config.get("adam_eps", 1.0e-8))
    parser.add_argument(
        "--val_split",
        type=float,
        default=config.get("val_split", 0.1),
        help="Episode-level validation split.",
    )
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--num_workers", type=int, default=config.get("num_workers", 4))
    parser.add_argument("--device", type=str, default=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument(
        "--max_train_episodes",
        type=int,
        default=config.get("max_train_episodes", 0),
        help="Optional cap on training episodes. 0 means all.",
    )
    parser.add_argument(
        "--max_val_episodes",
        type=int,
        default=config.get("max_val_episodes", 0),
        help="Optional cap on validation episodes. 0 means all.",
    )
    parser.add_argument("--n_obs_steps", type=int, default=config.get("n_obs_steps", 2))
    parser.add_argument("--horizon", type=int, default=config.get("horizon", 16))
    parser.add_argument("--n_action_steps", type=int, default=config.get("n_action_steps", 8))
    parser.add_argument("--num_inference_steps", type=int, default=config.get("num_inference_steps", 100))
    parser.add_argument(
        "--rotation_repr",
        type=str,
        default=config.get("rotation_repr", "rot6d"),
        choices=["rotvec", "rot6d", "rot9d"],
        help="Training representation for the 3D rotational action delta.",
    )
    parser.add_argument(
        "--vision_encoder",
        type=str,
        default=config.get("vision_encoder", "cnn"),
        choices=["cnn", "escnn_cyclic", "resnet18_gn"],
        help="Vision encoder backend. escnn_cyclic requires the escnn package.",
    )
    parser.add_argument(
        "--image_normalization",
        type=str,
        default=config.get("image_normalization", "minus_one_one"),
        choices=["minus_one_one", "imagenet", "zero_one"],
        help="Image normalization applied before the encoder.",
    )
    parser.add_argument("--image_embed_dim", type=int, default=config.get("image_embed_dim", 128))
    parser.add_argument(
        "--obs_feature_dim",
        type=int,
        default=config.get("obs_feature_dim", 128),
        help="Per-observation fused feature width.",
    )
    parser.add_argument(
        "--diffusion_step_embed_dim",
        type=int,
        default=config.get("diffusion_step_embed_dim", 128),
    )
    parser.add_argument("--down_dims", type=int, nargs="+", default=config.get("down_dims", [256, 512, 1024]))
    parser.add_argument("--kernel_size", type=int, default=config.get("kernel_size", 5))
    parser.add_argument("--n_groups", type=int, default=config.get("n_groups", 8))
    parser.add_argument("--cond_predict_scale", type=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, default=config.get("cond_predict_scale", True))
    parser.add_argument("--num_diffusion_steps", type=int, default=config.get("num_diffusion_steps", 100))
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default=config.get("scheduler_type", "ddim"),
        choices=["ddim", "ddpm"],
        help="Diffusion scheduler type.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=config.get("prediction_type", "sample"),
        choices=["sample", "epsilon"],
        help="Diffusion training target.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=config.get("lr_scheduler", "cosine"),
        choices=["constant", "cosine"],
        help="Learning-rate schedule.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=config.get("lr_warmup_steps", 500))
    parser.add_argument("--gradient_accumulate_every", type=int, default=config.get("gradient_accumulate_every", 1))
    parser.add_argument("--max_train_steps", type=int, default=config.get("max_train_steps"))
    parser.add_argument("--max_val_steps", type=int, default=config.get("max_val_steps"))
    parser.add_argument("--resume", type=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, default=config.get("resume", False))
    parser.add_argument("--use_ema", type=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, default=config.get("use_ema", True))
    parser.add_argument("--ema_update_after_step", type=int, default=config.get("ema_update_after_step", 0))
    parser.add_argument("--ema_inv_gamma", type=float, default=config.get("ema_inv_gamma", 1.0))
    parser.add_argument("--ema_power", type=float, default=config.get("ema_power", 0.75))
    parser.add_argument("--ema_min_value", type=float, default=config.get("ema_min_value", 0.0))
    parser.add_argument("--ema_max_value", type=float, default=config.get("ema_max_value", 0.9999))
    parser.add_argument("--sample_every", type=int, default=config.get("sample_every", 10000))
    parser.add_argument(
        "--log_interval",
        type=int,
        default=config.get("log_interval", 50),
        help="Print progress every N train batches. 0 disables.",
    )
    args = parser.parse_args(remaining_argv)
    if not args.dataset:
        parser.error("dataset must be provided in the YAML config or via --dataset")
    if not args.output_dir:
        parser.error("output_dir must be provided in the YAML config or via --output_dir")
    return args


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


def _estimate_eta(elapsed_seconds: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= 0 or completed > total:
        return "--:--:--"
    rate = elapsed_seconds / completed
    remaining = max(total - completed, 0) * rate
    return _format_duration(remaining)


@dataclass
class EpisodeSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class DatasetStats:
    state_center: list[float]
    state_scale: list[float]
    action_center: list[float]
    action_scale: list[float]


class ExponentialMovingAverage:
    def __init__(
        self,
        model: nn.Module,
        shadow_model: nn.Module,
        update_after_step: int,
        inv_gamma: float,
        power: float,
        min_value: float,
        max_value: float,
    ):
        if update_after_step < 0:
            raise ValueError(f"update_after_step must be >= 0: {update_after_step}")
        if inv_gamma <= 0.0:
            raise ValueError(f"inv_gamma must be > 0: {inv_gamma}")
        if power < 0.0:
            raise ValueError(f"power must be >= 0: {power}")
        if not 0.0 <= min_value <= max_value < 1.0:
            raise ValueError(f"EMA bounds must satisfy 0 <= min <= max < 1: {(min_value, max_value)}")
        self.shadow = shadow_model.eval()
        self.shadow.load_state_dict(model.state_dict())
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.optimization_step = 0
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def get_decay(self) -> float:
        step = max(0, self.optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return min(max(value, self.min_value), self.max_value)

    def update(self, model: nn.Module):
        self.optimization_step += 1
        decay = self.get_decay()
        with torch.no_grad():
            shadow_params = dict(self.shadow.named_parameters())
            model_params = dict(model.named_parameters())
            for name, shadow_param in shadow_params.items():
                shadow_param.lerp_(model_params[name].detach(), 1.0 - decay)

            shadow_buffers = dict(self.shadow.named_buffers())
            model_buffers = dict(model.named_buffers())
            for name, shadow_buffer in shadow_buffers.items():
                if name in model_buffers:
                    shadow_buffer.copy_(model_buffers[name].detach())

    def state_dict(self) -> dict:
        return {
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "optimization_step": self.optimization_step,
            "current_decay": self.get_decay(),
            "shadow": self.shadow.state_dict(),
        }


def _rotation_vector_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors to rotation matrices."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / torch.clamp(angle, min=1.0e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(*rotvec.shape[:-1], 3, 3)
    ident = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand_as(k)
    sin_term = torch.sin(angle)[..., None] * k
    cos_term = (1.0 - torch.cos(angle))[..., None] * torch.matmul(k, k)
    rot = ident + sin_term + cos_term
    small_angle = angle.squeeze(-1) < 1.0e-8
    if torch.any(small_angle):
        rot = torch.where(small_angle[..., None, None], ident, rot)
    return rot


def _matrix_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    return rotmat[..., :, :2].reshape(*rotmat.shape[:-2], 6)


def _convert_action_repr_torch(action: torch.Tensor, rotation_repr: str) -> torch.Tensor:
    dpos = action[..., :3]
    drot = action[..., 3:]
    if rotation_repr == "rotvec":
        return action
    rotmat = _rotation_vector_to_matrix(drot)
    if rotation_repr == "rot6d":
        rot_repr = _matrix_to_rot6d(rotmat)
    elif rotation_repr == "rot9d":
        rot_repr = rotmat.reshape(*rotmat.shape[:-2], 9)
    else:
        raise ValueError(f"Unsupported rotation_repr: {rotation_repr}")
    return torch.cat((dpos, rot_repr), dim=-1)


def _convert_action_repr_numpy(action: np.ndarray, rotation_repr: str) -> np.ndarray:
    action_tensor = torch.from_numpy(action.astype(np.float32))
    return _convert_action_repr_torch(action_tensor, rotation_repr).cpu().numpy()


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    start = int(rows.min())
    end = int(rows.max()) + 1
    block = np.asarray(dataset[start:end])
    return block[rows - start]


def _state_from_h5(h5_file: h5py.File, rows: np.ndarray) -> np.ndarray:
    eef_pos = np.asarray(_read_h5_rows(h5_file["eef_pos"], rows), dtype=np.float32)
    eef_quat = np.asarray(_read_h5_rows(h5_file["eef_quat"], rows), dtype=np.float32)
    gripper_pos = np.asarray(_read_h5_rows(h5_file["gripper_pos"], rows), dtype=np.float32)
    return np.concatenate((eef_pos, eef_quat, gripper_pos), axis=-1)


def _load_episode_spans(h5_path: str) -> list[EpisodeSpan]:
    with h5py.File(h5_path, "r") as h5_file:
        if "done" not in h5_file:
            raise KeyError("Dataset is missing required key: done")
        done = np.asarray(h5_file["done"], dtype=np.bool_)
        spans: list[EpisodeSpan] = []
        start = 0
        for idx, is_done in enumerate(done):
            if is_done:
                spans.append(EpisodeSpan(start=start, end=idx + 1))
                start = idx + 1
        if start < len(done):
            spans.append(EpisodeSpan(start=start, end=len(done)))
    if not spans:
        raise ValueError("Dataset contains no complete episodes.")
    return spans


def _split_episodes(
    spans: list[EpisodeSpan],
    val_split: float,
    seed: int,
    max_train_episodes: int,
    max_val_episodes: int,
) -> tuple[list[EpisodeSpan], list[EpisodeSpan]]:
    episode_ids = np.arange(len(spans))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    if len(episode_ids) == 1:
        train_ids = episode_ids
        val_ids = np.array([], dtype=np.int64)
    else:
        num_val = max(1, int(round(len(episode_ids) * val_split)))
        num_val = min(num_val, len(episode_ids) - 1)
        val_ids = episode_ids[:num_val]
        train_ids = episode_ids[num_val:]

    if max_train_episodes > 0:
        train_ids = train_ids[:max_train_episodes]
    if max_val_episodes > 0:
        val_ids = val_ids[:max_val_episodes]

    train_spans = [spans[int(idx)] for idx in train_ids]
    val_spans = [spans[int(idx)] for idx in val_ids]
    return train_spans, val_spans


def _compute_stats(
    h5_path: str,
    spans: list[EpisodeSpan],
    rotation_repr: str,
    normalization_mode: str,
) -> DatasetStats:
    state_sum = None
    state_sq_sum = None
    action_sum = None
    action_sq_sum = None
    state_min = None
    state_max = None
    action_min = None
    action_max = None
    num_state_rows = 0
    num_action_rows = 0

    with h5py.File(h5_path, "r") as h5_file:
        for span in spans:
            rows = np.arange(span.start, span.end, dtype=np.int64)
            state = _state_from_h5(h5_file, rows)
            action = np.asarray(h5_file["action"][rows], dtype=np.float32)
            action_repr = _convert_action_repr_numpy(action, rotation_repr)

            if state_sum is None:
                state_sum = state.sum(axis=0, dtype=np.float64)
                state_sq_sum = np.square(state, dtype=np.float64).sum(axis=0)
                action_sum = action_repr.sum(axis=0, dtype=np.float64)
                action_sq_sum = np.square(action_repr, dtype=np.float64).sum(axis=0)
                state_min = state.min(axis=0)
                state_max = state.max(axis=0)
                action_min = action_repr.min(axis=0)
                action_max = action_repr.max(axis=0)
            else:
                state_sum += state.sum(axis=0, dtype=np.float64)
                state_sq_sum += np.square(state, dtype=np.float64).sum(axis=0)
                action_sum += action_repr.sum(axis=0, dtype=np.float64)
                action_sq_sum += np.square(action_repr, dtype=np.float64).sum(axis=0)
                state_min = np.minimum(state_min, state.min(axis=0))
                state_max = np.maximum(state_max, state.max(axis=0))
                action_min = np.minimum(action_min, action_repr.min(axis=0))
                action_max = np.maximum(action_max, action_repr.max(axis=0))

            num_state_rows += state.shape[0]
            num_action_rows += action_repr.shape[0]

    if num_state_rows == 0 or num_action_rows == 0:
        raise ValueError("Cannot compute normalization stats from an empty split.")

    if normalization_mode == "zscore":
        state_center = state_sum / num_state_rows
        state_var = state_sq_sum / num_state_rows - np.square(state_center)
        state_scale = np.sqrt(np.maximum(state_var, 1.0e-6))

        action_center = action_sum / num_action_rows
        action_var = action_sq_sum / num_action_rows - np.square(action_center)
        action_scale = np.sqrt(np.maximum(action_var, 1.0e-6))
    elif normalization_mode == "limits":
        state_center = 0.5 * (state_max + state_min)
        state_scale = np.maximum(0.5 * (state_max - state_min), 1.0e-6)
        action_center = 0.5 * (action_max + action_min)
        action_scale = np.maximum(0.5 * (action_max - action_min), 1.0e-6)
    else:
        raise ValueError(f"Unsupported state_action_normalization: {normalization_mode}")

    return DatasetStats(
        state_center=np.asarray(state_center, dtype=np.float32).tolist(),
        state_scale=np.asarray(state_scale, dtype=np.float32).tolist(),
        action_center=np.asarray(action_center, dtype=np.float32).tolist(),
        action_scale=np.asarray(action_scale, dtype=np.float32).tolist(),
    )


class H5SequenceDataset(Dataset):
    """Sequence dataset that yields padded observation windows and future action horizons."""

    def __init__(
        self,
        h5_path: str,
        spans: list[EpisodeSpan],
        n_obs_steps: int,
        horizon: int,
        state_center: np.ndarray,
        state_scale: np.ndarray,
        action_center: np.ndarray,
        action_scale: np.ndarray,
        rotation_repr: str,
        image_normalization: str,
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

        wrist = np.asarray(_read_h5_rows(self._file["wrist_rgb"], obs_rows), dtype=np.float32) / 255.0
        if self.image_normalization == "minus_one_one":
            wrist = wrist * 2.0 - 1.0
        elif self.image_normalization == "imagenet":
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            wrist = (wrist - mean) / std
        elif self.image_normalization == "zero_one":
            pass
        else:
            raise ValueError(f"Unsupported image_normalization: {self.image_normalization}")
        state = _state_from_h5(self._file, obs_rows)
        state = (state - self.state_center) / self.state_scale

        raw_action = np.asarray(_read_h5_rows(self._file["action"], action_rows), dtype=np.float32)
        action = _convert_action_repr_numpy(raw_action, self.rotation_repr)
        action = (action - self.action_center) / self.action_scale

        return {
            "obs": {
                "wrist": torch.from_numpy(wrist),
                "state": torch.from_numpy(state.astype(np.float32)),
            },
            "action": torch.from_numpy(action.astype(np.float32)),
        }


class SpatialSoftmax2D(nn.Module):
    """Channel-wise spatial softmax pooling into expected xy coordinates."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_pos_x", torch.empty(0), persistent=False)
        self.register_buffer("_pos_y", torch.empty(0), persistent=False)
        self._grid_hw: tuple[int, int] | None = None

    def _maybe_build_grid(self, height: int, width: int, device: torch.device, dtype: torch.dtype):
        if self._grid_hw == (height, width) and self._pos_x.device == device and self._pos_x.dtype == dtype:
            return
        ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self._pos_x = grid_x.reshape(-1)
        self._pos_y = grid_y.reshape(-1)
        self._grid_hw = (height, width)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        self._maybe_build_grid(height, width, feature_map.device, feature_map.dtype)
        logits = feature_map.reshape(batch_size, channels, height * width)
        attn = torch.softmax(logits, dim=-1)
        exp_x = torch.sum(attn * self._pos_x.view(1, 1, -1), dim=-1)
        exp_y = torch.sum(attn * self._pos_y.view(1, 1, -1), dim=-1)
        return torch.cat((exp_x, exp_y), dim=-1)


class SimpleVisionEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.pool = SpatialSoftmax2D()
        self.encoder = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(image)
        pooled = self.pool(feature_map)
        return self.encoder(pooled)


class ResNet18GroupNormEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        norm_layer = functools.partial(nn.GroupNorm, 32)
        backbone = models.resnet18(weights=None, norm_layer=norm_layer)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.imagenet_mean) / self.imagenet_std
        feat = self.backbone(image)
        return self.proj(feat)


_EquiResBlockBase = escnn_nn.EquivariantModule if escnn_nn is not None else nn.Module


class EquiResBlock(_EquiResBlockBase):
    """Residual block built from escnn equivariant convolutions."""

    def __init__(self, group, in_channels: int, hidden_channels: int, out_channels: int | None = None):
        super().__init__()
        if escnn_nn is None:
            raise ImportError("escnn is required for EquiResBlock.")
        out_channels = hidden_channels if out_channels is None else out_channels
        self.in_type = escnn_nn.FieldType(group, in_channels * [group.regular_repr])
        hidden_type = escnn_nn.FieldType(group, hidden_channels * [group.regular_repr])
        self.out_type = escnn_nn.FieldType(group, out_channels * [group.regular_repr])
        self.block1 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(self.in_type, hidden_type, kernel_size=3, padding=1),
            escnn_nn.ReLU(hidden_type, inplace=True),
        )
        self.block2 = escnn_nn.R2Conv(hidden_type, self.out_type, kernel_size=3, padding=1)
        self.skip = escnn_nn.R2Conv(self.in_type, self.out_type, kernel_size=1) if in_channels != out_channels else None
        self.out_relu = escnn_nn.ReLU(self.out_type, inplace=True)

    def forward(self, x):
        assert x.type == self.in_type
        residual = x if self.skip is None else self.skip(x)
        out = self.block1(x)
        out = self.block2(out)
        out = out + residual
        out = self.out_relu(out)
        assert out.type == self.out_type
        return out

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (input_shape[0], self.out_type.size, input_shape[2], input_shape[3])


class ESCNNCyclicVisionEncoder(nn.Module):
    def __init__(self, out_dim: int, obs_channels: int = 3, n_rotations: int = 8):
        super().__init__()
        try:
            from escnn import gspaces, nn as enn
        except ImportError as exc:
            raise ImportError("escnn is required for --vision_encoder escnn_cyclic.") from exc

        self._enn = enn
        self.group = gspaces.rot2dOnR2(n_rotations)
        n1 = max(out_dim // 16, 8)
        n2 = max(out_dim // 8, 16)
        n3 = max(out_dim // 4, 32)
        n4 = max(out_dim // 2, 64)
        n5 = max(out_dim, 64)
        in_type = enn.FieldType(self.group, obs_channels * [self.group.trivial_repr])
        self.in_type = in_type
        t1 = enn.FieldType(self.group, n1 * [self.group.regular_repr])
        t2 = enn.FieldType(self.group, n2 * [self.group.regular_repr])
        t3 = enn.FieldType(self.group, n3 * [self.group.regular_repr])
        t4 = enn.FieldType(self.group, n4 * [self.group.regular_repr])
        t5 = enn.FieldType(self.group, n5 * [self.group.regular_repr])
        self.net = enn.SequentialModule(
            enn.R2Conv(in_type, t1, kernel_size=5, padding=0),
            enn.ReLU(t1, inplace=True),
            EquiResBlock(self.group, n1, n1),
            enn.PointwiseMaxPool(t1, 2),
            EquiResBlock(self.group, n1, n2, out_channels=n2),
            enn.PointwiseMaxPool(t2, 2),
            EquiResBlock(self.group, n2, n3, out_channels=n3),
            enn.PointwiseMaxPool(t3, 2),
            EquiResBlock(self.group, n3, n4, out_channels=n4),
            enn.PointwiseMaxPool(t4, 2),
            EquiResBlock(self.group, n4, n5, out_channels=n5),
            enn.PointwiseMaxPool(t5, 2),
            EquiResBlock(self.group, n5, n5),
            enn.PointwiseMaxPool(t5, 3),
            enn.R2Conv(t5, t5, kernel_size=3, padding=1),
            enn.ReLU(t5, inplace=True),
        )
        self.proj = nn.Sequential(
            nn.LazyLinear(out_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self._enn.GeometricTensor(image, self.in_type)
        feat = self.net(x).tensor
        feat = feat.mean(dim=(-2, -1))
        return self.proj(feat)


class ObservationEncoder(nn.Module):
    def __init__(self, state_dim: int, image_embed_dim: int, obs_feature_dim: int, n_obs_steps: int, vision_encoder: str):
        super().__init__()
        if vision_encoder == "cnn":
            self.image_encoder = SimpleVisionEncoder(image_embed_dim)
        elif vision_encoder == "resnet18_gn":
            self.image_encoder = ResNet18GroupNormEncoder(image_embed_dim)
        elif vision_encoder == "escnn_cyclic":
            self.image_encoder = ESCNNCyclicVisionEncoder(image_embed_dim)
        else:
            raise ValueError(f"Unsupported vision_encoder: {vision_encoder}")

        self.obs_fusion = nn.Sequential(
            nn.Linear(image_embed_dim + state_dim, obs_feature_dim),
            nn.LayerNorm(obs_feature_dim),
            nn.SiLU(),
        )
        self.n_obs_steps = n_obs_steps
        self.global_cond_dim = obs_feature_dim * n_obs_steps

    def forward(self, wrist: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size, n_obs_steps, height, width, channels = wrist.shape
        wrist = wrist.permute(0, 1, 4, 2, 3).reshape(batch_size * n_obs_steps, channels, height, width).contiguous()
        image_feat = self.image_encoder(wrist).reshape(batch_size, n_obs_steps, -1)
        fused = self.obs_fusion(torch.cat((image_feat, state), dim=-1))
        return fused.reshape(batch_size, -1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -factor)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _valid_group_count(num_channels: int, max_groups: int) -> int:
    max_groups = max(1, min(max_groups, num_channels))
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            nn.GroupNorm(_valid_group_count(in_channels, n_groups), in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(_valid_group_count(out_channels, n_groups), out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )
        self.cond_predict_scale = cond_predict_scale
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2 if cond_predict_scale else out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        if self.cond_predict_scale:
            scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        else:
            shift = self.cond_proj(cond)
            h = h + shift.unsqueeze(-1)
        h = self.block2(h)
        return h + self.residual(x)


class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int,
        down_dims: list[int],
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = global_cond_dim + diffusion_step_embed_dim

        dims = [action_dim, *down_dims]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_blocks.append(
                ConditionalResidualBlock1D(
                    dims[i],
                    dims[i + 1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                )
            )
            if i < len(dims) - 2:
                self.downsamples.append(nn.Conv1d(dims[i + 1], dims[i + 1], kernel_size=4, stride=2, padding=1))

        self.mid_block1 = ConditionalResidualBlock1D(
            dims[-1], dims[-1], cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
        )
        self.mid_block2 = ConditionalResidualBlock1D(
            dims[-1], dims[-1], cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
        )

        rev_dims = list(reversed(down_dims))
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(rev_dims) - 1):
            in_channels = rev_dims[i] + rev_dims[i + 1]
            out_channels = rev_dims[i + 1]
            self.up_blocks.append(
                ConditionalResidualBlock1D(
                    in_channels,
                    out_channels,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                )
            )
            self.upsamples.append(nn.ConvTranspose1d(rev_dims[i], rev_dims[i], kernel_size=4, stride=2, padding=1))

        self.final_block = nn.Sequential(
            nn.GroupNorm(_valid_group_count(down_dims[0], n_groups), down_dims[0]),
            nn.SiLU(),
            nn.Conv1d(down_dims[0], action_dim, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = sample.transpose(1, 2).contiguous()
        t_emb = self.timestep_encoder(timesteps)
        cond = torch.cat((global_cond, t_emb), dim=-1)

        skips = []
        for i, block in enumerate(self.down_blocks):
            x = block(x, cond)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        skips = list(reversed(skips[:-1]))
        for block, upsample, skip in zip(self.up_blocks, self.upsamples, skips):
            x = upsample(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            x = block(x, cond)

        x = self.final_block(x)
        return x.transpose(1, 2).contiguous()


class EquiDiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_obs_steps: int,
        image_embed_dim: int,
        obs_feature_dim: int,
        vision_encoder: str,
        diffusion_step_embed_dim: int,
        down_dims: list[int],
        kernel_size: int,
        n_groups: int,
        horizon: int,
        n_action_steps: int,
        num_inference_steps: int,
        cond_predict_scale: bool,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.action_dim = action_dim
        self.obs_encoder = ObservationEncoder(
            state_dim=state_dim,
            image_embed_dim=image_embed_dim,
            obs_feature_dim=obs_feature_dim,
            n_obs_steps=n_obs_steps,
            vision_encoder=vision_encoder,
        )
        self.noise_pred_net = ConditionalUNet1D(
            action_dim=action_dim,
            global_cond_dim=self.obs_encoder.global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.register_buffer("action_center", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_scale", torch.ones(action_dim), persistent=True)

    def forward(self, wrist: torch.Tensor, state: torch.Tensor, noisy_action: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        global_cond = self.obs_encoder(wrist, state)
        return self.noise_pred_net(noisy_action, timesteps, global_cond)

    def set_normalizer(self, action_center: np.ndarray | torch.Tensor, action_scale: np.ndarray | torch.Tensor):
        action_center_tensor = torch.as_tensor(action_center, dtype=self.action_center.dtype, device=self.action_center.device)
        action_scale_tensor = torch.as_tensor(action_scale, dtype=self.action_scale.dtype, device=self.action_scale.device)
        self.action_center.copy_(action_center_tensor)
        self.action_scale.copy_(action_scale_tensor)

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale.view(1, 1, -1) + self.action_center.view(1, 1, -1)

    def conditional_sample(self, obs_dict: dict[str, torch.Tensor], scheduler: DDIMScheduler) -> torch.Tensor:
        wrist = obs_dict["wrist"]
        state = obs_dict["state"]
        sample = torch.randn(
            wrist.shape[0],
            self.horizon,
            self.action_dim,
            device=wrist.device,
            dtype=wrist.dtype,
        )
        scheduler.set_timesteps(self.num_inference_steps, device=wrist.device)
        for timestep in scheduler.timesteps:
            timestep_batch = torch.full((sample.shape[0],), int(timestep), device=wrist.device, dtype=torch.long)
            model_output = self.forward(wrist, state, sample, timestep_batch)
            step_output = scheduler.step(model_output, timestep, sample)
            sample = step_output.prev_sample
        return sample

    def predict_action(self, obs_dict: dict[str, torch.Tensor], scheduler: DDIMScheduler) -> dict[str, torch.Tensor]:
        action_pred_norm = self.conditional_sample(obs_dict, scheduler)
        start = min(max(self.n_obs_steps - 1, 0), max(self.horizon - 1, 0))
        end = min(start + self.n_action_steps, self.horizon)
        action_norm = action_pred_norm[:, start:end]
        return {
            "action": self._denormalize_action(action_norm),
            "action_pred": action_pred_norm,
            "action_pred_denorm": self._denormalize_action(action_pred_norm),
        }


def _create_dataloaders(args, stats: DatasetStats, train_spans: list[EpisodeSpan], val_spans: list[EpisodeSpan]):
    state_center = np.asarray(stats.state_center, dtype=np.float32)
    state_scale = np.asarray(stats.state_scale, dtype=np.float32)
    action_center = np.asarray(stats.action_center, dtype=np.float32)
    action_scale = np.asarray(stats.action_scale, dtype=np.float32)

    train_dataset = H5SequenceDataset(
        args.dataset,
        train_spans,
        args.n_obs_steps,
        args.horizon,
        state_center,
        state_scale,
        action_center,
        action_scale,
        args.rotation_repr,
        args.image_normalization,
    )
    val_dataset = H5SequenceDataset(
        args.dataset,
        val_spans,
        args.n_obs_steps,
        args.horizon,
        state_center,
        state_scale,
        action_center,
        action_scale,
        args.rotation_repr,
        args.image_normalization,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _make_lr_lambda(total_steps: int, warmup_steps: int, schedule_name: str):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def _lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(warmup_steps, 1))
        if schedule_name == "constant":
            return 1.0
        progress_numerator = current_step - warmup_steps
        progress_denominator = max(total_steps - warmup_steps, 1)
        progress = min(max(progress_numerator / progress_denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lr_lambda


def _build_diffusion_scheduler(args):
    scheduler_cls = DDIMScheduler if args.scheduler_type == "ddim" else DDPMScheduler
    scheduler_kwargs = dict(
        num_train_timesteps=args.num_diffusion_steps,
        beta_start=1.0e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type=args.prediction_type,
    )
    if args.scheduler_type == "ddim":
        scheduler_kwargs["set_alpha_to_one"] = True
        scheduler_kwargs["steps_offset"] = 0
    else:
        scheduler_kwargs["variance_type"] = "fixed_small"
    return scheduler_cls(**scheduler_kwargs)


def _compute_diffusion_loss(
    model: nn.Module,
    scheduler: DDIMScheduler,
    wrist: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(action)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (action.shape[0],), device=action.device).long()
    noisy_action = scheduler.add_noise(action, noise, timesteps)
    pred = model(wrist, state, noisy_action, timesteps)
    target = action if scheduler.config.prediction_type == "sample" else noise
    return F.mse_loss(pred, target)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scheduler: DDIMScheduler,
    device: torch.device,
    max_val_steps: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            wrist = batch["obs"]["wrist"].to(device)
            state = batch["obs"]["state"].to(device)
            action = batch["action"].to(device)

            loss = _compute_diffusion_loss(model, scheduler, wrist, state, action)
            total_loss += loss.item() * action.shape[0]
            total_count += action.shape[0]
            if max_val_steps is not None and batch_idx >= max_val_steps:
                break
    return total_loss / max(total_count, 1)


def _load_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage | None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if ema is not None and checkpoint.get("ema_model") is not None:
        ema.shadow.load_state_dict(checkpoint["ema_model"])
    if ema is not None and checkpoint.get("ema_state") is not None:
        ema.optimization_step = checkpoint["ema_state"].get("optimization_step", ema.optimization_step)
    return (
        checkpoint.get("history", []),
        int(checkpoint.get("epoch", 0)) + 1,
        float(checkpoint.get("best_val", float("inf"))),
    )


def _initialize_policy_modules(
    model: nn.Module,
    dataset: Dataset,
    action_dim: int,
    horizon: int,
    device: torch.device,
):
    sample = dataset[0]
    wrist = sample["obs"]["wrist"].unsqueeze(0).to(device)
    state = sample["obs"]["state"].unsqueeze(0).to(device)
    noisy_action = torch.zeros((1, horizon, action_dim), device=device, dtype=state.dtype)
    timesteps = torch.zeros((1,), device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        model(wrist, state, noisy_action, timesteps)


def main():
    args = parse_args()
    set_seed(args.seed)
    run_start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Rotation representation: {args.rotation_repr}")
    print(f"[INFO] Observation window: {args.n_obs_steps}")
    print(f"[INFO] Action horizon: {args.horizon}")

    spans = _load_episode_spans(args.dataset)
    train_spans, val_spans = _split_episodes(
        spans, args.val_split, args.seed, args.max_train_episodes, args.max_val_episodes
    )
    if not train_spans:
        raise ValueError("Training split is empty.")

    print(f"[INFO] Episodes: total={len(spans)} train={len(train_spans)} val={len(val_spans)}")
    stats = _compute_stats(args.dataset, train_spans, args.rotation_repr, args.state_action_normalization)
    state_dim = len(stats.state_center)
    action_dim = len(stats.action_center)
    print(f"[INFO] State dim: {state_dim}")
    print(f"[INFO] Action dim after conversion: {action_dim}")

    train_dataset, val_dataset, train_loader, val_loader = _create_dataloaders(args, stats, train_spans, val_spans)
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Validation samples: {len(val_dataset)}")

    device = torch.device(args.device)
    model = EquiDiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=args.n_obs_steps,
        image_embed_dim=args.image_embed_dim,
        obs_feature_dim=args.obs_feature_dim,
        vision_encoder=args.vision_encoder,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.down_dims,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        cond_predict_scale=args.cond_predict_scale,
    ).to(device)
    ema_model = None
    if args.use_ema:
        ema_model = EquiDiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            n_obs_steps=args.n_obs_steps,
            image_embed_dim=args.image_embed_dim,
            obs_feature_dim=args.obs_feature_dim,
            vision_encoder=args.vision_encoder,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.down_dims,
            kernel_size=args.kernel_size,
            n_groups=args.n_groups,
            horizon=args.horizon,
            n_action_steps=args.n_action_steps,
            num_inference_steps=args.num_inference_steps,
            cond_predict_scale=args.cond_predict_scale,
        ).to(device)
    _initialize_policy_modules(model, train_dataset, action_dim, args.horizon, device)
    if ema_model is not None:
        _initialize_policy_modules(ema_model, train_dataset, action_dim, args.horizon, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.adam_betas),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    diffusion_scheduler = _build_diffusion_scheduler(args)
    batches_per_epoch = len(train_loader) if args.max_train_steps is None else min(len(train_loader), args.max_train_steps)
    optimizer_steps_per_epoch = max(math.ceil(batches_per_epoch / max(args.gradient_accumulate_every, 1)), 1)
    total_train_steps = max(optimizer_steps_per_epoch * args.epochs, 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(total_train_steps, args.lr_warmup_steps, args.lr_scheduler),
    )
    ema = (
        ExponentialMovingAverage(
            model,
            ema_model,
            update_after_step=args.ema_update_after_step,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            min_value=args.ema_min_value,
            max_value=args.ema_max_value,
        )
        if args.use_ema
        else None
    )
    action_center = np.asarray(stats.action_center, dtype=np.float32)
    action_scale = np.asarray(stats.action_scale, dtype=np.float32)
    model.set_normalizer(action_center, action_scale)
    if ema_model is not None:
        ema_model.set_normalizer(action_center, action_scale)

    config = vars(args).copy()
    config.update(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "num_train_episodes": len(train_spans),
            "num_val_episodes": len(val_spans),
            "resolved_total_train_steps": total_train_steps,
        }
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    best_val = float("inf")
    history = []
    start_epoch = 1
    train_sampling_batch = None
    resume_path = checkpoint_dir / "last.pt"
    if args.resume and resume_path.exists():
        history, start_epoch, best_val = _load_training_state(resume_path, model, optimizer, lr_scheduler, ema)
        print(f"[INFO] Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        total_count = 0
        optimizer.zero_grad(set_to_none=True)
        effective_train_steps = len(train_loader) if args.max_train_steps is None else min(len(train_loader), args.max_train_steps)

        for batch_idx, batch in enumerate(train_loader, start=1):
            wrist = batch["obs"]["wrist"].to(device)
            state = batch["obs"]["state"].to(device)
            action = batch["action"].to(device)
            if train_sampling_batch is None:
                train_sampling_batch = {
                    "obs": {"wrist": wrist.detach().clone(), "state": state.detach().clone()},
                    "action": action.detach().clone(),
                }

            raw_loss = _compute_diffusion_loss(model, diffusion_scheduler, wrist, state, action)
            loss = raw_loss / max(args.gradient_accumulate_every, 1)
            loss.backward()

            is_accum_boundary = (batch_idx % max(args.gradient_accumulate_every, 1) == 0) or (batch_idx == effective_train_steps)
            if is_accum_boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if ema is not None:
                    ema.update(model)

            running_loss += raw_loss.item() * action.shape[0]
            total_count += action.shape[0]

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                epoch_elapsed = time.time() - epoch_start
                run_elapsed = time.time() - run_start_time
                print(
                    f"[INFO] Epoch {epoch:03d} Batch {batch_idx:04d}: "
                    f"train_loss={running_loss / max(total_count, 1):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"epoch_elapsed={_format_duration(epoch_elapsed)} "
                    f"epoch_eta={_estimate_eta(epoch_elapsed, batch_idx, effective_train_steps)} "
                    f"run_elapsed={_format_duration(run_elapsed)} "
                    f"run_eta={_estimate_eta(run_elapsed, epoch - start_epoch + 1, args.epochs - start_epoch + 1)}"
                )

            if args.max_train_steps is not None and batch_idx >= args.max_train_steps:
                break

        train_loss = running_loss / max(total_count, 1)
        eval_model = ema.shadow if ema is not None else model
        val_loss = (
            evaluate(eval_model, val_loader, diffusion_scheduler, device, max_val_steps=args.max_val_steps)
            if len(val_dataset) > 0
            else train_loss
        )
        epoch_time = time.time() - epoch_start
        sample_mse = None
        if train_sampling_batch is not None and epoch % max(args.sample_every, 1) == 0:
            eval_model.eval()
            with torch.inference_mode():
                result = eval_model.predict_action(train_sampling_batch["obs"], diffusion_scheduler)
                sample_mse = F.mse_loss(result["action_pred"], train_sampling_batch["action"]).item()
        history_entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if sample_mse is not None:
            history_entry["train_action_mse_error"] = sample_mse
        history.append(history_entry)
        sample_mse_text = f"sample_mse={sample_mse:.6f} " if sample_mse is not None else ""
        run_elapsed = time.time() - run_start_time
        print(
            f"[INFO] Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"{sample_mse_text}"
            f"lr={optimizer.param_groups[0]['lr']:.6e} "
            f"epoch_time={_format_duration(epoch_time)} "
            f"run_elapsed={_format_duration(run_elapsed)} "
            f"run_eta={_estimate_eta(run_elapsed, epoch - start_epoch + 1, args.epochs - start_epoch + 1)}"
        )

        is_best = val_loss <= best_val
        if is_best:
            best_val = val_loss

        checkpoint = {
            "model": model.state_dict(),
            "ema_model": ema.shadow.state_dict() if ema is not None else None,
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "config": config,
            "stats": asdict(stats),
            "history": history,
            "epoch": epoch,
            "best_val": best_val,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    elapsed = time.time() - run_start_time
    print(f"[INFO] Training finished in {_format_duration(elapsed)}")
    print(f"[INFO] Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
