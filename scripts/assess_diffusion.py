#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Assess an offline diffusion visuomotor policy inside Isaac Lab."""

from __future__ import annotations

import argparse
import dill
import hydra
import importlib
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
DP_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "third_party", "diffusion_policy")
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)


parser = argparse.ArgumentParser(description="Assess an offline diffusion policy in Isaac Lab.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to offline diffusion checkpoint (*.pt).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during assessment.")
parser.add_argument(
    "--video_src",
    type=str,
    default="pov",
    choices=["pov", "zed", "both", "side_grid"],
    help="Video source: `pov` records env-0 side-view, `side_grid` records a 3x3 side-view grid, `zed` records wrist, `both` writes side-view+wrist.",
)
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--side_view_grid_9",
    action="store_true",
    default=False,
    help="Force num_envs=9 and record a 3x3 side-view camera grid. Implies --video --video_src side_grid.",
)
parser.add_argument(
    "--num_loops",
    type=int,
    default=1,
    help="Number of episode-length loops to run before stopping. Use <= 0 to run until closed.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--random_orn",
    type=float,
    default=None,
    help="Enable random EE roll/pitch initialization with +/- this many degrees for tasks that support it.",
)
parser.add_argument(
    "--fixed_eef_init",
    action="store_true",
    default=False,
    help="Disable random EEF position/orientation initialization noise for assessment.",
)
parser.add_argument(
    "--fixed_asset_yaw_deg",
    type=float,
    default=None,
    help="Override the fixed asset nominal yaw in degrees.",
)
parser.add_argument(
    "--fixed_asset_yaw_range_deg",
    type=float,
    default=None,
    help="Override the fixed asset yaw randomization range in degrees. Use 0 for fixed yaw.",
)
parser.add_argument(
    "--fixed_asset_height",
    action="store_true",
    default=False,
    help="Disable fixed-asset Z-position randomization while keeping XY position randomization unchanged.",
)
parser.add_argument(
    "--fixed_held_asset_height",
    action="store_true",
    default=False,
    help="Disable held-asset Z-position randomization while keeping XY held-asset randomization unchanged.",
)
parser.add_argument(
    "--ft",
    action="store_true",
    default=False,
    help="Append left/right 6D FT wrench readings to the low-dimensional state during inference.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--height_diff_log_interval",
    type=int,
    default=50,
    help="Print held-base height difference vector every N policy steps. Use <= 0 to disable.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401


def _set_default_factory_video_view(env_cfg, task_name: str | None):
    if "Factory" not in (task_name or ""):
        return
    if not hasattr(env_cfg, "viewer") or env_cfg.viewer is None:
        return
    if hasattr(env_cfg.viewer, "eye"):
        env_cfg.viewer.eye = (1.4, -0.015, 0.28)
    if hasattr(env_cfg.viewer, "lookat"):
        env_cfg.viewer.lookat = (0.60, 0.00, 0.12)


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _get_current_success_rate(env):
    if not hasattr(env, "_get_curr_successes"):
        return None
    check_rot = getattr(env.cfg_task, "name", None) == "nut_thread"
    curr_successes = env._get_curr_successes(success_threshold=env.cfg_task.success_threshold, check_rot=check_rot)
    return torch.count_nonzero(curr_successes).float() / env.num_envs


def _get_episode_success_rate(env):
    if not hasattr(env, "ep_succeeded"):
        return None
    return torch.count_nonzero(env.ep_succeeded).float() / env.num_envs


def _get_height_diff_vector(env):
    required_attrs = ("held_pos", "held_quat", "fixed_pos", "fixed_quat", "cfg_task")
    if not all(hasattr(env, attr) for attr in required_attrs):
        return None

    try:
        factory_utils = importlib.import_module(env.__class__.__module__.rsplit(".", 1)[0] + ".factory_utils")
    except (ImportError, ValueError):
        return None

    held_base_pos, _ = factory_utils.get_held_base_pose(
        env.held_pos,
        env.held_quat,
        env.cfg_task.name,
        env.cfg_task.fixed_asset_cfg,
        env.num_envs,
        env.device,
    )
    target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
        env.fixed_pos,
        env.fixed_quat,
        env.cfg_task.name,
        env.cfg_task.fixed_asset_cfg,
        env.num_envs,
        env.device,
    )
    return held_base_pos[:, 2] - target_held_base_pos[:, 2]


def _print_height_diff_vector(env, step: int):
    z_disp = _get_height_diff_vector(env)
    if z_disp is None:
        return
    z_disp_np = z_disp.detach().cpu().numpy()
    fixed_cfg = env.cfg_task.fixed_asset_cfg
    if env.cfg_task.name in ("peg_insert", "gear_mesh"):
        threshold = fixed_cfg.height * env.cfg_task.success_threshold
    elif env.cfg_task.name == "nut_thread":
        threshold = fixed_cfg.thread_pitch * env.cfg_task.success_threshold
    else:
        threshold = None
    threshold_text = f", threshold={threshold:.6f} m" if threshold is not None else ""
    print(
        f"[INFO] Step {step}: height_diff_m held_base_z-target_z"
        f"{threshold_text}: {np.array2string(z_disp_np, precision=6, separator=', ')}"
    )


def _apply_factory_init_overrides(env_cfg):
    task_cfg = getattr(env_cfg, "task", None)
    if task_cfg is None:
        return

    if args_cli.fixed_eef_init:
        task_cfg.hand_init_pos_noise = [0.0, 0.0, 0.0]
        task_cfg.hand_init_orn_noise = [0.0, 0.0, 0.0]
        if hasattr(task_cfg, "randomize_hand_init_tilt"):
            task_cfg.randomize_hand_init_tilt = False
        print("[INFO] Fixed EEF init enabled: zeroed hand init position/orientation noise.")

    if args_cli.fixed_asset_yaw_deg is not None:
        task_cfg.fixed_asset_init_orn_deg = float(args_cli.fixed_asset_yaw_deg)
        print(f"[INFO] Fixed asset nominal yaw set to {task_cfg.fixed_asset_init_orn_deg:.2f} deg.")

    if args_cli.fixed_asset_yaw_range_deg is not None:
        task_cfg.fixed_asset_init_orn_range_deg = float(args_cli.fixed_asset_yaw_range_deg)
        print(f"[INFO] Fixed asset yaw range set to {task_cfg.fixed_asset_init_orn_range_deg:.2f} deg.")

    if args_cli.fixed_asset_height and hasattr(task_cfg, "fixed_asset_init_pos_noise"):
        pos_noise = list(task_cfg.fixed_asset_init_pos_noise)
        if len(pos_noise) < 3:
            raise ValueError("task.fixed_asset_init_pos_noise must have at least 3 elements [x, y, z].")
        pos_noise[2] = 0.0
        task_cfg.fixed_asset_init_pos_noise = pos_noise
        print("[INFO] Fixed asset height enabled: zeroed fixed-asset Z-position randomization noise.")

    if args_cli.fixed_held_asset_height and hasattr(task_cfg, "held_asset_pos_noise"):
        pos_noise = list(task_cfg.held_asset_pos_noise)
        if len(pos_noise) < 3:
            raise ValueError("task.held_asset_pos_noise must have at least 3 elements [x, y, z].")
        pos_noise[2] = 0.0
        task_cfg.held_asset_pos_noise = pos_noise
        print("[INFO] Fixed held-asset height enabled: zeroed held-asset Z-position randomization noise.")


def _to_uint8_rgb(frame: torch.Tensor):
    frame = frame.detach().cpu()
    if frame.dtype != torch.uint8:
        frame = torch.clamp(frame, 0, 255).to(torch.uint8)
    return frame.numpy()


def _to_uint8_rgb_array(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        return _to_uint8_rgb(frame)
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _make_zed_grid(rgb_batch: torch.Tensor, crop_size: int | None = None):
    if crop_size is not None:
        rgb_batch = _center_crop_torch(rgb_batch[..., :3], crop_size)
    else:
        rgb_batch = rgb_batch[..., :3]
    frames = [_to_uint8_rgb(rgb_batch[i]) for i in range(min(rgb_batch.shape[0], 10))]
    if not frames:
        raise RuntimeError("No wrist camera frames available for ZED video recording.")
    frame_h, frame_w, frame_c = frames[0].shape
    blank = torch.zeros((frame_h, frame_w, frame_c), dtype=torch.uint8).numpy()
    while len(frames) < 10:
        frames.append(blank.copy())
    top = frames[0:5]
    bottom = frames[5:10]
    return np.concatenate((np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)), axis=0)


def _make_rgb_grid(
    rgb_batch: torch.Tensor, rows: int, cols: int, crop_size: int | None = None, label: str = "camera"
) -> np.ndarray:
    if crop_size is not None:
        rgb_batch = _center_crop_torch(rgb_batch[..., :3], crop_size)
    else:
        rgb_batch = rgb_batch[..., :3]
    num_frames = rows * cols
    frames = [_to_uint8_rgb(rgb_batch[i]) for i in range(min(rgb_batch.shape[0], num_frames))]
    if not frames:
        raise RuntimeError(f"No {label} frames available for grid video recording.")
    frame_h, frame_w, frame_c = frames[0].shape
    blank = np.zeros((frame_h, frame_w, frame_c), dtype=np.uint8)
    while len(frames) < num_frames:
        frames.append(blank.copy())
    grid_rows = []
    for row_idx in range(rows):
        start = row_idx * cols
        grid_rows.append(np.concatenate(frames[start : start + cols], axis=1))
    return np.concatenate(grid_rows, axis=0)


def _center_crop_numpy(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if target_h > height or target_w > width:
        raise ValueError(
            f"Cannot crop frame of size {(height, width)} to larger target {(target_h, target_w)}."
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return frame[top : top + target_h, left : left + target_w, :]


def _make_side_view_wrist_side_by_side(
    side_view_rgb_batch: torch.Tensor, wrist_rgb_batch: torch.Tensor, crop_size: int | None = None
) -> np.ndarray:
    side_view = side_view_rgb_batch[..., :3]
    if crop_size is not None:
        side_view = _center_crop_torch(side_view, crop_size)
        wrist_rgb_batch = _center_crop_torch(wrist_rgb_batch[..., :3], crop_size)
    else:
        side_view = side_view[..., :3]
        wrist_rgb_batch = wrist_rgb_batch[..., :3]
    side_view = _to_uint8_rgb(side_view[0])
    wrist = _to_uint8_rgb(wrist_rgb_batch[0])
    wrist_h, wrist_w = wrist.shape[:2]
    if side_view.shape[0] != wrist_h or side_view.shape[1] != wrist_w:
        side_view = _center_crop_numpy(side_view, wrist_h, wrist_w)
    return np.concatenate((side_view, wrist), axis=1)


def _center_crop_torch(images: torch.Tensor, crop_size: int | None) -> torch.Tensor:
    """Center crop NHWC image batches to a square size."""
    if crop_size is None:
        return images
    if crop_size <= 0:
        raise ValueError(f"image_crop_size must be positive when set, got {crop_size}")
    height, width = images.shape[1:3]
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"image_crop_size={crop_size} exceeds image dimensions {(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, top : top + crop_size, left : left + crop_size, :]


def _resolve_ft_body_indices(env) -> tuple[object, int, int]:
    """Resolve fingertip link ids whose incoming fixed-joint wrench is used as FT."""
    robot = env.scene["robot"]
    left_name = "fr3_left_ft"
    right_name = "fr3_right_ft"
    try:
        left_id = robot.body_names.index(left_name)
        right_id = robot.body_names.index(right_name)
    except ValueError as exc:
        raise ValueError(
            f"Could not resolve fingertip FT bodies '{left_name}' and '{right_name}'. "
            f"Available bodies: {robot.body_names}"
        ) from exc
    return robot, left_id, right_id


class OfflineDiffusionInferencePolicy:
    def __init__(self, checkpoint_path: str, device: torch.device):
        payload = torch.load(open(checkpoint_path, "rb"), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = workspace_cls(cfg, output_dir=os.path.dirname(checkpoint_path))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        model = workspace.ema_model if cfg.training.use_ema else workspace.model

        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.image_crop_size = cfg.task.dataset.get("image_crop_size")
        self.sample_obs_cfg = cfg.shape_meta.get("sample", {}).get("obs", {}).get("sparse", {})
        self.n_obs_steps = int(cfg.get("n_obs_steps", cfg.task.dataset.n_obs_steps))
        self.rgb_keys = [
            key for key, attr in cfg.shape_meta.obs.items() if attr.get("type", "low_dim") == "rgb"
        ]
        self.low_dim_keys = [
            key for key, attr in cfg.shape_meta.obs.items() if attr.get("type", "low_dim") == "low_dim"
        ]
        self.wrench_keys = [key for key in self.low_dim_keys if "wrench" in key]
        self.use_ft = bool(self.wrench_keys) or bool(cfg.task.dataset.get("use_ft", False)) or args_cli.ft
        self.key_horizons = {
            key: int(self.sample_obs_cfg.get(key, {}).get("horizon", self.n_obs_steps))
            for key in [*self.rgb_keys, *self.low_dim_keys]
        }
        if "wrist" not in self.rgb_keys:
            raise ValueError(f"Offline diffusion assessment requires a wrist RGB obs key, got {self.rgb_keys}.")
        if "state" not in self.low_dim_keys:
            raise ValueError(f"Offline diffusion assessment requires a state low-dim obs key, got {self.low_dim_keys}.")
        if args_cli.ft and not self.wrench_keys:
            print("[WARN] --ft was passed, but checkpoint shape_meta has no separate wrench obs keys.")
        self._obs_histories = None
        self._action_plan = None
        self._action_step = 0
        self._robot = None
        self._left_ft_body_idx = None
        self._right_ft_body_idx = None

    def reset(self):
        self._obs_histories = None
        self._action_plan = None
        self._action_step = 0

    def _normalize_images(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = _center_crop_torch(rgb, self.image_crop_size)
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        return rgb.float() / 255.0

    def _maybe_init_ft(self, env):
        if not self.use_ft or self._robot is not None:
            return
        self._robot, self._left_ft_body_idx, self._right_ft_body_idx = _resolve_ft_body_indices(env)

    def _build_current_obs(self, env) -> dict[str, torch.Tensor]:
        if env._wrist_camera is None:
            raise RuntimeError("Offline diffusion policy requires wrist camera data, but no wrist camera is configured.")
        obs = {"wrist": self._normalize_images(env._wrist_camera.data.output["rgb"][..., :3])}
        if "side_view" in self.rgb_keys:
            if getattr(env, "_side_view_camera", None) is None:
                raise RuntimeError(
                    "Offline diffusion policy requires side_view image data, but no side-view camera is configured."
                )
            obs["side_view"] = self._normalize_images(env._side_view_camera.data.output["rgb"][..., :3])
        gripper_pos = torch.mean(env.joint_pos[:, 7:], dim=1, keepdim=True)
        obs["state"] = torch.cat((env.fingertip_midpoint_pos, env.fingertip_midpoint_quat, gripper_pos), dim=-1)
        if self.wrench_keys:
            self._maybe_init_ft(env)
            obs["left_ft_wrench"] = self._robot.data.body_incoming_joint_wrench_b[:, self._left_ft_body_idx]
            obs["right_ft_wrench"] = self._robot.data.body_incoming_joint_wrench_b[:, self._right_ft_body_idx]
        elif args_cli.ft:
            self._maybe_init_ft(env)
            left_ft_wrench = self._robot.data.body_incoming_joint_wrench_b[:, self._left_ft_body_idx]
            right_ft_wrench = self._robot.data.body_incoming_joint_wrench_b[:, self._right_ft_body_idx]
            obs["state"] = torch.cat((obs["state"], left_ft_wrench, right_ft_wrench), dim=-1)
        return obs

    def _init_history(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.unsqueeze(1).repeat(1, self.key_horizons[key], *([1] * (value.ndim - 1)))
            for key, value in obs.items()
        }

    def _update_history(self, obs: dict[str, torch.Tensor]):
        for key, value in obs.items():
            self._obs_histories[key] = torch.roll(self._obs_histories[key], shifts=-1, dims=1)
            self._obs_histories[key][:, -1] = value

    @torch.inference_mode()
    def act(self, env) -> torch.Tensor:
        obs = self._build_current_obs(env)
        if self._obs_histories is None:
            self._obs_histories = self._init_history(obs)
        else:
            self._update_history(obs)

        if self._action_plan is None or self._action_step >= self._action_plan.shape[1]:
            obs_dict = {key: self._obs_histories[key] for key in [*self.rgb_keys, *self.low_dim_keys]}
            result = self.model.predict_action(obs_dict)
            self._action_plan = result["action"]
            self._action_step = 0

        action = self._action_plan[:, self._action_step]
        self._action_step += 1
        return action


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    del agent_cfg

    if args_cli.side_view_grid_9:
        args_cli.video = True
        args_cli.video_src = "side_grid"
        args_cli.num_envs = 9
        print("[INFO] Side-view 3x3 grid enabled: forcing --video --video_src side_grid --num_envs 9.")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.random_orn is not None and hasattr(env_cfg, "task"):
        if hasattr(env_cfg.task, "randomize_hand_init_tilt"):
            env_cfg.task.randomize_hand_init_tilt = True
            env_cfg.task.hand_init_tilt_noise_deg = args_cli.random_orn
        if hasattr(env_cfg.task, "hand_init_orn_noise"):
            orn_noise = list(env_cfg.task.hand_init_orn_noise)
            if len(orn_noise) < 3:
                raise ValueError("task.hand_init_orn_noise must have at least 3 elements [roll, pitch, yaw].")
            tilt_noise_rad = float(np.deg2rad(args_cli.random_orn))
            orn_noise[0] = tilt_noise_rad
            orn_noise[1] = tilt_noise_rad
            env_cfg.task.hand_init_orn_noise = orn_noise
            print(
                f"[INFO] Random EEF roll/pitch init enabled: +/- {args_cli.random_orn:.2f} deg; "
                f"yaw noise remains +/- {np.rad2deg(orn_noise[2]):.2f} deg."
            )
    _apply_factory_init_overrides(env_cfg)

    env_cfg.scene.clone_in_fabric = False

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading offline diffusion checkpoint from: {checkpoint_path}")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    video_root = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir) == "checkpoints" else checkpoint_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    max_assessment_steps = None
    effective_video_length = args_cli.video_length
    if args_cli.num_loops > 0:
        steps_per_loop = max(base_env.max_episode_length - 1, 1)
        max_assessment_steps = args_cli.num_loops * steps_per_loop
        if args_cli.video:
            effective_video_length = max_assessment_steps
        print(
            f"[INFO] Assessment will stop after {args_cli.num_loops} loop(s): "
            f"{max_assessment_steps} steps ({steps_per_loop} steps per loop)."
        )

    if args_cli.video:
        video_dir = os.path.join(video_root, "videos")
        os.makedirs(video_dir, exist_ok=True)
        if args_cli.video_src == "pov":
            pov_video_path = os.path.join(video_dir, f"{args_cli.task.replace(':', '_')}_side_view.mp4")
            print(f"[INFO] Recording side-view video to: {pov_video_path}")
            pov_writer = imageio.get_writer(pov_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1))
            zed_writer = None
            both_writer = None
            side_grid_writer = None
        elif args_cli.video_src == "side_grid":
            side_grid_video_path = os.path.join(video_dir, f"{args_cli.task.replace(':', '_')}_side_view_3x3.mp4")
            print(f"[INFO] Recording 3x3 side-view grid video to: {side_grid_video_path}")
            side_grid_writer = imageio.get_writer(
                side_grid_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1)
            )
            pov_writer = None
            zed_writer = None
            both_writer = None
        elif args_cli.video_src == "both":
            both_video_path = os.path.join(video_dir, f"{args_cli.task.replace(':', '_')}_both.mp4")
            print(f"[INFO] Recording side-by-side side-view+wrist video to: {both_video_path}")
            both_writer = imageio.get_writer(both_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1))
            pov_writer = None
            zed_writer = None
            side_grid_writer = None
        else:
            zed_video_path = os.path.join(video_dir, f"{args_cli.task.replace(':', '_')}_zed.mp4")
            print(f"[INFO] Recording ZED wrist video to: {zed_video_path}")
            zed_writer = imageio.get_writer(zed_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1))
            pov_writer = None
            both_writer = None
            side_grid_writer = None
    else:
        pov_writer = None
        zed_writer = None
        both_writer = None
        side_grid_writer = None

    device = torch.device(args_cli.device or env_cfg.sim.device)
    policy = OfflineDiffusionInferencePolicy(checkpoint_path, device)

    env.reset()
    policy.reset()
    timestep = 0
    completed_loops = 0
    loop_success_rates = []

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy.act(base_env)
            _, _, terminated, truncated, extras = env.step(actions)
            dones = torch.logical_or(terminated, truncated)
            if pov_writer is not None:
                if getattr(base_env, "_side_view_camera", None) is None:
                    raise RuntimeError("POV video requested, but no side-view camera is configured on the environment.")
                side_view_batch = base_env._side_view_camera.data.output["rgb"]
                side_view_frame = side_view_batch[..., :3]
                if policy.image_crop_size is not None:
                    side_view_frame = _center_crop_torch(side_view_frame, policy.image_crop_size)
                pov_writer.append_data(_to_uint8_rgb(side_view_frame[0]))
            if zed_writer is not None:
                zed_batch = base_env._wrist_camera.data.output["rgb"]
                zed_writer.append_data(_make_zed_grid(zed_batch, policy.image_crop_size))
            if side_grid_writer is not None:
                if getattr(base_env, "_side_view_camera", None) is None:
                    raise RuntimeError("Side-view grid requested, but no side-view camera is configured on the environment.")
                side_view_batch = base_env._side_view_camera.data.output["rgb"]
                side_grid_writer.append_data(
                    _make_rgb_grid(side_view_batch, rows=3, cols=3, crop_size=policy.image_crop_size, label="side-view")
                )
            if both_writer is not None:
                if getattr(base_env, "_side_view_camera", None) is None:
                    raise RuntimeError("Combined video requested, but no side-view camera is configured on the environment.")
                side_view_batch = base_env._side_view_camera.data.output["rgb"]
                wrist_batch = base_env._wrist_camera.data.output["rgb"]
                both_writer.append_data(
                    _make_side_view_wrist_side_by_side(side_view_batch, wrist_batch, policy.image_crop_size)
                )

            if args_cli.height_diff_log_interval > 0 and (timestep + 1) % args_cli.height_diff_log_interval == 0:
                _print_height_diff_vector(base_env, timestep + 1)

            if len(dones) > 0 and torch.all(dones).item():
                completed_loops += 1
                episode_success_rate = _get_episode_success_rate(base_env)
                episode_text = (
                    f"episode success rate = {_to_float(episode_success_rate):.4f}"
                    if episode_success_rate is not None
                    else "episode success rate = unavailable"
                )
                print(f"[INFO] Loop {completed_loops}: {episode_text}")
                if episode_success_rate is not None:
                    loop_success_rates.append(episode_success_rate)
                policy.reset()
                if args_cli.num_loops > 0 and completed_loops >= args_cli.num_loops:
                    break

        timestep += 1
        if max_assessment_steps is not None and timestep >= max_assessment_steps:
            break

        sleep_time = base_env.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if loop_success_rates:
        mean_success_rate = torch.stack(loop_success_rates).mean()
        print(f"[INFO] Mean episode success rate over {len(loop_success_rates)} loop(s): {_to_float(mean_success_rate):.4f}")

    if pov_writer is not None:
        pov_writer.close()
    if zed_writer is not None:
        zed_writer.close()
    if both_writer is not None:
        both_writer.close()
    if side_grid_writer is not None:
        side_grid_writer.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
