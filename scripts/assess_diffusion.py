#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Assess an offline diffusion visuomotor policy inside Isaac Lab."""

from __future__ import annotations

import argparse
import dill
import hydra
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
    choices=["pov", "zed", "both"],
    help="Video source: `pov` records the viewer perspective, `zed` records the wrist camera stream, `both` writes a side-by-side POV+wrist video.",
)
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
    "--ft",
    action="store_true",
    default=False,
    help="Append left/right 6D FT wrench readings to the low-dimensional state during inference.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
        env_cfg.viewer.eye = (1.7, -0.02, 0.34)
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


def _center_crop_numpy(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if target_h > height or target_w > width:
        raise ValueError(
            f"Cannot crop frame of size {(height, width)} to larger target {(target_h, target_w)}."
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return frame[top : top + target_h, left : left + target_w, :]


def _make_pov_wrist_side_by_side(
    pov_frame, wrist_rgb_batch: torch.Tensor, crop_size: int | None = None
) -> np.ndarray:
    pov = _to_uint8_rgb_array(pov_frame)
    if crop_size is not None:
        wrist_rgb_batch = _center_crop_torch(wrist_rgb_batch[..., :3], crop_size)
    else:
        wrist_rgb_batch = wrist_rgb_batch[..., :3]
    wrist = _to_uint8_rgb(wrist_rgb_batch[0])
    wrist_h, wrist_w = wrist.shape[:2]
    if pov.shape[0] != wrist_h or pov.shape[1] != wrist_w:
        pov = _center_crop_numpy(pov, wrist_h, wrist_w)
    return np.concatenate((pov, wrist), axis=1)


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
    """Resolve the left/right FT body ids on the robot articulation."""
    robot = env.scene["robot"]
    left_ids, _ = robot.find_bodies("fr3_left_ft")
    right_ids, _ = robot.find_bodies("fr3_right_ft")
    if len(left_ids) == 0 or len(right_ids) == 0:
        raise ValueError("Could not resolve FT bodies 'fr3_left_ft' and 'fr3_right_ft' on scene['robot'].")
    return robot, int(left_ids[0]), int(right_ids[0])


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
        self.use_ft = bool(cfg.task.dataset.get("use_ft", False)) or args_cli.ft
        self.n_obs_steps = int(cfg.n_obs_steps)
        self._wrist_history = None
        self._state_history = None
        self._action_plan = None
        self._action_step = 0
        self._robot = None
        self._left_ft_body_idx = None
        self._right_ft_body_idx = None

    def reset(self):
        self._wrist_history = None
        self._state_history = None
        self._action_plan = None
        self._action_step = 0

    def _normalize_images(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        wrist_rgb = _center_crop_torch(wrist_rgb, self.image_crop_size)
        return wrist_rgb.float() / 255.0

    def _maybe_init_ft(self, env):
        if not self.use_ft or self._robot is not None:
            return
        self._robot, self._left_ft_body_idx, self._right_ft_body_idx = _resolve_ft_body_indices(env)

    def _build_current_obs(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        if env._wrist_camera is None:
            raise RuntimeError("Offline diffusion policy requires wrist camera data, but no wrist camera is configured.")
        wrist = self._normalize_images(env._wrist_camera.data.output["rgb"][..., :3])
        gripper_pos = torch.mean(env.joint_pos[:, 7:], dim=1, keepdim=True)
        state_parts = [env.fingertip_midpoint_pos, env.fingertip_midpoint_quat, gripper_pos]
        if self.use_ft:
            self._maybe_init_ft(env)
            left_ft_wrench = self._robot.data.body_incoming_joint_wrench_b[:, self._left_ft_body_idx]
            right_ft_wrench = self._robot.data.body_incoming_joint_wrench_b[:, self._right_ft_body_idx]
            state_parts.extend((left_ft_wrench, right_ft_wrench))
        return wrist, torch.cat(state_parts, dim=-1)

    @torch.inference_mode()
    def act(self, env) -> torch.Tensor:
        wrist, state = self._build_current_obs(env)
        if self._wrist_history is None or self._state_history is None:
            self._wrist_history = wrist.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1, 1)
            self._state_history = state.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
        else:
            self._wrist_history = torch.roll(self._wrist_history, shifts=-1, dims=1)
            self._wrist_history[:, -1] = wrist
            self._state_history = torch.roll(self._state_history, shifts=-1, dims=1)
            self._state_history[:, -1] = state

        if self._action_plan is None or self._action_step >= self._action_plan.shape[1]:
            result = self.model.predict_action({"wrist": self._wrist_history, "state": self._state_history})
            self._action_plan = result["action"]
            self._action_step = 0

        action = self._action_plan[:, self._action_step]
        self._action_step += 1
        return action


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    del agent_cfg

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.random_orn is not None and hasattr(env_cfg, "task") and hasattr(env_cfg.task, "randomize_hand_init_tilt"):
        env_cfg.task.randomize_hand_init_tilt = True
        env_cfg.task.hand_init_tilt_noise_deg = args_cli.random_orn

    env_cfg.scene.clone_in_fabric = False
    if args_cli.video and args_cli.video_src in ("pov", "both"):
        _set_default_factory_video_view(env_cfg, args_cli.task)

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
            video_kwargs = {
                "video_folder": video_dir,
                "step_trigger": lambda step: step == 0,
                "video_length": effective_video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording POV videos during assessment.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            base_env = env.unwrapped
            zed_writer = None
            both_writer = None
        elif args_cli.video_src == "both":
            both_video_path = os.path.join(video_dir, f"{args_cli.task.replace(':', '_')}_both.mp4")
            print(f"[INFO] Recording side-by-side POV+wrist video to: {both_video_path}")
            both_writer = imageio.get_writer(both_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1))
            zed_writer = None
        else:
            zed_video_path = os.path.join(video_dir, f"{args_cli.task.replace(':', '_')}_zed.mp4")
            print(f"[INFO] Recording ZED wrist video to: {zed_video_path}")
            zed_writer = imageio.get_writer(zed_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1))
            both_writer = None
    else:
        zed_writer = None
        both_writer = None

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
            if zed_writer is not None:
                zed_batch = base_env._wrist_camera.data.output["rgb"]
                zed_writer.append_data(_make_zed_grid(zed_batch, policy.image_crop_size))
            if both_writer is not None:
                pov_frame = env.render()
                wrist_batch = base_env._wrist_camera.data.output["rgb"]
                both_writer.append_data(_make_pov_wrist_side_by_side(pov_frame, wrist_batch, policy.image_crop_size))

            if len(dones) > 0 and torch.all(dones).item():
                completed_loops += 1
                episode_success_rate = _get_episode_success_rate(base_env)
                final_success_rate = extras.get("successes") if isinstance(extras, dict) else None
                if final_success_rate is None:
                    final_success_rate = _get_current_success_rate(base_env)

                episode_text = (
                    f"episode success rate = {_to_float(episode_success_rate):.4f}"
                    if episode_success_rate is not None
                    else "episode success rate = unavailable"
                )
                final_text = (
                    f"final-step success rate = {_to_float(final_success_rate):.4f}"
                    if final_success_rate is not None
                    else "final-step success rate = unavailable"
                )
                print(f"[INFO] Loop {completed_loops}: {episode_text}, {final_text}")
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

    if zed_writer is not None:
        zed_writer.close()
    if both_writer is not None:
        both_writer.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
