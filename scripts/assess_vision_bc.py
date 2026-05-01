#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Assess an offline vision behavior cloning policy inside Isaac Lab."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Assess an offline BC policy in Isaac Lab.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to offline BC checkpoint (*.pt).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during assessment.")
parser.add_argument(
    "--video_src",
    type=str,
    default="pov",
    choices=["pov", "zed"],
    help="Video source: `pov` records the viewer perspective, `zed` records the wrist camera stream.",
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# BC assessment always needs cameras
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import imageio.v2 as imageio
import numpy as np

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401
from train_vision_bc import VisionBCPolicy


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


def _make_zed_grid(rgb_batch: torch.Tensor):
    """Tile up to 10 env wrist images into a 2x5 grid."""
    frames = [_to_uint8_rgb(rgb_batch[i, ..., :3]) for i in range(min(rgb_batch.shape[0], 10))]
    if not frames:
        raise RuntimeError("No wrist camera frames available for ZED video recording.")

    frame_h, frame_w, frame_c = frames[0].shape
    blank = torch.zeros((frame_h, frame_w, frame_c), dtype=torch.uint8).numpy()
    while len(frames) < 10:
        frames.append(blank.copy())

    top = frames[0:5]
    bottom = frames[5:10]
    return np.concatenate((np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)), axis=0)


class OfflineBCInferencePolicy:
    def __init__(self, checkpoint_path: str, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        stats = checkpoint["stats"]

        proprio_dim = len(stats["proprio_mean"])
        action_dim = len(stats["action_mean"])
        self.model = VisionBCPolicy(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            image_embed_dim=config["image_embed_dim"],
            hidden_dims=config["hidden_dims"],
            encoder_type=config.get("encoder_type", "custom_cnn"),
            freeze_encoder=config.get("freeze_encoder", False),
        ).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.device = device
        self.proprio_mean = torch.tensor(stats["proprio_mean"], dtype=torch.float32, device=device)
        self.proprio_std = torch.tensor(stats["proprio_std"], dtype=torch.float32, device=device)

    @torch.inference_mode()
    def act(self, env) -> torch.Tensor:
        if env._wrist_camera is None:
            raise RuntimeError("Offline BC policy requires wrist camera data, but no wrist camera is configured.")

        camera_rgb = env._wrist_camera.data.output["rgb"][..., :3].float() / 255.0
        gripper_pos = torch.mean(env.joint_pos[:, 7:], dim=1, keepdim=True)
        proprio = torch.cat((env.joint_pos[:, 0:7], gripper_pos, env.prev_action_obs), dim=-1)
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        return self.model(camera_rgb, proprio)


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    del agent_cfg

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # TiledCamera needs real cloned USD prims, not Fabric-only clones.
    env_cfg.scene.clone_in_fabric = False
    # Pull the viewer closer for recorded perspective videos when viewer settings are available.
    if hasattr(env_cfg, "viewer") and env_cfg.viewer is not None:
        if hasattr(env_cfg.viewer, "eye"):
            env_cfg.viewer.eye = (1.8, 1.2, 1.1)
        if hasattr(env_cfg.viewer, "lookat"):
            env_cfg.viewer.lookat = (0.0, 0.0, 0.25)

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading offline BC checkpoint from: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    if getattr(base_env, "_wrist_camera", None) is None:
        raise RuntimeError("Visuomotor environment did not create a wrist camera. Camera input is required for BC.")

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
        if args_cli.video_src == "pov":
            video_kwargs = {
                "video_folder": os.path.join(os.path.dirname(checkpoint_path), "videos", "offline_bc_assess"),
                "step_trigger": lambda step: step == 0,
                "video_length": effective_video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording POV videos during assessment.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            base_env = env.unwrapped
        else:
            zed_video_dir = os.path.join(os.path.dirname(checkpoint_path), "videos", "offline_bc_assess_zed")
            os.makedirs(zed_video_dir, exist_ok=True)
            zed_video_path = os.path.join(zed_video_dir, f"{args_cli.task.replace(':', '_')}.mp4")
            print(f"[INFO] Recording ZED wrist video to: {zed_video_path}")
            zed_writer = imageio.get_writer(zed_video_path, fps=max(int(round(1.0 / base_env.step_dt)), 1))
    else:
        zed_writer = None

    device = torch.device(args_cli.device or env_cfg.sim.device)
    policy = OfflineBCInferencePolicy(checkpoint_path, device)

    env.reset()
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
                zed_writer.append_data(_make_zed_grid(zed_batch))

            if len(dones) > 0 and torch.all(dones).item():
                completed_loops += 1
                episode_success_rate = _get_episode_success_rate(base_env)
                final_success_rate = extras.get("successes") if isinstance(extras, dict) else None
                if final_success_rate is None:
                    final_success_rate = _get_current_success_rate(base_env)

                if episode_success_rate is not None:
                    loop_success_rates.append(episode_success_rate)
                    episode_text = f"episode success rate = {_to_float(episode_success_rate):.4f}"
                else:
                    episode_text = "episode success rate = unavailable"

                if final_success_rate is not None:
                    final_text = f"final-step success rate = {_to_float(final_success_rate):.4f}"
                else:
                    final_text = "final-step success rate = unavailable"

                print(f"[INFO] Loop {completed_loops}: {episode_text}, {final_text}")

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

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
