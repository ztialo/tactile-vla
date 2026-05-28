# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--fixed_eef_init",
    action="store_true",
    default=False,
    help="Disable random EEF position/orientation initialization noise for reset.",
)
parser.add_argument(
    "--fixed_asset_height",
    action="store_true",
    default=False,
    help="Disable fixed-asset Z-position randomization while keeping XY position randomization unchanged.",
)
parser.add_argument(
    "--reset_interval_steps",
    type=int,
    default=0,
    help="Force a full env reset every N control steps. Use 0 to disable periodic resets.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after N control steps. Use 0 to run until the simulator closes.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import fr3_manipulation.tasks  # noqa: F401


def _apply_task_overrides(env_cfg):
    task_cfg = getattr(env_cfg, "task", None)
    if task_cfg is None:
        return

    if args_cli.fixed_eef_init:
        task_cfg.hand_init_pos_noise = [0.0, 0.0, 0.0]
        task_cfg.hand_init_orn_noise = [0.0, 0.0, 0.0]
        if hasattr(task_cfg, "randomize_hand_init_tilt"):
            task_cfg.randomize_hand_init_tilt = False
        print("[INFO]: Fixed EEF init enabled: zeroed hand init position/orientation noise.")

    if args_cli.fixed_asset_height and hasattr(task_cfg, "fixed_asset_init_pos_noise"):
        pos_noise = list(task_cfg.fixed_asset_init_pos_noise)
        if len(pos_noise) < 3:
            raise ValueError("task.fixed_asset_init_pos_noise must have at least 3 elements [x, y, z].")
        pos_noise[2] = 0.0
        task_cfg.fixed_asset_init_pos_noise = pos_noise
        print("[INFO]: Fixed asset height enabled: zeroed fixed-asset Z-position randomization noise.")


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "eye"):
        env_cfg.viewer.eye = (0.6, 0.6, 0.6)
    _apply_task_overrides(env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                print(f"[INFO]: Reached max_steps={args_cli.max_steps}. Stopping.")
                break
            if args_cli.reset_interval_steps > 0 and step_count > 0 and step_count % args_cli.reset_interval_steps == 0:
                print(f"[INFO]: Resetting environment at step {step_count}.")
                env.reset()
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            step_count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
