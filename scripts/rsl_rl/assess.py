# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assess a checkpoint of an RSL-RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Assess a checkpoint of an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during assessment.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--num_loops",
    type=int,
    default=1,
    help="Number of task episode-length loops to run before stopping. Use a value <= 0 to run until closed.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or "Visuomotor" in (args_cli.task or ""):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401


def _to_float(value):
    """Convert a scalar tensor or scalar-like value to a float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _get_current_success_rate(env):
    """Compute the current Factory success rate from the raw environment state."""
    if not hasattr(env, "_get_curr_successes"):
        return None
    check_rot = getattr(env.cfg_task, "name", None) == "nut_thread"
    curr_successes = env._get_curr_successes(success_threshold=env.cfg_task.success_threshold, check_rot=check_rot)
    return torch.count_nonzero(curr_successes).float() / env.num_envs


def _get_episode_success_rate(env):
    """Compute episode success rate when the raw environment tracks ep_succeeded."""
    if not hasattr(env, "ep_succeeded"):
        return None
    return torch.count_nonzero(env.ep_succeeded).float() / env.num_envs


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Assess with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    max_assessment_steps = None
    if args_cli.num_loops > 0:
        steps_per_loop = max(base_env.max_episode_length - 1, 1)
        max_assessment_steps = args_cli.num_loops * steps_per_loop
        print(
            f"[INFO] Assessment will stop after {args_cli.num_loops} loop(s): "
            f"{max_assessment_steps} steps ({steps_per_loop} steps per loop)."
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "assess"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during assessment.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    completed_loops = 0
    loop_success_rates = []

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

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

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if loop_success_rates:
        mean_success_rate = torch.stack(loop_success_rates).mean()
        print(f"[INFO] Mean episode success rate over {len(loop_success_rates)} loop(s): {_to_float(mean_success_rate):.4f}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()