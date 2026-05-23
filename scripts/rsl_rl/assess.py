# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assess a checkpoint of an RSL-RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Assess a checkpoint of an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during assessment.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of the recorded video (in steps). Defaults to the assessed rollout length when unset.",
)
parser.add_argument(
    "--num_loops",
    type=int,
    default=1,
    help="Number of task episode-length loops to run before stopping. Use a value <= 0 to run until closed.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override task episode length in seconds for assessment.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
    "--privileged_actor",
    action="store_true",
    default=False,
    help="Load a checkpoint whose actor consumes privileged critic observations.",
)
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
import numpy as np
import torch
import isaacsim.core.utils.torch as torch_utils
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


def _set_default_factory_video_view(env_cfg, task_name: str | None):
    """Place the default viewer in front of env_0 for Factory assessment videos."""
    if "Factory" not in (task_name or ""):
        return
    if not hasattr(env_cfg, "viewer") or env_cfg.viewer is None:
        return
    if hasattr(env_cfg.viewer, "eye"):
        env_cfg.viewer.eye = (1.4, -0.015, 0.28)
    if hasattr(env_cfg.viewer, "lookat"):
        env_cfg.viewer.lookat = (0.60, 0.00, 0.12)


def _to_float(value):
    """Convert a scalar tensor or scalar-like value to a float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


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


def _override_actor_target_xy_noise_for_eval(env_cfg):
    """Use fixed actor-side XY target noise for student-style evaluation/logging."""
    curriculum_cfg = getattr(env_cfg, "actor_target_perturb_curriculum", None)
    if curriculum_cfg is None or args_cli.privileged_actor:
        return
    curriculum_cfg.enabled = True
    curriculum_cfg.start_xy_noise_m = 0.01
    curriculum_cfg.end_xy_noise_m = 0.01
    curriculum_cfg.total_steps = 1
    print("[INFO] Actor target XY noise override enabled: uniform reset noise in x,y within +/- 10.0 mm.")


def _apply_episode_length_override(env_cfg):
    """Override episode length for assessment when requested."""
    if args_cli.episode_length_s is None:
        return
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    print(f"[INFO] Episode length override set to {env_cfg.episode_length_s:.2f} s.")


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


def _print_env0_eef_euler(env, label: str):
    """Print env_0 end-effector Euler offsets from nominal in degrees."""
    if not hasattr(env, "fingertip_midpoint_quat"):
        return
    quat = env.fingertip_midpoint_quat[0:1]
    roll, pitch, yaw = torch_utils.get_euler_xyz(quat)
    rpy_deg = np.rad2deg(torch.stack((roll, pitch, yaw), dim=-1)[0].detach().cpu().numpy())
    nominal_deg = np.rad2deg(np.asarray(env.cfg_task.hand_init_orn, dtype=np.float32))
    delta_deg = (rpy_deg - nominal_deg + 180.0) % 360.0 - 180.0
    print(
        f"[INFO] {label} env0 EEF delta from nominal (deg): "
        f"roll={delta_deg[0]:+.2f}, pitch={delta_deg[1]:+.2f}, yaw={delta_deg[2]:+.2f}"
    )


def _print_env0_target_pos(env, label: str):
    """Print env_0 target info and actor-side sampled target noise when available."""
    if not hasattr(env, "fixed_pos_obs_frame"):
        return
    exact_target = env.fixed_pos_obs_frame[0].detach().cpu().numpy()
    if hasattr(env, "actor_held_xy_offset") and not args_cli.privileged_actor:
        actor_xy_noise = (
            env.last_actor_held_xy_offset[0].detach().cpu().numpy()
            if hasattr(env, "last_actor_held_xy_offset")
            else env.actor_held_xy_offset[0].detach().cpu().numpy()
        )
        actor_noise_level_mm = float(getattr(env, "current_actor_xy_noise", 0.0)) * 1000.0
        print(
            f"[INFO] {label} env0 target pos exact={np.array2string(exact_target, precision=6, separator=', ')} "
            f"actor_xy_offset={np.array2string(actor_xy_noise, precision=6, separator=', ')} "
            f"actor_xy_noise_level_mm=+/- {actor_noise_level_mm:.1f}"
        )
        return

    noise = np.zeros_like(exact_target)
    noisy_target = exact_target.copy()
    if hasattr(env, "init_fixed_pos_obs_noise"):
        noise = env.init_fixed_pos_obs_noise[0].detach().cpu().numpy()
        noisy_target = noisy_target + noise
    print(
        f"[INFO] {label} env0 target pos exact={np.array2string(exact_target, precision=6, separator=', ')} "
        f"noisy={np.array2string(noisy_target, precision=6, separator=', ')} "
        f"noise={np.array2string(noise, precision=6, separator=', ')}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Assess with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.random_orn is not None and hasattr(env_cfg, "task") and hasattr(env_cfg.task, "randomize_hand_init_tilt"):
        env_cfg.task.randomize_hand_init_tilt = True
        env_cfg.task.hand_init_tilt_noise_deg = args_cli.random_orn
    _apply_factory_init_overrides(env_cfg)
    _override_actor_target_xy_noise_for_eval(env_cfg)
    _apply_episode_length_override(env_cfg)
    if args_cli.privileged_actor:
        agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.video:
        _set_default_factory_video_view(env_cfg, args_cli.task)

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
    # TiledCamera sensors need real cloned USD prims. Fabric-only cloning breaks multi-env camera indexing.
    if hasattr(env_cfg, "wrist_camera") and getattr(env_cfg, "wrist_camera", None) is not None:
        env_cfg.scene.clone_in_fabric = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    if hasattr(base_env, "current_actor_xy_noise") and not args_cli.privileged_actor:
        print(f"[INFO] Actor target XY noise level before sim start: +/- {float(base_env.current_actor_xy_noise) * 1000.0:.1f} mm.")
    max_assessment_steps = None
    if args_cli.num_loops > 0:
        steps_per_loop = max(base_env.max_episode_length - 1, 1)
        max_assessment_steps = args_cli.num_loops * steps_per_loop
        print(
            f"[INFO] Assessment will stop after {args_cli.num_loops} loop(s): "
            f"{max_assessment_steps} steps ({steps_per_loop} steps per loop)."
        )
    else:
        steps_per_loop = max(base_env.max_episode_length - 1, 1)

    # wrap for video recording
    if args_cli.video:
        if args_cli.video_length is None:
            video_length = max_assessment_steps if max_assessment_steps is not None else steps_per_loop
        else:
            video_length = args_cli.video_length
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "assess"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
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
    _print_env0_eef_euler(base_env, "Reset")
    _print_env0_target_pos(base_env, "Reset")
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
                _print_env0_eef_euler(base_env, f"Loop {completed_loops} end")
                _print_env0_target_pos(base_env, f"Loop {completed_loops} end")

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
