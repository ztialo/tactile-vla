# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
parser.add_argument(
    "--log_path",
    type=str,
    default=None,
    help="Path to an HDF5 file for logging vision rollout data. If omitted, logging is disabled.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Maximum number of simulator steps to log. Use 0 to run until the simulator closes.",
)
parser.add_argument(
    "--log_env_ids",
    type=str,
    default="all",
    help="Comma-separated env ids to log, or 'all'.",
)
parser.add_argument(
    "--no_log_images",
    action="store_true",
    default=False,
    help="Skip wrist RGB image logging and only log state/action labels.",
)
parser.add_argument(
    "--progress_interval",
    type=int,
    default=100,
    help="Print logging progress every N simulator steps. Use 0 to disable periodic progress prints.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video or vision rollouts
if args_cli.video or (args_cli.log_path and not args_cli.no_log_images) or "Visuomotor" in (args_cli.task or ""):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import h5py
import numpy as np
import os
import time
import torch
from tensordict import TensorDict

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
from isaaclab.utils import math as torch_utils
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401


def _parse_env_ids(spec: str, num_envs: int, device: torch.device) -> torch.Tensor:
    """Parse a comma-separated env id list or 'all'."""
    if spec.lower() == "all":
        return torch.arange(num_envs, device=device)
    env_ids = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not env_ids:
        raise ValueError("--log_env_ids must be 'all' or a non-empty comma-separated id list.")
    if min(env_ids) < 0 or max(env_ids) >= num_envs:
        raise ValueError(f"--log_env_ids contains an id outside [0, {num_envs - 1}].")
    return torch.tensor(env_ids, dtype=torch.long, device=device)


def _tensor_to_numpy(value: torch.Tensor, env_ids: torch.Tensor | None = None, dtype=None) -> np.ndarray:
    """Select env rows and move a tensor to CPU numpy."""
    if env_ids is not None:
        value = value[env_ids]
    array = value.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _append_h5_batch(h5_file: h5py.File, batch: dict[str, np.ndarray]):
    """Append a batch of row-major data into extendable HDF5 datasets."""
    for name, array in batch.items():
        if name not in h5_file:
            compression = "gzip" if array.ndim >= 4 else None
            h5_file.create_dataset(
                name,
                data=array,
                maxshape=(None, *array.shape[1:]),
                chunks=True,
                compression=compression,
            )
        else:
            dataset = h5_file[name]
            old_size = dataset.shape[0]
            dataset.resize(old_size + array.shape[0], axis=0)
            dataset[old_size:] = array


def _get_timeout_tensor(extras: dict, dones: torch.Tensor) -> torch.Tensor:
    """Best-effort extraction of timeout flags from RSL/IsaacLab extras."""
    for key in ("time_outs", "timeouts", "truncated", "truncations"):
        if key in extras:
            value = extras[key]
            if isinstance(value, torch.Tensor):
                return value.to(device=dones.device, dtype=torch.bool)
    return torch.zeros_like(dones, dtype=torch.bool)


def _format_duration(seconds: float) -> str:
    """Format seconds as hh:mm:ss."""
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _compute_teacher_policy_obs(base_env) -> torch.Tensor:
    """Build the privileged teacher actor observation from a visuomotor env instance."""
    noisy_fixed_pos = base_env.fixed_pos_obs_frame + base_env.init_fixed_pos_obs_noise
    prev_actions = base_env.actions.clone()
    return torch.cat(
        (
            base_env.fingertip_midpoint_pos - noisy_fixed_pos,
            base_env.fingertip_midpoint_quat,
            base_env.ee_linvel_fd,
            base_env.ee_angvel_fd,
            prev_actions,
        ),
        dim=-1,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.no_log_images and hasattr(env_cfg, "wrist_camera"):
        env_cfg.wrist_camera = None
    elif hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False

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
    base_env = env.unwrapped

    if agent_cfg.class_name == "DistillationRunner":
        if "Visuomotor-" not in args_cli.task:
            raise ValueError("DistillationRunner fallback is only supported for visuomotor teacher data collection.")

        teacher_task = args_cli.task.replace("Visuomotor-", "Privileged-")
        teacher_agent_cfg = cli_args.parse_rsl_rl_cfg(teacher_task, args_cli)
        teacher_agent_cfg.obs_groups = {"policy": ["teacher_policy"], "critic": ["critic"]}

        original_get_observations = base_env._get_observations

        def _patched_get_observations():
            obs_dict = original_get_observations()
            obs_dict["teacher_policy"] = _compute_teacher_policy_obs(base_env)
            return obs_dict

        base_env._get_observations = _patched_get_observations
        runner_cfg = teacher_agent_cfg
        runner_class = OnPolicyRunner
    else:
        runner_cfg = agent_cfg
        runner_class = OnPolicyRunner

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner = runner_class(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    base_env = env.unwrapped

    h5_file = None
    log_env_ids = None
    episode_ids = None
    timestep_in_episode = None
    run_start_time = time.time()
    total_samples_written = 0
    total_episodes_finished = 0
    if args_cli.log_path:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.log_path)), exist_ok=True)
        h5_file = h5py.File(args_cli.log_path, "w")
        log_env_ids = _parse_env_ids(args_cli.log_env_ids, base_env.num_envs, base_env.device)
        episode_ids = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        timestep_in_episode = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        h5_file.attrs["task"] = args_cli.task
        h5_file.attrs["checkpoint"] = resume_path
        h5_file.attrs["num_envs"] = base_env.num_envs
        h5_file.attrs["logged_env_ids"] = _tensor_to_numpy(log_env_ids, dtype=np.int64)
        h5_file.attrs["action_order"] = "dx,dy,dz,droll,dpitch,dyaw"
        h5_file.attrs["quat_order"] = "w,x,y,z"
        print(f"[INFO] Vision rollout HDF5 log: {os.path.abspath(args_cli.log_path)}")
        print(f"[INFO] Logging env ids: {h5_file.attrs['logged_env_ids'].tolist()}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, extras = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)

                if h5_file is not None:
                    dones_mask = dones.to(dtype=torch.bool)
                    timeout = _get_timeout_tensor(extras, dones_mask)
                    gripper_pos = torch.mean(base_env.joint_pos[:, 7:], dim=1, keepdim=True)
                    prev_action = (
                        base_env.prev_action_obs
                        if hasattr(base_env, "prev_action_obs")
                        else base_env.actions
                    )

                    env_id_batch = _tensor_to_numpy(log_env_ids, dtype=np.int64)
                    batch_size = env_id_batch.shape[0]
                    batch = {
                        "env_id": env_id_batch,
                        "global_step": np.full((batch_size,), timestep, dtype=np.int64),
                        "episode_id": _tensor_to_numpy(episode_ids, log_env_ids, dtype=np.int64),
                        "timestep_in_episode": _tensor_to_numpy(timestep_in_episode, log_env_ids, dtype=np.int64),
                        "done": _tensor_to_numpy(dones_mask, log_env_ids, dtype=np.bool_),
                        "timeout": _tensor_to_numpy(timeout, log_env_ids, dtype=np.bool_),
                        "joint_pos": _tensor_to_numpy(base_env.joint_pos[:, 0:7], log_env_ids, dtype=np.float32),
                        "gripper_pos": _tensor_to_numpy(gripper_pos, log_env_ids, dtype=np.float32),
                        "prev_action": _tensor_to_numpy(prev_action, log_env_ids, dtype=np.float32),
                        "action": _tensor_to_numpy(actions, log_env_ids, dtype=np.float32),
                        "fingertip_pos_rel_fixed": _tensor_to_numpy(
                            base_env.fingertip_midpoint_pos - base_env.fixed_pos_obs_frame,
                            log_env_ids,
                            dtype=np.float32,
                        ),
                        "held_pos_rel_fixed": _tensor_to_numpy(
                            base_env.held_pos - base_env.fixed_pos_obs_frame,
                            log_env_ids,
                            dtype=np.float32,
                        ),
                    }
                    if not args_cli.no_log_images:
                        if not hasattr(base_env, "_wrist_camera"):
                            raise AttributeError(
                                "The env has no _wrist_camera. Use a visuomotor task or pass --no_log_images."
                            )
                        wrist_rgb = base_env._wrist_camera.data.output["rgb"][..., :3]
                        if wrist_rgb.dtype.is_floating_point:
                            wrist_rgb = torch.clamp(wrist_rgb * 255.0, 0.0, 255.0).to(torch.uint8)
                        else:
                            wrist_rgb = wrist_rgb.to(torch.uint8)
                        batch["wrist_rgb"] = _tensor_to_numpy(wrist_rgb, log_env_ids)
                    _append_h5_batch(h5_file, batch)
                    h5_file.flush()
                    total_samples_written += batch_size

                    timestep_in_episode += 1
                    if torch.any(dones_mask):
                        total_episodes_finished += int(torch.count_nonzero(dones_mask).item())
                        episode_ids[dones_mask] += 1
                        timestep_in_episode[dones_mask] = 0

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            else:
                timestep += 1
            if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                break

            if args_cli.progress_interval > 0 and timestep % args_cli.progress_interval == 0:
                elapsed = time.time() - run_start_time
                rate = timestep / elapsed if elapsed > 0 else 0.0
                status_parts = [
                    f"step={timestep}",
                    f"samples={total_samples_written}",
                    f"episodes={total_episodes_finished}",
                    f"elapsed={_format_duration(elapsed)}",
                ]
                if "successes" in extras:
                    success_value = extras["successes"]
                    if isinstance(success_value, torch.Tensor):
                        success_value = float(success_value.item())
                    status_parts.append(f"success={success_value:.4f}")
                if args_cli.max_steps > 0 and rate > 0:
                    remaining_steps = args_cli.max_steps - timestep
                    eta = remaining_steps / rate
                    status_parts.append(f"eta={_format_duration(eta)}")
                print("[INFO] " + " | ".join(status_parts))

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if h5_file is not None:
            h5_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
