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
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
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
    help="Disable random EEF position/orientation initialization noise for data collection.",
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
    "--privileged_actor",
    action="store_true",
    default=False,
    help="Load a checkpoint whose actor consumes privileged critic observations.",
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
    "--log_depth",
    action="store_true",
    default=False,
    help="Also log wrist depth as distance_to_image_plane.",
)
parser.add_argument(
    "--progress_interval",
    type=int,
    default=100,
    help="Print logging progress every N simulator steps. Use 0 to disable periodic progress prints.",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=1.0,
    help="Multiply policy actions by this factor before stepping the env. Values < 1 slow the policy down.",
)
parser.add_argument(
    "--action_scale_z",
    type=float,
    default=1.0,
    help="Multiply only the Z-axis action (dz) by this factor before stepping the env.",
)
parser.add_argument(
    "--action_smoothing_alpha",
    type=float,
    default=1.0,
    help="Low-pass smoothing factor for actions in [0, 1]. 1.0 disables smoothing.",
)
parser.add_argument(
    "--log_success_only",
    action="store_true",
    default=False,
    help="Buffer full episodes and only write rollouts for environments that finish successfully.",
)
parser.add_argument(
    "--successful_episodes_target",
    type=int,
    default=0,
    help="Stop after writing this many successful episodes. Use 0 to disable the success-count stop condition.",
)
parser.add_argument(
    "--success_tail_seconds",
    type=float,
    default=1.0,
    help="After first success, keep logging this many additional seconds before truncating and writing the episode.",
)
parser.add_argument(
    "--pre_action_wait_seconds",
    type=float,
    default=1.0,
    help="Hold the robot still for this many seconds at the start of each episode before running the policy.",
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


def _get_checkpoint_actor_input_dim(checkpoint_path: str) -> int | None:
    """Read the actor input dimension from an RSL-RL checkpoint."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Failed to inspect checkpoint actor input dim from {checkpoint_path}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        return None

    actor_weight = model_state_dict.get("actor.0.weight")
    if actor_weight is None or getattr(actor_weight, "ndim", None) != 2:
        return None

    return int(actor_weight.shape[1])


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


def _append_episode_step(storage: dict[int, list[dict[str, np.ndarray]]], env_ids: np.ndarray, batch: dict[str, np.ndarray]):
    """Append one row per env into in-memory episode buffers."""
    batch_size = env_ids.shape[0]
    for idx in range(batch_size):
        env_id = int(env_ids[idx])
        row = {}
        for name, array in batch.items():
            row[name] = array[idx : idx + 1].copy()
        storage[env_id].append(row)


def _slice_batch_rows(batch: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Select a subset of env rows from a batch dict."""
    return {name: array[mask] for name, array in batch.items()}


def _flush_episode_rows(h5_file: h5py.File, rows: list[dict[str, np.ndarray]], force_terminal: bool = False):
    """Write a buffered episode to HDF5 as one contiguous batch."""
    if not rows:
        return 0
    episode_batch = {}
    for key in rows[0]:
        episode_batch[key] = np.concatenate([row[key] for row in rows], axis=0)
    if force_terminal:
        if "done" in episode_batch:
            episode_batch["done"][-1] = True
        if "timeout" in episode_batch:
            episode_batch["timeout"][-1] = False
    _append_h5_batch(h5_file, episode_batch)
    first_key = next(iter(episode_batch))
    return episode_batch[first_key].shape[0]


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


def _center_crop(image: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    """Crop an NHWC image tensor to a centered window while excluding the bottom artifact row."""
    height, width = image.shape[-3], image.shape[-2]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"Requested crop ({crop_height}, {crop_width}) exceeds image size ({height}, {width})."
        )
    effective_height = height - 1
    if crop_height > effective_height:
        raise ValueError(
            f"Requested crop height {crop_height} exceeds artifact-trimmed image height {effective_height}."
        )
    top = (effective_height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[..., top : top + crop_height, left : left + crop_width, :]


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


def _configure_teacher_policy_obs(task_name: str, checkpoint_path: str, base_env, teacher_agent_cfg) -> None:
    """Configure teacher actor observations, including legacy 51-dim checkpoint compatibility."""
    if "Visuomotor-" not in task_name:
        teacher_agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}
        return

    actor_input_dim = _get_checkpoint_actor_input_dim(checkpoint_path)
    if actor_input_dim == 51:
        teacher_agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}
        print(
            "[INFO] Detected legacy teacher checkpoint with 51-dim actor input; "
            "using critic observations for policy loading compatibility."
        )
        return

    teacher_agent_cfg.obs_groups = {"policy": ["teacher_policy"], "critic": ["critic"]}
    original_get_observations = base_env._get_observations

    def _patched_get_observations():
        obs_dict = original_get_observations()
        obs_dict["teacher_policy"] = _compute_teacher_policy_obs(base_env)
        return obs_dict

    base_env._get_observations = _patched_get_observations


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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
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
    if args_cli.privileged_actor:
        agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.no_log_images and hasattr(env_cfg, "wrist_camera"):
        env_cfg.wrist_camera = None
    elif args_cli.log_depth and hasattr(env_cfg, "wrist_camera") and env_cfg.wrist_camera is not None:
        data_types = list(getattr(env_cfg.wrist_camera, "data_types", []))
        if "distance_to_image_plane" not in data_types:
            data_types.append("distance_to_image_plane")
        env_cfg.wrist_camera.data_types = data_types
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
        if args_cli.privileged_actor:
            teacher_agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}
        else:
            _configure_teacher_policy_obs(args_cli.task, resume_path, base_env, teacher_agent_cfg)
        runner_cfg = teacher_agent_cfg
        runner_class = OnPolicyRunner
    elif args_cli.privileged_actor and "Visuomotor-" in args_cli.task:
        teacher_task = args_cli.task.replace("Visuomotor-", "Privileged-")
        teacher_agent_cfg = cli_args.parse_rsl_rl_cfg(teacher_task, args_cli)
        _configure_teacher_policy_obs(args_cli.task, resume_path, base_env, teacher_agent_cfg)
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
    timestep_in_episode = None
    episode_rows = None
    run_start_time = time.time()
    total_samples_written = 0
    total_episodes_finished = 0
    total_successful_episodes_written = 0
    if not (0.0 <= args_cli.action_smoothing_alpha <= 1.0):
        raise ValueError("--action_smoothing_alpha must be in [0, 1].")
    if args_cli.action_scale <= 0.0:
        raise ValueError("--action_scale must be > 0.")
    last_actions = None
    if args_cli.log_path:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.log_path)), exist_ok=True)
        h5_file = h5py.File(args_cli.log_path, "w")
        log_env_ids = _parse_env_ids(args_cli.log_env_ids, base_env.num_envs, base_env.device)
        timestep_in_episode = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        if args_cli.log_success_only:
            episode_rows = {int(env_id): [] for env_id in log_env_ids.detach().cpu().tolist()}
            suppress_after_success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            pending_success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            success_tail_remaining = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
            success_tail_steps = max(int(round(args_cli.success_tail_seconds / dt)), 0)
        pre_action_wait_steps = max(int(round(args_cli.pre_action_wait_seconds / dt)), 0)
        rgb_crop_height = 240
        rgb_crop_width = 240
        h5_file.attrs["task"] = args_cli.task
        h5_file.attrs["checkpoint"] = resume_path
        h5_file.attrs["num_envs"] = base_env.num_envs
        h5_file.attrs["logged_env_ids"] = _tensor_to_numpy(log_env_ids, dtype=np.int64)
        h5_file.attrs["action_order"] = "dx,dy,dz,droll,dpitch,dyaw"
        h5_file.attrs["quat_order"] = "w,x,y,z"
        if args_cli.log_depth:
            h5_file.attrs["depth_key"] = "distance_to_image_plane"
        h5_file.attrs["log_success_only"] = args_cli.log_success_only
        h5_file.attrs["successful_episodes_target"] = args_cli.successful_episodes_target
        h5_file.attrs["success_tail_seconds"] = args_cli.success_tail_seconds
        h5_file.attrs["pre_action_wait_seconds"] = args_cli.pre_action_wait_seconds
        h5_file.attrs["wrist_rgb_resolution"] = np.asarray([rgb_crop_width, rgb_crop_height], dtype=np.int64)
        h5_file.attrs["side_view_rgb_resolution"] = np.asarray([rgb_crop_width, rgb_crop_height], dtype=np.int64)
        h5_file.attrs["replay_center_crop"] = "216x216"
        h5_file.attrs["action_scale"] = args_cli.action_scale
        h5_file.attrs["action_scale_z"] = args_cli.action_scale_z
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
                prev_ep_succeeded = None
                if args_cli.log_success_only and hasattr(base_env, "ep_succeeded"):
                    prev_ep_succeeded = base_env.ep_succeeded.clone().to(dtype=torch.bool)
                # agent stepping
                raw_actions = policy(obs) * args_cli.action_scale
                if args_cli.action_scale_z != 1.0:
                    raw_actions = raw_actions.clone()
                    raw_actions[:, 2] *= args_cli.action_scale_z
                hold_mask = None
                if h5_file is not None:
                    hold_mask = timestep_in_episode < pre_action_wait_steps
                    if args_cli.log_success_only:
                        hold_mask = torch.logical_or(hold_mask, pending_success)
                if hold_mask is not None and torch.any(hold_mask):
                    raw_actions = raw_actions.clone()
                    raw_actions[hold_mask] = 0.0
                if last_actions is None or args_cli.action_smoothing_alpha >= 1.0:
                    actions = raw_actions
                else:
                    # First-order low-pass filter: smooth high-frequency action changes.
                    actions = (
                        args_cli.action_smoothing_alpha * raw_actions
                        + (1.0 - args_cli.action_smoothing_alpha) * last_actions
                    )
                if hold_mask is not None and torch.any(hold_mask):
                    actions = actions.clone()
                    actions[hold_mask] = 0.0
                last_actions = actions
                # env stepping
                obs, _, dones, extras = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)

                if h5_file is not None:
                    dones_mask = dones.to(dtype=torch.bool)
                    timeout = _get_timeout_tensor(extras, dones_mask)
                    gripper_pos = torch.mean(base_env.joint_pos[:, 7:], dim=1, keepdim=True)
                    env_id_batch = _tensor_to_numpy(log_env_ids, dtype=np.int64)
                    batch_size = env_id_batch.shape[0]
                    batch = {
                        "timestep": _tensor_to_numpy(timestep_in_episode, log_env_ids, dtype=np.int64),
                        "done": _tensor_to_numpy(dones_mask, log_env_ids, dtype=np.bool_),
                        "timeout": _tensor_to_numpy(timeout, log_env_ids, dtype=np.bool_),
                        "gripper_pos": _tensor_to_numpy(gripper_pos, log_env_ids, dtype=np.float32),
                        "action": _tensor_to_numpy(actions, log_env_ids, dtype=np.float32),
                        "eef_pos": _tensor_to_numpy(base_env.fingertip_midpoint_pos, log_env_ids, dtype=np.float32),
                        "eef_quat": _tensor_to_numpy(base_env.fingertip_midpoint_quat, log_env_ids, dtype=np.float32),
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
                        wrist_rgb = _center_crop(wrist_rgb, rgb_crop_height, rgb_crop_width)
                        batch["wrist_rgb"] = _tensor_to_numpy(wrist_rgb, log_env_ids)
                        if not hasattr(base_env, "_side_view_camera") or base_env._side_view_camera is None:
                            raise AttributeError(
                                "The env has no _side_view_camera. Update the visuomotor robot USD/config or pass --no_log_images."
                            )
                        side_view_rgb = base_env._side_view_camera.data.output["rgb"][..., :3]
                        if side_view_rgb.dtype.is_floating_point:
                            side_view_rgb = torch.clamp(side_view_rgb * 255.0, 0.0, 255.0).to(torch.uint8)
                        else:
                            side_view_rgb = side_view_rgb.to(torch.uint8)
                        side_view_rgb = _center_crop(side_view_rgb, rgb_crop_height, rgb_crop_width)
                        batch["side_view_rgb"] = _tensor_to_numpy(side_view_rgb, log_env_ids)
                        if args_cli.log_depth:
                            if "distance_to_image_plane" not in base_env._wrist_camera.data.output:
                                raise KeyError(
                                    "Depth logging requested, but wrist camera has no 'distance_to_image_plane' output."
                                )
                            wrist_depth = base_env._wrist_camera.data.output["distance_to_image_plane"]
                            wrist_depth = torch.nan_to_num(
                                wrist_depth,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).to(torch.float32)
                            wrist_depth = wrist_depth.unsqueeze(-1)
                            wrist_depth = _center_crop(wrist_depth, rgb_crop_height, rgb_crop_width).squeeze(-1)
                            batch["wrist_depth"] = _tensor_to_numpy(wrist_depth, log_env_ids, dtype=np.float32)
                    if args_cli.log_success_only:
                        active_mask = ~suppress_after_success[log_env_ids].detach().cpu().numpy()
                        if np.any(active_mask):
                            _append_episode_step(
                                episode_rows,
                                env_id_batch[active_mask],
                                _slice_batch_rows(batch, active_mask),
                            )
                    else:
                        _append_h5_batch(h5_file, batch)
                        h5_file.flush()
                        total_samples_written += batch_size

                    timestep_in_episode += 1
                    if args_cli.log_success_only and prev_ep_succeeded is not None:
                        curr_ep_succeeded = base_env.ep_succeeded.to(dtype=torch.bool)
                        first_success_mask = torch.logical_and(curr_ep_succeeded, torch.logical_not(prev_ep_succeeded))
                        first_success_env_ids = log_env_ids[first_success_mask[log_env_ids]]
                        for env_id_tensor in first_success_env_ids:
                            env_id = int(env_id_tensor.item())
                            if (
                                args_cli.successful_episodes_target > 0
                                and (
                                    total_successful_episodes_written
                                    + int(torch.count_nonzero(pending_success).item())
                                )
                                >= args_cli.successful_episodes_target
                            ):
                                episode_rows[env_id].clear()
                                suppress_after_success[env_id] = True
                                continue
                            pending_success[env_id] = True
                            success_tail_remaining[env_id] = success_tail_steps

                        pending_logged = pending_success[log_env_ids]
                        if torch.any(pending_logged):
                            pending_logged_env_ids = log_env_ids[pending_logged]
                            success_tail_remaining[pending_logged_env_ids] -= 1
                            ready_env_ids = pending_logged_env_ids[success_tail_remaining[pending_logged_env_ids] <= 0]
                            for env_id_tensor in ready_env_ids:
                                env_id = int(env_id_tensor.item())
                                total_samples_written += _flush_episode_rows(
                                    h5_file, episode_rows[env_id], force_terminal=True
                                )
                                total_successful_episodes_written += 1
                                episode_rows[env_id].clear()
                                pending_success[env_id] = False
                                success_tail_remaining[env_id] = 0
                                suppress_after_success[env_id] = True
                            if len(ready_env_ids) > 0:
                                h5_file.flush()
                    if torch.any(dones_mask):
                        total_episodes_finished += int(torch.count_nonzero(dones_mask).item())
                        if last_actions is not None:
                            last_actions[dones_mask] = 0.0
                        if args_cli.log_success_only:
                            done_env_ids = log_env_ids[dones_mask[log_env_ids]]
                            for env_id_tensor in done_env_ids:
                                env_id = int(env_id_tensor.item())
                                if pending_success[env_id]:
                                    total_samples_written += _flush_episode_rows(
                                        h5_file, episode_rows[env_id], force_terminal=True
                                    )
                                    total_successful_episodes_written += 1
                                episode_rows[env_id].clear()
                                pending_success[env_id] = False
                                success_tail_remaining[env_id] = 0
                            suppress_after_success[dones_mask] = False
                            h5_file.flush()
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
                if args_cli.log_success_only:
                    status_parts.append(f"successful_episodes={total_successful_episodes_written}")
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

            if (
                args_cli.log_success_only
                and args_cli.successful_episodes_target > 0
                and total_successful_episodes_written >= args_cli.successful_episodes_target
            ):
                print(f"[INFO] Reached {total_successful_episodes_written} successful episodes. Stopping collection.")
                break

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
