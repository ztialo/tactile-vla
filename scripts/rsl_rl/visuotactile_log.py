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
    help="Disable random EEF position/orientation initialization noise.",
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
    "--hand_init_height",
    type=float,
    default=None,
    help="Override task.hand_init_pos[2] in meters while keeping the task's existing XY hand-init offsets.",
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
    help=(
        "Path to an HDF5 file for logging visuotactile rollout data. If only a filename is provided, "
        "it is written under logs/vistac_rollouts. If omitted, logging is disabled."
    ),
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
parser.add_argument(
    "--action_scale",
    type=float,
    default=1.0,
    help="Multiply policy actions by this factor before stepping the env. Values < 1 slow the policy down.",
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
parser.add_argument(
    "--physics_hz",
    type=float,
    default=120.0,
    help="Physics stepping frequency for visuotactile logging. Policy/images/state remain at --policy_hz.",
)
parser.add_argument(
    "--policy_hz",
    type=float,
    default=15.0,
    help="Policy, image, and state logging frequency. Must divide --physics_hz.",
)
parser.add_argument(
    "--ft_log_hz",
    type=float,
    default=None,
    help=(
        "Saved FT sampling frequency. Defaults to --physics_hz. Must divide --physics_hz "
        "and be greater than or equal to --policy_hz."
    ),
)
parser.add_argument(
    "--flush_partial_episodes_on_exit",
    action="store_true",
    default=False,
    help="When stopping via max_steps/app close, flush in-progress (non-terminal) episode fragments to HDF5.",
)
parser.add_argument(
    "--hide_held_asset",
    action="store_true",
    default=False,
    help="Teleport the held asset far below each environment and keep it there. Useful for empty-gripper FT bias logs.",
)
parser.add_argument(
    "--hold_finger_position",
    action="store_true",
    default=False,
    help="Keep finger joint targets fixed at their current positions instead of commanding gripper close/open.",
)
parser.add_argument(
    "--ft_grasp_width_control",
    action="store_true",
    default=False,
    help=(
        "After the reset-time grasp fit, override gripper commands with a fixed per-finger width "
        "derived from the held-asset diameter. Intended for FR3 fingertip-FT experiments."
    ),
)
parser.add_argument(
    "--ft_grasp_width_scale",
    type=float,
    default=0.9,
    help="Scale factor for fixed-width FT grasp control: commanded_width = (asset_diameter / 2) * scale.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.hold_finger_position and args_cli.ft_grasp_width_control:
    raise ValueError("--hold_finger_position and --ft_grasp_width_control are mutually exclusive.")
if args_cli.ft_log_hz is not None:
    if args_cli.ft_log_hz < args_cli.policy_hz:
        raise ValueError("--ft_log_hz must be greater than or equal to --policy_hz.")
task_name_cli = args_cli.task or ""
is_visuo_task = ("Visuomotor" in task_name_cli) or ("Visuotactile" in task_name_cli)
# always enable cameras to record video or visuotactile rollouts
if args_cli.video or (args_cli.log_path and not args_cli.no_log_images) or is_visuo_task:
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
import omni

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


LEFT_FT_PRIM_PATH = "/World/fr3/fr3_left_ft_pad"
RIGHT_FT_PRIM_PATH = "/World/fr3/fr3_right_ft_pad"
FT_ATTR_CANDIDATES = (
    "body_incoming_joint_wrench_b",
    "state:force",
    "state:linearForce",
    "physxJoint:force",
    "physxJoint:normalForce",
)


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


def _get_episode_success_rate(env) -> float | None:
    """Return the ever-succeeded episode success rate tracked by the raw Factory env."""
    if not hasattr(env, "ep_succeeded"):
        return None
    return float(torch.count_nonzero(env.ep_succeeded).float().item() / env.num_envs)


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


def _resolve_log_path(log_path: str) -> str:
    """Place bare visuotactile log filenames under the VISTAC rollout root."""
    if os.path.isabs(log_path) or os.path.dirname(log_path):
        return log_path
    return os.path.join("logs", "vistac_rollouts", log_path)


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


def _extract_force_vector(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray([float(v) for v in value], dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size >= 6:
            return arr[:6]
        if arr.size >= 3:
            return np.concatenate([arr[:3], np.zeros(3, dtype=np.float32)], axis=0)
        return None
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        try:
            return np.asarray([float(value.x), float(value.y), float(value.z), 0.0, 0.0, 0.0], dtype=np.float32)
        except Exception:
            return None
    for attr_name in ("force", "linear", "linear_force"):
        nested = getattr(value, attr_name, None)
        if nested is not None:
            vec = _extract_force_vector(nested)
            if vec is not None:
                return vec
    for getter_name in ("GetForce", "GetLinear"):
        getter = getattr(value, getter_name, None)
        if callable(getter):
            try:
                vec = _extract_force_vector(getter())
            except Exception:
                vec = None
            if vec is not None:
                return vec
    return None


def _read_ft_wrench_from_prim(prim_path: str) -> np.ndarray | None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    for attr_name in FT_ATTR_CANDIDATES:
        try:
            attr = prim.GetAttribute(attr_name)
            if not attr or not attr.IsValid():
                continue
            vec = _extract_force_vector(attr.Get())
            if vec is not None:
                return vec
        except Exception:
            continue
    return None


def _read_ft_wrenches_from_prim_paths(num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    left = _read_ft_wrench_from_prim(LEFT_FT_PRIM_PATH)
    right = _read_ft_wrench_from_prim(RIGHT_FT_PRIM_PATH)
    if left is None or right is None:
        return None
    left_tensor = torch.from_numpy(left).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    right_tensor = torch.from_numpy(right).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    return left_tensor, right_tensor


def _drop_startup_ft_substep(ft_wrench: torch.Tensor, timestep_in_episode: torch.Tensor) -> torch.Tensor:
    """For episode-start rows, drop substep 0 by shifting substeps left and repeating the last sample."""
    if ft_wrench is None or ft_wrench.ndim != 3 or ft_wrench.shape[1] < 2:
        return ft_wrench
    startup_mask = timestep_in_episode == 0
    if not torch.any(startup_mask):
        return ft_wrench
    ft_wrench = ft_wrench.clone()
    ft_wrench[startup_mask, :-1, :] = ft_wrench[startup_mask, 1:, :]
    ft_wrench[startup_mask, -1, :] = ft_wrench[startup_mask, -2, :]
    return ft_wrench


def _downsample_ft_wrench(ft_wrench: torch.Tensor, physics_hz: float, ft_log_hz: float) -> torch.Tensor:
    """Downsample per-physics-step FT samples while keeping simulator dt unchanged."""
    if ft_wrench is None or ft_wrench.ndim != 3:
        return ft_wrench
    physics_hz = float(physics_hz)
    ft_log_hz = float(ft_log_hz)
    if ft_log_hz <= 0.0:
        raise ValueError("--ft_log_hz must be positive.")
    stride = physics_hz / ft_log_hz
    stride_int = int(round(stride))
    if abs(stride - stride_int) > 1.0e-6:
        raise ValueError(f"--ft_log_hz must divide --physics_hz, got {ft_log_hz} vs {physics_hz}.")
    if stride_int <= 1:
        return ft_wrench
    # Use the last sample in each stride-sized bin to keep a consistent causal sample timing.
    start_idx = stride_int - 1
    return ft_wrench[:, start_idx::stride_int, :]


def _resolve_ft_body_indices(base_env) -> tuple[object, int, int]:
    """Resolve fingertip link ids whose incoming fixed-joint wrench is used as FT."""
    robot = base_env.scene["robot"]
    left_candidates = ("fr3_left_ft_pad", "fr3_left_ft_base", "fr3_left_ft")
    right_candidates = ("fr3_right_ft_pad", "fr3_right_ft_base", "fr3_right_ft")
    try:
        left_name = next(name for name in left_candidates if name in robot.body_names)
        right_name = next(name for name in right_candidates if name in robot.body_names)
        left_id = robot.body_names.index(left_name)
        right_id = robot.body_names.index(right_name)
    except StopIteration as exc:
        raise ValueError(
            "Could not resolve fingertip FT bodies from "
            f"{left_candidates} and {right_candidates}. "
            f"Available bodies: {robot.body_names}"
        ) from exc
    return robot, left_id, right_id


def _hide_held_asset(base_env):
    """Move the held asset out of the workspace and zero its velocity."""
    held_asset = getattr(base_env, "_held_asset", None)
    if held_asset is None:
        return
    held_state = held_asset.data.default_root_state.clone()
    held_state[:, 0:3] = base_env.scene.env_origins
    held_state[:, 2] -= 10.0
    held_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_env.device, dtype=held_state.dtype)
    held_state[:, 7:] = 0.0
    held_asset.write_root_pose_to_sim(held_state[:, 0:7])
    held_asset.write_root_velocity_to_sim(held_state[:, 7:])
    held_asset.reset()


def _install_hold_finger_position(base_env):
    """Patch gripper control so finger targets stay at the current joint positions."""
    held_finger_pos = base_env._robot.data.joint_pos[:, 7:9].clone()
    original_generate_ctrl_signals = base_env.generate_ctrl_signals

    def _generate_ctrl_signals_hold_fingers(
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos,
    ):
        del ctrl_target_gripper_dof_pos
        return original_generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=held_finger_pos,
        )

    base_env.generate_ctrl_signals = _generate_ctrl_signals_hold_fingers
    return held_finger_pos


def _install_fixed_grasp_width_control(base_env, width_scale: float):
    """Patch gripper control to command a fixed post-grasp width derived from asset diameter."""
    asset_diameter = float(base_env.cfg_task.held_asset_cfg.diameter)
    commanded_width = float(asset_diameter * 0.5 * width_scale)
    commanded_width_tensor = torch.full(
        (base_env.num_envs, 2),
        commanded_width,
        device=base_env.device,
        dtype=base_env._robot.data.joint_pos.dtype,
    )
    original_generate_ctrl_signals = base_env.generate_ctrl_signals

    def _generate_ctrl_signals_fixed_grasp_width(
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos,
    ):
        del ctrl_target_gripper_dof_pos
        return original_generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=commanded_width_tensor,
        )

    base_env.generate_ctrl_signals = _generate_ctrl_signals_fixed_grasp_width
    return commanded_width


def _apply_factory_init_overrides(env_cfg):
    task_cfg = getattr(env_cfg, "task", None)
    if task_cfg is None:
        return

    if args_cli.hand_init_height is not None and hasattr(task_cfg, "hand_init_pos"):
        hand_init_pos = list(task_cfg.hand_init_pos)
        if len(hand_init_pos) < 3:
            raise ValueError("task.hand_init_pos must have at least 3 elements [x, y, z].")
        hand_init_pos[2] = float(args_cli.hand_init_height)
        task_cfg.hand_init_pos = hand_init_pos
        print(f"[INFO] Hand-init height override set to {task_cfg.hand_init_pos[2]:.4f} m.")

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


def _apply_multirate_timing(env_cfg):
    physics_hz = float(args_cli.physics_hz)
    policy_hz = float(args_cli.policy_hz)
    if physics_hz <= 0.0 or policy_hz <= 0.0:
        raise ValueError("--physics_hz and --policy_hz must be positive.")
    decimation = physics_hz / policy_hz
    decimation_int = int(round(decimation))
    if abs(decimation - decimation_int) > 1.0e-6:
        raise ValueError(f"--physics_hz must be divisible by --policy_hz, got {physics_hz} / {policy_hz}.")
    env_cfg.sim.dt = 1.0 / physics_hz
    env_cfg.decimation = decimation_int
    print(
        f"[INFO] Multirate timing: physics={physics_hz:.1f} Hz, "
        f"policy/log={policy_hz:.1f} Hz, decimation={decimation_int}."
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
    if args_cli.random_orn is not None and hasattr(env_cfg, "task") and hasattr(env_cfg.task, "randomize_hand_init_tilt"):
        env_cfg.task.randomize_hand_init_tilt = True
        env_cfg.task.hand_init_tilt_noise_deg = args_cli.random_orn
    _apply_factory_init_overrides(env_cfg)
    _apply_multirate_timing(env_cfg)
    if args_cli.privileged_actor:
        agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.no_log_images:
        if hasattr(env_cfg, "wrist_camera"):
            env_cfg.wrist_camera = None
        if hasattr(env_cfg, "side_view_camera"):
            env_cfg.side_view_camera = None
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
    robot, left_ft_body_idx, right_ft_body_idx = _resolve_ft_body_indices(base_env)
    if args_cli.hide_held_asset:
        _hide_held_asset(base_env)
        print("[INFO] Hidden held asset enabled: moved held asset below each environment.")

    if agent_cfg.class_name == "DistillationRunner":
        if args_cli.task is None:
            raise ValueError("DistillationRunner fallback requires a task name.")
        if args_cli.task.startswith("Visuomotor-"):
            teacher_task = args_cli.task.replace("Visuomotor-", "Privileged-", 1)
        elif args_cli.task.startswith("Visuotactile-"):
            teacher_task = args_cli.task.replace("Visuotactile-", "Privileged-", 1)
        else:
            raise ValueError(
                "DistillationRunner fallback is only supported for Visuomotor-* or Visuotactile-* teacher data collection."
            )
        teacher_agent_cfg = cli_args.parse_rsl_rl_cfg(teacher_task, args_cli)
        if args_cli.privileged_actor:
            teacher_agent_cfg.obs_groups = {"policy": ["critic"], "critic": ["critic"]}
        else:
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
    timestep_in_episode = None
    episode_id_in_env = None
    episode_rows = None
    run_start_time = time.time()
    total_samples_written = 0
    total_episodes_finished = 0
    total_successful_episodes_written = 0
    printed_ft_source_info = False
    if not (0.0 <= args_cli.action_smoothing_alpha <= 1.0):
        raise ValueError("--action_smoothing_alpha must be in [0, 1].")
    if args_cli.action_scale <= 0.0:
        raise ValueError("--action_scale must be > 0.")
    last_actions = None
    if args_cli.log_path:
        log_path = _resolve_log_path(args_cli.log_path)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        h5_file = h5py.File(log_path, "w")
        log_env_ids = _parse_env_ids(args_cli.log_env_ids, base_env.num_envs, base_env.device)
        ft_log_hz = float(args_cli.ft_log_hz) if args_cli.ft_log_hz is not None else float(args_cli.physics_hz)
        timestep_in_episode = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        episode_id_in_env = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        episode_rows = {int(env_id): [] for env_id in log_env_ids.detach().cpu().tolist()}
        if args_cli.log_success_only:
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
        h5_file.attrs["log_success_only"] = args_cli.log_success_only
        h5_file.attrs["successful_episodes_target"] = args_cli.successful_episodes_target
        h5_file.attrs["success_tail_seconds"] = args_cli.success_tail_seconds
        h5_file.attrs["pre_action_wait_seconds"] = args_cli.pre_action_wait_seconds
        h5_file.attrs["ft_wrench_order"] = "fx,fy,fz,tx,ty,tz"
        h5_file.attrs["physics_hz"] = args_cli.physics_hz
        h5_file.attrs["policy_hz"] = args_cli.policy_hz
        h5_file.attrs["ft_log_hz"] = ft_log_hz
        h5_file.attrs["ft_samples_per_policy_step"] = int(round(ft_log_hz / float(args_cli.policy_hz)))
        h5_file.attrs["ft_layout"] = "per_policy_step_substeps"
        h5_file.attrs["ft_prim_paths"] = np.asarray([LEFT_FT_PRIM_PATH, RIGHT_FT_PRIM_PATH], dtype="S")
        h5_file.attrs["episode_layout"] = "contiguous_per_env_episode"
        h5_file.attrs["wrist_rgb_resolution"] = np.asarray([rgb_crop_width, rgb_crop_height], dtype=np.int64)
        h5_file.attrs["side_view_rgb_resolution"] = np.asarray([rgb_crop_width, rgb_crop_height], dtype=np.int64)
        h5_file.attrs["replay_center_crop"] = "216x216"
        print(f"[INFO] Visuotactile rollout HDF5 log: {os.path.abspath(log_path)}")
        print(f"[INFO] Logging env ids: {h5_file.attrs['logged_env_ids'].tolist()}")

    # reset environment
    obs = env.get_observations()
    if args_cli.hold_finger_position:
        held_finger_pos = _install_hold_finger_position(base_env)
        print(
            "[INFO] Hold finger position enabled after reset-time grasp: "
            f"target={held_finger_pos[0].detach().cpu().tolist()}.",
        )
    if args_cli.ft_grasp_width_control:
        body_names = tuple(getattr(base_env._robot, "body_names", []) or [])
        ft_body_candidates = ("fr3_left_ft_pad", "fr3_right_ft_pad")
        if not all(name in body_names for name in ft_body_candidates):
            print(
                "[WARN] FT grasp-width control requested, but not all fingertip FT pad bodies were found. "
                f"Expected {ft_body_candidates}, available bodies: {body_names}",
            )
        commanded_width = _install_fixed_grasp_width_control(base_env, args_cli.ft_grasp_width_scale)
        print(
            "[INFO] Fixed FT grasp-width control enabled after reset-time grasp: "
            f"per_finger_width={commanded_width:.6f} m "
            f"(diameter={float(base_env.cfg_task.held_asset_cfg.diameter):.6f} m, "
            f"scale={float(args_cli.ft_grasp_width_scale):.3f}).",
        )
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
                if args_cli.hide_held_asset:
                    _hide_held_asset(base_env)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)

                if h5_file is not None:
                    dones_mask = dones.to(dtype=torch.bool)
                    timeout = _get_timeout_tensor(extras, dones_mask)
                    gripper_pos = torch.mean(base_env.joint_pos[:, 7:], dim=1, keepdim=True)
                    ft_from_prim_paths = _read_ft_wrenches_from_prim_paths(base_env.num_envs, base_env.device)
                    if ft_from_prim_paths is not None:
                        left_ft_wrench, right_ft_wrench = ft_from_prim_paths
                        if not printed_ft_source_info:
                            print(
                                "[INFO] FT logging source: prim paths "
                                f"('{LEFT_FT_PRIM_PATH}', '{RIGHT_FT_PRIM_PATH}').",
                            )
                            printed_ft_source_info = True
                    else:
                        left_ft_wrench = getattr(base_env, "left_ft_wrench_substeps", None)
                        right_ft_wrench = getattr(base_env, "right_ft_wrench_substeps", None)
                        if left_ft_wrench is not None and right_ft_wrench is not None and not printed_ft_source_info:
                            body_names = getattr(robot, "body_names", [])
                            left_body_name = body_names[left_ft_body_idx] if left_ft_body_idx < len(body_names) else left_ft_body_idx
                            right_body_name = body_names[right_ft_body_idx] if right_ft_body_idx < len(body_names) else right_ft_body_idx
                            print(
                                "[INFO] FT logging source: env substep body_incoming_joint_wrench_b "
                                f"(left={left_body_name}, right={right_body_name}).",
                            )
                            printed_ft_source_info = True
                    if left_ft_wrench is None or right_ft_wrench is None:
                        left_ft_wrench = robot.data.body_incoming_joint_wrench_b[:, left_ft_body_idx]
                        right_ft_wrench = robot.data.body_incoming_joint_wrench_b[:, right_ft_body_idx]
                        if not printed_ft_source_info:
                            body_names = getattr(robot, "body_names", [])
                            left_body_name = body_names[left_ft_body_idx] if left_ft_body_idx < len(body_names) else left_ft_body_idx
                            right_body_name = body_names[right_ft_body_idx] if right_ft_body_idx < len(body_names) else right_ft_body_idx
                            print(
                                "[INFO] FT logging source: robot.data.body_incoming_joint_wrench_b "
                                f"(left={left_body_name}, right={right_body_name}).",
                            )
                            printed_ft_source_info = True
                    left_ft_wrench = _drop_startup_ft_substep(left_ft_wrench, timestep_in_episode)
                    right_ft_wrench = _drop_startup_ft_substep(right_ft_wrench, timestep_in_episode)
                    left_ft_wrench = _downsample_ft_wrench(left_ft_wrench, args_cli.physics_hz, ft_log_hz)
                    right_ft_wrench = _downsample_ft_wrench(right_ft_wrench, args_cli.physics_hz, ft_log_hz)
                    env_id_batch = _tensor_to_numpy(log_env_ids, dtype=np.int64)
                    batch = {
                        "env_id": env_id_batch,
                        "episode_id": _tensor_to_numpy(episode_id_in_env, log_env_ids, dtype=np.int64),
                        "timestep": _tensor_to_numpy(timestep_in_episode, log_env_ids, dtype=np.int64),
                        "done": _tensor_to_numpy(dones_mask, log_env_ids, dtype=np.bool_),
                        "timeout": _tensor_to_numpy(timeout, log_env_ids, dtype=np.bool_),
                        "gripper_pos": _tensor_to_numpy(gripper_pos, log_env_ids, dtype=np.float32),
                        "action": _tensor_to_numpy(actions, log_env_ids, dtype=np.float32),
                        "eef_pos": _tensor_to_numpy(base_env.fingertip_midpoint_pos, log_env_ids, dtype=np.float32),
                        "eef_quat": _tensor_to_numpy(base_env.fingertip_midpoint_quat, log_env_ids, dtype=np.float32),
                        "left_ft_wrench": _tensor_to_numpy(left_ft_wrench, log_env_ids, dtype=np.float32),
                        "right_ft_wrench": _tensor_to_numpy(right_ft_wrench, log_env_ids, dtype=np.float32),
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
                        if not hasattr(base_env, "_side_view_camera"):
                            raise AttributeError(
                                "The env has no _side_view_camera. Use a visuotactile task with side view or pass --no_log_images."
                            )
                        side_view_rgb = base_env._side_view_camera.data.output["rgb"][..., :3]
                        if side_view_rgb.dtype.is_floating_point:
                            side_view_rgb = torch.clamp(side_view_rgb * 255.0, 0.0, 255.0).to(torch.uint8)
                        else:
                            side_view_rgb = side_view_rgb.to(torch.uint8)
                        side_view_rgb = _center_crop(side_view_rgb, rgb_crop_height, rgb_crop_width)
                        batch["side_view_rgb"] = _tensor_to_numpy(side_view_rgb, log_env_ids)
                    if args_cli.log_success_only:
                        active_mask = ~suppress_after_success[log_env_ids].detach().cpu().numpy()
                        if np.any(active_mask):
                            _append_episode_step(
                                episode_rows,
                                env_id_batch[active_mask],
                                _slice_batch_rows(batch, active_mask),
                            )
                    else:
                        _append_episode_step(episode_rows, env_id_batch, batch)

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
                        done_env_ids = log_env_ids[dones_mask[log_env_ids]]
                        if args_cli.log_success_only:
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
                        else:
                            for env_id_tensor in done_env_ids:
                                env_id = int(env_id_tensor.item())
                                total_samples_written += _flush_episode_rows(h5_file, episode_rows[env_id])
                                episode_rows[env_id].clear()
                            if len(done_env_ids) > 0:
                                h5_file.flush()
                        if len(done_env_ids) > 0:
                            episode_id_in_env[done_env_ids] += 1
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
                episode_success_rate = _get_episode_success_rate(base_env)
                if episode_success_rate is not None:
                    status_parts.append(f"episode_success={episode_success_rate:.4f}")
                elif "successes" in extras:
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
            if args_cli.flush_partial_episodes_on_exit and episode_rows is not None:
                partial_written = 0
                for env_id, rows in episode_rows.items():
                    if rows:
                        partial_written += _flush_episode_rows(h5_file, rows, force_terminal=False)
                        rows.clear()
                if partial_written > 0:
                    total_samples_written += partial_written
                    h5_file.flush()
                    print(f"[INFO] Flushed {partial_written} rows of partial episode data on exit.")
            h5_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
