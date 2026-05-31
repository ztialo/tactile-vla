# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect rollout data, including FT by default, from an RL-Games checkpoint into HDF5."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Log rollout data from an RL-Games checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--privileged_actor",
    action="store_true",
    default=False,
    help="When using a visuomotor task, load a privileged teacher checkpoint and map compatible actor observations.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable_action_shaping",
    action="store_true",
    default=False,
    help="Disable task-space workspace clipping and upright overwrite in _apply_action().",
)
parser.add_argument(
    "--enable_action_shaping",
    action="store_true",
    default=False,
    help="Force task-space workspace clipping and upright overwrite in _apply_action().",
)
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
parser.add_argument(
    "--ft_log_hz",
    type=float,
    default=None,
    help="Saved FT frequency. Defaults to physics_hz.",
)
parser.add_argument(
    "--log_ft",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Log force-torque wrench data. Enabled by default.",
)
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


import math
import random
import time

import gymnasium as gym
import h5py
import numpy as np
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401


def _load_rl_games_agent_cfg_from_task(task_name: str, agent_entry_point: str) -> dict:
    task_spec = gym.spec(task_name)
    if agent_entry_point not in task_spec.kwargs:
        raise KeyError(f"Task '{task_name}' has no agent entry point '{agent_entry_point}'.")
    cfg_ref = task_spec.kwargs[agent_entry_point]
    if ":" not in cfg_ref:
        raise ValueError(f"Unexpected config reference '{cfg_ref}'. Expected '<module>:<relative_cfg_path>'.")
    module_name, rel_cfg_path = cfg_ref.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    module_dir = os.path.dirname(module.__file__)
    cfg_path = os.path.join(module_dir, rel_cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Loaded RL-Games config is not a dictionary: {cfg_path}")
    return cfg


def _get_checkpoint_actor_input_dim(checkpoint_path: str) -> int | None:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Failed to inspect checkpoint actor input dim from {checkpoint_path}: {exc}")
        return None

    if not isinstance(payload, dict):
        return None
    model_state_dict = payload.get("model")
    if not isinstance(model_state_dict, dict):
        return None

    rnn_input = model_state_dict.get("a2c_network.rnn.rnn.weight_ih_l0")
    if getattr(rnn_input, "ndim", None) == 2:
        return int(rnn_input.shape[1])

    running_mean = model_state_dict.get("running_mean_std.running_mean")
    if getattr(running_mean, "ndim", None) == 1:
        return int(running_mean.shape[0])

    first_actor_weight = model_state_dict.get("a2c_network.actor_mlp.0.weight")
    if getattr(first_actor_weight, "ndim", None) == 2:
        return int(first_actor_weight.shape[1])

    return None


def _configure_visuomotor_obs_groups_for_checkpoint(agent_cfg: dict, actor_input_dim: int | None):
    env_cfg = agent_cfg.setdefault("params", {}).setdefault("env", {})
    env_cfg["concate_obs_groups"] = True
    if actor_input_dim == 19:
        # Keep wrapper keys on registered groups; teacher_policy is aliased into "policy" at runtime.
        env_cfg["obs_groups"] = {"obs": ["policy"], "states": ["critic"]}
        print(
            "[INFO] Using privileged teacher-policy observations via policy alias "
            "for checkpoint compatibility (19-dim input)."
        )
    elif actor_input_dim in (51, 72):
        env_cfg["obs_groups"] = {"obs": ["critic"], "states": ["critic"]}
        print("[INFO] Using critic observations for privileged actor checkpoint compatibility.")
    elif actor_input_dim == 272:
        env_cfg["obs_groups"] = {"obs": ["policy"], "states": ["critic"]}
        print("[INFO] Using visuomotor policy observations for student checkpoint (272-dim input).")
    else:
        env_cfg["obs_groups"] = {"obs": ["policy"], "states": ["critic"]}
        print(
            "[WARN] Could not infer exact actor input dim; defaulting to policy observations. "
            "If using a privileged teacher checkpoint, pass --privileged_actor and verify actor input dim logs."
        )


def _normalize_wrapper_obs_groups(obs_groups: dict | None) -> dict[str, list[str]] | None:
    if obs_groups is None:
        return None
    if "obs" in obs_groups or "states" in obs_groups:
        return {"obs": list(obs_groups.get("obs", [])), "states": list(obs_groups.get("states", []))}
    # Compatibility with configs/scripts that use policy/critic naming.
    return {"obs": list(obs_groups.get("policy", ["policy"])), "states": list(obs_groups.get("critic", []))}


def _alias_teacher_policy_as_policy(base_env, policy_dim_hint: int | None = None):
    """Route teacher_policy through policy and patch policy observation space shape."""
    original_get_observations = base_env._get_observations

    def _patched_get_observations():
        obs_dict = original_get_observations()
        teacher = obs_dict.get("teacher_policy")
        if teacher is None:
            raise KeyError("Expected 'teacher_policy' in observations for privileged actor mode.")
        obs_dict["policy"] = teacher
        return obs_dict

    base_env._get_observations = _patched_get_observations

    # RL-Games wrapper validates groups against single_observation_space; patch policy shape to teacher dim.
    obs_space = base_env.single_observation_space
    if not hasattr(obs_space, "spaces") or not isinstance(obs_space.spaces, dict):
        raise TypeError("single_observation_space is expected to expose a mutable '.spaces' dictionary.")
    teacher_space = obs_space.spaces.get("teacher_policy")
    if policy_dim_hint is not None:
        policy_dim = int(policy_dim_hint)
    if teacher_space is not None and hasattr(teacher_space, "shape") and teacher_space.shape is not None:
        policy_dim = int(teacher_space.shape[-1])
    elif policy_dim_hint is None:
        raise KeyError(
            "Could not infer teacher_policy observation dim from observation space. "
            "Pass a policy_dim_hint from the checkpoint inspection path."
        )
    obs_space.spaces["policy"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(policy_dim,), dtype=np.float32)
    print(f"[INFO] Aliased teacher_policy -> policy with policy obs dim = {policy_dim}.")


def _parse_env_ids(spec: str, num_envs: int, device: torch.device) -> torch.Tensor:
    if spec.lower() == "all":
        return torch.arange(num_envs, device=device)
    env_ids = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not env_ids:
        raise ValueError("--log_env_ids must be 'all' or a non-empty comma-separated id list.")
    if min(env_ids) < 0 or max(env_ids) >= num_envs:
        raise ValueError(f"--log_env_ids contains an id outside [0, {num_envs - 1}].")
    return torch.tensor(env_ids, dtype=torch.long, device=device)


def _tensor_to_numpy(value: torch.Tensor, env_ids: torch.Tensor | None = None, dtype=None) -> np.ndarray:
    if env_ids is not None:
        value = value[env_ids]
    array = value.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _append_h5_batch(h5_file: h5py.File, batch: dict[str, np.ndarray]):
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
    batch_size = env_ids.shape[0]
    for idx in range(batch_size):
        env_id = int(env_ids[idx])
        row = {}
        for name, array in batch.items():
            row[name] = array[idx : idx + 1].copy()
        storage[env_id].append(row)


def _slice_batch_rows(batch: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: array[mask] for name, array in batch.items()}


def _flush_episode_rows(h5_file: h5py.File, rows: list[dict[str, np.ndarray]], force_terminal: bool = False):
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
    for key in ("time_outs", "timeouts", "truncated", "truncations"):
        if key in extras:
            value = extras[key]
            if isinstance(value, torch.Tensor):
                return value.to(device=dones.device, dtype=torch.bool)
    return torch.zeros_like(dones, dtype=torch.bool)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _center_bottom_crop(image: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    height, width = image.shape[-3], image.shape[-2]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"Requested crop ({crop_height}, {crop_width}) exceeds image size ({height}, {width})."
        )
    top = height - crop_height
    left = (width - crop_width) // 2
    return image[..., top : top + crop_height, left : left + crop_width, :]


def _drop_startup_ft_substep(ft_wrench: torch.Tensor, timestep_in_episode: torch.Tensor) -> torch.Tensor:
    if ft_wrench.ndim != 3 or ft_wrench.shape[1] < 2:
        return ft_wrench
    start_mask = timestep_in_episode == 0
    if not torch.any(start_mask):
        return ft_wrench
    ft_wrench = ft_wrench.clone()
    start_indices = start_mask.nonzero(as_tuple=False).squeeze(-1)
    ft_wrench[start_indices, :-1, :] = ft_wrench[start_indices, 1:, :].clone()
    ft_wrench[start_indices, -1, :] = ft_wrench[start_indices, -2, :].clone()
    return ft_wrench


def _downsample_ft_wrench(ft_wrench: torch.Tensor, physics_hz: float, ft_log_hz: float) -> torch.Tensor:
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
    start_idx = stride_int - 1
    return ft_wrench[:, start_idx::stride_int, :]


def _to_torch_bool_mask(dones, device: torch.device) -> torch.Tensor:
    if isinstance(dones, torch.Tensor):
        return dones.to(device=device, dtype=torch.bool)
    return torch.as_tensor(dones, device=device, dtype=torch.bool)


def _apply_factory_action_interface_overrides(env_cfg):
    task_cfg = getattr(env_cfg, "task", None)
    if task_cfg is None or not hasattr(task_cfg, "disable_action_shaping"):
        return
    if args_cli.disable_action_shaping and args_cli.enable_action_shaping:
        raise ValueError("Pass at most one of --disable_action_shaping or --enable_action_shaping.")
    if args_cli.disable_action_shaping:
        task_cfg.disable_action_shaping = True
        print("[INFO] Action shaping disabled for RL-Games rollout logging.")
    elif args_cli.enable_action_shaping:
        task_cfg.disable_action_shaping = False
        print("[INFO] Action shaping enabled for RL-Games rollout logging.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    runner_agent_cfg = agent_cfg
    if args_cli.privileged_actor and "Visuomotor-" in args_cli.task:
        teacher_task = args_cli.task.replace("Visuomotor-", "Privileged-")
        runner_agent_cfg = _load_rl_games_agent_cfg_from_task(teacher_task, args_cli.agent)
        print(f"[INFO] Privileged actor mode enabled. Loaded RL-Games config from task: {teacher_task}")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_factory_action_interface_overrides(env_cfg)
    if args_cli.no_log_images and hasattr(env_cfg, "wrist_camera"):
        env_cfg.wrist_camera = None
    elif args_cli.log_depth and hasattr(env_cfg, "wrist_camera") and env_cfg.wrist_camera is not None:
        data_types = list(getattr(env_cfg.wrist_camera, "data_types", []))
        if "distance_to_image_plane" not in data_types:
            data_types.append("distance_to_image_plane")
        env_cfg.wrist_camera.data_types = data_types
    elif hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    runner_agent_cfg["params"]["seed"] = (
        args_cli.seed if args_cli.seed is not None else runner_agent_cfg["params"]["seed"]
    )
    env_cfg.seed = runner_agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", runner_agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        run_dir = runner_agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            checkpoint_file = f"{runner_agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    actor_input_dim = None
    if args_cli.privileged_actor and "Visuomotor-" in args_cli.task:
        actor_input_dim = _get_checkpoint_actor_input_dim(resume_path)
        _configure_visuomotor_obs_groups_for_checkpoint(runner_agent_cfg, actor_input_dim)

    env_cfg.log_dir = log_dir

    rl_device = runner_agent_cfg["params"]["config"]["device"]
    clip_obs = runner_agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = runner_agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = _normalize_wrapper_obs_groups(runner_agent_cfg["params"]["env"].get("obs_groups"))
    concate_obs_groups = runner_agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # keep a direct handle for logging states/images
    base_env = env.unwrapped
    if args_cli.privileged_actor and "Visuomotor-" in args_cli.task and actor_input_dim == 19:
        _alias_teacher_policy_as_policy(base_env, policy_dim_hint=actor_input_dim)
    if (
        args_cli.log_path
        and not args_cli.no_log_images
        and getattr(base_env, "_wrist_camera", None) is None
    ):
        args_cli.no_log_images = True
        print("[INFO] No wrist camera found for this task. Falling back to state/FT-only logging.")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "vision_log"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during rollout.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner_agent_cfg["params"]["load_checkpoint"] = True
    runner_agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {runner_agent_cfg['params']['load_path']}")

    runner_agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(runner_agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    if not (0.0 <= args_cli.action_smoothing_alpha <= 1.0):
        raise ValueError("--action_smoothing_alpha must be in [0, 1].")
    if args_cli.action_scale <= 0.0:
        raise ValueError("--action_scale must be > 0.")

    dt = env.unwrapped.step_dt

    h5_file = None
    log_env_ids = None
    timestep_in_episode = None
    episode_rows = None
    total_samples_written = 0
    total_episodes_finished = 0
    total_successful_episodes_written = 0
    run_start_time = time.time()
    last_actions = None
    if args_cli.log_path:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.log_path)), exist_ok=True)
        h5_file = h5py.File(args_cli.log_path, "w")
        log_env_ids = _parse_env_ids(args_cli.log_env_ids, base_env.num_envs, base_env.device)
        timestep_in_episode = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        episode_rows = {int(env_id): [] for env_id in log_env_ids.detach().cpu().tolist()}
        if args_cli.log_success_only:
            suppress_after_success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            pending_success = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            success_tail_remaining = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
            success_tail_steps = max(int(round(args_cli.success_tail_seconds / dt)), 0)
        pre_action_wait_steps = max(int(round(args_cli.pre_action_wait_seconds / dt)), 0)
        physics_hz = 1.0 / float(base_env.physics_dt)
        policy_hz = 1.0 / float(base_env.step_dt)
        ft_log_hz = float(args_cli.ft_log_hz) if args_cli.ft_log_hz is not None else physics_hz
        rgb_crop_height = 256
        rgb_crop_width = 256
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
        h5_file.attrs["wrist_rgb_crop"] = "centerbottom_256x256"
        h5_file.attrs["action_scale"] = args_cli.action_scale
        h5_file.attrs["action_scale_z"] = args_cli.action_scale_z
        h5_file.attrs["physics_hz"] = physics_hz
        h5_file.attrs["policy_hz"] = policy_hz
        h5_file.attrs["ft_log_hz"] = ft_log_hz
        h5_file.attrs["ft_samples_per_policy_step"] = int(round(ft_log_hz / policy_hz))
        h5_file.attrs["ft_layout"] = "per_policy_step_substeps"
        h5_file.attrs["ft_wrench_order"] = "fx,fy,fz,tx,ty,tz"
        print(f"[INFO] Vision rollout HDF5 log: {os.path.abspath(args_cli.log_path)}")
        print(f"[INFO] Logging env ids: {h5_file.attrs['logged_env_ids'].tolist()}")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    timestep = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                prev_ep_succeeded = None
                if args_cli.log_success_only and hasattr(base_env, "ep_succeeded"):
                    prev_ep_succeeded = base_env.ep_succeeded.clone().to(dtype=torch.bool)

                obs_torch = agent.obs_to_torch(obs)
                raw_actions = agent.get_action(obs_torch, is_deterministic=agent.is_deterministic)
                if args_cli.action_scale != 1.0:
                    raw_actions = raw_actions * args_cli.action_scale
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
                    actions = (
                        args_cli.action_smoothing_alpha * raw_actions
                        + (1.0 - args_cli.action_smoothing_alpha) * last_actions
                    )
                if hold_mask is not None and torch.any(hold_mask):
                    actions = actions.clone()
                    actions[hold_mask] = 0.0
                last_actions = actions

                obs, _, dones, extras = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]

                # RL-Games LSTM handling: clear hidden states for terminated envs.
                if agent.is_rnn and agent.states is not None and len(dones) > 0:
                    dones_rnn_mask = _to_torch_bool_mask(dones, agent.states[0].device)
                    for s in agent.states:
                        s[:, dones_rnn_mask, :] = 0.0

                if h5_file is not None:
                    dones_mask = _to_torch_bool_mask(dones, base_env.device)
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
                    if args_cli.log_ft:
                        left_ft_wrench = getattr(base_env, "left_ft_wrench_substeps", None)
                        right_ft_wrench = getattr(base_env, "right_ft_wrench_substeps", None)
                        if left_ft_wrench is None or right_ft_wrench is None:
                            left_ft_body_idx = getattr(base_env, "left_ft_body_idx", None)
                            right_ft_body_idx = getattr(base_env, "right_ft_body_idx", None)
                            if left_ft_body_idx is not None and right_ft_body_idx is not None:
                                body_wrench = getattr(base_env._robot.data, "body_incoming_joint_wrench_b", None)
                                if body_wrench is not None:
                                    left_ft_wrench = body_wrench[:, left_ft_body_idx].unsqueeze(1)
                                    right_ft_wrench = body_wrench[:, right_ft_body_idx].unsqueeze(1)
                        if left_ft_wrench is not None and right_ft_wrench is not None:
                            left_ft_wrench = _drop_startup_ft_substep(left_ft_wrench, timestep_in_episode)
                            right_ft_wrench = _drop_startup_ft_substep(right_ft_wrench, timestep_in_episode)
                            left_ft_wrench = _downsample_ft_wrench(left_ft_wrench, physics_hz, ft_log_hz)
                            right_ft_wrench = _downsample_ft_wrench(right_ft_wrench, physics_hz, ft_log_hz)
                            batch["left_ft_wrench"] = _tensor_to_numpy(left_ft_wrench, log_env_ids, dtype=np.float32)
                            batch["right_ft_wrench"] = _tensor_to_numpy(right_ft_wrench, log_env_ids, dtype=np.float32)
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
                        wrist_rgb = _center_bottom_crop(wrist_rgb, rgb_crop_height, rgb_crop_width)
                        batch["wrist_rgb"] = _tensor_to_numpy(wrist_rgb, log_env_ids)
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
                            wrist_depth = _center_bottom_crop(wrist_depth, rgb_crop_height, rgb_crop_width).squeeze(-1)
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
                        else:
                            done_env_ids = log_env_ids[dones_mask[log_env_ids]]
                            for env_id_tensor in done_env_ids:
                                env_id = int(env_id_tensor.item())
                                total_samples_written += _flush_episode_rows(
                                    h5_file, episode_rows[env_id], force_terminal=False
                                )
                                episode_rows[env_id].clear()
                            if len(done_env_ids) > 0:
                                h5_file.flush()
                        timestep_in_episode[dones_mask] = 0

            timestep += 1
            if args_cli.video and timestep == args_cli.video_length:
                break
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

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if h5_file is not None and episode_rows is not None and not args_cli.log_success_only:
            for env_id in sorted(episode_rows):
                total_samples_written += _flush_episode_rows(h5_file, episode_rows[env_id], force_terminal=False)
                episode_rows[env_id].clear()
            h5_file.flush()
        if h5_file is not None:
            h5_file.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
