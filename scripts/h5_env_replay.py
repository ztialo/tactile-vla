#!/usr/bin/env python3
"""Replay one logged H5 demo inside Isaac Lab with exact episode-setup restore."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay one logged H5 demo inside Isaac Lab.")
parser.add_argument("--h5", type=str, required=True, help="Path to the logged H5 file.")
parser.add_argument("--demo", type=int, default=1, help="1-indexed demo number from done-delimited episodes.")
parser.add_argument(
    "--task",
    type=str,
    default="Visuotactile-Factory-GearMesh-Direct-v0",
    help="Task name used to construct the Isaac Lab env.",
)
parser.add_argument(
    "--replay_mode",
    type=str,
    choices=("controller_targets", "executed_action", "action", "controller_effective_action"),
    default="controller_targets",
    help="Replay source. controller_targets bypasses action shaping and injects logged controller targets directly.",
)
parser.add_argument("--seed", type=int, default=None, help="Optional env seed override.")
parser.add_argument(
    "--output_root",
    type=str,
    default="logs/demo_replay_env",
    help="Where to write replay traces and summary metrics.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to instantiate. Replay currently expects 1.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs != 1:
    raise ValueError("--num_envs must be 1 for exact per-demo replay.")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401


def _episode_bounds(done: np.ndarray) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for i, flag in enumerate(done):
        if flag:
            bounds.append((start, i))
            start = i + 1
    return bounds


def _quat_angle_error_deg(q_logged: np.ndarray, q_replayed: np.ndarray) -> np.ndarray:
    dots = np.sum(q_logged * q_replayed, axis=-1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _load_demo_payload(h5_path: Path, demo_idx: int):
    with h5py.File(h5_path, "r") as h5_file:
        done = np.asarray(h5_file["done"][:], dtype=np.bool_)
        bounds = _episode_bounds(done)
        if demo_idx < 1 or demo_idx > len(bounds):
            raise IndexError(f"demo {demo_idx} is out of range for {h5_path} (found {len(bounds)} complete demos).")
        start, end = bounds[demo_idx - 1]
        rows = np.arange(start, end + 1, dtype=np.int64)

        required_setup = (
            "setup_robot_joint_pos",
            "setup_robot_joint_vel",
            "setup_ctrl_target_joint_pos",
            "setup_fixed_root_state",
            "setup_held_root_state",
            "setup_fixed_pos_obs_frame",
            "setup_init_fixed_pos_obs_noise",
            "setup_actions",
            "setup_prev_actions",
            "setup_prev_action_obs",
            "setup_controller_effective_action",
            "setup_prev_joint_pos",
            "setup_prev_fingertip_pos",
            "setup_prev_fingertip_quat",
            "setup_task_prop_gains",
            "setup_task_deriv_gains",
            "setup_actor_held_xy_offset",
            "setup_last_actor_held_xy_offset",
        )
        missing = [name for name in required_setup if name not in h5_file]
        if missing:
            raise KeyError(f"H5 file is missing required setup datasets: {missing}")
        if args_cli.replay_mode == "controller_targets":
            for name in ("controller_target_pos", "controller_target_quat", "controller_target_gripper"):
                if name not in h5_file:
                    raise KeyError(f"H5 file is missing required controller-target dataset: {name}")
        elif args_cli.replay_mode not in h5_file:
            raise KeyError(f"H5 file is missing required action dataset for replay_mode={args_cli.replay_mode!r}.")

        first_row = int(rows[0])
        setup = {
            "robot_joint_pos": np.asarray(h5_file["setup_robot_joint_pos"][first_row : first_row + 1], dtype=np.float32),
            "robot_joint_vel": np.asarray(h5_file["setup_robot_joint_vel"][first_row : first_row + 1], dtype=np.float32),
            "ctrl_target_joint_pos": np.asarray(
                h5_file["setup_ctrl_target_joint_pos"][first_row : first_row + 1], dtype=np.float32
            ),
            "fixed_root_state": np.asarray(h5_file["setup_fixed_root_state"][first_row : first_row + 1], dtype=np.float32),
            "held_root_state": np.asarray(h5_file["setup_held_root_state"][first_row : first_row + 1], dtype=np.float32),
            "fixed_pos_obs_frame": np.asarray(
                h5_file["setup_fixed_pos_obs_frame"][first_row : first_row + 1], dtype=np.float32
            ),
            "init_fixed_pos_obs_noise": np.asarray(
                h5_file["setup_init_fixed_pos_obs_noise"][first_row : first_row + 1], dtype=np.float32
            ),
            "actions": np.asarray(h5_file["setup_actions"][first_row : first_row + 1], dtype=np.float32),
            "prev_actions": np.asarray(h5_file["setup_prev_actions"][first_row : first_row + 1], dtype=np.float32),
            "prev_action_obs": np.asarray(h5_file["setup_prev_action_obs"][first_row : first_row + 1], dtype=np.float32),
            "controller_effective_action": np.asarray(
                h5_file["setup_controller_effective_action"][first_row : first_row + 1], dtype=np.float32
            ),
            "prev_joint_pos": np.asarray(h5_file["setup_prev_joint_pos"][first_row : first_row + 1], dtype=np.float32),
            "prev_fingertip_pos": np.asarray(
                h5_file["setup_prev_fingertip_pos"][first_row : first_row + 1], dtype=np.float32
            ),
            "prev_fingertip_quat": np.asarray(
                h5_file["setup_prev_fingertip_quat"][first_row : first_row + 1], dtype=np.float32
            ),
            "task_prop_gains": np.asarray(h5_file["setup_task_prop_gains"][first_row : first_row + 1], dtype=np.float32),
            "task_deriv_gains": np.asarray(
                h5_file["setup_task_deriv_gains"][first_row : first_row + 1], dtype=np.float32
            ),
            "actor_held_xy_offset": np.asarray(
                h5_file["setup_actor_held_xy_offset"][first_row : first_row + 1], dtype=np.float32
            ),
            "last_actor_held_xy_offset": np.asarray(
                h5_file["setup_last_actor_held_xy_offset"][first_row : first_row + 1], dtype=np.float32
            ),
        }
        if "setup_small_gear_root_state" in h5_file:
            setup["small_gear_root_state"] = np.asarray(
                h5_file["setup_small_gear_root_state"][first_row : first_row + 1], dtype=np.float32
            )
        if "setup_large_gear_root_state" in h5_file:
            setup["large_gear_root_state"] = np.asarray(
                h5_file["setup_large_gear_root_state"][first_row : first_row + 1], dtype=np.float32
            )

        payload = {
            "rows": rows,
            "setup": setup,
            "logged_eef_pos": np.asarray(h5_file["eef_pos"][rows], dtype=np.float32),
            "logged_eef_quat": np.asarray(h5_file["eef_quat"][rows], dtype=np.float32),
            "logged_done": np.asarray(h5_file["done"][rows], dtype=np.bool_),
            "logged_env_id": np.asarray(h5_file["env_id"][rows], dtype=np.int64),
            "logged_episode_id": np.asarray(h5_file["episode_id"][rows], dtype=np.int64),
        }
        if args_cli.replay_mode == "controller_targets":
            payload["controller_target_pos"] = np.asarray(h5_file["controller_target_pos"][rows], dtype=np.float32)
            payload["controller_target_quat"] = np.asarray(h5_file["controller_target_quat"][rows], dtype=np.float32)
            payload["controller_target_gripper"] = np.asarray(
                h5_file["controller_target_gripper"][rows], dtype=np.float32
            )
        else:
            payload["actions"] = np.asarray(h5_file[args_cli.replay_mode][rows], dtype=np.float32)
        if "controller_effective_action" in h5_file:
            payload["logged_controller_effective_action"] = np.asarray(
                h5_file["controller_effective_action"][rows], dtype=np.float32
            )
        return payload


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    del agent_cfg
    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "wrist_camera"):
        env_cfg.wrist_camera = None
    if hasattr(env_cfg, "side_view_camera"):
        env_cfg.side_view_camera = None
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False

    h5_path = Path(args_cli.h5).expanduser().resolve()
    payload = _load_demo_payload(h5_path, args_cli.demo)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    if isinstance(base_env, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        base_env = env.unwrapped

    obs, _ = env.reset()
    del obs
    env_ids = torch.tensor([0], dtype=torch.long, device=base_env.device)
    setup_tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=base_env.device) for key, value in payload["setup"].items()
    }
    base_env.restore_episode_setup(env_ids, setup_tensors)

    replayed_eef_pos = []
    replayed_eef_quat = []
    replayed_controller_effective_action = []
    replayed_done = []

    zero_action = torch.zeros((1, base_env.cfg.action_space), dtype=torch.float32, device=base_env.device)
    for step_idx in range(payload["rows"].shape[0]):
        if args_cli.replay_mode == "controller_targets":
            base_env.set_direct_ctrl_target_replay(
                torch.as_tensor(payload["controller_target_pos"][step_idx : step_idx + 1], device=base_env.device),
                torch.as_tensor(payload["controller_target_quat"][step_idx : step_idx + 1], device=base_env.device),
                torch.as_tensor(payload["controller_target_gripper"][step_idx : step_idx + 1], device=base_env.device),
                env_ids,
            )
            _, _, terminated, truncated, _ = env.step(zero_action)
        else:
            base_env.clear_direct_ctrl_target_replay(env_ids)
            action = torch.as_tensor(payload["actions"][step_idx : step_idx + 1], device=base_env.device)
            _, _, terminated, truncated, _ = env.step(action)
        replayed_eef_pos.append(base_env.fingertip_midpoint_pos[0].detach().cpu().numpy())
        replayed_eef_quat.append(base_env.fingertip_midpoint_quat[0].detach().cpu().numpy())
        replayed_controller_effective_action.append(base_env.controller_effective_action[0].detach().cpu().numpy())
        replayed_done.append(bool((terminated[0] | truncated[0]).item()))

    replayed_eef_pos = np.asarray(replayed_eef_pos, dtype=np.float32)
    replayed_eef_quat = np.asarray(replayed_eef_quat, dtype=np.float32)
    replayed_controller_effective_action = np.asarray(replayed_controller_effective_action, dtype=np.float32)
    replayed_done = np.asarray(replayed_done, dtype=np.bool_)

    pos_err = np.linalg.norm(replayed_eef_pos - payload["logged_eef_pos"], axis=-1)
    quat_err_deg = _quat_angle_error_deg(payload["logged_eef_quat"], replayed_eef_quat)
    summary = {
        "h5": str(h5_path),
        "demo": int(args_cli.demo),
        "replay_mode": args_cli.replay_mode,
        "steps": int(payload["rows"].shape[0]),
        "source_env_id": int(payload["logged_env_id"][0]),
        "source_episode_id": int(payload["logged_episode_id"][0]),
        "eef_pos_err_mean_m": float(np.mean(pos_err)),
        "eef_pos_err_max_m": float(np.max(pos_err)),
        "eef_quat_err_mean_deg": float(np.mean(quat_err_deg)),
        "eef_quat_err_max_deg": float(np.max(quat_err_deg)),
        "done_match": bool(np.array_equal(payload["logged_done"], replayed_done)),
    }

    out_dir = Path(args_cli.output_root) / h5_path.name / str(args_cli.demo) / args_cli.replay_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "replay_trace.npz",
        logged_eef_pos=payload["logged_eef_pos"],
        replayed_eef_pos=replayed_eef_pos,
        logged_eef_quat=payload["logged_eef_quat"],
        replayed_eef_quat=replayed_eef_quat,
        logged_done=payload["logged_done"],
        replayed_done=replayed_done,
        replayed_controller_effective_action=replayed_controller_effective_action,
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Replayed demo {args_cli.demo} from {h5_path.name} with mode={args_cli.replay_mode}.")
    print(f"[INFO] Source episode: env_id={summary['source_env_id']} episode_id={summary['source_episode_id']}")
    print(
        "[INFO] EEF position error: "
        f"mean={summary['eef_pos_err_mean_m']:.6f} m max={summary['eef_pos_err_max_m']:.6f} m"
    )
    print(
        "[INFO] EEF orientation error: "
        f"mean={summary['eef_quat_err_mean_deg']:.4f} deg max={summary['eef_quat_err_max_deg']:.4f} deg"
    )
    print(f"[INFO] Done flags match: {summary['done_match']}")
    print(f"[INFO] Wrote trace:   {out_dir / 'replay_trace.npz'}")
    print(f"[INFO] Wrote summary: {out_dir / 'summary.json'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
