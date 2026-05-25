#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard teleop recorder for visuotactile gear-mesh demos.")
parser.add_argument("--task", type=str, default="Visuotactile-Factory-GearMesh-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is currently supported.")
parser.add_argument(
    "--log_path",
    type=str,
    required=True,
    help="Output HDF5 path. Bare filenames are written under logs/vistac_rollouts.",
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
    help="Override the fixed asset yaw randomization range in degrees.",
)
parser.add_argument(
    "--fixed_asset_height",
    action="store_true",
    default=False,
    help="Disable fixed-asset Z-position randomization while keeping XY randomization unchanged.",
)
parser.add_argument(
    "--hand_init_height",
    type=float,
    default=None,
    help="Override task.hand_init_pos[2] in meters.",
)
parser.add_argument(
    "--physics_hz",
    type=float,
    default=120.0,
    help="Physics stepping frequency. Must divide policy_hz through env decimation.",
)
parser.add_argument(
    "--policy_hz",
    type=float,
    default=15.0,
    help="Policy and logging frequency. Must divide physics_hz.",
)
parser.add_argument(
    "--ft_log_hz",
    type=float,
    default=None,
    help="Saved FT frequency. Defaults to physics_hz.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    raise ValueError("Keyboard teleop requires a visible Isaac Sim window. Do not pass --headless.")
if args_cli.num_envs != 1:
    raise ValueError("This teleop recorder currently supports --num_envs 1 only.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import carb
import gymnasium as gym
import h5py
import numpy as np
import omni.appwindow
import omni.ui as ui
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import fr3_manipulation.tasks  # noqa: F401


KEYBOARD_FORWARD_KEYS = {"PERIOD", "DOT", "GREATER", "GREATERTHAN", ">"}
KEYBOARD_BACKWARD_KEYS = {"COMMA", "LESS", "LESSTHAN", "<"}


def _resolve_log_path(log_path: str) -> str:
    if os.path.isabs(log_path) or os.path.dirname(log_path):
        return log_path
    return os.path.join("logs", "vistac_rollouts", log_path)


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


def _center_crop(image: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    height, width = image.shape[-3], image.shape[-2]
    effective_height = height - 1
    top = (effective_height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[..., top : top + crop_height, left : left + crop_width, :]


def _drop_startup_ft_substep(ft_wrench: torch.Tensor, timestep_in_episode: int) -> torch.Tensor:
    if ft_wrench.ndim != 3 or ft_wrench.shape[1] < 2 or timestep_in_episode != 0:
        return ft_wrench
    ft_wrench = ft_wrench.clone()
    shifted = ft_wrench[:, 1:, :].clone()
    ft_wrench[:, :-1, :] = shifted
    ft_wrench[:, -1, :] = ft_wrench[:, -2, :].clone()
    return ft_wrench


def _downsample_ft_wrench(ft_wrench: torch.Tensor, physics_hz: float, ft_log_hz: float) -> torch.Tensor:
    stride = physics_hz / ft_log_hz
    stride_int = int(round(stride))
    if abs(stride - stride_int) > 1.0e-6:
        raise ValueError(f"--ft_log_hz must divide --physics_hz, got {ft_log_hz} vs {physics_hz}.")
    if stride_int <= 1:
        return ft_wrench
    start_idx = stride_int - 1
    return ft_wrench[:, start_idx::stride_int, :]


def _apply_factory_init_overrides(env_cfg):
    task_cfg = getattr(env_cfg, "task", None)
    if task_cfg is None:
        return
    if args_cli.hand_init_height is not None and hasattr(task_cfg, "hand_init_pos"):
        hand_init_pos = list(task_cfg.hand_init_pos)
        hand_init_pos[2] = float(args_cli.hand_init_height)
        task_cfg.hand_init_pos = hand_init_pos
        print(f"[INFO] Hand-init height override set to {task_cfg.hand_init_pos[2]:.4f} m.")
    if args_cli.fixed_eef_init:
        task_cfg.hand_init_pos_noise = [0.0, 0.0, 0.0]
        task_cfg.hand_init_orn_noise = [0.0, 0.0, 0.0]
        if hasattr(task_cfg, "randomize_hand_init_tilt"):
            task_cfg.randomize_hand_init_tilt = False
        print("[INFO] Fixed EEF init enabled.")
    if args_cli.fixed_asset_yaw_deg is not None:
        task_cfg.fixed_asset_init_orn_deg = float(args_cli.fixed_asset_yaw_deg)
    if args_cli.fixed_asset_yaw_range_deg is not None:
        task_cfg.fixed_asset_init_orn_range_deg = float(args_cli.fixed_asset_yaw_range_deg)
    if args_cli.fixed_asset_height and hasattr(task_cfg, "fixed_asset_init_pos_noise"):
        pos_noise = list(task_cfg.fixed_asset_init_pos_noise)
        pos_noise[2] = 0.0
        task_cfg.fixed_asset_init_pos_noise = pos_noise
        print("[INFO] Fixed asset height enabled.")


class KeyboardTeleopController:
    def __init__(self):
        self._pressed_keys: set[str] = set()
        self._input_iface = None
        self._keyboard = None
        self._keyboard_sub = None
        self._install()

    def _install(self):
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("No Isaac Sim app window available for keyboard input.")
        self._keyboard = app_window.get_keyboard()
        self._input_iface = carb.input.acquire_input_interface()
        self._keyboard_sub = self._input_iface.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        print("[INFO] Keyboard teleop enabled. Focus the Isaac Sim window to capture key presses.")

    def shutdown(self):
        if self._input_iface is not None and self._keyboard is not None and self._keyboard_sub is not None:
            try:
                self._input_iface.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass
        self._keyboard = None
        self._keyboard_sub = None
        self._input_iface = None

    @staticmethod
    def _normalize_key(event) -> str:
        key_name = getattr(event.input, "name", None)
        if callable(key_name):
            key_name = key_name()
        if not key_name:
            key_name = str(event.input)
        key_id = str(key_name).upper()
        for delimiter in (".", ":", " "):
            if delimiter in key_id:
                key_id = key_id.split(delimiter)[-1]
        return key_id

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        del args, kwargs
        key_id = self._normalize_key(event)
        if int(event.type) == int(carb.input.KeyboardEventType.KEY_PRESS):
            self._pressed_keys.add(key_id)
        elif int(event.type) == int(carb.input.KeyboardEventType.KEY_RELEASE):
            self._pressed_keys.discard(key_id)
        return True

    def get_action(self, device: torch.device) -> torch.Tensor:
        action = torch.zeros((1, 6), dtype=torch.float32, device=device)
        if "UP" in self._pressed_keys or "UP_ARROW" in self._pressed_keys:
            action[:, 2] += 1.0
        if "DOWN" in self._pressed_keys or "DOWN_ARROW" in self._pressed_keys:
            action[:, 2] -= 1.0
        if "LEFT" in self._pressed_keys or "LEFT_ARROW" in self._pressed_keys:
            action[:, 1] += 1.0
        if "RIGHT" in self._pressed_keys or "RIGHT_ARROW" in self._pressed_keys:
            action[:, 1] -= 1.0
        if self._pressed_keys & KEYBOARD_FORWARD_KEYS:
            action[:, 0] += 1.0
        if self._pressed_keys & KEYBOARD_BACKWARD_KEYS:
            action[:, 0] -= 1.0
        if "Q" in self._pressed_keys:
            action[:, 3] += 1.0
        if "W" in self._pressed_keys:
            action[:, 3] -= 1.0
        if "A" in self._pressed_keys:
            action[:, 4] += 1.0
        if "S" in self._pressed_keys:
            action[:, 4] -= 1.0
        if "Z" in self._pressed_keys:
            action[:, 5] += 1.0
        if "X" in self._pressed_keys:
            action[:, 5] -= 1.0
        return torch.clamp(action, -1.0, 1.0)


class VisuotactileEpisodeRecorder:
    def __init__(self, env, log_path: str):
        self.env = env
        self.base_env = env.unwrapped
        self.log_path = _resolve_log_path(log_path)
        self.current_rows: list[dict[str, np.ndarray]] = []
        self.recording = False
        self.saved_episodes = 0
        self.next_episode_id = 0
        self.active_episode_id: int | None = None
        self.timestep_in_episode = 0
        self.rgb_crop_height = 240
        self.rgb_crop_width = 240
        self.ft_log_hz = float(args_cli.ft_log_hz) if args_cli.ft_log_hz is not None else float(args_cli.physics_hz)
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self.h5_file = h5py.File(self.log_path, "w")
        self._write_attrs()
        print(f"[INFO] Recording teleop demos to: {os.path.abspath(self.log_path)}")

    def _write_attrs(self):
        self.h5_file.attrs["task"] = args_cli.task
        self.h5_file.attrs["num_envs"] = 1
        self.h5_file.attrs["logged_env_ids"] = np.asarray([0], dtype=np.int64)
        self.h5_file.attrs["action_order"] = "dx,dy,dz,droll,dpitch,dyaw"
        self.h5_file.attrs["quat_order"] = "w,x,y,z"
        self.h5_file.attrs["ft_wrench_order"] = "fx,fy,fz,tx,ty,tz"
        self.h5_file.attrs["physics_hz"] = args_cli.physics_hz
        self.h5_file.attrs["policy_hz"] = args_cli.policy_hz
        self.h5_file.attrs["ft_log_hz"] = self.ft_log_hz
        self.h5_file.attrs["ft_samples_per_policy_step"] = int(round(self.ft_log_hz / float(args_cli.policy_hz)))
        self.h5_file.attrs["ft_layout"] = "per_policy_step_substeps"
        self.h5_file.attrs["episode_layout"] = "contiguous_per_env_episode"
        self.h5_file.attrs["wrist_rgb_resolution"] = np.asarray([self.rgb_crop_width, self.rgb_crop_height], dtype=np.int64)
        self.h5_file.attrs["side_view_rgb_resolution"] = np.asarray([self.rgb_crop_width, self.rgb_crop_height], dtype=np.int64)
        self.h5_file.attrs["replay_center_crop"] = "216x216"

    def close(self):
        self.h5_file.close()

    def start(self):
        if self.recording:
            return
        if self.active_episode_id is None:
            self.active_episode_id = self.next_episode_id
            self.timestep_in_episode = 0
        self.recording = True
        print(f"[INFO] Recording started for episode {self.active_episode_id}.")

    def delete_current(self):
        discarded = len(self.current_rows)
        self.current_rows.clear()
        self.recording = False
        if self.active_episode_id is not None:
            self.next_episode_id += 1
        self.active_episode_id = None
        self.timestep_in_episode = 0
        print(f"[INFO] Deleted current buffered recording ({discarded} step(s)).")

    def finish_current(self):
        if not self.current_rows:
            self.recording = False
            self.active_episode_id = None
            self.timestep_in_episode = 0
            return
        self.current_rows[-1]["done"][0] = True
        self.current_rows[-1]["timeout"][0] = False
        batch = {}
        for key in self.current_rows[0]:
            batch[key] = np.concatenate([row[key] for row in self.current_rows], axis=0)
        _append_h5_batch(self.h5_file, batch)
        self.h5_file.flush()
        self.saved_episodes += 1
        print(f"[INFO] Saved episode {self.active_episode_id} with {len(self.current_rows)} step(s).")
        self.current_rows.clear()
        self.recording = False
        self.next_episode_id += 1
        self.active_episode_id = None
        self.timestep_in_episode = 0

    def reset_env(self):
        if self.current_rows:
            print("[INFO] Reset requested. Discarding current unsaved recording.")
        self.current_rows.clear()
        self.recording = False
        if self.active_episode_id is not None:
            self.next_episode_id += 1
        self.active_episode_id = None
        self.timestep_in_episode = 0
        self.env.reset()

    def capture_step(self, action: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        if not self.recording or self.active_episode_id is None:
            return
        gripper_pos = torch.mean(self.base_env.joint_pos[:, 7:], dim=1, keepdim=True)
        left_ft_wrench = _drop_startup_ft_substep(self.base_env.left_ft_wrench_substeps, self.timestep_in_episode)
        right_ft_wrench = _drop_startup_ft_substep(self.base_env.right_ft_wrench_substeps, self.timestep_in_episode)
        left_ft_wrench = _downsample_ft_wrench(left_ft_wrench, args_cli.physics_hz, self.ft_log_hz)
        right_ft_wrench = _downsample_ft_wrench(right_ft_wrench, args_cli.physics_hz, self.ft_log_hz)

        wrist_rgb = self.base_env._wrist_camera.data.output["rgb"][..., :3]
        if wrist_rgb.dtype.is_floating_point:
            wrist_rgb = torch.clamp(wrist_rgb * 255.0, 0.0, 255.0).to(torch.uint8)
        else:
            wrist_rgb = wrist_rgb.to(torch.uint8)
        wrist_rgb = _center_crop(wrist_rgb, self.rgb_crop_height, self.rgb_crop_width)

        side_view_rgb = self.base_env._side_view_camera.data.output["rgb"][..., :3]
        if side_view_rgb.dtype.is_floating_point:
            side_view_rgb = torch.clamp(side_view_rgb * 255.0, 0.0, 255.0).to(torch.uint8)
        else:
            side_view_rgb = side_view_rgb.to(torch.uint8)
        side_view_rgb = _center_crop(side_view_rgb, self.rgb_crop_height, self.rgb_crop_width)

        row = {
            "env_id": np.asarray([0], dtype=np.int64),
            "episode_id": np.asarray([self.active_episode_id], dtype=np.int64),
            "timestep": np.asarray([self.timestep_in_episode], dtype=np.int64),
            "done": terminated.detach().cpu().numpy().astype(np.bool_),
            "timeout": truncated.detach().cpu().numpy().astype(np.bool_),
            "gripper_pos": gripper_pos.detach().cpu().numpy().astype(np.float32),
            "action": action.detach().cpu().numpy().astype(np.float32),
            "eef_pos": self.base_env.fingertip_midpoint_pos.detach().cpu().numpy().astype(np.float32),
            "eef_quat": self.base_env.fingertip_midpoint_quat.detach().cpu().numpy().astype(np.float32),
            "left_ft_wrench": left_ft_wrench.detach().cpu().numpy().astype(np.float32),
            "right_ft_wrench": right_ft_wrench.detach().cpu().numpy().astype(np.float32),
            "wrist_rgb": wrist_rgb.detach().cpu().numpy(),
            "side_view_rgb": side_view_rgb.detach().cpu().numpy(),
        }
        self.current_rows.append(row)
        self.timestep_in_episode += 1

        if bool(torch.any(terminated | truncated).item()):
            print("[INFO] Episode terminated while recording. Auto-saving current demo.")
            self.finish_current()


class TeleopControlWindow:
    def __init__(self, recorder: VisuotactileEpisodeRecorder):
        self.recorder = recorder
        self._status = ui.SimpleStringModel("")
        self._window = ui.Window("Visuotactile Teleop Recorder", width=420, height=260)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Gear-Mesh Visuotactile Keyboard Teleop")
                ui.Label("Move: arrows=Y/Z, comma/period=X")
                ui.Label("Rotate: Q/W=roll, A/S=pitch, Z/X=yaw")
                ui.Label("Buttons affect the current buffered demo.")
                ui.Label("", model=self._status, word_wrap=True, height=60)
                with ui.HStack(height=32):
                    ui.Button("Start Recording", clicked_fn=self.recorder.start)
                    ui.Button("Delete Current", clicked_fn=self.recorder.delete_current)
                with ui.HStack(height=32):
                    ui.Button("Finish + Save", clicked_fn=self.recorder.finish_current)
                    ui.Button("Reset Env", clicked_fn=self.recorder.reset_env)

    def refresh(self):
        steps = len(self.recorder.current_rows)
        episode = "none" if self.recorder.active_episode_id is None else str(self.recorder.active_episode_id)
        status = (
            f"recording={self.recorder.recording} | active_episode={episode} | "
            f"buffered_steps={steps} | saved_episodes={self.recorder.saved_episodes}"
        )
        self._status.set_value(status)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.sim.dt = 1.0 / float(args_cli.physics_hz)
    decimation = float(args_cli.physics_hz) / float(args_cli.policy_hz)
    decimation_int = int(round(decimation))
    if abs(decimation - decimation_int) > 1.0e-6:
        raise ValueError(f"--physics_hz must be divisible by --policy_hz, got {args_cli.physics_hz} / {args_cli.policy_hz}.")
    env_cfg.decimation = decimation_int
    _apply_factory_init_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    controller = KeyboardTeleopController()
    recorder = VisuotactileEpisodeRecorder(env, args_cli.log_path)
    control_window = TeleopControlWindow(recorder)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                action = controller.get_action(env.unwrapped.device)
                _, _, terminated, truncated, _ = env.step(action)
                recorder.capture_step(action, terminated, truncated)
                control_window.refresh()
    finally:
        controller.shutdown()
        recorder.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
