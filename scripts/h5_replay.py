#!/usr/bin/env python3
"""Replay wrist RGB frames from an H5 demo and plot 6-axis FT wrenches."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one demo from an H5 file and plot FT wrench data.")
    parser.add_argument("--h5", type=str, required=True, help="Path to H5 file (must contain done + wrist_rgb).")
    parser.add_argument(
        "--demo_num",
        type=int,
        default=1,
        help="1-based demo index to replay. Demo boundaries are inferred from done==True rows.",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for output MP4.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="logs/demo_replay",
        help="Output root folder. Final outputs go to <output_root>/<h5_file_name>/<demo_num>/",
    )
    return parser.parse_args()


def _episode_bounds(done: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive episode bounds from done flags."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for i, flag in enumerate(done):
        if flag:
            bounds.append((start, i))
            start = i + 1
    return bounds


def _get_wrench_axis_names(h5_file: h5py.File) -> list[str]:
    order = h5_file.attrs.get("ft_wrench_order", "fx,fy,fz,tx,ty,tz")
    if isinstance(order, bytes):
        order = order.decode("utf-8")
    names = [token.strip() for token in str(order).split(",") if token.strip()]
    if len(names) != 6:
        names = ["fx", "fy", "fz", "tx", "ty", "tz"]
    return names


def main():
    args = _parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if args.demo_num < 1:
        raise ValueError("--demo_num must be >= 1.")

    out_dir = Path(args.output_root) / h5_path.name / str(args.demo_num)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "replay.mp4"
    plot_path = out_dir / "ft_wrench.png"

    with h5py.File(h5_path, "r") as h5_file:
        required = ["done", "wrist_rgb", "left_ft_wrench", "right_ft_wrench"]
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise KeyError(f"H5 file is missing required datasets: {missing}")

        done = np.asarray(h5_file["done"][:], dtype=np.bool_)
        bounds = _episode_bounds(done)
        if not bounds:
            raise ValueError("No complete demos found (no done=True rows).")
        if args.demo_num > len(bounds):
            raise IndexError(f"--demo_num={args.demo_num} is out of range. File has {len(bounds)} complete demos.")

        start, end = bounds[args.demo_num - 1]
        rows = slice(start, end + 1)

        wrist_rgb = np.asarray(h5_file["wrist_rgb"][rows], dtype=np.uint8)
        left_ft = np.asarray(h5_file["left_ft_wrench"][rows], dtype=np.float32)
        right_ft = np.asarray(h5_file["right_ft_wrench"][rows], dtype=np.float32)
        axis_names = _get_wrench_axis_names(h5_file)
        x = np.arange(wrist_rgb.shape[0], dtype=np.int64)

    with imageio.get_writer(str(video_path), fps=max(args.fps, 1), macro_block_size=1) as writer:
        for frame in wrist_rgb:
            writer.append_data(frame)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for row in range(3):
        force_idx = row
        torque_idx = row + 3

        ax_force = axes[row, 0]
        ax_force.plot(x, left_ft[:, force_idx], color="tab:blue", linewidth=1.6, label="left")
        ax_force.plot(x, right_ft[:, force_idx], color="tab:orange", linewidth=1.6, label="right")
        ax_force.set_title(f"Force {axis_names[force_idx]}")
        ax_force.grid(alpha=0.3)
        ax_force.set_ylabel("force/torque")

        ax_torque = axes[row, 1]
        ax_torque.plot(x, left_ft[:, torque_idx], color="tab:blue", linewidth=1.6, label="left")
        ax_torque.plot(x, right_ft[:, torque_idx], color="tab:orange", linewidth=1.6, label="right")
        ax_torque.set_title(f"Torque {axis_names[torque_idx]}")
        ax_torque.grid(alpha=0.3)

    axes[2, 0].set_xlabel("step")
    axes[2, 1].set_xlabel("step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{h5_path.name} | demo {args.demo_num} | rows {start}-{end}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"[INFO] Wrote video: {video_path}")
    print(f"[INFO] Wrote plot:  {plot_path}")
    print(f"[INFO] Demo rows:  {start}..{end} ({end - start + 1} steps)")


if __name__ == "__main__":
    main()
