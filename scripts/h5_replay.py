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
        "--vision",
        action="store_true",
        default=False,
        help="Replay RGB only and do not require tactile FT datasets.",
    )
    parser.add_argument(
        "--demo",
        type=int,
        nargs="+",
        default=[1],
        help="One demo index, or two indices [start end] for an inclusive concatenated replay range.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS for output MP4. Defaults to H5 playback FPS when available, otherwise 15.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Optional center-crop size applied to wrist RGB before writing the MP4.",
    )
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


def _center_crop_numpy(images: np.ndarray, crop_size: int | None) -> np.ndarray:
    """Center crop NHWC image batches to a square size."""
    if crop_size is None:
        return images
    if crop_size <= 0:
        raise ValueError(f"image_crop_size must be positive when set, got {crop_size}")
    height, width = images.shape[1:3]
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"image_crop_size={crop_size} exceeds image dimensions {(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, top : top + crop_size, left : left + crop_size, :]


def _default_fps_from_h5(h5_file: h5py.File) -> int:
    """Infer playback FPS from H5 metadata, with a conservative fallback."""
    for key in ("fps", "playback_fps", "env_fps"):
        if key in h5_file.attrs:
            value = int(h5_file.attrs[key])
            if value > 0:
                return value
    return 15


def main():
    args = _parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if len(args.demo) not in (1, 2):
        raise ValueError("--demo expects one index or two indices [start end].")
    if min(args.demo) < 1:
        raise ValueError("--demo indices must be >= 1.")

    demo_start = args.demo[0]
    demo_end = args.demo[0] if len(args.demo) == 1 else args.demo[1]
    if demo_end < demo_start:
        raise ValueError("--demo end index must be >= start index.")

    demo_label = str(demo_start) if demo_start == demo_end else f"{demo_start}_{demo_end}"
    out_dir = Path(args.output_root) / h5_path.name / demo_label
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "replay.mp4"
    plot_path = out_dir / "ft_wrench.png"

    with h5py.File(h5_path, "r") as h5_file:
        required = ["done", "wrist_rgb"]
        if not args.vision:
            required.extend(["left_ft_wrench", "right_ft_wrench"])
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise KeyError(f"H5 file is missing required datasets: {missing}")

        done = np.asarray(h5_file["done"][:], dtype=np.bool_)
        bounds = _episode_bounds(done)
        if not bounds:
            raise ValueError("No complete demos found (no done=True rows).")
        if demo_end > len(bounds):
            raise IndexError(
                f"--demo={demo_start} {demo_end} is out of range. File has {len(bounds)} complete demos."
            )

        selected_bounds = bounds[demo_start - 1 : demo_end]
        row_arrays = [np.arange(start, end + 1, dtype=np.int64) for start, end in selected_bounds]
        rows = np.concatenate(row_arrays, axis=0)
        start = int(selected_bounds[0][0])
        end = int(selected_bounds[-1][1])

        wrist_rgb = np.asarray(h5_file["wrist_rgb"][rows], dtype=np.uint8)
        wrist_rgb = _center_crop_numpy(wrist_rgb, args.img_size)
        fps = args.fps if args.fps is not None else _default_fps_from_h5(h5_file)
        left_ft = None
        right_ft = None
        axis_names = None
        if not args.vision:
            left_ft = np.asarray(h5_file["left_ft_wrench"][rows], dtype=np.float32)
            right_ft = np.asarray(h5_file["right_ft_wrench"][rows], dtype=np.float32)
            axis_names = _get_wrench_axis_names(h5_file)
        x = np.arange(wrist_rgb.shape[0], dtype=np.int64)

    with imageio.get_writer(str(video_path), fps=max(fps, 1), macro_block_size=1) as writer:
        for frame in wrist_rgb:
            writer.append_data(frame)

    if not args.vision:
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
        fig.suptitle(f"{h5_path.name} | demo {demo_label} | rows {start}-{end}", fontsize=12)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    print(f"[INFO] Wrote video: {video_path}")
    if not args.vision:
        print(f"[INFO] Wrote plot:  {plot_path}")
    print(f"[INFO] Demo rows:  {start}..{end} ({rows.shape[0]} concatenated steps)")
    print(f"[INFO] Replay FPS:  {fps}")


if __name__ == "__main__":
    main()
