#!/usr/bin/env python3
"""Replay demo RGB frames from an H5 file and optionally plot 6-axis FT wrenches."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one demo from an H5 file and plot FT wrench data.")
    parser.add_argument(
        "--h5", type=str, required=True, help="Path to H5 file (must contain done + wrist_rgb, and usually side_view_rgb)."
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        default=True,
        help="Replay RGB only and do not require tactile FT datasets. Uses both side-view and wrist RGB when available.",
    )
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="Replay wrist RGB with left/right FT wrench plots. Requires left_ft_wrench and right_ft_wrench datasets.",
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
        "--res",
        type=str,
        choices=("raw", "crop", "both"),
        default="raw",
        help="Replay resolution mode. `raw` writes the stored image, `crop` writes the center-cropped image, `both` writes both videos.",
    )
    parser.add_argument(
        "--square_wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center-crop wrist frames to a square using min(height, width). Applied before --img_size.",
    )
    parser.add_argument(
        "--sync_height",
        type=int,
        default=720,
        help="Optional output height for the side-by-side sync video. Upscales/downscales both panels.",
    )
    parser.add_argument(
        "--ft_ma_window",
        type=int,
        default=10,
        help="Moving-average window for FT wrench overlay when --filter ma. 1 disables smoothing.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=("ma", "lfilter"),
        default="ma",
        help="FT smoothing filter. 'ma' uses the current moving average, 'lfilter' uses scipy.signal.lfilter.",
    )
    parser.add_argument(
        "--lfilter_order",
        type=int,
        default=2,
        help="Butterworth low-pass order when --filter lfilter.",
    )
    parser.add_argument(
        "--lfilter_cutoff_hz",
        type=float,
        default=8.0,
        help="Butterworth low-pass cutoff frequency in Hz when --filter lfilter.",
    )
    parser.add_argument(
        "--allow_partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat a file with no done=True rows as one partial demo. Useful for fixed-duration bias logs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="logs/demo_replay",
        help="Output root folder. Final outputs go to <output_root>/<h5_file_name>/<demo_num>/",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        default=False,
        help="Disable all MP4 generation, including replay, crop/raw replay, and FT sync videos.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=("left", "right", "both"),
        default=None,
        help=(
            "Save extra 2x1 FT wrench plots: top fx/fy/fz, bottom tx/ty/tz. "
            "'both' saves separate left and right figures."
        ),
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


def _center_square_numpy(images: np.ndarray) -> np.ndarray:
    """Center crop NHWC image batches to square using min(height, width)."""
    height, width = images.shape[1:3]
    crop_size = min(height, width)
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


def _default_crop_size_from_h5(h5_file: h5py.File) -> int | None:
    """Infer default replay crop size from H5 metadata."""
    value = h5_file.attrs.get("replay_center_crop", None)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).lower().strip()
    if "x" in text:
        first, second = text.split("x", 1)
        if first == second and first.isdigit():
            return int(first)
    if text.isdigit():
        return int(text)
    return None


def _resize_image_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an HWC uint8 image with nearest-neighbor sampling."""
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid resize target: {(target_h, target_w)}")
    src_h, src_w = image.shape[:2]
    if src_h == target_h and src_w == target_w:
        return image
    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int64)
    return image[y_idx][:, x_idx]


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to target height preserving aspect ratio."""
    if target_h <= 0:
        raise ValueError(f"sync_height must be positive, got {target_h}")
    src_h, src_w = image.shape[:2]
    target_w = max(int(round(src_w * target_h / src_h)), 1)
    return _resize_image_nearest(image, target_h, target_w)


def _concat_views(left: np.ndarray, right: np.ndarray, target_h: int | None = None) -> np.ndarray:
    """Concatenate two HWC RGB frames side-by-side, resizing to a common height when needed."""
    if target_h is not None:
        left = _resize_to_height(left, target_h)
        right = _resize_to_height(right, target_h)
    elif left.shape[0] != right.shape[0]:
        right = _resize_to_height(right, left.shape[0])
    return np.concatenate((left, right), axis=1)


def _moving_average_ft(values: np.ndarray, window: int) -> np.ndarray:
    """Apply edge-padded moving-average filter to [T, 6] FT wrench values."""
    if window <= 1:
        return values.copy()
    if window < 0:
        raise ValueError(f"ft_ma_window must be >= 1, got {window}")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
    filtered = np.empty_like(values, dtype=np.float32)
    for axis_idx in range(values.shape[1]):
        filtered[:, axis_idx] = np.convolve(padded[:, axis_idx], kernel, mode="valid")
    return filtered


def _lfilter_ft(values: np.ndarray, order: int, cutoff_hz: float, sample_hz: float) -> np.ndarray:
    """Apply a causal Butterworth low-pass filter using scipy.signal.lfilter."""
    if order <= 0:
        raise ValueError(f"lfilter_order must be >= 1, got {order}")
    if cutoff_hz <= 0.0:
        raise ValueError(f"lfilter_cutoff_hz must be > 0, got {cutoff_hz}")
    if sample_hz <= 0.0:
        raise ValueError(f"sample_hz must be > 0 for --filter lfilter, got {sample_hz}")
    try:
        from scipy.signal import butter, lfilter
    except Exception as exc:
        raise RuntimeError("scipy is required for --filter lfilter.") from exc

    nyquist_hz = 0.5 * sample_hz
    normalized_cutoff = cutoff_hz / nyquist_hz
    normalized_cutoff = float(np.clip(normalized_cutoff, 1e-6, 0.999))

    filtered_input = values.copy()
    tx_axis_idx = 3
    filtered_input[:, tx_axis_idx] = np.clip(filtered_input[:, tx_axis_idx], -0.12, 0.12)

    b, a = butter(order, normalized_cutoff, btype="low")
    filtered = lfilter(b, a, filtered_input, axis=0)
    return np.asarray(filtered, dtype=np.float32)


def _filter_ft(
    values: np.ndarray,
    filter_name: str,
    ma_window: int,
    lfilter_order: int,
    lfilter_cutoff_hz: float,
    sample_hz: float | None = None,
) -> np.ndarray:
    """Apply the requested FT smoothing filter."""
    if filter_name == "ma":
        return _moving_average_ft(values, ma_window)
    if filter_name == "lfilter":
        return _lfilter_ft(values, lfilter_order, lfilter_cutoff_hz, float(sample_hz or 0.0))
    raise ValueError(f"Unsupported filter '{filter_name}'.")


def _flatten_multirate_ft(values: np.ndarray) -> tuple[np.ndarray, int]:
    if values.ndim == 3:
        return values.reshape(values.shape[0] * values.shape[1], values.shape[2]), int(values.shape[1])
    return values, 1


def _expand_boundaries_for_ft(episode_boundaries: list[int], samples_per_step: int) -> list[int]:
    return [int(boundary) * samples_per_step for boundary in episode_boundaries]


def _figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render a Matplotlib figure to an HWC RGB uint8 image."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[..., :3].copy()


def _pad_to_even_hw(image: np.ndarray) -> np.ndarray:
    """Pad an HWC RGB image to even height/width for yuv420p encoders (e.g., libx264)."""
    height, width = image.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return image
    out = np.zeros((height + pad_h, width + pad_w, image.shape[2]), dtype=image.dtype)
    out[:height, :width] = image
    if pad_w:
        out[:height, width:] = image[:height, width - 1 : width]
    if pad_h:
        out[height:, :width] = image[height - 1 : height, :width]
    if pad_h and pad_w:
        out[height:, width:] = image[height - 1 : height, width - 1 : width]
    return out


def _make_ft_plot_renderer(
    left_ft: np.ndarray,
    right_ft: np.ndarray,
    left_ft_filtered: np.ndarray,
    right_ft_filtered: np.ndarray,
    axis_names: list[str],
    title: str,
    episode_boundaries: list[int] | None = None,
) -> tuple[callable, callable]:
    """Create a renderer that returns an FT plot frame for each timestep."""
    num_steps = left_ft.shape[0]
    x = np.arange(num_steps, dtype=np.int64)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=12)

    line_specs: list[tuple[object, object, object, object, int]] = []
    cursors: list[object] = []
    for row in range(3):
        force_idx = row
        torque_idx = row + 3
        for col, axis_idx, prefix in ((0, force_idx, "Force"), (1, torque_idx, "Torque")):
            ax = axes[row, col]
            left_raw_line, = ax.plot([], [], color="tab:blue", linewidth=1.3, alpha=0.45, label="left raw")
            right_raw_line, = ax.plot([], [], color="tab:orange", linewidth=1.3, alpha=0.45, label="right raw")
            left_filt_line, = ax.plot([], [], color="tab:blue", linewidth=1.8, linestyle="--", label="left filtered")
            right_filt_line, = ax.plot(
                [], [], color="tab:orange", linewidth=1.8, linestyle="--", label="right filtered"
            )
            combined = np.concatenate(
                (
                    left_ft[:, axis_idx],
                    right_ft[:, axis_idx],
                    left_ft_filtered[:, axis_idx],
                    right_ft_filtered[:, axis_idx],
                ),
                axis=0,
            )
            value_min = float(np.nanmin(combined))
            value_max = float(np.nanmax(combined))
            if np.isclose(value_min, value_max):
                padding = 1.0 if np.isclose(value_min, 0.0) else 0.1 * abs(value_min)
            else:
                padding = 0.08 * (value_max - value_min)
            ax.set_xlim(0, max(num_steps - 1, 1))
            ax.set_ylim(value_min - padding, value_max + padding)
            ax.set_title(f"{prefix} {axis_names[axis_idx]}")
            ax.set_ylabel("force/torque")
            ax.grid(alpha=0.3)
            for boundary in episode_boundaries or []:
                ax.axvline(boundary - 0.5, color="0.25", linestyle="-", linewidth=1.1, alpha=0.55)
            cursor = ax.axvline(0, color="k", linestyle="--", linewidth=0.9, alpha=0.75)
            cursors.append(cursor)
            line_specs.append((left_raw_line, right_raw_line, left_filt_line, right_filt_line, axis_idx))

    axes[2, 0].set_xlabel("step")
    axes[2, 1].set_xlabel("step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    def _render(step_idx: int) -> np.ndarray:
        end = step_idx + 1
        for left_raw, right_raw, left_filt, right_filt, axis_idx in line_specs:
            left_raw.set_data(x[:end], left_ft[:end, axis_idx])
            right_raw.set_data(x[:end], right_ft[:end, axis_idx])
            left_filt.set_data(x[:end], left_ft_filtered[:end, axis_idx])
            right_filt.set_data(x[:end], right_ft_filtered[:end, axis_idx])
        for cursor in cursors:
            cursor.set_xdata([step_idx, step_idx])
        return _figure_to_rgb(fig)

    return _render, fig


def _save_side_ft_plot(
    side: str,
    ft_wrench: np.ndarray,
    ft_wrench_filtered: np.ndarray,
    axis_names: list[str],
    title: str,
    output_path: Path,
    sample_hz: float | None = None,
):
    """Save a single-side 2x1 wrench plot."""
    if ft_wrench.ndim != 2 or ft_wrench.shape[1] != 6:
        raise ValueError(f"Expected [N, 6] FT wrench for {side}, got {ft_wrench.shape}.")
    if ft_wrench_filtered.shape != ft_wrench.shape:
        raise ValueError(
            f"Filtered FT wrench shape {ft_wrench_filtered.shape} does not match raw shape {ft_wrench.shape}."
        )
    if sample_hz is not None and sample_hz > 0.0:
        x = np.arange(ft_wrench.shape[0], dtype=np.float32) / float(sample_hz)
        x_label = "time (s)"
    else:
        x = np.arange(ft_wrench.shape[0], dtype=np.int64)
        x_label = "sample"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=12)

    axes[0].plot(x, ft_wrench[:, 0], color="tab:blue", linewidth=1.2, alpha=0.25, label=f"{axis_names[0]} raw")
    axes[0].plot(
        x,
        ft_wrench_filtered[:, 0],
        color="tab:blue",
        linewidth=1.8,
        linestyle="--",
        label=f"{axis_names[0]} filtered",
    )
    axes[0].plot(x, ft_wrench[:, 1], color="tab:orange", linewidth=1.2, alpha=0.25, label=f"{axis_names[1]} raw")
    axes[0].plot(
        x,
        ft_wrench_filtered[:, 1],
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        label=f"{axis_names[1]} filtered",
    )
    axes[0].plot(x, ft_wrench[:, 2], color="tab:green", linewidth=1.2, alpha=0.25, label=f"{axis_names[2]} raw")
    axes[0].plot(
        x,
        ft_wrench_filtered[:, 2],
        color="tab:green",
        linewidth=1.8,
        linestyle="--",
        label=f"{axis_names[2]} filtered",
    )
    axes[0].set_title(f"{side.capitalize()} force")
    axes[0].set_ylabel("force")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(x, ft_wrench[:, 3], color="tab:red", linewidth=1.2, alpha=0.25, label=f"{axis_names[3]} raw")
    axes[1].plot(
        x,
        ft_wrench_filtered[:, 3],
        color="tab:red",
        linewidth=1.8,
        linestyle="--",
        label=f"{axis_names[3]} filtered",
    )
    axes[1].plot(x, ft_wrench[:, 4], color="tab:purple", linewidth=1.2, alpha=0.25, label=f"{axis_names[4]} raw")
    axes[1].plot(
        x,
        ft_wrench_filtered[:, 4],
        color="tab:purple",
        linewidth=1.8,
        linestyle="--",
        label=f"{axis_names[4]} filtered",
    )
    axes[1].plot(x, ft_wrench[:, 5], color="tab:brown", linewidth=1.2, alpha=0.25, label=f"{axis_names[5]} raw")
    axes[1].plot(
        x,
        ft_wrench_filtered[:, 5],
        color="tab:brown",
        linewidth=1.8,
        linestyle="--",
        label=f"{axis_names[5]} filtered",
    )
    axes[1].set_title(f"{side.capitalize()} torque")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("torque")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = _parse_args()
    if args.ft:
        args.vision = False

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
    raw_video_path = out_dir / "replay_raw.mp4"
    crop_video_path = out_dir / "replay_crop.mp4"
    plot_path = out_dir / "ft_wrench.png"
    sync_video_path = out_dir / "replay_ft_sync.mp4"
    extra_plot_paths: list[Path] = []

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
            if args.allow_partial and done.shape[0] > 0:
                bounds = [(0, done.shape[0] - 1)]
                print("[INFO] No done=True rows found; treating the full file as one partial demo.")
            else:
                raise ValueError("No complete demos found (no done=True rows).")
        if demo_end > len(bounds):
            raise IndexError(
                f"--demo={demo_start} {demo_end} is out of range. File has {len(bounds)} complete demos."
            )

        selected_bounds = bounds[demo_start - 1 : demo_end]
        row_arrays = [np.arange(start, end + 1, dtype=np.int64) for start, end in selected_bounds]
        rows = np.concatenate(row_arrays, axis=0)
        episode_boundaries = np.cumsum([row_array.shape[0] for row_array in row_arrays[:-1]]).astype(np.int64).tolist()
        start = int(selected_bounds[0][0])
        end = int(selected_bounds[-1][1])
        local_step_start = 0
        local_step_end = int(rows.shape[0] - 1)
        demo_step_text = f"policy steps {local_step_start}-{local_step_end}"

        wrist_rgb_raw = np.asarray(h5_file["wrist_rgb"][rows], dtype=np.uint8)
        side_view_rgb_raw = None
        if "side_view_rgb" in h5_file:
            side_view_rgb_raw = np.asarray(h5_file["side_view_rgb"][rows], dtype=np.uint8)
        fps = args.fps if args.fps is not None else _default_fps_from_h5(h5_file)
        crop_size = args.img_size if args.img_size is not None else _default_crop_size_from_h5(h5_file)
        wrist_rgb = wrist_rgb_raw
        if args.square_wrist:
            wrist_rgb = _center_square_numpy(wrist_rgb)
        wrist_rgb = _center_crop_numpy(wrist_rgb, crop_size)
        side_view_rgb = None
        if side_view_rgb_raw is not None:
            side_view_rgb = _center_crop_numpy(side_view_rgb_raw, crop_size)
        left_ft = None
        right_ft = None
        axis_names = None
        ft_sample_hz = None
        if not args.vision:
            # IsaacLab body_incoming_joint_wrench_b is the incoming joint wrench on the link.
            # Negate it here to visualize the reaction wrench measured by the CoinFT/environment.
            left_ft_raw = np.asarray(h5_file["left_ft_wrench"][rows], dtype=np.float32)
            right_ft_raw = np.asarray(h5_file["right_ft_wrench"][rows], dtype=np.float32)
            left_ft_flat, ft_samples_per_step = _flatten_multirate_ft(left_ft_raw)
            right_ft_flat, _ = _flatten_multirate_ft(right_ft_raw)
            left_ft = -left_ft_flat
            right_ft = -right_ft_flat
            ft_episode_boundaries = _expand_boundaries_for_ft(episode_boundaries, ft_samples_per_step)
            axis_names = _get_wrench_axis_names(h5_file)
            ft_sample_hz = float(h5_file.attrs.get("ft_log_hz", h5_file.attrs.get("physics_hz", 0.0)))
            left_ft_filtered = _filter_ft(
                left_ft,
                args.filter,
                args.ft_ma_window,
                args.lfilter_order,
                args.lfilter_cutoff_hz,
                sample_hz=ft_sample_hz,
            )
            right_ft_filtered = _filter_ft(
                right_ft,
                args.filter,
                args.ft_ma_window,
                args.lfilter_order,
                args.lfilter_cutoff_hz,
                sample_hz=ft_sample_hz,
            )
        x = np.arange(left_ft.shape[0] if not args.vision else wrist_rgb.shape[0], dtype=np.int64)

    wrote_raw_video = False
    if (not args.no_video) and args.vision and args.res in ("raw", "both"):
        target_path = raw_video_path if args.res == "both" else video_path
        raw_sync_height = args.sync_height if side_view_rgb_raw is not None else None
        with imageio.get_writer(str(target_path), fps=max(fps, 1), macro_block_size=1) as writer:
            for frame_idx, wrist_frame in enumerate(wrist_rgb_raw):
                if side_view_rgb_raw is not None:
                    frame = _concat_views(side_view_rgb_raw[frame_idx], wrist_frame, raw_sync_height)
                else:
                    frame = wrist_frame
                writer.append_data(_pad_to_even_hw(frame))
        wrote_raw_video = True

    wrote_crop_video = False
    if (not args.no_video) and args.res in ("crop", "both"):
        target_path = crop_video_path if args.res == "both" else video_path
        with imageio.get_writer(str(target_path), fps=max(fps, 1), macro_block_size=1) as writer:
            for frame_idx, wrist_frame in enumerate(wrist_rgb):
                if args.vision and side_view_rgb is not None:
                    frame = _concat_views(side_view_rgb[frame_idx], wrist_frame, args.sync_height)
                else:
                    frame = wrist_frame
                writer.append_data(_pad_to_even_hw(frame))
        wrote_crop_video = True

    if not args.vision:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for row in range(3):
            force_idx = row
            torque_idx = row + 3

            ax_force = axes[row, 0]
            ax_force.plot(x, left_ft[:, force_idx], color="tab:blue", linewidth=1.3, alpha=0.45, label="left raw")
            ax_force.plot(
                x,
                left_ft_filtered[:, force_idx],
                color="tab:blue",
                linewidth=1.8,
                linestyle="--",
                label="left filtered",
            )
            ax_force.plot(x, right_ft[:, force_idx], color="tab:orange", linewidth=1.3, alpha=0.45, label="right raw")
            ax_force.plot(
                x,
                right_ft_filtered[:, force_idx],
                color="tab:orange",
                linewidth=1.8,
                linestyle="--",
                label="right filtered",
            )
            ax_force.set_title(f"Force {axis_names[force_idx]}")
            ax_force.grid(alpha=0.3)
            ax_force.set_ylabel("force/torque")
            for boundary in ft_episode_boundaries:
                ax_force.axvline(boundary - 0.5, color="0.25", linestyle="-", linewidth=1.1, alpha=0.55)

            ax_torque = axes[row, 1]
            ax_torque.plot(x, left_ft[:, torque_idx], color="tab:blue", linewidth=1.3, alpha=0.45, label="left raw")
            ax_torque.plot(
                x,
                left_ft_filtered[:, torque_idx],
                color="tab:blue",
                linewidth=1.8,
                linestyle="--",
                label="left filtered",
            )
            ax_torque.plot(
                x, right_ft[:, torque_idx], color="tab:orange", linewidth=1.3, alpha=0.45, label="right raw"
            )
            ax_torque.plot(
                x,
                right_ft_filtered[:, torque_idx],
                color="tab:orange",
                linewidth=1.8,
                linestyle="--",
                label="right filtered",
            )
            ax_torque.set_title(f"Torque {axis_names[torque_idx]}")
            ax_torque.grid(alpha=0.3)
            for boundary in ft_episode_boundaries:
                ax_torque.axvline(boundary - 0.5, color="0.25", linestyle="-", linewidth=1.1, alpha=0.55)

        axes[2, 0].set_xlabel("step")
        axes[2, 1].set_xlabel("step")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{h5_path.name} | demo {demo_label} | {demo_step_text}", fontsize=12)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

        render_ft_frame, ft_fig = _make_ft_plot_renderer(
            left_ft,
            right_ft,
            left_ft_filtered,
            right_ft_filtered,
            axis_names,
            f"{h5_path.name} | demo {demo_label} | {demo_step_text}",
            episode_boundaries=ft_episode_boundaries,
        )
        resized_for_sync = False
        if not args.no_video:
            with imageio.get_writer(str(sync_video_path), fps=max(fps, 1), macro_block_size=1) as writer:
                for frame_idx, wrist_frame in enumerate(wrist_rgb):
                    ft_frame_idx = min((frame_idx + 1) * ft_samples_per_step - 1, left_ft.shape[0] - 1)
                    ft_frame = render_ft_frame(ft_frame_idx)
                    if args.sync_height is not None:
                        wrist_frame = _resize_to_height(wrist_frame, args.sync_height)
                        ft_frame = _resize_to_height(ft_frame, args.sync_height)
                    if ft_frame.shape[0] != wrist_frame.shape[0]:
                        target_w = max(int(round(ft_frame.shape[1] * wrist_frame.shape[0] / ft_frame.shape[0])), 1)
                        ft_frame = _resize_image_nearest(ft_frame, wrist_frame.shape[0], target_w)
                        resized_for_sync = True
                    synced = np.concatenate((wrist_frame, ft_frame), axis=1)
                    writer.append_data(_pad_to_even_hw(synced))
        plt.close(ft_fig)
        if args.plot is not None:
            plot_sides = ("left", "right") if args.plot == "both" else (args.plot,)
            title_prefix = f"{h5_path.name} | demo {demo_label} | {demo_step_text}"
            for side in plot_sides:
                side_ft = left_ft if side == "left" else right_ft
                side_ft_filtered = left_ft_filtered if side == "left" else right_ft_filtered
                side_plot_path = out_dir / f"{side}_ft_wrench.png"
                _save_side_ft_plot(
                    side=side,
                    ft_wrench=side_ft,
                    ft_wrench_filtered=side_ft_filtered,
                    axis_names=axis_names,
                    title=f"{title_prefix} | {side} FT wrench",
                    output_path=side_plot_path,
                    sample_hz=ft_sample_hz if ft_sample_hz > 0.0 else None,
                )
                extra_plot_paths.append(side_plot_path)

    if (not args.no_video) and args.res in ("raw", "crop"):
        print(f"[INFO] Wrote video: {video_path}")
    if wrote_raw_video and (not args.no_video) and args.res == "both":
        print(f"[INFO] Wrote raw:   {raw_video_path}")
    if wrote_crop_video and (not args.no_video) and args.res == "both":
        print(f"[INFO] Wrote crop:  {crop_video_path}")
    if not args.vision:
        print(f"[INFO] Wrote plot:  {plot_path}")
        for extra_plot_path in extra_plot_paths:
            print(f"[INFO] Wrote plot:  {extra_plot_path}")
        if not args.no_video:
            print(f"[INFO] Wrote sync:  {sync_video_path}")
        if args.square_wrist:
            print("[INFO] Sync video note: wrist camera center-cropped to square.")
        if args.sync_height is not None:
            print(f"[INFO] Sync video note: upscaled/downscaled panels to {args.sync_height}px height.")
        print("[INFO] FT sign convention: plotted -left/right_ft_wrench as CoinFT reaction wrench.")
        print(f"[INFO] FT filter: {args.filter}")
        if args.filter == "ma":
            print(f"[INFO] FT moving-average window: {args.ft_ma_window}")
        else:
            print(f"[INFO] FT lfilter order: {args.lfilter_order}")
            print(f"[INFO] FT lfilter cutoff Hz: {args.lfilter_cutoff_hz}")
        if resized_for_sync:
            print("[INFO] Sync video note: resized FT plot panel to match wrist frame height.")
        elif not args.no_video:
            print("[INFO] Sync video note: no resize needed (matched panel heights).")
    if args.no_video:
        print("[INFO] Video generation disabled by --no_video.")
    print(
        f"[INFO] Demo policy steps: {local_step_start}..{local_step_end} "
        f"({rows.shape[0]} concatenated steps)"
    )
    print(f"[INFO] Source rows:       {start}..{end}")
    print(f"[INFO] Replay FPS:  {fps}")


if __name__ == "__main__":
    main()
