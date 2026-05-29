#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DP = REPO_ROOT / "third_party" / "diffusion_policy"
if str(THIRD_PARTY_DP) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DP))

try:
    import zarr  # noqa: F401
except ModuleNotFoundError:
    sys.modules["zarr"] = types.SimpleNamespace(Array=type("Array", (), {}))

from diffusion_policy.dataset.isaaclab_hdf5_image_dataset import (  # noqa: E402
    _convert_action_repr_numpy,
    _prepare_ft_wrench,
    _read_h5_rows,
    load_episode_spans,
)


def _load_task_cfg(config_path: str):
    cfg = OmegaConf.load(config_path)
    if "task" in cfg:
        return cfg.task
    wrapped = OmegaConf.create({"task": cfg})
    return wrapped.task


def _instantiate_dataset(task_cfg, dataset_override: str | None):
    import hydra

    if dataset_override is not None:
        task_cfg.dataset.dataset_path = dataset_override
    return hydra.utils.instantiate(task_cfg.dataset)


def _flatten_ft(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError(f"Expected FT array with shape [T, S, 6], got {values.shape}")
    return values.reshape(values.shape[0] * values.shape[1], values.shape[2])


def _plot_wrench(normalized: np.ndarray, out_path: Path, title: str):
    axis_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    x = np.arange(normalized.shape[0], dtype=np.int64)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i in range(3):
        axes[0].plot(x, normalized[:, i], color=colors[i], linewidth=1.3, label=axis_names[i])
    for i in range(3, 6):
        axes[1].plot(x, normalized[:, i], color=colors[i], linewidth=1.3, label=axis_names[i])

    axes[0].set_ylabel("Normalized force")
    axes[1].set_ylabel("Normalized torque")
    axes[1].set_xlabel("FT substep index")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_action(normalized: np.ndarray, out_path: Path, title: str):
    axis_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    x = np.arange(normalized.shape[0], dtype=np.int64)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i in range(3):
        axes[0].plot(x, normalized[:, i], color=colors[i], linewidth=1.3, label=axis_names[i])
    for i in range(3, 6):
        axes[1].plot(x, normalized[:, i], color=colors[i], linewidth=1.3, label=axis_names[i])

    axes[0].set_ylabel("Normalized translation")
    axes[1].set_ylabel("Normalized rotation")
    axes[1].set_xlabel("Policy step")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect and plot exact DP preprocessing for one H5 demo.")
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/configs/task/gear_mesh_visuotactile_timm_ft.yaml",
        help="Task config YAML or full training config YAML.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional H5 override.")
    parser.add_argument("--demo", type=int, required=True, help="1-indexed done-delimited demo number.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to logs/demo_replay/<h5 name>/<demo>/dp_preprocess_plots.",
    )
    args = parser.parse_args()

    task_cfg = _load_task_cfg(args.config)
    dataset = _instantiate_dataset(task_cfg, args.dataset)
    normalizer = dataset.get_normalizer()
    dataset_path = Path(dataset.dataset_path).expanduser().resolve()

    spans = load_episode_spans(str(dataset_path))
    demo_index = int(args.demo) - 1
    if demo_index < 0 or demo_index >= len(spans):
        raise IndexError(f"--demo must be in [1, {len(spans)}], got {args.demo}")
    span = spans[demo_index]

    if args.out_dir is None:
        out_dir = REPO_ROOT / "logs" / "demo_replay" / dataset_path.name / str(args.demo) / "dp_preprocess_plots"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    with h5py.File(dataset_path, "r") as h5_file:
        rows = span.rows
        raw_action = np.asarray(_read_h5_rows(h5_file["action"], rows), dtype=np.float32)
        action_processed = _convert_action_repr_numpy(raw_action, dataset.rotation_rep).astype(np.float32)
        action_normalized = normalizer["action"].normalize(action_processed).detach().cpu().numpy()
        arrays["action_processed"] = action_processed
        arrays["action_normalized"] = action_normalized

        for key in dataset.wrench_keys:
            wrench_raw = np.asarray(_read_h5_rows(h5_file[key], rows), dtype=np.float32)
            wrench_processed = _prepare_ft_wrench(
                wrench_raw,
                negate=dataset.negate_ft,
                ma_window=dataset.ft_ma_window,
            ).astype(np.float32)
            wrench_normalized = (
                normalizer[key].normalize(wrench_processed).detach().cpu().numpy().astype(np.float32)
            )
            arrays[f"{key}_raw"] = wrench_raw
            arrays[f"{key}_processed"] = wrench_processed
            arrays[f"{key}_normalized"] = wrench_normalized

    _plot_action(
        arrays["action_normalized"],
        out_dir / "action_processed_normalized.png",
        f"Demo {args.demo} action normalized for DP",
    )
    for key in dataset.wrench_keys:
        _plot_wrench(
            _flatten_ft(arrays[f"{key}_normalized"]),
            out_dir / f"{key}_processed_normalized.png",
            f"Demo {args.demo} {key} normalized for DP",
        )

    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_path": str(dataset_path),
        "demo": int(args.demo),
        "demo_length": span.length,
        "ft_ma_window": int(dataset.ft_ma_window),
        "negate_ft": bool(dataset.negate_ft),
        "rotation_rep": str(dataset.rotation_rep),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote {out_dir / 'action_processed_normalized.png'}")
    for key in dataset.wrench_keys:
        print(f"[INFO] Wrote {out_dir / f'{key}_processed_normalized.png'}")


if __name__ == "__main__":
    main()
