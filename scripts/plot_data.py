#!/usr/bin/env python3
"""Plot left/right finger FT wrench data from a CSV log."""

import argparse
import ast
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_wrench(raw: str):
    """Parse a serialized wrench list into a 6D numpy vector."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        return None
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot FT wrench CSV from data_collect_play.py")
    parser.add_argument("csv_path", type=Path, help="Path to FT CSV file")
    parser.add_argument(
        "--x-axis",
        choices=["step", "time"],
        default="step",
        help="Use step index or wall-time seconds on x-axis",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path (default: alongside CSV as .png)",
    )
    parser.add_argument("--show", action="store_true", help="Display plot window")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    steps = []
    times = []
    left_wrenches = []
    right_wrenches = []

    with args.csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left = parse_wrench(row.get("left_ft_joint"))
            right = parse_wrench(row.get("right_ft_joint"))
            if left is None or right is None:
                continue

            steps.append(int(row["step"]))
            times.append(datetime.fromisoformat(row["wall_time_iso"]))
            left_wrenches.append(left)
            right_wrenches.append(right)

    if not left_wrenches:
        raise ValueError("No valid left/right wrench rows found in CSV.")

    left = np.vstack(left_wrenches)
    right = np.vstack(right_wrenches)

    if args.x_axis == "time":
        t0 = times[0]
        x = np.array([(t - t0).total_seconds() for t in times], dtype=float)
        x_label = "Time (s)"
    else:
        x = np.asarray(steps, dtype=float)
        x_label = "Step"

    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.suptitle(f"FT Wrench Over Time\n{args.csv_path.name}")

    for i, ax in enumerate(axes.flat):
        ax.plot(x, left[:, i], label="Left finger", linewidth=1.2)
        ax.plot(x, right[:, i], label="Right finger", linewidth=1.2, alpha=0.9)
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
        if i >= 3:
            ax.set_xlabel(x_label)

    axes[0, 0].legend(loc="best")
    plt.tight_layout()

    save_path = args.save if args.save is not None else args.csv_path.with_suffix(".png")
    fig.savefig(save_path, dpi=160)
    print(f"Saved plot to: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
