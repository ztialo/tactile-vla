#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_episode_spans(h5_path: Path) -> list[np.ndarray]:
    with h5py.File(h5_path, "r") as h5_file:
        done = np.asarray(h5_file["done"], dtype=np.bool_)
        env_ids = np.asarray(h5_file["env_id"], dtype=np.int64)
        spans: list[np.ndarray] = []
        for env_id in np.unique(env_ids):
            env_rows = np.nonzero(env_ids == env_id)[0]
            env_done = done[env_rows]
            start_idx = 0
            for i, is_done in enumerate(env_done):
                if is_done:
                    spans.append(env_rows[start_idx : i + 1].astype(np.int64, copy=True))
                    start_idx = i + 1
    return spans


def build_row_index(spans: list[np.ndarray], num_episodes: int) -> list[np.ndarray]:
    if num_episodes < 0:
        raise ValueError(f"num_episodes must be non-negative, got {num_episodes}")
    if len(spans) < num_episodes:
        raise ValueError(f"Requested {num_episodes} episodes but file only contains {len(spans)}.")
    return spans[:num_episodes]


def rewrite_episode_metadata(data: dict[str, np.ndarray], episode_lengths: list[int]) -> None:
    total_rows = sum(episode_lengths)
    data["env_id"] = np.zeros((total_rows,), dtype=np.int64)

    episode_id = np.empty((total_rows,), dtype=np.int64)
    timestep = np.empty((total_rows,), dtype=np.int64)
    cursor = 0
    for episode_idx, length in enumerate(episode_lengths):
        episode_id[cursor : cursor + length] = episode_idx
        timestep[cursor : cursor + length] = np.arange(length, dtype=np.int64)
        cursor += length
    data["episode_id"] = episode_id
    data["timestep"] = timestep


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix complete visuotactile HDF5 episodes from two source files.")
    parser.add_argument("--uniform_path", type=Path, required=True)
    parser.add_argument("--gaussian_path", type=Path, required=True)
    parser.add_argument("--uniform_episodes", type=int, default=5)
    parser.add_argument("--gaussian_episodes", type=int, default=195)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    uniform_spans = load_episode_spans(args.uniform_path)
    gaussian_spans = load_episode_spans(args.gaussian_path)
    selected_uniform = build_row_index(uniform_spans, args.uniform_episodes)
    selected_gaussian = build_row_index(gaussian_spans, args.gaussian_episodes)

    selected = [(args.uniform_path, rows) for rows in selected_uniform]
    selected.extend((args.gaussian_path, rows) for rows in selected_gaussian)

    keys: list[str] | None = None
    merged: dict[str, list[np.ndarray]] = {}
    episode_lengths: list[int] = []

    for source_path, rows in selected:
        with h5py.File(source_path, "r") as h5_file:
            if keys is None:
                keys = list(h5_file.keys())
                merged = {key: [] for key in keys}
            elif list(h5_file.keys()) != keys:
                raise ValueError(f"HDF5 schema mismatch in {source_path}")
            for key in keys:
                merged[key].append(np.asarray(h5_file[key][rows]))
        episode_lengths.append(int(rows.shape[0]))

    if keys is None:
        raise ValueError("No episodes selected.")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stacked = {key: np.concatenate(chunks, axis=0) for key, chunks in merged.items()}
    rewrite_episode_metadata(stacked, episode_lengths)

    with h5py.File(args.output_path, "w") as out_h5:
        for key in keys:
            out_h5.create_dataset(key, data=stacked[key])

        with h5py.File(args.gaussian_path, "r") as gaussian_h5:
            for attr_key, attr_value in gaussian_h5.attrs.items():
                out_h5.attrs[attr_key] = attr_value

        out_h5.attrs["mixed_dataset"] = True
        out_h5.attrs["uniform_source_path"] = str(args.uniform_path)
        out_h5.attrs["gaussian_source_path"] = str(args.gaussian_path)
        out_h5.attrs["uniform_episodes_used"] = args.uniform_episodes
        out_h5.attrs["gaussian_episodes_used"] = args.gaussian_episodes
        out_h5.attrs["successful_episodes_target"] = args.uniform_episodes + args.gaussian_episodes


if __name__ == "__main__":
    main()
