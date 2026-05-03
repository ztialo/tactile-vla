#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline Diffusion-style visuomotor policy training with the CNN baseline."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_equi_df as train_impl


train_impl.DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs").joinpath("train_diffusion.yaml")


if __name__ == "__main__":
    train_impl.main()
