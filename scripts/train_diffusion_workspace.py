#!/usr/bin/env python3

from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffusion_workspace import TrainDiffusionUnetImageWorkspace  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(SCRIPT_DIR / "configs"),
    config_name="train_diffusion_workspace",
)
def main(cfg: OmegaConf):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
