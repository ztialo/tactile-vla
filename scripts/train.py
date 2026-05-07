#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DP_ROOT = REPO_ROOT / "third_party" / "diffusion_policy"

if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(SCRIPT_DIR / "configs"),
    config_name="train_diffusion_policy_image",
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
