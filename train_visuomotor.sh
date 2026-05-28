#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python scripts/train.py \
  --config-name train_dp_timm_visuomotor \
  task=gear_mesh_vision_wrist_sideview \
  task.dataset_path=logs/vistac_rollouts/gear_mesh_200demo_wait1p5_0p8speed_0p15height.h5 \
  exp_name=gear_mesh_visuomotor_timm_wait1p5_0p8speed_0p15height \
  'hydra.run.dir=/data/dp_output/${now:%Y.%m.%d-%H.%M.%S}_gear_mesh_visuomotor'
