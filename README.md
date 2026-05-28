# train visuomotor:
```bash
python scripts/train.py   --config-name train_dp_timm_visuomotor   task=gear_mesh_vision_wrist_sideview   task.dataset_path=logs/vistac_rollouts/gear_mesh_200demo_wait1p5_0p8speed_0p15height.h5   exp_name=gear_mesh_visuomotor_timm_wait1p5_0p8speed_0p15height
```

# train visuotactile:
```bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party:$PWD/third_party/multimodal_representation/multimodal:$PWD
python scripts/train.py \
  --config-name train_dp_timm_vistuotactile \
  task=gear_mesh_visuotactile_timm_ft \
  task.dataset_path=logs/vistac_rollouts/gear_mesh_200demo_wait1p5_0p8speed_0p15height.h5 \
  exp_name=gear_mesh_vistac_timm_ft_wait1p5_0p8speed_0p15height

```

# evaluate visuotactile:
```bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party:$PWD/third_party/multimodal_representation/multimodal:$PWD

python scripts/assess_diffusion.py \
  --checkpoint logs/diffusion/gear_mesh_visuotactile_timm_ft/2026.05.28-01.31.57/checkpoints/epoch=0045-val_loss=0.0367.ckpt \
  --task Visuotactile-Factory-GearMesh-Direct-v0 \
  --fixed_asset_height \
  --num_loops 3 \
  --side_view_grid_9
```
May need: 
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/lib/x86_64-linux-gnu/libgcc_s.so.1
```