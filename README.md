# train visuomotor:
```bash
python scripts/train.py   --config-name train_dp_timm_visuomotor   task=gear_mesh_vision_wrist_sideview   task.dataset_path=logs/vistac_rollouts/gear_mesh_200demo_wait1p5_0p8speed_0p15height.h5   exp_name=gear_mesh_visuomotor_timm_wait1p5_0p8speed_0p15height
```

# train visuotactile:
```bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party:$PWD/third_party/multimodal_representation/multimodal:$PWD do i need to run this severytihme
```
```bash
python scripts/train.py   --config-name train_dp_timm_vistuotactile   task=gear_mesh_visuotactile_timm_ft   task.dataset_path=logs/vistac_rollouts/gear_mesh_200demo_wait1p5_0p8speed_0p15height.h5   exp_name=gear_mesh_vistac_timm_ft_wait1p5_0p8speed_0p15height

```