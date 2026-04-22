# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class FactoryVisuomotorDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 200
    save_interval = 50
    experiment_name = "factory_privileged"
    run_name = "visuomotor_distillation"

    obs_groups = {"student": ["policy"], "teacher": ["critic"]}

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=1.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
