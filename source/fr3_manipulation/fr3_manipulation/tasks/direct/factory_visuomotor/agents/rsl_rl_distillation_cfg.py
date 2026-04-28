# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class FactoryVisuomotorDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 200
    save_interval = 50
    experiment_name = "factory_privileged"
    run_name = "visuomotor_distillation"

    # The distillation runner expects the student obs group under "policy" and the teacher obs group under "teacher".
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}

    # Offline BC initialization checkpoint for the student MLP head. The train script copies this into the student.
    student_init_checkpoint: str = ""

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 128, 64],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=1.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
