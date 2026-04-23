# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class RslRlPpoAlgorithmCompatCfg:
    """PPO config matching the rsl-rl version installed in this environment."""

    class_name: str = "PPO"
    num_learning_epochs: int = 4
    num_mini_batches: int = 32
    clip_param: float = 0.2
    gamma: float = 0.995
    lam: float = 0.95
    value_loss_coef: float = 2.0
    entropy_coef: float = 0.0
    learning_rate: float = 1.0e-4
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.008
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None


@configclass
class FactoryPrivilegedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 200
    save_interval = 100
    experiment_name = "factory_privileged"
    run_name = "teacher"
    obs_groups = {"policy": ["critic"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 128, 64],
        critic_hidden_dims=[512, 128, 64],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=1024,
        rnn_num_layers=2,
    )

    algorithm = RslRlPpoAlgorithmCompatCfg()
