# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO training configuration for the Open Duck Mini v2.

Follows the H1 humanoid biped PPO config with identical hyperparameters,
as these have been validated extensively for bipedal locomotion.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OpenDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training configuration for Open Duck Mini v2 rough locomotion."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "open_duck_ppo"

    # MLP policy: [512, 256, 128] with ELU activation (same as H1)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO hyperparameters (identical to H1 — proven for bipedal locomotion)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
