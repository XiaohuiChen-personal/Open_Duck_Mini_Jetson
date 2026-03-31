# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO training configuration for the Open Duck Mini v2.

Tuned based on Open Duck Playground parameters and first training run analysis.
Key changes from H1 baseline:
- Observation normalization enabled (critical for mixed-scale observations)
- Gamma reduced to 0.97 (shorter horizon, faster convergence for locomotion)
- Entropy coefficient reduced to 0.005 (less exploration, more exploitation)
- Initial noise std reduced to 0.5 (prevents entropy increase during training)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OpenDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training configuration for Open Duck Mini v2 locomotion."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "open_duck_ppo"

    # MLP policy: [512, 256, 128] with ELU activation
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Reduced from 1.0 — prevents entropy runaway
        actor_obs_normalization=True,  # Enable — critical for mixed-scale obs
        critic_obs_normalization=True,  # Enable
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO hyperparameters (tuned for small biped locomotion)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Reduced from 0.01 — less exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.97,  # Reduced from 0.99 — shorter horizon, faster convergence
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
