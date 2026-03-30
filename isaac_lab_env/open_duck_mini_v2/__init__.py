# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Register Open Duck Mini v2 environments with Gymnasium.

Follows the Isaac Lab registration pattern (see H1 config at
isaaclab_tasks/manager_based/locomotion/velocity/config/h1/__init__.py).

Environments:
    Isaac-Velocity-Rough-OpenDuck-v0 : Training environment (4096 envs, domain randomization)
    Isaac-Velocity-Rough-OpenDuck-Play-v0 : Evaluation/playback (50 envs, no randomization)
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-OpenDuck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-OpenDuck-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckPPORunnerCfg",
    },
)
