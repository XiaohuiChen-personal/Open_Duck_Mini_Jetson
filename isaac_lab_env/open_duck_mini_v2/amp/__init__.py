# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) environments for the Open Duck Mini v2.

Follows the Isaac Lab direct-workflow registration pattern (see
isaaclab_tasks/direct/humanoid_amp/__init__.py). The agent config kwarg key
must be ``skrl_amp_cfg_entry_point``: the skrl train script resolves it when
launched with ``--algorithm AMP``.

Environments:
    Isaac-OpenDuck-AMP-PureStyle-v0 : Pure style imitation (single forward
        clip, no commands, style reward only) — AMP sanity-check task.
    Isaac-OpenDuck-AMP-v0 : Command-conditioned locomotion (full clip
        library; task/style reward mix set in skrl_amp_command_cfg.yaml,
        currently 0.6/0.4) — deployable policy.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-OpenDuck-AMP-PureStyle-v0",
    entry_point=f"{__name__}.duck_amp_env:DuckAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.duck_amp_env_cfg:DuckAmpPureStyleEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_purestyle_cfg.yaml",
    },
)

gym.register(
    id="Isaac-OpenDuck-AMP-v0",
    entry_point=f"{__name__}.duck_amp_env:DuckAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.duck_amp_env_cfg:DuckAmpCommandEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_command_cfg.yaml",
    },
)

gym.register(
    id="Isaac-OpenDuck-AMP-Video-v0",
    entry_point=f"{__name__}.duck_amp_env:DuckAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.duck_amp_env_cfg:DuckAmpVideoEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_command_cfg.yaml",
    },
)
