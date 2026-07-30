# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Register Open Duck Mini v2 environments with Gymnasium.

Follows the Isaac Lab registration pattern (see H1 config at
isaaclab_tasks/manager_based/locomotion/velocity/config/h1/__init__.py).

Environments:
    Isaac-Velocity-Rough-OpenDuck-v0 : Training environment (4096 envs; observation
        noise only — dynamics randomization is currently disabled: push_robot /
        add_base_mass / base_com are set to None in env_cfg.py __post_init__)
    Isaac-Velocity-Rough-OpenDuck-Play-v0 : Evaluation/playback (50 envs, no randomization)
    Isaac-Velocity-Rough-OpenDuck-Robust-v0 / -Robust-Play-v0 : v4-robust track
        (dynamics DR + asymmetric 59-dim actor obs) and its play twin
    Isaac-Velocity-Rough-OpenDuck-PushEval-v0 / -PlainPushEval-v0 : push-recovery
        eval tasks (Task 2.7), 59-dim robust and 62-dim plain variants
    Isaac-Velocity-Rough-OpenDuck-Contact-v0 / -Contact-Play-v0 : v5 contact-rich
        track (Task 2.8) — obstacles, sustained wrenches, sustained fast rotation,
        fall-only terminations, disturbance-gated rewards — and its play twin
    Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0 : sustained-press gate (plan gate 5)
    Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0 : obstacle-graze gate / video audit
    Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0 : push eval under v5's
        fall-only termination (companion to the v4-comparable PushEval number)

(The skrl AMP tasks were removed 2026-07-26 with the course-study archive;
see the open-duck-ppo-vs-amp repo.)
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

# --- v4-robust track (Run B): dynamics DR + asymmetric actor/critic obs ---

gym.register(
    id="Isaac-Velocity-Rough-OpenDuck-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckRobustPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-OpenDuck-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckRobustEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckRobustPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-OpenDuck-PushEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckPushEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckRobustPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-OpenDuck-PlainPushEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckPlainPushEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckPPORunnerCfg",
    },
)

# --- v5 contact-rich track (Task 2.8) ---

_V5_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckContactPPORunnerCfg"

for _task_id, _cfg_class in (
    ("Isaac-Velocity-Rough-OpenDuck-Contact-v0", "OpenDuckContactEnvCfg"),
    ("Isaac-Velocity-Rough-OpenDuck-Contact-Play-v0", "OpenDuckContactEnvCfg_PLAY"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactUngated-v0", "OpenDuckContactUngatedEnvCfg"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactMinimal-v0", "OpenDuckContactMinimalEnvCfg"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactMinimal-Play-v0", "OpenDuckContactMinimalEnvCfg_PLAY"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0", "OpenDuckContactWrenchEnvCfg"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0", "OpenDuckContactWrenchEnvCfg_PLAY"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactUngated-Play-v0", "OpenDuckContactUngatedEnvCfg_PLAY"),
    ("Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0", "OpenDuckWrenchEvalEnvCfg"),
    ("Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0", "OpenDuckObstacleEvalEnvCfg"),
    ("Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0", "OpenDuckContactPushEvalEnvCfg"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_cfg_class}",
            "rsl_rl_cfg_entry_point": _V5_AGENT,
        },
    )
