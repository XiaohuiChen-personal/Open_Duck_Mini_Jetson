# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab locomotion environment for the Open Duck Mini v2.

Reward design based on:
- Disney BDX paper (Grandia et al., arXiv:2501.05204v1) — survival bonus, action rate
- Open Duck Playground (apirrone/Open_Duck_Playground) — velocity ranges, tracking sigma
- First training run analysis — removed H1-specific penalty bloat

Key differences from the H1 humanoid biped config:
- Alive bonus (+5.0/step) — structurally positive reward
- Sharper velocity tracking (std=0.1 vs 0.5)
- Conservative velocity ranges matching a 42cm robot
- Reduced action scale (0.25 vs 0.5)
- Fewer penalty terms (8 total vs 13 in H1-derived config)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG


@configclass
class DuckRewards(RewardsCfg):
    """Reward function for Open Duck Mini v2 locomotion.

    Designed from first principles for a small (42cm, 2.75kg) bipedal duck robot
    with STS3250 position-controlled servos. 8 reward terms: 4 positive, 4 negative.

    References:
    - Disney BDX: survival bonus structure (weight=20, we use 5.0 adjusted for Isaac Lab)
    - Open Duck Playground: velocity ranges, tracking sigma=0.01, action_scale=0.25
    - Cassie/H1: biped gait rewards (feet_air_time_positive_biped)
    """

    # ========== POSITIVE REWARDS ==========

    # Survival bonus: +5.0 per step for staying alive.
    # Disney BDX uses +20.0, Open Duck Playground uses +20.0 (multiplied by dt=0.02 = 0.4 effective).
    # Isaac Lab does NOT multiply rewards by dt, so +5.0 here gives a similar positive budget.
    # This makes the reward structurally positive — the policy is always incentivized to survive.
    alive_bonus = RewTerm(func=mdp.is_alive, weight=5.0)

    # Velocity tracking: exponential reward for matching commanded linear velocity.
    # std=0.1 (sharper than H1's 0.5; Playground uses 0.01 but different implementation).
    # Weight 2.0 (Playground uses 2.5 but their reward is per-dt).
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.1},
    )

    # Yaw velocity tracking: exponential reward for matching commanded turn rate.
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # Biped gait: reward alternating single-stance phases.
    # threshold=0.2s (reduced from H1's 0.4 — small robot takes faster, shorter steps).
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot_assembly", "foot_assembly_2"]
            ),
            "threshold": 0.2,
        },
    )

    # ========== NEGATIVE REWARDS (penalties) ==========

    # Termination penalty: strong negative signal for falling.
    # Universal across all biped configs. -200.0 is standard.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Flat orientation: penalize tilting. Weight -1.0 (same as H1).
    # Important for the top-heavy duck (64% mass above hips).
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Action smoothness: penalize jerky motor commands.
    # Weight -0.005 (same as H1, validated in first training run as not dominant
    # once other penalties are removed).
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # Joint position limits: penalize ankle/knee joints approaching hard limits.
    # Protects real STS3250 servos from hitting hard stops.
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_ankle", "left_ankle",
                    "right_knee", "left_knee",
                ],
            )
        },
    )

    # ========== DISABLED BASE-CLASS REWARDS ==========
    # Explicitly disable rewards from LocomotionVelocityRoughEnvCfg that don't
    # apply to our robot or were causing issues in the first training run.

    lin_vel_z_l2 = None           # Bouncing penalty — not needed for biped
    ang_vel_xy_l2 = None          # Redundant with flat_orientation_l2
    dof_torques_l2 = None         # Negligible at -1e-5, adds noise
    dof_acc_l2 = None             # Negligible at -1.25e-7, adds noise
    feet_air_time_base = None     # We override with biped version above
    undesired_contacts = None     # Re-enable after verifying body names
    dof_pos_limits = None         # We define our own joint_pos_limits above


@configclass
class OpenDuckRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Open Duck Mini v2 locomotion environment configuration.

    Designed for a 42cm, 2.75kg bipedal duck with STS3250 servos.
    Velocity ranges and reward structure based on Open Duck Playground.
    """

    rewards: DuckRewards = DuckRewards()

    def __post_init__(self):
        super().__post_init__()

        # --- Scene: swap robot asset ---
        self.scene.robot = OPEN_DUCK_MINI_V2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        # --- Simulation timing ---
        self.sim.dt = 0.005  # 200 Hz physics (matching MuJoCo timestep)
        self.decimation = 4  # Policy at 50 Hz (200 / 4)
        self.episode_length_s = 20.0

        # --- Height scanner: disabled for flat-ground training ---
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # --- Contact sensor: fix prim path for MJCF-converted USD ---
        # The MJCF→USD converter nests bodies under /base/, so the default
        # sensor path "{ENV_REGEX_NS}/Robot/.*" doesn't find them.
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/.*",
            history_length=3,
            track_air_time=True,
        )

        # --- Commands: velocity ranges for a 42cm duck ---
        # Open Duck Playground uses lin_vel_x [-0.15, 0.15] m/s.
        # We allow slightly more forward (0.3 m/s) for initial exploration.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # --- Action scale: reduce from default 0.5 to 0.25 ---
        # Matches Open Duck Playground. Conservative control for position servos.
        self.actions.joint_pos.scale = 0.25

        # --- Terminations: use duck trunk body ---
        self.terminations.base_contact.params["sensor_cfg"].body_names = (
            "trunk_assembly"
        )

        # --- Events: adjust for duck-specific bodies ---
        self.events.push_robot = None  # Disable initially; enable after basic walking
        self.events.add_base_mass = None  # Disable mass randomization initially
        self.events.base_com = None  # Disable CoM randomization initially
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            "trunk_assembly"
        ]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class OpenDuckRoughEnvCfg_PLAY(OpenDuckRoughEnvCfg):
    """Playback configuration with fewer envs and no randomization."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # Disable terrain curriculum
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Fixed forward walk command (within trained range)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Disable observation noise and external disturbances
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
