# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab locomotion environment for the Open Duck Mini v2.

Extends the built-in LocomotionVelocityRoughEnvCfg following the H1 humanoid
biped pattern. Overrides rewards for biped-specific gait control and adjusts
simulation/scene parameters for the duck robot.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG


@configclass
class DuckRewards(RewardsCfg):
    """Biped-specific rewards for the Open Duck Mini v2.

    Follows the H1/G1/Digit humanoid biped reward pattern with duck-specific
    body names and weight tuning for Feetech STS3250 servos.
    """

    # --- Termination penalty (strong negative signal for falling) ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # --- Override velocity tracking with yaw-frame versions (biped best practice) ---
    lin_vel_z_l2 = None  # Disable default bouncing penalty — not needed for biped
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # --- Biped-specific gait reward — encourages alternating single-stance ---
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_foot", "right_foot"]
            ),
            "threshold": 0.4,
        },
    )

    # --- Penalize feet sliding on ground ---
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_foot", "right_foot"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_foot", "right_foot"]
            ),
        },
    )

    # --- Joint deviation penalties ---

    # Penalize deviation of non-locomotion joints (head/antennas) from default
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                    "left_antenna", "right_antenna",
                ],
            )
        },
    )

    # Penalize hip yaw/roll deviation — prevents unnecessary hip splaying
    # and conserves torque on the Feetech STS3250 servos.
    # Following H1/G1/Digit biped configs which all penalize hip deviation.
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_hip_yaw", "left_hip_yaw",
                    "right_hip_roll", "left_hip_roll",
                ],
            )
        },
    )

    # --- Joint position limits — protects real Feetech STS3250 servos ---
    # All Isaac Lab biped configs (H1, G1, Cassie, Digit) include this.
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


@configclass
class OpenDuckRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Open Duck Mini v2 locomotion environment configuration.

    Extends the base rough locomotion config with duck-specific:
    - Robot asset (OPEN_DUCK_MINI_V2_CFG)
    - Biped reward overrides (DuckRewards)
    - Simulation timing (200 Hz physics, 50 Hz policy)
    - Velocity command ranges for a small biped
    - Duck-specific body names for termination and events
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

        # --- Commands: velocity ranges for a small biped ---
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

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

        # --- Rewards: tune base-class weights for duck ---
        self.rewards.undesired_contacts = None  # Re-enable after verifying body names
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_torques_l2.weight = -1e-5  # Enable now that STS3250 servos have headroom
        # Increase pitch/roll angular velocity penalty (default -0.05) to
        # compensate for top-heavy trunk after Jetson relocation (+316g).
        self.rewards.ang_vel_xy_l2.weight = -0.1


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

        # Fixed forward walk command for evaluation
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Disable observation noise and external disturbances
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
