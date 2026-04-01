# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab locomotion environment for the Open Duck Mini v2.

Reward design v4 — imitation reward using Open Duck Playground reference motions.

Evolution:
- v1 (Run 1): H1-derived, structurally negative, 13 penalties → failed
- v2 (Run 2): Alive bonus → crouching/shuffling → failed
- v3 (Run 3): Isaac Lab biped pattern, no alive bonus, height control → improved
- v4 (Run 4): Added polynomial imitation reward for natural gait quality

v4 adds reference motion tracking from the Open Duck Playground gait library
(240 polynomial walking gaits, degree-15, 0.54s period). The imitation reward
(weight=10.0) is the dominant positive signal — it rewards matching reference
leg joint positions via exp(-2 * squared_error). Velocity tracking and
feet_air_time are retained as secondary signals.

Phase observation [cos(phase), sin(phase)] added to policy obs so the network
knows where in the gait cycle to aim.

Penalties reduced from v3 since the imitation reward implicitly enforces
upright posture and proper gait:
- base_height: -5.0 → -2.0
- flat_orientation: -2.0 → -1.0
- joint_deviation_hips: removed (imitation handles hip movement)

Ground truth from Open Duck Playground placo_defaults.json (hardware-tuned):
- walk_com_height = 0.20 m
- walk_foot_height = 0.02 m
- walk_trunk_pitch = 0 degrees
- single_support_duration = 0.17 s
- feet_spacing = 0.16 m
"""

from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaac_lab_env.open_duck_mini_v2.imitation_reward import (
    ImitationReward,
    gait_phase_observation,
)
from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG


@configclass
class DuckRewards(RewardsCfg):
    """Reward function v4 for Open Duck Mini v2 locomotion.

    11 terms: 4 positive + 7 negative. No alive bonus.
    Imitation reward is the dominant positive signal (weight=10.0).

    Evolution from v3:
    - Added imitation_reward (+10.0) for reference motion tracking
    - Reduced feet_air_time (1.0→0.25, imitation already enforces stepping)
    - Reduced base_height (-5.0→-2.0, imitation enforces correct posture)
    - Reduced flat_orientation (-2.0→-1.0, imitation keeps upright)
    - Removed joint_deviation_hips (imitation handles hip movement)
    """

    # ====================================================================
    # POSITIVE REWARDS (4 terms)
    # ====================================================================

    # Imitation reward: DOMINANT positive signal.
    # Tracks reference leg joint positions from polynomial gait library.
    # exp(-2 * squared_error) over 10 leg joints. Weight 10.0 makes this
    # the strongest incentive — the policy earns most reward by matching
    # the reference walking gait.
    imitation_reward = RewTerm(
        func=ImitationReward,
        weight=10.0,
        params={"command_name": "base_velocity"},
    )

    # Velocity tracking: secondary objective (kept from v3).
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # Yaw tracking: tertiary objective (kept from v3).
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Biped gait: reward alternating foot lifting.
    # Reduced from v3's 1.0 — imitation reward already encourages stepping.
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

    # ====================================================================
    # NEGATIVE REWARDS — penalties (7 terms)
    # ====================================================================

    # --- Termination penalty ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # --- Posture control (lighter than v3 — imitation helps) ---

    # Height control: reduced from -5.0 since imitation enforces correct posture.
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.20},
    )

    # Orientation: reduced from -2.0 since imitation keeps robot upright.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Vertical bouncing penalty (kept from v3).
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    # --- Smoothness ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # --- Joint protection ---
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

    # --- Head stabilization ---
    # Head is 21% of body mass — uncontrolled flailing destabilizes the robot.
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

    # ====================================================================
    # DISABLED BASE-CLASS REWARDS
    # ====================================================================

    ang_vel_xy_l2 = None       # Redundant with flat_orientation_l2
    dof_torques_l2 = None      # Not needed at this stage
    dof_acc_l2 = None           # Not needed at this stage
    undesired_contacts = None   # Re-enable after verifying body names
    dof_pos_limits = None       # We define our own joint_pos_limits


@configclass
class OpenDuckRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Open Duck Mini v2 locomotion environment configuration.

    Flat-ground training with conservative velocity ranges.
    """

    rewards: DuckRewards = DuckRewards()

    # Camera close to robot for video recording.
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.0, 1.0, 0.5),
        lookat=(0.0, 0.0, 0.15),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        # --- Scene: swap robot asset ---
        self.scene.robot = OPEN_DUCK_MINI_V2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5

        # --- Simulation timing ---
        self.sim.dt = 0.005  # 200 Hz physics
        self.decimation = 4  # Policy at 50 Hz
        self.episode_length_s = 20.0

        # --- Observations: add gait phase ---
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.gait_phase = ObsTerm(func=gait_phase_observation)

        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/.*",
            history_length=3,
            track_air_time=True,
        )

        # --- Terrain: FLAT ground for initial training ---
        # Disable rough terrain generator. The robot must learn to walk
        # on flat ground before encountering obstacles.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # --- Commands: conservative for 42cm duck ---
        self.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # --- Action scale: 0.25 matching Open Duck Playground ---
        self.actions.joint_pos.scale = 0.25

        # --- Terminations ---
        self.terminations.base_contact.params["sensor_cfg"].body_names = (
            "trunk_assembly"
        )

        # --- Events ---
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
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

        # --- Disable terrain curriculum (we use flat ground) ---
        self.curriculum.terrain_levels = None


@configclass
class OpenDuckRoughEnvCfg_PLAY(OpenDuckRoughEnvCfg):
    """Playback configuration for evaluation."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.0, 1.0, 0.5),
        lookat=(0.0, 0.0, 0.15),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # Fixed forward walk command
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
