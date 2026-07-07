# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab locomotion environment for the Open Duck Mini v2.

Reward design v3 — BDX-aligned composite imitation reward, corrected.

Based on the Disney BDX paper ("Design and Control of a Bipedal Robotic
Character", Jan 2025) and the Open Duck Playground reward structure.

v3 changes from v2 (the run archived in exported_policies/v2_bdx_imitation_ppo):
- PHASE BUG FIX in imitation_reward.py: the polynomial reference is now
  evaluated at normalized phase t in [0, 1) instead of seconds in [0, 0.54)
  — v2 imitated only the first 54% of the gait cycle (an asymmetric limp).
  See imitation_reward.py module docstring for the full story.
- Reference joint positions clamped to soft joint limits (knee swing peaks
  in the library exceed the model's limits).
- Imitation reward gated to zero for near-zero commands (upstream parity).
- Command ranges clipped to the reference-motion grid hull: the library
  covers vx in [-0.148, 0.222], vy in [-0.111, 0.111] — commanding beyond
  it silently saturates the nearest-motion match.
- track_lin_vel_xy_exp is now declared EXPLICITLY: v2's docstring claimed it
  was removed, but the silently-inherited base term remained active during
  the v2 run (confirmed in the archived env.yaml). It is kept deliberately:
  it tracks the commanded velocity while the imitation base-vel term tracks
  the reference gait's instantaneous velocity profile.

v2 changes from v1 (kept):
- Raw quadratic joint tracking (BDX weight 15.0) instead of exp(-2*L2)*10.0
- Joint velocity / base velocity / foot contact terms from polynomial
  dims 16-31 / 34-36 / 32-33
- Alive bonus +10.0; action_rate -1.0; flat_orientation -2.0
- Removed base_height, lin_vel_z, feet_air_time
"""

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from isaac_lab_env.open_duck_mini_v2.imitation_reward import (
    ImitationReward,
    gait_phase_observation,
)
from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG


@configclass
class DuckRewards(RewardsCfg):
    """Reward function v3 — BDX-aligned composite imitation, corrected.

    9 terms: 4 positive + 5 penalties (the imitation composite is itself
    net-negative raw quadratic, offset by the alive bonus). Follows the
    Disney BDX paper (Table I) and the Open Duck Playground weights.
    """

    # ====================================================================
    # POSITIVE REWARDS (4 terms)
    # ====================================================================

    # Alive bonus: +10.0 per step. BDX uses +20.0.
    # Safe with strong imitation signal — policy can't earn reward by
    # crouching because imitation demands precise joint matching.
    alive_bonus = RewTerm(func=mdp.is_alive, weight=10.0)

    # BDX-style composite imitation reward (weight=1.0 since BDX weights
    # are baked into the class: joint_pos=15.0, joint_vel=0.001,
    # base_vel=1.0, contacts=1.0).
    imitation_reward = RewTerm(
        func=ImitationReward,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )

    # Yaw tracking from velocity command (BDX weight 0.5).
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Commanded-velocity tracking. Declared explicitly in v3: this base-class
    # term was silently active during the v2 run (same func/weight/std as the
    # inherited default) and is kept deliberately — it tracks the COMMAND
    # while the imitation base-vel sub-term tracks the reference gait's
    # instantaneous (waddling) velocity profile. Playground keeps both too.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ====================================================================
    # NEGATIVE REWARDS — penalties (5 terms)
    # ====================================================================

    # Termination penalty.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Action smoothness: -1.0 (BDX uses -1.5, Playground uses -0.5).
    # Increased 200x from v1's -0.005 — critical for natural motion.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    # Orientation: -2.0 (doubled from v1 to counteract forward lean).
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # Joint limits: protect servos.
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

    # Head stabilization (21% of body mass).
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

    ang_vel_xy_l2 = None
    dof_torques_l2 = None
    dof_acc_l2 = None
    undesired_contacts = None
    dof_pos_limits = None
    lin_vel_z_l2 = None        # Handled by imitation base_vel tracking
    feet_air_time = None       # Handled by imitation contact matching


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

        # --- Commands: clipped to the reference-motion grid hull ---
        # The polynomial library covers vx in [-0.148, 0.222] and
        # vy in [-0.111, 0.111]; commands beyond that saturate the
        # nearest-motion match (the gait can't follow them anyway).
        # ang_vel_z +/-0.5 is well inside the library's +/-1.222.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.148, 0.222)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.111, 0.111)
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
class DuckRobustObservationsCfg(ObservationsCfg):
    """Asymmetric observations for the v4-robust track.

    The actor group drops ``base_lin_vel`` (the BNO055 cannot measure base
    linear velocity on hardware — the 62-dim v3 policy has an unfixable
    deployment gap); the critic keeps the full privileged set. Groups are
    mapped to the algorithm via ``obs_groups`` in
    :class:`agents.rsl_rl_ppo_cfg.OpenDuckRobustPPORunnerCfg`.
    """

    @configclass
    class CriticCfg(ObservationsCfg.PolicyCfg):
        """Privileged critic observations: full set, no corruption."""

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    critic: CriticCfg = CriticCfg()


@configclass
class OpenDuckRobustEnvCfg(OpenDuckRoughEnvCfg):
    """v4-robust training config (Run B): v3 + dynamics DR + asymmetric obs.

    One consolidated robustness bundle on top of the v3 recipe (which Run A
    re-trains unchanged on the corrected model):

    - dynamics domain randomization (v3 trained at a single dynamics point:
      push/mass/CoM events were disabled and friction was fixed 0.8/0.6)
    - asymmetric actor/critic observations (actor loses base_lin_vel)
    - BAM-measured joint velocity limit (8.94 rad/s, params_sts3250_id008)

    DR ranges are duck-scaled (2.66 kg robot): literature ranges for
    human-scale robots (e.g. the parent cfg's +/-5 kg base mass) would be
    absurd here.
    """

    observations: DuckRobustObservationsCfg = DuckRobustObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- Asymmetric observations ---
        # Actor: drop base_lin_vel (unmeasurable on hardware); keep the rest.
        self.observations.policy.base_lin_vel = None
        # Critic mirrors the policy group setup (height_scan off, gait phase on).
        self.observations.critic.height_scan = None
        self.observations.critic.gait_phase = ObsTerm(func=gait_phase_observation)

        # --- Dynamics domain randomization ---
        # NOTE: the conversion's root body ("base") is massless — all
        # body-targeted DR must point at trunk_assembly.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 14.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
                "mass_distribution_params": (-0.10, 0.15),
                "operation": "add",
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
                "com_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.005, 0.005),
                },
            },
        )
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        # enforce dynamic <= static per bucket (independent sampling would
        # otherwise produce non-physical dynamic > static combinations)
        self.events.physics_material.params["make_consistent"] = True
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        # --- Actuator realism: BAM-measured max servo speed ---
        for actuator in self.scene.robot.actuators.values():
            actuator.velocity_limit_sim = 8.94


@configclass
class OpenDuckRobustEnvCfg_PLAY(OpenDuckRobustEnvCfg):
    """Playback/evaluation configuration for the robust track (no DR)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


@configclass
class OpenDuckPushEvalEnvCfg(OpenDuckRobustEnvCfg_PLAY):
    """Push-recovery evaluation (the Task 2.7 gate that was never run).

    Playback determinism EXCEPT interval pushes stay enabled — measures
    whether the policy survives lateral/longitudinal shoves while tracking
    a forward command. Results feed docs/jetson-mod/validation_results.md.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )


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


@configclass
class OpenDuckPlainPushEvalEnvCfg(OpenDuckRoughEnvCfg_PLAY):
    """Push-recovery eval for the PLAIN (62-dim, no-DR) policy track.

    Identical push schedule to :class:`OpenDuckPushEvalEnvCfg` but on the
    plain v3-recipe task, so a policy trained WITHOUT domain randomization
    (Run A / v4_inertials) can be measured under the same pushes as the DR
    policy (Run B / v4_robust) — isolating the robustness contribution of
    the DR (Task 2.7 comparison). Requires evaluate_policies.py --keep-pushes.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )
