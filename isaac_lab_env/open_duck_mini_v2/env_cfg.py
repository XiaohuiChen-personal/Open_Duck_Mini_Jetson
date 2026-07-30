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

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from isaac_lab_env.open_duck_mini_v2 import contact_events, gated_rewards
from isaac_lab_env.open_duck_mini_v2.duck_commands import BandedWzVelocityCommandCfg
from isaac_lab_env.open_duck_mini_v2.imitation_reward import (
    ImitationReward,
    gait_phase_observation,
)
from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG

# --- v5 impulse-push ramp -------------------------------------------------
# Endpoint 0.7 m/s per axis is where three independent lines converge (see
# docs/jetson-mod/v5_retrain_plan.md section 11.4): it reproduces the upstream
# Open Duck Playground push distribution for THIS robot to within 3%
# (magnitude_range=[0.1, 1.0] applied as a uniformly-directed 2D magnitude vs
# Isaac Lab's independent per-axis box, for which E|dv| = a*0.7652), it matches
# the Froude-scaled median across seven published legged-RL configs
# (0.741 m/s), and it matches the capture-point limit for the robot's measured
# ~0.10-0.12 m leg reach at a 0.17 m CoM height. Task 2.8's written 1.3 m/s is
# amended: it would demand a 24 cm capture step on a robot whose CoM sits at
# 17 cm, i.e. unrecoverable by construction.
PUSH_START = 0.4
PUSH_END = 0.7
PUSH_RAMP_START_STEPS = 0
PUSH_RAMP_END_STEPS = 36_000  # ~1.5k iterations at 24 steps/iter


def ramp_push_velocity(env, env_ids, old_value, start_steps, end_steps, start_mag, end_mag):
    """Linearly ramp the planar push magnitude with training progress.

    Used as ``modify_fn`` for ``mdp.modify_env_param`` against the push event's
    ``velocity_range`` dict. Angular keys are left untouched — the rotational
    push is at full magnitude from the start, since it is the regime the
    benchmark falls came from and the policy already walks.
    """
    step = env.common_step_counter
    if step >= end_steps:
        frac = 1.0
    elif step <= start_steps:
        frac = 0.0
    else:
        frac = (step - start_steps) / max(1, end_steps - start_steps)
    mag = start_mag + frac * (end_mag - start_mag)
    new_value = dict(old_value)
    new_value["x"] = (-mag, mag)
    new_value["y"] = (-mag, mag)
    return new_value


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


# ======================================================================
# v5 CONTACT-RICH TRACK (Task 2.8)
# ======================================================================
#
# v4_robust is excellent in the world it was trained for — flat, empty, and
# disturbed only by instantaneous velocity kicks: gait gate 5/5, 0.00% unpushed
# falls, 6.84% pushed falls. The Duck Embody benchmark then put it in a
# furnished apartment and measured 10 falls in 12 trials (1.58 per
# policy-minute), split 7 rotation-under-contact / 1 free-space rotation /
# 2 sustained press. None of those three regimes exists anywhere in v4
# training. This track adds them; docs/jetson-mod/v5_retrain_plan.md carries
# the full rationale and the evidence for every constant below.


@configclass
class DuckContactRewards(DuckRewards):
    """v5 rewards: v3 composite, made survivable under contact.

    Four changes, each traceable to a measured failure mode:

    - the two tracking terms and the imitation composite are **disturbance
      gated**, so recovering from a shove is not scored as tracking error
      (Hartmann et al., ETH CRL 2024);
    - ``flat_orientation_l2`` gains a dead zone, because leaning into a
      sustained load is how a biped survives one;
    - ``ang_vel_xy_l2`` and ``feet_slide`` are switched on: every benchmark
      fall was a tip-over (``fell_low`` never once fired), and tilt rate is its
      earliest proprioceptive signature, while pressed robots skate their
      stance feet;
    - head/lower-leg ground contact is penalised, but **only where there is no
      obstacle** — see ``gated_rewards.ground_contact_penalty`` for why a
      blanket contact penalty would re-create the failure this task exists to
      remove.
    """

    track_lin_vel_xy_exp = RewTerm(
        func=gated_rewards.GatedTrackLinVel,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=gated_rewards.GatedTrackAngVel,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    imitation_reward = RewTerm(
        func=ImitationReward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "reference_pkl": "polynomial_coefficients_v2.pkl",
            "normalized_match": True,
            "disturbance_gate_scale": 0.1,
        },
    )

    # Replaced by the dead-zoned variant below.
    flat_orientation_l2 = None
    flat_orientation_deadzone = RewTerm(
        func=gated_rewards.flat_orientation_deadzone,
        weight=-1.0,
        params={"deadzone": 0.1},
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot_assembly", "foot_assembly_2"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["foot_assembly", "foot_assembly_2"]
            ),
        },
    )

    ground_contact = RewTerm(
        func=gated_rewards.ground_contact_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["head", "knee_and_ankle_assembly.*"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class OpenDuckContactEnvCfg(OpenDuckRobustEnvCfg):
    """v5 trainer: v4_robust + obstacles, sustained wrenches, sustained rotation.

    Inherits the whole v4-robust bundle (dynamics DR, 59-dim asymmetric actor,
    BAM velocity limit) so the observation/action contract — and therefore
    checkpoint compatibility for fine-tuning and the frozen Jetson deployment
    interface — is untouched.
    """

    rewards: DuckContactRewards = DuckContactRewards()

    def __post_init__(self):
        super().__post_init__()

        # --- Scene: one kinematic box per env -------------------------------
        # A single shared geometry keeps `replicate_physics=True`; per-env
        # variety would force it off (interactive_scene.py:57-68) at real cost.
        # 0.7 m matches the apartment's wall height and is taller than the
        # 0.42 m robot, so head, torso and shin contact are all reachable.
        self.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.7),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.3)),
            ),
            # Parked below the plane until ContactRegimeEvent places it.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
        )

        # --- Commands: sustained, banded rotation ---------------------------
        # 30% of envs hold a directly sampled wz (rather than the decaying
        # heading servo), 40% of those draws land in the killer band, sampling
        # reaches +/-0.7 so the deployment limit of 0.5 is interior, and the
        # heading servo stays deployment-exact at +/-0.5.
        self.commands.base_velocity = BandedWzVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 10.0),
            rel_standing_envs=0.08,
            rel_heading_envs=0.7,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            wz_band_frac=0.4,
            wz_band_abs=(0.35, 0.7),
            heading_wz_limit=0.5,
            ranges=BandedWzVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.148, 0.222),
                lin_vel_y=(-0.111, 0.111),
                ang_vel_z=(-0.7, 0.7),
                heading=(-math.pi, math.pi),
            ),
        )

        # --- Terminations: falling, not touching ----------------------------
        # v4 terminated on ANY trunk contact above 1 N with a -200 penalty, so
        # 3,000 iterations taught "contact = death" and never taught recovery.
        # v5 terminates on the deployment's own fall definition instead
        # (duck-embody embody_env_cfg.py:120-122): tilt 60 deg or height 0.09 m.
        self.terminations.base_contact = None
        self.terminations.tilt = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
        )
        self.terminations.fell_low = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.09}
        )

        # --- Events ---------------------------------------------------------
        # ContactRegimeEvent is the sole owner of the permanent wrench
        # composer, so the stock reset-mode wrench term must go.
        self.events.base_external_force_torque = None

        self.events.push_robot = EventTerm(
            func=contact_events.push_and_mark,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={
                "velocity_range": {
                    "x": (-PUSH_START, PUSH_START),
                    "y": (-PUSH_START, PUSH_START),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-1.0, 1.0),
                }
            },
        )

        self.events.contact_regime = EventTerm(
            func=contact_events.ContactRegimeEvent,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            is_global_time=True,
            params={
                "force_frac_range": (0.05, 0.30),
                "duration_range_s": (2.0, 6.0),
                "torque_z_range": (0.05, 0.15),
                "lateral_bias": 0.7,
                "active_frac": 0.5,
                "cooccurrence_frac": 0.3,
                "rotating_cmd_wz": 0.25,
                "obstacle_frac": 0.25,
                "obstacle_ahead_range": (0.3, 0.9),
                "obstacle_lateral_range": (0.10, 0.35),
                "obstacle_parked_z": -2.0,
                "command_name": "base_velocity",
            },
        )

        # --- Curriculum: ramp the impulse push -------------------------------
        # Hartmann withholds pushes until the tracking reward is strong,
        # warning that pushing too early "can prompt an excessive caution in
        # the agent". Fine-tuning starts from an already-competent policy, so a
        # step-indexed ramp is enough here; the scratch control delays it.
        self.curriculum.push_ramp = CurrTerm(
            func=mdp.modify_env_param,
            params={
                "address": "event_manager.cfg.push_robot.params.velocity_range",
                "modify_fn": ramp_push_velocity,
                "modify_params": {
                    "start_steps": PUSH_RAMP_START_STEPS,
                    "end_steps": PUSH_RAMP_END_STEPS,
                    "start_mag": PUSH_START,
                    "end_mag": PUSH_END,
                },
            },
        )


@configclass
class OpenDuckContactMinimalEnvCfg(OpenDuckRobustEnvCfg):
    """v5c: v4_robust plus EXACTLY two changes. The smallest real experiment.

    Why descend this far
    --------------------
    Measured double-support fraction (stance duty L + R - 100), same protocol:

        v4_robust  32%  -> gait gate 6/6   (real walking)
        v5b        62%  -> gait gate 3/6   (cautious shuffle, right foot drags)
        v5a        99%  -> gait gate 0/6   (standing)

    Each pressure removed from the curriculum moved the policy back toward
    walking, which says the remaining bundle is still too aggressive for a
    single fine-tune — the "excessive caution" local optimum Hartmann et al.
    explicitly warn about when disturbances arrive too early or too hard. The
    reference-library patch is exonerated: its synthesized cells carry the same
    70.4/66.7% reference duty and comparable L/R range of motion as the
    original cells, so the asymmetry is learned, not prescribed.

    So this arm keeps ONLY the two changes that address the measured benchmark
    failure, and reverts everything else to v4_robust:

      1. Trunk contact is no longer instant death — terminate on falling
         (tilt > 60 deg or height < 0.09 m), the deployment's own definition.
         This is the root cause of all 10 benchmark falls: 3,000 iterations
         taught "contact = death", never "contact = survivable".
      2. An obstacle exists, in 25% of episodes.

    Reverted vs v5a/v5b: no sustained wrench, no rotational pushes, v4's push
    magnitude and 8-14 s schedule, v4's heading-only +/-0.5 commands, v4's
    reward set (so the original frozen reference library), no extra penalties,
    no curriculum ramp. If this recovers v4's duty while surviving contact,
    every other lever in v5 was unnecessary complexity.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Change 1: falling ends the episode; touching things does not ---
        self.terminations.base_contact = None
        self.terminations.tilt = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
        )
        self.terminations.fell_low = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.09}
        )

        # --- Change 2: an obstacle exists ---
        self.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.7),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
        )
        # ContactRegimeEvent is carried only for obstacle placement and for the
        # in-training gait canary; the wrench is off (active_frac = 0) and the
        # gate it maintains is read by nothing in this cfg's reward set.
        self.events.base_external_force_torque = None
        self.events.contact_regime = EventTerm(
            func=contact_events.ContactRegimeEvent,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            is_global_time=True,
            params={
                "active_frac": 0.0,
                "obstacle_frac": 0.25,
                "obstacle_ahead_range": (0.3, 0.9),
                "obstacle_lateral_range": (0.10, 0.35),
                "obstacle_parked_z": -2.0,
                "command_name": "base_velocity",
                "contact_raises_gate": False,
            },
        )
        # Everything else — rewards, commands, pushes, reference library — is
        # v4_robust's, inherited untouched.


@configclass
class OpenDuckContactWrenchEnvCfg(OpenDuckContactMinimalEnvCfg):
    """v5d: the minimal arm plus ONE lever — the sustained wrench.

    Why exactly this lever, and only this one
    -----------------------------------------
    `v5c_contact_only` (v4 + fall-only terminations + obstacles) came back as a
    genuine candidate: gait gate 6/6, ref RMS 4.49 against v4's 4.48, falls
    0.00%, and duty asymmetry 1.1 pp — *better* than v4's 5.0 pp. Push recovery
    improved dramatically, 0.29% falls against v4's historical 6.84%.

    But it fell in **100% of episodes** on the sustained-wrench gate. That is
    not a surprise, it is a direct consequence of its own design: v5c trains
    with `active_frac = 0`, so it has never once felt a force held for
    2-6 seconds. Sustained pressing is 2 of the 10 benchmark falls and the
    mechanism behind the duck-embody S2 counter-press, so it cannot be left
    unaddressed.

    This arm therefore adds the wrench and nothing else. Everything v5c
    established stays fixed: v4's reward set and original reference library,
    heading-only +/-0.5 commands, v4's +/-0.3 m/s pushes on the 8-14 s
    schedule, no rotational pushes, no extra penalties, no curriculum ramp.
    If the gait gate survives, the wrench was the missing piece; if the gait
    degrades the way v5b's did, the wrench is what costs it, and that is worth
    knowing on its own.

    Force magnitudes are deliberately gentler than v5a/v5b used (5-20% of body
    weight rather than 5-30%): Hartmann et al. apply 10 N and 20 N to a 117.7 N
    Go1, i.e. 8.5% and 17%, and the whole lesson of v5a/v5b is that this
    curriculum tips into "excessive caution" when disturbances are too strong.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.contact_regime.params.update(
            {
                "active_frac": 0.5,
                "force_frac_range": (0.05, 0.20),
                "duration_range_s": (2.0, 6.0),
                "torque_z_range": (0.05, 0.15),
                "lateral_bias": 0.7,
                "cooccurrence_frac": 0.3,
                "rotating_cmd_wz": 0.25,
            }
        )


@configclass
class OpenDuckContactWrenchEnvCfg_PLAY(OpenDuckContactWrenchEnvCfg):
    """Deterministic playback twin of the wrench arm."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.contact_regime.params["active_frac"] = 0.0
        self.events.contact_regime.params["obstacle_frac"] = 0.0


@configclass
class OpenDuckContactMinimalEnvCfg_PLAY(OpenDuckContactMinimalEnvCfg):
    """Deterministic playback twin of the minimal arm."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.contact_regime.params["obstacle_frac"] = 0.0


@configclass
class DuckContactUngatedRewards(DuckContactRewards):
    """v5b rewards: the contact curriculum WITHOUT the disturbance gate.

    Why this arm exists, and why it is now the mainline rather than an ablation
    ---------------------------------------------------------------------------
    Run ``v5a_gated_ft`` (2026-07-28) trained beautifully and evaluated as a
    standing policy: gait gate **0/6**, stance duty 99.7/99.6% (both feet down
    essentially always), energy 11.1 W vs v4's 20.9 W, wz error 0.575 rad/s.
    The v4_robust control on the identical task scored 6/6 at 68.7/63.7% duty,
    so the harness was sound and the regression was real.

    The mechanism is legible in the training logs: trunk-contact duty reached
    50%, and trunk contact was a gate trigger, so gate duty hit 83%. With the
    tracking terms frozen and imitation scaled to x0.1 for 83% of steps, the
    episode-average imitation reward collapsed to +0.008 — the only term that
    forces a real gait was effectively deleted, leaving the +10 alive bonus as
    the dominant objective. Standing maximises that.

    The deeper lesson: "gate only on external events" is not enough once the
    world contains obstacles, because the policy can go and *acquire* contact.
    The trigger is now off by default (``contact_raises_gate=False``), but this
    arm removes the gate altogether — after a mechanism produces a degenerate
    optimum, deleting it is a better next experiment than tuning its constants.
    Everything else in the contact curriculum (obstacles, sustained wrenches,
    rotational pushes, banded wz, fall-only terminations, patched reference
    library) is unchanged, so this run tests whether the curriculum works
    without the one novel reward mechanism.
    """

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    imitation_reward = RewTerm(
        func=ImitationReward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "reference_pkl": "polynomial_coefficients_v2.pkl",
            "normalized_match": True,
            # No gating: the imitation prior is the only term that demands a
            # real gait, and v5a showed what happens when it is switched off.
            "disturbance_gate_scale": None,
        },
    )


@configclass
class OpenDuckContactUngatedEnvCfg(OpenDuckContactEnvCfg):
    """v5b trainer: contact curriculum, no disturbance gating."""

    rewards: DuckContactUngatedRewards = DuckContactUngatedRewards()


@configclass
class OpenDuckContactUngatedEnvCfg_PLAY(OpenDuckContactUngatedEnvCfg):
    """Deterministic playback twin of the v5b trainer."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.wz_band_frac = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.contact_regime.params["active_frac"] = 0.0
        self.events.contact_regime.params["obstacle_frac"] = 0.0
        self.curriculum.push_ramp = None


@configclass
class OpenDuckContactEnvCfg_PLAY(OpenDuckContactEnvCfg):
    """Deterministic playback for the v5 track (no disturbances, no obstacle)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.wz_band_frac = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # Keep the regime term (it owns the gate the rewards read) but disable
        # every disturbance it can produce.
        self.events.contact_regime.params["active_frac"] = 0.0
        self.events.contact_regime.params["obstacle_frac"] = 0.0
        self.curriculum.push_ramp = None


@configclass
class OpenDuckWrenchEvalEnvCfg(OpenDuckContactEnvCfg_PLAY):
    """Sustained-press gate: deterministic playback with the wrench ON.

    The scripted furniture proxy of plan gate 5 — a constant lateral force held
    2-6 s while the robot tracks its command, including at max rotation rate.
    Force fraction is overridden per tier (0.1 / 0.2 / 0.3 x body weight).
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.contact_regime.params["active_frac"] = 0.9
        self.events.contact_regime.params["force_frac_range"] = (0.2, 0.2)
        self.events.contact_regime.params["cooccurrence_frac"] = 0.0


@configclass
class OpenDuckObstacleEvalEnvCfg(OpenDuckContactEnvCfg_PLAY):
    """Obstacle-graze gate: box in play at grazing offset, wrench off.

    Exists because no other eval condition puts the policy against real
    geometry — the "compliant slide along the obstacle" item of the frame-by
    -frame checklist had nothing to audit without it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.contact_regime.params["obstacle_frac"] = 1.0
        self.events.contact_regime.params["obstacle_lateral_range"] = (0.10, 0.15)


@configclass
class OpenDuckContactPushEvalEnvCfg(OpenDuckContactEnvCfg_PLAY):
    """Push eval under v5's own fall definition.

    The v4 ``PushEval`` task counts any trunk contact above 1 N as a fall,
    which v5 is explicitly trained to treat as survivable. Both numbers get
    reported: the v4-comparable one (on the v4 task, for continuity with the
    6.84% baseline) and this one, so a gate miss can be attributed to the
    definition rather than to genuine falls.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=contact_events.push_and_mark,
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
