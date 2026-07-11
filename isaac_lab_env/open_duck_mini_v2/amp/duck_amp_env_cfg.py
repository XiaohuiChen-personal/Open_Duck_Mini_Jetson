# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP environment configurations for the Open Duck Mini v2.

Adapted from the Isaac Lab template at
``isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env_cfg.py`` with
duck-specific values throughout.

AMP observation frame (51 dims), built by ``duck_amp_env.compute_obs``::

    16  joint positions
    16  joint velocities
     1  root height (z)
     6  root rotation as tangent + normal vectors
     3  root linear velocity
     3  root angular velocity
     6  key body positions relative to root (2 feet x 3)
    --
    51

Key deltas vs the template, all deliberate:

- ``sim.dt=0.005`` with ``decimation=4`` gives a 50 Hz policy step that
  EQUALS the reference-motion frame rate (50 fps). The template runs the
  policy at 30 Hz but spaces AMP observation history at the motion dt
  (1/60 s), so sim-collected frame pairs and dataset frame pairs have
  different time spacing — a systematic gap the discriminator can exploit.
  Matching the rates eliminates that mismatch by construction.
- ``robot = OPEN_DUCK_MINI_V2_CFG`` is reused as-is: it carries the
  BAM-identified STS3250 actuator model (kp=45.53, kd=1.346) that the
  PPO pipeline already validated. The template instead zeroes stiffness/
  damping and drives raw torques — wrong for position-controlled servos.
- ``termination_height=0.10``: the duck's base sits at ~0.17 m when
  standing; below 0.10 m it has irrecoverably collapsed.
- ``heading_localize_amp_obs=True``: reference clips walk along +x, but
  the robot must be free to walk in any world heading. Rotating root
  velocities and orientation into the yaw-heading frame makes the AMP
  observation heading-invariant.

Two concrete tasks are derived from the base config:

- :class:`DuckAmpPureStyleEnvCfg` — pure style reward (the skrl agent uses
  task_reward_weight=0.0), single forward-walk clip, no velocity commands.
  Sanity-check task: "can AMP make the duck walk like the reference at all?"
- :class:`DuckAmpCommandEnvCfg` — velocity-command tracking task reward mixed
  50/50 with style reward, full clip library, 3-dim command appended to the
  POLICY observation only (never to the AMP observation — the discriminator
  must judge style, not intent).
"""

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class DuckAmpEnvCfgBase(DirectRLEnvCfg):
    """Open Duck Mini v2 AMP environment config (base class)."""

    # env
    episode_length_s = 20.0
    decimation = 4  # 200 Hz physics / 4 = 50 Hz policy = motion fps

    # action scaling: target = default_joint_pos + action_scale * action
    # (duck/playground convention, matching the PPO pipeline and deployment)
    action_scale = 0.25
    # raw actions are clipped to +/-action_clip before scaling (run-12 fix):
    # bounds the effective joint-target offset to +/-1.25 rad (covers the full
    # reference gait ROM) and prevents the run-11 action-magnitude divergence.
    action_clip = 5.0

    # spaces
    observation_space = 51
    action_space = 16
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 51

    early_termination = True
    termination_height = 0.10

    # Reference motion clips (glob patterns, expanded by the env at startup).
    motion_files: list[str] = [os.path.join(MOTIONS_DIR, "*.npz")]
    reference_body = "base"
    key_body_names: list[str] = ["foot_assembly", "foot_assembly_2"]
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (duck's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # Rotate root velocities and orientation into the yaw-heading frame when
    # building AMP observations, making them invariant to world heading.
    heading_localize_amp_obs = True

    # velocity commands (disabled in the base/pure-style config)
    include_command_obs = False
    """If True, a 3-dim (vx, vy, wz) command is appended to the POLICY observation
    (never the AMP observation) and the task reward tracks it."""
    command_vx_range: tuple[float, float] = (-0.148, 0.222)
    command_vy_range: tuple[float, float] = (-0.111, 0.111)
    command_wz_range: tuple[float, float] = (-0.5, 0.5)
    """Command ranges match the reference-motion grid hull of the Open Duck
    Playground gait library (see env_cfg.py v3 notes) — commanding beyond the
    library would ask for styles the discriminator has never seen."""
    command_resampling_time_s = 10.0

    # simulation — 200 Hz physics, matching the PPO pipeline (isaac-lab.md)
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene — 4096 envs on DGX Spark; 2.5 m spacing is plenty for a 42 cm robot
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot — reuse the validated articulation config as-is (BAM STS3250
    # actuators: kp=45.53, kd=1.346, armature=0.040, effort_limit=8.716 Nm)
    robot: ArticulationCfg = OPEN_DUCK_MINI_V2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Contact sensor: not consumed by training (the template env has none);
    # present so evaluate_policies.py can compute the same contact-based gait
    # metrics (stance duty, asymmetry) for AMP policies as for PPO ones.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class DuckAmpPureStyleEnvCfg(DuckAmpEnvCfgBase):
    """Pure style imitation: single forward-walk clip, no velocity commands.

    Reward is constant 1.0 in the env; the skrl AMP agent runs with
    task_reward_weight=0.0 / style_reward_weight=1.0, so learning is driven
    entirely by the discriminator.
    """

    observation_space = 51
    include_command_obs = False
    # The near-forward gait clip (vx=0.148, smallest |vy| and |wz| in the
    # library) exported by scripts/convert_gait_library_to_amp.py.
    motion_files: list[str] = [os.path.join(MOTIONS_DIR, "duck_gait_0.148_-0.037_-0.074.npz")]


@configclass
class DuckAmpCommandEnvCfg(DuckAmpEnvCfgBase):
    """Command-conditioned AMP: velocity tracking task + style from full library.

    The policy observes 51 AMP dims + 3 command dims = 54. The skrl AMP agent
    runs with task_reward_weight=0.5 / style_reward_weight=0.5, mixing the
    env's velocity-tracking reward with the discriminator's style reward.
    """

    observation_space = 54  # 51 AMP dims + 3-dim velocity command (policy obs only)
    include_command_obs = True
    motion_files: list[str] = [os.path.join(MOTIONS_DIR, "*.npz")]
    # Run-9 lever: 4-frame AMP history (80 ms window) — with 2 frames the
    # engaged-but-fooled discriminator (run 8: loss 1.4-1.7) could not
    # distinguish a 2-8 deg shuffle from the ~30 deg reference waddle.
    # Run-10: 14 frames = 280 ms > half gait cycle (0.27 s) — alternation is a
    # cycle-level statistic; shorter windows make one-legged strides
    # style-optimal (run 9). Discriminator input 14 x 51 = 714 dims.
    num_amp_observations = 14


@configclass
class DuckAmpVideoEnvCfg(DuckAmpCommandEnvCfg):
    """Video-audit variant of the command env (evaluation protocol step 2).

    Adds the PPO play tracking camera and PINS the velocity command via
    degenerate ranges (resampling then always redeals the same command).
    Used by scripts/play_amp.py through ``Isaac-OpenDuck-AMP-Video-v0``.
    Defaults render the forward condition (vx=0.2); override per condition,
    e.g. the turn condition:
    ``'env.command_vx_range=[0.0,0.0]' 'env.command_wz_range=[0.3,0.3]'``.
    num_amp_observations must match the checkpoint (2 for runs 4-8 era).
    """

    # Same robot-tracking camera as the PPO play tasks (env_cfg.py).
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.0, 1.0, 0.5),
        lookat=(0.0, 0.0, 0.15),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    command_vx_range: tuple = (0.2, 0.2)
    command_vy_range: tuple = (0.0, 0.0)
    command_wz_range: tuple = (0.0, 0.0)
