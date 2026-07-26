# Reinforcement Learning Training

## RL Overview for This Project

The robot learns to walk through trial-and-error in simulation (Isaac Sim). A neural network "policy" observes the robot's state and outputs motor commands. PPO is the primary algorithm, with reference motion imitation as the dominant reward signal.

## Observation Space

Standard Isaac Lab locomotion observations plus gait phase:

| Component | Dimensions | Description |
|---|---|---|
| Base linear velocity | 3 | Inherited from `LocomotionVelocityRoughEnvCfg` — NOT directly measurable by the BNO055 on hardware (deployment gap; needs an estimator or a retrain without it) |
| Base angular velocity | 3 | Gyro (measurable on hardware) |
| Projected gravity | 3 | Gravity direction in robot frame (from IMU) |
| Joint positions | 16 | Current angle of each joint (rad) |
| Joint velocities | 16 | Current speed of each joint (rad/s) |
| Previous action | 16 | Last motor command sent |
| Velocity command | 3 | Desired (vx, vy, yaw_rate) from user or VLM |
| Gait phase | 2 | [cos(phase), sin(phase)] of gait cycle |

**Total: 62 dims** (verified against `exported_policies/v3_bdx_imitation_ppo/env.yaml`
— the ONNX exporter's OBS_LAYOUT_62 that mirrored it was removed with the AMP track; RSL-RL exports ONNX natively).

## Action Space (16 dimensions)

Joint position targets for all 16 actuators. Scaled by `action_scale` (0.25, matching Open Duck Playground) and offset by `init_pos`.

## Joint Orders

**MuJoCo / MJCF order** (joint declaration AND actuator order in
`robot.xml` and `robot_motors.xml` — verified; identical to the Playground
polynomial order below):
```
0: left_hip_yaw      8: head_roll
1: left_hip_roll     9: left_antenna
2: left_hip_pitch   10: right_antenna
3: left_knee        11: right_hip_yaw
4: left_ankle       12: right_hip_roll
5: neck_pitch       13: right_hip_pitch
6: head_pitch       14: right_knee
7: head_yaw         15: right_ankle
```
WARNING: an earlier version of this table listed a right-leg-first order —
that was the legacy 15-joint BDX layout from `rl_utils.py`, NOT this model.
Never hardcode a joint order from documentation; derive mappings from joint
names (as `imitation_reward.py` does) or assert against the model file
(as the archived `convert_gait_library_to_amp.py` did).

**Isaac Lab / USD order** (from MJCF→USD conversion, interleaved):
```
0: left_hip_yaw      8: right_hip_pitch
1: neck_pitch        9: left_knee
2: right_hip_yaw    10: head_roll
3: left_hip_roll    11: right_knee
4: head_pitch       12: left_ankle
5: right_hip_roll   13: left_antenna
6: left_hip_pitch   14: right_antenna
7: head_yaw         15: right_ankle
```

**Playground polynomial order** (used in polynomial_coefficients.pkl):
```
0: left_hip_yaw      8: head_roll
1: left_hip_roll     9: left_antenna
2: left_hip_pitch   10: right_antenna
3: left_knee        11: right_hip_yaw
4: left_ankle       12: right_hip_roll
5: neck_pitch       13: right_hip_pitch
6: head_pitch       14: right_knee
7: head_yaw         15: right_ankle
```

`mini_bdx/mini_bdx/utils/rl_utils.py` contains MuJoCo↔IsaacGym conversion tables for the LEGACY 15-joint BDX robot (no head_roll) — do not reuse them for this 16-joint model. The Isaac Lab imitation reward (`imitation_reward.py`) builds the Playground↔Isaac Lab mapping dynamically from joint names.

## RL Algorithm

PPO via RSL-RL on DGX Spark. 4096 parallel envs. Environment extends `LocomotionVelocityRoughEnvCfg`.

## Reward Functions (v3 — corrected BDX-aligned composite imitation)

SOURCE OF TRUTH: `isaac_lab_env/open_duck_mini_v2/env_cfg.py` (DuckRewards)
and `imitation_reward.py`. 9 terms: 4 positive + 5 penalties.

**Positive rewards:**
- `alive_bonus` (mdp.is_alive): weight=+10.0 (BDX paper uses +20)
- `imitation_reward` (ImitationReward class): RewTerm weight=1.0, BDX
  sub-weights baked in: joint_pos -L2*15.0 (raw quadratic), joint_vel
  -L2*0.001, base_vel exp(-8e)*1.0, contact match *1.0. Reference evaluated
  at NORMALIZED phase t=(i % nb_steps)/nb_steps (the v2 phase bug imitated
  only 54% of the cycle — see experiment_journal.md run 1); reference
  clamped to soft joint limits; gated off for near-zero commands.
- `track_lin_vel_xy_exp`: command tracking (std=0.5, weight=1.0)
- `track_ang_vel_z_world_exp`: yaw tracking (std=0.5, weight=0.5)

**Penalties:**
- `is_terminated`: fall penalty (-200)
- `flat_orientation_l2`: stay upright (-2.0)
- `action_rate_l2`: smoothness (-1.0)
- `joint_pos_limits`: servo protection for ankle/knee (-1.0)
- `joint_deviation_head`: head stabilization (-0.1) — head is 21% of body mass

**Reward evolution** (full history in `docs/jetson-mod/experiment_journal.md`):
- iter1: H1-derived, 13 penalties → structurally negative, failed
- iter2: alive bonus +5 → crouching/shuffling exploit, failed
- iter3: no alive bonus + height control → unnatural gait
- iter4 (= policy "v1"): exp-kernel imitation ×10 + phase obs → walks
- v2 (BDX-aligned): raw-quadratic composite — but the phase-in-seconds bug
  produced a measurable limp (stance asym 14.6pp, ROM ratio 0.82)
- v3 (current): phase fix + clamp + gating + command clip → G1 PASS
  (stance asym 4.3pp, ROM ratio 1.02, 4.59° RMS vs true reference)

Velocity command ranges (clipped to the reference-motion grid hull):
- lin_vel_x: (-0.148, 0.222) m/s
- lin_vel_y: (-0.111, 0.111) m/s
- ang_vel_z: (-0.5, 0.5) rad/s

Action scale: 0.25 (matching Open Duck Playground)

## v4 Tracks (post-CAD retrain)

- **Run A ("v4-inertials")**: task `Isaac-Velocity-Rough-OpenDuck-v0`
  unchanged — retrains the v3 recipe on the layout-v2.1 model (new masses,
  frame-correct fullinertia, cut meshes). Isolates the model-change effect.
- **Run B ("v4-robust")**: task `Isaac-Velocity-Rough-OpenDuck-Robust-v0`
  (`OpenDuckRobustEnvCfg` + `OpenDuckRobustPPORunnerCfg`):
  - dynamics DR: pushes (±0.3 m/s, 8-14 s), trunk mass ±(-0.10,+0.15) kg,
    trunk CoM ±10/±5 mm, friction 0.4-1.0/0.3-0.8, joint-reset scale 0.9-1.1
  - asymmetric obs: actor = 59 dims (NO base_lin_vel — the BNO055 cannot
    measure it), critic = 62 dims privileged, uncorrupted
    (`obs_groups={"actor": ["policy"], "critic": ["critic"]}`)
  - actuator velocity_limit_sim = 8.94 rad/s (BAM sts3250 id008)
- **Push-recovery gate**: task `Isaac-OpenDuck...PushEval-v0`
  (`OpenDuckPushEvalEnvCfg`) — play determinism with interval pushes ON;
  feeds docs/jetson-mod/validation_results.md (Task 2.7).
- Deployed ONNX for Run B consumes the 59-dim actor layout (v3's was 62).
  Surviving term order (the Jetson obs builder must emit exactly this):
  base_ang_vel(3), projected_gravity(3), velocity_commands(3),
  joint_pos(16), joint_vel(16), actions(16), gait_phase(2).

## AMP Track (removed 2026-07-26)

The skrl AMP track (envs, motion clips, train/play/measure/export scripts)
was removed with the course-study archive. Full implementation + run history:
archive repo `open-duck-ppo-vs-amp` (in-repo history at tag
`course-study-freeze`). Study run entries remain in the experiment journal
as historical record.

## PPO Hyperparameters

- gamma: 0.97
- entropy_coef: 0.005
- init_noise_std: 0.5
- obs_normalization: True (actor and critic)

## Actuator Configuration (STS3250)

```python
stiffness = 45.53     # kp from BAM (STS3250 id008)
damping = 1.346       # kd from BAM (STS3250 id008)
armature = 0.040      # From BAM id008
friction = 0.200      # frictionloss from BAM id008
effort_limit = 8.716  # Torque limit in Nm (BAM forcerange)
```

## Policy Network Architecture

MLP with 3 hidden layers: [512, 256, 128], ELU activation. Separate policy and value networks.

## Training Video Recording

Always record training progress videos for debugging and documentation. Use these flags on all training runs:

```bash
--video --video_length 200 --video_interval 5000
```

Videos saved to `logs/<workflow>/<task>/<run>/videos/train/`. Use `play.py --video` for evaluation clips.

## Training → Deployment Pipeline

```
Isaac Lab (DGX Spark) → .pt checkpoint → ONNX export → TensorRT engine → Jetson inference
```

## Key Files

- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — Environment config (rewards, observations, terrain)
- `isaac_lab_env/open_duck_mini_v2/imitation_reward.py` — ImitationReward class + gait_phase_observation
- `isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl` — 240 polynomial walking gaits
- `isaac_lab_env/open_duck_mini_v2/agents/rsl_rl_ppo_cfg.py` — PPO hyperparameters
- `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — Robot articulation config
- `exported_policies/*.onnx` — Exported policies
- `experiments/v2/params_sts3250_id008.json` — BAM motor identification parameters (STS3250)
- `mini_bdx/mini_bdx/utils/rl_utils.py` — Joint order conversion, action scaling
