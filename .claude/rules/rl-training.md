# Reinforcement Learning Training

## RL Overview for This Project

The robot learns to walk through trial-and-error in simulation (Isaac Sim). A neural network "policy" observes the robot's state and outputs motor commands. PPO is the primary algorithm.

## Observation Space (56 dimensions for AWD policy)

| Component | Dimensions | Description |
|---|---|---|
| Projected gravity | 3 | Gravity direction in robot frame (from IMU) |
| Joint positions | 16 | Current angle of each joint (rad) |
| Joint velocities | 16 | Current speed of each joint (rad/s) |
| Feet contacts | 2 | Binary: is left/right foot touching ground? |
| Previous action | 16 | Last motor command sent |
| Velocity command | 3 | Desired (vx, vy, yaw_rate) from user or VLM |

## Action Space (16 dimensions)

Joint position targets for all 16 actuators. Scaled by `action_scale` (typically 0.25-0.5) and offset by `init_pos`.

## Joint Order (MuJoCo convention)

```
0: right_hip_yaw     5: left_hip_yaw     10: neck_pitch
1: right_hip_roll    6: left_hip_roll     11: head_pitch
2: right_hip_pitch   7: left_hip_pitch    12: head_yaw
3: right_knee        8: left_knee         13: head_roll
4: right_ankle       9: left_ankle        14: left_antenna
                                          15: right_antenna
```

Note: Isaac Gym uses a different joint order (left leg first). See `mini_bdx/mini_bdx/utils/rl_utils.py` for conversion functions.

## RL Algorithms (Phase 2)

Training is done on DGX Spark with Isaac Lab. The environment extends `LocomotionVelocityRoughEnvCfg` (H1 biped pattern).

| Algorithm | Framework | Type | Status |
|---|---|---|---|
| **PPO** | RSL-RL | On-policy | Primary — built-in ONNX export for Jetson |
| **AMP** | SKRL | On-policy + imitation | Optional stretch goal — requires DirectRLEnv + reference motions |

Note: The Isaac Lab SKRL training script only supports `--algorithm PPO` and `--algorithm AMP`. Other algorithms (SAC, TRPO, RPO, TD3) would require custom training scripts.

## Reward Functions

Inherited from `LocomotionVelocityRoughEnvCfg` base + biped-specific overrides (following H1/G1/Digit patterns):

- `track_lin_vel_xy_yaw_frame_exp`: Exponential reward for tracking commanded velocity in yaw frame
- `track_ang_vel_z_world_exp`: Exponential reward for tracking angular velocity
- `feet_air_time_positive_biped`: Reward alternating single-stance phases (biped gait)
- `feet_slide`: Penalize feet sliding on ground
- `flat_orientation_l2`: Penalize non-upright orientation
- `action_rate_l2`: Penalize jerky actions (smoothness)
- `joint_deviation_l1` (head): Penalize head/antenna deviation from default
- `joint_deviation_l1` (hips): Penalize hip yaw/roll deviation — conserves servo torque
- `joint_pos_limits`: Penalize ankle/knee joints approaching position limits — protects servos
- `ang_vel_xy_l2`: Penalize pitch/roll angular velocity (increased to -0.1 for top-heavy trunk)
- `joint_acc_l2`: Penalize joint accelerations
- `is_terminated`: Strong penalty (-200) for falling

Conditional (add after initial training if needed):
- `base_height_l2`: Add if policy learns excessively crouched gait (target ~0.15-0.17m)
- `stand_still_joint_deviation_l1`: Add if robot can't stand still on zero velocity command

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

This records a 200-step clip every 2000 training steps. Videos are saved to `logs/<workflow>/<task>/<run>/videos/train/`. Use `play.py --video` after training for clean evaluation clips of the final policy.

## Training → Deployment Pipeline

```
Isaac Lab (DGX Spark) → .pt checkpoint → ONNX export → TensorRT engine → Jetson inference
```

## Key Files

- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — Environment config
- `isaac_lab_env/open_duck_mini_v2/train_cfg.py` — Training hyperparameters
- `exported_policies/*.onnx` — Exported policies
- `experiments/v2/params_sts3250_id008.json` — BAM motor identification parameters (STS3250)
- `experiments/v2/params_m6.json` — Legacy BAM motor parameters (STS3215, for reference)
- `mini_bdx/mini_bdx/utils/rl_utils.py` — Joint order conversion, action scaling
