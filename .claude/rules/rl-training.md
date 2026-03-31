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

Joint position targets for all 16 actuators. Scaled by `action_scale` (0.25, matching Open Duck Playground) and offset by `init_pos`.

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

Redesigned from first principles based on Disney BDX paper and Open Duck Playground.
8 reward terms (4 positive, 4 negative). Removed all H1-specific penalty bloat.

Positive rewards:
- `is_alive`: Survival bonus, weight +5.0 — structurally positive reward budget
- `track_lin_vel_xy_yaw_frame_exp`: Velocity tracking (std=0.1, weight=2.0)
- `track_ang_vel_z_world_exp`: Yaw tracking (std=0.25, weight=1.0)
- `feet_air_time_positive_biped`: Biped gait (threshold=0.2, body_names=foot_assembly/foot_assembly_2)

Penalties:
- `is_terminated`: Fall penalty (-200)
- `flat_orientation_l2`: Stay upright (-1.0)
- `action_rate_l2`: Smoothness (-0.005)
- `joint_pos_limits`: Servo protection for ankle/knee (-1.0)

Removed (H1-specific, not applicable to duck):
- joint_deviation_head, joint_deviation_hips, dof_acc_l2, dof_torques_l2, ang_vel_xy_l2, feet_slide

Velocity command ranges (conservative for 42cm robot):
- lin_vel_x: (-0.15, 0.3) m/s
- lin_vel_y: (-0.15, 0.15) m/s
- ang_vel_z: (-0.5, 0.5) rad/s

Action scale: 0.25 (reduced from 0.5, matching Open Duck Playground)

## PPO Hyperparameters

- gamma: 0.97
- entropy_coef: 0.005
- init_noise_std: 0.5
- obs_normalization: True

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
