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

## Multi-Algorithm Experiment (Phase 2)

Training is done on DGX Spark with Isaac Lab. We compare these algorithms:

| Algorithm | Framework | Parallel Envs | Type | Key Feature |
|---|---|---|---|---|
| **PPO** | RSL-RL | 4096 | On-policy | Proven baseline for locomotion |
| **RPO** | SKRL | 4096 | On-policy | PPO + random perturbation (outperforms PPO in 93% of envs) |
| **AMP** | SKRL | 4096 | On-policy + imitation | Uses reference motion discriminator for natural gaits |
| **SAC** | SKRL | 512 | Off-policy | Sample-efficient but needs smaller env count (replay buffer) |
| **TRPO** | SKRL | 4096 | On-policy | Conservative trust region updates, more stable |
| **TD3** | SKRL | 512 | Off-policy | Conditional on SAC results |

## Reward Functions

Ported from the original project + Isaac Lab locomotion best practices:

- `survival`: +0.05 per timestep alive
- `smoothness`: Penalize action jerk (consecutive action differences)
- `init_pose`: Penalize deviation from natural standing pose
- `velocity_tracking`: Reward matching commanded velocity
- `upright`: Reward keeping trunk Z-axis aligned with world up
- `walking_height`: Reward maintaining trunk at ~0.15m height

## Policy Network Architecture

MLP with 3 hidden layers: [512, 256, 128], ELU activation. Separate policy and value networks.

## Training → Deployment Pipeline

```
Isaac Lab (DGX Spark) → .pt checkpoint → ONNX export → TensorRT engine → Jetson inference
```

## Key Files

- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — Environment config
- `isaac_lab_env/open_duck_mini_v2/train_cfg.py` — Training hyperparameters
- `exported_policies/*.onnx` — Exported policies
- `experiments/v2/params_m6.json` — BAM motor identification parameters
- `mini_bdx/mini_bdx/utils/rl_utils.py` — Joint order conversion, action scaling
