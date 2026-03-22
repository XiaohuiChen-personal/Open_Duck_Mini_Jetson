---
paths:
  - "isaac_lab_env/**"
---

# Isaac Lab Environment

Rules and context for the Isaac Lab RL training environment.

## Environment Config

The environment is defined in `isaac_lab_env/open_duck_mini_v2/env_cfg.py`:

- **Sim timestep:** 0.005 s (200 Hz physics)
- **Policy frequency:** 50 Hz (decimation = 4)
- **Parallel envs:** 4096 on DGX Spark (512 for off-policy algorithms)
- **Robot USD:** `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`

## Actuator Configuration

Use `ImplicitActuatorCfg` or `IdealPDActuatorCfg` with BAM-identified parameters:

```python
stiffness = 6.55      # kp from BAM
damping = 0.65         # kd from BAM
armature = 0.027       # From robot_motors.xml
friction = 0.083       # frictionloss from robot_motors.xml
effort_limit = 3.57    # Torque limit in Nm
```

## Domain Randomization

Critical for sim2real transfer. Apply during training:

- Mass: +/-10% on all bodies
- Friction: range [0.5, 2.0]
- Motor strength: +/-10%
- Random pushes: [-3.0, 3.0] N every 5-10 seconds
- Sensor noise on joint positions and IMU

## Termination Conditions

- Trunk height < 0.08 m (fallen)
- Trunk tilt > 90 degrees (flipped)

## Reference Motions (for AMP)

The original project has a reference motion generator: https://github.com/apirrone/Open_Duck_reference_motion_generator

This produces `polynomial_coefficients.pkl` containing parametric walking gaits. Convert to AMP dataset format (sequence of joint positions per timestep) for the AMP algorithm in SKRL.

## Training Commands

```bash
# PPO via RSL-RL (baseline)
python -m isaaclab.train --task OpenDuckLocomotion-v0 --headless --num_envs 4096

# AMP via SKRL (best gait quality expected)
python -m isaaclab.train --task OpenDuckLocomotion-v0 --algorithm AMP --headless --num_envs 4096

# SAC via SKRL (off-policy, fewer envs)
python -m isaaclab.train --task OpenDuckLocomotion-v0 --algorithm SAC --headless --num_envs 512

# Play/visualize a trained policy
python -m isaaclab.play --task OpenDuckLocomotion-v0 --checkpoint <path> --num_envs 4
```

## Policy Export

```bash
# RSL-RL exports ONNX automatically after play
# SKRL: use torch.onnx.export on the policy network
# Then convert to TensorRT on Jetson: trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16
```
