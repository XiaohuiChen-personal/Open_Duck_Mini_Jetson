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
- **Parallel envs:** 4096 on DGX Spark
- **Robot USD:** `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`

## Actuator Configuration

Use `ImplicitActuatorCfg` or `IdealPDActuatorCfg` with BAM-identified parameters:

```python
stiffness = 45.53     # kp from BAM (STS3250 id008)
damping = 1.346       # kd from BAM (STS3250 id008)
armature = 0.040      # From BAM id008
friction = 0.200      # frictionloss from BAM id008
effort_limit = 8.716  # Torque limit in Nm (BAM forcerange)
```

## Domain Randomization

Critical for sim2real transfer. The implemented, gate-validated DR lives in
`OpenDuckRobustEnvCfg` (env_cfg.py, v4-robust track) — duck-scaled, NOT the
generic literature values. Source of truth: `.claude/rules/rl-training.md`
"v4 Tracks". Summary:

- Velocity pushes: +/-0.3 m/s every 8-14 s (not force-based)
- Trunk mass: additive (-0.10, +0.15) kg; trunk CoM +/-10/+/-5 mm
- Friction: static 0.4-1.0 / dynamic 0.3-0.8
- Joint reset scale: 0.9-1.1
- Sensor noise on joint positions and IMU (obs corruption, actor only)

## Termination Conditions

- Trunk height < 0.08 m (fallen)
- Trunk tilt > 90 degrees (flipped)

## Reference Motions (Imitation Reward)

The Open Duck Playground reference motion generator (https://github.com/apirrone/Open_Duck_reference_motion_generator) produces `polynomial_coefficients.pkl` — 240 parametric walking gaits as degree-15 polynomials over a 0.54s period.

This data is used directly by the `ImitationReward` class in `isaac_lab_env/open_duck_mini_v2/imitation_reward.py` — a BDX-style composite (joint pos -L2*15.0, joint vel -L2*0.001, base vel exp(-8e), contact match; RewTerm weight=1.0). It matches the closest velocity command to a reference motion and evaluates the polynomials at NORMALIZED phase t = (i % nb_steps) / nb_steps (see `.claude/rules/rl-training.md` for the current v3 design and `docs/jetson-mod/experiment_journal.md` for the phase-bug history). (The AMP clip-conversion path that also consumed this library was removed 2026-07-26 with the course-study archive.)

The polynomial data lives at: `isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl`

## Training Commands

```bash
# PPO via RSL-RL (primary)
cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/train_ppo.py \
    --task Isaac-Velocity-Rough-OpenDuck-v0 \
    --headless --video --video_length 200 --video_interval 5000

# Play/evaluate a trained policy
cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/play_policy.py \
    --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
    --num_envs 4 --checkpoint <path> --headless --video --video_length 500
```

## Policy Export

```bash
# RSL-RL exports ONNX automatically after play
# Then convert to TensorRT on Jetson: trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16
```
