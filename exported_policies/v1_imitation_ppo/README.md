# v1 Imitation PPO — Training Report

**Date:** 2026-03-31
**Hardware:** DGX Spark (Grace Blackwell)
**Iterations:** 3000 (max_iterations)
**Parallel envs:** 4096

## Checkpoint

- `model_2999.pt` — Final policy checkpoint (4.6 MB)
- `agent.yaml` — RSL-RL PPO hyperparameters
- `env.yaml` — Isaac Lab environment configuration

## Architecture

- **Algorithm:** PPO via RSL-RL
- **Network:** MLP [512, 256, 128], ELU activation, separate actor/critic
- **Observation normalization:** Enabled (actor + critic)
- **Policy frequency:** 50 Hz (physics at 200 Hz, decimation=4)
- **Action scale:** 0.25 (matching Open Duck Playground)

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| gamma | 0.97 |
| lam (GAE) | 0.95 |
| entropy_coef | 0.005 |
| learning_rate | 1e-3 (adaptive schedule) |
| clip_param | 0.2 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |
| init_noise_std | 0.5 |

## Reward Function (v1 — internal iteration v4)

11 terms: 4 positive + 7 negative. No alive bonus.

The imitation reward is the dominant positive signal — it tracks reference leg joint positions from the Open Duck Playground polynomial gait library (240 motions, degree-15 polynomials, 0.54s period). Velocity command is matched to the nearest reference motion.

### Positive Rewards

| Term | Weight | Description |
|------|--------|-------------|
| `imitation_reward` | **+10.0** | `exp(-2 * sum((q_leg - q_ref)^2))` over 10 leg joints. Uses `ImitationReward(ManagerTermBase)` with per-env phase tracking. Matches closest velocity command to polynomial library |
| `track_lin_vel_xy_exp` | +2.0 | Velocity tracking in yaw frame, std=0.25 |
| `track_ang_vel_z_exp` | +1.0 | Yaw rate tracking, std=0.5 |
| `feet_air_time` | +0.25 | Biped stepping reward, threshold=0.2s, bodies: foot_assembly/foot_assembly_2 |

### Penalties

| Term | Weight | Description |
|------|--------|-------------|
| `termination_penalty` | -200.0 | Fall penalty (is_terminated) |
| `base_height` | -2.0 | L2 deviation from target_height=0.20m |
| `flat_orientation_l2` | -1.0 | Trunk tilt penalty |
| `lin_vel_z_l2` | -1.0 | Vertical velocity penalty |
| `action_rate_l2` | -0.005 | Action smoothness |
| `joint_pos_limits` | -1.0 | Soft joint limit proximity (ankle/knee) |
| `joint_deviation_head` | -0.1 | Head/neck/antenna L1 deviation from default |

### Observations

Standard Isaac Lab locomotion observations plus gait phase:

| Component | Dims |
|-----------|------|
| Base linear velocity | 3 |
| Base angular velocity | 3 |
| Projected gravity | 3 |
| Joint positions | 16 |
| Joint velocities | 16 |
| Previous action | 16 |
| Velocity command | 3 |
| Gait phase (cos, sin) | 2 |

## Training Results

### Final Metrics (Iteration 3000)

| Metric | Value |
|--------|-------|
| Mean reward | **~235** |
| Mean episode length | **~1000 steps** (20s, max) |
| Imitation reward | 9.3 / 10.0 |
| Velocity tracking (XY) | 1.86 |
| Yaw tracking | 0.80 |
| Flat orientation penalty | -0.05 |
| Base height penalty | -0.003 |
| Fall rate | **~1.3%** |
| Timeout rate | ~98.7% |

### Training Progression

| Iteration | Reward | Imitation | Fall Rate | Episode Length |
|-----------|--------|-----------|-----------|---------------|
| 19 | 24.7 | 1.26 | 99.9% | 237 |
| 175 | 201.8 | 9.01 | 7.9% | 1000 |
| 500 | 208.8 | 8.88 | 2.9% | 936 |
| 1000 | 228.3 | 8.82 | 2.5% | 986 |
| 1500 | 229.6 | 9.20 | 2.2% | 986 |
| 2000 | 232.8 | 9.27 | 1.7% | 993 |
| 2500 | 231.1 | 9.21 | 1.6% | 975 |
| 3000 | 234.8 | 9.27 | 1.3% | 988 |

## Visual Assessment

The robot successfully walks with bipedal gait, maintains upright posture, and survives 98.7% of episodes. However, visual analysis of the trained policy reveals several areas for improvement:

### Issues Identified

Note: The bent-knee posture is **correct for this robot** — the Open Duck Mini
is a BDX-derived bird-like biped where the standing pose has 78° of knee
flexion by design. The polynomial reference gaits encode this bent-knee walk.
"Crouching" is not a defect for this morphology.

The actual issues are:

1. **Short stride length** — Steps are small and shuffling rather than full strides. Feet barely move forward/backward between frames.

2. **Minimal foot clearance** — Feet appear to skim just above the ground during swing phase rather than lifting clearly (target should be ~2cm per placo_defaults).

3. **Constant forward lean** — Trunk tilted forward in most frames, making the robot look hunched. Head droops down.

4. **Stiff/monotonous motion** — No visible weight shift, lateral sway, or dynamic body movement. The posture barely changes over 10 seconds.

### Root Cause Analysis

| Issue | Root Cause |
|-------|-----------|
| Short steps | No reference velocity or contact tracking. Only joint positions tracked (dims 0-15), missing dynamic trajectory from dims 16-36 |
| No clearance | No foot height reward. `feet_air_time` at 0.25 only measures contact duration, not lift height |
| Forward lean | `flat_orientation` at -1.0 too weak vs imitation at 10.0. No explicit trunk pitch penalty |
| Stiff motion | `action_rate` at -0.005 is 100-300x weaker than BDX (-1.5) and Playground (-0.5). Exp kernel saturates, allowing imprecise tracking |

## What Needs to Improve (v2 Design Direction)

Based on research into the Disney BDX paper ("Design and Control of a Bipedal Robotic Character", Jan 2025) and the Open Duck Playground reward structure:

1. **Switch from exp kernel to raw quadratic** for joint position tracking (matching BDX's `-L2 * 15.0`)
2. **Add reference velocity tracking** using polynomial dims 16-31 (joint velocities) and 34-36 (base velocity)
3. **Add reference contact matching** using polynomial dims 32-33 (foot contacts)
4. **Re-add alive bonus** (+10 to +20) — safe with strong imitation signal preventing survival-only exploit
5. **Increase action_rate penalty** from -0.005 to -0.5 or -1.0 for smooth, natural-looking motion
6. **Add action acceleration penalty** for second-order smoothness (BDX uses this)
7. **Remove base_height, lin_vel_z, feet_air_time** — these are handled implicitly by the comprehensive BDX-style imitation

## Internal Reward Iteration History

Prior to this first usable policy, three failed reward designs were attempted and discarded:

| Internal | Approach | Outcome |
|----------|----------|---------|
| iter 1 | H1-derived, 13 penalties | Structurally negative, reward -5.7, failed |
| iter 2 | Added alive bonus (+5.0) | Crouching/shuffling exploit, failed |
| iter 3 | No alive bonus, height control | Improved but unnatural gait |
| **iter 4 (this)** | **Imitation reward (exp kernel)** | **First usable policy — walking but forward lean, short steps, stiff motion** |
| v2 (planned) | BDX-aligned composite imitation | Expected: natural bipedal gait |

## How to Use This Checkpoint

```bash
# Evaluate with Isaac Lab
cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/play_policy.py \
    --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
    --num_envs 50 \
    --checkpoint ~/Projects/Open_Duck_Mini_Jetson/exported_policies/v1_imitation_ppo/model_2999.pt \
    --headless --video --video_length 500
```

## References

- Disney BDX Paper: https://arxiv.org/html/2501.05204v1
- Open Duck Playground: https://github.com/apirrone/Open_Duck_Playground
- Open Duck Mini: https://github.com/apirrone/Open_Duck_Mini
