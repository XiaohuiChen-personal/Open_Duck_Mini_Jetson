# SERVO-1 / SERVO-2 fix — results

**Executed 2026-08-15/16** against
[`v7_servo_fix_plan.md`](v7_servo_fix_plan.md). Two training iterations, both
fully gated. Every bar in this document was fixed **before** the runs and is
scored mechanically by `scripts/check_v7_acceptance.py`.

## VERDICT

**SERVO-1 is FIXED.** **SERVO-2 is substantially improved but its ≤1.0 N·m bar
is not reachable by reward weight alone** — pushing for it breaks two other bars
that matter more.

**Recommended policy: `v7_servo_safe` (weight −1e-2).** It passes every bar
except the aspirational leg-RMS target, and it *improves* on v6d in five
separate places.

## 1. What was changed

| | change |
|---|---|
| SERVO-1 | the four head joints added to `joint_pos_limits`, which protected only ankles and knees |
| SERVO-2 | `dof_torques_l2` enabled — it was `None`, so **nothing priced torque at all** — using a new **pre-clip** term (`torque_rewards.joint_torques_commanded_l2`), because the stock term squares post-clip torque and has no gradient where the policy is saturated |

## 2. SERVO-1 — fixed, decisively

| criterion | bar | v6d | **v7** |
|---|---|---|---|
| `neck_pitch` travel, straight | ≥ 2.0° | 0.005° | **3.775°** |
| % of steps on the end stop | ≤ 10 % | **100.0 %** | **0.0 %** |
| `neck_pitch` torque RMS | ≤ 1.5 N·m | 3.164 | **0.445** |
| `head_yaw` travel (not frozen) | ≥ 10° | 25.9° | **19.2°** |
| turn: neck travel / head_yaw | ≥2 / ≥5 | 6.52 / 11.2 | **6.488 / 16.7** |

The neck went from **207 % of its continuous thermal rating to 28 %**, and from
jammed on its stop 100 % of the time to **never touching it**. It now rests at
−10.99°, mid-range, instead of −20.00°. Visible in the video: the head sits
level rather than pitched down.

## 3. SERVO-2 — the frontier, measured

Three operating points, same plant, same protocol, 3,840 episodes per cell:

| | leg RMS ≤1.0 | neck travel ≥2.0 | head_yaw ≥10 | push v4 ≤1.0 | energy | jerk |
|---|---|---|---|---|---|---|
| v6d (no penalty) | 2.735 ✗ | 0.005 ✗ | 25.9 ✓ | 0.104 ✓ | 19.59 W | 0.0749 |
| **v7 (−1e-2)** | 2.060 ✗ | **3.775 ✓** | **19.2 ✓** | **0.417 ✓** | **12.40 W** | **0.0407** |
| v7b (−4e-2) | **1.587** ✗ | 2.115 ✓ | **9.101 ✗** | **1.875 ✗** | **7.15 W** | **0.0213** |

**The bar is real but unreachable this way.** −4e-2 got leg RMS to 1.587 N·m —
101 % of Feetech's 1.569 N·m nameplate, down from 174 % — but it bought that by
suppressing the head (`head_yaw` 19.2° → 9.1°) and losing push recovery
(0.417 % → 1.875 %). Both are pre-registered bars; both fail. Extrapolation from
these two points says even −8e-2 lands near 1.24 N·m, still above 1.0.

**So the last stretch is a hardware problem, not a reward problem** — consistent
with `servo_torque_budget.md` §4, which put the gait floor at ≈0.69 N·m RMS and
pointed at mass and gearing. That is Task **S.8**'s territory.

## 4. What v7 improves over v6d

Not just torque:

| | v6d | v7 | |
|---|---|---|---|
| mechanical power draw | 19.59 W | **12.40 W** | **−37 %** |
| jerk | 0.0749 | **0.0407** | **−46 %** |
| duty asymmetry | 2.609 pp | **1.982 pp** | better |
| sustained wrench falls | 70.365 % | **62.969 %** | **−7.4 pp** |
| obstacle graze falls | 7.318 % | **6.562 %** | **−0.76 pp** |

**The torque penalty made the robot more robust, not less.** Two of four contact
gates improved. The likely mechanism is the smoother, lower-impulse gait: 46 %
less jerk is harder to destabilise. `energy_proxy_w` is computed from applied
torque × joint velocity, so the 37 % power reduction is the closest thing to a
measured thermal benefit this project has.

Locomotion held throughout: gait 6/6 and 0.000 % open-field falls at every
operating point, and all five video conditions pass frame-by-frame audit
(`eval_results_v7/videos/AUDIT.md`).

## 5. Honest limits

- **No hardware.** All of this is the simulator's implicit-PD actuator model. A
  real STS3250 runs its own firmware PD. S.8 must confirm on a bench.
- **PLANT-6 is open** — dry friction inactive during motion — so simulated
  torque is not faithful motor torque.
- **`armature = 0.040 kg·m²` is unverified** and is 1,056× the driven link
  inertia at `hip_roll_assembly_2`. If it is too high the sim overstates torque
  demand and the real envelope is easier than this suggests.
- **A middle weight was never tested.** −2e-2 sits between the two measured
  points and might hold all bars at ~1.7-1.8 N·m. One more 5-hour iteration
  would answer it.
- **Get-up from a completed fall remains untrained and untested.**
