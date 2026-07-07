# Algorithm Comparison — Open Duck Mini v2 (Task 2.5)

Standardized evaluation protocol applied uniformly to all trained policies
(PPO v2, PPO v3, AMP variants). Produced by `scripts/evaluate_policies.py`;
per-policy raw numbers live in `docs/jetson-mod/eval_results/<name>.json`
(each JSON records the exact protocol parameters used for that run).

## Protocol (defaults — all CLI-overridable)

- **Velocity conditions** (vx m/s, vy m/s, wz rad/s):
  (0.2, 0, 0), (-0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.3), (0.15, 0.05, 0.2)
- **Per condition:** 10 rollout windows x 30 s, 64 parallel envs
  (each window yields 64 env-episodes; only each env's FIRST episode per
  window is scored — post-fall auto-reset data is discarded)
- **Policy:** deterministic (mean) actions, no observation corruption,
  no external pushes, fixed command (degenerate command ranges)
- **Contact threshold:** 1 N on the foot contact force norm

## Metric definitions

| Metric | Definition |
|---|---|
| Fall rate (%) | episodes terminated early (termination, not time-out) |
| Ep len (s) | mean episode length; survivors count the full window |
| Ref RMS (deg) | RMS over 10 leg joints vs the nearest library motion's polynomial reference, evaluated at normalized phase t = (i % nb_steps) / nb_steps, coefficients constant-term-first (the corrected v3 phase convention) |
| Jerk | mean over steps of sum over action dims of (a_t - 2a_(t-1) + a_(t-2))^2 |
| Action std | per-env std of actions over time, averaged over dims and envs |
| Duty L / R (%) | share of steps with foot contact force > 1 N, per foot |
| Duty asym (pp) | abs(duty_L - duty_R) — gait-symmetry indicator (the v2 phase bug showed 78/52) |
| ROM ratio L/R | per joint-pair ratio of the p5-p95 range of motion, left/right (1.0 = symmetric) |
| v_xy err (m/s) | mean L2 error between achieved base-frame planar velocity and command |
| wz err (rad/s) | mean abs error between world-frame yaw rate and command |
| Energy (W) | mean of sum_j abs(tau_j * qdot_j) over applied joint torques |
| Gait valid | contact-pattern validity gate: both feet's stance duty within [40, 90]% for the condition. Below the band the policy is not load-bearing on its feet (crawl), above it a foot is dragging. Applied BEFORE any quality ranking |

**Metric hierarchy (added 2026-07-06 after the amp_v4 crawl audit):** the
gait-validity gate comes first; only gate-passing conditions are ranked on
the quality metrics. The gate is necessary, not sufficient — a policy can
pass it and still fail on falls or on the qualitative video audit, which
remains a mandatory protocol step.

Aggregate values below are means over the five conditions.

<!-- BEGIN AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
## Results

_Last regenerated: 2026-07-07T00:12:14_

### Aggregate over all conditions

Rank policies on the quality columns only after they pass the gait gate (see Metric definitions); gate-failing entries are not walking.

| Policy | Framework | Gait valid | Fall rate (%) | Ep len (s) | Ref RMS (deg) | Jerk | Action std | Duty L (%) | Duty R (%) | Duty asym (pp) | ROM ratio L/R | v_xy err (m/s) | wz err (rad/s) | Energy (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_v3 | rsl_rl | 5/5 | 0.0 | 30.0 | 4.59 | 0.0687 | 0.234 | 68.9 | 64.6 | 4.3 | 1.02 | 0.155 | 0.078 | 21.61 |
| v4_inertials | rsl_rl | 5/5 | 0.0 | 30.0 | 4.60 | 0.0704 | 0.238 | 70.0 | 64.9 | 5.1 | 0.99 | 0.155 | 0.073 | 21.42 |

### Fall rate (%) per condition

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 |
|---|---|---|---|---|---|
| ppo_v3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v4_inertials | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Gait validity per condition (duty L/R, ok = both in [40, 90]%)

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 |
|---|---|---|---|---|---|
| ppo_v3 | ok (70/63) | ok (69/65) | ok (70/63) | ok (67/67) | ok (68/64) |
| v4_inertials | ok (70/63) | ok (70/66) | ok (70/66) | ok (69/63) | ok (70/67) |
<!-- END AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
