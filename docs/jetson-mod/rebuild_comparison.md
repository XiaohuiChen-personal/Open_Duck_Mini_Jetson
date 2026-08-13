# Algorithm Comparison — Open Duck Mini v2

Standardized evaluation protocol applied uniformly to the policies whose
JSONs live in this report's results directory (passed via --output_dir;
one table = one robot model — never mix models across result dirs).
Produced by `scripts/evaluate_policies.py`; each JSON records the exact
protocol parameters used for that run.

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

_Last regenerated: 2026-08-13T03:36:17_

_Plant mass(es) simulated: 2.729035 kg — obs/action dims: 53/14 — conditions per entry: [6]_

### Aggregate over all conditions

Rank policies on the quality columns only after they pass the gait gate (see Metric definitions); gate-failing entries are not walking.

| Policy | Framework | Gait valid | Fall rate (%) | Ep len (s) | Ref RMS (deg) | Jerk | Action std | Duty L (%) | Duty R (%) | Duty asym (pp) | ROM ratio L/R | v_xy err (m/s) | wz err (rad/s) | Energy (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6_robust_grid6 | rsl_rl | 6/6 | 0.0 | 30.0 | 4.61 | 0.0601 | 0.270 | 66.7 | 64.2 | 2.6 | 0.99 | 0.165 | 0.107 | 19.97 |

### Fall rate (%) per condition

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 | vx+0.00_vy+0.00_wz+0.50 |
|---|---|---|---|---|---|---|
| v6_robust_grid6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Gait validity per condition (duty L/R, ok = both in [40, 90]%)

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 | vx+0.00_vy+0.00_wz+0.50 |
|---|---|---|---|---|---|---|
| v6_robust_grid6 | ok (63/60) | ok (70/63) | ok (63/63) | ok (68/68) | ok (66/63) | ok (70/68) |
<!-- END AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
