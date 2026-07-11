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
remains a mandatory protocol step. Motivation: amp_v4 passed every
aggregate quality metric while locomoting in a crawl (stance duty
0.7%/0.4%); the gate turns that duty signal into an explicit verdict, and
it also flags amp_v1's one-foot-dragging conditions (98–100% duty).

Aggregate values below are means over the five conditions.

<!-- BEGIN AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
## Results

_Last regenerated: 2026-07-11T03:03:36_

### Aggregate over all conditions

Rank policies on the quality columns only after they pass the gait gate (see Metric definitions); gate-failing entries are not walking.

| Policy | Framework | Gait valid | Fall rate (%) | Ep len (s) | Ref RMS (deg) | Jerk | Action std | Duty L (%) | Duty R (%) | Duty asym (pp) | ROM ratio L/R | v_xy err (m/s) | wz err (rad/s) | Energy (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amp_v1 | skrl_amp | 2/5 | 0.2 | 29.9 | 10.89 | 14.4823 | 0.203 | 69.6 | 60.8 | 29.1 | 1.00 | 0.026 | 0.062 | 18.36 |
| amp_v2 | skrl_amp | 4/5 | 2.4 | 29.3 | 13.50 | 12.8383 | 0.327 | 59.8 | 72.6 | 13.0 | 3.67 | 0.035 | 0.129 | 17.65 |
| amp_v3 | skrl_amp | 5/5 | 5.8 | 28.3 | 18.07 | 0.6328 | 0.175 | 79.1 | 78.8 | 0.3 | 0.85 | 0.120 | 0.113 | 11.72 |
| amp_v4 | skrl_amp | 0/5 | 1.3 | 29.6 | 27.20 | 2.2803 | 0.677 | 0.7 | 0.4 | 0.3 | 12.92 | 0.135 | 0.078 | 170.55 |
| amp_v5 | skrl_amp | 0/5 | 0.8 | 29.8 | 12.13 | 1.2574 | 0.308 | 99.2 | 90.9 | 8.3 | 1.12 | 0.118 | 0.059 | 3.62 |
| ppo_v2 | rsl_rl | 5/5 | 0.0 | 30.0 | 10.27 | 0.0513 | 0.247 | 75.6 | 61.0 | 14.6 | 0.82 | 0.134 | 0.086 | 14.41 |
| ppo_v3 | rsl_rl | 5/5 | 0.0 | 30.0 | 4.59 | 0.0687 | 0.234 | 68.9 | 64.6 | 4.3 | 1.02 | 0.155 | 0.078 | 21.61 |

### Fall rate (%) per condition

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 |
|---|---|---|---|---|---|
| amp_v1 | 0.3 | 0.0 | 0.0 | 0.2 | 0.5 |
| amp_v2 | 3.0 | 2.3 | 2.5 | 1.6 | 2.5 |
| amp_v3 | 8.9 | 5.3 | 3.1 | 5.0 | 6.7 |
| amp_v4 | 1.9 | 1.6 | 1.2 | 0.9 | 0.9 |
| amp_v5 | 1.4 | 0.5 | 0.2 | 1.1 | 0.6 |
| ppo_v2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ppo_v3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Gait validity per condition (duty L/R, ok = both in [40, 90]%)

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 |
|---|---|---|---|---|---|
| amp_v1 | ok (50/50) | FAIL (99/52) | FAIL (50/100) | FAIL (99/51) | ok (50/51) |
| amp_v2 | ok (53/52) | ok (49/82) | ok (49/81) | FAIL (98/98) | ok (50/50) |
| amp_v3 | ok (83/82) | ok (81/81) | ok (81/81) | ok (76/76) | ok (75/75) |
| amp_v4 | FAIL (1/0) | FAIL (1/0) | FAIL (1/0) | FAIL (1/0) | FAIL (1/0) |
| amp_v5 | FAIL (99/98) | FAIL (100/100) | FAIL (100/100) | FAIL (99/64) | FAIL (98/94) |
| ppo_v2 | ok (74/63) | ok (75/59) | ok (76/61) | ok (77/59) | ok (76/63) |
| ppo_v3 | ok (70/63) | ok (69/65) | ok (70/63) | ok (67/67) | ok (68/64) |
<!-- END AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
