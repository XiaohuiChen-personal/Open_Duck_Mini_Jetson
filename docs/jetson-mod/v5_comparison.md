# Algorithm Comparison — Open Duck Mini v2

Standardized evaluation protocol applied uniformly to the policies whose
JSONs live in this report's results directory (passed via --output_dir;
one table = one robot model — never mix models across result dirs).
Produced by `scripts/evaluate_policies.py`; each JSON records the sampling
parameters used for that run (conditions, windows, envs, seed) plus its
`task_id`. It does NOT record `--keep-pushes` or any disturbance setting, so a
`*PushEval*` JSON on its own cannot say whether pushes were active — the runs
here passed `--keep-pushes` (`scripts/v5_pipeline.sh:143-144`).

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

Aggregate values below are means over the six conditions (640 episodes each,
3,840 per policy).

> **PLANT-1 was fixed on 2026-08-11** — `base` is merged into `trunk_assembly`,
> the USD is regenerated, and PhysX now simulates 2.657067 kg. Caveat (1) below
> therefore describes the plant these rows were measured on, **not** the current
> one. Every number on this page needs re-measuring on the corrected plant before
> it is quoted again.
>
> **Caveats on every row below.** (1) PhysX simulated a **3.657067 kg** plant,
> not the 2.657067 kg the MJCF and USD author: the articulation root `base` has
> no `<inertial>` and takes PhysX's 1.000 kg default (`known_issues.md`
> PLANT-1). These are results for a 3.657 kg plant; do not quote them as
> hardware numbers. (2) The `*_wrencheval` rows scale the wrench by that
> inflated mass (`contact_events.py:112-113`), so the configured 20% of body
> weight is ~27.5% of the 2.657 kg design mass. (3) On the `*_obstacleeval`
> rows the obstacle is placed twice per episode (`known_issues.md` CFG-2), so
> the box's realized offset is the second draw's, taken after the robot has
> already stepped; exposure itself is unaffected there because
> `obstacle_frac = 1.0` (`env_cfg.py:973`).

<!-- BEGIN AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
## Results

_Last regenerated: 2026-07-29T13:51:01_

### Aggregate over all conditions

Rank policies on the quality columns only after they pass the gait gate (see Metric definitions); gate-failing entries are not walking.

| Policy | Framework | Gait valid | Fall rate (%) | Ep len (s) | Ref RMS (deg) | Jerk | Action std | Duty L (%) | Duty R (%) | Duty asym (pp) | ROM ratio L/R | v_xy err (m/s) | wz err (rad/s) | Energy (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4_robust_grid6 | rsl_rl | 6/6 | 0.0 | 30.0 | 4.48 | 0.0720 | 0.228 | 68.7 | 63.7 | 5.0 | 0.99 | 0.154 | 0.067 | 20.85 |
| v4_robust_obstacleeval | rsl_rl | 6/6 | 32.6 | 20.9 | 4.63 | 0.0735 | 0.240 | 67.8 | 63.7 | 4.1 | 1.03 | 0.170 | 0.142 | 20.51 |
| v4_robust_pusheval_v4def | rsl_rl | 6/6 | 6.4 | 29.2 | 4.53 | 0.0710 | 0.232 | 68.6 | 63.7 | 4.9 | 0.98 | 0.156 | 0.086 | 20.58 |
| v4_robust_pusheval_v5def | rsl_rl | 6/6 | 11.1 | 28.5 | 4.49 | 0.0723 | 0.230 | 68.6 | 63.8 | 4.8 | 0.98 | 0.156 | 0.078 | 20.87 |
| v4_robust_wrencheval | rsl_rl | 6/6 | 100.0 | 1.8 | 5.21 | 0.0688 | 0.250 | 63.1 | 68.0 | 5.0 | 1.02 | 0.212 | 0.440 | 20.29 |
| v5a_gated_ft | rsl_rl | 0/6 | 0.0 | 30.0 | 6.18 | 0.0207 | 0.229 | 99.7 | 99.6 | 0.1 | 0.98 | 0.108 | 0.575 | 11.10 |
| v5b_ungated_ft | rsl_rl | 3/6 | 0.0 | 30.0 | 4.79 | 0.0738 | 0.242 | 76.7 | 85.4 | 9.8 | 1.16 | 0.139 | 0.156 | 20.68 |
| v5c_contact_only | rsl_rl | 6/6 | 0.0 | 30.0 | 4.49 | 0.0773 | 0.234 | 65.9 | 66.2 | 1.1 | 1.03 | 0.156 | 0.081 | 21.29 |
| v5c_contact_only_obstacleeval | rsl_rl | 6/6 | 6.2 | 28.2 | 4.73 | 0.0759 | 0.246 | 65.0 | 66.4 | 1.4 | 1.02 | 0.174 | 0.142 | 20.32 |
| v5c_contact_only_pusheval_v4def | rsl_rl | 6/6 | 0.3 | 30.0 | 4.58 | 0.0765 | 0.237 | 65.7 | 65.8 | 0.7 | 1.02 | 0.157 | 0.091 | 21.05 |
| v5c_contact_only_pusheval_v5def | rsl_rl | 6/6 | 3.7 | 29.4 | 4.49 | 0.0776 | 0.236 | 65.9 | 66.0 | 1.0 | 1.03 | 0.157 | 0.088 | 21.29 |
| v5c_contact_only_wrencheval | rsl_rl | 6/6 | 100.0 | 1.6 | 5.46 | 0.0747 | 0.268 | 60.1 | 67.4 | 7.3 | 1.03 | 0.225 | 0.394 | 20.33 |
| v5d_contact_wrench | rsl_rl | 6/6 | 0.0 | 30.0 | 4.67 | 0.0836 | 0.240 | 71.5 | 73.9 | 3.6 | 1.03 | 0.146 | 0.098 | 20.33 |
| v5d_contact_wrench_obstacleeval | rsl_rl | 6/6 | 0.3 | 29.9 | 5.22 | 0.0760 | 0.251 | 74.9 | 76.6 | 3.4 | 0.99 | 0.160 | 0.196 | 17.64 |
| v5d_contact_wrench_pusheval_v4def | rsl_rl | 6/6 | 1.1 | 29.9 | 4.67 | 0.0838 | 0.242 | 71.4 | 73.4 | 3.1 | 1.03 | 0.147 | 0.103 | 20.35 |
| v5d_contact_wrench_pusheval_v5def | rsl_rl | 6/6 | 0.0 | 30.0 | 4.69 | 0.0833 | 0.243 | 71.6 | 73.6 | 3.0 | 1.03 | 0.147 | 0.106 | 20.24 |
| v5d_contact_wrench_wrencheval | rsl_rl | 6/6 | 47.1 | 21.9 | 4.98 | 0.0816 | 0.290 | 66.6 | 69.2 | 2.6 | 0.99 | 0.145 | 0.319 | 19.82 |

### Fall rate (%) per condition

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 | vx+0.00_vy+0.00_wz+0.50 |
|---|---|---|---|---|---|---|
| v4_robust_grid6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v4_robust_obstacleeval | 99.4 | 4.7 | 6.2 | 5.2 | 74.7 | 5.5 |
| v4_robust_pusheval_v4def | 13.9 | 5.8 | 3.3 | 2.0 | 9.2 | 3.9 |
| v4_robust_pusheval_v5def | 27.5 | 8.6 | 5.0 | 2.8 | 18.4 | 4.4 |
| v4_robust_wrencheval | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| v5a_gated_ft | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v5b_ungated_ft | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v5c_contact_only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v5c_contact_only_obstacleeval | 12.8 | 3.9 | 4.2 | 3.9 | 8.3 | 4.2 |
| v5c_contact_only_pusheval_v4def | 0.2 | 1.2 | 0.0 | 0.2 | 0.0 | 0.2 |
| v5c_contact_only_pusheval_v5def | 9.7 | 2.2 | 1.9 | 1.4 | 5.5 | 1.6 |
| v5c_contact_only_wrencheval | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| v5d_contact_wrench | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v5d_contact_wrench_obstacleeval | 0.0 | 0.2 | 0.6 | 0.3 | 0.5 | 0.3 |
| v5d_contact_wrench_pusheval_v4def | 2.0 | 1.6 | 0.9 | 0.2 | 1.2 | 0.5 |
| v5d_contact_wrench_pusheval_v5def | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| v5d_contact_wrench_wrencheval | 18.6 | 61.6 | 61.6 | 55.8 | 26.6 | 58.6 |

### Gait validity per condition (duty L/R, ok = both in [40, 90]%)

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 | vx+0.00_vy+0.00_wz+0.50 |
|---|---|---|---|---|---|---|
| v4_robust_grid6 | ok (68/63) | ok (68/66) | ok (69/64) | ok (69/63) | ok (69/63) | ok (69/63) |
| v4_robust_obstacleeval | ok (64/64) | ok (68/66) | ok (69/64) | ok (69/63) | ok (68/63) | ok (69/63) |
| v4_robust_pusheval_v4def | ok (69/64) | ok (68/65) | ok (68/63) | ok (69/64) | ok (69/63) | ok (69/63) |
| v4_robust_pusheval_v5def | ok (68/63) | ok (68/65) | ok (68/64) | ok (69/64) | ok (69/63) | ok (69/63) |
| v4_robust_wrencheval | ok (66/67) | ok (60/68) | ok (60/73) | ok (63/67) | ok (65/68) | ok (64/66) |
| v5a_gated_ft | FAIL (100/100) | FAIL (100/100) | FAIL (99/98) | FAIL (100/100) | FAIL (100/100) | FAIL (100/100) |
| v5b_ungated_ft | ok (73/70) | ok (79/86) | FAIL (74/96) | FAIL (82/93) | ok (71/78) | FAIL (81/91) |
| v5c_contact_only | ok (67/65) | ok (65/67) | ok (65/65) | ok (67/67) | ok (65/67) | ok (66/67) |
| v5c_contact_only_obstacleeval | ok (62/67) | ok (65/67) | ok (65/66) | ok (67/67) | ok (65/66) | ok (66/67) |
| v5c_contact_only_pusheval_v4def | ok (68/66) | ok (65/66) | ok (64/64) | ok (66/67) | ok (66/66) | ok (66/66) |
| v5c_contact_only_pusheval_v5def | ok (67/65) | ok (65/66) | ok (65/66) | ok (66/67) | ok (65/67) | ok (66/66) |
| v5c_contact_only_wrencheval | ok (62/65) | ok (59/65) | ok (57/72) | ok (60/67) | ok (60/70) | ok (62/65) |
| v5d_contact_wrench | ok (72/70) | ok (74/74) | ok (71/80) | ok (74/72) | ok (69/73) | ok (69/74) |
| v5d_contact_wrench_obstacleeval | ok (86/82) | ok (75/75) | ok (71/80) | ok (74/73) | ok (73/75) | ok (70/74) |
| v5d_contact_wrench_pusheval_v4def | ok (72/70) | ok (74/74) | ok (70/79) | ok (74/72) | ok (69/72) | ok (70/73) |
| v5d_contact_wrench_pusheval_v5def | ok (72/70) | ok (74/74) | ok (71/79) | ok (74/72) | ok (69/72) | ok (70/73) |
| v5d_contact_wrench_wrencheval | ok (66/66) | ok (70/72) | ok (65/73) | ok (68/68) | ok (65/67) | ok (65/69) |
<!-- END AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
