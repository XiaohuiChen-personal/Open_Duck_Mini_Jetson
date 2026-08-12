# Re-gate on the PLANT-1-corrected plant — Open Duck Mini v2

**This table is the `eval_results_m2657/` robot model and nothing else.**
2.657067 kg, articulation root `trunk_assembly`, 21 rigid bodies,
observation 59 / action 16, USD asset hash `10ab887fe4d412b22d3d7c857a9d7f12`,
MJCF sha256 `4cd217d3...e87832`. Every JSON behind the table carries its own
`plant` block and the generated line below re-states it — **if that line ever
shows two masses or two dim pairs, the table is mixing robot models and must be
split.**

Campaign: Tasks R1 / R1b of `docs/jetson-mod/task_plan_v2.md`. Verdict document:
[`m2657_regate.md`](m2657_regate.md). Directory contract and gate table:
[`eval_results_m2657/README.md`](eval_results_m2657/README.md).

> **Do not compare the absolute contact numbers in this table to
> `v5_comparison.md`.** Those were measured on the 3.657 kg plant. Three
> effects move numbers here for reasons that have nothing to do with the
> policy — the wrench press scaled down with body weight (7.18 N -> 5.21 N),
> the obstacle did **not** shrink so a 27 %-lighter robot meets it with less
> momentum, and CFG-2 places that obstacle twice per episode. See
> `m2657_regate.md` for the full list.

## Protocol actually used for every row in this table

Six conditions, not the script's five-condition default:

- **Velocity conditions** (vx m/s, vy m/s, wz rad/s):
  (0.2, 0, 0), (-0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.3), (0.15, 0.05, 0.2),
  **(0, 0, 0.5)**
- **Per condition:** 10 rollout windows x 30 s, 64 parallel envs
  = 640 env-episodes per condition, **3,840 per policy**
  (only each env's FIRST episode per window is scored — post-fall auto-reset
  data is discarded)
- **Policy:** deterministic (mean) actions, no observation corruption,
  fixed command (degenerate command ranges), seed 42
- **External pushes:** OFF for the two open-field rows in this table.
  The push / wrench / obstacle batteries are separate JSONs in the same
  directory and are deliberately **excluded** from this table by
  `--include v5d_contact_wrench --include v4_robust_grid6`. Read them with
  `scripts/regate_report.py`, which pairs each battery gate against its
  same-plant control.
- **Contact threshold:** 1 N on the foot contact force norm

Each policy is evaluated on its own registered play twin — v5d on
`ContactWrench-Play-v0`, v4_robust on `Contact-Play-v0` — exactly as in the
v5 campaign, so the plant is the only changed variable.

## Metric definitions

| Metric | Definition |
|---|---|
| Fall rate (%) | episodes terminated early (termination, not time-out) |
| Ep len (s) | mean episode length; survivors count the full window |
| Ref RMS (deg) | RMS over 10 leg joints vs the nearest library motion's polynomial reference, evaluated at normalized phase t = (i % nb_steps) / nb_steps, coefficients constant-term-first (the corrected v3 phase convention) |
| Jerk | mean over steps of sum over action dims of (a_t - 2a_(t-1) + a_(t-2))^2 |
| Action std | per-env std of actions over time, averaged over dims and envs |
| Duty L / R (%) | share of steps with foot contact force > 1 N, per foot |
| Duty asym (pp) | abs(duty_L - duty_R) — gait-symmetry indicator |
| ROM ratio L/R | per joint-pair ratio of the p5-p95 range of motion, left/right (1.0 = symmetric) |
| v_xy err (m/s) | mean L2 error between achieved base-frame planar velocity and command |
| wz err (rad/s) | mean abs error between world-frame yaw rate and command |
| Energy (W) | mean of sum_j abs(tau_j * qdot_j) over applied joint torques |
| Gait valid | contact-pattern validity gate: both feet's stance duty within [40, 90]% for the condition. Applied BEFORE any quality ranking |

**Metric hierarchy.** The gait-validity gate comes first; only gate-passing
conditions are ranked on the quality metrics. The gate is necessary, not
sufficient — a policy can pass it and still fail on falls or on the
qualitative video audit, which remains a mandatory protocol step
(`amp_command7` passed every aggregate metric while crawling).

Aggregate values below are means over the **six** conditions.

<!-- BEGIN AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
## Results

_Last regenerated: 2026-08-12T06:17:58_

_Plant mass(es) simulated: 2.657067 kg — obs/action dims: 59/16 — conditions per entry: [6]_

### Aggregate over all conditions

Rank policies on the quality columns only after they pass the gait gate (see Metric definitions); gate-failing entries are not walking.

| Policy | Framework | Gait valid | Fall rate (%) | Ep len (s) | Ref RMS (deg) | Jerk | Action std | Duty L (%) | Duty R (%) | Duty asym (pp) | ROM ratio L/R | v_xy err (m/s) | wz err (rad/s) | Energy (W) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4_robust_grid6 | rsl_rl | 6/6 | 24.2 | 23.1 | 4.70 | 0.0724 | 0.238 | 64.3 | 62.3 | 5.5 | 1.00 | 0.190 | 0.151 | 19.57 |
| v5d_contact_wrench | rsl_rl | 6/6 | 0.0 | 30.0 | 4.68 | 0.0845 | 0.243 | 65.7 | 65.4 | 3.2 | 0.99 | 0.156 | 0.104 | 19.86 |

### Fall rate (%) per condition

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 | vx+0.00_vy+0.00_wz+0.50 |
|---|---|---|---|---|---|---|
| v4_robust_grid6 | 100.0 | 0.0 | 0.0 | 0.0 | 45.0 | 0.0 |
| v5d_contact_wrench | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Gait validity per condition (duty L/R, ok = both in [40, 90]%)

| Policy | vx+0.20_vy+0.00_wz+0.00 | vx-0.10_vy+0.00_wz+0.00 | vx+0.00_vy+0.10_wz+0.00 | vx+0.00_vy+0.00_wz+0.30 | vx+0.15_vy+0.05_wz+0.20 | vx+0.00_vy+0.00_wz+0.50 |
|---|---|---|---|---|---|---|
| v4_robust_grid6 | ok (51/61) | ok (68/64) | ok (67/63) | ok (67/63) | ok (63/59) | ok (70/63) |
| v5d_contact_wrench | ok (64/64) | ok (69/65) | ok (63/69) | ok (68/63) | ok (63/67) | ok (66/65) |
<!-- END AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->
