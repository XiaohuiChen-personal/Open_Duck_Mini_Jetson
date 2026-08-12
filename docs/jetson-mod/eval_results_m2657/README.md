# `eval_results_m2657/` — the PLANT-1-only plant

**Created by Task R0 (`docs/jetson-mod/task_plan_v2.md`) on 2026-08-12.**

This directory holds the re-gate campaign (Tasks R1, R1b, R1c) for the robot as
it exists **after the PLANT-1 fix and before any Phase-M change**.

> **No JSON measured on the 3.657 kg plant may be copied into this directory.**
>
> **This directory is the PLANT-1-only plant. Post-Phase-M results go to
> `eval_results_rebuild/`.**

---

## The plant this directory belongs to

`m2657` means "the 2.657 kg, PLANT-1-only plant". Every JSON written here must
carry a `plant` block (added to `scripts/evaluate_policies.py` by Task R0) whose
fields match the table below. If one does not, it was measured on a different
robot and does not belong here.

| Field | Value |
|---|---|
| `simulated_total_mass_kg` | `2.657067` |
| `root_body` | `trunk_assembly` |
| `num_bodies` | `21` |
| `obs_dim` / `action_dim` | `59` / `16` |
| `mjcf_sha256` | `4cd217d32fe5b7affda0b60b2249c3542fb1d06c3a9f783a973fa2dd83e87832` |
| `usd_asset_hash` | `10ab887fe4d412b22d3d7c857a9d7f12` |

Sources, re-runnable:

```bash
sha256sum mini_bdx/robots/open_duck_mini_v2/robot_motors.xml
cat mini_bdx/robots/open_duck_mini_v2/usd/.asset_hash     # no trailing newline
cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/audit_plant_mass.py --headless
cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/verify_action_contract.py --headless
```

**Why `m2657` is not a name for the post-Phase-M plant.** Task M2 re-derives the
trunk inertial for PLANT-10 and Task M0b changes the obs/action dims to 53/14,
so the post-M robot is a *different model* even if its total mass happens to
land back on 2.657067 kg. One results directory and one comparison table per
robot model, never mixed (`AGENTS.md`, "Locomotion Policy Evaluation Protocol").

**Known limitation, true of every number in this directory.**
`known_issues.md` **PLANT-10** is still open: the Part-2 CAD delta is booked at
−88.48 g but measures −6.60 g, so `trunk_assembly` is 54–82 g light and this
"corrected" 2.657 kg plant is itself provisional. Phase R corrects a 37.6 %
error; PLANT-10 is a further ~2 %, and Phase M Task M2 owns it. State this in
every artefact that quotes a number from here.

---

## The frozen protocol for this campaign

Do not vary it between entries in this directory — the whole point of the
campaign is a controlled comparison.

| Parameter | Value |
|---|---|
| Conditions | `0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2;0,0,0.5` (6) |
| Windows per condition | 10 |
| Environments | 64 |
| Episode length | 30 s |
| Episodes per policy | 6 × 10 × 64 = **3,840** |
| Seed | 42 |
| Deterministic policy | yes |
| Observation corruption | off |
| External pushes | **off** unless `--keep-pushes` (push/wrench/obstacle batteries only) |
| Comparison table | `docs/jetson-mod/m2657_comparison.md` |
| Verdict document | `docs/jetson-mod/m2657_regate.md` |

Always pass `--include` when regenerating the comparison table. Without it every
`*.json` here is injected, and the push/wrench/obstacle JSONs land in a table
whose preamble says "no external pushes" — that is `known_issues.md` **EVAL-1**,
which has already fired once in `eval_results_v4/`.

---

## Gate thresholds (copied verbatim from `task_plan_v2.md`, Phase R)

| Gate | Bar |
|---|---|
| **G-R1** gait validity | ≥ **5 of 6** conditions gait-valid (both feet's stance duty inside `GAIT_DUTY_BAND_PCT = (40.0, 90.0)`, `scripts/evaluate_policies.py`). AGENTS.md's "≥ 4/5" is the same rule stated for the 5-condition grid |
| **G-R2** falls, open field | aggregate `fall_rate_pct` < **1.0 %** (AGENTS.md); `v5_retrain_plan.md` §7 gate 1 tightens the bar to ≤ **0.5 %** |
| **G-R3** video audit | PASS in ≥ 2 conditions (AGENTS.md mandates forward `vx=0.2` **and** turn `wz=0.3`), judged frame-by-frame against the fixed checklist |
| **G-R4** contact battery | each of the four fall rates ≤ the control measured under the **same protocol on the same plant** (R1b for this campaign). Relative gate — an absolute number alone means nothing here |
| **G-R5** open-field regression | `reference_tracking_rms_deg` within **1.0°** of the same-plant control (`v5_retrain_plan.md` §7 gate 1) |

**Reference numbers from the OLD 3.657 kg plant** are in `task_plan_v2.md`
(Phase R) and in `eval_results_v5/`. They are for delta reporting only and are
**not** pass bars on this plant.

---

## Plant-coupled effects to name in every write-up

Not bugs — consequences of the PLANT-1 fix. Someone will otherwise read a moved
number as a policy regression.

- **The wrench got weaker in newtons.** `ContactRegimeEvent` sizes the press off
  `default_mass.sum() * 9.81`, so the banner reads `26.07 N (2.657 kg)` where it
  read `35.88 N (3.657 kg)`. At `force_frac_range = (0.2, 0.2)` the press drops
  from **7.18 N to 5.21 N** while the robot drops by the same 27 %. The
  dimensionless ratio is preserved; the absolute newtons are not.
- **The obstacle did not shrink.** `obstacle_frac = 1.0`, lateral range
  `(0.10, 0.15)` — fixed geometry. A 27 %-lighter robot meets the same box with
  less momentum. Also **CFG-2**: the obstacle is placed *twice per episode* with
  two independent draws, still open during R1/R1b. Same defect on both sides of
  the comparison, so the comparison holds; the absolute number does not.
- **`add_base_mass` DR is `(-0.10, +0.15)` kg on `trunk_assembly`** — was
  −2.7 %/+4.1 % of the plant, now −3.8 %/+5.6 %. Left alone deliberately so this
  stays a controlled re-run.
- **`torque_z_range` in the wrench recipe is dead code** (**CFG-1**). It was dead
  for v5d too, so the comparison is unaffected — but do not describe the wrench
  as applying a yaw torque until Phase M Task M0 fixes it.
- **`Gait/duty_in_band_frac` is measured off a 15 ms ContactSensor history**
  (**CFG-5**), not 60 ms, so the in-training canary reads a systematically
  higher duty than `evaluate_policies.py`. Training watchdog only; never a gate
  number.
