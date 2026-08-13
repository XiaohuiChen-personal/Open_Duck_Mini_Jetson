# `eval_results_rebuild/` — the post-Phase-M plant

**Created by Task R2 on 2026-08-13.** This directory holds the rebuild campaign
(Tasks R2, R2b, R3) for the robot **after every Phase-M change**.

> **This is a DIFFERENT ROBOT MODEL from `eval_results_m2657/`. Nothing may be
> copied between them**, in either direction. The two differ in mass, in
> inertia on every body, in geometry, and in the shape of the policy interface.

---

## The plant this directory belongs to

Measured, not assumed — `audit_plant_mass.py` exits 0 with these numbers and
`verify_action_contract.py` reads the dims back off a live environment.

| field | value |
|---|---|
| `simulated_total_mass_kg` | `2.729035` |
| `root_body` | `trunk_assembly` |
| `num_bodies` | `21` |
| `obs_dim` / `action_dim` | `53` / `14` |
| `mjcf_sha256` | `c444493a651b4745eafe5f52a84783ae02e2ed14abd5f24bd3eb2117039931a5` |
| `usd_asset_hash` | `767f2415d1b3a056d95e9c310434dbbc` |

## What changed from `eval_results_m2657/` (2.657067 kg, 59/16)

| task | change |
|---|---|
| **M2** | PLANT-10: CAD-mod deltas are whole-part **measured**, no assumed density. Trunk +74.53 g |
| **M3** | all six 18650 cells are real geometry; rear hump deepened; `holder_6cell` added |
| **M4** | every one of the 17 real-inertial bodies rebuilt **bottom-up per part**. Robot 2.657067 -> 2.729035 kg |
| **M0** | PLANT-3 (spawn +20 mm, below-ground resets 61.7 % -> 0.0 %), PLANT-5 (effort ceiling 8.716 -> **4.903 N·m**, the datasheet stall), PLANT-7 (**new** 0-40 ms observation latency), PLANT-8, PLANT-9, CFG-1 (`torque_z_range` was dead), CFG-2 (obstacle drawn twice per episode) |
| **M0b** | antennas dropped from the interface: obs **59 -> 53**, action **16 -> 14**, critic 62 -> 56. Closes DEPLOY-3 |
| **M6** | USD regenerated; phase gate `audit_plant_mass.py` exits 0 |

**Every v1-v5d checkpoint fails to load against this model** (shape mismatch on
the 53/14 interface). That is deliberate and irreversible; their re-gate on the
2.657 kg plant is `m2657_regate.md` and it is the last measurement of them that
will ever exist.

## The frozen protocol for this campaign

Identical to the m2657 campaign, so the two campaigns are comparable in
*protocol* even though their plants are not comparable in *absolute numbers*.

| parameter | value |
|---|---|
| conditions | `0.2,0,0; -0.1,0,0; 0,0.1,0; 0,0,0.3; 0.15,0.05,0.2; 0,0,0.5` (6) |
| windows x envs x length | 10 x 64 x 30 s = **3,840 episodes** per entry |
| seed / policy | 42 / deterministic, no observation corruption |
| pushes | off unless `--keep-pushes` |
| comparison table | `docs/jetson-mod/rebuild_comparison.md` |
| results doc | `docs/jetson-mod/rebuild_results.md` |

Always pass `--include` when regenerating the comparison table (EVAL-1).

## Gate thresholds

| gate | bar |
|---|---|
| **G-R1** gait validity | >= **5 of 6** conditions gait-valid, duty inside (40, 90) % |
| **G-R2** falls, open field | aggregate `fall_rate_pct` < **1.0 %**; tightened bar <= **0.5 %** |
| **G-R3** video audit | PASS in >= 2 conditions, forward `vx=0.2` **and** turn `wz=0.3` mandatory |
| **G-R4** contact battery | each of the four fall rates <= the **same-plant** control, which for this campaign is **R2's `v6_robust`** |
| **G-R5** open-field regression | `reference_tracking_rms_deg` within **1.0 deg** of the same-plant control |

**`v4_robust` is NOT the control here.** It cannot even load against a 53/14
interface, and on the 2.657 kg plant it already fell 24.167 % of the time in the
open field (`m2657_regate.md`). R2's `v6_robust` is the control for R2b.

## Effects to name in every write-up

- **The wrench scales with body weight.** `ContactRegimeEvent` sizes it off
  `default_mass.sum() * 9.81`, so at 2.729035 kg the banner reads ~26.77 N and
  the 0.2-fraction press is ~5.35 N, against 26.07 N / 5.21 N on the m2657 plant
  and 35.88 N / 7.18 N on the original 3.657 kg one. The dimensionless ratio is
  what is comparable.
- **The obstacle is fixed geometry** and does not scale with the robot.
- **CFG-2 is FIXED here** — the obstacle is now drawn once per episode, not
  twice. Obstacle numbers are therefore **not** comparable to either earlier
  campaign.
- **CFG-1 is FIXED here** — the wrench now actually applies its yaw torque.
- **PLANT-7 latency is live**: observations are delayed 0-2 control steps. The
  value is provisional until Task S.6 measures the real loop.
