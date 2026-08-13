# Print-process decision

> # ⚠ AMENDED 2026-08-12 — read this before the sections below
>
> **The process choice stands. The recorded profile and mass did not survive
> vendor research, and have been corrected.**
>
> | | original | **amended** |
> |---|---|---|
> | process | `fdm-asa` | **`fdm-asa`** (unchanged) |
> | perimeters / infill | 2 / 15 % | **3 / 20 %** |
> | set mass | 1004.29 g | **1163.14 g** |
> | reference vendor | — | **Protolabs Network (hubs.com)** |
>
> **Why.** "2 perimeters / 15 % infill" is a hobbyist slicer profile and is not
> orderable anywhere. Seven bureaus were examined against their own published
> pages: **zero of six expose a wall-count field at all**, and **15 % is below
> every published infill floor** (20 % at Protolabs Network, Craftcloud and
> PCBWay; Xometry sells tiers, not percentages). Sections 1, 3, 4 and 6 below
> were written before that was known and are superseded where they conflict.
>
> **What replaces it.** Protolabs Network is the only bureau that **publishes an
> unconditional shell count** — *"All parts are printed with 3 outline /
> perimeter shells or a wall thickness of 1.2 mm"* — alongside a closed infill
> enumeration `[20, 30, 40, 60, 80]`. Both axes of the mass model are therefore
> pinned **by publication, before the order is placed**. 3 shells / 20 % infill
> measures **1163.14 g** with zero residual profile uncertainty. Full shortlist
> and the quote question list: [`print_vendors.md`](print_vendors.md).
>
> **The sensitivity that decided it.** One extra perimeter costs **+130.39 g**;
> five more infill points cost **+37.66 g**. **82 % of the exposure sits on the
> perimeter axis — the one no vendor lets you choose.** §4 below argued that
> pinning infill would close the risk. It would not: it removes ~18 % of it. A
> *published* shell count outranks an adjustable infill slider.
>
> **Two errors in the original text, corrected.** (1) It described `body_front`
> as a near-solid slab; at 2 perim / 15 % it is ~38 % dense. (2) A working note
> cited a "JLC3DP default 50 % infill" — **that figure is unsubstantiated**, its
> source being a customer Q&A page with zero answers. JLC3DP is in any case
> disqualified: 20 of 37 distinct parts fall under its published 30 × 30 × 15 mm
> minimum build size.
>
> **Not reverted to `mjf-pa12`.** The §4 trigger — "if the bureau will not commit
> to a profile" — is satisfiable, just not in the form written there. A vendor
> committed in public, in writing, ahead of the order. `fdm-asa` at 1163.14 g is
> still **434 g** under MJF PA12's 1597.57 g, and ASA remains the only commodity
> FDM material inside the 90–100 °C window the Jetson cavity needs. `mjf-pa12`
> is now the recorded **fallback**, displacing `fdm-abs`.
>
> **One recommendation carried to Task M0/R2:** R2/R2b retrain before any part
> physically exists, so the plant should carry a **mass DR band of roughly
> 2.50–2.76 kg** (the p2/i15 → p4/i20 span) rather than a point value. Given a
> phantom 1.000 kg root mass and a 54–82 g CAD-mod delta already in this
> project's history, a policy brittle to ±5 % plant mass will not survive
> Phase S whoever prints the parts. Recorded in `print_process.json` as
> `mass_dr_band_kg`.

**Task M1 of `docs/jetson-mod/task_plan_v2.md`. Measured and drafted
2026-08-12.** Every mass in this document was produced by
`scripts/measure_print_mass.py` on this machine on that date; the per-process
reports are committed alongside it as `print_mass_<process>[_pNiM].txt`.

The machine-readable form of this decision is **`scripts/print_process.json`**,
which is the single source of truth that M2, M4 and M5 read. This document is
the reasoning behind it.

> ## ⚠ One item is open and it is the owner's: the quote
>
> Requesting and accepting a quote commits real money to an outside bureau.
> `scripts/print_process.json` carries `"vendor": null` and `"quote_ref": null`
> until that happens, and the schema test asserts they are either both null or
> both filled. **The process choice below is made on engineering evidence and
> does not depend on price** — but see [§6](#6-what-the-owner-must-confirm-at-quote-time),
> which lists three things the quote must establish before M2 books a single
> gram.

---

## 1. The candidates, measured

52 pieces, **1571.94 cm³** of solid volume. `foot_bottom_tpu` ×2 (47.20 cm³) is
exempt from the process density on every row and booked at TPU 95A 1.22 g/cm³,
as `measure_print_mass.py` already does.

| process | kind | infill a lever? | set mass (g) | vs lightest |
|---|---|---|---|---|
| `fdm-abs` 2 perim / 15 % | fdm | yes | **977.15** | — |
| `fdm-asa` 2 perim / 15 % | fdm | yes | **1004.29** | +27 |
| `fdm-abs` 3 perim / 20 % | fdm | yes | 1131.58 | +154 |
| `fdm-pla` 2 perim / 15 % | fdm | yes | 1158.18 | +181 |
| `fdm-asa` 3 perim / 20 % | fdm | yes | 1163.14 | +186 |
| `fdm-petg` 2 perim / 15 % | fdm | yes | 1185.35 | +208 |
| `sls-pa12` | solid | **no** | 1536.58 | +559 |
| `mjf-pa12` | solid | **no** | 1597.57 | +620 |
| `sla-tough` | solid | **no** | 1811.03 | +834 |
| `mjf-pa12-gb` | solid | **no** | 2039.75 | +1063 |

**The full spread is 1,062.60 g on a 2.657 kg robot — 40 % of the machine.**

FDM rows: PrusaSlicer 2.7.2 (linux-arm64), supports / brim / skirt / raft /
wipe-tower all off, so the slicer's `[g]` header line is the part mass.
Reproduce any row with
`python3 scripts/measure_print_mass.py --process <p> [--perimeters N --infill M]`.

## 2. The two constraints that actually decide this

### 2a. Servo torque — measured, and already exceeded

`scripts/measure_joint_torque.py` (new, this task) rolls out the shipped v5d
policy for 30 s × 32 envs at vx = 0.2 and reports per-joint torque. On the
**current 2.657 kg plant**, before any print-process choice:

| joint | peak N·m | RMS N·m | % of steps over continuous |
|---|---|---|---|
| `right_hip_pitch` | **6.723** | 2.827 | 44.1 % |
| `left_hip_pitch` | 6.510 | 2.737 | 55.8 % |
| `left_ankle` | 6.161 | 2.187 | 36.9 % |
| `right_ankle` | 5.886 | 2.386 | 33.2 % |
| `right_knee` | 3.897 | 2.191 | 62.9 % |
| `left_knee` | 3.874 | 2.136 | 59.9 % |

The Feetech STS3250 stalls at **4.903 N·m** (50 kg·cm) and is rated
**1.569 N·m continuous** (16 kg·cm). So the shipped policy commands **137 % of
datasheet stall** and **428 % of continuous** at the hip pitch, on the lightest
plant the project has ever had.

**This is process-independent and no material choice fixes it.** It is
`known_issues.md` **PLANT-5** made concrete: `effort_limit_sim = 8.716 N·m` is
1.78× the datasheet stall, so the simulator has been letting the policy borrow
torque the hardware cannot produce. The fix is Task M0 step 2 (a defensible
ceiling plus DR over it) followed by the R2/R2b retrain.

What the print process *does* control is how much torque the retrained gait must
find. First-order linear scaling of the worst leg joint:

| process | plant kg | worst-leg RMS N·m | vs continuous |
|---|---|---|---|
| `fdm-abs` | 2.476 | 2.635 | 168 % |
| `fdm-asa` | 2.503 | 2.664 | **170 %** |
| `fdm-pla` | 2.657 | 2.827 | 180 % |
| `sls-pa12` | 3.035 | 3.230 | 206 % |
| `mjf-pa12` | 3.096 | 3.295 | **210 %** |

The continuous rating is a **thermal** limit — the torque a servo can hold
indefinitely without overheating. Every option is over it at this gait, but the
solid processes are over it by 40 percentage points more. With 14 servos to
protect through a Phase-S bring-up, that margin is worth buying.

### 2b. Temperature — the currently documented material is wrong

`docs/print_guide.md` specifies **PLA**. The trunk parts *are* the cavity that
encloses the Jetson Orin Nano, which dissipates 20–22 W at 25 W mode with its
**heatsink reaching 55–80 °C** (the reason the thermal-partition design exists
at all).

Heat-deflection temperature at 0.45 MPa:

| material | HDT | verdict for the trunk |
|---|---|---|
| PLA | ~60 °C | **disqualified** — at or below the heatsink range, and PLA creeps under sustained load well below its HDT |
| PETG | ~70 °C | marginal |
| ABS | ~90 °C | adequate |
| ASA | ~91 °C (82–105 range) | adequate |
| PA12 (MJF/SLS) | ~175 °C | far clear |

**Switching the FDM branch off PLA is strictly an improvement on both axes at
once**: ASA is 154 g *lighter* than PLA *and* ~30 °C more heat-tolerant.

## 3. Decision: `fdm-asa`, 2 perimeters, 15 % infill

Ranked reasons:

1. **Servo thermal headroom (§2a).** 170 % of continuous versus 210 % for MJF.
   The torque problem is real, measured, and the print process is the only lever
   available before the retrain. −594 g against MJF is the largest single
   improvement any choice in this phase can make.
2. **Trunk temperature (§2b).** ASA at ~91 °C clears the 55–80 °C cavity with
   margin. This is what removes PLA, which is what the repo currently specifies.
3. **ASA over ABS** for 27 g: better layer adhesion and toughness, which is
   exactly where FDM is weak on a biped that will fall during bring-up. ABS is
   the fallback if the bureau does not offer ASA.

### What this decision costs, stated plainly

- **Anisotropy.** FDM parts are weaker across layer lines. A biped that falls
  will eventually crack one. Mitigation: orientation guidance in §6; and with no
  printer in the house, a replacement part is a bureau order either way, so FDM
  buys no iteration speed here.
- **Warping.** `body_front` is a 143 × 110 mm near-flat slab and `trunk_top` is
  similar — the classic ABS/ASA warp geometry. Mitigation: require an enclosed,
  heated-chamber machine and put both parts through DFM review (§6).
- **Mass is not deterministic.** See §4. This is the serious one.

## 4. The risk this decision takes on, and how it is closed

A powder process would have made the mass model deterministic: mass = volume ×
density, no profile to get wrong. **FDM re-opens exactly the failure class that
produced PLANT-10**, where an assumed density booked the trunk 54–82 g wrong.

The exposure is measured, not hypothetical:

```
fdm-asa  2 perimeters / 15 % infill  ->  1004.29 g
fdm-asa  3 perimeters / 20 % infill  ->  1163.14 g
                                         --------
                                         +158.85 g
```

**158.85 g between two entirely ordinary bureau profiles — nearly twice the
error PLANT-10 records.** Many online FDM services print to a fixed house
profile and do not let the customer specify perimeter count.

**This is closed by a hard gate, not by hope.** `scripts/print_process.json`
carries `"profile_confirmed_with_vendor": false`. M2 and M4 must not book a
single gram until the bureau's actual perimeter count and infill are known,
written into that file, and `measure_print_mass.py` re-run with them. If the
bureau will not state its profile, **the decision reverts to `mjf-pa12`**, whose
mass needs no such promise.

## 5. Consequence for the rest of Phase M

| task | effect of choosing `fdm-asa` |
|---|---|
| **M2** | Books the measured per-part FDM masses. Trap 1 applies in full: on FDM one density cannot describe both sides of a diff, and the sign can even flip. Use the whole-part scheme. |
| **M3** | New `holder_6cell.stl` is booked at the FDM profile like everything else. |
| **M5** | **Does not run.** M5 is explicitly "solid-process branch only" — there is no solid material to hollow out of an FDM part, and infill already does that job. `max_wall_mm` is therefore `null`. |
| **M4** | Per-part density table comes from the slicer, not from a single constant. |
| **M6** | Unchanged; USD regen + `audit_plant_mass.py` as the phase gate. |

### A correction to the plan's premise for M5, recorded here because M1 owns the wall-thickness field

`task_plan_v2.md` states *"HP's design guide caps walls at 3 mm because thicker
sections accumulate heat and deform."* **Three primary sources were checked on
2026-08-12 and none supports that.** 2–3 mm is the recommended **shell wall when
hollowing**, not a maximum on solid wall:

| source | min wall | hollowing guidance |
|---|---|---|
| Materialise, PA12 (MJF) design guidelines | 1 mm | hollow when wall **exceeds 20 mm**; shell 2–3 mm; ≥2 escape holes ≥2 mm dia. |
| Proto3000, MJF design guidelines | 0.3 mm XY / 0.5 mm Z | no stated maximum; hollow shell min 2 mm |
| Secondary summaries | — | "7 mm max for bulky parts" |

The thickest part in this set is `trunk_top` at **T_eff 7.99 mm**, well under
Materialise's 20 mm threshold. **So on a solid process, shelling this part set
would have been a mass-and-cost optimisation, not a print-quality requirement.**
That does not change the decision — it is recorded so that whoever revisits the
solid branch does not inherit a wrong premise.

## 6. What the owner must confirm at quote time

Three things, in priority order. The first is a gate on M2.

1. **The exact print profile.** Perimeter/wall count, infill percentage and
   pattern, and layer height. Write them into `scripts/print_process.json`, set
   `profile_confirmed_with_vendor` to `true`, and re-run
   `python3 scripts/measure_print_mass.py --process fdm-asa --perimeters N --infill M`.
   **If the bureau will not commit to a profile, switch to `mjf-pa12`** (§4).
2. **An enclosed, heated-chamber machine**, and a DFM opinion on `body_front`
   (143 × 110 mm) and `trunk_top` for warp. These are the two parts most likely
   to come back out of flat.
3. **Print orientation** for the load-bearing parts, so layer lines are not in
   the primary load path — in particular the knee/ankle sheets, `foot_top`,
   `foot_side` and the roll/pitch brackets.

Price and lead time are recorded here when known:

| field | value |
|---|---|
| vendor | **PENDING — owner** |
| quoted price | **PENDING — owner** |
| lead time | **PENDING — owner** |
| quote reference | **PENDING — owner** |
| date quoted | **PENDING — owner** |

## 7. Reproduce every number in this document

```bash
# whole-set mass, any process
python3 scripts/measure_print_mass.py --process fdm-asa --perimeters 2 --infill 15
python3 scripts/measure_print_mass.py --process mjf-pa12

# the torque measurement behind §2a (needs a free GPU)
cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/measure_joint_torque.py \
  --checkpoint <repo>/exported_policies/v5d_contact_wrench_ppo/model_5998.pt --headless

# the decision record and its schema
python3 -m pytest tests/test_print_process.py -v
```
