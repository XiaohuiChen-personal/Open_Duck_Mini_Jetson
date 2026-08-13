# Rebuild results — `v6d_contact_wrench` on the post-Phase-M plant

**Decided 2026-08-13.** Task R2b of `docs/jetson-mod/task_plan_v2.md`.

> **Plant banner.** Every number in this document was measured on the
> **post-Phase-M** plant: **2.729035 kg**, root
> `trunk_assembly`, 21 rigid bodies, obs **53** /
> action **14**, USD asset hash `767f2415d1b3a056d95e9c310434dbbc`.
> This is a **different robot model** from every number in
> [`v5_contact_results.md`](v5_contact_results.md) (3.657 kg) and
> [`m2657_regate.md`](m2657_regate.md) (2.657 kg).

## VERDICT: `v6d_contact_wrench` PASSES — beats its same-plant control on 4 of 4 contact gates

Gait gate **6/6** (bar: >= 5/6), open-field falls
**0.000 %**. Scored mechanically by `scripts/regate_report.py`;
the video gate G-R3 passed a frame-by-frame audit recorded in
[`eval_results_rebuild/videos/AUDIT.md`](eval_results_rebuild/videos/AUDIT.md).

**The finding is not that v6d wins — it is where the win comes from.** v6d
matches the control on open-field locomotion and separates from it only under
contact, which is exactly the acceptance rule R2b set in advance: *it cannot win
on open-field locomotion, only match it; the win must come from the contact
gates.*

---

## 1. What this comparison is, and what it is not

**Controlled:** `v6d_contact_wrench` versus `v6_robust`. Same plant, same
Phase-M fixes, same protocol, same seed. `v6d` is a fine-tune *of* `v6_robust`'s
own checkpoint, so the contact curriculum is the only variable between them.
Every verdict in this document rests on that pair.

**Not controlled:** anything versus `v5d`. That comparison crosses, at minimum:

| | v5d | v6d |
|---|---|---|
| plant | 3.657 kg (PLANT-1 present) | **2.729035 kg** |
| interface | obs 59 / action 16 | **obs 53 / action 14** (M0b) |
| torque ceiling | 8.716 N·m | **4.903 N·m** (datasheet, PLANT-5) |
| observation latency | none | **0-40 ms** per-env (PLANT-7) |
| below-ground resets | 61.7 % | **0.0 %** (PLANT-3) |
| wrench yaw torque | **silently discarded** (CFG-1) | applied |
| obstacle draws/episode | 2 (CFG-2) | 1 |

A plant change, an interface change and fifteen defect fixes at once.
**v5d numbers appear in this document as context only and are never subtracted
from a v6 number.**

## 2. Protocol

Frozen for the whole campaign; identical for both policies and every battery.

| parameter | value |
|---|---|
| conditions | `0.2,0,0; -0.1,0,0; 0,0.1,0; 0,0,0.3; 0.15,0.05,0.2; 0,0,0.5` (6) |
| per condition | 10 windows x 64 envs x 30 s |
| episodes per entry | **3,840** |
| seed / policy | 42 / deterministic, no observation corruption |
| contact threshold | 1.0 N |

Each policy ran on its own registered play twin — `v6d` on
`ContactWrench-Play-v0`, `v6_robust` on `Contact-Play-v0` — so the curriculum is
the only changed variable.

The plant block is stamped into every JSON by `evaluate_policies.py` and checked
by `tests/test_regate_results.py`; `regate_report.py` exits **2** if a results
directory mixes robot models. That guard is why this document can assert one
plant for all ten JSONs rather than hoping.

## 3. Open-field grid

| metric | `v6_robust` (control) | **`v6d_contact_wrench`** |
|---|---|---|
| gait valid conditions | 6 / 6 | **6 / 6** |
| fall rate (%) | 0.000 | **0.000** |
| ref tracking RMS (deg) | 4.612 | **4.719** |
| stance duty asymmetry (pp) | 2.593 | **2.609** |
| lin vel error (m/s) | 0.1652 | **0.1495** |
| ang vel error (rad/s) | 0.1075 | **0.1151** |
| mean squared jerk | 0.0601 | **0.0749** |
| energy proxy (W) | 19.969 | **19.586** |
| ROM ratio mean | 0.9898 | **1.0845** |

**v6d matches the control on locomotion; it does not beat it.** Reference RMS
differs by **0.107 deg**, inside gate G-R5's 1.0 deg band, and both policies
fall 0.000 % of the time walking in the open field. Per condition, all six valid
and zero falls for both:

| command | control duty L/R | v6d duty L/R | control RMS | v6d RMS |
|---|---|---|---|---|
| `vx+0.20_vy+0.00_wz+0.00` | 63.25/59.58 | 68.06/65.38 | 5.031 | 5.083 |
| `vx-0.10_vy+0.00_wz+0.00` | 69.67/63.18 | 73.42/73.88 | 4.497 | 4.711 |
| `vx+0.00_vy+0.10_wz+0.00` | 63.21/63.43 | 69.90/76.68 | 4.599 | 4.629 |
| `vx+0.00_vy+0.00_wz+0.30` | 67.97/68.20 | 72.55/69.84 | 4.545 | 4.655 |
| `vx+0.15_vy+0.05_wz+0.20` | 66.04/62.77 | 69.92/70.63 | 4.580 | 4.645 |
| `vx+0.00_vy+0.00_wz+0.50` | 70.02/68.33 | 71.49/69.19 | 4.422 | 4.593 |

## 4. Contact battery

Fall rate (%), candidate against the same-plant control. **This is G-R4, and it
is a RELATIVE gate — an absolute number here means nothing without the control
column.** `v5_pipeline.sh` produces the candidate's battery only; the control
column cost four extra evaluations that no task step names.

| gate | `v6_robust` (control) | **`v6d_contact_wrench`** | delta | verdict |
|---|---|---|---|---|
| Push recovery, v4 fall rule | 10.990 | **0.104** | -10.885 | **beats control** |
| Push recovery, v5 fall rule | 13.359 | **0.026** | -13.333 | **beats control** |
| Sustained wrench | 100.000 | **70.365** | -29.635 | **beats control** |
| Obstacle graze | 34.245 | **7.318** | -26.927 | **beats control** |

**v6d beats the control on 4 of 4.**

The wrench row carries the campaign's argument. The control falls
**100.000 %** of the time under a sustained press at 0.2x body weight held 90 %
of every episode; v6d falls **70.365 %**. Both policies are otherwise
identical in lineage — v6d *is* the control, fine-tuned for 3,000 iterations
with obstacles and a sustained wrench switched on. Nothing else differs.

The absolute number is high, and it should be read against what the test now
does: the press applies a yaw torque for the first time (CFG-1 fixed), on a
robot whose actuator ceiling was cut 1.78x by PLANT-5, with 0-40 ms of
observation latency. It is a materially harder test than any v5-era wrench
measurement.

### The wrench row, per condition

Mean episode length is the sharper statistic: the control is not marginally
worse, it is overwhelmed immediately.

| command | control falls | control ep len | v6d falls | v6d ep len |
|---|---|---|---|---|
| `vx+0.20_vy+0.00_wz+0.00` | 100.000 % | 2.54 s | **73.750 %** | **16.68 s** |
| `vx-0.10_vy+0.00_wz+0.00` | 100.000 % | 2.34 s | **73.594 %** | **16.94 s** |
| `vx+0.00_vy+0.10_wz+0.00` | 100.000 % | 3.43 s | **74.062 %** | **15.96 s** |
| `vx+0.00_vy+0.00_wz+0.30` | 100.000 % | 3.72 s | **64.062 %** | **18.70 s** |
| `vx+0.15_vy+0.05_wz+0.20` | 100.000 % | 2.73 s | **69.219 %** | **17.51 s** |
| `vx+0.00_vy+0.00_wz+0.50` | 100.000 % | 3.42 s | **67.500 %** | **17.91 s** |

**v6d's best condition is `vx+0.00_vy+0.00_wz+0.30` at 64.062 %, and both
pure-yaw conditions sit at the top of its ranking.** The Duck Embody benchmark
attributed **7 of 10 falls to rotation-under-contact**; that is exactly where
v6d's margin over the control is widest. The control, by contrast, falls
100.000 % in all six conditions — there is no command under which it survives
this press.


### Caveats that must be stated with these numbers

A reader who sees a v6 wrench number next to v5d's 43.047 % will try to
subtract them. These are the reasons that subtraction is meaningless, each
verified against the configs and logs rather than recalled.

1. **The wrench eval itself got harder, in two independent ways.** The m2657
   run logged `measured robot weight 26.07 N (2.657 kg) -> wrench 5.21-5.21 N
   (active_frac=0.9)`; the v6 run logs **26.77 N (2.729 kg) -> 5.35-5.35 N**.
   That is the smaller change. The larger one is that **CFG-1 was open for every
   earlier measurement**: `torque_z_range = (0.05, 0.15)` was configured and
   then silently discarded, because the warp kernel behind
   `set_forces_and_torques_at_position` assigned the composed torque and then
   overwrote it from the position-induced moment. `contact_events.py` now calls
   `add_forces_and_torques`, so **the yaw torque reaches the robot for the first
   time**. Every wrench number before this campaign was measured under a press
   that could not twist. Both v6 columns share the fix, so v6d-vs-`v6_robust`
   remains controlled.
2. **CFG-2 is fixed here and was open there.** The obstacle was placed twice per
   episode with two independent draws, the second silently replacing the first
   (`known_issues.md` CFG-2, fixed by Task M0). Both v6 columns share the fix;
   no obstacle number from the v5 campaign or the m2657 re-gate is comparable.
3. **The two push rows measure different things and must be read together.**
   The v4 rule terminates on **any** `trunk_assembly` contact above 1 N. The v5
   rule is the deployment's own fall definition — tilt > 60 deg or root height
   < 0.09 m. `env_cfg.py` records why the v4 rule was abandoned: it "taught
   contact = death and never taught recovery". A high v4-rule number beside a
   low v5-rule number says the recovery involves the trunk touching down, not
   that the robot fell.
4. **The obstacle geometry did not scale with the robot.** `obstacle_frac = 1.0`
   and lateral range `(0.10, 0.15)` are fixed in
   `OpenDuckObstacleEvalEnvCfg`, so a robot of a different mass meets the same
   box with different momentum. Controlled within v6; not across plants.
5. **The plant is 2.729 kg, not 2.657 or 3.657.** `add_base_mass` DR remains
   `(-0.10, +0.15)` kg on `trunk_assembly` — as a fraction of this plant,
   -3.7 %/+5.5 %. Left alone deliberately so this stayed a controlled re-run of
   the v5 levers.
6. **The actuator ceiling fell by 1.78x.** PLANT-5 replaced the 8.716 N·m BAM
   electrical-stall figure with the datasheet's **4.903 N·m**. A policy that
   recovers here is recovering with substantially less torque available than any
   v5-era policy had.
7. **PLANT-3 was still open when the m2657 numbers were measured, and it
   contaminates the v4-rule push row specifically.** 61.7 % of resets started
   *inside* the ground plane; the fix (a +20 mm spawn `z`) landed 2026-08-13,
   while the m2657 evals are stamped 2026-08-12. The v4 fall rule counts **any**
   `trunk_assembly` contact above 1 N as a fall, so a robot spawned below the
   floor scores a fall on its first step. v5d's 55.443 % on that row is
   therefore not a measurement of v5d's push recovery, and the gap to v6d's
   0.104 % must not be read as a policy improvement. Both v6 columns were
   measured after the fix (deepest reset now +4.40 mm above ground).

   The v4-rule column collapsed on **both** sides of the fix, which is the
   signature you would expect if it was measuring spawn position rather than
   push recovery:

   | v4-rule push | @m2657 (PLANT-3 open) | @rebuild (fixed) |
   |---|---|---|
   | control | `v4_robust` 93.151 % | `v6_robust` **10.990 %** |
   | contact-trained | `v5d` 55.443 % | `v6d` **0.104 %** |

   The policies differ across columns, so this is consistent-with rather than
   proof. It is enough to disqualify the v4-rule row from any cross-campaign
   reading.


## 5. Video audit (G-R3)

Five conditions rendered at `--num_envs 2`, 1000 control steps each, audited
frame by frame in
[`eval_results_rebuild/videos/AUDIT.md`](eval_results_rebuild/videos/AUDIT.md).
This gate exists because `amp_command7` (journal Run 12) passed every aggregate
metric while crawling; **the video verdict outranks the metrics.**

| clip | verdict |
|---|---|
| `forward_vx02` | **PASS** — alternating bipedal gait, trunk vertical, real swing clearance |
| `turn_wz03` (AGENTS.md-mandated) | **PASS** — stepping rotation, no pivot-scrape |
| `turn_wz05` | **PASS** — same at the faster yaw rate |
| `press_wrench` | **PASS** — see below |
| `obstacle_graze` | **PASS** — contacts the slab, continues upright |

**The press clip shows a behaviour v5d did not have.** At ~7 s the trunk pitches
steeply forward, visibly close to the 60 deg tilt termination, with the head low
and one leg extended behind — and the policy **catches the stumble and returns
to upright within about a second**, then keeps stepping for the remaining 13 s.

The m2657 audit of v5d records the opposite on the same clip type: *"walks
upright for roughly the first 12 s under the sustained press, then topples and
stays down for the remainder — it does not self-right."*

That clip is 2 envs and one draw; the measured rate is 70.365 %. It shows the
mechanism, not its frequency.

## 6. Verdict

**`v6d_contact_wrench` PASSES.** Gait 6/6,
open-field falls 0.000 %, and it beats its same-plant control on
4 of 4 contact gates while matching it on locomotion.

It is exported to `exported_policies/v6d_contact_wrench_ppo/`
(`model_5998.pt`, md5 `37da88d08bf5d593a3febf74bce96dbe`) with a
`deployment_contract.json` sidecar, because the ONNX graph does not contain
`action_scale` or `q_default` (`known_issues.md` DEPLOY-1).

**What this replicates.** The v5 campaign's finding was that *the curriculum has
to contain the disturbance it is meant to survive* — `v5c` (obstacles, no
wrench) fell in 100.000 % of wrench episodes, identically to its control. That
result was measured on a plant 37.6 % heavier than the model on disk, with the
wrench's yaw torque silently discarded and the obstacle placed twice per
episode. **It reproduces on the corrected plant with those three defects fixed,
against a control that is the same policy before fine-tuning.** The conclusion
survived its own evidence being rebuilt.

## 7. Consequences

- **`v6d_contact_wrench` is the locomotion mainline.** `v5d` is superseded: its
  59/16 checkpoint no longer loads after M0b, and its numbers describe a robot
  that does not exist in the simulator.
- **Phase S must assume a completed fall ends the run.** The press clip shows
  recovery from a deep *lean*, not get-up from the floor. No policy in this
  project has ever been trained or tested for get-up.
- **The wrench gate is where the remaining headroom is.** 70.365 % is the
  worst of the four gates by a wide margin, and it is the one that most
  resembles the Duck Embody failure mode (7 of 10 falls were
  rotation-under-contact).

## 8. Limits of this result

1. **Simulation only.** Nothing here has touched hardware. The plant is a
   bottom-up mass rebuild validated against a slicer to 1 %, not a weighed
   robot.
2. **One seed.** Both policies are single runs. The v5 campaign's `amp_v8_seed`
   showed a reproduced mechanism can still miss a gate bar on a different seed.
3. **The wrench eval is one point on a curve.** `force_frac_range` is fixed at
   (0.2, 0.2) with `active_frac = 0.9`. The tiers at 0.1 and 0.3 were never run,
   so the shape of the failure against press magnitude is unmeasured.
4. **`add_base_mass` DR is +/-0.10/0.15 kg** — -3.7 %/+5.5 % of this plant.
   Neither policy has seen the mass uncertainty a real build will carry.
5. **The v4-rule push row cannot be compared across campaigns**, and §4's
   caveat 7 shows why with the measurement.

## Reproduce

```bash
python3 scripts/regate_report.py \
    --results_dir docs/jetson-mod/eval_results_rebuild \
    --candidate v6d_contact_wrench --control v6_robust --markdown
```

