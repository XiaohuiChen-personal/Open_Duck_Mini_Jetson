# Re-gate verdict — v5d on the PLANT-1-corrected plant

> **Archive note (2026-08-15).** `exported_policies/v5d_contact_wrench_ppo/` no longer exists: `exported_policies/` now keeps only the mainline policy (`v6d_contact_wrench_ppo/`) — see [`locomotion_selection.md`](locomotion_selection.md). The v5d checkpoint remains in git history. Paths below are left as written, because they record what was measured at the time.


**Decided 2026-08-12.** Task R1c of `docs/jetson-mod/task_plan_v2.md`.

## VERDICT: v5d PASSES every gate on the corrected plant

`scripts/regate_report.py` exits **0**. G-R1, G-R2, G-R4 and G-R5 all pass
mechanically; G-R3 (video) passed a frame-by-frame audit recorded in
[`eval_results_m2657/videos/AUDIT.md`](eval_results_m2657/videos/AUDIT.md).

**But the headline of this campaign is not v5d's pass. It is the control's
collapse:** `v4_robust`, measured under the identical protocol on the identical
plant, now **falls 24.167 % of the time simply walking in the open field**,
against 0.000 % on the 3.657 kg plant. The plant correction did not merely move
some numbers — it destroyed one of the two policies outright and left the other
untouched.

---

## 1. What changed, and why this document exists

On 2026-08-11 commit `11b1690` merged the massless MJCF root `base` into
`trunk_assembly`. PhysX had been filling that unauthored body with its own
1.000 kg default, so every policy from v1 through the shipped
`v5d_contact_wrench` was trained and gated on a robot **37.6 % heavier** than
the one on disk. See [`known_issues.md#plant-1`](known_issues.md#plant-1).

Their checkpoints still load — obs/action stayed 59/16 — but every published
gate number described a robot that no longer exists in the simulator. This
document records the re-measurement and the decision that follows from it.

## 2. Protocol

Frozen for the whole campaign; identical for both policies and every battery.

| parameter | value |
|---|---|
| conditions | `0.2,0,0; -0.1,0,0; 0,0.1,0; 0,0,0.3; 0.15,0.05,0.2; 0,0,0.5` (6) |
| per condition | 10 windows x 64 envs x 30 s |
| episodes per entry | **3,840** |
| seed / policy | 42 / deterministic, no observation corruption |
| plant | **2.657067 kg**, root `trunk_assembly`, 21 rigid bodies, obs 59 / action 14... see below |

Plant identity, stamped into all ten JSONs by `evaluate_policies.py` and checked
by `tests/test_regate_results.py`: **2.657067 kg**, root `trunk_assembly`,
21 rigid bodies, **obs 59 / action 16**, USD asset hash
`10ab887fe4d412b22d3d7c857a9d7f12`.

Each policy ran on its own registered play twin — v5d on
`ContactWrench-Play-v0`, v4_robust on `Contact-Play-v0` — exactly as in the v5
campaign, so the plant is the only changed variable.

## 3. Open-field grid

| metric | v4_robust @2.657 | v5d @2.657 | v5d @3.657 (stale) |
|---|---|---|---|
| gait valid | 6/6 | **6/6** | 6/6 |
| fall rate (%) | **24.167** | **0.000** | 0.000 |
| mean episode length (s) | 23.06 | **30.00** | 30.00 |
| ref RMS (deg) | 4.704 | **4.676** | 4.669 |
| duty L / R (%) | 64.3 / 62.3 | 65.7 / 65.4 | 71.5 / 73.9 |
| duty asym (pp) | 5.49 | **3.23** | 3.60 |
| wz err (rad/s) | 0.151 | **0.104** | 0.098 |
| energy (W) | 19.57 | 19.86 | 20.33 |
| jerk | 0.0724 | 0.0845 | 0.0836 |

**v5d's open-field behaviour is essentially unchanged by the plant fix.** Ref
RMS moved +0.007 deg. Falls stayed at zero. Stance duty fell ~6-8 pp and energy
fell 0.47 W, both of which are what a 27 %-lighter robot should do.

**v4_robust's did not survive.** 0.000 % -> 24.167 % falls, and mean episode
length 30.00 -> 23.06 s. It still passes the *gait* gate 6/6, which is a useful
reminder that the gait gate measures contact-pattern validity on the episodes
that survive and is not a substitute for the fall rate.

## 4. Contact battery

Fall rate (%), candidate against the same-plant control. **This is G-R4, and it
is a relative gate — an absolute number here means nothing without the control
column.**

| gate | v4_robust @2.657 | v5d @2.657 | delta | v5d @3.657 (stale) |
|---|---|---|---|---|
| Push recovery, v4 fall rule | 93.151 | **55.443** | −37.708 | 1.068 |
| Push recovery, v5 fall rule | 93.672 | **0.339** | −93.333 | 0.000 |
| Sustained wrench | 100.000 | **43.047** | −56.953 | 47.109 |
| Obstacle graze | 46.979 | **0.885** | −46.094 | 0.312 |

v5d beats the control on all four. **G-R4 PASS.**

### Caveats that must be stated with these numbers

A reader who sees "wrench 47.1 % -> 43.0 %" without this section will conclude
the policy improved. It did not; the test got easier in newtons.

1. **The wrench got weaker.** `ContactRegimeEvent` sizes the press off
   `default_mass.sum() * 9.81`. The banner moved from
   `measured robot weight 35.88 N (3.657 kg) -> wrench 1.79-7.18 N` to
   `26.07 N (2.657 kg) -> wrench 5.21-5.21 N`. The press dropped **7.18 N ->
   5.21 N** while the robot dropped by the same 27 %. The *dimensionless* ratio
   (0.2 x body weight) is preserved, which is what makes the gate comparable
   across plants; the absolute newtons are not.
2. **The obstacle did not shrink.** `obstacle_frac = 1.0`, lateral range
   `(0.10, 0.15)` — fixed geometry. A 27 %-lighter robot meets the same box with
   less momentum. Obstacle numbers move for reasons unrelated to the policy.
3. **CFG-2 is still open.** The obstacle is placed **twice per episode with two
   independent draws** (`known_issues.md` CFG-2; Phase M Task M0 step 7 fixes
   it). The defect is present on *both* sides of this comparison, so the
   comparison is controlled — but the absolute obstacle number is not
   trustworthy in either column.
4. **The two push rows measure different things and must be read together.**
   The v4 rule terminates on **any** `trunk_assembly` contact above 1 N. The v5
   rule is the deployment's own fall definition — tilt > 60 deg or root height
   < 0.09 m. `env_cfg.py` records why the v4 rule was abandoned: it "taught
   contact = death and never taught recovery". By the definition the hardware
   will actually be judged on, v5d went **0.000 % -> 0.339 %**. The 55.443 %
   says its recovery now involves the trunk touching down, not that it fell.
5. **`add_base_mass` DR is `(-0.10, +0.15)` kg on `trunk_assembly`** — as a
   fraction of the plant that was −2.7 %/+4.1 % and is now −3.8 %/+5.6 %. Left
   alone deliberately so this stayed a controlled re-run.
6. **`torque_z_range` in the wrench recipe is dead code** (CFG-1). It was dead
   for the old measurements too, so the comparison holds — but do not describe
   the wrench as applying a yaw torque until M0 fixes it.

## 5. Video audit (G-R3)

Full per-condition write-up in
[`eval_results_m2657/videos/AUDIT.md`](eval_results_m2657/videos/AUDIT.md).

| clip | verdict |
|---|---|
| `forward_vx02` (AGENTS.md-mandated) | **PASS** — alternating bipedal gait, real swing clearance, trunk vertical, stance foot loaded |
| `turn_wz03` (AGENTS.md-mandated) | **PASS** — stepping rotation, not a pivot-scrape |
| `turn_wz05` | **PASS** |
| `press_wrench` | robot walks ~12 s, topples, and **does not self-right** |
| `obstacle_graze` | one adverse draw; aggregate is 0.885 %, so it shows the failure mode, not its rate |

**G-R3: PASS** (bar is >= 2 conditions, and both mandated conditions pass).

**Carry this into Phase S: v5d has no get-up behaviour.** Anything that puts the
robot down ends the run until a human rights it.

## 6. Verdict

Machine-generated report, pasted verbatim from
`python3 scripts/regate_report.py --results_dir docs/jetson-mod/eval_results_m2657
--candidate v5d_contact_wrench --control v4_robust --markdown` (exit 0):

```
Results dir entries : 10
Candidate           : v5d_contact_wrench
Control             : v4_robust_grid6
Plant               : 2.657067 kg, root trunk_assembly, 21 bodies, obs/action 59/16
USD asset hash      : 10ab887fe4d412b22d3d7c857a9d7f12

-- Open field (6 conditions x 10 windows x 64 envs x 30 s) ----------

metric                 v4_robust_grid6            v5d_contact_wrench
--------------------------------------------------------------------
gait valid                         6/6                           6/6
fall rate (%)                   24.167                         0.000
ref RMS (deg)                    4.704                         4.676
duty L (%)                        64.3                          65.7
duty R (%)                        62.3                          65.4
duty asym (pp)                    5.49                          3.23
wz err (rad/s)                   0.151                         0.104
energy (W)                       19.57                         19.86
jerk                            0.0724                        0.0845

-- Contact battery (fall rate %, candidate vs same-plant control) --

gate                            control   candidate       delta  verdict
---------------------------------------------------------------------------
Push recovery, v4 fall rule      93.151      55.443     -37.708  ok
Push recovery, v5 fall rule      93.672       0.339     -93.333  ok
Sustained wrench                100.000      43.047     -56.953  ok
Obstacle graze                   46.979       0.885     -46.094  ok

-- Gates ------------------------------------------------------------

G-R1  PASS    gait gate 6/6 (bar >=5/6)
G-R2  PASS    open-field fall rate 0.000% (bar <1.0%) (also inside the tightened <=0.5% bar)
G-R3  MANUAL  video audit — MANUAL, see the verdict doc. A script cannot watch a video and this gate outranks every metric above.
G-R4  PASS    contact battery: all four candidate rates <= control
G-R5  PASS    ref RMS 4.676 deg vs control 4.704 (-0.028, bar |delta| <= 1.0)

OVERALL: PASS
(G-R3 is excluded from this verdict — it is a manual audit.)
```

**In prose: v5d survives the plant correction. v4_robust does not.**

Every scriptable gate passes and the video audit passes. Under the plan's own
decision rule this is the first branch:

> All of G-R1, G-R2, G-R4, G-R5 pass and the video audit passes → v5d survives
> the plant correction. It is still not a hardware candidate (Phase M will
> change the plant again), but **the recipe is known to transfer**, which lowers
> the risk of R2b.

So **no "NOT VALID ON THE CURRENT PLANT" banner is added** to
`exported_policies/v5d_contact_wrench_ppo/README.md`. The checkpoint remains
valid both as the 3.657 kg artefact it was gated as and as a policy that has now
been shown to walk on the corrected plant.

### The finding that matters more than the verdict

The contact-rich v5 curriculum did not just win the v5-vs-v4 comparison on the
plant it was tuned for. **It produced a policy that is robust to a 27 % change
in the plant itself**, which is a different and more valuable property, and it
is the property sim-to-real actually needs.

- v5d open field: 0.000 % -> **0.000 %**
- v4_robust open field: 0.000 % -> **24.167 %**

Neither policy was trained with any mass domain randomisation beyond
`add_base_mass (-0.10, +0.15) kg`. v5d's robustness came from the contact-rich
curriculum — fall-only terminations, obstacles, sustained wrench — not from mass
DR. That is worth carrying into R2b's recipe.

### One weakness in this result, stated plainly

**G-R5 compares v5d's reference-tracking RMS against a control that is falling
24 % of the time.** 4.676 vs 4.704 deg is inside the 1.0 deg bar, but the
control's RMS is computed over the episodes and steps it survived, so it is a
weaker baseline than the gate's authors assumed. The gate passes; it just
carries less information than it did when the control walked.

## 7. Consequences

1. **v5d is not the policy that ships to hardware, and this verdict does not
   change that.** `task_plan_v2.md`'s central decision — "a retrain is already
   forced, spend it once, on everything" — already commits the project to a
   post-Phase-M retrain. There are 15 open plant-fidelity defects and Phase M
   changes the plant again (M2 re-derives the trunk inertial, M0b takes
   obs/action to 53/14). v5d is a 59/16 artefact and will not even load after
   M0b.
2. **The risk of R2b just went down.** The v5d recipe is now known to transfer
   across a large plant change, so retraining it on the post-Phase-M plant is a
   lower-variance bet than it looked before this measurement.
3. **v4_robust is retired as a control for anything on this plant.** A baseline
   that falls 24 % of the time in the open field cannot anchor G-R4 or G-R5 for
   the rebuild campaign. **R2's `v6_robust` becomes the control for R2b**, which
   is what the plan already specifies — this measurement is the justification.
4. **Phase S must assume no get-up behaviour** (section 5).
5. **Phase S must also assume trunk-down recoveries.** The trunk carries the
   Jetson, the battery pack and the BMS; a recovery style that puts it on the
   floor in ~55 % of pushed episodes is a hardware risk even though the
   deployment fall definition passes it. Keep the tether and padding on longer,
   and watch in R2b whether the retrained policy recovers without trunk
   touchdown.

## 8. Limits of this result

None of these are fixed by this task; Phase M owns them.

- **PLANT-10** — the Part-2 CAD delta is booked at −88.48 g but measures
  −6.60 g at the documented FDM profile, so `trunk_assembly` is **54-82 g
  light** and this "corrected" 2.657 kg plant is itself provisional. Phase R
  corrected a 37.6 % error; this is a further ~2 %. Task M2 owns it.
- **PLANT-5** — the simulated torque ceiling is `effort_limit_sim = 8.716 N·m`,
  **1.78x the 50 kg·cm datasheet stall**, pinned to 12.1 V, and **no actuator
  parameter is randomised at all**. The policy may be leaning on torque the
  hardware cannot deliver, independent of everything above.
- **DEPLOY-3** — 4 of the 59 observation dims (`joint_pos_rel[22, 23]`,
  `joint_vel_rel[38, 39]`) are antenna joints driven by open-loop SG90 servos
  with no position feedback, and **cannot be measured on the real robot at
  all**. Task M0b removes them.
- **PLANT-4** — those same antennas are simulated with the STS3250 parameter
  set: ~48x the torque and **10,336x** the armature they should have.
- **CFG-1 / CFG-2** — dead `torque_z_range`, and the obstacle placed twice per
  episode. Both were live for every number in this document.
- **PLANT-3** — 61.7 % of resets start with a collision vertex inside the ground
  plane.
- **CFG-5** — the in-training `Gait/duty_in_band_frac` canary reads a 15 ms
  ContactSensor history, not 60 ms, so it is systematically optimistic. Never
  quote it as a gate number.

## Reproduce

```bash
python3 scripts/regate_report.py \
  --results_dir docs/jetson-mod/eval_results_m2657 \
  --candidate v5d_contact_wrench --control v4_robust
echo "exit=$?"          # 0 = every scriptable gate passed
python3 -m pytest tests/test_regate_results.py tests/test_regate_report.py -q
```
