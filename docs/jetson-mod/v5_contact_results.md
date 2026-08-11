# v5 Contact-Rich Retrain Results (Task 2.8)

Engineering record of the v5 contact-rich retraining campaign (runs executed
2026-07-28 → 2026-07-30). **Deliberately NOT part of
`experiment_journal.md`'s study numbering** — study runs 1-16 are the
EN.665.645 course record, frozen in the archive repo `open-duck-ppo-vs-amp`
(tag `course-study-freeze`); this document serves the robot project, and is the
companion to `v4_retrain_results.md` for the generation that followed it. Note
that this page does **not** satisfy AGENTS.md's per-run journal requirement —
`experiment_journal.md` still has zero v5 entries (`known_issues.md` DOC-2).

**Provenance of this page.** Every figure here was taken from the evaluation
JSONs in `docs/jetson-mod/eval_results_v5/`, from the run records written inline
in `docs/jetson-mod/v5_retrain_plan.md` §§14-25, from the training logs under
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/`, or — where a claim belongs to the
LLM benchmark — from the sibling repo `/home/xiaohui_chen/Projects/duck-embody`
(every such claim is labelled as such below, with its file). Where the plan and
a JSON disagree in decimal places, the JSON value is used. Where two sources
disagree on substance, both are named and neither is quoted as fact.

**What changed under these runs vs v4_robust:** the campaign added *contact* —
walking beside, brushing and being pressed into obstacles, including while
rotating — without regressing v4_robust's flat-ground and push record. The
motivating evidence was the Duck Embody benchmark: 10 falls in 12 trials,
1.58 falls/policy-min, of which 7 were rotation-under-contact, 1 free-space
rotation and 2 sustained press (`v5_retrain_plan.md:18-19`; the per-model split
in `duck-embody/results/scores.json` is fable5 0.75, opus5 1.00, gpt56sol 0.75
falls/trial × 4 trials each = 10).

> **Caveat that applies to every simulation number on this page.** All v5 runs
> and all v5 evaluations were executed on the **old 3.657067 kg plant**: the
> articulation root `base` carried no `<inertial>` and took PhysX's 1.000 kg
> default, so the simulator ran a robot 1.000 kg heavier than the MJCF/USD
> author (`known_issues.md` PLANT-1). **PLANT-1 was fixed on 2026-08-11** —
> `base` was merged into `trunk_assembly` (commit `11b1690`), the USD was
> regenerated, and `audit_plant_mass.py` now exits 0 with PhysX simulating
> 2.657067 kg. Re-check with:
>
> ```bash
> cd ~/IsaacLab && ./isaaclab.sh -p \
>   ~/Projects/Open_Duck_Mini_Jetson/scripts/audit_plant_mass.py --headless
> ```
>
> (the script imports `isaaclab` and **cannot** run under a bare `python3`; it
> exits 1 if simulated and MJCF totals differ by more than `--tolerance`.)
>
> The obs/action contract is unchanged (59/16) so these checkpoints still load,
> but **every gate number below is stale and must be re-measured on the
> corrected plant before it is quoted again**, and none of them may be quoted as
> a hardware number.
>
> Three consequences of the old plant are specific to this campaign: (a) the
> sustained-wrench magnitudes are a fraction of the *inflated* body weight
> (`contact_events.py:112-113` sums `default_mass` and multiplies by 9.81), so
> the configured 20% was ≈27.5% of the 2.657 kg design mass — the event now
> reports 26.07 N where it previously reported 35.88 N; (b) on the
> `*_obstacleeval` rows the obstacle is placed twice per episode from two
> independent draws (`known_issues.md` CFG-2), so the realized offset is the
> second draw's, taken after the robot has already stepped; (c) `torque_z_range`
> is configured but silently discarded in all v5 arms including v5d (CFG-1).

---

## Evaluation protocol used by this campaign

All v5-era evaluations ran `scripts/evaluate_policies.py` on a **6-condition**
grid — the five legacy conditions plus turn-in-place at wz = 0.5, the command
5 of the 10 benchmark falls sat on (`v5_retrain_plan.md:84`):

(0.2, 0, 0) · (−0.1, 0, 0) · (0, 0.1, 0) · (0, 0, 0.3) · (0.15, 0.05, 0.2) ·
(0, 0, 0.5)

Per condition: 10 rollout windows × 30 s × 64 parallel envs = 640 env-episodes,
so **3,840 episodes per policy per gate** (`protocol` block of every JSON in
`eval_results_v5/`). Only each env's **first** episode per window is scored;
post-fall auto-reset data is discarded (`v5_comparison.md`, "Per condition").
Deterministic (mean) actions, no observation corruption, seed 42, contact
threshold 1 N on the foot contact-force norm. Reference RMS is scored against
the **original** frozen reference library for every row (`protocol.reference_pkl`
= `isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl` in all 17
JSONs), including for the v5a/v5b arms that *trained* against the patched
`polynomial_coefficients_v2.pkl`.

> **This is not the protocol `AGENTS.md` documents** (`known_issues.md`
> EVAL-2). AGENTS.md mandates 5 conditions × 10 windows × 64 envs = 3,200
> episodes and a ≥4/5 gate, and specifies the turn video at wz = 0.3; the v5
> pipeline ran 6 conditions, 3,840 episodes, an automated gate of ≥5/6
> (`scripts/v5_pipeline.sh:131`) and rendered its turn video at wz = 0.5.
> Neither protocol is wrong; the difference was simply never written down.
> Consequently v5 numbers are **not** episode-for-episode comparable with the
> `eval_results_v4/` tables — which is why `v4_robust` was re-run on the
> 6-condition grid and on all four contact gates, and why v5-era results live in
> their own directory with their own comparison table (`eval_results_v5/`,
> `v5_comparison.md`; `v4_comparison.md` is frozen and must never be
> regenerated — EVAL-1).

Four contact gates were run as separate registered tasks, each on the same
6-condition grid. Registration lives in
`isaac_lab_env/open_duck_mini_v2/__init__.py:81-115`; the cfg classes are in
`isaac_lab_env/open_duck_mini_v2/env_cfg.py`:

| Gate | Task | Cfg class | What it applies |
|---|---|---|---|
| Push (v4 fall definition) | `...-PushEval-v0` `--keep-pushes` | `OpenDuckPushEvalEnvCfg` (← `OpenDuckRobustEnvCfg_PLAY`) | v4's ±0.3 m/s interval pushes; trunk-contact > 1 N counts as a fall |
| Push (tilt/height definition) | `...-ContactPushEval-v0` `--keep-pushes` | `OpenDuckContactPushEvalEnvCfg` | same push schedule (`interval_range_s=(4.0, 7.0)`, x/y ±0.3 m/s), v5's fall-only terminations (tilt > 60°, height < 0.09 m) |
| Sustained wrench | `...-WrenchEval-v0` | `OpenDuckWrenchEvalEnvCfg` | `active_frac=0.9`, `force_frac_range=(0.2, 0.2)` — a fixed 0.2 × measured body weight — 2-6 s holds, `lateral_bias=0.7`, `cooccurrence_frac=0.0` |
| Obstacle graze | `...-ObstacleEval-v0` | `OpenDuckObstacleEvalEnvCfg` | one kinematic 0.5 × 0.5 × 0.7 m box, `obstacle_frac=1.0`, `obstacle_lateral_range=(0.10, 0.15)` |

**Trap to know before re-running any of these:** all three of `WrenchEval`,
`ObstacleEval` and `ContactPushEval` descend from `OpenDuckContactEnvCfg_PLAY`
— i.e. the **v5a/v5b gated** contact cfg — not from the v5d training cfg. That
is fine for scoring (the gate metrics are falls and duty, not rewards), but it
means the eval env's reward set is not the env the shipped policy trained in,
and it is *not* a contradiction of "v5d does not pass through
`OpenDuckContactEnvCfg`", which is a statement about the **training** chain.

**Second trap:** the eval JSONs do **not** record `--keep-pushes` or any
disturbance setting, so a `*PushEval*` JSON on its own cannot say whether
pushes were active. The runs on this page passed it — see
`scripts/v5_pipeline.sh:143-144` and `scripts/v5_chain.sh:60-61`.

---

## Run 0 — `v5_smoke` (mechanism check, executed 2026-07-28)

**Config**: task `Isaac-Velocity-Rough-OpenDuck-Contact-v0`, 4096 envs,
`--resume --load_run 0000-00-00_v4robust_seed --checkpoint model_2999.pt
--max_iterations 100 --video --video_length 200 --video_interval 1000`.

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-28_03-10-51/`, console
`.training_runs/v5_smoke.log` (repo root), TB event file
`events.out.tfevents.1785226504.spark-2a60.1624246.0`.

**Resume mechanics — confirmed exactly as the plan review predicted.** The run
logged `Learning iteration 2999/3099` and its final checkpoint is
**`model_3098.pt`** (the only other checkpoint in the dir is the interval save
`model_3000.pt`). `model_3099.pt` does not exist; a completion check keyed to
`loaded_iter + max_iterations` would have failed on a healthy run.

**Mechanism diagnostics** (TB `Regime/*`, first → last of 100 iterations):

| Metric | Value | Criterion | Verdict |
|---|---|---|---|
| `wrench_rotating_cooccurrence` | 0.776 → 0.642 | ≥ 0.30 | PASS — the modal fall regime is trained on purpose, not by luck |
| `obstacle_envs_frac` | 0.246 | ~0.25 | PASS |
| `wrench_duty` | 0.035 → 0.309 | reaches target band | PASS |
| `wrench_force_n` | 6.31 → 5.49 N | inside the sampled band | PASS |
| `gate_duty` | 0.035 → **0.398** | plan said < 0.30 | **EXCEEDS the stated bound** |
| `trunk_contact_duty` | 0.0039 | floor to be set here | **thin — 0.39% of steps** |

Terminations were fall-only as designed: `tilt` 0.78, `fell_low` 0.15,
`time_out` 0.22 — `base_contact` is gone. All **twelve** reward terms reported:
`DuckContactRewards` is v4's nine terms minus `flat_orientation_l2` plus the
four new ones (`ang_vel_xy_l2`, `feet_slide`, `flat_orientation_deadzone`,
`ground_contact`) — `env_cfg.py`, class `DuckContactRewards`.

**Two items settled before Run 1, not silently accepted:**

1. **Gate duty 0.398 vs the plan's "< 0.30"** — a consequence of
   `active_frac = 0.5` plus the 1.0 s hold, not of policy exploitation; the gate
   was event-driven at this point. Hartmann's own episode structure is 2 s walk
   / 1 s recovery / 1 s post = 50% disturbed. Resolution: the documented bound
   was raised to < 0.45 rather than the miss ignored.
2. **Trunk-contact duty 0.39%** — thin, and understood: the smoke starts from a
   policy trained for 3,000 iterations to treat trunk contact as death, so it
   still actively avoids the box. Resolution: carried as the Run 1 watch-metric
   with a floor of 1.0% by iteration 1,500.

**Adaptation signal** (expected): reward 2.81 → 125.84 and mean episode length
12.6 → 580 steps across 100 iterations. v4_robust's converged reward under its
own much milder DR was 249.51.

**Verdict: SMOKE PASS** on every mechanism, with the two bounds above amended in
the open rather than waived.

---

## Run 1 — `v5a_gated_ft`: **FAIL** (gait gate 0/6, standing policy)

**Config**: task `Isaac-Velocity-Rough-OpenDuck-Contact-v0`
(`OpenDuckContactEnvCfg`), 4096 envs, the full v5 recipe fine-tuned from
`v4_robust/model_2999.pt` — fall-only terminations, obstacles
(`obstacle_frac=0.25`), sustained wrenches (`force_frac_range=(0.05, 0.30)`,
`active_frac=0.5`), rotational + stronger pushes, banded wz to ±0.7,
disturbance-gated imitation/tracking (`disturbance_gate_scale=0.1`), patched
reference library (`polynomial_coefficients_v2.pkl`), four new penalties. 3000
iterations (2999 → 5998).

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-28_03-25-43/`, final checkpoint
`model_5998.pt` (31 `.pt` files in the dir), console
`.training_runs/v5a_gated_ft.log`, wall-clock **2.22 h** (TB first-to-last
event timestamps). Zero errors.

**Training signals** (TB last-100 means):

| | v5a_gated_ft | v4_robust |
|---|---|---|
| `Train/mean_reward` | 197.71 | 249.51 |
| `Train/mean_episode_length` | 977.46 | 991.76 |
| `Metrics/.../error_vel_xy` | 0.2629 | 0.3140 |
| `Metrics/.../error_vel_yaw` | **1.0898** | 0.3157 |
| `Policy/mean_std` | 0.0977 | 0.0565 |

Terminations at the end: `time_out` 0.966, `tilt` 0.034, `fell_low` 0.0005 —
the robot had all but stopped falling under a far harsher regime than v4 ever
saw. `Regime/trunk_contact_duty` climbed 0.0039 → **0.499**. On the training
curves alone this reads as a success.

**Gated evaluation** (`eval_results_v5/v5a_gated_ft.json`, task
`...-Contact-Play-v0`) — it is not walking:

| Metric | v5a_gated_ft | v4_robust (same task, same protocol) |
|---|---|---|
| Gait gate | **0 / 6** | **6 / 6** |
| Stance duty L/R | **99.7 / 99.6 %** | 68.7 / 63.7 % |
| Fall rate | 0.00 % | 0.00 % |
| Ref RMS | 6.183° | 4.478° |
| wz error | 0.575 rad/s | 0.067 rad/s |
| Energy proxy | 11.10 W | 20.85 W |
| Mean squared jerk | 0.0207 | 0.0720 |

Both feet loaded ~100% of the time is the exact signature the journal records
for standing (study runs 13/14: gate 0/5 at ~99% duty). Half the energy and a
third the jerk corroborate minimal motion, and at the wz = 0.5 condition the yaw
error is 0.509 against a 0.5 command — it rotates at essentially zero. It never
falls because it never really walks; fall rate alone would have called this a
triumph.

**The control matters:** v4_robust evaluated on the *identical* task and
protocol scores 6/6 at 68.7/63.7% duty (`eval_results_v5/v4_robust_grid6.json`).
The harness is sound; the regression is real.

**Root cause — the disturbance gate became self-inducible.**
`Regime/gate_duty` reached **0.83**. Trunk contact was one of the gate triggers;
with obstacles in the world the policy can *acquire* contact; contact duty
reached 0.50; the gate therefore stood up 83% of the time, freezing the tracking
terms at a favourable constant and scaling imitation ×0.1. The episode-average
`imitation_reward` ended at **+0.008** — the only term demanding a real gait was
effectively deleted, leaving `alive_bonus` (+9.58) as the dominant objective.
Standing maximises exactly that. Trunk contact reads as external on an empty
plane and stops being external the moment the curriculum adds obstacles; that is
the error.

**Verdict: GATE 1 FAIL.** No benchmark re-run triggered, no spend. The negative
result isolates the one novel mechanism in the recipe.

**Fix taken:** `contact_raises_gate` now defaults to **False**
(`contact_events.py:206`), so only env-scheduled wrenches and pushes can raise
the gate; trunk contact is still counted for diagnostics.

**Also learned, and load-bearing for the rest of the campaign:** v4_robust
passes wz = 0.5 open-field **6/6** (duty 68.9/63.2, wz err 0.070, 0% falls —
`v4_robust_grid6.json`, condition `vx+0.00_vy+0.00_wz+0.50`). The eval grid had
never tested above 0.3. v4 is *not* rate-limited at 0.5 rad/s in free space —
its benchmark falls at |wz| = 0.5 required the **contact** context. The target
regime is turning **while loaded**, not turning fast.

---

## Run 2 — `v5b_ungated_ft`: **FAIL** (gait gate 3/6, asymmetric shuffle)

**Config**: task `Isaac-Velocity-Rough-OpenDuck-ContactUngated-v0`
(`DuckContactUngatedRewards`): stock `track_lin_vel_xy_exp` /
`track_ang_vel_z_world_exp`, `disturbance_gate_scale=None`. Everything else in
the contact curriculum unchanged — obstacles, sustained wrenches, rotational
pushes, banded wz to ±0.7, fall-only terminations, patched reference library.
Same fine-tune from `model_2999.pt`, 3000 iterations, final checkpoint
`model_5998.pt`.

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-28_22-54-19/`, eval JSON
`eval_results_v5/v5b_ungated_ft.json`. Driven end-to-end by
`scripts/v5_pipeline.sh`, which measured the gait gate first and **skipped the
contact battery** on the 3/6 result (`v5_pipeline.sh:131-132`) — saving roughly
an hour of GPU that would have characterised the contact robustness of a policy
that does not walk well enough to qualify.

| Metric | v5a | **v5b** | v4_robust |
|---|---|---|---|
| Gait gate | 0/6 | **3/6** | 6/6 |
| Stance duty L/R | 99.7/99.6 | **76.7/85.4** | 68.7/63.7 |
| Double support | ~99 % | **~62 %** | ~32 % |
| Duty asymmetry | 0.1 pp | **9.8 pp** | 5.0 pp |
| Ref RMS | 6.183° | **4.790°** | 4.478° |
| Energy proxy | 11.10 W | **20.68 W** | 20.85 W |
| Mean squared jerk | 0.0207 | **0.0738** | 0.0720 |
| Fall rate | 0 % | **0 %** | 0 % |

Removing the disturbance gate fixed the specific defect it was meant to fix: the
imitation reward stayed alive all run (**0.67** vs v5a's terminal 0.008), energy
and jerk returned to v4's values, and reference tracking recovered to within
0.32° of v4. **This policy walks.** It simply does not walk well enough: ~62%
double support against v4's ~32%, and a 9.8 pp left/right asymmetry — a
cautious, heavy-footed shuffle whose right foot drags. The three failing
conditions are precisely the ones with lateral or rotational motion
(vy = 0.1 → duty 74.2/95.7, wz = 0.3 → 81.6/92.6, wz = 0.5 → 81.5/90.7); pure
fore/aft walking passes.

**Reference library exonerated.** Its synthesized cells carry the same
70.4/66.7% reference contact duty and comparable per-side range of motion as the
original cells, so the learned asymmetry is not prescribed by the patch.

**Read across three runs, the trend is monotone in one variable:**

```
double support   99 % (v5a) -> 62 % (v5b) -> 32 % (v4_robust)
gait gate         0/6        ->  3/6      ->  6/6
```

Every pressure removed from the curriculum moved the policy back toward real
walking — the signature of a curriculum still too aggressive for a single
fine-tune (Hartmann et al.'s "excessive caution" local optimum), not of a broken
reward term.

**Verdict: GATE 1 FAIL.** Next arm descends the contact axis instead of retuning
the reward.

---

## Run 3 — `v5c_contact_only`: **PASSES the gait gate 6/6**, fails the wrench gate

**Config**: task `Isaac-Velocity-Rough-OpenDuck-ContactMinimal-v0`
(`OpenDuckContactMinimalEnvCfg`, `env_cfg.py:660`) — **v4_robust plus exactly
two changes**:

1. trunk contact is no longer instant death — terminate on falling
   (`bad_orientation` at 60°, `root_height_below_minimum` at 0.09 m), the
   deployment's own fall definition;
2. an obstacle exists (0.5 × 0.5 × 0.7 m kinematic cuboid), in 25% of episodes
   (`obstacle_frac: 0.25`, `active_frac: 0.0`).

Everything else reverts to v4_robust: its reward set (hence the **original**
frozen reference library), its heading-only ±0.5 commands, its ±0.3 m/s pushes
on the 8-14 s schedule, no sustained wrench, no rotational pushes, no extra
penalties, no curriculum ramp. Verified by config dump: rewards are exactly v4's
nine terms, `cmd_wz = (−0.5, 0.5)`, `curriculum = []`.

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_01-46-04/`,
`model_5998.pt`. Driven end-to-end by `scripts/v5_pipeline.sh`.

**Two process fixes this run forced.** The first launch died in
`ContactRegimeEvent.__init__` with `KeyError: 'force_frac_range'` — the minimal
cfg passes no such param, inside a timeline PLAY callback whose exceptions are
swallowed (the constructor's own comment, `contact_events.py:134-146`, warns
about exactly this). Every param is now read with a default
(`contact_events.py:157`). And **runtime smoke became mandatory before any new
task consumes GPU hours**: config parsing had passed; only stepping the env
exposed the crash. The ~3 min smoke also validated the gait canary end-to-end —
with null actions it correctly reported duty 99.5/99.6 and
`Gait/duty_in_band_frac = 0.0`; at iteration ~3000 of the real run it read
68.0/78.2 with `duty_in_band_frac = 0.59`.

**Open-field grid** (`eval_results_v5/v5c_contact_only.json`) — v5c reproduces
v4's gait:

| Metric | v4_robust | **v5c** | v5b | v5a |
|---|---|---|---|---|
| Gait gate | 6/6 | **6/6** | 3/6 | 0/6 |
| Fall rate | 0.00 % | **0.00 %** | 0.00 % | 0.00 % |
| Ref RMS | 4.478° | **4.486°** | 4.790° | 6.183° |
| Duty L/R | 68.7/63.7 | **65.9/66.2** | 76.7/85.4 | 99.7/99.6 |
| Duty asymmetry | 5.0 pp | **1.1 pp** | 9.8 pp | 0.1 pp |
| Energy proxy | 20.85 W | 21.29 W | 20.68 W | 11.10 W |
| Mean squared jerk | 0.0720 | 0.0773 | 0.0738 | 0.0207 |
| wz error | 0.067 | 0.081 | 0.156 | 0.575 |

Reference tracking is within 0.009° of v4 and the gait is **more symmetric than
v4's** (1.1 pp vs 5.0 pp). The hypothesis that the whole v5a/v5b bundle was
unnecessary complexity is supported: two changes reproduce v4's locomotion.

**Contact battery — one clear win, one total failure** (falls out of 3,840
episodes in parentheses):

| Gate | v5c | reference available at the time |
|---|---|---|
| Push recovery (v4 fall definition) | **0.29 %** (11) | v4 historical 6.84 % |
| Push recovery (tilt/height definition) | 3.70 % (142) | — |
| Obstacle graze | 6.22 % (239) | — |
| **Sustained wrench** | **100.00 %** (3840) | — |

Push recovery improves by more than an order of magnitude and the gait gate
stays 6/6 under every contact condition. But **v5c falls in every single episode
of the sustained-wrench gate** — exactly what its own design predicts: it trains
with `active_frac = 0` and has never felt a force held for 2-6 s. Sustained
pressing is 2 of the 10 benchmark falls and the mechanism behind duck-embody's
S2 counter-press, so this gap is disqualifying on its own.

Note on the two push numbers: the v5 definition (3.70%) reads *higher* than the
v4 definition (0.29%) because tilt > 60° is a more sensitive fall detector than
trunk-contact > 1 N — a robot can pitch past 60° without its trunk touching
anything. The stricter number is the honest one.

**The missing control.** Every v5c contact number above lacked a v4 counterpart
measured under the same protocol. v4's only record was **6.84% pushed falls
from a 5-condition run** (`eval_results_v4/v4_robust_pusheval.json`, aggregate
6.84375%). That run used the **same task id**
(`Isaac-Velocity-Rough-OpenDuck-PushEval-v0`) — the difference is the condition
grid (5 vs 6), not the task, and the added wz = 0.5 condition is most of why the
6-condition re-measurement came back lower at 6.35%. Since "better than
v4_robust" is the acceptance rule, the battery could not be scored without a
matched control. `scripts/v5_chain.sh:59-63` was written to run v4_robust
through all four contact gates before anything else — the control battery
reported in Run 4.

**Verdict: GATE 1 PASS, GATE 5 FAIL.** Not a deployment candidate.

---

## Run 4 — `v5d_contact_wrench`: **BEATS v4_robust on every contact gate**

**Config**: task `Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0`
(`OpenDuckContactWrenchEnvCfg`, `env_cfg.py:742`) — `v5c` **plus one lever**:
the sustained wrench (`active_frac = 0.5`, `force_frac_range = (0.05, 0.20)`
i.e. 5-20% of measured body weight, `duration_range_s = (2.0, 6.0)`,
`lateral_bias = 0.7`, `cooccurrence_frac = 0.3` with `rotating_cmd_wz = 0.25`).
Force is deliberately gentler than v5a/v5b's 5-30% — Hartmann applies 10 N and
20 N to a 117.7 N Go1, i.e. 8.5% and 17%, and the v5a/v5b lesson is that this
curriculum tips into "excessive caution" when disturbances are too strong.
Everything v5c established stays fixed. Config chain:
`OpenDuckContactWrenchEnvCfg → OpenDuckContactMinimalEnvCfg →
OpenDuckRobustEnvCfg → OpenDuckRoughEnvCfg` — it does **not** pass through the
v5a/v5b `OpenDuckContactEnvCfg`.

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25/`,
`model_5998.pt` (md5 `0333e68a4cd9ed3817310ed80f6715e4`, sha256
`301e24e336b2eab0ba387beb50fc16b03e6062b26622bc9a3e98588216a12c54`; both
re-verified 2026-08-11 against the archived copy). Ran unattended via
`scripts/v5_chain.sh`: v4 control battery → train → gate → contact battery →
four audit videos. Archived at `exported_policies/v5d_contact_wrench_ppo/`.

**Pre-launch runtime smoke** (the rule v5c's `KeyError` established): wrench
fires on 51.6% of envs, co-occurrence 0.833, gait canary emitting.

**Early-abort watchdog** (new with this run, `v5_chain.sh:78-92`): thirty
minutes in it reads `Gait/duty_in_band_frac` from the training log and kills the
run below 0.30. It read **0.9852** and correctly let the run continue. v5a spent
2.22 h converging to a standing policy that every training signal called
healthy; this metric would have exposed it in minutes. **Do not treat this
watchdog as a safety net on a future run:** it fails open — any parse failure
leaves its `BAD` variable empty and it prints "healthy, letting it run"
(`known_issues.md` SHELL-5). Two related orchestration traps are also live:
SHELL-1 (`v5_pipeline.sh` picks the run dir by mtime, not by run name —
verified correct for *this* run from `.training_runs/chain_v5d_contact_wrench.log`)
and SHELL-2 (`v5_chain.sh` deletes the launcher's pidfile, disarming its
duplicate-run guard).

**Open-field grid** (6 conditions × 3,840 episodes,
`eval_results_v5/v5d_contact_wrench.json`):

| Metric | v4_robust | v5c_minimal | **v5d_wrench** |
|---|---|---|---|
| Gait gate | 6/6 | 6/6 | **6/6** |
| Fall rate | 0.00 % | 0.00 % | **0.00 %** |
| Ref RMS | 4.478° | 4.486° | **4.669°** |
| Duty L/R | 68.7/63.7 | 65.9/66.2 | **71.5/73.9** |
| Duty asymmetry | 5.0 pp | 1.1 pp | **3.6 pp** |
| wz error | 0.067 | 0.081 | **0.098** |
| Energy proxy | 20.85 W | 21.29 W | **20.33 W** |
| Mean squared jerk | 0.0720 | 0.0773 | **0.0836** |

**Contact battery — every gate measured against the SAME v4 control** (falls out
of 3,840 episodes in parentheses):

| Gate | v4_robust | v5c | **v5d** |
|---|---|---|---|
| Push recovery (v4 fall definition) | 6.35 % (244) | 0.29 % (11) | **1.07 % (41)** |
| Push recovery (tilt/height definition) | 11.12 % (427) | 3.70 % (142) | **0.00 % (0)** |
| **Sustained wrench** | **100.00 % (3840)** | 100.00 % (3840) | **47.11 % (1809)** |
| **Obstacle graze** | **32.60 % (1252)** | 6.22 % (239) | **0.31 % (12)** |

The gait gate holds 6/6 under all four contact conditions for all three
policies — the discriminator is the fall rate, exactly as Task 2.7 found.

**This is the first time the project measured v4_robust's contact fragility
directly, and it is severe: 100% falls under a sustained press and 32.6% on an
obstacle graze**, against 0.00% in the open field. That is the benchmark's
10-falls-in-12-trials, reproduced in a controlled protocol. Per condition,
v4's obstacle-graze falls concentrate where forward motion drives it into the
box: 99.4% at vx = 0.2 and 74.7% at the mixed (0.15, 0.05, 0.2) condition,
against 4.7-6.3% on the others. v5d's residual wrench falls are spread the other
way — 18.6% at vx = 0.2 and 26.6% at the mixed condition, but 55.8-61.6% on the
reverse, lateral and rotational conditions.

v5d cuts obstacle-graze falls **~104×** (1,252 falls → 12, i.e. 32.60 → 0.31%),
eliminates tilt-definition push falls entirely (11.12 → 0.00%), and halves
sustained-wrench falls (100 → 47.11%) — while matching v4's open-field
locomotion.

**Video audit (mandatory; video outranks metrics).** Four mp4s were rendered
into `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25/videos/play/`
(`v5d_contact_wrench_forward_vx02.mp4`, `..._turn_wz05.mp4`,
`..._press_wrench.mp4`, `..._obstacle_graze.mp4`). Three carry a written
verdict; filmstrips at 1.5 fps:

- **forward vx = 0.2** — trunk upright, legs alternate with real ground
  clearance, no drag or glide. **PASS.**
- **turn wz = 0.5 sustained** — heading advances steadily and the rotation is
  produced by *stepping*, not a planted pivot-scrape. This is the command 5 of
  10 benchmark falls died under. **PASS.**
- **sustained press** — the trunk visibly leans into the applied load while the
  feet keep cycling, then recovers. That lean is the intended compliance, and it
  is what v5c/v5d's fall-only terminations permit. Note the dead-zoned
  orientation penalty was a v5a/v5b term and is **not** in this arm, which
  carries v4's stock `flat_orientation_l2` at −2.0. **PASS.**
- **obstacle graze** — rendered, **no written verdict on record.** The 0.31%
  fall rate is a metric, not a video audit; if this arm is ever re-certified,
  audit this clip.

**Verdict: GATE PASS — v5d_contact_wrench is better than v4_robust** under the
owner's acceptance rule: it matches open-field locomotion (gate 6/6, 0.00%
falls, ref RMS within 0.19°) and is strictly better on all four contact gates.

Honest caveats, none disqualifying but all carried to hardware:

1. **Jerk +16% vs v4** (0.0836 vs 0.0720). v4_robust was already documented as
   the least smooth policy on record; v5d is less smooth still. This is the
   clearest regression.
2. **wz error 0.098 vs 0.067** and ref RMS 4.669 vs 4.478 — both inside the
   stated margins, but v5c was closer to v4 on both, so the wrench lever costs a
   little tracking precision.
3. **Sustained wrench is still 47.11%.** Halving v4's 100% is real progress, but
   half of all pressed episodes still end in a fall. This is the remaining
   weakness and the obvious v5e/v6 target.
4. v5d maintains a per-env disturbance-gate state machine every step that
   **nothing reads** — it inherits v4's nine reward terms, not
   `DuckContactRewards` (`known_issues.md` CFG-3). Harmless to the result, pure
   overhead.

---

## Gate scorecard (Task 2.8's five gates, as scored for v5d)

Bars are quoted from `v5_retrain_plan.md` §7. Where a gate's evidence lives in
the sibling repo, the file is named; nothing in this table is inferred.

| # | Gate | Bar (plan §7) | v5d result | Status |
|---|---|---|---|---|
| 1 | Parent eval, 6-condition grid | legacy 5 conditions gait 5/5, unpushed falls ≤ 0.5%, ref RMS within 1.0° of v4; **wz = 0.5 mandatory**: falls ≤ 0.5%, duty valid, wz err comparable to v4 @ wz = 0.3, video "stepping rotation" | 6/6 gait-valid, 0.00% falls, RMS 4.669 vs 4.478 (Δ 0.19°); at wz = 0.5: valid, 0 falls, wz err 0.095 vs v4's 0.063 at wz = 0.3; turn video PASS | **PASS** |
| 2 | Push recovery | ≤ 3% under the v4-comparable rule (v4: 6.84%) | 1.07% (v4 rule) · 0.00% (tilt/height rule) | **PASS** |
| 3 | Duck-embody smokes | physics-pass PASS; S0/S1/S3/S4/S5 green; S2 "no fall within budget" = the v5 pass | physics-pass **PASS** (8/8 transits, `duck-embody/results/logs/physics_pass_v5d_20260729-220122.json`, `acceptance: PASS`); gap-hunt 2026-07-29 (`results/logs/gate_v5d_gap_hunt.log`): S0 PASS, S1 PASS, S2 INCONCLUSIVE/no fall (= desired), S3 PASS, S4 PASS, **S5 FAIL**; S5 re-run green 2026-08-03 (`results/logs/gap_hunt_20260802-214801/gap_hunt_report.json`, verdict PASS) | **PASS (S5 needed a harness fix and a re-run)** |
| 4 | Fall regression suite | precondition: v4 falls in ≥ 8/10 scenarios; pass: v5 survives ≥ 4/5 reps in ≥ 9/10 | precondition met — baseline survived 1/10 (`replay_baseline_20260729-205807`); v5d survived **10/10 scenarios, 3/3 reps each**, mean survival fraction 1.0 (`duck-embody/results/logs/replay_v5d_20260729-220122/replay_falls_report.json`). Note **reps = 3**, not the plan's 5 (`transfer_gates.sh` default) | **PASS on scenarios; rep count below the stated bar** |
| 5 | Sustained wrench + chatter | survives 0.1/0.2 × BW presses during max-wz tracking; chatter condition survived | aggregate 47.11% falls at the single configured tier (`force_frac_range=(0.2, 0.2)`); the 0.1 × BW tier and the command-chatter condition were never run | **NOT SCORED against its own bar** |
| ✔ | Benchmark re-run | falls/policy-min < 0.3 (v4: 1.58) | **0 falls in 12 trials** — see below | **PASS (recorded in duck-embody, not in this repo)** |

---

## Selection

**Deployment candidate: `v5d_contact_wrench` (Run 4)** — owner-confirmed
2026-07-29, and the policy archived in
`exported_policies/v5d_contact_wrench_ppo/`.

**Baseline identity for the comparison:** `duck-embody/policy/model_2999.pt`,
sha256 `b1ebf3a5d7d866efc7496f6cfe511dbe0c1f57e64049da7f1e53f1856b48cbe9` —
re-verified 2026-08-11 as byte-identical to
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`,
i.e. v4_robust, the policy that produced the frozen 10-falls-in-12-trials
record.

**Why v5d and not v5c, stated honestly.** "Best" is not unidimensional; v5c is
the better *gait* reproduction:

| | v5c | v5d |
|---|---|---|
| open-field ref RMS | **4.486°** | 4.669° |
| open-field duty asymmetry | **1.13 pp** | 3.60 pp |
| push, v4 fall definition | **0.29 %** | 1.07 % |
| push, tilt/height definition | 3.70 % | **0.00 %** |
| sustained press | 100.00 % | **47.11 %** |
| obstacle graze | 6.22 % | **0.31 %** |

The benchmark measures falls in a furnished apartment, and the two gates that
map onto it are the ones v5d dominates: obstacle graze ~20× better (239 falls →
12 — the mechanism behind 7 of the 10 original falls) and sustained press halved
rather than total (the other 2). v5c falling in 100% of press episodes would
walk straight back into the wedge failure the benchmark already found. The price
is ~0.18° of reference tracking; both pass the gait gate 6/6 at 0.00%
open-field falls, so neither is a gait regression.

Running both in the apartment would have cost roughly double to isolate the
wrench lever; deferred rather than dropped, and still not done.

---

## Post-selection: the benchmark re-run

All evidence in this section lives in
`/home/xiaohui_chen/Projects/duck-embody`, **not** in this repo. Nothing here
was re-derived; every claim names its file.

### Attempt 1 (2026-07-30) — aborted, two findings

Launched with v5d after the duck-embody re-freeze (`config_hash 6a65f335…`)
with the transfer gates green — replay 10/10, physics PASS, calibration
`k_velocity_realisation = 0.9617`
(`duck-embody/results/calibrations/v5d_contact_wrench.json`), provenance sha
recorded. **Stopped by hand after two partial trials**
(`duck-embody/results/raw_v5d/README.md`, titled "ABORTED BATCH — NOT A
RESULT"). Two findings:

**1 — the locomotion retrain works under LLM control.** In `fable5_seed101`,
v5d took repeated furniture contact and **never fell**. In the frozen v4 batch
the same seed and spawn fell on turn 2, 3.74 policy-seconds into its first
`move`, torso on the sofa (`replay_falls_report.json`, scenario
`fable5_seed101`, `original.fall_turn = 2`, `original.total_policy_s = 3.74`,
mechanism "sustained press"). A frame-by-frame audit of the recorded frames
found a trunk upright, legs alternating with real ground clearance, no drag.
**Do not quote turn / bump / cost counts from this trial**: duck-embody's own
README says 34 turns and 35 bump events for ~$4.87, while the trial's `final`
block records `turns_used = 40`, `bumps = 63` and a `$6.57` token estimate for
that trial alone. The discrepancy is unreconciled and the batch is explicitly
marked "not a result". The **zero falls** is the finding.

**2 — ~95% of the observed "dead-reckoning drift" was a harness accounting bug,
not a policy property.** The trial's believed position ended 26.65 m from truth
in a 4.8 × 3.6 m apartment:

| source | believed | true | inflation |
|---|---|---|---|
| `send_velocity` (49 calls) | 27.09 m | 1.99 m | **25.10 m** |
| `move` (19 calls) | 3.66 m | 2.42 m | 1.24 m |
| clean moves only | 1.10 m | 0.97 m | 0.13 m |

A duck wedged against furniture with its legs cycling was credited
`commanded_speed × time`. Genuine policy-tracking error was 0.13 m, ~0.5% of the
total. Fixed in duck-embody `e0ac862` ("Stop crediting travel to a wedged robot",
contact-time discounting). Consequence for this project: **v5d's calibration
constants are sound**, but no drift figure from this trial is a v5d number, and
the position estimates in `results/raw_v5d/` are corrupted by construction.

### The completed batch (2026-08-03) — the gate is met

The benchmark was subsequently re-run to completion. Recorded in
`duck-embody/results/scores_raw_v5d_r3.json` and
`duck-embody/results/summary_table_raw_v5d_r3.md`:

- 12 trials, 3 models × 4 seeds (101-104), manifest
  `results/manifests/v5d-r3-final-prod.json` status `complete`, `config_hash`
  `d30462d03c76…`, freeze commit `56bd08a68d92…`.
- Checkpoint sha256 `301e24e336b2eab0…` — the **same v5d checkpoint archived in
  this repo**. The manifest also pins this repo's parent commit `7dde4ba9…`.
- **falls / trial = 0.00 for all three models** (`n_defined = 4` each), i.e.
  **zero falls across all 12 trials**, against the frozen v4 batch's 10 falls in
  12 trials (`results/scores.json`: fable5 0.75, opus5 1.00, gpt56sol 0.75 falls
  per trial). The gate `falls/policy-min < 0.3` is met with a zero numerator.

Two honest qualifications:

1. **Roster change.** The r3 matrix is sonnet5 / opus5 / gpt56sol; the frozen v4
   batch was fable5 / opus5 / gpt56sol. fable5 was run as a separate 4-trial
   companion batch (`results/raw_v5d_r3_fable5`, scored in
   `results/scores_raw_v5d_r3_fable5.json`) and also recorded **0.00 falls per
   trial**, so all three of the original models have a zero-fall v5d record.
2. **Harness drift.** r3 runs on a later harness than the frozen v4 batch — it
   includes the contact-time discounting fix (`e0ac862`) and subsequent
   furniture-wedge work. Falls are comparable across that change; the navigation
   and localisation metrics are not.

**Cost, measured rather than extrapolated.** The completed 12-trial batch cost
**$26.56** (sum of `final.tokens.cost_usd_estimate` over
`results/raw_v5d_r3/*.json`; per model: sonnet5 $8.22, opus5 $12.50, gpt56sol
$5.84). The plan's ~$10 estimate was low by ~2.7×, because it silently assumed
v4's fall-shortened trials. Do not re-quote the "$50-60" figure extrapolated
from the aborted attempt — the completed batch supersedes it.

---

## Carried forward

1. **Re-gate everything on the corrected plant.** PLANT-1 was fixed 2026-08-11;
   every simulation number on this page was measured at 3.657067 kg. The winning
   recipe should be retrained and re-gated on the 2.657067 kg plant into a new
   per-model results directory before any hardware work.
2. **Sustained press is the one unsolved failure mode in simulation**: 47.11%,
   halved from v4's 100% but far from safe. Highest-value lever: longer/stronger
   wrench exposure, or simply more iterations — v5d saw the wrench for only
   3,000 fine-tune iterations.
3. **Smoothness regression**: jerk 0.0836 vs v4's 0.0720 (+16%) on a policy
   already documented as the least smooth on record. Worth an explicit
   `action_rate_l2` / `dof_acc_l2` lever before hardware.
4. **v5c-vs-v5d in the apartment** — the paired benchmark this decision
   deferred, and which the r3 batch (v5d only) still does not answer.
5. **Gate 5 was never scored against its own bar** (the 0.1 × BW tier — the eval
   cfg hardcodes `force_frac_range = (0.2, 0.2)` — and the 0.2 s command-chatter
   condition). Gate 4's rep count (3) is also below the stated 5.
6. **This page is not a journal entry.** `experiment_journal.md` still has zero
   v5 entries (DOC-2, journal half).
7. Defects touching these results — CFG-1 (dead `torque_z_range`), CFG-2 (double
   obstacle placement), CFG-3 (dead gate in v5d), EVAL-1/EVAL-2 (comparison-table
   injection, protocol drift), SHELL-1/2/5 (run-dir selection, disarmed guard,
   fail-open watchdog), ART-1 (`policy.onnx` gitignored) — are tracked in
   `docs/jetson-mod/known_issues.md`, the single defect register. **Do not
   re-describe them here.**
