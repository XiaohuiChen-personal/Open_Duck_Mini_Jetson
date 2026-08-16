# Plan — `v7_servo_safe`: fix SERVO-1 and SERVO-2, retrain, validate

**Drafted 2026-08-15. Revised after adversarial review. NOT YET APPROVED.**

> **Review record.** The first draft was attacked by 5 independent reviewers and
> every reported defect was independently re-verified; 2 BLOCKERs and 9 further
> real defects survived arbitration and are fixed below. The two blockers would
> each have wasted the entire GPU budget:
> **(1)** the resume command pointed `--load_run` into a log root that does not
> exist, so training would have died after Kit startup into a detached log while
> the launcher exited 0; **(2)** the torque-penalty weight was justified by
> arithmetic wrong by 1000×, making the term worth 0.01 % of the alive bonus —
> the entire SERVO-2 lever would have done nothing.
> A changelog is at the end.

Fixes two confirmed defects in one retrain, because both are reward-function
changes and the expensive part is paid once either way.

| | defect | fix |
|---|---|---|
| **SERVO-1** | `neck_pitch` driven onto its lower end stop during forward walking — 100 % of steps, 0.003° travel, 3.28 N·m RMS | add the 4 head joints to `joint_pos_limits` |
| **SERVO-2** | worst leg joint 2.735 N·m RMS = 174 % of rated; `dof_torques_l2` and `dof_acc_l2` both `None` — nothing prices torque | add a torque penalty on **commanded** torque |

---

## 0. Design decisions

**D1 — Fine-tune from `v6d`, not scratch.** Reward shaping only, so the
checkpoint loads unchanged (obs 53 / action 14). 3,000 iterations is the budget
that took `v6_robust` → `v6d` and moved tilt terminations 63.4 % → 11.5 %.
~2 h 06 m vs ~4 h.

**D2 — Do NOT change `effort_limit_sim`.** The plant stays byte-identical to
`v6d`'s, so the ten existing JSONs in `eval_results_rebuild/` remain a valid
control and we skip a **5-eval** control battery (~2 h 30 m).

> ⚠ **Gap this exposes.** `evaluate_policies.py` stamps `plant` = mass, bodies,
> obs/action dims, USD hash, MJCF sha256. It does **not** stamp
> `effort_limit_sim` or reward weights, so `regate_report.py`'s exit-2 mixing
> guard would **not** catch a comparison broken by changing them. **Stage 8 must
> extend the plant stamp before it runs.**

**D3 — Penalise `computed_torque` (pre-clip), not `applied_torque`.** Verified in
`actuator_pd.py`: `ImplicitActuator.compute()` stores
`computed_effort = stiffness·err_pos + damping·err_vel` and then
`applied_effort = _clip_effort(computed_effort)`. Isaac Lab's stock
`mdp.joint_torques_l2` squares the **clipped** tensor — and four leg joints
already sit at the clip, so over those steps the penalty is a **constant no
action can lower**: the policy is charged for torque *delivered*, not *demanded*,
and gets no gradient until it drops below 4.903 N·m. A 3-line custom term on
`computed_torque` keeps the gradient alive exactly where the problem is.

**D4 — Bundle SERVO-1 and SERVO-2 into one arm**, departing from the one-lever
rule deliberately: both are servo-protection reward terms and the validation
measures them independently (neck travel vs leg RMS), so a partial failure is
diagnosable without a bisect retrain.

---

## 1. Code changes — four edits

### 1a. `env_cfg.py` — SERVO-1

```python
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "right_ankle", "left_ankle", "right_knee", "left_knee",
+           "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        ])},
    )
```

**Why it bites immediately.** `mdp.joint_pos_limits` penalises violation of the
**soft** limits, set by `soft_joint_pos_limit_factor=0.9`. For `neck_pitch`
(hard −20.00°…+65.00°) the soft range is **−15.75°…+60.75°**. The policy sits at
**−20.00°, 4.25° beyond it**, so the term is active from iteration 1.

### 1b. NEW `isaac_lab_env/open_duck_mini_v2/torque_rewards.py` — SERVO-2

```python
def joint_torques_commanded_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """Sum of squared COMMANDED joint torque (pre-clip).

    Stock mdp.joint_torques_l2 squares `applied_torque`, which is post-clip.
    Four leg joints sit at the 4.903 N.m clip, so there the stock term is a
    constant and provides no gradient. `computed_torque` is what the actuator
    was ASKED for and stays differentiable above the clip.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.computed_torque[:, asset_cfg.joint_ids]), dim=1)
```

```python
-   dof_torques_l2 = None
+   dof_torques_l2 = RewTerm(func=torque_rewards.joint_torques_commanded_l2, weight=-1.0e-2)
```

**Weight, derived from measurement — not transplanted.**

```
measured over 14 actioned joints, v6d, forward walk:  mean Στ² = 51.84
  weight −1.0e-2  ->  penalty 0.518 / step
```

Against the current per-step reward budget (`alive_bonus` +9.37,
`imitation_reward` +1.27, tracking +0.84/+0.32), that is **≈4.7 %** — material,
not dominant. If the policy reaches the 1.0 N·m target, Στ² ≈ 14 → penalty 0.14,
so **the available gain is ≈0.38/step**, comparable to a third of the tracking
reward. That is the number that has to be worth earning.

**Why mainstream weights do not transfer.** legged_gym `-1e-5`, Isaac Lab
velocity `-1.0e-5`, G1 `-1.5e-7` are calibrated on robots whose joint torques are
10–50× ours. The term is **squared**, so transplanting the weight transplants
100–2500× less penalty. `-1e-5` here yields 0.0005/step — 90× smaller than the
smallest penalty already in the reward.

### 1c. `env_cfg.py` — housekeeping

Remove `left_antenna, right_antenna` from `joint_deviation_head`; M0b removed
them from the action space. Changes no reward (they hold default); stops the
config from implying they are controlled.

### 1d. `scripts/measure_joint_torque.py` and `measure_head_motion.py` — make the acceptance criteria measurable

The first draft set two criteria **neither script computes**. Add before any GPU
time:

- `measure_joint_torque.py`: retain the per-step torque tensor and report
  **longest continuous run above 3.923 N·m** (max consecutive-True run × 0.02 s),
  plus `--out <npz>`.
- `measure_head_motion.py`: add **`% of steps within 0.5° of a joint limit`** and
  an `--out <path>` flag, so the two Stage-5b conditions do not overwrite each
  other's `head_kinematics.npz` (the first draft's two runs would have).

Then **re-measure `v6d` with the new code** to establish real baselines. The
first draft's "0.08 s" figure was computed ad hoc in a chat session and exists
nowhere in the repo — it is not a citable baseline.

---

## 2. Tests, before any GPU time

New `tests/test_servo_protection.py`, `@pytest.mark.phase2`, parsing the AST of
`env_cfg.py` rather than grepping literals (TEST-1):

1. `test_head_joints_are_limit_protected` — all four head joints in the
   `joint_pos_limits` list. **Fails today.**
2. `test_torque_penalty_is_enabled` — `dof_torques_l2` is not `None`, weight
   negative and within `[-5e-2, -1e-3]` (the band the §1b derivation supports).
3. `test_torque_penalty_uses_preclip_torque` — the term's func resolves to
   `joint_torques_commanded_l2`, guarding D3 against a future "simplification"
   back to the stock term.
4. `test_soft_limit_actually_binds` — computes the soft limit from the MJCF range
   and `soft_joint_pos_limit_factor` and asserts −20.00° lies outside it. This is
   the test proving the fix *can* work.
5. `test_effort_limit_unchanged` — `STS3250_EFFORT_LIMIT_NM == 4.903`, pinning D2
   so an edit cannot silently invalidate the control comparison.

Gate: `python3 -m pytest tests/ -q` → 229 existing + 5 new, all passing.

---

## 3. Seed the v7 log root — **do not skip; the first draft did**

rsl-rl resolves `--load_run` **inside the current experiment's log root**
(`train.py:149,181`). `agent.experiment_name=open_duck_ppo_v7` moves that root to
`logs/rsl_rl/open_duck_ppo_v7`, which does not exist — `get_checkpoint_path`
raises `FileNotFoundError` after ~2 min of Kit startup, into a detached log,
while the launcher exits 0. This is why `0000-00-00_v6robust_seed` exists.

```bash
SRC=$ISAACLAB/logs/rsl_rl/open_duck_ppo_v6/2026-08-13_03-50-54_v6d_contact_wrench
DST=$ISAACLAB/logs/rsl_rl/open_duck_ppo_v7/0000-00-00_v6d_seed
mkdir -p $DST && cp $SRC/model_5998.pt $DST/ && cp -r $SRC/params $DST/
ls -l $DST/model_5998.pt    # pre-flight: must exist before launching
```

Keep the `experiment_name` override — Stage 5c's `LOGROOT` and the SHELL-1
log-root separation both require it.

---

## 4. Smoke — 100 iterations, ~5 min

```bash
cd $REPO && ./scripts/launch_training_detached.sh v7_smoke \
  --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0 --headless \
  --max_iterations 100 --resume \
  --load_run 0000-00-00_v6d_seed --checkpoint model_5998.pt \
  --run_name v7_smoke --video --video_length 200 --video_interval 1000 \
  agent.experiment_name=open_duck_ppo_v7
```

**At ~3 min, confirm it did not die** — the launcher exits 0 either way:

```bash
grep -c "Loading model checkpoint from:.*0000-00-00_v6d_seed/model_5998.pt" \
  .training_runs/v7_smoke.log     # must be >= 1
grep -c "Traceback" .training_runs/v7_smoke.log   # must be 0
```

Then verify all four:
- iteration counter starts at 5998
- `ContactRegimeEvent` still reads `26.77 N (2.729 kg) -> 1.34-5.35 N` (plant
  unchanged; if this moved, stop)
- `Episode_Reward/dof_torques_l2` appears **and is ≈ −0.5**, not ≈ −0.001 — the
  check that would have caught the 1000× error
- `Episode_Reward/joint_pos_limits` is more negative than v6d's −0.0040

Final checkpoint **`model_6097.pt`** (5998 + 100 − 1; verified against v6d, which
resumed at 2999 with `max_iterations 3000` and produced `model_5998.pt`).

---

## 5. Fine-tune — 3,000 iterations, ~2 h 06 m

```bash
cd $REPO && ./scripts/launch_training_detached.sh v7_servo_safe \
  --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0 --headless \
  --max_iterations 3000 --resume \
  --load_run 0000-00-00_v6d_seed --checkpoint model_5998.pt \
  --run_name v7_servo_safe --video --video_length 200 --video_interval 5000 \
  agent.experiment_name=open_duck_ppo_v7
```

Final checkpoint **`model_8997.pt`**.

**Watchdog every 15 min — TensorBoard only, NO GPU probe.** The first draft
proposed running the torque probe mid-run; that is a second Isaac Sim process on
a GPU already held by training, which violates the project's one-Isaac-job rule.
Instead:

```bash
~/IsaacLab/_isaac_sim/python.sh scripts/tb_summary.py --last 100 \
  --run_dir <v7 run dir> --tags Gait/duty_in_band_frac \
  --tags Episode_Reward/dof_torques_l2 --tags Train/mean_reward
```

- abort if `Gait/duty_in_band_frac` < 0.30 (v6d healthy: 0.9891)
- abort if `Train/mean_reward` < 150 sustained 300 iterations
- **`Episode_Reward/dof_torques_l2` trending toward zero is the SERVO-2 progress
  signal** — it is proportional to Στ², so it tracks the thing being fixed
  without a second GPU process.

**Tuning branch.** If at ~1500 iterations that tag has not fallen ≥30 % from its
start, the weight is too weak: kill, raise to −2e-2, restart from the seed.
Budget **one** restart.

> **EXECUTION DEVIATION, 2026-08-15, recorded because it departs from the rule
> above.** At iteration 7562 (1,564 into the fine-tune) the tag had fallen
> **20.0 %** (−0.2898 → −0.2318), i.e. short of the 30 % bar. The run was
> **allowed to finish anyway.** Why:
> 1. The acceptance criterion is a *direct* measurement (§6a, worst-leg RMS)
>    one hour away. The 30 % bar was a heuristic written before any weight→effect
>    data existed; killing on a proxy when the real number is imminent is the
>    weaker choice.
> 2. Finishing yields a measured **(weight → RMS)** point, turning the next
>    weight into an interpolation rather than a blind 2× guess — which matters
>    because overshooting risks collapsing the gait with no way to tell which
>    side of the target we are on.
> 3. The proxy is confounded: the tag is logged during *training* with wrench
>    and obstacles active, while the 2.735 N·m baseline was measured on the PLAY
>    task with disturbances off. They are not the same scale.
> 4. No safety signal — reward 229.38 (above v6d's 225.95), duty 0.9894.
> 5. SERVO-1 already reads as fixed (`joint_pos_limits` −0.0133 → −0.0018, an
>    86 % fall); that result is worth measuring rather than discarding.
>
> Cost if this is wrong: one extra hour before the same restart. The restart
> budget is untouched.

---

## 6. Validation — acceptance criteria fixed in advance

### 6a. Torque envelope (SERVO-2)

```bash
cd ~/IsaacLab && ./isaaclab.sh -p $REPO/scripts/measure_joint_torque.py \
  --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 \
  --checkpoint $CKPT --out $REPO/v7_torque.npz --headless
```
where `CKPT=$(ls $RUNDIR/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)`
— derived, not hard-coded, matching `v5_pipeline.sh:85`.

| criterion | bar | v6d today |
|---|---|---|
| worst **leg** joint RMS | **≤ 1.0 N·m** | 2.735 |
| any leg joint p99 at the 4.903 clip | **none** | 4 joints |
| longest continuous run > 3.923 N·m | **< 2.0 s** | re-measure in 1d |

### 6b. Neck kinematics (SERVO-1), both conditions, separate output files

```bash
... measure_head_motion.py --vx 0.2 --wz 0.0 --out $REPO/v7_head_straight.npz
... measure_head_motion.py --vx 0.0 --wz 0.5 --out $REPO/v7_head_turn.npz
```

| criterion | bar | v6d today (straight) |
|---|---|---|
| `neck_pitch` travel | **≥ 2.0°** | 0.003° |
| % of steps within 0.5° of the stop | **≤ 10 %** | 100 % |
| `neck_pitch` torque RMS | **≤ 1.5 N·m** | 3.28 |
| `head_yaw` travel — must NOT be suppressed | **≥ 10°** | 21.7° |

The last row matters: the fix must not buy thermal safety by freezing the head.

### 6c. Locomotion — the gate battery

```bash
LOGROOT=$ISAACLAB/logs/rsl_rl/open_duck_ppo_v7 \
RESULTS=$REPO/docs/jetson-mod/eval_results_rebuild \
COMPARISON_MD=$REPO/docs/jetson-mod/rebuild_comparison.md \
INCLUDE_ARGS="--include v7_servo_safe --include v6d_contact_wrench --include v6_robust_grid6" \
$REPO/scripts/v5_pipeline.sh v7_servo_safe \
  Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0
```

5 evals + 4 videos, ~2 h 55 m. **The pipeline renders `turn_wz05`, NOT
`turn_wz03`** — and `turn_wz03` is an AGENTS.md-mandated audit condition, so it
needs its own step (the first draft gated on a clip nothing produced):

```bash
cd ~/IsaacLab && ./isaaclab.sh -p $REPO/scripts/play_policy.py \
  --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 --num_envs 2 \
  --checkpoint $CKPT --headless --video --video_length 1000 \
  'env.commands.base_velocity.ranges.ang_vel_z=[0.3,0.3]'      # ~4 min
```

**HARD gates — v7 vs `v6d`, not vs `v6_robust`.** The first draft gated against
`v6_robust`, which falls **100.000 %** on the wrench and **34.245 %** on
obstacles — so v7 could regress from 70.365 % to 99.9 % and still "pass". That
is vacuous in exactly the direction a torque penalty is most likely to break
things. Bars, set now:

| gate | bar |
|---|---|
| gait valid | **≥ 5/6** |
| open-field falls | **≤ 1.0 %** |
| push, v4 rule | **≤ 1.0 %** (v6d 0.104) |
| push, v5 rule | **≤ 1.0 %** (v6d 0.026) |
| sustained wrench | **≤ 80 %** (v6d 70.365) |
| obstacle graze | **≤ 12 %** (v6d 7.318) |
| ref RMS | within **1.0°** of `v6_robust` 4.612 |
| video | 5 conditions PASS, incl. `turn_wz03` |

`v6_robust` remains an additional floor (v7 must still beat it on all four), but
it cannot be the only floor.

---

## 7. If it passes / if it fails

**Passes:** export to `exported_policies/v7_servo_safe_ppo/`, **replacing**
`v6d_contact_wrench_ppo/` (the repo keeps one locomotion) after backing the
outgoing archive up outside the repo; regenerate and verify
`deployment_contract.json`; update `AGENTS.md`, `locomotion_selection.md`,
journal Runs 25–26; mark SERVO-1/SERVO-2 FIXED with measured numbers and retire
their checks; add a "measured after fix" section to `servo_torque_budget.md`.

**Fails:** `v6d` remains the mainline; nothing is deleted until v7 passes.
6a/6b separate the diagnosis:

| symptom | reading | next |
|---|---|---|
| leg RMS > 1.0, neck fixed | penalty too weak | raise to −2e-2 |
| neck still pinned, legs fixed | limit penalty loses to the balance benefit | raise `joint_pos_limits` weight, or widen the neck's hard range in CAD |
| both fixed, gait < 5/6 | the robot needs that torque | ~1.0 N·m infeasible; escalate to mass/gearing (S.8) |
| both fixed, contact gates regress past the bars | robustness was bought with torque | owner decision |

## 8. CONDITIONAL — lower the effort ceiling

Only if 6a shows sustained excursions > 3.923 N·m. Requires **first** extending
the `plant` stamp to include `effort_limit_sim`, then re-gating `v6_robust`
**and** `v6d` under the new ceiling — **+10 evals (5 per policy: `_grid6` plus
four battery files), ~5 h**. Not in this plan's budget.

---

## Budget

| stage | wall clock |
|---|---|
| 1–2 code, tests, script instrumentation | ~1 h, no GPU |
| 1d re-measure v6d baselines | ~10 min |
| 3 seed | <1 min |
| 4 smoke | ~5 min |
| 5 fine-tune | **~2 h 06 m** |
| 6a + 6b probes (3 runs) | ~15 min |
| 6c battery + 4 videos | **~2 h 55 m** |
| 6c `turn_wz03` | ~4 min |
| 7 export + docs | ~45 min, no GPU |
| **total** | **≈ 7 h 20 m**, of which ~5 h 25 m is GPU |
| contingency: one weight-tuning restart | +2 h 06 m |

---

## Changelog — what the adversarial review changed

| # | severity | defect | fix |
|---|---|---|---|
| 1 | BLOCKER | `--load_run` resolved into a non-existent v7 log root; training dies after Kit startup, launcher exits 0 | Stage 3 seed step + post-launch death check |
| 2 | BLOCKER | torque weight justified by arithmetic wrong by 1000× (−2e-5 × 52 = 0.001, not 1.1) | re-derived from measured Στ²=51.84 → **−1e-2**; smoke now checks the tag reads ≈−0.5 |
| 3 | MAJOR | final checkpoints named `model_6098/8998` — off by one | `6097` / `8997`, and derived dynamically in probe commands |
| 4 | MAJOR | stock `joint_torques_l2` squares **post-clip** torque → no gradient where saturated | custom pre-clip term (D3) |
| 5 | MAJOR | G-R4 gated against `v6_robust` (100 % wrench) → vacuous | explicit bars against `v6d` |
| 6 | MAJOR | `turn_wz03` is a hard gate but nothing in the plan renders it | explicit `play_policy.py` step |
| 7 | MAJOR | mid-run GPU torque probe violates one-Isaac-job-at-a-time | TensorBoard-only watchdog |
| 8 | MAJOR | two acceptance criteria measured by neither script | Stage 1d adds them first |
| 9 | MAJOR | both 6b runs would overwrite `head_kinematics.npz` | `--out` flag |
| 10 | MINOR | control battery is 5 evals not 4 | D2 saving ~2 h 30 m; Stage 8 ~5 h |
| 11 | MINOR | "0.08 s" baseline had no provenance in the repo | re-measure in 1d |
