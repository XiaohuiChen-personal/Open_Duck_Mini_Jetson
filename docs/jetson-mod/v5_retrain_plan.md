# v5 Contact-Rich Retrain — Execution Plan (Task 2.8)

> **Defects do not live in this file.** Every issue mentioned here is recorded,
> verified and tracked in **`docs/jetson-mod/known_issues.md`** — that is the
> single register. This document is the v5 *plan and run record*: it is
> currently the only place the v5a–v5d runs are written up at all
> (`known_issues.md` DOC-2), so it is kept for that history.

**Status:** PLAN v2 (drafted 2026-07-28; revised same day after a 3-lens
adversarial review — code reality, protocol coverage, experiment design — all
findings folded in below and marked "(review)"). Executes `task_plan.md`
Task 2.8.
**Goal:** a locomotion policy that survives *contact* — walking beside,
brushing, and being pressed into obstacles, including while rotating — without
regressing the v4_robust flat-ground/push record.
**Baseline:** `v4_robust` = `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`
(gait gate 5/5, 0.00% unpushed falls, 6.84% pushed falls, ref RMS 4.49°).
**Motivating evidence:** Duck Embody benchmark, 10 falls / 12 trials, 1.58
falls/policy-min — 7 rotation-under-contact, 1 free-space rotation, 2
sustained press (`task_plan.md:1466-1486`; `duck-embody/results/audit_notes.md`).

Everything below is grounded in a line-level survey of both repos and the
installed Isaac Lab (v2.3.2, isaaclab 0.54.3, rsl-rl-lib 5.0.1) taken
2026-07-28, plus the review pass. Where a mechanism was verified in source,
the file is named.

---

## 1. Decision: role of the Duck Embody apartment scene

**Question asked:** should training use the scene created in `../duck-embody`?

**Decision: NO for training; YES (unchanged) as the held-out evaluation gate.**
Training uses procedural primitive obstacles whose *statistics* come from the
apartment. Three reasons:

1. **Benchmark validity.** The apartment is the benchmark world. If v5 trains
   in it, the planned benchmark re-run stops measuring generalization and
   starts measuring memorization of one furniture layout. The Hartmann et al.
   result the task plan builds on (Deep Compliant Control, ETH CRL 2024) is
   exactly that this is unnecessary: base-wrench + push training generalized
   zero-shot to box collisions, drags, and clutter that were never meshed.
   Keep the apartment out-of-distribution and let it certify transfer.
2. **Architecture.** The duck-embody scene is Isaac Lab-native but
   single-instance by design: absolute `/World/Apartment/...` prim paths,
   `collision_group=-1`, `num_envs=1` (`duck-embody/duck_embody/env/scene_builder.py:383,421`,
   `embody_env_cfg.py:160`). Per-env replication for 4096 envs does not exist
   and would be new machinery for negative scientific value (reason 1).
3. **Throughput.** ~52 prims + SimReady convex-decomposition colliders × 4096
   envs is pure overhead when the training signal ("rotation while loaded",
   "sustained press") needs only a box face and a corner.

**What we DO take from duck-embody (all pure-python, no kit dependency):**
- Obstacle statistics for the training boxes: wall height 0.7 m, thickness
  0.03 m, doorway width 0.35 m, furniture heights up to the 0.746 m fridge
  (`apartment_layout.py:50-58`, `assets/manifest.json`); measured swept gait
  half-width 0.11–0.15 m and the 9 cm clean-clearance boundary
  (`configs/benchmark.yaml:131-139`).
- The deployment thresholds as training terminations (§3.5): tilt 60°, height
  0.09 m (`embody_env_cfg.py:120-122`) — train-time fall definition ==
  deployment fall definition.
- The evaluation stack: T2.4 physics pass; gap-hunt S0/S1/S3/S4/S5 (S4 is a
  corridor `move()`-semantics check at measured 9/3 cm clearances — adjacent
  to, not a substitute for, a policy graze course, which §8 adds); the 10
  frozen fall recordings as a regression suite (§3.10); the paid benchmark
  re-run as the final gate (§7). **Correction from review:** gap-hunt S2 is a
  forced-fall *harness* diagnostic (it PASSES only if the robot falls and the
  diagnostics/video plumbing is intact, `smoke_gap_hunt.py:927-931`) — for a
  press-surviving v5, S2 reporting "no fall within budget" is the *desired
  policy outcome*; §7 gate 3 scores it accordingly.

---

## 2. v5 design summary (delta vs v4_robust)

v5 inherits everything from `OpenDuckRobustEnvCfg` (dynamics DR, 59-dim
asymmetric actor, 8.94 rad/s velocity limit) and changes:

| Axis | v4_robust (verified current) | v5 |
|---|---|---|
| Impulse pushes | linear ±0.3 m/s x/y, every 8–14 s | every **5–10 s** (upstream-exact); linear ramped **0.4 → 0.7 m/s** per axis over the first ~1.5k ft-iters; **+ rotational: yaw ±1.0, roll/pitch ±0.5 rad/s** (native: extra keys in `velocity_range`, `events.py:1067`). *Amends Task 2.8's "0.5→1.3 m/s" — full evidence in §11.4: a=0.70 reproduces the upstream Open Duck Playground push distribution for this exact robot to within 3%, matches the Froude-scaled median (0.741 m/s) across seven published configs, and matches the capture-point limit for the measured ~0.10–0.12 m leg reach. Ramp advances only while TB push-survival stays above ~70%* |
| Sustained wrench | none (`base_external_force_torque` zeroed) | custom event: 2–6 s constant force, 5–30% of measured robot weight, lateral-biased, random torso point, τz ±0.05–0.15 N·m; ~50% of envs; **≥30% of activations forced while \|wz cmd\| ≥ 0.25** |
| Obstacles | none | one kinematic box per env, active in ~25% of envs per episode, spawned 0.3–0.9 m ahead **tangent to the commanded path**, ±0.10–0.35 m lateral offset (parked underground otherwise); obstacle envs' commands biased toward \|wz\| ≥ 0.25; **realized contact rate logged to TB with a smoke-run floor** (review: placement alone does not guarantee contact — the modal fall mechanism must be measurably trained) |
| Commands | heading-servo wz 100% of envs, resample 10 s, standing 2% | `rel_heading_envs=0.7` (native per-env split, `velocity_command.py:139-159`); **direct-wz envs sample ±0.7 with ~40% of draws banded into ±(0.35–0.7)**, while heading-servo envs stay clipped at the deployment-exact ±0.5 (§11.2 — upstream trains this robot at ±1.0, and 5 of 10 falls sat exactly on our ±0.5 boundary, so 0.5 must become an interior point); resample (2, 10) s, standing 8% |
| Termination | trunk contact > 1 N = death (−200) | **fall-only**: tilt > 60° or root height < 0.09 m (−200 kept). Trunk/obstacle contact carries gradient, not death |
| Imitation | always on (gated only for zero cmd) | **disturbance-gated** (Hartmann): during wrench / push / trunk-contact events and 1.0 s after → imitation ×0.1, tracking terms frozen to pre-disturbance running mean. *Review fix: NO \|ang_vel_xy\| trigger — a rate trigger is self-inducible and gate-freezing is provably profitable under hard tracking; gate only on external event flags, and log per-env gate duty to TB* |
| Reference match | unweighted L2 over (vx,vy,wz) → wz dominates | span-normalized per-axis L2 |
| Reference library | 240 cells, **no vy=0, no wz=0 rows** (verified) | patched pkl v2: synthesized vy=0 / wz=0 rows + (0,0,±0.5) pair (§5) — training only. *Review re-rating: this is label hygiene with a small effect (the synthesized (0,0,0.5) cell differs from the original nearest cell by only 1.19° RMS, and the composite has no wz term at all) — the load-bearing fix for the wz=0.5 falls is the command-exposure change above, and gains will be attributed there* |
| New penalties | — | `ang_vel_xy_l2` −0.05; `feet_slide` −0.1; head/lower-leg **ground**-contact −0.5 (scoped to obstacle-free envs, §3.5); `flat_orientation` softened −2.0 → −1.0 **with a 0.1 rad dead zone** (custom term — the dead zone is what permits lean-against-load; review restored it from Task 2.8) |
| Init | from scratch | **fine-tune from v4_robust `model_2999.pt`** (optimizer + iteration restored by rsl-rl 5.0.1 `load()`; PPF anchor as fallback) |

Obs/action contract untouched: 59-dim actor / 62-dim critic / 16 actions —
required both by `strict=True` checkpoint loading (`rsl_rl/algorithms/ppo.py:457-462`)
and by the frozen Jetson/duck-embody deployment interface.

---

## 3. Code changes — file by file

### 3.1 New: `isaac_lab_env/open_duck_mini_v2/contact_events.py`

1. **`SustainedWrenchEvent(ManagerTermBase)`** — class-based interval event,
   `interval_range_s=(step_dt, step_dt)` so it ticks every env step and owns
   its own per-env timers (verified pattern: `event_manager.py:206-232` gives
   class terms `(env, env_ids, **params)` and a `reset(env_ids)` hook).
   Per-env state: `active`, `time_remaining`, `cooldown`.
   - Activation: sample candidate envs at a rate targeting ~50% duty-of-envs;
     **enforced co-occurrence**: ≥30% of each activation batch drawn only from
     envs whose current command has \|wz\| ≥ 0.25 (read
     `env.command_manager.get_command("base_velocity")`).
   - Wrench: force magnitude 5–30% of robot weight **measured at init** from
     the articulation's body masses — do NOT hardcode: the task plan's
     "0.8–4.8 N" assumed a 1.6 kg robot; the sim robot is ~2.66 kg →
     ~1.3–7.8 N. Direction biased lateral (body frame), application point
     uniform over the trunk AABB, τz ∈ ±(0.05–0.15) N·m, duration 2–6 s.
   - Apply via `robot.permanent_wrench_composer.set_forces_and_torques(forces,
     torques, positions=..., body_ids=[trunk_id], env_ids=..., is_global=False)` —
     persistent until overwritten; auto-cleared per env on episode reset by
     `scene.reset` (`rigid_object.py:113-132`, `interactive_scene.py:443-455`).
     On expiry, set zeros for those envs. **Sole owner of the robot's
     permanent composer** → v5 cfg sets
     `events.base_external_force_torque = None` (same composer).
   - Exposes ramp targets (`max_force_frac`, …) as attributes for curriculum
     (`modify_env_param` address
     `"event_manager.cfg.sustained_wrench.func.<attr>"`, the documented
     pattern in `curriculums.py:57-60`).
   - Maintains the **disturbance-gate state** (§3.2): `gate=1` for its envs
     while active and 1.0 s after. Additional triggers via
     `mark_disturbed(env_ids)`: impulse pushes (below) and **trunk contact
     force > 1 N** from the existing `contact_forces` sensor (external,
     covers obstacle-contact recovery windows). *No ang-vel trigger — see §2
     (review): a self-measurable rate trigger lets the policy induce the gate
     and harvest the frozen tracking constant while abandoning tracking.*
   - **TB logging** (via `env.extras` / episode sums): gate duty per episode,
     wrench-active duty, co-occurrence fraction, push-survival rate.
2. **`push_and_mark`** — thin wrapper around `mdp.push_by_setting_velocity`
   that also calls `mark_disturbed(env_ids)`.
3. **`place_obstacle`** — reset-mode event function (declared *after*
   `reset_base` — reset terms run in cfg declaration order — so the robot
   pose and freshly resampled command are readable): with p=0.25 mark the env
   obstacle-active and place its box at robot pos + R(yaw)·[fwd U(0.3, 0.9),
   lat ±U(0.10, 0.35)], **yaw aligned tangent to the commanded velocity
   direction** (review: maximize graze probability); else park at (0, 0, −2)
   under the plane. Uses `obstacle.write_root_pose_to_sim`. Exposes the
   per-env `obstacle_active` mask (module registry) for §3.2/§3.5 scoping,
   and logs **obstacle-contact steps per episode** to TB (trunk sensor
   already exists, `env_cfg.py:198-202`).

### 3.2 New: `isaac_lab_env/open_duck_mini_v2/gated_rewards.py`

- **`DisturbanceState`** — module-level per-env registry (same pattern as
  `imitation_reward._instances`): gate flag + hold timer + `obstacle_active`
  mask; written by §3.1, read by reward terms.
- **`GatedTrackLinVel` / `GatedTrackAngVel`** (class RewTerms): compute the
  stock exp-kernel reward; maintain a per-env EMA (α ≈ 0.01 at 50 Hz ≈ 2 s
  window) while `gate=0`; while `gate=1` return the frozen EMA (recovery not
  punished as tracking error; the constant carries no action gradient —
  Hartmann's freeze). Gate is event-driven only (§3.1), so the freeze cannot
  be self-triggered on the open plane.
- **`flat_orientation_deadzone`** — penalize `max(0, ‖g_xy‖ − 0.1)²`
  (restores Task 2.8's dead zone; leaning into a press inside 0.1 rad is
  free, review).
- **`ground_contact_penalty`** — head + lower-leg contact > 1 N, **active
  only in obstacle-free envs** (where the only touchable thing IS the
  ground, so no filtered sensor is needed; in obstacle envs the term is off,
  matching Task 2.8's "no contact penalty in obstacle envs" rule — review:
  the earlier global `undesired_contacts` version would have trained
  obstacle avoidance, the exact forbidden failure mode, since the 0.7 m box
  is taller than the robot).
- `ImitationReward` gains optional `disturbance_gate_scale` (default `None`
  = off; v3/v4 tasks and tests untouched): composite ×=
  `(1 − (1−scale)·gate)`, scale ≈ 0.1.

### 3.3 New: `isaac_lab_env/open_duck_mini_v2/duck_commands.py`

`BandedWzVelocityCommand(UniformVelocityCommand)` + cfg: for non-heading
(direct-wz) envs, draw wz from a mixture — 60% uniform over (−0.5, 0.5), 40%
uniform over ±(0.35–0.5) magnitude band (review: with plain uniform sampling
only ~30%×30% ≈ 9% of envs would hold killer-band wz at any time vs Task
2.8's "~30% holding ±0.35–0.5"). Heading envs unchanged. Fallback if this
class misbehaves in smoke: revert to the native sampler and record the
diluted exposure explicitly.

### 3.4 Modified: `imitation_reward.py`

- **Span-normalized nearest-motion match**: divide each axis of
  `vel_cmd − self._velocities` by the library's per-axis span (vx 0.370,
  vy 0.222, wz 2.333) before the squared sum at `IM:201-205`. Behind
  `normalized_match: bool` (default False) — v3/v4 reproducibility and the
  phase-bug regression tests (`tests/test_isaac_lab_env.py:255-282`) intact.
- **Configurable library path** (`reference_pkl` param, default current
  file): v5 trains on the patched pkl; the original stays the immutable
  dataset its README declares.
- The `disturbance_gate_scale` hook from §3.2.

### 3.5 Modified: `env_cfg.py` — new cfg classes

```
OpenDuckContactEnvCfg(OpenDuckRobustEnvCfg)             # v5 trainer
OpenDuckContactEnvCfg_PLAY(OpenDuckContactEnvCfg)       # deterministic playback (wrench/obstacles/pushes off)
OpenDuckWrenchEvalEnvCfg(OpenDuckContactEnvCfg_PLAY)    # sustained-wrench eval (wrench ON, deterministic tiers)
OpenDuckObstacleEvalEnvCfg(OpenDuckContactEnvCfg_PLAY)  # obstacle-graze eval (box ON at fixed offsets — feeds §8 cond. 7)
OpenDuckContactPushEvalEnvCfg(OpenDuckContactEnvCfg_PLAY) # push eval under v5 fall-only terminations (gate-2 companion)
```

`OpenDuckContactEnvCfg.__post_init__` (values per §2):
- scene: `obstacle` — single `RigidObjectCfg`, kinematic cuboid
  0.5 × 0.5 × 0.7 m (one geometry for all envs keeps
  `replicate_physics=True`; `MultiAssetSpawnerCfg` variety is a later lever —
  heterogeneous spawns force `replicate_physics=False`,
  `interactive_scene.py:57-68`).
- events: `push_robot` → `push_and_mark`, interval (4, 8) s, ramped linear
  ±0.4→0.7 + roll/pitch ±0.5 + yaw ±1.0; `sustained_wrench`;
  `base_external_force_torque = None`; `place_obstacle`.
- commands: swap in `BandedWzVelocityCommand` cfg; `rel_heading_envs=0.7`,
  `rel_standing_envs=0.08`, `resampling_time_range=(2.0, 10.0)`.
- terminations: `base_contact = None`; `tilt = bad_orientation(1.047)`;
  `fell_low = root_height_below_minimum(0.09)` (flat-terrain-only term —
  fine, we stay on plane terrain).
- rewards: gated tracking terms; imitation `disturbance_gate_scale=0.1`,
  `normalized_match=True`, `reference_pkl=v2`; `ang_vel_xy_l2` −0.05;
  `feet_slide` (g1-style) −0.1 on `foot_assembly` / `foot_assembly_2`;
  `ground_contact_penalty` −0.5 with
  `body_names=["head", "knee_and_ankle_assembly.*"]` (review: no body is
  named "knee" — the links are `knee_and_ankle_assembly{,_2,_3,_4}` and the
  penalty covers the whole lower-leg link); `flat_orientation_l2 = None`,
  replaced by `flat_orientation_deadzone` −1.0.
- curriculum: `modify_env_param` ramps for push magnitude and wrench force,
  gated on `env.common_step_counter` (starts at 0 on launch; 1.5k iters ×
  24 steps/iter = 36k steps), parameterized by a `ramp_start_steps` class
  attribute (0 for fine-tunes; delayed for the scratch control, §6).

Crawl-exploit watch: removing contact-death re-opens the run-2-era
low-posture exploit in principle. Mitigations: the imitation composite still
demands the reference gait, `fell_low` terminates true crumples, and the
eval duty gate + video audit catch it. Escalation path if it appears: trunk
*ground*-contact termination via a dedicated one-body
`ContactSensorCfg(prim_path=".../trunk_assembly",
filter_prim_paths_expr=[".../Obstacle"])` and net-minus-filtered force
(`contact_sensor_cfg.py:52-72`) — spec'd, built only if needed.

### 3.6 Modified: `agents/rsl_rl_ppo_cfg.py`

`OpenDuckContactPPORunnerCfg(OpenDuckRobustPPORunnerCfg)`:
`experiment_name = "open_duck_ppo_v5"`. Hyperparameters unchanged
(adaptive-KL LR is the fine-tune-friendly default already in use).

### 3.7 Modified: `open_duck_mini_v2/__init__.py`

Register `Isaac-Velocity-Rough-OpenDuck-Contact-v0` / `-Contact-Play-v0` /
`-WrenchEval-v0` / `-ObstacleEval-v0` / `-ContactPushEval-v0` (copy the
Robust pattern `I:53-81`), update the docstring index.

### 3.8 Modified: `tests/test_isaac_lab_env.py`

Add class/registration assertions for the new cfgs + runner cfg (mirroring
`:92-101`, `:408-417`); add a grid-content test for the patched pkl (vy=0,
wz=0, (0,0,±0.5) rows exist; original pkl untouched, md5 pinned); keep every
existing assertion green (new classes only *add* strings — verified
compatible with the whole-file greps).

### 3.9 New: eval/audit tooling (main repo)

- **`scripts/eval_common.py`** — the env-building/policy-adapter/condition
  helpers factored out so both `evaluate_policies.py` and the new audit
  script can use them. (Review: `evaluate_policies.py` is
  explicitly not importable — module-level AppLauncher parsing exits on
  import, `evaluate_policies.py:80,1156-1170` — so "reuse" means refactor,
  not import.)
- **`scripts/audit_rollout.py`** — §8.
- **`scripts/make_filmstrip.py`** — §8.

### 3.10 duck-embody (sibling repo, post-freeze work)

- `scripts/replay_falls.py` — **new** (every building block verified
  present: `SimSession.reset(seed, spawn)`, `scripted_drive`, macros,
  `Recorder`): parameterized scenario runner (`--checkpoint`, spawn pose,
  command/macro sequence) encoding the 10 frozen falls: seed-101 sofa
  `move(1.5)` (+ opus5 `move(1.2)` twin), the 5 bump-then-turn-at-±0.5
  sites (command sequences lifted from `results/raw/*.json` `tool_calls`),
  the 3 remaining move-topples, and the S2 counter press. Emits
  per-scenario survive/fall + mp4 + filmstrip. **Validity protocol
  (review):** the suite is meaningful only once **v4_robust reproduces
  ≥8/10 of the original falls under replay** — re-encode any scenario that
  doesn't reproduce before it counts; and because Kit is not
  bitwise-deterministic, every scenario runs **5 repetitions** per policy
  and gates on survival fraction, never a single binary trial.
- Add `--checkpoint` passthrough to `smoke_physics_pass.py` and
  `smoke_gap_hunt.py` (both hardwire `policy/model_2999.pt` via
  `session.py:23`; neither is in `FROZEN_FILES`, so this does not disturb
  the freeze).
- Final acceptance only: vendor v5 into `policy/` (checkpoint + params +
  checksums + README provenance — the freeze hash does NOT cover checkpoint
  bytes, so vendoring is the actual traceability), bump the parent-commit
  pin, `--freeze`, new batch dir, re-run. **Prerequisite (review): the
  freeze refuses any dirty tracked file, and the pin bump edits a frozen
  file — so this step requires owner-approved commits in BOTH repos (the
  parent v2 commit containing v5 = the new pin target; the duck-embody
  vendoring + pin commit). Both repos carry a no-commit-without-asking
  rule; the approval is requested explicitly at that point.**

---

## 4. Why each mechanism is safe to build (verified feasibility)

1. **Rotational pushes** — `push_by_setting_velocity` reads keys
   `["x","y","z","roll","pitch","yaw"]`, missing keys → (0,0)
   (`events.py:1067`). Zero new code.
2. **Sustained wrench** — permanent `WrenchComposer` persists across steps,
   re-applied every physics substep by `scene.write_data_to_sim` inside the
   decimation loop (`manager_based_rl_env.py:182-189`), auto-cleared on env
   reset. No built-in duration event exists (grep verified) → §3.1's class
   term is required and sufficient.
3. **Direct-wz env fraction** — `rel_heading_envs` is natively a per-env
   split re-drawn each resample (`velocity_command.py:139`); §3.3 adds the
   banded magnitude bias on top.
4. **Fine-tune** — `--resume --load_run <dir> --checkpoint model_2999.pt`
   restores actor+critic+optimizer+iteration (rsl-rl-lib 5.0.1 `load()`
   defaults, `ppo.py:444-466`); env cfg is NOT in the checkpoint, so changed
   rewards/events/terminations/scene are legal; obs/action dims must match
   (they do). Resume resolves only under the current experiment's log root
   and `load_run=".*"` would pick the just-created empty dir — hence the
   §6 copied-run-dir + explicit `--load_run/--checkpoint` (resolution
   verified against `get_checkpoint_path`). **Iteration/checkpoint naming
   (review, verified in rsl-rl 5.0.1):** `learn()` runs
   `range(2999, 2999+3000)` and the final save is
   `model_{current}.pt` = **`model_5998.pt`** — `model_5999.pt` never
   exists; interval saves land at 3000, 3100, …, 5900; the first resumed
   log line re-uses index 2999.
5. **Curriculum ramps** — `mdp.modify_env_param` addresses event params and
   class-term attributes by dotted path, gated on `env.common_step_counter`
   (counts policy steps; starts at 0 on any launch, including resumes).
6. **Reset-event ordering** — reset terms execute in cfg declaration order,
   so `place_obstacle` declared after `reset_base` sees the final robot pose
   and the freshly resampled command (verified in event manager dispatch).

---

## 5. Reference-library patch (the wz=0 / vy=0 gap)

Verified grid: full 6×4×10 product, vx ∈ {−0.148…0.222}, vy ∈ {±0.111,
±0.037}, wz ∈ {−1.111…+1.222 in 0.259 steps} — **no vy=0, no wz=0**; all
240 cells share period 0.54 s / 27 steps @ 50 Hz. Turn-in-place (0,0,0.5)
snaps to (0, −0.037, 0.444); straight walk snaps to wz=−0.074.

**Honest sizing (review).** The synthesized (0,0,0.5) cell differs from the
original nearest cell by **1.19° RMS** over the 10 leg joints, and the
imitation composite contains **no wz term** (it tracks xy linear velocity
only — no pkl dim encodes yaw rate). So the "imitation fights the tracking
term" effect at wz=0.5 is real but small (~0.07/step of joint-pos penalty
vs ±1.5 tracking and +10 alive). The patch is kept as **cheap label
hygiene**; the load-bearing fix for the wz=0.5 falls is the sustained
direct-wz command exposure (§2), and any wz=0.5 improvement is attributed
there unless an ablation says otherwise.

**Patch method — coefficient interpolation** (new pkl, original untouched):
evaluation is linear in coefficients and all cells share period/step count,
so convex combinations are well-defined; blend order (vy first vs wz first)
is immaterial by linearity:
- vy=0 rows: mean of (vx, +0.037, wz) and (vx, −0.037, wz).
- wz=0 rows: blend wz=−0.074 / +0.185 at (0.7143, 0.2857).
- (0,0,±0.5): at the synthesized vy=0 slice, blend wz=0.444 / 0.704 at
  (0.7846, 0.2154); mirror for −0.5. (Weights verified exact by review.)
- Contact dims (32–33) blend continuously; re-binarize at 0.5 (review
  verified the parent contact schedules are phase-aligned; only 2–3/27
  frames land near threshold).

**Checker (before any run; no Isaac):** assert each synthesized cell stays
**within its parent cells' envelopes** (true for convex combinations) — NOT
within soft joint limits: the parents themselves exceed the knee soft limit
(peaks 1.78–1.94 rad vs ±1.5708; the runtime clamps, `IM:232-235`), so a
soft-limit assertion would reject every row (review). Also assert contact
duty in [40,90]% and L/R symmetry of vy=0 rows, and plot (0,0,0.5) vs its
parents for the eye check. **Known limit:** no pkl dim encodes yaw rate, so
the checker cannot verify the blended cell truly turns at 0.5 rad/s —
that's confirmed by the §8 frame audit of the trained policy, not by the
checker. Fallback if blends look kinematically wrong: run the upstream
reference-motion generator (placo) for the missing rows — budgeted spike.

**Eval comparability rule (rewritten after review — was a blocker):**
- The 5 legacy conditions are scored against the **original pkl** — true
  1:1 with v4_robust's 4.49°.
- The **wz=0.5 condition is excluded from the RMS-parity bar**: the
  original library's nearest cell there is the side-stepping reference the
  patch exists to move away from, so a correct v5 would carry a ~1.19°
  built-in penalty — larger than the whole 1.0° margin — while v4 is
  favored by construction. wz=0.5 is gated on reference-free metrics
  (falls, gait duty, wz tracking error, video), and RMS at wz=0.5 is
  reported against **both** pkls for **both** policies (separate
  `--reference_pkl` invocations, distinct JSON names) as descriptive data.

---

## 6. Run matrix and sequencing

All runs: 4096 envs, `--video --video_length 200 --video_interval 5000`,
seed 42, launched via `./scripts/launch_training_detached.sh` from tmux
session "duck", ONE Isaac/GPU job at a time, evals in the gaps (AGENTS.md
rules 6–7). Fine-tunes: `--max_iterations 3000` ≈ 1.9 h measured on this
GB10.

**Prep (one-time):**
```bash
mkdir -p ~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5
cp -r ~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43 \
      ~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/0000-00-00_v4robust_seed
```
(explicit `--load_run 0000-00-00_v4robust_seed --checkpoint model_2999.pt`
targets it unambiguously; the copied name sorts before any new timestamped
run dir.)

**Run 0 — `v5_smoke`** (resumed, `--max_iterations 100`, ~10 min).
Pass criteria (review-hardened, all from TB/log, not vibes):
- resume banner starts at iteration **2999**; final checkpoint
  **`model_3098.pt`** (2999+100−1);
- wrench co-occurrence fraction ≥ 0.30; gate duty in a sane band (>0, <30%);
- obstacle-env trunk-contact rate logged — **record the baseline and set
  the Run-1 floor from it** (placement→contact is otherwise unmeasured);
- direct-wz band exposure ≈ 12% of envs (0.3 × 0.4);
- obstacle placement + a push visible in the train video; no NaNs; dims 59/62.

**Run 1 — `v5a_gated_ft` (primary):** the full §2 recipe, fine-tuned.
```bash
./scripts/launch_training_detached.sh v5a_gated_ft \
  --task Isaac-Velocity-Rough-OpenDuck-Contact-v0 --headless \
  --max_iterations 3000 --resume \
  --load_run 0000-00-00_v4robust_seed --checkpoint model_2999.pt \
  --video --video_length 200 --video_interval 5000
```
Final checkpoint: `model_5998.pt`. Journal compute-parity note (the AMP
run-8 precedent): the resumed policy totals 6000 training iterations —
report accordingly.

**Run 2 — `v5b_ungated_ft` (gate ablation):** identical except
`disturbance_gate_scale=None` and stock tracking terms. Isolates the one
novel reward mechanism. **Conclusion discipline (review):** one training
seed per arm — the journal verdict is recorded as a *single-seed
observation*; if the v5a-vs-v5b delta would change the deployed
configuration, re-run the deciding pair on one more seed (~2 h each) before
recording a win/retire verdict.

**Run 3 — `v5c_scratch` (control, de-confounded per review):**
`--max_iterations 6000` (compute parity with v5a's 3000+3000) and
`ramp_start_steps` delayed by 72k steps (=3000 iters) so DR arrives only
after a basic gait can exist — otherwise the scratch arm faces from step 0
the pushes/wrenches/obstacles that v4 never had while learning to walk, and
"fine-tuning preserves gait quality" would be predetermined. ~3.8 h.

**Decision-gated, not a blind queue:** evaluate Run 1 before finalizing
Run 2/3 configs. Contingency **v5d_ppf_anchor** (only if v5a/b degrade the
gait: legacy-condition RMS drift ≫ v4, gate < 4/5 there, or visible limp):
add the PPF-style L2 anchor — frozen v4_robust actor inside a class
RewTerm, penalty ‖μ_θ(o) − μ_v4(o)‖², weight ~5 annealed to 0.

**Per-candidate eval batch (GPU gaps, sequenced by hand):**
1. `evaluate_policies.py`, 6-condition grid (`--conditions` appends
   `0,0,0.5` — zero code change): the v5 candidate on `-Contact-Play-v0`
   AND **`v4_robust` re-run on the same 6 conditions** (JSON
   `v4_robust_grid6`). Plus the dual-pkl wz=0.5 RMS runs (§5).
   **Results dir (review — was a blocker):** all v5-era JSONs go to a NEW
   `docs/jetson-mod/eval_results_v5/` with its own `v5_comparison.md`
   (6-condition protocol header). `v4_comparison.md` is **frozen — never
   regenerate it**: its results dir already contains pusheval JSONs that
   any regeneration would silently inject into a table whose header says
   "no external pushes" (a latent hazard that predates v5 — flagged in
   AGENTS.md by this plan; an allowlist arg for
   `write_comparison_markdown` is an optional hardening task).
2. Push eval, both definitions (review): (a)
   `Isaac-Velocity-Rough-OpenDuck-PushEval-v0` + `--keep-pushes` —
   v4-comparable number (its trunk-contact termination counts trunk grazes
   as falls; v4's 6.84% was measured under this rule); (b)
   `-ContactPushEval-v0` — same pushes under v5's fall-only terminations.
   A gate-2 miss is attributed (definition vs genuine falls) before any
   candidate is rejected.
3. Wrench eval: `-WrenchEval-v0`, tiers 0.1/0.2/0.3 × measured body weight,
   2–6 s holds, during wz-tracking sweeps incl. wz=0.5; **plus the
   command-chatter condition** (§8 cond. 8 — review: root cause 6, the
   0.2 s re-command regime, was otherwise tested by no gate before the
   paid benchmark).
4. Video + frame-by-frame audit per §8.
5. Best candidate only → duck-embody regression (§7 gates 3–4).

---

## 7. Acceptance gates (Task 2.8's five, made concrete and review-corrected)

| # | Gate | Tool | Pass bar |
|---|---|---|---|
| 1 | Parent eval, 6-condition grid | `evaluate_policies.py` (3,840 eps) → `eval_results_v5/` | Legacy 5 conditions: gait gate 5/5, unpushed falls ≤ 0.5%, ref RMS within 1.0° of v4_robust (original pkl). **wz=0.5 condition: mandatory pass** on falls ≤ 0.5%, gait-duty valid, wz err comparable to v4@wz=0.3, video "stepping rotation" item; RMS there is dual-pkl descriptive only (§5) |
| 2 | Push recovery | PushEval (v4 rule) + ContactPushEval (v5 rule), `--keep-pushes` | ≤ 3% under the v4-comparable rule (v4: 6.84%); both numbers journaled; definitional misses attributed before rejection |
| 3 | Duck-embody smokes | `smoke_physics_pass.py`, `smoke_gap_hunt.py` (with `--checkpoint`) | physics-pass PASS; S0/S1/S3/S4/S5 green; **S2: "no fall within budget" is the v5 PASS** (it is a forced-fall harness diagnostic — its fall-plumbing halves are certified by running S2 once with the v4 checkpoint) |
| 4 | Fall regression suite | `replay_falls.py`, 10 scenarios × **5 reps** | **Precondition:** v4_robust falls in ≥8/10 scenarios under replay (re-encode until true). Pass: v5 survives ≥4/5 reps in ≥9/10 scenarios |
| 5 | Sustained-wrench + chatter | `-WrenchEval-v0` | survives 0.1/0.2 × BW presses during max-wz tracking; 0.3 × BW reported (stretch); command-chatter condition survived |
| ✔ | Final: benchmark re-run | duck-embody new freeze, 12 trials (~$10, ~1 h) | falls/policy-min < 0.3 (v4: 1.58); fall-dominated headline gone |

Regression baselines (from `eval_results_v4/v4_robust.json`): aggregate RMS
4.49°, jerk 0.0753, duty 68.6/63.8, asym 4.9 pp, ROM 0.99, energy 21.8 W;
pusheval per-condition falls 13.9/5.8/3.3/2.0/9.2%. Watch-items carried from
v4: jerk (already the least smooth policy) and forward-push falls (13.9%).

The benchmark re-run costs real money (~$10 LLM spend) **and requires
owner-approved commits in both repos for the freeze (§3.10)** — it launches
only on explicit owner approval, after gates 1–5 are green.

---

## 8. Frame-by-frame validation protocol (mandatory, every candidate)

Today's audits are solid but manual, and the extraction commands were never
preserved (zero ffmpeg usage in main-repo scripts; duck-embody's 24-tile
audit mosaics have no checked-in generator). v5 makes the "quantitative twin
of the video checklist" (AGENTS.md forward-plan) real:

**New `scripts/audit_rollout.py`** (runs under `isaaclab.sh -p`, built on
`scripts/eval_common.py` — §3.9): deterministic rollout of a given
checkpoint × condition, emitting side by side:
- the mp4 (1280×720@50, robot-tracking viewer — same camera as today), and
- a per-control-step trace CSV: trunk height, tilt angle, base ang-vel,
  per-foot contact force + binary contact, stance-foot slip distance, leg
  joint positions vs reference, commanded vs achieved (vx, vy, wz), action
  rate, **trunk contact force, applied wrench, and the disturbance-gate
  flag** (review: without the gate flag, spurious/exploited gating is
  invisible to the audit).

**New `scripts/make_filmstrip.py`** (plain python + pinned
`~/.local/bin/ffmpeg`, cv2 available): (a) uniform 24-tile contact sheet
with burned-in timestamps (the duck-embody auditors' proven format, now
scripted); (b) **auto-flagged dense bursts** — any window where tilt > 15°,
slip spikes, or a contact-force step lands gets a 0.4 s-spacing strip (the
`smoke_physics_pass.py` event-strip pattern, generalized).

**Conditions rendered per candidate** (seed 42, ≥20 s each):
1. forward vx=0.2 (legacy) · 2. turn wz=0.3 (legacy) · 3. **turn-in-place
wz=0.5 sustained 8 s** (the killer command) · 4. wz=0.5 immediately after a
push · 5. sustained lateral press 0.2×BW while walking · 6. press during
wz=0.5 (co-occurrence case) · 7. **obstacle graze at 0.10–0.15 m lateral
offset with a turn command** (`-ObstacleEval-v0`; review: no prior condition
exercised a real obstacle, so the "compliant slide" checklist item had
nothing to audit) · 8. **0.2 s command chatter** alternating wz ±0.3 around
a heading (the deployment `turn_to_heading` texture; root cause 6).

**Checklist — legacy five + four contact items** (each with its trace twin):
- trunk upright near 0.17 m (height mean/min)
- both feet alternate swing with real clearance (contact binaries)
- feet loaded in stance, no drag/glide/crawl (slip-in-stance)
- heading straight / command respected (wz error)
- no action dither (action rate)
- **press response is stable**: for sub-critical presses (≤ ~15% BW),
  minimal-motion bracing **passes** — that is the gated-reward optimum for a
  statically stable duck (review: demanding visible stepping would fail the
  exact behavior the reward trains); recovery **steps are required** only
  when displacement/tilt exceeds threshold (tilt > ~10° or CoM shift >
  ~3 cm), and toppling always fails
- **compliant slide along obstacle** — keeps walking, no freeze/bounce
  (condition 7)
- **turn-in-place is a stepping rotation**, not a pivot-scrape (both feet
  keep cycling at wz=0.5)
- **gait resumes ≤ ~2 cycles after disturbance end** (RMS re-converges;
  gate flag back to 0)

Audit output: filmstrips + bursts read frame-by-frame (agent), verdict
cross-checked against the trace CSV; both stored in the run's log dir and
cited in the journal entry. Video verdict outranks metrics, per protocol.

---

## 9. Ops, journaling, artifacts

- **Journal:** every run (incl. smoke) gets an entry in
  `experiment_journal.md` (engineering record) **and the run-index table is
  updated** (protocol requirement); full results in a new
  `docs/jetson-mod/v5_contact_results.md` following the
  `v4_retrain_results.md` template (config delta | TB last-100 means via
  EventAccumulator | gated eval + JSON paths | video audit | verdict).
  Wall-clock from event-file timestamps.
- **Run-exit verification before any eval** (AGENTS rule 6, corrected):
  pidfile process group dead AND the newest `model_*.pt` index ==
  loaded_iter + max_iterations − 1 (**`model_5998.pt`** for fine-tunes,
  `model_5999.pt` does not exist; `model_3098.pt` for the smoke;
  `model_5999.pt`-style off-by-one was a review catch).
- **Comparison tables:** v5-era results live in `eval_results_v5/` +
  `v5_comparison.md` (6-condition header). `v4_comparison.md` is frozen;
  never regenerate it (§6.1).
- **Docs that must change with the code (review):** AGENTS.md — Termination
  Conditions, DR summary, RL Training "v4 Tracks" → add the v5 track, the
  eval-protocol section (6-condition grid, new results dir, the
  `--report-only` hazard on `eval_results_v4/`); `task_plan.md` Task 2.8
  status + evidence when done.
- **Archive:** `exported_policies/v5_contact/` (checkpoint + agent.yaml +
  env.yaml + README) for the winner — and **backfill the missing
  `exported_policies/v4_robust/`** (the AGENTS.md-mandated archive was never
  created; v3's README is also missing).
- **ONNX:** export via play (automatic), verify the 59-dim input; TensorRT
  parity remains a Phase 4 item.
- Cleanup opportunity (optional): `launch_training_detached.sh` still
  dispatches `--algorithm` runs to the deleted `train_amp.py`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Catastrophic forgetting during fine-tune | ramped DR, gate-1 legacy-condition regression bar, v5c control, v5d PPF anchor |
| Gate-freeze exploited by the policy | event-only triggers (no self-measurable rate trigger); TB gate-duty per episode watched in smoke + training |
| Restored low exploration noise under-explores new regimes | v5c comparison; if v5a plateaus on wrench survival, bump `entropy_coef` or re-init std as follow-up lever |
| Push/wrench too strong → −200 noise, no learning | ramp capped at 0.7 m/s (capture-point argument, §2), survival-gated ramp advance |
| Obstacle never actually contacted → regime 1 untrained | path-tangent placement + wz-biased commands + TB contact-rate floor set from smoke |
| Crawl exploit re-opens (no contact-death) | duty gate + video audit; escalation: filtered ground-contact termination (§3.5) |
| Wrench event fights other composer users | `base_external_force_torque=None`; single-owner rule in the term |
| Synthesized reference cells wrong | §5 checker (parent-envelope, duty, symmetry) + policy-level frame audit; placo fallback |
| Eval bias against the intended fix | wz=0.5 excluded from RMS-parity; dual-pkl descriptive reporting (§5) |
| Comparison-table corruption | new `eval_results_v5/` dir; `v4_comparison.md` frozen (§6.1) |
| Replay suite vacuous | v4 must reproduce ≥8/10; 5 reps per scenario (§3.10) |
| Single-seed over-claiming | v5a/v5b verdicts recorded as single-seed observations; deciding pair re-run on a second seed before config-changing conclusions |
| GPU contention | one job at a time; evals in gaps; queue runner only for unattended sequential trainings |

## 11. Resolved decisions (owner-directed, 2026-07-28)

**1. Benchmark re-run — conditional, owner's rule.** The re-run happens ONLY
after every trained candidate has been evaluated and one is demonstrably
better than v4_robust; that candidate is the one benchmarked. **If no
candidate beats v4_robust, do not re-run the benchmark — document the
findings and stop.** "Better than v4_robust" is defined as: gates 1–5 all
pass (§7) AND no regression on the legacy-condition quality metrics beyond
the stated margins. A candidate that merely ties v4 does not earn the spend.
The freeze-prerequisite commits in both repos (§3.10) are requested at the
same moment as the spend approval, not before.

**2. wz command range — train wider than deployment: direct-wz envs at
±0.7, heading-servo envs clipped at ±0.5.** Evidence: upstream Open Duck
Playground trains *this robot* at `ang_vel_yaw=[-1.0, 1.0]`
(`playground/open_duck_mini_v2/joystick.py`) — double our current ±0.5; the
reference library spans wz ∈ [−1.111, 1.222], so ±0.7 is well inside it; and
5 of 10 benchmark falls occurred at exactly |wz| = 0.5, i.e. on the boundary
of trained experience. Training to ±0.7 makes the deployment hull limit an
interior point with 40% margin while keeping command density concentrated
near the regime we actually deploy (rather than spreading it to upstream's
±1.0 on a robot that is now 2.66 kg with a raised trunk CoM). Heading-servo
envs keep the deployment-exact ±0.5 clip, which `BandedWzVelocityCommand`
(§3.3) enforces separately from the sampling range. Backlog lever if wz=0.5
stays marginal: widen to ±1.0 (upstream-exact).

**3. Sequencing — decision-gated, with a measurement first.** Order: push
calibration sweep (§6.0, ~35 min GPU, v4_robust only) → Run 0 smoke → Run 1
v5a → evaluate → then decide Runs 2/3. Rationale: v5a changes many things at
once; a pathology found after 2 h is cheap, three blind runs producing
uninterpretable ablations is not. v5c is 6000 iters (~3.8 h), so it is the
one worth queueing overnight once its config is settled.

**4. Push magnitude — ramp 0.4 → 0.7 m/s per axis, interval 5–10 s.**
Researched and settled; Task 2.8's written "0.5 → 1.3 m/s" is amended.
Three independent lines converge on 0.68–0.75 m/s:
- **Exact-robot precedent (decisive).** Upstream Open Duck Playground trains
  this robot with `push_config(magnitude_range=[0.1, 1.0],
  interval_range=[5.0, 10.0])`, applied as a uniformly-directed 2D magnitude
  added to base velocity. Isaac Lab samples x and y independently over a box,
  for which E|Δv| = a·0.7652 and max = a·√2; matching upstream's mean 0.55 /
  max 1.0 gives a = 0.719 → **a = 0.70 reproduces the upstream push
  distribution to within 3%**. Same 20 s episode length, same additive
  semantics (`events.py:1069`), so the numbers are directly comparable.
- **Froude scaling** (v ∝ √(g·h), identical to the capture-point law since
  v/√(gh) ≡ d_step/h): duck-equivalents across seven published configs
  (G1, Berkeley Humanoid, Booster T1, ANYmal/legged_gym, Go1) have median
  **0.741 m/s**.
- **Capture point** at h = 0.17 m (ω₀ = 7.60 s⁻¹): 0.68–0.76 m/s for a
  9–10 cm reactive step. Measured leg geometry supports this — hip-pitch
  travel is 100° (−70°/+30°, `robot.xml`), giving ~0.10–0.12 m of reach,
  vs a 6 cm nominal stride.

Task 2.8's 1.3 m/s per axis would mean a 1.84 m/s resultant needing a
**24 cm** capture step on a robot whose CoM sits at 17 cm — unrecoverable by
construction, injecting −200 termination noise with no learnable gradient.
Interval moves to upstream's 5–10 s rather than the plan's earlier 4–8 s, to
remove a free variable while two other disturbance sources are being added.
The survival-gated ramp is retained regardless: Hartmann §III-D withholds
pushes until the tracking reward exceeds 85% of its maximum, warning that
pushing too early or too hard "can prompt an excessive caution in the agent,
possibly steering it into an unfavorable local optimum".

**Corrections to earlier drafts of this plan, from the same research:**
Hartmann et al. train on a **Unitree Go1 quadruped** (12 kg, 0.31 m base
height), not ANYmal; their **training** push was 1.0 m/s and ±1.5 m/s was
the *evaluation* grid (their Go1 training value Froude-scales to 0.741 m/s
for the duck — precisely the recommendation). Their sustained-wrench
magnitudes were 10 N and 20 N on a 117.7 N robot = **8.5% and 17% of body
weight**, which brackets nicely inside this plan's 5–30% BW range. Disney
BDX publishes **no** disturbance magnitudes at all — it is a structural
precedent (forces and torques on torso, head, hips and feet) and must not be
cited as a numeric anchor.

## 12. DECISION (2026-07-28): the phantom kilogram is not fixed inside v5

The full analysis, measurement and fix options for the 1.000 kg phantom mass on
the articulation root now live in **`docs/jetson-mod/known_issues.md`
(PLANT-1)**, together with the DR term that fails to cover it (PLANT-2). The
duplicate write-up that used to sit here was removed on 2026-08-09 — it had
already begun to drift from the register.

What belongs to *this* document is the decision it forced:

**Do NOT fix it inside v5.** Changing the robot's mass changes the plant, which
would (a) confound the contact-rich experiment with a dynamics change, (b) break
fine-tuning from the v4_robust checkpoint as a controlled comparison, and (c)
make every v4 eval number cross-model — which this project's own rule forbids
("Evaluate on the SAME robot model/USD the policy was trained on", AGENTS.md).
v5 continues on the current plant.

It does **not** invalidate the v5 experiment: v5 fine-tunes from v4_robust on
the same plant, so v5-vs-v4 remains a controlled comparison. It does mean no v5
number may be quoted as a hardware number.

**Follow-up, owner decision, should block Phase 4:** a separate model-correction
task — see PLANT-1 "Fix options" — then retrain the winning recipe on the
corrected plant and re-run the full gate protocol into a new per-model results
dir.

Also observed, lower severity: `enable_external_forces_every_iteration` is
False, so an applied wrench is integrated once per physics step rather than
per solver iteration. Acceptable for a wrench held 2-6 s; recorded so the
sustained-press numbers are not over-read.

## 13. Backlog / deferred levers

- Widen wz to ±1.0 (upstream-exact) if wz=0.5 remains marginal after v5a.
- Replace the random-interval push eval with a deterministic
  magnitude × direction grid reporting the 80%-success contour (Hartmann
  Fig. 2 style) — strictly more informative than a single fall percentage,
  and the natural portfolio artifact.
- Add a lower floor to the push magnitude (upstream never emits below
  0.1 m/s; the Isaac box emits arbitrarily small ones).
- Competence-gated DR onset (reward-threshold trigger rather than a fixed
  step count) — Hartmann's actual mechanism, and a cleaner de-confounder for
  v5c than the fixed 3000-iteration ramp delay currently specified.
- An allowlist argument for `write_comparison_markdown` so a results dir can
  hold push/no-push JSONs without corrupting its table (§6.1 hazard).

---

## 14. Run 0 — `v5_smoke` (executed 2026-07-28)

**Config:** task `Isaac-Velocity-Rough-OpenDuck-Contact-v0`, 4096 envs,
`--resume --load_run 0000-00-00_v4robust_seed --checkpoint model_2999.pt
--max_iterations 100 --video --video_length 200 --video_interval 1000`.
**Provenance:** log dir `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-28_03-10-51/`,
console `.training_runs/v5_smoke.log`, TB event file
`events.out.tfevents.1785226504.spark-2a60.1624246.0`.

**Resume mechanics — confirmed exactly as the review predicted.** The run
logged `Learning iteration 2999/3099` and its final checkpoint is
**`model_3098.pt`** (interval save at `model_3000.pt`). `model_3099.pt` does
not exist; a completion check keyed to `loaded_iter + max_iterations` would
have failed on a healthy run.

**Mechanism diagnostics (TB `Regime/*`, first -> last of 100 iterations):**

| Metric | Value | Criterion | Verdict |
|---|---|---|---|
| `wrench_rotating_cooccurrence` | 0.776 -> 0.642 | >= 0.30 | PASS — the modal fall regime is being trained on purpose, not by luck |
| `obstacle_envs_frac` | 0.246 | ~0.25 | PASS |
| `wrench_duty` | 0.035 -> 0.309 | reaches target band | PASS |
| `wrench_force_n` | 6.31 -> 5.49 N | inside the sampled band | PASS |
| `gate_duty` | 0.035 -> **0.398** | plan said < 0.30 | **EXCEEDS the stated bound** |
| `trunk_contact_duty` | 0.0039 | floor to be set here | **thin — 0.39% of steps** |

Terminations are fall-only as designed: `tilt` 0.78, `fell_low` 0.15,
`time_out` 0.22 — `base_contact` is gone. All twelve reward terms report,
including the four new ones (`ang_vel_xy_l2`, `feet_slide`,
`flat_orientation_deadzone`, `ground_contact`).

**Two items to settle before Run 1 (not silently accepted):**

1. **Gate duty 0.398 vs the plan's "< 0.30".** This is a consequence of
   `active_frac = 0.5` plus the 1.0 s hold, not of policy exploitation — the
   gate is event-driven only, so it cannot be self-induced. Hartmann's own
   episode structure is 2 s walk / 1 s recovery / 1 s post = 50% disturbed, so
   0.40 is within the precedent the design is copied from. **Resolution: raise
   the documented bound to < 0.45 and keep `active_frac = 0.5`, rather than
   quietly ignore the miss.** If Run 1 shows tracking quality stalling, drop
   `active_frac` to 0.35 as the first lever.
2. **Trunk-contact duty 0.39%.** Thin, and this is exactly the metric the
   review demanded a floor for. Cause is understood: the smoke starts from a
   policy trained for 3,000 iterations to treat trunk contact as death, so it
   still actively avoids the box. It is nonzero and the mechanism is proven
   live. **Resolution: carry it as the Run 1 watch-metric with a floor of
   1.0% by iteration 1,500; if it does not climb, the obstacle placement is
   not producing contact and the fix is to bias the initial command straight
   at the box rather than merely tangent to the path.**

**Adaptation signal (expected, not alarming):** reward 2.81 -> 125.84 and mean
episode length 12.6 -> 580 steps across 100 iterations. The starting shock is
real — the policy meets stronger pushes, sustained wrenches and obstacles it
has never seen — but both curves are climbing steeply. v4_robust's converged
reward under its own (much milder) DR was 249.5, so Run 1 has roughly half its
reward to recover plus the new regime to learn. This is the number to watch
for the "excessive caution" local optimum Hartmann warns about.

---

## 15. Run 1 — `v5a_gated_ft`: FAIL (gait gate 0/6, standing policy)

**Config:** `Isaac-Velocity-Rough-OpenDuck-Contact-v0`, 4096 envs, fine-tuned
from `v4_robust/model_2999.pt`, 3000 iterations (2999 -> 5998).
**Provenance:** `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-28_03-25-43/`,
final checkpoint `model_5998.pt` (31 checkpoints), console
`.training_runs/v5a_gated_ft.log`, wall-clock **2.22 h** (TB first-to-last
event timestamps). Zero errors.

### Training signals (TB last-100 means)

| | v5a_gated_ft | v4_robust |
|---|---|---|
| `Train/mean_reward` | 197.71 | 249.51 |
| `Train/mean_episode_length` | 977.46 | 991.76 |
| `Metrics/.../error_vel_xy` | 0.2629 | 0.3140 |
| `Metrics/.../error_vel_yaw` | **1.0898** | 0.3157 |
| `Policy/mean_std` | 0.0977 | 0.0565 |

Terminations at the end: `time_out` 0.966, `tilt` 0.034, `fell_low` 0.0005 —
i.e. the robot had all but stopped falling under a far harsher regime than v4
ever saw. `Regime/trunk_contact_duty` climbed 0.0039 -> **0.499**. On the
training curves this reads as a success.

### Gated evaluation — it is not walking

`scripts/evaluate_policies.py`, 6 conditions x 10 windows x 64 envs x 30 s =
3,840 episodes, seed 42, deterministic, into
`docs/jetson-mod/eval_results_v5/` (a NEW per-protocol dir; `v4_comparison.md`
was deliberately not regenerated).

| Metric | v5a_gated_ft | v4_robust (same task, same protocol) |
|---|---|---|
| **Gait gate** | **0 / 6** | **6 / 6** |
| Stance duty L/R | **99.7 / 99.6 %** | 68.7 / 63.7 % |
| Fall rate | 0.00 % | 0.00 % |
| Ref RMS | 6.18 deg | 4.48 deg |
| wz error | 0.575 rad/s | 0.067 rad/s |
| Energy | 11.10 W | 20.85 W |
| Jerk | 0.0207 | (v4 historical 0.0753) |

Both feet are loaded ~100 % of the time — the exact signature the journal
records for standing (study runs 13/14: gate 0/5 at ~99 % duty). Half the
energy and a third the jerk corroborate minimal motion, and at the wz = 0.5
condition the yaw error is 0.509 against a 0.5 command, i.e. **it rotates at
essentially zero**. It never falls because it never really walks. Fall rate
alone would have called this a triumph.

**The control matters:** v4_robust evaluated on the *identical* task and
protocol scores 6/6 at 68.7/63.7 % duty. The harness is sound; the regression
is real.

### Root cause — the disturbance gate became self-inducible

`Regime/gate_duty` reached **0.83**. The chain: trunk contact was one of the
gate triggers; with obstacles in the world the policy can *acquire* contact;
contact duty reached 0.50; so the gate stood up 83 % of the time, freezing the
tracking terms at a favourable constant and scaling imitation to x0.1. The
episode-average `imitation_reward` ended at **+0.008** — the only term that
demands a real gait was effectively deleted, leaving `alive_bonus` (+9.58) as
the dominant objective. Standing maximises exactly that.

The adversarial review had flagged the principle ("the gate must only be raised
by externally caused events") and the `ang_vel_xy` trigger was removed for it.
The trunk-contact trigger was kept because contact reads as external — **which
is true on an empty plane and false the moment the curriculum adds obstacles.**
That is the error.

### Verdict and consequences

- **v5a FAILS gate 1.** It does not beat v4_robust, so under the owner's rule
  **no benchmark re-run is triggered** and no spend occurs.
- The negative result is informative rather than wasted: it isolates the one
  novel mechanism in the recipe (gated imitation/tracking) as the failure, with
  everything else held fixed.
- Newly learned, and useful: **v4_robust passes wz = 0.5 open-field, 6/6**
  (duty 68.9/63.2, wz err 0.070, 0 % falls). The eval grid had never tested
  above 0.3. So root cause 4 in Task 2.8 needs refining — v4 is not
  rate-limited at 0.5 rad/s in free space; its benchmark falls at |wz| = 0.5
  required the *contact* context. The target regime is turning **while loaded**,
  not turning fast.

### Fix taken

`contact_raises_gate` is now a parameter defaulting to **False**
(`contact_events.py`), so only env-scheduled wrenches and pushes — which the
policy cannot cause — can raise the gate; trunk contact is still counted for
diagnostics. Run 2 goes further and removes the gate entirely rather than
retuning its constants: after a mechanism produces a degenerate optimum,
deleting it is the better next experiment.

## 16. Run 2 — `v5b_ungated_ft` (launched 2026-07-28)

Task `Isaac-Velocity-Rough-OpenDuck-ContactUngated-v0`
(`DuckContactUngatedRewards`): stock `track_lin_vel_xy_exp` /
`track_ang_vel_z_world_exp`, `disturbance_gate_scale=None`. Everything else in
the contact curriculum is unchanged — obstacles, sustained wrenches,
rotational pushes, banded wz to +/-0.7, fall-only terminations, patched
reference library — so this isolates whether the curriculum works *without*
the novel reward mechanism. Same fine-tune from `model_2999.pt`, 3000
iterations, final checkpoint will be `model_5998.pt`.

Watch metrics: `Regime/gate_duty` should now be irrelevant (nothing reads it);
`Episode_Reward/imitation_reward` must stay materially non-zero — its collapse
to +0.008 was the leading indicator of the v5a failure and is the cheapest
early warning available.

---

## 17. Variation structure for the remaining v5 runs (revised after v5a)

### What was wrong with how v5a/v5b were built

Both are **bundles**: v5a changed eight things at once against v4_robust
(fall-only terminations, obstacles, sustained wrenches, rotational + stronger
pushes, banded wz, disturbance gating, patched library, four new penalties).
The journal protocol asks for "one-lever config delta" per run, and this is
exactly why: when v5a failed, only the fact that the failure signature was
*specific* (99.7% stance duty + imitation collapsed to +0.008) made the cause
identifiable. A less legible failure would have left eight suspects.
Subsequent arms are therefore built as a ladder, not as bundles.

### Two axes, because the evidence names two different risks

**Axis I — contact exposure (the thing being added).** Ordered by how directly
each rung attacks the measured falls:

| Rung | Content | Targets |
|---|---|---|
| L1 | fall-only terminations only (tilt/height replace contact-death) | "training taught contact = death" — regime 2, the root of all 10 falls |
| L2 | L1 + obstacles | rotation-under-contact, 7/10 falls |
| L3 | L2 + sustained wrench | sustained press, 2/10 falls |
| L4 | L3 + rotational/stronger pushes | general disturbance robustness |

**Axis II — gait protection (the thing that must not break).** v5a proved
this axis is not optional: fall-only termination is simultaneously the
essential fix AND the removal of the rail that used to make leaning/standing
fatal. Something must hold the gait up in its place.

| Guard | Content | Cost |
|---|---|---|
| G0 | imitation at full weight, never gated | free (v5b) |
| G1 | G0 + explicit anti-standing term (`feet_air_time` positive, or a duty penalty) | one lever |
| G2 | G0 + PPF-style L2 anchor of the policy mean to the frozen v4 actor | preserves gait by construction; structurally different approach |

Axis III (init: fine-tune vs scratch) is **deprioritised**. v5a's failure was
reward design, not initialisation, so a scratch arm would spend 3.8 h
answering a question nothing currently points at.

Two levers are also deprioritised by measurement: **banded wz and the patched
reference library**. v4_robust scores 6/6 including wz = 0.5 open-field, and
the synthesized turn-in-place cell differs from the original nearest cell by
only 1.19 deg. Neither can be responsible for the benchmark falls, so neither
should consume a run of its own.

### The decision tree, driven by failure signature rather than guesswork

`v5b_ungated_ft` (running) is **L4 + G0**. What happens next is determined by
*how* it fails, not merely whether:

- **Passes gate 1** (>= 5/6, duty in band, no gait regression) -> promote to
  the full gate battery: push eval both definitions, wrench eval, obstacle
  eval, fall replays, frame-by-frame audit. Only then is it a benchmark
  candidate.
- **Fails by standing** (duty -> ~100%, imitation collapsing) -> the guard
  axis is the binding constraint, not the curriculum. Next run is
  **L2 + G1**: minimal contact exposure plus an explicit anti-standing term.
- **Fails by falling / poor gait** (duty in band but falls high, or RMS badly
  degraded) -> the curriculum is too aggressive for one fine-tune. Descend the
  contact axis: **L2 + G0**, and add L3/L4 back one rung at a time.
- **Fails both ways, or gait degrades on every rung** -> stop tuning this
  formulation and switch to **L2 + G2** (PPF anchor), which keeps the gait by
  construction instead of by incentive.

### Early abort, so a wrong arm costs ~20 min instead of 2.2 h

v5a's whole cost came from the fact that stance duty was only computed by the
eval, i.e. after the run. `Gait/stance_duty_left_pct`,
`Gait/stance_duty_right_pct` and `Gait/duty_in_band_frac` are now logged every
step by `ContactRegimeEvent` (diagnostic only — they feed no reward).

**Abort rule for every subsequent arm:** at iteration ~500, kill the run if
`Gait/duty_in_band_frac` < 0.5 or `Episode_Reward/imitation_reward` has fallen
below ~10% of its early-run value. Either condition means the policy is
leaving the space of real gaits, and no amount of further training in this
project's history has ever brought one back.

### What "beats v4_robust" has to mean

v4_robust on the 6-condition grid is 6/6, 0.00% falls, RMS 4.48, duty
68.7/63.7, energy 20.85 W — and it already handles wz = 0.5 in the open field.
A v5 candidate therefore cannot win on open-field locomotion; it can only
match it. **The win has to come from the contact gates** (wrench eval,
obstacle graze, the 10 frozen fall replays) while not regressing the open-field
numbers beyond the stated margins. An arm that improves contact survival but
drops the gait gate below 5/6 is not a candidate, and under the owner's rule
triggers no benchmark spend.

---

## 18. Run 2 result — `v5b_ungated_ft`: FAIL (gait gate 3/6, asymmetric shuffle)

**Provenance:** `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-28_22-54-19/`,
final checkpoint `model_5998.pt`, eval JSON
`docs/jetson-mod/eval_results_v5/v5b_ungated_ft.json`. Driven end-to-end by
`scripts/v5_pipeline.sh`, which measured the gait gate first and **skipped the
contact battery** on the 3/6 result — saving roughly an hour of GPU that would
have characterised the contact robustness of a policy that does not walk well
enough to qualify.

| Metric | v5a | **v5b** | v4_robust |
|---|---|---|---|
| Gait gate | 0/6 | **3/6** | 6/6 |
| Stance duty L/R | 99.7/99.6 | **76.7/85.4** | 68.7/63.7 |
| Double support | ~99 % | **~62 %** | ~32 % |
| Duty asymmetry | 0.1 pp | **9.8 pp** | 5.0 pp |
| Ref RMS | 6.18 | **4.79** | 4.48 |
| Energy | 11.1 W | **20.7 W** | 20.9 W |
| Jerk | 0.021 | **0.074** | 0.072 |
| Falls | 0 % | **0 %** | 0 % |

Removing the disturbance gate worked as intended and fixed the specific defect
it was meant to fix: the imitation reward stayed alive all run (0.67 vs v5a's
terminal 0.008), energy and jerk returned to v4's values, and reference
tracking recovered to within 0.3 deg of v4. **This policy walks.** It simply
does not walk well enough: it carries ~62 % double support against v4's ~32 %,
and a 9.8 pp left/right asymmetry — a cautious, heavy-footed shuffle whose
right foot drags. The three failing conditions are precisely the ones with
lateral or rotational motion (vy = 0.1, wz = 0.3, wz = 0.5); pure fore/aft
walking passes.

**Reference library exonerated.** Its synthesized cells carry the same
70.4/66.7 % reference contact duty and comparable per-side range of motion as
the original cells, so the learned asymmetry is not prescribed by the patch.

**Read across the three runs, the trend is monotone in one variable:**

    double support   99 % (v5a) -> 62 % (v5b) -> 32 % (v4_robust)
    gait gate         0/6        ->  3/6      ->  6/6

Every pressure removed from the curriculum moved the policy back toward real
walking. That is the signature of a curriculum that is still too aggressive for
a single fine-tune — the "excessive caution" local optimum Hartmann et al. warn
about when disturbances arrive too early or too hard — not of a broken reward
term.

## 19. Run 3 — `v5c_contact_only` (launched 2026-07-29)

Task `Isaac-Velocity-Rough-OpenDuck-ContactMinimal-v0`. **v4_robust plus
exactly two changes**, per the decision tree's "descend the contact axis"
branch:

1. trunk contact is no longer instant death — terminate on falling (tilt 60
   deg / height 0.09 m), the deployment's own definition;
2. an obstacle exists, in 25 % of episodes.

Everything else reverts to v4_robust: its reward set (hence the **original**
frozen reference library), its heading-only +/-0.5 commands, its +/-0.3 m/s
pushes on the 8-14 s schedule, no sustained wrench, no rotational pushes, no
extra penalties, no curriculum ramp. Verified by config dump: rewards are
exactly v4's nine terms, `cmd_wz=(-0.5, 0.5)`, `curriculum=[]`.

If this recovers v4's duty while surviving contact, every other lever in v5 was
unnecessary complexity — which is the cheapest possible outcome to act on.

### Two process fixes this run forced

**A latent crash, caught before it cost anything.** The first launch died in
`ContactRegimeEvent.__init__` with `KeyError: 'force_frac_range'` — the minimal
cfg passes no such param. The constructor's own comment warns that it runs
inside a timeline PLAY callback whose exceptions are **swallowed**, and I then
wrote code that violated exactly that. Every param is now read with a default.

**Runtime smoke is now mandatory before any new task consumes GPU hours.**
Config parsing had passed; only stepping the env exposed the crash. The smoke
(~3 min) now also asserts that the gait canary emits, and it validated the
canary end-to-end: with null actions it correctly reported duty 99.5/99.6 and
`Gait/duty_in_band_frac = 0.0`, i.e. "this is not walking". At iteration ~3000
of the real run it reads 68.0/78.2 with `duty_in_band_frac = 0.59` — already
distinguishing v4-like walking from the v5b shuffle, live, instead of 2.2 h
later.

---

## 20. Run 3 result — `v5c_contact_only`: **PASSES the gait gate 6/6**

**Provenance:** `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_01-46-04/`,
`model_5998.pt`. Driven end-to-end by `scripts/v5_pipeline.sh` (gate first,
then the contact battery, then four audit videos).

### Open-field grid — v5c reproduces v4's gait

| Metric | v4_robust | **v5c** | v5b | v5a |
|---|---|---|---|---|
| Gait gate | 6/6 | **6/6** | 3/6 | 0/6 |
| Falls | 0.00 % | **0.00 %** | 0.00 % | 0.00 % |
| Ref RMS | 4.478 | **4.486** | 4.79 | 6.18 |
| Duty L/R | 68.7/63.7 | **65.9/66.2** | 76.7/85.4 | 99.7/99.6 |
| Duty asymmetry | 5.0 pp | **1.1 pp** | 9.8 pp | 0.1 pp |
| Energy | 20.85 W | 21.29 W | 20.68 W | 11.10 W |
| Jerk | 0.0720 | 0.0773 | 0.0738 | 0.0207 |
| wz error | 0.067 | 0.081 | 0.156 | 0.575 |

Reference tracking is within 0.008 deg of v4 and the gait is **more symmetric
than v4's** (1.1 pp vs 5.0 pp). Energy, jerk and wz error are marginally worse
but all well inside the stated margins. The hypothesis that the whole v5a/v5b
bundle was unnecessary complexity is, so far, supported: two changes reproduce
v4's locomotion.

### Contact battery — one clear win, one total failure

| Gate | v5c | reference |
|---|---|---|
| Push recovery (v4 fall definition) | **0.29 %** | v4 historical 6.84 % |
| Push recovery (v5 fall definition: tilt/height) | 3.70 % | — |
| Obstacle graze | 6.22 % | — |
| **Sustained wrench** | **100.00 %** | — |

Push recovery improves by more than an order of magnitude, and the gait gate
stays 6/6 under every one of those conditions. But **v5c falls in every single
episode of the sustained-wrench gate** — which is exactly what its own design
predicts: it trains with `active_frac = 0` and has never felt a force held for
2-6 s. Sustained pressing is 2 of the 10 benchmark falls and the mechanism
behind duck-embody's S2 counter-press, so this gap is disqualifying on its own.

Note on the two push numbers: the v5 definition (3.70 %) reads *higher* than
the v4 definition (0.29 %) because tilt > 60 deg is a more sensitive fall
detector than trunk-contact > 1 N — a robot can pitch past 60 degrees without
its trunk touching anything. The stricter number is the honest one.

### The missing control

Every v5c contact number above lacks a v4 counterpart measured under the same
protocol; v4's only record is 6.84 % pushed falls from a 5-condition run on a
different task. Since "better than v4_robust" is the acceptance rule, the
battery cannot be scored without it. `scripts/v5_chain.sh` now runs v4_robust
through all four contact gates before anything else.

## 21. Run 4 — `v5d_contact_wrench` (launched 2026-07-29)

`v5c` **plus one lever**: the sustained wrench (`active_frac = 0.5`,
5-20 % of measured body weight, 2-6 s, >= 30 % co-occurring with |wz| >= 0.25).
Force is deliberately gentler than v5a/v5b's 5-30 % — Hartmann applies 8.5 %
and 17 % of body weight to a Go1, and the v5a/v5b lesson is that this
curriculum tips into "excessive caution" when disturbances are too strong.
Everything v5c established stays fixed.

Runtime-smoked before launch (the rule v5c's `KeyError` established): wrench
fires on 51.6 % of envs, co-occurrence 0.833, gait canary emitting.

Two automation changes shipped with it, both paid for by earlier mistakes:

- **`scripts/v5_chain.sh`** serialises v4 controls -> train -> evaluate into one
  unattended background process, because the GPU takes one Isaac job at a time
  and nothing progresses while waiting to be asked.
- **An early-abort watchdog.** Thirty minutes in, it reads
  `Gait/duty_in_band_frac` from the training log and kills the run if it is
  below 0.30. v5a spent 2.22 h converging to a standing policy that every
  training signal called healthy; this metric would have exposed it in minutes.

---

## 22. Run 4 result — `v5d_contact_wrench`: **BEATS v4_robust on every contact gate**

**Provenance:** `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25/`,
`model_5998.pt`. Ran unattended via `scripts/v5_chain.sh`: v4 control battery
-> train -> gate -> contact battery -> four audit videos. The early-abort
watchdog read `Gait/duty_in_band_frac = 0.985` at 30 min and correctly let the
run continue.

### Open-field grid (6 conditions x 3,840 episodes)

| Metric | v4_robust | v5c_minimal | **v5d_wrench** |
|---|---|---|---|
| Gait gate | 6/6 | 6/6 | **6/6** |
| Falls | 0.00 % | 0.00 % | **0.00 %** |
| Ref RMS | 4.478 | 4.486 | **4.669** |
| Duty L/R | 68.7/63.7 | 65.9/66.2 | **71.5/73.9** |
| Duty asymmetry | 5.0 pp | 1.1 pp | **3.6 pp** |
| wz error | 0.067 | 0.081 | **0.098** |
| Energy | 20.85 W | 21.29 W | **20.33 W** |
| Jerk | 0.0720 | 0.0773 | **0.0836** |

### Contact battery — every gate measured against the SAME v4 control

| Gate | v4_robust | v5c | **v5d** |
|---|---|---|---|
| Push recovery (v4 fall definition) | 6.35 % | 0.29 % | **1.07 %** |
| Push recovery (tilt/height definition) | 11.12 % | 3.70 % | **0.00 %** |
| **Sustained wrench** | **100.00 %** | 100.00 % | **47.11 %** |
| **Obstacle graze** | **32.60 %** | 6.22 % | **0.31 %** |

The gait gate holds 6/6 under all four contact conditions for all three
policies — the discriminator is the fall rate, exactly as Task 2.7 found.

**This is the first time the project has measured v4_robust's contact
fragility directly, and it is severe: 100 % falls under a sustained press and
32.6 % on an obstacle graze**, against 0.00 % in the open field. That is the
benchmark's 10-falls-in-12-trials, reproduced in a controlled protocol.

v5d cuts obstacle-graze falls **105x** (32.60 -> 0.31 %), eliminates
tilt-definition push falls entirely (11.12 -> 0.00 %), and halves sustained
-wrench falls (100 -> 47.11 %) — while matching v4's open-field locomotion.

### Video audit (mandatory; video outranks metrics)

Filmstrips at 1.5 fps from the four rendered conditions:

- **forward vx=0.2** — trunk upright, legs alternate with real ground
  clearance, no drag or glide. PASS.
- **turn wz=0.5 sustained** — heading advances steadily and the rotation is
  produced by *stepping*, not a planted pivot-scrape. This is the command 5 of
  10 benchmark falls died under. PASS.
- **sustained press** — the trunk visibly leans into the applied load while the
  feet keep cycling, then recovers. That lean is the intended compliance, and
  it is what the dead-zoned orientation penalty was meant to permit. PASS.

### Verdict

**v5d_contact_wrench is better than v4_robust** under the owner's acceptance
rule: it matches open-field locomotion (gate 6/6, 0.00 % falls, ref RMS within
0.19 deg) and is strictly better on all four contact gates.

Honest caveats, none disqualifying but all worth carrying to hardware:

1. **Jerk +16 % vs v4** (0.0836 vs 0.0720). v4_robust was already documented as
   "the least smooth of the three, worth watching on hardware"; v5d is less
   smooth still. This is the clearest regression.
2. **wz error 0.098 vs 0.067** and ref RMS 4.669 vs 4.478 — both inside the
   stated margins, but v5c was closer to v4 on both, so the wrench lever costs
   a little tracking precision.
3. **Sustained wrench is still 47 %.** Halving v4's 100 % is real progress, but
   half of all pressed episodes still end in a fall. This is the remaining
   weakness and the obvious target for a v5e (longer/stronger wrench exposure,
   or more iterations at the current setting).
4. Gates 3 and 4 of the plan — the duck-embody smoke suites and the 10 frozen
   fall reproductions — have **not** been run. They are the actual transfer
   test, they cost no money, and they require `--checkpoint` passthrough in the
   sibling repo's smoke scripts plus the `replay_falls.py` that does not yet
   exist.

**No benchmark re-run has been triggered.** Under the owner's rule that needs a
policy demonstrably better than v4 (satisfied) AND explicit approval for the
~$10 spend and the freeze-prerequisite commits in both repos (not requested
yet, and correctly gated behind the transfer tests above).

---

## 23. Benchmark candidate: v5d only (owner-confirmed 2026-07-29)

**Baseline:** `duck-embody/policy/model_2999.pt`, sha256 `b1ebf3a5d7d866ef` —
verified byte-identical to
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`,
i.e. v4_robust, the policy that produced the frozen 10-falls-in-12-trials
record. Reproducing falls against anything else would not be a regression suite.

**Candidate:** `v5d_contact_wrench`,
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25/model_5998.pt`,
sha256 `301e24e336b2eab0`. Run-dir identity confirmed from its own startup
banner (`active_frac=0.5, obstacle_frac=0.25` — the wrench arm).

**Why v5d and not v5c, stated honestly.** "Best" is not unidimensional; v5c is
the better *gait* reproduction:

| | v5c | v5d |
|---|---|---|
| open-field ref RMS | **4.486** | 4.669 |
| open-field duty asymmetry | **1.13 pp** | 3.61 pp |
| push, v4 fall definition | **0.29 %** | 1.07 % |
| push, tilt/height definition | 3.70 % | **0.00 %** |
| sustained press | 100.00 % | **47.11 %** |
| obstacle graze | 6.22 % | **0.31 %** |

The benchmark measures falls in a furnished apartment, and the two gates that
map onto it are the ones v5d dominates: obstacle graze 20x better (the
mechanism behind 7 of the 10 original falls) and sustained press halved rather
than total (the other 2). v5c falling in 100 % of press episodes would walk
straight back into the wedge failure the benchmark already found. The price is
~0.18 deg of reference tracking; both pass the gait gate 6/6 at 0.00 %
open-field falls, so neither is a gait regression.

**Decision: one candidate, one batch (~$10).** Running both would have cost
~$20 to isolate the wrench lever in the real apartment; deferred rather than
dropped — see the v6 backlog below.

## 24. v6 backlog (deferred, not dropped)

Recorded now so the v5 evidence is not lost when this thread ends:

1. **Sustained press is the one unsolved failure mode**: 47.11 % falls, halved
   from v4's 100 % but far from safe. Highest-value v6 lever: longer/stronger
   wrench exposure, or simply more iterations at v5d's current setting, since
   v5d saw the wrench for only 3,000 fine-tune iterations.
2. **Smoothness regression**: v5d jerk 0.0836 vs v4's 0.0720 (+16 %) on a
   policy already documented as the least smooth on record. Worth an explicit
   `action_rate_l2` / `dof_acc_l2` lever before hardware.
3. **v5c-vs-v5d in the apartment** — the paired benchmark this decision
   deferred. Cheap to answer later if v5d's benchmark result is ambiguous.
4. **The 1 kg phantom root-link mass** (section 12) — affects every policy
   trained so far, identically, so it does not confound v5-vs-v4. But it is a
   sim-to-real fidelity error and should be fixed with a full retrain on the
   corrected plant before Phase 4 hardware work.
5. Competence-gated DR onset (Hartmann's actual mechanism) instead of the
   fixed step-count ramp, as a cleaner de-confounder than v5c's approach.
6. Full `build_scores.py` parity for out-of-tree batches (it hardcodes
   `RAW = results/raw`), so a candidate batch can be scored by the repo's own
   scorer rather than the headline-metric computation in `auto_pipeline.sh`.

---

## 25. Benchmark attempt 1 (2026-07-30): stopped early, two findings

Launched with v5d after the re-freeze (`config_hash 6a65f335`), all gates green:
replay 9/9, physics PASS, calibration K=0.9617 matching the measurement,
provenance sha recorded. Stopped by hand after **one partial trial, $4.87**.

### Finding 1 — the locomotion retrain works under LLM control

`fable5_seed101`: **34 turns, 35 bump events, ZERO falls.** In the frozen v4
batch the same seed and spawn fell on turn 2, 3.74 policy-seconds into its first
`move`, torso on the sofa. Frame-by-frame audit of 543 recorded frames: trunk
upright, legs alternating with real ground clearance, no drag, tracking straight
along the sofa face. The replay-suite result (v5d survives 9/9 gates v4 fell in)
reproduces in the live LLM-driven setting.

This is the Task 2.8 objective met. Everything below is a different problem.

### Finding 2 — ~95% of the "dead-reckoning drift" was an accounting bug

The trial's believed position ended **26.65 m** from truth in a 4.8 x 3.6 m
apartment. Attribution, re-derived from the trial JSON:

| source | believed | true | inflation |
|---|---|---|---|
| `send_velocity` (49 calls) | 27.09 m | 1.99 m | **25.10 m** |
| `move` (19 calls) | 3.66 m | 2.42 m | 1.24 m |
| clean moves only | 1.10 m | 0.97 m | 0.13 m |

A duck wedged against furniture with its legs cycling was credited
`commanded_speed x time` — 0.60 m for 0.01 m of real motion, 49 times. Genuine
policy-tracking error was 0.13 m, ~0.5% of the total. Fixed in duck-embody
`e0ac862` (contact-time discounting in both the reported distance and the
integrator). Consequence for this project: **v5d's calibration constants are
sound** — the 4.3% k shortfall is real and the recalibration was correct — but
the drift figure quoted from this trial was not a policy property.

### Why the batch was stopped, and the cost lesson

The benchmark had stopped measuring locomotion and started measuring a harness
bug. Also: a non-falling trial costs ~$4.9 because it runs to the caps, against
the frozen batch's ~$0.80 average for fall-shortened trials. **12 such trials is
$50-60, not the ~$10 this plan estimated** — that estimate silently assumed v4's
early deaths. Any future full batch needs that budget stated up front.

### Why `correct_position` is never called (0 uses, 3 models x 13 trials)

Investigated because loop closure is not optional in SLAM. Two independent
blockers, both verified:

1. **The tool demands a coordinate the system cannot produce.** `correct_position`
   requires `x` and `y` as numbers; the map's `Room` dataclass is
   `['name', 'description', 'landmarks']` — no coordinate field anywhere, and no
   observation payload supplies one. Recognition yields a NAME. It is the only
   tool of twelve whose required arguments are absolute world quantities the
   harness never provides. Corroboration: 14 free-text writes across the trials
   smuggle coordinates into `description` strings, i.e. the models wanted metric
   anchors badly enough to hide them in prose.
2. **The trigger almost never fired.** The prompt gates it on "when you recognize
   a place you already mapped"; exactly **1 of 13 trials** ever revisited a
   mapped room, and 8 of 13 mapped only one. The single trial that did close a
   loop named the coordinate in its reasoning and called `set_current_room`
   instead — that tool accepts a name.

SLAM mapping: place recognition present, **data association broken**, correction
unreachable. The map holds semantics without geometry.

Honest claim: zero uptake is explained by an affordance gap, not model
incapacity. Nothing here bounds whether these models *can* close loops.

### v6 backlog additions

7. **Doorway anchors, not room anchors.** A per-place beacon is the right
   granularity (per-object is impossible: monocular camera, no depth, so object
   coordinates would be invented). But a room-centroid anchor leaves ~1.8 m of
   intra-room error against a 0.35 m success radius, whereas a **doorway** is a
   0.35 m gap — anchoring on `Exit` (already marked by the models, 24 calls)
   gives ~0.175 m. Requires an `anchor_xy` written from the integrator when the
   place is first mapped, and rendered in the memory block, so
   `correct_position`'s argument becomes copy-from-the-block. Touches frozen
   files.
8. Replace the recognition-gated prompt trigger with an observable one, and
   un-gate the `Re-anchored: N times` line (it currently renders only after a
   correction has happened, so the null action is self-reinforcing).
9. Re-run the benchmark only after 7-8, with the $50-60 budget acknowledged.
