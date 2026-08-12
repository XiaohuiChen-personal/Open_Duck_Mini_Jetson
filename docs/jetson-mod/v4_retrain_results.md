# v4 Retrain Results

Engineering record of the post-CAD retraining campaign (originally
branch-local on `cad-redesign`; merged to `v2` 2026-07-26). **Deliberately
NOT part of `experiment_journal.md`'s study numbering** — study runs 1-16
are the EN.665.645 course record, frozen in the archive repo
`open-duck-ppo-vs-amp` (tag `course-study-freeze`); this document serves
the robot project. Data-sourcing follows
the same discipline: training numbers are last-100-iteration TensorBoard
means, behavioral numbers come from the gated `evaluate_policies.py`
protocol (JSONs in `eval_results_v4/`), and every run gets a mandatory
video audit.

**What changed under these runs vs every earlier policy:** layout v2.1
(`component_layout_v2.md`) + the inertia frame correction (upstream
principal-frame diaginertia had been read as body-frame — trunk inertia
was off +78%/−10%/−27% for all of study runs 1-16 — every pre-correction
policy: v1-v3 PPO and all AMP runs) + the Part-2 CAD mesh cuts
(robot 2.657067 kg, −88.5 g). USD regenerated and hash-guarded.

> **Caveat added 2026-08-09.** 2.657067 kg is what the MJCF/USD *author*; PhysX
> simulates **3.657067 kg** because the articulation root `base` carries no
> `<inertial>` and gets a 1.000 kg default. Both v4 runs below trained on the
> heavier plant. See `known_issues.md` PLANT-1.

> **Re-gated 2026-08-12 on the corrected 2.657 kg plant — see [`m2657_regate.md`](m2657_regate.md).** v5d passes every gate there; `v4_robust` does not (0.000 % -> 24.167 % open-field falls). The historical numbers on this page are unchanged and remain the 3.657 kg record.

> **Update 2026-08-11: PLANT-1 is fixed.** `base` was merged into
> `trunk_assembly` and the USD regenerated; `audit_plant_mass.py` now exits 0
> with PhysX simulating 2.657067 kg. The caveat above still applies to the
> numbers on this page, which were produced on the old 3.657 kg plant and are
> kept as the historical record. **They must be re-measured on the corrected
> plant before any of them is quoted again.**

## Run A — `v4_inertials` (model-correction isolation)

**Config**: task `Isaac-Velocity-Rough-OpenDuck-v0`, UNCHANGED v3 recipe
(one lever: the corrected model/USD). 4096 envs, 3000 iterations,
`--video` per protocol.

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo/2026-07-06_20-58-16/`, final
checkpoint `model_2999.pt`; console log
`.training_runs/v4_inertials.log`; launched 2026-07-06 20:58, wall-clock
1.85 h (first-to-last TB event timestamps, 21:02→22:53).

**Training signals** (TB `EventAccumulator`, `size_guidance={'scalars': 0}`):
- `Train/mean_reward`: last-100 mean **253.30** (peak 254.88, n=3000)
- `Train/mean_episode_length`: last-100 mean **998.27** / 1000
- Reference: v3's last-100 mean reward was 253.0 — statistically identical.

**Gated evaluation** (full 5-condition protocol, 3200 episodes,
`eval_results_v4/v4_inertials.json`; report `v4_comparison.md`). The
`ppo_v3` column is that policy's ORIGINAL 2026-06-12 evaluation on the
pre-correction model (`open_duck_ppo/2026-06-12_00-44-12`), copied in and
re-scored by the gate from its stored duty — NOT re-run on the corrected
model. That is intentional: each policy is measured on the model it was
trained for, which is exactly the model-correction isolation. Numbers are
code-comparable (RMS/duty/energy fns unchanged since that eval):

| Metric | v4_inertials (Run A) | ppo_v3 (baseline) |
|---|---|---|
| Gait-validity gate | **5/5 conditions** | 5/5 (re-gated) |
| Fall rate | 0.00% (0/3200) | 0.00% (0/3200) |
| Ref tracking RMS | 4.60° | 4.59° |
| Stance duty L/R | 70.0 / 64.9% | 68.9 / 64.6% |
| Duty asymmetry | 5.09 pp | 4.32 pp |
| ROM ratio L/R | 0.99 | 1.02 |
| v_xy err | 0.155 m/s | 0.155 m/s |
| wz err | 0.073 rad/s | 0.078 rad/s |
| Energy proxy | 21.4 W | 21.6 W |
| Mean squared jerk | 0.0704 | 0.0687 |

**Video audit (mandatory, per the gated protocol)**: deterministic rollout
at fixed vx=0.2 (`eval_results_v4/v4_inertials_play.mp4`, from
`play_policy.py --video`, 4 envs, 10 s) and the final training filmstrip
(`videos/train/rl-video-step-70000.mp4`). Both show upright bipedal
walking: high trunk, clear single-support phases with the swing foot off
the ground, steady forward progression across the grid, stable head. No
crawl (run-12 failure mode), no shuffle, no foot dragging — coherent with
the measured 65-70% stance duty.

**Verdict: GATE PASS.** The combined model correction (layout v2.1 +
frame-correct inertia + −88.5 g CAD cuts + CoM moved ~7 mm rearward/down)
retrains to a gait statistically equivalent to v3 on every protocol
metric. The corrected model is confirmed trainable and the v3 recipe
transfers unchanged.

## Run B — `v4_robust` (deployment candidate)

**Config**: task `Isaac-Velocity-Rough-OpenDuck-Robust-v0` — Run A recipe +
dynamics domain randomization (interval pushes ±0.3 m/s, trunk mass
+(−0.10,+0.15) kg, trunk CoM ±10/±5 mm, friction 0.4-1.0/0.3-0.8 with
`make_consistent`, joint-reset scale 0.9-1.1) + asymmetric observations
(actor 59-dim, drops the hardware-unmeasurable `base_lin_vel`; privileged
critic 62-dim uncorrupted) + `velocity_limit_sim` 8.94 rad/s. 4096 envs,
3000 iterations.

**Provenance**: log dir
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/`, final
checkpoint `model_2999.pt`; console log `.training_runs/v4_robust.log`;
launched 2026-07-07 00:15, wall-clock 1.91 h (TB timestamps 00:19→02:14).

**Training signals** (TB last-100 means):
- `Train/mean_reward`: **249.51** (peak 252.61, n=3000) — measured UNDER
  active DR, so ~4 below Run A's unperturbed 253.30 as expected.
- `Train/mean_episode_length`: **991.76** / 1000.

**Gated evaluation** (task `...-Robust-Play-v0`, 59-dim actor, DR off,
3200 episodes, `eval_results_v4/v4_robust.json`):

| Metric | v4_robust (Run B) | v4_inertials (Run A) | ppo_v3 |
|---|---|---|---|
| Gait-validity gate | **5/5** | 5/5 | 5/5 (re-gated) |
| Fall rate | 0.00% | 0.00% | 0.00% |
| Ref tracking RMS | 4.49° | 4.60° | 4.59° |
| Stance duty L/R | 68.6 / 63.8% | 70.0 / 64.9% | 68.9 / 64.6% |
| Duty asymmetry | 4.85 pp | 5.09 pp | 4.32 pp |
| ROM ratio L/R | 0.99 | 0.99 | 1.02 |
| v_xy err | 0.153 m/s | 0.155 m/s | 0.155 m/s |
| wz err | 0.067 rad/s | 0.073 rad/s | 0.078 rad/s |
| Energy proxy | 21.8 W | 21.4 W | 21.6 W |
| Mean squared jerk | 0.0753 | 0.0704 | 0.0687 |

**Video audit**: deterministic rollout at fixed vx=0.2
(`eval_results_v4/v4_robust_play.mp4`). Upright bipedal walking — high
trunk, clear single-support swing phases, steady forward progression,
stable head. No crawl/shuffle/drag; coherent with the measured 64-69%
stance duty.

**Push-recovery evaluation** (Task 2.7 gate, first ever): under active
interval pushes (±0.3 m/s, 4-7 s), v4_robust falls in **6.84%** of episodes
vs **46.28%** for the Run A (v3-recipe) policy under the identical schedule
— an 85% relative reduction from the robust bundle (DR + asym obs +
velocity limit; both are 0% falls unpushed). Full results, the fairness
caveat (bundle vs per-lever), and verdict in
`docs/jetson-mod/validation_results.md`.

**Verdict: GATE PASS.** Run B walks with gait quality statistically
equivalent to Run A and v3 (all three within ~0.1°/2% on ref-RMS — a
tie, not a ranking) while adding two
deployment-critical properties v3 never had: robustness to dynamics
randomization (the policy was trained under pushes/mass/CoM/friction
variation) and a hardware-realizable 59-dim observation (no dependence on
the unmeasurable base linear velocity). **Run B is the deployment
candidate.**

## Selection

**Deployment candidate: `v4_robust` (Run B).** It matches v3-quality gait
on the corrected robot model, is the only policy trained with dynamics
domain randomization (a real robustness margin — see
`validation_results.md` for the push-recovery quantification), and is the
only policy whose actor observation is realizable on the BNO055-equipped
hardware. Run A confirms the model correction itself is sound; Run B is
what should be exported to ONNX/TensorRT for the Jetson (59-dim actor
layout — see `AGENTS.md`, RL Training section).
