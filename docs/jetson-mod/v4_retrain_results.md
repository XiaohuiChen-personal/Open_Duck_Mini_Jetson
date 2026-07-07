# v4 Retrain Results (cad-redesign branch)

Branch-local record of the post-CAD retraining campaign. **Deliberately NOT
part of `experiment_journal.md`** — the journal and its run numbering
(runs 1-14) are course-project artifacts on the v2 branch (user directive
2026-07-07); this document serves the robot project. Data-sourcing follows
the same discipline: training numbers are last-100-iteration TensorBoard
means, behavioral numbers come from the gated `evaluate_policies.py`
protocol (JSONs in `eval_results_v4/`), and every run gets a mandatory
video audit.

**What changed under these runs vs every earlier policy:** layout v2.1
(`component_layout_v2.md`) + the inertia frame correction (upstream
principal-frame diaginertia had been read as body-frame — trunk inertia
was off +78%/−10%/−27% for all of runs 1-12) + the Part-2 CAD mesh cuts
(robot 2.657067 kg, −88.5 g). USD regenerated and hash-guarded.

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
`eval_results_v4/v4_inertials.json`; v3 baseline re-gated in the same
table, `eval_results_v4/ppo_v3.json`; report `v4_comparison.md`):

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

Pending (launches immediately after Run A's closeout). Config: task
`Isaac-Velocity-Rough-OpenDuck-Robust-v0` — Run A + dynamics DR (pushes,
trunk mass/CoM, friction with `make_consistent`), asymmetric 59/62-dim
actor/critic observations (actor drops the hardware-unmeasurable
base_lin_vel), velocity_limit_sim 8.94 rad/s. Gates: the 5-condition
gated protocol + the first-ever push-recovery evaluation
(`Isaac-Velocity-Rough-OpenDuck-PushEval-v0`), feeding
`validation_results.md` (Task 2.7).
