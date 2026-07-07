# Validation Results — Task 2.7 (Best Policy in Isaac Sim)

Whole-project deliverable (the Open Duck Mini Jetson robot project). This is
the gate the task plan names as "the critical gate before committing to
hardware." It certifies the selected locomotion policy in Isaac Sim against
gait quality, contact validity, and — for the first time in this project —
**push recovery**.

All numbers are measured (never estimated): gait/contact metrics from
`scripts/evaluate_policies.py` (JSONs in `docs/jetson-mod/eval_results_v4/`);
the push-recovery numbers from the same protocol with `--keep-pushes` on the
PushEval tasks; the video observations from deterministic rollouts. Model:
layout v2.1, frame-correct inertia, 2.657 kg (see `component_layout_v2.md`).

## Selected policy

**`v4_robust` (Run B)** — PPO, RSL-RL, checkpoint
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`.
Trained on the corrected robot model with dynamics domain randomization,
asymmetric actor/critic observations (actor 59-dim, no base_lin_vel — the
BNO055 cannot measure it), and a BAM-measured joint velocity limit. Full
training/eval provenance in `docs/jetson-mod/v4_retrain_results.md`.

Why this policy over v3 / Run A: identical gait quality (below), plus the
only measured robustness margin (push section) and the only
hardware-realizable observation vector.

## 1. Gait quality + contact validity (gated protocol)

5 velocity conditions × 640 episodes = 3200 episodes, DR off,
`eval_results_v4/v4_robust.json`. Gate = both feet's stance duty in
[40, 90]% per condition (applied before ranking; see
`algorithm_comparison.md`).

| Metric | v4_robust | acceptance | result |
|---|---|---|---|
| Gait-validity gate | **5 / 5 conditions** | ≥ 4/5 | PASS |
| Fall rate (no push) | **0.00%** (0/3200) | < 1% | PASS |
| Reference tracking RMS | **4.49°** | ≤ v3's 4.59° | PASS |
| Stance duty L / R | 68.6 / 63.8% | both in [40,90] | PASS |
| Duty asymmetry | 4.85 pp | — | (v3: 4.32) |
| ROM ratio L/R | 0.99 | ≈ 1.0 | PASS |
| Velocity tracking err | 0.153 m/s | — | (v3: 0.155) |
| Yaw-rate err | 0.067 rad/s | — | (v3: 0.078) |
| Energy proxy | 21.8 W | — | (v3: 21.6) |

**Video audit** (`eval_results_v4/v4_robust_play.mp4`, deterministic vx=0.2
rollout): upright bipedal walking — high trunk, clear single-support swing
phases with the swing foot off the ground, steady forward progression,
stable head. No crawl / shuffle / foot-drag. Consistent with the 64-69%
stance duty. This audit is a mandatory protocol step (it is what exposed
the amp_v4 crawl that passed every aggregate metric).

## 2. Push recovery (Task 2.7 — measured for the first time)

The v3 evaluation, and the AMP campaign, never tested push recovery
(`evaluate_policies.py` disables pushes by default so quality metrics
reflect steady-state gait). Here pushes are held ACTIVE (`--keep-pushes`):
interval shoves every 4-7 s of ±0.3 m/s in x and y, ~4-7 per 30 s episode,
3200 episodes.

**Controlled DR-benefit comparison** — the same push schedule applied to the
DR policy (Run B) and the no-DR policy (Run A / `v4_inertials`, identical
recipe except no domain randomization):

| Policy | Falls, no push | Falls, under push | Gate under push |
|---|---|---|---|
| **v4_robust (DR)** | 0.00% | **6.84%** (219/3200) | 5/5 |
| v4_inertials (no DR) | 0.00% | 46.28% (1481/3200) | 5/5 |

**Domain randomization reduced the push-induced fall rate by 85% relative
(46.3% → 6.8%).** Both policies are identical (0% falls) without pushes, so
the difference is entirely attributable to the DR added in the v4-robust
config. Per-condition (DR / no-DR): forward 13.9/41.1, backward 5.8/60.2,
lateral 3.3/44.2, turn 2.0/45.9, combined 9.2/40.0% — DR helps in every
condition, most in the lateral/turn/backward cases the no-DR policy never
saw perturbed.

Under pushes, v4_robust's surviving episodes still walk validly (gate 5/5)
with only mild quality degradation (ref RMS 4.49 → 4.55°, yaw-rate err
0.067 → 0.087). Raw data: `eval_results_v4/v4_robust_pusheval.json`,
`eval_results_v4/v4_inertials_pusheval.json`.

## 3. Verdict

**GATE PASS — v4_robust is cleared for hardware (Phase 4), sim-side.** It
walks with v3-equivalent gait quality on the corrected 2.657 kg model,
survives 93.2% of push episodes (vs 53.7% for the no-DR policy), and its
59-dim actor observation is realizable on the BNO055-equipped robot.

## 4. Honest limitations (must carry into Phase 4)

- **6.84% is a real, moderate robustness — not bulletproof.** A ±0.3 m/s
  shove is a firm nudge; harder impacts were not tested. The forward
  condition (13.9%) is the weakest and worth a targeted look.
- **Sim only.** These are PhysX results on the modified model; sim-to-real
  gap (unmodeled cabling stiffness, servo thermal/backlash, IMU noise,
  contact/friction fidelity at duck-foot scale) is not captured here.
- **No ONNX/TensorRT parity check yet.** The 59-dim actor still needs
  ONNX export + a numerical parity check vs the PyTorch policy, and the
  Jetson observation builder must emit the exact 59-dim term order
  (documented in `.claude/rules/rl-training.md`) — Phase 4 / Task 2.6.
- The push schedule (±0.3 m/s, 4-7 s interval) is a design choice, not a
  hardware-derived spec; revisit against real disturbance measurements.

## 5. Reproduction

```bash
cd ~/IsaacLab
# gait + contact gate (DR off):
./isaaclab.sh -p <repo>/scripts/evaluate_policies.py \
  --policies "v4_robust=Isaac-Velocity-Rough-OpenDuck-Robust-Play-v0:rsl_rl:<ckpt>" \
  --output_dir <repo>/docs/jetson-mod/eval_results_v4 \
  --comparison_md <repo>/docs/jetson-mod/v4_comparison.md --headless
# push recovery (pushes ON):
./isaaclab.sh -p <repo>/scripts/evaluate_policies.py \
  --policies "v4_robust_pusheval=Isaac-Velocity-Rough-OpenDuck-PushEval-v0:rsl_rl:<ckpt>" \
  --keep-pushes --output_dir <repo>/docs/jetson-mod/eval_results_v4 ...
# no-DR baseline under identical pushes:
#   task Isaac-Velocity-Rough-OpenDuck-PlainPushEval-v0, checkpoint = v4_inertials
```
