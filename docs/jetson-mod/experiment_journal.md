# Experiment Journal — PPO vs AMP Locomotion (Open Duck Mini v2)

Chronological record of every training run in the "Designed vs. Learned
Imitation" study (EN.665.645 research project). One config lever changes per
run; each entry records the delta, the training signals, the gate evaluation,
and the verdict. Quantitative protocol details live in
`algorithm_comparison.md`; per-policy raw metrics in `eval_results/*.json`.

**Compute parity:** full PPO runs are 3000 iterations x 24 steps/env x 4096
envs (294.9M env-steps, ~1.8-1.9 h on DGX Spark GB10). AMP runs are 72,000
skrl timesteps x 4096 envs (294.9M env-steps — exactly matched; ~2.2-2.3 h
wall-clock). All runs use
`--video --video_length 200 --video_interval 5000` (training filmstrips in
each run dir under `videos/train/`).

**Gates:** G1 = corrected PPO baseline beats v2 on gait symmetry. G2 = AMP
pure-style produces recognizable forward walking with healthy discriminator.
G3 = command-conditioned AMP tracks velocity within ~2x of PPO v3 error.

---

## Run index

| # | Run | Algorithm | Date | One-lever delta | Verdict |
|---|---|---|---|---|---|
| 1 | `ppo_v2` | PPO (RSL-RL) | 2026-04-13 | (baseline, phase bug present) | Walks; limp later quantified |
| 2 | `v3_smoke` | PPO | 2026-06-12 | phase fix bundle, 100-iter validation | PASS — proceed to full run |
| 3 | `ppo_v3` | PPO | 2026-06-12 | phase fix bundle, full 3000 iters | **G1 PASS** — new baseline |
| 4 | `amp_purestyle` | AMP (skrl) | 2026-06-12 | first AMP run (template config) | **G2 FAIL** — mode collapse |
| 5 | `amp_purestyle2` | AMP | 2026-06-12 | reference velocity-consistency fix | running |

---

## Run 1 — `ppo_v2` (archived baseline, trained 2026-04-13)

- **Config:** v2 BDX-aligned reward; imitation reward evaluated phase-in-seconds
  (the bug, undiscovered at training time). `exported_policies/v2_bdx_imitation_ppo/`.
- **Training:** mean reward 238.8 (last-100 mean); imitation term
  hard-plateaued at +1.03 (51% of the ~2.0 ceiling) from iter ~360 onward;
  falls 0.065%; 1.93 h.
- **Gate evaluation (standardized protocol, 3,200 episodes):**
  fall rate 0.0% | ref-tracking RMS **10.27 deg** (vs corrected reference) |
  stance-duty asymmetry **14.6 pp** (L 75.6 / R 61.0) | leg ROM ratio **0.82** |
  action std 0.247 | jerk 0.051 | vel error 0.134 m/s | energy 14.4 W
- **Post-hoc finding (the paper's bug case study):** the policy faithfully
  learned the corrupted reference — left-leg hip ROM 17 deg vs right 43 deg
  measured in rollout; tracks the *buggy* reference at 4.06 deg RMS but the
  *true* gait at 13.66 deg. Root cause: polynomials fit over normalized
  t in [0,1] but evaluated at seconds in [0, 0.54) — 54% of the gait cycle,
  asymmetric contact schedule, +0.34 rad reference teleport per cycle.

## Run 2 — `v3_smoke` (100-iteration validation of the fix bundle)

- **Delta:** normalized integer-counter phase evaluation + reference clamped
  to soft limits + zero-command gating + command ranges clipped to grid hull.
- **Result @ iter 99:** mean reward 219.5, imitation +0.845 — vs v2's 195.9 /
  +0.146 at the same point. **PASS** (corrected reference is more learnable).

## Run 3 — `ppo_v3` (corrected baseline, run `2026-06-12_00-44-12`)

- **Delta:** same as run 2, full 3000 iterations, seed 42, identical PPO
  hyperparameters to v2 (clean reward-semantics ablation).
- **Training:** mean reward 253.0 (last-100 mean; peak 254.8); imitation
  term +1.61 last-100 (81% of ceiling, peak +1.70) — v2's +1.03 plateau
  surpassed by iter 107; falls 0.04%; 1.78 h.
- **Gate G1 evaluation (3,200 episodes, identical protocol to run 1):**

| Metric | ppo_v2 | **ppo_v3** | fixed reference demands |
|---|---|---|---|
| Ref-tracking RMS (deg) | 10.27 | **4.59** | — |
| Stance-duty asymmetry (pp) | 14.6 | **4.3** | ~4.6 |
| Leg ROM ratio L/R | 0.82 | **1.02** | 1.0 |
| Fall rate (%) | 0.0 | 0.0 | — |
| Action std | 0.247 | 0.234 | — |
| Mean squared jerk | 0.051 | 0.069 | — |
| Vel error (m/s) | 0.134 | 0.155 | — |
| Energy proxy (W) | 14.4 | 21.6 | — |

- **Verdict: G1 PASS.** The limp is gone (asymmetry matches the reference's
  own 4.6 pp; both legs swing equally). Honest trade-offs: ~50% more energy
  and slightly higher jerk/velocity error — the cost of full strides and of
  faithfully reproducing the reference's waddle dynamics.
- **Artifacts:** `exported_policies/v3_bdx_imitation_ppo/`; ONNX in run dir
  `exported/`; eval video `videos/play/`.

## Run 4 — `amp_purestyle` (first AMP run, `2026-06-12_03-49-57_amp_torch`)

- **Config:** DuckAmpEnv (DirectRLEnv port), single forward clip
  (`duck_gait_0.148_-0.037_-0.074.npz`, 10 tiled cycles @ 50 fps), pure style
  (task_reward_weight 0.0), template hyperparameters ([1024,512] nets,
  lr 5e-5, gradient penalty 5.0, disc batch 4096), 72k timesteps.
- **Training signals:** ran mechanically end-to-end (~2h20m, no crashes).
  Discriminator loss 1.42 -> ~0.05 last-100 (range 0.04-0.11; near-perfect
  real/fake separation); mean episode length 500 -> ~750/1000 plateau;
  policy std fixed 0.055.
- **Gate G2 instrumented rollout (64 envs x 20 s):** forward velocity
  **-0.002 m/s** (clip: 0.148); hip-pitch ROM ~0 deg; 85% stable standing.
- **Verdict: G2 FAIL — mode collapse to standing.** Discriminator saturation:
  policy gets no style gradient, standing is the safest behavior.
- **Root cause (measured):** the converter clamped reference joint positions
  to soft limits but kept the *unclamped* analytic velocity — 90 frames per
  knee per clip (33% of frames) in a physically impossible state: position
  pinned at the limit while the velocity field claims up to 25.3 rad/s
  (mean 5.7-8.1) of motion. States the policy can never produce — a trivial,
  unbeatable real/fake separator. Paper relevance: a data-engineering
  artifact breaking AMP loudly, mirroring the reward-engineering artifact
  breaking PPO quietly (H2).

## Run 5 — `amp_purestyle2` (`2026-06-12_06-17-30_amp_torch`) — RUNNING

- **One-lever delta:** reference velocities are now the periodic central
  difference of the *clamped* position trajectory — every reference state
  physically self-consistent (pinned stretches read ~0 velocity). Converter
  fix documented in `convert_gait_library_to_amp.py`.
- **Early signals (at 65%):** discriminator loss 0.12-0.14 (vs run 4's
  0.04-0.10 — separation is harder, the intended direction); mean episode
  length ~350 (vs run 4's ~750 — consistent with a policy that *moves and
  sometimes falls* rather than freezing, but ambiguous until the G2 rollout).
- **Verdict:** pending G2 rollout at completion.

---

## Planned lever ladder (one per run, in order, each ~2 h)

1. ~~Reference data consistency~~ (run 5, in flight)
2. Discriminator gradient penalty 5 -> 10 (if saturation persists)
3. Per-dimension real-vs-fake distribution audit -> feature pruning of the
   AMP observation (measurement-driven, not blind)
4. Exploration: unfix log-std / raise to ~0.15
5. **Fallback of last resort:** regenerate the reference dataset from PPO v3
   rollouts (physically consistent by construction; reframes the study as
   two-stage distillation — documented trade-off, near-guaranteed learnable)

Phase B (`Isaac-OpenDuck-AMP-v0`, task 0.5/style 0.5, full clip set) launches
as soon as a pure-style run shows forward locomotion — or directly after
lever 2 regardless, since the task reward provides a non-adversarial gradient
that does not depend on winning the discriminator game.
