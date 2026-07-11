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
| 5 | `amp_purestyle2` | AMP | 2026-06-12 | reference velocity-consistency fix | **G2 FAIL** — moves+falls, no stride |
| 6 | `amp_command` | AMP | 2026-06-12 | Phase B: +task reward 0.5, 22-clip set | **G3 FAIL** — stand-and-dither exploit |
| 7 | `amp_command2` | AMP | 2026-06-12 | sharp tracking kernel + action-rate penalty + clamped targets | **partial** — legs move, in-place march, UNCONVERGED |
| 8 | `amp_command3` | AMP | 2026-06-12 | resume run 7 + 72k more timesteps (zero config changes) | **G3 PASS (tracking)** — walks, 0.023 m/s err, 0 falls; style = shuffle, not waddle; gate 2026-07-06: 2/5 (one-foot drag off-forward) |
| 9 | `amp_command4` | AMP | 2026-06-12 | num_amp_observations 2 -> 4 (discriminator temporal context) | **partial** — real strides appear, but one-legged |
| 10 | `amp_command5` | AMP | 2026-06-12 | num_amp_observations 4 -> 14 (window > half gait cycle) | **partial** — bilateral reference-amplitude gait, but task drowned (0.18 err, 14% falls) |
| 11 | `amp_command6` | AMP | 2026-06-12 | 14-frame window + task/style 0.7/0.3 (recover translation) | **DIVERGED** — action blowup, reward -> -6766 |
| 12 | `amp_command7` | AMP | 2026-06-15 | run-11 fix (raw actions clipped +/-5.0) + 14-frame + task/style 0.6/0.4 | ~~best striding AMP~~ REVISED 2026-07-06: **crawl** (see run-12 addendum) — converged, 0-1% falls, but locomotes on its body |
| 13 | `amp_v5` | AMP | 2026-07-10 | termination height 0.10 -> 0.13 m (run-12 config, hydra override) | **FAIL (gate 0/5)** — crawl excised, posture upright, but stand-and-pivot: no stepping; root-cause #1 (data physics) now primary |
| 14 | `amp_v6` | AMP | 2026-07-11 | motion library -> 22 physically-consistent clips recorded from ppo_v3 rollouts (0 dead AMP dims) | **FAIL (gate 0/5)** — stands upright; but discriminator ENGAGED all run (no tells): data fix healed the adversarial game without producing locomotion; exploration ceiling promoted |

---

## Artifact locations & evaluation coverage (read before citing numbers)

**Checkpoint / log dirs** (all training-loop logs under `~/IsaacLab/logs/`):
- PPO: `rsl_rl/open_duck_ppo/<timestamp>/` — final = `model_2999.pt`
- AMP: `skrl/open_duck_amp[_purestyle]/<timestamp>_amp_torch/` — final =
  `checkpoints/agent_72000.pt`
- Each run entry below names its exact `<timestamp>` for reproduction.

**Archived policies** (`exported_policies/`, checkpoint + agent.yaml + env.yaml):
v1_imitation_ppo, v2_bdx_imitation_ppo, v3_bdx_imitation_ppo (PPO only).
Two key AMP checkpoints are now also archived here (2026-06-15):
`amp_v1_run8_command/` (the precise command-following shuffler) and
`amp_v4_run12_command/` (converged command-follower; video audit 2026-07-06:
locomotes in a crawl — see the run-12 addendum), each with its
`params/{env,agent}.yaml`. Other AMP runs remain only in the IsaacLab logs.

**Full-protocol evaluation** (5 conditions x 10 windows x 64 envs, in
`eval_results/*.json`, tabulated in `algorithm_comparison.md`): exists for
**ppo_v2, ppo_v3, amp_v1 (run 8), amp_v2 (run 9), amp_v3 (run 10)** — these
5 are the directly-comparable rows. Runs 4/5/6/7/11 were characterized by the
single-command forensic rollout (`measure_amp_gait.py` / inline forensic,
numbers in their entries) but did NOT get a full eval — they failed their gate
and a full sweep was not warranted. Cite their forensic numbers as diagnostic,
not as protocol-comparable. Run 12 = `amp_v4` (full eval done 2026-06-15).
Full-eval set is now: ppo_v2, ppo_v3, amp_v1 (r8), amp_v2 (r9), amp_v3 (r10),
amp_v4 (r12), amp_v5 (r13), amp_v6 (r14).

**Pre-study / non-study runs on disk** (documented here so stray log dirs are
not mistaken for study runs):
- `rsl_rl/.../2026-03-31_22-48-29` = the original v1 imitation-PPO policy
  (exp-kernel reward), pre-dates the v2/v3 study; archived in
  `exported_policies/v1_imitation_ppo/` with its own README. Referenced as
  historical context only.
- `rsl_rl/.../2026-03-31_13-51-16` and `.../2026-04-13_00-08-18` = early
  100-iteration validation runs; `.../2026-04-13_00-03-29` = an aborted run.
  No final checkpoints; not part of the comparison.
- **(added 2026-07-10)** `rsl_rl/open_duck_ppo/2026-07-06_20-58-16`
  (v4_inertials) and `rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43`
  (v4_robust) = robot-project runs launched from the `cad-redesign` branch
  (worktree `Open_Duck_Mini_Jetson-cad`), **NOT study runs — excluded from
  the PPO-vs-AMP comparison** (user directive 2026-07-07, reconfirmed
  2026-07-10). They train on a *different robot model* (CAD layout v2.1,
  trunk-inertia frame correction, regenerated USD; v4_robust additionally
  enables domain randomization), so their metrics are not comparable to
  runs 1-12 and they carry no run number here. Branch-local record:
  `v4_retrain_results.md` + `eval_results_v4/` on `cad-redesign`.
  Paper-relevant side fact from that work: runs 1-12 all trained on a
  trunk inertia mis-read as body-frame (+78/-10/-27% per axis). The study
  comparison remains internally valid — all six evaluated policies share
  that same model, and v4_inertials shows the corrected model retrains to
  v3 parity (mean reward 253.3 vs 253.0) — but disclose this as a
  threats-to-validity note in the final paper.

---

## Run 1 — `ppo_v2` (archived baseline, log `2026-04-13_00-25-01`)

- **Config:** v2 BDX-aligned reward; imitation reward evaluated phase-in-seconds
  (the bug, undiscovered at training time). Log dir
  `~/IsaacLab/logs/rsl_rl/open_duck_ppo/2026-04-13_00-25-01`; checkpoint
  archived at `exported_policies/v2_bdx_imitation_ppo/`; full eval
  `eval_results/ppo_v2.json`.
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

## Run 2 — `v3_smoke` (100-iteration validation, log `2026-06-12_00-26-05`)

- **Delta:** normalized integer-counter phase evaluation + reference clamped
  to soft limits + zero-command gating + command ranges clipped to grid hull.
- **Result @ iter 99:** mean reward 219.5, imitation +0.845 — vs v2's 195.9 /
  +0.146 at the same point. **PASS** (corrected reference is more learnable).
- Log dir `~/IsaacLab/logs/rsl_rl/open_duck_ppo/2026-06-12_00-26-05` (100-iter
  validation, no final-3000 checkpoint; not archived — superseded by run 3).

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

## Run 5 — `amp_purestyle2` (`2026-06-12_06-17-30_amp_torch`)

- **One-lever delta:** reference velocities are now the periodic central
  difference of the *clamped* position trajectory — every reference state
  physically self-consistent (pinned stretches read ~0 velocity). Converter
  fix documented in `convert_gait_library_to_amp.py`.
- **Training (TB, last-100/last-10 means):** completed 72,000 timesteps,
  2.33 h. Discriminator loss 1.43 -> held 0.12-0.14 mid-run (vs run 4's
  fast saturation) -> re-saturated to 0.072 by the end. Mean episode length
  ~357 (run 4: ~750).
- **Gate G2 rollout (final `agent_72000.pt`, 64 envs x 20 s, idle GPU):**
  forward velocity **+0.003 m/s** (clip: 0.148); hip-pitch ROM ~0 deg;
  34.3% alive, **67.2% of envs fell** (source: /tmp/g2_run5_final.txt via
  `scripts/measure_amp_gait.py`).
- **Verdict: G2 FAIL — but a *different* failure mode than run 4.** The
  data fix demonstrably changed the equilibrium: the discriminator took
  far longer to saturate, and the policy attempts motion (falls) instead
  of freezing in a safe stand. It still never strides; the discriminator
  ultimately wins again.
- **Method note:** `best_agent.pt` selection uses total env reward, which
  for pure-style equals episode length — a criterion that *favors standing*.
  Final-step checkpoints are the honest evaluation artifact for pure-style
  runs. (best_agent and agent_72000 had different weights but bit-identical
  seeded-rollout metrics — both behaviorally collapsed.)

---

## Run 6 — `amp_command` (Phase B, `2026-06-12_08-47-30_amp_torch`)

- **Config:** DuckAmpCommandEnvCfg — velocity command in policy obs (54 dims),
  task_reward_weight 0.5 / style 0.5, 22 curated clips, 72k timesteps, 2.42 h.
- **Training signals (TB):** episode length ~952/1000 sustained from early on;
  env task reward 0.854/1.0 (looked like excellent tracking); discriminator
  saturated (last-10 mean 0.046).
- **Gate G3 forensic rollout (fixed cmd vx=0.2, 64 envs x 20 s,
  `agent_72000.pt`, /tmp/g3_forensic.txt):** survival 97%, BUT achieved
  vx = -0.006 m/s (tracking error 0.209 = the full command), ALL leg-joint
  ROMs 0.0 deg, action rate mean |da| = 26.1/step with direction flips on
  65% of steps.
- **Verdict: G3 FAIL — stand-and-dither reward exploit, fully diagnosed.**
  Two compounding flaws in the env's task reward (mine, not skrl's):
  (1) kernel too flat for duck-scale commands — standing still earns
  0.5*exp(-8*||cmd||^2)+0.5*exp(-2*wz^2) ~= 0.87/1.0 for |cmd| <= 0.26 m/s,
  matching the observed 0.854 training reward almost exactly; walking risks
  falls for ~0.13 marginal reward. (2) No smoothness term — the saturated
  discriminator provides no styling gradient, so violent action dithering
  (PD-filtered into a stable stance) is free. Reproduces the BDX paper's
  no-gait-signal ablation ("rapidly shuffles the feet") in exploit form.
- **Paper relevance (H2):** even in the "learned reward" method, the
  *hand-designed task component* was the exploitable part — reward
  engineering errors simply moved, not disappeared.

## Run 7 — `amp_command2` (`2026-06-12_11-26-06_amp_torch`)

- **Composite lever (targeting run 6's measured exploit):** tracking kernel
  sharpened to std 0.1 m/s linear / 0.25 rad/s angular (standing under a
  0.2 m/s command now earns ~0.02 of the linear half, was ~0.87 overall);
  action-rate penalty -0.05*mean(da^2) added to the env task reward; PD
  targets clamped to soft joint limits.
- **Training signals (TB):** task reward 0.32 -> 0.54, STILL CLIMBING at
  budget exhaustion (no plateau). Episode length ~952 sustained.
  **Discriminator loss 1.7-2.2 SUSTAINED — first run where the
  discriminator never saturated**: with dithering gone and legs moving,
  the adversarial game finally has gradient on both sides. Policy std
  pinned at 0.055 (fixed_log_std, the remaining exploration ceiling).
- **Forensic rollout (fixed cmd vx=0.2, agent_72000.pt):** action rate
  26.1 -> **0.003** (dithering eliminated); leg ROMs now real (hip pitch
  14.8/9.9 deg, ankle 14.2/10.0, knee 7.1/6.1 — was 0.0 across the board);
  survival 95.5%; BUT achieved vx = -0.000 — it marches **in place**,
  tracking error 0.202.
- **Verdict: partial pass — both run-6 pathologies fixed, gait articulated
  and smooth, locomotion not yet learned, and the run was unconverged.**
  Decision: cheapest-hypothesis-first — run 8 resumes this agent for
  another 72k timesteps with ZERO config changes before reaching for the
  exploration lever. (Compute-parity note for the paper: run 8's policy
  totals 144k timesteps = 2x the PPO budget; report accordingly.)

## Run 8 — `amp_command3` (`2026-06-12_13-47-33_amp_torch`, resume of run 7)

- **Lever:** none — run 7's agent resumed (optimizer + preprocessors intact)
  for 72k more timesteps. Total policy budget 144k timesteps = 2x PPO
  (report in any compute-parity comparison).
- **Training signals (TB):** task reward 0.54 -> 0.80; episode length ->
  995/1000; discriminator loss 1.42-1.75 SUSTAINED (engaged, never
  saturated — but also never winning).
- **Forensic rollout (fixed cmd vx=0.2, agent_72000.pt, /tmp/g3_run8 logs):**
  **achieved vx +0.192 m/s (tracking error 0.023 m/s), 100% survival,
  0 falls.** THE DUCK WALKS WHERE COMMANDED. Gait style, however: leg ROMs
  1.5-8 deg (reference demands ~27-43 deg), action direction flips ~every
  step at small amplitude — a smooth high-frequency shuffle/glide, not the
  reference waddle. The action-rate penalty bounds the oscillation
  (mean |da| 0.408) but does not produce striding.
- **Verdict: G3 PASS on command tracking; style criterion open.** The
  unconverged-budget hypothesis was correct — no new levers were needed for
  locomotion. Style finding: the discriminator, though engaged, FAILS to
  separate a ~2-8 deg shuffle from the ~30 deg reference waddle (loss
  1.4-1.7 = fooled half the time). With only num_amp_observations=2
  (a 20 ms window), stride amplitude is nearly invisible in its features —
  motivating run 9's lever: temporal context 2 -> 4 frames.
- **Paper relevance (H1):** functional locomotion achieved by AMP+task at
  2x compute, but gait naturalness so far DOESN'T come for free from the
  discriminator — style fidelity tracks discriminator feature adequacy,
  not just engagement.
- **Full-protocol evaluation (5 conditions x 10 windows x 64 envs, as
  `amp_v1` in algorithm_comparison.md / eval_results/amp_v1.json; required
  adding a contact sensor to DuckAmpEnv + a Direct-env command hook in the
  evaluator):** velocity tracking 0.026 m/s — **6x better than both PPO
  policies** (v2 0.134 / v3 0.155); ang vel error 0.062 (best); fall rate
  0.19%; BUT reference-tracking RMS 10.9 deg (v3: 4.59), stance asymmetry
  29.1 pp (worst), mean squared jerk **14.5 vs PPO's 0.05-0.07 (200x)**.
  Clean style-vs-tracking trade-off: AMP's policy obeys the command better
  than the designed-reward policies and looks far less like a duck doing it.
- **ADDENDUM 2026-07-06 (gait-validity gate).** The retrofitted gate (both
  feet's stance duty in [40, 90]%; see run-12 addendum) scores this policy
  **2/5 conditions valid** (eval_results/amp_v1.json): forward (50.1/50.1)
  and mixed (50.0/51.1) pass, but in the backward (98.7/52.0), lateral
  (50.0/99.7), and turn (99.1/51.1) conditions **one foot drags at 98-100%
  duty** — the aggregate 29.1 pp asymmetry is the mean of near-zero and
  ~47-50 pp conditions, not a uniform limp. Status revised accordingly:
  run 8 is the only AMP policy that walks at all, and its micro-shuffle
  degrades to one-foot dragging outside the forward/mixed conditions.

## Run 9 — `amp_command4` (`2026-06-12_16-40-01_amp_torch`)

- **Lever:** num_amp_observations 2 -> 4 (80 ms discriminator window),
  fresh init (discriminator input 204 dims). Also carries two inert infra
  changes from the eval work: contact sensor in the scene (not consumed by
  training) and the evaluator's Direct-env command hook.
- **Forensic rollout (fixed cmd vx=0.2, agent_72000.pt):** 100% survival,
  0 falls; vx +0.148 (tracking error 0.057 — striding costs precision vs
  run 8's gliding 0.023); wz drift -0.093 (asymmetric gait veers).
  **Leg ROMs: left knee 38.0 deg, left ankle 37.4 deg, hip pitch 17.8/20.2
  — real strides in the reference's ~27-43 deg range for the first time.**
  BUT right knee 9.7 / right ankle 9.3 deg: a one-legged stride.
- **Verdict: partial — the temporal-context lever produced striding, and
  the failure that remains is precisely diagnostic.** A 4-frame (80 ms)
  window cannot observe gait alternation: every policy window resembles
  some reference left-swing window, so a left-only stride is style-optimal
  under ANY window shorter than half the gait cycle (0.27 s). Alternation
  is a cycle-level statistic.
- **Run 10 lever (pre-registered here):** num_amp_observations 4 -> 14
  (280 ms > half cycle) — the discriminator's receptive field must span
  the alternation period to penalize one-legged gaits. Paper angle: AMP
  style fidelity as a function of discriminator temporal receptive field
  vs gait period — runs 8/9/10 form a clean sweep (40/80/280 ms).

## Run 10 — `amp_command5` (`2026-06-12_19-48-52_amp_torch`)

- **Lever:** num_amp_observations 4 -> 14 (280 ms window > half gait cycle
  0.27 s; discriminator input 714 dims), fresh init. 2.68 h, 57 W (vs ~34 W
  for 2-frame runs — the 7x bigger discriminator input is real compute).
- **Training (TB):** discriminator loss oscillates 0.7-1.35, NEVER saturates
  (the larger window keeps it genuinely competitive); episode length ~882
  (down from run 8/9's ~950-995). Instantaneous reward 0.52.
- **Forensic (fixed cmd vx=0.2, agent_72000.pt):** survival 86.4% (14% fell);
  vx +0.025 (tracking error 0.179 — much worse than run 8's 0.023);
  **leg ROMs hip pitch L27.3/R54.3, ankle L38.2/R75.9 deg — bilateral and
  in/above the reference's ~27-43 deg range, the most reference-like
  amplitude of any run.** action rate 0.070 (smooth).
- **Verdict: partial — full receptive-field sweep now complete (runs
  8/9/10 = 40/80/280 ms windows).** Clean monotone trend:
  | window | gait structure | task tracking | falls |
  | 40 ms (run 8) | symmetric micro-shuffle, ROM ~5 deg | 0.023 (best) | 0% |
  | 80 ms (run 9) | one-legged stride, ROM 38/10 | 0.057 | 0% |
  | 280 ms (run 10) | bilateral big-amplitude gait | 0.179 (worst) | 14% |
  As the discriminator's temporal receptive field grows to span the gait
  cycle, it enforces progressively more reference-like gait STRUCTURE — but
  at a fixed 0.5/0.5 task/style weight the now-stronger style reward
  increasingly overrides the velocity-tracking task, degrading forward
  translation and stability. The gait "performs in place."
- **Paper finding (novel, beyond the proposal):** AMP style fidelity is
  governed by the discriminator's temporal receptive field relative to the
  gait period; style and task compete, and the balance that suffices for a
  short window over-weights style for a long one.
- **Run 11 lever:** keep the 14-frame window (it produces real gait
  amplitude) but shift task/style 0.5/0.5 -> 0.7/0.3 to recover forward
  translation — the "best of both" hypothesis.

## Run 11 — `amp_command6` (`2026-06-12_22-39-45_amp_torch`)

- **Lever:** task/style 0.5/0.5 -> 0.7/0.3, 14-frame window kept. Intent:
  recover run 10's lost forward translation. ONLY this weight changed.
- **Training (TB):** instantaneous reward 0.34 -> -119 (50%) -> **-6766
  (end)** — monotonic DIVERGENCE, not convergence. Discriminator saturated
  (0.11). Episode length pinned at 999 (never falls).
- **Forensic (fixed cmd vx=0.2, agent_72000.pt):** vx -0.002 (no
  locomotion); ALL leg ROMs ~0.2-0.8 deg (frozen); **action rate mean |da|
  = 64.3/step (raw), p95 149** — far worse than run 6's 26; 0 falls.
- **Verdict: FAIL — action-magnitude divergence, root cause identified.**
  Mechanism: `_apply_action` clamps PD TARGETS to soft limits (added run 7),
  which DECOUPLES raw policy actions from robot motion — the GaussianMixin
  mean is unbounded, so the network can emit ever-larger thrashing actions
  that all clamp to a standing pose, with no physical feedback to correct
  them. The -0.05 action-rate penalty was the only brake; dropping style
  weight to 0.3 removed the imitation pressure that had incidentally kept
  actions sane in run 10, the penalty alone could not contain the blowup,
  and a positive-feedback loop drove actions (and reward) to divergence.
- **Lesson (design flaw, not a tuning miss):** clamping targets while
  leaving raw actions unbounded + only softly penalized is unstable. The
  principled fix is to BOUND the actions (clip_actions / tanh squashing in
  the policy), not to reweight rewards. This invalidates reward-reweighting
  as a lever until actions are bounded.
- **Paper relevance:** a third, distinct AMP failure mode (after data
  saturation in run 4 and reward-hacking in run 6) — adversarial training's
  instability surfacing as raw-action divergence (supports H3: AMP is less
  stable than PPO under massive parallelism).

## Run 12 — `amp_command7` (`2026-06-15_01-42-56_amp_torch`) — the capstone

- **Lever (composite, the run-11 fix + recommended params):** raw actions
  clipped to +/-5.0 in `_pre_physics_step` (bounds joint-target offset to
  +/-1.25 rad, kills the run-11 divergence channel); 14-frame discriminator
  window kept; task/style rebalanced 0.7/0.3 -> **0.6/0.4**.
- **Training (TB):** instantaneous reward 0.34 -> 0.63 -> **0.81 (converged,
  monotonic increase)** — the action clip fully resolved run 11's divergence
  (which went to -6766). Episode length -> 985; discriminator loss 1.08 ->
  0.16 (engaged). 2.79 h.
- **Forensic (fixed cmd vx=0.2, agent_72000.pt, heading-frame velocity):**
  vx +0.172 (2D err 0.055), 100% survival / 0 falls, action rate 0.358
  (smooth, NO divergence/dither), leg ROM hip-pitch L35.9/R25.9, knee
  L20.0/R0.6, ankle L20.7/R14.4 deg — **real bilateral stride amplitude,
  the best striding command-follower of the campaign.**
- **Full 5-condition eval (`amp_v4`):** fall 1.3% (best of any AMP run),
  stance asymmetry 0.33 pp (near-symmetric, ~ run 10), jerk 2.28 (6x smoother
  than the short-window runs 8/9). BUT: ref-tracking RMS 27.2 deg (worst of
  all policies — the strided gait does not match the SPECIFIC reference
  poses), energy proxy **170.5 W (8x PPO v3's 21.6 W — mechanically
  aggressive: action_std 0.73, large near-clip actions drive high steady
  torque)**, and lin-vel error 0.135 m/s aggregate.
- **MEASUREMENT DISCREPANCY (flagged, unresolved):** the fixed-forward
  forensic reports vel error 0.055 m/s (heading-frame) while the full eval's
  forward condition reports 0.223 m/s (world-frame). The 4x gap is almost
  certainly a frame/command-application difference between `g3_forensic.py`
  (heading-localized) and `evaluate_policies.py` (world-frame) — NOT two
  measurements of the same quantity. Until reconciled, cite the heading-frame
  forensic for "does it walk forward" and the full eval only for cross-policy
  relative comparison. TODO: align the two on frame convention.
- **Verdict: best striding AMP policy, and a partial "best of both."** The
  action-clip fix worked completely (stable training, 0-1% falls, smooth,
  real stride ROM). It did NOT cleanly dominate: vs run 8 (`amp_v1`, the
  precise shuffler, 0.026 err) it trades tracking precision and energy for
  gait amplitude and symmetry. PPO v3 remains the most balanced policy
  overall (best ref-fidelity 4.6 deg, lowest energy, 0 falls, decent
  tracking).
- **Paper relevance (H1/H3):** after 9 AMP runs, the learned-reward method
  produced either precise-but-unnatural (run 8) or natural-but-imprecise-and-
  costly (run 12) gaits, never strictly dominating the hand-designed PPO
  reward at equal quality. AMP's style fidelity is real but bought with
  substantial tuning, training instability, and energy cost — a nuanced
  result stronger than a simple "AMP wins/loses."
- **ADDENDUM 2026-07-06 (video audit — verdict revised).** A deterministic
  rollout video at fixed cmd vx=0.2 (agent_72000.pt, robot-tracking camera,
  rendered for the en665.645 midpoint deliverable) shows run 12 locomotes in
  a low forward **crawl**: trunk riding just above the 0.1 m termination
  height, feet rarely loaded past 1 N. This resolves the "stance duty
  0.7%/0.4% anomaly" flagged in the full eval (eval_results/amp_v4.json) as
  real behavior, and coherently explains the 170.5 W energy and 27.2° ref
  RMS. The same posture appears in the final training filmstrip
  (videos/train/rl-video-step-70000.mp4), so this is converged behavior,
  not a render artifact. The "best striding command-follower" verdict is
  therefore revised: the forensic's large joint ROMs are crawl motion, not
  strides; run 8 (amp_v1) remains the only walking AMP policy. Videos: all
  four policies at fixed forward cmd 0.2 m/s, hosted on Google Drive as
  individually shared files (per-file links in the en665.645 midpoint paper
  Sec. 5.5 / notebook Sec. 5; local copies in ~/gait_videos_midpoint/ on
  the Spark). Protocol consequence: a gait-validity
  gate (both feet's stance duty in [40, 90]%) was added to
  `evaluate_policies.py` as metric 9 and to `algorithm_comparison.md`
  (regenerable via `--report-only`); on the archived JSONs it scores
  ppo_v2 5/5, ppo_v3 5/5, amp_v1 2/5, amp_v2 4/5, amp_v3 5/5, amp_v4 0/5.
  The gate is necessary, not sufficient — the qualitative video audit
  remains a mandatory protocol step (see algorithm_comparison.md).

## Run 13 — `amp_v5` (`2026-07-10_23-47-45_amp_torch`)

- **Lever (single, pre-registered in the forward plan):** termination height
  0.10 -> 0.13 m, applied as a hydra CLI override
  (`env.termination_height=0.13`) on the unchanged run-12 config (14-frame
  window, task/style 0.6/0.4, action clip +/-5.0, 22 synthetic clips, 72k
  timesteps). Tests root-cause #2: is the crawl a termination-geometry
  reachability artifact? Startup verified from the run's dumped
  `params/env.yaml` (termination_height: 0.13).
- **Training (TB, last-100 means):** instantaneous reward 0.34 -> 0.4985
  (run 12 reached 0.81 — the crawl's reward is no longer earnable);
  episode length -> 929.9/1000 (the policy survives the raised floor);
  discriminator loss 1.09 -> 0.99 last-100 (min 0.217 mid-run, 1.33 at
  budget end — engaged throughout, policy fooling it at the end); policy
  std fixed 0.055; 2.61 h.
- **Full-protocol eval (3,200 episodes, `eval_results/amp_v5.json`):**
  falls 0.75% | gait gate **0/5** — stance duty L 99.2 / R 90.9% (feet
  essentially never swing; the opposite sign of amp_v4's 0.7/0.4%) |
  ref RMS 12.13 deg | jerk 1.26 | action std 0.31 | lin vel err 0.118
  aggregate but **0.210 in the forward condition** (~ the full 0.2 m/s
  command: no forward translation; backward/lateral likewise untracked) |
  ang vel err 0.059 (turn condition 0.026 — the only tracked command) |
  energy 3.62 W (lowest of any policy — near-static).
- **Video audit (mandatory; deterministic forward vx=0.2 + turn wz=0.3,
  `videos/play/play_forward_vx0.2.mp4` / `play_turn_wz0.3.mp4`):** posture
  is upright at proper height — **the crawl is gone** — but under the
  forward command the duck stands in place for the full 20 s (no steps, no
  swing, feet planted); under the turn command it pivots in place.
  Checklist: upright PASS; alternating swing FAIL; gait stance FAIL (feet
  loaded ~99%); heading PASS; no-dither PASS. Verdict: upright
  stand-and-pivot, not walking.
- **Verdict: FAIL vs the acceptance bar (gate 0/5), and the hypothesis
  test is decisive in the informative sense pre-registered:** the crawl
  was indeed reachability-dependent (excised by the floor), yet no walker
  emerges — the policy reverts to the next-safest non-walking optimum.
  Root-cause #2 (termination geometry) is real but **not sufficient**;
  root-cause #1 (reference-data physics -> no usable style gradient) is
  now the primary suspect, consistent with the 2026-07-10 data forensics
  (20 exploitable constant dims measured in the synthetic clips' AMP
  features, cross-clip; plus per-clip-constant root z and wz).
- **Paper relevance (H2/H3):** fourth distinct degenerate optimum from
  the same learned-reward setup (stand -> in-place march -> shuffle ->
  crawl -> stand again as geometry changes): adjusting task/termination
  geometry relocates the exploit rather than eliminating it — the binding
  constraint is the data. Run 14 (`amp_v6`, reference clips recorded from
  deterministic PPO v3 rollouts, physically consistent by construction)
  is the decisive test.

## Run 14 — `amp_v6` (`2026-07-11_03-16-59_amp_torch`) — the data-physics test

- **Lever (single, the decisive test of root-cause #1):** motion library
  swapped to 22 physically-consistent clips recorded from deterministic
  ppo_v3 rollouts (`scripts/record_ppo_rollouts_to_amp.py`;
  `amp/motions_ppo/VALIDATION.md` = PASS, **0 data-dead AMP dims vs the
  synthetic set's 20**, root z live 0.164-0.177 m). Termination back at the
  0.10 default so the data is the only lever. Overrides
  (`env.motion_files=[.../motions_ppo/*.npz]`) verified in the run's
  dumped `params/env.yaml`.
- **Training (TB, last-100 means):** instantaneous reward 0.34 -> 0.478;
  episode length -> 894.6/1000; **discriminator loss 1.08 -> 1.09 last-100
  (min 0.315 mid-run, 1.36 at budget end) — engaged for the entire run and
  never saturated. With honest reference data the adversarial game is
  finally played on style rather than on physics tells** — the first AMP
  run of the campaign with a healthy discriminator start-to-finish. Policy
  std fixed 0.055; 2.57 h.
- **Full-protocol eval (3,200 episodes, `eval_results/amp_v6.json`):**
  falls 4.19% | gait gate **0/5** — stance duty L 99.0 / R 96.7% | ref RMS
  10.15 deg | jerk 0.088 (PPO-grade smoothness — 165x below amp_v1) |
  action std 0.157 | lin vel err 0.115 aggregate, **0.201 in the forward
  condition (~ the full command: no translation)** | ang vel err 0.058
  (turn 0.032) | energy 3.56 W (near-static).
- **Video audit (forward vx=0.2 + turn wz=0.3,
  `videos/play/play_forward_vx0.2.mp4` / `play_turn_wz0.3.mp4`):** upright
  at proper height, stands in place for the full 20 s under the forward
  command; pivots in place under the turn command. Checklist: upright
  PASS; alternating swing FAIL; gait stance FAIL; heading PASS; no-dither
  PASS. Same behavioral endpoint as run 13, now on honest data.
- **Post-eval diagnosis (pre-registered branch D: diagnose before
  retraining):** (a) the data-composition hypothesis — "the nine vx=0
  clips contain standing, legitimizing it" — is **REFUTED** by direct
  measurement: every clip including all vx=0 ones contains genuine
  stepping (leg-velocity RMS 1.5-2.5 rad/s, foot clearance 2.3-3.5 cm,
  base speed 0.15-0.29 m/s). The reference contains no standing to
  imitate. (b) Discriminator ENGAGED, not saturated -> the indicated
  constraint is the **exploration ceiling**: policy log-std has been
  pinned at sigma=0.055 for all eleven AMP runs, while the PPO baseline
  trains with initial noise std 0.5 (adaptive) — a 9x exploration
  asymmetry between the study's two arms that was never equalized.
  Discovering coordinated stepping from damped standing under fall risk +
  action-rate penalty with 0.055-sigma noise is the remaining untested
  bottleneck.
- **Verdict: FAIL vs the acceptance bar (gate 0/5), and root-cause #1
  resolves as necessary-but-not-sufficient:** honest data healed the
  adversarial game (engaged discriminator, zero exploitable tells, no
  crawl, no collapse) but did not by itself produce locomotion. The
  campaign's last untested lever — exploration (#4) — is promoted.
- **Run 15 lever (pre-registered):** `agent.models.policy.initial_log_std`
  -2.9 -> -1.9 (sigma 0.055 -> 0.15), everything else identical to this
  run (recorded clips, termination 0.10, task/style 0.6/0.4, 14-frame
  window, action clip +/-5).
- **Paper relevance (H2/H3):** with a healthy discriminator the learned
  reward still provides no curriculum from standing toward stepping at
  low exploration, whereas PPO's dense per-joint tracking reward does —
  and the two arms' exploration budgets were silently unequal (0.5 vs
  0.055) via framework defaults, itself a finding about hidden
  configuration asymmetries in method comparisons.

## Campaign status — 2026-07-06 (post-audit synthesis)

Consolidated picture after the video audit and the gait-validity gate; this
section supersedes the per-run verdicts above where they conflict and the
lever ladder below. Sources: eval_results/*.json (full protocol),
algorithm_comparison.md (gate columns), the four rollout videos, and the
en665.645 midpoint paper.

**Bottom line: the AMP campaign underperformed expectations — no AMP run
produced an acceptable walking gait.** Per policy (gate = gait-valid
conditions out of 5; falls over 3,200 episodes):

| Policy | Gate | Falls | Defining defect |
|---|---|---|---|
| amp_v1 (r8) | 2/5 | 0.19% | only AMP policy that walks; micro-shuffle (ROM ~2-8° vs ref 27-43°), jerk 14.5 (~210x ppo_v3), one-foot dragging in backward/lateral/turn conditions, 2x compute budget |
| amp_v2 (r9) | 4/5 | 2.4% | one-legged stride (ROM ratio 3.67) |
| amp_v3 (r10) | 5/5 | 5.8% | bilateral amplitude but style-dominated; walks in place (vel err 0.120) |
| amp_v4 (r12) | 0/5 | 1.3% | **crawls** (video audit); passed every aggregate quality metric while doing so |

PPO v3 (gate 5/5, 0 falls, ref RMS 4.59°, energy 21.6 W) remains the only
deployment-quality policy and the Task 2.5/2.6 selection.

**Hypothesis status (for the paper):** H1 (AMP smoother at comparable
stability) — not supported: the only stability-comparable AMP policy
(amp_v1, fall-rate CI overlapping PPO's) has the highest jerk. H2 (less
reward engineering) — nuanced: the imitation term disappeared but
engineering reappeared as the data-consistency fix (run 5), task-kernel
hardening (run 7), and action bounding (run 12), and the end state can
still fail silently (run 12's crawl). H3 (PPO more stable/faster) —
supported: 2/2 PPO runs converged first try; 9 AMP runs yielded 1
divergence, one 2x-budget resume, and zero acceptable gaits.

**Root-cause hypotheses, ranked by evidence:**
1. **Reference-data physics.** The synthetic polynomial gaits are
   kinematically plausible but dynamically imperfect (finite-difference
   velocities, limit-clamped joints, corrupt angular-velocity dims, 27
   samples/cycle). A discriminator can separate real-vs-fake on those
   inconsistencies, so "match the style" stops implying "walk naturally".
   Run 4's collapse was the loud version of this failure; the shuffle/crawl
   optima may be its quiet versions.
2. **Task/termination geometry.** Termination at base z < 0.10 m vs a
   0.17 m nominal height leaves a reachable crawl equilibrium (run 12
   lives in that gap); the task reward has no upright/posture term.
3. **Receptive-field/weight coupling.** Runs 8/9/10 show gait structure
   requires a >=half-cycle window, but at any fixed task/style weight the
   long window either drowns the task (run 10) or, rebalanced, finds the
   crawl (run 12).
4. **Exploration ceiling.** Policy log-std fixed at 0.055 for every run;
   this lever was never pulled.

### Forward plan (adopted 2026-07-06)

> **STATUS 2026-07-11:** Phase-2 item 4 (amp_v5 / run 13) is COMPLETE —
> crawl excised but reverts to stand-and-pivot, gate 0/5 (see Run 13);
> root-cause #2 tested and insufficient alone. Item 5 (amp_v6 / run 14)
> in progress: the rollout-recorder (`scripts/record_ppo_rollouts_to_amp.py`)
> is authored, its Isaac-free validator verified against the synthetic
> clips (correctly fails them with 20 exploitable dead AMP dims), and clip
> recording is next on the GPU. Phase-1 statistics items remain open.

Principle: no open-ended gait-chasing. Every further run must test a named
root-cause hypothesis and be informative in BOTH outcomes; the paper's
negative-result story is already complete without them.

**Phase 1 — evidence work, no new training (~1-2 days, fills the midpoint
paper's committed placeholders):**
1. Extend `evaluate_policies.py` with a per-episode dump and posture
   metrics (mean base height, trunk orientation) as a second,
   contact-independent validity axis; re-run the protocol on the six
   existing checkpoints (minutes each on GPU). Unlocks bootstrap CIs and
   significance tests for H1/H3.
2. Convergence analysis (H3) from the existing TensorBoard logs — no GPU.
3. Filmstrip figures from the four existing rollout videos.

**Phase 2 — two hypothesis-driven AMP runs (one overnight each, ~2.8 h
train + ~40 min eval; ONE Isaac job at a time):**
4. **amp_v5 / run 13 — tests root-cause #2 (termination geometry).**
   Run-12 config, ONE lever: termination height 0.10 -> ~0.13 m (or an
   upright-posture term) to excise the crawl equilibrium. Walker emerges =
   crawl was a reachability artifact; reverts to in-place/shuffle = the
   problem sits deeper.
5. **amp_v6 / run 14 — tests root-cause #1 (reference-data physics); the
   decisive experiment.** Author a rollout-recorder that emits
   MotionLoader `.npz` clips from deterministic PPO v3 rollouts across the
   command grid (~1 day), then retrain the same AMP config on the
   physically-consistent clips. Good gait = failure pinned on the
   synthetic polynomial data; still fails = adversarial imitation shown
   hard on this platform even with clean demonstrations.
6. Optional run 15: unfix log-std (0.055 -> ~0.15) ONLY if v5 or v6 shows
   a partial improvement worth amplifying.

**Acceptance bar for any new AMP policy** (all three, else it is another
data point, not a candidate): gait gate >= 4/5 conditions, fall rate < 1%
over the full protocol, and an upright walking gait on the video audit.

**Phase 3 — final paper:** integrate Phase-1 statistics and the v5/v6
outcomes, settle H1/H2/H3 verdicts, add filmstrips next to the video
links.

Expectation on record: matching ppo_v3's overall quality is unlikely in
this timeline; the prize is evidence, not a deployable AMP policy. PPO v3
stays the deployment selection — ONNX export and Isaac Sim validation
(Tasks 2.6/2.7) proceed independently of all of the above.

## Planned lever ladder (one per run, in order, each ~2 h)

> **STATUS 2026-07-06:** historical. Levers 1-2 were consumed by runs 5-12;
> the ladder is superseded by the campaign-status section above (its item 5
> "fallback of last resort" is now promoted to next-candidate #2).

1. ~~Reference data consistency~~ (run 5 — changed the failure mode from
   freezing to attempting motion, but did not produce striding)
2. **AMENDED 2026-06-12, reordered ahead of gradient-penalty:** Phase B
   (task 0.5/style 0.5) promoted to next run. Rationale: run 5 shows the
   policy now *attempts* locomotion; what's missing is a gradient toward
   walking that survives discriminator saturation — exactly what the task
   reward provides. Pure-style levers continue afterward as ablations.
3. Discriminator gradient penalty 5 -> 10 (if saturation persists)
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
