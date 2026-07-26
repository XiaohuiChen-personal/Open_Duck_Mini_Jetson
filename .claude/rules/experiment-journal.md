# Experiment Journal Protocol

Every training run MUST get an entry in `docs/jetson-mod/experiment_journal.md`
before the next run launches. The journal's numbers must be reproducible
from primary sources, never from memory or mid-training log greps. (The
EN.665.645 paper's copy of this journal was frozen 2026-07-26 into the
archive repo `open-duck-ppo-vs-amp` — tag `course-study-freeze`; entries here
from that point on serve the robot project's engineering record.)

## Data-sourcing rules (non-negotiable)

1. **Training metrics = last-100-iteration means from TensorBoard event
   files** (`EventAccumulator` with `size_guidance={'scalars': 0}`), with
   peaks noted separately. NEVER quote a single-iteration value from a
   console-log grep as a final result — this exact mistake corrupted the
   first journal draft (e.g. v3 "253.8 / +1.69" were single-iteration
   samples; the true last-100 means were 253.0 / +1.61).
2. **Wall-clock = first-to-last event timestamps** (or checkpoint file
   mtimes), never estimates and never reused from a different run.
3. **Behavioral metrics are measured, not inferred**: gait/gate numbers come
   from `scripts/evaluate_policies.py` (JSONs in the per-model results dir —
   `eval_results_v4/` for the corrected layout-v2.1 model; never mix robot
   models in one table; pre-CAD study JSONs are archive-repo only)
   or targeted diagnostic rollouts (study-era `measure_amp_gait.py` is in
   the archive repo). Reward curves alone cannot
   certify a gait (run 4 had healthy-looking episode lengths while standing
   still).
4. **Every number names its source** (TB tag, JSON field, file path/mtime)
   at least once per entry.
5. **Claims about data artifacts must be measured before journaling** (the
   "~8 rad/s" impossible-velocity claim was actually 25.3 rad/s max when
   measured).
6. **Verify a run has actually exited before evaluating or journaling it**:
   check the pidfile process group is dead AND the final checkpoint exists
   (`agent_<final_timestep>.pt` / `model_<final_iter>.pt`). Elapsed time is
   not completion; a fired watcher is the signal, not a guess. (A mid-training
   G2 measurement was once run against a live training under GPU contention
   because completion was assumed from wall-clock.)
7. **ONE Isaac Sim / GPU job at a time.** Never launch an evaluation
   (`evaluate_policies.py`, `play_policy.py`) while a
   training run is active — two Isaac Sim processes collide on GPU/kit
   resources during startup and the second dies in its init banner (run-10
   amp_v3 eval failed this way, exit 2, while run 11 was training). Evals run
   in the gap AFTER a run exits and BEFORE the next launches; batch deferred
   evals there. The queue runner enforces this for trainings — evals are
   launched by hand, so the human/agent must sequence them.

## Entry structure

Per run: one-lever config delta | training signals (TB, per rule 1) | gate
evaluation (per rule 3) | verdict with gate name | artifact paths (checkpoint,
ONNX, videos, log dir). Update the run-index table at the top of the journal.

## Log dirs

- PPO (RSL-RL): `~/IsaacLab/logs/rsl_rl/open_duck_ppo/<timestamp>/`
- PPO robust track (Run B+): `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/<timestamp>/`
- AMP (skrl, archived study era): `~/IsaacLab/logs/skrl/<experiment.directory>/<timestamp>_amp_torch/`
- Detached run console logs + PIDs: `.training_runs/<run_name>.{log,pid}`

## Companion documents

- Comparison tables are PER MODEL: corrected-model (v2.1) policies go to
  `v4_comparison.md` + `eval_results_v4/` (or a successor per-model pair for
  future models). Add every gate-passing policy to the table matching its
  robot model. (The pre-CAD study table `algorithm_comparison.md` +
  `eval_results/` was removed 2026-07-26 — it lives in the archive repo
  `open-duck-ppo-vs-amp` and at tag `course-study-freeze`.)
- `exported_policies/<name>/` — archive checkpoint + `agent.yaml` +
  `env.yaml` + README for any deployment-candidate policy.
