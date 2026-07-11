# Open Duck Mini — Jetson Modification

This is a fork of the [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) bipedal robot project. We are modifying the robot to replace the Raspberry Pi Zero 2W with an NVIDIA Jetson Orin Nano Super Developer Kit, and migrating the entire simulation/training/deployment pipeline to the NVIDIA stack.

## Quick Reference

- **Robot:** Open Duck Mini v2, ~42cm tall bipedal duck, 14x Feetech STS3250 servos, ~2.75 kg (after mod)
- **Onboard computer:** NVIDIA Jetson Orin Nano Super (8 GB, 67 TOPS) — relocated from head to trunk
- **Training hardware:** NVIDIA DGX Spark (Grace Blackwell)
- **Simulation:** NVIDIA Isaac Sim (PhysX 5) — replacing MuJoCo
- **RL framework:** NVIDIA Isaac Lab with RSL-RL (PPO) and SKRL (SAC, AMP, RPO, TRPO)
- **On-device AI:** Cosmos Reason2-2B (W4A16 quantized) for physical AI reasoning + TensorRT locomotion policy
- **Task plan:** See `docs/jetson-mod/task_plan.md` for the full 5-phase, 28-task implementation plan
- **Experiment journal:** EVERY training run gets an entry in `docs/jetson-mod/experiment_journal.md` (a paper artifact). Data-sourcing protocol in `.claude/rules/experiment-journal.md` — last-100 TensorBoard means (never single-iteration log samples), measured gate rollouts, every number names its source.

## Build & Test

```bash
# Install the package
pip install -e ".[all]"

# Run MuJoCo model viewer (original model)
python3 -m mujoco.viewer --mjcf=mini_bdx/robots/open_duck_mini_v2/scene.xml

# Run tests
pytest tests/ -v

# Run tests by phase
pytest -m "phase1" -v   # Model integrity
pytest -m "phase2" -v   # Isaac Lab / training
pytest -m "phase5" -v   # Cosmos VLM integration
```

## Locomotion Policy Evaluation Protocol (mandatory, all future policies)

Every trained locomotion policy — PPO, AMP, or any future method — gets the same
three-step validation before any verdict lands in the journal. Canonical details:
`scripts/evaluate_policies.py` docstring and `docs/jetson-mod/algorithm_comparison.md`
("Metric hierarchy").

1. **Quantitative eval (3,200 episodes).** `scripts/evaluate_policies.py`: 5 command
   conditions x 10 windows x 64 envs, 30 s episodes, deterministic, seed 42, obs
   corruption and pushes disabled. Emits one JSON per policy into
   `docs/jetson-mod/eval_results/` with metrics 1-9. Metric 9 is the gait-validity
   gate: BOTH feet's stance duty inside [40, 90]% per condition (`GAIT_DUTY_BAND_PCT`
   is the single source of truth). `--report-only` regenerates
   `algorithm_comparison.md` from archived JSONs without Isaac.
2. **Video audit (mandatory — aggregate metrics alone are NOT sufficient).** Render
   deterministic rollout mp4s (robot-tracking camera, ~20 s, seed 42) in TWO
   conditions: fixed forward vx=0.2 AND turn wz=0.3 — defects like one-foot dragging
   only show off-forward. PPO: `scripts/play_policy.py`. AMP: no committed wrapper
   yet — recreate one (camera-enabled AMP play env; overrides must match the
   checkpoint's `num_amp_observations` / `action_clip`) before the next AMP run.
   Extract a filmstrip (ffmpeg) and check against the fixed checklist: trunk upright
   near 0.17 m; both feet alternate swing with real ground clearance; feet loaded
   during stance (no drag / glide / crawl); heading straight; no action dither.
   Store the mp4s in the run's log dir (`videos/play/`) and reference them from the
   journal entry.
3. **Verdict + journal.** A policy is acceptable only if ALL THREE hold: gait gate
   >= 4/5 conditions, fall rate < 1% over the full protocol, and upright walking on
   the video audit. **When metrics and video disagree, the video wins** — run 12
   (amp_v4) passed every aggregate metric while crawling; only the video caught it.
   The journal entry records both the metric row and the video verdict.

Rules:
- Evaluate on the SAME robot model/USD the policy was trained on (the PPO-vs-AMP
  study runs use the v2-branch model); cross-model numbers are not comparable.
- Runs that fail their gate get forensic-rollout numbers (`scripts/measure_amp_gait.py`)
  cited as diagnostic only — never as protocol-comparable results.
- Planned extension (forward-plan Phase 1): per-episode dumps + posture metrics
  (mean base height, trunk orientation) join the eval JSON as the quantitative twin
  of the video checklist.

## Key Directories

- `mini_bdx/robots/open_duck_mini_v2/` — Robot model files (MJCF, URDF, USD, STL meshes)
- `isaac_lab_env/` — Isaac Lab RL environment definitions and training configs
- `jetson_runtime/` — Jetson deployment code (TensorRT, Cosmos Reason2, GPIO)
- `exported_policies/` — Trained ONNX and TensorRT policy files
- `experiments/` — Legacy MuJoCo-based experiment scripts (reference only)
- `docs/jetson-mod/` — Modification documentation and task plan
- `tests/` — Automated test suite
