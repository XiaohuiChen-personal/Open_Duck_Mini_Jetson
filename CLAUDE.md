# Open Duck Mini — Jetson Modification

This is a fork of the [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) bipedal robot project. We are modifying the robot to replace the Raspberry Pi Zero 2W with an NVIDIA Jetson Orin Nano Super Developer Kit, and migrating the entire simulation/training/deployment pipeline to the NVIDIA stack.

## Quick Reference

- **Robot:** Open Duck Mini v2, ~42cm tall bipedal duck, 14x Feetech STS3250 servos, ~2.66 kg (after mod)
- **Onboard computer:** NVIDIA Jetson Orin Nano Super (8 GB, 67 TOPS) — relocated from head to trunk
- **Training hardware:** NVIDIA DGX Spark (Grace Blackwell)
- **Simulation:** NVIDIA Isaac Sim (PhysX 5) — replacing MuJoCo
- **RL framework:** NVIDIA Isaac Lab with RSL-RL (PPO) and SKRL (SAC, AMP, RPO, TRPO)
- **On-device AI:** Cosmos Reason2-2B (W4A16 quantized) for physical AI reasoning + TensorRT locomotion policy
- **Task plan:** See `docs/jetson-mod/task_plan.md` for the full 5-phase, 31-task implementation plan
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

## Key Directories

- `mini_bdx/robots/open_duck_mini_v2/` — Robot model files (MJCF, URDF, USD, STL meshes)
- `isaac_lab_env/` — Isaac Lab RL environment definitions and training configs
- `jetson_runtime/` — Jetson deployment code (TensorRT, Cosmos Reason2, GPIO)
- `exported_policies/` — Trained ONNX and TensorRT policy files
- `experiments/` — Legacy MuJoCo-based experiment scripts (reference only)
- `docs/jetson-mod/` — Modification documentation and task plan
- `tests/` — Automated test suite
