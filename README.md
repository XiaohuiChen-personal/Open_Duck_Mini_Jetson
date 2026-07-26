# Open Duck Mini v2 — Jetson Edition

<table>
  <tr>
    <td> <img src="https://github.com/user-attachments/assets/2a407765-70ad-48dd-8a5d-488f82503716" alt="1" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/3b8fe350-73a9-4c9f-ad29-efc781be7aee" alt="2" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/fd7e5949-1492-4d31-851f-feaa9b695557" alt="3" width="300px" ></td>
   </tr>
</table>

A miniature bipedal BDX Droid by Disney, about 42 cm tall. This fork migrates the Open Duck Mini v2 from a Raspberry Pi / MuJoCo / Stable-Baselines3 pipeline to the **NVIDIA stack** (Isaac Sim, Isaac Lab, Jetson Orin Nano, TensorRT, Cosmos Reason2). The work is being done in 5 phases over a longer arc; see status table below for what's currently shipped versus planned.

> Based on the original [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) by Antoine Pirrone.

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Simulation model update (Jetson mass/inertia in trunk) | ✅ Complete |
| Phase 2 | Isaac Lab environment + PPO locomotion training on DGX Spark | ✅ Complete (PPO baseline + v4_robust DR retrain shipped, push-recovery gate passed; ONNX export remaining) |
| Phase 3 | CAD redesign of trunk / body / battery for Jetson cavity | 🟠 In progress (layout v2.1 + corrected inertials merged 2026-07-26; remaining 3.x tasks pending) |
| Phase 4 | Hardware build, TensorRT deployment, real-robot walking | 🟡 Planned |
| Phase 5 | Cosmos Reason2 VLM for vision-language-action control | 🟡 Planned |

Full task breakdown (5 phases, 31 tasks): **[docs/jetson-mod/task_plan.md](docs/jetson-mod/task_plan.md)**

---

## What's Working Today

A trained PPO locomotion policy in Isaac Lab using a **BDX-style composite imitation reward**, based on the Disney BDX paper *"Design and Control of a Bipedal Robotic Character"* (Jan 2025) and the Open Duck Playground reward structure. Three reward iterations shipped (v1, v2, and the current v3, which fixed the v2 gait-phase bug and passes the G1 gait gate), plus a 16-run PPO-vs-AMP comparison study (archived in [open-duck-ppo-vs-amp](https://github.com/XiaohuiChen-personal/open-duck-ppo-vs-amp)). After the CAD-driven model correction, the deployment candidate is **v4_robust** — PPO retrained on the corrected layout-v2.1 model with dynamics domain randomization and a hardware-realizable 59-dim observation (`docs/jetson-mod/v4_retrain_results.md`, `validation_results.md`).

**Training setup:**
- NVIDIA Isaac Lab + RSL-RL PPO on DGX Spark (Grace Blackwell)
- 4096 parallel environments, 50 Hz policy / 200 Hz physics
- MLP [512, 256, 128] with ELU, observation normalization enabled
- 3000 iterations, ~20s episodes

**Imitation reward design** (`isaac_lab_env/open_duck_mini_v2/imitation_reward.py`):
- Polynomial gait library: 240 reference motions × 40 dimensions × 16 polynomial coefficients
- Joint position tracking (raw quadratic, BDX weight 15.0)
- Joint velocity tracking (raw quadratic, BDX weight 0.001)
- Base velocity tracking (exponential, BDX weight 1.0)
- Foot contact matching (binary, BDX weight 1.0)

**v2 final results (iteration 3000):**

| Metric | Value |
|---|---|
| Mean reward | ~239 |
| Fall rate | **0.0%** at final iteration (down from 99.9% at iter 31) |
| Reference tracking | **3.80° RMS** error across 6 leg joints |
| Mean episode length | 1000 steps (full 20s, no early termination) |
| Action std | 0.07 (precise, confident policy) |

**v1 → v2 reward design improvements:**

| Metric | v1 | v2 | Improvement |
|---|---|---|---|
| Mean reward | 235 | 239 | +2% |
| Fall rate | 1.3% | **0.0%** | Eliminated |
| Flat orientation | -0.05 | **-0.008** | 6× less tilt |
| Action std | 0.48 | **0.07** | 7× more precise |
| Convergence | ~500 iter to plateau | ~200 iter to plateau | 2.5× faster |

**Full training report:** [`exported_policies/v2_bdx_imitation_ppo/README.md`](exported_policies/v2_bdx_imitation_ppo/README.md)

**Showcase videos** (training progression, untrained → iter 2999): [`showcase_videos/`](showcase_videos/)

---

## What's Different in This Fork

| | Original | Jetson Edition |
|---|---|---|
| **Onboard computer** | Raspberry Pi Zero 2W (no GPU) | NVIDIA Jetson Orin Nano Super (67 TOPS) — *planned* |
| **Computer location** | Head | Trunk (relocated for better CoG) — *physics updated, hardware planned* |
| **Simulation** | MuJoCo | NVIDIA Isaac Sim (PhysX 5) — ✅ migrated |
| **RL training** | MuJoCo Playground + SB3 | NVIDIA Isaac Lab (RSL-RL PPO) — ✅ migrated |
| **Training hardware** | Single GPU | NVIDIA DGX Spark — ✅ in use |
| **Policy deployment** | ONNX on CPU | TensorRT on Jetson GPU — *planned* |
| **Physical AI** | None | Cosmos Reason2-2B VLM — *planned* |
| **Camera** | None | CSI camera (IMX219) in head — *planned* |

---

## Target Architecture

The diagram below describes the **end-state system after all 5 phases are complete**, not the current implementation. Currently only the locomotion policy (Phase 2) is shipped.

```
Voice / Text command: "Walk to the red cup"
         │
         v
┌─────────────────────────────────────────────────┐
│  Cosmos Reason2-2B (VLM)           ~2-3 Hz      │
│  Sees camera, reasons about scene,              │   PLANNED (Phase 5)
│  outputs velocity commands                       │
└────────────────────┬────────────────────────────┘
                     │  (vx, vy, yaw_rate)
                     v
┌─────────────────────────────────────────────────┐
│  Locomotion Policy (PPO)           50 Hz        │
│  Trained in Isaac Lab                            │   ✅ TRAINED (Phase 2)
│  TensorRT deployment on Jetson                   │   PLANNED (Phase 4)
│  Outputs 16 joint position targets              │
└────────────────────┬────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────┐
│  Feetech STS3250 Servos (14x)     200+ Hz      │
└─────────────────────────────────────────────────┘
```

---

## State of Sim2Real

Original sim2real results from the Pi Zero version (reference, not this fork):

https://github.com/user-attachments/assets/58721d0f-2f95-4088-8900-a5d02f41bba7

https://github.com/user-attachments/assets/4129974a-9d97-4651-9474-c078043bb182

Sim2real videos for the Jetson edition will be added after Phase 4 (hardware build).

---

## NVIDIA Stack

| Tool | Purpose | Status in This Fork |
|---|---|---|
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com) | Physics simulation (PhysX 5, GPU-accelerated) | ✅ In use |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab) | RL training framework | ✅ In use |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO training + built-in ONNX export | ✅ In use (PPO) |
| [SKRL](https://github.com/Toni-SM/skrl) | AMP (Adversarial Motion Priors) | ✅ In use (AMP) |
| [TensorRT](https://developer.nvidia.com/tensorrt) | On-device policy inference | 🟡 Planned (Phase 4) |
| [Cosmos Reason2](https://github.com/nvidia-cosmos/cosmos-reason2) | Physical AI reasoning VLM | 🟡 Planned (Phase 5) |
| [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | Training hardware (Grace Blackwell) | ✅ In use |

---

## RL Algorithms

PPO is the primary algorithm; the course study selected ppo_v3, since superseded for deployment by v4_robust (PPO retrained on the corrected post-CAD model — `docs/jetson-mod/validation_results.md`). AMP is implemented (custom `DirectRLEnv` + skrl) and was trained head-to-head against PPO in a 16-run study — nine June runs failed in instructive ways; the July campaign's amp_v7 passed the acceptance bar (gait gate 4/5, 0.34% falls, best command tracking of the study). The full study is archived in [open-duck-ppo-vs-amp](https://github.com/XiaohuiChen-personal/open-duck-ppo-vs-amp) (tag `course-study-freeze` in this repo marks the freeze point).

| Algorithm | Framework | Type | Status | Why |
|---|---|---|---|---|
| **PPO** | RSL-RL | On-policy | ✅ Shipped | Proven baseline for locomotion. All Isaac Lab locomotion examples use it. Built-in ONNX export for Jetson. |
| **AMP** | SKRL | On-policy + imitation | ✅ Implemented (16-run study; amp_v7 passed acceptance) | Adversarial Motion Priors: discriminator-learned style reward. Custom DirectRLEnv (`isaac_lab_env/open_duck_mini_v2/amp/`) + reference motion clips. |

> **Note:** The Isaac Lab SKRL training script's `--algorithm` flag accepts any skrl algorithm name, but Isaac Lab ships per-task agent configs only for PPO and AMP (plus multi-agent MAPPO/IPPO on a few tasks). Off-policy algorithms (SAC, TD3, DDPG, etc.) ship zero task configs — using them means authoring configs from scratch with no locomotion examples in the Isaac Lab ecosystem.

### Reference Motion Generation

Two reference-motion libraries exist for AMP: (a) the polynomial gait library from [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator), converted via `scripts/convert_gait_library_to_amp.py`; (b) 22 physically-consistent clips recorded from deterministic ppo_v3 rollouts via `scripts/record_ppo_rollouts_to_amp.py` — the library that made AMP walk (runs 14-16).

### Actuator Identification

Motor parameters identified using BAM system identification. STS3250 parameters from [kscalelabs/sysid](https://github.com/kscalelabs/sysid) in `experiments/v2/params_sts3250_id008.json`. Legacy STS3215 parameters (Rhoban's [BAM](https://github.com/Rhoban/bam)) in `experiments/v2/params_m6.json`.

---

## Engineering Rigor

- **Test suite:** ~1,276 lines of pytest tests across 5 files covering model integrity, mass/inertia validation, CAD dimensions, USD conversion, and Isaac Lab environment correctness ([`tests/`](tests/))
- **Phase markers:** Tests are tagged by phase (`pytest -m "phase1"`, `pytest -m "phase2"`, etc.)
- **Training monitoring:** TensorBoard event log parser ([`scripts/monitor_training.py`](scripts/monitor_training.py))
- **MJCF→USD pipeline:** Headless converter using Isaac Lab's MjcfConverter API ([`scripts/convert_mjcf_to_usd.py`](scripts/convert_mjcf_to_usd.py))

---

## Hardware

### BOM

Original BOM (Pi Zero version): https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing

Additional parts for Jetson modification (planned for Phase 4):

| Item | Qty | Est. Cost |
|---|---|---|
| NVIDIA Jetson Orin Nano Super Developer Kit | 1 | $249 |
| Feetech STS3250 servos (replacing STS3215) | 14 | $280 |
| 18650 Li-ion cells (e.g., Samsung 30Q) | 6 | $45 |
| 3S BMS (>=15A) | 1 | $8 |
| DC-DC boost converter (11.1V to 19V, 3A+) | 1 | $12 |
| CSI camera module (IMX219) | 1 | $15 |
| CSI ribbon cable 30cm | 1 | $5 |
| M3 standoffs + screws (assorted) | 1 set | $8 |

**Additional cost for Jetson mod: ~$627**

### CAD

Original CAD: https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05

Modified parts (layout v2.1, merged 2026-07-26 — see `docs/jetson-mod/component_layout_v2.md`): `body_front`, `body_back`, `body_middle_bottom`, `trunk_bottom`, plus new `print/thermal_partition.stl`. Still open: `battery_pack_lid` redesign, battery retention tray, partition rails (Phase 3/4).

---

## Build Guide

### Original Build (unmodified robot)

- Tnkr guide: https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home
- [Print guide](docs/print_guide.md)
- [Assembly guide (incomplete)](docs/assembly_guide.md)

### Jetson Modification Build

The Jetson-specific build guide will be added after Phase 4 (hardware assembly). See [docs/jetson-mod/task_plan.md](docs/jetson-mod/task_plan.md) — Phase 3 (CAD) and Phase 4 (assembly) for the planned scope.

---

## Training Your Own Policies

Training uses Isaac Lab on a DGX Spark (or any NVIDIA GPU with Isaac Sim installed):

```bash
# PPO via RSL-RL (primary, currently shipped)
./isaaclab.sh -p scripts/train_ppo.py \
    --task Isaac-Velocity-Rough-OpenDuck-v0 \
    --headless --video --video_length 200 --video_interval 5000

# Evaluate a trained checkpoint
./isaaclab.sh -p scripts/play_policy.py \
    --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
    --num_envs 50 \
    --checkpoint exported_policies/v2_bdx_imitation_ppo/model_2999.pt \
    --headless --video --video_length 500
```

See [docs/sim2real.md](docs/sim2real.md) for the original MuJoCo-based sim2real guide (for reference).

---

## Planned: Jetson Runtime (Phase 4-5)

Once Phase 4 (hardware build + TensorRT deployment) and Phase 5 (Cosmos Reason2 integration) are complete, the on-robot runtime will live in `jetson_runtime/`. The planned components are:

- `trt_infer.py` — TensorRT locomotion policy inference (Phase 4)
- `walk_controller.py` — Manual walk control via keyboard/gamepad (Phase 4)
- `cosmos_commander.py` — Cosmos Reason2 VLM interface (Phase 5)
- `autonomous_walk.py` — Main control loop combining VLM + locomotion (Phase 5)
- `prompts/` — Cosmos prompt library for different behaviors (Phase 5)

The legacy Pi Zero runtime (for reference) lives in a separate repo: https://github.com/apirrone/Open_Duck_Mini_Runtime

---

## Repository Structure

```
Open_Duck_Mini_Jetson/
├── mini_bdx/                              # Robot models (MJCF, URDF, USD, STL)
│   ├── robots/open_duck_mini_v2/          # Robot definition files
│   └── mini_bdx/utils/                    # MuJoCo utilities, joint mapping
├── isaac_lab_env/                         # Isaac Lab RL environment ✅
│   └── open_duck_mini_v2/
│       ├── env_cfg.py                     # Locomotion environment config
│       ├── robot_cfg.py                   # Articulation + actuator config
│       ├── imitation_reward.py            # BDX-style composite reward
│       └── agents/rsl_rl_ppo_cfg.py       # PPO hyperparameters
├── scripts/                               # Training + utility scripts ✅
│   ├── convert_mjcf_to_usd.py             # MJCF → USD pipeline
│   ├── train_ppo.py                       # PPO training entry point
│   ├── play_policy.py                     # Policy evaluation
│   └── monitor_training.py                # TensorBoard log parser
├── exported_policies/                     # Trained policies ✅
│   ├── v1_imitation_ppo/                  # First reward iteration (historical)
│   ├── v2_bdx_imitation_ppo/              # BDX-aligned reward (historical)
│   ├── v3_bdx_imitation_ppo/              # v3: gait-phase fix (study winner)
│   └── amp_v*_run*_command/               # Archived AMP study policies
│       (deployment candidate v4_robust: checkpoint in ~/IsaacLab/logs, see
│        docs/jetson-mod/v4_retrain_results.md — not yet exported here)
├── showcase_videos/                       # Training progression videos ✅
├── tests/                                 # Pytest suite ✅
├── experiments/                           # Legacy MuJoCo experiment scripts
├── print/                                 # 3D printable STL files
├── docs/
│   ├── jetson-mod/
│   │   ├── task_plan.md                   # 5-phase, 31-task plan
│   │   └── mass_inertia_calculations.md   # Phase 1 physics math
│   ├── assembly_guide.md
│   ├── sim2real.md
│   └── print_guide.md
└── jetson_runtime/                        # Planned (Phases 4–5, not yet implemented)
```

---

## Community

This fork builds on the work of the Open Duck Mini community.

![duck_collage](https://github.com/user-attachments/assets/e240c06e-769f-4c87-b65f-189a442cf1e9)

Join the discord: https://discord.gg/UtJZsgfQGe

---

## Acknowledgments

- [Antoine Pirrone](https://github.com/apirrone) and the Open Duck Mini community for the original robot design and the open-source foundation this fork builds on
- [HuggingFace](https://huggingface.co/) and [Pollen Robotics](https://www.pollen-robotics.com/) for sponsoring the original project
- [NVIDIA](https://developer.nvidia.com/isaac) for Isaac Sim, Isaac Lab, Cosmos, and the Jetson platform
- [Rhoban](https://github.com/Rhoban) and [kscalelabs/sysid](https://github.com/kscalelabs/sysid) for BAM actuator identification
- [Disney Research](https://la.disneyresearch.com/) for the BDX droid design and the BDX paper that informed the v2 reward design
- [BDX-R Isaac Lab](https://github.com/KaydenKnapik/BDX-R-Isaaclab) for prior art in BDX-style Isaac Lab environments
