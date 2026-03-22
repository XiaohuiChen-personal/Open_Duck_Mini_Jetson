# Open Duck Mini v2 — Jetson Edition

<table>
  <tr>
    <td> <img src="https://github.com/user-attachments/assets/2a407765-70ad-48dd-8a5d-488f82503716" alt="1" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/3b8fe350-73a9-4c9f-ad29-efc781be7aee" alt="2" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/fd7e5949-1492-4d31-851f-feaa9b695557" alt="3" width="300px" ></td>
   </tr>
</table>

A miniature bipedal BDX Droid by Disney, about 42 cm tall. This fork replaces the Raspberry Pi Zero 2W with an **NVIDIA Jetson Orin Nano Super** and migrates the entire simulation, training, and deployment pipeline to the **NVIDIA stack** (Isaac Sim, Isaac Lab, TensorRT, Cosmos Reason2).

> Based on the original [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) by Antoine Pirrone.

## What's Different in This Fork

| | Original | Jetson Edition |
|---|---|---|
| **Onboard computer** | Raspberry Pi Zero 2W (no GPU) | NVIDIA Jetson Orin Nano Super (67 TOPS) |
| **Computer location** | Head | Trunk (relocated for better CoG) |
| **Simulation** | MuJoCo | NVIDIA Isaac Sim (PhysX 5) |
| **RL training** | MuJoCo Playground + SB3 | NVIDIA Isaac Lab (RSL-RL + SKRL) |
| **Training hardware** | Single GPU | NVIDIA DGX Spark |
| **Policy deployment** | ONNX on CPU | TensorRT on Jetson GPU (<1 ms) |
| **Physical AI** | None | Cosmos Reason2-2B (vision + language + reasoning) |
| **Camera** | None | CSI camera (IMX219) in head |

## Architecture

```
Voice / Text command: "Walk to the red cup"
         │
         v
┌─────────────────────────────────────────────────┐
│  Cosmos Reason2-2B (VLM)           ~2-3 Hz      │
│  Sees camera, reasons about scene,              │
│  outputs velocity commands                       │
└────────────────────┬────────────────────────────┘
                     │  (vx, vy, yaw_rate)
                     v
┌─────────────────────────────────────────────────┐
│  Locomotion Policy (PPO/AMP)       50 Hz        │
│  Trained in Isaac Lab, runs via TensorRT        │
│  Outputs 16 joint position targets              │
└────────────────────┬────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────┐
│  Feetech STS3215 Servos (14x)     200+ Hz      │
└─────────────────────────────────────────────────┘
```

## State of Sim2Real

<!-- TODO: Add videos of Jetson-modified robot walking -->
<!-- Placeholder: sim2real videos will be added after Phase 4 (hardware build) -->

Original sim2real results (Pi Zero version):

https://github.com/user-attachments/assets/58721d0f-2f95-4088-8900-a5d02f41bba7

https://github.com/user-attachments/assets/4129974a-9d97-4651-9474-c078043bb182

## Task Plan

The full modification is documented in a 5-phase, 28-task plan: **[docs/jetson-mod/task_plan.md](docs/jetson-mod/task_plan.md)**

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Simulation model update (mass/inertia for Jetson in trunk) | Pending |
| Phase 2 | Isaac Lab setup + multi-algorithm RL training on DGX Spark | Pending |
| Phase 3 | CAD redesign of trunk/body parts | Pending |
| Phase 4 | Hardware build, assembly, real-robot walking | Pending |
| Phase 5 | Cosmos Reason2 VLM integration for autonomous behavior | Pending |

## NVIDIA Stack

| Tool | Purpose |
|---|---|
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com) | Physics simulation (PhysX 5, GPU-accelerated) |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab) | RL training framework |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO implementation for locomotion |
| [SKRL](https://github.com/Toni-SM/skrl) | SAC, AMP, RPO, TRPO, TD3 algorithms |
| [TensorRT](https://developer.nvidia.com/tensorrt) | On-device policy inference (<1 ms) |
| [Cosmos Reason2](https://github.com/nvidia-cosmos/cosmos-reason2) | Physical AI reasoning VLM |
| [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | Training hardware (Grace Blackwell) |

## RL Algorithms

We train and compare multiple RL algorithms to find the best walking gait:

| Algorithm | Framework | Type | Why |
|---|---|---|---|
| **PPO** | RSL-RL | On-policy | Proven baseline for locomotion |
| **RPO** | SKRL | On-policy | PPO + random perturbation, outperforms PPO in 93% of envs |
| **AMP** | SKRL | Imitation | Adversarial Motion Priors for natural-looking gaits |
| **SAC** | SKRL | Off-policy | Sample-efficient, potentially smoother gaits |
| **TRPO** | SKRL | On-policy | Conservative updates, stable convergence |

<!-- TODO: Add algorithm comparison results table after Phase 2 -->
<!-- See docs/jetson-mod/algorithm_comparison.md when available -->

### Reference Motion Generation

For AMP (imitation learning), reference walking motions are generated using [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator).

### Actuator Identification

Motor parameters identified using Rhoban's [BAM](https://github.com/Rhoban/bam). Results in `experiments/v2/params_m6.json`.

## Hardware

### BOM

Original BOM (Pi Zero version): https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing

Additional parts for Jetson modification:

| Item | Qty | Est. Cost |
|---|---|---|
| NVIDIA Jetson Orin Nano Super Developer Kit | 1 | $249 |
| 18650 Li-ion cells (e.g., Samsung 30Q) | 2 | $15 |
| DC-DC boost converter (7.4V to 19V, 3A+) | 1 | $12 |
| CSI camera module (IMX219) | 1 | $15 |
| CSI ribbon cable 30cm | 1 | $5 |
| M3 standoffs + screws (assorted) | 1 set | $8 |

**Additional cost for Jetson mod: ~$309**

### CAD

Original CAD: https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05

Modified parts (Phase 3): `trunk_top`, `trunk_bottom`, `body_back`, `body_middle_top/bottom`, `battery_pack_lid`

<!-- TODO: Link to modified CAD/STL files after Phase 3 -->

## Build Guide

### Original Build (unmodified robot)

- Tnkr guide: https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home
- [Print guide](docs/print_guide.md)
- [Assembly guide (incomplete)](docs/assembly_guide.md)

### Jetson Modification Build

<!-- TODO: Add Jetson-specific build guide after Phase 4 -->

See [docs/jetson-mod/task_plan.md](docs/jetson-mod/task_plan.md) — Phase 3 (CAD) and Phase 4 (assembly) for details.

## Deployment

### On-Robot Runtime (Jetson)

<!-- TODO: Add runtime setup instructions after Phase 4 -->

The Jetson runtime code lives in `jetson_runtime/`:
- `trt_infer.py` — TensorRT locomotion policy inference
- `cosmos_commander.py` — Cosmos Reason2 VLM interface
- `autonomous_walk.py` — Main control loop (Cosmos + locomotion)
- `walk_controller.py` — Manual walk control (keyboard/gamepad)
- `prompts/` — Cosmos prompt library for different behaviors

```bash
# Start Cosmos Reason2 server
docker run --rm -it --runtime=nvidia --network host --shm-size=4g \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve "embedl/Cosmos-Reason2-2B-W4A16-Edge2" \
    --max-model-len 2048 --gpu-memory-utilization 0.70 --max-num-seqs 1

# Run autonomous walking
python jetson_runtime/autonomous_walk.py "Walk to the door"

# Run manual walking (keyboard control)
python jetson_runtime/walk_controller.py
```

### Legacy Runtime (Pi Zero)

The original Pi Zero runtime is in a separate repo: https://github.com/apirrone/Open_Duck_Mini_Runtime

### Training Your Own Policies

Training uses Isaac Lab on a DGX Spark (or any NVIDIA GPU with Isaac Sim installed):

```bash
# PPO (baseline)
python -m isaaclab.train --task OpenDuckLocomotion-v0 --headless --num_envs 4096

# AMP (imitation learning, best gait quality)
python -m isaaclab.train --task OpenDuckLocomotion-v0 --algorithm AMP --headless --num_envs 4096

# SAC (off-policy)
python -m isaaclab.train --task OpenDuckLocomotion-v0 --algorithm SAC --headless --num_envs 512
```

See [docs/sim2real.md](docs/sim2real.md) for the original MuJoCo-based sim2real guide (for reference).

## Repository Structure

```
Open_Duck_Mini_Jetson/
├── mini_bdx/                         # Robot models and Python package
│   ├── robots/open_duck_mini_v2/     # MJCF, URDF, USD, STL meshes
│   └── mini_bdx/utils/               # MuJoCo utilities, joint mapping, action scaling
├── isaac_lab_env/                    # Isaac Lab RL environment (Phase 2)
│   └── open_duck_mini_v2/            # Env config, training configs, evaluation
├── jetson_runtime/                   # Jetson deployment code (Phase 4-5)
│   ├── trt_infer.py                  # TensorRT inference
│   ├── cosmos_commander.py           # Cosmos Reason2 interface
│   ├── autonomous_walk.py            # VLM + locomotion control loop
│   └── prompts/                      # Cosmos prompt library
├── exported_policies/                # Trained ONNX + TensorRT policies (Phase 2)
├── experiments/                      # Legacy MuJoCo experiment scripts
├── print/                            # 3D printable STL files
├── docs/
│   ├── jetson-mod/                   # Jetson modification docs
│   │   └── task_plan.md              # Full 5-phase, 28-task plan
│   ├── assembly_guide.md
│   ├── sim2real.md
│   └── print_guide.md
├── tests/                            # Automated test suite (Phase 1-5)
├── CLAUDE.md                         # Claude Code context
└── .cursor/rules/                    # Cursor IDE context
```

## Community

This fork builds on the amazing work of the Open Duck Mini community.

![duck_collage](https://github.com/user-attachments/assets/e240c06e-769f-4c87-b65f-189a442cf1e9)

Join the discord: https://discord.gg/UtJZsgfQGe

## Acknowledgments

- [Antoine Pirrone](https://github.com/apirrone) and the Open Duck Mini community for the original robot design
- [HuggingFace](https://huggingface.co/) and [Pollen Robotics](https://www.pollen-robotics.com/) for sponsoring the original project
- [NVIDIA](https://developer.nvidia.com/isaac) for Isaac Sim, Isaac Lab, Cosmos, and the Jetson platform
- [Rhoban](https://github.com/Rhoban) for the BAM actuator identification tool
- [Disney Research](https://la.disneyresearch.com/) for the original BDX droid design and paper
