# NVIDIA Stack

The entire pipeline uses NVIDIA tools. Here's what each tool does and how it fits.

## Stack Map

| Stage | Tool | Replaces |
|---|---|---|
| Physics simulation | **Isaac Sim** (PhysX 5) | MuJoCo |
| RL training framework | **Isaac Lab** | MuJoCo Playground + Stable-Baselines3 |
| RL algorithms | **RSL-RL** (PPO) + **SKRL** (PPO, AMP) | SB3 |
| Robot model format | **USD** (converted from URDF) | MJCF (.xml) |
| Training hardware | **DGX Spark** (Grace Blackwell, 1 PFLOP FP4) | Single GPU |
| Policy deployment | **TensorRT** on Jetson GPU | onnxruntime on Pi CPU |
| Physical AI reasoning | **Cosmos Reason2-2B** (W4A16) | None (new capability) |
| VLM serving | **vLLM** (Jetson-optimized Docker) | N/A |
| On-robot hardware | **Jetson Orin Nano Super** | Raspberry Pi Zero 2W |

## Isaac Sim

NVIDIA's physics simulator built on Omniverse. Uses PhysX 5 for GPU-accelerated rigid body simulation. Can run thousands of robot instances in parallel.

- Docs: https://docs.isaacsim.omniverse.nvidia.com
- Has built-in URDF and MJCF importers to convert to USD format
- The robot model lives at `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`

## Isaac Lab

RL training framework built on Isaac Sim. Provides environments, reward utilities, domain randomization, and integrations with RSL-RL and SKRL.

- Docs: https://isaac-sim.github.io/IsaacLab
- GitHub: https://github.com/isaac-sim/IsaacLab
- Our env config: `isaac_lab_env/open_duck_mini_v2/env_cfg.py`

## RSL-RL

Lightweight PPO implementation optimized for GPU parallel training. Default in Isaac Lab for locomotion.

- Used for PPO training with 4096 parallel envs

## SKRL

Modular RL library with the widest algorithm support in Isaac Lab. Only library with AMP (Adversarial Motion Priors) support.

- GitHub: https://github.com/Toni-SM/skrl
- Used for: AMP training (Isaac Lab ships no off-policy skrl task configs; only PPO/AMP + multi-agent MAPPO/IPPO exist)

## TensorRT

NVIDIA's inference optimizer. Converts ONNX models to optimized GPU engines for Jetson.

- Convert: `trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16`
- Locomotion policy inference: <1 ms on Jetson
- Docs: https://developer.nvidia.com/tensorrt

## Cosmos Reason2-2B

NVIDIA's physical AI reasoning VLM. Understands spatial relationships, physics, and can plan robot actions.

- Model: `embedl/Cosmos-Reason2-2B-W4A16-Edge2` (INT4 quantized for edge)
- Runs on Jetson Orin Nano Super at ~5.8 GB RAM, ~16-17 tok/s
- Served via vLLM Jetson Docker: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`
- Outputs text (reasoning + JSON velocity commands), NOT direct actions
- GitHub: https://github.com/nvidia-cosmos/cosmos-reason2

## DGX Spark

Training workstation with Grace Blackwell chip. 128 GB unified memory, up to 1 PFLOP FP4.

- All NVIDIA software pre-installed
- Used for: Isaac Lab RL training (Phase 2), VLM fine-tuning (if needed)
- NOT used on the robot — training only
