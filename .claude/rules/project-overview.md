# Project Overview

This project is modifying the Open Duck Mini v2 bipedal robot to use NVIDIA hardware and software throughout.

## What Changed and Why

The original robot uses a Raspberry Pi Zero 2W (65x30x5mm, 10g, no GPU) in the head. We are replacing it with a Jetson Orin Nano Super Dev Kit (103x90.5x34.77mm incl. heatsink+fan, 176g, 67 TOPS GPU) relocated to the trunk/body cavity.

**Why:** To add physical AI capabilities (vision, language understanding, autonomous navigation) and learn the NVIDIA robotics stack (Isaac Sim, Isaac Lab, TensorRT, Cosmos).

**Approach:** Partial body redesign (4-6 printed parts), not scaling up the entire robot. The cube-square law makes scaling impractical (mass scales as L^3, torque needed as L^4).

## Architecture

```
Cosmos Reason2-2B (VLM, 2-3 Hz)     — "where should I go?" (text reasoning)
         │ velocity commands
         v
Locomotion Policy (PPO/AMP, 50 Hz)  — "how do I walk there?" (joint control)
         │ joint position targets
         v
Feetech STS3250 Servos (14x)        — physical motors
```

## Key Decisions Made

1. **Jetson Dev Kit over module + mini carrier board** — Dev Kit has CSI camera ports and 40-pin GPIO header needed for peripherals
2. **Full NVIDIA stack migration** — Isaac Sim/Lab replaces MuJoCo for simulation and training
3. **Cosmos Reason2-2B for VLM** — Confirmed running on Orin Nano Super at ~5.8 GB RAM, ~16 tok/s
4. **Multi-algorithm RL experiment** — PPO vs AMP compared in a 12-run campaign (experiment_journal.md); PPO v3 selected. SAC/RPO/TRPO were not run — skrl's Isaac Lab integration only has locomotion-ready PPO/AMP paths (see README note)
5. **Sim-first approach** — All changes validated in simulation before hardware purchases

## Current Phase

Check `docs/jetson-mod/task_plan.md` for detailed progress. The plan has 5 phases:
- Phase 1: Simulation model update (mass/inertia for Jetson in trunk)
- Phase 2: Isaac Lab setup + multi-algorithm RL training on DGX Spark
- Phase 3: CAD redesign of trunk/body parts
- Phase 4: Hardware build and real-robot walking
- Phase 5: Cosmos Reason2 VLM integration for autonomous behavior
