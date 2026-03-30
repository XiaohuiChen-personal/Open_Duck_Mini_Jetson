# Jetson Orin Nano Dev Kit Modification — Task Plan

## Overview

This document describes the full plan to replace the Raspberry Pi Zero 2W onboard computer with an NVIDIA Jetson Orin Nano Super Developer Kit on the Open Duck Mini v2 bipedal robot. The Jetson will be relocated from the head (where the Pi currently lives) to the trunk/body cavity.

**Goal:** Validate feasibility entirely in simulation before committing to hardware purchases or 3D printing. Migrate the entire simulation and training pipeline from MuJoCo to the NVIDIA stack (Isaac Sim + Isaac Lab).

**Guiding principle:** Sim-first. Every physical change is validated in simulation before being built. The full NVIDIA stack (Isaac Sim, Isaac Lab, TensorRT) will be used for simulation, training, and deployment — replacing the current MuJoCo + Stable-Baselines3 pipeline.

**Training hardware:** NVIDIA DGX Spark (Grace Blackwell, 128 GB unified memory, up to 1 PFLOP FP4, 1000 TOPS). All NVIDIA software pre-installed.

---

## Context for AI Agents

### Repository Structure (Key Files)

```
Open_Duck_Mini_Jetson/
├── mini_bdx/
│   ├── robots/
│   │   └── open_duck_mini_v2/
│   │       ├── robot_motors.xml      # PRIMARY MuJoCo model (torque-controlled, 1086 lines)
│   │       ├── robot.xml             # MuJoCo model (position-controlled)
│   │       ├── robot.urdf            # URDF version of the model
│   │       ├── scene.xml             # Simulation scene (includes robot_motors.xml)
│   │       ├── scene_position.xml    # Alternate scene (includes robot.xml)
│   │       ├── config.json           # OnShape export config
│   │       ├── *.stl                 # Mesh files for all robot parts (meters)
│   │       ├── raspberrypizerow.stl  # Pi Zero mesh (to be removed)
│   │       └── usd/                  # NEW — Isaac Sim USD format
│   │           └── open_duck_mini_v2.usd  # Converted from URDF (Task 2.1)
│   └── mini_bdx/
│       └── utils/
│           ├── mujoco_utils.py       # Contact detection, actuator helpers
│           ├── rl_utils.py           # Joint order mapping (Isaac<->MuJoCo), action scaling
│           ├── poly_spline.py        # Trajectory splines
│           └── xbox_controller.py    # Gamepad input
├── experiments/
│   ├── v2/
│   │   ├── onnx_AWD_mujoco.py              # Policy inference in MuJoCo (position control)
│   │   ├── onnx_AWD_mujoco_motor_control.py # Policy inference (torque/PD control)
│   │   └── params_m6.json                   # BAM motor identification params
│   ├── RL/new/
│   │   ├── env.py                    # Gymnasium RL environment (86-dim obs, 15-dim action)
│   │   └── train.py                  # SB3 training script (SAC/TD3/PPO/A2C/TQC)
│   └── real_robot/
│       └── rl_walk.py                # Real robot RL deployment
├── BEST_WALK_ONNX.onnx              # Old pre-trained walking policy (MuJoCo-based, for reference)
├── BEST_WALK_ONNX_2.onnx            # Old pre-trained walking policy (MuJoCo-based, for reference)
├── exported_policies/                # NEW — Isaac Lab trained policies
│   ├── open_duck_walk_policy.onnx   # ONNX export from Isaac Lab PPO training
│   └── open_duck_walk_policy.trt    # TensorRT engine for Jetson deployment
├── isaac_lab_env/                    # NEW — Isaac Lab environment definition
│   └── open_duck_mini_v2/
│       ├── env_cfg.py               # Environment config (obs, actions, rewards, domain rand)
│       ├── train_cfg.py             # Training configs (PPO via RSL-RL, optional AMP via SKRL)
│       ├── evaluate_policies.py     # Multi-algorithm comparison script (Task 2.5)
│       └── __init__.py              # Register env with Isaac Lab
├── jetson_runtime/                   # NEW — Jetson deployment code
│   ├── trt_infer.py                 # TensorRT inference wrapper for locomotion policy
│   ├── thermal_manager.py           # Battery temp monitor + power mode management (Task 4.6)
│   ├── cosmos_commander.py          # Cosmos Reason2 interface + command parser (Phase 5)
│   ├── autonomous_walk.py           # Main control loop: Cosmos + locomotion (Phase 5)
│   ├── walk_controller.py           # Manual walk control loop (replaces Pi runtime)
│   ├── gpio_config.py              # Jetson GPIO pin mappings
│   └── prompts/                     # Cosmos Reason2 prompt library (Phase 5)
│       ├── navigate.txt             # "Walk to the ..." behavior
│       ├── follow.txt               # "Follow me" behavior
│       ├── avoid.txt                # Obstacle avoidance behavior
│       ├── explore.txt              # Room exploration behavior
│       ├── interact.txt             # Social interaction behavior
│       └── patrol.txt               # Patrol and report behavior
├── print/                            # 3D printable STL files (millimeters)
├── docs/
│   ├── sim2real.md                   # Sim-to-real transfer guide
│   ├── assembly_guide.md             # Build instructions + GPIO pin map
│   └── jetson-mod/                   # THIS MODIFICATION
│       ├── task_plan.md              # This file
│       ├── jetson_wiring_diagram.drawio  # Wiring diagram for Jetson modification (Task 4.3)
│       ├── algorithm_comparison.md   # Multi-algorithm evaluation results (Task 2.5 output)
│       ├── validation_results.md    # Final policy validation observations (Task 2.7 output)
│       └── prompt_engineering_results.md  # Cosmos prompt behavior test results (Task 5.4 output)
└── tests/                            # NEW — proposed test directory
    ├── conftest.py
    ├── test_model_integrity.py
    ├── test_mass_inertia.py
    ├── test_usd_conversion.py
    ├── test_isaac_lab_env.py
    ├── test_policy_inference.py
    └── test_tensorrt_inference.py
```

### Key Constants

| Constant | Value | Source |
|---|---|---|
| Total robot DOFs | 15 (+ 1 head_roll = 16 actuators) | robot_motors.xml actuator section |
| MuJoCo timestep | 0.005 s | onnx_AWD_mujoco.py line 72 |
| Control decimation | 4 (policy runs every 4 sim steps = 50 Hz) | onnx_AWD_mujoco.py line 96 |
| Observation dim (AWD policy) | 56 (+ 18 zeros padded = 74 input) | onnx_AWD_mujoco.py lines 86, 209 |
| Observation dim (RL env) | 86 | experiments/RL/new/env.py |
| Action dim | 15 or 16 (depending on policy) | experiments/RL/new/env.py |
| STL unit in robots/ dir | Meters | Verified via bounding box analysis |
| STL unit in print/ dir | Millimeters | Verified via bounding box analysis |
| Current total robot mass | 2,062 g | Sum of all body masses in robot_motors.xml |
| trunk_assembly mass | 698.526 g | robot_motors.xml line 115 |
| head_assembly mass | 352.583 g | robot_motors.xml line 652 |
| Servo model | Feetech STS3250 (12V, 50 kg.cm stall) | setup.cfg, params_sts3250_id008.json |
| Servo weight | 74.5g each, 14 servos total | Datasheet |

### Hardware Change Summary

| Parameter | Raspberry Pi Zero 2W (remove) | Jetson Orin Nano Super Dev Kit (add) |
|---|---|---|
| Dimensions | 65 x 30 x 5 mm | 103 x 90.5 x 34.77 mm (full dev kit incl. heatsink+fan) |
| Weight | 10 g | 176 g |
| Power (typical) | ~1-3 W | 7-25 W |
| Current location | Head (head_assembly body) | Trunk (trunk_assembly body) — NEW |
| AI compute | None | 67 TOPS (1024 CUDA + 32 Tensor cores) |

### Additional Hardware

- 4x extra 18650 battery cells (6 total, 3S2P at 11.1V): ~45g each, 18mm diameter x 65mm long
- 3S BMS (>=15A): ~10g
- DC-DC boost converter (11.1V to 19V): ~15g, ~43x21x14mm (XL6009-based module)
- CSI camera module: ~3g, mounted in head
- Thermal partition wall (PLA, 2mm thick, ~110x90mm): ~25g
- Mica insulation sheet (1mm thick, ~110x90mm): ~12g
- NTC 10kΩ thermistor: ~0.5g (negligible)
- Additional wiring/cables: ~8g estimated

**Power system note (STS3250 + Jetson migration):**
- Servo operating voltage: 11.1V nominal (3S LiPo / 3S2P 18650)
- DC-DC boost: 11.1V → 19V for Jetson barrel jack input
- BMS: 3S rated, >=15A continuous (handles servo + Jetson draw)
- STS3250 servos operate at 6-12.6V; 11.1V nominal is within spec

### NVIDIA Stack Overview

The entire simulation, training, and deployment pipeline is being migrated to NVIDIA tools:

| Stage | Old Stack | New NVIDIA Stack |
|---|---|---|
| **Physics simulator** | MuJoCo / MuJoCo MJX | **Isaac Sim** (PhysX 5, GPU-accelerated) |
| **RL framework** | MuJoCo Playground + Stable-Baselines3 / RSL-RL | **Isaac Lab** (built on Isaac Sim) |
| **RL algorithm** | PPO (via RSL-RL) | **PPO** (via RSL-RL or rl_games — same algorithm) |
| **Robot model format** | MJCF (.xml) / URDF | **USD** (.usd) — converted from URDF via Isaac Sim importer |
| **Training hardware** | Single GPU | **DGX Spark** (Grace Blackwell, 4096+ parallel envs) |
| **Policy export** | ONNX (CPU inference) | **ONNX → TensorRT** (GPU-accelerated on Jetson) |
| **On-robot inference** | onnxruntime on Pi Zero CPU | **TensorRT on Jetson GPU** (67 TOPS) |
| **Robot middleware** | Custom Python scripts | **Isaac ROS** (optional, ROS2-based) |

**Key NVIDIA tools and what they do:**

- **Isaac Sim** — The physics simulator. Built on NVIDIA Omniverse, uses PhysX 5 for GPU-accelerated rigid body simulation. Replaces MuJoCo. Can simulate thousands of robot instances in parallel on one GPU. Docs: https://docs.isaacsim.omniverse.nvidia.com
- **Isaac Lab** — The RL training framework built on top of Isaac Sim. Provides pre-built environments for locomotion, manipulation, etc. Includes domain randomization, reward utilities, and integrations with RSL-RL and rl_games. Replaces Open_Duck_Playground + Stable-Baselines3. Docs: https://isaac-sim.github.io/IsaacLab
- **RSL-RL** — A lightweight PPO implementation optimized for GPU training. Used by both the old MuJoCo pipeline AND Isaac Lab. The RL algorithm itself doesn't change.
- **TensorRT** — NVIDIA's inference optimizer. Converts ONNX models into highly optimized engines for Jetson GPU inference. Sub-millisecond latency for small MLPs. Docs: https://developer.nvidia.com/tensorrt
- **Isaac ROS** — Optional ROS2-based framework for robot deployment on Jetson. Provides sensor drivers, perception pipelines, and policy execution nodes.
- **JetPack** — The OS/SDK for Jetson. Ubuntu-based, includes CUDA, cuDNN, TensorRT, and all NVIDIA libraries pre-installed.

### Reinforcement Learning (RL) Concepts

This section explains how RL is used to make the robot walk, for readers who are new to the concept.

**What is RL?**

Reinforcement Learning trains a neural network (the "policy") to control the robot through trial and error in simulation. Instead of hand-coding motor commands, the robot learns by:
1. Observing its current state (joint angles, IMU, foot contacts)
2. Taking an action (setting motor targets)
3. Receiving a reward signal (did I stay upright? Am I walking smoothly?)
4. Updating the neural network to maximize future rewards

**The RL Loop:**

```
                    ┌──────────────────────────┐
                    │       ENVIRONMENT         │
                    │  (Isaac Sim: simulated    │
                    │   robot + physics)        │
                    └───────┬─────────▲─────────┘
                            │         │
               observation  │         │  action
             (what I sense) │         │  (motor targets)
                            ▼         │
                    ┌──────────────────────────┐
                    │         POLICY            │
                    │  (neural network, MLP     │
                    │   [512, 256, 128] units)  │
                    └───────┬─────────▲─────────┘
                            │         │
                      reward│         │update weights
                  (+0.05 per│         │(gradient descent)
                   timestep)│         │
                            ▼         │
                    ┌──────────────────────────┐
                    │      PPO ALGORITHM        │
                    │  (Proximal Policy         │
                    │   Optimization)           │
                    └──────────────────────────┘
```

**Key RL terms as they apply to this project:**

| Term | Meaning in this project |
|---|---|
| **Observation** | What the robot senses: joint angles (15), joint velocities (15), gravity direction (3), foot contacts (2), previous action (16), walk command (3) = 54-86 dimensions depending on policy |
| **Action** | Motor position targets for all 15-16 joints, output by the neural network |
| **Reward** | A scalar score per timestep. Current rewards: +0.05 for surviving, penalty for jerky motion, bonus for staying near natural pose. Advanced: imitation reward (match reference walking motion from Disney BDX paper) |
| **Episode** | One attempt at walking. Ends when robot falls (trunk Z < 0.08m or tilts > 90 deg) or time limit reached |
| **Policy** | The neural network (MLP with 3 hidden layers). Maps observation → action |
| **PPO** | Proximal Policy Optimization — the RL algorithm. Collects batches of experience from thousands of parallel environments, then updates the policy network in small steps. Very stable for locomotion. |
| **Domain randomization** | During training, randomly vary mass, friction, motor strength, sensor noise. This makes the policy robust to real-world uncertainty (key for sim2real transfer). |
| **Sim2real** | The challenge of transferring a policy trained in simulation to the real robot. Requires accurate physics, motor models, and domain randomization. |
| **ONNX / TensorRT** | The trained policy is exported as an ONNX file, then optimized with TensorRT for fast inference on the Jetson GPU |

**PPO specifically (the algorithm used):**

PPO is the dominant RL algorithm for robot locomotion. Here's why:
1. It runs thousands of robots in parallel (4096+ in Isaac Lab on DGX Spark)
2. Each robot collects experience (observations, actions, rewards) for N timesteps
3. The experience is used to compute a "surrogate objective" — an estimate of how much better the new policy is
4. The policy network is updated via gradient descent, but the update is **clipped** to prevent too-large changes (the "proximal" part) — this makes training stable
5. Repeat for millions of timesteps until the robot walks reliably

**Other RL algorithms found in this repo's history:**

| Algorithm | Where | Notes |
|---|---|---|
| **SAC** (Soft Actor-Critic) | `experiments/RL/new/train.py` | Maximizes reward + entropy. Good exploration but slower per-step than PPO. |
| **TD3** (Twin Delayed DDPG) | `experiments/RL/new/train.py` | Off-policy, uses replay buffer. Less common for locomotion. |
| **TQC** (Truncated Quantile Critics) | `experiments/RL/new/train.py` | Advanced distributional RL. Experimental. |
| **A2C** (Advantage Actor-Critic) | `experiments/RL/new/train.py` | Simpler PPO variant. Faster but less stable. |

For this project going forward, we use **PPO exclusively** via Isaac Lab's RSL-RL integration — it's the proven choice for bipedal locomotion.

---

## Proposed Test Structure

### Directory: `tests/`

Create a new `tests/` directory at the repository root with the following structure:

```
tests/
├── conftest.py                     # Shared pytest fixtures (model loading, paths)
├── test_model_integrity.py         # Phase 1: XML/URDF well-formedness & asset checks
├── test_mass_inertia.py            # Phase 1: Mass/inertia correctness validation
├── test_usd_conversion.py          # Phase 2: USD model validates in Isaac Sim
├── test_isaac_lab_env.py           # Phase 2: Isaac Lab env creates, steps, and resets
├── test_policy_inference.py        # Phase 2: ONNX/TensorRT policy loads and produces valid actions
├── test_cad_dimensions.py          # Phase 3: STL bounding box & clearance checks
├── test_tensorrt_inference.py      # Phase 4: TensorRT engine runs on Jetson
└── fixtures/
    └── expected_values.json        # Ground-truth mass, inertia, dimension values
```

### Test Dependencies

Add to `setup.cfg` under `[options.extras_require]`:
```
test =
    pytest>=7.0
    mujoco>=3.1.5
    numpy
    onnxruntime

# On DGX Spark (for Isaac Lab tests):
#   Isaac Sim + Isaac Lab (installed via isaaclab.sh)
#   torch, rsl_rl, tensorboard

# On Jetson (for TensorRT tests):
#   tensorrt (pre-installed with JetPack)
#   pycuda
#   onnxruntime-gpu
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only model integrity tests (Phase 1)
pytest tests/test_model_integrity.py -v

# Run simulation stability tests (Phase 2)
pytest tests/test_simulation_stability.py -v

# Run with markers
pytest -m "phase1" -v
pytest -m "phase2" -v
```

---

## Phase 1: Simulation Model Update

### Task 1.1 — Calculate New Mass and Inertia Values

**Description:**
Compute the updated mass, center-of-mass (CoM), and diagonal inertia tensor for `trunk_assembly` and `head_assembly` bodies after removing the Pi Zero 2W from the head and adding the Jetson Orin Nano Dev Kit + extra batteries + thermal partition to the trunk.

**Inputs:**
- Current `trunk_assembly`: mass=0.698526 kg, pos=(-0.0483259, -9.97823e-05, 0.0384971), diaginertia=(0.00344489, 0.00292719, 0.00167606)
- Current `head_assembly`: mass=0.352583 kg, pos=(0.00761779, 0.00018098, 0.0242575), diaginertia=(0.00207104, 0.00144128, 0.000909578)
- Raspberry Pi Zero 2W: 10g, 65x30x5mm, located at pos=(0.03205, 0.048, 0.00595) in head_assembly frame
- Jetson Orin Nano Dev Kit (full kit incl. heatsink+fan): 176g, 103x90.5x34.77mm
- 2x extra 18650 cells: 45g each, 18mm diameter x 65mm long
- DC-DC boost converter (XL6009): 15g, 43x21x14mm
- Thermal partition wall (PLA, 2mm thick): ~25g, ~110x90x2mm
- Mica insulation sheet (1mm thick, bonded to partition): ~12g, ~110x90x1mm
- Additional wiring/cables: ~8g (estimated lump mass at trunk CoM)

**Steps:**
1. Determine Jetson placement position in the trunk_assembly coordinate frame. Target: forward-mid cavity, approximately pos=(-0.03, 0.0, 0.035) in trunk frame (between the neck motor and the battery pack, above the trunk bottom). Note: the full dev kit is 103x90.5x35mm — verify clearance against trunk cavity bounds.
2. Compute Jetson's inertia tensor as a rectangular solid:
   - Ixx = m/12 * (height^2 + depth^2) = 0.176/12 * (0.03477^2 + 0.0905^2)
   - Iyy = m/12 * (height^2 + width^2) = 0.176/12 * (0.03477^2 + 0.103^2)
   - Izz = m/12 * (width^2 + depth^2) = 0.176/12 * (0.103^2 + 0.0905^2)
3. Compute extra battery inertia (2 cylinders, I_axial = m*r^2/2, I_transverse = m/12*(3r^2+h^2))
4. Compute DC-DC converter inertia (rectangular solid, 0.015 kg, 43x21x14mm)
5. Compute thermal partition + mica inertia (thin slab, ~0.037 kg total, 110x90x3mm composite at partition position X ≈ -0.08m)
6. Use the parallel axis theorem to compute the new composite trunk_assembly inertia:
   - I_composite = I_existing + I_jetson_at_new_pos + I_batteries_at_pos + I_dcdc_at_pos + I_partition_at_pos + I_wiring
   - New CoM = (m_existing * pos_existing + m_jetson * pos_jetson + ...) / m_total
7. Subtract Pi Zero contribution from head_assembly inertia using parallel axis theorem in reverse
8. Document all values with full derivation

**Output:**
A file `docs/jetson-mod/mass_inertia_calculations.md` containing:
- All input parameters
- Step-by-step calculations
- Final values for robot_motors.xml

Also create `tests/fixtures/expected_values.json` with the computed values.

**How to test:**

*Automated test — `tests/test_mass_inertia.py`:*

```python
import pytest
import json
import numpy as np

@pytest.mark.phase1
class TestMassInertiaCalculations:

    def test_total_mass_is_correct(self, expected_values):
        """Total mass should equal sum of all body masses."""
        # Sum all body masses from the updated XML
        # Compare against expected total (~2406g)
        assert abs(total_mass - expected_values["total_mass_kg"]) < 0.001

    def test_trunk_mass_increased(self, expected_values):
        """trunk_assembly mass should reflect Jetson + extra batteries + thermal partition + wiring."""
        expected = expected_values["trunk_assembly_mass_kg"]
        # Should be ~1.0545 kg (original 0.6985 + 0.176 + 0.090 + 0.015 + 0.037 + 0.008)
        assert abs(trunk_mass - expected) < 0.001

    def test_head_mass_decreased(self, expected_values):
        """head_assembly mass should decrease by ~10g (Pi removed)."""
        expected = expected_values["head_assembly_mass_kg"]
        # Should be ~0.3426 kg
        assert abs(head_mass - expected) < 0.001

    def test_inertia_tensor_positive_definite(self, updated_model):
        """All diagonal inertia values must be positive."""
        for body_id in range(updated_model.nbody):
            inertia = updated_model.body_inertia[body_id]
            if np.sum(inertia) > 0:  # skip zero-mass bodies
                assert all(i >= 0 for i in inertia), f"Body {body_id} has negative inertia"

    def test_triangle_inequality_holds(self, updated_model):
        """Inertia tensor must satisfy triangle inequality: Ixx+Iyy >= Izz, etc."""
        for body_id in range(updated_model.nbody):
            I = updated_model.body_inertia[body_id]
            if np.sum(I) > 0:
                assert I[0] + I[1] >= I[2] - 1e-10
                assert I[0] + I[2] >= I[1] - 1e-10
                assert I[1] + I[2] >= I[0] - 1e-10

    def test_com_within_body_bounds(self, updated_model):
        """Center of mass should be within reasonable bounds of the body geometry."""
        trunk_id = updated_model.body("trunk_assembly").id
        com = updated_model.body_ipos[trunk_id]
        # trunk extends roughly X: [-0.14, 0.02], Y: [-0.055, 0.055], Z: [-0.024, 0.088]
        assert -0.15 < com[0] < 0.03, f"trunk CoM X out of bounds: {com[0]}"
        assert -0.06 < com[1] < 0.06, f"trunk CoM Y out of bounds: {com[1]}"
        assert -0.03 < com[2] < 0.10, f"trunk CoM Z out of bounds: {com[2]}"
```

*Manual verification:*
- [ ] Open the calculations spreadsheet/document and verify each formula step
- [ ] Cross-check: new trunk mass ≈ 0.6985 + 0.176 + 0.090 + 0.015 + 0.037 + 0.008 = 1.0245 kg
- [ ] Cross-check: new head mass ≈ 0.3526 - 0.010 = 0.3426 kg
- [ ] Cross-check: new total mass ≈ 2.062 + 0.176 + 0.090 + 0.015 + 0.037 + 0.008 - 0.010 = 2.378 kg

---

### Task 1.2 — Create Jetson Placeholder STL Mesh

**Description:**
Create a simple box-shaped STL file representing the Jetson Orin Nano Dev Kit for use in the MuJoCo simulation. The mesh units must be in **meters** (matching other meshes in the `robots/open_duck_mini_v2/` directory).

**Dimensions:** 0.103 x 0.0905 x 0.03477 meters (103 x 90.5 x 34.77 mm) — full dev kit including carrier board + module + heatsink + fan

**Steps:**
1. Write a Python script to generate a box STL with the correct dimensions
2. Save as `mini_bdx/robots/open_duck_mini_v2/jetson_orin_nano.stl`
3. Create `mini_bdx/robots/open_duck_mini_v2/thermal_partition.stl` — thin slab (0.110 x 0.090 x 0.003 m) representing the PLA partition wall + mica insulation
4. Optionally create `mini_bdx/robots/open_duck_mini_v2/dcdc_converter.stl` (0.043 x 0.021 x 0.014 m)

**How to test:**

*Automated test — `tests/test_model_integrity.py`:*

```python
import pytest
import struct
import os
import numpy as np

ROBOT_DIR = "mini_bdx/robots/open_duck_mini_v2"

@pytest.mark.phase1
class TestJetsonMesh:

    def test_jetson_stl_exists(self):
        """Jetson STL mesh file must exist."""
        assert os.path.exists(os.path.join(ROBOT_DIR, "jetson_orin_nano.stl"))

    def test_jetson_stl_dimensions(self):
        """Jetson STL bounding box must match 103x90.5x34.77mm (in meters)."""
        vertices = load_stl_vertices(os.path.join(ROBOT_DIR, "jetson_orin_nano.stl"))
        dims = vertices.max(axis=0) - vertices.min(axis=0)
        # Tolerance of 0.5mm
        assert abs(dims[0] - 0.103) < 0.0005, f"X dimension wrong: {dims[0]}"
        assert abs(dims[1] - 0.0905) < 0.0005, f"Y dimension wrong: {dims[1]}"
        assert abs(dims[2] - 0.03477) < 0.0005, f"Z dimension wrong: {dims[2]}"

    def test_jetson_stl_units_are_meters(self):
        """All coordinates should be < 0.2 (meters, not mm)."""
        vertices = load_stl_vertices(os.path.join(ROBOT_DIR, "jetson_orin_nano.stl"))
        assert vertices.max() < 0.2, "STL appears to be in millimeters, not meters"
        assert vertices.min() > -0.2, "STL appears to be in millimeters, not meters"
```

*Manual verification:*
- [ ] Open in MuJoCo viewer and visually confirm the box is approximately the right size relative to the robot trunk
- [ ] Compare visually against the `raspberrypizerow.stl` mesh — the Jetson box should be noticeably larger

---

### Task 1.3 — Update `robot_motors.xml`

**Description:**
Modify the primary MuJoCo simulation model to reflect the Jetson modification. This is the most critical file change.

**File:** `mini_bdx/robots/open_duck_mini_v2/robot_motors.xml`

**Changes required:**

1. **Add Jetson mesh asset** (near line 65):
   ```xml
   <mesh name="jetson_orin_nano" file="jetson_orin_nano.stl" />
   <mesh name="thermal_partition" file="thermal_partition.stl" />
   ```

2. **Remove Pi Zero mesh asset** (line 61):
   ```xml
   <!-- REMOVE: <mesh name="raspberrypizerow" file="raspberrypizerow.stl" /> -->
   ```

3. **Add Jetson geom in trunk_assembly** (after line 191, inside the trunk_assembly body):
   ```xml
   <geom
       pos="CALCULATED_X CALCULATED_Y CALCULATED_Z"
       quat="CALCULATED_QUATERNION"
       type="mesh"
       rgba="0.1 0.6 0.1 1"
       mesh="jetson_orin_nano"
   />
   ```
   The position values come from Task 1.1. Note: the full dev kit bounding box is 103x90.5x35mm — verify it fits within the trunk cavity.

4. **Add thermal partition geom in trunk_assembly** (near Jetson geom):
   ```xml
   <geom
       pos="-0.08 0 CALCULATED_Z"
       type="mesh"
       rgba="0.8 0.2 0.2 0.5"
       mesh="thermal_partition"
   />
   ```
   Position X ≈ -0.08m (between battery zone and compute zone). The partition mass (~37g) is included in the trunk_assembly inertial from Task 1.1.

5. **Remove Pi Zero geom from head_assembly** (lines 682-688):
   ```xml
   <!-- REMOVE the raspberrypizerow geom -->
   ```

6. **Update trunk_assembly inertial** (lines 112-117):
   Replace mass, pos, quat, and diaginertia with values from Task 1.1 (includes Jetson + batteries + DC-DC + thermal partition + wiring).

7. **Update head_assembly inertial** (lines 649-653):
   Replace mass, pos, quat, and diaginertia with values from Task 1.1.

**How to test:**

*Automated test — `tests/test_model_integrity.py`:*

```python
import pytest
import mujoco
import xml.etree.ElementTree as ET

@pytest.mark.phase1
class TestRobotMotorsXML:

    def test_xml_parses_without_error(self):
        """The modified XML must parse and compile in MuJoCo."""
        model = mujoco.MjModel.from_xml_path(
            "mini_bdx/robots/open_duck_mini_v2/scene.xml"
        )
        assert model is not None

    def test_correct_number_of_actuators(self, model):
        """Must still have exactly 16 actuators."""
        assert model.nu == 16

    def test_correct_number_of_joints(self, model):
        """Must still have exactly 16 joints (+ 1 freejoint = 16 named joints)."""
        named_joints = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        ]
        # Exclude the freejoint (unnamed or auto-named)
        expected_joints = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "neck_pitch", "head_pitch", "head_yaw", "head_roll",
            "left_antenna", "right_antenna",
        ]
        for j in expected_joints:
            assert j in named_joints, f"Missing joint: {j}"

    def test_no_raspberrypi_mesh(self):
        """raspberrypizerow mesh should not exist in the model."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")
        root = tree.getroot()
        meshes = [m.get("name") for m in root.iter("mesh")]
        assert "raspberrypizerow" not in meshes

    def test_jetson_mesh_exists(self):
        """jetson_orin_nano mesh must be declared in assets."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")
        root = tree.getroot()
        meshes = [m.get("name") for m in root.iter("mesh")]
        assert "jetson_orin_nano" in meshes

    def test_jetson_geom_in_trunk(self):
        """A geom referencing jetson_orin_nano must exist inside trunk_assembly body."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")
        root = tree.getroot()
        # Find trunk_assembly body
        for body in root.iter("body"):
            if body.get("name") == "trunk_assembly":
                geom_meshes = [g.get("mesh") for g in body.findall("geom")]
                assert "jetson_orin_nano" in geom_meshes
                break

    def test_no_pi_geom_in_head(self):
        """No geom referencing raspberrypizerow should exist in head_assembly."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")
        root = tree.getroot()
        for body in root.iter("body"):
            if body.get("name") == "head_assembly":
                geom_meshes = [g.get("mesh") for g in body.findall("geom")]
                assert "raspberrypizerow" not in geom_meshes
                break

    def test_trunk_mass_updated(self, model):
        """trunk_assembly mass should be approximately 1.0-1.06 kg."""
        trunk_id = model.body("trunk_assembly").id
        mass = model.body_mass[trunk_id]
        assert 0.95 < mass < 1.15, f"trunk mass out of expected range: {mass}"

    def test_head_mass_updated(self, model):
        """head_assembly mass should be approximately 0.34 kg."""
        head_id = model.body("head_assembly").id
        mass = model.body_mass[head_id]
        assert 0.30 < mass < 0.36, f"head mass out of expected range: {mass}"

    def test_total_mass_in_range(self, model):
        """Total robot mass should be approximately 2.2-2.5 kg."""
        total = sum(model.body_mass)
        assert 2.1 < total < 2.6, f"Total mass out of expected range: {total}"

    def test_simulation_does_not_diverge(self):
        """Stepping the simulation 1000 times should not produce NaN or Inf."""
        model = mujoco.MjModel.from_xml_path(
            "mini_bdx/robots/open_duck_mini_v2/scene.xml"
        )
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), "NaN in qpos after 1000 steps"
        assert not np.any(np.isinf(data.qpos)), "Inf in qpos after 1000 steps"
```

*Manual verification:*
- [ ] Open in MuJoCo viewer: `python3 -m mujoco.viewer --mjcf=mini_bdx/robots/open_duck_mini_v2/scene.xml`
- [ ] Visually confirm: Jetson box is visible inside the trunk area
- [ ] Visually confirm: No Pi Zero visible in the head
- [ ] Visually confirm: Robot doesn't immediately collapse or explode
- [ ] Visually confirm: Robot proportions look correct (no giant or tiny parts)

---

### Task 1.4 — Update `robot.xml`

**Description:**
Apply the same changes from Task 1.3 to the position-controlled robot model.

**File:** `mini_bdx/robots/open_duck_mini_v2/robot.xml`

**Changes:** Identical to Task 1.3 (mesh assets, geoms, inertials). The only difference is this file uses `<position>` actuators instead of `<motor>` actuators — the actuator section does NOT change.

**How to test:**

*Automated test — `tests/test_model_integrity.py`:*

```python
@pytest.mark.phase1
class TestRobotXML:

    def test_xml_parses_without_error(self):
        """robot.xml must compile when loaded via scene_position.xml."""
        model = mujoco.MjModel.from_xml_path(
            "mini_bdx/robots/open_duck_mini_v2/scene_position.xml"
        )
        assert model is not None

    def test_mass_matches_robot_motors(self):
        """Total mass in robot.xml must match robot_motors.xml."""
        model_motors = mujoco.MjModel.from_xml_path(
            "mini_bdx/robots/open_duck_mini_v2/scene.xml"
        )
        model_pos = mujoco.MjModel.from_xml_path(
            "mini_bdx/robots/open_duck_mini_v2/scene_position.xml"
        )
        assert abs(sum(model_motors.body_mass) - sum(model_pos.body_mass)) < 0.001
```

---

### Task 1.5 — Update `robot.urdf`

**Description:**
Apply corresponding changes to the URDF file. This is lower priority than the XML files since the RL pipeline uses MuJoCo XML directly, but it should be kept in sync.

**File:** `mini_bdx/robots/open_duck_mini_v2/robot.urdf`

**Changes:**
1. Remove all `<visual>` and `<collision>` elements referencing `raspberrypizerow.stl` from the `trunk_assembly` link
2. Add `<visual>` and `<collision>` elements for `jetson_orin_nano.stl` in the `trunk_assembly` link
3. Update `<inertial>` elements for `trunk_assembly` and `head_assembly` links (mass, origin xyz, inertia ixx/iyy/izz)

**How to test:**

*Automated test — `tests/test_model_integrity.py`:*

```python
@pytest.mark.phase1
class TestRobotURDF:

    def test_urdf_parses(self):
        """URDF must be valid XML."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot.urdf")
        assert tree.getroot().tag == "robot"

    def test_no_raspberrypi_reference(self):
        """No reference to raspberrypizerow.stl in URDF."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot.urdf")
        content = ET.tostring(tree.getroot(), encoding="unicode")
        assert "raspberrypizerow" not in content

    def test_jetson_reference_exists(self):
        """URDF should reference jetson_orin_nano.stl."""
        tree = ET.parse("mini_bdx/robots/open_duck_mini_v2/robot.urdf")
        content = ET.tostring(tree.getroot(), encoding="unicode")
        assert "jetson_orin_nano" in content
```

---

### Task 1.6 — Create Test Infrastructure

**Description:**
Set up the `tests/` directory with pytest configuration, shared fixtures, and the expected values file.

**Files to create:**

1. `tests/__init__.py` — empty
2. `tests/conftest.py` — shared fixtures:
   ```python
   import pytest
   import json
   import mujoco
   import os

   REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   ROBOT_DIR = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")

   @pytest.fixture
   def model():
       """Load the torque-controlled MuJoCo model."""
       return mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, "scene.xml"))

   @pytest.fixture
   def model_position():
       """Load the position-controlled MuJoCo model."""
       return mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, "scene_position.xml"))

   @pytest.fixture
   def updated_model(model):
       """Alias for clarity in test names."""
       return model

   @pytest.fixture
   def expected_values():
       """Load expected mass/inertia values."""
       with open(os.path.join(REPO_ROOT, "tests", "fixtures", "expected_values.json")) as f:
           return json.load(f)
   ```
3. `tests/fixtures/expected_values.json` — populated with values from Task 1.1

**How to test:**

```bash
# Verify test infrastructure works
pytest tests/ --collect-only
# Should list all test functions without running them
```

---

## Phase 2: Isaac Sim/Lab Setup & RL Training (on DGX Spark)

> This phase converts the robot to the NVIDIA stack and trains a fresh walking policy from scratch using Isaac Lab on the DGX Spark. The environment extends Isaac Lab's built-in `LocomotionVelocityRoughEnvCfg` (following the H1 humanoid biped pattern) to inherit battle-tested defaults for observations, rewards, terminations, domain randomization, and curriculum.

### Task 2.1 — Convert MJCF to USD for Isaac Sim

**Description:**
Convert the robot model to USD format using Isaac Lab's converter API. We use the MJCF importer (from `robot_motors.xml`) as the primary path since it preserves actuator parameters (kp, kd, armature, frictionloss) more directly than the URDF importer.

**Prerequisites:**
- Isaac Sim 5.1.0+ built from source on DGX Spark
- Isaac Lab 2.3.0+ installed
- Updated `robot_motors.xml` from Phase 1 with correct mass/inertia and Jetson mesh

**Steps:**
1. Use the Isaac Lab MJCF converter API:
   ```python
   from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

   cfg = MjcfConverterCfg(
       asset_path="mini_bdx/robots/open_duck_mini_v2/robot_motors.xml",
       usd_path="mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd",
       fix_base=False,
       make_instanceable=True,
       import_inertia_tensor=True,
       import_sites=True,
   )
   converter = MjcfConverter(cfg)  # conversion happens in __init__
   ```
   Alternatively, use the low-level Isaac Sim API as fallback:
   ```python
   import omni.kit.commands
   status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
   import_config.set_fix_base(False)
   import_config.set_import_inertia_tensor(True)
   omni.kit.commands.execute(
       "MJCFCreateAsset",
       mjcf_path="mini_bdx/robots/open_duck_mini_v2/robot_motors.xml",
       import_config=import_config,
       dest_path="mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd",
   )
   ```
2. Open the resulting USD in Isaac Sim viewer — verify visual appearance
3. Save the USD file to the repository

**Output file:** `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`

**How to test:**

*Automated test — `tests/test_usd_conversion.py`:*

```python
import pytest
import os

@pytest.mark.phase2
class TestUSDConversion:

    def test_usd_file_exists(self):
        """Converted USD file must exist."""
        assert os.path.exists("mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd")

    def test_usd_file_not_empty(self):
        """USD file must not be empty."""
        size = os.path.getsize("mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd")
        assert size > 1000, f"USD file suspiciously small: {size} bytes"
```

*Manual verification (on DGX Spark):*
- [ ] Open USD in Isaac Sim viewer — robot should appear with correct geometry
- [ ] Verify all 16 joints are present and have correct names
- [ ] Verify joint limits match the MJCF values
- [ ] Verify mass properties are preserved (check in physics inspector)
- [ ] Drag the robot around — physics should behave reasonably (no explosions, no interpenetration)
- [ ] Compare visual appearance against the MuJoCo viewer rendering

---

### Task 2.2 — Create Robot ArticulationCfg

**Description:**
Define the Isaac Lab `ArticulationCfg` for the Open Duck Mini v2, including the actuator model with BAM-identified Feetech STS3250 servo parameters. This config is used by the environment to spawn the robot.

**Motor parameters (from `experiments/v2/params_sts3250_id008.json` — kscalelabs/sysid STS3250 id008):**

| Parameter | Value | Source |
|---|---|---|
| armature | 0.040 | BAM id008 params |
| frictionloss | 0.200 | BAM id008 params |
| kp (position gain) | 45.53 | BAM id008 mujoco_export.kp |
| kd (velocity gain) | 1.346 | BAM id008 mujoco_export.damping |
| effort_limit | 8.716 Nm | BAM id008 mujoco_export.forcerange |
| kt (torque constant) | 1.0006 | BAM id008 params |
| R (resistance) | 1.3890 | BAM id008 params |

**Steps:**
1. Create the robot asset configuration:
   ```python
   # isaac_lab_env/open_duck_mini_v2/robot_cfg.py

   from isaaclab.actuators import ImplicitActuatorCfg
   from isaaclab.assets.articulation import ArticulationCfg
   import isaaclab.sim as sim_utils

   OPEN_DUCK_MINI_V2_CFG = ArticulationCfg(
       spawn=sim_utils.UsdFileCfg(
           usd_path="mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd",
           rigid_props=sim_utils.RigidBodyPropertiesCfg(
               disable_gravity=False,
               retain_accelerations=False,
           ),
           articulation_props=sim_utils.ArticulationRootPropertiesCfg(
               enabled_self_collisions=False,
           ),
       ),
       init_state=ArticulationCfg.InitialStateCfg(
           pos=(0.0, 0.0, 0.17),  # spawn height (same as MuJoCo base pos)
           joint_pos={
               "right_hip_yaw": -0.014, "right_hip_roll": 0.079,
               "right_hip_pitch": 0.533, "right_knee": -1.623,
               "right_ankle": 0.915,
               "left_hip_yaw": 0.014, "left_hip_roll": 0.077,
               "left_hip_pitch": 0.593, "left_knee": -1.631,
               "left_ankle": 0.862,
               "neck_pitch": -0.175, "head_pitch": -0.175,
               "head_yaw": 0.0, "head_roll": 0.0,
               "left_antenna": 0.0, "right_antenna": 0.0,
           },
       ),
       actuators={
           "legs": ImplicitActuatorCfg(
               joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch",
                                 ".*_knee", ".*_ankle"],
               stiffness=45.53,
               damping=1.346,
               armature=0.040,
               friction=0.200,
               effort_limit=8.716,
           ),
           "head": ImplicitActuatorCfg(
               joint_names_expr=["neck_pitch", "head_pitch", "head_yaw",
                                 "head_roll", ".*_antenna"],
               stiffness=45.53,
               damping=1.346,
               armature=0.040,
               friction=0.200,
               effort_limit=8.716,
           ),
       },
   )
   ```
2. Verify that a single joint responds similarly to MuJoCo when given the same step input

**How to test:**

*Manual verification:*
- [ ] In Isaac Sim, command a single leg joint to move from 0 to 0.5 rad
- [ ] Compare the response curve (speed, overshoot, settling time) to the same test in MuJoCo
- [ ] The responses should be qualitatively similar — not identical but same ballpark

---

### Task 2.3 — Create Isaac Lab Locomotion Environment

**Description:**
Create an Isaac Lab RL environment for the Open Duck Mini v2 by **extending the built-in `LocomotionVelocityRoughEnvCfg`** — following the same pattern as the H1 humanoid biped config. This gives us correct observations, actions, terminations, domain randomization, commands, and curriculum out of the box. We only override what's specific to our duck robot.

**Architecture decision:** We inherit from `LocomotionVelocityRoughEnvCfg` rather than building from scratch. This base class (located at `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`) provides:
- Observations: `base_lin_vel` (3), `base_ang_vel` (3), `projected_gravity` (3), `velocity_commands` (3), `joint_pos_rel` (N), `joint_vel_rel` (N), `last_action` (N) — with noise for sim2real
- Actions: `JointPositionActionCfg` with `scale=0.5` and `use_default_offset=True`
- Commands: `UniformVelocityCommandCfg` with `rel_standing_envs=0.02`
- Rewards: `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2`, `action_rate_l2`, `joint_torques_l2`, `joint_acc_l2`, `feet_air_time`, `undesired_contacts`
- Terminations: `time_out` (with `time_out=True`), `base_contact`, `root_height_below_minimum`
- Events: mass/friction/CoM randomization, push disturbances, joint/base reset randomization
- Curriculum: terrain difficulty based on velocity tracking performance
- Scene: terrain, robot, contact sensors, height scanner

**Reference implementation:** `isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py`

**Steps:**
1. Create a new directory: `isaac_lab_env/open_duck_mini_v2/`
2. Define duck-specific reward overrides (biped gait rewards):

   ```python
   # isaac_lab_env/open_duck_mini_v2/env_cfg.py

   import math
   from isaaclab.managers import RewardTermCfg as RewTerm
   from isaaclab.managers import SceneEntityCfg
   from isaaclab.utils import configclass

   import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
   from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
       LocomotionVelocityRoughEnvCfg, RewardsCfg,
   )
   from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG

   @configclass
   class DuckRewards(RewardsCfg):
       """Biped-specific rewards for the Open Duck Mini v2."""

       # Termination penalty (strong negative signal for falling)
       termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

       # Override velocity tracking with yaw-frame versions (biped best practice)
       lin_vel_z_l2 = None  # Disable default — bouncing penalty not needed for biped
       track_lin_vel_xy_exp = RewTerm(
           func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
           params={"command_name": "base_velocity", "std": 0.5},
       )
       track_ang_vel_z_exp = RewTerm(
           func=mdp.track_ang_vel_z_world_exp, weight=1.0,
           params={"command_name": "base_velocity", "std": 0.5},
       )

       # Biped-specific gait reward — encourages alternating single-stance
       feet_air_time = RewTerm(
           func=mdp.feet_air_time_positive_biped, weight=0.25,
           params={
               "command_name": "base_velocity",
               "sensor_cfg": SceneEntityCfg("contact_forces",
                   body_names=["left_foot", "right_foot"]),
               "threshold": 0.4,
           },
       )

       # Penalize feet sliding on ground
       feet_slide = RewTerm(
           func=mdp.feet_slide, weight=-0.25,
           params={
               "sensor_cfg": SceneEntityCfg("contact_forces",
                   body_names=["left_foot", "right_foot"]),
               "asset_cfg": SceneEntityCfg("robot",
                   body_names=["left_foot", "right_foot"]),
           },
       )

       # Penalize deviation of non-locomotion joints from default
       joint_deviation_head = RewTerm(
           func=mdp.joint_deviation_l1, weight=-0.1,
           params={"asset_cfg": SceneEntityCfg("robot",
               joint_names=["neck_pitch", "head_pitch", "head_yaw", "head_roll",
                            "left_antenna", "right_antenna"])},
       )

       # Penalize hip yaw/roll deviation from default — prevents unnecessary
       # hip splaying and conserves torque on the weak Feetech servos.
       # Following H1/G1/Digit biped configs which all penalize hip deviation.
       joint_deviation_hips = RewTerm(
           func=mdp.joint_deviation_l1, weight=-0.2,
           params={"asset_cfg": SceneEntityCfg("robot",
               joint_names=["right_hip_yaw", "left_hip_yaw",
                            "right_hip_roll", "left_hip_roll"])},
       )

       # Penalize joints approaching position limits — protects real Feetech
       # STS3250 servos from hitting hard stops and causing gear damage.
       # All Isaac Lab biped configs (H1, G1, Cassie, Digit) include this.
       joint_pos_limits = RewTerm(
           func=mdp.joint_pos_limits, weight=-1.0,
           params={"asset_cfg": SceneEntityCfg("robot",
               joint_names=["right_ankle", "left_ankle",
                            "right_knee", "left_knee"])},
       )

   @configclass
   class OpenDuckRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
       """Open Duck Mini v2 locomotion environment configuration."""

       rewards: DuckRewards = DuckRewards()

       def __post_init__(self):
           super().__post_init__()

           # --- Scene: swap robot asset ---
           self.scene.robot = OPEN_DUCK_MINI_V2_CFG.replace(
               prim_path="{ENV_REGEX_NS}/Robot"
           )
           self.scene.num_envs = 4096
           self.scene.env_spacing = 2.5

           # --- Simulation timing ---
           self.sim.dt = 0.005       # 200 Hz physics
           self.decimation = 4       # Policy at 50 Hz
           self.episode_length_s = 20.0

           # --- Height scanner (optional, set to None for flat-ground only) ---
           self.scene.height_scanner = None

           # --- Commands: velocity ranges for the duck ---
           self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
           self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
           self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

           # --- Terminations: adjust body names for duck ---
           self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk_assembly"

           # --- Events: adjust for duck-specific bodies ---
           self.events.push_robot = None  # Disable push initially, enable after basic walking works
           self.events.base_external_force_torque.params["asset_cfg"].body_names = ["trunk_assembly"]
           self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
           self.events.reset_base.params = {
               "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
               "velocity_range": {
                   "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
               },
           }

           # --- Rewards: tune weights for duck ---
           self.rewards.undesired_contacts = None  # Re-enable after verifying body names
           self.rewards.flat_orientation_l2.weight = -1.0
           self.rewards.action_rate_l2.weight = -0.005
           self.rewards.dof_acc_l2.weight = -1.25e-7
           self.rewards.dof_torques_l2.weight = 0.0  # Disable torque penalty initially
           # Increase pitch/roll angular velocity penalty (default -0.05) to
           # compensate for top-heavy trunk after Jetson relocation (+316g).
           # Digit biped (also top-heavy) uses -0.1.
           self.rewards.ang_vel_xy_l2.weight = -0.1

           # --- Conditional rewards (add after initial training if needed) ---
           # base_height_l2: Add if policy learns excessively crouched gait.
           #   Target ~0.15-0.17m. The original MuJoCo env used tight height
           #   control (weight 0.15, target 0.15m). Not added initially to
           #   avoid over-constraining early training.
           # stand_still_joint_deviation_l1: Add if robot can't stand still on
           #   zero velocity command (shuffles/wiggles in place). Digit uses
           #   this at weight -0.4. Not needed until basic walking works.

   @configclass
   class OpenDuckRoughEnvCfg_PLAY(OpenDuckRoughEnvCfg):
       """Playback configuration with fewer envs and no randomization."""
       def __post_init__(self):
           super().__post_init__()
           self.scene.num_envs = 50
           self.scene.env_spacing = 2.5
           self.episode_length_s = 40.0
           self.scene.terrain.max_init_terrain_level = None
           self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
           self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
           self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
           self.observations.policy.enable_corruption = False
           self.events.base_external_force_torque = None
           self.events.push_robot = None
   ```

3. Register the environment with Gymnasium (following the H1 pattern):
   ```python
   # isaac_lab_env/open_duck_mini_v2/__init__.py
   import gymnasium as gym
   from . import agents

   gym.register(
       id="Isaac-Velocity-Rough-OpenDuck-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckRoughEnvCfg",
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckPPORunnerCfg",
       },
   )
   gym.register(
       id="Isaac-Velocity-Rough-OpenDuck-Play-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.env_cfg:OpenDuckRoughEnvCfg_PLAY",
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckPPORunnerCfg",
       },
   )
   ```

**Observation space (inherited from base, ~60 dims):**
- `base_lin_vel` (3) — body linear velocity in robot frame
- `base_ang_vel` (3) — body angular velocity in robot frame
- `projected_gravity` (3) — gravity direction in robot frame
- `velocity_commands` (3) — commanded (vx, vy, ωyaw)
- `joint_pos_rel` (16) — joint positions relative to default pose
- `joint_vel_rel` (16) — joint velocities
- `last_action` (16) — previous policy output

**How to test:**

*Automated test — `tests/test_isaac_lab_env.py`:*

```python
@pytest.mark.phase2
class TestIsaacLabEnv:

    def test_env_creates_successfully(self):
        """Environment must instantiate without errors."""
        import gymnasium as gym
        env = gym.make("Isaac-Velocity-Rough-OpenDuck-v0", num_envs=2)
        assert env is not None
        env.close()

    def test_env_step_produces_valid_output(self):
        """A single env.step() must return obs, reward, done, info."""
        import gymnasium as gym
        env = gym.make("Isaac-Velocity-Rough-OpenDuck-v0", num_envs=2)
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward.shape[0] == 2  # num_envs
        env.close()

    def test_env_resets_after_fall(self):
        """Environment must auto-reset when robot falls."""
        import gymnasium as gym
        import torch
        env = gym.make("Isaac-Velocity-Rough-OpenDuck-v0", num_envs=4)
        env.reset()
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(
                torch.zeros(4, 16, device="cuda")
            )
            if terminated.any():
                break
        assert terminated.any(), "Robot never fell — termination condition may be broken"
        env.close()
```

*Manual verification (on DGX Spark):*
- [ ] Launch env with `num_envs=16` without `--headless` to open Isaac Sim viewer
- [ ] Visually confirm 16 robots spawn on a flat plane
- [ ] Send random actions — robots should move chaotically and fall
- [ ] Confirm auto-reset: fallen robots reappear in standing position
- [ ] Check that the Jetson box is visible in the trunk of each robot

---

### Task 2.4 — Train Walking Policies

**Description:**
Train locomotion policies using PPO (primary) and optionally AMP (stretch goal). All training runs on the DGX Spark.

**Algorithm Plan:**

| Priority | Algorithm | Framework | Parallel Envs | Est. Time | Type | Why |
|---|---|---|---|---|---|---|
| 1 | **PPO** | RSL-RL | 4096 | ~30 min | On-policy | Proven baseline. All Isaac Lab locomotion examples use it. Built-in ONNX export for Jetson. |
| 2 (optional) | **AMP** | SKRL | 4096 | ~1-2 hr | On-policy + imitation | Produces natural gaits by imitating reference motions. Requires separate `DirectRLEnv` implementation + reference motion data. |

**Note on algorithm scope:** The Isaac Lab SKRL training script only supports `--algorithm PPO` and `--algorithm AMP` via the built-in CLI (verified in `scripts/reinforcement_learning/skrl/train.py` line 54: `choices=["AMP", "PPO", "IPPO", "MAPPO"]`). Other algorithms (SAC, TRPO, RPO, TD3) would require custom training scripts with no existing locomotion examples. We focus on PPO as the primary policy.

#### 2.4a — PPO (Proximal Policy Optimization) via RSL-RL

The primary policy. All Isaac Lab locomotion work is built on this. RSL-RL provides built-in ONNX export for TensorRT deployment on Jetson.

```python
# isaac_lab_env/open_duck_mini_v2/train_cfg.py

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class DuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training configuration for Open Duck Mini v2."""
    seed = 42
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "open_duck_ppo"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

```bash
# Train PPO via RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-OpenDuck-v0 \
    --headless --video --video_length 200 --video_interval 5000
```

#### 2.4b — AMP (Adversarial Motion Priors) via SKRL — optional stretch goal

AMP combines PPO-style RL with a GAN-like discriminator that rewards the policy for producing motions that resemble reference data. This produces more natural-looking gaits.

**Important:** AMP requires a **`DirectRLEnv`** (not `ManagerBasedRLEnv`). This means a separate environment implementation following the `humanoid_amp` example at `isaaclab_tasks/direct/humanoid_amp/`. This is a significant additional effort.

**Extra requirements:**
- Reference motion dataset from [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator)
- Separate `DirectRLEnv` environment class with AMP observation buffer
- SKRL YAML config file (not Python dict) with `models`, `memory`, `agent`, `trainer` sections

```yaml
# isaac_lab_env/open_duck_mini_v2/agents/skrl_amp_cfg.yaml
seed: 42
models:
  separate: False
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    network:
      - name: net
        input: OBSERVATIONS
        layers: [512, 256, 128]
        activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    network:
      - name: net
        input: OBSERVATIONS
        layers: [512, 256, 128]
        activations: elu
    output: ONE
  discriminator:
    class: DeterministicMixin
    network:
      - name: net
        input: AMP_OBSERVATIONS
        layers: [1024, 512]
        activations: relu
    output: ONE
memory:
  class: RandomMemory
  memory_size: 24
agent:
  class: AMP
  rollouts: 24
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 1.0e-03
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.01
  amp_batch_size: 512
  discriminator_batch_size: 4096
  discriminator_reward_scale: 2.0
  discriminator_loss_scale: 5.0
trainer:
  class: SequentialTrainer
  timesteps: 72000
```

```bash
# Train AMP via SKRL (only if DirectRLEnv is implemented)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Velocity-Rough-OpenDuck-AMP-v0 \
    --algorithm AMP --headless --video --video_length 200 --video_interval 5000
```

**Steps (for PPO, the primary path):**
1. Ensure Isaac Lab + RSL-RL are installed on DGX Spark
2. Run training with TensorBoard logging:
   ```bash
   tensorboard --logdir logs/
   ```
3. Key metrics to track:
   - `mean_reward` — should increase steadily
   - `mean_episode_length` — should increase (robot surviving longer)
   - `policy_loss` and `value_loss` — should decrease
4. Save the best checkpoint

**How to test:**

*Automated test — `tests/test_policy_inference.py`:*

```python
import pytest
import glob

@pytest.mark.phase2
class TestTrainedPolicies:

    def test_ppo_training_produces_checkpoint(self):
        """PPO training should produce at least one checkpoint."""
        checkpoints = glob.glob("logs/*open_duck_ppo*/model_*.pt") + \
                      glob.glob("logs/*open_duck_ppo*/checkpoints/*.pt")
        assert len(checkpoints) > 0, "No PPO checkpoint found"

    def test_ppo_final_episode_length_above_minimum(self):
        """PPO trained policy should survive > 5 seconds on average."""
        # 5 seconds at 50 Hz = 250 steps
        # Parse training logs for final mean_episode_length
        # assert final_ep_length > 250
        pass
```

*Manual verification (on DGX Spark):*
- [ ] PPO training run completes without crashes
- [ ] TensorBoard shows increasing reward and episode length
- [ ] Play PPO's best policy in Isaac Sim viewer (briefly)
- [ ] (Optional) AMP training completes if DirectRLEnv was implemented

---

### Task 2.5 — Evaluate and Select Best Policy

**Description:**
Evaluate the trained PPO policy (and AMP if available) using quantitative metrics and visual inspection. Confirm the policy is ready for deployment.

**Evaluation metrics:**

| Metric | How to Measure | Why It Matters |
|---|---|---|
| **Survival time** | Mean episode length from training logs | Basic viability — can the robot walk without falling? |
| **Velocity tracking** | RMS error between commanded and actual velocity over 30s | Can the robot follow speed/direction commands? |
| **Gait smoothness** | Mean squared jerk of joint actions (∑(a_t - 2*a_{t-1} + a_{t-2})²) | Smooth gaits are quieter, less stressful on servos, better for sim2real |
| **Push recovery** | Max lateral push velocity (m/s) survived without falling | Robustness to disturbances in the real world |
| **Energy efficiency** | Total torque integral per meter traveled (Nm·s/m) | Lower is better for battery life |
| **Visual quality** | Subjective 1-5 score from video review | Does it look like a duck walking or a robot convulsing? |

**Steps:**
1. Create an evaluation script `isaac_lab_env/open_duck_mini_v2/evaluate_policies.py` that:
   - Loads the best PPO checkpoint (and AMP if available)
   - Runs 10 episodes of 30 seconds each with forward walk command
   - Records all metrics above
   - Exports a comparison table
2. For push recovery: enable `push_by_setting_velocity` with increasing velocity magnitude and record the threshold
3. Record evaluation videos using:
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
       --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
       --checkpoint <path_to_best_model.pt> \
       --video --video_length 500
   ```
   Training progress videos are automatically saved by the `--video --video_length 200 --video_interval 5000` flags on the training run
4. Fill in the evaluation table:

```markdown
| Algorithm | Survival (s) | Vel. Track (m/s) | Smoothness | Push (m/s) | Energy | Visual (1-5) |
|-----------|-------------|-------------------|------------|------------|--------|--------------|
| PPO       |             |                   |            |            |        |              |
| AMP       |             |                   |            |            |        |              |
```

**How to test:**

*Automated test — `tests/test_policy_inference.py`:*

```python
@pytest.mark.phase2
class TestPolicyEvaluation:

    def test_evaluation_results_exist(self):
        """Evaluation results must be documented."""
        import os
        assert os.path.exists("docs/jetson-mod/algorithm_comparison.md"), \
            "Algorithm comparison document not found"

    def test_best_policy_survives_30_seconds(self):
        """The selected best policy must survive 30+ seconds in simulation."""
        # Load the chosen best policy
        # Run 10 episodes, each 30 seconds
        # All 10 should survive the full duration
        pass
```

*Manual verification:*
- [ ] Review evaluation metrics
- [ ] Watch policy videos — gait should look natural
- [ ] Document observations in `docs/jetson-mod/algorithm_comparison.md`
- [ ] Select **primary policy** (PPO) for deployment

**Output:** `docs/jetson-mod/algorithm_comparison.md` with evaluation results and observations.

---

### Task 2.6 — Export Best Policy to ONNX

**Description:**
Export the selected PPO policy to ONNX format for Jetson deployment.

**Steps:**
1. For RSL-RL trained PPO policy:
   ```python
   from isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx

   export_policy_as_onnx(
       policy=loaded_policy,
       path="exported_policies/",
       normalizer=obs_normalizer,  # if observation normalization was used
       filename="open_duck_ppo_policy.onnx",
   )
   ```
2. For SKRL trained AMP policy (if available):
   ```python
   import torch

   # Load the trained agent
   agent.load("path/to/best_checkpoint.pt")
   # Export the actor network
   dummy_input = torch.zeros(1, 60, device="cuda")  # obs dim from base config
   torch.onnx.export(
       agent.policy,
       dummy_input,
       "exported_policies/open_duck_amp_policy.onnx",
       input_names=["obs"],
       output_names=["actions"],
       opset_version=18,
   )
   ```
3. Verify each ONNX file:
   ```bash
   python -c "
   import onnxruntime as ort, numpy as np
   sess = ort.InferenceSession('exported_policies/open_duck_ppo_policy.onnx')
   print(f'inputs={[(i.name, i.shape) for i in sess.get_inputs()]}')
   dummy = np.zeros((1, 60), dtype=np.float32)
   out = sess.run(None, {sess.get_inputs()[0].name: dummy})
   print(f'output shape={out[0].shape}')
   assert out[0].shape[-1] == 16
   "
   ```

**How to test:**

*Automated test — `tests/test_policy_inference.py`:*

```python
@pytest.mark.phase2
class TestONNXExport:

    def test_at_least_one_onnx_exists(self):
        """At least one exported ONNX policy must exist."""
        import glob
        onnx_files = glob.glob("exported_policies/*.onnx")
        assert len(onnx_files) >= 1, "No ONNX files found"

    def test_onnx_produces_valid_actions(self):
        """ONNX model must accept observations and output 16-dim actions."""
        import onnxruntime as ort
        import numpy as np, glob
        for onnx_path in glob.glob("exported_policies/*.onnx"):
            sess = ort.InferenceSession(onnx_path)
            obs_dim = sess.get_inputs()[0].shape[-1]
            dummy = np.random.randn(1, obs_dim).astype(np.float32)
            out = sess.run(None, {sess.get_inputs()[0].name: dummy})
            assert out[0].shape[-1] == 16, f"{onnx_path}: unexpected output dim {out[0].shape}"
            assert not np.any(np.isnan(out[0])), f"{onnx_path}: produced NaN"
            assert not np.any(np.isinf(out[0])), f"{onnx_path}: produced Inf"
```

---

### Task 2.7 — Validate Best Policy in Isaac Sim

**Description:**
Run the selected best policy in Isaac Sim with full rendering to visually validate walking quality and confirm readiness for sim2real transfer. This is the critical gate before committing to hardware.

**Steps:**
1. Play the best policy with the Isaac Sim viewer:
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
       --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
       --checkpoint <path_to_best_checkpoint> \
       --num_envs 4
   ```
2. Test all walking commands (modify velocity ranges in `OpenDuckRoughEnvCfg_PLAY`):
   - Forward walk (0.5 m/s)
   - Backward walk (-0.3 m/s)
   - Sideways walk (0.3 m/s left/right)
   - Turning in place (0.5 rad/s)
   - Standing still (0 velocity)
3. Test robustness:
   - Re-enable push disturbances in the play config
   - Vary terrain roughness (if configured)

**How to test:**

*Manual verification (on DGX Spark — this is the critical gate):*
- [ ] Robot walks forward smoothly for 30+ seconds
- [ ] Robot can turn left and right while walking
- [ ] Robot recovers from small pushes
- [ ] Gait looks natural (no excessive wobbling, jerking, or foot dragging)
- [ ] Record videos for documentation
- [ ] Save observations to `docs/jetson-mod/validation_results.md`

**Pass/Fail criteria:**
- **PASS:** Policy walks in all directions for 30+ seconds, recovers from pushes → proceed to Phase 3
- **MARGINAL:** Policy walks but gait is poor → adjust reward weights and retrain
- **FAIL:** No walking achieved → debug actuator model (Task 2.2), check USD conversion (Task 2.1)

---

## Phase 3: CAD Redesign

> **Prerequisite:** Phase 2 must pass (either existing policies work, or retraining succeeded).

### Task 3.1 — Redesign `trunk_top`

**Description:**
Modify the trunk_top STL to add mounting provisions for the Jetson Orin Nano Dev Kit.

**Changes:**
- Add 4x M3 standoff mounting holes matching the Jetson dev kit hole pattern (58mm x 86mm, with 3.2mm holes)
- Add cable routing channels for: USB cable to servo driver, CSI ribbon to head, power cable
- Ensure clearance above and below the Jetson for the heatsink (top) and connectors (bottom/sides)

**Tools:** OnShape (web CAD, original project), FreeCAD, or Fusion 360

**How to test:**

*Automated test — `tests/test_cad_dimensions.py`:*

```python
@pytest.mark.phase3
class TestTrunkTopSTL:

    def test_trunk_top_stl_exists(self):
        """Modified trunk_top.stl must exist."""
        assert os.path.exists("print/trunk_top.stl")

    def test_jetson_fits_inside_trunk(self):
        """
        The trunk_top internal cavity must be at least
        113x100.5x43.77mm (Jetson 103x90.5x34.77 + 5mm clearance each side, + 4mm height for standoffs).
        """
        # Load trunk_top STL and compute internal cavity dimensions
        # This is approximate — measure bounding box and subtract wall thickness
        vertices = load_stl_vertices("print/trunk_top.stl")
        dims = vertices.max(axis=0) - vertices.min(axis=0)
        # Trunk top should be at least as large as before (125x100x41)
        assert dims[0] >= 125, f"trunk_top X too small: {dims[0]}"
        assert dims[1] >= 100, f"trunk_top Y too small: {dims[1]}"
```

*Manual verification:*
- [ ] Open the modified STL in a slicer (Cura, PrusaSlicer) — verify it is printable without support issues
- [ ] Measure the M3 hole spacing in the slicer — must match 58x86mm
- [ ] Physically test-fit a Jetson dev kit mockup (cardboard cutout 103x90.5mm) into a print of the part
- [ ] Verify USB-C, DC barrel jack, and CSI connectors are accessible

---

### Task 3.2 — Redesign `body_back` and `body_front` for Directed Ventilation

**Description:**
Add ventilation slots to `body_back` and matching inlet holes on `body_front` to create cross-flow ventilation through the **compute zone only** (not the battery zone). The airflow path should direct hot exhaust air away from the battery compartment and out the back of the robot.

**Context — Thermal Safety:**
The Jetson dissipates 20-22W at 25W mode. Its heatsink surface can reach 55-80°C under load. 18650 Li-ion cells degrade above 45°C and risk venting/damage above 60°C. The trunk cavity must be thermally zoned to protect batteries. This task works together with Task 3.6 (thermal partition wall) and Task 3.4 (battery pack isolation) to form a complete thermal management solution.

**Changes to `body_back`:**
- Add hex-pattern ventilation cutouts (~40x20mm) **positioned only on the compute zone side** (the side where the Jetson sits, opposite the battery compartment)
- Keep the battery zone side of `body_back` **solid** (no vents) to prevent hot air recirculating toward batteries
- Optional: add a cutout for the DC barrel jack or USB-C connector on the compute side

**Changes to `body_front`:**
- Add matching inlet ventilation slots (~30x15mm) on the compute zone side of `body_front`
- Inlet holes should be positioned to create a cross-flow path: air enters front → flows over Jetson heatsink → exits rear vents
- Keep the battery zone side of `body_front` **solid**

**Airflow diagram (top view):**
```
         body_front                              body_back
    ┌────────────────────────────────────────────────────┐
    │  SOLID (battery zone)          SOLID (battery zone)│
    │────────────── PARTITION WALL ──────────────────────│
    │░░░INLET░░░  →  JETSON + FAN  →  ░░░EXHAUST░░░░░░░│
    │  (compute zone)                (compute zone)      │
    └────────────────────────────────────────────────────┘
         AIR IN →                              → AIR OUT
```

**How to test:**

*Manual verification:*
- [ ] Print both parts and visually inspect ventilation holes
- [ ] Confirm holes do not weaken the structural integrity of either part
- [ ] Blow air through — vents should allow free cross-flow airflow on compute side only
- [ ] Verify battery zone side has no openings that would allow hot air ingress
- [ ] With Jetson fan running, hold tissue paper near rear vents to confirm exhaust flow

---

### Task 3.3 — Redesign `body_middle_top` and `body_middle_bottom` (if needed)

**Description:**
If clearance analysis from Phase 2 shows the Jetson protrudes beyond the current body cavity, extend these parts by 10-15mm in the X (depth) dimension.

**Decision criteria:** Almost certainly needed — the full Jetson dev kit is 34.77mm tall (including heatsink + fan), so with standoffs (~3-5mm) the total is ~38-40mm. This is likely to exceed the available vertical space between trunk_bottom and body_middle_top.

**How to test:**

*Automated test:*

```python
@pytest.mark.phase3
class TestBodyMiddleDimensions:

    def test_body_fits_together(self):
        """body_middle_top and body_middle_bottom Y dimensions must match."""
        top_verts = load_stl_vertices("print/body_middle_top.stl")
        bot_verts = load_stl_vertices("print/body_middle_bottom.stl")
        top_y = top_verts.max(axis=0)[1] - top_verts.min(axis=0)[1]
        bot_y = bot_verts.max(axis=0)[1] - bot_verts.min(axis=0)[1]
        assert abs(top_y - bot_y) < 1.0, "Top and bottom Y dimensions don't match"
```

*Manual verification:*
- [ ] Print both parts and test-fit together
- [ ] Verify screw holes still align with trunk mounting points

---

### Task 3.4 — Redesign Battery Pack (Thermally Isolated)

**Description:**
Redesign the battery_pack_lid to hold 6x 18650 cells (3S2P configuration, 11.1V nominal) instead of 2x, and ensure the battery pack is thermally isolated from the Jetson compute zone. Note: the DC-DC boost converter is placed in the **compute zone** (near the Jetson) for a short 19V cable run, not in the battery zone.

**Thermal isolation requirements:**
- The battery pack must sit entirely within the **battery zone** (behind the thermal partition wall from Task 3.6)
- Add a **slot or lip** on the battery pack housing that mates with the thermal partition wall, creating a sealed boundary
- Include a **thermistor mounting point** (small clip or channel) to hold a NTC 10kΩ thermistor against one of the 18650 cells for runtime temperature monitoring (see Task 4.6)
- Route power cables from battery zone to compute zone through a **small notch** in the partition wall (just large enough for wires, not airflow)

**How to test:**

*Manual verification:*
- [ ] Print and test-fit with 6x 18650 cells (3S2P)
- [ ] Verify cells are held securely and cannot rattle
- [ ] Confirm DC-DC converter mounting location and wire routing
- [ ] Verify partition wall slot aligns with Task 3.6 thermal partition
- [ ] Confirm thermistor mounting clip holds sensor snugly against a cell
- [ ] Verify cable routing notch is <5mm diameter (minimal thermal leakage)

---

### Task 3.5 — Update Simulation Model with Final CAD Meshes

**Description:**
Replace the placeholder Jetson box STL with the actual redesigned part meshes. Re-run all Phase 2 tests.

**How to test:**

Run the full Phase 2 test suite:
```bash
pytest tests/test_simulation_stability.py tests/test_policy_inference.py -v
```

All tests must pass with the final meshes.

---

### Task 3.6 — Design Thermal Partition Wall

**Description:**
Design and print a partition wall that divides the trunk cavity into two thermally isolated zones: a **battery zone** (back/rear of trunk) and a **compute zone** (front/mid of trunk where the Jetson is mounted). This is the primary thermal safety measure preventing Jetson heat from reaching the 18650 battery cells.

**Context — Why this is critical:**
The Jetson Orin Nano dissipates 20-22W at 25W power mode. Its heatsink surface temperature can reach 55-80°C under sustained AI workloads. 18650 Li-ion cells begin accelerated degradation above 45°C, risk gas formation and venting above 60°C, and enter thermal runaway above ~130°C. While the gap between heatsink temp and battery danger zone is not immediately catastrophic, sustained operation in an enclosed trunk cavity without a thermal barrier could push battery temperatures into the damage zone (60°C+), especially during prolonged Cosmos VLM inference at 25W mode.

**Design specifications:**

```
    TRUNK CAVITY (top view)
    ┌──────────────────┬───────────────────────────────┐
    │  BATTERY ZONE    │  COMPUTE ZONE                 │
    │  (cool, sealed)  │  (hot, ventilated)            │
    │                  │                                │
    │  [6x 18650]      │  [JETSON ORIN NANO]            │
    │  [BMS]           │  [DC-DC conv]                  │
    │                  │  [SERVO DRIVER]                │
    │                  │  [IMU]                         │
    │                  │  [FAN → exhaust vents]         │
    │                  │                                │
    │  ~50mm depth     │  ~100mm depth                  │
    └──────────────────┴───────────────────────────────┘
         BACK (-X)    PARTITION     FRONT (+X)
                     (~X = -0.08)
```

**Partition wall construction:**
- **Base material:** PLA, 2mm thick, printed to span the full width (Y: 110mm) and height (Z: ~90mm) of the trunk cavity
- **Thermal insulation:** Bond a 1mm **mica sheet** (cut to size) to the compute-zone-facing side of the partition wall
  - Mica thermal conductivity: ~0.5-0.7 W/mK (good insulator; note: silicone thermal pads are designed to *conduct* heat, not insulate — mica is the correct choice here)
  - Mica density: ~2.2-2.3 g/cm³ → a 110x90x1mm sheet weighs ~12g
  - Mica is flame resistant (rated to 500°C+), electrically insulating
  - Total partition assembly weight: ~37g (25g PLA wall + 12g mica sheet)
- **Mounting:** Slot into grooves in `trunk_top` and `trunk_bottom` (add matching slots in Task 3.1 and existing trunk_bottom redesign)
- **Cable pass-through:** Small notch (~5mm wide) at the bottom for power cables from battery zone to compute zone. Notch should be as small as possible to minimize thermal leakage; fill remaining gap with thermal insulation tape if needed
- **Estimated position:** X ≈ -0.08m in trunk frame (between BMS at -0.075 and servo driver at 0.001)

**New file:** `print/thermal_partition.stl`

**How to test:**

*Automated test — `tests/test_cad_dimensions.py`:*

```python
@pytest.mark.phase3
class TestThermalPartition:

    def test_partition_stl_exists(self):
        """Thermal partition wall STL must exist."""
        assert os.path.exists("print/thermal_partition.stl")

    def test_partition_spans_cavity(self):
        """Partition must span the full width and height of the trunk cavity."""
        vertices = load_stl_vertices("print/thermal_partition.stl")
        dims = vertices.max(axis=0) - vertices.min(axis=0)
        # Should span at least 100mm in Y (width) and 80mm in Z (height)
        assert dims[1] >= 100, f"Partition Y (width) too small: {dims[1]}"
        assert dims[2] >= 80, f"Partition Z (height) too small: {dims[2]}"
        # Should be thin: 2-4mm in X (depth)
        assert dims[0] <= 5.0, f"Partition X (thickness) too large: {dims[0]}"

    def test_partition_has_cable_notch(self):
        """Partition should not be a perfect rectangle — it needs a cable pass-through notch."""
        vertices = load_stl_vertices("print/thermal_partition.stl")
        # A simple rectangle would have 12 triangles (2 per face × 6 faces)
        # A notch adds geometry — expect more than 12 triangles
        assert len(vertices) > 36, "Partition appears to be a simple box — missing cable notch?"
```

*Manual verification:*
- [ ] Print the partition wall and verify it fits snugly into the trunk cavity
- [ ] Verify the mica/silicone sheet can be bonded flush to the compute-side face
- [ ] Verify cable notch allows power cables through but is not oversized
- [ ] Test-fit with battery pack (Task 3.4) — partition should mate with battery housing slot
- [ ] Test-fit with trunk_top and trunk_bottom — partition should slide into grooves
- [ ] With Jetson running at 25W for 10 minutes, measure temperature on both sides of the partition with an IR thermometer — battery side should be >15°C cooler than compute side

---

### Task 3.7 — Update `trunk_top` and `trunk_bottom` for Partition Wall Slots

**Description:**
Add matching slots/grooves in `trunk_top` and `trunk_bottom` to accept the thermal partition wall from Task 3.6. This is a follow-up modification to Task 3.1.

**Changes:**
- Add a 2.5mm wide slot (for 2mm wall + tolerance) running across the full Y-width of both `trunk_top` and `trunk_bottom` at X ≈ -0.08m
- Slot depth: 2-3mm into the wall (enough to hold the partition securely)
- Ensure slots align vertically so the partition slides in during assembly

**How to test:**

*Manual verification:*
- [ ] Print modified trunk_top and trunk_bottom
- [ ] Thermal partition from Task 3.6 slides into slots with light friction fit
- [ ] Partition stands upright without additional fasteners
- [ ] No interference with Jetson mounting standoffs (Task 3.1) or battery pack (Task 3.4)

---

## Phase 4: Hardware Build & Deployment

> **Prerequisite:** Phase 2 and Phase 3 fully validated.

### Task 4.1 — Procure Hardware

**Shopping list:**

| Item | Qty | Est. Cost | Source |
|---|---|---|---|
| NVIDIA Jetson Orin Nano Super Developer Kit | 1 | $249 | Amazon (B0BZJTQ5YP) |
| Feetech STS3250 servos | 14 | $280 | Feetech/AliExpress |
| 18650 Li-ion cells (e.g., Samsung 30Q) | 6 | $45 | Amazon/18650batterystore |
| 3S BMS (>=15A) | 1 | $8 | Amazon/AliExpress |
| DC-DC boost converter XL6009/XL6019 (11.1V to 19V, 4A+) | 1 | $5 | Amazon/AliExpress |
| CSI camera module (IMX219) | 1 | $15 | Amazon/Arducam |
| CSI ribbon cable 30cm | 1 | $5 | Amazon |
| M3 standoffs + screws (assorted) | 1 set | $8 | Amazon |
| XT30 to DC barrel adapter cable | 1 | $5 | Amazon |
| Mica insulation sheet (50x100mm, 1mm thick) | 1 | $3 | Amazon/AliExpress |
| NTC 10kΩ thermistor (3950B, with leads) | 1 | $1 | Amazon/AliExpress |
| Thermal insulation tape (kapton or silicone) | 1 roll | $5 | Amazon |

**Total estimated additional cost:** ~$627

**How to test:**
- [ ] Verify all items received and undamaged
- [ ] Power on Jetson dev kit standalone — verify it boots to JetPack OS
- [ ] Check CSI camera connects and captures frames: `nvgstcapture-1.0`

---

### Task 4.2 — 3D Print Modified Parts

**Parts to print:**

| Part | Material | Est. Time | Notes |
|---|---|---|---|
| trunk_top (modified) | PLA | ~3 hr | M3 inserts for Jetson standoffs + partition wall slot |
| trunk_bottom (modified) | PLA | ~2 hr | Cable pass-throughs + partition wall slot |
| body_back (modified) | PLA | ~1.5 hr | Ventilation slots (compute zone side only) |
| body_front (modified) | PLA | ~1.5 hr | Inlet vents (compute zone side only) |
| body_middle_top (if modified) | PLA | ~3 hr | Extended depth |
| body_middle_bottom (if modified) | PLA | ~3 hr | Extended depth |
| battery_pack_lid (modified) | PLA | ~1.5 hr | 6-cell (3S2P) capacity + thermistor clip |
| thermal_partition (new) | PLA | ~1 hr | Thermal barrier wall between battery and compute zones |

**How to test:**
- [ ] Each printed part passes visual inspection — no warping, layer adhesion good
- [ ] M3 inserts install cleanly with soldering iron
- [ ] Test-fit all modified parts together before full assembly
- [ ] Verify screw holes align with existing unmodified parts (legs, hip mounts, etc.)
- [ ] Thermal partition slides into trunk_top/trunk_bottom slots with light friction fit
- [ ] Verify body_front inlet vents and body_back exhaust vents are aligned to compute zone only

---

### Task 4.3 — Assemble Modified Robot

**Wiring reference:** See `docs/jetson-mod/jetson_wiring_diagram.drawio` for the full wiring diagram. Open in [diagrams.net](https://app.diagrams.net) or VS Code draw.io extension.

**Steps:**
1. Disassemble the head: remove Pi Zero 2W, retain SG90 servos and wiring
2. **Install thermal partition wall:** Slide the thermal partition (Task 3.6) into the trunk_top/trunk_bottom slots. Bond the mica insulation sheet to the compute-zone-facing side using high-temp adhesive or thermal tape
3. Mount Jetson dev kit in trunk **compute zone** using M3 standoffs — ensure the dev kit fan exhaust points toward the body_back ventilation slots
4. Mount the 6-cell (3S2P) battery pack in the **battery zone** (behind the partition wall). Attach the NTC thermistor to one of the middle cells using thermal tape or the printed clip (Task 3.4)
5. Wire power: Battery (3S2P, 11.1V) → 3S BMS → DC-DC boost (11.1V → 19V) → route power cable through partition wall notch → Jetson DC barrel jack
6. Wire thermistor: Route thermistor leads through partition wall notch → Jetson GPIO ADC pin (or external ADC like ADS1115 on I2C)
7. Wire servo driver board: USB cable from trunk-mounted servo driver to Jetson USB port
8. Wire IMU (BNO055): I2C from trunk-mounted IMU to Jetson GPIO (SDA=pin 3, SCL=pin 5)
9. Route CSI ribbon cable from head through neck to Jetson CSI connector
10. Wire foot switches: GPIO wires from feet to Jetson GPIO header
11. Wire antenna servos: PWM from Jetson GPIO through neck to head SG90s
12. Wire eye LEDs: GPIO from Jetson through neck to head LEDs (or use a small I2C LED driver in the head)
13. Seal the partition wall cable notch with thermal insulation tape (kapton or silicone tape) around the wire bundle
14. Reassemble body panels — verify body_front inlet and body_back exhaust vents are unobstructed

**How to test:**

*Integration tests (on the Jetson):*
- [ ] Jetson boots from battery power (not wall adapter)
- [ ] `i2cdetect -y 1` shows BNO055 at address 0x28 or 0x29
- [ ] `ls /dev/ttyUSB0` or `/dev/ttyACM0` shows servo driver connected
- [ ] Camera test: `nvgstcapture-1.0` shows live preview
- [ ] GPIO test: Toggle each LED on/off with `gpioset`
- [ ] GPIO test: Read foot switch state with `gpioget`
- [ ] Servo test: Command each servo to center position and verify movement
- [ ] **Thermal test:** Run Jetson at 25W mode for 15 minutes (e.g., `stress-ng --cpu 6 --gpu 1`). Measure temperature on both sides of the thermal partition with IR thermometer. Battery side must stay below 40°C
- [ ] **Thermistor test:** Read NTC thermistor value via GPIO/ADC — should report reasonable ambient temperature (~20-30°C). Verify value rises when battery is warmed manually

---

### Task 4.4 — Port Runtime Software to Jetson with TensorRT

**Description:**
Adapt the [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) to run on the Jetson Orin Nano, using TensorRT for GPU-accelerated policy inference.

**Key changes:**

| Component | Pi Zero 2W | Jetson Orin Nano |
|---|---|---|
| OS | Raspberry Pi OS | JetPack (Ubuntu-based) |
| Python | System python | Conda/venv with CUDA support |
| Policy inference | `onnxruntime` (CPU) | **TensorRT** (GPU, sub-ms latency) |
| GPIO library | `RPi.GPIO` or `gpiod` | `Jetson.GPIO` (compatible API) |
| I2C | `smbus2` | `smbus2` (same) |
| Serial (servos) | `pyserial` via `/dev/ttyUSB0` | `pyserial` via `/dev/ttyUSB0` (same) |
| PWM (antennas) | Hardware PWM on GPIO 12/13 | Jetson PWM on GPIO 32/33 (remap needed) |

**Step 1: Convert ONNX to TensorRT engine:**
```bash
# On the Jetson (TensorRT must run on target hardware)
/usr/src/tensorrt/bin/trtexec \
    --onnx=open_duck_walk_policy.onnx \
    --saveEngine=open_duck_walk_policy.trt \
    --fp16 \
    --verbose
```
This produces a `.trt` engine file optimized for the Jetson's GPU. FP16 mode gives ~2x speedup with negligible accuracy loss for an MLP policy.

**Step 2: Create TensorRT inference wrapper:**
```python
# jetson_runtime/trt_infer.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # Allocate GPU buffers for input/output
        self.d_input = cuda.mem_alloc(1 * 56 * 4)   # 56 floats, FP32
        self.d_output = cuda.mem_alloc(1 * 16 * 4)   # 16 floats, FP32

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32).ravel()
        cuda.memcpy_htod(self.d_input, obs)
        self.context.execute_v2([int(self.d_input), int(self.d_output)])
        output = np.empty(16, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)
        return output
```

**Step 3: Pin mapping (Pi header → Jetson 40-pin header):**

The Jetson Orin Nano dev kit's 40-pin header is largely Pi-compatible, but pin functions differ. Create a new mapping file.

| Function | Pi Zero GPIO | Jetson GPIO | Jetson Pin |
|---|---|---|---|
| Left Eye LED | GPIO 23 | GPIO 23 (or remap) | Pin 16 |
| Right Eye LED | GPIO 24 | GPIO 24 (or remap) | Pin 18 |
| Projector LED | GPIO 25 | GPIO 25 (or remap) | Pin 22 |
| Left Antenna PWM | GPIO 12 | GPIO 12 (or remap) | Pin 32 |
| Right Antenna PWM | GPIO 13 | GPIO 13 (or remap) | Pin 33 |
| Left Foot Switch | GPIO 22 | GPIO 22 (or remap) | Pin 15 |
| Right Foot Switch | GPIO 27 | GPIO 27 (or remap) | Pin 13 |
| IMU SDA | GPIO 2 | GPIO 2 | Pin 3 |
| IMU SCL | GPIO 3 | GPIO 3 | Pin 5 |

> NOTE: Verify exact pin mappings against the Jetson Orin Nano dev kit carrier board pinout diagram before wiring.

**How to test:**

*Unit tests (run on Jetson):*
```bash
# Test TensorRT engine loads and runs
python -c "
from jetson_runtime.trt_infer import TRTInfer
import numpy as np
policy = TRTInfer('open_duck_walk_policy.trt')
obs = np.random.randn(56).astype(np.float32)
action = policy.infer(obs)
print('Action shape:', action.shape)
print('Action values:', action)
assert action.shape == (16,)
assert not np.any(np.isnan(action))
print('PASS: TensorRT inference works')
"
```

```bash
# Test TensorRT inference latency (must be < 1ms for 50 Hz control loop)
python -c "
from jetson_runtime.trt_infer import TRTInfer
import numpy as np, time
policy = TRTInfer('open_duck_walk_policy.trt')
obs = np.random.randn(56).astype(np.float32)
# Warm up
for _ in range(100): policy.infer(obs)
# Benchmark
t0 = time.time()
for _ in range(1000): policy.infer(obs)
avg_ms = (time.time() - t0) / 1000 * 1000
print(f'Average inference: {avg_ms:.3f} ms')
assert avg_ms < 1.0, f'Too slow: {avg_ms:.3f} ms'
print('PASS: TensorRT inference under 1ms')
"
```

```bash
# Fallback: Test ONNX GPU inference (if TensorRT conversion fails)
python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers()
sess = ort.InferenceSession('open_duck_walk_policy.onnx',
                            providers=['CUDAExecutionProvider'])
import numpy as np
inp = np.zeros((1, 56), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
print('Output shape:', out[0].shape)
print('PASS: ONNX GPU inference works')
"
```

```bash
# Test Jetson GPIO
python -c "
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
GPIO.output(23, GPIO.HIGH)
import time; time.sleep(0.5)
GPIO.output(23, GPIO.LOW)
GPIO.cleanup()
print('PASS: GPIO works')
"
```

```bash
# Test servo communication
python -c "
from mini_bdx_runtime.hwi import HWI
hwi = HWI('/dev/ttyUSB0')
positions = hwi.get_present_position()
print('Servo positions:', positions)
print('PASS: Servo communication works')
"
```

*Integration test:*
- [ ] Run the full walk script with TensorRT policy
- [ ] Robot should stand up and respond to gamepad/keyboard commands
- [ ] Measure control loop frequency — must be ≥ 50 Hz (target: 200+ Hz on Jetson with TensorRT)
- [ ] Monitor Jetson thermals: `tegrastats` — temperature should stay below 70C

---

### Task 4.5 — Real Robot Walking Test

**Description:**
Deploy the walking policy on the physical Jetson-modified robot and verify sim2real transfer.

**How to test:**

*Manual verification:*
- [ ] Place robot on flat surface with safety support (held loosely by hand or on a stand)
- [ ] Start the walking script
- [ ] Robot should balance and stand
- [ ] Command forward walk — robot should take steps
- [ ] Record video from multiple angles
- [ ] Compare against videos of the original Pi Zero version
- [ ] Test battery life — record time from full charge to Jetson shutdown

**Pass/Fail criteria:**
- **PASS:** Robot walks forward for 10+ seconds on flat ground without falling
- **MARGINAL:** Robot can stand but walking is unstable — may need more sim2real tuning
- **FAIL:** Robot cannot stand — review Phase 2 validation, check wiring, re-measure mass

---

### Task 4.6 — Software Thermal Management

**Description:**
Implement runtime software that monitors battery temperature via the NTC thermistor and manages Jetson power modes to prevent battery overheating. This complements the physical thermal management (partition wall + ventilation) with an active software safety layer.

**Context:**
Even with the thermal partition wall (Task 3.6) and directed ventilation (Task 3.2), software thermal management adds defense-in-depth. The Jetson's built-in thermal sensors only monitor the SoC — they don't know how hot the batteries are. An external thermistor on the battery pack closes this gap.

**Power mode strategy:**

| Mode | TDP | Heatsink Temp | When to Use |
|---|---|---|---|
| 7W eco | 7W | ~35-40°C | Idle / standing / battery temp > 38°C |
| 15W default | 15W | ~45-55°C | Locomotion policy only (normal operation) |
| 25W super | 25W | ~65-80°C | Cosmos VLM inference (duty-cycled, battery temp < 35°C) |

**Implementation:**

```python
# jetson_runtime/thermal_manager.py
import subprocess
import time
import threading
import logging

logger = logging.getLogger(__name__)

# Temperature thresholds (°C)
BATTERY_TEMP_WARNING = 35.0    # Switch to 15W if above this
BATTERY_TEMP_CRITICAL = 40.0   # Switch to 7W if above this
BATTERY_TEMP_SHUTDOWN = 50.0   # Emergency shutdown if above this
JETSON_TEMP_THROTTLE = 85.0    # Reduce to 15W if Jetson SoC above this

# NVP model names for Jetson Orin Nano Super
POWER_MODES = {
    "7W":  "MAXN",       # Adjust based on actual nvpmodel names
    "15W": "15W",
    "25W": "25W_SUPER",
}


class ThermalManager:
    """
    Monitors battery temperature (via NTC thermistor) and Jetson SoC temperature.
    Automatically adjusts power mode to keep batteries safe.
    """

    def __init__(self, thermistor_adc_channel=0, poll_interval_s=5.0):
        self.thermistor_channel = thermistor_adc_channel
        self.poll_interval = poll_interval_s
        self.current_mode = "15W"
        self.battery_temp = 25.0
        self.jetson_temp = 40.0
        self._running = False
        self._thread = None

    def read_battery_temp(self) -> float:
        """Read NTC thermistor via ADC (ADS1115 on I2C or Jetson ADC pin).

        Returns temperature in °C.
        """
        # TODO: Implement actual ADC reading
        # For ADS1115: import board, busio, adafruit_ads1x15.ads1115
        # Convert ADC voltage to temperature using Steinhart-Hart equation
        # For NTC 10kΩ 3950B: R = R0 * exp(B * (1/T - 1/T0))
        raise NotImplementedError("Wire up ADC reading for thermistor")

    def read_jetson_temp(self) -> float:
        """Read Jetson SoC temperature from thermal zones."""
        try:
            with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            return 40.0  # Safe default

    def set_power_mode(self, mode: str):
        """Switch Jetson power mode via nvpmodel."""
        if mode == self.current_mode:
            return
        # Map mode name to nvpmodel ID (verify with `nvpmodel -p --all`)
        mode_ids = {"7W": 0, "15W": 1, "25W": 2}  # Adjust for actual system
        mode_id = mode_ids.get(mode, 1)
        try:
            subprocess.run(
                ["sudo", "nvpmodel", "-m", str(mode_id)],
                check=True, capture_output=True, timeout=10
            )
            logger.info(f"Power mode changed: {self.current_mode} -> {mode}")
            self.current_mode = mode
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set power mode {mode}: {e}")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.battery_temp = self.read_battery_temp()
                self.jetson_temp = self.read_jetson_temp()

                # Emergency shutdown
                if self.battery_temp >= BATTERY_TEMP_SHUTDOWN:
                    logger.critical(
                        f"BATTERY TEMP CRITICAL: {self.battery_temp}°C — SHUTTING DOWN"
                    )
                    subprocess.run(["sudo", "shutdown", "-h", "now"])
                    return

                # Thermal throttling logic
                if self.battery_temp >= BATTERY_TEMP_CRITICAL:
                    self.set_power_mode("7W")
                elif (self.battery_temp >= BATTERY_TEMP_WARNING
                      or self.jetson_temp >= JETSON_TEMP_THROTTLE):
                    self.set_power_mode("15W")
                # Don't auto-escalate to 25W — that's done explicitly by VLM code

            except NotImplementedError:
                logger.warning("Thermistor ADC not implemented — using Jetson temp only")
                self.jetson_temp = self.read_jetson_temp()
                if self.jetson_temp >= JETSON_TEMP_THROTTLE:
                    self.set_power_mode("15W")
            except Exception as e:
                logger.error(f"Thermal monitor error: {e}")

            time.sleep(self.poll_interval)

    def start(self):
        """Start background thermal monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Thermal manager started (polling every %.1fs)", self.poll_interval)

    def stop(self):
        """Stop background thermal monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def request_boost(self) -> bool:
        """Request 25W mode for VLM inference. Returns True if safe to boost."""
        if self.battery_temp < BATTERY_TEMP_WARNING:
            self.set_power_mode("25W")
            return True
        else:
            logger.warning(
                f"Boost denied — battery temp {self.battery_temp}°C >= {BATTERY_TEMP_WARNING}°C"
            )
            return False

    def release_boost(self):
        """Return to 15W after VLM inference completes."""
        self.set_power_mode("15W")
```

**Integration with Cosmos VLM (Task 5.3):**

The `autonomous_walk.py` control loop should use `ThermalManager.request_boost()` before each VLM inference call and `release_boost()` after:

```python
# In the main control loop:
thermal = ThermalManager()
thermal.start()

# When VLM query is needed:
if thermal.request_boost():
    vlm_result = cosmos.query(camera_frame, prompt)
    thermal.release_boost()
else:
    # Use last known VLM command or stop
    vlm_result = last_known_command
```

**How to test:**

*Unit tests (run on Jetson):*
```bash
# Test Jetson temperature reading
python -c "
from jetson_runtime.thermal_manager import ThermalManager
tm = ThermalManager()
jetson_temp = tm.read_jetson_temp()
print(f'Jetson SoC temp: {jetson_temp}°C')
assert 20 < jetson_temp < 100, f'Implausible temperature: {jetson_temp}'
print('PASS: Jetson temp reading works')
"
```

```bash
# Test power mode switching
python -c "
from jetson_runtime.thermal_manager import ThermalManager
tm = ThermalManager()
tm.set_power_mode('15W')
print(f'Current mode: {tm.current_mode}')
assert tm.current_mode == '15W'
tm.set_power_mode('7W')
assert tm.current_mode == '7W'
tm.set_power_mode('15W')  # Restore default
print('PASS: Power mode switching works')
"
```

*Integration test:*
- [ ] Start thermal manager, run Jetson at 25W for 10 minutes
- [ ] Verify battery temperature stays below 35°C with partition wall installed
- [ ] Simulate high battery temp (warm the thermistor with a hair dryer) — verify auto-throttle to 7W
- [ ] Verify `request_boost()` returns `False` when battery temp is above threshold
- [ ] Monitor with `tegrastats` — verify power mode transitions are logged

---

## Phase 5: Cosmos Reason2 VLM Integration (Physical AI)

> **Prerequisite:** Phase 4 complete — robot walks with locomotion policy on Jetson. Camera mounted and working.

### Background

The locomotion policy (Phase 2-4) makes the robot walk, but it's blind — someone must manually send velocity commands. Cosmos Reason2-2B adds vision and reasoning: the robot sees through its camera, understands natural language commands, reasons about the physical world, and autonomously decides where to walk.

**Architecture:**

```
                   ┌──────────────────────────────────────┐
                   │  Voice / Text command                 │
                   │  "Walk to the red cup"                │
                   └──────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│  COSMOS REASON2-2B (W4A16-Edge2)              ~2-3 Hz, ~5.8 GB │
│                                                                 │
│  Input:  camera frame (640x480) + text command                  │
│  Processing: chain-of-thought reasoning about scene             │
│  Output: text containing JSON velocity command + behavior flags │
│                                                                 │
│  Example output:                                                │
│  "I see a red cup on the right side, about 1.5m away.          │
│   I should walk forward and slightly right.                     │
│   {"forward": 0.2, "lateral": 0.0, "turn": -0.1}              │
│   BEHAVIOR: normal"                                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ parsed: vx=0.2, vy=0.0, yaw=-0.1
┌─────────────────────────────────▼───────────────────────────────┐
│  LOCOMOTION POLICY (TensorRT)                 50 Hz, <0.1 GB   │
│  Input:  velocity cmd + joint angles + IMU + foot contacts      │
│  Output: 16 joint position targets → servos                     │
└─────────────────────────────────────────────────────────────────┘
```

**Memory budget on Jetson Orin Nano Super (8 GB):**

```
Cosmos Reason2 (W4A16-Edge2):  ~5.8 GB
Locomotion policy (TensorRT):  ~0.1 GB
Camera pipeline:               ~0.3 GB
OS + system:                   ~1.5 GB
────────────────────────────────────────
Total:                         ~7.7 GB  ✓ fits
```

---

### Task 5.1 — Set Up Cosmos Reason2 on Jetson

**Description:**
Install and run the quantized Cosmos Reason2-2B model on the Jetson Orin Nano Super using the Jetson-optimized vLLM Docker container.

**Model:** `embedl/Cosmos-Reason2-2B-W4A16-Edge2` (INT4 weights, FP16 activations)

**Steps:**
1. Pull the Jetson-optimized vLLM Docker image:
   ```bash
   docker pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
   ```
2. Test basic text+image inference:
   ```bash
   docker run --rm -it \
     --runtime=nvidia \
     --network host \
     --shm-size=4g \
     -e HF_TOKEN=$HF_TOKEN \
     ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
     vllm serve "embedl/Cosmos-Reason2-2B-W4A16-Edge2" \
       --max-model-len 2048 \
       --gpu-memory-utilization 0.70 \
       --max-num-seqs 1
   ```
3. From another terminal, test with a sample image:
   ```python
   import requests, base64

   image_b64 = base64.b64encode(open("test_image.jpg", "rb").read()).decode()
   response = requests.post("http://localhost:8000/v1/chat/completions", json={
       "model": "embedl/Cosmos-Reason2-2B-W4A16-Edge2",
       "messages": [{"role": "user", "content": [
           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
           {"type": "text", "text": "Describe what you see in this image."},
       ]}],
       "max_tokens": 256,
   })
   print(response.json()["choices"][0]["message"]["content"])
   ```
4. Verify memory usage with `tegrastats` — total should be < 7.5 GB
5. Verify inference speed — should get ~16-17 tokens/sec

**How to test:**

*Automated test (run on Jetson):*

```python
@pytest.mark.phase5
class TestCosmosSetup:

    def test_vllm_server_responds(self):
        """vLLM server must respond to health check."""
        import requests
        resp = requests.get("http://localhost:8000/health")
        assert resp.status_code == 200

    def test_text_only_inference(self):
        """Model must respond to text-only prompt."""
        import requests
        resp = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "embedl/Cosmos-Reason2-2B-W4A16-Edge2",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 32,
        })
        assert resp.status_code == 200
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0

    def test_image_inference(self):
        """Model must reason about a camera image."""
        import requests, base64
        # Capture a test frame from CSI camera
        image_b64 = capture_and_encode_frame()
        resp = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "embedl/Cosmos-Reason2-2B-W4A16-Edge2",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "Describe what you see."},
            ]}],
            "max_tokens": 128,
        })
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["message"]["content"]
        assert len(text) > 20  # Should produce a meaningful description

    def test_memory_under_limit(self):
        """Total system memory usage must be < 7.5 GB during inference."""
        # Parse tegrastats or /proc/meminfo
        # Total used RAM should be < 7.5 GB
        pass
```

*Manual verification:*
- [ ] vLLM server starts without OOM errors
- [ ] `tegrastats` shows RAM < 7.5 GB during inference
- [ ] Point camera at various objects — model describes them correctly
- [ ] Inference responds within ~500ms per query (16-17 tok/s)

---

### Task 5.2 — Build the Command Parser

**Description:**
Create a Python module that takes Cosmos Reason2's text output and extracts structured velocity commands for the locomotion policy.

**File:** `jetson_runtime/cosmos_commander.py`

**Implementation:**

```python
# jetson_runtime/cosmos_commander.py
import re
import json
import requests
import base64
import numpy as np
from dataclasses import dataclass

@dataclass
class RobotCommand:
    forward: float = 0.0     # m/s, range [-0.2, 0.3]
    lateral: float = 0.0     # m/s, range [-0.2, 0.2]
    turn: float = 0.0        # rad/s, range [-0.3, 0.3]
    behavior: str = "normal" # normal, excited, cautious, stop

    def clamp(self):
        """Clamp values to safe ranges."""
        self.forward = np.clip(self.forward, -0.2, 0.3)
        self.lateral = np.clip(self.lateral, -0.2, 0.2)
        self.turn = np.clip(self.turn, -0.3, 0.3)
        return self

SYSTEM_PROMPT = """You are the brain of a small bipedal duck robot (42cm tall).
You see through its forward-facing camera. You receive voice/text commands and
must output motion commands to control the robot.

Output format — you MUST end your response with exactly one JSON line:
{"forward": float, "lateral": float, "turn": float, "behavior": string}

Ranges:
- forward: -0.2 (backward) to 0.3 (forward) m/s
- lateral: -0.2 (right) to 0.2 (left) m/s
- turn: -0.3 (clockwise) to 0.3 (counter-clockwise) rad/s
- behavior: "normal", "excited", "cautious", "stop"

Think step by step about what you see, then output the JSON command.
If no command is active or you're unsure, output all zeros with "stop"."""

class CosmosCommander:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        self.current_command = "stand still"
        self.last_cmd = RobotCommand()

    def set_command(self, command: str):
        """Set the current high-level command (from voice or text)."""
        self.current_command = command

    def get_velocity_command(self, camera_frame_jpeg: bytes) -> RobotCommand:
        """Send camera frame + command to Cosmos, parse velocity output."""
        image_b64 = base64.b64encode(camera_frame_jpeg).decode()

        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": "embedl/Cosmos-Reason2-2B-W4A16-Edge2",
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": (
                            f"{SYSTEM_PROMPT}\n\n"
                            f"Current command: {self.current_command}")},
                    ]}],
                    "max_tokens": 256,
                    "temperature": 0.1,  # Low temp for consistent outputs
                },
                timeout=2.0,  # 2 second timeout
            )
            text = resp.json()["choices"][0]["message"]["content"]
            return self._parse_response(text)

        except Exception as e:
            # On any failure, return last known good command (safety)
            print(f"Cosmos inference failed: {e}, using last command")
            return self.last_cmd

    def _parse_response(self, text: str) -> RobotCommand:
        """Extract JSON velocity command from model's text output."""
        # Find the last JSON object in the response
        json_matches = re.findall(r'\{[^}]+\}', text)
        if not json_matches:
            return RobotCommand()  # Default: stop

        try:
            data = json.loads(json_matches[-1])
            cmd = RobotCommand(
                forward=float(data.get("forward", 0.0)),
                lateral=float(data.get("lateral", 0.0)),
                turn=float(data.get("turn", 0.0)),
                behavior=str(data.get("behavior", "normal")),
            ).clamp()
            self.last_cmd = cmd
            return cmd
        except (json.JSONDecodeError, ValueError):
            return self.last_cmd
```

**How to test:**

*Automated test — `tests/test_cosmos_commander.py`:*

```python
@pytest.mark.phase5
class TestCommandParser:

    def test_parse_valid_json(self):
        """Parser should extract valid velocity commands from model output."""
        commander = CosmosCommander()
        text = 'I see a door ahead. {"forward": 0.2, "lateral": 0.0, "turn": -0.1, "behavior": "normal"}'
        cmd = commander._parse_response(text)
        assert abs(cmd.forward - 0.2) < 0.01
        assert abs(cmd.turn - (-0.1)) < 0.01

    def test_parse_clamps_values(self):
        """Parser should clamp out-of-range values."""
        commander = CosmosCommander()
        text = '{"forward": 999, "lateral": -999, "turn": 0.0, "behavior": "normal"}'
        cmd = commander._parse_response(text)
        assert cmd.forward == 0.3   # clamped to max
        assert cmd.lateral == -0.2  # clamped to min

    def test_parse_handles_garbage(self):
        """Parser should return safe default on unparseable output."""
        commander = CosmosCommander()
        cmd = commander._parse_response("this is not json at all")
        assert cmd.forward == 0.0
        assert cmd.lateral == 0.0
        assert cmd.turn == 0.0

    def test_parse_extracts_last_json(self):
        """If multiple JSON objects in output, use the last one."""
        commander = CosmosCommander()
        text = ('First thought {"forward": 0.1, "turn": 0.0}\n'
                'Better idea {"forward": 0.25, "turn": -0.1, "lateral": 0.0, "behavior": "normal"}')
        cmd = commander._parse_response(text)
        assert abs(cmd.forward - 0.25) < 0.01

    def test_fallback_on_timeout(self):
        """On server timeout, should return last known good command."""
        commander = CosmosCommander(server_url="http://localhost:99999")
        commander.last_cmd = RobotCommand(forward=0.15)
        cmd = commander.get_velocity_command(b"fake_image_data")
        assert cmd.forward == 0.15  # Returns last known good
```

---

### Task 5.3 — Integrate Cosmos with Locomotion Controller

**Description:**
Create the main control loop that runs Cosmos Reason2 and the locomotion policy concurrently on the Jetson, with Cosmos providing high-level velocity commands and the locomotion policy executing the walking.

**File:** `jetson_runtime/autonomous_walk.py`

**Key design: Two-thread architecture**

```
Thread 1: Cosmos reasoning loop (~2-3 Hz)
    - Capture camera frame
    - Send to Cosmos Reason2 via vLLM API
    - Parse velocity command
    - Update shared velocity_command variable

Thread 2: Locomotion control loop (50 Hz)
    - Read shared velocity_command
    - Read joint positions, IMU, foot contacts from hardware
    - Build observation vector
    - Run locomotion policy (TensorRT)
    - Send joint targets to servos

Thread 3 (optional): Behavior effects
    - Read behavior flag from Cosmos output
    - Control antenna servos, eye LEDs, speaker
```

**Implementation skeleton:**

```python
# jetson_runtime/autonomous_walk.py
import threading
import time
import numpy as np
from jetson_runtime.cosmos_commander import CosmosCommander, RobotCommand
from jetson_runtime.trt_infer import TRTInfer
from mini_bdx_runtime.hwi import HWI

class AutonomousWalkController:
    def __init__(self):
        self.cosmos = CosmosCommander()
        self.locomotion = TRTInfer("exported_policies/open_duck_walk_policy.trt")
        self.hwi = HWI("/dev/ttyUSB0")

        # Shared state (thread-safe via GIL for simple reads/writes)
        self.velocity_cmd = RobotCommand()
        self.running = True

    def cosmos_loop(self):
        """High-level reasoning at ~2-3 Hz."""
        while self.running:
            frame = self.capture_camera_frame()
            self.velocity_cmd = self.cosmos.get_velocity_command(frame)
            # No sleep needed — Cosmos inference itself takes ~300-500ms

    def locomotion_loop(self):
        """Low-level walking at 50 Hz."""
        prev_action = np.zeros(16)
        while self.running:
            t0 = time.time()

            # Build observation from hardware
            obs = self.build_observation(
                self.velocity_cmd, prev_action
            )

            # Run locomotion policy
            action = self.locomotion.infer(obs)
            prev_action = action.copy()

            # Send to servos
            self.hwi.set_position_all(self.action_to_servo_dict(action))

            # Maintain 50 Hz
            elapsed = time.time() - t0
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)

    def run(self, command: str):
        """Start autonomous walking with a voice/text command."""
        self.cosmos.set_command(command)

        t_cosmos = threading.Thread(target=self.cosmos_loop, daemon=True)
        t_locomotion = threading.Thread(target=self.locomotion_loop, daemon=True)

        t_cosmos.start()
        t_locomotion.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
            self.hwi.turn_off()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="e.g., 'Walk to the door'")
    args = parser.parse_args()

    controller = AutonomousWalkController()
    controller.run(args.command)
```

**Usage:**
```bash
# Start vLLM server in one terminal
docker run ... vllm serve "embedl/Cosmos-Reason2-2B-W4A16-Edge2" ...

# Start autonomous walking in another terminal
python jetson_runtime/autonomous_walk.py "Walk to the red cup"
```

**How to test:**

*Automated test — `tests/test_autonomous_integration.py`:*

```python
@pytest.mark.phase5
class TestAutonomousIntegration:

    def test_cosmos_and_locomotion_share_gpu(self):
        """Both models must coexist on Jetson without OOM."""
        # Start vLLM server, load locomotion TRT engine
        # Run one Cosmos inference + 50 locomotion inferences
        # Check: no OOM, no NaN in outputs
        pass

    def test_velocity_command_reaches_locomotion(self):
        """Velocity commands from Cosmos must propagate to the locomotion loop."""
        controller = AutonomousWalkController()
        controller.velocity_cmd = RobotCommand(forward=0.2)
        obs = controller.build_observation(controller.velocity_cmd, np.zeros(16))
        # The observation vector should contain the commanded velocity
        # Check that obs[command_indices] == [0.2, 0.0, 0.0]
        pass

    def test_locomotion_runs_at_target_frequency(self):
        """Locomotion loop must maintain >= 40 Hz even while Cosmos runs."""
        # Start both loops, measure locomotion loop timing for 5 seconds
        # Assert mean loop time < 25ms (40 Hz minimum)
        pass
```

*Manual verification (on Jetson, with robot):*
- [ ] Start vLLM server + autonomous_walk.py with command "walk forward"
- [ ] Robot should walk forward continuously
- [ ] Change command to "turn left" — robot should start turning
- [ ] Point camera at an obstacle — robot should avoid it (if prompted with avoidance command)
- [ ] Monitor `tegrastats` — RAM < 7.8 GB, GPU not thermal throttling
- [ ] Measure: locomotion loop holds 50 Hz while Cosmos runs at 2-3 Hz

---

### Task 5.4 — Prompt Engineering for Robot Behaviors

**Description:**
Design and test a library of system prompts for different robot behaviors. The quality of Cosmos Reason2's output depends heavily on the prompt.

**Prompt library to create in `jetson_runtime/prompts/`:**

| Prompt File | Behavior | Trigger |
|---|---|---|
| `navigate.txt` | Walk toward a named object/location | "Walk to the ..." |
| `follow.txt` | Follow a person at ~1m distance | "Follow me" |
| `avoid.txt` | Walk forward while avoiding obstacles | "Walk forward safely" |
| `explore.txt` | Wander and describe surroundings | "Explore the room" |
| `interact.txt` | React to people (wave, approach, retreat) | "Be friendly" |
| `patrol.txt` | Walk a pattern and report what's seen | "Patrol this area" |

**Each prompt must specify:**
1. The robot's physical constraints (42cm tall, bipedal, no arms, forward camera only)
2. The exact JSON output format expected
3. Safety rules (don't walk off edges, don't exceed speed limits)
4. The specific behavior logic for that mode

**How to test:**

*Manual verification (on Jetson, with robot — requires camera):*
- [ ] **Navigate:** Say "Walk to the door" — robot walks toward door, stops when close
- [ ] **Follow:** Say "Follow me", walk around — robot follows at ~1m distance
- [ ] **Avoid:** Place obstacles in path — robot steers around them
- [ ] **Explore:** Robot wanders, prints descriptions of what it sees
- [ ] **Interact:** Wave at robot from 3m — robot walks toward you
- [ ] **Safety:** Place robot near table edge — robot does NOT walk off

For each prompt, document:
- Success rate (out of 10 attempts)
- Common failure modes
- Latency from visual change to robot response

**Output:** `docs/jetson-mod/prompt_engineering_results.md`

---

### Task 5.5 — Add Voice Input (Optional)

**Description:**
Add speech-to-text so the robot can receive voice commands hands-free, completing the natural interaction loop.

**Options (ranked by practicality on Orin Nano):**

| Method | Model | Extra RAM | Latency |
|---|---|---|---|
| **Whisper Tiny** (on-device) | `openai/whisper-tiny` (39M params) | ~0.1 GB | ~1-2s per utterance |
| **Google Speech API** (cloud) | Cloud-based | 0 | ~0.5-1s (needs WiFi) |
| **Vosk** (on-device, lightweight) | vosk-model-small-en | ~0.05 GB | ~0.5s |

Whisper Tiny or Vosk are recommended since they run on-device (no cloud dependency).

**Steps:**
1. Install whisper or vosk
2. Add a USB microphone or use the speaker/mic header in the head
3. Create a voice command listener that feeds text to `CosmosCommander.set_command()`

**How to test:**

*Manual verification:*
- [ ] Say "walk forward" — robot walks forward within 2 seconds
- [ ] Say "stop" — robot stops within 1 second
- [ ] Say "follow me" — robot enters follow mode
- [ ] Test in quiet and moderately noisy environments
- [ ] Measure: time from end of speech to robot motion

---

## Dependency Graph

```
Task 1.1 (Mass/Inertia Calc)
    ├── Task 1.2 (Jetson STL) ─── can run in parallel
    └── Task 1.6 (Test Infra) ─── can run in parallel
         │
         v
Task 1.3 (Update robot_motors.xml) ── depends on 1.1, 1.2
    │
    ├── Task 1.4 (Update robot.xml) ── can run in parallel with 1.3
    └── Task 1.5 (Update robot.urdf) ── depends on 1.1, 1.2 (critical for Isaac Sim import)
         │
         v
Task 2.1 (URDF → USD conversion) ── depends on 1.5
    │
    v
Task 2.2 (Configure actuator model) ── depends on 2.1
    │
    v
Task 2.3 (Create Isaac Lab env) ── depends on 2.1, 2.2
    │
    v
Task 2.4 (Train PPO + optional AMP) ── depends on 2.3
    │   ├── 2.4a PPO  (RSL-RL, 4096 envs, ~1 hr)
    │   └── 2.4b AMP  (SKRL,   4096 envs, ~1-2 hr)  ── optional stretch goal, requires DirectRLEnv
    │   ├── 2.4c AMP  (SKRL,   4096 envs, ~1-2 hr)  ── parallel (needs ref motion data)
    │
    │
    v
Task 2.5 (Compare algorithms & select best) ── depends on 2.4
    │
    v
Task 2.6 (Export best policies to ONNX) ── depends on 2.5
    │
    v
Task 2.7 (Validate best policy in Isaac Sim) ── depends on 2.6
         │
         v  [PASS → proceed]
Task 3.1 (Redesign trunk_top)
    │
    ├── Task 3.2 (Redesign body_back + body_front ventilation) ── parallel
    ├── Task 3.3 (Redesign body_middle) ── parallel, conditional
    ├── Task 3.4 (Redesign battery pack, thermally isolated) ── parallel
    ├── Task 3.6 (Design thermal partition wall) ── parallel
    └── Task 3.7 (Add partition slots to trunk_top/bottom) ── depends on 3.6
         │
         v
Task 3.5 (Update sim with final STLs) ── depends on 3.1-3.4, 3.6, 3.7
         │
         v
Task 4.1 (Procure Hardware) ── can start after 2.6 PASS (includes mica sheet, thermistor)
    │
    v
Task 4.2 (3D Print) ── depends on 3.1-3.4, 3.6, 3.7 (includes thermal_partition)
    │
    v
Task 4.3 (Assemble) ── depends on 4.1, 4.2 (includes thermal partition + thermistor install)
    │
    v
Task 4.4 (Port Runtime + TensorRT) ── depends on 4.1, 2.6
    │
    ├── Task 4.6 (Software Thermal Management) ── parallel with 4.4 (depends on 4.3)
    v
Task 4.5 (Real Robot Walk Test) ── depends on 4.3, 4.4, 4.6
         │
         v  [PASS → robot walks autonomously]
Task 5.1 (Setup Cosmos Reason2 on Jetson) ── depends on 4.5 PASS
    │
    v
Task 5.2 (Build command parser) ── depends on 5.1
    │
    v
Task 5.3 (Integrate Cosmos + locomotion + thermal_manager) ── depends on 5.2, 4.4, 4.6
    │
    v
Task 5.4 (Prompt engineering for behaviors) ── depends on 5.3
    │
    v
Task 5.5 (Voice input — optional) ── depends on 5.3
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| URDF/MJCF → USD conversion loses joint properties | Medium | High | Manually verify every joint in Isaac Sim inspector. Cross-reference with MuJoCo model. Fix USD manually if needed. |
| Isaac Sim PhysX behaves differently from MuJoCo | Medium | High | Carefully tune actuator model (Task 2.2). Compare step responses. PhysX and MuJoCo will never match exactly — domain randomization compensates. |
| PPO training fails to produce walking gait | Low-Med | High | Use proven Isaac Lab locomotion configs (ANYmal, humanoid) as starting points. Iterate on reward weights. Add imitation reward from reference motions. |
| Jetson doesn't physically fit in trunk cavity | Low | High | CAD clearance checks (Task 3.1). Worst case: extend body_middle parts by 10-15mm. |
| STS3250 servos lack torque for heavier robot | Low | Medium | **Mitigated by servo upgrade.** STS3250 provides 50 kg.cm stall torque (vs 19.5 kg.cm STS3215), a 2.5x increase. Domain randomization in training includes mass variation. Monitor servo current in real tests. |
| Thermal issues — Jetson heat damages batteries | Medium | High | **Three-layer defense:** (1) Thermal partition wall with mica insulation between battery and compute zones (Task 3.6), (2) Directed cross-flow ventilation — inlet on body_front, exhaust on body_back, compute zone only (Task 3.2), (3) Software thermal management — NTC thermistor on battery pack + auto power-mode throttling (7W/15W/25W) with emergency shutdown at 50°C (Task 4.6). |
| TensorRT conversion fails for the policy MLP | Low | Medium | Fallback to `onnxruntime-gpu` (CUDAExecutionProvider). Slightly slower but still fast enough. |
| DC-DC converter introduces electrical noise | Low | Medium | Use shielded cables. Add capacitors to servo power lines. |
| Sim2real gap is too large (trained policy doesn't work on real robot) | Medium | High | Aggressive domain randomization during training. Start with conservative commands (slow walk). Iteratively adjust sim parameters based on real robot behavior. |
| Battery life too short at 15W+ | Medium | Medium | Default to 7W eco mode for walking. 67 TOPS is still available at 7W. |
| Cosmos Reason2 + locomotion policy OOM on 8 GB | Low-Med | High | Cosmos W4A16-Edge2 benchmarked at 5.8 GB. Reduce `max-model-len` from 2048 to 1024 if tight. Locomotion policy is only ~0.1 GB. |
| Cosmos output is unparseable / inconsistent JSON | Medium | Medium | Robust parser with fallback to last known command (Task 5.2). Low temperature (0.1) for consistency. Structured prompt with explicit JSON format. |
| Cosmos reasoning too slow for reactive obstacles | Medium | Medium | Cosmos runs at 2-3 Hz — fine for navigation, too slow for fast obstacles. Locomotion policy handles balance. Add a fast YOLO-based obstacle detector (~30 Hz) if needed later. |
| vLLM Docker container doesn't fit / has compatibility issues | Low | Medium | Fallback: run Cosmos directly with `transformers` library + manual quantization via `auto-gptq` or `llama.cpp`. Slower but simpler. |
| Cosmos hallucinates obstacles or misidentifies objects | Medium | Low-Med | Safety: clamp all velocity commands. Never allow speeds > 0.3 m/s. Add foot-contact + IMU safety stops in locomotion layer regardless of Cosmos output. |

---

## Estimated Timeline

| Phase | Duration | Can Start | Hardware |
|---|---|---|---|
| Phase 1 (Sim Model Update) | 3-5 days | Immediately | Local machine |
| Phase 2 (Isaac Lab Setup + PPO Training) | 1-2 weeks | After Phase 1 | **DGX Spark** |
| Phase 3 (CAD Redesign) | 1-3 weeks | After Phase 2 passes | Local + CAD software |
| Phase 4 (Hardware Build) | 2-4 weeks | After Phase 3 + procurement | **Jetson + 3D printer** |
| Phase 5 (Cosmos Reason2 VLM Integration) | 1-2 weeks | After Phase 4 (robot walks) | **Jetson Orin Nano** |
| **Total** | **~9-15 weeks** | | |

Phase 2 breakdown:
- Task 2.1-2.3 (USD conversion, actuator tuning, env setup): ~3-5 days
- Task 2.4 (PPO training + optional AMP): ~1-2 days
- Task 2.5-2.7 (comparison, export, validation): ~2-3 days

Phase 5 breakdown:
- Task 5.1 (Cosmos setup on Jetson): ~1-2 days
- Task 5.2 (Command parser): ~1 day
- Task 5.3 (Integration with locomotion): ~2-3 days
- Task 5.4 (Prompt engineering): ~3-5 days (iterative testing)
- Task 5.5 (Voice input — optional): ~1-2 days

**Phase 1** is local work (edit XML/URDF, create STL). **Phase 2** requires the DGX Spark — Isaac Sim/Lab setup and RL training. **Phase 3** is CAD work (OnShape/Fusion). **Phase 4** requires purchased hardware. **Phase 5** is software-only on the Jetson — no additional hardware needed (CSI camera already installed in Phase 4).

The first critical gate is **Task 2.7** — if the trained policy walks in Isaac Sim, we have high confidence the physical modification will work. The second gate is **Task 4.5** — the robot walks in the real world. Phase 5 builds on top of a working walking robot.

The estimated spend before the first gate is $0 (DGX Spark is already available).
