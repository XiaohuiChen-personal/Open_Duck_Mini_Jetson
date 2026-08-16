# AGENTS.md — Rules & Context (single source of truth)

Every AI agent working in this repo (Claude Code, Cursor, or otherwise) reads
this file first. `CLAUDE.md` and `.cursor/rules/agents.mdc` are pointers to
this file — never duplicate rules there. **Update this file whenever a decision
changes.** Consolidated 2026-07-27 from `CLAUDE.md` + nine near-duplicate rule
files that used to live in both `.claude/rules/` and `.cursor/rules/` and had
already drifted apart — the failure this layout exists to prevent.

---

This is a fork of the [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) bipedal robot project. We are modifying the robot to replace the Raspberry Pi Zero 2W with an NVIDIA Jetson Orin Nano Super Developer Kit, and migrating the entire simulation/training/deployment pipeline to the NVIDIA stack.

## Quick Reference

- **Robot:** Open Duck Mini v2, ~42cm tall bipedal duck, 14x Feetech STS3250 servos, **2.729 kg** (after mod; Task M4 rebuilt every inertial bottom-up per part on 2026-08-13, up from 2.657). Isaac simulates the same — the PhysX phantom kilogram on the massless MJCF root was **fixed 2026-08-11** (PLANT-1). **Every policy through v5d was trained on the old 3.66 kg plant and its gate numbers are stale:** `docs/jetson-mod/known_issues.md`
- **Onboard computer:** NVIDIA Jetson Orin Nano Super (8 GB, 67 TOPS) — relocated from head to trunk
- **Training hardware:** NVIDIA DGX Spark (Grace Blackwell)
- **Simulation:** NVIDIA Isaac Sim (PhysX 5) — replacing MuJoCo
- **RL framework:** NVIDIA Isaac Lab with RSL-RL (PPO). (SKRL was used only for the archived PPO-vs-AMP course study — see `open-duck-ppo-vs-amp`; Isaac Lab ships no off-policy skrl task configs, only PPO/AMP plus multi-agent MAPPO/IPPO)
- **On-device AI:** Cosmos Reason2-2B (W4A16 quantized) for physical AI reasoning + TensorRT locomotion policy
- **Task plan:** `docs/jetson-mod/task_plan_v2.md` is the **active plan** for the current rebuild (Phase M mass/CAD/USD, Phase R re-gate+retrain, Phase S sim-to-real, Phase V VLM). `docs/jetson-mod/task_plan.md` remains the original 5-phase, 32-task plan and the only record of the Phase 3/4 design intent, thermal rationale and risk register — **where the two disagree, v2 wins**
- **Experiment journal:** EVERY training run gets an entry in `docs/jetson-mod/experiment_journal.md`. Data-sourcing protocol in the [Experiment Journal Protocol](#experiment-journal-protocol) section below — last-100 TensorBoard means (never single-iteration log samples), measured gate rollouts, every number names its source. **Note (2026-07-26): the EN.665.645 course-paper record is FROZEN in the archive repo `open-duck-ppo-vs-amp` (tag `course-study-freeze` marks the freeze commit); from here on this journal is the robot project's engineering record, and study runs 1-16 in it are historical.**

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
`scripts/evaluate_policies.py` docstring and `docs/jetson-mod/v4_comparison.md`
("Metric hierarchy").

1. **Quantitative eval (3,200 episodes).** `scripts/evaluate_policies.py`: 5 command
   conditions x 10 windows x 64 envs, 30 s episodes, deterministic, seed 42, obs
   corruption and pushes disabled. Emits one JSON per policy into the per-model
   results dir (`docs/jetson-mod/eval_results_v4/` for the current layout-v2.1
   model; one dir + one table per robot model, never mixed) with metrics 1-9.
   Metric 9 is the gait-validity gate: BOTH feet's stance duty inside [40, 90]%
   per condition (`GAIT_DUTY_BAND_PCT` is the single source of truth).
   `--report-only` regenerates the comparison table (`v4_comparison.md`) from
   archived JSONs without Isaac.
2. **Video audit (mandatory — aggregate metrics alone are NOT sufficient).** Render
   deterministic rollout mp4s (robot-tracking camera, ~20 s, seed 42) in TWO
   conditions: fixed forward vx=0.2 AND turn wz=0.3 — defects like one-foot dragging
   only show off-forward. PPO: `scripts/play_policy.py`.
   (AMP tooling was removed 2026-07-26 with the course-study archive; if a
   future method needs a video wrapper, recreate one from `open-duck-ppo-vs-amp`.)
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
  study runs used the pre-CAD model, frozen in the archive repo; v4+ policies use
  the corrected layout-v2.1 model); cross-model numbers are not comparable.
- Runs that fail their gate get forensic-rollout numbers (targeted diagnostic
  rollouts, e.g. per-condition metrics from `evaluate_policies.py`; the study-era
  `measure_amp_gait.py` lives in the archive repo) cited as diagnostic only —
  never as protocol-comparable results.
- Planned extension (forward-plan Phase 1): per-episode dumps + posture metrics
  (mean base height, trunk orientation) join the eval JSON as the quantitative twin
  of the video checklist.

## Key Directories

- `mini_bdx/robots/open_duck_mini_v2/` — Robot model files (MJCF, URDF, USD, STL meshes)
- `isaac_lab_env/` — Isaac Lab RL environment definitions and training configs
- `jetson_runtime/` — Jetson deployment code (TensorRT, Cosmos Reason2, GPIO) — planned, Phases 4-5; not yet created
- `exported_policies/` — Trained ONNX and TensorRT policy files
- `experiments/` — Legacy MuJoCo-based experiment scripts (reference only)
- `docs/jetson-mod/` — Modification documentation and task plan
  - `known_issues.md` — **the consolidated register of every known defect**, each
    mechanically re-verified before inclusion (2026-08-09). Read this first;
    it carries severity, scope, evidence and a reproduction command per issue,
    plus a list of claims that were tested and *refuted* so they are not raised
    again. Start here before filing or fixing anything.
- `tests/` — Automated test suite

## CAD, mesh and robot-description work (mandatory for any geometry task)

If your task touches a `.stl`, `.step`, `.3mf`, `.glb`, `.urdf`, `.sdf` or
`.dxf` file, or slices a mesh, **an installed agent skill already owns that
workflow** — `cad`, `urdf`, `cad-viewer`, `step-parts`, `gcode`, `dxf`,
`sendcutsend`. Invoke the skill rather than writing your own pipeline.

**Run the setup gate first. It is idempotent and takes seconds:**

```bash
scripts/setup_cad_tools.sh --check     # verify only
scripts/setup_cad_tools.sh             # install what is missing, then verify
```

**Then invoke every skill script with the venv interpreter, never plain
`python3`:**

```bash
~/.venvs/cad-viewer/bin/python3 ~/.claude/skills/cad/scripts/snapshot --input part.stl --output out.png
~/.venvs/cad-viewer/bin/python3 ~/.claude/skills/cad/scripts/inspect refs part.step --facts
~/.venvs/cad-viewer/bin/python3 ~/.claude/skills/urdf/scripts/validate robot.urdf
```

Four things that will otherwise cost you an hour, all verified 2026-08-12:

1. **Plain `python3` renders `.stl` but fails on `.step`** with
   `No module named 'OCP'`. The toolchain looks healthy until the first STEP —
   which is exactly what `task_plan_v2.md` Task M3 produces. The venv carries
   `cadgen`, `OCP`, `build123d` and `playwright`; the system Python does not.
2. **An agent can *see* geometry, not just link to it.** `scripts/snapshot`
   writes a PNG (or `--mode orbit` GIF) that you then read directly. Use that as
   your review step. Do not ask the owner to open a browser and describe it.
3. **There is no STEP for any robot part.** The robot is 48 STLs in
   `mini_bdx/robots/open_duck_mini_v2/` plus 37 in `print/`; the only `.step`
   files belong to a third-party head mod. An STL has no B-rep, so **you cannot
   shell, fillet or feature-edit an existing part** — `inspect refs` refuses it.
   Author *new* parts in build123d; modify *existing* ones with `trimesh`
   (see `scripts/generate_cad_mods.py` for the worked example).
4. **`robot.urdf` cannot be rendered at all** until its 240 malformed
   `package:///name.stl` mesh URIs are fixed (`task_plan_v2.md` Task M7).
   `urdf scripts/snapshot` fails with `No link mesh loaded for robot`. That is a
   data defect, not a tooling problem.

**Do not put setup, rules or a venv inside `~/.agents/skills/<name>/`.** Skills
are installed from `github.com/earthtojake/text-to-cad` and tracked in
`~/.agents/.skill-lock.json`, so that directory is replaced wholesale on update
and anything stored there is lost silently. The venv lives at
`~/.venvs/cad-viewer` for exactly this reason; the rules live here.

**Course-study archive:** the EN.665.645 PPO-vs-AMP study (runs 1-16, journal, eval results,
policies, training-log evidence) is frozen in the dedicated repo
https://github.com/XiaohuiChen-personal/open-duck-ppo-vs-amp (`~/Projects/open-duck-ppo-vs-amp`);
tag `course-study-freeze` marks the freeze commit here. Post-merge state of this repo is NOT
the study record.

---

## Project Overview

_High-level project overview, architecture, and key decisions for the Open Duck Mini Jetson modification_

This project is modifying the Open Duck Mini v2 bipedal robot to use NVIDIA hardware and software throughout.

### What Changed and Why

The original robot uses a Raspberry Pi Zero 2W (65x30x5mm, 10g, no GPU) in the head. We are replacing it with a Jetson Orin Nano Super Dev Kit (103x90.5x34.77mm incl. heatsink+fan, 176 g as booked in the model (NVIDIA SP-11324-001 v1.3 §4 publishes 175 g), 67 TOPS GPU) relocated to the trunk/body cavity.

**Why:** To add physical AI capabilities (vision, language understanding, autonomous navigation) and learn the NVIDIA robotics stack (Isaac Sim, Isaac Lab, TensorRT, Cosmos).

**Approach:** Partial body redesign (4-6 printed parts), not scaling up the entire robot. The cube-square law makes scaling impractical (mass scales as L^3, torque needed as L^4).

### Architecture

```
Cosmos Reason2-2B (VLM, 2-3 Hz)     — "where should I go?" (text reasoning)
         │ velocity commands
         v
Locomotion Policy (PPO/AMP, 50 Hz)  — "how do I walk there?" (joint control)
         │ joint position targets
         v
Feetech STS3250 Servos (14x)        — physical motors
```

### Key Decisions Made

1. **Jetson Dev Kit over module + mini carrier board** — Dev Kit has CSI camera ports and 40-pin GPIO header needed for peripherals
2. **Full NVIDIA stack migration** — Isaac Sim/Lab replaces MuJoCo for simulation and training
3. **Cosmos Reason2-2B for VLM** — Confirmed running on Orin Nano Super at ~5.8 GB RAM, ~16 tok/s
4. **Multi-algorithm RL experiment** — PPO vs AMP compared head-to-head (16-run study, archived in open-duck-ppo-vs-amp); SAC/RPO/TRPO were not run — skrl's Isaac Lab integration only has locomotion-ready PPO/AMP paths (see README note). Study winner ppo_v3 has since been superseded for deployment by v4_robust, PPO retrained on the corrected layout-v2.1 model (docs/jetson-mod/validation_results.md)
5. **Sim-first approach** — All changes validated in simulation before hardware purchases

### Current Phase

Check `docs/jetson-mod/task_plan.md` for detailed progress. The plan has 5 phases:
- Phase 1: Simulation model update (mass/inertia for Jetson in trunk)
- Phase 2: Isaac Lab setup + multi-algorithm RL training on DGX Spark
- Phase 3: CAD redesign of trunk/body parts
- Phase 4: Hardware build and real-robot walking
- Phase 5: Cosmos Reason2 VLM integration for autonomous behavior

---

## Hardware Specs

_Hardware specifications for the robot, Jetson Orin Nano, servos, batteries, and motor parameters_

### Robot Physical Specs

| Parameter | Value |
|---|---|
| Total height | ~420 mm (legs extended) |
| Total mass (after mod) | ~2,657 g declared **and now simulated** — [PLANT-1](docs/jetson-mod/known_issues.md#plant-1) fixed 2026-08-11, `audit_plant_mass.py` exits 0. The true *build* is heavier: ~2,810 g once the Part-2 CAD delta is measured rather than assumed ([PLANT-10](docs/jetson-mod/known_issues.md#plant-10)) and the 4 booked-but-unmodelled 18650 cells are counted. Every policy through v5d was trained on the old 3,657 g plant and needs re-gating. |
| **Mass Isaac actually simulates** | **2,657 g** — matches the authored mass since the PLANT-1 fix (2026-08-11) merged the massless root frame `base` into `trunk_assembly`. It was 3,657 g for every run up to and including v5d, because PhysX substituted a 1,000 g default for the unauthored root; see `docs/jetson-mod/known_issues.md#plant-1` |
| DOFs | 15 joints + 1 head_roll = 16 actuators |
| Servos | 14x Feetech STS3250 (12V, 50 kg.cm stall, 74.5g each) |
| Ear servos | 2x SG90 micro servos (in head) |
| Battery | 6x 18650 Li-ion cells (3S2P, 11.1V) |
| IMU | BNO055 (I2C, mounted in trunk) |

### Jetson Orin Nano Super Developer Kit

| Spec | Value |
|---|---|
| Dimensions | 103 x 90.5 x 34.77 mm (full dev kit incl. heatsink+fan) |
| Weight | 176 g (design input carried in the MJCF; NVIDIA publishes 175 g) |
| GPU | 1024 CUDA + 32 Tensor cores (Ampere) |
| AI Performance | 67 TOPS |
| RAM | 8 GB LPDDR5 unified (shared CPU+GPU) |
| Memory bandwidth | 102 GB/s |
| CPU | 6-core ARM Cortex-A78AE @ 1.5 GHz |
| Power modes | 7W eco / 15W default / 25W max |
| Ports | 2x CSI camera, 4x USB 3.2, DisplayPort, M.2, 40-pin GPIO, DC barrel jack |
| Location on robot | Trunk cavity (relocated from head) |

### Memory Budget on Jetson (8 GB total)

```
Cosmos Reason2 (W4A16-Edge2):  ~5.8 GB
Locomotion policy (TensorRT):  ~0.1 GB
Camera pipeline:               ~0.3 GB
OS + system:                   ~1.5 GB
Total:                         ~7.7 GB  (fits)
```

### Mass Changes from Modification

| Body | Original (g) | Modified (g) | Change |
|---|---|---|---|
| trunk_assembly | 698.5 | **1,188.9** | +176 (Jetson) +58.5 (3 servo upgrades STS3215→STS3250 in trunk: +19.5g each) +180 (batteries: 4 extra 18650 cells, 2→6 total) +5 (BMS) +15 (DC-DC) +37 (thermal partition assembly) +8 (wiring) **−13.95** (Part-2 CAD, **measured** per part at the chosen process — `fdm-asa`, 3 perim / 20 % infill: trunk_bottom −23.45, body_middle_bottom −1.72, body_front **+0.11**, body_back **+11.11**. PLANT-10 **FIXED 2026-08-12** by Task M2: the old −88.5 came from one assumed density of 1.116 g/cm³ applied to a volume difference, and was 74.53 g wrong. Mass is now whole-part measured — `scripts/part_mass_table.json` — with no density assumed anywhere. Note the two positive terms: cutting slots in `body_front` *adds* mass because the new walls print near-solid, which is why no single density could ever work) |
| head_assembly | 352.6 | ~362.1 | -10 (Pi removed) +19.5 (head_roll servo upgrade) |
| All 14 servos | 770 (14x55g) | 1,043 (14x74.5g) | +273 g total (+19.5g each x14: 3 in trunk, 1 in head, 10 in limb/neck bodies) |
| Total robot | 2,062 | ~2,657 | +595 g (+28.9%) |

### Raspberry Pi Zero 2W (being removed)

| Spec | Value |
|---|---|
| Dimensions | 65 x 30 x 5 mm |
| Weight | 10 g |
| Location | Head (head_assembly body in MuJoCo model) |
| CPU | Quad-core ARM Cortex-A53 @ 1 GHz |
| RAM | 512 MB |
| AI compute | None |

### Motor Parameters (from BAM identification)

Source: `experiments/v2/params_sts3250_id008.json` (kscalelabs/sysid STS3250 id008)

| Parameter | Value | Used in |
|---|---|---|
| kp (position gain) | 45.53 | Isaac Lab actuator config |
| kd (velocity gain) | 1.346 | Isaac Lab actuator config |
| armature | 0.040 | MuJoCo/Isaac Sim joint model |
| frictionloss | 0.200 | MuJoCo/Isaac Sim joint model |
| torque limit | 8.716 Nm | Actuator effort_limit — **1.78x the 50 kg.cm (4.90 Nm) datasheet stall in the specs table above.** BAM's figure is electrical stall from its identified model (kt*V/R = 1.0006*12.1/1.389); the datasheet is rated stall. The simulator enforces BAM's, so a policy may lean on torque the hardware cannot deliver. Unresolved — see `docs/jetson-mod/known_issues.md` PLANT-5 |
| kt (torque constant) | 1.0006 | BAM motor model |
| R (resistance) | 1.3890 | BAM motor model |

---

## NVIDIA Stack

_NVIDIA software stack overview — Isaac Sim, Isaac Lab, TensorRT, Cosmos Reason2, DGX Spark_

The entire pipeline uses NVIDIA tools. Here's what each tool does and how it fits.

### Stack Map

| Stage | Tool | Replaces |
|---|---|---|
| Physics simulation | **Isaac Sim** (PhysX 5) | MuJoCo |
| RL training framework | **Isaac Lab** | MuJoCo Playground + Stable-Baselines3 |
| RL algorithms | **RSL-RL** (PPO; SKRL was used only for the archived AMP course study) | SB3 |
| Robot model format | **USD** (converted from **MJCF**, `robot_motors.xml`) | MJCF (.xml) |
| Training hardware | **DGX Spark** (Grace Blackwell, 1 PFLOP FP4) | Single GPU |
| Policy deployment | **TensorRT** on Jetson GPU | onnxruntime on Pi CPU |
| Physical AI reasoning | **Cosmos Reason2-2B** (W4A16) | None (new capability) |
| VLM serving | **vLLM** (Jetson-optimized Docker) | N/A |
| On-robot hardware | **Jetson Orin Nano Super** | Raspberry Pi Zero 2W |

### Isaac Sim

NVIDIA's physics simulator built on Omniverse. Uses PhysX 5 for GPU-accelerated rigid body simulation. Can run thousands of robot instances in parallel.

- Docs: https://docs.isaacsim.omniverse.nvidia.com
- Has built-in URDF and MJCF importers to convert to USD format
- The robot model lives at `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`

### Isaac Lab

RL training framework built on Isaac Sim. Provides environments, reward utilities, domain randomization, and integrations with RSL-RL and SKRL.

- Docs: https://isaac-sim.github.io/IsaacLab
- GitHub: https://github.com/isaac-sim/IsaacLab
- Our env config: `isaac_lab_env/open_duck_mini_v2/env_cfg.py`

### RSL-RL

Lightweight PPO implementation optimized for GPU parallel training. Default in Isaac Lab for locomotion.

- Used for PPO training with 4096 parallel envs

### SKRL

Modular RL library with the widest algorithm support in Isaac Lab; the only library with AMP (Adversarial Motion Priors) support. No longer used in this repo — the AMP track was removed 2026-07-26 with the course-study archive (`open-duck-ppo-vs-amp`).

- GitHub: https://github.com/Toni-SM/skrl
- Was used for: the PPO-vs-AMP study's AMP training (Isaac Lab ships no off-policy skrl task configs; only PPO/AMP + multi-agent MAPPO/IPPO exist)

### TensorRT

NVIDIA's inference optimizer. Converts ONNX models to optimized GPU engines for Jetson.

- Convert: `trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16`
- Locomotion policy inference: <1 ms on Jetson
- Docs: https://developer.nvidia.com/tensorrt

### Cosmos Reason2-2B

NVIDIA's physical AI reasoning VLM. Understands spatial relationships, physics, and can plan robot actions.

- Model: `embedl/Cosmos-Reason2-2B-W4A16-Edge2` (INT4 quantized for edge)
- Runs on Jetson Orin Nano Super at ~5.8 GB RAM, ~16-17 tok/s
- Served via vLLM Jetson Docker: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`
- Outputs text (reasoning + JSON velocity commands), NOT direct actions
- GitHub: https://github.com/nvidia-cosmos/cosmos-reason2

### DGX Spark

Training workstation with Grace Blackwell chip. 128 GB unified memory, up to 1 PFLOP FP4.

- All NVIDIA software pre-installed
- Used for: Isaac Lab RL training (Phase 2), VLM fine-tuning (if needed)
- NOT used on the robot — training only

---

## Isaac Lab

_Isaac Lab RL environment configuration — actuator params, domain randomization, training commands_

Rules and context for the Isaac Lab RL training environment.

### Environment Config

The environment is defined in `isaac_lab_env/open_duck_mini_v2/env_cfg.py`:

- **Sim timestep:** 0.005 s (200 Hz physics)
- **Policy frequency:** 50 Hz (decimation = 4)
- **Parallel envs:** 4096 on DGX Spark
- **Robot USD:** `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`

### Actuator Configuration

Use `ImplicitActuatorCfg` or `IdealPDActuatorCfg` with BAM-identified parameters:

```python
stiffness = 45.53     # kp from BAM (STS3250 id008)
damping = 1.346       # kd from BAM (STS3250 id008)
armature = 0.040      # From BAM id008
friction = 0.200      # frictionloss from BAM id008
effort_limit_sim = 8.716  # Torque limit in Nm (BAM forcerange). NOTE the _sim
                          # suffix: robot_cfg.py sets effort_limit_sim and leaves
                          # effort_limit at None. Pasting `effort_limit=` sets a
                          # different, currently-unused field.
```

### Domain Randomization

Critical for sim2real transfer. The implemented, gate-validated DR lives in
`OpenDuckRobustEnvCfg` (env_cfg.py, v4-robust track) — duck-scaled, NOT the
generic literature values. Source of truth: the [RL Training](#rl-training) section
"v4 Tracks". Summary:

- Velocity pushes: +/-0.3 m/s every 8-14 s (not force-based)
- Trunk mass: additive (-0.10, +0.15) kg; trunk CoM +/-10/+/-5 mm
- Friction: static 0.4-1.0 / dynamic 0.3-0.8
- Joint reset scale: 0.9-1.1
- Sensor noise on joint positions and IMU (obs corruption, actor only)

> **Known coverage gap — `add_base_mass` does not cover the base.** The term is
> named for the concept "base mass" but bound to the *body* `trunk_assembly`
> (`env_cfg.py:365`). The plant's single largest mass error is a phantom
> **1.000 kg** on the body actually called `base` — 6.7x this term's upper bound,
> constant rather than sampled, and on a different body — so no amount of this
> randomization exposes a policy to it. Full analysis, measurement, and fix
> options: **`docs/jetson-mod/known_issues.md`**.
>
> Generalize the lesson when adding terms: a randomization can be present,
> correctly configured and gate-validated, and still randomize the wrong body.
> Check the term against the body that actually carries the uncertainty.

**Not randomized at all** (name them honestly rather than implying full
coverage): control/observation latency, actuator gain and thermal drift, servo
backlash, IMU bias/drift and mounting misalignment, battery voltage sag under
load, per-joint friction spread, and mass distribution outside the trunk.

### Termination Conditions

Read from `env_cfg.py:591,594` (repeated at `700,703`) — verified 2026-08-02.
An earlier revision of this section said 0.08 m / 90 degrees; **both numbers
were wrong, and wrong in the permissive direction**, which matters because fall
rate is a headline gate in every policy comparison here. The gate has always
been stricter than this section described; the measurements are unaffected.

- Trunk height < **0.09 m** (`root_height_below_minimum`, `minimum_height=0.09`)
- Trunk tilt > **60 degrees** (`bad_orientation`, `limit_angle=radians(60.0)`)

### Reference Motions (Imitation Reward)

The Open Duck Playground reference motion generator (https://github.com/apirrone/Open_Duck_reference_motion_generator) produces `polynomial_coefficients.pkl` — 240 parametric walking gaits as degree-15 polynomials over a 0.54s period.

This data is used directly by the `ImitationReward` class in `isaac_lab_env/open_duck_mini_v2/imitation_reward.py` — a BDX-style composite (joint pos -L2*15.0, joint vel -L2*0.001, base vel exp(-8e), contact match; RewTerm weight=1.0). It matches the closest velocity command to a reference motion and evaluates the polynomials at NORMALIZED phase t = (i % nb_steps) / nb_steps (see the [RL Training](#rl-training) section for the current v3 design and `docs/jetson-mod/experiment_journal.md` for the phase-bug history). (The AMP clip-conversion path that also consumed this library was removed 2026-07-26 with the course-study archive.)

The polynomial data lives at: `isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl`

### Training Commands

```bash
# PPO via RSL-RL (primary)
cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/train_ppo.py \
    --task Isaac-Velocity-Rough-OpenDuck-v0 \
    --headless --video --video_length 200 --video_interval 5000

# Play/evaluate a trained policy
cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/play_policy.py \
    --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
    --num_envs 4 --checkpoint <path> --headless --video --video_length 500
```

### Policy Export

```bash
# RSL-RL exports ONNX automatically after play
# Then convert to TensorRT on Jetson: trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16
```

---

## RL Training

_Reinforcement learning training details — observation/action spaces, imitation reward, joint orders, and pipeline_

### RL Overview for This Project

The robot learns to walk through trial-and-error in simulation (Isaac Sim). A neural network "policy" observes the robot's state and outputs motor commands. PPO is the primary algorithm, with reference motion imitation as the dominant reward signal.

### Observation Space

**The row order below IS the concatenation order — do not reorder it.** A prior
revision of this table listed "Velocity command" second-to-last; it is actually
the 4th term in the v3 layout and the 3rd in v5d. Corrected 2026-08-02 against
the exported `env.yaml` term order.

**Current layout — 53 dims** (Task M0b, 2026-08-13). Read back off a freshly
built environment, not computed:

| # | Component | Dims | Slice | Isaac Lab term |
|---|---|---|---|---|
| 1 | Base angular velocity | 3 | `[0:3]` | `base_ang_vel` — gyro |
| 2 | Projected gravity | 3 | `[3:6]` | `projected_gravity` — from IMU |
| 3 | Velocity command | 3 | `[6:9]` | `generated_commands` — (vx, vy, yaw_rate) |
| 4 | Joint positions | **14** | `[9:23]` | **`joint_pos_rel`** — `q − q_default`, NOT absolute; **delayed 0–2 control steps** (PLANT-7) |
| 5 | Joint velocities | **14** | `[23:37]` | **`joint_vel_rel`** — minus default joint vel (zero, so numerically absolute); same delay |
| 6 | Previous action | **14** | `[37:51]` | `last_action` |
| 7 | Gait phase | 2 | `[51:53]` | `[cos(phase), sin(phase)]` |

> **The antennas are gone from the interface, and that was the point.**
> Terms 4 and 5 were 16 wide and included `left_antenna`/`right_antenna`. Those
> joints are driven on hardware by **SG90 micro servos with no position
> feedback**, so four of the old 59 dims — `joint_pos_rel[22, 23]` and
> `joint_vel_rel[38, 39]` — **could not be measured on the real robot at all**
> (DEPLOY-3). Feeding zeros is distribution shift; feeding the commanded angle
> is also distribution shift. There was no correct value.
>
> **53, not 55.** Term 6 is `last_action` called with `action_name=None`, so it
> returns the whole action tensor and shrinks 16 → 14 by itself. Subtracting
> only the four joint dims gives 55 and is wrong.
>
> The antennas remain in the model as passive links held at `q_default` by
> their own actuator group. **The hardware runtime must do the same** — do not
> drive them from the policy.

**Historical (v5d_contact_wrench and earlier) — 59 dims.** Terms 4/5/6 at 16
wide. Every checkpoint through v5d has this layout and **will not load against
the current config**; that is deliberate and irreversible (`task_plan_v2.md`
M0b). Their re-gate on the corrected plant is `m2657_regate.md`.

> **Deployment trap — read this before writing the Jetson obs builder.**
> Term 4 is `joint_pos_rel`, so the runtime must send `q − q_default`, not raw
> encoder angles. The offset is the entire standing pose: large, constant, and
> it would not present as noise. Export `q_default` (`robot_cfg.py`
> `init_state.joint_pos`) alongside the policy.

**Historical (v3, retired) — 62 dims.** Same terms with `base_lin_vel(3)`
prepended at `[0:3]`, shifting velocity command to `[9:12]`. `base_lin_vel` is
not directly measurable by the BNO055, which is why the v4/v5 layout drops it.

### Action Space (14 dimensions)

Joint position targets for the 14 **non-antenna** actuators (Task M0b). Scaled by `action_scale` (0.25, matching Open Duck Playground) and offset by `init_pos`.

### Joint Orders

**MuJoCo / MJCF order** (joint declaration AND actuator order in
`robot.xml` and `robot_motors.xml` — verified; identical to the Playground
polynomial order below):
```
0: left_hip_yaw      8: head_roll
1: left_hip_roll     9: left_antenna
2: left_hip_pitch   10: right_antenna
3: left_knee        11: right_hip_yaw
4: left_ankle       12: right_hip_roll
5: neck_pitch       13: right_hip_pitch
6: head_pitch       14: right_knee
7: head_yaw         15: right_ankle
```
WARNING: an earlier version of this table listed a right-leg-first order —
that was the legacy 15-joint BDX layout from `rl_utils.py`, NOT this model.
Never hardcode a joint order from documentation; derive mappings from joint
names (as `imitation_reward.py` does) or assert against the model file
(as the archived `convert_gait_library_to_amp.py` did).

**Isaac Lab / USD order** (from MJCF→USD conversion, interleaved):
```
0: left_hip_yaw      8: right_hip_pitch
1: neck_pitch        9: left_knee
2: right_hip_yaw    10: head_roll
3: left_hip_roll    11: right_knee
4: head_pitch       12: left_ankle
5: right_hip_roll   13: left_antenna
6: left_hip_pitch   14: right_antenna
7: head_yaw         15: right_ankle
```

**Playground polynomial order** (used in polynomial_coefficients.pkl):
**identical to the MuJoCo / MJCF order above** — verified byte-for-byte, which is
why only one listing is kept here. Read it from the MJCF block above rather than
from a second copy that can drift out of sync with it.

`mini_bdx/mini_bdx/utils/rl_utils.py` contains MuJoCo↔IsaacGym conversion tables for the LEGACY 15-joint BDX robot (no head_roll) — do not reuse them for this 16-joint model. The Isaac Lab imitation reward (`imitation_reward.py`) builds the Playground↔Isaac Lab mapping dynamically from joint names.

### RL Algorithm

PPO via RSL-RL on DGX Spark. 4096 parallel envs. Environment extends `LocomotionVelocityRoughEnvCfg`.

### Reward Functions (v3 — corrected BDX-aligned composite imitation)

SOURCE OF TRUTH: `isaac_lab_env/open_duck_mini_v2/env_cfg.py` (DuckRewards)
and `imitation_reward.py`. 9 terms: 4 positive + 5 penalties.

**Positive rewards:**
- `alive_bonus` (mdp.is_alive): weight=+10.0 (BDX paper uses +20)
- `imitation_reward` (ImitationReward class): RewTerm weight=1.0, BDX
  sub-weights baked in: joint_pos -L2*15.0 (raw quadratic), joint_vel
  -L2*0.001, base_vel exp(-8e)*1.0, contact match *1.0. Reference evaluated
  at NORMALIZED phase t=(i % nb_steps)/nb_steps (the v2 phase bug imitated
  only 54% of the cycle — see experiment_journal.md run 1); reference
  clamped to soft joint limits; gated off for near-zero commands.
- `track_lin_vel_xy_exp`: command tracking (std=0.5, weight=1.0)
- `track_ang_vel_z_world_exp`: yaw tracking (std=0.5, weight=0.5)

**Penalties:**
- `is_terminated`: fall penalty (-200)
- `flat_orientation_l2`: stay upright (-2.0)
- `action_rate_l2`: smoothness (-1.0)
- `joint_pos_limits`: servo protection for ankle/knee (-1.0)
- `joint_deviation_head`: head stabilization (-0.1) — head is 21% of body mass

**Reward evolution** (full history in `docs/jetson-mod/experiment_journal.md`):
- iter1: H1-derived, 13 penalties → structurally negative, failed
- iter2: alive bonus +5 → crouching/shuffling exploit, failed
- iter3: no alive bonus + height control → unnatural gait
- iter4 (= policy "v1"): exp-kernel imitation ×10 + phase obs → walks
- v2 (BDX-aligned): raw-quadratic composite — but the phase-in-seconds bug
  produced a measurable limp (stance asym 14.6pp, ROM ratio 0.82)
- v3 (current): phase fix + clamp + gating + command clip → G1 PASS
  (stance asym 4.3pp, ROM ratio 1.02, 4.59° RMS vs true reference)

Velocity command ranges (clipped to the reference-motion grid hull):
- lin_vel_x: (-0.148, 0.222) m/s
- lin_vel_y: (-0.111, 0.111) m/s
- ang_vel_z: (-0.5, 0.5) rad/s

Action scale: 0.25 (matching Open Duck Playground)

### v4 Tracks (post-CAD retrain)

- **Run A ("v4-inertials")**: task `Isaac-Velocity-Rough-OpenDuck-v0`
  unchanged — retrains the v3 recipe on the layout-v2.1 model (new masses,
  frame-correct fullinertia, cut meshes). Isolates the model-change effect.
- **Run B ("v4-robust")**: task `Isaac-Velocity-Rough-OpenDuck-Robust-v0`
  (`OpenDuckRobustEnvCfg` + `OpenDuckRobustPPORunnerCfg`):
  - dynamics DR: pushes (±0.3 m/s, 8-14 s), trunk mass ±(-0.10,+0.15) kg,
    trunk CoM ±10/±5 mm, friction 0.4-1.0/0.3-0.8, joint-reset scale 0.9-1.1
  - asymmetric obs: actor = 59 dims (NO base_lin_vel — the BNO055 cannot
    measure it), critic = 62 dims privileged, uncorrupted
    (`obs_groups={"actor": ["policy"], "critic": ["critic"]}`)
  - actuator velocity_limit_sim = 8.94 rad/s (BAM sts3250 id008)
- **Push-recovery gate**: task `Isaac-OpenDuck...PushEval-v0`
  (`OpenDuckPushEvalEnvCfg`) — play determinism with interval pushes ON;
  feeds docs/jetson-mod/validation_results.md (Task 2.7).
- Deployed ONNX for Run B consumes the 59-dim actor layout (v3's was 62).
  Surviving term order (the Jetson obs builder must emit exactly this):
  base_ang_vel(3), projected_gravity(3), velocity_commands(3),
  **joint_pos_rel**(16), **joint_vel_rel**(16), actions(16), gait_phase(2).
  The `_rel` matters and was missing here until 2026-08-02: the policy consumes
  `q - q_default`, not raw encoder angles (`env.yaml:472` binds `joint_pos_rel`).
  Ship `q_default` with the policy. Full table: [Observation Space](#observation-space).

### AMP Track (removed 2026-07-26)

The skrl AMP track (envs, motion clips, train/play/measure/export scripts)
was removed with the course-study archive. Full implementation + run history:
archive repo `open-duck-ppo-vs-amp` (in-repo history at tag
`course-study-freeze`). Study run entries remain in the experiment journal
as historical record.

### PPO Hyperparameters

- gamma: 0.97
- entropy_coef: 0.005
- init_noise_std: 0.5
- obs_normalization: True (actor and critic)

### Actuator Configuration (STS3250)

See [Actuator Configuration](#actuator-configuration) above — the snippet was
duplicated verbatim here and is now kept in one place so the two copies cannot
drift. The torque ceiling it sets carries a caveat: see
[known_issues.md PLANT-5](docs/jetson-mod/known_issues.md#plant-5).

### Policy Network Architecture

MLP with 3 hidden layers: [512, 256, 128], ELU activation. Separate policy and value networks.

### Training Video Recording

Always record training progress videos for debugging and documentation. Use these flags on all training runs:

```bash
--video --video_length 200 --video_interval 5000
```

Videos saved to `logs/<workflow>/<task>/<run>/videos/train/`. Use `play.py --video` for evaluation clips.

### Training → Deployment Pipeline

```
Isaac Lab (DGX Spark) → .pt checkpoint → ONNX export → TensorRT engine → Jetson inference
```

### Key Files

- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — Environment config (rewards, observations, terrain)
- `isaac_lab_env/open_duck_mini_v2/imitation_reward.py` — ImitationReward class + gait_phase_observation
- `isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl` — 240 polynomial walking gaits
- `isaac_lab_env/open_duck_mini_v2/agents/rsl_rl_ppo_cfg.py` — PPO hyperparameters
- `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — Robot articulation config
- `exported_policies/v7_servo_safe_ppo/` — **THE locomotion policy.** Supersedes v6d on 2026-08-16; see [`servo_fix_results.md`](docs/jetson-mod/servo_fix_results.md) and [`locomotion_selection.md`](docs/jetson-mod/locomotion_selection.md). obs 53 / action 14, plant 2.729 kg, `model_8997.pt` md5 `2e7dda7a771768052d856612843b100b`. Fixes SERVO-1 (the neck no longer stalls against its end stop: 100 % -> 0 % of steps, 207 % -> 28 % of the servo's continuous rating) and mitigates SERVO-2 (worst leg RMS 174 % -> 131 % of rated). Every other policy is **retired**. `*.onnx` is gitignored, so the dir's `README.md` md5 is the only provenance. **`deployment_contract.json` is required to deploy.** SERVO-2 is NOT closed — the remaining gap is a mass/gearing problem for Task S.8.
- `experiments/v2/params_sts3250_id008.json` — BAM motor identification parameters (STS3250)
- `mini_bdx/mini_bdx/utils/rl_utils.py` — Joint order conversion, action scaling

---

## Jetson Deployment

_Jetson Orin Nano deployment details — TensorRT, Cosmos vLLM, GPIO pins, safety rules, power_

Rules and context for code that runs on the Jetson Orin Nano Super.

### Two-Thread Architecture

The robot runs two concurrent loops on the Jetson GPU:

```
Thread 1: Cosmos Reason2 (2-3 Hz) — high-level reasoning via vLLM API
Thread 2: Locomotion policy (50 Hz) — low-level walking via TensorRT
```

They share the GPU. Locomotion takes <1 ms, Cosmos takes ~300-500 ms. They time-share without conflict.

### GPIO Pin Mapping (Pi → Jetson)

| Function | Pi GPIO | Jetson GPIO | Pin |
|---|---|---|---|
| Left Eye LED | 23 | 23 | 16 |
| Right Eye LED | 24 | 24 | 18 |
| Projector LED | 25 | 25 | 22 |
| Left Antenna PWM | 12 | 12 | 32 |
| Right Antenna PWM | 13 | 13 | 33 |
| Left Foot Switch | 22 | 22 | 15 |
| Right Foot Switch | 27 | 27 | 13 |
| IMU SDA | 2 | 2 | 3 |
| IMU SCL | 3 | 3 | 5 |

**Important:** Verify against the actual Jetson Orin Nano dev kit carrier board pinout before wiring. Use `Jetson.GPIO` library (not `RPi.GPIO`).

### TensorRT Policy Inference

```python
# Convert ONNX to TensorRT on the Jetson itself:
trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16

# Inference wrapper: jetson_runtime/trt_infer.py
#
# READ exported_policies/v6d_contact_wrench_ppo/deployment_contract.json FIRST.
# It is generated with the policy and machine-checked by
# scripts/verify_deployment_contract.py; the numbers below are a summary of it,
# and the JSON wins if they ever disagree.
#
# Input: 53-dim float32 — v6d actor layout, exact order: base_ang_vel(3),
# projected_gravity(3), velocity_commands(3), joint_pos_rel(14),
# joint_vel_rel(14), actions(14), gait_phase(2).
#   joint_pos_rel = q - q_default, NOT raw encoder angles. Absolute angles put
#   the knees 13-14 sigma outside the training distribution (DEPLOY-2).
# Output: 14-dim float32 action vector.  Batch dim is HARD-FIXED at 1.
#
# THE GRAPH IS NOT THE POLICY (known_issues.md DEPLOY-1, measured on this very
# export): action_scale = 0.25 appears in 0 of 193,784 initializer scalars, and
# q_default is absent -- the closest action-width tensor differs by 1.374409 rad.
#   q_target = q_default + 0.25 * action
# lives inside Isaac Lab's JointPositionAction. Commanding the ONNX output
# directly is wrong by a 4x gain AND the whole standing pose: total, silent
# failure. Take both from the contract.
# The normaliser epsilon (0.01) is likewise not in the checkpoint; a
# reimplementation that divides by _std alone is off by 28.0 % on one channel.
#
# Joint counts are 14, not 16: Task M0b removed the two antenna joints from the
# action and observation spaces, which also closed DEPLOY-3.
# (Historical: v4/v5 were 59-dim obs / 16-dim action; v3 was 62-dim incl.
# base_lin_vel. Those checkpoints no longer load.)
# Latency: <1 ms
```

### Cosmos Reason2 Deployment

```bash
# Start vLLM server:
docker run --rm -it --runtime=nvidia --network host --shm-size=4g \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve "embedl/Cosmos-Reason2-2B-W4A16-Edge2" \
    --max-model-len 2048 --gpu-memory-utilization 0.70 --max-num-seqs 1

# Query via HTTP API from Python:
# POST http://localhost:8000/v1/chat/completions
# Send camera frame as base64 image + text prompt
# Parse JSON velocity command from text response
```

### Safety Rules

- **All velocity commands MUST be clamped to the trained command hull:**
  forward `[-0.148, 0.222]` m/s, lateral `[-0.111, 0.111]` m/s, turn
  `[-0.3, 0.3]` rad/s. The forward and lateral figures are the hull the
  policy was trained on (`env_cfg.py:301-303`, and the shipped policy's
  archived `env.yaml`). **The turn clamp is deliberately tighter than the
  trained ±0.5** — a conservative deployment choice, not the hull. Do not
  "correct" it to ±0.5.
- On any Cosmos inference failure, fall back to last known good command (not zero — that could cause mid-stride fall)
- IMU-based emergency stop: if tilt > 60 degrees, cut all motors
- Servo current monitoring: if any servo exceeds safe current, reduce velocity

### Power

- Battery: 6x 18650 Li-ion cells (3S2P, 11.1V) → DC-DC boost (11.1V → 19V) for Jetson barrel jack
- Run Jetson at 7W eco mode for maximum battery life (67 TOPS still available)
- Estimated battery life: ~1-2 hours at 7W, ~30 min at 25W

---

## Experiment Journal Protocol

_Experiment journal protocol — every training run must be journaled with last-100 TensorBoard means, measured gate rollouts, and named data sources_

Every training run MUST get an entry in `docs/jetson-mod/experiment_journal.md`
before the next run launches. The journal's numbers must be reproducible
from primary sources, never from memory or mid-training log greps. (The
EN.665.645 paper's copy of this journal was frozen 2026-07-26 into the
archive repo `open-duck-ppo-vs-amp` — tag `course-study-freeze`; entries here
from that point on serve the robot project's engineering record.)

### Data-sourcing rules (non-negotiable)

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

### Entry structure

Per run: one-lever config delta | training signals (TB, per rule 1) | gate
evaluation (per rule 3) | verdict with gate name | artifact paths (checkpoint,
ONNX, videos, log dir). Update the run-index table at the top of the journal.

### Log dirs

- PPO (RSL-RL): `~/IsaacLab/logs/rsl_rl/open_duck_ppo/<timestamp>/`
- PPO robust track (Run B+): `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/<timestamp>/`
- AMP (skrl, archived study era): `~/IsaacLab/logs/skrl/<experiment.directory>/<timestamp>_amp_torch/`
- Detached run console logs + PIDs: `.training_runs/<run_name>.{log,pid}`

### Companion documents

- Comparison tables are PER MODEL: corrected-model (v2.1) policies go to
  `v4_comparison.md` + `eval_results_v4/` (or a successor per-model pair for
  future models). Add every gate-passing policy to the table matching its
  robot model. (The pre-CAD study table `algorithm_comparison.md` +
  `eval_results/` was removed 2026-07-26 — it lives in the archive repo
  `open-duck-ppo-vs-amp` and at tag `course-study-freeze`.)
- `exported_policies/<name>/` — archive checkpoint + `agent.yaml` +
  `env.yaml` + README for any deployment-candidate policy.

---

## Task Plan Reference

_Task plan summary — 5 phases, 32 tasks, dependencies, and output files reference_

The detailed implementation plan is in `docs/jetson-mod/task_plan.md` (2800+ lines, 32 tasks). For the rebuild triggered by the 2026-08-11 PLANT-1 fix, use `docs/jetson-mod/task_plan_v2.md` instead — it is written task-at-a-time for an implementing agent and supersedes v1 wherever they overlap.

### Phase Summary

| Phase | Tasks | Status | Where |
|---|---|---|---|
| Phase 1: Sim Model Update | 1.1-1.6 | Complete | Local machine |
| Phase 2: Isaac Lab + RL Training | 2.1-2.8 | Complete (Task 2.7 push-recovery gate PASSED — validation_results.md, v4_robust; Task 2.8 v5 contact-rich retrain shipped v5d_contact_wrench) | DGX Spark |
| Phase 3: CAD Redesign + Thermal | 3.1-3.7 | In progress (cad-redesign merged to v2 2026-07-26) | OnShape/Fusion |
| Phase 4: Hardware Build + Thermal Mgmt | 4.1-4.6 | Pending | Jetson + 3D printer |
| Phase 5: Cosmos VLM Integration | 5.1-5.5 | Pending | Jetson |

### Key Task Dependencies

- Phase 1 → Phase 2: `robot_motors.xml` (MJCF) must be updated before Isaac Sim import
- Phase 2 → Phase 3: RL policy must walk in sim before committing to CAD changes
- Phase 3 → Phase 4: Parts must be redesigned before printing
- Phase 4 → Phase 5: Robot must walk physically before adding VLM

### When Working on Tasks

1. Read the relevant task in `docs/jetson-mod/task_plan.md` for full context
2. Each task has: description, steps, automated test code, and manual verification checklist
3. Mark tests as passing before moving to the next task
4. Document results in `docs/jetson-mod/validation_results.md` or `docs/jetson-mod/v4_comparison.md`

### Output Files Created by Tasks

| Task | Output |
|---|---|
| 1.1 | `docs/jetson-mod/mass_inertia_calculations.md` |
| 1.2 | `mini_bdx/robots/open_duck_mini_v2/jetson_orin_nano.stl` |
| 2.1 | `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd` |
| 2.5 | `docs/jetson-mod/v4_comparison.md` (study-era `algorithm_comparison.md` archived in `open-duck-ppo-vs-amp`) |
| 2.6 | `exported_policies/<name>/policy.onnx` (currently `v5d_contact_wrench_ppo/`) |
| 2.7 | `docs/jetson-mod/validation_results.md` |
| 3.6 | `print/thermal_partition.stl` |
| 4.6 | `jetson_runtime/thermal_manager.py` |
| 5.4 | `docs/jetson-mod/prompt_engineering_results.md` |

---

## Tooling

_Tooling instructions for diagram export and other project utilities_

### Exporting draw.io Diagrams to PNG

The wiring diagram at `docs/jetson-mod/jetson_wiring_diagram.drawio` has a corresponding PNG render at `docs/jetson-mod/jetson_wiring_diagram.png`. When the `.drawio` file is modified, the PNG must be re-exported.

#### Method: Headless Export via Puppeteer + draw.io Viewer

This machine has no draw.io desktop app. Use the npm `@mattiash/drawio-export` package or the following puppeteer-based approach:

```bash
# Requires: node, npm, chromium (all available on DGX Spark)
# One-time setup (in /tmp):
cd /tmp && npm install puppeteer

# Export script:
cat > /tmp/export_drawio.mjs << 'JSEOF'
import puppeteer from 'puppeteer';
import fs from 'fs';

const drawioFile = process.argv[2];
const outputFile = process.argv[3];
const xmlContent = fs.readFileSync(drawioFile, 'utf8');
const match = xmlContent.match(/<mxGraphModel[\s\S]*?<\/mxGraphModel>/);
if (!match) { console.error('No mxGraphModel found'); process.exit(1); }
const b64 = Buffer.from(match[0]).toString('base64');

const html = `<!DOCTYPE html>
<html><head><style>body{margin:0;padding:10px;background:white;}</style></head><body>
<div id="graph" class="mxgraph"></div>
<script>
var div=document.getElementById('graph');
div.setAttribute('data-mxgraph',JSON.stringify({highlight:'#0000ff',nav:false,resize:true,toolbar:'',edit:'_blank',xml:atob("${b64}")}));
</script>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body></html>`;

const browser = await puppeteer.launch({
  executablePath: '/snap/bin/chromium',
  headless: 'new',
  args: ['--no-sandbox','--disable-setuid-sandbox','--disable-gpu']
});
const page = await browser.newPage();
await page.setViewport({width:2200,height:1800});
await page.setContent(html,{waitUntil:'networkidle0',timeout:30000});
await new Promise(r=>setTimeout(r,6000));
await page.screenshot({path:outputFile,type:'png',fullPage:true});
console.log('Exported '+fs.statSync(outputFile).size+' bytes');
await browser.close();
JSEOF

# Run:
cd /tmp && node /tmp/export_drawio.mjs \
  /path/to/diagram.drawio \
  /path/to/output.png
```

#### When to Re-export

Re-export the PNG whenever `jetson_wiring_diagram.drawio` is modified (e.g., voltage changes, component additions). The PNG is used in documentation and should always match the drawio source.
