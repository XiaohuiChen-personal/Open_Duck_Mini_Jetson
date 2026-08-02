# Simulated Plant Fidelity — Where Isaac Differs From the Robot

Recorded 2026-08-02. Raised while auditing the plant for the EN.535.782 haptic
course project, which uses the v5d policy as a fixed baseline.

**Nothing here has been fixed.** Every fix changes the plant and therefore
invalidates the trained policies, so each is a decision for whoever owns the
next training campaign. This document exists so the decision is made knowingly
instead of by default.

Reproduce the headline number:

```bash
cd ~/IsaacLab && ./isaaclab.sh -p \
  ~/Projects/Open_Duck_Mini_Jetson/scripts/audit_plant_mass.py --headless
```

---

## 1. The phantom kilogram — Isaac simulates 3.657 kg, the robot is 2.657 kg

### 1.1 Every file on disk is correct

This is not a data-entry error, which is why it survived a CAD revision, a USD
regeneration, three policy generations, and an adversarial review:

| Source | Total mass | `base` body |
|---|---:|---:|
| CAD (`docs/jetson-mod/mass_inertia_calculations.md`) | 2.657067 kg | — |
| MJCF `robot_motors.xml`, summing `<inertial>` | 2.657067 kg | no `<inertial>` tag |
| MuJoCo, building a model from that MJCF | 2.657067 kg | 0.000000 kg |
| USD, summing its own authored `MassAPI` values | 2.657067 kg | unauthored |
| **Isaac / PhysX at runtime** | **3.657067 kg** | **1.000000 kg** |

The USD is also not stale — `pytest tests/test_usd_conversion.py` passes all 5
checks, including the MJCF asset-hash and the per-STL mesh manifest.

### 1.2 What actually happens

The MJCF root body `base` is a pure kinematic frame. It carries the
`<freejoint>` and declares no `<inertial>` and no `<geom>`; the real trunk mass
lives on its child `trunk_assembly`, which is welded to it (no joint of its own):

```xml
<body name="base" pos="0 0 0.22">
  <freejoint />
  <body name="trunk_assembly">
    <inertial ... mass="1.089544" .../>
```

MuJoCo handles a massless frame natively and reports total 2.657067 kg. PhysX
does not: it fills the unauthored value with its own defaults. Isaac Lab's
converter documents this path explicitly —
`MjcfConverterCfg.import_inertia_tensor`:

> "Import the inertia tensor from mjcf. Defaults to True. **If the `inertial`
> tag is missing, then it is imported as an identity.**"

The other converter knob that could have covered it does not apply:
`link_density` is `0.0` ("auto-compute"), but auto-computing mass from geometry
requires geometry, and `base` has no `<geom>`.

Result, measured by `scripts/audit_plant_mass.py`:

```
base                                    none      1.000000   +1.000000   NO <inertial> IN MJCF -> PhysX default
...
TOTAL                               2.657067      3.657067   +1.000000

unauthored body 'base': PhysX assigned mass=1.000000 kg, diagonal inertia=(4.000e-03, 4.000e-03, 4.000e-03) kg*m^2
VERDICT: FAIL — simulated total is +1.000000 kg (+37.6%) versus the MJCF (tolerance 0.001 kg)
```

`base` is **27.3%** of the simulated robot, and it is a body that does not
physically exist.

The inertia matters as much as the mass and is easier to overlook: the phantom
isotropic `4.0e-3 kg·m²` is comparable to `trunk_assembly`'s own real diagonal
(2.2–5.4e-3), so the **root link's rotational inertia is close to doubled**.

### 1.3 The simulator has been saying so on every single run

This is the uncomfortable part. PhysX emits a warning that names the exact prim,
every time the robot loads — every training run, every evaluation, every play:

```
[Warning] [omni.physx.plugin] The rigid body at /World/envs/env_0/Robot/base/base
has a possibly invalid inertia tensor of {1.0, 1.0, 1.0} and a negative mass,
small sphere approximated inertia was used. Either specify correct values in the
mass properties, or add collider(s) to any shape(s) that you wish to
automatically compute mass properties for.
```

"Negative mass" is USD's `-1.0` unauthored sentinel; `{1.0, 1.0, 1.0}` is the
identity inertia the converter warned it would write. The warning sits in the
several hundred lines of Kit startup noise that every run produces, next to
genuinely benign messages about MaterialX and X servers, so it was read as more
of the same.

### 1.4 The file already contains the fix pattern

The MJCF's other massless frames do not have this problem, because they carry an
explicit placeholder inertial rather than nothing at all:

| Body | mass |
|---|---:|
| `trunk` | 1e-09 |
| `left_foot` | 1e-09 |
| `right_foot` | 1e-09 |
| `head` | 1e-09 |

`base` is the **only** one of all 22 bodies with no `<inertial>` element at all,
and therefore the only one PhysX has to invent numbers for. Whether
`onshape-to-robot` treats the root differently by design or by omission is not
knowable from here — but the practical point is that the shape of the fix is
already demonstrated four times in the same file. See §5 for why `base` still
cannot simply copy the `1e-09` value.

### 1.5 It comes from upstream, and it is not upstream's bug

The massless `base` wrapper is present in the very first commit of
`robot_motors.xml` in this repository (`64c9940`), inherited from
`apirrone/Open_Duck_Mini`. It is simply how `onshape-to-robot` emits a
free-floating root.

Upstream trains in MuJoCo/MJX, where mass 0 on that frame is correct and
harmless. **The defect is created at the MuJoCo→PhysX boundary**, which only
this fork crosses. Any project converting this robot — or any
`onshape-to-robot` MJCF with a bare root frame — into Isaac Sim inherits the
same phantom kilogram.

### 1.6 Confirmed physically, not just from metadata

Metadata could in principle be misread; the dynamics cannot. Over a 30 s
steady walk at `vx = 0.2 m/s` (v5d, Play task, 1 env), mean total ground
reaction force is **37.50 N**. In steady walking that must equal `m·g`:

| Candidate mass | Predicted weight | Error vs measured 37.50 N |
|---|---:|---:|
| 2.657067 kg (CAD) | 26.06 N | **+43.9%** |
| 3.657067 kg (PhysX) | 35.87 N | **+4.5%** |

The residual +4.5% is the 3-vector force norm slightly over-reading the vertical
component. The physics follows PhysX.

---

## 2. Domain randomization does not cover it

The startup event that would be the natural mitigation misses on both axes.
`env_cfg.py:361`, in `OpenDuckRobustEnvCfg`:

```python
self.events.add_base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="trunk_assembly"),
        "mass_distribution_params": (-0.10, 0.15),
        "operation": "add",
    },
)
```

1. **Wrong body.** It perturbs `trunk_assembly`. The phantom mass is on `base`.
2. **Wrong magnitude.** Its range is −100 g to +150 g. The error is **1000 g —
   6.7× the upper bound** — and it is a constant offset, not a sampled one, so
   it does not even appear as tail variance.

v5d inherits this term unchanged: `OpenDuckContactWrenchEnvCfg` →
`OpenDuckContactMinimalEnvCfg` → `OpenDuckRobustEnvCfg`, with no override. Only
the `_PLAY` variants set `add_base_mass = None`.

So every Isaac-trained policy in this repo learned to walk a plant 37.6% heavier
than the robot, with no randomization exposure to the discrepancy and therefore
no learned robustness to it.

### 2.1 The general lesson

`add_base_mass` is named for the *concept* "base mass" but bound to the *body*
`trunk_assembly`. That naming gap is exactly what hid the miss. Any
randomization term that names a body should be checked against the body that
actually carries the uncertainty — a term can be present, correctly configured,
gate-validated, and still randomize the wrong thing.

---

## 3. Impact

**Affects every policy trained in Isaac against this USD** — v3, v4_robust and
v5d alike. It is systematic, not a v5d regression. The v5d checkpoint
(`model_5998.pt`, 2026-07-29) postdates the USD regeneration (`8f09c31`,
2026-07-06), so it certainly trained on the affected asset.

**In simulation, nothing is broken.** Training and evaluation share one USD, so
policies and plant are self-consistent and all sim-only results remain
internally valid — they are results for a 3.657 kg robot. The EN.535.782 course
project explicitly accepted this on 2026-08-02 and did not retrain; it quotes
3.657 kg as the plant mass and names the gap as a limitation.

**On hardware, this is a first-order transfer risk.** A policy trained to lift
37.6% more mass, with root rotational inertia near doubled, will command
systematically excessive torque. It compounds with §4.5: the simulator also
allows each servo 1.78× its datasheet stall torque, so the policy has two
independent reasons to ask a real servo for more than it can give.

---

## 4. Other fidelity gaps found in the same audit

Each was found by reading the artifact and re-checked with the command shown.
Ordered by what would hurt most on hardware.

> **§4.1–4.3 are documentation defects and were FIXED in the same commit as this
> document** — `AGENTS.md` now states the correct values. They are recorded here
> anyway because the wrong values were load-bearing: they were the contract a
> future Jetson runtime would have been built against. Quoted line numbers refer
> to `AGENTS.md` *before* that fix. **§4.4–4.12 are plant/behaviour/deployment defects and remain
> open** — fixing any of them changes the plant and forces a retrain.

### 4.1 The deployment contract omits that joint positions are *relative* — HIGH

`exported_policies/v5d_contact_wrench_ppo/env.yaml:472` binds the `joint_pos`
observation to **`joint_pos_rel`**, not `joint_pos`:

```yaml
    joint_pos:
      func: isaaclab.envs.mdp.observations:joint_pos_rel
```

`joint_pos_rel` is `q − q_default`. But both places that specify the deployment
contract describe an absolute angle:

- `AGENTS.md:398` — "Joint positions | 16 | Current angle of each joint (rad)"
- `AGENTS.md:521-523` — "the Jetson obs builder must emit exactly this:
  ... joint_pos(16), joint_vel(16), ..."

A Jetson obs builder written to that contract would feed absolute encoder
angles into 16 inputs the policy expects centred on zero. The offset equals the
whole standing pose, so the error is large, constant, and would not look like
noise — it would look like the policy is broken. **Write `joint_pos_rel` and the
`q_default` vector into the contract before anyone builds the runtime.**

(`joint_vel` is bound to `joint_vel_rel`, which subtracts the default joint
velocity — zero — so that one is numerically absolute. Name it anyway.)

### 4.2 The observation-space table lists terms in the wrong order — HIGH

`AGENTS.md:393-405` puts "Velocity command" **7th**, after "Previous action".
It is actually **4th** in the v3 layout the table describes, at dims `[9:12]`,
and **3rd** at dims `[6:9]` in the current v5d 59-dim layout (env.yaml term
order: `base_ang_vel, projected_gravity, velocity_commands, joint_pos,
joint_vel, actions, gait_phase`).

The table is also still captioned "Total: 62 dims" against the retired v3
policy. `AGENTS.md:521-523` states the correct order for the live policy, so the
repo contains one right answer and one wrong one — the failure mode is someone
finding the wrong one first.

### 4.3 Documented termination thresholds match nothing in the code — HIGH

`AGENTS.md:345-348` claims:

> - Trunk height < 0.08 m (fallen)
> - Trunk tilt > 90 degrees (flipped)

`env_cfg.py:591,594` (and again at `700,703`) actually use:

```python
func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
func=mdp.root_height_below_minimum, params={"minimum_height": 0.09}
```

Both numbers are wrong and both are wrong in the *permissive* direction, which
matters because "fall rate" is a headline gate in every policy comparison in
this repo. The real gate terminates earlier (60° tilt, 9 cm) than documented, so
published fall rates were measured against a stricter bar than the docs
describe. The measurements are fine; the description is not.

### 4.4 The antennas are simulated as 50 kg·cm servos — MEDIUM

`robot_cfg.py:96-105` puts `.*_antenna` in the `head` actuator group with full
STS3250 parameters:

```python
"head": ImplicitActuatorCfg(
    joint_names_expr=["neck_pitch", "head_pitch", "head_yaw",
                      "head_roll", ".*_antenna"],
    stiffness=45.53, damping=1.346, armature=0.040,
    friction=0.200, effort_limit_sim=8.716,
```

`AGENTS.md:160-161` records the real hardware as "14x Feetech STS3250" **plus
"2x SG90 micro servos"** for the ears/antennas. An SG90 stalls around
0.18 N·m, so the simulated antenna actuator has roughly **48× the real torque
ceiling**, plus an armature (0.040) sized for a servo an order of magnitude
heavier.

The `armature` is the sharper problem. Armature is reflected rotor inertia,
added to the joint's own inertia. Each antenna link weighs 4.216 g with
principal inertias of `2.45e-07` to `3.87e-06 kg·m²`:

```
left_antenna_holder:  mass=0.004216 kg  diaginertia=[3.80836e-06, 3.59678e-06, 2.45273e-07]
right_antenna_holder: mass=0.004216 kg  diaginertia=[3.86998e-06, 3.65850e-06, 2.45146e-07]
```

So `armature = 0.040` is **roughly four orders of magnitude larger than the
link it is attached to** (≈10,000× against the largest principal axis, more
against the smallest). Those two joints are not simulating an antenna driven by
a micro servo; they are simulating a flywheel. Their dynamics are set almost
entirely by a fictitious inertia.

Low consequence for locomotion — the antennas are 4.2 g each and the policy
barely uses them — but any antenna motion the policy learns is not reproducible
on hardware, and both joints are in the 16-dim action vector the Jetson will
send to real servos.

### 4.5 The simulated torque ceiling is 1.78× the datasheet stall torque — MEDIUM

`robot_cfg.py:94` sets `effort_limit_sim=8.716` N·m from BAM's `forcerange`
(`experiments/v2/params_sts3250_id008.json` → `mujoco_export.forcerange:
8.716130441407099`). `AGENTS.md:160` documents the STS3250 as **50 kg·cm
stall = 4.90 N·m**. The ratio is **1.78×**.

The two are not measuring the same thing and both are defensible in isolation:
BAM's figure is the electrical stall torque from its identified model
(`kt·V/R = 1.0006 × 12.1 / 1.389 = 8.716`), while the datasheet number is the
rated stall at the manufacturer's test voltage. But the simulator enforces the
BAM number, so **the policy may be relying on torque the hardware cannot
deliver**, and nothing in the repo records that the two disagree by 78%.

Worth resolving before deployment, because it compounds with §1: a policy
trained to lift 37.6% extra mass *and* allowed 1.78× the datasheet torque has
two independent reasons to over-command on a real servo. The cheap check is a
stall-torque measurement on one servo at the deployment battery voltage.

**And the ceiling is pinned to a voltage the robot rarely sees.** BAM identified
at `vin: 12.1` V. The robot runs a 3S2P 18650 pack — `AGENTS.md:168`, "11.1V"
nominal. Since stall torque scales with supply:

| Supply | `kt·V/R` |
|---|---:|
| 12.1 V (BAM identification) | **8.716 N·m** ← what Isaac enforces |
| 12.0 V (servo rated) | 8.644 N·m |
| 11.1 V (pack nominal) | 7.996 N·m |
| 9.9 V (pack near empty) | 7.131 N·m |

So the real ceiling drifts ~8–18% below the simulated one over a discharge, and
**no domain randomization touches any actuator parameter** — not stiffness, not
damping, not effort limit. The simulated servo is a constant-strength ideal at
the top of its voltage range for the whole of training.

> The remaining BAM parameters check out exactly: `kp 45.53`, `damping 1.346`,
> `armature 0.040`, `friction 0.200` in `robot_cfg.py` all match
> `params_sts3250_id008.json`. Only the ceiling is contentious.

### 4.6 The head half of `ground_contact_penalty` can never fire — MEDIUM

`env_cfg.py:515-524` binds the v5 ground-contact penalty to body `head`:

```python
ground_contact = RewTerm(
    func=gated_rewards.ground_contact_penalty,
    weight=-0.5,
    params={"sensor_cfg": SceneEntityCfg(
        "contact_forces", body_names=["head", "knee_and_ankle_assembly.*"]),
        "threshold": 1.0},
)
```

`head` is one of the four 1e-09 kg marker frames from §1.4. It has **no
`<geom>`**, therefore no collider, therefore its contact force is identically
zero — confirmed by `scripts/audit_plant_mass.py`, which lists `head` as a real
PhysX rigid body of mass 0.000000. The real head is a different body,
`head_assembly` (0.362083 kg, 24 geoms).

This was not intentional. `docs/jetson-mod/v5_retrain_plan.md:157` specifies the
term as "**head** + lower-leg contact > 1 N". The lower-leg half works; the head
half has never been able to fire.

**Scope — the pinned v5d policy is NOT affected.** The term lives in
`DuckContactRewards`, which only `OpenDuckContactEnvCfg` and its
`ContactUngated` subclass use. `OpenDuckContactWrenchEnvCfg` (v5d) descends
`ContactMinimal → Robust` and inherits `DuckRewards` instead —
`grep ground_contact exported_policies/v5d_contact_wrench_ppo/env.yaml` returns
nothing. The dead half therefore affected the `Contact`/`ContactUngated` arms of
the v5 selection ablation, not the policy that won it.

That bounds the damage but does not dismiss it: the ablation that chose v5d
compared arms one of which had a reward term running at half its documented
strength. Fix is one word — `head` → `head_assembly` — and it belongs to the
next campaign, since it changes the reward.

### 4.7 Most training resets spawn the robot inside the ground plane — MEDIUM

`robot_cfg.py:47` spawns at `pos=(0.0, 0.0, 0.17)`, "Spawn height matching
MuJoCo base pos". At the nominal default pose that leaves only **+3.2 mm** of
clearance between the lowest collision vertex and the ground — measured by
transforming every one of the 130 collision meshes' vertices by its
forward-kinematic pose in MuJoCo, using the joint defaults parsed straight out
of `robot_cfg.py`.

Training then randomizes the reset pose (`env_cfg.py:387`,
`reset_robot_joints.position_range = (0.9, 1.1)`), which scales the joint
angles and therefore changes how tall the robot stands. 3.2 mm does not absorb
that. Over 4000 sampled resets:

| | lowest collision vertex |
|---|---:|
| nominal (scale 1.0) | **+3.2 mm** |
| mean over random resets | **−1.9 mm** |
| deepest | **−16.4 mm** |
| highest | +12.2 mm |
| **resets starting below ground** | **61.4%** |

So the majority of training episodes begin with the robot interpenetrating the
floor, and PhysX resolves it by pushing the robot out at up to
`max_depenetration_velocity = 1.0 m/s` (`robot_cfg.py:34`). Every such episode
opens with a depenetration impulse that has no counterpart on hardware.

Scope, precisely: **training only.** All three `_PLAY` configs pin
`position_range = (1.0, 1.0)` (`env_cfg.py:280, 413`), so evaluation, the gate
rollouts, and the course project's baseline all start from the clean +3.2 mm
pose. Published gate numbers are unaffected; what is affected is what the policy
was trained through.

Cheap fix: raise the spawn height a centimetre, or clamp the randomization. Both
change the training distribution, so both belong to the next campaign.


### 4.8 Joint dry friction is inactive while the joints move — MEDIUM

The MJCF gives every joint dry friction in its `<default>` block:

```xml
<joint damping="0.0" armature="0.040" frictionloss="0.200"/>
```

MuJoCo's `frictionloss` opposes motion at **all** joint speeds. Isaac Lab splits
that one number into three, and only the first is set
(`robot_cfg.py:93,104` — frozen into `env.yaml:193-195, 211-213`):

```yaml
friction: 0.2
dynamic_friction: null
viscous_friction: null
```

`None` does not mean "unset, use the sibling"; per `actuator_base.py:176-192` it
means "read the value from the USD". The USD authors **no joint friction at
all** — the only PhysX joint attribute on any of the 16 revolute joints is
`physxJoint:armature`, and the string "friction" does not appear in any of the
four composed USD layers. So the read-back is PhysX's default `0.0`, and Isaac
ends up applying `PxJointFrictionParams(0.200, 0.0, 0.0)`.

Isaac Lab's own semantics (`articulation.py:888-889`) are that **static friction
caps effort only at rest, while dynamic friction is what acts during motion.**
With dynamic friction at zero, the BAM-identified dry friction is present when
the robot is standing still and absent for the entire gait.

Direction of the error: the simulated joints are **easier to move than the real
servos** for the whole of locomotion, which is the unhelpful direction — it
flatters the policy in sim and under-prepares it for hardware. It also means
this fork's plant differs from upstream's MuJoCo plant in a way that is not
recorded anywhere, despite both nominally using "the BAM parameters".

### 4.9 There is no latency model of any kind — MEDIUM

The plant has no control latency, action delay, or observation staleness. Real
deployment runs TensorRT inference on the Jetson and pushes 16 joint targets
over a serial servo bus; `AGENTS.md` budgets "<1 ms" for the inference alone and
says nothing about the bus, and the simulator models neither.

This is an absence rather than a wrong number, so it cannot be quoted as a
discrepancy — but it belongs on this list because it is invisible: nothing in
the config will ever look wrong, and the gap only appears on hardware. It is
also the cheapest of these to address, since Isaac Lab supports action delay
without touching the asset.

### 4.10 v5d never saw a constant yaw-rate command in training — HIGH for teleop

`exported_policies/v5d_contact_wrench_ppo/env.yaml:984-987`:

```yaml
heading_command: true
heading_control_stiffness: 0.5
rel_standing_envs: 0.02
rel_heading_envs: 1.0
```

`rel_heading_envs: 1.0` means **every** environment is a heading environment.
Isaac Lab's `UniformVelocityCommand._update_command`
(`velocity_command.py:150-159`) then overwrites the yaw channel every single
step:

```python
heading_error = wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
self.vel_command_b[env_ids, 2] = torch.clip(
    self.cfg.heading_control_stiffness * heading_error, ...)
```

So the `ang_vel_z` range `(-0.5, 0.5)` was never sampled as a yaw command.
Observation dim 8 always carried a **decaying heading-error signal** — a
P-controller output with stiffness 0.5, i.e. a ~2 s time constant, shrinking
toward zero as the robot turns to face its target.

Consequence: **a sustained constant yaw rate is out of distribution.** Any
consumer that writes `vel_command_b[:, 2]` directly — a joystick, a VLM, the
haptic teleop layer — is feeding the policy a signal shape it never trained on.
The policy still turns, but its yaw tracking should not be assumed to be
characterized by the training ranges.

This is not a plant defect and nothing is misconfigured; it is an undocumented
property of the training distribution that matters the moment anything drives
yaw open-loop. Worth either documenting as a known limitation or fixing in the
next campaign by setting `rel_heading_envs` below 1.0 so some envs train on
direct yaw-rate commands.

### 4.11 Four of the 59 observation dims cannot be measured on hardware — HIGH for deployment

The 16-joint vector includes the two antennas (`robot_cfg.py` `init_state.joint_pos`):

```
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle,
neck_pitch, head_pitch, head_yaw, head_roll, left_antenna, right_antenna
```

`joint_pos` and `joint_vel` therefore each carry an antenna pair, so **4 of the
59 actor inputs are antenna state**. The antennas are SG90 micro servos
(`AGENTS.md:161`) — open-loop PWM hobby servos with no position feedback. The 14
Feetech STS3250s report position and velocity over the serial bus; the SG90s
report nothing.

The Jetson obs builder will therefore have four inputs it cannot fill. Neither
option is free: feeding zeros or the commanded angle is a distribution shift on
those dims, and the policy was trained with them live (and with the wrong
actuator model behind them, §4.4).

This is a deployment blocker that has to be decided before the runtime is
written, and the cheapest resolution is upstream — drop the antennas from the
action and observation spaces in the next training campaign, since they
contribute nothing to locomotion.

### 4.12 The planned TensorRT wrapper is specified against 56 dims — MEDIUM

`docs/jetson-mod/task_plan.md:2076` and `:2092`, in the Phase-4 acceptance
snippet for `jetson_runtime/trt_infer.py`:

```python
obs = np.random.randn(56).astype(np.float32)
action = policy.infer(obs)
```

The pinned v5d actor takes **59**. The number is stale from an earlier obs
layout and nothing has reconciled it. Harmless today because
`jetson_runtime/` does not exist yet — which is exactly why it should be fixed
now, before it becomes the spec someone codes against.

### 4.13 The pattern behind §2, §4.6, and arguably §1

Three independent config sites bind to a body whose **name reads correctly** but
whose **physics is wrong**:

| Site | Names | Should name | Symptom |
|---|---|---|---|
| `add_base_mass` (`env_cfg.py:365`) | `trunk_assembly` | `base` | randomizes a body that has no error, misses the one that does |
| `ground_contact` (`env_cfg.py:521`) | `head` | `head_assembly` | penalizes a collider-less frame; never fires |
| MJCF root (`robot_motors.xml`) | `base` | — | a name suggesting the main body, attached to an empty frame |

The common cause is that this model carries **two parallel naming families**:
real `*_assembly` bodies that own the mass and geometry, and short kinematic
marker frames (`base`, `trunk`, `head`, `left_foot`, `right_foot`) that own
nothing. The short names are the intuitive ones to reach for, and they are
almost always the wrong ones.

**Rule for anything that names a body by string** — reward terms, event terms,
sensor configs, termination terms: confirm the named body actually has the mass
or the collider the term depends on. `scripts/audit_plant_mass.py` prints the
mass of all 22; a body at 0.000000 kg is a marker frame and almost certainly not
what you meant.

---

## 5. Fix options for the mass

None applied. Note that `base` is the free-floating articulation root, so the
1e-09 placeholder used for the leaf marker frames is **not** automatically safe
here — a near-massless articulation root is numerically poor in PhysX, and this
is the body the solver hangs the whole floating chain from.

1. **Merge `base` into `trunk_assembly` in the MJCF** — move the `<freejoint>`
   onto `trunk_assembly` and delete the empty wrapper. Physically cleanest: the
   two are already rigidly welded, so this changes nothing about the intended
   robot. Renames the root body, so audit anything that refers to it by name.
   Isaac Lab's `MjcfConverterCfg` exposes no merge-fixed-joints option
   (`link_density`, `import_inertia_tensor`, `fix_base`, `import_sites`,
   `self_collision` only), so this must be done in the MJCF.
2. **Author a small `<inertial>` on `base`** so the converter emits real values
   and PhysX stops defaulting. Smallest diff, matches the existing in-file
   convention. Choose the mass for solver conditioning, and verify against the
   PhysX warning in §1.3 disappearing.
3. **Override at load time** in `robot_cfg.py`, leaving the USD alone. Fastest
   to test, but the asset stays misleading for every other consumer.

Whichever is chosen, **retrain before quoting any hardware number** — the plant
changes, so existing checkpoints no longer match it.

### 5.1 Add the regression guard with the fix, not before

The natural guard is a no-GPU assertion that every MJCF body declares
`<inertial>`, alongside the existing staleness checks in
`tests/test_usd_conversion.py`. It is deliberately **not** added yet: it would
fail on `base` from the moment it is committed, and a permanently red test
teaches everyone to ignore the suite. Add it in the same change that fixes the
asset. Until then `scripts/audit_plant_mass.py` is the check, and it is a real
gate — it exits nonzero on FAIL (verified in both directions).

---

## 6. What this does *not* affect

- **The MJCF, MuJoCo, and MJX paths are correct** and need no change. Upstream's
  model is right for upstream's simulator.
- **The CAD record is correct.** 2.657067 kg remains the accurate robot mass and
  is the right number for hardware, BOM, and battery/torque budgeting.
- **Per-body masses other than `base` are exact.** All 21 authored bodies match
  the MJCF to float32 resolution (`scripts/audit_plant_mass.py` reports zero
  mismatches).
- **The USD is not stale.** The asset-hash and mesh-manifest guards pass.
- **The BAM actuator parameters are faithfully applied.** `kp`, `damping`,
  `armature` and `friction` in `robot_cfg.py` match
  `params_sts3250_id008.json` exactly; only the torque ceiling (§4.5) and the
  antenna group (§4.4) are in question.
- **Published gate numbers stand.** The reset penetration in §4.7 is
  training-only; every `_PLAY` config spawns from the clean pose.
- **The 59-dim observation width and term order are correct in the code.** The
  defects in §4.1 and §4.2 are in the documentation of that contract, not in
  the trained policy.

---

## 7. Scope of this audit

Method: five parallel read-only auditors over the asset chain, actuator model,
domain randomization, observation/action contract, and docs-vs-code, each
finding adversarially re-checked by an independent verifier prompted to refute
it. Findings that survived were then reproduced by hand before being written
here — every number above was re-measured directly, not copied from an agent.

**Checked:** MJCF/URDF/USD/CAD mass and inertia consistency; unauthored physical
properties; USD staleness guards; the converter's settings; BAM actuator
parameters versus `robot_cfg.py` and the MJCF; the effective event/randomization
set along the full v5d inheritance chain; observation term order, width and
binding against the exported `env.yaml`; reset spawn geometry against the
collision meshes; and the numeric claims in `AGENTS.md`.

**Not checked:** reward-function correctness beyond the two terms named above;
the imitation reference-motion library; training hyperparameters; anything in
`jetson_runtime/` (it does not exist yet); and real-hardware measurements of any
kind. Nothing here is a claim about the physical robot's behaviour — only about
what the simulator does and whether the repo describes it accurately.

**Nothing here is unverified.** 25 candidate findings were raised; 9 went
through adversarial verification and 1 of those was refuted (as a duplicate).
Everything written up above was then re-measured by hand. Findings that were
neither verified nor hand-checked were **left out entirely** rather than
included with a caveat.

**Candidates raised but not pursued**, listed so they are not lost. Each is
plausible and unconfirmed — treat as leads, not findings:

- Isaac disables self-collision entirely (`enabled_self_collisions=False`);
  a claimed 71.5% of in-limit poses self-collide in MuJoCo but not in Isaac.
- Trunk CoM randomization reportedly delivers ~3.4× less whole-body CoM shift
  than its ±10/±5 mm range suggests, because it moves one body of 22.
- Observation corruption is i.i.d. zero-mean only — no IMU bias, drift, or
  mounting misalignment — and no gate is measured with corruption on.
- The three in-repo plants (`robot.xml`, `robot_motors.xml`, `robot_cfg.py`)
  may disagree on joint damping and actuator gain, with no test covering it.
- The documented deployment velocity clamp may permit commands outside the
  trained hull.
- The eval protocol is documented as 5 conditions / 3,200 episodes; v5-era
  results appear to have used 6 / 3,840.
- `AGENTS.md` never mentions v5 in its plant/DR description, which predates the
  pinned policy by two curriculum changes.
