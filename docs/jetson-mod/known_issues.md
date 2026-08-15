# Known Issues — consolidated, verified register

**Compiled 2026-08-09.** This is the **only** issue list in this repo. It absorbed
and replaced `sim-plant-fidelity.md` (deleted 2026-08-09) and supersedes the
issue fragments scattered through `v5_retrain_plan.md`, `task_plan.md` and
`AGENTS.md`. Those remain authoritative for their own subject — the retrain
plan, the phase plan, the rules — but **defects live here and nowhere else**.

If you find yourself writing an issue into another document, put it here instead
and link to it. The failure this file exists to prevent is the one that already
happened twice: the same defect described in two places, drifting apart, with no
way to tell which copy is current.

## The rule this document is built on

**Every issue below was mechanically re-verified on 2026-08-09 before being
written down.** Nothing was carried over from a summary, a prior agent report,
or a previous draft. Claims that failed re-verification are in
[§ Not issues](#not-issues-verified-refutations) rather than deleted, so they
are not raised again.

Two verification passes exist, and both are runnable:

| Pass | Covers | Command |
|---|---|---|
| Static / data | 34 checks over source, MJCF, USD, ONNX, JSON, pickles, and the archived eval results | `python3 scripts/verify_known_issues.py` |
| Runtime (GPU) | plant mass and per-body inertia only — the CFG-1..CFG-5 runtime figures came from ad-hoc probes, not from this script (see the Appendix) | `cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/audit_plant_mass.py --headless` |

`verify_known_issues.py` needs no GPU and no Isaac Sim — plain `python3` with
`numpy` and `mujoco`, the same dependencies `tests/` already uses. Current state:

```
CONFIRMED 18 / 19
INCONCLUSIVE -> could not be evaluated here: ['DEPLOY-1']
   the ONNX checks need an interpreter with `onnx` installed, e.g.
   ~/IsaacLab/_isaac_sim/python.sh scripts/verify_known_issues.py DEPLOY-1
```

The denominator falls as issues are fixed: a fixed issue keeps its entry (with a
`FIXED` block) and **loses its check**, so 34 checks on 2026-08-09 became 29 on
2026-08-11 (PLANT-1, PLANT-2, DEPLOY-5, ART-3, DOC-4 retired) and 28 on 2026-08-12 (EVAL-1 retired by Task R0) and **21 on 2026-08-13**, when
Tasks M0/M0b fixed and retired PLANT-3, PLANT-4, PLANT-5, PLANT-7, CFG-1,
CFG-2 and DEPLOY-3.

Exit codes: **0** all confirmed · **1** a claim was refuted (the issue was fixed,
or the check has rotted — either way, act on it) · **2** a check could not be
evaluated. Run a single check by id: `python3 scripts/verify_known_issues.py PLANT-3`.

Where an issue's evidence required a live simulator, it is marked **[GPU]** and
the measurement is quoted verbatim.

**When you fix an issue,** the corresponding check flips to REFUTED and the
script exits 1. That is the signal to delete both the check and the entry — the
register is meant to shrink.

## Legend

**Severity** is about what the issue *blocks*, not how ugly it is:

| | Meaning |
|---|---|
| **CRITICAL** | Invalidates hardware transfer for policies already trained |
| **HIGH** | Blocks a phase, or silently corrupts the engineering record |
| **MEDIUM** | Real defect; forces a retrain or a rewrite when fixed |
| **LOW** | Correct-but-misleading, dead code, or cosmetic |

**Scope** matters more than severity here, because several defects live only in
abandoned experiment arms. The shipped policy is **v5d**
(`exported_policies/v5d_contact_wrench_ppo/`), whose config chain is:

```
OpenDuckContactWrenchEnvCfg -> OpenDuckContactMinimalEnvCfg -> OpenDuckRobustEnvCfg -> OpenDuckRoughEnvCfg
```

It does **not** pass through `OpenDuckContactEnvCfg`, the v5a/v5b arm — verified,
so anything scoped to `DuckContactRewards` does not touch the shipped policy.

---

## Summary

| ID | Issue | Sev | Scope |
|---|---|---|---|
| [PLANT-1](#plant-1) | Phantom 1.000 kg on the articulation root `base` | CRITICAL | **FIXED 2026-08-11**; every policy v1–v5d was trained with it |
| [PLANT-2](#plant-2) | The one DR term that could cover PLANT-1 misses on both axes | CRITICAL | **RESOLVED 2026-08-11** by the PLANT-1 fix |
| [PLANT-3](#plant-3) | 61.7% of training resets start inside the ground plane | MEDIUM | **FIXED 2026-08-13** (M0/M0b) |
| [PLANT-4](#plant-4) | Antennas simulated with STS3250 parameters (~48× torque, ~10⁴× armature) | MEDIUM | **FIXED 2026-08-13** (M0/M0b) |
| [PLANT-5](#plant-5) | Torque ceiling is 1.78× datasheet stall, pinned to 12.1 V, never randomized | MEDIUM | **FIXED 2026-08-13** (M0/M0b) |
| [PLANT-6](#plant-6) | Joint dry friction is inactive during motion; BAM's viscous term is dropped | MEDIUM | all |
| [PLANT-7](#plant-7) | No latency model of any kind | MEDIUM | **FIXED 2026-08-13** (M0/M0b) |
| [PLANT-8](#plant-8) | The yaw command is a heading servo, never an open-loop rate | MEDIUM | teleop / VLM consumers |
| [PLANT-9](#plant-9) | The push curriculum is sized against a "0.17 m CoM height"; 0.17 m is the root spawn height and the measured CoM is 0.203 m | LOW | v5a/v5b only |
| [PLANT-10](#plant-10) | CAD-mod mass deltas were booked at 0.9× solid PLA; `trunk_assembly` was **74.53 g light** | HIGH | **FIXED 2026-08-12** (Task M2) — mass is now whole-part measured, no density assumed |
| [CFG-1](#cfg-1) | `torque_z_range` is silently discarded — a dead configured parameter | MEDIUM | **FIXED 2026-08-13** (M0/M0b) |
| [CFG-2](#cfg-2) | Obstacles are placed twice per episode, with two independent draws | MEDIUM | **FIXED 2026-08-13** (M0/M0b) |
| [CFG-3](#cfg-3) | In v5c/v5d the disturbance gate is maintained every step and read by nothing | LOW | v5c, v5d |
| [CFG-4](#cfg-4) | The `head` half of `ground_contact_penalty` can never fire | MEDIUM | v5a/v5b **only** |
| [CFG-5](#cfg-5) | ContactSensor history spans 15 ms, not 60 ms | LOW | all consumers of the sensor |
| [EVAL-1](#eval-1) | `--report-only` injects every JSON in the directory — already fired | HIGH | **FIXED 2026-08-12** by the `--include` allowlist (Task R0) |
| [EVAL-2](#eval-2) | Documented eval protocol is not the protocol that ran | MEDIUM | all v5 results |
| [SHELL-1](#shell-1) | ~~selects the run directory by mtime~~ **FIXED 2026-08-13** | MEDIUM | future runs |
| [SHELL-2](#shell-2) | `v5_chain.sh` deletes the pidfile its own duplicate-run guard depends on | MEDIUM | future runs |
| [SHELL-3](#shell-3) | Launcher dispatches `--algorithm` to a deleted script and records success | LOW | dead code |
| [SHELL-4](#shell-4) | Queue's GPU-busy predicate is blind to eval and play jobs | LOW | dormant script |
| [SHELL-5](#shell-5) | Training watchdog fails open on a parse error | MEDIUM | future runs |
| [ART-1](#art-1) | `policy.onnx` is gitignored while its weight sidecar is committed | HIGH | reproducibility |
| [ART-2](#art-2) | The shipped policy's provenance parent (`v6_robust`) is not in git | MEDIUM | **REFRAMED 2026-08-15** |
| [ART-3](#art-3) | `usd/config.yaml` records a deleted worktree as the asset source | LOW | **FIXED 2026-08-11** by the USD regen |
| [ART-4](#art-4) | `.gitignore` ignores `*.txt` repo-wide — any future prompt library or fixture is silently untracked | MEDIUM | Phase 4/5 |
| [DEPLOY-1](#deploy-1) | The ONNX omits `action_scale` and `q_default` entirely | HIGH | **MITIGATED 2026-08-13** (R3: sidecar ships with the policy; graph unchanged) |
| [DEPLOY-2](#deploy-2) | The normalizer epsilon is in the graph but not in the checkpoint | MEDIUM | Phase 4 |
| [DEPLOY-3](#deploy-3) | 4 of the 59 observation dims cannot be measured on hardware | HIGH | **FIXED 2026-08-13** (M0/M0b) |
| [DEPLOY-4](#deploy-4) | Exported ONNX has a hard-fixed batch dimension of 1 | LOW | offline tooling |
| [DEPLOY-5](#deploy-5) | Documented velocity clamp exceeds the trained command hull | MEDIUM | **FIXED 2026-08-11** |
| [TEST-1](#test-1) | The test suite is source-text grepping; 5 of 7 real regressions pass | HIGH | all future changes |
| [TEST-2](#test-2) | A bare `pytest` from the repo root fails at collection | LOW | developer experience |
| [TEST-3](#test-3) | `requires_isaac_sim` is documented but applied to zero tests | LOW | false assurance |
| [REF-1](#ref-1) | Reference library has no `vy=0` and no `wz=0` cell; contact dims are constant | MEDIUM | v3–v5d |
| [REF-2](#ref-2) | 33.6% of reachable knee references are clamped; velocity targets are not | MEDIUM | v3–v5d |
| [DOC-2](#doc-2) | The experiment journal has zero v5 entries | MEDIUM | engineering record; **results-doc half done 2026-08-11** |
| [DOC-3](#doc-3) | Task 2.7 docs still assert a 2.657 kg plant | HIGH | **RESOLVED 2026-08-11** — the plant IS 2.657 kg now |
| [DOC-4](#doc-4) | `task_plan.md` is stale on four axes | MEDIUM | **FIXED 2026-08-11** (all four) |
| [DOC-5](#doc-5) | `AGENTS.md` says the USD came from URDF; it came from MJCF | LOW | **FIXED 2026-08-09** |
| [DOC-6](#doc-6) | `AGENTS.md` actuator snippet sets a field the code does not use | LOW | **FIXED 2026-08-09** |
| [SERVO-1](#servo-1) | `neck_pitch` commands 207 % of continuous — caused by leg-sized PD gains on the head group, not by head mass | **HIGH** | **CORRECTED 2026-08-15**; fix is config, not mechanical |
| [SERVO-2](#servo-2) | The shipped policy saturates the leg actuators: p99 = peak stall on four joints | **HIGH** | blocks sustained operation |

---

# PLANT — the simulated robot differs from the design

<a id="plant-1"></a>
## PLANT-1 · Phantom 1.000 kg on the articulation root — CRITICAL

**Scope:** every policy ever trained in Isaac against this USD — v1, v2, v3,
v4_inertials, v4_robust, v5a–v5d.

### Every file on disk is correct

This is not a data-entry error, which is why it survived a CAD revision, a USD
regeneration, three policy generations and an adversarial review:

| Source | Total mass | `base` body |
|---|---:|---:|
| CAD (`mass_inertia_calculations.md`) | 2.657067 kg | — |
| MJCF `robot_motors.xml`, summing `<inertial>` | 2.657067 kg | no `<inertial>` tag |
| MuJoCo, building a model from that MJCF | 2.657067 kg | 0.000000 kg |
| USD, summing its own authored `MassAPI` values | 2.657067 kg | unauthored |
| **Isaac / PhysX at runtime** | **3.657067 kg** | **1.000000 kg** |

The USD is not stale either — `pytest tests/test_usd_conversion.py` passes all 5
checks, including the MJCF asset-hash and the per-STL mesh manifest.

### What actually happens

The MJCF root body `base` is a pure kinematic frame. It carries the
`<freejoint>` and declares no `<inertial>` and no `<geom>`; the real trunk mass
lives on its child `trunk_assembly`, welded to it with no joint of its own.

MuJoCo handles a massless frame natively. PhysX cannot, and fills the unauthored
value with its own defaults. Isaac Lab's converter documents this path —
`MjcfConverterCfg.import_inertia_tensor`: *"If the `inertial` tag is missing,
then it is imported as an identity."* The other knob that could have covered it
does not apply: `link_density = 0.0` means auto-compute-from-geometry, and `base`
has no geometry.

**Evidence [GPU]** — `scripts/audit_plant_mass.py`, exit code **1**:

```
base                                    none      1.000000   +1.000000   NO <inertial> IN MJCF -> PhysX default
TOTAL                               2.657067      3.657067   +1.000000

unauthored body 'base': PhysX assigned mass=1.000000 kg,
  diagonal inertia=(4.000e-03, 4.000e-03, 4.000e-03) kg*m^2
VERDICT: FAIL — simulated total is +1.000000 kg (+37.6%) versus the MJCF
```

All 21 other bodies match the MJCF to float32 precision, so the entire error is
on this one body. `base` is **27.3%** of the simulated robot and does not
physically exist.

The inertia matters as much as the mass and is easier to overlook: the phantom
isotropic `4.0e-3 kg·m²` is **1.81×** `trunk_assembly`'s own Ixx (2.215e-03), so
the root link's rotational inertia is roughly doubled.

### Confirmed physically, not just from metadata

Metadata could be misread; the dynamics cannot. Over a 30 s steady walk at
`vx = 0.2 m/s` (v5d, Play task, 1 env), mean total ground reaction force is
**37.50 N**. In steady walking that must equal `m·g`:

| Candidate mass | Predicted weight | Error vs measured 37.50 N |
|---|---:|---:|
| 2.657067 kg (CAD) | 26.06 N | **+43.9%** |
| 3.657067 kg (PhysX) | 35.87 N | **+4.5%** |

The residual +4.5% is the 3-vector force norm slightly over-reading the vertical
component. The physics follows PhysX.

### The simulator has been saying so on every run

PhysX names the exact prim every time the robot loads — every training run,
every evaluation, every play:

```
[Warning] [omni.physx.plugin] The rigid body at /World/envs/env_0/Robot/base/base
has a possibly invalid inertia tensor of {1.0, 1.0, 1.0} and a negative mass,
small sphere approximated inertia was used. Either specify correct values in the
mass properties, or add collider(s) to any shape(s) that you wish to
automatically compute mass properties for.
```

"Negative mass" is USD's `-1.0` unauthored sentinel; `{1.0, 1.0, 1.0}` is the
identity inertia the converter warned it would write. It sits in several hundred
lines of Kit startup noise next to genuinely benign MaterialX and X-server
messages, so it was read as more of the same. It appears in **53** of the logs
under `.training_runs/`.

### The fix pattern is already in the file

The MJCF's other massless frames do not have this problem — they carry an
explicit placeholder rather than nothing at all:

| Body | mass |
|---|---:|
| `trunk` | 1e-09 |
| `left_foot` | 1e-09 |
| `right_foot` | 1e-09 |
| `head` | 1e-09 |

`base` is the only one of 22 bodies with no `<inertial>` element at all, and
therefore the only one PhysX has to invent numbers for.

### It comes from upstream, and it is not upstream's bug

The massless `base` wrapper is in the first commit of `robot_motors.xml` here
(`64c9940`), inherited from `apirrone/Open_Duck_Mini`. It is simply how
`onshape-to-robot` emits a free-floating root. Upstream trains in MuJoCo/MJX,
where mass 0 on that frame is correct and harmless. **The defect is created at
the MuJoCo→PhysX boundary**, which only this fork crosses — as would any project
converting an `onshape-to-robot` MJCF with a bare root frame into Isaac Sim.

### Impact

**In simulation, nothing is broken.** Training and evaluation share one USD, so
policies and plant are self-consistent and all sim-only results remain internally
valid — they are results for a 3.657 kg robot. The EN.535.782 course project
explicitly accepted this on 2026-08-02 and did not retrain; it quotes 3.657 kg as
the plant mass and names the gap as a limitation.

**On hardware, this is a first-order transfer risk.** A policy trained to lift
37.6% more mass, with root rotational inertia near doubled, will command
systematically excessive torque. It compounds with [PLANT-5](#plant-5): the
simulator also allows each servo 1.78× its datasheet stall, so the policy has two
independent reasons to ask a real servo for more than it can give.

**And it leaks into the disturbance curriculum** — a consequence not recorded
before this pass. The v5 sustained wrench is a fraction of `default_mass.sum()`
(`contact_events.py:112-113`), so v5d's configured "5–20% of body weight" is
**6.9–27.5%** of the real robot, overshooting the Hartmann et al. reference the
code cites it against.

### Fix options

None applied. Note that `base` is the free-floating articulation root, so the
`1e-09` placeholder used for the leaf marker frames is **not** automatically safe
here — a near-massless articulation root is numerically poor in PhysX, and this
is the body the solver hangs the whole floating chain from.

1. **Merge `base` into `trunk_assembly` in the MJCF** — move the `<freejoint>`
   onto `trunk_assembly` and delete the empty wrapper. Physically cleanest: the
   two are already rigidly welded. Renames the root body, so audit everything
   that refers to it by name. `MjcfConverterCfg` exposes no merge-fixed-joints
   option, so this must be done in the MJCF.
2. **Author a small `<inertial>` on `base`** so the converter emits real values.
   Smallest diff, matches the existing in-file convention. Choose the mass for
   solver conditioning and verify the PhysX warning disappears.
3. **Override at load time** in `robot_cfg.py`, leaving the USD alone. Fastest to
   test, but the asset stays misleading for every other consumer.

Whichever is chosen, **retrain before quoting any hardware number** — the plant
changes, so existing checkpoints no longer match it.

### Add the regression guard with the fix, not before

The natural guard is a no-GPU assertion that every MJCF body declares
`<inertial>`, alongside the staleness checks in `tests/test_usd_conversion.py`.
It is deliberately **not** added yet: it would fail from the moment it is
committed, and a permanently red test teaches everyone to ignore the suite. Add
it in the same change that fixes the asset. Until then `audit_plant_mass.py` is
the check, and it is a real gate — it exits nonzero on FAIL, verified in both
directions.

### FIXED 2026-08-11 — option 1 (merge `base` into `trunk_assembly`)

`robot_motors.xml` now carries the `<freejoint/>` and `pos="0 0 0.17"` on
`trunk_assembly` directly; the `base` wrapper is gone. The merge is a pure
structural rename because `trunk_assembly` had **no `pos` and no `quat`** — an
identity transform. Verified against the pre-fix model at a randomized state:

```
                     OLD (base wrapper)   NEW (merged)
nbody                                24             23
nq / nv                         23 / 22        23 / 22
total mass (kg)             2.657067284    2.657067284
root body                          base   trunk_assembly

max |M_old - M_new|       = 0.000e+00   (mass matrix, 22x22)
max |qacc_old - qacc_new| = 0.000e+00
max |CoM_old - CoM_new|   = 5.551e-17 m
```

USD regenerated from the merged MJCF. `audit_plant_mass.py` now **exits 0**:

```
TOTAL     MJCF 2.657067    PhysX 2.657067    -0.000000
VERDICT: PASS — simulated total is -0.000000 kg (-0.0%)
```

The PhysX `possibly invalid inertia tensor` warning no longer appears (grep
count 0 across the run), and the articulation has **21 rigid bodies instead of
22**. Runtime smoke test on `ContactWrench-v0`, 16 envs × 60 steps: root body
`trunk_assembly`, contact sensor resolves all 21 bodies, feet register
152.5 N / 87.7 N, and **obs/action stay 59/16** so exported checkpoints remain
loadable.

Downstream sites updated in the same change:
`robot_cfg.py` `articulation_root_prim_path` `/base/base` →
`/trunk_assembly/trunk_assembly`; `env_cfg.py` contact-sensor `prim_path`
`Robot/base/.*` → `Robot/trunk_assembly/.*`; `scripts/debug_contact_sensor.py`;
and the `tests/test_isaac_lab_env.py` assertion.

The wrench curriculum leak closed with it — `ContactRegimeEvent` now reports
`measured robot weight 26.07 N (2.657 kg)` where it previously reported
35.88 N (3.657 kg).

> **Every policy through v5d was trained on the 3.657 kg plant and none has been
> retrained on the corrected one.** Existing checkpoints load, but their gate
> numbers describe a robot that no longer exists in the simulator.

> ### Re-gated 2026-08-12 — [`m2657_regate.md`](m2657_regate.md)
>
> The re-run this entry called for is done (task_plan_v2.md Tasks R1/R1b/R1c).
> Ten evaluations, 3,840 episodes each, both policies on the corrected plant
> under one frozen protocol. **The outcome was not symmetric:**
>
> | open-field falls | 3.657 kg | 2.657 kg |
> |---|---|---|
> | `v5d_contact_wrench` | 0.000 % | **0.000 %** |
> | `v4_robust` | 0.000 % | **24.167 %** |
>
> **v5d passes every gate on the corrected plant** (`scripts/regate_report.py`
> exits 0; G-R3 video audit passed frame-by-frame). Its reference tracking moved
> 4.669 -> 4.676 deg and it beats the same-plant control on all four contact
> gates. **`v4_robust` no longer walks reliably** and is retired as a control on
> this plant.
>
> The phantom kilogram was therefore not merely a bookkeeping error: it was
> load-bearing for `v4_robust`, and removing it destroyed that policy. The
> contact-rich v5 curriculum — fall-only terminations, obstacles, sustained
> wrench — is what made v5d survive a 27 % plant change, since neither policy
> had mass DR beyond `add_base_mass (-0.10, +0.15) kg`.
>
> **This does not make v5d a hardware candidate.** Phase M changes the plant
> again and Task M0b takes obs/action to 53/14, after which the v5d checkpoint
> will not load at all. The retrain is Task R2/R2b.

<a id="plant-2"></a>
## PLANT-2 · The DR term that could cover PLANT-1 misses on both axes — CRITICAL

**Scope:** `OpenDuckRobustEnvCfg` and everything below it — v4_robust, v5a, v5b, v5c, v5d.

**Evidence [GPU]**, resolved at runtime:

```
add_base_mass  body_names=['trunk_assembly'] -> resolved ids=[1]
               randomized body mass = 1.089544 kg
               body 'base' mass     = 1.000000 kg   <-- NOT randomized
               range (-0.10, +0.15) kg vs a 1.000 kg constant offset -> 6.7x too small
```

1. **Wrong body.** It perturbs `trunk_assembly`, which has no error.
2. **Wrong magnitude.** ±(−0.10, +0.15) kg against a 1.000 kg offset, and the
   offset is *constant*, so it does not even appear as tail variance.

`base_com` binds `trunk_assembly` too. Nothing in any config targets `base`.

**The general lesson** (see [the naming-families pattern](#naming-families)): the term is named
for the *concept* "base mass" but bound to the *body* `trunk_assembly`. Any term
that names a body by string should be checked against the body that actually
carries the uncertainty.

### RESOLVED 2026-08-11 by the PLANT-1 fix

Both halves dissolve once `base` is merged away:

1. **Wrong body — no longer wrong.** `trunk_assembly` *is* the articulation root
   now, so `add_base_mass` and `base_com` perturb exactly the body their names
   claim. The concept and the binding agree.
2. **Wrong magnitude — no longer wrong.** There is no 1.000 kg constant offset
   left to cover, so a ±(−0.10, +0.15) kg range is a reasonable mass DR band on a
   1.09 kg link rather than a 6.7×-too-small correction for a systematic error.

Nothing in the config changed; the defect was in the plant the config pointed at.
The naming-families lesson still stands and is why the entry is kept rather than
deleted.

<a id="plant-3"></a>
## PLANT-3 · 61.7% of training resets start inside the ground plane — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> 61.7 % -> **0.0 %**. Task M0 added a +20 mm spawn `z` to `reset_base.pose_range`. Re-measured with `verify_known_issues.py PLANT-3` before retiring the check: deepest reset is now **+4.40 mm** above ground, mean +17.73 mm. The check was also taught to READ the spawn offset — before that it simulated the pose only and could never have observed the fix.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


**Scope: training only.** All `_PLAY` configs pin `position_range=(1.0, 1.0)`,
so every published gate number is unaffected.

`reset_robot_joints` scales each joint's default angle by U(0.9, 1.1) and clamps
to soft limits, while the root stays pinned at z = 0.17 m. The nominal pose
leaves only +3.2 mm of clearance, which that scaling does not fit inside.

**Evidence** — transform every collision-mesh vertex by its forward-kinematic
pose (MuJoCo, CPU, 4000 samples). The figures in brackets are the independent
measurement made during the 2026-08-02 plant audit:

```
NOMINAL pose lowest collision vertex = +3.17 mm    (doc: +3.2 mm -> methods agree)
collidable mesh geoms considered     = 130
mean lowest collision vertex         = -2.27 mm    (doc: -1.9 mm)
deepest                              = -15.60 mm   (doc: -16.4 mm)
FRACTION starting below ground       = 61.7%       (doc: 61.4%)
```

Every such episode opens with a PhysX depenetration impulse of up to
`max_depenetration_velocity = 1.0 m/s`, which has no hardware counterpart.

> **Note:** an earlier draft of this register wrongly listed §4.7 as refuted.
> That refutation measured link-frame *origins* rather than collision *vertices*
> — a foot's origin sits ~3.2 mm above its sole, so the metric could not see the
> penetration. §4.7 is correct.

<a id="plant-4"></a>
## PLANT-4 · Antennas are simulated as STS3250 servos — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> Task **M0b** removed `.*_antenna` from the `head` actuator group and gave the antennas their own: `effort_limit_sim` 8.716 -> **0.18 N.m** (an SG90 stalls there; 8.716 was ~48x) and `armature` 0.040 -> **4.0e-06** against the link's 3.31e-06 principal inertia (0.040 was 12,082x — a flywheel). They are also out of the action and observation spaces entirely.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


`robot_cfg.py` places `.*_antenna` in the `head` actuator group with the full
STS3250 parameter set. The real hardware uses SG90 micro servos.

**Evidence:**
```
robot_cfg 'head' actuator group includes '.*_antenna': True
  -> effort_limit_sim=8.716 N.m, armature=0.040 kg.m^2 applied to the antennas
AGENTS.md documents the real antennas as SG90 micro servos: True
  (SG90 stall ~0.18 N.m -> sim ceiling is ~48x)
antenna link inertials: left 0.004216 kg diag [3.80836e-06, 3.59678e-06, 2.45273e-07]
                        right 0.004216 kg diag [3.86998e-06, 3.65850e-06, 2.45146e-07]
  -> armature 0.040 vs largest antenna principal inertia 3.870e-06 = 10336x
```

The armature is the sharper problem: those two joints are not simulating an
antenna on a micro servo, they are simulating a flywheel. Low consequence for
locomotion, but both joints are in the 16-dim action vector destined for real
servos. See also [DEPLOY-3](#deploy-3).

<a id="plant-5"></a>
## PLANT-5 · Torque ceiling is 1.78× datasheet, pinned to 12.1 V — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> Task M0 replaced the bare `8.716` — BAM id008's **electrical** stall at 12.1 V, 1.78x the datasheet — with `STS3250_EFFORT_LIMIT_NM = 4.903`, the 50 kg·cm datasheet stall, and recorded `STS3250_CONTINUOUS_NM = 1.569` beside it. `scripts/measure_joint_torque.py` had measured v5d commanding **6.723 N·m**, 137 % of datasheet stall, against the old ceiling.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


**Evidence:**
```
BAM: kt=1.00058746, R=1.38904625, vin=12.1 V  ->  kt*V/R = 8.7161 N.m
robot_cfg effort_limit_sim = 8.716 N.m  (matches BAM exactly)
AGENTS.md documents 50 kg.cm stall = 4.903 N.m
  -> simulated ceiling is 1.78x the datasheet stall
pack nominal 11.1 V -> 7.996 N.m (8.3% below the simulated ceiling)
```

Both numbers are defensible in isolation — BAM's is electrical stall from its
identified model, the datasheet's is rated stall — but the simulator enforces
BAM's, so the policy may lean on torque the hardware cannot deliver. **And no
domain randomization touches any actuator parameter** — not stiffness, not
damping, not the effort limit. This compounds with [PLANT-1](#plant-1): two
independent reasons to over-command a real servo.

<a id="plant-6"></a>
## PLANT-6 · Joint dry friction is inactive during motion — MEDIUM

Isaac splits MuJoCo's single `frictionloss` into three parameters; only the
first is set.

**Evidence** (actuator blocks of the shipped `env.yaml`, both groups):
```
legs: friction=0.2, dynamic_friction=null, viscous_friction=null
head: friction=0.2, dynamic_friction=null, viscous_friction=null
any USD layer authors a joint friction attribute: False
```

`None` does not mean "use the sibling"; per `actuator_base.py:176-192` it means
"read from the USD", and the USD authors no joint friction at all, so the
read-back is `0.0`. Isaac's semantics (`articulation.py:888-889`): static
friction caps effort only **at rest**; dynamic friction is what acts during
motion. So the BAM-identified dry friction is present when the robot stands
still and absent for the entire gait — the unhelpful direction, since it
flatters the policy in sim.

Related: BAM identifies `friction_viscous = 0.6256393187557033` and nothing
applies it.

<a id="plant-7"></a>
## PLANT-7 · No latency model of any kind — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> Task M0 added `isaac_lab_env/open_duck_mini_v2/latency.py`. The policy's `joint_pos`/`joint_vel` observations are delayed **0–2 control steps (0–40 ms)**, redrawn per environment at reset; the privileged critic keeps the undelayed state. **The VALUE is provisional** until Task S.6 measures the real sensor->inference->actuation loop — the mechanism is not.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


**Evidence:** no `latency`, `delay`, or observation-staleness term appears in
`env_cfg.py` or the shipped `env.yaml`. (The single textual "delay" hit is the
word *delays* inside a comment at `env_cfg.py:642`.)

Deployment runs TensorRT inference plus a serial servo bus; the simulator models
neither. This is an *absence*, so nothing in the config will ever look wrong —
which is exactly why it belongs on a list. It is also the cheapest of the plant
gaps to close, since Isaac Lab supports action delay without touching the asset.

<a id="plant-8"></a>
## PLANT-8 · The yaw command is a heading servo, never an open-loop rate — MEDIUM

**Scope:** teleop / VLM / haptic consumers of the velocity command. Not a plant
defect and nothing is misconfigured — an undocumented property of the training
distribution.

`rel_heading_envs: 1.0` means **every** environment is a heading environment, so
`UniformVelocityCommand._update_command` overwrites the yaw channel every step
with `wz = clip(0.5 · heading_error, ±0.5)`. The sampled `ang_vel_z` value never
survives.

**What this does NOT mean.** It does not mean the endpoint was never trained.
The servo **saturates whenever |heading_error| > 1.0 rad**, and in that regime
its output is a hard constant, not a decaying signal. Heading targets resample
uniformly on (−π, π), so a fresh target usually starts saturated.

**Evidence [GPU]** — the shipped checkpoint rolled out in its own training env,
1500 steps × 64 envs on `ContactWrench-v0`:

```
|wz| commanded: mean 0.1849, p50 0.1079, max 0.5000
  fraction of samples |wz| >= 0.49 : 0.1751
  longest CONSECUTIVE |wz| >= 0.49 : mean 3.35 s, max 12.72 s
heading error decays 2.03 rad (first 100 steps) -> 0.29 rad (last 100)
corr(commanded wz, measured yaw rate) = 0.4888
```

At the measured ~0.27 rad/s turn rate, closing a 2 rad error takes ~3.7 s of
pinned ±0.5 — which is what the 3.35 s mean run length shows. **A sustained
constant yaw rate at the range endpoint is IN distribution.**

**The finding that survives.** The servo's output is always
≤ 0.5·|heading_error|, so a held 0.5 *implies* a persistent ≥1 rad misalignment.
An open-loop consumer can hold 0.5 while the robot is **already aligned** — turn
indefinitely rather than turn *toward* something. That correlation structure
between the command and the robot's own heading error is what the policy never
saw.

**Fix:** set `rel_heading_envs` below 1.0 in the next campaign so some envs train
on directly sampled yaw-rate commands. The justification is
persistence-after-alignment, not "the endpoint was never trained".

> **Correction history.** The predecessor of this entry
> (`sim-plant-fidelity.md` §4.10, since deleted) claimed dim 8 "always carried a
> decaying heading-error signal" and concluded that a sustained yaw rate was out
> of distribution. That was wrong; the disproof was the `torch.clip` in the
> section's own quoted snippet. Corrected 2026-08-09 by the measurement above.

<a id="plant-9"></a>
## PLANT-9 · The push curriculum is sized against the spawn height, not the CoM — LOW

`isaac_lab_env/open_duck_mini_v2/env_cfg.py:73` derives the recoverable
push velocity from a capture-point argument that names **"a 0.17 m CoM height"**.
0.17 m is not the CoM height — it is the **root spawn height**
(`init_state.pos=(0, 0, 0.17)` in `robot_cfg.py`).

Measured by forward kinematics at the `robot_cfg.py` standing pose, transforming
every collision-mesh vertex to world:

```
lowest collision vertex   = +3.17 mm
CoM above the sole        = 203.1 mm      <- the h the capture point wants
value used in the comment = 170   mm
```

Capture-point velocity goes as `sqrt(g/h)`, so using 0.17 m where the CoM is
0.203 m overstates the recoverable velocity by `sqrt(203.1/170) - 1` ≈ **9%**.

**Scope.** `PUSH_END` is consumed only by the v5a/v5b arms, which were
abandoned; the shipped v5d wrench does not use it. No published gate number
moves. Recorded because the same constant is the natural thing to reach for when
sizing a future push curriculum, and because it reads as a measured property of
the robot when it is not one.

**Fix:** use the measured 0.203 m, or compute the CoM height at reset rather
than hard-coding either number.

<a id="plant-10"></a>
## PLANT-10 · CAD-mod mass deltas are booked at a density the parts are not printed at — HIGH — ✅ **FIXED 2026-08-12**

`scripts/generate_cad_mods.py:51` converts Part-2 CAD volume deltas to mass with

```python
PLA_EFFECTIVE_DENSITY = 1116.0  # kg/m^3 (0.9 x solid)
```

and writes the result to `scripts/cad_mod_deltas.json` (`density_kg_m3: 1116.0`,
total **−88.48 g**). That number is the `−88.5` term in the `AGENTS.md` trunk
mass chain and in `component_layout_v2.md`.

The script's own comment is honest that this is a consistency choice rather than
a measurement. It can now be measured. Slicing the **adbc082 baseline geometry**
and the **current geometry** with identical settings (PrusaSlicer 2.7.2, PLA
1.24 g/cm³, supports/brim/skirt/raft/wipe-tower all off, so the reported
`filament used [g]` is the part) gives:

```
part                 baseline    current     delta
body_front             62.23 g    63.64 g    +1.41
body_middle_bottom     90.96 g    92.83 g    +1.87
trunk_bottom           34.40 g    13.60 g   -20.80
body_back              70.11 g    81.03 g   +10.92
                                  MEASURED   -6.60 g      BOOKED  -88.48 g
```

The sign and magnitude of the error survive every plausible print profile, so
this is not an artefact of the settings chosen:

| profile | measured delta | booked | error |
|---|---|---|---|
| 2 perim / 15% infill (`print_guide.md`'s profile) | −6.60 g | −88.48 g | **+81.88 g** |
| 3 perim / 15% | −10.73 g | −88.48 g | +77.75 g |
| 2 perim / 25% | −17.49 g | −88.48 g | +70.99 g |
| 3 perim / 40% | −34.70 g | −88.48 g | +53.78 g |

**Mechanism.** One density is applied to both sides of the diff, but the two
sides do not print at the same density. Material *removed* came from bulky
interiors that are mostly infill — `trunk_bottom`'s removal measures **0.28×
solid**. Material *added* is thin vents, bosses and the hump extension, which are
nearly all perimeter and print close to solid. Both errors push the same way.
Reaching −88.48 g would require printing essentially solid, which contradicts
`docs/print_guide.md:5` ("standard PLA with 15% infill").

**Consequence.** `trunk_assembly` is **54–82 g heavier** than the 1.089544 kg the
MJCF declares — on top of, and independent of, [PLANT-1](#plant-1)'s phantom
kilogram. Both errors are in the same direction for the real robot: the hardware
will be heavier than the number every mass document quotes.

**Fix:** stop deriving the delta from an assumed density. Slice the baseline and
current STLs and book the measured difference, or weigh the printed parts. Note
that the delta is robust to the print profile *because both sides use the same
profile*, so this works even before the print process is finally chosen.

**Tooling added 2026-08-11.** `scripts/measure_print_mass.py` replaces the
assumption with a measurement and is parameterised by process:

```
python3 scripts/measure_print_mass.py --cad-delta --sweep      # this measurement
python3 scripts/measure_print_mass.py --process mjf-pa12       # no slicer needed
python3 scripts/measure_print_mass.py --process fdm-pla --perimeters 2 --infill 15
```

It counts **pieces, not files** (`print_guide.md` has ×2/×4 on nine rows, so the
set is 52 pieces / 1571.94 cm³; summing the 37 distinct STLs undercounts by 13%)
and auto-reorients parts whose authored pose gives the slicer a knife-edge first
layer. Whole-set totals it produces: **1158 g** FDM PLA at 2 perim/15%,
**1598 g** MJF PA12 (solid — powder processes have no infill parameter).

~~`PLA_EFFECTIVE_DENSITY` is deliberately **left unchanged** for now.~~
**Superseded — see the FIXED block below.** That reasoning was right at the time:
the correction moves the plant and the right value depended on the print process,
which was open. Task M1 settled the process on 2026-08-12 and Task M2 applied the
correction the same day, ahead of the R2/R2b retrain, exactly as this paragraph
asked.

> **Scope caution.** The 698.5 g upstream trunk booking carries its own unknown
> density, so this fix corrects the *delta*, not the absolute baseline. The
> absolute number is only settled by the bottom-up rebuild or by weighing.

### FIXED 2026-08-12 — Task M2

**The assumed density is gone, not replaced.** `scripts/generate_cad_mods.py` no
longer derives mass from volume at all. Each modified part now contributes two
**measured whole-part** terms — `<name>_baseline` at negative mass and
`<name>_current` at positive mass — taken from `scripts/part_mass_table.json`,
which `scripts/measure_print_mass.py --emit-table` produces by slicing every
part at the process recorded in `scripts/print_process.json`, including the four
CAD-mod parts as they were at baseline commit `adbc082`.

The net is therefore the measured delta by construction, and the two tensors of
a pair cancel over the regions the edit did not touch.

**Measured at the chosen process (`fdm-asa`, 3 perimeters / 20 % infill):**

| part | baseline g | current g | delta g |
|---|---|---|---|
| `trunk_bottom` | 36.99 | 13.54 | **−23.45** |
| `body_middle_bottom` | 93.30 | 91.58 | −1.72 |
| `body_front` | 68.06 | 68.17 | **+0.11** |
| `body_back` | 74.63 | 85.74 | **+11.11** |
| **net** | | | **−13.95** |

against the **−88.48 g** previously booked. **`trunk_assembly` was 74.53 g
light**, inside the 54–82 g band this entry predicted.
`scripts/compute_trunk_inertial.py` now emits **1.164076 kg** for
`trunk_assembly`, against 1.089544 kg before.

**Why no constant could have worked, now visible in the data.** Two of the four
deltas are **positive**. Cutting the inlet slots in `body_front` removes
7.2 cm³ of geometry and *adds* 0.11 g, because the new slot walls contribute
perimeters and solid skins worth more than the infill they displace. A scalar
density times a volume difference cannot change sign. The measured effective
densities across this part set span **0.452 g/cm³** (`trunk_top`) to
**1.102 g/cm³** (`knee_to_ankle_right_sheet`) — a factor of 2.4 against the
single 1.116 that stood in for all of them.

Five thin parts measure *above* the 1.07 g/cm³ filament density. That is real
slicer behaviour, not an error: at 3 perimeters a ~2 mm sheet asks for ~2.7 mm
of wall per side, so the walls pack solid and overlap, and PrusaSlicer reports
mass from extruded filament length. The volume-weighted mean across the set is
**0.7399 g/cm³ (69.2 % of filament)**, which is what confirms infill is actually
being applied.

**Guarded by** `tests/test_cad_mod_deltas.py` (11 tests): the booked per-part
sum must equal the measured difference to 1e-6 kg; `generate_cad_mods` must not
regain the constant (an attribute check, not a grep); `cad_mod_deltas.json` must
record the same process as `print_process.json`; positive-mass tensors must be
positive-definite and satisfy the triangle inequality; and the set must not be
secretly solid.

**Two things this fix deliberately does NOT do.**
1. **The model files are untouched.** `robot_motors.xml`, `robot.xml`,
   `robot.urdf` and `tests/fixtures/expected_values.json` still declare the old
   trunk inertial and a 2.657067 kg robot. **Task M4 rewrites every body's
   inertial from one composer** and consumes this output; writing it twice would
   guarantee the two disagree. Applying the M2 delta alone would put the robot
   at **2.731599 kg**.
2. **The absolute baseline is still unsettled.** The scope caution above stands:
   the 698.5 g upstream trunk booking carries its own unknown density, so this
   corrects the *delta*, not the absolute. M4 owns that.

**The provisional-profile caveat.** The masses above are measured at Protolabs
Network's *published* profile, not at a profile a vendor has confirmed for this
order (`print_process.json: profile_confirmed_with_vendor: false`). One extra
perimeter is worth ~130 g across the set. If a quote contradicts the published
standard, re-run `--emit-table`, `generate_cad_mods.py` and M4. No geometry is
touched, so the re-run is cheap.

---

<a id="mass-1"></a>
## MASS-1 · A uniform density per body inflates every tensor ~30 % — DOCUMENTED TRAP

Not a defect in the repo — a trap that will catch the next person who rebuilds
the mass model. Recorded by Task M4, 2026-08-12.

Fusing a body's meshes and smearing its declared mass over them at one uniform
density inflates the principal moments by a **median of 1.30x and up to 1.45x**:

```
body                            mass g   I_uniform / I_declared (desc)
hip_roll_assembly                85.98      1.363  1.450  0.995
left_roll_to_pitch_assembly      94.66      1.282  1.413  1.009
neck_yaw_assembly               111.31      1.400  1.302  1.258
head_assembly                   362.08      1.192  1.104  1.338
head_pitch_to_yaw                16.94      1.000  1.000  1.000   <- anchor
left_antenna_holder               4.22      1.000  1.000  0.999   <- anchor
```

The cause is physical, not numerical: the servos are dense masses sitting close
to the joint axes and the printed shells are hollow relative to a fused mesh, so
a uniform density moves mass outward. **A composer that assigns one density per
body is wrong by ~30 % and looks entirely plausible while doing it.** The
rebuild must be per-part with per-part densities, combined by parallel axes.

<a id="mass-2"></a>
## MASS-2 · The upstream export did not book printed parts at one density — DOCUMENTED TRAP

Three bodies are pure printed parts whose declared inertia is *exactly*
uniform-density inertia, and they pin the printed convention at **1250 kg/m³**:

```
head_pitch_to_yaw    m=0.0169378 kg / V=13.5490 cm³  ->  1250.11 kg/m³
left_antenna_holder  m=0.00421629 kg / V=3.3694 cm³  ->  1251.33 kg/m³
```

A fourth anchor pins the servo at the same time: `neck_yaw_assembly` composed
from printed meshes at 1250 plus **one servo at 74.5 g** gives 111.29 g against
111.31 declared — 0.02 %.

**But that convention does not hold across the robot.** `foot_assembly`'s
declared mass implies **748 kg/m³** over its fused meshes and
`hip_roll_assembly`'s printed remainder roughly **650**. So the upstream export
booked some bodies at solid density and some at as-printed. **The four anchors
verify a composer's arithmetic; they do not license "printed = 1250" as a model
of the whole machine.**

<a id="mass-3"></a>
## MASS-3 · A servo's mass splits by volume; its INERTIA does not — DOCUMENTED TRAP

`scripts/compose_body_inertials.py` divides one STS3250's 74.5 g across its five
case meshes in proportion to volume. Measured 2026-08-12, that reproduces the
servo's **total mass** (`neck_yaw_assembly` 111.29 g vs 111.31 declared, 0.02 %)
but **not its inertia distribution**: the same body's principal moments come out
at **0.906–0.981** of declared.

A real servo concentrates its motor and gearbox; volume-splitting spreads them
over the case. So `--verify-composer` gates `neck_yaw_assembly` on mass and only
*reports* its inertia. Closing this needs a servo mass-distribution model the
repo does not have — Task S.8 is where real hardware could supply one.

Two related counting traps, both verified:

- **De-duplicate on `(body, mesh, pos, quat)`.** The MJCF has 271 mesh geoms
  collapsing to **137** unique instances. Keying without the body name collapses
  the mirrored legs and silently deletes a leg's worth of mass; iterating raw
  `<geom>` elements double-counts every collision/visual twin.
- **Group servo meshes by `(body, quat)`, never by `(pos, quat)`.**
  `passive_palonier` carries a different `pos` from the other four in **every**
  servo, so a pos-based key splits it off as a phantom sixth part.
  `(body, quat)` yields exactly 14 groups of exactly 5.

---

# CFG — configuration and reward-wiring defects

<a id="cfg-1"></a>
## CFG-1 · `torque_z_range` is silently discarded — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> Task M0 switched the wrench from `set_forces_and_torques(..., positions=)` to `add_forces_and_torques(..., positions=)`, preceded by a composer `reset()` for those envs so `add` behaves as `set`. The `set_` variant's warp kernel assigns the composed torque and then assigns it again from the position moment, so `torque_z_range` was dead configuration for v5c and v5d alike.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


**Scope:** every v5 arm including the shipped v5d. The parameter is configured,
documented, threaded through the cfg, and frozen into
`exported_policies/v5d_contact_wrench_ppo/env.yaml:831-833` — and has never had
any effect.

`contact_events.py:345-352` passes `forces`, `torques` **and** `positions` in one
`set_forces_and_torques` call. In the warp kernel, the torque is assigned, then
assigned **again** (`=`, not `+=`) with `skew(r)×f`.

**Evidence [GPU]** — running the exact call the event makes:
```
requested τ = [0,0,0.15],  F = [3,0,0] at r = [0,0.04,0]
r × F                    = [0,0,-0.12]
EXPECTED if summed       = [0,0,+0.03]
ACTUAL composed torque   = [0,0,-0.12]    matches SUM? False   matches OVERWRITE? True
CONTROL, no `positions`  = [0,0,+0.15]    ✓
```

**Fix:** use `add_forces_and_torques_at_position` (the summing variant), or drop
`positions` and apply the offset moment manually.

<a id="cfg-2"></a>
## CFG-2 · Obstacles are placed twice per episode — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> Task M0 changed the fresh-episode predicate from `episode_length_buf <= 1` — true at **both** buf 0 and buf 1, so every episode drew its obstacle twice and the second draw silently replaced the first — to `== 1`.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


`_place_obstacles_for_fresh_episodes` triggers on `episode_length_buf <= 1`,
which is true both on the reset step (buf = 0) and the following step (buf = 1).
Each call re-rolls the Bernoulli `obstacle_active` and re-places the box — the
second time from a robot that has already taken a step.

**Evidence [GPU]** — instrumented rollout, placement calls at:
```
(50,51) (100,101) (124,125) (130,131) (140,141) (142,143)
CONSECUTIVE-STEP placement pairs sharing env ids (= double placement): 7
```
i.e. exactly one pair per reset.

**Fix:** trigger on `episode_length_buf == 1` only, or latch per episode.

<a id="cfg-3"></a>
## CFG-3 · v5c/v5d maintain a disturbance gate that nothing reads — LOW

**Scope:** v5c and v5d, i.e. the shipped policy.

Because v5d descends `ContactMinimal → Robust`, it inherits `DuckRewards` (v4's
reward set), not `DuckContactRewards`. `ContactRegimeEvent` still runs its full
per-env state machine every step across 4096 envs.

**Evidence [GPU]** — live reward manager on `ContactWrench-v0` lists 9 terms, all
stock `isaaclab.envs.mdp.rewards:*` plus `ImitationReward(params={'command_name'})`:
```
reward terms that READ the disturbance gate: NONE
mean gate duty (on-policy, 30 s)           : 0.4261
```
Static confirmation — occurrences in the shipped `env.yaml`:
```
ground_contact=0  GatedTrack=0  flat_orientation_deadzone=0  feet_slide=0
disturbance_gate=0  polynomial_coefficients_v2=0  normalized_match=0
```

Cost is wasted computation plus a misleading docstring; the wrench itself still
acts as physics, which is what v5d actually learned from.

<a id="cfg-4"></a>
## CFG-4 · The `head` half of `ground_contact_penalty` can never fire — MEDIUM

**Scope: v5a/v5b only.** The shipped v5d does not carry this term (see CFG-3).

The term binds body `head`, which is a 1e-09 kg marker frame with no `<geom>`
and therefore no collider. The real head is `head_assembly`.

**Evidence [GPU]:** over a rollout, max ‖F‖ on `head` = **0.000000 N**, while the
feet on the same sensor recorded **821.5 N / 227.0 N**.

The damage is bounded but not nil: the ablation that selected v5d compared arms
one of which ran a reward term at half its documented strength. Fix is one word,
`head` → `head_assembly`, and belongs to the next campaign since it changes the
reward.

<a id="cfg-5"></a>
## CFG-5 · ContactSensor history spans 15 ms, not 60 ms — LOW

`history_length=3` with `update_period=0.0` rolls once per **physics** substep,
not per policy step.

**Evidence [GPU]:** 2 of the 3 history slots already differ from slot 0 *within a
single policy step*. With `sim.dt=0.005` and `decimation=4`, the buffer spans
3 × 5 ms = **15 ms**, less than one 20 ms control step.

Any code treating it as "the last 3 policy steps" is off by 4×. Today this makes
the in-training gait canary (`.amax` over history) systematically read a higher
duty than the evaluator (instantaneous `net_forces_w`).

---

# EVAL — measurement harness

<a id="eval-1"></a>
## EVAL-1 · `--report-only` injects every JSON in the directory — HIGH

`write_comparison_markdown` enumerates the results directory and appends **every**
`*.json` with no filter of any kind:

```python
747:     entries = []
748:     if os.path.isdir(results_dir):
749:         for fname in sorted(os.listdir(results_dir)):
750:             if fname.endswith(".json"):
751:                 with open(os.path.join(results_dir, fname)) as f:
752:                     entries.append(json.load(f))
```

**This has already fired.** `eval_results_v4/` contains
`v4_inertials_pusheval.json` and `v4_robust_pusheval.json`, while
`v4_comparison.md` states "no external pushes". Any regeneration silently mixes
push-eval rows into a table documented as push-free.

**Fix:** add an allowlist or task-id filter argument; until then, treat
`v4_comparison.md` as frozen — do not regenerate it.

> **FIXED 2026-08-12** (task_plan_v2.md Task R0).
> `write_comparison_markdown(md_path, results_dir, include=None)` now takes an
> allowlist of policy names, exposed as the repeatable CLI flag
> `--include NAME` (exact match on each JSON's `name` key). `include=None`
> keeps the old unfiltered behaviour, so `v4_comparison.md` and
> `v5_comparison.md` regenerate exactly as before and the archived tables are
> not disturbed.
>
> Evidence — the real script, run in a subprocess over a two-JSON directory:
>
> ```
> $ python3 scripts/evaluate_policies.py --report-only \
>     --output_dir $TMP --comparison_md $TMP/t.md --include keep_me
> [report-only] regenerated .../t.md
> keep=1 drop=0          # with --include
> keep=1 drop=1          # without it
> ```
>
> Guarded by `tests/test_eval_report_filter.py` (7 tests), which executes the
> script rather than grepping it — TEST-1 is the reason. The check is retired
> from `scripts/verify_known_issues.py`.
>
> `allow_abbrev=False` is set on that parser, so `--incl` will not silently
> work; always spell `--include` in full.
>
> **`eval_results_v4/` is still mixed.** The fix stops new corruption; it does
> not retroactively clean `v4_comparison.md`, which remains frozen.

<a id="eval-2"></a>
## EVAL-2 · The documented protocol is not the protocol that ran — MEDIUM

**Evidence:**
```
v5_pipeline.sh CONDITIONS has 6 entries; gate is 'PASSED -lt 5' (i.e. >= 5 of 6)
eval_results_v4 JSONs condition counts: [5]   (5 files)
eval_results_v5 JSONs condition counts: [6]   (17 files)
episodes per v5 file: [3840]
```
against `AGENTS.md`, which mandates 5 conditions × 10 windows × 64 envs = 3,200
episodes and a gate of ≥ 4/5. The v5 campaign additionally rendered its turn
video at wz = 0.5 where `AGENTS.md` specifies wz = 0.3.

Neither protocol is wrong; they are simply different, and the difference is
undocumented. Note the "five conditions" preamble inside `v5_comparison.md` is
**machine-generated** (`PROTOCOL_HEADER`, `evaluate_policies.py:573-615`) and
describes the script's *defaults*, not the run — so the fix belongs in the
generator, which should render the header from each entry's own `protocol` block.

---

# SHELL — orchestration

<a id="shell-1"></a>
## SHELL-1 · Run directory selected by mtime, not by run name — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task R2b preparation
>
> The pipeline is handed a run name and now uses it:
> `RUNDIR="${RUNDIR_OVERRIDE:-$(ls -td "$LOGROOT"/*_"$RUN_NAME"/ …)}"`.
> An mtime fallback survives for the case where no name-matching
> directory exists, but it fires only then and **prints a warning
> naming this issue** when it does.
>
> `LOGROOT`, `RESULTS`, `COMPARISON_MD`, `CONDITIONS` and a new
> `INCLUDE_ARGS` are environment-overridable, so the rebuild campaign
> reuses this pipeline rather than editing its constants inline — which
> would have destroyed the v5 campaign's reproduction path. Defaults are
> unchanged, so a bare invocation still reproduces v5 exactly.
>
> `INCLUDE_ARGS` also closes the EVAL-1 exposure inside the pipeline:
> without it every `*.json` in the results dir is injected into the
> comparison table.
>
> Check retired; the entry stays here.


```
RUNDIR=$(ls -td "$LOGROOT"/*/ 2>/dev/null | grep -v v4robust_seed | head -1)
references RUN_NAME: False
candidate dirs under open_duck_ppo_v5: 6
```

The pipeline is passed a run name but never uses it to find the run. It is
partially protected upstream — lines 54-58 block on `$STATE/${RUN_NAME}.pid`, so
selection cannot begin until the *named* run's process group is dead — but any
write into an older run dir between that point and line 65 reorders the
selection. Consequence is not a crash: results for run B get written under run
A's name.

**Verified correct for the shipped run** — `chain_v5d_contact_wrench.log` records
`2026-07-29_08-59-25//model_5998.pt`, matching the archived checkpoint.

<a id="shell-2"></a>
## SHELL-2 · `v5_chain.sh` disarms the launcher's duplicate-run guard — MEDIUM

`v5_chain.sh` runs `rm -f "$STATE/${RUN_NAME}.pid"` immediately before invoking
the launcher, whose guard begins `[[ -f "$PIDFILE" ]] && kill -0 ...`. Deleting
the file short-circuits the guard before the liveness half ever runs.

Collateral if it fires: the launcher's `> "$LOG"` truncates the live run's log,
and the new pidfile overwrites the only handle to the original process group.
`wait_for_gpu` on the preceding line narrows the window but does not close it.

<a id="shell-3"></a>
## SHELL-3 · `--algorithm` dispatches to a deleted script, recorded as success — LOW

```
scripts/train_amp.py exists: False
launcher references it: True
deleted by: 34f70fd  "Remove the AMP track; PPO (v4_robust) is the sole locomotion mainline"
```
The launcher still writes a pidfile for the already-dead process and exits 0, so
`run_experiment_queue.sh` records the run as `completed (process group exited)`.

**Scope: dead code.** The queue last ran 2026-06-15, before the deletion, and
nothing would pass `--algorithm` again without resurrecting AMP.

<a id="shell-4"></a>
## SHELL-4 · Queue's GPU-busy predicate is blind to eval and play — LOW

```
TRAIN_PATTERN = 'train_(ppo|amp)\.py'
matches: train_ppo.py=True, evaluate_policies.py=False, play_policy.py=False
```
So the queue's "one GPU job at a time" guarantee is false for the two other
Isaac entry points.

**Scope: dormant.** v5 ran on `v5_chain.sh`, whose `wait_for_gpu` greps
`train_*` **and** `evaluate_policies`; the residual gap in the live harness is
`play_policy.py` only.

<a id="shell-5"></a>
## SHELL-5 · The training watchdog fails open — MEDIUM

The 30-minute gait canary parses the last `Gait/duty_in_band_frac` with
`grep -oE "[0-9]+\.[0-9]+"` and feeds it to `python3 -c` with `2>/dev/null`. Any
parse failure leaves `BAD` empty, `[ "$BAD" = "1" ]` is false, and control falls
into the `else` branch, which prints **"healthy, letting it run"**. A multi-decimal
or scientific-notation log line therefore reports a doomed run as healthy.

Historical note, verified: `v5a_gated_ft.log` and `v5b_ungated_ft.log` contain
**zero** occurrences of the metric because it was introduced in the same commit
as the watchdog (`2fc57c9`). The watchdog could not have run against them. On
the one run it did guard, it worked (`duty_in_band_frac=0.9852 — healthy`).

---

# ART — artifacts and provenance

<a id="art-1"></a>
## ART-1 · `policy.onnx` is gitignored while its weight sidecar is tracked — HIGH

```
git check-ignore: .gitignore:19:*.onnx  exported_policies/v5d_contact_wrench_ppo/policy.onnx
tracked in that dir: README.md, agent.yaml, env.yaml, model_5998.pt, policy.onnx.data, policy.pt
```

A fresh clone receives **788 KB of orphaned external weights and no graph to
load them with** — the offsets and shapes live only in the `.onnx`. This breaks
the `trtexec --onnx=policy.onnx` instruction in `AGENTS.md`. Note the two
root-level `BEST_WALK_ONNX*.onnx` files *are* tracked, so the ignore rule is
inconsistent about which ONNX matters.

<a id="art-2"></a>
## ART-2 · The shipped policy's provenance parent is not in git — MEDIUM

**Reframed 2026-08-15.** The original complaint was that `exported_policies/v4_robust/`
was never created. [`locomotion_selection.md`](locomotion_selection.md) retired
`v4_robust` outright and pruned `exported_policies/` to the single mainline
archive, so that framing would now be CONFIRMED forever for an **intentional**
reason — a check describing no defect. The concern itself is real and moved.

`exported_policies/` contains exactly `v6d_contact_wrench_ppo/`. Its
`README.md` records the provenance *"fine-tuned from `v6_robust`
(`model_2999.pt`), itself trained from scratch on the post-Phase-M plant"* —
and `v6_robust` is **not in the repository**. It survives only at
`~/IsaacLab/logs/rsl_rl/open_duck_ppo_v6/2026-08-13_01-05-40_v6_robust/`.

So a clean clone can load and deploy the shipped policy, but **cannot reproduce
or verify the control it was measured against**. Every contact-gate number in
`rebuild_results.md` is relative to that run.

Two honest options, neither taken yet: archive `v6_robust`'s checkpoint too
(≈4.7 MB, and it is the control for every published delta), or accept that the
comparison is reproducible only from the eval JSONs already committed — which
do carry the control's full aggregate and per-condition numbers.

<a id="art-3"></a>
## ART-3 · `usd/config.yaml` records a deleted worktree — LOW

The committed converter record names
`/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson-cad/...` as `asset_path`,
a worktree that no longer exists. The staleness guard still passes because
`tests/test_usd_conversion.py` pops the path keys before hashing — so the asset
is fine and only its recorded provenance is unverifiable.

<a id="art-4"></a>
## ART-4 · `.gitignore` swallows every `.txt` in the repo — MEDIUM

`.gitignore:18` is a bare `*.txt`, with no negation anywhere. It applies to the
whole tree, not just to the training-queue files it was presumably written for.

```
$ sed -n '18p' .gitignore
*.txt
$ git check-ignore -v jetson_runtime/prompts/navigate.txt
.gitignore:18:*.txt	jetson_runtime/prompts/navigate.txt
```

Only **one** `.txt` file is tracked today
(`experiments/RL/new/bdx_walk_in_place.txt`), so nothing is lost yet. The defect
is latent and fires in Phase 5: the planned Cosmos prompt library, any recorded
VLM transcript fixture, and any plain-text golden file would be created,
apparently succeed, and never be committed. `git status` stays clean, which is
exactly what makes it dangerous — the loss is silent and is only discovered on a
fresh clone.

**Fix:** narrow the rule to where it is needed (`.training_runs/*.txt`), or add
an explicit negation for the directories that must be tracked, e.g.

```
*.txt
!jetson_runtime/**/*.txt
!tests/fixtures/**/*.txt
```

Verify with `git check-ignore -v <path>` — a clean `git status` is not evidence
the file is tracked.

---

# DEPLOY — the Phase-4 contract

`jetson_runtime/` does not exist and no file in the repo imports onnxruntime, so
none of these are live bugs. They are the specification the runtime must satisfy,
with the measured cost of each mistake.

<a id="deploy-1"></a>
## DEPLOY-1 · The ONNX omits `action_scale` and `q_default` — HIGH

> **MITIGATED 2026-08-13 by Task R3 — NOT fixed, and the check stays.** A
> sidecar now ships with the policy:
> `exported_policies/v6d_contact_wrench_ppo/deployment_contract.json`, generated
> by `scripts/make_deployment_contract.py` from the repo's own sources and
> machine-checked by `scripts/verify_deployment_contract.py` (10 checks, exit 0).
> It carries `action_scale`, `q_default_rad`, the joint order, the normaliser
> epsilon and the batch-dim constraint.
>
> **The ONNX graph itself is unchanged.** Re-measured against the v6d export:
> **0 of 193,784** initializer scalars equal 0.25, and the closest action-width
> tensor (`mlp.6.bias`) differs from `q_default` by **1.374409 rad**. A runtime
> that ignores the sidecar and commands the graph output directly is still wrong
> by a 4x gain and the whole standing pose. Do not retire this check.

**Evidence** — exhaustive enumeration of the graph:
```
input : obs [1, 59]        output: actions [1, 16]
nodes : Sub, Div, Gemm, Elu, Gemm, Elu, Gemm, Elu, Gemm
total scalars across all initializers: 197126
values equal to 0.25 (atol 1e-6)     : 0
16-element tensors (could hold q_default): ['mlp.6.bias']
  max|mlp.6.bias - q_default| = 1.461333
```

The action pipeline `q_target = q_default + 0.25·a` lives entirely in Isaac
Lab's `JointPositionAction`. A runtime that commands the ONNX output directly is
wrong by a **4× gain and a standing-pose offset up to 1.379 rad** — total, silent
failure. Related trap: the observation's joint block is `joint_pos_rel`
(`q − q_default`), not raw encoder angles; feeding absolute angles puts the knees
13–14 σ outside the training distribution.

> **The values already exist in this repo — do not re-type them.**
> `scripts/duck_init_pos.json` carries the canonical `joint_order` (16 names) and
> `init_pos_rad` (16 values), which is exactly the `q_default` the graph omits.
> A hardware runtime should load that file rather than transcribe angles, and
> should key by joint NAME, not by index: Isaac Lab's joint order is interleaved
> and differs from both the MJCF order and the Playground polynomial order (see
> `AGENTS.md` "Joint Orders", which also warns against a legacy 15-joint table).
> `action_scale` is 0.25 and is not in the file — it is in the env config.

<a id="deploy-2"></a>
## DEPLOY-2 · The normalizer epsilon is not in the checkpoint — MEDIUM

The graph divides by `_std + 0.01`. That `0.01` is a plain Python attribute in
`rsl_rl/modules/normalization.py`, never a registered buffer, so it is
structurally absent from `model_5998.pt` (`divisor − _std = 0.01` to 9.5e-09 on
all 59 channels). Dividing by `_std` alone is off by **28.0%** on
`joint_pos_rel:right_hip_yaw` and 19.9% on `projected_gravity[y]`; median 3.7%.

Measured action cost: max 0.0229 rad (1.3°), mean 0.0030 rad. Modest, but it
presents as gait asymmetry — i.e. it looks like a calibration problem rather
than a code bug. **Only affects a hand-rolled reimplementation**; rsl-rl's own
loader reconstructs the same default, so `policy.pt` and `policy.onnx` are correct.

<a id="deploy-3"></a>
## DEPLOY-3 · 4 of the 59 observation dims cannot be measured on hardware — HIGH — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task M0 / M0b
>
> Task **M0b** removed the antennas from the interface: **obs 59 -> 53, action 16 -> 14, critic 62 -> 56**, all three read back off a freshly built environment rather than computed. The four dims that could never be measured on hardware no longer exist. 53, not 55: the `actions` observation term is `last_action` with `action_name=None`, so it returns the whole action tensor and shrinks with it.
>
> Check retired from `scripts/verify_known_issues.py`; the entry stays here.


```
antenna joint indices: [13, 14]
  -> obs dims joint_pos_rel [22, 23], joint_vel_rel [38, 39]; action outputs [13, 14]
```
The antennas are open-loop SG90 PWM servos with no position feedback. Neither
zeros nor the commanded angle is a free choice — both are distribution shift,
and they were trained live with the wrong actuator model behind them
([PLANT-4](#plant-4)).

**This must be decided before the obs builder is written.** Cheapest resolution
is upstream: drop the antennas from the action and observation spaces in the
next campaign.

<a id="deploy-4"></a>
## DEPLOY-4 · Exported ONNX has a hard-fixed batch dimension of 1 — LOW

`obs: float32[1,59]` is a literal, not a symbolic dim, because the exporter
passes `dynamic_axes={}`. Batch 2 and 4 raise `InvalidArgument`. Irrelevant
on-robot (one duck steps at batch 1) but blocks batched offline replay. Patching
`dim_param="B"` reproduces looped batch-1 output bit-for-bit.

<a id="deploy-5"></a>
## DEPLOY-5 · Documented velocity clamp exceeds the trained hull — MEDIUM

```
AGENTS.md clamp : forward [-0.2, 0.3], lateral [-0.2, 0.2], turn [-0.3, 0.3]
trained hull    : vx (-0.148, 0.222), vy (-0.111, 0.111), wz (-0.5, 0.5)
forward overshoot 0.30/0.222 = 1.35x ; backward 0.20/0.148 = 1.35x ; lateral 1.80x
```
The turn clamp (±0.3) is correctly *inside* the trained ±0.5. Two numbers 158
lines apart in the same document disagree.

> **FIXED 2026-08-11.** `AGENTS.md` now documents the clamp as the trained hull
> exactly — forward 1.00×, backward 1.00×, lateral 1.00×, turn still correctly
> inside. The verifier check has been retired per this register's convention
> that a fixed issue carries no check.

---

# TEST

<a id="test-1"></a>
## TEST-1 · The suite is source-text grepping; 5 of 7 regressions pass — HIGH

33 of the 47 tests in `tests/test_isaac_lab_env.py` are
`assert "<literal>" in open(file).read()`; the file imports no project code.
Because the literals are not unique in the 1046-line `env_cfg.py`, several
assertions cannot fail for the reason their docstrings claim.

```
literal counts in env_cfg.py:
  '(-0.5, 0.5)'=5   'weight=1.0'=7   '(-0.148, 0.222)'=2   '"std": 0.5'=6   'scale = 0.25'=1
```

Mutation results (mutate → run → `git checkout --`), baseline `101 passed`:

| Mutation | Full suite |
|---|---|
| Delete `imitation_reward` from `DuckRewards` entirely | **101 passed** |
| `lin_vel_x` hull → `(-9.9, 9.9)` | **101 passed** |
| `armature` 0.040 → 0.0499 and `friction` 0.200 → 0.2999 | **101 passed** |
| head-group `stiffness` 45.53 → 99.99, legs unchanged | **101 passed** |
| `limit_angle` `radians(60.0)` → `radians(89.0)` | **101 passed** |

The first is the sharpest: `"ImitationReward" in content` is satisfied by the
bare **import at line 60**, and `"weight=1.0"` by a **comment at line 122**. That
is the reward class the shipped v5d trained with.

**The template for the fix already exists in the file:**
`test_v4_and_v3_tracks_unchanged` (`:533`) asserts the *full assignment
statement*, not the bare tuple — which is why it caught the `ang_vel_z` mutation.

<a id="test-2"></a>
## TEST-2 · A bare `pytest` from the repo root fails at collection — LOW

`pytest.ini` sets no `testpaths`, so pytest walks into `experiments/RL/old_test.py`
→ `ModuleNotFoundError: No module named 'gymnasium'`. Only `pytest tests/` works.
Any bare pass-count claim in this register implicitly means `python3 -m pytest tests/`.

> **Baseline moved 2026-08-11: it is now `103 passed`**, not 101. The PLANT-1 fix
> added two regression tests to `tests/test_mass_inertia.py`. The mutation table
> in [TEST-1](#test-1) records `101 passed` because that was the baseline when
> those mutations were run; the conclusion is unchanged, but re-running them
> today should compare against 103.

<a id="test-3"></a>
## TEST-3 · `requires_isaac_sim` is documented but applied to zero tests — LOW

```
textual mentions in tests/: 1 (all prose, the module docstring); real decorators: 0
registered in pytest.ini: False
pytest's own view -> 101 deselected
```
The docstring's "and are skipped by default" implies a deferred GPU-side test
set that was never written.

---

# REF — imitation reference library

<a id="ref-1"></a>
## REF-1 · No `vy=0` / `wz=0` cell; contact dims are constant — MEDIUM

```
cells: 240
vy axis: [-0.111, -0.037, 0.037, 0.111]   contains 0.0? False
wz axis: [-1.111, -0.852, -0.593, -0.333, -0.074, 0.185, 0.444, 0.704, 0.963, 1.222]   contains 0.0? False
contact dims 32/33 identical across ALL cells: True
```

Two consequences. First, the "straight walk" command (0.2, 0, 0) snaps to
`0.222_-0.037_-0.074`, so the reference has a baked-in lateral and yaw
component and the reference-tracking metric has a floor that is a property of
the grid, not the policy. Second, the contact sub-reward compares against a
single fixed schedule for every command, so it carries no command-dependent
information.

<a id="ref-2"></a>
## REF-2 · Knee references are clamped; velocity targets are not — MEDIUM

```
reachable cells under the shipped hull (|wz| <= 0.593): 120 of 240
KNEE reference samples outside soft limits (clamped at runtime): 2177/6480 = 33.60%
max |reference joint velocity| over reachable cells: 24.10 rad/s vs actuator limit 8.94 (2.70x)
reference velocity samples exceeding 8.94 rad/s: 3.89%
```

Positions are clamped at `imitation_reward.py:265-268`; velocities are not. So
terms 1 and 2 of the composite ask for different trajectories: term 1 for a
clamped pose, term 2 for the velocity of the unclamped one.

> Restricting to cells the shipped command hull can actually select makes the
> knee clamp **worse** (33.60% vs 31.98% over all 240), while the headline
> velocity figure shrinks from 41.3 to 24.1 rad/s. Earlier drafts quoted the
> all-cell numbers, which include commands no shipped policy can reach.

---

# DOC — documentation defects

<a id="doc-2"></a>
## DOC-2 · The experiment journal has zero v5 entries — MEDIUM — ✅ **FIXED 2026-08-13**

> ### FIXED 2026-08-13 — Task R4
>
> `experiment_journal.md` carries **Runs 17–21** (`v5_smoke`,
> `v5a_gated_ft`, `v5b_ungated_ft`, `v5c_contact_only`,
> `v5d_contact_wrench`), each with last-100-iteration TensorBoard means
> from `scripts/tb_summary.py` — **not** console greps, per AGENTS.md
> rule 1 — its gate JSON path, a verdict naming the gate, artifact
> paths, and a **3.657 kg plant banner** pointing at `m2657_regate.md`.
> The run-index table gained five rows.
>
> `docs/jetson-mod/v5_contact_results.md` now exists, with the campaign's
> finding stated plainly: **the curriculum has to contain the disturbance
> it is meant to survive** — `v5c` (obstacles, no wrench) fell in 100 % of
> wrench episodes, identically to the control, and adding the wrench took
> `v5d` to 47.109 %.
>
> Check retired; the entry stays here.


```
v5 run names in experiment_journal.md: v5a_gated_ft=0, v5b_ungated_ft=0,
  v5c_contact_only=0, v5d_contact_wrench=0, v5_smoke=0
docs/jetson-mod/v5_contact_results.md exists: False
```
`AGENTS.md` makes a journal entry mandatory per run and `v5_retrain_plan.md`
mandates the results doc. Five executed runs — including the one that produced
the shipped policy — exist only inside a file titled "Execution Plan".


> **Half discharged 2026-08-11.** `docs/jetson-mod/v5_contact_results.md` now
> exists, built by extracting the run records from `v5_retrain_plan.md` §§14-25
> and the JSONs in `eval_results_v5/`, and mirroring `v4_retrain_results.md` so
> the two generations are comparable. **The journal half is still open** — all
> five v5 run names still appear zero times in `experiment_journal.md`, and
> `AGENTS.md` requires an entry per training run. The verifier check has been
> narrowed to the journal alone, so half a fix can no longer turn it green.
> Task R4 in `task_plan_v2.md` closes the remainder.

<a id="doc-3"></a>
## DOC-3 · Task 2.7 docs still assert a 2.657 kg plant — HIGH

`validation_results.md` and `v4_retrain_results.md` both still describe the plant
as 2.657 kg after the 2026-08-02 audit measured 3.657 kg in simulation. A reader
taking the Task 2.7 certification at face value inherits the wrong plant mass.
A caveat has been added to both.

<a id="doc-4"></a>
## DOC-4 · `task_plan.md` is stale on four axes — MEDIUM

Verified present: the 56-dim TensorRT wrapper spec (against a 59-dim actor), the
push ramp "0.5→1.3 m/s" (superseded by 0.4→0.7 in `v5_retrain_plan.md` §11.4),
and the wrench "0.8–4.8 N at ~1.6 kg" (the sim robot is 3.657 kg). Task 2.8 is
still marked `PLANNED` after four runs shipped a winning policy.

The 56-dim spec has been corrected, since it is the number someone would code
Phase 4 against.

> **FIXED 2026-08-11 — all four axes.** The push ramp now reads 0.4→0.7 m/s with
> a pointer to `v5_retrain_plan.md` §11.4; the wrench row is sized against the
> declared 2.657 kg rather than "~1.6 kg"; Task 2.8 is marked COMPLETE with v5d
> named as the shipped winner. The verifier check has been retired.

<a id="doc-5"></a>
## DOC-5 · `AGENTS.md` says the USD came from URDF — LOW

`convert_mjcf_to_usd.py` and `usd/config.yaml` both name `robot_motors.xml`; the
URDF is in no code path. Corrected.

<a id="doc-6"></a>
## DOC-6 · `AGENTS.md` actuator snippet sets an unused field — LOW

The copy-pasteable snippet writes `effort_limit = 8.716`; the code sets
`effort_limit_sim=8.716` and leaves `effort_limit` at `null`. For implicit
actuators upstream treats these as aliases, so behaviour matches — but pasting
the documented snippet sets a different, currently-null field. Corrected.

---

<a id="naming-families"></a>
# The pattern behind PLANT-1, PLANT-2 and CFG-4

Three independent config sites bind to a body whose **name reads correctly** but
whose **physics is wrong**:

| Site | Names | Should name | Symptom |
|---|---|---|---|
| `add_base_mass` (`env_cfg.py:365`) | `trunk_assembly` | `base` | randomizes a body that has no error, misses the one that does |
| `ground_contact` (`env_cfg.py:521`) | `head` | `head_assembly` | penalises a collider-less frame; never fires |
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

# What these issues do NOT affect

Stated explicitly, because the register is long and it would be easy to conclude
more is broken than is.

- **The MJCF, MuJoCo and MJX paths are correct** and need no change. Upstream's
  model is right for upstream's simulator; PLANT-1 is created at the
  MuJoCo→PhysX boundary that only this fork crosses.
- **The CAD record is correct.** 2.657067 kg remains the accurate robot mass and
  is the right number for hardware, BOM, and battery/torque budgeting.
- **Per-body masses other than `base` are exact** — all 21 authored bodies match
  the MJCF to float32 resolution, zero mismatches.
- **The USD is not stale.** The asset-hash and mesh-manifest guards pass.
- **The BAM actuator parameters are faithfully applied.** `kp`, `damping`,
  `armature` and `friction` match `params_sts3250_id008.json` exactly; only the
  torque ceiling ([PLANT-5](#plant-5)) and the antenna group
  ([PLANT-4](#plant-4)) are in question.
- **Published gate numbers stand.** PLANT-3 is training-only; every `_PLAY`
  config spawns from the clean pose.
- **The 59-dim observation width and term order are correct in the code.** The
  historical defects were in the *documentation* of that contract, and are fixed.
- **In simulation nothing is broken.** Training and evaluation share one USD, so
  every sim-only result is internally valid — for a 3.657 kg robot.
- **The exported inference artifacts are mutually consistent.** ONNX, TorchScript
  and a from-scratch numpy reimplementation agree to ~1e-6, and all eight weight
  tensors are bit-identical to the checkpoint.

---

<a id="not-issues-verified-refutations"></a>
# Not issues — verified refutations

Recorded so they are not raised again. Each was proposed, tested, and failed.

| Proposed | Verdict | Evidence |
|---|---|---|
| The sustained-wrench gate never tests rotation-under-load | **REFUTED** | The gate JSON contains a `wz=0.5` condition: 640 episodes, 375 falls (58.59%). `evaluate_policies.py:apply_condition` overrides the cfg's `ang_vel_z=(0,0)` per condition. **Never judge an eval task from its EnvCfg alone.** |
| L/R foot swap between the training reward and the eval metric | **REFUTED** | `find_bodies(FOOT)` returns `[7, 20]` both with and without `preserve_order=True` |
| `joint_pos_limits` is active at the nominal standing pose | **REFUTED** | Term value 0.000000; knees sit 0.046 / 0.035 rad from the soft limit |
| §4.7 below-ground resets do not reproduce | **REFUTED** (the refutation was wrong) | See [PLANT-3](#plant-3) — measured with link origins instead of collision vertices |

---

# Open leads — plausible, not verified

Do **not** cite these as findings. Carried forward from the 2026-08-02 plant
audit plus this pass.

- **Self-collision is disabled** (`enabled_self_collisions=False` — *verified as a
  setting*). The claim that 71.5% of in-limit poses self-collide in MuJoCo is
  **not measured**; the impact is unknown.
- Trunk CoM randomization may deliver ~3.4× less whole-body CoM shift than its
  ±10/±5 mm range implies, because it moves one body of 22.
- Observation corruption is i.i.d. zero-mean only — no IMU bias, drift, or
  mounting misalignment — and no gate has been measured with corruption on.
- The three in-repo plants (`robot.xml`, `robot_motors.xml`, `robot_cfg.py`) may
  disagree on joint damping and actuator gain, with no test covering it.
- Closed-loop cost of DEPLOY-1/DEPLOY-2 is unmeasured; all figures are
  open-loop, single-step.
- Whether TensorRT specifically rejects DEPLOY-4's fixed batch dim is inferred
  from the graph, not measured (`trtexec` is not on PATH).

---

# Resolved

- **The deployment-contract documentation defects** (relative joint positions,
  observation term order, termination thresholds) were fixed in `AGENTS.md` and
  re-verified still correct on 2026-08-09: `joint_pos_rel` is named, 60° / 0.09 m are documented,
  and the code still uses `math.radians(60.0)` and `minimum_height: 0.09`.

---

# Appendix — reproducing this register

```
# static, no GPU -- 34 checks over source, MJCF, USD, ONNX, JSON and pickles
python3 scripts/verify_known_issues.py
python3 scripts/verify_known_issues.py PLANT-3        # a single check
~/IsaacLab/_isaac_sim/python.sh scripts/verify_known_issues.py DEPLOY-1   # needs onnx

# runtime, requires the GPU and exactly one Isaac job at a time
cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/audit_plant_mass.py --headless
```

The runtime measurements marked **[GPU]** for CFG-1..CFG-5 came from ad-hoc
probes rather than a committed script. `audit_plant_mass.py` is the only
committed runtime gate; the rest are quoted verbatim above and would need
re-instrumenting to re-measure.

**Sequencing rule for anyone re-running the runtime checks:** one Isaac job at a
time (`AGENTS.md` journal-protocol rule 7), no process may modify tracked source
while a probe is running, and build one env per process — a second `gym.make`
after `env.close()` in the same Kit process hangs.

---

# Provenance and method

This register has two ancestries, both preserved here.

**The 2026-08-02 plant audit** (formerly `docs/jetson-mod/sim-plant-fidelity.md`,
deleted 2026-08-09 when its content was absorbed): five parallel read-only
auditors over the asset chain, actuator model, domain randomization,
observation/action contract, and docs-vs-code, each finding adversarially
re-checked by an independent verifier prompted to refute it. 25 candidates
raised, 9 adversarially verified, everything written up re-measured by hand.
Findings neither verified nor hand-checked were left out entirely.

**The 2026-08-09 consolidation pass** (this document): a full re-read of the
repository and the relevant upstream Isaac Lab machinery, then mechanical
re-verification of every claim before inclusion. Four claims were **refuted** and
moved to [§ Not issues](#not-issues-verified-refutations); one entry inherited
from the plant audit was found factually wrong and rewritten
([PLANT-8](#plant-8)); several severity figures were reduced after checking
whether the worst-case numbers came from configurations the shipped policy can
actually reach.

**Not checked by either pass:** training hyperparameters, reward-function
correctness beyond the terms named above, anything in `jetson_runtime/` (it does
not exist), and **real-hardware measurements of any kind**. Nothing in this
document is a claim about the physical robot's behaviour — only about what the
simulator does and whether the repo describes it accurately.

## SERVO-1 · `neck_pitch` commands 207 % of continuous — and it is the GAINS, not the head mass — HIGH

> **CORRECTED 2026-08-15, same day as filing.** This entry first said the cause
> was "the head assembly held against gravity, i.e. a **static** load", that
> "retraining cannot fix it", and that "the candidates are mechanical". **All
> three were wrong**, and they were wrong in the direction that costs money —
> they implied a head redesign before the print order. The measurement below
> replaces them.

**The measurement stands.** `scripts/measure_joint_torque.py` on
`v6d_contact_wrench`, current plant, its own PLAY task, vx = 0.2, all
disturbances off (`v6d_torque_measurement.md`):

```
joint          peak     p99     rms   % steps over continuous
neck_pitch    4.710   4.523   3.241        94.70%
```

3.241 / 1.569 = **207 % of the STS3250 continuous rating**, sustained.

**The cause is not gravity.** Computed from the MJCF with full forward
kinematics (body quaternions applied — a flat position sum gets this wrong):

| | |
|---|---|
| mass outboard of `neck_pitch` | 541.6 g (neck + head + antenna holders) |
| horizontal CoM offset from the pitch axis | **14.4 mm** — the head is well balanced |
| **static gravity torque** | **0.077 N·m = 5 % of continuous, 2 % of stall** |
| measured RMS while walking | 3.241 N·m = **42× the static requirement** |

Holding the head up is nearly free. The torque is **dynamic**.

**The cause is the control gains.** `robot_cfg.py:118-131` gives the `head`
group *the same numbers as the `legs` group*:

```
legs: stiffness=45.53  damping=1.346  armature=0.040
head: stiffness=45.53  damping=1.346  armature=0.040   <- copied verbatim
```

With `stiffness = 45.53 N·m/rad`, torque is `45.53 × tracking_error`:

| torque | implied tracking error |
|---|---|
| 0.077 N·m (static hold) | 0.10° |
| **3.241 N·m (measured RMS)** | **4.08°** |
| 4.903 N·m (clip ceiling) | 6.17° |

So the neck is producing 207 % of its thermal rating because the head lags its
commanded angle by **about four degrees** while the trunk bobs and rotates, and
a leg-sized proportional gain turns four degrees into 3.2 N·m. The policy
actively commands `neck_pitch`/`head_pitch`/`head_yaw`/`head_roll` (they are 4
of the 14 actions) and `joint_deviation_head` rewards holding them at default,
so the servo is actively fighting body motion to keep the head still.

`armature = 0.040 kg·m²` compounds it: the head link's own principal inertia is
**0.0020 kg·m²**, so the actuator model attributes to the neck a reflected rotor
inertia **~20× the thing it is moving**. That is defensible as a servo property
(same servo, same gearbox) but it means the head group inherited a tuning sized
for a load 20-40× heavier.

**Candidate fixes, cheapest first — none of them mechanical:**

1. **Retune the `head` group's `stiffness`/`damping`** for its actual load and
   retrain. This is a config change.
2. **Let the head track the trunk** instead of resisting it — relax or reshape
   `joint_deviation_head` so the policy is not paying reward to hold the head
   still against body motion.
3. Add an explicit torque penalty (shared with [SERVO-2](#servo-2)).
4. Only if 1-3 fail: mechanical work.

**Do NOT redesign the head or delay the print order on account of this entry.**
The head is balanced to 14 mm and holds itself for 5 % of the servo's continuous
rating.

**Epistemic status.** This is the *simulator's* actuator model — an implicit PD
law with a clip. A real STS3250 runs its own firmware PD with its own gains, so
these torques are a red flag to investigate on the bench, not a measured
hardware failure. Task **S.8** is where that gets settled.

## SERVO-2 · The shipped policy saturates the leg actuators — HIGH

**Evidence** — same run as [SERVO-1](#servo-1):

```
joint             peak     p99     rms   % steps over continuous
left_hip_pitch   4.903   4.903   2.735        60.59%
right_hip_pitch  4.903   4.903   2.591        50.91%
left_ankle       4.903   4.903   2.475        40.77%
right_ankle      4.903   4.903   2.417        34.85%
left_knee        4.903   3.734   2.269        62.20%
```

`effort_limit_sim` is **4.903 N.m** since Task M0 fixed PLANT-5. Four leg joints
reach exactly that value **and their p99 is also exactly that value** — so this
is not a transient at heel-strike, it is where the policy lives. Worst-leg RMS
is **2.735 N.m = 174 % of the 1.569 N.m continuous rating**, again on the
undisturbed walk; every contact gate is worse.

**Read this together with the alternative, not alone.** v5d commands
**6.723 N.m = 137 % of datasheet stall** because it was trained against the
stale 8.716 N.m ceiling — it asks for torque the hardware cannot produce at all.
v6d was trained against the true ceiling, so it never exceeds what the servo can
deliver; it simply uses all of it. **The 4.903 N.m ceiling is the corrective,
and v6d is the first policy trained under it.** So this issue argues for a
mechanical or reward change, never for reverting to an older policy.

**Candidate responses**, in the order a retrain should try them: raise
`joint_torque_l2` / add an explicit torque-limit penalty; lower the commanded
velocity envelope; reduce plant mass (the M1 print process is already the
lightest orderable option at 2.729 kg); gear or upgrade the hip-pitch joints.
