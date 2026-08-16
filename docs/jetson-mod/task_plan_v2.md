# Task Plan v2 — Complete Rebuild

**Created 2026-08-11.** This plan supersedes `task_plan.md` for everything it
covers. `task_plan.md` remains the record of Phases 1–5 as originally designed
and is still the only home for the thermal-zoning rationale, the risk register
and the Phase 3/4/5 design intent — read it for background, but where the two
disagree, **this document wins**.

---

## How to use this document

This plan is written to be executed **one task at a time by an AI coding agent
that has no memory of the other tasks**. Each task is therefore self-contained
and repeats context rather than referring back.

Every task has exactly five parts:

| Part | What it gives you |
|---|---|
| **1. Context** | Why the task exists, which files to read first, what breaks if it is skipped, and the repo-specific traps that will bite you |
| **2. Low-level implementation plan** | File-by-file, symbol-by-symbol. Every file created or modified is named |
| **3. Unit tests** | What to assert, or "None applicable" with a reason |
| **4. Smoke test** | The end-to-end check and the specific observable that proves it worked |
| **5. Done when** | An objectively checkable list |

Each task is also marked **AI-agent suitable: YES / NO / PARTIAL**. Take this
seriously — several tasks in Phase S cannot be done by an agent at all, because
they require physically holding the robot.

One exception to the five-part rule: a task whose heading ends in **MOVED** is a
redirect, not a task. It keeps its id so that an implementer told "do S.0" can
find it and be sent to the right place, and it deliberately has no
implementation plan. There is exactly one today (Task S.0 → Phase R).

### Rules that apply to every task

1. **Do not trust `tests/` as a safety net.** 5 of 7 seeded regressions pass the
   existing suite, because most of it is `assert "<literal>" in open(file).read()`
   (`known_issues.md` TEST-1). Tests you add should parse or execute, not grep.
2. **`pytest` from the repo root fails at collection** (TEST-2). Run
   `python3 -m pytest tests/ -q`.
3. **Read `docs/jetson-mod/known_issues.md` before changing anything.** It is
   the single consolidated defect register, every entry mechanically verified.
   `python3 scripts/verify_known_issues.py` re-checks it in seconds with no GPU.
4. **One Isaac Sim / GPU job at a time**, and never mutate tracked source while a
   GPU job is running — a probe that reads a file mid-edit produces a confident
   wrong answer. Build one env per process; building a second after `env.close()`
   in the same Kit process hangs.
5. **Never commit or push unless the owner explicitly asks.**
6. **RL training runs always pass `--video`.** Aggregate metrics have passed a
   crawling policy before; the video caught it when the numbers did not.
7. **Line numbers in this repo go stale, so locate things by heading, symbol or
   `grep`, never by line number.** Some references below still carry a
   `file.py:NNN` form because that is the only practical way to point at a spot
   inside a long function — treat the number as a hint and the surrounding quote
   as the truth. If they disagree, the quote wins. (References into
   `task_plan.md` and `known_issues.md` had their line numbers removed entirely
   for exactly this reason: both files were edited after this plan was drafted
   and every number in them shifted.)
8. **Never quote a number you have not run.** This plan cites measurements; if
   you need a new one, produce it and say what command produced it.
9. **Use the installed skills for geometry and robot descriptions.** If your
   task touches a `.step`, `.stl`, `.3mf`, `.glb`, `.urdf`, `.sdf`, `.srdf` or
   `.dxf` file, or slices a mesh, a skill already owns that workflow — `cad`,
   `urdf`, `cad-viewer`, `step-parts`, `gcode`, `dxf`, `sendcutsend`. Invoke it
   instead of writing your own pipeline, and read
   [CAD and robot-description tooling](#cad-and-robot-description-tooling--read-before-any-geometry-task)
   first: it records the one constraint that decides how each geometry task must
   be done (**there is no STEP for any robot part — the STLs have no B-rep, so
   they cannot be feature-edited**). Both `cad` and `urdf` require handing the
   changed file to `cad-viewer` and reporting the link.

---

## The state you are starting from

Verified by running code on 2026-08-11. These are facts, not assumptions.

| Fact | Value | How to re-check |
|---|---|---|
| Plant mass, authored **and** simulated | **2.657067 kg** | `./isaaclab.sh -p scripts/audit_plant_mass.py --headless` exits 0 |
| Articulation root | `trunk_assembly` (was the massless `base`) | `robot_motors.xml` line ~69 |
| Rigid bodies in the articulation | 21 (was 22) | the audit table |
| Observation / action dims | **59 / 16** | unchanged by the PLANT-1 fix, so checkpoints still load |
| Shipped locomotion policy | `exported_policies/v6d_contact_wrench_ppo/` | **superseded v5d on 2026-08-15** (`locomotion_selection.md`); contains `policy.onnx`, `policy.pt`, `model_5998.pt`, `deployment_contract.json`. The v5d archive was pruned; it remains in git history. |
| Printed part set | **52 pieces, 1571.94 cm³** | `python3 scripts/measure_print_mass.py --process mjf-pa12` |
| Printed mass, FDM PLA 2 perim/15% | 1158 g | `--process fdm-pla --perimeters 2 --infill 15` |
| Printed mass, MJF PA12 (solid) | 1598 g | `--process mjf-pa12` |
| Register status | 28/29 CONFIRMED, 0 refuted | `python3 scripts/verify_known_issues.py` |
| Test suite | 103 passing | `python3 -m pytest tests/ -q` |

### What changed on 2026-08-11, and why it matters to you

**PLANT-1 is fixed.** The MJCF root `base` was a massless frame carrying the
`<freejoint>`. PhysX cannot represent that, so it substituted its own default —
1.000 kg with an isotropic 4.0e-3 tensor — and every policy from v1 to v5d
trained on a robot **37.6 % heavier** than the one being built. `base` is now
merged into `trunk_assembly`; the merge is bit-identical in MuJoCo (mass matrix
and `qacc` deltas both exactly 0.000e+00) because `trunk_assembly` had an
identity transform.

**The consequence you inherit: every published gate number is stale.** v5d's
47.11 % aggregate fall rate, the push-recovery numbers, the v4/v5 comparison
tables — all measured on a 3.657 kg plant that no longer exists. They are kept
as historical record and annotated as such. **Re-gating is Phase R, task 1, and
nothing downstream should be trusted until it is done.**

---

## Phase order, and what gates what

```
  Phase M — mass model, CAD, USD          ← changes the plant
        │
        ├── M1 print-process decision  (HUMAN: this is a spend decision)
        │        └── gates M5 (shelling) and the final mass numbers
        │
  Phase R — re-gate and retrain           ← must follow every plant change
        │
        ├── R1 re-gate v5d on the corrected plant   ← do this FIRST, before M
        │        (it is cheap, and it tells you whether the plant fix alone
        │         broke the policy)
        │
  Phase S — sim-to-real                   ← needs hardware; many tasks are NOT
        │                                   agent-suitable
  Phase V — VLM                           ← needs a working Phase S bring-up
```

**⛔ Read this ordering caveat carefully — it is the one irreversible mistake in
this plan.** R1 (re-gate the existing v5d checkpoint on the corrected plant)
comes *before* Phase M, even though M is listed first.

There are two reasons, and the second is the one that matters:

1. R1 is a few GPU-hours and answers "did fixing the phantom kilogram alone
   break the shipped policy?" — which determines how much of Phase M is urgent.
   Doing M first means you change the plant twice and cannot attribute the
   result to either change.
2. **Task M0b makes R1 permanently impossible.** It changes the observation
   space 59 → 53 and the action space 16 → 14, after which
   `OnPolicyRunner.load()` fails on the shape mismatch for every existing
   checkpoint. There is no way back: the measurement is simply gone.

Reason 1 costs you attribution. **Reason 2 costs you the measurement itself.**
Before starting any Phase M task, confirm R1 has run:
`ls docs/jetson-mod/eval_results_m2657/*.json`.

Every plant change invalidates every trained policy's gate numbers. Batch them:
do **all** of Phase M's plant edits, then retrain **once**. Do not retrain
between M2, M3 and M5.

---

## Do the CAD and the USD need rebuilding?

Answered explicitly, because it was an open question when this plan was written.

**`robot.urdf` — YES, whenever the mass model changes.** As of Task M7 the URDF
is validated, renders, and is checked against the MJCF by
`tests/test_urdf_consistency.py`. It carries its own copy of every link's mass,
CoM and inertia tensor, so **any Phase M task that rewrites the MJCF inertials
must rewrite the URDF's too, in the same change.** The test will fail until you
do — that is the reminder working, not a broken test.

**USD — YES, unconditionally, every time `robot_motors.xml` changes.**
The USD is generated, not authored: `scripts/convert_mjcf_to_usd.py` runs
`MjcfConverter` over `mini_bdx/robots/open_duck_mini_v2/robot_motors.xml`.
It was regenerated on 2026-08-11 for the PLANT-1 fix and must be regenerated
again after any Phase M mass edit. The gate is `scripts/audit_plant_mass.py`
exiting 0.

**CAD (the STL meshes) — CONDITIONALLY, and the condition is the print process.**

| Change | STL edit needed? | Why |
|---|---|---|
| Correcting the mass/inertia numbers | **No** | Mass lives in the MJCF `<inertial>` tags, not in the geometry |
| Modelling the 4 missing 18650 cells | **Yes** | Only 2 of 6 cells exist as geometry. The `holder` mesh measures 20 × 41 × 76 mm and fits **exactly two** cells, so a 6-cell pack needs a **new** holder, not a stretched one |
| Choosing FDM | No further changes | Mass is a print-setting outcome |
| Choosing MJF/SLS | **Yes — shelling** | Powder parts are solid. `body_front` is **86 % of its bounding box** (a near-solid 143 × 110 × 10 mm slab, T_eff 7.52 mm) and `trunk_top` is T_eff 7.99 mm. HP's design guide caps walls at 3 mm because thicker sections accumulate heat and deform. Shelling is required for print *quality*, and it happens to remove mass and cost on the same edit |

Measured wall thickness (`T_eff = 2V/A`) for the shelling candidates, worst first:

```
trunk_top           T_eff 7.99 mm   145 cm3   (27.9% of AABB)
body_front          T_eff 7.52 mm   135 cm3   (86.1% of AABB - a solid slab)
body_middle_bottom  T_eff 4.99 mm   146 cm3   (10.2% of AABB)
head                T_eff 3.38 mm   229 cm3   (marginal)
body_back           T_eff 3.19 mm    95 cm3   (marginal)
body_middle_top     T_eff 2.83 mm    76 cm3   (already under the cap)
```

Reproduce with `trimesh`: `2 * m.volume / m.area` on `print/<part>.stl`.

**Good news, already verified:** every one of the 37 STLs is watertight, none
exceeds 200 mm in any dimension, and **no part has more than one shell**, so
there are no enclosed voids and therefore no MJF powder traps. The set is
printable as drawn by any process.

---

## CAD and robot-description tooling — read before any geometry task

Installed agent skills cover this work. **Use them; do not hand-roll geometry
code when a skill owns the workflow.** Invoke a skill by name (e.g. the `cad`
skill, the `urdf` skill).

| Skill | Owns | Use it in |
|---|---|---|
| `cad` | STEP-first parametric parts/assemblies in build123d Python; `scripts/gen`, `export`, `inspect`, `snapshot` | **M3** (new battery holder), **M5** (re-authored parts) |
| `step-parts` | Catalog of purchasable parts — servos, cells, boards, fasteners | **M3** (18650 cell), **M4** (servo envelopes) |
| `urdf` | `.urdf` authoring + `scripts/validate` + `scripts/snapshot` | **M7** (new — see below) |
| `cad-viewer` | Live review links for `.step`/`.stl`/`.urdf`/`.3mf` | after **every** geometry change |
| `gcode` | Slicing meshes to G-code via real slicer CLIs | **M2**, print-mass measurement |
| `dxf`, `sendcutsend` | 2D cut profiles and service preflight | not needed unless a laser-cut part is added |

### The constraint that decides how every geometry task must be done

**This repo contains no STEP file for any robot part.** The main robot is 48
STLs in `mini_bdx/robots/open_duck_mini_v2/` and 37 in `print/`. The only
`.step` files anywhere are in `print/mods/Justins_Park_Head_Mod/`, a
third-party head mod that is not part of the build. The true parametric source
is an **Onshape document** that this repo does not contain and cannot reach.

Verified consequence — run it yourself and see:

```
$ python3 ~/.claude/skills/cad/scripts/inspect refs print/roll_motor_top.stl --facts
{"ok":false, ... "CAD STEP ref not found for 'print/roll_motor_top.stl'."}
```

An STL is a triangle soup with **no B-rep**: no faces, edges, sketches or
features. So:

- You **cannot** shell, fillet, offset a face, or edit a feature on an existing
  part with the `cad` skill. There is nothing to select.
- You **can** author a *new* part in build123d, validate it, and export STL —
  that is the normal `cad` workflow and it is the right tool for M3.
- To modify an *existing* part you have exactly three options, and the plan must
  say which one each task takes:
  1. **Mesh operation** (`trimesh`, `manifold3d`) — boolean, hollow, decimate.
     Keeps the original surface. No parametric intent. This is what
     `scripts/generate_cad_mods.py` already does, and it works.
  2. **Re-author in CAD** from measurements — full parametric control, and you
     own the result forever. Expensive per part; correct for a part being
     redesigned anyway.
  3. **Go back to Onshape** — the real fix, unavailable to an agent.

**Do not attempt option 2 on a part you are only trying to lighten.** Use a mesh
operation. Re-authoring is justified only where the part is being redesigned
(the 6-cell battery holder) or where its geometry is wrong.

### Mandatory review handoff

Both the `cad` and `urdf` skills require it: after creating or modifying any
`.step`, `.stl`, `.3mf`, `.glb` or `.urdf`, hand the explicit path to the
`cad-viewer` skill and include the returned link. If `cad-viewer` will not
start, say so — do not silently skip it. Snapshot review is **mandatory** after
a visible geometry change, not optional because deterministic checks passed.

> **Setup is a script now — run it, do not follow prose.**
>
> ```bash
> scripts/setup_cad_tools.sh --check     # verify only, exits 1 if not ready
> scripts/setup_cad_tools.sh             # install what is missing, then verify
> ```
>
> It is idempotent, it creates the venv, installs the skill's own
> `requirements.txt`, fetches chromium, and **proves the toolchain by actually
> rendering a part** — a skipped render probe is treated as a failure, not a
> pass. Use it as a gate at the top of any geometry task. `AGENTS.md` carries
> the same rule. The diagnosis below is kept for context.
>
> **CAD Viewer setup — resolved 2026-08-12. `npm` is NOT the way in.**
> The skill documents `npm --prefix scripts/viewer run start`, and that path is
> broken in build 0.4.5: `package.json`'s `start` runs
> `node scripts/start-viewer.mjs`, and **that file does not exist anywhere in
> the skill** — hence `MODULE_NOT_FOUND`. `package.json` also declares **zero
> dependencies**, so `npm install` installs nothing and fixes nothing. (A
> relative `--prefix` also resolves against your *current* directory, so running
> it from `~` fails with a confusing `ENOENT` on `~/scripts/viewer/package.json`
> — a red herring on top of the real problem.)
>
> The viewer is a **Python** backend. Its real dependency is `cadgen==0.4.5`
> (`scripts/viewer/requirements.txt`), and Ubuntu 24.04 refuses a system-wide
> pip install under PEP 668, so it needs a venv. One-time setup:
>
> ```bash
> python3 -m venv ~/.venvs/cad-viewer
> cd ~/.agents/skills/cad-viewer/scripts/viewer
> ~/.venvs/cad-viewer/bin/pip install -r requirements.txt
> ```
>
> Put the venv **outside** the skill directory as shown — `.agents/skills/` is
> replaced when skills update, which would wipe a venv kept inside it.
>
> To run it:
>
> ```bash
> cd ~/.agents/skills/cad-viewer/scripts/viewer
> ~/.venvs/cad-viewer/bin/python3 -m server_py.start_viewer --port 3245
> ```
>
> A URL's **path is the absolute directory** to browse and `?file=` selects one
> artifact inside it, so one server reviews any folder:
> `http://127.0.0.1:3245/<abs-dir>?file=<relative/path.stl>`.

> **Run every `cad` and `urdf` script with that venv's interpreter, not
> `python3`.** Verified 2026-08-12: the system Python has `playwright` but no
> `OCP`, so it renders STL and **fails on STEP** with `No module named 'OCP'`,
> and `inspect refs` on a STEP dies the same way. The venv has all of it
> (`cadgen`, `OCP 7.9.3.1`, `build123d 0.11.1`, `playwright`) — the CAD skill's
> own `requirements.txt` is exactly `cadgen==0.4.5` + `playwright`. Chromium is
> a separate one-time download:
>
> ```bash
> ~/.venvs/cad-viewer/bin/pip install playwright
> ~/.venvs/cad-viewer/bin/python3 -m playwright install chromium
> ```
>
> Then always: `~/.venvs/cad-viewer/bin/python3 ~/.claude/skills/cad/scripts/<tool> ...`
>
> **An agent can see the geometry, not just link to it.** `scripts/snapshot`
> writes a PNG (or an orbit GIF with `--mode orbit`) that the agent then reads
> directly — confirmed working on both `.stl` and `.step`. Use it as the review
> step in M3 and M5 instead of asking the owner to look at a browser.
>
> **The URDF additionally cannot be rendered until Task M7 fixes its mesh
> URIs.** All 240 are `package:///name.stl` with an empty package name, so no
> link mesh resolves and both the viewer and the `urdf` skill's `scripts/snapshot`
> will fail with "No link mesh loaded for robot". Fix M7 first, then review.

## The single most important decision in this plan

**A retrain is already forced. Spend it once, on everything.**

Fixing PLANT-1 changed the plant, so every trained policy's gate numbers are
stale and a retrain is unavoidable. A retrain of the v5d recipe is the most
expensive single operation in this project. There are currently **15 open
plant-fidelity defects**, and each one, fixed separately, forces its own retrain.

| ID | Sev | Defect | Fix cost |
|---|---|---|---|
| PLANT-10 | HIGH | CAD-mod deltas booked at 0.9× solid; trunk is 54–82 g light | constant + re-derive |
| PLANT-3 | MEDIUM | 61.7 % of resets start inside the ground plane | raise spawn height |
| PLANT-4 | MEDIUM | Antennas simulated as STS3250: ~48× torque, **10,336×** armature | actuator group split |
| PLANT-5 | MEDIUM | Torque ceiling 1.78× datasheet stall, pinned to 12.1 V, never randomized | one constant + DR |
| PLANT-6 | MEDIUM | Joint dry friction inactive during motion; BAM viscous term dropped | actuator cfg |
| PLANT-7 | MEDIUM | No latency model of any kind | one event term |
| PLANT-8 | MEDIUM | Yaw command is a heading servo, never an open-loop rate | `rel_heading_envs < 1.0` |
| PLANT-9 | LOW | Push curriculum sized against the spawn height, not the CoM | one constant |
| CFG-1 | MEDIUM | `torque_z_range` silently discarded — dead parameter | use the summing warp variant |
| CFG-2 | MEDIUM | Obstacles placed twice per episode, two independent draws | one predicate |
| CFG-3 | LOW | v5c/v5d gate maintained every step, read by nothing | delete or consume |
| CFG-4 | MEDIUM | `head` half of `ground_contact_penalty` can never fire | retarget to `head_assembly` |
| CFG-5 | LOW | ContactSensor history spans 15 ms, not 60 ms | document or resample |
| REF-1 | MEDIUM | Reference library has no `vy=0` / `wz=0` cell | regenerate library |
| REF-2 | MEDIUM | 33.6 % of reachable knee references are clamped | regenerate library |

Almost every one is a few lines of config. **Fixing all 15 costs barely more
than fixing one, and saves fourteen retrains.**

Two of them are worth more than the rest, because they are the only ones that
change what the *hardware* will receive:

- **PLANT-4 + DEPLOY-3 together.** The antennas are open-loop SG90 PWM servos
  with **no position feedback**, so 4 of the 59 observation dims
  (`joint_pos_rel[22, 23]`, `joint_vel_rel[38, 39]`) cannot be measured on the
  real robot at all — a Phase-4 blocker. Meanwhile the sim drives those two
  joints with `effort_limit_sim=8.716 N·m` (an SG90 stalls at ~0.18) and
  `armature=0.040` against a link inertia of 3.87e-06 — **10,336× too high**.
  It is not simulating an antenna; it is simulating a flywheel.
  **Dropping the antennas from the action and observation spaces fixes both, and
  it can only be done at a retrain.** It shrinks the policy to 14 actions and
  removes the 4 unmeasurable dims. This is the single highest-value change in
  the whole rebuild, and this retrain is the only cheap moment to make it.
- **PLANT-7, latency.** There is no latency model anywhere in the sim. Real
  sensor→inference→actuation delay on a serial-bus servo chain is tens of
  milliseconds against a 20 ms control step. Adding it later means another
  retrain; adding it now costs one event term.

### Task M0 — Assemble the batched plant-fix changeset — ✅ **DONE 2026-08-13**

> **Completed together with M0b, as the plan requires** (M0b supersedes M0
> step 1). Seven issues fixed in one changeset, so one retrain covers them all.
>
> | issue | fix | evidence |
> |---|---|---|
> | **PLANT-3** | `reset_base` gains a +20 mm spawn `z` | below-ground resets **61.7 % → 0.0 %**, deepest now **+4.40 mm** |
> | **PLANT-4** | antennas get their own SG90-scale actuator group | effort 8.716 → **0.18 N·m**, armature 0.040 → **4.0e-06** (was 12,082× the link inertia) |
> | **PLANT-5** | ceiling = datasheet stall, not electrical | 8.716 → **4.903 N·m**; `measure_joint_torque.py` had measured v5d at 6.723 |
> | **PLANT-7** | new `latency.py` observation delay | **0–2 control steps (0–40 ms)**, per-env at reset; critic undelayed |
> | **PLANT-8** | `rel_heading_envs = 0.7` | 30 % of envs train on open-loop `wz`, which the Phase-V VLM needs |
> | **PLANT-9** | CoM height 0.17 → **0.203 m** | 0.17 was the *spawn* height of the root body, not the CoM |
> | **CFG-1** | `add_forces_and_torques` + composer `reset()` | the `set_` kernel assigned torque twice, discarding `torque_z_range` |
> | **CFG-2** | predicate `<= 1` → `== 1` | the obstacle was drawn **twice per episode** |
>
> **CFG-4 is recorded WON'T-FIX**, per the plan's own scope check: the `head`
> half of `ground_contact_penalty` exists only in `DuckContactRewards`
> (v5a/v5b), and that arm is not being revived. Editing it would change nothing
> that runs.
>
> **PLANT-3's check could never have seen its own fix.** It simulated the joint
> scaling and measured the lowest collision vertex, but never read
> `reset_base`'s spawn `z` — so it measured the POSE, not the RESET. It now
> reads the offset from `env_cfg.py`, and only then does it report 0.0 %.
>
> **A stale grep test failed and was rewritten, not re-pinned.**
> `test_robot_cfg_has_correct_actuator_params` asserted the literal
> `effort_limit_sim=8.716`. It now parses the AST, asserts the *constant*, and
> additionally asserts the old literal is absent — TEST-1 in miniature.
>
> Seven checks retired from `scripts/verify_known_issues.py` and marked FIXED in
> the register: PLANT-3/4/5/7, CFG-1/2, DEPLOY-3. Verifier now reads
> **`CONFIRMED 20 / 21`**, no refutations. Suite **202 passed**.
>
> **No training has been run.** The retrain is Task R2, and **M6 must run first**
> — Isaac Lab loads the USD, not the MJCF.


> **⛔ Ordering check before you start.** If your changeset will include
> [Task M0b](#task-m0b--drop-the-antennas-from-the-action-and-observation-spaces)
> (dropping the antennas), then **Task R1 must have run first** — M0b changes
> obs/action to 53/14 and every existing checkpoint stops loading, making the
> re-gate permanently impossible. Verify with
> `ls docs/jetson-mod/eval_results_m2657/*.json`; empty means R1 has not run.

**AI-agent suitable: YES** (except the PLANT-7 latency *value*, which needs a
hardware measurement from Task S.6 — use a literature default and mark it).

**1. Context for the implementing agent**

You are preparing a single changeset that fixes every plant-fidelity defect at
once, so that exactly one retrain is needed. Do not train anything in this task.
Do not fix them one at a time and retrain between fixes — that is the failure
this task exists to prevent.

Read first, in this order:
- `docs/jetson-mod/known_issues.md` — the full text of PLANT-3 through PLANT-10
  and CFG-1 through CFG-5. Each entry carries its own evidence and reproduction
  command. **Do not act on the summary table alone**; the per-issue sections
  contain scope caveats that change what the right fix is (several defects live
  only in the abandoned v5a/v5b arms and must NOT be "fixed" in v5d's chain).
- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — the config chain. v5d is
  `OpenDuckContactWrenchEnvCfg → OpenDuckContactMinimalEnvCfg →
  OpenDuckRobustEnvCfg → OpenDuckRoughEnvCfg`. It does **not** pass through
  `OpenDuckContactEnvCfg`, which is the v5a/v5b arm.
- `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — actuator groups.
- `isaac_lab_env/open_duck_mini_v2/contact_events.py` — the wrench composer.

**Traps specific to this repo:**
- A term's *name* and the *body it binds* routinely disagree. `add_base_mass`
  binds `trunk_assembly`; the `head` contact term binds a 1e-09 kg marker frame
  and not `head_assembly`. Always resolve body names against
  `robot.data.body_names`, never against the term's name.
- Editing `OpenDuckContactEnvCfg` does nothing to the shipped policy. Check
  which class you are in before editing.
- `scripts/verify_known_issues.py` will flip a check to REFUTED the moment you
  fix its issue and the script will exit 1. That is success, not failure — retire
  the check and mark the issue FIXED in the register in the same commit.

**2. Low-level implementation plan**

Work through these in order; each is independent, so a mistake in one does not
corrupt the others.

1. `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — **PLANT-4.** Remove
   `.*_antenna` from the `head` `ImplicitActuatorCfg` group and add a separate
   `antenna` group with SG90-appropriate values: `effort_limit_sim≈0.18`,
   `armature` on the order of the link inertia (3.87e-06), and stiffness/damping
   scaled accordingly. If Task M0b (dropping the antennas entirely) is taken,
   this step is superseded — do M0b instead, not both.
2. `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — **PLANT-5.** The
   `effort_limit_sim=8.716` is `kt·V/R` from BAM at 12.1 V and is 1.78× the
   50 kg·cm datasheet stall. Decide a defensible ceiling and, more importantly,
   add DR over it — the real servo's ceiling moves with battery voltage across a
   3S discharge. Record the reasoning in the commit; do not silently change it.
3. `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — **PLANT-3.** Raise the reset
   spawn height so the nominal pose does not start 61.7 % of episodes with a
   collision vertex below the ground plane. The nominal lowest vertex is
   +3.17 mm at the standing pose; the randomised joint scaling is what pushes it
   under. Re-measure with the PLANT-3 check after changing it.
4. `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — **PLANT-9.** Replace the
   hard-coded 0.17 m "CoM height" in the push-magnitude derivation with the
   measured 0.203 m, or compute it at reset.
5. `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — **PLANT-8.** Set
   `rel_heading_envs` below 1.0 so some envs train on directly sampled yaw-rate
   commands. This is what makes the policy safe for an open-loop consumer such
   as the Phase-V VLM.
6. `isaac_lab_env/open_duck_mini_v2/contact_events.py` — **CFG-1.** The warp
   kernel behind `set_forces_and_torques_at_position` *assigns* torque twice, so
   passing `positions` discards `torque_z_range` entirely. Switch to the summing
   variant (`add_forces_and_torques_at_position`) or drop `positions`.
7. `isaac_lab_env/open_duck_mini_v2/contact_events.py` — **CFG-2.** The
   fresh-episode predicate is `episode_length_buf <= 1`, true at both buf=0 and
   buf=1, so obstacles are placed twice per episode with two independent draws.
   Make it fire exactly once.
8. `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — **CFG-4.** Retarget the `head`
   half of `ground_contact_penalty` to `head_assembly`. Scope check first: this
   term exists only in `DuckContactRewards` (v5a/v5b). If you are not reviving
   that arm, record it as won't-fix instead of editing it.
9. **PLANT-7, latency.** Add a latency/delay event term. Use a literature
   default now and mark the value as provisional; Task S.6 measures the real
   loop and replaces it.
10. **PLANT-10.** Covered by Task M2 — do not duplicate it here.

**3. Unit tests**

Add to `tests/test_isaac_lab_env.py`, but **parse or execute — do not grep**:
- Resolve the actuator groups from `robot_cfg.py` by importing it, and assert
  no antenna joint is in a group whose `effort_limit_sim` exceeds 1.0 N·m.
- Assert `armature` on any antenna group is within 100× of the antenna link
  inertia read from the MJCF (3.87e-06), not 10,000×.
- For CFG-2, assert the fresh-episode predicate cannot be true on two
  consecutive steps — construct the buffer values and call the predicate.

**4. Smoke test**

`python3 scripts/verify_known_issues.py`. Every issue you fixed must flip to
REFUTED. That is the pass condition for this task: the script exiting 1 with
exactly the set of issues you intended to fix, and no others.

**5. Done when**

- [ ] `python3 scripts/verify_known_issues.py` reports REFUTED for every issue in
      the batch and CONFIRMED for every issue you did not touch
- [ ] Each fixed issue is marked FIXED in `docs/jetson-mod/known_issues.md` with
      the evidence, and its check is retired from the verifier
- [ ] `python3 -m pytest tests/ -q` passes, including the new assertions
- [ ] `./isaaclab.sh -p scripts/audit_plant_mass.py --headless` still exits 0
- [ ] No training has been run yet — that is Task R2

### Task M0b — Drop the antennas from the action and observation spaces — ✅ **DONE 2026-08-13**

> **Completed.** R1/R1b had run (10 JSONs in `eval_results_m2657/`), so the
> re-gate this task destroys was already taken.
>
> **Measured off a freshly built environment, not computed:**
>
> ```
> obs    dim: 53
> action dim: 14
> critic dim: 56
> action joints: left_hip_yaw, neck_pitch, right_hip_yaw, left_hip_roll,
>                head_pitch, right_hip_roll, left_hip_pitch, head_yaw,
>                right_hip_pitch, left_knee, head_roll, right_knee,
>                left_ankle, right_ankle          <- 14, no antenna
> stepped 60x with zero actions: OK
> ```
>
> **53, exactly as the plan warned — not 55.** `last_action` is called with
> `action_name=None` so it returns the whole action tensor and shrinks 16 → 14
> by itself: 3 + 3 + 3 + 14 + 14 + 14 + 2.
>
> **The critic needed the same filter and I missed it first time.** `CriticCfg`
> subclasses `PolicyCfg` but is a *separate instance*, so filtering
> `self.observations.policy.*` left the critic at **60**, carrying the four
> antenna dims. Caught by reading the dim off the env rather than trusting the
> edit. It is 56 now, and the critic deliberately gets the **undelayed**
> observation — it is privileged, never runs on hardware, and that is the
> standard asymmetric arrangement.
>
> Filtered **by name** via `^(?!.*antenna).*$`, never by index: Isaac Lab's
> joint order is interleaved and matches neither the MJCF nor the Playground
> order, so an index filter would remove the wrong joints. A test runs the
> actual regex against the actual `duck_init_pos.json` order and asserts 14.
>
> Option **(a)** was taken, as recommended: the antennas stay in the model as
> passive links held at `q_default` by their own actuator group, which is what
> the real robot does when the runtime does not drive them. The plant does not
> move, so `audit_plant_mass.py` is unaffected.
>
> `AGENTS.md`'s observation table and action-space section now read 53 / 14,
> with the old 59-dim layout kept as explicitly historical and marked
> **will not load** against the current config.


> ## ⛔ STOP — this task destroys a measurement you cannot take later
>
> **Task R1 (re-gate the shipped v5d on the corrected plant) MUST have run
> before this task. Check that it has. If it has not, stop and run it first.**
>
> This task changes the observation space 59 → 53 and the action space 16 → 14.
> After it lands, `OnPolicyRunner.load()` fails on the shape mismatch for
> **every existing checkpoint** — v5d and v4_robust included. Re-gating them on
> the PLANT-1-corrected plant then becomes **permanently impossible**, and the
> question "did fixing the phantom kilogram alone break the shipped policy?"
> can never be answered.
>
> R1 costs a few GPU-hours. Losing it costs the only clean before/after
> measurement of the plant fix.
>
> **How to check in one command** — R1's results directory must exist and be
> non-empty:
>
> ```bash
> ls docs/jetson-mod/eval_results_m2657/*.json 2>/dev/null | head
> ```
>
> Empty or missing → **R1 has not run. Do not proceed.**

**AI-agent suitable: YES.** No hardware needed. This is a config + retrain change.

> **This is the single highest-value change in the rebuild.** It closes a HIGH
> Phase-4 blocker (DEPLOY-3) and a MEDIUM plant defect (PLANT-4) at once, it can
> only be done at a retrain, and a retrain is already forced by the PLANT-1 fix.
> If you do nothing else discretionary in Phase M, do this.

**1. Context for the implementing agent**

The robot has two antenna joints (`left_antenna`, `right_antenna`) driven on
hardware by **SG90 micro servos with no position feedback**. Two separate
defects follow from that:

- **DEPLOY-3 (HIGH, Phase-4 blocker).** Those joints contribute 4 of the 59
  observation dims — `joint_pos_rel[22, 23]` and `joint_vel_rel[38, 39]` — and
  **none of the four can be measured on the real robot.** Feeding zeros is
  distribution shift; feeding the commanded angle is also distribution shift.
  There is no correct value to supply, so the obs vector cannot be honestly
  built on hardware while these dims exist.
- **PLANT-4 (MEDIUM).** In simulation those joints are driven with the STS3250
  parameter set: `effort_limit_sim=8.716 N·m` where an SG90 stalls at ~0.18
  (≈48×), and `armature=0.040` against a link inertia of 3.87e-06 — **10,336×
  too high**. The sim is not modelling an antenna, it is modelling a flywheel
  bolted to the head.

Both live in the 16-dim action vector destined for real servos.

Read first:
- `docs/jetson-mod/known_issues.md`, entries DEPLOY-3 and PLANT-4 (full text,
  not the summary rows)
- `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — the `head` actuator group is
  where `.*_antenna` is matched
- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — action and observation terms
- `isaac_lab_env/open_duck_mini_v2/imitation_reward.py` — read this before you
  panic about the gait library (see below)

**The trap you would otherwise hit, already checked for you.** The gait
reference library is keyed to a hardcoded 16-entry `PLAYGROUND_JOINT_ORDER` that
*includes* both antennas, which looks like a blocker. It is not.
`_build_joint_mapping()` only ever maps `LEG_JOINT_NAMES` — **10 leg joints,
verified to contain no antenna** — and it resolves the Isaac side at runtime via
`isaac_joint_names.index(name)`. `PLAYGROUND_JOINT_ORDER` indexes the *reference
library*, which is independent of the robot's joint set. Removing the antennas
from the robot changes the Isaac-side indices, which are looked up by name, and
leaves the reference-library indices untouched. **The imitation reward keeps
working.** Verify this yourself before relying on it:

```bash
python3 -c "
import re; s=open('isaac_lab_env/open_duck_mini_v2/imitation_reward.py').read()
leg=re.search(r'LEG_JOINT_NAMES = \[(.*?)\]', s, re.S).group(1)
print('antenna in tracked set:', 'antenna' in leg)"
# expect: False
```

**Decide the mechanism first.** Two options, and they are not equivalent:

| | (a) Keep joints, exclude from action/obs | (b) Remove the joints from the MJCF |
|---|---|---|
| Action dim | 16 → 14 | 16 → 14 |
| Obs dim | 59 → 53 | 59 → 53 |
| Antennas in sim | still present, unactuated — must be held | gone |
| MJCF / USD change | none | yes, plus USD regen and a new `audit_plant_mass` run |
| Mass change | none | −8.4 g (two 4.216 g links) — changes the plant again |
| Risk | low | higher; touches the model |

**Recommended: (a).** It achieves the deployment goal — the policy neither reads
nor writes the antennas — without moving the plant a second time. The antennas
remain in the model as passive links, which is also what the real robot does
when the runtime simply does not drive them. Choose (b) only if the owner wants
the antennas physically deleted from the design.

**2. Low-level implementation plan** (for option (a))

1. `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` — remove `.*_antenna` from the
   `head` `ImplicitActuatorCfg` `joint_names_expr`. Add a separate `antenna`
   actuator group matching `.*_antenna` with SG90-plausible values
   (`effort_limit_sim` ≈ 0.18, `armature` on the order of 4e-06, stiffness and
   damping scaled to match). The joints must still have *an* actuator or PhysX
   will treat them as free-swinging.
2. `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — restrict the action term's
   `joint_names` to the 14 non-antenna joints. Do **not** filter by index;
   filter by name or by a regex that excludes `.*_antenna`, because the joint
   order is not the MJCF order (see `AGENTS.md` "Joint Orders" — Isaac Lab's
   order is interleaved and differs from both the MJCF and the Playground order).
3. `isaac_lab_env/open_duck_mini_v2/env_cfg.py` — restrict the `joint_pos_rel`
   and `joint_vel_rel` observation terms the same way. **Note the `actions`
   observation term is `isaaclab.envs.mdp.observations:last_action` called with
   `params: {}`, i.e. `action_name=None`, which returns the ENTIRE action tensor
   — so that block shrinks 16 → 14 by itself.** Total observation therefore goes
   **59 → 53** (`3 + 3 + 3 + 14 + 14 + 14 + 2`), and the critic group, which
   adds `base_lin_vel(3)`, goes 62 → 56. Do not write 55; that is the figure you
   get by subtracting only the four antenna joint dims and forgetting
   `last_action`.
4. Hold the antennas at their default: either a constant position target on the
   new actuator group, or `q_default` for those joints. State which you chose in
   a comment; the hardware runtime must do the same thing.
5. `AGENTS.md` — update the observation-layout table (the canonical one; there
   are two further copies that are pointers to it) and the deployment contract
   from 59 to 53 dims, and the action dim from 16 to 14.
6. `docs/jetson-mod/known_issues.md` — mark DEPLOY-3 and PLANT-4 FIXED with
   evidence; retire the PLANT-4 check in `scripts/verify_known_issues.py`.

**3. Unit tests**

In `tests/test_isaac_lab_env.py`, and **parse, do not grep** — this repo's suite
already passes real regressions because it greps:

- Import `robot_cfg.py` and assert no actuator group whose `joint_names_expr`
  matches an antenna has `effort_limit_sim > 1.0`.
- Assert the antenna actuator group's `armature` is within 100× of the antenna
  link inertia read from the MJCF (3.87e-06) — the current value is 10,336×.
- Assert the configured action term resolves to exactly 14 joint names and that
  none contains `"antenna"`.

**4. Smoke test**

Build the training env and read the dimensions back from the running env, not
from the config:

```python
obs, _ = env.reset()
assert obs["policy"].shape[-1] == 53
assert env.action_manager.total_action_dim == 14
```

Then step 60 times with zero actions and confirm the episode does not terminate
immediately and the feet still register contact force — i.e. you changed the
interface, not the physics.

**5. Done when**

- [ ] A freshly built env reports obs **53** and action 14, read from the env
- [ ] `python3 scripts/verify_known_issues.py PLANT-4` reports REFUTED
- [ ] `docs/jetson-mod/known_issues.md` marks DEPLOY-3 and PLANT-4 FIXED, with
      the measured evidence, and the retired check is deleted
- [ ] `./isaaclab.sh -p scripts/audit_plant_mass.py --headless` still exits 0
      (option (a) must not move the mass at all)
- [ ] `python3 -m pytest tests/ -q` passes including the new assertions
- [ ] The observation layout in `AGENTS.md` says 53, and no stale 59 remains:
      `grep -rn "59" AGENTS.md docs/jetson-mod/known_issues.md | grep -i obs`
- [ ] **No training has been run yet** — the retrain is Task R2, and it must
      pick up this change together with every other Phase-M change

---

# Phase M — Rebuild the mass model, CAD and USD

## What actually needs rebuilding, and when

Read this before starting any task in the phase. It is the rule the whole phase hangs on, and three of the six tasks below are shaped by it.

**The STL meshes are the geometry. They do not change unless a part is redesigned.**
Correcting a *density*, a *mass* or an *inertia tensor* touches zero STL bytes — those numbers live in `mini_bdx/robots/open_duck_mini_v2/robot_motors.xml`, `mini_bdx/robots/open_duck_mini_v2/robot.xml`, `mini_bdx/robots/open_duck_mini_v2/robot.urdf`, `scripts/cad_mod_deltas.json` and `tests/fixtures/expected_values.json`. Only two tasks in this phase redesign parts and therefore rewrite STLs:

| Task | Rewrites STLs? | Why |
|---|---|---|
| M1 choose print process | no | a decision record only |
| M2 fix PLANT-10 density | **no** | changes how existing geometry is *weighed*, not the geometry |
| M3 six-cell battery pack | **yes** | a new `holder_6cell.stl`, and probably a deeper hump in `body_back.stl` |
| M4 bottom-up mass/inertia | **no** | changes `<inertial>` blocks only |
| M5 shell thick parts (solid processes only) | **yes** | hollowing `body_front` etc. is a geometry change |
| M6 regenerate USD + gate | no | consumes whatever M2–M5 produced |

### Three repo facts you cannot infer and will get wrong if you assume

These were measured in this repo on 2026-08-11. Do not skip them.

**1. `print/` and `mini_bdx/robots/open_duck_mini_v2/` are two different mesh sets, not two copies of one set.**
There are 37 `.stl` files in `print/` (millimetre scale) and 47 `<mesh>` entries in the MJCF `<asset>` block (metre scale). Only **28 names appear in both**.

- 19 MJCF meshes have **no** `print/` file: `antenna`, `bms`, `bno055`, `board`, `cell`, `dcdc_converter`, `drive_palonier`, `holder`, `jetson_orin_nano`, `left_knee_to_ankle_left_sheet`, `left_knee_to_ankle_right_sheet`, `passive_palonier`, `power_switch`, `roll_bearing`, `sg90`, `usb_c_charger`, `wj-wk00-0122topcabinetcase_95`, `wj-wk00-0123middlecase_56`, `wj-wk00-0124bottomcase_45`.
- 9 `print/` parts are **not** in the MJCF: `bulb`, `flash_light_module`, `flash_reflector_interface`, `knee_to_ankle_left_sheet`, `knee_to_ankle_right_sheet`, `left_eye`, `right_eye`, `speaker_interface`, `speaker_stand`.
- Note in particular that the MJCF calls the shin sheets `left_knee_to_ankle_*_sheet` while `print/` calls them `knee_to_ankle_*_sheet`. **The name mapping between the two sets is not the identity.**

**2. Even where a name appears in both directories, the geometry is not always the same.** Measured volumes:

| part | sim copy (cm³) | print copy (cm³) |
|---|---|---|
| `head` | 217.23 | **229.35** |
| `thermal_partition` | 20.71 | **13.07** |
| `left_roll_to_pitch` / `right_roll_to_pitch` | 29.38 | **29.64** |
| `head_bot_sheet` | 65.89 | 65.91 |
| `body_front`, `body_back`, `body_middle_bottom`, `trunk_bottom`, `trunk_top`, `body_middle_top`, `battery_pack_lid`, `left_cache`, `right_cache`, … | identical | identical |

The four parts `generate_cad_mods.py` owns are guaranteed identical because its `save()` writes both copies from one mesh. `head` and `thermal_partition` are not, and **any generator that `save()`s a sim mesh into `print/` will silently destroy the print geometry.** See M5.

**3. There are 21 `<body>` elements in the MJCF, but only 17 carry a real `<inertial>`.** Four are 1e-9 kg marker frames: `trunk`, `left_foot`, **`head`**, `right_foot`. (`head` is easy to miss because `head_assembly` is a different, real body.) MuJoCo reports `nbody 23` for `scene.xml` — the 21 plus `world` and `floor`.

**The USD is generated, never edited.** `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd` is produced by `scripts/convert_mjcf_to_usd.py` from `robot_motors.xml`. It **must be regenerated whenever `robot_motors.xml` changes, and also whenever any STL that `robot_motors.xml` references changes.** This is not a style preference: Isaac Lab training loads the *USD*, not the MJCF, so a stale USD trains a policy on the old plant and nothing complains at runtime.

Two executing tests already enforce this and are the phase's cheap tripwires (both are `@pytest.mark.phase2`, in `tests/test_usd_conversion.py`):

- `test_usd_not_stale` — recomputes Isaac Lab's asset hash (md5 of `usd/config.yaml` minus its three path keys, plus the `robot_motors.xml` bytes) and compares it to `usd/.asset_hash`.
- `test_usd_meshes_not_stale` — md5s every STL named in the MJCF `<asset>` block against `usd/.mesh_manifest.json`, which `convert_mjcf_to_usd.py::write_mesh_manifest` writes at conversion time. The manifest currently holds 47 entries.

**Dependency order.** M1 → M2 → (M3, M5 in either order) → M4 → M6.
M4 must come after every task that changes geometry or per-part density, because it reads both. **If M5 runs, M2 must be re-run after it** (M5 changes the geometry M2 weighs) — M5's own step list says so; do not treat M2 as done-once. M6 is last and is the phase gate.

**Standing warning for every task below.** `scripts/generate_cad_mods.py` restores `body_front`, `body_middle_bottom`, `trunk_bottom` and `body_back` from git commit `adbc082` at the *start of every run*, then re-applies its cut list. Any hand edit to those four STLs (in `mini_bdx/robots/open_duck_mini_v2/` **or** their mm-scale twins in `print/`) is silently reverted the next time anyone runs that script. Every geometry change to those four parts must be written as a step *inside* `generate_cad_mods.py`.

**Second standing warning.** `pytest` run bare from the repo root **errors at collection** (`experiments/RL/old_test.py` imports `gymnasium`, which the system Python does not have — `known_issues.md` TEST-2). Always name the test files. Run pytest **from the repo root**, because the new tests below `import scripts.measure_print_mass` and that only resolves with the repo root on `sys.path` (there is no `scripts/__init__.py`; it works as a namespace package — verified).

---

### Task M1 — Choose the print process and record it as a machine-readable decision — ✅ **DONE 2026-08-12** (amended same day; quote pending owner)

> ## ⚠ AMENDED after vendor research — the profile below the fold is superseded
>
> **`fdm-asa` stands. `2 perimeters / 15 % infill = 1004.29 g` does not.**
>
> Seven bureaus were checked against their own published pages. **Zero of six
> expose a wall-count field**, and **15 % infill is below every published floor**
> (20 % at Protolabs Network, Craftcloud and PCBWay; Xometry sells tiers, not
> percentages). The booked profile was a hobbyist slicer default that is not
> orderable anywhere.
>
> **Amended to 3 perimeters / 20 % infill = 1163.14 g**, on
> **Protolabs Network**'s published standard — *"All parts are printed with
> 3 outline / perimeter shells or a wall thickness of 1.2 mm"* plus a closed
> infill enumeration. It is the only bureau found that pins **both** axes by
> publication, so the mass is exact **before** the order. Shortlist and quote
> questions: [`print_vendors.md`](print_vendors.md).
>
> **Why not just pin infill:** one extra perimeter is **+130.39 g**, five infill
> points are **+37.66 g** — **82 % of the exposure is the perimeter axis**, the
> one nobody sells. Pinning infill removes ~18 % of the risk.
>
> **Why not revert to `mjf-pa12`:** at 1163.14 g `fdm-asa` is still **434 g**
> under MJF, and ASA is the only commodity FDM material inside the 90–100 °C
> window the Jetson cavity needs. `mjf-pa12` becomes the **fallback** (displacing
> `fdm-abs`) for the case where a vendor states neither axis.
>
> **Corrections to the original M1 write-up:** `body_front` was described as a
> near-solid slab — at 2 perim/15 % it is ~38 % dense; and a "JLC3DP default
> 50 % infill" figure was quoted that is **unsubstantiated** (source Q&A page has
> zero answers). JLC3DP is separately disqualified: 20 of 37 distinct parts fall
> under its published 30 × 30 × 15 mm minimum build size.
>
> **Carried to Task M0/R2:** R2/R2b retrain before any part exists, so the plant
> should carry a **mass DR band of ~2.50–2.76 kg** rather than a point value.
> Recorded as `mass_dr_band_kg` in `print_process.json`.
>
> **`profile_confirmed_with_vendor` is now a SOFT gate**: M2/M4 proceed on the
> published standard and book masses as provisional; the re-run trigger is a
> quote that contradicts it. No geometry is touched, so the re-run is cheap.

> **DECISION: `fdm-asa`, 2 perimeters, 15 % infill. NOT PLA, and not a powder
> process.** Machine-readable record: `scripts/print_process.json`. Reasoning:
> `docs/jetson-mod/print_process_decision.md`. All ten per-process reports are
> committed as `docs/jetson-mod/print_mass_*.txt`.
>
> **The decision turned on a measurement this plan did not have.** New tool
> `scripts/measure_joint_torque.py` rolled out the shipped v5d policy and found
> that on the **current 2.657 kg plant** it already commands **6.723 N·m peak at
> `right_hip_pitch` — 137 % of the STS3250's 4.903 N·m datasheet stall and 428 %
> of its 1.569 N·m continuous rating**, with the knees above continuous for
> ~60 % of steps. That is PLANT-5 made concrete, it is process-independent, and
> no material choice fixes it (Task M0 step 2 plus the retrain does). What the
> process *does* control is how much torque the retrained gait must find:
> `fdm-asa` lands at 170 % of continuous versus `mjf-pa12` at 210 %.
>
> **Measured set mass, 52 pieces / 1571.94 cm³** (TPU sole exempt at 1.22):
>
> | process | set mass (g) | | process | set mass (g) |
> |---|---|---|---|---|
> | `fdm-abs` p2/i15 | 977.15 | | `sls-pa12` | 1536.58 |
> | **`fdm-asa` p2/i15** | **1004.29** | | `mjf-pa12` | 1597.57 |
> | `fdm-abs` p3/i20 | 1131.58 | | `sla-tough` | 1811.03 |
> | `fdm-pla` p2/i15 | 1158.18 | | `mjf-pa12-gb` | 2039.75 |
> | `fdm-asa` p3/i20 | 1163.14 | | | |
> | `fdm-petg` p2/i15 | 1185.35 | | | |
>
> **The plan's `fdm-pla` baseline is thermally wrong and was never a candidate
> once checked.** The trunk parts *are* the cavity enclosing a Jetson whose
> heatsink reaches 55–80 °C; PLA's HDT is ~60 °C. ASA is ~91 °C **and** 154 g
> lighter at the same profile — better on both axes at once.
>
> **The risk this takes on, and the gate that closes it.** FDM re-opens the
> PLANT-10 failure class: `fdm-asa` measures **1004.29 g at 2 perim / 15 % and
> 1163.14 g at 3 / 20 — a 158.85 g swing**, nearly twice PLANT-10's error, and
> many bureaus print to a fixed house profile. `print_process.json` therefore
> carries `profile_confirmed_with_vendor: false`, which is a **hard gate on M2
> and M4**: do not book a gram until the bureau's real profile is written in and
> `measure_print_mass.py` re-run with it. **If the bureau will not commit to a
> profile, the decision reverts to `mjf-pa12`.**
>
> **M5 does not run on this branch** (it is solid-process-only), so
> `max_wall_mm` is `null`.
>
> ### Correction to this plan, recorded because M1 owns `max_wall_mm`
>
> This document states *"HP's design guide caps walls at 3 mm because thicker
> sections accumulate heat and deform."* **Three primary sources were checked on
> 2026-08-12 and none supports it.** 2–3 mm is the recommended **shell wall when
> hollowing**, not a cap on solid wall: Materialise's PA12 (MJF) guidelines
> advise hollowing when wall thickness **exceeds 20 mm** (shell 2–3 mm, ≥2
> escape holes ≥2 mm dia.); Proto3000 states no maximum at all. The thickest
> part here is `trunk_top` at **T_eff 7.99 mm**. So on the solid branch,
> shelling this part set would have been a mass-and-cost optimisation, **not a
> print-quality requirement** — M5's stated premise is weaker than written.
> Recorded so a future solid-branch revisit does not inherit it.
>
> **Open, and it is the owner's:** step 3, the quote. `vendor`, `quote_ref`,
> `quoted_price` and `lead_time` are `null`, and
> `tests/test_print_process.py::test_vendor_and_quote_are_filled_together`
> enforces that they stay consistent. §6 of the decision document lists the
> three things the quote must establish. The process choice does not depend on
> price.
>
> Delivered: `scripts/print_process.json`, `print_process_decision.md`, ten
> `print_mass_*.txt` reports, `scripts/measure_joint_torque.py`,
> `tests/test_print_process.py` (12 tests), and `docs/print_guide.md` corrected
> — PLA superseded, perimeter TODO resolved to 2, and the MJF figure fixed from
> 1,588 g to 1,598 g.

**AI-agent suitable:** PARTIAL. The agent can and should produce every measured row, the whole comparison document, the JSON, and the tests. It must stop before step 3: requesting and accepting a quote commits real money to an outside bureau and is the owner's call. Nothing here needs hardware or a GPU.

**1. Context for the implementing agent**

Every downstream number in this phase is a function of how the parts are made. The same 52-piece set masses **1,158 g** in FDM PLA at 2 perimeters / 15 % infill and **1,598 g** in MJF PA12 — a 440 g swing on a 2.66 kg robot. (Both figures are from `known_issues.md` PLANT-10; the MJF one was re-measured this session as 1,597.57 g and rounds to 1,598.) For powder processes (MJF, SLS) and resin (SLA) the parts are *solid* — infill is not a parameter you get to choose — which is why M5 (shelling) exists at all and only exists on that branch. If this task is skipped, M2 has no density to book, M4 has no per-part density table, and M5 has no trigger condition; all three would go back to guessing, which is exactly the failure PLANT-10 records.

The owner does **not** own a 3D printer. Parts will be bought from an online printing service.

Read first:

- `scripts/measure_print_mass.py` — the whole file (356 lines). In particular the `PROCESS` dict (lines 82–91): it is the closed set of process names, each with a *finished-part* density in g/cm³ and a `kind` of `"fdm"` or `"solid"`. Note the comment that solid-process figures include porosity and that using powder bulk density instead is a ~40 % error. Also note `TPU_PART = "foot_bottom_tpu"` and `TPU_DENSITY = 1.22` (lines 92–94): the TPU sole is **exempt from the process density on every branch**.
- `docs/jetson-mod/known_issues.md`, section `PLANT-10` (search for `<a id="plant-10">`, line 558). It states the measured whole-set totals and the counting rule.
- `docs/print_guide.md` — the parts list and the block quote at the top flagging that perimeter count is unspecified.

**Trap 1 — the set is 52 pieces, not 37 files.** `print_guide.md` carries `x2` and `x4` on nine rows. Summing the 37 distinct STL files undercounts the robot by 13 %. `measure_print_mass.py::part_quantities()` already reads the quantities out of the guide; do not re-implement the count by listing `print/`. Verified this session: `sum(part_quantities().values()) == 52`, total solid volume `1571.94 cm³`.

**Trap 2 — `part_quantities()` has a silent fallback.** Any `print/*.stl` with no row in `print_guide.md` is added at quantity 1 (lines 105–109). That is how `thermal_partition` got counted. It also means **M3 dropping `holder_6cell.stl` into `print/` will change the piece count to 53 without anyone editing the guide.** Whatever count assertion you write here must be updated in the same commit as any new print part, and the guide must be edited too — see the test below.

**Trap 3 — PrusaSlicer is NOT installed on this machine.** Verified: `which prusa-slicer` and `which prusa-slicer-console` both fail. FDM rows therefore **cannot be re-measured here**; `--process fdm-*` exits 2 with an explicit error. Solid processes need no slicer. Your options, in order of preference:
   1. install PrusaSlicer and re-measure; or
   2. cite the recorded PLANT-10 measurement (PrusaSlicer 2.7.2, PLA 1.24 g/cm³, supports/brim/skirt/raft/wipe-tower all off) as the FDM row and label it **"recorded 2026-08-11, not re-measured"**.
   Do **not** substitute an estimate.

**2. Low-level implementation plan**

1. For each candidate process, capture a full report to a file:
   ```bash
   cd ~/Projects/Open_Duck_Mini_Jetson
   python3 scripts/measure_print_mass.py --process mjf-pa12  > docs/jetson-mod/print_mass_mjf-pa12.txt
   python3 scripts/measure_print_mass.py --process sls-pa12  > docs/jetson-mod/print_mass_sls-pa12.txt
   python3 scripts/measure_print_mass.py --process fdm-pla --perimeters 2 --infill 15 \
                                                     > docs/jetson-mod/print_mass_fdm-pla_p2_i15.txt
   ```
   The last one exits 2 with the "ERROR: FDM needs PrusaSlicer" message on this machine (Trap 3). Record that outcome, then apply option 1 or 2 above.
2. Create `docs/jetson-mod/print_process_decision.md` with one row per candidate: process, whole-set mass (the `TOTAL` line of each report), whether infill is a lever, vendor, quoted price, lead time, and a one-line consequence for the robot (mass budget, servo torque headroom, whether M5 is required).
3. **HUMAN STEP:** request quotes, choose, and record the vendor, quote reference and date in that document.
4. Create `scripts/print_process.json` — this file, not a constant in any script, becomes the single source of truth:
   ```json
   {
     "process": "mjf-pa12",
     "kind": "solid",
     "density_g_cm3": 1.01,
     "perimeters": null,
     "infill_pct": null,
     "tpu_part": "foot_bottom_tpu",
     "tpu_density_g_cm3": 1.22,
     "max_wall_mm": null,
     "decided_on": "YYYY-MM-DD",
     "vendor": "<name>",
     "quote_ref": "<ref or URL>"
   }
   ```
   - `kind` and `density_g_cm3` must be copied from `PROCESS[process]`.
   - For an FDM choice, `perimeters` and `infill_pct` are integers and `max_wall_mm` stays `null`.
   - For a solid choice, `max_wall_mm` is the bureau's / process guide's maximum recommended wall thickness in mm, cited in `print_process_decision.md`. **M5 reads this field**; putting it here now saves M5 from retro-fitting the schema.
5. Fix two documentation defects found while grounding this plan (both re-verified this session):
   - `docs/print_guide.md` line 14 states MJF PA12 lands at "~1,588 g". That is the all-PA12 figure (1571.94 cm³ × 1.01 = 1587.66 g); it omits the TPU sole. The correct figure is **1,597.57 g** — 47.20 cm³ of the total is `foot_bottom_tpu` ×2 at 1.22 g/cm³ rather than 1.01, worth +9.91 g. Change it to 1,598 g so it agrees with `known_issues.md`.
   - Resolve the perimeter-count TODO in the same block quote (lines 7–10): either write the chosen perimeter count, or write "N/A — parts are bought from a powder/resin process; there is no infill or perimeter parameter."

**3. Unit tests**

New file `tests/test_print_process.py`, marked `@pytest.mark.phase3`. All assertions execute, none grep:

- `test_process_json_parses_and_is_a_known_process` — `json.load` the file; `import scripts.measure_print_mass as m`; assert `cfg["process"] in m.PROCESS`.
- `test_density_and_kind_match_the_process_table` — assert `abs(cfg["density_g_cm3"] - m.PROCESS[cfg["process"]]["density"]) < 1e-9` and `cfg["kind"] == m.PROCESS[cfg["process"]]["kind"]`. This is the assertion that stops someone typing a bulk powder density.
- `test_tpu_fields_match_the_script_constants` — assert `cfg["tpu_part"] == m.TPU_PART` and `abs(cfg["tpu_density_g_cm3"] - m.TPU_DENSITY) < 1e-9`.
- `test_fdm_fields_are_consistent_with_kind` — if `kind == "fdm"`, assert `perimeters` and `infill_pct` are `int`; else assert both are `None` and `max_wall_mm` is a positive number.
- `test_every_print_stl_has_an_explicit_guide_row` — read `docs/print_guide.md` with the same regex `part_quantities()` uses; assert the set of matched names equals `{basename for print/*.stl}`. **This is the assertion that kills Trap 2's silent fallback** — it fails the moment M3 adds a print part without updating the guide.
- `test_piece_count_matches_the_guide` — `assert sum(m.part_quantities().values()) == EXPECTED_PIECES`, with `EXPECTED_PIECES = 52` as a module constant carrying the comment "bump this in the same commit that adds a row to print_guide.md". Guards the 13 % undercount permanently without becoming a landmine for M3.

**4. Smoke test**

```bash
cd ~/Projects/Open_Duck_Mini_Jetson
python3 scripts/measure_print_mass.py --process "$(python3 -c 'import json;print(json.load(open("scripts/print_process.json"))["process"])')"
echo "exit=$?"
```
Observable: exit code 0, and a `TOTAL` line whose piece column reads `52` and whose volume column reads `1571.94` (before M3/M5 change anything).

**5. Done when**

- [ ] `scripts/print_process.json` exists and `pytest tests/test_print_process.py -v` is all-green (run from the repo root).
- [ ] `docs/jetson-mod/print_process_decision.md` names a vendor, a quoted price and a date.
- [ ] One `docs/jetson-mod/print_mass_<process>.txt` exists per candidate considered; if the FDM one is the exit-2 error message, that file contains that message and the decision document says where the FDM number came from instead.
- [ ] `docs/print_guide.md` says 1,598 g and no longer carries an unresolved perimeter TODO.

---

### Task M2 — Replace `generate_cad_mods.py`'s assumed density with the measured per-part mass (PLANT-10) — ✅ **DONE 2026-08-12**

> **Completed. PLANT-10 is FIXED.** The assumed density is not replaced by a
> better constant — it is **gone**. `generate_cad_mods.py` no longer derives mass
> from volume at all.
>
> **Measured at the chosen process** (`fdm-asa`, 3 perim / 20 % infill, from
> `scripts/print_process.json`):
>
> | part | baseline g | current g | delta g |
> |---|---|---|---|
> | `trunk_bottom` | 36.99 | 13.54 | **−23.45** |
> | `body_middle_bottom` | 93.30 | 91.58 | −1.72 |
> | `body_front` | 68.06 | 68.17 | **+0.11** |
> | `body_back` | 74.63 | 85.74 | **+11.11** |
> | **net** | | | **−13.95** vs the booked **−88.48** |
>
> **`trunk_assembly` was 74.53 g light** — inside the 54–82 g band this plan
> predicted. `compute_trunk_inertial.py` now emits **1.164076 kg**, against
> 1.089544 kg. Predicted before running, as the plan demands:
> `1.089544 + (−0.01395 + 0.0884824) = 1.1640764`. Matched to the digit.
>
> **Two of the four deltas are POSITIVE**, which is the whole argument made
> visible: cutting the inlet slots in `body_front` removes 7.2 cm³ and *adds*
> 0.11 g, because the new slot walls print near-solid. No scalar density times a
> volume difference can change sign.
>
> Measured effective densities span **0.452** (`trunk_top`) to **1.102 g/cm³**
> (`knee_to_ankle_right_sheet`) — a factor of 2.4 against the single 1.116 that
> stood in for all of them. Volume-weighted mean **0.7399 g/cm³ (69.2 % of
> filament)**.
>
> **A measurement artefact worth carrying to M4:** five thin parts measure
> *above* the 1.07 g/cm³ filament density, worst 1.1021. That is real slicer
> behaviour — at 3 perimeters a ~2 mm sheet asks ~2.7 mm of wall per side, so
> walls pack solid and overlap, and PrusaSlicer reports mass from extruded
> filament length. It is not a leak of a solid figure; the aggregate test proves
> that.
>
> Delivered: `--emit-table` on `measure_print_mass.py` (defaults now read from
> `print_process.json`), `BOOKED_CAD_DELTA_G` replaced by `booked_cad_delta_g()`
> which reads `cad_mod_deltas.json` so `--cad-delta` now self-reports
> `error in the model −0.00`, `whole_part_terms()` replacing
> `signed_delta_terms()`, `scripts/part_mass_table.json`, and
> `tests/test_cad_mod_deltas.py` (11 tests).
>
> **Verified:** STL bytes unmoved (`git status --porcelain` empty for both mesh
> dirs, after two consecutive runs), `cad_mod_deltas.json` byte-identical across
> re-runs (md5 `2d37399c…`), suite **164 passed / 5 skipped / 1 xfailed**.
>
> **The model files are deliberately UNTOUCHED** — `robot_motors.xml`,
> `robot.xml`, `robot.urdf` and `expected_values.json` still declare 1.089544 kg
> and a 2.657067 kg robot. **Task M4 rewrites every inertial from one composer**
> and consumes this output; writing them twice would guarantee they disagree.
> Applying the M2 delta alone would put the robot at **2.731599 kg**.
>
> Consequently `tests/test_cad_dimensions.py::test_inertial_matches_generator`
> now carries a **`strict=True` xfail** for `trunk_assembly`, naming M4 and the
> exact 0.074532 kg gap. Strict means it becomes a hard FAILURE the moment M4
> lands, forcing the marker's removal. **Do not close it by loosening a
> tolerance.**
>
> **Provisional-profile caveat.** These masses sit on Protolabs Network's
> *published* profile, not a vendor-confirmed one
> (`profile_confirmed_with_vendor: false`). One extra perimeter is ~130 g across
> the set. If a quote contradicts the published standard, re-run `--emit-table`,
> `generate_cad_mods.py` and M4 — no geometry is touched, so it is cheap.

**AI-agent suitable:** YES, with one caveat: if M1 chose an FDM process, every step here needs PrusaSlicer on `PATH`, which this machine does not have (M1 Trap 3). On a solid process no slicer is needed and the task is fully local.

> **Ordering note.** This task moves the plant mass, which invalidates every
> existing policy's gate numbers. It does **not** break checkpoint loading, so
> it is recoverable — unlike [Task M0b](#task-m0b--drop-the-antennas-from-the-action-and-observation-spaces).
> But running it before [Task R1](#task-r1--re-gate-the-shipped-v5d-against-the-corrected-plant)
> means the plant has changed twice and you can no longer attribute a result to
> the PLANT-1 fix alone. Confirm R1 has run:
> `ls docs/jetson-mod/eval_results_m2657/*.json`


**1. Context for the implementing agent**

`scripts/generate_cad_mods.py` books the Part-2 CAD volume deltas as mass using one constant, `PLA_EFFECTIVE_DENSITY = 1116.0` kg/m³ (line 70). At the FDM profile `print_guide.md` documents, slicing the baseline geometry against the current geometry measures the true delta as **−6.60 g**, not the booked **−88.48 g**. That error propagates into `scripts/cad_mod_deltas.json` → `scripts/compute_trunk_inertial.py` → the `<inertial>` block of `trunk_assembly` in all three model files → the USD → every trained policy. If this is skipped, M4's rebuild inherits a wrong trunk delta and the phase gate passes on a wrong number.

**The size of the error depends on the chosen process.** Re-measured this session against `adbc082`:

| part | baseline cm³ | current cm³ | delta at MJF PA12 (1.01 g/cm³) |
|---|---|---|---|
| `body_front` | 142.64 | 135.44 | −7.27 g |
| `body_middle_bottom` | 147.49 | 145.80 | −1.71 g |
| `trunk_bottom` | 82.56 | 22.24 | −60.93 g |
| `body_back` | 105.53 | 95.45 | −10.18 g |
| **net** | | | **−80.09 g** vs booked −88.48 g → error **+8.39 g** |

At the FDM profile the same net is −6.60 g → error **+81.88 g**. **Do not carry a single expected trunk mass into this task.** Compute the expectation from your own measured table.

Read first, all in full:

- `scripts/generate_cad_mods.py` (257 lines) — the `PLA_EFFECTIVE_DENSITY` block (lines **46–70**, which already documents that the constant is wrong and *why it was deliberately not fixed yet*), `signed_delta_terms()` (lines 113–141), `save()` (lines 104–110), and `main()` (lines 194–253).
- `scripts/measure_print_mass.py` — `cad_delta()` (lines 257–306), `slice_mass_g()` (125–161), `solid_volume_cm3()` (113–116), and `CAD_MOD_PARTS` / `BOOKED_CAD_DELTA_G` (lines 76–77).
- `scripts/compute_trunk_inertial.py::_load_shell_deltas` (lines 133–145, wired in at line 148) — the consumer of `cad_mod_deltas.json`. Note that it negates the tensor whenever `mass < 0`.

Depends on: **M1** (`scripts/print_process.json` must exist).

**Trap 1 — the obvious fix is wrong on FDM.** Do not simply replace `1116.0` with a different single number *if the chosen process is FDM*. One density is applied to both sides of each diff, and the two sides do not print at the same density: material *removed* came from bulky interiors that are mostly infill (measured 0.28× solid on `trunk_bottom`), while material *added* is thin vents and bosses that print nearly solid. The signs even disagree — at 2 perimeters / 15 % infill, cutting the inlet slots in `body_front` *increases* its sliced mass by +1.41 g, because the new slot walls add perimeters and solid skins that outweigh the infill removed. No single density can reproduce a sign flip. (On a solid process there is no sign flip — `body_front` loses 7.27 g — but the whole-part scheme below is still the right shape, because it is what M5 and M4 consume.)

**Trap 2 — two scales.** The `print/` copies are millimetre-scale, the `mini_bdx/robots/.../` copies are metre-scale. `save()` writes both. `measure_print_mass.py` slices/measures the `print/` copies. Never mix the two when computing volumes. For these four parts the two copies are byte-equivalent geometry (verified), so either is a valid source of *shape* — but the units differ by 1000×.

**Trap 3 — `--cad-delta` defaults to `fdm-pla`.** `main()` sets `--process` default `"fdm-pla"` (line 335). Every `--cad-delta` invocation in this task must pass `--process` explicitly or it will demand a slicer.

**Trap 4 — `density_kg_m3` in `cad_mod_deltas.json` is read by nothing.** Verified by grep: `_load_shell_deltas` reads only `data["terms"]`. Replacing that key is safe. Also verified: `scripts/verify_known_issues.py` has **no** PLANT-10 check, so nothing there breaks.

**2. Low-level implementation plan**

1. **`scripts/measure_print_mass.py`** — add an exporter so the measurement becomes a data file instead of stdout that someone has to retype.
   - Add `--emit-table PATH`. It writes `scripts/part_mass_table.json`:
     ```json
     {
       "process": "mjf-pa12",
       "generated_on": "YYYY-MM-DD",
       "parts": {
         "body_front": {"volume_cm3": 135.44, "mass_g": 136.79, "how": "solid",
                        "effective_density_g_cm3": 1.010},
         "...": {}
       },
       "baseline": {
         "body_front": {"volume_cm3": 142.64, "mass_g": 144.06, "effective_density_g_cm3": 1.010}
       }
     }
     ```
     The `parts` block is every piece in `part_quantities()` at current geometry; the `baseline` block is the four `CAD_MOD_PARTS` at commit `adbc082`, measured through the same code path `cad_delta()` already uses. `effective_density_g_cm3 = mass_g / volume_cm3`.
     **`foot_bottom_tpu` is a special case on every branch** — `mass_for_part()` already substitutes `TPU_DENSITY = 1.22`, so its effective density will not equal the process density. Record it faithfully; the tests below must exempt it.
   - Make `--emit-table` read defaults (`--process`, `--perimeters`, `--infill`) from `scripts/print_process.json` when they are not given on the command line.
   - Change `BOOKED_CAD_DELTA_G` (line 77) from the hard-coded `-88.48` to a function that reads `sum(t["mass"] for t in json.load(open(cad_mod_deltas.json))["terms"]) * 1000`, falling back to `-88.48` with a printed warning if the file is missing. Update the printed label on line 302 accordingly (it currently says `generate_cad_mods.py PLA_EFFECTIVE_DENSITY=1116`, which will be false). Without this change the `error in the model` line keeps comparing against a constant this task deletes.

2. **`scripts/generate_cad_mods.py`** — stop deriving mass from an assumed density; book the measured mass of the whole part before and after.
   - Delete the `PLA_EFFECTIVE_DENSITY` constant and its comment block (lines 46–70). Replace with `PART_MASS_TABLE = os.path.join(REPO, "scripts", "part_mass_table.json")` and a `load_part_masses()` helper that raises a clear error if the file is missing: `"run python3 scripts/measure_print_mass.py --emit-table scripts/part_mass_table.json first"`.
   - Replace `signed_delta_terms(name, before, after)` with `whole_part_terms(name, before, after, table)`, which emits **two** terms per part instead of two diff terms:
     - `f"{name}_baseline"`: `mass = -table["baseline"][name]["mass_g"]/1000`, `pos` = centroid of `before` in body frame (`before.center_mass + OFF`), `tensor` = tensor of `before` at uniform density `baseline.effective_density_g_cm3 * 1000`.
     - `f"{name}_current"`: `mass = +table["parts"][name]["mass_g"]/1000`, `pos` = centroid of `after` in body frame, `tensor` = tensor of `after` at uniform density `parts.effective_density_g_cm3 * 1000`.
     Keep the existing `volume_cm3` field and the existing sign convention: `_load_shell_deltas` negates the tensor whenever `mass < 0`, so **emit both tensors positive** and let the consumer negate. A pre-negated tensor would be double-negated.
     Why whole-part instead of diff: the net mass is then the measured delta by construction, the two tensors cancel exactly over the unchanged regions of the part, and the FDM sign-flip problem disappears because nothing is derived from the diff any more.
   - Keep the printed per-part line, but print the measured masses and the net so a human reading the run log sees the new number. Format:
     `f"  {name}: {v0:.2f} -> {v1:.2f} cm^3, {m0:.2f} -> {m1:.2f} g ({m1-m0:+.2f} g)"`.
   - In the `cad_mod_deltas.json` payload (line 246), replace `density_kg_m3=PLA_EFFECTIVE_DENSITY` with `process=<from print_process.json>` and `part_mass_table_generated_on=<from the table>`, so a future reader can tell which process the file describes.

3. **`scripts/compute_trunk_inertial.py`** — no code change is required if step 2 preserves the term schema (`name`, `mass`, `pos`, `tensor`, `volume_cm3`). Update the comment block at lines 124–131 to say the terms are now measured whole-part replacements, not density-assumed diffs, and point at `scripts/part_mass_table.json`.

4. Run, in this order, from the repo root:
   ```bash
   python3 scripts/measure_print_mass.py --emit-table scripts/part_mass_table.json
   python3 scripts/generate_cad_mods.py          # rewrites cad_mod_deltas.json; STL bytes unchanged
   python3 scripts/compute_trunk_inertial.py     # prints the new trunk/head <inertial> lines
   ```
   `generate_cad_mods.py` re-exports the four STLs from the same baseline with the same cuts, so `git status` should show the STLs unchanged (byte-identical) and only `cad_mod_deltas.json` modified. **If an STL shows as modified, stop** — something else changed and the USD mesh manifest will now be stale for a reason unrelated to this task. (Note: the files are *rewritten* every run, so their mtimes always change; only the bytes are expected to be stable. Judge by `git status`, never by mtime.)

5. Do **not** hand-write the new `<inertial>` values into the model files here — M4 rewrites every body's inertial from one composer and will consume this output. If M4 is being deferred, then and only then paste `compute_trunk_inertial.py`'s emitted MJCF line into `robot_motors.xml` line 71, the URDF block into the matching `<inertial>` in `robot.urdf`, the MJCF line into `robot.xml`, update `tests/fixtures/expected_values.json` (all seven entries), and run M6.

6. Update `docs/jetson-mod/known_issues.md` PLANT-10: append a `**FIXED <date>**` subsection with the before/after numbers, the same way PLANT-1's entry is written (see line 285, `### FIXED 2026-08-11 — option 1`). Update `AGENTS.md` line 202, the "Mass Changes from Modification" row for `trunk_assembly` (the `−88.5` term and its long parenthetical). Update `docs/jetson-mod/component_layout_v2.md` line 137's "Part-2 status" paragraph, which currently states "the CAD cuts removed ~88.5 g of PLA" and quotes `printed-PLA effective density 1.116 g/cm³`.

**3. Unit tests**

New file `tests/test_cad_mod_deltas.py`, `@pytest.mark.phase3`:

- `test_deltas_json_matches_the_measured_table` — load `scripts/cad_mod_deltas.json` and `scripts/part_mass_table.json`; for each of the four `CAD_MOD_PARTS`, assert the sum of that part's term masses equals `(parts[p].mass_g - baseline[p].mass_g)/1000` to within 1e-6 kg. This is the assertion PLANT-10 was missing: it makes the booked delta *provably* the measured delta.
- `test_no_assumed_density_constant_remains` — `import scripts.generate_cad_mods as g; assert not hasattr(g, "PLA_EFFECTIVE_DENSITY")`. An attribute check, not a text grep, so it cannot be defeated by moving the constant into a string. (Importing the module is safe — `main()` is guarded by `if __name__ == "__main__"`.)
- `test_delta_terms_are_physically_valid` — for every term with positive mass, build the 3×3 from `tensor` and assert it is positive semi-definite and satisfies the triangle inequality on its eigenvalues. Do **not** apply this to negative-mass terms; they are bookkeeping subtractions, not bodies.
- `test_effective_density_matches_the_process` — for every entry in `parts` and `baseline` **except `foot_bottom_tpu`**: if the chosen process kind is `"solid"`, assert `effective_density_g_cm3` equals the process density to 1e-9; if `"fdm"`, assert it is strictly less than the filament density. Assert `foot_bottom_tpu`'s effective density against `TPU_DENSITY` on a solid process. Catches a bulk-vs-finished density mix-up in either direction.

**4. Smoke test**

```bash
cd ~/Projects/Open_Duck_Mini_Jetson
P=$(python3 -c 'import json;print(json.load(open("scripts/print_process.json"))["process"])')
python3 scripts/measure_print_mass.py --cad-delta --process "$P"
python3 scripts/compute_trunk_inertial.py | head -12
git status --porcelain -- 'mini_bdx/robots/open_duck_mini_v2/*.stl' 'print/*.stl'
```
Observables:
- `--cad-delta` prints an `error in the model` line of `+0.00` — it now compares the measurement against `cad_mod_deltas.json`'s own total (step 1's change to `BOOKED_CAD_DELTA_G`).
- `compute_trunk_inertial.py` prints a `trunk_assembly` mass equal to `1.089544 kg + (measured_net_delta − (−0.08848) kg)`. **Compute that number from your own table before running and compare** — it is ≈1.098 kg on MJF PA12 and ≈1.171 kg on FDM PLA at 2 perim/15 %. Do not accept a number you did not predict.
- The `git status` line prints **nothing** (no STL bytes moved).

**5. Done when**

- [ ] `PLA_EFFECTIVE_DENSITY` survives only as history. `grep -rn PLA_EFFECTIVE_DENSITY . --exclude-dir=.git --exclude-dir=__pycache__` returns hits **only** in `docs/jetson-mod/known_issues.md` (lines ~564, ~629) and, if you left it, the historical note in the `scripts/measure_print_mass.py` module docstring. It must not appear in `scripts/generate_cad_mods.py`, in `scripts/cad_mod_deltas.json`, or in any executable line of `measure_print_mass.py`.
- [ ] `scripts/part_mass_table.json` exists and its `process` equals `scripts/print_process.json`'s `process`.
- [ ] `pytest tests/test_cad_mod_deltas.py -v` all-green.
- [ ] Re-running `generate_cad_mods.py` twice in a row leaves `git status --porcelain` empty for the STLs and produces a byte-identical `cad_mod_deltas.json` (idempotence preserved).
- [ ] `known_issues.md` PLANT-10, `AGENTS.md` line 202 and `component_layout_v2.md` all quote the same new delta.

---

### Task M3 — Model the four missing 18650 cells as real geometry — ✅ **DONE 2026-08-12**

> **Completed.** All six 18650s are real geometry and the pack is **proved** to
> fit, not asserted to.
>
> **The bore was measured, ending the repo's two-answer disagreement.**
> `scripts/measure_battery_bay.py` (new) reports the pre-M3 bore as
> **42 (x) × 41 (y) × 69 (z) mm** — this document gave 28.4 mm in one place and
> implied 26.4 mm in another. Every layout the plan lists (3×2, 2×3, 6×1, 1×6)
> failed against it, best case **2384 mm³** of interference.
>
> **The hump was DEEPENED, not widened, and that reversed my first choice.**
> Widening looked cheaper until the measurement showed `bms` (y −28.5…−24.9)
> and `usb_c_charger` (y −30.9…−25.7) hugging the old bore's −y wall, with no
> obvious home elsewhere in the trunk. Dodging them asymmetrically would offset
> the 270 g pack +6.6 mm in y ≈ **0.65 mm lateral CoM shift**; deepening keeps
> the pack 2 wide, clears both untouched, stays laterally symmetric, and costs
> ~**1.6 mm aft CoM shift** plus 23 mm of rear protrusion. Aft is the safer
> error — sagittal shift is along the walking direction the gait manages, while
> lateral shift biases frontal-plane balance, the axis a biped actually falls on.
>
> | | before | after |
> |---|---|---|
> | bore x | −167 … −125 (42 mm) | **−190 … −123 (67 mm)** |
> | bore y | ±20.5 (41 mm) | **±22.0 (44 mm)** |
> | outer x | −170 mm | **−193 mm** |
> | outer y | ±24.0 | unchanged, so `bms` keeps its clearance |
>
> **Result: `measure_battery_bay.py --from-mjcf` exits 0 with WORST = 0.00 mm³**
> — better than the shipped 2-cell layout, which fouls `battery_pack_lid` by
> 384.69 mm³ and whose `holder` fouls `body_back` by 513.67 mm³. Those
> pre-existing interferences are recorded in
> `docs/jetson-mod/battery_bay_before.txt`.
>
> Pack: 3 (x) × 2 (y) at 20.6 mm pitch, centred (−156.5, 0, +32.5) mm.
> New `holder_6cell` — 62.8 × 42.2 × 22 mm, Ø18.6 bores, 2.0 mm webs,
> watertight, single-component, binary STL, byte-identical on re-run.
>
> **Two traps I hit that the plan did not name:**
> 1. **Sweeping cells alone found a 6 mm-wide "clear" band the 62.8 mm tray did
>    not share.** The tray must be in the sweep. The final bore carries 4.2 mm
>    of slack because a 1 mm window is a coincidence, not a clearance.
> 2. **Sizing the tray as `ROWS × PITCH + 2 × WALL`** adds a full pitch of empty
>    material at each end and gave 65.8 × 45.2 mm, which did not fit. It is
>    sized to the cells: `(ROWS−1) × PITCH + BORE + 2 × WALL`.
>
> **Deviation from the plan, deliberate:** the old 2-cell `holder` **geom** is
> removed rather than "retained alongside" the new tray. A robot has one battery
> tray; keeping both adds a phantom 7.5 cm³ part and double-books it. The
> `holder` mesh asset and `holder.stl` both remain for the 2-cell configuration.
> So the unique-instance count is **133 − 3 + 7 = 137**, not the 140 the plan
> predicted.
>
> **Mass moved twice and both moves are booked.** The deepened hump adds 25.44 g
> of `body_back` shell, so the Part-2 CAD delta went from M2's **−13.95 g** to
> **+0.38 g**, and the trunk from 1.164076 to **1.178406 kg**. The set is now
> **53 pieces / 1598.61 cm³ / 1199.50 g** (was 52 / 1571.94 / 1163.14):
> `part_mass_table.json`, `cad_mod_deltas.json`, `print_process.json`,
> `print_guide.md` and `EXPECTED_PIECES` were all re-run or bumped in this
> commit.
>
> **`cells_extra_4x45g` is kept with a `TODO(M4)`**, per the plan's branch for
> "M4 has not landed": its position and tensor are corrected from the phantom
> (−0.145, 0, 0.0306) / 38×38×65 mm to the measured (−0.1565, 0, 0.0325) /
> 59.2×38.6×65 mm. `tests/test_battery_pack.py` asserts the marker is present
> while the lump is, so M4 cannot silently double-book 180 g.
>
> **USD is now stale, by design.** Both `test_usd_not_stale` and
> `test_usd_meshes_not_stale` carry `strict=True` xfails naming **Task M6**,
> which is the Phase-M gate that regenerates it once at the end. **Nothing may
> train before M6** — Isaac Lab loads the USD, not the MJCF.
>
> Suite: **174 passed, 5 skipped, 3 xfailed**.
>
> **Still owner-facing** (the plan's PARTIAL): confirming a real 6-cell pack
> plus BMS and wiring physically goes in. Cell diameter varies 18.3–18.6 mm with
> the wrap (the bore is cut for the max) and **the wiring loom is not modelled
> at all**.

**AI-agent suitable:** PARTIAL. The measurement, the new holder mesh, the interference proof and the MJCF/URDF edits are all scriptable and the agent should do all of them. What needs a human: confirming that a real 6-cell pack plus BMS plus wiring physically goes into the resulting bay (cell diameter varies 18.3–18.6 mm with the wrap, and the wiring loom is not modelled at all), and approving any hump enlargement, which changes the robot's silhouette and its printed cost.

> **Ordering note.** This task moves the plant, which invalidates every existing
> policy's gate numbers. It does **not** break checkpoint loading, so it is
> recoverable — unlike [Task M0b](#task-m0b--drop-the-antennas-from-the-action-and-observation-spaces).
> But running it before [Task R1](#task-r1--re-gate-the-shipped-v5d-against-the-corrected-plant)
> means the plant has changed twice and you can no longer attribute a result to
> the PLANT-1 fix alone. Confirm R1 has run:
> `ls docs/jetson-mod/eval_results_m2657/*.json`

**Skills to use — this task creates NEW geometry, which is the one case the CAD
skill is exactly right for.**

| Step | Skill | Why |
|---|---|---|
| Find an 18650 model | `step-parts` | Search the catalog **before** modelling a placeholder cylinder. The skill's own rule: search first, record the miss, only then use a documented envelope. |
| Author the 6-cell holder | `cad` | Write `battery_holder_6cell.step.py` with `gen_step()` in build123d. Parametric: cell diameter, count, wall, pitch. Keep the `.step.py` and its `.step` in the same directory with the same basename. |
| Export the printable mesh | `cad` | `scripts/export` → STL. STL is a **secondary** artifact here; the STEP is the source of truth and must be committed with it. |
| Validate | `cad` | `scripts/inspect refs <step> --facts --planes --positioning`, then `scripts/inspect validate <step>`. Note `refs --facts` only proves refs resolve — an open shell passes it. Run `validate` too. |
| Review | `cad` + `cad-viewer` | `scripts/snapshot` is **mandatory** after visible geometry change; then hand the path to `cad-viewer` and include the link. |

**Do not** try to edit the existing `holder.stl` with the `cad` skill. It is a
mesh with no B-rep and `inspect refs` will refuse it (verified). The holder is
being redesigned from 2 cells to 6, so authoring it fresh in build123d is both
the correct option and the cheaper one — and it leaves you a parametric source
the project has never had for any part.

**Cell dimensions to model, measured from the existing mesh:** the `cell` mesh
is 18.00 × 18.00 × 65.00 mm with its axis of revolution on **Z**, and the two
existing instances sit at trunk-frame `(-129.6, ±10.5, 32.5)` mm. A real 18650
is 18.3–18.6 mm across the wrap, so model the *pocket* with clearance, not the
nominal 18.0. The current `holder` mesh spans 20 × 41 × 76 mm — it fits exactly
two cells, which is why it must be replaced rather than stretched.

**1. Context for the implementing agent**

The robot is specified as 6× 18650 in 3S2P, and the mass is already booked: `scripts/compute_trunk_inertial.py::TRUNK_COMPONENTS` carries, at line 107,
```python
("cells_extra_4x45g", 0.180, [-0.145, 0.0, 0.0306], box_tensor(0.180, 0.038, 0.038, 0.065)),
```
But `robot_motors.xml` contains only **two** `cell` geom poses. So 180 g sits in the trunk as a lumped box at a position no geometry occupies. Every interference test, every clearance number and every future CAD change is being computed against a bay that is four cells emptier than the robot will be. If this is skipped, M4 composes the trunk from geometry that is missing 180 g of its densest payload, and the hardware build discovers the problem with parts already bought.

Read first:

- `mini_bdx/robots/open_duck_mini_v2/robot_motors.xml` **lines 103–108** — the two `cell` geom poses and the one `holder` geom pose, each present **twice** (a collision geom and a visual twin with `contype="0" conaffinity="0"` at the identical pose). Verified: those exact six lines.
- The mirror block in `mini_bdx/robots/open_duck_mini_v2/robot.xml` **lines 97–99** (untwinned — `robot.xml` has one geom per pose), and in `robot.urdf` **lines 219–258** (`<visual>`/`<collision>` pairs for `cell`, `holder`, `cell`).
- `docs/jetson-mod/component_layout_v2.md`, the "Battery pack 6× 18650 (3S2P) + BMS" row of the Layout v2.1 table, and cut-list item 4. These already record the measured problem.
- `scripts/generate_cad_mods.py` section 4 (`body_back` hump extension, lines 225–231) and `check_partition_fit()` (lines 176–191) — the latter is the exact-boolean fit-proof pattern to copy.
- `tests/test_cad_dimensions.py` — the `NESTING_WHITELIST` at lines 22–41, and `_trunk_geoms()` at lines 82–96.

Depends on: M1 (for the holder's print density) and, if the hump has to move, M2 (so the new geometry is weighed correctly).

**Measured facts to build on** (all re-verified this session; reproduce them, do not take them on faith):

```
cell.stl    18.00 × 18.00 × 65.00 mm, 16.43 cm³   — a bare 18650; long axis is the mesh's local z
holder.stl  76.00 × 41.00 × 20.00 mm,  7.49 cm³   — a 2-cell tray
body_back   56.00 × 110.00 × 125.00 mm, 95.45 cm³ — outer bounding box, post-hump-extension
```

At their MJCF poses, in the trunk body frame (mm):

```
cell  #1  x[-138.6, -120.6]  y[-19.5,  -1.5]  z[  0.0, 65.0]
cell  #2  x[-138.6, -120.6]  y[  1.5,  19.5]  z[  0.0, 65.0]
holder    x[-139.6, -119.6]  y[-20.5,  20.5]  z[ -5.5, 70.5]
```

So both cells stand **vertically** (65 mm along body z), side by side in y, and the tray's 41 mm width takes exactly two 18 mm bores. Note that despite the very different `pos` z values in the XML (`0.0650134` vs `1.34322e-05`) the two cells occupy the *same* z band — the quats differ, so you cannot read placement off `pos` alone. Always pose the mesh and take its bounds.

**Trap 1 — the holder cannot be stretched.** A three-wide row needs 54 mm of bore plus walls. This is a **new part**, `holder_6cell.stl`, not an edit to `holder.stl` — and `holder.stl` is an upstream part still used by the 2-cell configuration, so do not overwrite it.

**Trap 2 — "the bay floor is ~56 × 110 mm" is the wrong number, and the repo gives two different right ones.** 56 × 110 mm is `body_back.stl`'s *outer bounding box*. `component_layout_v2.md` records the usable bore after the Part-2 hump extension **inconsistently**: the Layout table says "the bore behind the existing pair is only **28.4 mm** deep", while cut-list item 4 says the interior was deepened "from x≈−0.1396 to ≈−0.166", i.e. **26.4 mm**. **Do not quote either. Measure it** with the new script in step 1. Either way, a 3×2 vertical pack needs a ≥54 mm floor in one axis, and the bore is roughly 26–28 mm along x by 41 mm along y (`y ±20.5`), so **54 mm fits neither axis**. The same document also records that the declared 4-cell block at (−0.145, 0, 0.0306) *overlaps each existing cell by ~11.9 cm³*. The declared placement is a placeholder that does not fit; this task must measure the real bore and then either deepen the hump or relocate the existing pair — and prove it, not assert it.

**Trap 3 — the de-duplication key includes the body.** The MJCF has 264 `<geom>` elements, 263 of them `type="mesh"`. Those 263 collapse to **133 unique `(body, mesh, pos, quat)` instances**: 130 twinned collision+visual pairs plus 3 visual-only singletons (`jetson_orin_nano`, `thermal_partition`, `dcdc_converter`, all on `trunk_assembly`). **De-duplicating on `(mesh, pos, quat)` without the body name collapses to only 102** — because the mirrored left and right legs carry identical local poses — and would delete an entire leg's worth of mass. Adding four cells means adding **eight** geoms to `robot_motors.xml`. Adding only four silently makes the new cells collision-only.

**Trap 4 —** `generate_cad_mods.py` restores `body_back.stl` from `adbc082` on every run. A hump change made anywhere else disappears.

**Trap 5 — a new `print/` part changes the piece count.** `holder_6cell.stl` in `print/` will be picked up by `part_quantities()`'s on-disk fallback and push the set from 52 to 53 pieces. **Add a `- holder_6cell.stl x1` row to `docs/print_guide.md`** and bump `EXPECTED_PIECES` in `tests/test_print_process.py` in the same commit, or M1's tests go red.

**Trap 6 — `tests/test_cad_dimensions.py` requires binary STL.** `_load_tris()` asserts `len(data) == 84 + 50*n` and fails loudly on ASCII STL. `trimesh.Trimesh.export()` to `.stl` writes binary by default, which is what `generate_cad_mods.py::save()` already relies on — do the same.

**2. Low-level implementation plan**

1. **New file `scripts/measure_battery_bay.py`.** Copy the structure of `generate_cad_mods.py::check_partition_fit()`.
   - Load, at their MJCF geom poses in the trunk body frame (`OFF = [-0.019, 0, 0.0648909]` applies to the *shell* geoms only — read the actual `pos`/`quat` off each geom rather than assuming): `body_back`, `body_middle_bottom`, `body_middle_top`, `trunk_bottom`, `trunk_top`, `battery_pack_lid`, plus the existing `cell` ×2, `holder`, `bms`, `usb_c_charger`, `power_switch`.
   - Print the measured free bore: sample the interior of the hump and report its x/y/z extents. This is the number Trap 2 says the repo disagrees about; produce it once and cite it everywhere after.
   - Accept a candidate pack pose and layout on the command line (`--rows 2 --cols 3 --origin x,y,z --pitch 20.5`), build the pack as the union of six Ø18 × 65 mm cylinders plus the tray shell, and `trimesh.boolean.boolean_manifold([...], "intersection")` it against every loaded mesh in turn.
   - Print one line per mesh with the intersection volume in mm³, and a final `WORST` line. Exit 1 if `WORST >= 5.0` mm³ (the same threshold `check_partition_fit` uses).
   - Add a `--from-mjcf` mode that reads the pack poses out of `robot_motors.xml` instead of the command line; the unit test and the smoke test both use it.
2. Run it against the currently declared position to reproduce the known failure, and save that output to `docs/jetson-mod/battery_bay_before.txt` — it is the evidence that the change was needed.
3. Search for a fitting pose. In order of increasing cost, try: (a) 3×2 with the existing pair relocated into the array; (b) 2×3 rotated 90°; (c) 6×1 along x; (d) deepening the hump. Report which ones pass. Record all results, including the failures.
4. If the hump must move: edit **section 4 of `scripts/generate_cad_mods.py`** (lines 226–231) — the two `body_box` calls that union the outer hump and subtract the bore. Change the numbers only; keep the union-then-difference structure and the `save("body_back", bb)` call, which writes both the metre and millimetre copies. Re-run `generate_cad_mods.py`.
5. **New file `scripts/generate_battery_pack.py`** — author `holder_6cell` procedurally with trimesh so it is reproducible, exporting a metre-scale copy to `mini_bdx/robots/open_duck_mini_v2/holder_6cell.stl` and a millimetre-scale copy to `print/holder_6cell.stl` (copy `generate_cad_mods.py::save`). Assert watertight and single-component before export, as `save()` does.
   **Pitch constraint — the naive numbers do not print.** With bore Ø18.6 mm (0.3 mm radial clearance on a wrapped cell) a 19.0 mm pitch leaves a **0.4 mm** web between adjacent bores. Enforce `pitch >= bore_dia + 2 * min_wall` with `min_wall >= 1.0 mm`, i.e. pitch ≥ 20.6 mm, and assert it in the script. Recompute the tray envelope from your actual parameters (a 3×2 array at Ø18.6 / pitch 20.6 / wall 2.0 / floor 2.0 is ≈63.8 × 43.2 × 22 mm) and feed that envelope, not a remembered number, into step 1's fit check.
6. **`robot_motors.xml`:**
   - `<asset>`: add `<mesh name="holder_6cell" file="holder_6cell.stl" />`. The asset block goes from **47 to 48** meshes.
   - In `trunk_assembly`, replace lines 103–108 with six `cell` geom **pairs** and one `holder_6cell` geom pair. Each pair is two `<geom>` lines with byte-identical `pos`/`quat`, the second carrying `contype="0" conaffinity="0"`. Keep the existing `rgba` values so the render stays readable.
7. Mirror the same poses into `mini_bdx/robots/open_duck_mini_v2/robot.xml` (its `trunk_assembly` body starts at line 75; the cell/holder block is lines 97–99 — **one geom per pose there, not two**) and into `robot.urdf` (`<visual>` + `<collision>` pairs on the trunk link, pattern at lines 219–258). `tests/test_model_integrity.py::test_mass_matches_robot_motors` cross-checks total mass between `scene.xml` and `scene_position.xml`; `robot.urdf` is checked only for reference presence, but it is the documented source of the upstream body-frame tensors and must not drift.
8. **`scripts/compute_trunk_inertial.py`:** delete the `cells_extra_4x45g` entry (line 107) from `TRUNK_COMPONENTS`, and update the ledger comment at lines **90–94** (which currently reads `0.698526 + 0.0585 + 0.176 + 0.180 + 0.005 + 0.015 + 0.037 + 0.008`). The four cells' mass then arrives through geometry in M4's composer. **If M4 has not landed yet**, do not delete it — instead correct its `pos` to the measured pack centroid and its tensor to the exact box tensor of the as-placed array, and leave a `# TODO(M4): delete when compose_body_inertials.py owns this` comment. Booking it both ways double-counts 180 g.
9. **`tests/test_cad_dimensions.py`:** extend `NESTING_WHITELIST` (lines 22–41). Names come from `_trunk_geoms()`, which parses **`robot.xml`** and suffixes repeats in document order: `cell`, `cell_2`, `cell_3`, … `cell_6`, plus `holder_6cell`. Do not guess the pairs — run `pytest tests/test_cad_dimensions.py::TestComponentLayout::test_component_pairwise_no_overlap -v` first and whitelist exactly the legitimate container/content pairs it reports (expect `cell_N` ↔ `holder_6cell`, `cell_N` ↔ `battery_pack_lid`, `cell_N` ↔ `cell_M`, `cell_N` ↔ `bms`, `cell_N` ↔ `usb_c_charger`, `holder_6cell` ↔ `bms`/`usb_c_charger`/`power_switch`/`battery_pack_lid`). Note the test compares **AABBs**, not exact meshes, so nesting whitelists are expected; the exact-boolean proof is step 1's script.

**3. Unit tests**

New file `tests/test_battery_pack.py`, `@pytest.mark.phase3`:

- `test_six_cells_are_modelled` — parse `robot_motors.xml`; count geoms with `mesh="cell"`; assert exactly **12** (six poses × collision+visual). Assert `robot.xml` has exactly **6**.
- `test_every_mesh_geom_has_a_visual_twin` — group **all** `type="mesh"` geoms in `robot_motors.xml` by `(body_name, mesh, pos, quat)`; assert every group has size 2 and that exactly one member of each carries `contype="0"`, whitelisting the three known visual-only singletons by name (`jetson_orin_nano`, `thermal_partition`, `dcdc_converter`). Assert the instance count is **133 + 7 = 140** after this task (133 today; six new cell poses replace two, and one `holder_6cell` pose is added alongside the retained `holder`, so recompute from your own edit rather than trusting this arithmetic — the assertion should be a literal you derived and can defend).
- `test_pack_does_not_intersect_the_shells` — import `measure_battery_bay` and call its fit function on the as-authored poses read out of the MJCF; assert worst intersection `< 5.0` mm³. An executing boolean, not a docstring claim.
- `test_holder_6cell_is_manifold_and_accepts_six_cells` — load `holder_6cell.stl`; assert `is_watertight`; assert `len(mesh.split(only_watertight=False)) == 1`; for each of the six nominal bore centres, assert a Ø18.0 × 60 mm probe cylinder placed there has zero intersection with the tray.
- `test_holder_6cell_webs_are_printable` — assert `pitch - bore_diameter >= 2.0` mm from the generator's own parameters (import them from `scripts.generate_battery_pack`).
- `test_total_mass_ledger` — load `scene.xml` with mujoco; assert `sum(model.body_mass)` equals `tests/fixtures/expected_values.json`'s `total_mass_kg` to 1e-3, and that `trunk_assembly`'s mass changed from its pre-task value by exactly the pack's booked mass. **Regenerate `tests/fixtures/expected_values.json` in the same commit** or `tests/test_mass_inertia.py` fails on stale fixtures.

**4. Smoke test**

```bash
cd ~/Projects/Open_Duck_Mini_Jetson
python3 scripts/generate_battery_pack.py
python3 scripts/measure_battery_bay.py --from-mjcf ; echo "fit exit=$?"
python3 -c "import mujoco;m=mujoco.MjModel.from_xml_path('mini_bdx/robots/open_duck_mini_v2/scene.xml');print('nbody',m.nbody,'total',sum(m.body_mass))"
```
Observables: fit exit code `0` with a `WORST` line under 5 mm³; the mujoco line loads without error, prints `nbody 23` (unchanged — this task adds geoms, not bodies), and a total mass that accounts for six 45 g cells exactly once.

**5. Done when**

- [ ] `robot_motors.xml`, `robot.xml` and `robot.urdf` each contain six cells and one `holder_6cell` — twinned in `robot_motors.xml`, single in `robot.xml`, visual+collision in `robot.urdf`.
- [ ] The MJCF `<asset>` block has 48 `<mesh>` entries.
- [ ] `mini_bdx/robots/open_duck_mini_v2/holder_6cell.stl` and `print/holder_6cell.stl` exist, are binary, watertight, single-component, and are regenerable by re-running `scripts/generate_battery_pack.py` to byte-identical output.
- [ ] `scripts/measure_battery_bay.py --from-mjcf` exits 0, and `docs/jetson-mod/battery_bay_before.txt` records the pre-change failure.
- [ ] `pytest tests/test_battery_pack.py tests/test_cad_dimensions.py tests/test_print_process.py -v` all-green.
- [ ] `cells_extra_4x45g` appears exactly zero times (if M4 landed) or exactly one time with a `TODO(M4)` comment (if it has not), and `compute_trunk_inertial.py`'s ledger comment at lines 90–94 matches what the file actually sums.
- [ ] `docs/print_guide.md` has a `holder_6cell.stl x1` row and its totals line is updated from 52 pieces.
- [ ] `docs/jetson-mod/component_layout_v2.md`'s battery row and cut-list item 4 are rewritten from "does not fit" / two conflicting bore depths to the single measured, proven placement.
- [ ] If `body_back.stl` changed, the USD is regenerated (M6).

---

### Task M4 — Rebuild every body's mass and inertia bottom-up, per part — ✅ **DONE 2026-08-13**

> **Completed.** Every one of the 17 real-inertial bodies is now composed
> bottom-up from parts and densities. The four 1e-9 kg marker frames are
> untouched, as PLANT-1b requires.
>
> | | before | after |
> |---|---|---|
> | robot total | 2.657067 kg | **2.729035 kg** (+71.97 g, +2.7 %) |
> | `trunk_assembly` | 1.089544 | **1.188873** |
> | `head_assembly` | 0.362083 | **0.341706** |
>
> **The change is far smaller than this plan predicted** (+653 g at printed=1250)
> because the process M1 chose is `fdm-asa` at 1.07 g/cm³ and the per-part masses
> are measured hollow — volume-weighted mean 0.74 g/cm³. `--check` exits **0**:
> no body's principal moment differs from declared by more than 2×, so there was
> no per-body adjudication to make.
>
> **`--verify-composer` PASSES**, and the anchor split is by evidence rather
> than by tolerance-shopping. The three pure-printed bodies are gated on mass
> AND every principal moment at 0.5 % (they reproduce to 0.999).
> `neck_yaw_assembly` is gated on **mass only** (0.9998): its inertia comes out
> at 0.906–0.981 because this composer splits a servo's 74.5 g across five case
> meshes **by volume**, and a real STS3250 concentrates its motor and gearbox.
> The plan asserted all four on inertia, but its own evidence only ever
> demonstrated *mass* for the servo body. Recorded as `known_issues.md` MASS-3.
>
> **Three bugs I hit, all caught by a test rather than shipped:**
> 1. **`--verify-composer` was self-referential.** It compared against the live
>    MJCF, which `--write` had just replaced with the composed values, so it
>    would pass unconditionally forever. The upstream export is now frozen at
>    `tests/fixtures/upstream_declared_inertials.json` and the anchors judge
>    against that. The anchors are a property of the upstream export — historical
>    data that does not move.
> 2. **The URDF writer reported success and wrote nothing.** Its regex anchored
>    on `<link name="X">\s*<inertial>`, but `<inertial>` sits *after* the visual
>    and collision blocks. Replaced with block surgery that **refuses to write**
>    unless all 17 land, so a partial URDF cannot escape.
> 3. **Most bodies use `diaginertia` + a principal-frame `quat`, not
>    `fullinertia`** — and reading one as the other is the exact frame-permutation
>    bug this repo already paid for. Both forms are now parsed, and `--write`
>    emits `fullinertia` everywhere and deletes the `diaginertia`/`quat` pair so
>    the ambiguity is closed permanently. That broke `verify_known_issues.py`'s
>    PLANT-4 check (INCONCLUSIVE, not REFUTED); it now reads both forms too.
>
> **The human acceptance step was made by the agent** under the owner's standing
> instruction, and signed as such in
> `docs/jetson-mod/inertial_rebuild_report.txt`. It states plainly what it does
> not claim: **9 of 48 part entries carry an `ASSUMED` source** (no mass for them
> exists anywhere in the repo and no hardware exists to weigh), servo inertia is
> volume-split, and nothing has been on a scale. The owner should read those two
> points.
>
> **Cross-check, recorded not gated** (the plan forbids gating one method on the
> other): `compute_trunk_inertial.py` 1.178406 kg vs the composer's 1.188870 —
> a **+10.46 g / +0.9 %** gap, far better than the 11–26 % the plan warned to
> expect. `compute_trunk_inertial.py` is kept and its header now says it is a
> cross-check, not the source of truth.
>
> `tests/test_cad_dimensions.py::test_inertial_matches_generator` was
> **repointed, not deleted**: it compared the model files against the old
> generator, which is backwards now. It compares against the composer across all
> 17 bodies, which also retired M2's strict xfail.
>
> Delivered: `scripts/part_densities.json` (48 entries, every one with a
> `source`), `scripts/compose_body_inertials.py`,
> `tests/fixtures/upstream_declared_inertials.json`,
> `docs/jetson-mod/inertial_rebuild_report.txt`,
> `tests/test_inertial_composer.py` (12 tests), MASS-1/2/3 in `known_issues.md`.
> Suite **186 passed, 5 skipped, 2 xfailed** (both USD, awaiting M6).

**AI-agent suitable:** PARTIAL. Writing the composer, the density table and the golden test is fully suitable for an agent. Accepting the result is not: **this rebuild moves the robot's mass by hundreds of grams** (measured below), and deciding "the composed value is better than the declared one" needs a human. The task is written so the agent stops at a report and a passing composer-verification test, and a human types the acceptance.

> **Ordering note.** This task moves the plant mass, which invalidates every
> existing policy's gate numbers. It does **not** break checkpoint loading, so
> it is recoverable — unlike [Task M0b](#task-m0b--drop-the-antennas-from-the-action-and-observation-spaces).
> But running it before [Task R1](#task-r1--re-gate-the-shipped-v5d-against-the-corrected-plant)
> means the plant has changed twice and you can no longer attribute a result to
> the PLANT-1 fix alone. Confirm R1 has run:
> `ls docs/jetson-mod/eval_results_m2657/*.json`


**1. Context for the implementing agent**

Right now `trunk_assembly` and `head_assembly` are *composed* (by `compute_trunk_inertial.py`, from an upstream baseline plus signed deltas) while the other fifteen real-inertial bodies carry inertials inherited unchanged from the upstream Onshape export. Nobody has ever checked those fifteen against the geometry, and the two composed ones now sit on a density that M2 just corrected. A bottom-up rebuild makes every body's mass and inertia derivable from parts and densities, which is the only form in which the plant can be re-derived after a print-process change without redoing this analysis by hand.

Read first:

- `scripts/compute_trunk_inertial.py` in full (217 lines). `compose()` (lines 166–177) is the parallel-axis combiner to reuse; `report()` (lines 180–205) already emits correctly formatted MJCF `fullinertia` and URDF `<inertia>` lines; `check_base()` (lines 73–78) is the pattern for a self-check that fails loudly; `quat_to_mat()` (lines 62–70) is the quaternion convention the MJCF uses.
- `mini_bdx/robots/open_duck_mini_v2/robot_motors.xml` — **21 `<body>` elements, of which 17 carry a real `<inertial>`. Four are 1e-9 kg marker frames: `trunk`, `left_foot`, `head`, `right_foot`.** Leave those four alone. `scripts/verify_known_issues.py`'s `PLANT-1b` check asserts they keep their 1e-9 placeholder — do not remove it.
- `tests/fixtures/expected_values.json` — **seven** entries (trunk mass/com/diaginertia, head mass/com/diaginertia, total mass), all of which must be regenerated with the model or `tests/test_mass_inertia.py` fails.
- `tests/test_model_integrity.py` lines 145–160 — **three hard-coded range assertions that this task will break** (see the trap below).
- `docs/jetson-mod/mass_inertia_calculations.md`.

Depends on: M1 (densities), M2 (per-part measured masses), M3 (six cells present as geometry). Run last of the four.

**TRAP 1 — a naive rebuild inflates every tensor by about 30 %.** Reproduced this session: for each body, fuse all its meshes at one uniform density chosen to hit the *declared* mass, then divide the composed principal moments by the declared ones:

```
body                            mass g   I_uniform / I_declared (principal, descending)
hip_roll_assembly                85.98      1.363  1.450  0.995
left_roll_to_pitch_assembly      94.66      1.282  1.413  1.009
neck_pitch_assembly              85.68      1.351  1.163  1.413
neck_yaw_assembly               111.31      1.400  1.302  1.258
head_assembly                   362.08      1.192  1.104  1.338
knee_and_ankle_assembly         143.57      1.070  1.011  1.290
knee_and_ankle_assembly_2        92.09      1.206  1.104  1.297
foot_assembly                    75.24      1.043  1.029  1.120
trunk_assembly                 1089.54      0.899  0.877  1.086
head_pitch_to_yaw                16.94      1.000  1.000  1.000   <- golden
left_antenna_holder               4.22      1.000  1.000  0.999   <- golden
right_antenna_holder              4.22      1.000  1.000  0.999   <- golden
```
(The `_2`/`_3`/`_4` mirror bodies reproduce their partners to ±0.01.) Median of each body's worst axis is **1.297**; the worst single figure is **1.450** on `hip_roll_assembly`. The cause is physical, not numerical: the servos are dense masses sitting close to the joint axes, and the printed shells are hollow relative to a fused mesh, so smearing the body's total mass uniformly over its fused volume moves mass outward. **A composer that assigns one density per body is therefore wrong by ~30 % and will look plausible.** The rebuild must be per-part, with a per-part density or a per-part absolute mass, combined by the parallel-axis theorem.

**TRAP 2 — the composer anchors, and what they do and do not prove.** Three bodies are entirely printed parts with no bought-part mass, and their declared inertia is *exactly* uniform-density inertia over their meshes. Measured:

```
head_pitch_to_yaw    m=0.0169378 kg / V=13.5490 cm³  ->  rho = 1250.11 kg/m³   ratio 1.000/1.000/1.000
left_antenna_holder  m=0.00421629 kg / V=3.3694 cm³  ->  rho = 1251.33 kg/m³   ratio 1.000/1.000/0.999
   (that body is TWO meshes: left_antenna_holder 2.9355 cm³ + antenna 0.4339 cm³)
```

A **fourth** anchor pins the servo mass at the same time: composing `neck_yaw_assembly` from its printed meshes at 1250 kg/m³ plus **one servo at 74.5 g** gives **111.29 g** against a declared 111.31 g — a 0.02 % match. That simultaneously validates the 1250 kg/m³ printed convention and the 74.5 g STS3250 figure from `AGENTS.md` line 168.

**But do not generalise from these four.** Composing *every* body the same way (printed meshes at 1250, bought parts at their booked masses, servos at 74.5 g) gives:

```
                                declared      bottom-up @1250     delta
trunk_assembly                  1089.54 g       1374.20 g        +284.66
head_assembly                    362.08 g        471.78 g        +109.69
foot_assembly (each)              75.24 g        125.74 g         +50.50
knee_and_ankle_assembly (each)   143.57 g        182.58 g         +39.01
hip_roll_assembly (each)          85.98 g         96.59 g         +10.61
neck_yaw_assembly                111.31 g        111.29 g          -0.02
head_pitch_to_yaw                 16.94 g         16.94 g          -0.00
antenna holders (each)             4.22 g          4.21 g          -0.00
TOTAL                           2657.07 g       3310.82 g        +653.75
```

`foot_assembly`'s declared mass implies **748 kg/m³** over its fused meshes, and `hip_roll_assembly`'s printed remainder implies roughly **650 kg/m³**. So the upstream export did **not** book every printed part at solid PLA — some bodies were booked at as-printed (hollow) masses and some at solid. **The four anchors verify your composer's mechanics; they do not license "printed = 1250" as a model of the whole robot.** At the chosen MJF density (1.01 g/cm³) the same rebuild gives 2936 g total and a 1214 g trunk — still +279 g on the robot. Whatever you do, **this task changes the plant materially and deliberately**, and that must be stated in the report and accepted by a human.

**TRAP 3 — the golden check and the production density are in direct conflict.** If the composer uses the chosen process density for printed parts (1.01 for MJF), `head_pitch_to_yaw` composes to 13.68 g against a declared 16.94 g — 19 % off — and a `--check` that gates on the goldens at 0.5 % would exit 1 forever, making `--write` unreachable. **Split the two concerns**: a `--verify-composer` mode that forces printed density to 1250 kg/m³ and servo mass to 74.5 g and asserts the four anchors, and a normal run that uses `scripts/print_process.json`. `--write` gates on `--verify-composer`, never on the goldens under production densities.

**TRAP 4 — three hard-coded range tests will fail.** `tests/test_model_integrity.py` asserts `1.05 < trunk_mass < 1.30` (line 149), `0.30 < head_mass < 0.40` (line 155), and `2.5 < total < 3.0` (line 160). At printed=1250 the rebuild produces trunk 1.374 kg and total 3.311 kg — **two of the three fail**. Update those bounds in the same commit, with a one-line comment naming this task and the new expected values. Do not widen them to meaninglessness.

**TRAP 5 — the de-duplication key includes the body name.** 263 mesh geoms → **133 unique `(body, mesh, pos, quat)` instances**: 130 twinned pairs + 3 visual-only singletons on `trunk_assembly`. Keying on `(mesh, pos, quat)` alone gives 102 and silently deletes the mirrored right leg. Iterating raw `<geom>` elements doubles every mass and every tensor.

**TRAP 6 — one servo is five meshes, and they do not share a `pos`.** An STS3250 appears as `wj-wk00-0122topcabinetcase_95` + `wj-wk00-0123middlecase_56` + `wj-wk00-0124bottomcase_45` + `drive_palonier` + `passive_palonier`. That is **one 74.5 g servo**, not five parts. There are 14 servo instances (70 unique mesh instances, 14 of each mesh). **`passive_palonier` carries a different `pos` from the other four in every single servo** (e.g. `0.0125 0 -0.020515` vs `0.0096 0 -0.020515`), so grouping "by shared `(pos, quat)`" splits it off into a phantom sixth part. Verified working key: **`(body_name, quat)`** — it yields exactly 14 groups of exactly 5 meshes each, including the three servos that share `trunk_assembly` (they have three distinct quats).

**TRAP 7 — `antenna` must be treated as printed, not bought.** The `left_antenna_holder` body is two meshes and reproduces its declared inertia only when *both* are at ~1250 kg/m³. Classing `antenna` as a bought part with its own mass breaks the anchor.

**TRAP 8 — nine part masses do not exist anywhere in this repo.** Verified by grep: there is no absolute mass for `bms` (only a +5 g *delta* in `compute_trunk_inertial.py` line 109), `board`, `bno055`, `roll_bearing`, `sg90`, `usb_c_charger`, `power_switch`, `holder`, or `antenna`. No hardware exists to weigh. The rule: book them as `{"class": "printed"}` or at the process density and set `"source": "ASSUMED — no measurement or datasheet in repo as of <date>"`. The validator must count how many entries carry an `ASSUMED` source and print that count at the top of the report, so the human accepting the rebuild knows how much of it is guessed.

**2. Low-level implementation plan**

1. **New file `scripts/part_densities.json`** — one entry per mesh name in the MJCF `<asset>` block (**47 today, 48 after M3**). Each entry is exactly one of:
   - `{"class": "printed", "source": "..."}` — mass comes from `scripts/part_mass_table.json` if the part has a `print/` file under that exact name, else from `print_process.json`'s density × mesh volume. **Remember the name mapping is not the identity** (`left_knee_to_ankle_left_sheet` ↔ `knee_to_ankle_left_sheet`, and 19 MJCF meshes have no `print/` file at all) — carry an explicit `"print_name"` field wherever they differ, and let the validator fail on a missing mapping rather than silently falling back.
   - `{"class": "bought", "mass_g": <number>, "source": "<where the number comes from>"}` — verified-in-repo values: `cell` 45 g (`compute_trunk_inertial.py` line 107, 0.180 kg / 4), `jetson_orin_nano` 176 g (`AGENTS.md` line 119 and line 178: 176 g is the design input carried in the MJCF; NVIDIA SP-11324-001 v1.3 §4 publishes 175 g — keep 176 and record the discrepancy in `source`), `dcdc_converter` 15 g (line 111), `thermal_partition` 37 g (line 116, an assembly budget including mica and rails). Everything else in the bought class falls under Trap 8.
   - `{"class": "servo_case", "servo": "sts3250", "mass_g": 74.5, "source": "AGENTS.md:168"}` — for the five servo meshes; the composer splits `mass_g` across the five meshes of one servo instance by volume.
   Every entry carries a `source` string.
2. **New file `scripts/compose_body_inertials.py`.**
   - Parse `robot_motors.xml`. For each body with a real `<inertial>` (skip the four 1e-9 kg markers `trunk`, `left_foot`, `head`, `right_foot`), collect its `<geom type="mesh">` children and **de-duplicate on `(body_name, mesh, pos, quat)`**.
   - Group servo-case meshes into servo instances by **`(body_name, quat)`**; assert exactly 14 groups of exactly 5 meshes.
   - For each unique instance: load the STL from `mini_bdx/robots/open_duck_mini_v2/`, apply `pos`/`quat` (reuse `compute_trunk_inertial.quat_to_mat`), set `mesh.density` to the part's density (bought parts: `density = mass_g / volume`), and take `mesh.moment_inertia` (about the mesh centroid, in body-aligned axes) and `mesh.center_mass`.
   - Combine with the existing `compose()`, which already does the parallel-axis sum correctly.
   - Emit through the existing `report()` so the MJCF and URDF lines are formatted identically to today's.
   - **`--verify-composer` mode**: force printed density to 1250 kg/m³ and servo mass to 74.5 g; assert `head_pitch_to_yaw`, `left_antenna_holder`, `right_antenna_holder` and `neck_yaw_assembly` reproduce declared mass and each declared principal moment to within 0.5 %. Exit 1 on failure. This proves the composer's mechanics and nothing else.
   - **`--check` mode**: uses production densities. For every body print `declared mass | composed mass | Δg | declared principal | composed principal | ratio per axis`, plus a header line with the total declared and composed robot mass and the number of `ASSUMED` sources. Exit 2 (a distinct code) if any body's composed principal moment differs from declared by more than 2×, so a human reviews it rather than a script rubber-stamping it. **`--check` must not fail merely because the mass moved** — moving the mass is the point.
   - **`--write` mode**: rewrite the `<inertial>` element of every non-marker body in `robot_motors.xml`, `robot.xml` and `robot.urdf`, and regenerate all seven entries of `tests/fixtures/expected_values.json`. `--write` must refuse to run unless `--verify-composer` exits 0 and the report file contains a human acceptance line.
3. **`scripts/compute_trunk_inertial.py`** — keep the file (it holds the frame-correction history and the upstream baseline tensors, which are the only record of the `quat`-vs-body-frame bug), and add a header note that it is now a *cross-check* and that `compose_body_inertials.py` is the source of truth. **Do not add a 5 % agreement assertion between the two.** They are structurally different methods (upstream-baseline-plus-deltas vs bottom-up) and measurably disagree by 11–26 % on the trunk; a 5 % gate would be an unpassable blocker. Instead have `compose_body_inertials.py --check` print both numbers side by side and the gap in grams, for the record.
4. Run:
   ```bash
   cd ~/Projects/Open_Duck_Mini_Jetson
   python3 scripts/compose_body_inertials.py --verify-composer ; echo "verify exit=$?"
   python3 scripts/compose_body_inertials.py --check | tee docs/jetson-mod/inertial_rebuild_report.txt
   ```
   Read the report. **HUMAN ACCEPTANCE STEP:** for every body flagged >2×, and for the whole-robot mass change, decide whether the composed value or the declared value is right, and append the decisions to `docs/jetson-mod/inertial_rebuild_report.txt` under a heading `## Accepted by <name> on <date>`. Then:
   ```bash
   python3 scripts/compose_body_inertials.py --write
   ```
5. Update `tests/test_model_integrity.py` lines 145–160 to the new expected bands (Trap 4). Update `docs/jetson-mod/mass_inertia_calculations.md` to describe the per-part method and to reproduce the anchor table above. Update `AGENTS.md`'s "Robot Physical Specs" and "Mass Changes from Modification" tables with the new totals. Add the ~30 %-inflation finding, the four anchors, and the "upstream did not use one density" finding to `known_issues.md` as documented traps so the next person does not rediscover them.

**3. Unit tests**

New file `tests/test_inertial_composer.py`, `@pytest.mark.phase1`:

- `test_composer_reproduces_the_four_anchors` — the load-bearing test. Run the composer in `--verify-composer` semantics; assert `head_pitch_to_yaw`, `left_antenna_holder`, `right_antenna_holder` composed mass within 0.5 % of declared **and** each composed principal moment within 0.5 % of declared, with implied density within 1 % of 1250 kg/m³; assert `neck_yaw_assembly` within 0.5 % with the servo at 74.5 g.
- `test_mesh_instances_are_deduplicated` — assert the composer's instance count for the whole model equals the count of unique `(body, mesh, pos, quat)` tuples, that this is **133** before M3, and that summing raw `<geom type="mesh">` elements gives **263**. Also assert that keying without the body name gives a *smaller* number (102 before M3), so the trap is encoded as a fact rather than a comment.
- `test_one_servo_is_one_mass` — assert exactly **14** servo instances grouped by `(body, quat)`, each of exactly 5 meshes, totalling `14 × 74.5 = 1043 g`, matching `AGENTS.md` line 204's "All 14 servos" row.
- `test_every_asset_mesh_has_a_density_entry` — set difference between the MJCF `<asset>` mesh names and the keys of `part_densities.json`; assert empty in both directions. Prevents a part silently massing zero.
- `test_every_density_entry_cites_a_source` — assert every entry has a non-empty `source`, and assert the count of entries whose source starts with `ASSUMED` matches a literal constant in the test, so silently adding a new guess fails.
- `test_printed_parts_resolve_to_a_print_name` — for every `{"class": "printed"}` entry, assert the resolved `print/` filename exists or the entry explicitly declares `"print_name": null` with a source explaining why it is priced from mesh volume instead.
- `test_all_bodies_satisfy_triangle_inequality` — after `--write`, load `scene.xml` with mujoco and assert the eigenvalue triangle inequality for the **17** real bodies (this duplicates an existing assertion in `test_mass_inertia.py` deliberately, because the composer is the new thing that can break it).
- Regenerate `tests/fixtures/expected_values.json` (all seven entries) in the same commit, or `tests/test_mass_inertia.py` will fail on stale fixtures.

**4. Smoke test**

```bash
cd ~/Projects/Open_Duck_Mini_Jetson
python3 scripts/compose_body_inertials.py --verify-composer ; echo "verify exit=$?"
python3 scripts/compose_body_inertials.py --check ; echo "check exit=$?"
python3 -c "import mujoco;m=mujoco.MjModel.from_xml_path('mini_bdx/robots/open_duck_mini_v2/scene.xml');print('total kg',sum(m.body_mass))"
pytest tests/test_mass_inertia.py tests/test_inertial_composer.py tests/test_model_integrity.py -v
```
Observables: `--verify-composer` exits 0 and its four anchor rows show ratios of `1.000`; `--check` exits 0 (or 2 with a body list that a human then adjudicates); the mujoco total mass equals `expected_values.json`'s `total_mass_kg`; the pytest run is all-green.

**5. Done when**

- [ ] `scripts/part_densities.json` covers every mesh in the MJCF `<asset>` block, each with a `source`, and the count of `ASSUMED` sources is stated at the top of the rebuild report.
- [ ] `scripts/compose_body_inertials.py --verify-composer` exits 0.
- [ ] `scripts/compose_body_inertials.py --check` has been run and its report is committed at `docs/jetson-mod/inertial_rebuild_report.txt`.
- [ ] That report carries an `## Accepted by <name> on <date>` section covering every body flagged >2× **and** the whole-robot mass change.
- [ ] `robot_motors.xml`, `robot.xml`, `robot.urdf` and all seven entries of `tests/fixtures/expected_values.json` were written by `--write` in one run and agree with each other.
- [ ] `tests/test_model_integrity.py`'s three hard-coded mass ranges have been updated to the new plant with a comment naming this task.
- [ ] `pytest tests/test_inertial_composer.py tests/test_mass_inertia.py tests/test_model_integrity.py -v` all-green.
- [ ] The ~30 %-inflation trap, the four anchors, and the "upstream did not use one density" finding are written into `known_issues.md`.
- [ ] `docs/jetson-mod/mass_inertia_calculations.md` describes the per-part method, not the old lumped one.

---

### Task M5 — Shell the thick parts (solid-process branch only) — ⏭️ **N/A 2026-08-13** — `print_process.json` `kind` is `fdm`; recorded in `print_process_decision.md`

**AI-agent suitable:** PARTIAL. The boolean geometry, the drain holes and the mass measurement are scriptable and the agent should do them. What needs a human: judging that a hollowed part is still stiff enough to carry servo loads, and getting a DFM check from the printing bureau. A hollowed shell that cracks under a servo mount is not something either the mass model or the simulator will catch.

> **Ordering note.** This task moves the plant, which invalidates every existing
> policy's gate numbers. It does **not** break checkpoint loading, so it is
> recoverable — unlike [Task M0b](#task-m0b--drop-the-antennas-from-the-action-and-observation-spaces).
> But running it before [Task R1](#task-r1--re-gate-the-shipped-v5d-against-the-corrected-plant)
> means the plant has changed twice and you can no longer attribute a result to
> the PLANT-1 fix alone. Confirm R1 has run:
> `ls docs/jetson-mod/eval_results_m2657/*.json`

**Skills, and the constraint that decides the method — read before coding.**

You will be tempted to reach for the `cad` skill and call `shell()` or offset a
face. **You cannot.** These parts exist only as STLs. An STL is a triangle soup
with no B-rep: no faces to select, no features to offset. Verified:

```
$ python3 ~/.claude/skills/cad/scripts/inspect refs print/body_front.stl --facts
{"ok":false, ... "CAD STEP ref not found for 'print/body_front.stl'."}
```

**Therefore this task is a MESH operation, not a CAD operation.** Use `trimesh`
+ `manifold3d`, exactly as `scripts/generate_cad_mods.py` already does — that
script is your worked example for boolean edits on these meshes, including how
it re-exports to both the sim mesh directory and the mm-scale `print/` copy.

Where the skills still apply:

| Purpose | Skill | Note |
|---|---|---|
| Visual review of every shelled part | `cad` `scripts/snapshot`, then `cad-viewer` | snapshot **accepts `.stl`** even though `inspect refs` does not. Review before and after. |
| Measuring the mass you actually saved | `gcode`, or `scripts/measure_print_mass.py` | for a solid process, mass is volume × density — no slicer needed |
| If you decide to re-author a part properly | `cad` + `step-parts` | only justified for a part being redesigned anyway. It buys a parametric source the project has never had — but it is a *per-part* cost, and shelling five parts is not a reason to pay it five times. |

**Wall-thickness measurement to target the work.** `T_eff = 2·V/A` is a cheap
proxy for local wall thickness and is computable directly from the mesh:

```python
import trimesh
m = trimesh.load('print/body_front.stl', force='mesh')
print(2 * m.volume / m.area)   # mm
```

Measured on this part set, worst first — use it to pick which parts are worth
shelling at all:

```
trunk_top           T_eff 7.99 mm   145 cm3   (27.9% of AABB)
body_front          T_eff 7.52 mm   135 cm3   (86.1% of AABB — a solid slab)
body_middle_bottom  T_eff 4.99 mm   146 cm3   (10.2% of AABB)
head                T_eff 3.38 mm   229 cm3   (marginal)
body_back           T_eff 3.19 mm    95 cm3   (marginal)
body_middle_top     T_eff 2.83 mm    76 cm3   (already thin — leave it)
```

**Drain holes are mandatory on powder processes.** Today **no part in this set
has more than one shell**, i.e. there are no enclosed voids and no powder traps
— verified across all 37 STLs. Hollowing a part *creates* an enclosed void and
therefore creates a trap. If you shell it, you must add escape holes in the same
edit, and the done-when check for this task must re-run the shell-count test:

```python
import trimesh
m = trimesh.load(path, force='mesh')
print(len(m.split(only_watertight=False)))   # >1 after shelling == a sealed void
```

**1. Context for the implementing agent**

**Run this task only if `scripts/print_process.json`'s `kind` field is `"solid"`.** Test the `kind`, not the name — `mjf-pa12-gb` and `sla-tough` are both solid, and a future FDM entry could be named anything. If `kind` is `"fdm"`, write "N/A — FDM infill already hollows the interior; a 15 % gyroid does what shelling would do" into `docs/jetson-mod/print_process_decision.md` and skip to M6. Nothing else in the phase depends on M5 having run.

Why it exists on the solid branch: powder and resin parts are solid, so a thick part is thick *material* you pay for and carry. `body_front` is 135.44 cm³ inside a 10.00 × 110.00 × 143.00 mm envelope (157.3 cm³) — 86 % dense, effectively a solid 10 mm plate. Process design guides cap recommended wall thickness because heat and trapped powder in thick sections cause sink and warp, so shelling is a *print-quality* requirement on this branch, not only a mass optimisation. **The cap number is not in this repo. Get it from the bureau or the current published process guide at implementation time, put it in `print_process.json`'s `max_wall_mm` (M1 created the field), and cite the source in the commit message.** Do not carry a remembered figure.

Read first:

- `scripts/generate_cad_mods.py` — `body_box()` (86–91), `body_cyl()` (94–97), `boolean()` (100–101), `save()` (104–110), and section 3 (`body_front`, lines 215–223), which already cuts slots in this part.
- `scripts/print_process.json` and `scripts/part_mass_table.json` from M1/M2.

Measured thickness ranking of the **`print/`** set (effective thickness = 2·Volume/Area in mm; reproducible with trimesh and independent of any ray-sampling choice). Re-verified this session:

```
trunk_top             144.68 cm³   7.99 mm
body_front            135.44 cm³   7.52 mm
foot_bottom_tpu        23.60 cm³   5.09 mm   <- TPU, not in scope
body_middle_bottom    145.80 cm³   4.99 mm
foot_top               39.65 cm³   4.97 mm
foot_bottom_pla        15.88 cm³   4.63 mm
head_yaw_to_roll       29.42 cm³   4.61 mm
battery_pack_lid       23.64 cm³   4.59 mm
foot_side              21.45 cm³   4.49 mm
trunk_bottom           22.24 cm³   4.47 mm
left/right_roll_to_pitch 29.64 cm³ 3.84 mm each
neck_left_sheet         8.06 cm³   3.75 mm
leg_spacer              7.05 cm³   3.73 mm
roll_motor_top          9.18 cm³   3.46 mm
head                  229.35 cm³   3.38 mm   <- largest single volume in the set
body_back              95.45 cm³   3.19 mm
knee_to_ankle_left_sheet 8.26 cm³  3.12 mm
roll_motor_bottom       8.49 cm³   3.12 mm
left/right_cache       64.61 cm³   3.08 mm each
head_roll_mount        13.37 cm³   2.96 mm
body_middle_top        76.26 cm³   2.83 mm
```

**Trap 1 — `generate_cad_mods.py` will revert you.** `body_front`, `body_middle_bottom`, `trunk_bottom` and `body_back` are restored from commit `adbc082` at the start of every run of that script. Shelling for those four **must** be implemented as steps inside `generate_cad_mods.py`. `trunk_top`, `head`, `left_cache`, `right_cache`, `body_middle_top` and everything else are *not* owned by that script; shelling them needs a new sibling generator with the same restore-from-git idempotency, or they will not be reproducible.

**Trap 2 — a sibling generator that copies `save()` will destroy `print/head.stl`.** `save()` writes the metre mesh to `mini_bdx/robots/.../` and a ×1000 copy to `print/`. But for `head` those two files are **different geometry today** (217.23 cm³ sim vs 229.35 cm³ print) — as are `thermal_partition` (20.71 vs 13.07), `left_roll_to_pitch` and `right_roll_to_pitch` (29.38 vs 29.64). For any part in that list, the new generator must restore and shell **each copy from its own baseline** and write it back to its own directory, never derive one from the other. Add an assertion: after shelling, `print/<n>.stl` volume × 1000 must still differ from `sim/<n>.stl` volume by the same amount it did before, for the parts where they legitimately differ.

**Trap 3 — a sealed pocket is already rejected, and that is your drain-hole test.** Verified experimentally with trimesh: a box with a fully enclosed interior void reports `is_watertight == True` but `len(mesh.split(only_watertight=False)) == 2` and `euler_number == 4`; after a through-hole is cut it is 1 component with `euler_number == 0`. `save()`'s existing `assert len(mesh.split(only_watertight=False)) == 1` therefore already fails on any sealed void. Rely on it, and add the genus check as a second, more specific signal.

**Trap 4 — powder must be able to escape.** Every pocket needs drain holes on a face that will be downward or hidden in the assembly, or the bureau rejects the part — or worse, ships it with powder still inside, which is mass you paid for, did not model, and cannot remove.

**2. Low-level implementation plan**

1. **New file `scripts/measure_wall_thickness.py`** — for every STL in `print/`, print volume, area, `2V/A`, and the median **and 95th percentile** of `trimesh.proximity.thickness` over a few thousand surface samples (verified available in trimesh 4.11.4 in this environment). Commit its output as `docs/jetson-mod/wall_thickness.txt`. This is the evidence that selects the shelling candidates, rather than a guess.
2. Pick the candidate list from that output using `max_wall_mm` from `print_process.json`. Start with `body_front` and `trunk_top` — the two worst — and treat the rest as a second pass. Exclude `foot_bottom_tpu` (different material, different process).
3. **`scripts/generate_cad_mods.py`, new section 3b (`body_front`):** subtract explicit interior pocket solids using the existing `body_box()` helper, sized to leave the target wall on every face and to avoid every existing feature (the six inlet slots cut in section 3, and any mounting boss or screw column). Use explicit pockets rather than a generic inward offset: trimesh has no robust mesh offset, whereas the existing exact-boolean pocket approach is what the whole file already does and what `check_partition_fit()` already proves against.
4. Add **two or more Ø5 mm drain holes per pocket**, via `body_cyl()`, on a face that is downward or hidden in the assembly.
5. Assert, as `save()` already does, that the result is watertight and a single component. Add an assertion that no pocket comes within `max_wall_mm` of any existing cut.
6. **New file `scripts/generate_shelled_parts.py`** for the parts `generate_cad_mods.py` does not own. Same structure: restore the pristine mesh from a named `BASELINE_COMMIT`, apply pockets, write each copy back to its own directory per Trap 2.
7. **Save the pre-shelling mass table before overwriting it**, then measure the result — do not quote an estimate:
   ```bash
   cd ~/Projects/Open_Duck_Mini_Jetson
   cp scripts/part_mass_table.json docs/jetson-mod/part_mass_table_pre_shelling.json
   python3 scripts/measure_print_mass.py --emit-table scripts/part_mass_table.json
   ```
8. Re-run **M2** (`generate_cad_mods.py`, then `compute_trunk_inertial.py`) and **M4** (`compose_body_inertials.py --verify-composer`, `--check`, `--write`), because the geometry the mass model reads has changed. Note that `generate_cad_mods.py`'s `--emit-table` input must be regenerated *before* it runs, so the order is: measure → generate → compose.
9. **HUMAN STEP:** send the shelled STLs to the bureau for a DFM check before committing the design.

**3. Unit tests**

New file `tests/test_shelling.py`, `@pytest.mark.phase3`:

- `test_shelled_parts_are_manifold` — for each shelled part in both directories, assert `is_watertight` and `len(split(only_watertight=False)) == 1`.
- `test_no_wall_exceeds_the_process_cap` — assert the **95th percentile** of `trimesh.proximity.thickness` over surface samples of each shelled part is at or below `print_process.json`'s `max_wall_mm`, and print the max for the record. A median-only assertion passes while a solid boss remains; do not use it as the gate.
- `test_every_pocket_is_drained` — for each shelled part, assert `euler_number` did not increase relative to its pre-shelling baseline (a sealed void raises it; a drained one does not) **and** that the part is a single component. Executes on geometry rather than checking a comment.
- `test_shelling_did_not_move_the_outer_surface` — boolean-difference the pre-shelling part from the shelled part and assert the result has zero volume, i.e. only interior material was removed and every mating face is untouched.
- `test_mass_table_reflects_the_shelling` — assert the new `scripts/part_mass_table.json` mass for each shelled part is strictly less than the corresponding entry in `docs/jetson-mod/part_mass_table_pre_shelling.json`.
- `test_print_and_sim_copies_kept_their_relationship` — for `head`, `thermal_partition`, `left_roll_to_pitch`, `right_roll_to_pitch`: assert the sim/print volume relationship is unchanged in kind (both shelled, neither replaced by the other). Encodes Trap 2 as an executing check.

**4. Smoke test**

```bash
cd ~/Projects/Open_Duck_Mini_Jetson
python3 scripts/generate_cad_mods.py && python3 scripts/generate_shelled_parts.py
python3 scripts/measure_print_mass.py --process "$(python3 -c 'import json;print(json.load(open("scripts/print_process.json"))["process"])')"
```
Observable: the `TOTAL` gram column drops relative to the pre-shelling report saved in M1 (`docs/jetson-mod/print_mass_<process>.txt`), by a figure the commit message quotes; every part reports `solid` in the `how` column and none are listed under `NOT COUNTED`.

**5. Done when**

- [ ] `scripts/print_process.json`'s `kind` is `"solid"` and its `max_wall_mm` is set and sourced (otherwise this task is correctly skipped and that is recorded in `print_process_decision.md`).
- [ ] `docs/jetson-mod/wall_thickness.txt` exists and lists the candidates with median and 95th-percentile thickness.
- [ ] Every shelled part is produced by a script that restores from a named baseline commit and is byte-reproducible across two consecutive runs (`git status --porcelain` empty on the second run).
- [ ] `pytest tests/test_shelling.py tests/test_cad_dimensions.py -v` all-green.
- [ ] `docs/jetson-mod/part_mass_table_pre_shelling.json` is committed alongside the new table.
- [ ] The measured whole-set mass drop is recorded in `docs/jetson-mod/print_process_decision.md` alongside the price delta from the bureau.
- [ ] M2 and M4 have been re-run **after** the shelling, not before.
- [ ] A DFM response from the printing bureau is recorded.

---

### Task M6 — Regenerate the USD and re-run `audit_plant_mass.py` as the phase gate — ✅ **DONE 2026-08-13** — 🚦 **PHASE M COMPLETE**

> **THE PHASE GATE PASSED.**
>
> ```
> TOTAL   MJCF 2.729035   PhysX 2.729035   +0.000000
> VERDICT: PASS — simulated total is +0.000000 kg (+0.0%) versus the MJCF
> AUDIT EXIT=0
> ```
>
> The pipeline was re-run in order before the conversion and proved
> **idempotent**: `--emit-table` → `generate_cad_mods.py` →
> `--verify-composer` (PASS) → `--check` (exit 0, declared == composed ==
> 2729.04 g, delta **+0.00 g**), with `git status` clean on every mesh and model
> file. Only then was the USD regenerated. New asset hash
> **`767f2415d1b3a056d95e9c310434dbbc`**, mesh manifest **48** entries;
> `config.yaml`, `.asset_hash`, `.mesh_manifest.json` and the USD are all
> committed together from one conversion run.
>
> **The strict xfails did their job.** Both USD staleness tests turned
> `XPASS(strict)` the instant the USD was regenerated — a hard failure that
> forced the markers' removal rather than letting them rot. Removed.
>
> **`verify_action_contract.py` then caught a real gap M0b left behind.**
> `duck_init_pos.json` still declared 16 action joints while the policy now
> commands 14, so the check failed — correctly. That file **is** the deployment
> contract, so it now carries **both** orders explicitly:
> `joint_order` (16, the articulation / state order) and
> `action_joint_order` (14, what the policy commands), plus `action_dim`,
> `obs_dim` and `action_scale`. The check compares each against its own and
> passes:
>
> ```
> joint order MATCHES duck_init_pos.json (16 joints)
> action joint order MATCHES the articulation order minus the antennas (14 joints)
> observation dim: 53   action dim: 14   mass 2.729035 kg
> VERDICT: PASS
> ```
>
> One more test needed repointing, not silencing:
> `test_usd_asset_hash_is_the_corrected_model` compared the archived
> `eval_results_m2657/` JSONs against the **live** repo hash. That directory is a
> historical record of the pre-Phase-M plant, so once M6 regenerated the USD the
> live hash legitimately moved on. It now compares against the hash the
> directory's own README declares, which still catches the thing that matters —
> a result from a different model landing in there.
>
> Suite **204 passed, 5 skipped, 0 xfailed**. Verifier **CONFIRMED 20 / 21**.
>
> **Training is now unblocked.** Task R2 may run.


**AI-agent suitable:** YES. It needs a GPU and an Isaac Lab install; both are present (`~/IsaacLab/isaaclab.sh` exists, GPU is an NVIDIA GB10). There is no purchase and no physical step.

> **Ordering note.** This task moves the plant mass, which invalidates every
> existing policy's gate numbers. It does **not** break checkpoint loading, so
> it is recoverable — unlike [Task M0b](#task-m0b--drop-the-antennas-from-the-action-and-observation-spaces).
> But running it before [Task R1](#task-r1--re-gate-the-shipped-v5d-against-the-corrected-plant)
> means the plant has changed twice and you can no longer attribute a result to
> the PLANT-1 fix alone. Confirm R1 has run:
> `ls docs/jetson-mod/eval_results_m2657/*.json`


**1. Context for the implementing agent**

Isaac Lab loads `mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd`, not the MJCF. Everything M2–M5 changed is invisible to training until the USD is regenerated. This task is the only place in the phase where the change is proven end to end against the simulator that will actually run it.

Read first:

- `scripts/convert_mjcf_to_usd.py` (124 lines) — note `force_usd_conversion=True`, `import_inertia_tensor=True`, and `write_mesh_manifest()` (lines 89–120), which writes `usd/.mesh_manifest.json` and then calls `simulation_app.close()`.
- `scripts/audit_plant_mass.py` — the whole file (175 lines), especially the `finally` block (lines **155–175**). `simulation_app.close()` never returns; the script uses `os._exit(code)` after an explicit flush so its exit status survives, and Kit leaves stdout block-buffered on redirect. **Do not "clean that up".**
- `docs/jetson-mod/known_issues.md` PLANT-1, section `### FIXED 2026-08-11 — option 1` (line 285), for what a passing audit looks like.

Depends on: whichever of M2–M5 ran.

Traps:

- **One Isaac job at a time, one env per process.** Do not start a second Isaac process while the audit runs, and do not edit any source file that Isaac has loaded while a probe is in flight.
- The audit runs task `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0` with `num_envs=1` (lines 69, 88), so it exercises the real env config and the real USD, not a toy scene.
- `usd/config.yaml` is **not** a stale-path problem any more. `known_issues.md` ART-3 is marked **FIXED 2026-08-11**, and the file now names the live repo path. If you see a foreign path there, something regenerated it from the wrong working copy — that *is* worth chasing.
- `usd/config.yaml` is an **input to `test_usd_not_stale`'s hash**. `config.yaml`, `.asset_hash`, `.mesh_manifest.json` and `open_duck_mini_v2.usd` must all be committed together, from one conversion run.
- The tests in `tests/` are largely source-text grepping and 5 of 7 seeded regressions pass them (`known_issues.md` TEST-1). **The pytest run below is a smoke check, not the gate. The gate is `audit_plant_mass.py`'s exit code.**
- A bare `pytest` from the repo root errors at collection (`known_issues.md` TEST-2 — verified: `experiments/RL/old_test.py` imports `gymnasium`, absent from the system Python). Always name the files.

**2. Low-level implementation plan**

1. If any of M2/M3/M5 touched geometry, re-run the generators first so the STLs on disk are the ones the USD will embed:
   ```bash
   cd ~/Projects/Open_Duck_Mini_Jetson
   python3 scripts/generate_cad_mods.py
   python3 scripts/generate_battery_pack.py      # if M3 ran
   python3 scripts/generate_shelled_parts.py     # if M5 ran
   ```
2. Re-run the mass model and write it into the model files:
   ```bash
   python3 scripts/measure_print_mass.py --emit-table scripts/part_mass_table.json
   python3 scripts/generate_cad_mods.py          # consumes the table it was just handed
   python3 scripts/compose_body_inertials.py --verify-composer \
     && python3 scripts/compose_body_inertials.py --check \
     && python3 scripts/compose_body_inertials.py --write
   ```
   (`--emit-table` with no `--process` reads `scripts/print_process.json`; that defaulting is added in M2 step 1.)
3. Regenerate the USD — **this must be the last thing that touches the MJCF or any STL**:
   ```bash
   cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/convert_mjcf_to_usd.py
   ```
   Expect `SUCCESS: USD conversion complete.` and `Wrote mesh manifest: .../usd/.mesh_manifest.json (N meshes)`, where N is the count of `<mesh>` entries in the MJCF — **47 today, 48 after M3**.
4. Run the staleness and model tests (smoke, not gate):
   ```bash
   cd ~/Projects/Open_Duck_Mini_Jetson
   pytest tests/test_usd_conversion.py tests/test_mass_inertia.py tests/test_model_integrity.py \
          tests/test_cad_dimensions.py tests/test_inertial_composer.py tests/test_cad_mod_deltas.py \
          tests/test_print_process.py tests/test_battery_pack.py -v
   ```
   Drop any file whose task did not run. (For reference, the four pre-existing files pass in ~2.3 s.)
5. **Run the gate:**
   ```bash
   cd ~/IsaacLab && ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/audit_plant_mass.py --headless \
     > ~/Projects/Open_Duck_Mini_Jetson/docs/jetson-mod/audit_plant_mass_phaseM.txt 2>&1
   echo "audit exit=$?"
   ```
   `isaaclab.sh` collapses any nonzero status to 1, which is all a gate needs.
6. Confirm in the captured output that there are **zero** `unauthored body` lines, **zero** `NO <inertial> IN MJCF` lines, and **zero** PhysX `possibly invalid inertia tensor` warnings (`grep -c` each), and that the `TOTAL` row's delta column is `-0.000000` or `+0.000000`.
7. Documentation sweep, all in the same commit:
   - `AGENTS.md`: line 16 (the `~2.66 kg` / "Isaac now simulates" sentence), the "Robot Physical Specs" total-mass row, and the "Mass Changes from Modification" table at lines 202–204.
   - `docs/jetson-mod/known_issues.md`: mark PLANT-10 FIXED with the measured numbers; add the M4 inflation trap, the four composer anchors, and the "upstream did not use one density" finding.
   - `docs/jetson-mod/component_layout_v2.md`: the "Part-2 status" paragraph (line ~137), the Whole-robot-mass sentence, and the battery-pack row and cut-list item 4.
   - `docs/jetson-mod/mass_inertia_calculations.md`, `docs/jetson-mod/task_plan.md`, `docs/print_guide.md`.
8. Add one line to `docs/jetson-mod/experiment_journal.md` recording the new plant mass and the date, because the next training run's numbers are only interpretable against it.
9. **State plainly in the commit message and in `known_issues.md`:** every existing policy — including the shipped `exported_policies/v5d_contact_wrench_ppo/` — was trained on a plant that no longer exists, first because of PLANT-1 and now again because of Phase M. Their gate numbers are stale. Re-gating and retraining are out of scope for Phase M and belong to the next phase.

**3. Unit tests**

None new. This task's job is to run the existing ones, and the two that matter (`test_usd_not_stale`, `test_usd_meshes_not_stale`) already execute real hash comparisons rather than grepping source. If either fails after step 3, the conversion did not happen or a generator ran afterwards and re-wrote a mesh — re-run step 3 last.

**4. Smoke test**

The gate itself is the smoke test. Specific observables, all from `docs/jetson-mod/audit_plant_mass_phaseM.txt`:

- A final line matching `VERDICT: PASS — simulated total is ±0.000000 kg (±0.0%) versus the MJCF (tolerance 0.001 kg)`. **Either sign is a pass** — the recorded PLANT-1 fix printed `-0.000000 kg (-0.0%)`.
- Shell exit status `0` from step 5.
- No line containing `NO <inertial> IN MJCF` and no line containing `unauthored body`.
- The `TOTAL` row's MJCF and PhysX columns agree to six decimal places and equal `tests/fixtures/expected_values.json`'s `total_mass_kg`.

**5. Done when**

- [ ] `pytest tests/test_usd_conversion.py -v` all-green, **including both staleness tests**. (Use these, not file modification times: `generate_cad_mods.py` rewrites its STLs on every run even when the bytes are identical, so an mtime comparison gives false failures. The hash tests compare bytes.)
- [ ] `usd/open_duck_mini_v2.usd`, `usd/.asset_hash`, `usd/.mesh_manifest.json` and `usd/config.yaml` are all committed from the same conversion run, and the manifest holds the expected mesh count (47, or 48 after M3).
- [ ] `audit_plant_mass.py` exits **0** and its captured output is committed at `docs/jetson-mod/audit_plant_mass_phaseM.txt`.
- [ ] Zero `unauthored body` lines, zero `NO <inertial> IN MJCF` lines and zero `possibly invalid inertia tensor` warnings in that output.
- [ ] The mass quoted in `AGENTS.md`, `known_issues.md`, `component_layout_v2.md`, `tests/fixtures/expected_values.json` and the audit output are the **same number**, checked against all five.
- [ ] A journal line records the new plant mass and the date.
- [ ] The staleness of every existing policy's gate numbers is stated in writing.

---

### Task M7 — Decide the fate of `robot.urdf` — ✅ **DONE 2026-08-12 (branch A)**

> **Completed. Do not re-run this task.** The owner chose branch (A), fix and
> gate. What landed:
>
> - **266 mesh references** rewritten from `package:///<name>.stl` (an empty
>   package name, unresolvable by any consumer) to plain paths relative to the
>   URDF. All 47 distinct meshes verified present on disk.
> - **The four invalid inertials removed.** `trunk`, `left_foot`, `right_foot`
>   and `head` are pure kinematic frames; their zero-diagonal tensors were
>   illegal URDF. They now carry no `<inertial>` at all, which the `urdf` skill
>   endorses for frame-only links — rather than inventing an epsilon.
> - **A design ledger** added at the top of the file, recording units, frame
>   semantics, topology, the repairs, and — explicitly — that the inertials are
>   inherited from the Onshape export and are NOT certified by this task.
> - **`tests/test_urdf_consistency.py`** added: **10 tests** that PARSE both
>   descriptions and compare movable joint names, joint parent/child topology,
>   **joint limits and axes**, the body/link set, the root link, **per-link mass,
>   centre of mass and full inertia tensors**, total mass, and mesh resolution.
>
> **Scope of that test, stated precisely — the next agent needs this.**
> An earlier draft of this task said inertia tensors and joint limits would be
> left uncompared because Phase M is about to change them. That reasoning was
> wrong and the measurement disproved it: the two files **already agree
> exactly**. `trunk_assembly`'s full tensor and CoM match to every digit;
> per-link masses differ by at most **1.2e-8 kg**; joint limits by at most
> **4.0e-6 rad**, which is just the MJCF exporting to 6 significant figures
> (URDF `1.1344640137963142` → MJCF `1.13446`). So the comparison is enforced,
> with tolerances derived from those measurements rather than guessed.
>
> **This means the suite WILL fail when Phase M changes the mass model. That is
> the intended signal — regenerate the URDF inertials. Do not loosen the
> tolerance to make it pass.**
>
> What it deliberately does **not** cover, and why:
> 1. **Actuator parameters** (gear, damping, armature, frictionloss) — MuJoCo
>    concepts with no URDF equivalent. The simulated actuator model lives in
>    `robot_cfg.py`, so there is nothing to cross-check between the two files.
> 2. **The USD** — generated from the MJCF and verified separately by
>    `scripts/audit_plant_mass.py`, which reads back what PhysX actually loaded.
>    That catches a class of defect this test structurally cannot: in PLANT-1
>    every file on disk was correct and only the *loaded* plant was wrong.
> 3. **The four frame links** — no `<inertial>` in the URDF vs `mass=1e-09` in
>    the MJCF. An intentional representation difference, documented in the URDF
>    ledger, skipped rather than treated as drift.
>
> Validator result: **0 errors.** The remaining warnings are only the deliberate
> `<mujoco>` and `<joint_properties>` MuJoCo extensions.
>
> ```
> OK robot 'onshape', root 'trunk_assembly', 21 links, 20 joints (16 movable),
>    266 resolved mesh references, total mass 2.657 kg
> ```
>
> **That mass is an independent confirmation of the MJCF**, which declares
> 2.657067 kg — two descriptions produced by different toolchains agreeing.
>
> The URDF now **renders**: `urdf scripts/snapshot` produces a coherent robot
> where it previously failed with `No link mesh loaded for robot`.
>
> The consistency suite was **mutation-tested**, because a test that cannot fail
> is worse than no test. **Ten independent mutations, each caught by exactly one
> assertion, all reverted**, and the suite green again afterwards: rename a joint
> in the URDF only; restore one `package:///` URI; rewire a joint's parent; point
> a mesh at a missing file; delete a link; change a link's mass by 1 g; perturb
> one inertia component; shift a CoM by 1 mm; widen a joint limit by 0.01 rad;
> flip a joint axis sign.
>
> The original task text follows, kept as the record of the decision.

#### M7 — the original task text (SUPERSEDED, kept only as the record of the decision)

> This is not a task. It is what M7 said before it was executed, retained so the
> reasoning behind branch (A) is auditable. **Do not implement from it.**

**AI-agent suitable:** YES for the whole task. The *decision* in step 1 is the
owner's if they want the URDF kept as a supported artifact; the agent can make
the recommendation and do every edit either way.

**1. Context for the implementing agent**

`mini_bdx/robots/open_duck_mini_v2/robot.urdf` is an Onshape export that sits in
the repo, is **not used by anything**, and **does not validate**. All three
claims are checkable:

*Not used.* Every `.urdf` reference in live code points at a **different robot** —
`mini_bdx/robots/bdx/robot.urdf`, the legacy 15-joint BDX — and all of them are
under `experiments/`, which `AGENTS.md` labels "Legacy MuJoCo-based experiment
scripts (reference only)". The Isaac pipeline builds its USD from
`robot_motors.xml` via `scripts/convert_mjcf_to_usd.py`; the URDF is in no code
path.

```bash
grep -rn "\.urdf" --include=*.py --include=*.yaml --include=*.sh . | grep -v experiments/
```

*Does not validate.* Using the `urdf` skill's validator:

```bash
cd ~/.claude/skills/urdf
python3 scripts/validate <repo>/mini_bdx/robots/open_duck_mini_v2/robot.urdf
```

gives **244 blocking findings** in two classes:

| Count | Code | What it is |
|---|---|---|
| 240 | `invalid_mesh_uri` | every mesh is `package:///name.stl` — a `package://` URI with an **empty package name**. Malformed; RViz, MoveIt and Gazebo cannot resolve it. |
| 4 | `nonpositive_inertia_diagonal` | `trunk`, `left_foot`, `right_foot`, `head` have a zero inertia diagonal |
| 17 (warn) | `unknown_element` | `<mujoco>` and `<joint_properties>` — **intentional** MuJoCo extensions, not defects. Leave them. |

*The four zero-inertia links are the same four marker frames as in the MJCF*
(`known_issues.md` PLANT-1b). Both descriptions inherited the defect from the
same Onshape export. On the PhysX side this turned out to be benign — PhysX
substitutes `4.000e-12` — but a URDF consumer has no such fallback, and the
validator is right to block.

**One thing the URDF gets right that the MJCF got wrong.** Its root link is
`trunk_assembly`, with 21 links and **no `base` wrapper at all** — which is
exactly what the PLANT-1 fix made the MJCF on 2026-08-11. The URDF was never
wrong about this; the MJCF's massless `base` was the anomaly. That makes the
URDF a genuinely useful cross-check, which is the main argument for keeping it.

**Skills:** use the `urdf` skill for every step (it owns `scripts/validate` and
`scripts/snapshot`), and hand the file to `cad-viewer` afterwards — the `urdf`
skill requires that handoff.

**2. Low-level implementation plan**

1. **Decide, and write the decision down.** Two defensible options:
   - **(A) Fix and gate it** — recommended. It is cheap (one URI rewrite plus
     four inertials), and it buys an independent check on the MJCF that has
     already proved its worth once.
   - **(B) Mark it non-authoritative** — add a header comment saying it is an
     unmaintained Onshape export, not in any code path, and not to be used as a
     source of truth; then leave it alone. Choose this only if nobody wants a
     ROS/MoveIt path.
   Record the choice in `docs/jetson-mod/print_process_decision.md` or a sibling
   decision doc, with the date.
2. **(A only) Fix the mesh URIs.** Decide the scheme first — either a real
   package name (`package://open_duck_mini_v2/meshes/foo.stl`) or plain relative
   paths (`foo.stl`), which is what actually works given the STLs sit beside the
   URDF. Relative is simpler and needs no ROS package. Apply to all 240.
3. **(A only) Fix the four inertials.** `trunk`, `left_foot`, `right_foot` and
   `head` are pure frame links. Per the `urdf` skill, a frame-only link may
   omit mass and geometry entirely — that is cleaner than inventing a tiny
   nonzero tensor, and it removes the finding at the source. If a consumer needs
   them to be links, give them a small positive diagonal and say why in the
   ledger.
4. **(A only) Add the design ledger** the `urdf` skill requires: a comment block
   at the top of the `.urdf` recording frames, joint axes, units, mesh scale and
   assumptions.
5. **Re-validate to clean**, then snapshot and hand to `cad-viewer`.
6. **(A only) Decide whether it is kept in sync.** A second description that
   drifts is worse than none. Either add a CI check that the URDF and MJCF agree
   on link count, joint names, joint types and parent/child topology, or state
   in the header that it is a point-in-time export. **Do not leave this
   unstated** — that is exactly how the two descriptions in this repo drifted in
   the first place.

**3. Unit tests**

If option (A): add `tests/test_urdf_consistency.py` that **parses both files**
(no source-text grepping — `known_issues.md` TEST-1) and asserts the URDF and
`robot_motors.xml` agree on: the set of joint names, each joint's type, each
joint's parent and child link, and the total link count. That test is the thing
that stops the two descriptions drifting.

If option (B): none applicable — an explicitly unmaintained file needs no test.

**4. Smoke test**

```bash
cd ~/.claude/skills/urdf
python3 scripts/validate <repo>/mini_bdx/robots/open_duck_mini_v2/robot.urdf
```

Observable, option (A): `0 blocking findings` (the 17 `<mujoco>` /
`<joint_properties>` warnings may remain — they are intentional). Option (B):
the count is unchanged and the header comment is present.

**5. Done when**

- [ ] The decision is written down with its date and reasoning
- [ ] **(A)** the validator reports zero *errors*; the remaining warnings are
      only `unknown_element` for `<mujoco>` / `<joint_properties>`
- [ ] **(A)** every mesh URI resolves to a file that exists on disk
- [ ] **(A)** `tests/test_urdf_consistency.py` passes and genuinely fails if a
      joint is renamed in one file only — verify by mutating one name, running,
      and reverting
- [ ] **(B)** the header states plainly that the file is unmaintained, is in no
      code path, and is not a source of truth
- [ ] The file was handed to `cad-viewer` and the link reported
- [ ] `python3 -m pytest tests/ -q` still passes

---

# Phase R — Re-gate and retrain on the corrected plant

> **Where this block goes.** Insert it at the the end of the Phase R section marker in
> `docs/jetson-mod/task_plan_v2.md`. That document already names two of these
> tasks in its cross-references — "R1 re-gate v5d on the corrected plant" in
> *Phase order, and what gates what*, and "the retrain is Task R2" in the
> done-when lists of **M0** and **M0b**. The task ids below preserve those two
> names. Do not renumber them.

> **Why this phase exists.** On 2026-08-11 commit `11b1690` merged the massless
> MJCF root `base` into `trunk_assembly`. PhysX now simulates **2.657067 kg**
> instead of 3.657067 kg (`scripts/audit_plant_mass.py` exits 0). Every policy
> from v1 through the shipped `v5d_contact_wrench` was trained and gated on the
> 3.657 kg plant. Their checkpoints still *load* (obs/action are still 59/16),
> but **every published gate number describes a robot that no longer exists in
> the simulator.** `docs/jetson-mod/known_issues.md` PLANT-1 closes with:
> *"Re-running the gates on the corrected plant is the first task of the
> rebuild."* `docs/jetson-mod/v5_retrain_plan.md` §12 pre-authorised exactly
> this work: *"a separate model-correction task … then retrain the winning
> recipe on the corrected plant and re-run the full gate protocol into a new
> per-model results dir."* This is that phase.

## The ordering constraint that outranks everything else in this phase

This phase straddles Phase M. Read this before starting any task.

```
R0   tooling (no GPU-hours at risk)
 │
R1   re-gate v5d           ─┐
R1b  re-gate v4_robust      │  ALL of these MUST finish BEFORE Phase M lands.
R1c  verdict                │
 │                         ─┘
Phase M  (M0 batched plant fixes, M0b antenna removal, M2 PLANT-10)
 │
R2   train v6_robust from scratch   ─┐
R2b  fine-tune v6d_contact_wrench    │  ALL of these MUST run AFTER Phase M.
R3   export + deployment contract    │
 │                                  ─┘
R4   journal backfill (CPU-only; may be done in any GPU gap)
```

Two hard reasons, both verified:

1. **After Phase M's Task M0b the v5d and v4_robust checkpoints stop loading.**
   M0b takes the observation space from 59 → **53** dims and the action space
   from 16 → **14**. `OnPolicyRunner.load()` will fail on the shape mismatch, so
   R1 and R1b become *impossible* the moment M0b lands. R1 is a few GPU-hours
   and it is the only chance to answer "did fixing the phantom kilogram alone
   break the shipped policy?".
2. **`task_plan_v2.md` states the project's central decision: "A retrain is
   already forced. Spend it once, on everything."** There are 15 open
   plant-fidelity defects. Training `v6_robust` before Phase M would burn ~7
   GPU-hours on a policy Phase M immediately invalidates. R2/R2b therefore run
   *after* M0, M0b and M2, not before.

**If you are the agent executing R1 and Phase M has already landed, stop and
report.** Do not "adapt" the checkpoint. The re-gate is no longer possible and
that is a finding, not a task to improvise around.

## Constants every task in this phase uses verbatim

| Name | Value |
|---|---|
| `REPO` | `/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson` |
| `ISAACLAB` | `/home/xiaohui_chen/IsaacLab` |
| Results dir for R1/R1b (PLANT-1-only plant) | `$REPO/docs/jetson-mod/eval_results_m2657/` |
| Comparison table for R1/R1b | `$REPO/docs/jetson-mod/m2657_comparison.md` |
| Verdict doc for R1c | `$REPO/docs/jetson-mod/m2657_regate.md` |
| Results dir for R2/R2b (post-Phase-M plant) | `$REPO/docs/jetson-mod/eval_results_rebuild/` |
| Comparison table for R2/R2b | `$REPO/docs/jetson-mod/rebuild_comparison.md` |
| Results doc for R2/R2b | `$REPO/docs/jetson-mod/rebuild_results.md` |
| Conditions string (6 conditions, the v5 protocol) | `0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2;0,0,0.5` |
| v5d checkpoint (pinned) | `$REPO/exported_policies/v5d_contact_wrench_ppo/model_5998.pt`, md5 `0333e68a4cd9ed3817310ed80f6715e4` (verified 2026-08-11) |
| v4_robust checkpoint | `$ISAACLAB/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt` |
| USD asset hash of the PLANT-1-corrected model | `10ab887fe4d412b22d3d7c857a9d7f12` (`mini_bdx/robots/open_duck_mini_v2/usd/.asset_hash`, no trailing newline) |

**`m2657` means "the 2.657 kg, PLANT-1-only plant".** It is **not** a name for
the post-Phase-M plant. Phase M's Task M2 re-derives the trunk inertial for
PLANT-10 and Task M0b may change the obs/action dims, so the post-M robot is a
**different model** even if its mass happens to land back on 2.657067 kg. That is
why R2/R2b get their own directory. The repo rule is one results dir + one table
per robot model, never mixed (`AGENTS.md` "Locomotion Policy Evaluation
Protocol" rule 1, and the `### Companion documents` block at `AGENTS.md:761`).

**Do not write any Phase-R result into `eval_results_v4/`, `eval_results_v5/`,
`v4_comparison.md` or `v5_comparison.md`.** Those are the 3.657 kg model's
record.

## Standing rules that apply to every task here (from AGENTS.md)

Rule numbers below are `AGENTS.md` → "Experiment Journal Protocol" → "Data-sourcing
rules (non-negotiable)".

1. **ONE Isaac Sim / GPU job at a time** (rule 7). Never start an eval while a
   training run is alive. Two Isaac processes collide during Kit startup and the
   second dies in its init banner.
2. **Elapsed time is not completion** (rule 6). A training run is done only when
   its process group is dead *and* the final checkpoint file exists.
3. **All RL training runs pass `--video --video_length 200 --video_interval 5000`.**
   This is a standing owner rule; never drop `--video` to save time.
4. **Every training run gets an `experiment_journal.md` entry before the next
   run launches**, with numbers sourced from TensorBoard last-100 means, not
   console greps (rule 1). The tool for that is built in R0, not R4.
5. **Video verdict outranks metrics.** `amp_command7` (Run 12) passed every
   aggregate metric while crawling; only the video caught it.

## Gate thresholds this phase is judged against (frozen here; do not re-derive)

Sources: `AGENTS.md` "Locomotion Policy Evaluation Protocol" step 3,
`scripts/v5_pipeline.sh` (`if [ "$PASSED" -lt 5 ]`), and
`docs/jetson-mod/v5_retrain_plan.md` §7 (acceptance gates) and §17
("What 'beats v4_robust' has to mean").

| Gate | Bar |
|---|---|
| **G-R1** gait validity | ≥ **5 of 6** conditions gait-valid (both feet's stance duty inside `GAIT_DUTY_BAND_PCT = (40.0, 90.0)`, `scripts/evaluate_policies.py:167`). AGENTS.md's "≥ 4/5" is the same rule stated for the 5-condition grid |
| **G-R2** falls, open field | aggregate `fall_rate_pct` < **1.0 %** (AGENTS.md); v5_retrain_plan §7 gate 1 tightens the bar to ≤ **0.5 %** |
| **G-R3** video audit | PASS in ≥ 2 conditions (AGENTS.md mandates forward `vx=0.2` **and** turn `wz=0.3`), judged frame-by-frame against the fixed checklist |
| **G-R4** contact battery | each of the four fall rates ≤ the control measured under the **same protocol on the same plant** (R1b for the m2657 campaign, R2 for the rebuild campaign). Relative gate — an absolute number alone means nothing here |
| **G-R5** open-field regression | `reference_tracking_rms_deg` within **1.0°** of the same-plant control (v5_retrain_plan §7 gate 1) |

**Reference numbers from the OLD 3.657 kg plant** (verified by reading
`docs/jetson-mod/eval_results_v5/*.json` on 2026-08-11). These are for delta
reporting only — they are *not* pass bars on any corrected plant:

| Metric | v4_robust (3.657 kg) | v5d (3.657 kg) |
|---|---|---|
| Gait gate | 6/6 | 6/6 |
| Falls (open field) | 0.000 % | 0.000 % |
| Ref RMS (deg) | 4.478 | 4.669 |
| Duty L/R (%) | 68.7 / 63.7 | 71.5 / 73.9 |
| Duty asym (pp) | 4.99 | 3.60 |
| wz err (rad/s) | 0.067 | 0.098 |
| Energy (W) | 20.85 | 20.33 |
| Jerk | 0.0720 | 0.0836 |
| Push, v4 fall rule (%) | 6.354 | 1.068 |
| Push, v5 fall rule (%) | 11.120 | 0.000 |
| Sustained wrench (%) | 100.000 | 47.109 |
| Obstacle graze (%) | 32.604 | 0.312 |

## Plant-coupled effects that change what the eval measures

Name these in every write-up; they are not bugs, they are consequences of the
fix, and someone will otherwise read a moved number as a policy regression.

- **The wrench got weaker in newtons.** `isaac_lab_env/open_duck_mini_v2/contact_events.py`
  (the `ContactRegimeEvent.__init__` block around lines 108-113) computes
  `self._weight_n = float(self._robot.data.default_mass[0].sum()) * 9.81`. The
  event's banner line now reads `measured robot weight 26.07 N (2.657 kg)`; it
  read `35.88 N (3.657 kg)` before — that exact old line is in
  `.training_runs/v5d_contact_wrench.log:77`. `OpenDuckWrenchEvalEnvCfg`
  (`env_cfg.py:948-961`) presses at `force_frac_range = (0.2, 0.2)` with
  `active_frac = 0.9`, so the press drops from **7.18 N to 5.21 N** while the
  robot it pushes drops by the same 27 %. The *dimensionless* press ratio
  (0.2 × body weight) is preserved, which is what makes the gate comparable; the
  absolute newtons are not.
- **The obstacle did not shrink.** `OpenDuckObstacleEvalEnvCfg`
  (`env_cfg.py:964-976`) sets `obstacle_frac = 1.0` and
  `obstacle_lateral_range = (0.10, 0.15)` — fixed geometry, unchanged. A
  27 %-lighter robot meets the same box with less momentum. Expect
  obstacle-graze numbers to move for a reason that has nothing to do with the
  policy. Also note **CFG-2**: the obstacle is placed *twice per episode* with
  two independent draws, and that defect is still open during R1/R1b (Phase M
  Task M0 step 7 fixes it). It is the same defect on both sides of the R1-vs-R1b
  comparison, so the comparison is still controlled — but the absolute obstacle
  number is not trustworthy in either campaign until M0 lands.
- **`add_base_mass` DR is `(-0.10, +0.15)` kg on `trunk_assembly`**
  (`env_cfg.py:362-371`). As a fraction of the plant that was −2.7 %/+4.1 %; it
  is now −3.8 %/+5.6 %. Leave it alone for R1/R1b — changing it would stop this
  being a controlled re-run — but record it.
- **`torque_z_range` in the wrench recipe is dead code** (**CFG-1**): the warp
  kernel behind `set_forces_and_torques_at_position` assigns torque twice, so
  passing `positions` discards it. It was dead for v5d too, so it does not break
  the comparison — but do not describe the wrench as applying a yaw torque until
  Phase M Task M0 step 6 fixes it.
- **`Gait/duty_in_band_frac` is measured off the ContactSensor history buffer**,
  which spans 15 ms not 60 ms (**CFG-5**), so the in-training canary reads a
  systematically *higher* duty than `evaluate_policies.py` does. Use it as a
  training watchdog only; never quote it as a gate number.

## GPU budget (estimates, not commitments)

| Task | GPU jobs | Wall clock (est.) |
|---|---|---|
| R0 | 1 short audit + 1 short probe | ~15 min |
| R1 | 5 evals + 5 play/video runs | ~3.5 h |
| R1b | 5 evals | ~2.5 h |
| R1c | 0 | ~30 min |
| R2 | 1 training + 1 eval | ~2.5 h |
| R2b | 1 smoke training + 1 training + 5 evals + 4 videos | ~6 h |
| R3 | 1 play/export + 1 contract probe (no Isaac app) | ~20 min |
| R4 | 0 | ~2 h |

A single 6-condition `evaluate_policies.py` invocation (10 windows × 64 envs ×
30 s = 3,840 episodes) took roughly 30 minutes in the v5 campaign. Nothing in
R1–R2b may run concurrently.

**Known limitation to state in every artefact this phase produces:**
`known_issues.md` **PLANT-10** is still open at the time R1 runs — the Part-2 CAD
delta is booked at −88.48 g but measures −6.60 g, so `trunk_assembly` is
54–82 g light and the "corrected" 2.657 kg plant is itself provisional. Phase R
corrects a 37.6 % error; PLANT-10 is a further ~2 %, and Phase M Task M2 owns it.

---

### Task R0 — Make the evaluation and journaling tooling safe for a second robot model — ✅ **DONE 2026-08-12**

> **Completed. Do not re-run this task.** All five done-when boxes are ticked
> below and every number in them was produced by running the command named.
> What landed:
>
> - **EVAL-1 fixed.** `write_comparison_markdown()` takes `include=None`,
>   exposed as the repeatable `--include NAME` CLI flag. Default unchanged, so
>   `v4_comparison.md` / `v5_comparison.md` are undisturbed. The check is retired
>   from `scripts/verify_known_issues.py` and the register entry carries a
>   `FIXED 2026-08-12` block. Verifier now reads `CONFIRMED 27 / 28`.
> - **Every eval JSON now records its plant.** `evaluate_policy()` reads
>   `simulated_total_mass_kg`, `root_body`, `num_bodies`, `body_names`,
>   `joint_order`, `obs_dim`, `action_dim`, `mjcf_sha256` and `usd_asset_hash`
>   off the *running articulation* and stamps them into the result under
>   `"plant"`.
> - **The generated header renders provenance from the entries** (EVAL-2's
>   generator half): plant mass(es), obs/action dims and condition counts. Two
>   masses on that line means the table is mixing robot models.
> - **`scripts/verify_action_contract.py`** (new GPU probe) and
>   **`scripts/tb_summary.py`** (new, Isaac interpreter only).
> - **`docs/jetson-mod/eval_results_m2657/README.md`** — plant identity, frozen
>   protocol, gate table G-R1…G-R5, plant-coupled effects.
> - **`tests/test_eval_report_filter.py`** — 7 tests that run the real script in
>   a subprocess rather than grepping it (TEST-1 is why).
>
> **Measured smoke-test output, 2026-08-12:**
>
> ```
> audit_plant_mass.py      TOTAL 2.657067 / 2.657067   VERDICT: PASS   exit 0
> verify_action_contract   joint order MATCHES duck_init_pos.json (16 joints)
>                          worst default-pos delta 5.150e-08 rad on 'right_knee'
>                          obs 59, action 16, root trunk_assembly, 21 bodies
>                          VERDICT: PASS                              exit 0
> tb_summary.py            Gait/duty_in_band_frac last-100 mean 0.9836
>                          (open_duck_ppo_v5/2026-07-29_08-59-25, wall 2:05:02)
> pytest tests/ -q         120 passed
> ```
>
> **The joint order did NOT move.** That was the open question the probe
> existed to answer — the PLANT-1 fix removed a body, and nothing had verified
> that the 16 actuators still come out of the USD in the order
> `duck_init_pos.json` records. They do, and the default positions agree to
> 5.15e-08 rad. Phase R may proceed.
>
> **Still open on purpose:** step 4 of the plan below (hand-editing the
> `PROTOCOL_HEADER` preamble of a *new* comparison table) cannot run until R1
> generates `m2657_comparison.md` for the first time. It is listed in R1's work,
> not left undone here.

**AI-agent suitable:** YES

**1. Context for the implementing agent**

Four things must be true before a single GPU-hour is spent, and none of them is
true today.

- **The comparison-table generator injects every JSON in the directory** with no
  filter (`known_issues.md` **EVAL-1**, still CONFIRMED by
  `python3 scripts/verify_known_issues.py`). This has already fired once:
  `eval_results_v4/` contains `v4_robust_pusheval.json` and
  `v4_inertials_pusheval.json`, while `v4_comparison.md` says "no external
  pushes". The Phase-R results dirs will hold grid JSONs *and*
  push/wrench/obstacle JSONs side by side, so regenerating the comparison table
  will mix push-eval rows into a table documented as push-free — on day one, by
  construction.
- **No eval JSON records which plant it was measured on.** Verified: the
  top-level keys of `eval_results_v5/v5d_contact_wrench.json` are exactly
  `name, task_id, framework, checkpoint, agent_cfg, evaluated_at, protocol,
  per_condition, aggregate`. That missing field is precisely what let PLANT-1
  survive three policy generations.
- **The joint order may have moved.** The PLANT-1 fix removed a body. (Two
  different counts are quoted in the repo and they are both right: MuJoCo's
  `nbody` went **24 → 23** because it counts the worldbody, while the Isaac Lab
  articulation went **22 → 21** rigid bodies. Expect **21**.) The commit
  verified obs/action stay 59/16 — it did **not** verify that the 16 actuators
  still come out of the USD in the same order. If the order moved,
  `scripts/duck_init_pos.json` (which defines `q_target = init_pos + 0.25·action`
  for the whole project) is silently wrong and every number produced downstream
  is garbage. Check it before, not after.
- **There is no tool that produces last-100-iteration TensorBoard means.**
  `AGENTS.md` rule 1 makes those mandatory in every journal entry, and R2's
  done-when requires a journal entry *before* R2b launches. So the tool has to
  exist now, not in R4. **Measured: `tensorboard` is NOT importable from the
  system `python3`** (`ModuleNotFoundError: No module named 'tensorboard'`) but
  **is** importable from `~/IsaacLab/_isaac_sim/python.sh`. Plan for that.

Files to read first:

- `scripts/evaluate_policies.py` (1,610 lines) — read the module docstring
  (lines 60-81), then `write_comparison_markdown` (line 740),
  `render_results_section` (line 646), `build_arg_parser` (line 781), the
  `--report-only` short-circuit (lines 1127-1134), `evaluate_policy` (line 1505,
  note the returned dict at the end) and `main` (line 1574).
  **Trap: the file must never be imported** — everything below line ~1140
  launches Isaac Sim at module scope. The pure-python prefix is reachable only
  via `--self-test` / `--report-only`, which are dispatched by literal
  `sys.argv` checks (`if __name__ == "__main__" and "--self-test" in sys.argv`)
  and `sys.exit()` before the AppLauncher block.
- `docs/jetson-mod/known_issues.md` — sections EVAL-1 and EVAL-2.
- `scripts/duck_init_pos.json` — a 16-entry `joint_order` list **and** an
  `init_pos_rad` **dict keyed by joint name** (not a parallel list). Verified.
- `scripts/audit_plant_mass.py` (175 lines) — the shape of a GPU probe script
  that exits non-zero as a gate. Copy its structure exactly: argparse →
  `AppLauncher.add_app_launcher_args(parser)` → build the app → *then* import
  isaaclab; and copy its `sys.stdout.flush(); sys.stderr.flush(); os._exit(code)`
  ending, which the file documents as mandatory (`simulation_app.close()` never
  returns and pins the exit status to 0; Kit leaves stdout block-buffered on
  redirect).

Depends on: nothing. This is the first task of the phase.

**2. Low-level implementation plan**

1. **`scripts/evaluate_policies.py` — add an allowlist (fixes EVAL-1).**
   - In `build_arg_parser()`, add after the `--comparison_md` argument
     (currently at line 838):
     ```python
     parser.add_argument(
         "--include", action="append", default=None, metavar="NAME",
         help=("Only these policy names enter the comparison table "
               "(repeatable, exact match on the JSON's 'name'). Without it "
               "EVERY *.json in --output_dir is injected, which mixes "
               "push/wrench/obstacle rows into a table documented as "
               "push-free -- see known_issues.md EVAL-1."),
     )
     ```
     Note `allow_abbrev=False` is already set on this parser, so `--incl` will
     not silently work; always spell `--include` in full.
   - Change the signature to
     `def write_comparison_markdown(md_path: str, results_dir: str, include=None) -> str:`
     and insert, immediately after the `for fname in sorted(os.listdir(...))`
     loop that fills `entries` (lines 747-752):
     ```python
     if include:
         wanted = set(include)
         entries = [e for e in entries if e.get("name") in wanted]
     ```
   - Update both call sites: the `--report-only` block at line 1132 becomes
     `write_comparison_markdown(_args.comparison_md, _args.output_dir, include=_args.include)`,
     and the last line of `main()` (line 1607) becomes
     `write_comparison_markdown(args_cli.comparison_md, output_dir, include=args_cli.include)`.
   - Default stays `None` = current behaviour, so `v4_comparison.md` and
     `v5_comparison.md` regeneration is unchanged.

2. **`scripts/evaluate_policies.py` — stamp the plant into every JSON.**
   - Add near the other module constants, after `GAIT_DUTY_BAND_PCT` (line 167):
     ```python
     MJCF_PATH = os.path.join(REPO_ROOT, "mini_bdx", "robots",
                              "open_duck_mini_v2", "robot_motors.xml")
     USD_ASSET_HASH_PATH = os.path.join(REPO_ROOT, "mini_bdx", "robots",
                                        "open_duck_mini_v2", "usd", ".asset_hash")
     ```
   - Add a pure helper next to `_fmt` (line 620):
     ```python
     def _file_sha256(path: str) -> str:
         import hashlib
         h = hashlib.sha256()
         with open(path, "rb") as f:
             for chunk in iter(lambda: f.read(1 << 20), b""):
                 h.update(chunk)
         return h.hexdigest()
     ```
   - In `evaluate_policy()`, before the `return {` (line 1547), add — note
     `base_env` is `gym_env.unwrapped`, a `ManagerBasedRLEnv`, and the canonical
     accessors are `robot.data.body_names` / `robot.data.joint_names`, which is
     what `audit_plant_mass.py` uses:
     ```python
     robot = base_env.scene["robot"]
     body_names = list(robot.data.body_names)
     plant = {
         "simulated_total_mass_kg": float(robot.data.default_mass[0].sum()),
         "root_body": body_names[0],          # PhysX body order; root is index 0
         "num_bodies": len(body_names),
         "body_names": body_names,
         "joint_order": list(robot.data.joint_names),
         "obs_dim": int(base_env.observation_manager.group_obs_dim["policy"][0]),
         "action_dim": int(base_env.action_manager.total_action_dim),
         "mjcf_sha256": _file_sha256(MJCF_PATH),
         "usd_asset_hash": open(USD_ASSET_HASH_PATH).read().strip(),
     }
     ```
     and add `"plant": plant,` to the returned dict, directly after
     `"evaluated_at"`. Expected values on the PLANT-1-corrected plant:
     `simulated_total_mass_kg = 2.657067`, `root_body = "trunk_assembly"`,
     `num_bodies = 21`, `obs_dim = 59`, `action_dim = 16`,
     `usd_asset_hash = "10ab887fe4d412b22d3d7c857a9d7f12"`. **After Phase M's
     M0b these become `obs_dim = 53`, `action_dim = 14`** — that is expected and
     is exactly why the field exists.

3. **`scripts/evaluate_policies.py` — make the generated header honest (EVAL-2).**
   `render_results_section` currently builds `lines` as one list literal
   (lines 648-661). Restructure so the provenance lines can be appended: keep
   the first four entries (`"## Results"`, `""`, the `_Last regenerated: …_`
   line, `""`) in the literal, then append the block below, then append the
   `### Aggregate over all conditions` heading and the table header rows.
   ```python
   plants = sorted({
       f"{e['plant']['simulated_total_mass_kg']:.6f}"
       for e in entries if e.get("plant")
   })
   dims = sorted({
       f"{e['plant'].get('obs_dim', '?')}/{e['plant'].get('action_dim', '?')}"
       for e in entries if e.get("plant")
   })
   conds = sorted({len(e.get("protocol", {}).get("conditions", [])) for e in entries})
   lines += [
       f"_Plant mass(es) simulated: {', '.join(plants) or 'not recorded'} kg — "
       f"obs/action dims: {', '.join(dims) or 'not recorded'} — "
       f"conditions per entry: {sorted(conds) or 'n/a'}_",
       "",
   ]
   ```
   If that line ever shows two masses or two dim pairs, the table is mixing
   robot models and must be split.

4. **Fix the hand-written preamble of any NEW comparison table.**
   `write_comparison_markdown` writes the module constant `PROTOCOL_HEADER`
   (lines 573-615) verbatim when the target `.md` does not yet exist. That
   header hardcodes *five* conditions and the words "no external pushes" — it
   describes the script's defaults, not the run. Everything **above** the
   `<!-- BEGIN AUTO-GENERATED RESULTS ... -->` marker survives regeneration, so
   after the first table is generated in R1, hand-edit that preamble to state
   the six conditions actually used and the plant identity. Do this once per new
   table (`m2657_comparison.md`, `rebuild_comparison.md`).

5. **New file `scripts/verify_action_contract.py`** — a GPU probe, modelled
   line-for-line on `scripts/audit_plant_mass.py`:
   - args: `--task` (default `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0`),
     `--num_envs` (default 2).
   - Build the env with `parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)`
     then `gym.make(TASK, cfg=env_cfg).unwrapped`, `env.reset()`, and
     `robot = env.scene["robot"]` — exactly as `audit_plant_mass.py` does.
   - Load `scripts/duck_init_pos.json`.
   - Assert `list(robot.data.joint_names) == data["joint_order"]`; on mismatch
     print both lists with indices and set exit code 1.
   - For each name `n` at index `i` in `joint_order`, assert
     `abs(float(robot.data.default_joint_pos[0, i]) - data["init_pos_rad"][n]) < 1e-6`.
     (`init_pos_rad` is a dict keyed by joint name; the values match
     `robot_cfg.py:48-68` exactly, and float32 round-trip error is < 1e-7, so
     1e-6 is a safe tolerance.)
   - Assert `env.action_manager.total_action_dim == 16` and
     `env.observation_manager.group_obs_dim["policy"][0] == 59`. Make these two
     numbers CLI-overridable (`--expect_action_dim`, `--expect_obs_dim`) so the
     same script still works after Phase M's M0b changes them to 14/55.
   - Print a single summary block including the literal string
     `joint order MATCHES duck_init_pos.json (16 joints)` on success, then
     `env.close()`, `sys.stdout.flush()`, `os._exit(code)`.

6. **New file `scripts/tb_summary.py`** — moved here from R4 because R2's
   journal entry needs it. Plain stdlib + `tensorboard`:
   - **Run it with `~/IsaacLab/_isaac_sim/python.sh`, not the system `python3`.**
     Measured 2026-08-11: `EventAccumulator` imports fine under the Isaac
     interpreter and fails with `ModuleNotFoundError` under `python3`. Put that
     fact in the module docstring and in the usage string.
   - CLI: `--run_dir`, `--last N` (default 100), `--tags` (repeatable; default a
     built-in list of `Train/mean_reward`, `Train/mean_episode_length`,
     `Gait/duty_in_band_frac`, plus every `Episode_Reward/*` tag found),
     `--markdown`.
   - Load with `EventAccumulator(run_dir, size_guidance={'scalars': 0})` — the
     zero is load-bearing; the default down-samples and silently changes the
     means.
   - Print, per tag: last-N mean, peak value with its step, final step index.
     Also print wall-clock from first-to-last event timestamp (rule 2 — never
     estimate, never reuse another run's number).
   - `--markdown` emits a table ready to paste into a journal entry.

7. **Create the R1 results directory and its provenance stamp.**
   ```bash
   mkdir -p $REPO/docs/jetson-mod/eval_results_m2657
   ```
   Write `$REPO/docs/jetson-mod/eval_results_m2657/README.md` containing: the
   plant this directory belongs to (2.657067 kg, MJCF sha256, USD asset hash
   `10ab887fe4d412b22d3d7c857a9d7f12`, obs/action 59/16, 21 rigid bodies, root
   `trunk_assembly`), the frozen protocol for this campaign (6 conditions ×
   10 windows × 64 envs × 30 s = 3,840 episodes, seed 42, deterministic,
   corruption off, pushes off unless `--keep-pushes`), the gate table G-R1…G-R5
   copied verbatim from this plan, and two sentences: *"No JSON measured on the
   3.657 kg plant may be copied into this directory."* and *"This directory is
   the PLANT-1-only plant. Post-Phase-M results go to `eval_results_rebuild/`."*

**3. Unit tests**

New file `tests/test_eval_report_filter.py`, marked `@pytest.mark.phase2` (the
marker is registered in `pytest.ini`). These run the real code in a subprocess —
they do not grep source. `tests/` is otherwise source-text grepping and 5 of 7
seeded mutations pass it (`known_issues.md` **TEST-1**); do not add to that pile.

- `test_include_filters_the_table`: write two JSONs into `tmp_path` —
  `{"name": "keep_me", "framework": "rsl_rl", "aggregate": {}, "per_condition": {}}`
  and the same with `"drop_me"` — then
  `subprocess.run([sys.executable, "scripts/evaluate_policies.py", "--report-only", "--output_dir", str(tmp_path), "--comparison_md", str(tmp_path/"t.md"), "--include", "keep_me"], cwd=REPO)`.
  Assert returncode 0, `"keep_me" in t.md`, `"drop_me" not in t.md`.
- `test_no_include_keeps_current_behaviour`: same fixtures, no `--include`.
  Assert both names appear (guards the v4/v5 tables against this change).
- `test_plant_line_reports_missing_provenance`: entries with no `plant` key must
  render `not recorded` rather than crash.
- `test_self_test_still_passes`: run `scripts/evaluate_policies.py --self-test`
  and assert returncode 0 — this is the existing pure-numpy metric suite and
  must not be broken by the edits.

**4. Smoke test**

```bash
cd $ISAACLAB && ./isaaclab.sh -p $REPO/scripts/audit_plant_mass.py --headless; echo "audit exit=$?"
sleep 20
cd $ISAACLAB && ./isaaclab.sh -p $REPO/scripts/verify_action_contract.py --headless; echo "contract exit=$?"
~/IsaacLab/_isaac_sim/python.sh $REPO/scripts/tb_summary.py \
  --run_dir $ISAACLAB/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25 --markdown
```
Observable: the first prints a `TOTAL` row with `2.657067` in both the MJCF and
PhysX columns and `VERDICT: PASS`, and exits **0**; the second prints
`joint order MATCHES duck_init_pos.json (16 joints)` and exits **0**; the third
prints a markdown table whose `Gait/duty_in_band_frac` last-100 mean is
**0.9836** (measured from `.training_runs/v5d_contact_wrench.log`; if
`tb_summary.py` disagrees materially, the tag name or the accumulator config is
wrong).

If `verify_action_contract.py` exits 1, **stop the entire phase** and report —
the action mapping for every exported policy is wrong and no eval number is
meaningful until it is resolved.

**5. Done when**

- [ ] `python3 scripts/evaluate_policies.py --self-test` exits 0.
- [ ] `python3 -m pytest tests/test_eval_report_filter.py -q` passes (4 tests).
- [ ] `python3 -m pytest tests/ -q` passes with **at least 103** tests and no
      failures. (103 is the measured baseline on 2026-08-11. Run it scoped to
      `tests/`; a bare `pytest` from the repo root errors at collection —
      `known_issues.md` **TEST-2**.)
- [ ] `audit_plant_mass.py` exits 0 and prints `2.657067`.
- [ ] `verify_action_contract.py` exits 0.
- [ ] `scripts/tb_summary.py` produces a last-100 table for
      `open_duck_ppo_v5/2026-07-29_08-59-25` under the Isaac interpreter.
- [ ] `docs/jetson-mod/eval_results_m2657/README.md` exists and names the plant
      hash, the dims and the frozen protocol.
- [ ] `docs/jetson-mod/known_issues.md` EVAL-1 carries a
      `> **FIXED <date>.**` block describing the `--include` allowlist, its
      check is deleted from `scripts/verify_known_issues.py`, and the
      `CONFIRMED n / m` block near the top of `known_issues.md` is updated to
      the new count. (Precedent: DEPLOY-5 and PLANT-1 were closed exactly this
      way; the register's own "delete both the check and the entry" line at the
      top is superseded by that precedent — keep the entry, retire the check.)

---

### Task R1 — Re-gate the shipped v5d against the corrected plant — ✅ **DONE 2026-08-12**

> **Completed. Do not re-run this task.** All six done-when boxes are ticked.
> Five JSONs in `eval_results_m2657/`, five mp4s + filmstrips under
> `eval_results_m2657/videos/`, and a written per-condition video verdict in
> `eval_results_m2657/videos/AUDIT.md`.
>
> **The answer to the question this task exists to ask: the plant fix alone did
> NOT break the shipped policy's locomotion. It did degrade its disturbance
> rejection.**
>
> | metric | 3.657 kg | 2.657 kg | delta |
> |---|---|---|---|
> | gait valid | 6/6 | **6/6** | — |
> | open-field falls | 0.000 % | **0.000 %** | — |
> | ref RMS (deg) | 4.6689 | **4.6758** | +0.0068 |
> | duty L / R (%) | 71.46 / 73.88 | 65.71 / 65.39 | −5.74 / −8.49 |
> | duty asym (pp) | 3.6049 | 3.2344 | −0.37 |
> | energy (W) | 20.33 | 19.86 | −0.47 |
> | push, **v4 rule** | 1.068 % | **55.443 %** | +54.4 |
> | push, **v5 rule** | 0.000 % | **0.339 %** | +0.34 |
> | sustained wrench | 47.109 % | **43.047 %** | −4.06 |
> | obstacle graze | 0.312 % | **0.885 %** | +0.57 |
>
> **Read the two push rows together or you will misread this result.** The v4
> rule counts *any* `trunk_assembly` contact above 1 N as a fall; the v5 rule is
> the deployment's own definition (tilt > 60°, root height < 0.09 m). v5d still
> **recovers** from the pushes — 0.339 % by the deployment definition. What
> changed is that its recovery now involves the trunk touching down, and the
> legacy rule scores that as death. `env_cfg.py` says why the v4 rule was
> abandoned: it "taught contact = death and never taught recovery".
>
> Ruled out by measurement, not by argument: identical checkpoint
> (md5 `0333e68a4cd9ed3817310ed80f6715e4` for both the `exported_policies/`
> and training-log copies), byte-identical `protocol` block, termination still
> name-bound to `trunk_assembly`, `add_base_mass`/`base_com` both `None` in the
> PLAY chain, and `push_by_setting_velocity` **sets** root velocity so the push
> is mass-independent and did not change between plants. (The wrench, being a
> force, did: 7.18 N → 5.21 N, banner `measured robot weight 26.07 N (2.657 kg)`.)
>
> **G-R3 video audit: PASS.** `forward_vx02`, `turn_wz03` and `turn_wz05` all
> show alternating bipedal gait with real swing clearance, trunk vertical,
> stance foot loaded, and turn-in-place as a stepping rotation rather than a
> pivot-scrape. `press_wrench` shows the robot walking ~12 s then toppling and
> **not self-righting** — v5d has no get-up behaviour, which Phase S bring-up
> must assume. `obstacle_graze` shows one adverse draw; at 0.885 % aggregate it
> illustrates the failure mode, not its frequency, and CFG-2 (obstacle placed
> twice per episode) is still open.
>
> **Also fixed here (R0 step 4, which could not run until this table existed):**
> the hand-written preamble of `m2657_comparison.md` was the generator's
> `PROTOCOL_HEADER`, which describes the script's *defaults* — it claimed five
> conditions and "no external pushes". It now states the six conditions
> actually used, the plant identity, and why the absolute contact numbers must
> not be compared against `v5_comparison.md`.
>
> **Not decided here.** G-R1/G-R2/G-R5 pass and G-R3 passes, but **G-R4 is a
> relative gate** and its control is Task R1b. The verdict is Task R1c.

**AI-agent suitable:** PARTIAL — steps 1-5 are fully autonomous. The **video
audit (step 6) requires something that can actually look at images.** A
text-only agent must render the mp4s and filmstrips and then hand them to the
owner (or to a vision-capable model) rather than assert a verdict it cannot
form. Fabricating a video verdict is the single failure mode this protocol
exists to prevent (Run 12, `amp_command7`, passed every aggregate metric while
crawling).

**1. Context for the implementing agent**

`v5d_contact_wrench` is the shipped locomotion policy. Its entire gate record —
gait 6/6, 0.000 % falls, wrench 47.109 %, obstacle 0.312 % — was measured on a
3.657 kg robot. Until it is re-measured, the project does not know whether it
still walks. Nothing downstream (Phase S runtime, deployment contract, hardware
build) may proceed on the old numbers.

**This task must run before Phase M.** After Task M0b the checkpoint's 59/16
interface no longer matches the env and the run becomes impossible.

Read first:

- `scripts/v5_pipeline.sh` (174 lines) — the exact command shape used to produce
  the numbers being replaced. Read `run_eval()` (the
  `--policies NAME=TASK:rsl_rl:CKPT` form), the `CONDITIONS` string, the
  gate-parsing heredoc, and `play()`.
- `docs/jetson-mod/v5_retrain_plan.md` §22 — the v5d result being re-measured.
- `AGENTS.md` "Locomotion Policy Evaluation Protocol" — the three mandatory
  steps and the "video wins" rule.
- `isaac_lab_env/open_duck_mini_v2/__init__.py` — the task-id registry; the v5
  task ids are registered in the `for _task_id, _cfg_class in (…)` loop starting
  at line 104.

Depends on: **R0** (the `--include` flag and the `plant` JSON block are used by
every command below).

Traps, named explicitly:

- **Do not run `scripts/v5_pipeline.sh` for this task.** It hardcodes
  `LOGROOT`, `RESULTS=$REPO/docs/jetson-mod/eval_results_v5` and
  `--comparison_md "$REPO/docs/jetson-mod/v5_comparison.md"`, and it resolves
  the checkpoint from the newest run dir — none of which apply to an archived
  checkpoint. Run the eval commands directly, as written below.
- **`evaluate_policies.py`'s `--output_dir` default is `eval_results_v4/`**
  (`DEFAULT_OUTPUT_DIR`, line 145) and `--comparison_md` defaults to
  `v4_comparison.md`. Pass both explicitly on every single invocation.
- **The `NAME=` in `--policies` determines the JSON filename.**
  `write_policy_json` writes `<output_dir>/<name>.json`. To get
  `v5d_contact_wrench_wrencheval.json` the spec must read
  `--policies "v5d_contact_wrench_wrencheval=<task>:rsl_rl:<ckpt>"`.
- **Task ids are not interchangeable.** Evaluate each policy on the *same* task
  it used in `eval_results_v5/` so the plant is the only changed variable. For
  v5d the open-field task is `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0`
  (confirmed from `eval_results_v5/v5d_contact_wrench.json` → `task_id`).
- **`--keep-pushes` is required on the two push evals and forbidden on the other
  three.** The protocol default strips `push_robot` /
  `base_external_force_torque`; a push eval without `--keep-pushes` measures
  nothing.
- **`--conditions` must be passed.** The script's default
  (`DEFAULT_CONDITIONS_STR`, line 136) is the legacy **5**-condition string.
- **One Isaac job at a time.** Run these strictly sequentially with a `sleep 20`
  between them (`v5_pipeline.sh` does this for a reason: Kit startup collides).
- **`play_policy.py` writes into the checkpoint's own directory.** Isaac Lab's
  `play.py` sets `log_dir = os.path.dirname(resume_path)` (`play.py:130`) and
  the video folder to `<log_dir>/videos/play` (`play.py:145`); the ONNX/JIT
  export goes to `<log_dir>/exported` (`play.py:172`). Because the pinned v5d
  checkpoint lives in the **git-tracked** `exported_policies/v5d_contact_wrench_ppo/`,
  playing it will create `exported_policies/v5d_contact_wrench_ppo/videos/` and
  `.../exported/` inside the repo. Move the mp4s out and delete the leftover
  directories; do not commit them there.

**2. Low-level implementation plan**

1. Verify the checkpoint pin before spending GPU time:
   ```bash
   md5sum $REPO/exported_policies/v5d_contact_wrench_ppo/model_5998.pt
   # must print 0333e68a4cd9ed3817310ed80f6715e4
   ```
2. Re-run the R0 preflight (`audit_plant_mass.py` exit 0, `verify_action_contract.py`
   exit 0). Do not skip: if anything regenerated the USD in between, everything
   after this is void.
3. **Open-field grid** (the G-R1/G-R2/G-R5 gate):
   ```bash
   cd $ISAACLAB && ./isaaclab.sh -p $REPO/scripts/evaluate_policies.py \
     --policies "v5d_contact_wrench=Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0:rsl_rl:$REPO/exported_policies/v5d_contact_wrench_ppo/model_5998.pt" \
     --conditions "0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2;0,0,0.5" \
     --output_dir $REPO/docs/jetson-mod/eval_results_m2657 \
     --comparison_md $REPO/docs/jetson-mod/m2657_comparison.md \
     --include v5d_contact_wrench --include v4_robust_grid6 \
     --headless 2>&1 | tee $REPO/.training_runs/regate_v5d_grid.log
   ```
   Pass the *same two* `--include` names on every invocation in R1 and R1b —
   the table is regenerated by each run, and those two are the only open-field
   grid entries this campaign will produce (names with no matching file are
   simply skipped). Do **not** add `v6*` names here; those belong to a different
   plant and a different table.
4. **Contact battery** — four more invocations, identical except for the policy
   name, the task id and the extra flag. Keep the JSON basenames identical to
   `eval_results_v5/` so the two directories diff row-for-row:

   | `--policies` NAME (= JSON basename) | task id | extra flag |
   |---|---|---|
   | `v5d_contact_wrench_pusheval_v4def` | `Isaac-Velocity-Rough-OpenDuck-PushEval-v0` | `--keep-pushes` |
   | `v5d_contact_wrench_pusheval_v5def` | `Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0` | `--keep-pushes` |
   | `v5d_contact_wrench_wrencheval` | `Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0` | — |
   | `v5d_contact_wrench_obstacleeval` | `Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0` | — |

5. From the wrench-eval log, capture the `ContactRegimeEvent` banner. On the old
   plant it read (verbatim, `.training_runs/v5d_contact_wrench.log:77`):
   ```
   [v5] ContactRegimeEvent: measured robot weight 35.88 N (3.657 kg) -> wrench 1.79-7.18 N (active_frac=0.5); obstacle asset present, obstacle_frac=0.25
   ```
   It must now report `measured robot weight 26.07 N (2.657 kg)`. Grep for that
   substring rather than the whole line — the tail differs per task. Paste it
   into the R1c verdict doc as the proof that the curriculum leak closed.
6. **Videos** (G-R3). Render **five** clips. The PLAY configs default to
   `lin_vel_x = (0.2, 0.2)`, `ang_vel_z = (0.0, 0.0)`
   (`env_cfg.py:796-798`), so the un-overridden run *is* the forward vx=0.2
   condition:
   ```bash
   cd $ISAACLAB && ./isaaclab.sh -p $REPO/scripts/play_policy.py \
     --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 --num_envs 2 \
     --checkpoint $REPO/exported_policies/v5d_contact_wrench_ppo/model_5998.pt \
     --headless --video --video_length 1000
   ```
   | tag | change from the command above |
   |---|---|
   | `forward_vx02` | none |
   | `turn_wz03` | append `'env.commands.base_velocity.ranges.ang_vel_z=[0.3,0.3]'` (AGENTS.md-mandated condition) |
   | `turn_wz05` | append `'env.commands.base_velocity.ranges.ang_vel_z=[0.5,0.5]'` (the v5 campaign's tag) |
   | `press_wrench` | `--task Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0` |
   | `obstacle_graze` | `--task Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0` |

   After **each** run, immediately rename the newest mp4 — the next run writes
   into the same folder and the pattern to copy is `play()` in
   `v5_pipeline.sh`:
   ```bash
   VDIR=$REPO/exported_policies/v5d_contact_wrench_ppo/videos/play
   NEWEST=$(ls -t "$VDIR"/*.mp4 | head -1)
   mkdir -p $REPO/docs/jetson-mod/eval_results_m2657/videos
   mv "$NEWEST" $REPO/docs/jetson-mod/eval_results_m2657/videos/v5d_m2657_<tag>.mp4
   ```
   When all five are done, remove the now-empty
   `exported_policies/v5d_contact_wrench_ppo/videos/` and any `exported/`
   directory `play.py` created there. (Storing mp4s under
   `docs/jetson-mod/eval_results_*/` is precedented — `eval_results_v4/` tracks
   two.)
7. Extract filmstrips with ffmpeg (`/home/xiaohui_chen/.local/bin/ffmpeg`,
   version 7.0.2, verified present), e.g.
   `ffmpeg -i in.mp4 -vf "fps=1.5,scale=320:-1,tile=6x4" strip.png`, and audit
   against the fixed checklist from `AGENTS.md`: trunk upright near 0.17 m; both
   feet alternate swing with real ground clearance; feet loaded during stance
   (no drag / glide / crawl); heading straight; no action dither; turn-in-place
   is a stepping rotation, not a pivot-scrape.

**3. Unit tests**

New file `tests/test_regate_results.py`, marked `@pytest.mark.phase2`. It parses
the produced JSONs; it does not grep anything. Every test skips cleanly
(`pytest.skip`) when the results dir holds no JSON yet, so it is green before R1
runs and meaningful after. Parameterise the directory over both
`eval_results_m2657/` and `eval_results_rebuild/` so R2/R2b are covered by the
same file, and derive the expected mass from that directory's `README.md` rather
than hardcoding it twice.

- `test_every_json_records_a_plant`: for each `*.json`, assert a `plant` key
  exists with `simulated_total_mass_kg`, `root_body`, `joint_order`, `obs_dim`,
  `action_dim`.
- `test_directory_holds_exactly_one_plant`: assert all JSONs in the directory
  agree on `simulated_total_mass_kg` to 1e-3 **and** on `(obs_dim, action_dim)`.
  This is the per-model rule, mechanised.
- `test_no_stale_plant_leaked_in`: assert no JSON reports a mass within 1e-3 of
  3.657067 (the copy-paste guard).
- `test_m2657_dir_is_the_corrected_plant`: for `eval_results_m2657/` only,
  assert `abs(mass - 2.657067) < 1e-3` and `root_body == "trunk_assembly"`.
- `test_protocol_is_the_frozen_one`: assert
  `len(d["protocol"]["conditions"]) == 6`, `windows_per_condition == 10`,
  `num_envs == 64`, `episode_length_s == 30.0`, `seed == 42` for every JSON.
- `test_joint_order_matches_duck_init_pos`: for `eval_results_m2657/` only,
  assert `d["plant"]["joint_order"]` equals `scripts/duck_init_pos.json`'s
  `joint_order`. (Skip this for `eval_results_rebuild/`: if M0b lands, the
  action set is 14 joints and `duck_init_pos.json` must be regenerated by
  Phase M — flag it rather than assert it.)

**4. Smoke test**

```bash
ls $REPO/docs/jetson-mod/eval_results_m2657/*.json | wc -l   # expect 5 after R1
cd $REPO && python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob('docs/jetson-mod/eval_results_m2657/v5d*.json')):
    d = json.load(open(f)); a = d['aggregate']; p = d['plant']
    print(f"{os.path.basename(f):45s} gate {a['gait_valid_conditions']}/{a['conditions']} "
          f"falls {a['fall_rate_pct']:.3f}%  RMS {a['reference_tracking_rms_deg']:.3f}  "
          f"plant {p['simulated_total_mass_kg']:.6f}  dims {p['obs_dim']}/{p['action_dim']}")
PY
grep -l "measured robot weight 26.07 N (2.657 kg)" $REPO/.training_runs/*wrencheval*.log
```
Observable: five rows, each printing `plant 2.657067` and `dims 59/16`, and the
open-field row (`v5d_contact_wrench.json`) printing a gate count and a fall rate.
That row's `gait_valid_conditions` and `fall_rate_pct` are the numbers R1c judges.

**5. Done when**

- [ ] Exactly 5 JSONs exist in `eval_results_m2657/`, all named
      `v5d_contact_wrench*`.
- [ ] `python3 -m pytest tests/test_regate_results.py -q` passes.
- [ ] `m2657_comparison.md` exists, its plant line reads a single mass
      (`2.657067`) and a single dim pair (`59/16`), and its hand-written
      preamble above the `BEGIN AUTO-GENERATED` marker has been corrected to say
      six conditions (R0 step 4).
- [ ] The wrench eval log contains `measured robot weight 26.07 N (2.657 kg)`.
- [ ] Five mp4s exist under `eval_results_m2657/videos/` with their filmstrip
      PNGs, and `exported_policies/v5d_contact_wrench_ppo/` contains no
      `videos/` or `exported/` directory (`git status` clean apart from the
      intended additions).
- [ ] A written per-condition video verdict exists for at least `forward vx=0.2`
      **and** `turn wz=0.3` — or an explicit, recorded statement that the audit
      is pending because no vision-capable reviewer was available.

---

### Task R1b — Re-measure the v4_robust control battery on the corrected plant — ✅ **DONE 2026-08-12**

> **Completed. Do not re-run this task.** Five `v4_robust` evaluations on the
> corrected plant, same frozen protocol, into `eval_results_m2657/`. Ten JSONs
> total; `m2657_comparison.md` shows exactly the two open-field rows and no
> `*eval*` rows, which is the proof that R0's `--include` allowlist works.
>
> **The control did not survive the plant fix.**
>
> | | 3.657 kg | 2.657 kg |
> |---|---|---|
> | open-field falls | 0.000 % | **24.167 %** |
> | mean episode length | 30.00 s | 23.06 s |
> | push, v4 rule | 6.354 % | **93.151 %** |
> | push, v5 rule | 11.120 % | **93.672 %** |
> | sustained wrench | 100.000 % | 100.000 % |
> | obstacle graze | 32.604 % | **46.979 %** |
>
> Note `v4_robust` still passes the *gait* gate 6/6 while falling 24 % of the
> time — a reminder that the gait gate measures contact-pattern validity on the
> episodes that survive and is not a substitute for the fall rate.
>
> **Consequence, recorded in `m2657_regate.md`:** a baseline that falls 24 % of
> the time in the open field cannot anchor G-R4 or G-R5 for the rebuild
> campaign. `v4_robust` is retired as a control on this plant, and R2's
> `v6_robust` becomes the control for R2b.

**AI-agent suitable:** YES

**1. Context for the implementing agent**

G-R4 is a *relative* gate: v5d's contact numbers are only meaningful against
v4_robust measured under the same protocol on the same plant.
`scripts/v5_chain.sh` exists solely because that control was missing once
before — its header says so: *"v5c's contact numbers … have no v4 counterpart
measured under the same protocol, and 'better than v4' is the acceptance rule.
Without the control, none of v5c's battery can be scored."* The plant change
invalidates the control the same way it invalidated the candidate. Skipping this
task makes R1c unanswerable for four of its five gates.

Like R1, **this must run before Phase M** — the v4_robust checkpoint is also a
59/16 artefact.

Read first: `scripts/v5_chain.sh` step 1. Note it contains **four** `eval_ckpt`
calls (the contact battery), not five — the fifth control,
`v4_robust_grid6`, was produced by a separate hand-issued invocation. Its task
id is in `docs/jetson-mod/eval_results_v5/v4_robust_grid6.json` →
`Isaac-Velocity-Rough-OpenDuck-Contact-Play-v0`. That is deliberately **not** the
same play task v5d uses: each policy is evaluated on its own registered play
twin, as it was before.

Depends on: R0; must not overlap with R1 (one GPU job at a time). R1 and R1b may
be done in either order, but both must finish before R1c.

Trap: `v5_chain.sh` also hardcodes `RESULTS=…/eval_results_v5` and
`--comparison_md …/v5_comparison.md`. Do not run the script; issue the five
commands directly.

**2. Low-level implementation plan**

1. Preflight `audit_plant_mass.py` (exit 0).
2. Five invocations, identical in shape to R1 step 3, with
   `CKPT=$ISAACLAB/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`:

   | `--policies` NAME (= JSON basename) | task id | extra flag |
   |---|---|---|
   | `v4_robust_grid6` | `Isaac-Velocity-Rough-OpenDuck-Contact-Play-v0` | — |
   | `v4_robust_pusheval_v4def` | `Isaac-Velocity-Rough-OpenDuck-PushEval-v0` | `--keep-pushes` |
   | `v4_robust_pusheval_v5def` | `Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0` | `--keep-pushes` |
   | `v4_robust_wrencheval` | `Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0` | — |
   | `v4_robust_obstacleeval` | `Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0` | — |

   Same `--conditions`, same `--output_dir`, same `--comparison_md`, same two
   `--include` names, `--headless`, `sleep 20` between runs.
3. No videos are required for the control (it is a reference, not a candidate) —
   but if GPU time allows, render the `forward_vx02` clip so the R1c doc can
   show both policies side by side. Note that this checkpoint lives under
   `~/IsaacLab/logs/`, outside the repo, so `play.py` will write its `videos/`
   and `exported/` there — harmless.

**3. Unit tests**

None new. `tests/test_regate_results.py` from R1 now covers ten JSONs instead of
five — the plant-stamp, one-plant-per-directory, no-stale-plant, frozen-protocol
and joint-order assertions apply unchanged to the control files. Re-run it.

**4. Smoke test**

```bash
cd $REPO && python3 - <<'PY'
import json, glob, os
rows = {}
for f in glob.glob('docs/jetson-mod/eval_results_m2657/*.json'):
    d = json.load(open(f)); rows[os.path.basename(f)[:-5]] = d['aggregate']['fall_rate_pct']
for k in sorted(rows): print(f"{k:45s} {rows[k]:8.3f} % falls")
PY
```
Observable: ten rows. Both `*_wrencheval` rows and both `*_obstacleeval` rows are
present, so every G-R4 pair can be formed.

**5. Done when**

- [ ] Exactly 10 JSONs in `eval_results_m2657/` (5 × v5d, 5 × v4_robust).
- [ ] `python3 -m pytest tests/test_regate_results.py -q` passes over all 10.
- [ ] `m2657_comparison.md` shows exactly two open-field rows,
      `v5d_contact_wrench` and `v4_robust_grid6`, and no `*eval*` rows — the
      proof that the R0 `--include` fix works.

---

### Task R1c — Decide whether v5d still passes its gates, and record the verdict — ✅ **DONE 2026-08-12**

> **Completed. VERDICT: v5d PASSES every gate on the corrected plant.**
> `scripts/regate_report.py` exits **0**. Full write-up in
> [`m2657_regate.md`](m2657_regate.md) (8 sections, machine report pasted
> verbatim).
>
> ```
> G-R1  PASS    gait gate 6/6 (bar >=5/6)
> G-R2  PASS    open-field fall rate 0.000% (also inside the tightened <=0.5% bar)
> G-R3  PASS    video audit, frame-by-frame — eval_results_m2657/videos/AUDIT.md
> G-R4  PASS    all four candidate rates <= control (-37.7, -93.3, -57.0, -46.1)
> G-R5  PASS    ref RMS 4.676 vs control 4.704 (-0.028, bar <= 1.0)
> OVERALL: PASS
> ```
>
> Under the plan's own decision rule this is the first branch, so **no
> "NOT VALID ON THE CURRENT PLANT" banner was added** to
> `exported_policies/v5d_contact_wrench_ppo/README.md`, and no owner sign-off is
> pending.
>
> **The finding that matters more than the verdict.** The contact-rich v5
> curriculum produced a policy robust to a **27 % change in the plant itself** —
> v5d 0.000 % -> 0.000 % open-field falls, `v4_robust` 0.000 % -> 24.167 %.
> Neither had mass DR beyond `add_base_mass (-0.10, +0.15) kg`, so the
> robustness came from the curriculum (fall-only terminations, obstacles,
> sustained wrench), not from randomisation. Carry that into R2b's recipe.
>
> **Stated weakness:** G-R5 compares v5d's ref RMS against a control that is
> falling 24 % of the time. The gate passes, but it carries less information
> than its authors assumed.
>
> Delivered: `scripts/regate_report.py` (+ `tests/test_regate_report.py`,
> 13 tests incl. the mixed-plant and mixed-dims guards),
> `docs/jetson-mod/m2657_regate.md`, one-line verdict pointers added to
> `v5_comparison.md` / `validation_results.md` / `v4_retrain_results.md` with
> their historical numbers untouched, and `known_issues.md` PLANT-1's
> "re-running the gates is the first task of the rebuild" replaced with the
> actual outcome.

**AI-agent suitable:** PARTIAL — the arithmetic and the report are fully
autonomous. **Changing v5d's shipped status is an owner decision**: it changes
what Phase S is allowed to build against and whether the course-project baseline
pin (`exported_policies/v5d_contact_wrench_ppo/README.md`, which states it is the
EN.535.782 haptic-teleop baseline, pinned 2026-07-30) is still valid. Produce
the verdict, then get sign-off before editing that README.

**1. Context for the implementing agent**

The point of this task is that the decision must be *mechanical and recorded*,
not remembered. `docs/jetson-mod/known_issues.md` exists because verdicts that
lived only in a session transcript decayed into wrong documentation. Read
`docs/jetson-mod/v5_retrain_plan.md` §7 (gates), §17 ("What 'beats v4_robust'
has to mean") and §22 (the verdict being replaced) for the tone and structure of
an acceptable verdict write-up.

Depends on: R1 and R1b both complete (ten JSONs).

Trap: **the four contact numbers are not comparable to the 3.657 kg numbers.**
Report them as `new (old)` pairs and state the wrench-force, obstacle-momentum
and CFG-2 caveats from this plan's preamble in the document itself. A reader who
sees "wrench 47.1 % → 22 %" without that context will conclude the policy
improved.

**2. Low-level implementation plan**

1. **New file `scripts/regate_report.py`** — pure python + stdlib, no Isaac, no
   numpy needed. CLI: `--results_dir` (default
   `docs/jetson-mod/eval_results_m2657`), `--candidate` (default
   `v5d_contact_wrench`), `--control` (default `v4_robust`), `--markdown`.
   The `--control` value is a *prefix*: the open-field JSON is
   `<control>_grid6.json` for the v4 control and `<control>_grid6.json` for the
   v6 control too, while the battery files are `<control>_pusheval_v4def.json`
   etc. Resolve both patterns and fail loudly with the missing filename if one
   is absent.
   - Loads every `*.json` in the dir.
   - Prints an open-field block: for candidate and control, gait `valid/total`,
     `fall_rate_pct`, `reference_tracking_rms_deg`, duty L/R, asym, wz err,
     energy, jerk.
   - Prints a contact block: the four pairs
     (`pusheval_v4def`, `pusheval_v5def`, `wrencheval`, `obstacleeval`) as
     candidate vs control fall rates with the delta.
   - Evaluates the gates and prints one line per gate:
     `G-R1 gait gate 6/6 (bar >=5/6) PASS`, etc. G-R4 is PASS only if all four
     candidate rates are ≤ their control.
   - Prints `OVERALL: PASS` / `OVERALL: FAIL (G-Rx, G-Ry)` and exits 0 on PASS,
     1 on FAIL. G-R3 (video) is printed as `MANUAL — see the verdict doc` and is
     excluded from the exit code, because a script cannot watch a video.
   - Refuses to run (exit 2) if any JSON's `plant.simulated_total_mass_kg`
     differs from any other's by more than 1e-3, **or** if any two disagree on
     `(obs_dim, action_dim)` — the mixed-model guard. Print both offending
     values and both filenames.
2. **New file `docs/jetson-mod/m2657_regate.md`** with this structure:
   - *What changed and why this document exists* — the PLANT-1 fix, one
     paragraph, linking `known_issues.md#plant-1`.
   - *Protocol* — the frozen six conditions, 3,840 episodes, seed 42, and the
     statement that this is the `eval_results_m2657/` model (2.657067 kg, obs 59
     / action 16, USD hash `10ab887fe4d412b22d3d7c857a9d7f12`).
   - *Open-field grid* — a table with columns `metric | v4_robust @2.657 |
     v5d @2.657 | v5d @3.657 (stale) | delta`.
   - *Contact battery* — the same shape for the four gates, with the
     wrench-force, obstacle-momentum and CFG-2 caveats stated immediately under
     the table.
   - *Video audit* — per-condition verdicts from R1 step 7, or an explicit
     "PENDING — no vision-capable reviewer" line.
   - *Verdict* — the `regate_report.py` output pasted verbatim, then a prose
     verdict naming each failed gate.
   - *Consequences* — one of the two decision branches below, written out.
   - *Limits of this result* — PLANT-10 (54–82 g still unbooked), PLANT-5
     (servos still simulated at 1.78× datasheet stall, and no actuator parameter
     is randomised at all), DEPLOY-3 (4 obs dims unmeasurable on hardware),
     CFG-1/CFG-2 still open. None of these are fixed by this task; Phase M owns
     them.
3. **The decision rule, applied literally:**
   - **All of G-R1, G-R2, G-R4, G-R5 pass and the video audit passes** → v5d
     survives the plant correction. It is still not a hardware candidate (Phase M
     will change the plant again), but the recipe is known to transfer, which
     lowers the risk of R2b.
   - **Any gate fails** → add a banner to the top of
     `exported_policies/v5d_contact_wrench_ppo/README.md`: *"NOT VALID ON THE
     CURRENT PLANT — trained on the 3.657 kg plant; re-gated on the corrected
     2.657 kg plant on `<date>` and failed `<gates>`. See
     docs/jetson-mod/m2657_regate.md."* Keep the file and the checkpoint — the
     EN.535.782 baseline pin depends on it — and say explicitly that it remains
     valid *as a 3.657 kg artefact*. R2/R2b become blocking for Phase S.
   - In **either** branch, record that v5d is not the policy that ships to
     hardware: `task_plan_v2.md`'s "spend the retrain once" decision already
     commits the project to a post-Phase-M retrain.
4. **Propagate the verdict** to the three documents that would otherwise quote
   stale numbers as current: add a one-line pointer to `m2657_regate.md` in
   `docs/jetson-mod/v5_comparison.md` (its caveat block starts at line 49),
   `docs/jetson-mod/validation_results.md` (line 15) and
   `docs/jetson-mod/v4_retrain_results.md` (line 21). All three already carry a
   2026-08-11 caveat block from commit `11b1690` — **extend it, do not rewrite
   the historical numbers.**
5. Update `docs/jetson-mod/known_issues.md` PLANT-1's closing block: replace
   *"Re-running the gates on the corrected plant is the first task of the
   rebuild"* with the actual outcome and a link to `m2657_regate.md`.

**3. Unit tests**

New file `tests/test_regate_report.py`, `@pytest.mark.phase2`, driving
`scripts/regate_report.py` over synthetic JSONs in `tmp_path`. Unlike
`evaluate_policies.py` this script has no Isaac dependency, so a direct
`import` (or `runpy`) is safe and preferable to a subprocess:

- `test_gate_passes_on_clean_numbers`: candidate gate 6/6, falls 0.0, all four
  contact rates below control → exit 0, output contains `OVERALL: PASS`.
- `test_gait_gate_below_bar_fails`: candidate gait-valid in 4 of 6 conditions →
  exit 1, output contains `G-R1` and `FAIL`.
- `test_one_worse_contact_gate_fails`: candidate wrench 60 % vs control 55 % →
  exit 1, `G-R4 … FAIL`.
- `test_mixed_plant_refuses`: two JSONs with masses 2.657067 and 3.657067 → exit
  2, output names both masses. (This is the test that would have caught the
  original PLANT-1 class of error at the reporting layer.)
- `test_mixed_dims_refuses`: two JSONs with the same mass but `obs_dim` 59 and
  55 → exit 2. (This is the M0b-era guard.)
- `test_missing_control_file_names_it`: control battery file absent → non-zero
  exit and the missing filename in the output.

**4. Smoke test**

```bash
cd $REPO && python3 scripts/regate_report.py \
  --results_dir $REPO/docs/jetson-mod/eval_results_m2657 \
  --candidate v5d_contact_wrench --control v4_robust --markdown
echo "regate exit=$?"
```
Observable: a gate-by-gate table ending in `OVERALL: PASS` or
`OVERALL: FAIL (…)`, with exit code 0 or 1 matching. That exact output is pasted
into `m2657_regate.md`.

**5. Done when**

- [ ] `scripts/regate_report.py` exists and its six unit tests pass.
- [ ] `docs/jetson-mod/m2657_regate.md` exists with all eight sections and the
      pasted report output.
- [ ] Every one of G-R1, G-R2, G-R4, G-R5 has an explicit PASS/FAIL in that file,
      and G-R3 has a verdict or an explicit PENDING.
- [ ] `known_issues.md` PLANT-1 links to the verdict doc.
- [ ] The three companion docs each carry a one-line pointer to the verdict doc,
      with their historical numbers unchanged.
- [ ] If the verdict is FAIL: `exported_policies/v5d_contact_wrench_ppo/README.md`
      carries the banner, and the owner has signed off on it.

---

> ### ⛔ Phase M runs here
>
> Do not start Task R2 until `task_plan_v2.md` Tasks **M0**, **M0b** and **M2**
> are done (their own done-when lists all end with "no training has been run
> yet — the retrain is Task R2"). Every one of those tasks changes the plant or
> the interface, and each would otherwise force its own retrain.

---

### Task R2 — Train the robust seed on the post-Phase-M plant (`v6_robust`) — ✅ **DONE 2026-08-13** — gate 6/6, 0.000 % falls

**AI-agent suitable:** YES — long unattended GPU job; launch detached, monitor,
report proactively.

**1. Context for the implementing agent**

The v5d recipe is *defined* as a 3,000-iteration fine-tune from the v4_robust
checkpoint (`v5_retrain_plan.md` §6, §21: `--resume --load_run
0000-00-00_v4robust_seed --checkpoint model_2999.pt`). That seed was itself
trained on the 3.657 kg plant, and after Phase M's Task M0b its 59/16 interface
no longer even loads. So the rebuild is two stages, and this is stage one:
reproduce the v4_robust recipe **from scratch** on the post-Phase-M plant to
produce the seed stage two fine-tunes from.

**"From scratch" is load-bearing: this launch must NOT carry `--resume`.**
A resumed run would inherit the old plant's policy *and* fail on the shape
mismatch if M0b landed.

Read first:

- `AGENTS.md` "v4 Tracks (post-CAD retrain)" **Run B** — the recipe: task
  `Isaac-Velocity-Rough-OpenDuck-Robust-v0`, dynamics DR (pushes ±0.3 m/s on
  8-14 s, trunk mass (−0.10,+0.15) kg, CoM ±10/±5 mm, friction 0.4-1.0/0.3-0.8,
  joint-reset scale 0.9-1.1), asymmetric obs (actor 59 / critic 62 **before**
  M0b; actor 53 / critic 56 **after**)  (the line below restates this)
  M0b; 55 / 58 after), `velocity_limit_sim = 8.94 rad/s`.
- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` `OpenDuckRobustEnvCfg`
  (class at line 324, `__post_init__` at 342, DR block at 355-393).
- `isaac_lab_env/open_duck_mini_v2/agents/rsl_rl_ppo_cfg.py` —
  `OpenDuckPPORunnerCfg` sets `max_iterations = 3000` (line 28) and
  `save_interval = 100` (line 29); `OpenDuckRobustPPORunnerCfg.experiment_name`
  is `"open_duck_ppo_robust"` (line 69).
- `scripts/launch_training_detached.sh` (55 lines) — the launcher. It refuses to
  start if a live pidfile with the same run name exists, writes
  `.training_runs/<name>.log` and `.pid`, and `setsid`s the job so the pid in
  the file is also the process-group id.
- `scripts/monitor_training.py` — for progress reporting.

Depends on: R1c (the verdict must be recorded) **and** Phase M Tasks M0, M0b,
M2.

Traps:

- **Change no config in this task.** This must be the v4_robust recipe verbatim
  *as Phase M left it*; the only changed variable versus the v4_robust control is
  the plant plus Phase M's batched fixes. If you "improve" a reward or a DR range
  here, the comparison this whole phase exists to make is destroyed.
- **Log-root separation.** `OpenDuckRobustPPORunnerCfg.experiment_name` is
  `open_duck_ppo_robust`, whose directory already contains the stale
  `2026-07-07_00-15-43`. Isaac Lab computes
  `log_root_path = logs/rsl_rl/<agent_cfg.experiment_name>` relative to the cwd
  (`~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py:149`; the launcher
  `cd`s to `$ISAACLAB` first). Send this run to a fresh root with the hydra
  override `agent.experiment_name=open_duck_ppo_v6` so nothing can later resolve
  "the newest run" to a 3.657 kg one (`known_issues.md` **SHELL-1** selects run
  dirs by mtime). Hydra overrides work here because `train.py` is decorated with
  `@hydra_task_config(args_cli.task, args_cli.agent)` (line 114) and forwards
  unparsed args to Hydra (lines 41-48).
- **Always pass `--run_name`.** `train.py:157` appends
  `agent_cfg.run_name` to the timestamped log dir, which is the cheap half of the
  SHELL-1 fix. The v5 campaign did *not* pass it — its run dirs are bare
  timestamps, which is exactly why SHELL-1 matters.
- **`--video` is mandatory** (standing owner rule).

**2. Low-level implementation plan**

1. Confirm the GPU is idle: `pgrep -f train_ppo.py; pgrep -f evaluate_policies`
   must both print nothing.
2. Preflight `audit_plant_mass.py` (exit 0) and
   `verify_action_contract.py --expect_obs_dim <53|59> --expect_action_dim <14|16>`
   (exit 0) — use the dims Phase M actually produced, read back from
   `task_plan_v2.md` Task M0b's done-when evidence.
3. **Create the results directory for this model before anything else.**
   ```bash
   mkdir -p $REPO/docs/jetson-mod/eval_results_rebuild
   ```
   Write its `README.md` recording, measured not assumed: the mass printed by
   `audit_plant_mass.py`, the USD asset hash from
   `mini_bdx/robots/open_duck_mini_v2/usd/.asset_hash`, the obs/action dims read
   back from a live env, the list of Phase-M issue ids fixed, and the same
   frozen protocol + gate table as `eval_results_m2657/README.md`. State: *"This
   is a different robot model from `eval_results_m2657/`. Nothing may be copied
   between them."*
4. Launch:
   ```bash
   cd $REPO && ./scripts/launch_training_detached.sh v6_robust \
     --task Isaac-Velocity-Rough-OpenDuck-Robust-v0 --headless \
     --max_iterations 3000 --run_name v6_robust \
     --video --video_length 200 --video_interval 5000 \
     agent.experiment_name=open_duck_ppo_v6
   ```
   Logs: `$REPO/.training_runs/v6_robust.log`, pidfile
   `$REPO/.training_runs/v6_robust.pid`. Run dir:
   `$ISAACLAB/logs/rsl_rl/open_duck_ppo_v6/<timestamp>_v6_robust/`.
5. **Watchdog at +30 min** (the lever that would have saved v5a's wasted 2.22 h):
   ```bash
   grep -E "Gait/duty_in_band_frac" $REPO/.training_runs/v6_robust.log | tail -1
   ```
   If the value is `< 0.30`, kill the run
   (`kill -9 -- -$(cat $REPO/.training_runs/v6_robust.pid)`), record the number,
   and stop — a standing policy is not worth 2 hours. **Measured reference:**
   v5d's healthy reading at roughly the same point was **0.982**, and its
   last-100 mean was **0.9836** (recomputed 2026-08-11 from
   `.training_runs/v5d_contact_wrench.log`, 3,000 samples). Remember CFG-5: this
   canary reads high relative to `evaluate_policies.py`, so it is a
   go/no-go signal, not a gate number.
6. Report progress unprompted at ~30 min, ~60 min and on exit (iteration count,
   mean reward, `duty_in_band_frac`).
7. **Completion check** (AGENTS.md rule 6, both halves):
   ```bash
   kill -0 -- -$(cat $REPO/.training_runs/v6_robust.pid) 2>/dev/null; echo "alive=$?"   # want non-zero -> dead
   ls $ISAACLAB/logs/rsl_rl/open_duck_ppo_v6/*_v6_robust/model_2999.pt
   ```
8. **Create the fine-tune seed dir for R2b.** The pattern comes from
   `v5_retrain_plan.md` §6: rsl-rl's `get_checkpoint_path(log_root, load_run, …)`
   resolves `load_run` inside the *current* experiment's log root, so the seed
   must live under `open_duck_ppo_v6`, and giving it an explicit name avoids
   the empty-newest-run trap.
   ```bash
   SRC=$(ls -d $ISAACLAB/logs/rsl_rl/open_duck_ppo_v6/*_v6_robust)
   DST=$ISAACLAB/logs/rsl_rl/open_duck_ppo_v6/0000-00-00_v6robust_seed
   mkdir -p $DST && cp $SRC/model_2999.pt $DST/ && cp -r $SRC/params $DST/
   ```
9. **Journal it now.** Write the `v6_robust` entry in
   `docs/jetson-mod/experiment_journal.md` using `scripts/tb_summary.py` (built
   in R0, run under `~/IsaacLab/_isaac_sim/python.sh`) for the last-100 means.
   AGENTS.md makes this mandatory *before the next run launches*, and R2b is the
   next run.
10. **Gate the seed before using it.** Run the open-field grid only (one eval,
    ~30 min), named `v6_robust_grid6`, into `eval_results_rebuild/`, task
    `Isaac-Velocity-Rough-OpenDuck-Contact-Play-v0` (the play twin the v4 control
    used, so the two grids are directly comparable), `--include v6_robust_grid6
    --include v6d_contact_wrench`, `--comparison_md
    $REPO/docs/jetson-mod/rebuild_comparison.md`. A seed that does not walk must
    not be fine-tuned.

**3. Unit tests**

None new — this task trains a network. The mechanical checks live in the smoke
test and in `tests/test_regate_results.py` (R1), which will assert the new JSON
carries a consistent plant stamp for `eval_results_rebuild/`.

**4. Smoke test**

```bash
cd $REPO && python3 - <<'PY'
import json
d = json.load(open('docs/jetson-mod/eval_results_rebuild/v6_robust_grid6.json'))
a, p = d['aggregate'], d['plant']
print(f"gate {a['gait_valid_conditions']}/{a['conditions']}  falls {a['fall_rate_pct']:.3f}%  "
      f"RMS {a['reference_tracking_rms_deg']:.3f}  duty {a['stance_duty_left_pct']:.1f}/"
      f"{a['stance_duty_right_pct']:.1f}  plant {p['simulated_total_mass_kg']:.6f}  "
      f"dims {p['obs_dim']}/{p['action_dim']}")
PY
```
Observable: `gate >= 5/6`, `falls < 1.0%`, and a plant/dims line matching
`eval_results_rebuild/README.md`. If the gate is below 5/6, stop and report —
the v4 recipe does not transfer to the post-Phase-M plant unchanged, which is a
finding in its own right and changes R2b's design.

**5. Done when**

- [ ] `open_duck_ppo_v6/<timestamp>_v6_robust/model_2999.pt` exists and the
      process group is dead.
- [ ] `videos/train/` in that run dir is non-empty (the `--video` rule was
      honoured).
- [ ] `0000-00-00_v6robust_seed/` contains `model_2999.pt` and `params/`.
- [ ] `docs/jetson-mod/eval_results_rebuild/README.md` exists and its plant
      identity was **measured**, not copied from this plan.
- [ ] `eval_results_rebuild/v6_robust_grid6.json` exists, gate ≥ 5/6,
      falls < 1 %.
- [ ] An `experiment_journal.md` entry for `v6_robust` exists **before R2b
      launches**, with its training numbers sourced from `tb_summary.py`.

---

### Task R2b — Retrain the contact-wrench recipe on the post-Phase-M plant (`v6d_contact_wrench`) — ✅ **DONE 2026-08-13** — PASS, 4/4 contact gates beat the same-plant control

**AI-agent suitable:** PARTIAL — training and evaluation are autonomous; the
mandatory video audit needs a vision-capable reviewer, exactly as in R1.

**1. Context for the implementing agent**

This is the retrain the PLANT-1 register demands: *"Whichever [fix] is chosen,
retrain before quoting any hardware number — the plant changes, so existing
checkpoints no longer match it."* The recipe is v5d's, unchanged apart from
whatever Phase M altered: v5c (v4 + fall-only terminations + obstacles) **plus
one lever**, the sustained wrench. Verified values in
`OpenDuckContactWrenchEnvCfg.__post_init__` (`env_cfg.py:773-786`):
`active_frac = 0.5`, `force_frac_range = (0.05, 0.20)`,
`duration_range_s = (2.0, 6.0)`, `torque_z_range = (0.05, 0.15)`,
`lateral_bias = 0.7`, `cooccurrence_frac = 0.3`, `rotating_cmd_wz = 0.25`.
(`torque_z_range` was inert before Phase M — CFG-1 — so if M0 fixed it, this
run is **not** a pure plant-swap of v5d and the write-up must say so.)

Read first:

- `isaac_lab_env/open_duck_mini_v2/env_cfg.py` `OpenDuckContactWrenchEnvCfg`
  (class at line 742, docstring 743-772 explaining why it is exactly one lever,
  `__post_init__` 773-786) and `OpenDuckContactWrenchEnvCfg_PLAY` (789-807).
- `docs/jetson-mod/v5_retrain_plan.md` §21 (the launch) and §22 (the result).
- `scripts/v5_pipeline.sh` — you will parameterise and reuse it.

Depends on: R2 (the `0000-00-00_v6robust_seed` dir and its journal entry).

Traps:

- **`v5_pipeline.sh` hardcodes four things that must change**: `LOGROOT`
  (`$ISAACLAB/logs/rsl_rl/open_duck_ppo_v5`), `RESULTS`
  (`$REPO/docs/jetson-mod/eval_results_v5`), the literal `--comparison_md
  "$REPO/docs/jetson-mod/v5_comparison.md"` inside `run_eval()`, and
  `CONDITIONS`. It also selects the run dir by mtime, not by name (SHELL-1). Fix
  these by environment-variable override rather than by editing constants
  inline, so the v5 campaign's reproduction path is preserved.
- **The final checkpoint is `model_5998.pt`, not `model_5999.pt`.** rsl-rl
  restores the iteration counter on resume, so a 3,000-iteration fine-tune from
  `model_2999.pt` ends at 5998. The pipeline comments this; believe it.
- **`OpenDuckContactPPORunnerCfg.experiment_name` is `open_duck_ppo_v5`**
  (`rsl_rl_ppo_cfg.py:89`) — override it to `open_duck_ppo_v6` so the
  corrected-plant runs never share a log root with the stale ones.
- **Runtime-smoke the wrench before committing 2 hours** — the rule v5c's
  `KeyError` established. `ContactRegimeEvent.__init__` runs inside a timeline
  PLAY callback whose **exceptions are swallowed**: a bad param silently leaves
  the term as an uninstantiated class and training runs on without the wrench,
  looking perfectly healthy.
- **`run_eval()` skips any eval whose JSON already exists** in `$RESULTS`. That
  makes reruns cheap but also means a stale file silently short-circuits the
  gate. Delete a JSON before re-measuring it.

**2. Low-level implementation plan**

1. **Edit `scripts/v5_pipeline.sh`** — make the four constants overridable and
   fix SHELL-1. Replace the current assignments (around lines 33-40):
   ```bash
   LOGROOT="${LOGROOT:-$ISAACLAB/logs/rsl_rl/open_duck_ppo_v5}"
   RESULTS="${RESULTS:-$REPO/docs/jetson-mod/eval_results_v5}"
   COMPARISON_MD="${COMPARISON_MD:-$REPO/docs/jetson-mod/v5_comparison.md}"
   CONDITIONS="${CONDITIONS:-0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2;0,0,0.5}"
   INCLUDE_ARGS="${INCLUDE_ARGS:---include v5d_contact_wrench --include v4_robust_grid6}"
   ```
   In `run_eval()`, replace the literal comparison path with `"$COMPARISON_MD"`
   and add `$INCLUDE_ARGS` (unquoted, so it word-splits) to the same command.
   For SHELL-1, replace the run-dir selection line
   (`RUNDIR=$(ls -td "$LOGROOT"/*/ … | grep -v v4robust_seed | head -1)`) with:
   ```bash
   RUNDIR="${RUNDIR_OVERRIDE:-$(ls -td "$LOGROOT"/*_"$RUN_NAME"/ 2>/dev/null | head -1)}"
   [ -z "$RUNDIR" ] && RUNDIR=$(ls -td "$LOGROOT"/*/ 2>/dev/null | grep -v seed | head -1)
   ```
   Leave the gate logic (`PASSED -lt 5`) and the heredoc alone — it is correct
   and its comment records why (a previous version inverted the gate).
   Note the script runs under `set -uo pipefail` (no `-e`), and `RUN_NAME` /
   `PLAY_TASK` come from `$1` / `$2` with `${1:?}` guards.
2. **Runtime smoke, 100 iterations**:
   ```bash
   cd $REPO && ./scripts/launch_training_detached.sh v6_smoke \
     --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0 --headless \
     --max_iterations 100 --resume \
     --load_run 0000-00-00_v6robust_seed --checkpoint model_2999.pt \
     --run_name v6_smoke --video --video_length 200 --video_interval 1000 \
     agent.experiment_name=open_duck_ppo_v6
   ```
   In `.training_runs/v6_smoke.log`, verify **all three**: the resume banner
   loads `0000-00-00_v6robust_seed/model_2999.pt` and the iteration counter
   starts at 2999; a line matching
   `[v5] ContactRegimeEvent: measured robot weight <W> N (<M> kg) -> wrench <lo>-<hi> N (active_frac=0.5)`
   is present, with `<M>` equal to the mass `audit_plant_mass.py` reports and
   `<lo>-<hi>` equal to `0.05·W` and `0.20·W` (on the PLANT-1-only plant that
   was `26.07 N (2.657 kg) -> wrench 1.30-5.21 N`; Phase M may move it, so
   compute the expected numbers from the measured mass rather than copying
   these); and `Gait/duty_in_band_frac` is being emitted. Final checkpoint is
   `model_3098.pt`.
3. **Full fine-tune**:
   ```bash
   cd $REPO && ./scripts/launch_training_detached.sh v6d_contact_wrench \
     --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0 --headless \
     --max_iterations 3000 --resume \
     --load_run 0000-00-00_v6robust_seed --checkpoint model_2999.pt \
     --run_name v6d_contact_wrench \
     --video --video_length 200 --video_interval 5000 \
     agent.experiment_name=open_duck_ppo_v6
   ```
4. **Early-abort watchdog at +30 min**, same rule and same reference values as
   R2 step 5 (`duty_in_band_frac < 0.30` → kill).
5. **Evaluate via the parameterised pipeline** once training exits:
   ```bash
   LOGROOT=$ISAACLAB/logs/rsl_rl/open_duck_ppo_v6 \
   RESULTS=$REPO/docs/jetson-mod/eval_results_rebuild \
   COMPARISON_MD=$REPO/docs/jetson-mod/rebuild_comparison.md \
   INCLUDE_ARGS="--include v6d_contact_wrench --include v6_robust_grid6" \
   $REPO/scripts/v5_pipeline.sh v6d_contact_wrench \
     Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0
   ```
   This waits for the pidfile's process group, runs the gait gate first, aborts
   the battery if it fails (< 5/6), then runs the four contact evals and renders
   four audit videos into `<run_dir>/videos/play/`. It writes JSONs named
   `v6d_contact_wrench*`.
6. **Audit the videos** exactly as in R1 step 7, adding the two contact-specific
   checklist items from `v5_retrain_plan.md` §8: compliant slide along the
   obstacle (no freeze/bounce), and turn-in-place as a stepping rotation rather
   than a pivot-scrape. The pipeline renders `forward_vx02`, `turn_wz05`,
   `press_wrench` and `obstacle_graze`; render a fifth `turn_wz03` clip by hand
   because AGENTS.md names wz=0.3, not wz=0.5, as the mandatory turn condition.
7. **Score it** with
   `python3 scripts/regate_report.py --results_dir docs/jetson-mod/eval_results_rebuild --candidate v6d_contact_wrench --control v6_robust`,
   and write the result into `docs/jetson-mod/rebuild_results.md` (same eight
   sections as `m2657_regate.md`). The acceptance rule is unchanged from §17: it
   cannot win on open-field locomotion, only match it; **the win must come from
   the contact gates** without dropping the gait gate below 5/6.
8. **State plainly, in `rebuild_results.md`, what this comparison is and is
   not.** v6d-vs-v6_robust is controlled (same plant, same Phase-M fixes).
   v6d-vs-v5d is **not** controlled — it crosses a plant change *and* 15 defect
   fixes *and* possibly an interface change. Report the v5d numbers as context
   only, never as a delta.
9. Journal `v6_smoke` and `v6d_contact_wrench` before moving on.

**3. Unit tests**

New file `tests/test_v5_pipeline_overrides.py`, `@pytest.mark.phase2`. The
pipeline edits are shell, so test them by executing the script's variable
resolution rather than grepping it. **Note the header cannot simply be
`source`d without positional arguments**: `RUN_NAME="${1:?…}"` aborts the shell
if `$1` is unset. Supply them.

- `test_syntax_is_valid`: `bash -n scripts/v5_pipeline.sh` returns 0.
- `test_results_dir_is_overridable`: copy the script's header (everything above
  the first `mkdir -p "$RESULTS"`) into `tmp_path/header.sh`, append
  `echo "$RESULTS|$COMPARISON_MD|$LOGROOT"`, then run
  `bash tmp_path/header.sh myrun mytask` twice — once with `RESULTS`/
  `COMPARISON_MD` exported and once without — and assert the override wins and
  the fallback is `eval_results_v5` / `v5_comparison.md`.
- `test_rundir_prefers_the_named_run`: create `tmp_path/{ts1_other,ts2_myrun}`
  with `ts1_other` newer (`touch -d`), run the same header with
  `LOGROOT=tmp_path` and positional `$1=myrun`, and assert the resolved `RUNDIR`
  ends in `myrun`. This is the SHELL-1 regression that mtime-selection would
  otherwise reintroduce.
- `test_v5_reproduction_path_intact`: with no env overrides, assert the resolved
  `RESULTS` is `docs/jetson-mod/eval_results_v5` and `LOGROOT` ends in
  `open_duck_ppo_v5`.

**4. Smoke test**

```bash
grep -c "measured robot weight" $REPO/.training_runs/v6d_contact_wrench.log
grep -E "^\[pipeline\] GATE" $REPO/.training_runs/v6d_contact_wrench_pipeline.log
cd $REPO && python3 scripts/regate_report.py \
  --results_dir docs/jetson-mod/eval_results_rebuild \
  --candidate v6d_contact_wrench --control v6_robust
```
Observable: the grep count is ≥ 1 and the mass in that line matches
`audit_plant_mass.py` (the corrected plant reached the curriculum); the pipeline
log's `[pipeline] GATE n/6 …` line shows n ≥ 5; `regate_report.py` prints a full
gate table and exits 0 or 1 (never 2 — a 2 means the results dir is mixing
models).

**5. Done when**

- [ ] `open_duck_ppo_v6/<timestamp>_v6d_contact_wrench/model_5998.pt` exists,
      process group dead.
- [ ] `videos/train/` non-empty for the run.
- [ ] Five `v6d_contact_wrench*.json` files in `eval_results_rebuild/`, all
      carrying the same plant stamp as `v6_robust_grid6.json`.
- [ ] Five audit mp4s (including `turn_wz03`) + filmstrips stored and referenced.
- [ ] `docs/jetson-mod/rebuild_results.md` has a `v6d_contact_wrench` section
      with the gate table, a video verdict, and the explicit statement that
      v6d-vs-v5d is not a controlled comparison.
- [ ] `experiment_journal.md` entries exist for `v6_smoke` and
      `v6d_contact_wrench`.
- [ ] `tests/test_v5_pipeline_overrides.py` passes, and re-running the pipeline
      with no env overrides still resolves to `eval_results_v5` (the v5
      campaign's reproduction path is intact).

---

### Task R3 — Export the ONNX and verify the deployment contract — ✅ **DONE 2026-08-13** — verifier 10/10, exit 0

**AI-agent suitable:** YES

**1. Context for the implementing agent**

`jetson_runtime/` does not exist yet and no file in the repo imports
onnxruntime, so none of this is a live bug — it is the spec the Phase-S runtime
must satisfy, and it must be produced *with* the policy, not reconstructed from
memory later. Three measured facts define the contract
(`known_issues.md` **DEPLOY-1**, **DEPLOY-2**, **DEPLOY-4**):

1. **`action_scale = 0.25` and `q_default` are absent from the graph.** Of
   197,126 initializer scalars, zero equal 0.25, and the only 16-element tensor
   is `mlp.6.bias`, which differs from `q_default` by up to 1.461333 rad.
   `q_target = q_default + 0.25·a` lives entirely in Isaac Lab's
   `JointPositionAction`. A runtime that commands the ONNX output directly is
   wrong by a **4× gain and a standing-pose offset up to 1.379 rad** — total,
   silent failure.
2. **The observation's joint block is `joint_pos_rel` (`q − q_default`), not raw
   encoder angles.** Feeding absolute angles puts the knees 13-14 σ outside the
   training distribution.
3. **The normaliser epsilon (0.01) is not in the checkpoint** — it is a plain
   Python attribute in `rsl_rl/modules/normalization.py`, so a hand-rolled
   reimplementation that divides by `_std` alone is off by 28.0 % on one channel.
   rsl-rl's own loader reconstructs it, so `policy.pt` and `policy.onnx` are
   correct; only a reimplementation is at risk.

Read first: `exported_policies/v5d_contact_wrench_ppo/README.md` (the archive
format to copy — note its files are **flat**: `agent.yaml`, `env.yaml`,
`model_5998.pt`, `policy.pt`, `policy.onnx`, `policy.onnx.data`, `README.md`),
`scripts/duck_init_pos.json`, `isaac_lab_env/open_duck_mini_v2/env_cfg.py:266`
(`self.actions.joint_pos.scale = 0.25`), and
`~/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py:126-130` and `:170-195`
(the checkpoint resolution and the automatic export into
`os.path.dirname(resume_path)/exported/`).

Depends on: R2b (there must be a `v6d` checkpoint to export). If R2b's gate
failed, run this task against whatever checkpoint the owner designates and say
so in the README.

Traps:

- `*.onnx` is **gitignored** (`.gitignore:19`, `known_issues.md` **ART-1**) — the
  file will not be in git, so the README's md5 pin is the only provenance.
  Record it. Note `policy.onnx.data` (the external weights) *is* tracked, which
  is why ART-1 is filed as HIGH.
- ONNX/torch tooling lives in Isaac's interpreter. Verified 2026-08-11:
  `~/IsaacLab/_isaac_sim/python.sh` has `onnx 1.20.1`, `torch 2.9.0+cu130`,
  `numpy 1.26.0`. The system `python3` does not have `onnx`. Use the Isaac
  interpreter for the verifier.
- `onnx.load` on `policy.onnx` needs `policy.onnx.data` next to it (external
  data). Load with the file's own directory as cwd, or pass
  `load_external_data=True` after `onnx.load_model(..., load_external_data=False)`
  + `onnx.external_data_helper.load_external_data_for_model`.
- **`yaml.safe_load` FAILS on the archived `env.yaml`** — it contains 81
  `!!python/object` and `!!python/tuple` tags and raises
  `ConstructorError: could not determine a constructor for the tag
  'tag:yaml.org,2002:python/tuple'`. Verified. Use `yaml.unsafe_load`, which
  returns `d["actions"]["joint_pos"]["scale"] == 0.25` correctly, or a targeted
  regex. Do not "fix" the yaml.
- The batch dimension is hard-fixed at 1 (`dynamic_axes={}`); batch 2/4 raise
  `InvalidArgument`. Do not "fix" it — record it.
- **The obs/action dims depend on Phase M.** If M0b landed they are 53/14, not
  59/16, and the obs layout has 14-element joint blocks. Read them from the
  produced eval JSON's `plant` block; do not hardcode.

**2. Low-level implementation plan**

1. Export by running play once on the new checkpoint (this both renders a clip
   and writes `exported/policy.pt` + `exported/policy.onnx` next to the
   checkpoint):
   ```bash
   cd $ISAACLAB && ./isaaclab.sh -p $REPO/scripts/play_policy.py \
     --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 --num_envs 2 \
     --checkpoint $ISAACLAB/logs/rsl_rl/open_duck_ppo_v6/<ts>_v6d_contact_wrench/model_5998.pt \
     --headless --video --video_length 500
   ```
2. Create `$REPO/exported_policies/v6d_contact_wrench_ppo/` and copy in, **flat**
   (matching v5d's layout): `model_5998.pt`, `exported/policy.pt`,
   `exported/policy.onnx` (+ `policy.onnx.data` if present), and
   `params/agent.yaml` → `agent.yaml`, `params/env.yaml` → `env.yaml`. Record
   `md5sum` of `model_5998.pt` and `policy.onnx`.
3. **New file `exported_policies/v6d_contact_wrench_ppo/deployment_contract.json`**
   — the sidecar that closes DEPLOY-1 in practice. Generate it, do not
   hand-type it: read `joint_order` and `q_default_rad` from
   `scripts/duck_init_pos.json`, `action_scale` from the archived `env.yaml`,
   and the dims + plant mass from the run's eval JSON `plant` block.
   ```json
   {
     "obs_dim": 59,
     "obs_layout": ["base_ang_vel(3)", "projected_gravity(3)", "velocity_commands(3)",
                    "joint_pos_rel(N)", "joint_vel_rel(N)", "actions(N)", "gait_phase(2)"],
     "action_dim": 16,
     "action_scale": 0.25,
     "action_formula": "q_target = q_default + action_scale * action",
     "joint_order": ["…N names, copied from scripts/duck_init_pos.json…"],
     "q_default_rad": [ "… N floats in joint_order …" ],
     "joint_pos_obs_is_relative": true,
     "normalizer_epsilon": 0.01,
     "onnx_batch_dim": 1,
     "plant_mass_kg": 0.0,
     "usd_asset_hash": "…",
     "checkpoint_md5": "…",
     "onnx_md5": "…",
     "unmeasurable_obs_dims": {"note": "antenna joints — see known_issues.md DEPLOY-3",
                               "joint_pos_rel": [22, 23], "joint_vel_rel": [38, 39],
                               "action_outputs": [13, 14]}
   }
   ```
   **If Phase M's Task M0b landed, `unmeasurable_obs_dims` must become an empty
   object with a note that DEPLOY-3 was closed by removing the antennas** — and
   `obs_dim`/`action_dim`/`obs_layout` become 53/14. Getting this wrong is the
   exact class of stale-documentation error the register exists to prevent.
   (The index arithmetic for the 59-dim case, for reference: antenna joints are
   at indices 13 and 14 of `joint_order`; the joint_pos block starts at obs
   index 9 and the joint_vel block at 25, giving 22/23 and 38/39.)
4. **New file `scripts/verify_deployment_contract.py`** — runs under
   `~/IsaacLab/_isaac_sim/python.sh`, **no Isaac Sim app needed** (pure
   onnx + torch + numpy), takes `--policy_dir`:
   1. Load `policy.onnx`, enumerate `graph.node` op types — assert
      `["Sub","Div","Gemm","Elu","Gemm","Elu","Gemm","Elu","Gemm"]`.
   2. Assert graph input is `float32[1, obs_dim]` and output
      `float32[1, action_dim]`, with the dims read from the sidecar.
   3. Reimplement the forward pass in numpy from the initializers, load
      `policy.pt` with `torch.jit.load`, feed 2,000 samples of
      `np.random.randn(1, obs_dim).astype(np.float32)` through both, assert
      `max|Δ| < 1e-5`.
   4. Assert **zero** initializer scalars equal 0.25 (atol 1e-6) and that no
      `action_dim`-element initializer matches `q_default` — i.e. re-confirm
      DEPLOY-1 is still structurally true, so the sidecar is genuinely required.
   5. Load `deployment_contract.json`; assert `action_scale` equals the value
      parsed out of the archived `env.yaml` (via `yaml.unsafe_load`), and that
      `joint_order` and `q_default_rad` match `scripts/duck_init_pos.json`
      exactly.
   6. Attempt a batch-2 input, catch the exception and record whether DEPLOY-4
      still holds. Report it; do not fail on it.
   7. Print a one-line PASS/FAIL per check; `sys.exit(1)` if any FAIL.
5. Write `exported_policies/v6d_contact_wrench_ppo/README.md` in the same format
   as v5d's: checkpoint, md5, training log dir, train/play task ids, actor obs
   dim, runner cfg, eval JSON path, comparison table path, provenance
   (fine-tuned from `v6_robust`, itself trained from scratch on the
   post-Phase-M plant), and a **Plant** row stating the measured mass and the
   USD asset hash — the row whose absence made PLANT-1 invisible for three
   policy generations.
6. Update `AGENTS.md`: the "Key Files" line
   `exported_policies/<name>/policy.onnx — Exported policies (currently only
   v5d_contact_wrench_ppo/; *.onnx is gitignored)` to name the new directory,
   and the "TensorRT Policy Inference" block (line 651) to point at
   `deployment_contract.json` for `action_scale`/`q_default` and to carry the
   correct obs/action dims.

**3. Unit tests**

New file `tests/test_deployment_contract.py`, `@pytest.mark.phase4` (registered
in `pytest.ini`), all of which skip if
`exported_policies/v6d_contact_wrench_ppo/deployment_contract.json` is absent
(so the suite is green before R3 and meaningful after). These parse JSON — no
ONNX import, so they run under plain `python3`:

- `test_contract_matches_duck_init_pos`: `joint_order` and `q_default_rad` equal
  `scripts/duck_init_pos.json`'s `joint_order` / `init_pos_rad` (the latter is a
  **dict keyed by joint name**) to 1e-9.
- `test_action_scale_matches_env_cfg`: `action_scale == 0.25` **and** equals the
  value read from the archived `env.yaml` with `yaml.unsafe_load`.
- `test_dims_are_self_consistent`: `obs_dim` and `action_dim` agree with
  `len(joint_order)` and with the `obs_layout` string, and match the `plant`
  block of `eval_results_rebuild/v6d_contact_wrench.json`.
- `test_plant_mass_matches_the_results_dir`: `plant_mass_kg` equals the mass in
  `eval_results_rebuild/README.md` to 1e-3.
- `test_deploy3_is_addressed`: either `unmeasurable_obs_dims` names 4 obs dims
  and 2 action outputs (antennas still present), **or** it is empty and carries
  a note that DEPLOY-3 was closed by M0b. One of the two must be true —
  silence is a failure.

**4. Smoke test**

```bash
cd $REPO && ~/IsaacLab/_isaac_sim/python.sh scripts/verify_deployment_contract.py \
  --policy_dir exported_policies/v6d_contact_wrench_ppo
echo "contract exit=$?"
```
Observable: every check prints PASS, in particular
`ONNX vs TorchScript: max|delta| = <value> < 1e-5` and
`DEPLOY-1 still structurally true: 0 initializer scalars equal 0.25`, and the
script exits **0**.

**5. Done when**

- [ ] `exported_policies/v6d_contact_wrench_ppo/` contains `model_5998.pt`,
      `policy.pt`, `policy.onnx`, `agent.yaml`, `env.yaml`,
      `deployment_contract.json`, `README.md`.
- [ ] `verify_deployment_contract.py` exits 0.
- [ ] `python3 -m pytest tests/test_deployment_contract.py -q` passes (5 tests).
- [ ] The README records both md5s, the measured plant mass and the USD hash.
- [ ] `known_issues.md` DEPLOY-1 gains a note that a sidecar now ships with the
      policy. The ONNX graph itself is unchanged, so the issue is **mitigated,
      not fixed** — say exactly that, and **do not retire its check**.

---

### Task R4 — Backfill the experiment journal (`v5_contact_results.md` already exists) — ✅ **DONE 2026-08-13** — Runs 17–24 journalled

**AI-agent suitable:** YES

**1. Context for the implementing agent**

`known_issues.md` **DOC-2** is measured, not alleged — reproduce it with
`python3 scripts/verify_known_issues.py DOC-2`:
```
v5 run names in experiment_journal.md: v5a_gated_ft=0, v5b_ungated_ft=0,
  v5c_contact_only=0, v5d_contact_wrench=0, v5_smoke=0
docs/jetson-mod/v5_contact_results.md exists: False
```
Five executed training runs — including the one that produced the shipped
policy — exist only inside a file titled "Execution Plan"
(`v5_retrain_plan.md`), and `AGENTS.md` makes a journal entry mandatory per run.
Phase R adds three more runs (`v6_robust`, `v6_smoke`, `v6d_contact_wrench`),
which R2 and R2b journal as they happen. If the v5 backfill is not done, the
project's engineering record skips the generation that shipped.

Read first:

- `AGENTS.md` "Experiment Journal Protocol" (line 699) — the seven
  non-negotiable data-sourcing rules and the required entry structure
  (one-lever config delta | training signals from TensorBoard | gate evaluation |
  verdict with gate name | artifact paths), plus the requirement to update the
  run-index table at the top.
- `docs/jetson-mod/experiment_journal.md` — the run-index table starts at
  line 32 (`## Run index`, header row at 34-35). The last existing entry is
  **Run 16 — `amp_v8_seed`** at line 646-ish; copy the shape of "Run 15 —
  `amp_v7`" (line 571). Heading format is
  `## Run N — \`<run_name>\` (\`<log dir>\`) — optional trailing note`.
- `docs/jetson-mod/v5_retrain_plan.md` §14-§22 — the narrative source for the
  five v5 runs. **This is a source, not a substitute**: rule 1 requires
  last-100-iteration TensorBoard means, not numbers copied from prose.

Depends on: nothing hard. The v5 backfill is CPU-only and may be done in any GPU
gap — including while R1 or R2 occupies the GPU. Do it early if there is idle
time. The v6 entries are written by R2 and R2b themselves.

Run dirs (all under `$ISAACLAB/logs/rsl_rl/open_duck_ppo_v5/`, each verified to
contain a TensorBoard event file and a matching `.training_runs/<name>.log`):

| Run | Log dir | Final ckpt |
|---|---|---|
| `v5_smoke` | `2026-07-28_03-10-51` | `model_3098.pt` |
| `v5a_gated_ft` | `2026-07-28_03-25-43` | `model_5998.pt` |
| `v5b_ungated_ft` | `2026-07-28_22-54-19` | `model_5998.pt` |
| `v5c_contact_only` | `2026-07-29_01-46-04` | `model_5998.pt` |
| `v5d_contact_wrench` | `2026-07-29_08-59-25` | `model_5998.pt` |

Note these dirs carry **no `_<run_name>` suffix** — the v5 campaign did not pass
`--run_name`, which is exactly the SHELL-1 exposure R2/R2b fix.

Traps:

- Rule 1 is not decorative: *"NEVER quote a single-iteration value from a
  console-log grep as a final result — this exact mistake corrupted the first
  journal draft"*. Use `scripts/tb_summary.py` (built in R0) under
  `~/IsaacLab/_isaac_sim/python.sh`.
- **`v4_robust` and `v4_inertials` also have no journal entries.** DOC-2 only
  scopes the v5 gap, so do not silently renumber around them; if you add them,
  add them as their own runs and say so. Otherwise leave the numbering
  contiguous from Run 16.

**2. Low-level implementation plan**

1. **Backfill five v5 entries** in `docs/jetson-mod/experiment_journal.md`, in
   chronological order after Run 16, numbered **Run 17 through Run 21**. Each
   entry carries:
   - the one-lever delta (from `v5_retrain_plan.md` §17's two-axis structure),
   - training signals from `tb_summary.py` (rule 1) with the run dir named,
   - the gate evaluation from `eval_results_v5/*.json`, citing the JSON path and
     field (rules 3 and 4),
   - the verdict with its gate name — verified from the JSONs:
     v5a FAIL (gait gate 0/6, standing policy); v5b FAIL (gait gate 3/6,
     asymmetric shuffle); v5c PASS 6/6 but 100.000 % wrench falls;
     v5d PASS, beats v4_robust on all four contact gates (1.068 vs 6.354,
     0.000 vs 11.120, 47.109 vs 100.000, 0.312 vs 32.604),
   - artifact paths (checkpoint, `exported/`, `videos/play/`, log dir),
   - **and a plant banner**: *"Measured on the 3.657 kg plant (PLANT-1 present).
     Re-gated on the corrected 2.657 kg plant — see `m2657_regate.md`."*
2. **Update the run-index table** at the top of the journal with the five new
   rows (`# | Run | Algorithm | Date | One-lever delta | Verdict`), and confirm
   R2/R2b added theirs (Runs 22-24: `v6_robust`, `v6_smoke`,
   `v6d_contact_wrench`).
3. **New file `docs/jetson-mod/v5_contact_results.md`** — the results document
   `v5_retrain_plan.md` mandates and DOC-2 records as missing. It is the
   companion to `v4_retrain_results.md`; copy that file's structure:
   - What the v5 campaign changed vs v4_robust (contact-rich track: fall-only
     terminations, obstacles, sustained wrench, disturbance-gated rewards).
   - The four arms and why each existed (one lever per arm).
   - Open-field grid table and contact-battery table (from `eval_results_v5/`).
   - Video audit verdicts.
   - Verdict: v5d shipped.
   - **A prominent header block, above the first `## ` heading**: every number in
     this document was measured on the 3.657 kg plant; the corrected-plant
     re-gate is `m2657_regate.md`; the post-Phase-M rebuild is
     `rebuild_results.md`; pointer to `known_issues.md#plant-1`.
   - A note that CFG-1 (dead `torque_z_range`) and CFG-3 (a disturbance gate no
     reward reads) were both live in v5d, so the "one lever" description of the
     wrench arm overstates what the lever did.
4. **Update `docs/jetson-mod/task_plan_v2.md`**, not `task_plan.md`. Add Phase R
   with its per-task status at the the end of the Phase R section marker and keep the
   status line current. (`task_plan.md`'s Task 2.8 was already corrected to
   COMPLETE by the DOC-4 fix on 2026-08-11 — leave it alone except to add a
   pointer to `v5_contact_results.md` at line ~1448.)
5. **Retire the DOC-2 check** in `scripts/verify_known_issues.py` (it is
   registered by the `@issue("DOC-2", …)` decorator at line 559) and add the
   `> **FIXED <date>.**` block to the DOC-2 entry in `known_issues.md`, per the
   DEPLOY-5 precedent. **Then update the `CONFIRMED n / m` block near the top of
   `known_issues.md`** — it currently reads `CONFIRMED 33 / 34` while the
   verifier actually reports `CONFIRMED 28 / 29`, so it is already stale and
   must not be left more so.

**3. Unit tests**

New file `tests/test_journal_completeness.py`, `@pytest.mark.phase2`. This one is
unavoidably text-based, but it asserts *structure* rather than the presence of a
literal that might be non-unique — the TEST-1 trap is grepping for a literal
whose count is > 1 and calling it a semantic check:

- `test_every_v5_and_v6_run_has_an_entry`: for each of the eight run names,
  assert the journal contains a heading line matching
  `^## Run \d+ — \`<name>\`` (a heading, not a mention).
- `test_run_index_table_covers_every_entry`: parse the run-index table rows and
  the `## Run` headings; assert the two sets of run names are equal. This is the
  check that catches the real failure mode — an entry added without updating the
  index.
- `test_v5_contact_results_exists_and_declares_its_plant`: assert the file
  exists and that a line containing `3.657` appears above the first `## `
  section heading (the plant banner is at the top, where a reader will see it).
- `test_no_v5_entry_claims_the_corrected_plant`: for each v5 entry's body,
  assert `2.657` appears only inside a sentence that also contains
  `m2657_regate.md`. Prevents exactly the DOC-3 error (docs asserting a 2.657 kg
  plant for results measured at 3.657 kg).

**4. Smoke test**

```bash
~/IsaacLab/_isaac_sim/python.sh $REPO/scripts/tb_summary.py \
  --run_dir $ISAACLAB/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25 --markdown
cd $REPO && python3 -m pytest tests/test_journal_completeness.py -q
cd $REPO && python3 scripts/verify_known_issues.py 2>&1 | tail -8
```
Observable: `tb_summary.py` prints a markdown table with a last-100 mean for
`Train/mean_reward` and a `Gait/duty_in_band_frac` last-100 mean of **0.9836**
(measured independently from the console log; a wildly different value means the
tag name or the accumulator `size_guidance` is wrong). The pytest run passes.
`verify_known_issues.py` no longer lists DOC-2, reports one fewer check than
before, and shows no REFUTED entries.

**5. Done when**

- [ ] `experiment_journal.md` has eight new `## Run N — \`<name>\`` entries
      (Runs 17-24) and eight new run-index rows.
- [ ] Every new entry names its data source per rule 4 (TB tag / JSON path /
      log dir), and every training number came from `tb_summary.py`, not a grep.
- [ ] `docs/jetson-mod/v5_contact_results.md` exists with the 3.657 kg banner
      above the first section heading.
- [ ] `task_plan_v2.md` contains Phase R with per-task status; `task_plan.md`
      Task 2.8 points at `v5_contact_results.md`.
- [ ] `tests/test_journal_completeness.py` passes (4 tests).
- [ ] `known_issues.md` DOC-2 carries a FIXED block, its verifier check is
      deleted, and the `CONFIRMED n / m` block at the top of the register
      matches what the verifier actually prints.
- [ ] `python3 -m pytest tests/ -q` passes overall (≥ 103 + the new tests).

---

# Phase S — Sim-to-real

_Bridging the trained policy to the physical robot. Sits between Phase 3 (CAD)
and Phase 4 (hardware build), and interleaves with Phase 4: S.0–S.4 and S.8 are
pure software and can start today; S.5–S.7 and S.9–S.12 need hardware that does
not exist yet._

---

## Why this phase exists as its own block

`docs/jetson-mod/task_plan.md` Task 4.4 ("Port Runtime Software to Jetson with
TensorRT", line 1999) and Task 4.5 ("Real Robot Walking Test", line 2162) are a
sketch: a `trtexec` invocation, a `TRTInfer` class stub, and a checklist that
says the robot "should balance and stand". Everything that actually decides
whether the policy transfers — the observation contract, the four unmeasurable
observation dimensions, latency, the torque ceiling, calibration, and the abort
criteria — is not in them. Phase S is that missing content.

**Read before starting any task in this phase:**

| File | What to take from it |
|---|---|
| `docs/jetson-mod/known_issues.md` | The whole DEPLOY section (DEPLOY-1..5), PLANT-1, PLANT-4..PLANT-8, PLANT-10, ART-1, EVAL-1, SHELL-2, TEST-1..TEST-3, DOC-6. This is the only defect register in the repo. |
| `AGENTS.md` § Observation Space (`:421`), § Joint Orders (`:455`), § Jetson Deployment (`:618`) | The 59-dim layout, all three joint orders, the legacy-15-joint warning, the GPIO map, the safety rules. |
| `isaac_lab_env/open_duck_mini_v2/robot_cfg.py` | Actuator model (BAM params, `:92-112`), `init_state.joint_pos` = `q_default` (`:46-70`), `articulation_root_prim_path` (`:79`), `soft_joint_pos_limit_factor=0.9` (`:81`). |
| `exported_policies/v5d_contact_wrench_ppo/env.yaml` | The authoritative observation term order (`:428-515`), per-term training noise, `actions.joint_pos.scale: 0.25` (`:625`), `decimation: 4` (`:84`), `sim.dt: 0.005` (`:21`), `commands.base_velocity` (`:976-1000`). |
| `scripts/duck_init_pos.json` | **Already contains** the 16-entry Isaac joint order and `q_default`. Reuse it. Do not create a second copy. It is already consumed by `scripts/verify_known_issues.py:453`. |
| `docs/sim2real.md`, `docs/configure_motors.md`, `docs/feetech_identification.md` | Upstream's MuJoCo-era path. `configure_motors.md:22-38` has the servo-ID table. `sim2real.md` is marked "Not finalized yet" and points at an external runtime repo — treat as background, not as instructions. |

**Verified environment facts you may rely on (measured 2026-08-11 on this
machine — re-check if any of them matters and time has passed):**

- Isaac Lab `2.3.2` at `~/IsaacLab`; robot MJCF `mini_bdx/robots/open_duck_mini_v2/robot_motors.xml`.
- System `python3`: `numpy 2.3.5`, `mujoco 3.6.0`, `onnxruntime 1.24.1` present;
  **`onnx` is NOT importable**, and **`pypot` is NOT installed**.
- `jq` and `ffmpeg` are on `PATH`.
- `pytest tests/` baseline is **`103 passed`** (the register's "101 passed" predates
  the PLANT-1 fix commit — do not treat 101 as the target).
- `python3 scripts/verify_known_issues.py` currently **exits 2**, printing
  `CONFIRMED 28 / 29` with `DEPLOY-1` INCONCLUSIVE because `onnx` is missing.
  Exit 2 = "something could not be evaluated"; exit 1 = "a check came back
  REFUTED". Only exit 1 is a regression.

**Standing rules for every task below**

1. **Do not commit or push** unless the owner explicitly asks.
2. **ONE Isaac Sim / GPU job at a time** (`AGENTS.md` journal-protocol rule 7,
   `:740`). Never start an eval or a recording while a training is running.
3. **Run tests as `pytest tests/`, never bare `pytest`** — bare `pytest` from the
   repo root dies at collection on `experiments/RL/old_test.py`
   (`ModuleNotFoundError: gymnasium`, TEST-2).
4. **The existing test suite is not a safety net.** 33 of the 47 tests in
   `tests/test_isaac_lab_env.py` are `assert "<literal>" in open(file).read()`;
   5 of 7 seeded regressions pass it, including deleting the dominant imitation
   reward (TEST-1). Every test this phase adds must *import and execute* code or
   *parse* data, never grep source text.
5. **Every training run gets a journal entry** in
   `docs/jetson-mod/experiment_journal.md` with last-100-iteration TensorBoard
   means, and every training command carries `--video --video_length 200
   --video_interval 5000`.
6. **`logs/` does not exist and is not in `.gitignore`.** Several tasks below
   write to `logs/`. Create it and add `logs/` to `.gitignore` in the first task
   that needs it, or the bring-up logs will silently become tracked files.
7. **There is no git-lfs tracking in this repo** (`.gitattributes` is absent,
   even though lfs filters are configured globally). Anything you add is a plain
   git blob. `.gitignore` ignores `*.pkl` (with two `!` exceptions), `*.onnx`
   (`:19`), `*.txt` (`:18`) and `*.zip` (`:17`). **`.npz` is not ignored.**

**Ordering / dependencies (corrected)**

```
S.0 ──┐
S.1 ──┴─> S.2 ──┬─> S.3 ──┬─> S.4 ──> S.5 ──> S.6 ──> S.7 ──┐
                └─> S.8 ──┘                                  │
                                                             ├─> S.11 ──> S.12
                                       S.9  ──> S.10 ────────┘
```

Corrections to the original graph, and why:

- **S.8 must run before S.7, not after.** S.8 is pure numpy over the S.2 traces —
  it needs no hardware and no GPU — and its output (whether to change
  `effort_limit_sim` / add actuator DR / apply the BAM friction terms) is an
  *input* to the S.7 retrain campaign. In the original graph S.7 could not fold
  in a decision that had not been made yet.
- **S.3 must be closed before S.4 is written** — it changes the shape of the
  observation builder. That was right and is kept.
- **S.9 and S.10 do not depend on S.6/S.7.** They depend on S.1/S.4 (contract +
  decoder) and on having a physical robot. They can proceed in parallel with the
  latency work.
- **S.9 and S.10 must both be complete before S.11 rung 3.**
- **Retrain sequencing risk, name it explicitly:** S.0 may declare a retrain
  mandatory *today*, while S.6 (which tells S.7 whether latency must be modelled)
  cannot run until a Jetson and a powered servo bus exist. The owner must choose
  between (a) waiting for hardware so a single campaign bundles PLANT-1 re-gate +
  antennas + actuator fidelity + latency, or (b) running an interim campaign now
  and a second one later. Record the choice in `docs/jetson-mod/m2657_regate.md`
  (Task R1c's verdict document); do not pick it silently.

**AI-agent suitability at a glance**

| Task | Suitable | Run after | The part a human must do |
|---|---|---|---|
| S.0 Re-gate on the mass-fixed plant | PARTIAL | — | Sign off the video audit |
| S.1 Freeze the deployment contract | YES | — | — |
| S.2 Record Isaac reference traces | YES | S.1 | — |
| S.3 Decide DEPLOY-3 (antennas) | PARTIAL | S.2 | Make the call; it may cost a retrain |
| S.8 Torque / current / thermal envelope | PARTIAL | S.2 | Supply the STS3250 datasheet |
| S.4 Hardware-free runtime core | YES | S.1, S.2, S.3 | — |
| S.5 Jetson inference + parity | PARTIAL | S.4 | Own a Jetson; run it |
| S.6 Measure loop latency | PARTIAL | S.5 | Physical bench with servos powered |
| S.7 Latency response in sim | YES | S.6, S.3, S.8 | Approve spending a retrain |
| S.9 Calibration (zeros, signs, IMU frame) | **NO** | S.1, S.4 | All of it — hands on the robot |
| S.10 Safety layer | PARTIAL | S.8, S.9 | Wire the physical kill switch; verify trips |
| S.11 Staged bring-up ladder | **NO** | S.5, S.9, S.10 | Every rung — holding and watching the robot |
| S.12 Bring-up log analysis | YES | S.11 | — |

---

### Task S.0 — Re-gate the shipped policy on the mass-fixed plant — **MOVED**

**AI-agent suitable:** N/A — this task no longer exists here.

> **This work is Phase R, Tasks R0 → R1 → R1b → R1c. Do not run a second
> re-gate campaign.**
>
> An earlier draft of this phase specified its own re-gate of the same pinned
> checkpoint (`exported_policies/v5d_contact_wrench_ppo/model_5998.pt`) on the
> same task, the same six conditions and the same protocol as Task R1. Running
> both would fork the engineering record into two directories that disagree.
>
> **Where the outputs live** — use these paths, and do **not** create
> `eval_results_v6_massfix/`, `v6_massfix_comparison.md` or
> `sim2real/S0_regate.md`:
>
> | Output | Path |
> |---|---|
> | Result JSONs | `docs/jetson-mod/eval_results_m2657/` |
> | Comparison table | `docs/jetson-mod/m2657_comparison.md` |
> | Verdict | `docs/jetson-mod/m2657_regate.md` |
>
> **Anywhere later in Phase S that says "S.0's verdict", read it as R1c's
> verdict in `m2657_regate.md`.**
>
> **Ordering, which matters more here than anywhere else in the plan:** R1 must
> run **before** Phase M's Task M0b. M0b changes the observation and action
> shapes (59/16 → 53/14), after which `OnPolicyRunner.load()` fails on the v5d
> checkpoint and re-gating it becomes impossible. If you are reading this after
> Phase M has landed and R1 was never run, **stop and report** — the measurement
> is no longer available, and that is a finding, not something to improvise
> around.

### Task S.1 — Freeze the deployment contract as a generated artifact — ✅ **DONE 2026-08-16** — generator + loader + 8 tests, `--check` exits 0

**1. Context for the implementing agent**

The exported ONNX is a bare MLP. Everything around it — what the 59 inputs mean,
what the 16 outputs mean, how to turn an output into a servo command — exists
only as prose spread across three documents, and one of the required pieces
(the gait-phase clock) is written down **nowhere**. DEPLOY-1 measures the cost of
getting it wrong: commanding the ONNX output directly is wrong by a 4× gain and a
standing-pose offset of up to 1.379 rad (`right_knee`) — "total, silent failure".
If this task is skipped, the runtime author will re-derive the contract by reading
prose, and the prose has been wrong before (`AGENTS.md` once listed a
right-leg-first joint order that was the legacy 15-joint BDX table).

Read first:
- `docs/jetson-mod/known_issues.md` § DEPLOY-1 (`:905`), DEPLOY-2 (`:925`),
  DEPLOY-4 (`:955`), DEPLOY-5 (`:963`).
- `AGENTS.md:421` § Observation Space (the table IS the concatenation order) and
  `AGENTS.md:455` § Joint Orders (three different orders, one of which is a trap).
- `scripts/duck_init_pos.json` — **already has** `joint_order` (Isaac/USD order)
  and `init_pos_rad`.
- `exported_policies/v5d_contact_wrench_ppo/env.yaml`: observations `:428-515`,
  actions `:617-628`, commands `:976-1000`, `sim.dt` `:21`, `decimation` `:84`.
- `isaac_lab_env/open_duck_mini_v2/imitation_reward.py:337-350`
  (`gait_phase_observation`), `:47` (`_instances`), `:168` (registration),
  `:200-205` (reset), `:246` (phase), `:332` (the step-counter advance).

Depends on: nothing (can run in parallel with S.0).

Traps:
- **Three joint orders exist.** MJCF/actuator order (left leg, head, antennas,
  right leg), Isaac/USD order (interleaved), and the legacy 15-joint BDX order in
  `mini_bdx/mini_bdx/utils/rl_utils.py`. **The policy uses the Isaac/USD order for
  both the observation joint blocks and the 16 action outputs**, because the
  action term is `joint_names: ['.*']` with `preserve_order: false`
  (`env.yaml:623-627`), which resolves to articulation order. The servo bus uses
  names, so you need a name-keyed map, never a hardcoded index list.
- `clip_actions: null` in `agent.yaml` and `clip: null` on the action term
  (`env.yaml:622`): the policy output is **unbounded**, and Isaac Lab's
  `JointPositionAction.process_actions` applies no limit clamp at all
  (`~/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169-182`).
  In sim, PhysX joint limits absorb an out-of-range target. On hardware nothing
  does — the runtime must clamp.
- DEPLOY-2's `0.01` normalizer epsilon is *inside* the ONNX graph (the `Sub`/`Div`
  nodes, confirmed by running the graph). It only bites a hand-rolled
  reimplementation of `policy.pt`. Use the ONNX and this is a non-issue — but
  record it so nobody re-derives normalization.
- `yaml.safe_load` **fails** on this `env.yaml` — verified:
  `ConstructorError: could not determine a constructor for the tag
  'tag:yaml.org,2002:python/tuple'` at line 2. `yaml.unsafe_load` succeeds.
- **`AGENTS.md:686` is corrupted.** The first bullet of § Safety Rules is a
  leftover *edit instruction* ("Apply the same line edit — ... — but ONLY as part
  of a three-file change: ...") rather than the rule itself. The rule it embeds
  is: clamp to `forward [-0.148, 0.222], lateral [-0.111, 0.111], turn
  [-0.3, 0.3]`. Note that the **turn clamp ±0.3 is deliberately inside** the
  trained hull of ±0.5. Derive `vx`/`vy` from `env_cfg.py:261-263` and record the
  turn clamp as a separate, conservative *policy choice* field — do not conflate
  the two.

**2. Low-level implementation plan**

Facts that must end up in the artifact. Every one was checked against the repo:

| Field | Value | Primary source |
|---|---|---|
| `obs_dim` | 59 | `policy.onnx` input `obs float32[1,59]` |
| `act_dim` | 16 | `policy.onnx` output `actions float32[1,16]` |
| Obs term order | `base_ang_vel(3)`, `projected_gravity(3)`, `velocity_commands(3)`, `joint_pos(16)`, `joint_vel(16)`, `actions(16)`, `gait_phase(2)` | `env.yaml:436,449,462,472,485,498,508` |
| Obs term funcs | `joint_pos` → `joint_pos_rel`; `joint_vel` → `joint_vel_rel`; `actions` → `last_action` | the `func:` line under each term |
| Obs slices | `[0:3] [3:6] [6:9] [9:25] [25:41] [41:57] [57:59]` | derived; must sum to 59 |
| `action_scale` | `0.25` | `env.yaml:625` `actions.joint_pos.scale` |
| Action offset | `q_default` (`use_default_offset: true` `:628`, `offset: 0.0` `:626`) | `joint_actions.py:194-195` |
| `q_default` | 16 values | `scripts/duck_init_pos.json` / `robot_cfg.py:48-69` |
| Joint order | Isaac/USD 16-name list | `scripts/duck_init_pos.json` `joint_order` |
| Control period | `0.02 s` (50 Hz) = `sim.dt 0.005 × decimation 4` | `env.yaml:21,84`; corroborated by `protocol.step_dt: 0.02` in every eval JSON |
| Reference library | name from `rewards.imitation_reward.params.reference_pkl`; **absent for v5d**, so the default `polynomial_coefficients.pkl` applies (`imitation_reward.py:104`) | `env.yaml:885-889` |
| Gait period | **27 control steps** (0.54 s) — uniform across all 240 entries of `polynomial_coefficients.pkl` *and* all 388 of `polynomial_coefficients_v2.pkl` | measured from the pickles |
| Command hull (trained) | vx `(-0.148, 0.222)`, vy `(-0.111, 0.111)`, wz `(-0.5, 0.5)` | `env_cfg.py:261-263`; `env.yaml:989-997` |
| Turn clamp (deployment choice) | wz `(-0.3, 0.3)` — deliberately inside the hull | `AGENTS.md:686`, DEPLOY-5 |
| Joint hard limits | 16 `jnt_range` pairs | `robot_motors.xml` |
| Soft limits | `mean ± 0.5 · range · 0.9` per joint | `robot_cfg.py:81` + `~/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:765-768` |
| Training obs noise | gyro ±0.2 rad/s, gravity ±0.05, joint pos ±0.01 rad, joint vel ±1.5 rad/s (uniform, additive) | `env.yaml:441-447, 454-460, 477-483, 490-496` |
| Servo IDs | 14 entries; antennas `null` | `docs/configure_motors.md:22-38` and `experiments/v2/configure_motors.py:10-25` (verified identical) |
| Servo bus baud | 1,000,000 | `experiments/v2/configure_motors.py:29-32` |

Steps:

1. Create the package: `jetson_runtime/__init__.py` (empty),
   `jetson_runtime/README.md` (one paragraph: "this package is the deployment
   contract and the runtime; the contract JSON is generated, never hand-edited").
   **`setup.cfg` packages only `mini_bdx`**, so `jetson_runtime` is importable
   only from the repo root. Every command in this phase that imports it must run
   with the repo root as cwd or on `PYTHONPATH`; say so in the README.
2. Create `scripts/generate_policy_contract.py`. It re-derives every field above
   **from primary sources at run time** — it must not contain a literal copy of
   `q_default`, the joint order, or the joint limits:
   - joint order + `q_default`: `json.load(scripts/duck_init_pos.json)`.
   - `action_scale`, obs term order, per-term noise, `decimation`, `sim.dt`,
     command ranges, `reference_pkl`: parse
     `exported_policies/v5d_contact_wrench_ppo/env.yaml`. **Plain `yaml.safe_load`
     raises** (verified). Use a tag-tolerant loader — subclass `yaml.SafeLoader`
     and register `add_multi_constructor("", lambda loader, suffix, node: ...)`
     returning the node's plain value — or use `yaml.unsafe_load` and write one
     sentence saying why it is acceptable (the file is repo-local and trusted).
   - joint hard limits: `mujoco.MjModel.from_xml_path(
     'mini_bdx/robots/open_duck_mini_v2/robot_motors.xml')` → `m.jnt_range`,
     keyed by `mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)`. **Skip the
     free joint** — it is joint 0, its name is `None` and its range is `[0, 0]`.
   - `gait_nb_steps`: `pickle.load` the library named by the contract's
     `reference_pkl` field (for v5d: `isaac_lab_env/open_duck_mini_v2/data/
     polynomial_coefficients.pkl`); assert every entry's `nb_steps_in_period` is
     equal; write that single value.
   - soft limits: `mean ± 0.5 · (hi − lo) · 0.9` per joint, computed here so the
     runtime never recomputes it.
   Modes: default writes `jetson_runtime/policy_contract.json`;
   `--check` regenerates in memory and diffs against the file on disk, exiting 1
   on any difference (this is the anti-drift guard).
3. Create `jetson_runtime/contract.py` — a thin loader that reads
   `policy_contract.json` and exposes `JOINT_ORDER`, `Q_DEFAULT` (np.array in
   joint order), `ACTION_SCALE`, `OBS_SLICES` (dict of name → `slice`),
   `CONTROL_DT`, `GAIT_NB_STEPS`, `SOFT_LIMITS_LOW/HIGH`, `HARD_LIMITS_LOW/HIGH`,
   `CMD_HULL`, `CMD_CLAMP`, `SERVO_IDS`, `SERVO_BAUD`. No numbers literal in this
   file.
4. Servo IDs: copy the 14-entry table into the **generator** as the one place a
   literal is unavoidable, and mark it clearly:
   `{right_hip_yaw:10, right_hip_roll:11, right_hip_pitch:12, right_knee:13,
   right_ankle:14, left_hip_yaw:20, left_hip_roll:21, left_hip_pitch:22,
   left_knee:23, left_ankle:24, neck_pitch:30, head_pitch:31, head_yaw:32,
   head_roll:33}`. **The two antennas have no bus ID** — they are SG90 PWM
   servos on Jetson header pins 32/33 (`AGENTS.md:635` GPIO table). Record them
   as `null` so the absence is explicit. Have the generator assert its literal
   table equals the dict parsed out of `experiments/v2/configure_motors.py`
   (`ast.literal_eval` of the `joints = {...}` assignment), so the one literal is
   still checked against a primary source.
5. Write `docs/jetson-mod/sim2real/deployment_contract.md`: the human-readable
   contract with, for every field, the file and line it was derived from, plus a
   plain-language statement of the action pipeline:
   `q_target[j] = q_default[j] + 0.25 * a[j]`, then clamp to soft limits, then
   convert to servo units — and the observation trap:
   `joint_pos_rel = q_measured − q_default`, **not** raw encoder angles.
   Also record, as an explicit non-fact: **`ticks_per_rad` is not derivable from
   this repo.** Nothing here states the STS3250's encoder resolution or its
   firmware angle units. It must be measured in S.9 and written into
   `calibration.json`, not guessed.
6. Add a line to `docs/jetson-mod/known_issues.md` DEPLOY-1 noting the contract
   file now exists (do not mark DEPLOY-1 fixed — the ONNX still omits the
   values; the register's convention is that a fixed issue loses its check).

**3. Unit tests**

> **File-ownership note — read before creating anything.** Three tasks in this
> plan touch a "deployment contract", and they are NOT the same artifact:
>
> | Task | Owns | Artifact |
> |---|---|---|
> | **R3** | the *machine* contract emitted with the ONNX | `exported_policies/<run>/deployment_contract.json` + `scripts/verify_deployment_contract.py` |
> | **S.1** (this task) | the *human* contract for the hardware runtime | `docs/jetson-mod/sim2real/deployment_contract.md` |
> | **V.1** | the *command* contract the VLM must respect | `jetson_runtime/command_contract.py` |
>
> R3 runs first and produces the JSON. **This task consumes that JSON — it does
> not re-derive `action_scale` or `q_default` from the graph.** If R3 has not
> run, stop and run it.
>
> Both R3 and this task declare `tests/test_deployment_contract.py`. **R3
> creates it; this task extends it.** Do not overwrite R3's tests — add yours
> and keep the total passing. If the file already exists, append.

New file (or **extend R3's**) `tests/test_deployment_contract.py` — all of these
import and execute:

- `test_slices_tile_the_vector`: the 7 slices are contiguous, start at 0, end at
  59, and no two overlap.
- `test_action_scale_matches_env_yaml`: parse `env.yaml` independently in the
  test and assert `contract.ACTION_SCALE == 0.25` and equals the parsed value.
- `test_joint_order_matches_duck_init_pos`: `contract.JOINT_ORDER ==
  json.load(duck_init_pos.json)["joint_order"]` and has length 16.
- `test_joint_names_match_mjcf`: `set(contract.JOINT_ORDER)` equals the set of
  **named** joints in `robot_motors.xml` (loaded with `mujoco`, free joint
  excluded), proving no typo and no legacy 15-joint contamination.
- `test_q_default_matches_robot_cfg`: parse `robot_cfg.py` with `ast` (not
  regex), pull the `init_state.joint_pos` dict, assert every value equals the
  contract's to 1e-12.
- `test_gait_period_is_uniform`: load **both** pickles and assert
  `{e["nb_steps_in_period"] for e in data.values()} == {27}` for each
  (240 entries and 388 entries respectively).
- `test_soft_limits_inside_hard_limits`: elementwise
  `hard_low <= soft_low < soft_high <= hard_high` for all 16 joints, and
  `q_default` lies strictly inside the soft limits for all 16.
- `test_generator_is_idempotent`: run `generate_policy_contract.py --check` via
  `subprocess` and assert exit code 0.

**4. Smoke test**

```bash
cd $HOME/Projects/Open_Duck_Mini_Jetson
python3 scripts/generate_policy_contract.py            # writes the JSON
python3 scripts/generate_policy_contract.py --check ; echo "exit=$?"   # 0
python3 -c "from jetson_runtime.contract import OBS_SLICES; print(sum(s.stop-s.start for s in OBS_SLICES.values()))"
```
Observable: the last command prints `59`, and `--check` exits 0.

**5. Done when**

- [ ] `jetson_runtime/policy_contract.json` exists and contains every field in
      the table above.
- [ ] `grep -Ec '0\.25|1\.368|left_hip_yaw' jetson_runtime/contract.py` prints
      `0` (no literals — everything is loaded). Note `grep -c` **exits 1** when
      the count is zero; check the printed number, not the exit status, and do
      not run this inside `set -e`.
- [ ] `pytest tests/test_deployment_contract.py -v` → 8 passed.
- [ ] `python3 scripts/generate_policy_contract.py --check` exits 0.
- [ ] `docs/jetson-mod/sim2real/deployment_contract.md` names a file+line for
      every field, and states that `ticks_per_rad` is deliberately absent.
- [ ] `pytest tests/` still passes.

**AI-agent suitable:** YES — pure repo work, no hardware, no GPU.

---

### Task S.2 — Record Isaac reference traces as offline ground truth — ✅ **DONE 2026-08-16** — 3 traces, 21 tests, lag decomposition measured

**1. Context for the implementing agent**

Everything from S.4 onward is a reimplementation of what Isaac Lab does between
the sensors and the servos. Without recorded ground-truth traces you can only
test the reimplementation against your own reading of the code — which is exactly
how the 4× action-scale error in DEPLOY-1 would survive. This task produces
frozen `.npz` files that later tasks assert against, on any machine, with no GPU.

Read first:
- `scripts/evaluate_policies.py` — the AppLauncher bootstrap at `:1140-1230`
  (arg parsing before `AppLauncher`; `import isaac_lab_env` at `:1152` before
  `gym.make`; the `handle_deprecated_rsl_rl_cfg` shim at `:1189/1221` that is
  *required* with the installed rsl-rl), the corruption kill at `:1343`, the
  command pin at `:1386-1416`, and `gym.make` at `:1590`.
- `scripts/play_policy.py` — it simply `exec`s Isaac Lab's own `play.py`, so it
  is **not** a template for a custom recording loop; use `evaluate_policies.py`.
- `jetson_runtime/policy_contract.json` from S.1.

Depends on: S.1.

Traps:
- **Command pinning.** Do **not** write into `env.command_manager._terms[...]
  .vel_command_b`. `OpenDuckContactWrenchEnvCfg_PLAY` leaves `heading_command:
  true` with `rel_heading_envs: 1.0` (`env.yaml:984,987`), so
  `UniformVelocityCommand._update_command` overwrites the yaw channel **every
  step** with `clip(0.5·heading_error, ±0.5)` (PLANT-8) and any value you poke in
  is destroyed. Copy `apply_condition` (`evaluate_policies.py:1386-1416`)
  instead: set `term.cfg.heading_command = False`, `term.cfg.rel_standing_envs =
  0.0`, and the three `term.cfg.ranges.*` to degenerate `(c, c)` pairs, then
  reset the env so every env resamples.
- **Corruption.** The Play cfg already sets
  `observations.policy.enable_corruption = False` (`env_cfg.py:797`). Set it
  again explicitly before `gym.make` and assert it, so the trace cannot silently
  contain a training noise draw.
- **Episode boundaries destroy the trace.** `ImitationReward.reset`
  (`imitation_reward.py:200-205`) zeroes `_step_idx` on termination, and
  `last_action` resets too. The Play cfg's `episode_length_s = 40.0`
  (`env_cfg.py:794`) means 1500 steps × 0.02 s = 30 s fits inside one episode —
  but a fall would still truncate it. Record `terminated` and `truncated` per
  step and **fail the recording** if any is True.
- **The gait phase comes from a module-level `_instances` registry** keyed by
  `id(env)` (`imitation_reward.py:47,168,339`). It only advances because the
  *reward* term is evaluated (`:332`). If you build an env without the imitation
  reward, `gait_phase` silently returns zeros. Use the ContactWrench Play task
  unmodified.
- **`applied_torque` is an approximation, not a measurement.** The robot uses
  `ImplicitActuatorCfg`, and `ImplicitActuator.compute`
  (`~/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:118-141`)
  computes `computed_effort = kp·(target−q) + kd·(0−qd)` in Python and clips it
  to `effort_limit_sim` to produce `applied_torque`. PhysX's real internal torque
  is not exposed. It is also recomputed once per **physics** step inside the
  decimation loop (`manager_based_rl_env.py:182-195`), so a per-control-step read
  samples only the last of the four substeps. Record **both** `computed_torque`
  (unclipped) and `applied_torque` (clipped at 8.716 N·m) and state this caveat
  in the meta.
- Rule 7: one GPU job at a time.

**2. Low-level implementation plan**

1. Create `scripts/record_reference_trace.py`, modelled on the
   `evaluate_policies.py` bootstrap (argument parsing before `AppLauncher`,
   `import isaac_lab_env` before `gym.make`, `handle_deprecated_rsl_rl_cfg` when
   loading the agent cfg).
2. Arguments: `--task` (default
   `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0`), `--checkpoint`
   (default `exported_policies/v5d_contact_wrench_ppo/model_5998.pt`),
   `--num_envs 1`, `--steps 1500`, `--seed 42`, `--command "0.2,0,0"`,
   `--out docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz`.
3. Before stepping: disable corruption; apply the `apply_condition` pattern for
   the command; reset the env once so the pin takes effect.
4. Each of the `--steps` control steps, append to lists:
   - `obs` (59,) — the exact tensor handed to the policy
   - `action` (16,) — the raw policy output, before scaling
   - `joint_pos` (16,), `joint_vel` (16,) — **absolute**, from
     `env.scene["robot"].data.joint_pos[0]` / `joint_vel[0]`
   - `joint_pos_target` (16,) — `env.scene["robot"].data.joint_pos_target[0]`
   - `root_quat_w` (4,) (w,x,y,z), `root_ang_vel_b` (3,), `root_lin_vel_b` (3,),
     `projected_gravity_b` (3,)
   - `command` (3,), `gait_phase` (2,), `step_idx` (int)
   - `applied_torque` (16,), `computed_torque` (16,) — for S.8
   - `terminated` (bool), `truncated` (bool)
5. **Write down the indexing convention in the script's docstring and in the
   meta, because every later test depends on it:** row `i` of `obs` is the input
   that produced row `i` of `action`. Therefore `obs[i][41:57] == action[i-1]`
   and `obs[0][41:57]` is all zeros.
6. Save with `np.savez_compressed`, plus a `meta` dict as a 0-d object array:
   task id, checkpoint path, checkpoint MD5 (assert it equals
   `0333e68a4cd9ed3817310ed80f6715e4` for `model_5998.pt`), seed, command,
   Isaac Lab version (`2.3.2`), joint order (from the contract), git SHA, the
   `applied_torque` caveat, and `enable_corruption: false`.
7. Record **three** traces so later tests cover more than one command cell:
   `0.2,0,0`, `0,0,0.3`, and `0,0,0` (standing). Name them
   `reference_trace_v5d_fwd.npz`, `_turn.npz`, `_stand.npz`. Run them **one at a
   time** (rule 7).
8. Also emit `docs/jetson-mod/sim2real/reference_trace_stats.json` — per trace,
   per joint: `|a_t − a_{t−1}|` p50/p95/max, and `|q_target − q_default|` max.
   S.11 rung 2's abort criterion is defined against the sim action-rate p95, so
   that number has to exist before S.11 can be written.
9. Size and version control: `.npz` is **not** gitignored and this repo has **no
   `.gitattributes`, so git-lfs tracks nothing**. Expect roughly 1 MB per trace
   uncompressed (~170 float32 per step × 1500). Check the actual sizes; if a
   trace exceeds ~5 MB, downcast `computed_torque`/`applied_torque` to float16
   or add `docs/jetson-mod/sim2real/*.npz` to `.gitignore` and say which you did.

**3. Unit tests**

New `tests/test_reference_trace.py`, executed against the recorded files (skip
cleanly with `pytest.skip` if the files are absent, and register the marker):
- `test_trace_shapes`: `obs.shape == (steps, 59)`, `action.shape == (steps, 16)`.
- `test_no_episode_boundary`: `terminated` and `truncated` are all False.
- `test_obs_joint_block_is_relative`: for every step,
  `obs[:, 9:25] ≈ joint_pos − q_default` within 1e-5. **This is the single most
  valuable assertion in the phase** — it is DEPLOY-1's joint-block trap, proven
  from data rather than from prose.
- `test_obs_joint_vel_block_is_absolute`: `obs[:, 25:41] ≈ joint_vel` within
  1e-5 (the default joint velocity is zero, so `joint_vel_rel` is numerically
  absolute).
- `test_last_action_block_is_previous_action`: `obs[1:, 41:57] ≈ action[:-1]`
  within 1e-6 and `obs[0, 41:57]` is all zeros. Pins the indexing convention.
- `test_gait_phase_is_a_27_step_circle`:
  `obs[:, 57] ≈ cos(2π·step_idx/27)` and `obs[:, 58] ≈ sin(2π·step_idx/27)`
  within 1e-5, and `step_idx` increments by 1 mod 27.
- `test_command_block_matches_requested`: `obs[:, 6:9]` equals the `--command`
  triple on every step. This proves the heading servo was actually disabled —
  check the **yaw** channel specifically (PLANT-8).
- `test_checkpoint_md5`: the `meta` MD5 equals the value in
  `exported_policies/v5d_contact_wrench_ppo/README.md`.

**4. Smoke test**

```bash
cd $HOME/Projects/Open_Duck_Mini_Jetson && python3 -c "
import numpy as np; d=np.load('docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz', allow_pickle=True)
print(d['obs'].shape, d['action'].shape)
print(d['obs'][:3,57:59], (d['obs'][:,57]**2+d['obs'][:,58]**2).max())"
```
Observable: prints `(1500, 59) (1500, 16)` and three `[cos, sin]` pairs whose
squared norms are 1.0 to floating-point precision.

**5. Done when**

- [ ] Three `.npz` traces and one `reference_trace_stats.json` exist under
      `docs/jetson-mod/sim2real/`.
- [ ] `pytest tests/test_reference_trace.py -v` → 8 passed.
- [ ] Each trace's `meta` records the checkpoint MD5, seed, command, Isaac Lab
      version, git SHA, and the `applied_torque` caveat.
- [ ] `enable_corruption=False` is set explicitly in the recording script and
      asserted at runtime.
- [ ] No trace contains a terminated or truncated step.
- [ ] The trace file sizes are recorded, and the git decision (commit vs ignore)
      is stated in the S.2 section of `deployment_contract.md`.

**AI-agent suitable:** YES — GPU job plus analysis, no hardware.

---

### Task S.3 — Confirm DEPLOY-3 is closed (or execute the fallback)

**AI-agent suitable:** YES for the confirmation path. PARTIAL for the fallback
(the choice is the owner's; the measurement and the code are the agent's).

**1. Context for the implementing agent**

Four of the observation inputs cannot be measured on the physical robot: the two
antenna joints (indices 13, 14) drive open-loop SG90 servos with **no position
feedback**, and they occupy `joint_pos_rel[22, 23]` and `joint_vel_rel[38, 39]`
in the 59-dim vector. Feeding zeros is distribution shift; feeding the commanded
angle is also distribution shift. There is no correct value to supply. That is
`known_issues.md` DEPLOY-3, a Phase-4 blocker.

**In this plan that decision is already made, in Phase M.** Task M0b removes the
antennas from the action and observation spaces entirely (59/16 → 53/14), which
deletes the four dims rather than filling them, and closes PLANT-4 in the same
change. M0b runs long before Phase S. **This task is therefore a verification,
not a decision** — do not re-open it and do not offer the owner a menu of
options they have already been past.

Read first:
- `docs/jetson-mod/known_issues.md`, entry DEPLOY-3 (locate by heading, not line
  number)
- Task M0b in this document
- `docs/jetson-mod/task_plan_v2.md` Phase R, Task R1c's verdict document

**2. Low-level implementation plan**

1. Confirm M0b landed. Build any env and read the shapes back from the running
   env, not from a config file:
   ```python
   obs, _ = env.reset()
   print(obs["policy"].shape[-1], env.action_manager.total_action_dim)
   ```
   Expect `53 14`.
2. Confirm the register agrees: DEPLOY-3 and PLANT-4 are marked FIXED in
   `docs/jetson-mod/known_issues.md`, and `python3 scripts/verify_known_issues.py
   PLANT-4` reports REFUTED.
3. Confirm no antenna dim survives anywhere in the runtime contract produced by
   Task S.1 — the contract's `obs_layout` must contain no antenna entry.
4. **If and only if M0b was NOT taken** (the owner chose to keep the antennas in
   the policy), this becomes a real decision and it is the owner's, not yours.
   The options, worst to best, are: feed zeros; feed the last commanded angle;
   add position feedback to the antennas in hardware; or drop them from the
   spaces after all, which is M0b. Measure the cost of the first two before
   proposing either — substitute each candidate into the Task S.2 reference
   traces and report the resulting action delta in radians. **Do not pick
   silently, and do not proceed to S.4 until it is written down** in
   `docs/jetson-mod/m2657_regate.md` alongside the rest of the campaign record.

**3. Unit tests**

None applicable on the confirmation path — it asserts against a running env,
which is the smoke test below. On the fallback path, add a test that the
contract's `obs_layout` length equals the env's reported observation dimension.

**4. Smoke test**

```bash
cd ~/IsaacLab && ./isaaclab.sh -p /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/scripts/audit_plant_mass.py --headless
python3 scripts/verify_known_issues.py PLANT-4
```

Observable: the audit exits 0, and PLANT-4 reports **REFUTED** (meaning fixed).

**5. Done when**

- [ ] A freshly built env reports observation 53 and action 14
- [ ] `known_issues.md` marks DEPLOY-3 and PLANT-4 FIXED
- [ ] `scripts/verify_known_issues.py PLANT-4` reports REFUTED
- [ ] No antenna entry appears in the Task S.1 deployment contract
- [ ] If M0b was not taken: the chosen fallback is written down with its
      measured action-delta cost, and the owner has signed off

---

### Task S.8 — Establish the servo torque, current and thermal envelope — ✅ **DONE 2026-08-16** — verdict ACCEPT; firmware thresholds now CITED, not UNVERIFIED

> **Run this after S.2 and before S.7.** It is pure numpy over the S.2 traces —
> no hardware, no GPU — and its output is an input to the S.7 retrain campaign.
> The original plan placed it after S.7, which made that impossible.

**1. Context for the implementing agent**

The simulator lets the policy pull **8.716 N·m** per joint. Where that comes from
and what the servo actually does:

| Quantity | Value | Ratio to sim ceiling |
|---|---|---|
| Sim `effort_limit_sim` (BAM `kt·V/R`, at 12.1 V) | 8.716 N·m | 1.00× |
| Same, scaled to the pack's 11.1 V nominal | 7.996 N·m | 0.92× |
| STS3250 datasheet **stall** (50 kg·cm) | 4.903 N·m | 0.56× |
| STS3250 **continuous rated** (16 kg·cm) | 1.569 N·m | **0.18×** |

BAM source, verified in `experiments/v2/params_sts3250_id008.json`:
`kt = 1.0005874626213263`, `R = 1.3890462492623645`, `vin = 12.1`,
`mujoco_export.forcerange = 8.716130441407099`,
`friction_viscous = 0.6256393187557033`, `max_velocity = 8.93762009169868`.

So the simulated ceiling is **1.78× the datasheet stall and ≈5.6× the continuous
rating** (PLANT-5). And **no domain randomization touches any actuator
parameter** — not stiffness, not damping, not the effort limit (`AGENTS.md:365`,
"Not randomized at all").

The servo firmware is also believed to cut out on its own at **>3.8 A for 2 s,
>80 % stall for 2 s, and >70 °C** — a firmware cutout is a *silent torque-off*,
i.e. the leg goes limp mid-stride. **Those three thresholds appear nowhere in
this repo.** They are not verified facts here. Sourcing and citing them is part
of this task; until they are cited they must be marked `"source": "UNVERIFIED"`
in any artifact that carries them.

Two more actuator-fidelity defects belong in the same bundle:
- **PLANT-6**: `friction=0.2` is set but `dynamic_friction` and `viscous_friction`
  are `null` in both actuator groups. `None` does not mean "use the sibling"; per
  `actuator_base.py:176-192` it means "read from the USD", and the USD authors no
  joint friction, so the read-back is `0.0`. In Isaac
  (`articulation.py:888-889`), static friction caps effort **only at rest**;
  dynamic friction is what acts during motion. So the BAM dry friction is present
  when the robot stands still and **absent for the entire gait** — the direction
  that flatters the policy. BAM also identifies `friction_viscous = 0.6256` and
  nothing applies it.
- **PLANT-4**: the two antennas carry the full STS3250 parameter set.

Read first: `docs/jetson-mod/known_issues.md` § PLANT-5 (`:426`), PLANT-4
(`:404`), PLANT-6 (`:445`); `experiments/v2/params_sts3250_id008.json`;
`AGENTS.md:218` § Motor Parameters.

Depends on: S.2 (the traces carry `applied_torque` and `computed_torque`).

Traps:
- **The torque in the trace is a Python-side approximation, not a PhysX
  measurement.** `ImplicitActuator.compute`
  (`~/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:118-141`)
  computes `kp·(target − q) + kd·(0 − qd)` and clips it to `effort_limit_sim`.
  Consequences: (a) `applied_torque` **can never exceed 8.716 N·m by
  construction**, so "fraction above 8.716" is not a meaningful statistic —
  use `computed_torque` for that; (b) it excludes armature and friction effects;
  (c) it is refreshed once per 5 ms physics step inside the decimation loop, so a
  per-control-step read under-samples peaks by up to 4×. State all three in the
  document.
- **`effort_limit` vs `effort_limit_sim`.** `robot_cfg.py:99,110` sets
  `effort_limit_sim` and leaves `effort_limit` at `None`. Contrary to a common
  reading of DOC-6, in Isaac Lab 2.3.2 these are *aliased* for implicit
  actuators (`actuator_pd.py:59-79`): setting only `effort_limit` emits a
  deprecation warning and is then copied into `effort_limit_sim`. The real hazard
  is different and worse — **setting both to different values raises
  `ValueError` at construction**. So if you change the ceiling, change
  `effort_limit_sim` and leave `effort_limit` alone.

**2. Low-level implementation plan**

1. Write `scripts/measure_torque_headroom.py` (pure numpy, reads the S.2 traces):
   per joint, report p50 / p95 / p99 / max of `|applied_torque|` **and**
   `|computed_torque|`, plus the fraction of control steps above 1.569 N·m
   (continuous), above 4.903 N·m (datasheet stall) and above 7.996 N·m
   (pack-voltage ceiling). Also report the longest *consecutive* run above
   1.569 N·m in seconds — because the firmware protections are duration-based,
   a brief peak and a 2 s sustained load are completely different risks and a
   percentile alone cannot tell them apart.
2. Put the table in `docs/jetson-mod/sim2real/S8_torque_envelope.md`, together
   with the four rows of the table above, the three under-sampling caveats, and
   the three firmware thresholds each cited to the datasheet by page — or marked
   `UNVERIFIED` if no datasheet was obtained.
3. Decide, and record the decision with its rationale:
   - **If no joint sustains > 1.569 N·m for more than ~0.5 s and the max stays
     below 4.903 N·m** → the policy is not leaning on torque the hardware lacks.
     Accept, and rely on the S.10 runtime limiter as the guard.
   - **Otherwise** → fold an actuator-fidelity change into the S.7 campaign:
     set `effort_limit_sim = 4.903` (datasheet stall, both groups in
     `robot_cfg.py:99,110`), add actuator DR (`effort_limit_sim` ±15 %,
     `stiffness` ±20 %, `damping` ±20 %), set `dynamic_friction = 0.200` and
     `viscous_friction = 0.6256393187557033` from the BAM file, and apply the S.3
     antenna decision. Note honestly in the document that changing
     `effort_limit_sim` makes the resulting policy incomparable to every prior
     gate number — which is already true after PLANT-1, so this is the cheapest
     moment to do it.
4. Add to `jetson_runtime/policy_contract.json` **via S.1's generator** (not by
   hand-editing the JSON):
   `servo_continuous_torque_nm: 1.569`, `servo_stall_torque_nm: 4.903`,
   `sim_effort_limit_nm: 8.716`, `servo_current_limit_a: 3.8`,
   `servo_current_limit_window_s: 2.0`, `servo_temp_warn_c: 55`,
   `servo_temp_stop_c: 65`, `servo_firmware_temp_cutout_c: 70`, plus a
   `servo_limits_source` field naming the datasheet page or the string
   `"UNVERIFIED"`. **`servo_temp_warn_c` and `servo_temp_stop_c` are engineering
   choices, not measurements** — label them as such. The stop threshold is
   deliberately **below** the firmware cutout so the runtime stops the robot in a
   controlled way before the firmware drops torque without warning.
5. Update `docs/jetson-mod/known_issues.md` § PLANT-5 and § PLANT-6 with the
   measured headroom numbers and whichever decision was taken.

**3. Unit tests**

- `tests/test_torque_headroom.py::test_percentiles_and_run_length`: feed the
  analysis function a synthetic torque trace with a known 1.2 s excursion above
  threshold and assert the reported longest run is 1.2 s ± one sample.
- `::test_unit_conversions`: assert `50 kg·cm → 4.9033 N·m` and
  `16 kg·cm → 1.5691 N·m` to 1e-4 using the conversion helper, so the constant
  0.0980665 is exercised rather than trusted.
- `::test_contract_thresholds_ordered`: assert
  `temp_warn < temp_stop < firmware_cutout` and
  `continuous < stall < sim_effort_limit` in the contract file. Cheap, and it
  catches a transposed pair of numbers, which is the realistic failure.
- `::test_applied_torque_is_clipped_at_sim_limit`: over all three traces, assert
  `max|applied_torque| <= 8.716 + 1e-6`. This is not a robot fact — it is proof
  that you understood which buffer you are reading.

**4. Smoke test**

```bash
cd $HOME/Projects/Open_Duck_Mini_Jetson && python3 scripts/measure_torque_headroom.py \
  --trace docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz
```
Observable: a 16-row table. The decision-relevant cell is the maximum
`longest run above 1.569 N·m` across the 10 leg joints, in seconds — compare it
against the firmware's 2 s window.

**5. Done when**

- [ ] `docs/jetson-mod/sim2real/S8_torque_envelope.md` has the per-joint table,
      the four-row ceiling comparison, the three approximation caveats, and the
      three firmware thresholds either cited to a datasheet page or explicitly
      marked UNVERIFIED.
- [ ] The document states a decision (accept, or change + retrain) with rationale.
- [ ] `policy_contract.json` carries all the servo-limit fields plus
      `servo_limits_source`, written by the generator; `--check` exits 0.
- [ ] `pytest tests/test_torque_headroom.py -v` → 4 passed.
- [ ] `known_issues.md` PLANT-5 and PLANT-6 carry the measured numbers.

**AI-agent suitable:** PARTIAL — the agent does all the sim-side measurement, the
analysis and the config change. **A human must supply the STS3250 datasheet** for
the three firmware thresholds (they are not in the repo), and any bench
measurement of real stall current needs hardware and a current meter.

---

### Task S.8b — Bench-measure the servo thermal envelope — 🔧 **HARDWARE, parts ordered 2026-08-16**

**AI-agent suitable:** NO — this is a physical measurement. The agent wrote the
protocol, will process the logs, and will fold the results back into the
documents; a human runs the bench.

**Why this task was added.** S.8 is *"pure numpy over the S.2 traces — no
hardware"*, and **no other Phase-S task measures real servo current or
temperature**. That was a gap: every thermal number in this project — including
the ≤1.0 N·m target `v7_servo_safe` missed and a second retrain could not reach —
comes from a derating calculation over thermal behaviour **Feetech does not
publish**. One servo and an afternoon replaces the assumption with a measurement.

**The question:** the shipped policy commands **2.060 N·m RMS** at its worst leg
joint. On a real servo, does that plateau below the firmware's 70 °C torque-off,
or climb until the joint goes limp?

Full protocol, wiring, safety and the load table:
[`bench_test_servo.md`](bench_test_servo.md).

**Depends on:** parts only. Independent of S.1/S.2/S.8, which can run first and
should — S.8's sim-side envelope tells you what to look for on the bench.

**5. Done when**

- [ ] `docs/jetson-mod/bench_results_servo.md` carries the raw logs, the measured
      Kt, and a time-to-70 °C (or "stable at X °C") for each torque level.
- [ ] `servo_torque_budget.md`'s derating assumptions are replaced by measured
      numbers, or explicitly reaffirmed against them.
- [ ] `known_issues.md` **SERVO-2** is either closed or restated with a real
      target.
- [ ] The three firmware thresholds in S.8 lose their `"UNVERIFIED"` marking, or
      are shown to be wrong.

---

### Task S.4 — Build the hardware-free runtime core

**1. Context for the implementing agent**

This is the layer between the sensors and the ONNX, and between the ONNX and the
servos. It must be written so it can be fully tested **without a robot** —
otherwise its first test is the robot falling over. The whole point of S.2's
traces is that this module can be proven correct before any hardware exists.

Read first:
- `jetson_runtime/policy_contract.json` and
  `docs/jetson-mod/sim2real/deployment_contract.md` (S.1).
- `docs/jetson-mod/sim2real/S3_deploy3_decision.md` (S.3) — the chosen antenna
  option.
- `isaac_lab_env/open_duck_mini_v2/imitation_reward.py:337-350` — the phase term.
- `docs/jetson-mod/known_issues.md` § DEPLOY-1, DEPLOY-2, DEPLOY-4, ART-1.

Depends on: S.1, S.2, S.3.

Traps:
- **The gait-phase clock is documented nowhere in the repo and it is not
  optional** — it is `obs[57:59]`. In training, `ImitationReward` keeps a per-env
  integer `_step_idx` that increments once per control step (`:332`) and wraps at
  the matched motion's `nb_steps_in_period` (`:246`). Measured:
  `nb_steps_in_period == 27` for **all 240 entries** of
  `polynomial_coefficients.pkl` (the library v5d trained against) **and all 388**
  of `polynomial_coefficients_v2.pkl`, i.e. period 0.54 s. **So the matched
  motion does not affect the phase at all** and the runtime needs only a
  free-running mod-27 counter at 50 Hz:
  `phase = 2π·(k mod 27)/27`, `obs[57:59] = [cos(phase), sin(phase)]`.
  Verify that uniformity yourself in the test — do not take it on faith, because
  a future reference library could break it.
- `_step_idx` resets to 0 on episode termination in sim (`:200-205`). On hardware
  there are no episodes: reset it when the policy is (re)started, and **do not**
  reset it on a stumble unless you have decided that a stumble is an episode
  boundary. Write down whichever you choose, in `deployment_contract.md`.
- `base_ang_vel` is `asset.data.root_ang_vel_b` — the **body-frame** angular
  velocity of the root body — and since the PLANT-1 fix the root body is
  `trunk_assembly` (`robot_cfg.py:79`). `projected_gravity` is
  `asset.data.projected_gravity_b`, the world gravity direction expressed in that
  same body frame, i.e. `R_world→body · (0, 0, −1)`, a **unit** vector,
  ≈ `(0, 0, −1)` when upright.
- `last_action` is `env.action_manager.action`, the **raw, unscaled** policy
  output from the previous step, not the joint target. On the first step it is
  zeros. The S.2 traces pin the convention: `obs[i][41:57] == action[i-1]`.
- The policy output is unbounded (`clip_actions: null`, `clip: null`) and
  `JointPositionAction` applies **no clamp of any kind** — PhysX clamps to
  **hard** limits, not soft. So the recorded `joint_pos_target` may legitimately
  sit outside the soft limits, and the affine `q_default + 0.25·a` must be tested
  **before** clamping. Clamp in the runtime, count the clamps, and treat a
  nonzero count as a signal rather than a bug.
- **ART-1**: `exported_policies/v5d_contact_wrench_ppo/policy.onnx` is
  **gitignored** (`.gitignore:19 *.onnx`) while its weight sidecar
  `policy.onnx.data` (787,968 bytes) **is tracked**. The file is present in this
  working tree today, but a fresh clone gets an orphan `.data` and no graph, and
  `onnxruntime` fails with a confusing external-data error. Check it exists
  before starting; if it is missing, re-export it from `model_5998.pt` via Isaac
  Lab's play/export path, or fix the ignore rule — and say which you did in the
  document.

**2. Low-level implementation plan**

1. `jetson_runtime/gait_phase.py` — class `GaitPhase(nb_steps)`, methods
   `reset()`, `step() -> (cos, sin)`, property `k`. No dependencies.
2. `jetson_runtime/obs_builder.py` — class `ObsBuilder(contract)`:
   ```
   build(gyro_b,            # (3,) rad/s, body frame, already IMU-frame-corrected
         gravity_b,         # (3,) unit vector, body frame
         q_meas,            # (16,) rad, absolute, IN CONTRACT JOINT ORDER
         qd_meas,           # (16,) rad/s
         command,           # (3,) vx, vy, wz — already clamped
         last_action        # (16,) raw previous ONNX output
        ) -> np.ndarray (59,) float32
   ```
   Internals: `joint_pos_rel = q_meas − q_default`; `joint_vel_rel = qd_meas`
   (the default joint velocity is zero, so `_rel` is numerically absolute — say
   this in a comment so nobody "fixes" it); antenna dims filled per
   `deploy3_option`; phase appended from the `GaitPhase` instance.
   The builder must **assert the output dtype is float32 and shape (59,)** and
   raise on NaN — cheap, and it is the failure mode that shows up as a robot
   convulsing.
3. `jetson_runtime/action_decoder.py` — class `ActionDecoder(contract)`:
   - `raw_target(action_16) -> q_target_16` — the pure affine
     `q_default + 0.25·action`, **no clamping**. This is the function the trace
     test compares against.
   - `decode(action_16) -> (q_target_16, n_clamped)` — `raw_target` then
     `np.clip` to soft limits, returning the clamp count.
   - `to_servo_units(q_target)` returning per-servo firmware ticks using the
     calibration file from S.9 (accept a `calibration=None` default that applies
     identity offsets and signs so this module is usable before S.9 exists).
4. `jetson_runtime/imu.py` — pure math only in this task:
   `quat_to_projected_gravity(q_wxyz) -> (3,)` (state the quaternion convention
   explicitly: **w, x, y, z**, matching Isaac's `root_quat_w`; the BNO055's
   quaternion register order must be adapted at the driver, not here) and
   `rotate_gyro(gyro_imu, R_imu_to_body) -> (3,)`. No I2C code yet (that is
   S.9/S.10).
5. `jetson_runtime/policy.py` — an inference wrapper with two backends selected
   at construction: `onnxruntime` (works on any machine, used for all offline
   tests) and `tensorrt` (S.5). Both expose `infer(obs59) -> action16`.
   Reshape to `(1, 59)` — DEPLOY-4: the exported graph has a **hard-fixed batch
   dimension of 1**; batch 2 raises `onnxruntime.capi.onnxruntime_pybind11_state
   .InvalidArgument` (verified).
6. `jetson_runtime/loop.py` — the control loop skeleton with pluggable
   `SensorSource` and `ActuatorSink` protocols, plus a `ReplaySensorSource` that
   feeds a recorded trace, and a `--report` mode. No hardware imports at module
   scope, so the whole package imports on a laptop.

**3. Unit tests**

New `tests/test_runtime_core.py`, all executing against the S.2 traces:

- `test_obs_builder_reproduces_isaac`: for each of the three traces, feed the
  recorded `root_ang_vel_b`, the recorded `projected_gravity_b` (and separately
  the value recomputed from `root_quat_w`, asserting they agree to 1e-6),
  `joint_pos`, `joint_vel`, `command` and `action[i-1]` into `ObsBuilder`, and
  assert `max |obs_built − obs_recorded| < 1e-5` over all 59 dims and all steps
  — **excluding** dims 22/23/38/39 if the chosen DEPLOY-3 option is not C.
  Report the per-dim max error so a single bad dim is identifiable.
- `test_action_decoder_reproduces_joint_targets`: assert
  `max |raw_target(recorded_action) − recorded joint_pos_target| < 1e-5`.
  This is the DEPLOY-1 assertion: if `action_scale` or `q_default` is wrong, this
  test fails by ~1 rad, not by rounding. Use `raw_target`, **not** `decode` —
  Isaac does not clamp.
- `test_clamp_count_is_reported_not_assumed`: run `decode` over all three traces
  and assert the returned `n_clamped` matches an independently computed
  `np.sum(raw_target < soft_low) + np.sum(raw_target > soft_high)`. Record the
  actual number in `deployment_contract.md`. **Do not assert it is zero** — the
  policy is unbounded and Isaac never clamped it, so a nonzero count is a fact
  about the policy, not a bug in your code.
- `test_onnx_reproduces_recorded_actions`: run `policy.onnx` on the recorded
  observations, one row at a time, and assert
  `max |action_onnx − action_recorded| < 1e-4` (fp32 graph vs the torch policy;
  tighten if it comes out much smaller).
- `test_gait_phase_matches_trace`: the standalone `GaitPhase`, started at the
  trace's first `step_idx`, reproduces `obs[:, 57:59]` to 1e-6.
- `test_nb_steps_uniform`: re-derive `nb_steps_in_period` from **both** pickles
  and assert each set is `{27}`, so the mod-27 shortcut is guarded against a
  library change.
- `test_package_imports_without_hardware`: `import jetson_runtime.loop` succeeds
  in an environment with no `tensorrt`, no `pycuda`, no `Jetson.GPIO`,
  no `pyserial`, no `pypot` (none of which are installed here today).

**4. Smoke test**

```bash
cd $HOME/Projects/Open_Duck_Mini_Jetson
python3 -m jetson_runtime.loop --replay docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz \
  --onnx exported_policies/v5d_contact_wrench_ppo/policy.onnx --steps 1500 --report
```
Observable: prints `max|Δaction| = <value>` against the recorded actions.
**Pass threshold: < 1e-4.** A value near 3× the action magnitude means
`action_scale` was applied twice or not at all; a value near 1.4 rad on the joint
target means `q_default` is missing (the largest `q_default` is `right_knee`
1.379 rad).

**5. Done when**

- [ ] `python3 -c "import jetson_runtime.loop"` succeeds from the repo root on a
      machine with no Jetson packages installed.
- [ ] `pytest tests/test_runtime_core.py -v` → 7 passed.
- [ ] The replay smoke test prints `max|Δaction| < 1e-4` on all three traces.
- [ ] `grep -REc '0\.25|1\.368|1\.379|left_hip_yaw' jetson_runtime/*.py` prints 0
      for every file except `contract.py` (whose loader may name the JSON keys).
- [ ] `docs/jetson-mod/sim2real/deployment_contract.md` gains a "Reference
      implementation" section pointing at these modules, states the ART-1
      resolution taken, states the measured clamp counts, and states the
      gait-phase reset policy for a stumble.
- [ ] `pytest tests/` still passes.

**AI-agent suitable:** YES — this is the largest purely-software task in the
phase and an agent should do all of it.

---

### Task S.5 — Bring up inference on the Jetson and prove parity

**1. Context for the implementing agent**

TensorRT FP16 is a *different numerical implementation* of the same graph. The
locomotion policy is a small MLP (`Sub, Div, Gemm, Elu, Gemm, Elu, Gemm, Elu,
Gemm`), so the difference should be tiny — but "should be" is how a 1.379 rad
offset gets shipped. This task converts the engine and proves it produces the
same actions as the ONNX on the recorded traces before anything is connected to
a servo.

Read first: `docs/jetson-mod/task_plan.md` (locate by heading) (Task 4.4 Steps 1–2 — the
`trtexec` command and the `TRTInfer` sketch; note its buffer sizes already say
59/16); `docs/jetson-mod/known_issues.md` § DEPLOY-4 (`:955`) and § ART-1
(`:865`).

Depends on: S.4. Requires a physical Jetson Orin Nano Super with JetPack.

Traps:
- ART-1 again: `policy.onnx` is not in git. Copy it to the Jetson explicitly,
  **with its `policy.onnx.data` sidecar in the same directory** — the graph uses
  external data and will not load without it.
- DEPLOY-4: batch dimension is a literal 1. `trtexec` will build a static-shape
  engine, which is fine on-robot but means you cannot batch the parity check;
  loop instead.
- TensorRT engines are **not portable** — build on the Jetson itself, and rebuild
  after any JetPack/TensorRT upgrade. Record the TensorRT version in the engine's
  filename.
- The normalizer (`Sub`/`Div`) is **inside** the graph, so parity here also
  covers observation normalization. Do not add a second normalization step.

**2. Low-level implementation plan**

1. Copy to the Jetson: `policy.onnx`, `policy.onnx.data`,
   `jetson_runtime/policy_contract.json`, the three `.npz` traces, and the
   `jetson_runtime/` package.
2. Build both an FP32 and an FP16 engine (confirm the binary exists first —
   `ls /usr/src/tensorrt/bin/trtexec`, and fall back to `which trtexec`):
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=policy.onnx --saveEngine=policy_fp32_trt<VER>.engine
   /usr/src/tensorrt/bin/trtexec --onnx=policy.onnx --saveEngine=policy_fp16_trt<VER>.engine --fp16
   ```
3. Implement the `tensorrt` backend in `jetson_runtime/policy.py` (the class
   sketched in task_plan Task 4.4 Step 2, but as a backend of the existing
   wrapper, not a separate class — one inference interface, two implementations).
4. Write `jetson_runtime/tools/check_parity.py`: runs a trace through the ONNX
   (CPU), the FP32 engine and the FP16 engine, and reports per-backend
   `max|Δaction|` vs the recorded actions, plus FP16-vs-FP32.
5. Write `jetson_runtime/tools/bench_inference.py`: 100 warm-up + 1000 timed
   inferences, reporting p50/p95/p99/max in ms. **Report percentiles, not the
   mean** — the task_plan's existing benchmark asserts on the mean, and a 50 Hz
   loop is killed by the tail, not the average.
6. Record all results in `docs/jetson-mod/sim2real/S5_jetson_inference.md`
   together with the platform identification. `jetson_release -v` requires
   `jetson-stats`, which may not be installed; if it is absent, record
   `cat /etc/nv_tegra_release`, `dpkg -l | grep -i tensorrt`, and
   `python3 -c "import tensorrt; print(tensorrt.__version__)"` instead.

**3. Unit tests**

- `jetson_runtime/tools/check_parity.py` is itself the test; wrap its assertions
  in `tests/test_trt_parity.py` guarded by
  `@pytest.mark.skipif(importlib.util.find_spec("tensorrt") is None, ...)` **and**
  a `@pytest.mark.requires_tensorrt` marker. **Register the marker in
  `pytest.ini`** — it currently registers only `phase1`..`phase5`, and TEST-3
  records that `requires_isaac_sim` is documented in a docstring but applied to
  zero tests and registered nowhere, so that skip is fiction. Do not repeat that.
- Assertions: FP32 engine vs ONNX `max|Δaction| < 1e-5`; FP16 vs FP32
  `max|Δaction| < 2e-3` **and** `max|Δ(0.25·action)| < 1e-3 rad` at the joint.
  If FP16 exceeds this, ship FP32 — the policy takes well under 1 ms either way
  and there is no reason to trade accuracy for headroom you do not need.

**4. Smoke test**

`python3 -m jetson_runtime.tools.bench_inference --engine policy_fp16_trt<VER>.engine`
Observable: p99 latency in ms. **Budget: p99 < 2 ms**, which leaves 18 ms of the
20 ms control period for sensing and actuation.

**5. Done when**

- [ ] Both engines exist on the Jetson with the TensorRT version in the filename.
- [ ] `check_parity.py` reports FP32-vs-ONNX < 1e-5 and FP16-vs-FP32 < 2e-3 on
      all three traces.
- [ ] `bench_inference.py` reports p99 < 2 ms.
- [ ] `requires_tensorrt` is registered in `pytest.ini`, and
      `pytest tests/test_trt_parity.py -v` reports **skipped** (not "no tests
      ran") on a machine without TensorRT.
- [ ] `docs/jetson-mod/sim2real/S5_jetson_inference.md` records the numbers, the
      JetPack/TensorRT versions and how they were obtained, and which precision
      was selected.
- [ ] The selected engine's SHA256 is recorded in that document.

**AI-agent suitable:** PARTIAL — an agent can write every script, and can run
them if it has a shell on the Jetson. **A human must own and power the Jetson**,
and must copy the ONNX across since it is not in git.

---

### Task S.6 — Measure the real sensor → inference → actuation latency

**1. Context for the implementing agent**

PLANT-7: **there is no latency model of any kind in the simulator.** No
`latency`, `delay`, or observation-staleness term exists in `env_cfg.py` or the
shipped `env.yaml` (the single textual "delay" hit is the word *delays* in a
comment at `env_cfg.py:642`). The policy was trained believing the observation
it sees is the state *now* and that its command takes effect *instantly*. The
real loop is a BNO055 I²C read, a Feetech sync-read over a 1 Mbaud serial bus,
a TensorRT inference, a sync-write, and then the servo's own internal control
delay. None of that is zero.

This is an *absence*, so nothing in the config will ever look wrong. It has to be
measured on hardware, and it has to be measured before you can decide whether to
retrain.

Read first: `docs/jetson-mod/known_issues.md` § PLANT-7 (`:469`); `AGENTS.md:365`
§ Domain Randomization, the "Not randomized at all" paragraph — control and
observation latency are named there as known-missing.

Depends on: S.5 (you need working inference), and a partially assembled robot or
at minimum a powered servo bus.

Traps:
- Do **not** measure this with the robot standing. Measure with the robot on a
  bench, servos powered but either torque-disabled or with legs free in the air.
- The Feetech sync-read time scales with the number of servos on the bus and the
  baud rate. Measure with **all 14** servos on the bus at **1,000,000 baud**
  (`experiments/v2/configure_motors.py:29-32`); measuring one and multiplying is
  wrong because of per-packet overhead.
- Actuation latency is not the write call's return time. The write returns when
  the bytes are on the wire; the servo starts moving later.
- `logs/` does not exist yet and is not in `.gitignore`. Create it and add
  `logs/` to `.gitignore` in this task.

**2. Low-level implementation plan**

1. Write `jetson_runtime/tools/measure_latency.py` with two modes.
2. **Mode `--stages`** (software timing): run the real loop 2000 times against
   the real sensor path (not `ReplaySensorSource`), recording
   `time.perf_counter_ns()` around each stage:
   `imu_read`, `servo_sync_read`, `obs_build`, `infer`, `action_decode`,
   `servo_sync_write`, `loop_total`. Emit p50/p95/p99/max per stage to
   `logs/latency/stages_<timestamp>.json`.
3. **Mode `--transport`** (physical timing of the command→motion delay): pick one
   free-hanging leg joint. At a known timestamp, sync-write a step of +0.15 rad.
   Then sync-read present position as fast as the bus allows and find the first
   sample whose deviation from the pre-step mean exceeds 5× the pre-step noise
   standard deviation. The elapsed time is the **transport delay** (bus write +
   servo firmware + mechanical onset). Repeat 100 times, both directions, on
   three different joints (a hip, a knee, an ankle — different loads).
   Note explicitly in the output that the read path's own latency is included and
   is therefore an upper bound, and that the sampling interval bounds the
   resolution.
4. Compute the **total observation-to-actuation delay**:
   `sensor_read + obs_build + infer + decode + write + transport`, expressed both
   in milliseconds and in **control steps** (`/ 20 ms`).
5. Write `docs/jetson-mod/sim2real/S6_latency_budget.md`: the per-stage table, the
   total, the number in control steps, the bus baud rate used, the servo count,
   the Jetson power mode (`nvpmodel -q`), and the raw JSON paths.
6. Add the measured total to `jetson_runtime/policy_contract.json` as
   `measured_latency_ms` **through S.1's generator** (with the source JSON path
   recorded in the generator, not a hand edit) so downstream code and the sim
   change in S.7 read one number. Re-run `--check`.

**3. Unit tests**

- `tests/test_latency_tool.py::test_stage_stats_math`: feed the statistics
  function a synthetic array with known percentiles and assert p50/p95/p99 match
  `np.percentile` with the interpolation method you chose (state the method —
  percentile definitions differ and this is a real source of silent
  disagreement).
- `::test_onset_detector`: feed the transport-onset detector a synthetic step
  response with known onset index and added noise, assert it recovers the index
  within ±1 sample at SNR 5.
- None applicable for the measurement itself — it requires hardware by
  definition.

**4. Smoke test**

`python3 -m jetson_runtime.tools.measure_latency --stages --iters 2000`
Observable: a printed table ending in `loop_total p99 = <X> ms`. The loop is
viable only if `loop_total p99 < 20 ms`; if it is not, the control rate itself is
broken and S.11 cannot start.

**5. Done when**

- [ ] `logs/` exists and is in `.gitignore`.
- [ ] `logs/latency/stages_<ts>.json` and `logs/latency/transport_<ts>.json` exist.
- [ ] `docs/jetson-mod/sim2real/S6_latency_budget.md` states a single total
      observation-to-actuation delay in ms **and in control steps**, with the
      servo count, baud rate and Jetson power mode it was measured at.
- [ ] `loop_total p99 < 20 ms` is demonstrated, or the document explains why not
      and what was changed (baud rate, sync-read grouping, power mode).
- [ ] `policy_contract.json` carries `measured_latency_ms` and `--check` exits 0.
- [ ] `pytest tests/test_latency_tool.py -v` → 2 passed.

**AI-agent suitable:** PARTIAL — the agent writes both tools and does all the
analysis. **A human must power the bus and hang the robot on the bench**, and
must be present because a servo stepping 0.15 rad on an assembled robot can move
a limb into something.

---

### Task S.7 — Decide and implement the latency response in simulation

**1. Context for the implementing agent**

S.6 produces a number. This task turns it into a decision and, if the decision is
"model it", into a sim change plus a retrain. Do not do this before S.6 — the
whole failure mode this task exists to avoid is picking a latency value from a
paper instead of from the robot.

Read first: `docs/jetson-mod/sim2real/S6_latency_budget.md` (the measured total);
`docs/jetson-mod/sim2real/S3_deploy3_decision.md` and `S8_torque_envelope.md`
(the other two changes this campaign must carry);
`isaac_lab_env/open_duck_mini_v2/env_cfg.py` (`OpenDuckContactWrenchEnvCfg` at
`:742` and its PLAY twin at `:789` — the pattern for adding a track);
`isaac_lab_env/open_duck_mini_v2/robot_cfg.py` (`ImplicitActuatorCfg`);
`~/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:310`
(`DelayedPDActuator`; its cfg class lives in `actuator_pd_cfg.py`, not in
`actuator_pd.py`).

Depends on: **S.6, S.3 and S.8** (the antenna decision and the actuator-fidelity
decision are inputs to this campaign), plus S.0's verdict on whether a retrain
was already mandatory.

**Decision rule (write it into the document before you look at the number, so it
is not fitted to the answer):**
- measured total **< 10 ms (0.5 control step)** → record and accept; re-examine
  after S.12 if the bring-up logs show a correlated failure.
- measured total **≥ 10 ms** → model it in sim and retrain.

Traps:
- `DelayedPDActuatorCfg` looks like the obvious answer and is a trap in this
  repo: `DelayedPDActuator` subclasses `IdealPDActuator`
  (`actuator_pd.py:310`), i.e. it moves the PD loop **out of PhysX's implicit
  solver and into Python**. Every policy here was trained with
  `ImplicitActuatorCfg` (`robot_cfg.py:94,105`). Switching changes the plant well
  beyond latency and invalidates comparability with every prior run.
- Its delay unit is **physics steps, not control steps**: actuators are computed
  inside the `for _ in range(self.cfg.decimation)` loop
  (`~/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:182-195`,
  `apply_action()` at `:185`), so one unit = `sim.dt` = **5 ms**, and a 20 ms
  control-step delay is `max_delay = 4`. Getting this wrong by 4× is easy.
- `--video` is mandatory on every training run.
- **SHELL-2**: `scripts/v5_chain.sh` deletes the pidfile that the launcher's
  duplicate-run guard depends on. Do not use it here.

**2. Low-level implementation plan**

If the decision is **accept**: write
`docs/jetson-mod/sim2real/S7_latency_decision.md` with the measured number, the
rule, the verdict, and an explicit re-check item for S.12. Then check whether S.3
or S.8 independently require a campaign; if they do, run it without the latency
terms. If nothing requires one, stop and do not touch the env config.

If the decision is **model**:

1. Create `isaac_lab_env/open_duck_mini_v2/mdp/delayed_actions.py` with
   `DelayedJointPositionAction(JointPositionAction)` and
   `DelayedJointPositionActionCfg`: buffer the processed action in a ring of
   `max_delay_steps` **control** steps; on reset, draw an integer delay uniformly
   in `[min_delay_steps, max_delay_steps]` per env (mirroring
   `DelayedPDActuator.reset`'s randomization, which is the right pattern even
   though the class itself is wrong here). This keeps `ImplicitActuatorCfg`
   and therefore keeps the plant comparable.
2. Create `isaac_lab_env/open_duck_mini_v2/mdp/stale_observations.py` with a
   term wrapper that returns an observation from `n` control steps ago, for the
   `base_ang_vel`, `projected_gravity`, `joint_pos` and `joint_vel` terms only
   (the command, `actions` and `gait_phase` are generated on the Jetson and are
   not stale).
3. Add `OpenDuckLatencyEnvCfg` in `env_cfg.py`, **inheriting from
   `OpenDuckContactWrenchEnvCfg`** unless S.0 recorded a different current best
   track (name whichever you use in the docstring), with:
   - action delay `min_delay_steps=0`, `max_delay_steps=ceil(measured_ms/20)+1`
   - observation staleness `n` drawn from the same range
   Register `Isaac-Velocity-Rough-OpenDuck-Latency-v0` and its `-Play-v0` twin by
   adding two rows to the `(task_id, cfg_class)` list at
   `isaac_lab_env/open_duck_mini_v2/__init__.py:105-115`.
4. Fold in, in the *same* campaign, everything already decided: S.3's antenna
   option (E changes act to 14 and obs to 53 and therefore invalidates the S.1
   contract — regenerate it), S.8's actuator-fidelity changes, and, if the owner
   has settled the print process, PLANT-10's `PLA_EFFECTIVE_DENSITY` correction
   with the re-derived trunk inertial (PLANT-10's own fix note says it should be
   done in the retrain change, not separately). **One campaign, not four** — each
   costs GPU days.
5. Smoke-train, then train, both through the detached launcher. Its signature is
   `launch_training_detached.sh <run_name> [args passed to train_ppo.py]`, and it
   `cd`s to `~/IsaacLab` itself — do not prefix it with `isaaclab.sh`:
   ```bash
   cd $HOME/Projects/Open_Duck_Mini_Jetson
   ./scripts/launch_training_detached.sh latency_smoke \
     --task Isaac-Velocity-Rough-OpenDuck-Latency-v0 \
     --headless --max_iterations 50 \
     --video --video_length 200 --video_interval 5000
   # then, after it exits (journal rule 6: check the pidfile pgroup is dead
   # AND the final model_<iter>.pt exists):
   ./scripts/launch_training_detached.sh latency_full \
     --task Isaac-Velocity-Rough-OpenDuck-Latency-v0 \
     --headless --max_iterations <N> \
     --video --video_length 200 --video_interval 5000
   ```
   Monitor with `tail -f .training_runs/<run_name>.log`.
6. Gate the result with the full evaluation protocol into
   `docs/jetson-mod/eval_results_rebuild/` (Phase R's post-Phase-M results dir —
   do **not** invent a new one) using the **same six conditions** Task R1 used,
   and journal it with last-100 TensorBoard means
   (`EventAccumulator`, `size_guidance={'scalars': 0}`).

**3. Unit tests**

New `tests/test_delayed_action.py` — these must import and run the class, not
grep `env_cfg.py`:
- `test_zero_delay_is_identity`: with `min=max=0`, the wrapper's output equals
  its input for 100 random steps.
- `test_delay_of_n_shifts_by_n`: with `min=max=3`, output at step `t` equals
  input at step `t−3` for `t ≥ 3`, and equals the reset value before that.
- `test_delay_is_per_env`: with `min=0, max=4` and 8 envs, the drawn delays are
  not all equal across 20 resets (guards against a scalar draw broadcast to all
  envs — a real and silent bug).
- `test_stale_obs_shape_preserved`: the staleness wrapper returns the same shape
  and dtype as the wrapped term.

**4. Smoke test**

The 50-iteration run above completes, and the run's TensorBoard event file
contains training scalars. Observable: the launcher's log ends without a
traceback, `model_49.pt` (or the launcher's final-iteration checkpoint) exists,
and:
```bash
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator('<run_dir>', size_guidance={'scalars':0}); ea.Reload()
print(ea.Tags()['scalars'][:10])"
```
prints a non-empty tag list. **List the tags rather than assuming a name** — do
not hardcode `Train/mean_reward` before you have seen it in this rsl-rl version.
This proves registration and the new action term wire up before you spend the
full campaign.

**5. Done when**

- [ ] `docs/jetson-mod/sim2real/S7_latency_decision.md` states the rule, the
      measured number, and the verdict — in that order.
- [ ] If accepted and no other change forces a campaign: no env-config change,
      and an S.12 re-check item exists.
- [ ] If modelled: `pytest tests/test_delayed_action.py -v` → 4 passed; the new
      task is registered; the 50-iteration smoke run exits cleanly with a
      non-empty scalar tag list; the full campaign is journaled with last-100
      TensorBoard means; the new policy is gated on the same six conditions as
      S.0 and exported to `exported_policies/<name>/` with `agent.yaml`,
      `env.yaml`, `model_<iter>.pt`, a README carrying the MD5 — and the ONNX
      plus its `.data` sidecar are both present (ART-1: `*.onnx` is gitignored,
      so say explicitly whether the graph is committed or reproducible).
- [ ] If the obs/action dims changed (S.3 option E), `scripts/generate_policy_contract.py`
      was re-pointed at the new export and `--check` exits 0.
- [ ] `docs/jetson-mod/known_issues.md` § PLANT-7 is updated: either closed with
      the fix, or annotated with the measured latency and the accept decision.

**AI-agent suitable:** YES for the implementation, training and gating. The
owner should approve before the campaign launches, because it costs GPU days.

---

### Task S.9 — Calibrate the physical robot against the model

**1. Context for the implementing agent**

> **AI-agent suitable: NO.** Every step marked [HUMAN] below requires hands on
> the robot, eyes on the robot, and the ability to stop it. An agent's role is
> limited to writing the scripts, printing the checklist, and analysing the
> numbers that come back. Do not let an agent "complete" this task.

The policy commands joint angles in the MJCF's coordinate convention. The servos
know nothing about that convention. Five independent mappings must be established
and written down, or the robot will move confidently in the wrong directions:

1. **Servo ID ↔ joint name** (14 servos; the antennas are PWM and have no ID).
2. **Zero offset**: the servo firmware angle that corresponds to MJCF `q = 0`.
3. **Direction sign**: whether increasing firmware ticks is `+q` or `−q`.
4. **Ticks per radian.** **This is not in the repo.** Nothing here states the
   STS3250's encoder resolution or its firmware angle units. Determine it
   empirically (command two known angles far apart with torque on and read back
   the tick delta, or read the servo's angle-limit registers) and record it
   per joint. Do not assume a value.
5. **IMU mounting rotation**: the constant rotation from the BNO055's frame to
   the `trunk_assembly` body frame the policy's observation is expressed in.

**The sign convention is not "mirror the right side".** Read the model before
guessing — from `robot_motors.xml` `jnt_range` and `robot_cfg.py:48-69`:
- Hips **are** mirrored: `left_hip_pitch` range `[−1.22173, +0.523599]`, standing
  −0.630; `right_hip_pitch` range `[−0.523599, +1.22173]`, standing **+0.635**.
  `left_hip_roll` range `[−0.436332, +0.436332]` standing +0.053;
  `right_hip_roll` same range, standing **−0.065**.
- Knees and ankles are **not** mirrored: both knees have range
  `[−1.5708, +1.5708]` and stand at **+1.368** (left) / **+1.379** (right); both
  ankles have the same range and stand at **−0.784** / **−0.796**.

A blanket "negate everything on the right" therefore breaks the knees and ankles
while looking plausible on the hips — and the resulting gait would look like a
sim-to-real failure rather than a wiring error.

Read first: `docs/configure_motors.md` (the ID table at `:22-38` and the
horn-alignment procedure); `experiments/v2/configure_motors.py:10-32` (the same
table in code, plus `FeetechSTS3215IO(port, baudrate=1000000, use_sync_read=True)`
— the STS3215 IO class drives the STS3250, same protocol);
`scripts/duck_init_pos.json`; `AGENTS.md:455` § Joint Orders, including the
warning that an earlier revision of that very table was the legacy 15-joint BDX
order.

Depends on: S.1 (contract), S.4 (decoder). Blocks S.11.

Traps:
- `mini_bdx.hwi` and `mini_bdx_runtime.hwi_feetech_pypot`, imported by
  `experiments/real_robot/rl_walk.py:6`, `experiments/real_robot/move_test.py:1`
  and `experiments/v2/placo_walk_real_robot.py:1`, **do not exist in this repo**
  (`mini_bdx/mini_bdx/` contains only `old_walk_engine`, `placo_walk_engine`,
  `utils`; `import mini_bdx_runtime` fails). Those scripts are broken-import
  references only; the hardware interface lives in the external
  `apirrone/Open_Duck_Mini_Runtime` (branch `v2`). Either vendor it or write a
  minimal bus driver — do not assume the import works.
- **`pypot` is not installed** on this machine (`import pypot` →
  `ModuleNotFoundError`). `setup.cfg` lists it only under the `robot` extra, as a
  git URL to a fork (`apirrone/pypot@support_4B_registers`). Installing it is a
  prerequisite step, not an assumption.
- Servo configuration must be done **per motor, before assembly**, with only one
  motor on the bus (`configure_motors.py` exits if it finds more than one).
- `head_roll` (ID 33) is the 16th DOF added by this fork; some upstream tables
  predate it.

**2. Low-level implementation plan**

1. **[HUMAN]** Configure each of the 14 servos individually before assembly,
   following `docs/configure_motors.md`: set the ID from the table, drive to zero,
   fit the horn as closely aligned as possible.
2. **[AGENT]** Write `jetson_runtime/servo_bus.py`: a thin wrapper over the
   Feetech bus exposing `sync_read_position()`, `sync_read_velocity()`,
   `sync_read_current()`, `sync_read_temperature()`, `sync_write_position()`,
   `torque_enable(bool)`, all keyed by **joint name** via the contract, never by
   raw index. Add a `FakeBus` in the same module for the tests.
3. **[AGENT]** Write `jetson_runtime/tools/calibrate.py` with four subcommands:
   - `scan` — enumerate bus IDs, compare against the contract's table, print
     missing/extra/duplicate IDs.
   - `zero` — for one named joint at a time: torque off, prompt the operator to
     hold the joint at its mechanical reference, capture 200 samples, and record
     the mean firmware tick as `zero_tick`.
   - `scale` — for one named joint at a time: torque on, command two firmware
     positions a known distance apart, measure the resulting mechanical angle
     with the operator's protractor or from the servo's own reported units, and
     record `ticks_per_rad`.
   - `sign` — for one named joint at a time: torque on, command **+0.10 rad** in
     model convention, hold 1 s, return. Prompt the operator: "did the joint move
     in the same direction as the viewer?" while the same `q` is displayed in
     `python3 -m mujoco.viewer --mjcf=mini_bdx/robots/open_duck_mini_v2/scene.xml`
     (`scene.xml` includes `robot_motors.xml`, the position-controlled model whose
     joint ranges the contract was derived from). Record `+1` or `−1`.
4. **[HUMAN]** Run `zero`, `scale` and `sign` for all 14 joints, one joint at a
   time, with the robot suspended and the legs free. **0.10 rad is deliberately
   small** — big enough to see, small enough not to slam a limb into the chassis.
5. **[AGENT]** Write the results to `jetson_runtime/calibration.json`:
   `{joint: {servo_id, zero_tick, sign, ticks_per_rad, hard_low_rad,
   hard_high_rad, soft_low_rad, soft_high_rad}}`, where the limits come from the
   contract, not from the operator. Include the two antennas with
   `servo_id: null`.
6. **[HUMAN + AGENT]** IMU frame: mount the BNO055, place the robot in three
   known static orientations (level, +30° pitch on a wedge, +30° roll), capture
   200 samples of the gravity vector in each. **[AGENT]** solve the least-squares
   rotation `R_imu→body` (Kabsch / SVD) that maps the three measured gravity
   vectors onto the three expected body-frame vectors; store as a quaternion in
   `calibration.json` with the **w, x, y, z** convention stated explicitly, to
   match `jetson_runtime/imu.py`.
7. **[AGENT]** Write `docs/jetson-mod/sim2real/S9_calibration_record.md`: the full
   table, the date, the operator's name, the ambient temperature, and the raw
   sample files. Calibration drifts; an undated calibration is worthless.

**3. Unit tests**

Executable, hardware-free, against the `FakeBus`:
- `tests/test_calibration.py::test_roundtrip_rad_to_tick`: for every joint,
  `tick_to_rad(rad_to_tick(q)) ≈ q` within 1e-4 across the joint's full range,
  for both signs and a nonzero `zero_tick`.
- `::test_sign_flip_is_detected`: with `sign = −1`, `rad_to_tick(+0.1)` and
  `rad_to_tick(−0.1)` fall on opposite sides of `zero_tick`.
- `::test_no_command_exceeds_soft_limits`: sweep 10,000 random actions through
  `ActionDecoder.decode` + `rad_to_tick` and assert every tick lies inside the
  per-servo tick range implied by the soft limits. (Use `decode`, which clamps —
  not `raw_target`, which does not.)
- `::test_imu_solve_recovers_known_rotation`: generate three gravity vectors
  through a known rotation, add 1° of noise, and assert the solver recovers it
  within 2°.
- `::test_calibration_covers_all_joints`: `calibration.json` has exactly the 16
  contract joints; the 14 bus joints have integer `servo_id` matching the
  contract, and the two antennas have `servo_id: null`.

**4. Smoke test**

With the robot suspended, torque enabled, and a human hand on the power switch:
```bash
python3 -m jetson_runtime.tools.calibrate verify --hold q_default --seconds 5
```
The robot should assume the standing pose in the air. Observable: the printed
`max |q_measured − q_default|` across the 14 bus joints. **Pass: < 0.05 rad.** A
joint that lands at roughly `−q_default` instead has an inverted sign; a joint
that lands at a constant offset has a bad zero; a joint whose error scales with
`|q_default|` has a bad `ticks_per_rad`.

**5. Done when**

- [ ] `jetson_runtime/calibration.json` exists with all 16 joints, dated, and
      names the operator.
- [ ] `calibrate scan` reports all 14 expected IDs, no extras, no duplicates.
- [ ] `ticks_per_rad` is a **measured** value per joint, with the measurement
      method recorded — not a constant copied from anywhere.
- [ ] The suspended `verify --hold q_default` smoke test reports < 0.05 rad on
      every bus joint.
- [ ] `pytest tests/test_calibration.py -v` → 5 passed.
- [ ] `docs/jetson-mod/sim2real/S9_calibration_record.md` records the date,
      operator, ambient temperature, and raw sample file paths.
- [ ] The IMU rotation is verified by tilting the robot to a known +30° pitch and
      confirming `projected_gravity` matches the expected unit vector within 0.05
      — the same magnitude as the ±0.05 gravity noise the policy trained under
      (`env.yaml:454-460`).

**AI-agent suitable:** **NO.** Physical: fitting horns, holding joints at their
reference, judging the direction of motion, placing the robot on wedges, and
being ready to cut power. The agent writes `servo_bus.py`, `calibrate.py`, the
solver, the tests, and the record document — and analyses every number that comes
back — but it cannot perform a single step of the calibration itself.

---

### Task S.10 — Build the safety layer

**1. Context for the implementing agent**

The robot is ~2.66 kg with 14 servos that can each pull several amps and reach
70 °C. It will fall. The question is what happens when it does.

`AGENTS.md:684` § Safety Rules states three requirements — clamp all velocity
commands to the trained command hull; on a Cosmos inference failure fall back to
the last known good command (**not** zero, which could cause a mid-stride fall);
and cut all motors on an IMU tilt above 60 degrees. That last threshold is not
arbitrary: it is exactly the simulator's `bad_orientation` termination
(`env_cfg.py:592` and `:701`, `limit_angle = math.radians(60.0)`), so beyond it
the policy is outside every state it was ever trained in and its output is
meaningless.

The other sim termination — `root_height_below_minimum` with `minimum_height =
0.09` m (`env_cfg.py:595`, `:704`) — **cannot be evaluated on hardware**, because
nothing on the robot measures trunk height. The safety layer must substitute
something it can measure.

Read first: `AGENTS.md:684` § Safety Rules and `AGENTS.md:371` § Termination
Conditions; `docs/jetson-mod/sim2real/S8_torque_envelope.md` (thresholds, S.8);
`jetson_runtime/calibration.json` (S.9).

Depends on: S.8, S.9. Blocks S.11.

Traps:
- **`AGENTS.md:686` is corrupted.** The first Safety-Rules bullet is a leftover
  *edit instruction*, not a rule. The rule embedded in it is: clamp to
  `forward [-0.148, 0.222], lateral [-0.111, 0.111], turn [-0.3, 0.3]`. Note that
  the **turn clamp ±0.3 is deliberately narrower than the trained hull of ±0.5**
  (`env_cfg.py:263`, `env.yaml:995-997`) — DEPLOY-5 confirms that ±0.3 is
  correctly *inside* the hull. Implement the clamp from the contract's
  `CMD_CLAMP` field, not from a literal, and record in `S10_safety.md` that
  vx/vy clamp at the hull while wz clamps conservatively inside it. If the owner
  wants the full ±0.5, that is a decision to record, not a default.
- **On a fall, torque OFF — do not hold the pose.** A fallen robot holding a
  standing pose stalls its servos against the floor, which trips the stall and
  overcurrent protections and heats the coils. The firmware will then drop torque
  anyway, silently and unrepeatably.
- Distinguish two different fallbacks that are easy to conflate: a *command*
  fallback (Cosmos fails → hold the last good velocity command) and an *action*
  fallback (the policy or a sensor fails → hold the last joint target for a
  couple of steps, then ramp torque off). The `AGENTS.md` rule is about the first.
- Stop at 65 °C, below the believed firmware cutout of 70 °C. Letting the
  firmware win means the leg goes limp with no warning and no log line. **If S.8
  could not source the datasheet, the 70 °C figure is UNVERIFIED — say so in
  `S10_safety.md` rather than presenting it as a spec.**

**2. Low-level implementation plan**

1. `jetson_runtime/safety.py` — class `SafetyMonitor(contract, calibration)`,
   called once per control step with the current sensor bundle, returning one of
   `OK`, `DERATE`, `HOLD`, `TORQUE_OFF`, plus a reason string. Every threshold is
   read from the contract; none is a literal in this file. Rules:
   - **Tilt**: angle between measured `projected_gravity` and `(0,0,−1)` > 60° →
     `TORQUE_OFF`. Reason `"tilt"`.
   - **Fall proxy** (the height-termination substitute): both foot switches open
     **and** `|gravity_z| < 0.8` (i.e. tilted past ≈37°) for 5 consecutive steps →
     `TORQUE_OFF`. Reason `"fall_proxy"`. Both foot switches are in the GPIO map
     (`AGENTS.md:635`: left foot switch = header pin 15, right = pin 13).
   - **Current**: any servo above `servo_current_limit_a` for more than 1.0 s →
     `TORQUE_OFF`; above 70 % of it for more than 1.0 s → `DERATE`
     (scale the velocity command by 0.5). The 1.0 s window is deliberately half
     `servo_current_limit_window_s` so the runtime acts before the firmware.
   - **Temperature**: any servo ≥ `servo_temp_warn_c` → `DERATE` and log;
     ≥ `servo_temp_stop_c` → `TORQUE_OFF`. Reason includes the joint name.
   - **Loop overrun**: control step took > 2× `CONTROL_DT` → `HOLD`; three
     consecutive overruns → `TORQUE_OFF`.
   - **Sensor fault**: IMU read failure or NaN anywhere in the observation →
     `HOLD` for up to 3 steps, then `TORQUE_OFF`.
   - **Command clamp**: every incoming velocity command is clipped to
     `contract.CMD_CLAMP` before it enters the observation, and the clip is
     counted.
2. `TORQUE_OFF` implementation: ramp the goal position to the *measured* position
   over 2 control steps (so the servo does not fight), then call
   `torque_enable(False)` on all 14, then latch. Recovery requires an explicit
   operator `rearm()` — never automatic.
3. `jetson_runtime/tools/fault_injection.py` — replays a trace through the full
   loop while injecting each fault condition in turn, asserting the monitor
   returns the expected verdict within the expected number of steps. This is how
   the safety layer gets tested without endangering the robot. It must also run
   the **clean** trace and count false trips.
4. **[HUMAN]** A physical kill switch in series with the **servo power rail**,
   reachable without leaning over the robot, that does *not* cut the Jetson (so
   the logs survive the incident). Document its location and wiring in
   `docs/jetson-mod/sim2real/S10_safety.md` and add it to the wiring diagram.
5. Write `docs/jetson-mod/sim2real/S10_safety.md`: every rule, its threshold, the
   **source** of the threshold (contract field → S.8 document → datasheet page,
   or UNVERIFIED), and the observable that proves it fires.

**3. Unit tests**

New `tests/test_safety_monitor.py`, all pure-python against synthetic bundles and
a synthetic contract (so the thresholds under test are explicit, not whatever
happens to be in the JSON that day):
- `::test_tilt_trips_at_60_degrees`: 59° → `OK`, 61° → `TORQUE_OFF`.
- `::test_current_needs_duration`: over-limit for 0.5 s → `OK`; for 1.1 s →
  `TORQUE_OFF`. Guards against a threshold-only rule that trips on a single spike.
- `::test_temp_ladder`: below warn → `OK`, between warn and stop → `DERATE`,
  above stop → `TORQUE_OFF`.
- `::test_nan_obs_holds_then_stops`: NaN for 3 steps → `HOLD`, 4th → `TORQUE_OFF`.
- `::test_command_clamped_to_hull`: `(0.5, 0.5, 2.0)` clamps to the contract's
  `CMD_CLAMP` (with the shipped values, `(0.222, 0.111, 0.3)`) and increments the
  clip counter. Read the expected values from the contract, do not hardcode them.
- `::test_torque_off_is_latched`: after `TORQUE_OFF`, a subsequent healthy bundle
  still returns `TORQUE_OFF` until `rearm()` is called.
- `::test_fall_proxy_needs_both_conditions`: feet open but upright → `OK`;
  tilted but feet loaded → `OK`; both → `TORQUE_OFF` after 5 steps.

**4. Smoke test**

```bash
python3 -m jetson_runtime.tools.fault_injection \
  --trace docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz --all
```
Observable: a table of 7 injected faults with the step at which each tripped.
**Pass: 7/7 trip, and zero false trips on the clean trace.** The second half
matters as much as the first — a safety layer that trips during normal walking
will be disabled by the operator within an hour.

**5. Done when**

- [ ] `pytest tests/test_safety_monitor.py -v` → 7 passed.
- [ ] Fault injection: 7/7 injected faults trip, 0 false trips on all three clean
      traces.
- [ ] `grep -Ec '3\.8|65|60|0\.222' jetson_runtime/safety.py` prints 0 — every
      threshold comes from the contract.
- [ ] A physical kill switch exists on the servo rail, is documented in
      `S10_safety.md`, and is reflected in
      `docs/jetson-mod/jetson_wiring_diagram.drawio` (re-export
      `jetson_wiring_diagram.png` — `AGENTS.md:824` § Tooling has the puppeteer
      recipe).
- [ ] `docs/jetson-mod/sim2real/S10_safety.md` names the source of every
      threshold, and marks the ones that are UNVERIFIED.
- [ ] `TORQUE_OFF` is verified on the real robot, suspended, by tilting it past
      60° by hand — servos go limp, the reason string is logged, and re-arm is
      required.

**AI-agent suitable:** PARTIAL — the agent writes `safety.py`, the fault
injector, all tests and the document. **A human must wire the kill switch and
verify the trips on the real robot**, including the tilt test, which requires
holding a robot and watching it go limp.

---

### Task S.11 — Staged bring-up ladder

**1. Context for the implementing agent**

> **AI-agent suitable: NO.** Every rung requires a human holding, catching, or
> watching the robot. An agent may sit in the loop reading telemetry and may do
> all the analysis afterwards, but it must not execute a rung.

The purpose of a ladder is that each rung has a **stated abort criterion decided
in advance**. The existing plan (`task_plan.md (locate by heading)`, Task 4.5) says "place robot
on flat surface with safety support... robot should balance and stand", which has
no abort criterion at all and therefore permits an operator to keep trying while
the servos cook.

Read first: `docs/jetson-mod/task_plan.md` (locate by heading) Task 4.5; all of
`docs/jetson-mod/sim2real/` produced by S.6, S.8, S.9, S.10;
`docs/jetson-mod/sim2real/reference_trace_stats.json` (S.2 — rung 2's abort
criterion is defined against its action-rate p95); `AGENTS.md:371` § Termination
Conditions.

Depends on: S.5, S.9, S.10 all complete. S.0's (and S.7's) verdict determines
*which* policy is loaded — record its name and checkpoint MD5 in every log.

Traps:
- Every rung's log is the input to S.12. A rung run without logging is a rung
  that has to be repeated.
- The gait-phase counter is free-running. Restarting the policy mid-session
  restarts the phase; decide and record whether each rung starts from phase 0.
- Servo temperature rises across rungs. Record the starting temperature of every
  rung and give the servos time to cool — a rung 5 that follows rung 4 with hot
  servos is not the same experiment as one that follows a cold start.
- `logs/` must be in `.gitignore` (created in S.6) before rung logs start
  landing there.

**2. Low-level implementation plan**

**[AGENT]** builds `jetson_runtime/bringup.py`: one entry point,
`--rung {bench,suspended,assisted,standing,walking}`, that loads the policy,
applies the rung's own limits, logs every channel below at 50 Hz to
`logs/bringup/<rung>_<timestamp>.npz`, and enforces the rung's abort criterion by
calling the S.10 monitor with rung-specific thresholds. It prints a pre-flight
checklist and refuses to start until the operator confirms each line.

Logged channels (this exact list is the contract for S.12; a log missing one is
unusable): `t_ns`, `obs`, `action`, `q_target`, `q_measured`, `qd_measured`,
`current`, `temperature`, `foot_switch_l`, `foot_switch_r`, `safety_verdict`,
`safety_reason`, `stage_timings`, `command`, `n_clamped`.

**[HUMAN]** runs the rungs, in order, never skipping:

| Rung | Setup | Run | Abort criterion |
|---|---|---|---|
| **1. Bench** | Robot on a stand, legs free, no ground contact. Servos powered, **torque disabled**. | Closed loop for 5 min with actions computed but not written. | `loop_total` p95 exceeding the S.6 budget by > 4 ms; any sensor read failure; any servo > 45 °C at idle. |
| **2. Suspended** | Robot hanging in a harness, feet clear of the ground. Torque enabled. | Policy at command `(0,0,0)` for 60 s, then `vx=0.1` for 60 s. | Any joint reaching a soft limit; any current above 70 % of `servo_current_limit_a`; per-joint action rate exceeding the matching sim p95 from `reference_trace_stats.json` by 2×; any servo > 55 °C. |
| **3. Assisted stance** | Feet on the ground, harness taking roughly half the weight, operator's hands on the trunk. | Command `(0,0,0)` for 60 s. | Posture error `max\|q_meas − q_default\|` > 0.15 rad; any leg-joint current implying > 4.9 N·m; tilt > 20°. |
| **4. Free standing** | Harness slack, operator's hands within 10 cm but not touching. | Command `(0,0,0)` for 60 s. | Any fall; tilt > 30°; temperature rise > 15 °C in the 60 s; any `DERATE` verdict. |
| **5. Walking** | Foam mat, harness slack, operator walking alongside. | `vx=0.10` for 10 s → `vx=0.15` → `vx=0.222` → `vy=0.111` → `wz=0.3`, each 10 s, stopping between. | **First fall stops the session.** Also: any `DERATE`; any servo > 60 °C; visible one-foot dragging. |

(The rung-5 endpoints are the trained hull maxima for vx and vy, and the
deployment turn clamp for wz — see the contract's `CMD_HULL` and `CMD_CLAMP`.)

Additional rules:
- Rung `n+1` may not start until rung `n`'s log has been analysed and its
  criteria confirmed met.
- Between rungs, servos must return below 45 °C.
- Every rung is filmed. Video is a protocol requirement everywhere else in this
  repo for a reason: run 12 passed every aggregate metric while crawling.
- Never leave the robot commanded and unattended.

**3. Unit tests**

- `tests/test_bringup_runner.py::test_rung_limits_are_distinct`: each rung's
  threshold dict differs from the others and every threshold is inside the
  contract's absolute limits.
- `::test_logger_roundtrip`: run the runner against a `ReplaySensorSource` for
  100 steps, load the produced `.npz`, and assert **every channel in the list
  above** is present with the right shape and dtype. A bring-up log with a
  missing channel is discovered here, not after the robot falls.
- `::test_preflight_refuses_without_calibration`: with
  `jetson_runtime/calibration.json` absent, the runner exits non-zero with a
  clear message. This is the single most likely operator error.
- None applicable for the rungs themselves — they are physical procedures.

**4. Smoke test**

```bash
python3 -m jetson_runtime.bringup --rung bench \
  --replay docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz --steps 500 --dry-run
```
Observable: exit 0, and `logs/bringup/bench_<ts>.npz` contains every channel in
the list above for 500 steps. This validates the harness before any servo is
energised.

**5. Done when**

- [ ] All five rungs have a log under `logs/bringup/` and a video.
- [ ] Every log's metadata names the policy, its checkpoint path and MD5.
- [ ] Rung 4 completed 60 s of free standing without a fall or a `DERATE`.
- [ ] Rung 5 reached at least `vx = 0.10` for 10 s without a fall (this is the
      Task 4.5 **PASS** criterion, restated with an abort rule).
- [ ] `docs/jetson-mod/sim2real/S11_bringup_log.md` records, per rung: date,
      operator, start/end servo temperatures, outcome, abort reason if any, and
      the log and video paths.
- [ ] Any rung that aborted has its reason traced to a specific defect (existing
      or newly filed in `known_issues.md`) before it is re-attempted.
- [ ] `pytest tests/test_bringup_runner.py -v` → 3 passed.

**AI-agent suitable:** **NO.** Physical throughout: harnessing, catching,
watching for one-foot dragging, deciding it is safe to continue. The agent writes
`bringup.py`, the logger, the pre-flight checklist and the tests; it can watch
telemetry live and call out an abort; it does all of S.12's analysis. It must
never be the thing that decides the robot is safe to put on the floor.

---

### Task S.12 — Analyse the bring-up logs and decide the next retrain

**1. Context for the implementing agent**

Whatever happened in S.11 is now data. This task converts it into a
sim-versus-real divergence report and a single decision: ship, or retrain with
specific measured changes. Without this task the project has a robot that walks
or does not walk, and no explanation either way.

Read first: every log under `logs/bringup/`; the S.2 reference traces (the
matching sim rollouts); `docs/jetson-mod/sim2real/S6_latency_budget.md` and
`S8_torque_envelope.md`; `docs/jetson-mod/known_issues.md` in full.

Depends on: S.11.

Traps:
- Compare like with like: the sim traces were recorded at fixed commands with
  corruption off and the heading servo disabled. Slice the bring-up logs to the
  matching command segments before comparing anything.
- The bring-up logs were produced by a policy trained on the **old 3.657 kg
  plant** unless S.0/S.7 produced a replacement. Say which policy produced each
  log, by name and checkpoint MD5.
- Do not journal a number you have not measured — journal-protocol rule 5 exists
  because an "~8 rad/s" claim turned out to be 25.3 rad/s when someone finally
  measured it.
- Sim "tracking error" is not comparable to real tracking error without a
  caveat: in sim the PD loop is inside PhysX and `joint_pos_target` is exactly
  what was commanded, so `q − q_target` measures PhysX's solver, not a servo.

**2. Low-level implementation plan**

1. Write `scripts/compare_sim_real.py` (pure numpy; no Isaac, no hardware):
   loads one bring-up `.npz` and one reference trace, aligns by command segment,
   and reports per-joint and aggregate:
   - **tracking error** `q_measured − q_target`: p50/p95/max per joint, sim vs
     real. In sim this is near-zero; on hardware it is the servo's real
     compliance and the single best summary of the actuator-model gap.
   - **action rate** `|a_t − a_{t−1}|`: p50/p95, sim vs real. A real value far
     above sim means the policy is chattering against unmodelled dynamics.
   - **orientation**: `projected_gravity` distribution, sim vs real.
   - **stance duty** per foot from the foot switches, against the sim's
     `GAIT_DUTY_BAND_PCT` band of `[40, 90]` %
     (`scripts/evaluate_policies.py:167`).
   - **thermal**: temperature rise per joint per minute of walking.
   - **latency realised**: per-stage timings from the log vs the S.6 budget.
2. Write `docs/jetson-mod/sim2real/S12_divergence_report.md` with those tables
   and, for each metric that diverges, a named candidate cause drawn from the
   register (PLANT-4/DEPLOY-3 antennas, PLANT-5 torque ceiling, PLANT-6 missing
   dynamic friction, PLANT-7 latency, PLANT-8 heading-servo command structure,
   PLANT-10 trunk mass 54–82 g *heavier* than the 1.089544 kg the MJCF declares)
   — or a new issue filed in `known_issues.md` with evidence, following that
   document's stated convention that every entry is mechanically verified before
   being written down, and adding a matching check to
   `scripts/verify_known_issues.py`.
3. Produce **one** recommendation: ship as-is, or a single retrain campaign that
   bundles every change the data justifies. List the changes with the measured
   number that justifies each.
4. Add the bring-up outcome to `docs/jetson-mod/experiment_journal.md` as a
   deployment entry, with every number naming its source file.
5. Update `docs/jetson-mod/task_plan.md`'s phase table and `AGENTS.md:145`
   § Current Phase so the repo's own status is not stale — DOC-3 and DOC-4 record
   that letting this slide is a recurring failure here. While you are in
   `AGENTS.md`, fix the two known rot sites this phase surfaced: the corrupted
   Safety-Rules bullet at `:686`, and the § Domain Randomization note at `:355`
   that still describes the phantom kilogram on `base` as present tense.

**3. Unit tests**

- `tests/test_compare_sim_real.py::test_segment_alignment`: build a synthetic
  log with three command segments and assert the aligner returns the right index
  ranges, including at the boundaries.
- `::test_stance_duty_matches_known_signal`: feed a synthetic square-wave foot
  contact with a known 60 % duty and assert the computed duty is 60 ± 1 %.
- `::test_report_flags_divergence`: construct a synthetic pair where real
  tracking error is 10× sim and assert the report marks that metric as diverged.

**4. Smoke test**

```bash
python3 scripts/compare_sim_real.py \
  --real logs/bringup/walking_<ts>.npz \
  --sim docs/jetson-mod/sim2real/reference_trace_v5d_fwd.npz
```
Observable: a printed divergence table whose headline number is
**real p95 tracking error in radians**. Under ~0.05 rad means the actuator model
is roughly right; above ~0.15 rad means PLANT-5 and PLANT-6 are the first
suspects and the S.8 change is justified by data rather than by argument.

**5. Done when**

- [ ] `docs/jetson-mod/sim2real/S12_divergence_report.md` exists with all six
      metric families, sim and real side by side, and names the policy behind
      each log.
- [ ] Every diverging metric names a candidate cause by issue ID, or a new issue
      exists in `known_issues.md` with evidence and a reproduction command.
- [ ] The report ends with exactly one recommendation and the measured
      justification for each change in it.
- [ ] A deployment entry exists in `experiment_journal.md`.
- [ ] `python3 scripts/verify_known_issues.py` still exits **0 or 2** after any
      register edits. **Exit 2 is the current baseline** (it prints
      `CONFIRMED 28 / 29` with `DEPLOY-1` INCONCLUSIVE, because the system
      `python3` has no `onnx`). **Exit 1 means a check came back REFUTED** —
      either the issue was fixed or the check rotted; the script prints the
      evidence so you can tell which. For a full run including the ONNX checks:
      `~/IsaacLab/_isaac_sim/python.sh scripts/verify_known_issues.py`.
- [ ] `pytest tests/test_compare_sim_real.py -v` → 3 passed, and `pytest tests/`
      passes overall.
- [ ] `AGENTS.md` § Current Phase and `task_plan.md`'s phase table reflect
      reality, and `AGENTS.md:686` is no longer an edit instruction.

**AI-agent suitable:** YES — this is analysis of data that already exists, and an
agent should do all of it.

---

# Phase V — VLM / Cosmos Reason2 integration

> **This block replaces Tasks V.1–V.5 in `docs/jetson-mod/task_plan.md` (locate by heading).**
> The old block is kept for reference but its `RobotCommand` clamp values
> (`task_plan.md (locate by heading)`: `forward [-0.2, 0.3]`, `lateral [-0.2, 0.2]`, `turn [-0.3, 0.3]`)
> are **outside the trained hull on two of three axes** and must not be copied. Every number
> below was re-derived from the repo on 2026-08-11 by reading or running the named source.
>
> **⚠ NUMBERING COLLISION — READ BEFORE TOUCHING ANY TASK.** `task_plan.md` already contains
> a *different* Task V.1 (`:2446`, "Set Up Cosmos Reason2 on Jetson"), Task V.2 (`:2546`,
> "Build the Command Parser"), Task V.3 (`:2704`, "Integrate Cosmos with Locomotion
> Controller"), Task V.4 (`:2860`) and Task V.5 (`:2901`, "Add Voice Input"). An implementer
> reading one task at a time who opens `task_plan.md` looking for "Task V.2" will find the
> **wrong** task. **Task V.0 below is mandatory and must be done first**; it makes the repo
> self-consistent so that later tasks can be read in isolation.
>
> **Not this project:** the owner's *Duck Embody* work (LLM-as-SLAM harness + model
> benchmark) lives in a **different repository** and benchmarks language models against a
> simulated apartment. Phase V is the **on-robot** VLM: one small quantized VLM emitting
> velocity commands to a real 50 Hz locomotion policy. Do not import code, prompts, metrics
> or result numbers between the two. Do not cite Duck Embody numbers here. (`env_cfg.py:589`
> does legitimately cite `duck-embody embody_env_cfg.py:120-122` for the *fall definition* —
> that single borrowing predates this rule and stays.)

---

## Verified numbers (each was re-read or re-run on 2026-08-11; re-derive before trusting)

| Fact | Value | Source (read it) |
|---|---|---|
| Trained command hull | `vx (-0.148, 0.222)` m/s, `vy (-0.111, 0.111)` m/s, `wz (-0.5, 0.5)` rad/s | `exported_policies/v5d_contact_wrench_ppo/env.yaml:989-997`; the same three literals at `isaac_lab_env/open_duck_mini_v2/env_cfg.py:261-263` |
| Command term class | `UniformVelocityCommand` (`env.yaml:978`), `heading_command: true` (`:984`), `heading_control_stiffness: 0.5` (`:985`), `rel_standing_envs: 0.02` (`:986`), `rel_heading_envs: 1.0` (`:987`), `resampling_time_range: (10.0, 10.0)` (`:979-981`) | `env.yaml:976-1000` |
| Yaw is a heading servo | `wz = clip(0.5 · heading_error, ±0.5)` every step; the sampled `ang_vel_z` never survives, because `rel_heading_envs` is 1.0 | `known_issues.md` **PLANT-8** |
| Command dwell in training | **10.0 s, constant** for the shipped policy — a linear command changes at most once per 10 s, as a step | `env.yaml:979-981` |
| Standing is trained | 2% of envs get a zero command | `env.yaml:986` |
| Observation slice for the command | dims `[6:9]` of 59, and `velocity_commands` has `noise: None`, `scale: None`, `clip: None` — it is fed **raw** even though `enable_corruption: true` for the group | parsed from `env.yaml` `observations.policy`; table in `AGENTS.md:431-439` |
| Full obs layout | `base_ang_vel[0:3]`, `projected_gravity[3:6]`, `velocity_commands[6:9]`, `joint_pos_rel[9:25]`, `joint_vel_rel[25:41]`, `last_action[41:57]`, `gait_phase[57:59]` | `AGENTS.md:431-439`; matches the parsed `env.yaml` term order exactly |
| Action pipeline | `q_target = q_default + 0.25 · a`; **neither is in the ONNX** | `known_issues.md` **DEPLOY-1**; `env.yaml` `actions.joint_pos.scale = 0.25`, `use_default_offset: true` |
| `q_default` + joint order already exist on disk | `scripts/duck_init_pos.json` — keys `joint_order` (16 names) and `init_pos_rad`. **Do not re-type these values anywhere.** | read the file |
| Joint order is interleaved, not grouped | `0 left_hip_yaw, 1 neck_pitch, 2 right_hip_yaw, 3 left_hip_roll, 4 head_pitch, 5 right_hip_roll, 6 left_hip_pitch, 7 head_yaw, 8 right_hip_pitch, 9 left_knee, 10 head_roll, 11 right_knee, 12 left_ankle, 13 left_antenna, 14 right_antenna, 15 right_ankle` | `scripts/duck_init_pos.json` `joint_order` |
| ⇒ `head_yaw` telemetry dims | joint index **7** → `joint_pos_rel` obs dim **16**, `joint_vel_rel` obs dim **32**, action dim **7** (derived as `9+i` / `25+i` / `i`, the same arithmetic `verify_known_issues.py:457-458` uses) | derived; verify with `duck_init_pos.json` |
| Gait phase clock | all **240** reference motions have `nb_steps_in_period = 27` → `27 × 0.02 s = 0.54 s`; a free-running 50 Hz counter, command-independent | measured by unpickling `isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl` (240 entries, `Counter({27: 240})`); counter advanced at `imitation_reward.py:332`, consumed at `imitation_reward.py:337,343-346` |
| Control rate | `sim.dt = 0.005`, `decimation = 4` → 50 Hz, `dt = 0.02 s` | `env.yaml` `sim.dt`, `decimation` |
| Unmeasurable obs dims | antenna joints are indices `[13, 14]` → obs `joint_pos_rel [22, 23]`, `joint_vel_rel [38, 39]`; action outputs `[13, 14]` | `known_issues.md` **DEPLOY-3**; check code at `scripts/verify_known_issues.py:451-459` |
| Open-loop yaw is survivable | `wz = +0.50` for 30 s: **640 episodes, 0 falls**, `gait_valid: true`, duty 69.14/73.69%, `ang_vel_z_error 0.0947` rad/s. `wz = +0.30`: 640 episodes, 0 falls, duty 74.09/72.36%, err 0.1018. Aggregate over all 6 conditions: **3,840 episodes, 0 falls, 6/6 gait-valid** | `docs/jetson-mod/eval_results_v5/v5d_contact_wrench.json`, keys `per_condition["vx+0.00_vy+0.00_wz+0.50"]` / `["…wz+0.30"]` and `aggregate`. `evaluate_policies.py:1412` sets `term.cfg.heading_command = False`, so these rollouts *were* open-loop yaw |
| **Those numbers are stale** | measured on the 3.657 kg plant, before the PLANT-1 fix (commit `11b1690`, 2026-08-11) merged `base` into `trunk_assembly`; PhysX now simulates 2.657067 kg | `known_issues.md` PLANT-1 |
| VLM decode rate (claimed) | ~16–17 tok/s on Orin Nano Super | `AGENTS.md:296` |
| Memory budget (claimed) | Cosmos V.8 + TRT 0.1 + camera 0.3 + OS 1.5 = 7.7 GB of 8 GB | `AGENTS.md:188-196` |
| Serve command (claimed) | `vllm serve "embedl/Cosmos-Reason2-2B-W4A16-Edge2" --max-model-len 2048 --gpu-memory-utilization 0.70 --max-num-seqs 1` inside `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` | `AGENTS.md:671-682` |
| Camera | **IMX219 CSI module**, 1 unit, ~$15, 30 cm CSI ribbon; Phase-4 acceptance is `nvgstcapture-1.0` showing a live preview. `AGENTS.md` never names the sensor — only `task_plan.md` does | `task_plan.md (locate by heading), 1935, 1990`; mounted in head, ~3 g (`task_plan.md (locate by heading)`) |
| GPIO | eye LEDs BCM 23/24 (pins 16/18); antenna PWM BCM 12/13 (pins 32/33); foot switches BCM 22 (pin 15) / BCM 27 (pin 13); IMU SDA BCM 2 (pin 3), SCL BCM 3 (pin 5) | `AGENTS.md:635-649` |
| Fall / tilt termination | `bad_orientation` at `limit_angle = radians(60.0)`, and `root_height_below_minimum` at 0.09 m | `env_cfg.py:591-596` |
| Current test baseline | `python3 -m pytest tests/ -q` → **103 passed in ~2.4 s** (`known_issues.md` still says "101 passed"; that count is stale) | measured 2026-08-11 |
| Current defect-register baseline | `python3 scripts/verify_known_issues.py` → **CONFIRMED 28 / 29**, `DEPLOY-1` INCONCLUSIVE, **exit code 2** (the register header claiming "33 / 34" is stale) | measured 2026-08-11 |

**The arithmetic nobody has done yet.** The repo makes **three mutually inconsistent** claims
about VLM speed: `AGENTS.md:128` and `AGENTS.md:629` and `task_plan.md (locate by heading)` say **2–3 Hz**;
`AGENTS.md:633` says "**Cosmos takes ~300-500 ms**"; `AGENTS.md:296` says **~16–17 tok/s**.
At 16–17 tok/s, 2–3 Hz is 5–8 output tokens per cycle and 300–500 ms is 5–8 tokens too — so
the token-rate figure and the *latency* figures are only compatible if the model emits almost
nothing. The old Task V.2 asks for `max_tokens: 256` plus chain-of-thought, which decodes in
**~15 s** at that rate, before image prefill. **Every design decision below assumes the VLM
cycle is 0.2–1 Hz (1–5 s per decision), and Task V.2 measures the truth. The VLM is not a
reflex layer and must never be placed in a reflex path.**

---

## Repo traps that bite every task in this block

Read these once; they are not repeated in full under each task.

1. **`.gitignore:18` is `*.txt`.** Any `.txt` file you create — prompt files, captured model
   transcripts — is **silently untracked**. `git check-ignore -v jetson_runtime/prompts/navigate.txt`
   → `.gitignore:18:*.txt`. Tasks V.4 and V.7 depend on committing such files. Resolution is
   specified in Task V.4 step 0. `.gitignore:19` is `*.onnx` (that is `known_issues.md`
   **ART-1**, already filed).
2. **`env.yaml` contains `!!python/tuple` tags.** `yaml.safe_load()` **raises** on it. Use the
   loader given in Task V.1 step 3. Do **not** use `yaml.unsafe_load` — it tries to import
   Isaac Lab classes that are not installed on the Jetson.
3. **`env_cfg.py` has two different command configurations.** Lines 261-263 (in
   `OpenDuckRoughEnvCfg`, the base of the shipped v5d chain) carry the hull above. But
   `env_cfg.py:566-582`, inside `OpenDuckContactEnvCfg` (a *different* task,
   `Isaac-Velocity-Rough-OpenDuck-Contact-v0`), installs a `BandedWzVelocityCommandCfg` with
   `ang_vel_z=(-0.7, 0.7)`, `rel_heading_envs=0.7`, `rel_standing_envs=0.08`,
   `resampling_time_range=(2.0, 10.0)`. **The shipped policy did not train on that.**
   `OpenDuckContactWrenchEnvCfg` (`:742`) inherits `OpenDuckContactMinimalEnvCfg` (`:660`) →
   `OpenDuckRobustEnvCfg` (`:324`) → `OpenDuckRoughEnvCfg` (`:207`), never touching `:566`.
   **The exported `env.yaml` is the authority; grepping `env_cfg.py` for the hull is not.**
4. **`jetson_runtime/` does not exist.** `AGENTS.md:92` says "planned … not yet created" and
   `ls jetson_runtime` errors. You are creating it. `jetson_runtime/thermal_manager.py`
   (`AGENTS.md:815`, Task 4.6) and `jetson_runtime/trt_infer.py` (`AGENTS.md:657`) are
   Phase-4 outputs that also do not exist yet.
5. **`mini_bdx_runtime.hwi`, imported by the old skeleton at `task_plan.md (locate by heading), 2741`, is
   not in this repo.** `mini_bdx/mini_bdx/` contains only `old_walk_engine`,
   `placo_walk_engine`, `utils`. Do not import it.
6. **`tests/conftest.py` does `import mujoco` at module scope.** Collecting *anything* under
   `tests/` therefore requires `mujoco`. Your Phase-5 tests run on the dev machine, not on a
   bare Jetson, unless you install mujoco there.
7. **`onnx` is not installed in the system `python3`.** `import onnx` → `ModuleNotFoundError`.
   Any ONNX assertion must be `pytest.importorskip("onnx")`, or run under
   `~/IsaacLab/_isaac_sim/python.sh`.
8. **`pytest tests/ -v`, never a bare `pytest`** from the repo root — it walks into
   `experiments/RL/old_test.py` and dies on `ModuleNotFoundError: gymnasium`
   (`known_issues.md` **TEST-2**). The `phase5` marker **is** already registered in
   `pytest.ini`.
9. **Do not add grep-style tests.** `known_issues.md` **TEST-1**: 5 of 7 seeded regressions —
   including deleting the dominant `imitation_reward` outright — pass the existing suite,
   because 33 of 47 tests in `tests/test_isaac_lab_env.py` are
   `assert "<literal>" in open(file).read()`. Every test specified below parses or executes.
10. **One Isaac Sim / GPU job at a time** (`AGENTS.md:743-751`, Experiment Journal Protocol
    rule 7). Two Isaac processes collide during startup and the second dies in its init
    banner. Check before launching Task V.5.
11. **Every gate number quoted in this block predates the PLANT-1 fix.** The plant went
    3.657 kg → 2.657 kg on 2026-08-11 and v5d was trained on the heavy one. If a re-gate of
    v5d on the corrected plant lands before Phase 5 runs, re-read the hull from the *new*
    policy's `env.yaml` and update `jetson_runtime/command_contract.py` — that is the one
    file that changes, which is the whole point of Task V.1.
12. **Do not commit or push** unless the owner explicitly asks. (The one exception is Task
    V.7's protocol file, whose value depends on being committed before the trials — ask.)

---

### Task V.0 — Retire the superseded Phase 5 in `task_plan.md` (do this first)

**AI-agent suitable:** YES — pure documentation edit, no hardware, no GPU.

**1. Context for the implementing agent**

`docs/jetson-mod/task_plan.md` still contains a full Phase 5 ("Cosmos Reason2
VLM Integration"), written before this plan. It is superseded, and it is
actively dangerous to implement from: its velocity clamp exceeds the trained
command hull (`known_issues.md` DEPLOY-5), and it specifies a three-thread
single-process design that Task V.3 below deliberately rejects.

Because this plan is read one task at a time with no memory of the others, an
implementer who searches the repo for "Cosmos" will find the old section and
build the wrong thing. The tasks in *this* block are numbered `V.x` precisely so
they cannot be confused with `task_plan.md`'s `5.x` — do not renumber them back.

Read first:
- `docs/jetson-mod/task_plan.md`, the whole Phase 5 section. Locate it by
  heading (`## Phase 5: Cosmos Reason2 VLM Integration`), **not by line number** —
  line numbers in this repo drift and several have already gone stale.
- `AGENTS.md`, the Phase Summary table and the Output Files table.

No prior task.

**2. Low-level implementation plan**

1. In `docs/jetson-mod/task_plan.md`, move the entire existing Phase 5 section
   verbatim to the end of the file under a new heading:
   `## Appendix A — superseded Phase 5 (pre-2026-08-11), reference only`.
   Move it; do not delete it. It is the only record of the original VLM design
   intent, including the voice-input task that this plan drops.
2. Put a banner as the first line under that heading:
   *"Superseded by `docs/jetson-mod/task_plan_v2.md` Phase V. Its velocity clamp
   violates the trained command hull (`known_issues.md` DEPLOY-5) and its
   three-thread design was rejected. Do not implement from this appendix."*
3. In place of the moved section, leave a single pointer line:
   `Phase 5 (VLM) has moved to docs/jetson-mod/task_plan_v2.md Phase V. The
   superseded original is in Appendix A of this file.`
4. In `AGENTS.md`, update the Phase Summary row for Phase 5 and the Output Files
   table to point at `task_plan_v2.md` Phase V rather than at the moved section.
   Find both by their table headers, not by line number.
5. Do **not** touch `docs/jetson-mod/known_issues.md` or
   `scripts/verify_known_issues.py` in this task.

**3. Unit tests**

None applicable — this is a documentation move with no importable surface. Do
not invent a grep-based test for it; this repo already has a suite that greps
source text and passes through real regressions (`known_issues.md` TEST-1).

**4. Smoke test**

```bash
cd /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
grep -n "Appendix A — superseded Phase 5" docs/jetson-mod/task_plan.md
grep -c "task_plan_v2.md" AGENTS.md
python3 -m pytest tests/ -q
```

Observable: the first grep returns exactly one line; the second returns a count
of at least 1; pytest still reports `103 passed`.

**5. Done when**

- [ ] `task_plan.md`'s Phase 5 body appears exactly once, under `## Appendix A`,
      with its banner as the first line
- [ ] A pointer line to `task_plan_v2.md` Phase V sits where the section was
- [ ] `AGENTS.md` points at `task_plan_v2.md` Phase V for the VLM work
- [ ] `grep -rn "Task 5\." AGENTS.md` returns nothing that refers to VLM work
- [ ] `python3 -m pytest tests/ -q` still reports `103 passed`

---

### Task V.1 — Freeze the command contract in code

**AI-agent suitable:** YES — pure Python + docs, no hardware, no GPU.

**1. Context for the implementing agent**

Every later Phase-5 task needs the same handful of numbers: the velocity hull, the action
scale, the observation layout, the control rate. Today they exist in four places that have
already drifted (`AGENTS.md`, `task_plan.md`'s Phase-5 code skeleton, `env_cfg.py`, the
exported `env.yaml`), and the drift already produced a filed defect — `known_issues.md`
**DEPLOY-5**, "documented velocity clamp exceeds the trained command hull". DEPLOY-5's *text*
was fixed on 2026-08-11, but the fix left `AGENTS.md:686` as a malformed instruction rather
than a rule (see step 6), and `task_plan.md (locate by heading)` still carries the bad numbers in code.
If you skip this task, the VLM safety layer gets written against a lateral clamp that is
**1.80×** the trained hull, and the robot is commanded into a region the policy has never seen.

Read first:
- `exported_policies/v5d_contact_wrench_ppo/env.yaml` — the shipped policy's own frozen
  config. **This is the authority.** Look at `commands.base_velocity.ranges` (`:988-1000`),
  `actions.joint_pos.scale`, and the ordered keys of `observations.policy`.
- `scripts/duck_init_pos.json` — `joint_order` and `init_pos_rad` (q_default) already exist.
- `docs/jetson-mod/known_issues.md` sections **DEPLOY-1**, **DEPLOY-3**, **DEPLOY-5**,
  **PLANT-8**, **TEST-1**, **TEST-2**.
- `AGENTS.md:421-448` (§ Observation Space, incl. the deployment trap) and `AGENTS.md:618-689`
  (§ Jetson Deployment).

Depends on: Task V.0. Everything else in Phase V depends on this.

Traps: all twelve in the cross-cutting list, especially **2** (`!!python/tuple`), **3** (two
command configs in `env_cfg.py`), **4** (`jetson_runtime/` does not exist), **9** (no grep
tests).

**2. Low-level implementation plan**

1. Create `jetson_runtime/__init__.py` (empty) and `jetson_runtime/command_contract.py`.
2. In `command_contract.py`, define module-level constants — no classes, no I/O, no
   third-party imports, so it can be imported on the Jetson and in tests:
   ```python
   # Trained command hull — exported_policies/v5d_contact_wrench_ppo/env.yaml:989-997
   VX_MIN, VX_MAX = -0.148, 0.222      # m/s, base frame
   VY_MIN, VY_MAX = -0.111, 0.111      # m/s, base frame
   WZ_MIN, WZ_MAX = -0.5, 0.5          # rad/s, world-frame yaw

   # Operational yaw limit for open-loop VLM commands. Inside the hull on purpose:
   # see known_issues.md PLANT-8. Overridable per deployment.
   WZ_OPERATIONAL_LIMIT = 0.3

   # Reject-vs-clamp boundary: a value this far outside the hull is a parse or model
   # failure, not a saturated intent, and must not be silently clamped.
   REJECT_FACTOR = 3.0

   ACTION_SCALE = 0.25                 # env.yaml actions.joint_pos.scale (DEPLOY-1)
   OBS_DIM, ACT_DIM = 59, 16
   CMD_SLICE = slice(6, 9)             # velocity_commands occupies obs[6:9]
   OBS_TERM_ORDER = ("base_ang_vel", "projected_gravity", "velocity_commands",
                     "joint_pos", "joint_vel", "actions", "gait_phase")
   # Slice map, AGENTS.md:431-439. joint_pos is joint_pos_rel = q - q_default.
   OBS_SLICES = {
       "base_ang_vel": slice(0, 3), "projected_gravity": slice(3, 6),
       "velocity_commands": slice(6, 9), "joint_pos": slice(9, 25),
       "joint_vel": slice(25, 41), "actions": slice(41, 57),
       "gait_phase": slice(57, 59),
   }
   ANTENNA_JOINT_IDX = (13, 14)        # scripts/duck_init_pos.json joint_order
   ANTENNA_OBS_DIMS = (22, 23, 38, 39) # DEPLOY-3 — unmeasurable on hardware
   ANTENNA_ACTION_DIMS = (13, 14)
   HEAD_YAW_JOINT_IDX = 7              # duck_init_pos.json; obs 16 / 32, action 7

   CONTROL_HZ, CONTROL_DT = 50.0, 0.02 # env.yaml sim.dt 0.005 x decimation 4
   GAIT_PERIOD_STEPS = 27              # all 240 motions; 0.54 s at 50 Hz

   HEADING_STIFFNESS = 0.5             # env.yaml heading_control_stiffness
   HEADING_CLIP = 0.5                  # env.yaml ranges.ang_vel_z bound
   TRAINING_COMMAND_DWELL_S = 10.0     # env.yaml resampling_time_range

   # Set by the Task V.6 step-6 physical measurement. +1 is a placeholder.
   HEAD_YAW_SIGN = +1
   ```
3. Add a **loader**, not a copy, for `q_default`:
   `load_q_default() -> tuple[list[str], list[float]]` reads
   `scripts/duck_init_pos.json` (resolve the path relative to this file's parent's parent, so
   it works from any cwd), returns `(joint_order, [init_pos_rad[n] for n in joint_order])`,
   and raises if `len(joint_order) != ACT_DIM`. **Never re-type the 16 pose values.**
4. Add two pure functions in the same file:
   - `clamp_command(vx, vy, wz, wz_limit=WZ_OPERATIONAL_LIMIT) -> tuple[float,float,float]`
     — clamps each axis; raises `ValueError` on non-finite input.
   - `is_rejectable(vx, vy, wz) -> bool` — `True` if any axis is non-finite, or lies outside
     `REJECT_FACTOR ×` its hull half-width about the hull midpoint.
5. Add `jetson_runtime/README.md` stating in three lines: this package is the on-robot
   runtime; `command_contract.py` is the single source of truth; anyone changing a constant
   must change the shipped policy's `env.yaml` first (i.e. retrain), not this file.
6. Edit `AGENTS.md:686`. That bullet, under § Safety Rules, is **malformed**: it reads as an
   instruction to an agent ("Apply the same line edit … but ONLY as part of a three-file
   change"), embeds the clamp numbers inline, and cites stale line numbers
   (`known_issues.md` "789-798", `verify_known_issues.py` "EVAL-3 at 344-358"). Both of those
   have moved — DEPLOY-5 now lives at `known_issues.md (locate by heading)` and the check was retired at
   `verify_known_issues.py:328-332`. Replace the whole bullet with:
   *"All velocity commands from any autonomy layer MUST be clamped by
   `jetson_runtime/command_contract.clamp_command()`, which carries the trained hull. Do not
   re-type the numbers anywhere else."*
7. Edit `AGENTS.md:92` (§ Key Directories) — change the `jetson_runtime/` line from "planned,
   Phases 4-5; not yet created" to note that `command_contract.py` now exists and is the
   single source of truth for deployment constants.
8. Do **not** touch `scripts/verify_known_issues.py`. DEPLOY-5's check was already retired
   (`verify_known_issues.py:328-332`); adding one back would report REFUTED and change the
   script's exit code.

**3. Unit tests**

New file `tests/test_command_contract.py`, marked `@pytest.mark.phase5`. These **parse** the
exported config; none of them greps source text.

1. `test_hull_matches_shipped_policy` — load `env.yaml` with:
   ```python
   class _L(yaml.SafeLoader): pass
   _L.add_constructor("tag:yaml.org,2002:python/tuple",
                      lambda l, n: tuple(l.construct_sequence(n)))
   _L.add_multi_constructor("tag:yaml.org,2002:python/object", lambda l, s, n: None)
   _L.add_multi_constructor("tag:yaml.org,2002:python/name", lambda l, s, n: None)
   ```
   Assert `r["lin_vel_x"] == (VX_MIN, VX_MAX)`, `r["lin_vel_y"] == (VY_MIN, VY_MAX)`,
   `r["ang_vel_z"] == (WZ_MIN, WZ_MAX)` (floats written by the exporter; compare with
   `pytest.approx(rel=0, abs=1e-12)`).
2. `test_action_scale_matches` — assert
   `env["actions"]["joint_pos"]["scale"] == ACTION_SCALE` **and**
   `env["actions"]["joint_pos"]["use_default_offset"] is True` (that flag is what makes the
   offset `q_default` rather than the literal `offset: 0.0` in the same block — a real
   confusion trap).
3. `test_obs_term_order_matches` — take `observations.policy`, keep only keys whose value is
   a `dict` containing the key `"func"` (this correctly drops `concatenate_terms`,
   `concatenate_dim`, `enable_corruption`, `history_length`, `flatten_history_dim`, and the
   two disabled terms `base_lin_vel: None` and `height_scan: None`), and assert the resulting
   tuple equals `OBS_TERM_ORDER`. This is the test that catches a future retrain silently
   changing the layout.
4. `test_velocity_command_obs_is_raw` — assert `noise`, `scale` and `clip` are all `None` on
   the `velocity_commands` term, so the runtime must not pre-scale the command.
5. `test_heading_params_match` — assert `heading_control_stiffness == HEADING_STIFFNESS`,
   `rel_heading_envs == 1.0`, `heading_command is True`,
   `resampling_time_range == (TRAINING_COMMAND_DWELL_S,) * 2`.
6. `test_q_default_loads_and_matches_json` — call `load_q_default()`; assert 16 names, that
   `joint_order[13] == "left_antenna"` and `joint_order[14] == "right_antenna"` (so
   `ANTENNA_JOINT_IDX` is derived, not asserted), that `joint_order[7] == "head_yaw"`, and
   that every returned value equals the corresponding `init_pos_rad` entry.
7. `test_operational_yaw_inside_hull` — assert `WZ_OPERATIONAL_LIMIT <= WZ_MAX`.
8. `test_clamp_and_reject` — `clamp_command(9, -9, 9) == (VX_MAX, VY_MIN,
   WZ_OPERATIONAL_LIMIT)`; `clamp_command(float("nan"), 0, 0)` raises `ValueError`;
   `is_rejectable(999, 0, 0) is True`; `is_rejectable(0.25, 0, 0) is False`.
9. `test_onnx_io_shapes` — `pytest.importorskip("onnx")`, load
   `exported_policies/v5d_contact_wrench_ppo/policy.onnx`, assert input shape `[1, 59]` and
   output `[1, 16]`. **This will SKIP on the dev machine** — `onnx` is not installed in the
   system `python3`. That is expected and correct; to actually run it use
   `~/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_command_contract.py -k onnx`.
   (Note `policy.onnx` is gitignored per `known_issues.md` ART-1 but is present on disk.)

**4. Smoke test**

```bash
cd /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
python3 -m pytest tests/test_command_contract.py -v
python3 -c "from jetson_runtime.command_contract import clamp_command; print(clamp_command(0.9,-0.9,0.9))"
python3 -m pytest tests/ -q
```
Observable: the first command exits `0` with **9 tests collected**, 8 passed and
`test_onnx_io_shapes` reported `skipped`. The one-liner prints exactly
`(0.222, -0.111, 0.3)`. The last command reports **112 passed, 1 skipped** (103 pre-existing
+ 9 new, one skipping) — if the pre-existing count is not 103, record what it actually is
before your change and compare against that instead.

**5. Done when**

- [ ] `jetson_runtime/__init__.py`, `jetson_runtime/command_contract.py`,
      `jetson_runtime/README.md` exist.
- [ ] `python3 -m pytest tests/test_command_contract.py -v` exits 0; exactly one test skips
      (the ONNX one) and no test fails.
- [ ] `python3 -m pytest tests/ -q` shows the pre-existing count still passing plus the 9 new
      tests.
- [ ] `grep -rl -e "0\.148" jetson_runtime/ | wc -l` prints `1` — the hull numbers are typed
      in exactly one file. (Use `grep -rl`; a bare `grep -c` on a directory errors with
      `Is a directory`.)
- [ ] `python3 -c "from jetson_runtime.command_contract import load_q_default;
      print(load_q_default()[0][7])"` prints `head_yaw`, run from at least two different
      working directories.
- [ ] `AGENTS.md:686`'s malformed bullet is replaced by a plain rule referencing
      `clamp_command()`, with no numeric clamp values in it.
- [ ] `python3 scripts/verify_known_issues.py` reports the **same** result as before your
      change. Record the baseline first: today it prints `CONFIRMED 28 / 29`, lists
      `DEPLOY-1` as INCONCLUSIVE, and **exits 2** — 2 means "a check could not be evaluated
      here" (no `onnx` in the system interpreter), *not* failure. Do not expect exit 0.

---

### Task V.2 — Stand up the Cosmos server on the Jetson and measure the real budget

**AI-agent suitable:** PARTIAL — an agent can write every script and parse every log, but
the run itself requires the **physical Jetson Orin Nano Super with the IMX219 CSI camera
attached** (Phase 4 hardware, `task_plan.md (locate by heading), 1977, 1990`). A human must have the board
powered, the camera ribbon seated, and the board reachable. All numbers in this task are
*measurements*; none may be guessed.

**1. Context for the implementing agent**

The whole phase rests on one unverified claim: that a 2B VLM, a TensorRT locomotion engine, a
camera pipeline and the OS coexist in **8 GB of unified memory** — unified meaning the VLM's
"GPU memory" and the OS's RAM are the *same* pool, so vLLM's `--gpu-memory-utilization 0.70`
is taking 70% of the machine, not 70% of a separate card. If this is skipped, the first
symptom will be the OOM killer terminating whichever process the kernel likes least, and on a
walking biped that process might be the 50 Hz control loop.

Read first:
- `AGENTS.md:188-196` — the claimed budget (V.8 / 0.1 / 0.3 / 1.5 = 7.7 GB).
- `AGENTS.md:669-682` — the serve command, verbatim, and the query shape.
- `AGENTS.md:291-299` — the model id and the ~16–17 tok/s figure.
- `AGENTS.md:626-633` — the two-thread picture *and* its "~300-500 ms" claim, which is the
  third of three inconsistent speed claims you are here to resolve.
- `docs/jetson-mod/task_plan.md` (locate by heading) (old Task V.1) — the container pull and the smoke
  queries; keep those, replace the acceptance criteria.
- `docs/jetson-mod/task_plan.md` (locate by heading) (Task 4.6, `thermal_manager.py`) — the
  `request_boost()` / `release_boost()` contract (`:2334-2355`) you must exercise here.
  **`jetson_runtime/thermal_manager.py` does not exist yet**; it is a Phase-4 deliverable
  (`AGENTS.md:815`). If it is absent, do step 7 with a direct `nvpmodel` call and say so.

Depends on: Phase 4 complete (robot walks on the Jetson) and Task V.0. It does **not** depend
on V.1, and can run in parallel with V.1, V.3 and V.4's offline half.

Traps:
- **Start the locomotion process first, the VLM second.** vLLM pre-allocates its pool at
  startup; TensorRT allocates lazily. If vLLM claims memory first, the control loop's
  allocation failure happens *later*, at an arbitrary moment, possibly mid-stride. In the
  other order, the failure happens at VLM startup, where it is harmless and visible.
- The claimed ~V.8 GB and the flag `--gpu-memory-utilization 0.70` are two different
  quantities (weights+activations vs. a reservation fraction including the KV cache). Report
  both: resident RSS *and* what vLLM logs as its KV-cache size.
- A desktop session on Orin costs several hundred MB. Measure with the GUI both on and off.
- `--limit-mm-per-prompt` accepts different value syntaxes across vLLM releases (a JSON blob
  in newer builds, `image=1` in older ones). Read `vllm serve --help` **inside the pulled
  container** and use whatever that build accepts; record the container digest and the
  `vllm --version` output in the measurements doc so the flag set is reproducible.
- Plus the cross-cutting traps, especially **6** (`tests/conftest.py` imports `mujoco`, so the
  parser tests below need mujoco wherever you run them) and **8**.

**2. Low-level implementation plan**

1. Create `jetson_runtime/serve_cosmos.sh` — the docker+serve command from `AGENTS.md:673-676`
   verbatim, with every tunable exposed as an environment variable with the documented
   default: `GPU_MEM_UTIL=0.70`, `MAX_MODEL_LEN=2048`, `MAX_NUM_SEQS=1`, plus a
   single-image multimodal limit (syntax per the trap above) and an `ENFORCE_EAGER` switch
   that appends `--enforce-eager`. Log the full resolved command line, the container image
   digest and `vllm --version` to stdout on start.
2. Create `jetson_runtime/measure_budget.py`:
   - Runs `tegrastats --interval 1000` as a subprocess, parses the `RAM x/y MB` field, and
     records `(t, used_mb, total_mb, gpu_pct, soc_temp_c)` to CSV under
     `docs/jetson-mod/cosmos_measurements/`.
   - Exposes `--stage <name>` and `--duration <s>` so each measurement writes a labelled CSV.
   - Prints p50/p95/max used-MB for the stage on exit.
3. Measure the ladder, in this order, ~120 s each, appending to
   `docs/jetson-mod/cosmos_serving_measurements.md`:
   1. `os_idle` — nothing running, GUI disabled
      (`sudo systemctl set-default multi-user.target`, reboot).
   2. `os_idle_gui` — same with the GUI, for the delta.
   3. `camera` — IMX219 CSI capture at 640×480, 30 fps, JPEG-encoded, discarded.
   4. `locomotion` — the Phase-4 control loop at 50 Hz with the TRT engine loaded.
   5. `vllm_idle` — plus the served model, no requests.
   6. `full_query` — all of the above with one query per 2 s for 120 s.
4. Create `jetson_runtime/measure_latency.py`: sends `N=30` identical image+text queries for
   each `max_tokens ∈ {16, 32, 64, 128, 256}`, records time-to-first-token, total wall time,
   and `usage.completion_tokens` from the response; prints p50/p95 and the derived tok/s.
   Write the table into the same measurements doc.
5. If stage 5 or 6 OOMs, walk this ladder in order and record which rung fixed it:
   `GPU_MEM_UTIL 0.70 → 0.65 → 0.60`; `MAX_MODEL_LEN 2048 → 1536 → 1024`; `ENFORCE_EAGER=1`;
   camera JPEG longest side `640 → 512 → 448`; GUI off.
6. Protect the control loop from the VLM regardless of the outcome: in the Phase-4 locomotion
   entry point, call `mlockall(MCL_CURRENT | MCL_FUTURE)` via `ctypes`, launch it under
   `chrt -r 20`, and set its `oom_score_adj` to `-500`
   (`echo -500 > /proc/<pid>/oom_score_adj`, as root). Set the vLLM container's to `+500`.
   Record both in the measurements doc.
7. Measure the thermal interaction: with the locomotion loop running at 50 Hz, switch between
   the 15W and 25W power modes ten times — via `ThermalManager.set_power_mode()` if Task 4.6
   has landed, otherwise `sudo nvpmodel -m <id>` directly — and record the worst control-loop
   period observed across the transition.

**3. Unit tests**

`tests/test_measure_budget_parsers.py` (`@pytest.mark.phase5`) — the *parsers* are testable
without hardware, which is the only part that is:
- `test_tegrastats_line_parses` — feed a captured `tegrastats` line fixture to the parser;
  assert `used_mb`, `total_mb`, `soc_temp_c` come back as floats with expected values. Store
  the fixture as `tests/fixtures/tegrastats_sample.log` (20 real lines captured on the
  Jetson). **Not `.txt`** — `.gitignore:18` would swallow it (cross-cutting trap 1).
- `test_tegrastats_garbage_line_ignored` — a truncated line yields `None`, not an exception.
- `test_latency_summary_math` — feed a synthetic list of 30 durations; assert p50/p95 match
  `numpy.percentile` to 1e-9. (`numpy` and `scipy` are both present in the dev `python3`.)

The measurement itself is not unit-testable — it is hardware observation, covered by the
smoke test.

**4. Smoke test**

On the Jetson, with the control loop already running (see the ordering trap):
```bash
bash jetson_runtime/serve_cosmos.sh &            # wait for "Application startup complete"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
python3 jetson_runtime/measure_budget.py --stage full_query --duration 120
python3 jetson_runtime/measure_latency.py --max-tokens 48 --n 30
```
Observables that prove it worked:
- `/health` prints **200**.
- `measure_budget.py` prints a **p95 used-MB below 7,400** on an 8 GB board (i.e. ≥ 600 MB
  headroom) for stage `full_query`, and `dmesg | grep -i "oom"` is empty afterwards.
- `measure_latency.py` prints a **p95 end-to-end latency** and a **measured tok/s**; both land
  in `docs/jetson-mod/cosmos_serving_measurements.md` regardless of value. If p95 at
  `max_tokens=48` exceeds 5 s, that is a *finding*, not a failure — record it, and Task V.3
  sizes its watchdog against it.

**5. Done when**

- [ ] `docs/jetson-mod/cosmos_serving_measurements.md` exists and contains six labelled memory
      stages with p50/p95/max used-MB, each naming its CSV file.
- [ ] It contains a latency table for five `max_tokens` values with measured tok/s, and the
      container image digest + `vllm --version` used.
- [ ] It states the final chosen `serve_cosmos.sh` settings and which ladder rungs (if any)
      were needed.
- [ ] It states the measured worst-case 50 Hz loop period across a power-mode transition, and
      a written verdict: duty-cycle 15W↔25W, or pin one mode.
- [ ] It contains one sentence reconciling — or explicitly refuting — the three prior speed
      claims at `AGENTS.md:128`, `AGENTS.md:296` and `AGENTS.md:633`.
- [ ] `AGENTS.md:188-196`'s budget block is updated to the measured numbers, with the claimed
      numbers kept alongside and labelled as the prior estimate.
- [ ] No OOM kill in `dmesg` across a 10-minute `full_query` run.
- [ ] `python3 -m pytest tests/test_measure_budget_parsers.py -v` exits 0 on the dev machine.

---

### Task V.3 — Build the safety layer (clamp, dwell, watchdog, heading mode)

**AI-agent suitable:** YES — pure Python, fully testable off-robot with a fake clock.

**1. Context for the implementing agent**

This is the component that stands between a 2B quantized language model and a walking biped.
Nothing downstream of it may assume the VLM is sane, fast, alive, or numeric. If this is
skipped or written casually, the failure mode is not a bad trajectory — it is the policy
receiving `vy = -999`, or a command that has not changed since the VLM process died four
minutes ago.

Read first:
- `jetson_runtime/command_contract.py` (Task V.1) — every constant comes from there.
- `docs/jetson-mod/known_issues.md` § **PLANT-8** (`:481-529`) in full. Summary: the yaw
  channel was trained as a *heading servo*, `wz = clip(0.5 · heading_error, ±0.5)`, in 100%
  of environments (`rel_heading_envs: 1.0`). An open-loop consumer can hold `wz = 0.5` while
  the robot is **already pointed at the target** — a correlation between command and heading
  error that the policy never saw. Note what PLANT-8 explicitly does *not* say: it does not
  say the endpoint is untrained. Measured runs at `|wz| ≥ 0.49` lasted a mean **3.35 s**, max
  **12.72 s**, and PLANT-8's own conclusion is "**A sustained constant yaw rate at the range
  endpoint is IN distribution.**" Independently,
  `docs/jetson-mod/eval_results_v5/v5d_contact_wrench.json` shows `wz = +0.5` held open-loop
  for 30 s over 640 episodes with **0 falls** and `gait_valid: true`. So this is a
  *behavioural* risk (spinning past the target forever), not a demonstrated *stability* risk —
  on flat ground, on the pre-PLANT-1 plant.
- `AGENTS.md:687` § Safety Rules — "fall back to last known good command (not zero — that
  could cause mid-stride fall)". Treat this as a **hypothesis, not a fact**: `env.yaml:986`
  shows 2% of training envs held a zero command, so *standing* is in distribution; what is
  uncharacterised is the *transition* to zero mid-stride. Task V.5 measures it. Implement
  both behaviours behind a flag and let V.5 pick.

Depends on: Tasks V.0 and V.1. It does **not** depend on V.2 — but its `HOLD_S` / `STALE_S`
defaults must later be reconciled against V.2's measured p95 (see step 5).

Traps:
- Training resampled linear commands **once per 10 s** (`env.yaml:979-981`). A VLM at 1 Hz is
  already changing commands **10× faster than anything in the training distribution**. This
  is the single most important number in the task and it argues for a minimum dwell. (Do not
  be misled by `env_cfg.py:568`'s `resampling_time_range=(2.0, 10.0)` — that belongs to the
  *Contact* track, not the shipped v5d chain; cross-cutting trap 3.)
- Do not use wall-clock (`time.time()`); use `time.monotonic()`. NTP steps would otherwise
  make a fresh command look hours old.
- The gait-phase counter (`obs[57:59]`) is a free-running 27-step cycle owned by the
  locomotion loop. The arbiter must never touch, reset or gate it.
- Log files must not use a `.txt` extension (cross-cutting trap 1). `.jsonl` is fine.

**2. Low-level implementation plan**

1. Create `jetson_runtime/command_arbiter.py` with a `CommandArbiter` class that is a pure
   state machine — no I/O other than the append-only log in step 8, no threads, an injectable
   `clock` defaulting to `time.monotonic`, so tests can drive time by hand.
2. Public API, exactly three methods:
   - `submit(raw: dict, t_emit: float) -> SubmitResult` — called by the VLM side with the
     parsed model output.
   - `tick(t_now: float, yaw_rate_meas: float) -> tuple[float, float, float]` — called by the
     locomotion loop **every control step**, returns the `(vx, vy, wz)` to write into
     `obs[6:9]`.
   - `status() -> dict` — telemetry for logs and LEDs.
3. `submit()` validation pipeline, in this order, each stage counted in `status()`:
   1. Structural: `raw` must have numeric `forward` and `lateral`; missing → reject.
   2. Finite: `math.isfinite` on every axis; `NaN`/`inf` → reject.
   3. `is_rejectable()` from the contract → reject (`999` is a broken model, not a fast duck)
      and keep the previous command.
   4. `clamp_command()` → clamp, and increment `clamp_count` per axis.
   5. Deadband: if `|Δvx| < 0.02` and `|Δvy| < 0.02` and `|Δwz| < 0.05` versus the current
      command, keep the current command but refresh its timestamp (this is an *affirmation*,
      not a change — it must reset the watchdog).
   6. Dwell: if `t_emit - t_last_change < MIN_DWELL_S` (default **1.0 s**), queue the new
      command and apply it when the dwell expires; do not drop it.
4. **Yaw modes.** `CommandArbiter(yaw_mode=...)` supports two, and heading is the default:
   - `"heading"` (**default, and the answer to PLANT-8**): the VLM emits `heading_deg` — the
     *relative bearing to where it wants to go*, not a rate. The arbiter reproduces the
     training-time generator exactly:
     ```python
     err = wrap_to_pi(heading_target - yaw_integrated)   # rad
     wz  = clip(HEADING_STIFFNESS * err, -HEADING_CLIP, +HEADING_CLIP)
     wz  = clip(wz, -wz_limit, +wz_limit)
     ```
     where `yaw_integrated` accumulates `yaw_rate_meas · CONTROL_DT` since the bearing was
     emitted. `yaw_rate_meas` is the **z component of `base_ang_vel`, i.e. `obs[2]`** — pass it
     in; do not read the obs vector inside the arbiter. When the error is closed, `wz` decays
     to zero *by construction* — the robot stops turning because it is aligned, which is
     precisely the correlation PLANT-8 says the policy learned. A heading target expires after
     `HEADING_TTL_S` (default **3.0 s**): a stale bearing is worse than no bearing, because
     the world has moved.
   - `"rate"` (fallback, for teleop parity and A/B ablation): the VLM's `turn` is used
     directly, clamped to `WZ_OPERATIONAL_LIMIT` (0.3), plus a **sustained-yaw budget**: if
     `|wz| ≥ 0.25` continuously for more than `YAW_BUDGET_S` (default **3.0 s** — just under
     the 3.35 s mean run measured in PLANT-8), force `wz → 0` for `YAW_REST_S` (default
     1.0 s). This bounds the "spins forever" failure without leaving the hull.
5. **Watchdog**, evaluated in `tick()` from `age = t_now - t_last_accept`:
   | age | behaviour | `status().state` |
   |---|---|---|
   | `≤ HOLD_S` (default 2.0 s) | zero-order hold of the accepted command | `FRESH` |
   | `HOLD_S < age ≤ STALE_S` (default 3.0 s) | linear decay of all three axes to zero over `DECAY_S` (default 0.5 s) | `STALE` |
   | `> STALE_S` | hard zero, latched | `LOST` |
   Recovery out of `LOST` requires **two consecutive accepted submissions** before motion
   resumes (anti-flap on a thrashing server). Size the defaults against Task V.2's measured
   p95 latency: `HOLD_S ≥ 2 × p95`. If V.2 has not run yet, keep 2.0/3.0 and write
   `# TODO(V.2): resize against docs/jetson-mod/cosmos_serving_measurements.md` on the line.
6. **Application shape.** `apply_mode ∈ {"step", "slew"}`, default `"step"` (a step is the
   shape training used at every resample); `"slew"` ramps at `SLEW_MPS2` (default 0.4 m/s²,
   which crosses the 0.370 m/s vx span in ~0.93 s). Task V.5 chooses the shipped default;
   both must exist first.
7. `status()` returns: `state`, `age_s`, `command`, `clamp_count` per axis, `reject_count` by
   stage, `dwell_deferrals`, `watchdog_trips`, `yaw_budget_trips`, `heading_err_rad`. Tasks
   V.6 and V.7 consume exactly this dict — do not rename keys later.
8. Structured logging: one JSON line per accepted command and one per state transition, to
   `~/duck_logs/arbiter_<iso8601>.jsonl` (create the directory if absent). This file is the
   evidence base for V.5 and V.7, and its schema is the trace format V.5 replays.

**3. Unit tests**

`tests/test_command_arbiter.py`, `@pytest.mark.phase5`, all with an injected fake clock — no
`sleep`, no network, no hardware. The suite must run in under a second.

- `test_clamps_to_hull` — submit `(0.9, -0.9, 0.9)`; `tick()` returns
  `(0.222, -0.111, 0.3)` and `clamp_count` is 3.
- `test_rejects_garbage_keeps_last_good` — accept `(0.15, 0, 0)`, then submit `(999, 0, 0)`;
  `tick()` still returns `0.15` and `reject_count["magnitude"] == 1`.
- `test_rejects_nan` — `(nan, 0, 0)` rejected, no exception escapes.
- `test_deadband_refreshes_watchdog` — accept, advance 1.9 s, submit an identical command,
  advance 1.9 s more: state is still `FRESH` and the command is unchanged.
- `test_min_dwell_defers_not_drops` — two submissions 0.2 s apart with different values;
  assert the second is applied at `t_first + 1.0 s`, not discarded.
- `test_watchdog_decays_then_zeroes` — accept `(0.2, 0, 0)`, then only `tick()`. Assert
  `FRESH` at 1.9 s, output strictly between 0 and 0.2 at 2.25 s, exactly `(0,0,0)` at 3.1 s,
  and `state == "LOST"`.
- `test_lost_requires_two_good_to_recover` — from `LOST`, one submission leaves the output at
  zero; the second releases it.
- `test_heading_mode_decays_when_aligned` — `yaw_mode="heading"`, target 1.0 rad; feed
  `yaw_rate_meas` equal to the returned `wz` each tick; assert `|wz|` is monotonically
  non-increasing and `|wz| < 0.02` within 5 s. **This is the PLANT-8 regression test** — it
  asserts the command stops when alignment is achieved.
- `test_heading_mode_saturates_far_from_target` — target π rad → first `wz` is exactly `+0.3`
  (operational limit), and exactly `+0.5` when constructed with `wz_limit=WZ_MAX`.
- `test_heading_target_expires` — no new target for `HEADING_TTL_S + 0.1`; `wz == 0`.
- `test_rate_mode_yaw_budget` — hold `turn = 0.3` continuously; assert output drops to 0 after
  3.0 s and resumes after the 1.0 s rest.
- `test_slew_mode_respects_rate` — in `apply_mode="slew"`, step from 0 to `VX_MAX`; assert no
  consecutive `tick()` pair differs by more than `SLEW_MPS2 * CONTROL_DT` (+1e-9).
- `test_output_always_inside_hull` — property test: 10,000 pseudo-random submissions with a
  **fixed seed** (including `inf`, `nan`, strings, missing keys, huge values, wrong types)
  interleaved with random `tick()`s; assert **every** returned triple is finite and satisfies
  the hull. This is the one test that must never be deleted.
- `test_never_touches_gait_phase` — `tick()` returns exactly a 3-tuple; assert
  `{"phase", "gait", "step_idx"} & set(dir(CommandArbiter)) == set()`, and that a sentinel
  object passed in as `yaw_rate_meas`'s source is not mutated.

**4. Smoke test**

```bash
cd /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
python3 -m pytest tests/test_command_arbiter.py -v
python3 -m jetson_runtime.command_arbiter --demo   # scripted 60 s trace, fake clock
```
The `--demo` entry point replays a hard-coded 60 s script (good commands → garbage → silence →
recovery) on the fake clock and prints one line per simulated second. Observable: it prints a
state sequence containing `FRESH → STALE → LOST → FRESH`, the final line reports
`out_of_hull_violations: 0`, the process exits `0`, and it completes in well under a second of
real time (the fake clock means it must not actually take 60 s).

**5. Done when**

- [ ] `jetson_runtime/command_arbiter.py` exists and imports with only the standard library
      plus `jetson_runtime.command_contract` (verify: `python3 -c "import ast,sys;
      [print(n) for n in ...]"` or simply that it imports in a venv without numpy).
- [ ] `python3 -m pytest tests/test_command_arbiter.py -v` exits 0 with **≥ 14 tests**.
- [ ] `test_output_always_inside_hull` passes with 10,000 seeded randomized submissions.
- [ ] `--demo` prints `out_of_hull_violations: 0` and the four-state sequence, and finishes in
      under 5 s of wall time.
- [ ] A `.jsonl` log file is produced by the demo under `~/duck_logs/` and every line parses
      as JSON (`python3 -c "import json,sys;[json.loads(l) for l in open(p)]"`).
- [ ] `HOLD_S` and `STALE_S` defaults are annotated in the source with the p95 latency from
      Task V.2 — or, if V.2 has not run, with a `TODO(V.2)` naming
      `docs/jetson-mod/cosmos_serving_measurements.md`.
- [ ] `python3 -m pytest tests/ -q` still shows the pre-existing count passing.

---

### Task V.4 — Build the Cosmos client, prompt library and parser

**AI-agent suitable:** PARTIAL — the client, parser, prompts, renderer and all offline tests
are fully agent-suitable. Capturing the transcript fixtures requires **one session on the
Jetson with the served model** (Task V.2), because fixtures must be real model output. An
agent inventing plausible transcripts defeats the purpose of the test — say so in the file
header if you ever have to stub them.

**1. Context for the implementing agent**

The VLM's job is narrow: look at one camera frame, look at the standing instruction, and emit
*one* command that is already inside the hull. The prompt is where "inside the hull" is
taught; the parser is where it is enforced a second time; the arbiter (Task V.3) is where it
is enforced a third time. Three layers, because the model is 2B parameters and INT4-quantized
and will sometimes emit prose where JSON was requested.

Read first:
- `jetson_runtime/command_contract.py` and `jetson_runtime/command_arbiter.py`.
- `docs/jetson-mod/task_plan.md` (locate by heading) — the old parser. Keep its "take the **last** JSON
  object" idea; discard its clamp values (`:2566-2575`, wrong hull) and its regex `\{[^}]+\}`
  (fails on any nested object).
- `AGENTS.md:678-682` — the HTTP request shape (`POST http://localhost:8000/v1/chat/completions`,
  base64 image + text).
- `docs/jetson-mod/known_issues.md` **PLANT-8** — this is why the prompt asks for a *bearing*,
  not a turn rate.

Depends on: Tasks V.0, V.1 and V.3. Transcript capture (step 4) additionally depends on V.2.

Traps:
- **`.gitignore:18` is `*.txt`.** Prompt files and transcripts named `*.txt` will never be
  committed and a fresh clone will have no prompts. Step 0 fixes this. This is the single
  most likely silent failure in the whole block.
- The camera is in the **head**, and the head is driven by the locomotion policy — four head
  joints (`neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`) are policy-controlled
  actuators (`robot_cfg.py:101-112`), and `joint_deviation_head` is only a **−0.1** penalty
  (`env_cfg.py:179-191`), so the head does move. A bearing read off the image is in the
  **camera** frame; the velocity command is in the **trunk** frame. Task V.6 owns the
  correction — the prompt must therefore ask for a bearing *relative to the image centre*, and
  must **not** ask the model to reason about the robot's body frame.
- At the claimed decode rate, every output token costs ~60 ms. Chain-of-thought is not free —
  budget it explicitly rather than letting `max_tokens=256` set the loop rate.

**2. Low-level implementation plan**

0. **Un-ignore the text assets first.** Append to `.gitignore`, after line 18:
   ```
   # Phase 5 text assets are source, not scratch (the blanket *.txt above would eat them)
   !jetson_runtime/prompts/*.txt
   !tests/fixtures/cosmos_transcripts/*.txt
   ```
   Verify with `git check-ignore -v jetson_runtime/prompts/navigate.txt` → **exit 1, no
   output**. If negation does not take (it will not if a parent *directory* is ignored — it is
   not, here), fall back to naming every asset `.md` instead and adjust every path below.
1. Create `jetson_runtime/cosmos_commander.py`:
   - `@dataclass(frozen=True) class VlmCommand`: `forward: float`, `lateral: float`,
     `heading_deg: float`, `behavior: str`, `raw_text: str`, `t_emit: float`.
   - `class CosmosCommander(server_url, model_id, timeout_s, max_tokens)`. `timeout_s`
     defaults to `2 × p95` from Task V.2's table — name
     `docs/jetson-mod/cosmos_serving_measurements.md` in the comment, and use 4.0 s with a
     `TODO(V.2)` if that file does not exist yet.
   - `query(frame_jpeg: bytes, instruction: str) -> VlmCommand | None` — one blocking POST to
     `/v1/chat/completions`; returns `None` on any transport failure, timeout, or non-200.
     Returning `None` is correct: the arbiter's watchdog, not this class, decides what a
     missing command means.
   - `parse(text: str) -> dict | None` — a **brace-matching scanner** (not a regex): walk the
     string, track depth (and skip braces inside string literals), collect balanced `{...}`
     spans, `json.loads` each from last to first, return the first that parses **and** contains
     a numeric `forward`. Return `None` if none does.
   - No clamping here. Clamping lives in exactly one place (`clamp_command`). This class hands
     the dict to `CommandArbiter.submit()` untouched.
2. Create `jetson_runtime/prompts/` with one template per behaviour, mirroring the behaviour
   list at `task_plan.md (locate by heading)`: `navigate`, `follow`, `avoid`, `explore`, `interact`,
   `patrol`. Author each as `<name>.txt.tmpl` and render to `<name>.txt`. Every rendered file
   must contain, in the same wording:
   - the robot's physical facts: ~42 cm tall bipedal robot, forward camera in a **moving**
     head, no arms, cannot climb or step over anything taller than ~3 cm;
   - the output contract (numbers substituted from `command_contract.py`, never typed):
     ```
     End your reply with exactly one line of JSON and nothing after it:
     {"forward": <number>, "lateral": <number>, "heading_deg": <number>, "behavior": "<word>"}
     forward: {VX_MIN} to {VX_MAX} (metres per second; positive walks forward)
     lateral: {VY_MIN} to {VY_MAX} (metres per second; positive steps left)
     heading_deg: -60 to 60 (degrees; where you want to face, measured from the centre
                  of the image; positive is to the left. This is a DIRECTION, not a
                  spin rate — say 0 when you are already facing the target.)
     behavior: one of normal, cautious, stop
     ```
   - a hard rule: *"If you are unsure, output forward 0.0, lateral 0.0, heading_deg 0.0,
     behavior stop."*
   - an explicit reasoning budget: *"Give at most one short sentence of reasoning before the
     JSON."*
3. Add `jetson_runtime/render_prompts.py`, which substitutes the bounds from
   `command_contract.py` into the `*.txt.tmpl` templates and writes the `.txt` files, plus a
   `--check` mode that re-renders in memory and exits 1 if any committed `.txt` differs.
   Document in `jetson_runtime/prompts/_schema.md` that the numeric bounds are **generated,
   not typed** — this is how the prompts cannot drift from the hull. Commit both templates and
   rendered output.
4. Capture fixtures on the Jetson (needs V.2): run 30 real queries across 6 scenes, save every
   raw `content` string to `tests/fixtures/cosmos_transcripts/`. Include, and name accordingly,
   at least one of each observed failure: prose with no JSON, JSON with a trailing comment,
   two JSON objects, a value out of range, a missing key, a truncated reply (hit `max_tokens`).
   Also save one representative JPEG as `tests/fixtures/scene_cone.jpg` for the on-Jetson
   smoke test below.
5. Record the measured decode cost per prompt in
   `docs/jetson-mod/prompt_engineering_results.md` (`AGENTS.md:816` already assigns this file
   to this task): tokens emitted, seconds, and the JSON-well-formed rate out of 30.

**3. Unit tests**

`tests/test_cosmos_parser.py`, `@pytest.mark.phase5`. No network — the HTTP layer is covered
by a stub. (`requests` is installed in the dev `python3`; verified.)

- `test_fixture_dir_is_populated` — assert `tests/fixtures/cosmos_transcripts/` exists and
  contains ≥ 30 files, **and** that `git check-ignore` does not ignore them (shell out to
  `git check-ignore -q <one file>` and assert exit 1). This is the test that catches trap 1.
- `test_parses_every_captured_transcript` — parametrized over every fixture; assert `parse()`
  returns either a dict with a numeric `forward` or `None`, and **never raises**.
- `test_extracts_last_json_object` — two objects in one string; the second wins.
- `test_handles_nested_braces` — `{"a": {"b": 1}, "forward": 0.2, ...}` parses correctly. The
  old regex fails this; it is the reason for the scanner.
- `test_handles_brace_in_string` — `{"note": "a } b", "forward": 0.2}` parses correctly.
- `test_returns_none_on_prose` — the prose fixture → `None`.
- `test_truncated_json_returns_none` — the truncated fixture → `None`, no exception.
- `test_no_clamping_here` — `parse('{"forward": 999, "lateral": 0, "heading_deg": 0,
  "behavior": "normal"}')` returns `999.0` unchanged; clamping is the arbiter's job and
  duplicating it here would hide contract violations.
- `test_query_returns_none_on_timeout` — monkeypatch `requests.post` to raise
  `requests.Timeout`; assert `None` and no exception.
- `test_query_returns_none_on_500` — stubbed 500 response → `None`.
- `test_prompts_match_contract` — for every template, re-render with `render_prompts.py` in
  the test and assert **byte equality** with the committed `.txt`. This is a regeneration
  check, not a grep for a literal.
- `test_end_to_end_transcript_to_arbiter` — for each captured transcript, feed
  `parse()` → `CommandArbiter.submit()` → `tick()`, and assert the result is finite and inside
  the hull. This is the integration assertion that matters.

**4. Smoke test**

Off-robot:
```bash
cd /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
python3 -m pytest tests/test_cosmos_parser.py -v
python3 jetson_runtime/render_prompts.py --check     # exit 1 if rendered files are stale
git check-ignore -v jetson_runtime/prompts/navigate.txt ; echo "check-ignore exit=$?"
```
Observable: pytest exits 0; `--check` exits 0; `check-ignore` prints nothing and reports
`exit=1` (meaning *not* ignored).

On the Jetson with the server up:
```bash
python3 -m jetson_runtime.cosmos_commander --image tests/fixtures/scene_cone.jpg \
        --prompt navigate --instruction "walk to the orange cone" --n 10
```
Observable: prints 10 parsed commands and a summary line
`parsed 10/10, mean_latency_s <x>, mean_tokens <y>`. Acceptance for this task is
`parsed ≥ 9/10`; below that, the prompt needs work before Task V.7 can mean anything.

**5. Done when**

- [ ] `jetson_runtime/cosmos_commander.py`, `jetson_runtime/render_prompts.py`,
      `jetson_runtime/prompts/_schema.md`, six `*.txt.tmpl` and six rendered `*.txt` exist.
- [ ] `git check-ignore -q jetson_runtime/prompts/navigate.txt` exits **1** (not ignored), and
      `git status --short jetson_runtime/prompts/` lists all twelve prompt files.
- [ ] `python3 jetson_runtime/render_prompts.py --check` exits 0.
- [ ] `tests/fixtures/cosmos_transcripts/` contains ≥ 30 real captured transcripts, including
      all six named failure shapes, and none of them is git-ignored.
- [ ] `python3 -m pytest tests/test_cosmos_parser.py -v` exits 0.
- [ ] `docs/jetson-mod/prompt_engineering_results.md` exists with a per-prompt table: tokens,
      seconds, JSON-well-formed rate out of 30.
- [ ] No clamp constant appears anywhere in `cosmos_commander.py`:
      `grep -E "0\.148|0\.111|0\.222" jetson_runtime/cosmos_commander.py` produces no output.

---

### Task V.5 — Sim replay gate: prove the command *shape* is safe before the robot sees it

**AI-agent suitable:** YES — runs entirely on the DGX Spark in Isaac Lab. No hardware, no
purchases. This is the cheapest place to answer the three open design questions from V.3.

**1. Context for the implementing agent**

Three defaults in Task V.3 are currently guesses: `apply_mode` (step vs slew), the watchdog's
zero-vs-hold fallback, and `MIN_DWELL_S`. Guessing them on a physical robot costs servos.
Guessing them in Isaac Lab costs GPU time. This task replays **real recorded VLM command
traces** through the shipped policy in simulation and measures fall rate and tracking, so
V.3's defaults become measured rather than asserted. It also produces the only pre-hardware
evidence that the VLM layer is not actively harmful.

Read first:
- `scripts/evaluate_policies.py` — `apply_condition()` at **line 1386** and its docstring,
  `run_window()` at **1424**, `fall_metrics()` at **329**, `action_smoothness_metrics()` at
  **359**, `stance_duty_metrics()` at **385**, `gait_validity()` at **411**,
  `velocity_tracking_metrics()` at **479**, `GAIT_DUTY_BAND_PCT = (40.0, 90.0)` at **167**.
  You are writing a sibling of this script; reuse its metric functions rather than rewriting
  them. It also has a `--self-test` flag that unit-tests those pure-numpy functions **without
  Isaac Sim** — run it once before you start.
- `scripts/play_policy.py` (33 lines — it `exec`s Isaac Lab's own `rsl_rl/play.py`) for the
  launcher pattern, the `PYTHONPATH` requirement, and the video flags in its docstring.
- `AGENTS.md:43-86` § "Locomotion Policy Evaluation Protocol" — the three-step protocol, the
  verdict rule (gait gate ≥ 4/5 conditions, fall rate **< 1%**, upright on video), and the
  rule that **when metrics and video disagree, the video wins**.
- `docs/jetson-mod/known_issues.md` **EVAL-2** (`:765-784`) — `AGENTS.md` mandates 5 conditions
  with a ≥ 4/5 gate; the v5 campaign actually ran **6 conditions with a ≥ 5/6 gate**, and
  rendered its turn video at `wz = 0.5` where `AGENTS.md` says 0.3. Neither is wrong; they are
  different. **State explicitly which one you ran.**

Depends on: Tasks V.0, V.1, V.3. It needs at least one command trace; if Task V.4's on-Jetson
capture has not happened, synthesize traces using the *timing* from V.2's latency table (or,
if V.2 has not run either, a declared assumed cadence) and label the results **timing-only**
in the results doc.

Traps:
- **One Isaac Sim / GPU job at a time** (`AGENTS.md:743-751`). Launching this while a training
  or another eval runs kills the second process in its init banner. Check first.
- `apply_condition()` (`:1412-1413`) sets `heading_command = False` and
  `rel_standing_envs = 0.0`. That is what you want (you supply the yaw yourself), but it also
  means the published v5d gate numbers were measured with the heading servo off — say so.
- To drive a *time-varying* command you must stop the command manager from resampling: set
  `term.cfg.heading_command = False`, `term.cfg.rel_standing_envs = 0.0`,
  `term.cfg.resampling_time_range = (1e9, 1e9)`, then write `term.vel_command_b[:, :3]`
  directly after each `env.step()`. There is a one-control-step (20 ms) lag between the write
  and the observation that carries it; document it rather than fighting it.
- **The PLAY env overrides your intent.** `OpenDuckContactWrenchEnvCfg_PLAY`
  (`env_cfg.py:789-808`) sets `scene.num_envs = 50`, `episode_length_s = 40.0`, pins the
  command ranges to `(0.2, 0, 0)`, disables corruption, pushes, mass/CoM randomization, and
  sets `contact_regime` `active_frac`/`obstacle_frac` to 0. Your `--num_envs` and
  `--episode-s` must be applied **after** `__post_init__`, exactly the way
  `evaluate_policies.py` does it, or you will silently run 50 envs for 40 s.
- **The plant changed on 2026-08-11** (PLANT-1: 3.657 kg → 2.657 kg, commit `11b1690`). Every
  published v5d number predates that. Run your own `vx = 0.2` baseline in the same session so
  your comparison is internally consistent, and label all absolute numbers as
  post-PLANT-1-fix.

**2. Low-level implementation plan**

1. Create `scripts/replay_command_trace.py`. Launcher shape identical to
   `scripts/evaluate_policies.py` (AppLauncher first, project imports after). Arguments —
   deliberately mirroring `evaluate_policies.py`'s spelling so the two are not confusable:
   `--checkpoint`, `--task` (default **`Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0`**
   — this is the exact registered id, from `isaac_lab_env/open_duck_mini_v2/__init__.py:111`
   and confirmed by `docs/jetson-mod/eval_results_v5/v5d_contact_wrench.json`'s `task_id`),
   `--trace <file.jsonl>`, `--num_envs 64`, `--episodes 10` (windows per trace),
   `--episode_length 30.0`, `--apply-mode {step,slew}`, `--fallback {zero,hold}`,
   `--min-dwell 1.0`, `--seed 42`, `--output_dir docs/jetson-mod/eval_results_vlm`.
2. Trace format — one JSON object per line, **the same schema `CommandArbiter` logs in V.3**:
   `{"t": <float seconds>, "forward": .., "lateral": .., "heading_deg": .., "behavior": ..}`.
   A line whose payload is `null` means "the VLM produced nothing at this slot" (dropout).
3. Each control step: advance a `CommandArbiter` (**imported from `jetson_runtime`, not
   reimplemented**) with the simulated clock, get `(vx, vy, wz)`, write it into
   `term.vel_command_b[:, :3]`, step the env. Using the real arbiter is what makes the result
   mean anything.
4. Record with the existing helpers from `evaluate_policies.py`: `fall_metrics`,
   `stance_duty_metrics`, `gait_validity`, `velocity_tracking_metrics`,
   `action_smoothness_metrics`. Emit one JSON per configuration into
   `docs/jetson-mod/eval_results_vlm/` — a **new** directory, deliberately not
   `eval_results_v4/` (which is `evaluate_policies.py`'s `DEFAULT_OUTPUT_DIR`, `:145`) and not
   `eval_results_v5/`; `AGENTS.md:77-79` forbids mixing robot models or protocols in one
   results dir, and this is a different protocol.
5. Sweep these configurations against the same trace set (≥ 5 traces; aim for the v5 campaign's
   scale, 10 windows × 64 envs per condition):
   | Arm | apply_mode | fallback | min_dwell | purpose |
   |---|---|---|---|---|
   | `baseline_fixed` | — | — | — | constant `vx = 0.2`, the reference point |
   | `step_1s` | step | zero | 1.0 | the proposed default |
   | `slew_1s` | slew | zero | 1.0 | does ramping help or hurt? |
   | `step_hold` | step | hold | 1.0 | tests `AGENTS.md:687`'s "not zero" claim |
   | `step_5s` | step | zero | V.0 | closer to the 10 s training dwell |
   | `step_02s` | step | zero | 0.2 | deliberately too fast — the negative control |
6. Generate the two mandated videos per surviving arm (forward `vx = 0.2` and turning) per
   `AGENTS.md:59-69`, and do the filmstrip check. Use `wz = 0.3` (the documented value) and
   note if you also render `wz = 0.5` as the v5 campaign did (EVAL-2). **The video wins over
   the metrics.** Always pass `--video` to the rollout; never drop it.
7. Write `docs/jetson-mod/vlm_replay_results.md`: one row per arm with fall rate, gait-valid
   condition count (state the denominator), stance duty L/R, `wz` error, and the video verdict.
   End with an explicit sentence naming the chosen `apply_mode`, `fallback` and `MIN_DWELL_S`.
8. Edit `jetson_runtime/command_arbiter.py` to make those three the defaults, with a comment
   citing `docs/jetson-mod/vlm_replay_results.md` and the fall-rate delta that decided it.
9. Add the run to `docs/jetson-mod/experiment_journal.md` per the Experiment Journal Protocol
   (`AGENTS.md:699-772`). This is a measured rollout, so **rule 3** applies (behavioural
   metrics come from the eval script's JSONs, not from reward curves) and **rule 4** applies
   (every number names its source: JSON field + file path). Note the journal currently ends at
   run 16 and has **zero v5 entries** (`known_issues.md` **DOC-2**) — do not assume a v5
   template exists to copy; follow the structure at `AGENTS.md:754-758`.

**3. Unit tests**

`tests/test_replay_trace.py`, `@pytest.mark.phase5`, **no Isaac Sim** (the module under test
must therefore keep its pure-Python helpers importable without the AppLauncher — put them in a
module-level block that does not import `isaaclab`, or in a small `scripts/replay_trace_core.py`
that the Isaac script imports):
- `test_trace_roundtrip` — write a trace with the arbiter's logger, read it with the replay
  loader, assert field-for-field equality.
- `test_null_line_is_dropout` — a `null` line yields no `submit()` call, so the watchdog age
  keeps growing.
- `test_replay_command_sequence_matches_arbiter` — drive the arbiter directly with a fixed
  trace and a fake clock, and drive the replay script's *command generator* (factored out of
  the Isaac loop into a pure function `commands_for_trace(trace, n_steps, cfg)`) with the same
  input; assert the two 50 Hz command sequences are identical elementwise. This guarantees the
  thing you simulate is the thing you will ship.
- `test_all_replayed_commands_inside_hull` — over all committed traces and all six arms, assert
  every generated command satisfies the hull.

Also run `python3 scripts/evaluate_policies.py --self-test` once and record that it passes —
it exercises the metric functions you are reusing, without Isaac.

**4. Smoke test**

```bash
# 1. confirm no other Isaac job is live
pgrep -af "isaac|python.*train_ppo|evaluate_policies|play_policy" || echo "GPU free"

# 2. create the fixture trace this smoke test needs (it does not exist yet)
mkdir -p /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/tests/fixtures/vlm_traces
python3 -m jetson_runtime.command_arbiter --demo \
  --emit-trace /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/tests/fixtures/vlm_traces/cone_approach.jsonl

# 3. run one window. NOTE: all repo paths must be ABSOLUTE — cwd is ~/IsaacLab.
cd ~/IsaacLab && PYTHONPATH=/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson:$PYTHONPATH \
  ./isaaclab.sh -p /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/scripts/replay_command_trace.py \
  --checkpoint /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/exported_policies/v5d_contact_wrench_ppo/model_5998.pt \
  --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 \
  --trace /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/tests/fixtures/vlm_traces/cone_approach.jsonl \
  --apply-mode step --fallback zero --min-dwell 1.0 \
  --num_envs 64 --episodes 1 --episode_length 30 --seed 42 --headless \
  --output_dir /tmp/vlm_smoke
```
Observable: `/tmp/vlm_smoke/*.json` exists, its `per_condition` entry reports
`"episodes": 64` (proving your `--num_envs 64` beat the PLAY cfg's hard-coded 50) and a
`"fall_rate_pct"` field, and the process exits 0.

Gate for the full sweep: the chosen arm's fall rate is **< 1%** (`AGENTS.md:71`) and **not
worse than `baseline_fixed` by more than 1 percentage point**.

**5. Done when**

- [ ] `scripts/replay_command_trace.py` exists and runs headless to completion.
- [ ] `docs/jetson-mod/eval_results_vlm/` contains one JSON per arm; each JSON's `protocol`
      block states its own conditions, windows, envs and episode length, and the totals match.
- [ ] `docs/jetson-mod/vlm_replay_results.md` has a six-row table plus a video verdict per
      surviving arm, and states explicitly whether it ran the `AGENTS.md` 5-condition protocol
      or the v5 6-condition one (EVAL-2).
- [ ] The results doc names the chosen `apply_mode`, `fallback` and `MIN_DWELL_S` in one
      explicit sentence, with the fall-rate numbers that decided each.
- [ ] `command_arbiter.py`'s defaults match that sentence and cite the results file.
- [ ] Chosen arm: fall rate **< 1%**, gait-valid on ≥ 5 of 6 (or ≥ 4 of 5) conditions
      depending on the protocol you declared, and the turning video shows both feet swinging
      (no drag).
- [ ] All absolute numbers are labelled **post-PLANT-1-fix** and are not compared to the
      pre-fix v5d table.
- [ ] An entry exists in `docs/jetson-mod/experiment_journal.md` naming every source.
- [ ] `python3 -m pytest tests/test_replay_trace.py -v` exits 0.

---

### Task V.6 — On-robot integration: two processes, head-frame correction, thermal, LEDs

**AI-agent suitable:** PARTIAL — an agent writes all the code and all the tests; a **human
must be present** for every run, holding or spotting the robot, because this is the first time
the VLM commands real servos. The head-frame sign calibration (step 6) requires physically
placing an object, yawing the head, and judging the resulting motion.

**1. Context for the implementing agent**

This is where the pieces meet the hardware. The most important decision in the task is
structural and is a **deliberate departure from `task_plan.md (locate by heading)`**, which puts Cosmos,
locomotion and an optional behaviour-effects loop in **three threads of one Python process**
(`:2714`, `:2720`, `:2727`, started at `:2787-2788`). Do not do that. One process means a vLLM
client exception, a `requests` bug, a camera driver hang or an OOM kill takes the 50 Hz control
loop down with it, mid-stride, with 14 servos holding torque.

Read first:
- `jetson_runtime/command_arbiter.py`, `cosmos_commander.py`, `command_contract.py`.
- `docs/jetson-mod/cosmos_serving_measurements.md` (Task V.2) — timeout and watchdog constants
  and the power-mode verdict.
- `docs/jetson-mod/task_plan.md` (locate by heading) — `ThermalManager` and its
  `request_boost()` / `release_boost()` contract (`:2334-2355`). **The file
  `jetson_runtime/thermal_manager.py` is a Phase-4 deliverable and may not exist yet.**
- `AGENTS.md:635-649` — the GPIO map. `AGENTS.md:649` warns: **verify against the actual Orin
  Nano dev-kit carrier-board pinout before wiring**, and use `Jetson.GPIO`, not `RPi.GPIO`.
- `docs/jetson-mod/known_issues.md` **DEPLOY-3** (antenna dims) and **DEPLOY-1**
  (`q_target = q_default + 0.25·a`; also **DEPLOY-2** if you hand-roll the normalizer rather
  than using the exported ONNX/TRT engine).
- `AGENTS.md:441-446` — the deployment trap: the obs joint block is `joint_pos_rel`
  (`q − q_default`), not raw encoder angles.

Depends on: Tasks V.0–V.5 and Phase 4 (a robot that walks under teleop).

Traps:
- **The antennas are contested.** The locomotion policy already emits antenna targets on
  action dims `[13, 14]`, and a "behaviour" layer that also drives the antenna PWM pins
  (BCM 12/13) will fight it. Worse, obs dims `[22, 23, 38, 39]` are the antennas' and are
  **unmeasurable** (DEPLOY-3) — whatever constant Phase 4 chose to fill them with must stay
  constant. A behaviour flag must never change those four numbers.
- **The camera is on a moving head.** Four head joints are policy-controlled. A bearing read
  from the image is in the camera frame; `heading_deg` must be corrected by the measured head
  yaw before it reaches the arbiter.
- **`head_yaw` is joint index 7, not 13 or 14, and the joint order is interleaved, not
  grouped**: `scripts/duck_init_pos.json` `joint_order` is
  `[left_hip_yaw, neck_pitch, right_hip_yaw, left_hip_roll, head_pitch, right_hip_roll,
  left_hip_pitch, head_yaw, right_hip_pitch, left_knee, head_roll, right_knee, left_ankle,
  left_antenna, right_antenna, right_ankle]`. Anyone assuming legs-then-head will read the
  wrong servo. Use `command_contract.HEAD_YAW_JOINT_IDX`, never a literal.
- The gait-phase counter must free-run at 27 steps regardless of anything the VLM does.
- If Task V.2's verdict was "pin one power mode", do not implement boost duty-cycling.

**2. Low-level implementation plan**

1. Create `jetson_runtime/command_bus.py` — a single-slot lock-free channel over
   `multiprocessing.shared_memory`, 128 bytes:
   `seq(uint64) | t_emit(float64) | vx | vy | heading_deg (float64) | behavior(uint8) | pad`.
   Writer: `seq += 1` (now odd) → write payload → `seq += 1` (now even). Reader: read `seq`,
   read payload, re-read `seq`; retry if it changed or is odd. Never blocks the reader.
2. Create `jetson_runtime/vlm_node.py` — its own process. Loop: capture a frame from the
   IMX219 (GStreamer `nvarguscamerasrc`, 640×480, JPEG, longest side capped per V.2), read the
   standing instruction from `~/duck_logs/instruction.txt` (so a human can change it live),
   call `CosmosCommander.query()`, and on success write to the bus. On failure write nothing —
   silence is the signal, and the watchdog already understands it.
3. Create `jetson_runtime/locomotion_node.py` — the safety-critical process:
   - `mlockall(MCL_CURRENT|MCL_FUTURE)` via `ctypes`; set `oom_score_adj = -500`; launched
     under `chrt -r 20`.
   - Owns servos, IMU, TRT engine, and the free-running 27-step gait counter.
   - **Starts and runs with no VLM present** — if the bus is empty it stands (a zero command;
     `env.yaml:986` shows 2% of training envs were standing envs).
   - Each step: read the bus, `arbiter.submit()` on a new `seq`, `arbiter.tick()`, write the
     result into `obs[6:9]`, build the rest of the obs vector per `command_contract.OBS_SLICES`
     (remembering `joint_pos_rel = q − q_default` from `load_q_default()`), infer, apply
     `q_target = q_default + ACTION_SCALE·a`, send.
   - Emergency stop: IMU tilt > 60° cuts torque (`AGENTS.md:688`; it is also the policy's own
     termination angle, `env_cfg.py:591-593`).
4. Create `jetson_runtime/autonomous_walk.py` — a thin supervisor that starts both nodes as
   subprocesses, restarts `vlm_node` on exit with exponential backoff, and **never** restarts
   `locomotion_node` automatically (a crashed control loop is a stop-and-inspect event).
5. Head-frame correction, in `locomotion_node` before `submit()`:
   `heading_trunk = wrap_to_pi(heading_cam + HEAD_YAW_SIGN · head_yaw_measured)`, where
   `head_yaw_measured` is read from the `head_yaw` STS3250's position feedback — joint index
   `command_contract.HEAD_YAW_JOINT_IDX` (7). Only the two SG90 antennas lack feedback
   (DEPLOY-3); the other 14 servos are Feetech STS3250 with position feedback.
   `HEAD_YAW_SIGN` lives in `command_contract.py` as a calibration constant, set by step 6.
6. **Sign calibration (human required).** Place a cone straight ahead. Command the head to yaw
   +30° while the trunk is stationary. Log `heading_cam` (from the VLM) and `heading_trunk`
   (computed). Correct sign ⇒ `heading_trunk` stays near 0 while `heading_cam` swings by ~30°.
   If `heading_trunk` swings instead, flip `HEAD_YAW_SIGN`. Record both traces.
7. Antenna arbitration — pick **one** and write it into `jetson_runtime/README.md`:
   **(a)** the locomotion policy owns the antennas; the behaviour flag is expressed only
   through the eye LEDs and the speed style; or **(b)** the behaviour layer owns them and the
   policy's action dims `[13, 14]` are discarded. Default to **(a)** — it changes nothing about
   the trained action pipeline. Under either choice, obs dims `[22, 23, 38, 39]` keep the
   Phase-4 constant, always.
8. LED status on BCM 23/24 (pins 16/18): solid = `FRESH`, 2 Hz blink = `STALE`, off = `LOST`,
   fast double-blink = e-stop. This makes the watchdog observable from across the room during
   the Task V.7 trials. Use `Jetson.GPIO` (`AGENTS.md:649`).
9. Thermal: if V.2's verdict was "duty-cycle", wrap each `query()` in
   `request_boost()` / `release_boost()` per `task_plan.md (locate by heading)`; the boost request lives
   in `vlm_node`, **never** in `locomotion_node`.

**3. Unit tests**

`tests/test_command_bus.py` and `tests/test_locomotion_node_logic.py`, `@pytest.mark.phase5`,
off-hardware (all hardware behind an injected interface with a fake implementation):
- `test_bus_roundtrip` — write then read; fields match to 1e-12.
- `test_bus_torn_read_retries` — force `seq` odd mid-read; assert the reader retries and never
  returns a torn payload.
- `test_bus_reader_survives_dead_writer` — kill the writer process; the reader keeps returning
  the last `seq` and never raises, so the arbiter's watchdog can do its job.
- `test_locomotion_runs_with_no_vlm` — no writer ever attaches; assert 500 steps produce a
  `(0,0,0)` command and no exception.
- `test_head_frame_correction` — with `head_yaw = +0.5236` rad (30°) and `heading_cam = −30°`,
  assert `heading_trunk ≈ 0` for `HEAD_YAW_SIGN = +1`.
- `test_head_yaw_index_from_contract` — assert the code reads joint index
  `HEAD_YAW_JOINT_IDX` and that `load_q_default()[0][HEAD_YAW_JOINT_IDX] == "head_yaw"`.
  This catches the interleaved-order trap.
- `test_behavior_flag_does_not_touch_antenna_obs` — build the observation vector under all four
  behaviour flags; assert dims `[22, 23, 38, 39]` are bitwise identical across all four.
  **This is the DEPLOY-3 regression test.**
- `test_gait_phase_free_runs` — 270 steps with the command changing every step; assert the
  phase counter visits each of 27 values exactly 10 times.
- `test_action_pipeline_applies_scale_and_offset` — feed a known action vector; assert the
  commanded joint target equals `q_default + 0.25·a` elementwise, with `q_default` taken from
  `load_q_default()` and in `joint_order`. **This is the DEPLOY-1 regression test.**
- `test_obs_joint_block_is_relative` — feed known absolute encoder angles; assert
  `obs[9:25] == q_abs − q_default`, not `q_abs`. This is the `AGENTS.md:441-446` trap.

**4. Smoke test**

On the robot, on a stand, with a human holding it, and the servo bus powered:
```bash
python3 jetson_runtime/autonomous_walk.py --instruction "stand still and look around"
# in another shell:
kill -9 $(pgrep -f vlm_node)
```
Observables:
1. Before the kill: the arbiter `.jsonl` shows `state: FRESH` and the eye LEDs are solid.
2. Within `STALE_S + 0.2 s` of the kill: the LEDs blink, the log shows
   `FRESH → STALE → LOST`, and the robot **stands** — it does not lurch, does not keep its last
   command, and does not fall.
3. `locomotion_node`'s PID is unchanged throughout (`ps -o pid,etimes -p <pid>`), proving the
   crash was contained.
4. The supervisor restarts `vlm_node` and the state returns to `FRESH` after two submissions.

**5. Done when**

- [ ] `command_bus.py`, `vlm_node.py`, `locomotion_node.py`, `autonomous_walk.py` exist.
- [ ] `pgrep -c -f "vlm_node|locomotion_node"` returns `2` while running — they are two OS
      processes, not two threads.
- [ ] `kill -9` on `vlm_node` leaves `locomotion_node`'s PID unchanged and the robot standing;
      the transcript is in the log.
- [ ] `HEAD_YAW_SIGN` in `command_contract.py` is set by the step-6 measurement (no longer the
      placeholder), and the measured `heading_trunk` drift over a ±30° head sweep is recorded
      (target: < 5°).
- [ ] `jetson_runtime/README.md` names the antenna-ownership choice.
- [ ] `test_behavior_flag_does_not_touch_antenna_obs`,
      `test_action_pipeline_applies_scale_and_offset` and `test_obs_joint_block_is_relative`
      all pass.
- [ ] `docs/jetson-mod/validation_results.md` gains a Phase-5 integration section with the
      measured 50 Hz loop period p50/p95/max **while the VLM is querying**
      (requirement: p95 ≤ 25 ms, i.e. ≥ 40 Hz sustained).

---

### Task V.7 — Evaluate whether the VLM layer actually helps

**AI-agent suitable:** NO — an agent writes the analysis script and can score the logs, but the
trials themselves require a human to build a 3 × 3 m arena, place a target, set the robot on
six marked start poses, spot it against falls, and judge each trial's outcome. Video judgement
and intervention calls are human decisions. Budget: ~4 hours of a person's time plus a cone
and floor tape.

**1. Context for the implementing agent**

"The VLM works" is not a claim this project may make from a demo video. Every prior verdict in
this repo is a measured gate with a pre-registered threshold, and one of them — `amp_v4`,
run 12 — **passed every aggregate metric while crawling** (`AGENTS.md:72-73`;
`experiment_journal.md:398`). The VLM layer will look impressive long before it is useful,
because a model that emits `forward: 0.2` every second regardless of the image produces a robot
that walks confidently in one direction. The only way to tell that apart from vision-driven
navigation is a **blind control arm**.

This is also where the temptation to reach for the *Duck Embody* benchmark appears. Resist it:
that benchmark is a different repo, a simulated apartment, and an LLM-as-SLAM abstraction. Its
numbers do not transfer and must not be cited here.

Read first:
- `AGENTS.md:43-86` § "Locomotion Policy Evaluation Protocol" — the three-step shape (metrics,
  video audit, journal) and the rule that the video wins.
- `docs/jetson-mod/vlm_replay_results.md` (Task V.5) — the sim result this must be consistent
  with.
- `jetson_runtime/command_arbiter.py` `status()` — the telemetry dict this task consumes.

Depends on: Tasks V.5 and V.6.

Trap: pre-register the criterion **before** the first trial, in the results file, and commit it
(this is the one place in Phase 5 where committing is the point — ask the owner for permission
to commit, per the no-auto-commit rule). A threshold chosen after seeing the data is not a
threshold.

**2. Low-level implementation plan**

1. Write `docs/jetson-mod/vlm_eval_protocol.md` **first**, and commit it before any trial:
   - **Arena:** 3 × 3 m flat floor, one orange cone as the target. Six start marks taped down,
     each ≥ 1.5 m from the cone and ≥ 30° off the robot's initial facing (so walking blindly
     straight cannot succeed).
   - **Arms**, 20 trials each, start marks cycled in a fixed order:
     | Arm | Description |
     |---|---|
     | **A** blind | no VLM; constant `vx = 0.15`, `wz = 0` |
     | **B** VLM | full stack, `yaw_mode="heading"` |
     | **B′** VLM, no yaw | full stack with `heading_deg` forced to 0 — isolates whether the turn channel is what helps |
     | **C** teleop | a human drives, through the same arbiter — the practical upper bound |
   - **Success:** the trunk comes within 0.5 m of the cone within 60 s, with no fall and no
     human intervention.
   - **Pre-registered criterion — the VLM layer helps iff all four hold:**
     1. `success(B) − success(A) ≥ 0.35`, one-sided Fisher exact `p < 0.05`. (Verified with
        `scipy.stats.fisher_exact(..., alternative="greater")` on this machine: `14/20` vs
        `4/20` → `p = 0.00182`; `11/20` vs `4/20` → `p = 0.0242`; `10/20` vs `9/20` →
        `p = 0.50`. So a Δ of 0.35 at these counts clears p < 0.05, and the two gates are not
        redundant.)
     2. `falls(B) ≤ falls(C) + 2`;
     3. **zero** out-of-hull commands in the arbiter logs across all 80 trials;
     4. clamp rate in B < 30% of accepted commands (a higher rate means the prompt, not the
        robot, is doing the limiting — a Task V.4 defect).
   - **Secondary telemetry**, reported but not gating: p95 VLM latency, decisions per trial,
     watchdog trips per minute, mean command age at consumption, `B′` vs `B` delta.
2. Write `scripts/score_vlm_trials.py`: reads the arbiter `.jsonl` logs plus a `trials.csv`
   filled in by the human (`trial_id, arm, start_mark, outcome, fall, intervention,
   t_to_goal_s, notes`), and emits the four gate values, the Fisher exact p-value
   (`scipy.stats.fisher_exact(..., alternative="greater")` — `scipy` 1.17.0 is installed), and
   the telemetry table as markdown. Exit non-zero if any arm has fewer than the declared
   number of trials.
3. Run the 80 trials. Record video of every trial; keep them under
   `docs/jetson-mod/vlm_eval_videos/`.
4. Write `docs/jetson-mod/vlm_eval_results.md`: the pre-registered criterion quoted verbatim
   from the protocol file, the four gate values, the verdict, the telemetry table, the failure
   taxonomy (how many failures were perception, how many latency, how many locomotion), and
   links to the videos.
5. Update `docs/jetson-mod/prompt_engineering_results.md` with per-prompt success rates out of
   10, as `task_plan.md (locate by heading)` already asks.
6. Add an entry to `docs/jetson-mod/experiment_journal.md`. Every number names its source
   (rule 4). Note the journal has zero v5 entries (`known_issues.md` **DOC-2**) — you may be
   the first to add a post-v4 entry; follow the structure at `AGENTS.md:754-758`.
7. Update `AGENTS.md`'s Phase Summary table (`:788`) with the verdict, and its "Current Phase"
   line in § Quick Reference (`AGENTS.md:14-24`) if it still says Phase 4.

**3. Unit tests**

`tests/test_score_vlm_trials.py`, `@pytest.mark.phase5` — the *scoring* is testable even though
the trials are not:
- `test_fisher_known_table` — `14/20 vs 4/20` yields `p < 0.01`; `10/20 vs 9/20` yields
  `p > 0.3`. Guards against a one-sided/two-sided mix-up. (Expected exact values: 0.00182 and
  0.50.)
- `test_gate_all_four_required` — a synthetic input passing three gates and failing the
  clamp-rate gate must produce verdict `FAIL`.
- `test_out_of_hull_detection` — inject one log line with `vx = 0.5`; assert the script reports
  exactly 1 violation and fails gate 3.
- `test_missing_trial_rows_error` — a `trials.csv` with 19 rows for an arm errors out rather
  than silently scoring 19; incomplete arms must not be reported.
- `test_telemetry_percentiles` — synthetic latency list; p95 matches `numpy.percentile`.

**4. Smoke test**

Before the real trials, run the scorer on a two-trial dry run:
```bash
cd /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
python3 scripts/score_vlm_trials.py --logs ~/duck_logs \
        --trials docs/jetson-mod/trials_dryrun.csv --out /tmp/dryrun.md
```
Observable: `/tmp/dryrun.md` contains all four gate values and a `VERDICT:` line, and the
script exits 0. Then delete one row from `trials_dryrun.csv` and re-run: it must exit non-zero
and print which arm is short and by how many.

**5. Done when**

- [ ] `docs/jetson-mod/vlm_eval_protocol.md` was committed **before** the first trial
      (verifiable: `git log --format=%cI -1 -- docs/jetson-mod/vlm_eval_protocol.md` predates
      `git log --format=%cI -1 -- docs/jetson-mod/trials.csv`).
- [ ] 80 trials recorded, 20 per arm, with video for each under
      `docs/jetson-mod/vlm_eval_videos/`.
- [ ] `docs/jetson-mod/vlm_eval_results.md` reports all four pre-registered gate values and a
      one-word verdict.
- [ ] Gate 3 (zero out-of-hull commands across all 80 trials) is met — this one is not
      negotiable regardless of the verdict on the others.
- [ ] A failure taxonomy is present, attributing every failed trial to perception, latency, or
      locomotion.
- [ ] `docs/jetson-mod/prompt_engineering_results.md` has per-prompt success rates.
- [ ] An experiment-journal entry exists naming every source.
- [ ] `AGENTS.md`'s Phase Summary table shows Phase 5's real status.
- [ ] The results file contains an explicit sentence stating that these numbers are from the
      on-robot VLM layer and are **not** comparable to the Duck Embody benchmark.
- [ ] `python3 -m pytest tests/test_score_vlm_trials.py -v` exits 0.

---

## Cross-cutting notes for whoever implements any of these

1. **Run tests as `python3 -m pytest tests/ -v`**, never a bare `pytest` from the repo root — it
   fails at collection (`known_issues.md` **TEST-2**). The `phase5` marker is already registered
   in `pytest.ini`, and `AGENTS.md:41` documents `pytest -m "phase5" -v`.
2. **Measured baselines on 2026-08-11, before any Phase-5 work** — record yours before you
   start and compare against that, not against these:
   - `python3 -m pytest tests/ -q` → `103 passed in ~2.4 s`
   - `python3 scripts/verify_known_issues.py` → `CONFIRMED 28 / 29`, `DEPLOY-1` INCONCLUSIVE,
     **exit 2** (exit 2 means "could not evaluate", not "failed" — `onnx` is absent from the
     system interpreter). The register header's "33 / 34" and TEST-2's "101 passed" are stale.
3. **Do not add grep-style tests** (`known_issues.md` **TEST-1**). Every test above parses or
   executes.
4. **`.gitignore:18` is `*.txt` and `:19` is `*.onnx`.** Check every new text asset with
   `git check-ignore -v <path>` before declaring a task done.
5. **Every gate number quoted in this block predates the PLANT-1 fix** (3.657 kg → 2.657 kg,
   commit `11b1690`, 2026-08-11). The shipped v5d policy was trained on the heavy plant. If a
   re-gate of v5d on the corrected plant happens before Phase 5 runs, re-read the hull from the
   *new* policy's `env.yaml` and update `jetson_runtime/command_contract.py` — that is the one
   file that has to change, which is the point of Task V.1.
6. **Isaac jobs serialize.** One Isaac Sim / GPU job at a time; check before launching Task V.5.
7. **Always pass `--video`** to any rollout you render; never drop it to save time.
8. **Do not commit or push** unless the owner explicitly asks. The single exception worth asking
   about is Task V.7's protocol file, whose evidential value depends on being committed before
   the trials.

---

# Appendix A — Verified facts, with the command that produces each

Every row was produced by running the command on 2026-08-11. If you are about to
write a number into a document or a config, get it from here or re-run the
command. **Do not carry a number over from a summary.** That failure has already
happened twice in this project's history.

| Fact | Value | Command |
|---|---|---|
| Plant mass, MJCF and PhysX agree | 2.657067 kg | `cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/audit_plant_mass.py --headless` (exit 0) |
| Articulation root | `trunk_assembly` | same |
| Rigid bodies | 21 | same |
| Obs / action dims | 59 / 16 | build any env; `obs["policy"].shape[-1]`, `env.action_manager.total_action_dim` |
| Trained command hull | vx (−0.148, 0.222), vy (−0.111, 0.111), wz (−0.5, 0.5) | `grep -n "ranges.lin_vel_x = (\|ranges.lin_vel_y = (\|ranges.ang_vel_z = (" isaac_lab_env/open_duck_mini_v2/env_cfg.py` (lines ~261-263) |
| Gait gate band | stance duty ∈ [40, 90] % | `GAIT_DUTY_BAND_PCT`, `scripts/evaluate_policies.py:167` |
| Eval default output dir | `docs/jetson-mod/eval_results_v4` | `DEFAULT_OUTPUT_DIR`, same file line ~145 — **you must override this for a new plant** |
| Antenna joint indices | 13, 14 → obs 22, 23, 38, 39 | `known_issues.md` DEPLOY-3 |
| Antenna armature error | 0.040 vs link inertia 3.87e-06 = **10,336×** | `python3 scripts/verify_known_issues.py PLANT-4` |
| Printed set | 52 pieces, 1571.94 cm³ | `python3 scripts/measure_print_mass.py --process mjf-pa12` |
| Printed mass, FDM PLA 2p/15 % | 1158 g | `--process fdm-pla --perimeters 2 --infill 15` |
| Printed mass, MJF PA12 | 1598 g | `--process mjf-pa12` |
| CAD-mod delta, measured vs booked | −6.60 g vs −88.48 g | `python3 scripts/measure_print_mass.py --cad-delta --sweep` |
| Standing CoM above sole | 203.1 mm | FK over all collision-mesh vertices at the `robot_cfg.py` pose |
| Lowest collision vertex, nominal | +3.17 mm | same |
| Register status | 28/29 CONFIRMED under `python3` (**exit 2** — DEPLOY-1 needs `onnx`); 29/29 and **exit 0** under `~/IsaacLab/_isaac_sim/python.sh` | `python3 scripts/verify_known_issues.py` |
| Test suite | 103 passing | `python3 -m pytest tests/ -q` |
| Jetson memory budget | ~7.7 GB of 8 GB | `AGENTS.md` "Memory Budget on Jetson" — only ~0.3 GB headroom |

# Appendix B — Traps that have already caused wrong answers here

These are not hypothetical. Each one produced a confident, wrong result during
the audit that led to this plan.

1. **A term's name and the body it binds routinely disagree.** `add_base_mass`
   binds `trunk_assembly`. The `head` contact term binds a 1e-09 kg marker frame,
   not `head_assembly`, so it measured 0.000000 N while the feet saw 821 N.
   Always resolve against `robot.data.body_names`.
2. **Never judge an eval task from its `EnvCfg` alone.** `evaluate_policies.py`
   overrides the command term per condition, so a config that reads
   `ang_vel_z=(0,0)` still evaluates turning. This produced a false "the gate
   never tests rotation" finding.
3. **Measure the quantity the claim is about.** "Resets start below ground" is
   about the lowest *collision-mesh vertex*, not the body origin — a foot's
   origin sits ~3 mm above its sole. Measuring origins gave a confident false
   negative on a real defect.
4. **MuJoCo re-centres mesh vertices at compile time** (`mesh_pos`, `mesh_quat`).
   Transforming raw STL vertices by `geom_xpos`/`geom_xmat` produces a
   plausible-looking but wrong assembly. Use `model.mesh_vert`.
5. **Every mesh is instantiated twice** — a collision geom and a visual twin at
   the same pose. 133 unique instances, 130 twins. Counting both doubles every
   mass.
6. **Count pieces, not files.** `print_guide.md` has ×2/×4 on nine rows; summing
   the 37 distinct STL files undercounts the robot by 13 %.
7. **Infill is not a mass fraction.** A slicer lays solid perimeters and skins,
   so 15 % infill gives ~0.5–1.0 of solid depending on wall thickness, not 0.15.
   And for MJF/SLS there is no infill at all — parts are solid.
8. **`tests/` passes through real regressions.** 5 of 7 seeded mutations survive
   the suite. A green suite is not evidence your change is correct.
9. **PhysX substitutes silently.** Unauthored mass became 1.000 kg; zero inertia
   became 4.0e-12. Nothing on disk looked wrong. If a value matters, read it back
   from `robot.data.default_mass` / `default_inertia` at runtime.

# Appendix C — What "done" looks like for the whole rebuild

The rebuild is complete when all of the following hold simultaneously:

- [ ] `scripts/audit_plant_mass.py` exits 0 and the mass it reports equals the
      bottom-up BOM figure for the **chosen print process**, not for a
      placeholder
- [ ] `~/IsaacLab/_isaac_sim/python.sh scripts/verify_known_issues.py` exits 0,
      or every remaining CONFIRMED issue is one the owner has explicitly accepted.
      **Use the Isaac interpreter for this check** — under plain `python3` the
      script exits **2**, because the DEPLOY-1 ONNX checks need the `onnx`
      module and report INCONCLUSIVE without it. Exit 2 is not a failure.
- [ ] A policy has been trained on the corrected plant and passes the full
      three-part protocol (gait gate ≥ 4/5, fall rate < 1 %, **and** the video
      audit — when metrics and video disagree, the video wins)
- [ ] Its results live in a new per-model results directory, and no table mixes
      models
- [ ] `docs/jetson-mod/experiment_journal.md` has an entry per run
- [ ] The ONNX round-trips through the documented deployment contract, including
      `action_scale` and `q_default`, which the ONNX itself omits
- [ ] Either the antennas are out of the action/observation spaces, or there is a
      written decision for how the 4 unmeasurable dims are supplied on hardware
- [ ] The robot stands, unassisted, on hardware
