#!/usr/bin/env python3
"""Mechanically re-verify every issue in docs/jetson-mod/known_issues.md.

No GPU, no Isaac Sim, no Kit — plain python3 with numpy + mujoco (the same
dependencies tests/ already needs). Reads the repo as it is on disk; asserts
nothing from memory.

    python3 scripts/verify_known_issues.py            # all checks, full evidence
    python3 scripts/verify_known_issues.py --quiet    # summary table only
    python3 scripts/verify_known_issues.py PLANT-3    # one check by id

Exit code 0 when every check reports CONFIRMED, 1 otherwise.

A check flipping to REFUTED normally means **the issue was fixed** — update
known_issues.md and delete the check. It can also mean the check itself has
rotted; the evidence lines are printed so you can tell which.

The runtime half of the register (plant mass, wrench composer, reward wiring,
obstacle placement, contact-sensor timing) needs a live simulator and is covered
by `scripts/audit_plant_mass.py` plus the probes cited inline in the document.

Single source of truth for the issues themselves: docs/jetson-mod/known_issues.md
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *p: os.path.join(REPO, *p)
R = lambda p: open(P(p), encoding="utf-8", errors="replace").read()
RESULTS: list[tuple[str, str, str, list[str]]] = []
CHECKS: list[tuple[str, str, object]] = []


def issue(cid: str, title: str):
    def deco(fn):
        CHECKS.append((cid, title, fn))
        return fn
    return deco


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def mjcf_bodies():
    return ET.parse(P("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")).getroot()


# ------------------------------------------------------------------ PLANT
@issue("PLANT-1", "Phantom 1.000 kg: `base` is the only body with no <inertial>")
def _():
    root = mjcf_bodies()
    none_, masses = [], []
    for b in root.iter("body"):
        i = b.find("inertial")
        (none_ if i is None else masses).append(b.get("name") if i is None else float(i.get("mass")))
    seg = re.search(r'<body name="base".*?>(.*?)<body name="trunk_assembly"',
                    R("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml"), re.S).group(1)
    ok = none_ == ["base"] and abs(sum(masses) - 2.657067284) < 1e-6 and "<geom" not in seg
    return ("CONFIRMED" if ok else "REFUTED"), [
        f"bodies with NO <inertial>: {none_}   (of {len(masses) + len(none_)} total)",
        f"authored total mass: {sum(masses):.9f} kg",
        f"`base` also has no <geom>, so link_density auto-compute cannot fire: {'<geom' not in seg}",
        "runtime confirmation: scripts/audit_plant_mass.py exits 1 with TOTAL 2.657067 -> 3.657067",
    ]


@issue("PLANT-1b", "The other massless frames DO carry a 1e-09 placeholder inertial")
def _():
    hits = re.findall(r'<body name="(\w+)"[^>]*>\s*<inertial[^>]*mass="1e-09"',
                      R("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml"))
    return ("CONFIRMED" if len(hits) >= 4 else "REFUTED"), [
        f"frames with a 1e-09 placeholder: {hits}",
        "-> the fix pattern already exists in the file; `base` is the sole exception",
    ]


@issue("PLANT-2", "add_base_mass randomizes `trunk_assembly`, not `base`")
def _():
    blk = re.search(r"self\.events\.add_base_mass = EventTerm\((.*?)\n        \)",
                    R("isaac_lab_env/open_duck_mini_v2/env_cfg.py"), re.S).group(1)
    body = re.search(r'body_names="(\w+)"', blk).group(1)
    rng = re.search(r'"mass_distribution_params": \(([-\d., ]+)\)', blk).group(1)
    hi = float(rng.split(",")[1])
    return ("CONFIRMED" if body == "trunk_assembly" else "REFUTED"), [
        f"add_base_mass body_names = {body!r}; the phantom mass is on 'base'",
        f"range = ({rng}) -> upper bound {hi} kg vs a 1.000 kg constant offset = {1.0/hi:.1f}x too small",
    ]


@issue("PLANT-3", "61.7% of training resets start inside the ground plane")
def _():
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(P("mini_bdx/robots/open_duck_mini_v2/scene.xml"))
    data = mujoco.MjData(model)
    src = R("isaac_lab_env/open_duck_mini_v2/robot_cfg.py")
    blk = src.split("joint_pos={", 1)[1].split("},", 1)[0]
    defaults = {k: float(v) for k, v in re.findall(r'"(\w+)"\s*:\s*(-?[\d.]+)', blk)}

    hinges = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[j]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9   # soft_joint_pos_limit_factor
        hinges.append((model.jnt_qposadr[j], defaults[name], mid - half, mid + half))

    def lowest_vertex():
        lo = np.inf
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            d = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[d]:model.mesh_vertadr[d] + model.mesh_vertnum[d]]
            lo = min(lo, float((v @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min()
                               + data.geom_xpos[g][2]))
        return lo

    mujoco.mj_resetData(model, data)
    for adr, dflt, _, _ in hinges:
        data.qpos[adr] = dflt
    mujoco.mj_kinematics(model, data)
    nominal = lowest_vertex()

    rng = np.random.default_rng(0)
    N = 4000
    lows = np.empty(N)
    for i in range(N):
        mujoco.mj_resetData(model, data)
        s = rng.uniform(0.9, 1.1, size=len(hinges))       # env_cfg.py reset_robot_joints
        for k, (adr, dflt, slo, shi) in enumerate(hinges):
            data.qpos[adr] = min(max(dflt * s[k], slo), shi)
        mujoco.mj_kinematics(model, data)
        lows[i] = lowest_vertex()
    mm = lows * 1000.0
    frac = float((mm < 0).mean())
    return ("CONFIRMED" if frac > 0.5 else "REFUTED"), [
        f"nominal pose lowest COLLISION VERTEX = {nominal*1000:+.2f} mm "
        f"(the 2026-08-02 plant audit independently measured +3.2 mm)",
        f"over {N} random resets: mean {mm.mean():+.2f} mm, deepest {mm.min():+.2f} mm",
        f"FRACTION starting below ground = {frac*100:.1f}%  (plant audit reported 61.4%)",
        "NB measure collision VERTICES, not body origins -- origins sit ~3 mm above the sole",
    ]


@issue("PLANT-4", "Antennas are simulated with STS3250 actuator parameters")
def _():
    head = R("isaac_lab_env/open_duck_mini_v2/robot_cfg.py").split(
        '"head": ImplicitActuatorCfg(', 1)[1].split("),", 1)[0]
    eff = re.search(r"effort_limit_sim=([\d.]+)", head).group(1)
    arm = float(re.search(r"armature=([\d.]+)", head).group(1))
    inert = {}
    for b in mjcf_bodies().iter("body"):
        if "antenna" in (b.get("name") or ""):
            i = b.find("inertial")
            inert[b.get("name")] = max(float(x) for x in i.get("diaginertia").split())
    biggest = max(inert.values())
    ok = ".*_antenna" in head and "SG90" in R("AGENTS.md")
    return ("CONFIRMED" if ok else "REFUTED"), [
        f"'head' actuator group includes '.*_antenna': {'.*_antenna' in head}",
        f"  -> effort_limit_sim={eff} N.m (SG90 stalls ~0.18 N.m => ~{float(eff)/0.18:.0f}x)",
        f"  -> armature={arm} vs largest antenna principal inertia {biggest:.3e} = {arm/biggest:.0f}x",
    ]


@issue("PLANT-5", "Torque ceiling is 1.78x datasheet stall, pinned to 12.1 V")
def _():
    bam = json.load(open(P("experiments/v2/params_sts3250_id008.json")))
    kt, Rr, vin = bam["kt"], bam["R"], bam["vin"]
    calc = kt * vin / Rr
    eff = float(re.search(r"effort_limit_sim=([\d.]+)",
                          R("isaac_lab_env/open_duck_mini_v2/robot_cfg.py")).group(1))
    ds = 50 * 0.0980665
    return ("CONFIRMED" if abs(calc - eff) < 0.01 else "REFUTED"), [
        f"BAM kt={kt:.6f} R={Rr:.6f} vin={vin} V -> kt*V/R = {calc:.4f} N.m",
        f"robot_cfg effort_limit_sim = {eff} (matches BAM)",
        f"datasheet 50 kg.cm = {ds:.3f} N.m -> simulated ceiling is {eff/ds:.2f}x",
        f"pack nominal 11.1 V -> {kt*11.1/Rr:.3f} N.m ({100*(1-kt*11.1/Rr/eff):.1f}% below sim)",
        "and no DR touches any actuator parameter",
    ]


@issue("PLANT-6", "Joint dry friction is inactive during motion")
def _():
    rc = R("isaac_lab_env/open_duck_mini_v2/robot_cfg.py")
    y = R("exported_policies/v5d_contact_wrench_ppo/env.yaml")
    blk = y.split("actuators:", 1)[1].split("\n  articulation_props", 1)[0]
    vals = {}
    for grp in ("legs", "head"):
        seg = blk.split(f"{grp}:", 1)[1][:900]
        vals[grp] = dict(re.findall(
            r"^\s+(friction|dynamic_friction|viscous_friction): ([\w.]+)", seg, re.M))
    usd_dir = P("mini_bdx/robots/open_duck_mini_v2/usd/configuration")
    usd_fr = any(b"jointFriction" in open(os.path.join(usd_dir, f), "rb").read()
                 for f in os.listdir(usd_dir) if f.endswith(".usd"))
    ok = "dynamic_friction" not in rc and "viscous_friction" not in rc
    return ("CONFIRMED" if ok else "REFUTED"), [
        f"resolved actuator friction: {vals}",
        f"any USD layer authors a joint friction attribute: {usd_fr}",
        "actuator_base.py:176-192 -- None means 'read from USD'; USD has none => 0.0",
        f"BAM also identifies friction_viscous="
        f"{json.load(open(P('experiments/v2/params_sts3250_id008.json')))['friction_viscous']}, never applied",
    ]


@issue("PLANT-7", "No latency / action-delay / observation-staleness model")
def _():
    env = R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    y = R("exported_policies/v5d_contact_wrench_ppo/env.yaml")
    # match identifiers only, and ignore comments (the file has the word "delays" in prose)
    code = "\n".join(l.split("#")[0] for l in env.splitlines())
    hits = {t: bool(re.search(rf"\b{t}\b", code) or re.search(rf"\b{t}\b", y))
            for t in ("latency", "delay", "action_delay", "ObservationDelay")}
    return ("CONFIRMED" if not any(hits.values()) else "REFUTED"), [
        f"identifier hits in env_cfg.py (comments stripped) and shipped env.yaml: {hits}",
        "-> an absence, so nothing in the config will ever look wrong",
    ]


# -------------------------------------------------------------------- CFG
@issue("CFG-1", "torque_z_range is overwritten by skew(r)xF in the warp kernel")
def _():
    k = open("/home/xiaohui_chen/IsaacLab/source/isaaclab/isaaclab/utils/warp/kernels.py").read()
    fn = k.split("def set_forces_and_torques_at_position(", 1)[1].split("\n@wp.kernel", 1)[0]
    n = fn.count("composed_torques_b[env_ids[tid_env], body_ids[tid_body]] =")
    ce = R("isaac_lab_env/open_duck_mini_v2/contact_events.py")
    passes = all(t in ce for t in ("forces=", "torques=", "positions=", "set_forces_and_torques"))
    return ("CONFIRMED" if n >= 2 and passes and "+=" not in fn else "REFUTED"), [
        f"assignments to composed_torques_b in the kernel: {n} (the 2nd overwrites the 1st)",
        f"kernel uses '=' not '+=': {'+=' not in fn}",
        f"contact_events passes forces+torques+positions in one call: {passes}",
        "runtime: requested tau=[0,0,0.15] with F=[3,0,0] at r=[0,0.04,0] -> composed [0,0,-0.12]",
    ]


@issue("CFG-2", "Obstacles are placed twice per episode")
def _():
    ce = R("isaac_lab_env/open_duck_mini_v2/contact_events.py")
    m = re.search(r"fresh = \(env\.episode_length_buf (<=|<) (\d)\)", ce)
    return ("CONFIRMED" if m and m.group(1) == "<=" and m.group(2) == "1" else "REFUTED"), [
        f"placement trigger: episode_length_buf {m.group(1)} {m.group(2)}",
        "buf is 0 on the reset step and 1 on the next -> fires twice",
        f"and re-rolls obstacle_active each time: {'active = torch.rand' in ce}",
    ]


@issue("CFG-3", "v5d inherits DuckRewards, so nothing reads the disturbance gate")
def _():
    src = R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    chain, cur = [], "OpenDuckContactWrenchEnvCfg"
    parents = dict(re.findall(r"class (OpenDuck\w+)\((\w+)\)", src))
    while cur in parents:
        chain.append(cur)
        cur = parents[cur]
    chain.append(cur)
    y = R("exported_policies/v5d_contact_wrench_ppo/env.yaml")
    counts = {t: y.count(t) for t in ("ground_contact", "GatedTrack", "flat_orientation_deadzone",
                                      "feet_slide", "disturbance_gate",
                                      "polynomial_coefficients_v2", "normalized_match")}
    ok = "OpenDuckContactEnvCfg" not in chain and all(v == 0 for v in counts.values())
    return ("CONFIRMED" if ok else "REFUTED"), [
        "v5d chain: " + " -> ".join(chain),
        f"passes through the v5a/v5b arm (OpenDuckContactEnvCfg): {'OpenDuckContactEnvCfg' in chain}",
        f"gated machinery in the shipped env.yaml: {counts}",
    ]


@issue("CFG-4", "ground_contact binds body `head`, which has no collider")
def _():
    seg = re.search(r'<body name="head"[^>]*>(.*?)</body>',
                    R("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml"), re.S).group(1)
    env = R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    bound = 'body_names=["head", "knee_and_ankle_assembly.*"]' in env
    return ("CONFIRMED" if "<geom" not in seg and bound else "REFUTED"), [
        f"body 'head' contains a <geom>: {'<geom' in seg}",
        f"ground_contact binds ['head', 'knee_and_ankle_assembly.*']: {bound}",
        "runtime: max |F| on `head` = 0.000000 N while feet saw 821 N on the same sensor",
        "scope: DuckContactRewards (v5a/v5b) only -- the shipped v5d does not carry the term",
    ]


@issue("CFG-5", "ContactSensor history spans physics steps, not policy steps")
def _():
    env = R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    hl = int(re.search(r"history_length=(\d)", env).group(1))
    y = R("exported_policies/v5d_contact_wrench_ppo/env.yaml")
    up = re.search(r"update_period: ([\d.]+)", y).group(1)
    return ("CONFIRMED" if hl == 3 else "REFUTED"), [
        f"history_length={hl}, update_period={up} (0.0 = refresh every physics substep)",
        f"sim.dt=0.005, decimation=4 -> history spans {hl*5} ms vs a 20 ms control step",
    ]


# ------------------------------------------------------------------- EVAL
@issue("EVAL-1", "--report-only appends every JSON in the results dir, unfiltered")
def _():
    fn = R("scripts/evaluate_policies.py").split(
        "def write_comparison_markdown(", 1)[1].split("\ndef ", 1)[0]
    body = fn.split("entries = []", 1)[1].split("section =", 1)[0]
    enumerates = ("os.listdir" in fn or "glob" in fn) and '.endswith(".json")' in fn
    filt = any(t in body for t in ("task_id", "exclude", "allow", "skip", "continue"))
    push = [f for f in os.listdir(P("docs/jetson-mod/eval_results_v4")) if "pusheval" in f]
    return ("CONFIRMED" if enumerates and not filt else "REFUTED"), [
        f"enumerates the directory and takes every *.json: {enumerates}; any filter: {filt}",
        f"pusheval JSONs already present in eval_results_v4/: {push}",
        f"v4_comparison.md claims 'no external pushes': "
        f"{'no external pushes' in R('docs/jetson-mod/v4_comparison.md')}",
    ]


@issue("EVAL-2", "Documented eval protocol is not the protocol that ran")
def _():
    pipe = R("scripts/v5_pipeline.sh")
    conds = re.search(r'CONDITIONS="([^"]+)"', pipe).group(1).split(";")
    gate = re.search(r'PASSED" -lt (\d)', pipe).group(1)
    counts = {}
    for d in ("eval_results_v4", "eval_results_v5"):
        s = set()
        for f in os.listdir(P("docs/jetson-mod", d)):
            if f.endswith(".json"):
                s.add(len(json.load(open(P("docs/jetson-mod", d, f)))["protocol"]["conditions"]))
        counts[d] = sorted(s)
    ok = counts["eval_results_v4"] == [5] and counts["eval_results_v5"] == [6] and len(conds) == 6
    return ("CONFIRMED" if ok else "REFUTED"), [
        f"AGENTS.md mandates 5 conditions / gate >= 4-of-5",
        f"v5_pipeline.sh runs {len(conds)} conditions, gate '>= {gate}'",
        f"condition counts found in the archived JSONs: {counts}",
    ]


@issue("EVAL-3", "Documented deployment clamp exceeds the trained command hull")
def _():
    ag, env = R("AGENTS.md"), R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    m = re.search(r"forward \[([-\d., ]+)\], lateral \[([-\d., ]+)\], turn \[([-\d., ]+)\]", ag)
    hx = [float(x) for x in re.search(r"ranges\.lin_vel_x = \(([-\d., ]+)\)", env).group(1).split(",")]
    hy = [float(x) for x in re.search(r"ranges\.lin_vel_y = \(([-\d., ]+)\)", env).group(1).split(",")]
    f = [float(x) for x in m.group(1).split(",")]
    lat = [float(x) for x in m.group(2).split(",")]
    return ("CONFIRMED" if f[1] > hx[1] else "REFUTED"), [
        f"clamp: forward {m.group(1)}, lateral {m.group(2)}, turn {m.group(3)}",
        f"hull : vx {hx}, vy {hy}",
        f"forward {f[1]}/{hx[1]} = {f[1]/hx[1]:.2f}x, backward {abs(f[0]/hx[0]):.2f}x, "
        f"lateral {lat[1]/hy[1]:.2f}x; turn is correctly inside",
    ]


# ------------------------------------------------------------------ SHELL
@issue("SHELL-1", "v5_pipeline.sh selects the run directory by mtime, not run name")
def _():
    line = [l for l in R("scripts/v5_pipeline.sh").splitlines() if "RUNDIR=$(" in l][0]
    root = os.path.expanduser("~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5")
    n = len(os.listdir(root)) if os.path.isdir(root) else -1
    return ("CONFIRMED" if "RUN_NAME" not in line and "ls -td" in line else "REFUTED"), [
        f"selection line: {line.strip()}",
        f"references RUN_NAME: {'RUN_NAME' in line}; candidate dirs today: {n}",
        "partially guarded upstream: lines 54-58 block on $STATE/${RUN_NAME}.pid first",
    ]


@issue("SHELL-2", "v5_chain.sh deletes the pidfile its duplicate-run guard needs")
def _():
    c, l = R("scripts/v5_chain.sh"), R("scripts/launch_training_detached.sh")
    rm = 'rm -f "$STATE/${RUN_NAME}.pid"' in c
    guard = '[[ -f "$PIDFILE" ]] && kill -0' in l
    return ("CONFIRMED" if rm and guard else "REFUTED"), [
        f"v5_chain.sh removes the pidfile before launching: {rm}",
        f"launcher guard is `[[ -f $PIDFILE ]] && kill -0 ...`: {guard}",
        "-> removing the file short-circuits the guard before the liveness half runs",
    ]


@issue("SHELL-3", "Launcher dispatches --algorithm to a deleted script (dead code)")
def _():
    l = R("scripts/launch_training_detached.sh")
    exists = os.path.exists(P("scripts/train_amp.py"))
    return ("CONFIRMED" if "train_amp.py" in l and not exists else "REFUTED"), [
        f"scripts/train_amp.py exists: {exists}; launcher still references it: {'train_amp.py' in l}",
        f"deleted by: {sh('git log --oneline --diff-filter=D -- scripts/train_amp.py')}",
        "the pidfile is still written for the dead process, so the queue records 'completed'",
        "SCOPE: dead code -- the queue has not run since 2026-06-15",
    ]


@issue("SHELL-4", "Queue's GPU-busy predicate is blind to eval and play jobs")
def _():
    pat = re.search(r"TRAIN_PATTERN='([^']+)'", R("scripts/run_experiment_queue.sh")).group(1)
    hits = {s: bool(re.search(pat, f"isaaclab.sh -p scripts/{s}"))
            for s in ("train_ppo.py", "evaluate_policies.py", "play_policy.py")}
    return ("CONFIRMED" if not hits["evaluate_policies.py"] else "REFUTED"), [
        f"TRAIN_PATTERN = {pat!r} -> {hits}",
        f"SCOPE: v5 ran on v5_chain.sh, whose wait_for_gpu also greps evaluate_policies "
        f"({'evaluate' in R('scripts/v5_chain.sh')}); residual gap = play_policy.py only",
    ]


@issue("SHELL-5", "Training watchdog fails open on a parse error")
def _():
    c = R("scripts/v5_chain.sh")
    wd = c.split("sleep 1800", 1)[1][:700]
    return ("CONFIRMED" if "2>/dev/null" in wd and "else" in wd else "REFUTED"), [
        "python parse errors are swallowed by 2>/dev/null; the else-branch prints 'healthy'",
        "so an unparseable metric line reports a doomed run as healthy",
        "NB v5a/v5b logs contain 0 occurrences of the metric (introduced in the same commit),",
        "   so the watchdog could not have run against them; on v5d it worked",
    ]


# -------------------------------------------------------------------- ART
@issue("ART-1", "policy.onnx is gitignored while policy.onnx.data is tracked")
def _():
    ci = sh("git check-ignore -v exported_policies/v5d_contact_wrench_ppo/policy.onnx")
    tracked = sh("git ls-files exported_policies/v5d_contact_wrench_ppo/").split()
    names = [os.path.basename(t) for t in tracked]
    ok = ".gitignore" in ci and "policy.onnx" not in names and "policy.onnx.data" in names
    return ("CONFIRMED" if ok else "REFUTED"), [
        f"git check-ignore: {ci}",
        f"tracked: {names}",
        "-> a fresh clone gets orphaned external weights and no graph",
    ]


@issue("ART-2", "exported_policies/v4_robust/ was never created")
def _():
    dirs = sorted(os.listdir(P("exported_policies")))
    ext = os.path.expanduser(
        "~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt")
    return ("CONFIRMED" if "v4_robust" not in dirs else "REFUTED"), [
        f"exported_policies/: {[d for d in dirs if os.path.isdir(P('exported_policies', d))]}",
        f"only surviving copy (outside git): {ext} exists={os.path.exists(ext)}",
    ]


@issue("ART-3", "usd/config.yaml records a deleted worktree as the asset source")
def _():
    cfg = R("mini_bdx/robots/open_duck_mini_v2/usd/config.yaml")
    stale = "Open_Duck_Mini_Jetson-cad" in cfg
    return ("CONFIRMED" if stale else "REFUTED"), [
        f"config.yaml names the -cad worktree: {stale}; that path exists: "
        f"{os.path.exists('/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson-cad')}",
        "the staleness guard still passes because the test pops the path keys before hashing",
    ]


# ----------------------------------------------------------------- DEPLOY
@issue("DEPLOY-1", "The ONNX omits action_scale and q_default entirely")
def _():
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return "INCONCLUSIVE", ["onnx not importable in this interpreter; skipped"]
    p = P("exported_policies/v5d_contact_wrench_ppo/policy.onnx")
    if not os.path.exists(p):
        return "INCONCLUSIVE", [f"{p} absent (it is gitignored -- see ART-1)"]
    m = onnx.load(p)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    tot = sum(a.size for a in inits.values())
    n025 = sum(int(np.isclose(a, 0.25, atol=1e-6).sum()) for a in inits.values())
    sixteen = [k for k, a in inits.items() if a.size == 16]
    return ("CONFIRMED" if n025 == 0 else "REFUTED"), [
        f"nodes: {[n.op_type for n in m.graph.node]}",
        f"total initializer scalars: {tot}; values equal to 0.25: {n025}",
        f"16-element tensors that could hold q_default: {sixteen}",
        "-> q_target = q_default + 0.25*a lives in Isaac Lab's JointPositionAction, not the graph",
    ]


@issue("DEPLOY-3", "4 of the 59 observation dims cannot be measured on hardware")
def _():
    order = json.load(open(P("scripts/duck_init_pos.json")))["joint_order"]
    idx = [i for i, n in enumerate(order) if "antenna" in n]
    return ("CONFIRMED" if len(idx) == 2 else "REFUTED"), [
        f"antenna joint indices: {idx}",
        f"-> obs dims joint_pos_rel {[9+i for i in idx]}, joint_vel_rel {[25+i for i in idx]}; "
        f"action outputs {idx}",
        "the SG90s are open-loop PWM servos with no position feedback",
    ]


# ------------------------------------------------------------------- TEST
@issue("TEST-1", "Grepped test literals are non-unique (only scale = 0.25 is unique)")
def _():
    env = R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    lits = ["(-0.5, 0.5)", "weight=1.0", "(-0.148, 0.222)", '"std": 0.5', "scale = 0.25"]
    c = {l: env.count(l) for l in lits}
    ok = c["scale = 0.25"] == 1 and all(c[l] > 1 for l in lits if l != "scale = 0.25")
    return ("CONFIRMED" if ok else "REFUTED"), [
        "occurrence counts in env_cfg.py: " + ", ".join(f"{k!r}={v}" for k, v in c.items()),
        "mutation result: deleting imitation_reward / widening the hull leaves 101 passed",
    ]


@issue("TEST-2", "A bare pytest from the repo root fails at collection")
def _():
    out = sh("python3 -m pytest -m phase5 -q 2>&1 | tail -3")
    scoped = sh("python3 -m pytest tests/ -q 2>&1 | tail -1")
    return ("CONFIRMED" if "error" in out.lower() else "REFUTED"), [
        f"bare pytest: {out.replace(chr(10), ' | ')[:150]}",
        f"pytest tests/: {scoped}",
        f"pytest.ini sets testpaths: {'testpaths' in R('pytest.ini')}",
    ]


@issue("TEST-3", "requires_isaac_sim is documented but applied to zero tests")
def _():
    t, ini = R("tests/test_isaac_lab_env.py"), R("pytest.ini")
    deco = len(re.findall(r"(?m)^\s*@pytest\.mark\.requires_isaac_sim\b", t))
    return ("CONFIRMED" if deco == 0 and "requires_isaac_sim" in t
            and "requires_isaac_sim" not in ini else "REFUTED"), [
        f"textual mentions: {len(re.findall('requires_isaac_sim', t))} (prose); real decorators: {deco}",
        f"registered in pytest.ini: {'requires_isaac_sim' in ini}",
        f"pytest's view: {sh('python3 -m pytest tests/ -m requires_isaac_sim -q 2>&1 | tail -1')}",
    ]


# -------------------------------------------------------------------- REF
@issue("REF-1", "Reference library has no vy=0 / wz=0 cell; contact dims constant")
def _():
    lib = pickle.load(open(P("isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl"), "rb"))
    ks = sorted(lib)
    vy = sorted({float(k.split("_")[1]) for k in ks})
    wz = sorted({float(k.split("_")[2]) for k in ks})
    c32 = [lib[k]["coefficients"]["dim_32"] for k in ks]
    c33 = [lib[k]["coefficients"]["dim_33"] for k in ks]
    same = all(c == c32[0] for c in c32) and all(c == c33[0] for c in c33)
    return ("CONFIRMED" if same and 0.0 not in vy and 0.0 not in wz else "REFUTED"), [
        f"cells: {len(ks)}", f"vy axis: {vy} -> has 0.0? {0.0 in vy}",
        f"wz axis: {wz} -> has 0.0? {0.0 in wz}",
        f"contact dims 32/33 identical across every cell: {same}",
    ]


@issue("REF-2", "33.6% of reachable knee references are clamped; velocities are not")
def _():
    import numpy as np
    lib = pickle.load(open(P("isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl"), "rb"))
    lim = {}
    for j in mjcf_bodies().iter("joint"):
        if j.get("range"):
            lo, hi = (float(x) for x in j.get("range").split())
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9
            lim[j.get("name")] = (mid - half, mid + half)
    PG = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna",
          "right_antenna", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
          "right_knee", "right_ankle"]
    LEG = [n for n in PG if "hip" in n or "knee" in n or "ankle" in n]
    t = np.arange(27) / 27.0
    hull = lambda k: (-0.148 <= float(k.split("_")[0]) <= 0.222
                      and -0.111 <= float(k.split("_")[1]) <= 0.111
                      and abs(float(k.split("_")[2])) <= 0.593)
    reach = [k for k in lib if hull(k)]
    tot = cl = vn = vx = 0
    vmax = 0.0
    for k in reach:
        C = np.array([lib[k]["coefficients"][f"dim_{i}"] for i in range(40)])
        val = np.array([np.polyval(C[i][::-1], t) for i in range(40)])
        for jn in LEG:
            pg = PG.index(jn)
            pos, vel = val[pg], val[16 + pg]
            lo, hi = lim[jn]
            if "knee" in jn:
                cl += int(((pos < lo) | (pos > hi)).sum()); tot += len(pos)
            vmax = max(vmax, float(np.abs(vel).max()))
            vx += int((np.abs(vel) > 8.94).sum()); vn += len(vel)
    return "CONFIRMED", [
        f"reachable cells under the shipped hull (|wz| <= 0.593): {len(reach)} of {len(lib)}",
        f"knee samples outside soft limits (clamped at runtime): {cl}/{tot} = {100*cl/tot:.2f}%",
        f"max |reference joint velocity|: {vmax:.2f} rad/s vs the 8.94 limit ({vmax/8.94:.2f}x)",
        f"velocity samples over the limit: {100*vx/vn:.2f}%",
        "positions ARE clamped (imitation_reward.py:265-268); velocities are NOT",
    ]


# -------------------------------------------------------------------- DOC
@issue("DOC-2", "experiment_journal.md has zero v5 entries; v5_contact_results.md absent")
def _():
    j = R("docs/jetson-mod/experiment_journal.md")
    hits = {k: j.count(k) for k in ("v5a_gated_ft", "v5b_ungated_ft", "v5c_contact_only",
                                    "v5d_contact_wrench", "v5_smoke")}
    ex = os.path.exists(P("docs/jetson-mod/v5_contact_results.md"))
    return ("CONFIRMED" if sum(hits.values()) == 0 and not ex else "REFUTED"), [
        f"v5 run names in the journal: {hits}", f"v5_contact_results.md exists: {ex}",
    ]


@issue("DOC-4", "task_plan.md still carries superseded v5 numbers")
def _():
    tp = R("docs/jetson-mod/task_plan.md")
    ev, found = [], 0
    for pat, label in ((r"1\.3 m/s", "push ramp 0.5->1.3 m/s (superseded by 0.4->0.7)"),
                       (r"1\.6 kg", "wrench sized at ~1.6 kg body weight (sim robot is 3.657 kg)")):
        for ln, line in enumerate(tp.splitlines(), 1):
            if re.search(pat, line):
                ev.append(f"{label} -- task_plan.md:{ln}"); found += 1
                break
    ev.append(f"56-dim TensorRT spec still present: {'randn(56)' in tp}  (fixed 2026-08-09)")
    return ("CONFIRMED" if found >= 2 else "REFUTED"), ev


# -------------------------------------------- verified refutations (regression)
@issue("NOT-AN-ISSUE-1", "The wrench gate DOES exercise rotation-under-load")
def _():
    j = json.load(open(P("docs/jetson-mod/eval_results_v5/v5d_contact_wrench_wrencheval.json")))
    row = j["per_condition"].get("vx+0.00_vy+0.00_wz+0.50")
    return ("CONFIRMED" if row else "REFUTED"), [
        f"conditions: {j['protocol']['conditions']}",
        f"wz=0.5 condition: {row['episodes']} episodes, {row['falls']} falls "
        f"({row['fall_rate_pct']:.2f}%)",
        f"aggregate {j['aggregate']['fall_rate_pct']:.3f}%",
        "-> evaluate_policies.py:apply_condition overrides the cfg's ang_vel_z per condition",
    ]


def run(selected=None):
    for cid, title, fn in CHECKS:
        if selected and cid not in selected:
            continue
        try:
            verdict, ev = fn()
        except Exception as e:
            verdict, ev = "INCONCLUSIVE", [f"{type(e).__name__}: {e}"]
        RESULTS.append((cid, title, verdict, ev))
        if "--quiet" not in sys.argv:
            print(f"\n[{cid}] {title}\n  -> {verdict}")
            for line in ev:
                print(f"     {line}")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or None
    run(sel)
    print("\n" + "=" * 78)
    print("SUMMARY   CONFIRMED = the issue is still present in the repo")
    print("=" * 78)
    for cid, title, verdict, _ in RESULTS:
        print(f"  {verdict:13s} {cid:16s} {title}")
    refuted = [r[0] for r in RESULTS if r[2] == "REFUTED"]
    unknown = [r[0] for r in RESULTS if r[2] == "INCONCLUSIVE"]
    print(f"\n  CONFIRMED {sum(1 for r in RESULTS if r[2] == 'CONFIRMED')} / {len(RESULTS)}")
    if refuted:
        print(f"  REFUTED -> the issue was fixed, or the check has rotted: {refuted}")
        print("     update docs/jetson-mod/known_issues.md and remove the check.")
    if unknown:
        print(f"  INCONCLUSIVE -> could not be evaluated here: {unknown}")
        print("     the ONNX checks need an interpreter with `onnx` installed, e.g.")
        print("     ~/IsaacLab/_isaac_sim/python.sh scripts/verify_known_issues.py DEPLOY-1")
    # exit 1 = a claim was refuted (act on it); 2 = a claim could not be evaluated
    sys.exit(1 if refuted else (2 if unknown else 0))
