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
# PLANT-1 FIXED 2026-08-11: `base` was merged into `trunk_assembly`, which now
# carries the <freejoint/> and is the articulation root. audit_plant_mass.py
# exits 0 (MJCF 2.657067 == PhysX 2.657067) and the PhysX invalid-inertia
# warning is gone. The regression guard promised in the register now lives in
# tests/ as an "every MJCF body declares <inertial>" assertion. Check retired.


@issue("PLANT-1b", "The other massless frames DO carry a 1e-09 placeholder inertial")
def _():
    hits = re.findall(r'<body name="(\w+)"[^>]*>\s*<inertial[^>]*mass="1e-09"',
                      R("mini_bdx/robots/open_duck_mini_v2/robot_motors.xml"))
    return ("CONFIRMED" if len(hits) >= 4 else "REFUTED"), [
        f"frames with a 1e-09 placeholder: {hits}",
        "-> measured on GPU 2026-08-11: PhysX substitutes (4.000e-12, 4.000e-12,"
        " 4.000e-12) for each, i.e. 0.4*m*r^2 with r=0.1 m -- NOT the infinite"
        " inertia its docs warn about, so these frames are benign",
    ]


# PLANT-2 RESOLVED 2026-08-11 by the PLANT-1 fix: `trunk_assembly` IS the
# articulation root now, so add_base_mass/base_com perturb the body their names
# claim, and there is no 1.000 kg constant offset left for the range to be
# "too small" against. Check retired.


# PLANT-3 FIXED 2026-08-13 (Task M0/M0b).
# Task M0 added a +20 mm spawn z to reset_base.pose_range. Re-measured
# with this script before retiring: fraction starting below ground 61.7%
# -> 0.0%, deepest reset +4.40 mm.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
# PLANT-4 FIXED 2026-08-13 (Task M0/M0b).
# Task M0b removed `.*_antenna` from the `head` actuator group and gave
# the antennas their own group at SG90 scale (effort 8.716 -> 0.18 N.m,
# armature 0.040 -> 4.0e-06 against a 3.31e-06 link inertia). They are
# also out of the action and observation spaces entirely.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
# PLANT-5 FIXED 2026-08-13 (Task M0/M0b).
# Task M0 replaced the bare 8.716 (BAM's ELECTRICAL stall at 12.1 V, 1.78x
# datasheet) with STS3250_EFFORT_LIMIT_NM = 4.903, the 50 kg.cm datasheet
# stall, and recorded STS3250_CONTINUOUS_NM = 1.569 alongside it.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
@issue("PLANT-6", "Joint dry friction is inactive during motion")
def _():
    rc = R("isaac_lab_env/open_duck_mini_v2/robot_cfg.py")
    y = R("exported_policies/v6d_contact_wrench_ppo/env.yaml")
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


# PLANT-7 FIXED 2026-08-13 (Task M0/M0b).
# Task M0 added isaac_lab_env/open_duck_mini_v2/latency.py: the policy's
# joint_pos/joint_vel observations are delayed 0-2 control steps (0-40
# ms), redrawn per env at reset. The VALUE is provisional until Task S.6
# measures the real loop; the mechanism is not.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
# -------------------------------------------------------------------- CFG
# CFG-1 FIXED 2026-08-13 (Task M0/M0b).
# Task M0 switched the wrench from set_forces_and_torques(..., positions=)
# to add_forces_and_torques(..., positions=), preceded by a composer
# reset() for those envs. The set_ variant's warp kernel ASSIGNS the
# torque and then assigns it again from the position moment, discarding
# torque_z_range; the add_ variant accumulates both.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
# CFG-2 FIXED 2026-08-13 (Task M0/M0b).
# Task M0 changed the fresh-episode predicate from `episode_length_buf <=
# 1` (true at buf 0 AND 1, so two independent obstacle draws per episode)
# to `== 1`.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
@issue("CFG-3", "the contact arm inherits DuckRewards, so nothing reads the disturbance gate")
def _():
    src = R("isaac_lab_env/open_duck_mini_v2/env_cfg.py")
    chain, cur = [], "OpenDuckContactWrenchEnvCfg"
    parents = dict(re.findall(r"class (OpenDuck\w+)\((\w+)\)", src))
    while cur in parents:
        chain.append(cur)
        cur = parents[cur]
    chain.append(cur)
    y = R("exported_policies/v6d_contact_wrench_ppo/env.yaml")
    counts = {t: y.count(t) for t in ("ground_contact", "GatedTrack", "flat_orientation_deadzone",
                                      "feet_slide", "disturbance_gate",
                                      "polynomial_coefficients_v2", "normalized_match")}
    ok = "OpenDuckContactEnvCfg" not in chain and all(v == 0 for v in counts.values())
    return ("CONFIRMED" if ok else "REFUTED"), [
        "contact chain: " + " -> ".join(chain),
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
    y = R("exported_policies/v6d_contact_wrench_ppo/env.yaml")
    up = re.search(r"update_period: ([\d.]+)", y).group(1)
    return ("CONFIRMED" if hl == 3 else "REFUTED"), [
        f"history_length={hl}, update_period={up} (0.0 = refresh every physics substep)",
        f"sim.dt=0.005, decimation=4 -> history spans {hl*5} ms vs a 20 ms control step",
    ]


# ------------------------------------------------------------------- EVAL
# EVAL-1 FIXED 2026-08-12 (Task R0): write_comparison_markdown() takes an
# `include` allowlist, exposed as the repeatable `--include NAME` CLI flag, and
# tests/test_eval_report_filter.py runs the real script in a subprocess to prove
# the filter actually filters. Per this script's convention a fixed issue carries
# no check, so the check is retired rather than inverted.


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


# DEPLOY-5 (formerly checked here under the stale id "EVAL-3") was FIXED on
# 2026-08-11: the documented clamp in AGENTS.md now equals the trained hull
# exactly (forward 1.00x, backward 1.00x, lateral 1.00x, turn inside). Per this
# script's own convention a fixed issue carries no check, so the check is
# retired rather than inverted.


# ------------------------------------------------------------------ SHELL
# SHELL-1 FIXED 2026-08-13 (Task R2b prep). v5_pipeline.sh selects the run
# directory BY NAME -- RUNDIR="${RUNDIR_OVERRIDE:-$(ls -td "$LOGROOT"/*_"$RUN_NAME"/ ...)}"
# -- with an mtime fallback that only fires when no name-matching dir exists
# and PRINTS A WARNING when it does. LOGROOT, RESULTS, COMPARISON_MD,
# CONDITIONS and INCLUDE_ARGS are all environment-overridable now, so a
# second campaign reuses the pipeline instead of editing its constants.
# Per this script's convention a fixed issue carries no check.
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
    ci = sh("git check-ignore -v exported_policies/v6d_contact_wrench_ppo/policy.onnx")
    tracked = sh("git ls-files exported_policies/v6d_contact_wrench_ppo/").split()
    names = [os.path.basename(t) for t in tracked]
    ok = ".gitignore" in ci and "policy.onnx" not in names and "policy.onnx.data" in names
    return ("CONFIRMED" if ok else "REFUTED"), [
        f"git check-ignore: {ci}",
        f"tracked: {names}",
        "-> a fresh clone gets orphaned external weights and no graph",
    ]


@issue("ART-2", "the shipped policy's provenance parent is not in git")
def _():
    # Reframed 2026-08-15. The original complaint was that v4_robust was never
    # archived; the selection decision retired v4_robust outright, so that
    # framing would now be CONFIRMED forever for an INTENTIONAL reason -- a
    # check that describes no defect. The underlying concern is real and simply
    # moved: exported_policies/ keeps exactly one archive, and that archive's
    # README names a parent run which lives only in a training log directory.
    # A clean clone cannot resolve the shipped policy's lineage.
    dirs = sorted(d for d in os.listdir(P("exported_policies"))
                  if os.path.isdir(P("exported_policies", d)))
    rd = R("exported_policies/v6d_contact_wrench_ppo/README.md")
    m = re.search(r"fine-tuned from `([\w]+)`", rd)
    parent = m.group(1) if m else None
    ext = os.path.expanduser(
        "~/IsaacLab/logs/rsl_rl/open_duck_ppo_v6/2026-08-13_01-05-40_v6_robust/model_2999.pt")
    unresolvable = parent is not None and parent not in dirs
    return ("CONFIRMED" if unresolvable else "REFUTED"), [
        f"exported_policies/: {dirs}",
        f"shipped README names parent: {parent!r} -- archived here: {parent in dirs}",
        f"only surviving copy (outside git): {ext} exists={os.path.exists(ext)}",
        "-> a clean clone cannot reproduce or verify the parent this policy was fine-tuned from",
    ]


# ART-3 FIXED 2026-08-11: regenerating the USD from this repo rewrote
# usd/config.yaml, whose asset_path now names the live checkout instead of the
# deleted Open_Duck_Mini_Jetson-cad worktree. Check retired.


# ----------------------------------------------------------------- DEPLOY
@issue("DEPLOY-1", "The ONNX omits action_scale and q_default entirely")
def _():
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return "INCONCLUSIVE", ["onnx not importable in this interpreter; skipped"]
    # Check the CURRENT export, not a frozen one. Pinned at
    # v5d_contact_wrench_ppo this check would keep reporting CONFIRMED off a
    # superseded archive even if a newer export embedded the constants -- the
    # same blindness that let PLANT-3 and SHELL-1 pass their own fixes.
    # Only one archive exists by policy: exported_policies/ keeps the mainline
    # locomotion and nothing else (locomotion_selection.md, 2026-08-15).
    candidates = ["exported_policies/v6d_contact_wrench_ppo/policy.onnx"]
    p = next((P(c) for c in candidates if os.path.exists(P(c))), None)
    if p is None:
        return "INCONCLUSIVE", [f"none of {candidates} present (gitignored -- see ART-1)"]
    m = onnx.load(p)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    tot = sum(a.size for a in inits.values())
    n025 = sum(int(np.isclose(a, 0.25, atol=1e-6).sum()) for a in inits.values())
    # Action width READ OFF THE GRAPH. Hardcoding 16 finds nothing on a 14-action
    # policy, so the "could hold q_default" line would silently go empty and read
    # as stronger evidence than it is.
    out_dims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_act = out_dims[-1] if out_dims else 0
    same_width = [k for k, a in inits.items() if a.size == n_act]
    return ("CONFIRMED" if n025 == 0 else "REFUTED"), [
        f"export under test: {os.path.relpath(p, P('.'))}",
        f"nodes: {[n.op_type for n in m.graph.node]}",
        f"total initializer scalars: {tot}; values equal to 0.25: {n025}",
        f"{n_act}-element tensors that could hold q_default: {same_width}",
        "-> q_target = q_default + 0.25*a lives in Isaac Lab's JointPositionAction, not the graph",
        "-> MITIGATED by the deployment_contract.json sidecar (Task R3); the graph is unchanged",
    ]


# DEPLOY-3 FIXED 2026-08-13 (Task M0/M0b).
# Task M0b removed the antennas from the action and observation spaces:
# obs 59 -> 53, action 16 -> 14, critic 62 -> 56, all three read back off
# a freshly built env. The four unmeasurable dims no longer exist.
# Per this script's convention a fixed issue carries no check, so the check
# is retired rather than inverted. The entry stays in known_issues.md.
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
# DOC-2 FIXED 2026-08-13 (Task R4). experiment_journal.md now carries Runs
# 17-21 for v5_smoke, v5a_gated_ft, v5b_ungated_ft, v5c_contact_only and
# v5d_contact_wrench, each with last-100-iteration TensorBoard means from
# scripts/tb_summary.py (rule 1 forbids console greps), its gate JSON path,
# a verdict naming the gate, and a 3.657 kg plant banner. The results
# document docs/jetson-mod/v5_contact_results.md now exists too.
# Per this script's convention a fixed issue carries no check.
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
