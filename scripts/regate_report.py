#!/usr/bin/env python3
"""Score a re-gate campaign against the frozen G-R1..G-R5 gates.

Task R1c (`docs/jetson-mod/task_plan_v2.md`, Phase R). The point is that the
verdict is *mechanical and recorded*, not remembered: `known_issues.md` exists
because verdicts that lived only in a session transcript decayed into wrong
documentation.

    python3 scripts/regate_report.py \\
        --results_dir docs/jetson-mod/eval_results_m2657 \\
        --candidate v5d_contact_wrench --control v4_robust --markdown

Pure stdlib. No Isaac Sim, no numpy, no GPU — it reads the JSONs
`evaluate_policies.py` already wrote.

Exit status:
    0  every scriptable gate passed
    1  at least one scriptable gate failed, or a required file is missing
    2  the directory mixes robot models, or a plant block is absent

**G-R3 (video) is never part of the exit code.** A script cannot watch a video,
and the one failure this whole protocol exists to prevent is a policy that
passes every aggregate metric while crawling (Run 12, `amp_command7`). G-R3 is
printed as MANUAL and must be answered in the verdict document by something
that actually looked at the frames.

**Exit 2 is the PLANT-1 guard.** For three policy generations every published
gate number described a robot 37.6 % heavier than the one on disk, because no
result recorded which plant it was measured on and nothing cross-checked. This
script refuses to score a directory whose entries disagree on the plant.
"""

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(
    REPO_ROOT, "docs", "jetson-mod", "eval_results_m2657")

# Frozen in task_plan_v2.md, Phase R, "Gate thresholds this phase is judged
# against (frozen here; do not re-derive)". Do not tune these to make a run
# pass — that is the failure mode the word "frozen" is there to prevent.
GATE_GAIT_MIN_VALID = 5          # G-R1: >= 5 of 6 conditions gait-valid
GATE_GAIT_TOTAL = 6
GATE_FALL_RATE_PCT = 1.0         # G-R2: AGENTS.md bar
GATE_FALL_RATE_PCT_TIGHT = 0.5   # G-R2: v5_retrain_plan.md §7 gate 1
GATE_REF_RMS_DELTA_DEG = 1.0     # G-R5: within 1.0 deg of the same-plant control

# The four contact-battery pairs behind G-R4, in report order.
BATTERY = [
    ("pusheval_v4def", "Push recovery, v4 fall rule"),
    ("pusheval_v5def", "Push recovery, v5 fall rule"),
    ("wrencheval", "Sustained wrench"),
    ("obstacleeval", "Obstacle graze"),
]

PLANT_MASS_TOL_KG = 1e-3


class MissingResult(Exception):
    """A file the gates need is not in the results directory."""


def load_dir(results_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        out[os.path.basename(path)[:-len(".json")]] = json.load(open(path))
    return out


def check_one_plant(entries):
    """Refuse to score a directory that mixes robot models.

    Returns a list of complaint strings; empty means the directory is clean.
    """
    problems = []
    missing = [n for n, d in entries.items() if not d.get("plant")]
    if missing:
        problems.append(
            "no plant block in: " + ", ".join(f"{n}.json" for n in sorted(missing))
            + " — produced by a pre-R0 evaluate_policies.py; re-run, do not "
              "annotate by hand")
        return problems

    # Group at the tolerance, but report the value actually measured — a
    # message that says "2.657" when the file says 2.657067 sends the reader
    # looking for a number that is not in any file.
    by_mass = {}
    by_dims = {}
    for name, d in entries.items():
        p = d["plant"]
        mass = p["simulated_total_mass_kg"]
        key = round(mass / PLANT_MASS_TOL_KG)
        by_mass.setdefault(key, (mass, []))[1].append(name)
        by_dims.setdefault((p["obs_dim"], p["action_dim"]), []).append(name)

    if len(by_mass) > 1:
        detail = "; ".join(
            f"{mass:.6f} kg in {sorted(names)}"
            for _, (mass, names) in sorted(by_mass.items()))
        problems.append(f"the directory mixes {len(by_mass)} plant masses: {detail}")
    if len(by_dims) > 1:
        detail = "; ".join(
            f"{o}/{a} in {sorted(names)}" for (o, a), names in sorted(by_dims.items()))
        problems.append(f"the directory mixes obs/action dims: {detail}")
    return problems


def pick(entries, name):
    if name not in entries:
        raise MissingResult(f"{name}.json")
    return entries[name]


def gait_summary(entry):
    a = entry.get("aggregate", {})
    return a.get("gait_valid_conditions"), a.get("conditions")


def fmt(v, spec="{:.3f}"):
    return "n/a" if v is None else spec.format(v)


def build_report(entries, candidate, control):
    """Returns (lines, gate_results). gate_results maps id -> (bool|None, text)."""
    lines = []
    gates = {}

    cand = pick(entries, candidate)
    ctrl = pick(entries, f"{control}_grid6")

    plant = cand["plant"]
    lines += [
        f"Results dir entries : {len(entries)}",
        f"Candidate           : {candidate}",
        f"Control             : {control}_grid6",
        f"Plant               : {plant['simulated_total_mass_kg']:.6f} kg, "
        f"root {plant['root_body']}, {plant['num_bodies']} bodies, "
        f"obs/action {plant['obs_dim']}/{plant['action_dim']}",
        f"USD asset hash      : {plant.get('usd_asset_hash', 'not recorded')}",
        "",
        "-- Open field (6 conditions x 10 windows x 64 envs x 30 s) ----------",
        "",
    ]

    rows = [
        ("gait valid", "{}/{}".format(*gait_summary(ctrl)),
         "{}/{}".format(*gait_summary(cand))),
    ]
    for key, label, spec in [
        ("fall_rate_pct", "fall rate (%)", "{:.3f}"),
        ("reference_tracking_rms_deg", "ref RMS (deg)", "{:.3f}"),
        ("stance_duty_left_pct", "duty L (%)", "{:.1f}"),
        ("stance_duty_right_pct", "duty R (%)", "{:.1f}"),
        ("stance_duty_asymmetry_pp", "duty asym (pp)", "{:.2f}"),
        ("ang_vel_z_error_radps", "wz err (rad/s)", "{:.3f}"),
        ("energy_proxy_w", "energy (W)", "{:.2f}"),
        ("mean_squared_jerk", "jerk", "{:.4f}"),
    ]:
        rows.append((label,
                     fmt(ctrl["aggregate"].get(key), spec),
                     fmt(cand["aggregate"].get(key), spec)))

    w = max(len(r[0]) for r in rows)
    lines.append(f"{'metric':<{w}}  {control + '_grid6':>22}  {candidate:>28}")
    lines.append("-" * (w + 54))
    for label, c, k in rows:
        lines.append(f"{label:<{w}}  {c:>22}  {k:>28}")
    lines.append("")

    # ---- G-R1 gait -------------------------------------------------------
    valid, total = gait_summary(cand)
    if valid is None:
        gates["G-R1"] = (False, "gait gate n/a — the JSON has no "
                                "gait_valid_conditions")
    else:
        ok = valid >= GATE_GAIT_MIN_VALID and total == GATE_GAIT_TOTAL
        gates["G-R1"] = (ok, f"gait gate {valid}/{total} "
                             f"(bar >={GATE_GAIT_MIN_VALID}/{GATE_GAIT_TOTAL})")

    # ---- G-R2 falls ------------------------------------------------------
    falls = cand["aggregate"].get("fall_rate_pct")
    ok = falls is not None and falls < GATE_FALL_RATE_PCT
    tight = " (also inside the tightened <=0.5% bar)" if (
        falls is not None and falls <= GATE_FALL_RATE_PCT_TIGHT) else (
        f" (outside the tightened <={GATE_FALL_RATE_PCT_TIGHT}% bar of "
        "v5_retrain_plan §7)" if ok else "")
    gates["G-R2"] = (ok, f"open-field fall rate {fmt(falls)}% "
                         f"(bar <{GATE_FALL_RATE_PCT}%){tight}")

    # ---- G-R3 video ------------------------------------------------------
    gates["G-R3"] = (None, "video audit — MANUAL, see the verdict doc. "
                           "A script cannot watch a video and this gate "
                           "outranks every metric above.")

    # ---- G-R4 contact battery -------------------------------------------
    lines += ["-- Contact battery (fall rate %, candidate vs same-plant control) --", ""]
    missing = []
    pairs = []
    for suffix, label in BATTERY:
        cname, kname = f"{control}_{suffix}", f"{candidate}_{suffix}"
        for n in (cname, kname):
            if n not in entries:
                missing.append(f"{n}.json")
        if cname in entries and kname in entries:
            cv = entries[cname]["aggregate"].get("fall_rate_pct")
            kv = entries[kname]["aggregate"].get("fall_rate_pct")
            pairs.append((label, cv, kv))

    if missing:
        gates["G-R4"] = (False, "contact battery INCOMPLETE — missing "
                                + ", ".join(sorted(set(missing))))
        lines.append("  missing: " + ", ".join(sorted(set(missing))))
    else:
        w2 = max(len(p[0]) for p in pairs)
        lines.append(f"{'gate':<{w2}}  {'control':>10}  {'candidate':>10}  "
                     f"{'delta':>10}  verdict")
        lines.append("-" * (w2 + 48))
        all_ok = True
        for label, cv, kv in pairs:
            good = kv is not None and cv is not None and kv <= cv
            all_ok = all_ok and good
            lines.append(f"{label:<{w2}}  {fmt(cv):>10}  {fmt(kv):>10}  "
                         f"{fmt(kv - cv, '{:+.3f}'):>10}  "
                         f"{'ok' if good else 'WORSE THAN CONTROL'}")
        gates["G-R4"] = (all_ok, "contact battery: all four candidate rates "
                                 "<= control" if all_ok else
                                 "contact battery: at least one candidate rate "
                                 "exceeds its control")
    lines.append("")

    # ---- G-R5 open-field regression -------------------------------------
    crms = ctrl["aggregate"].get("reference_tracking_rms_deg")
    krms = cand["aggregate"].get("reference_tracking_rms_deg")
    if crms is None or krms is None:
        gates["G-R5"] = (False, "ref RMS missing from one of the two JSONs")
    else:
        d = krms - crms
        gates["G-R5"] = (abs(d) <= GATE_REF_RMS_DELTA_DEG,
                         f"ref RMS {krms:.3f} deg vs control {crms:.3f} "
                         f"({d:+.3f}, bar |delta| <= {GATE_REF_RMS_DELTA_DEG})")

    lines += ["-- Gates ------------------------------------------------------------", ""]
    for gid in ("G-R1", "G-R2", "G-R3", "G-R4", "G-R5"):
        ok, text = gates[gid]
        verdict = "MANUAL" if ok is None else ("PASS" if ok else "FAIL")
        lines.append(f"{gid}  {verdict:<7} {text}")
    lines.append("")

    failed = [g for g, (ok, _) in gates.items() if ok is False]
    if failed:
        lines.append(f"OVERALL: FAIL ({', '.join(sorted(failed))})")
    else:
        lines.append("OVERALL: PASS")
    lines.append("(G-R3 is excluded from this verdict — it is a manual audit.)")

    return lines, gates


def to_markdown(lines):
    """Wrap the fixed-width report in a fenced block for pasting into a doc."""
    return "```\n" + "\n".join(lines) + "\n```"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--candidate", default="v5d_contact_wrench")
    p.add_argument("--control", default="v4_robust",
                   help="Prefix: <control>_grid6.json is the open-field control "
                        "and <control>_<battery>.json the four contact controls.")
    p.add_argument("--markdown", action="store_true",
                   help="Fence the report for pasting into the verdict doc.")
    args = p.parse_args(argv)

    entries = load_dir(args.results_dir)
    if not entries:
        print(f"no *.json in {args.results_dir}", file=sys.stderr)
        return 1

    problems = check_one_plant(entries)
    if problems:
        print("REFUSING TO SCORE — the results directory does not describe one "
              "robot:", file=sys.stderr)
        for t in problems:
            print(f"  - {t}", file=sys.stderr)
        print("  One results dir + one comparison table per robot model "
              "(AGENTS.md). Split the directory.", file=sys.stderr)
        return 2

    try:
        lines, gates = build_report(entries, args.candidate, args.control)
    except MissingResult as exc:
        print(f"missing required result file: {exc} in {args.results_dir}",
              file=sys.stderr)
        return 1

    print(to_markdown(lines) if args.markdown else "\n".join(lines))
    return 1 if any(ok is False for ok, _ in gates.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
