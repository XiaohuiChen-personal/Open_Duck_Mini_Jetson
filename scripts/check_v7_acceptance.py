#!/usr/bin/env python3
"""Score `v7_servo_safe` against the acceptance bars fixed BEFORE the run.

`docs/jetson-mod/v7_servo_fix_plan.md` §6 sets every bar in advance, precisely so
the verdict cannot be argued afterwards. This script reads the produced
artifacts and prints PASS/FAIL per bar. Exit 0 only if every HARD bar passes.

    python3 scripts/check_v7_acceptance.py

Pure stdlib + numpy. No Isaac, no GPU.

Design note: the contact bars are stated against **v6d**, not `v6_robust`. The
adversarial review of the plan showed that gating on `v6_robust` is close to
vacuous — it falls 100.000 % on the sustained wrench, so v7 could regress from
70.365 % to 99.9 % and still "beat the control". `v6_robust` is kept as an
additional floor, never the only one.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The plant changed (PLANT-11/PLANT-6, 2026-08-22), so the same policy must be
# re-scored against freshly produced artifacts. Point these at the new run
# rather than overwriting the old-plant record, which is the evidence for what
# the inflated armature did.
RES = os.environ.get("V7_RESULTS_DIR",
                     os.path.join(REPO, "docs", "jetson-mod", "eval_results_rebuild"))
ART = os.environ.get("V7_ARTIFACT_PREFIX", os.path.join(REPO, "v7"))

CONT_NM = 1.569
OVERLOAD_NM = 3.923
CLIP_NM = 4.903
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")

rows: list[tuple[str, str, str, bool, bool]] = []   # section, bar, got, ok, hard


def check(section, bar, got, ok, hard=True):
    rows.append((section, bar, got, bool(ok), hard))


def _leg(names):
    return [i for i, n in enumerate(names) if any(k in n for k in LEG)]


def torque(path):
    if not os.path.isfile(path):
        check("6a", "torque artifacts present", f"MISSING {os.path.basename(path)}", False)
        return
    d = np.load(path, allow_pickle=True)
    tau, names = d["tau"], [str(x) for x in d["names"]]
    legs = _leg(names)
    rms = np.sqrt((tau[:, legs] ** 2).mean(axis=0))
    worst = int(np.argmax(rms))
    check("6a", "worst leg RMS <= 1.0 N.m",
          f"{rms[worst]:.3f} ({names[legs[worst]]})", rms[worst] <= 1.0)

    p99 = np.percentile(np.abs(tau[:, legs]), 99, axis=0)
    at_clip = [names[legs[i]] for i in range(len(legs)) if p99[i] >= CLIP_NM - 1e-3]
    check("6a", "no leg joint p99 at the 4.903 clip",
          f"{at_clip if at_clip else 'none'}", not at_clip)

    worst_run = 0.0
    for i in legs:
        above = np.abs(tau[:, i]) > OVERLOAD_NM
        best = cur = 0
        for v in above:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        worst_run = max(worst_run, best * 0.02)
    check("6a", "longest run > 3.923 N.m is < 2.0 s", f"{worst_run:.2f} s", worst_run < 2.0)


def head(path, tag, straight):
    if not os.path.isfile(path):
        check("6b", f"head artifacts present ({tag})", f"MISSING {os.path.basename(path)}", False)
        return
    d = np.load(path, allow_pickle=True)
    q, tau, names = d["q"], d["tau"], [str(x) for x in d["names"]]
    k = names.index("neck_pitch")
    trav = np.degrees(q[:, k].max() - q[:, k].min())
    # the stop is read from the MJCF, same source PhysX was built from
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(REPO, "mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")).getroot()
    rng = {j.get("name"): [float(v) for v in j.get("range").split()]
           for j in root.iter("joint") if j.get("name") and j.get("range")}
    lo, hi = rng["neck_pitch"]
    tol = np.radians(0.5)
    on_stop = ((q[:, k] <= lo + tol) | (q[:, k] >= hi - tol)).mean() * 100
    trms = float(np.sqrt((tau[:, k] ** 2).mean()))
    yaw = names.index("head_yaw")
    ytrav = np.degrees(q[:, yaw].max() - q[:, yaw].min())

    if straight:
        check("6b", "neck_pitch travel >= 2.0 deg", f"{trav:.3f} deg", trav >= 2.0)
        check("6b", "neck_pitch on-stop <= 10 %", f"{on_stop:.1f} %", on_stop <= 10.0)
        check("6b", "neck_pitch torque RMS <= 1.5 N.m", f"{trms:.3f} N.m", trms <= 1.5)
        check("6b", "head_yaw travel >= 10 deg (not frozen)", f"{ytrav:.1f} deg", ytrav >= 10.0)
    else:
        check("6b", f"[{tag}] neck travel not degraded (>= 2 deg)", f"{trav:.3f} deg", trav >= 2.0)
        check("6b", f"[{tag}] head_yaw still moves (>= 5 deg)", f"{ytrav:.1f} deg", ytrav >= 5.0)


def gates():
    def agg(name):
        p = os.path.join(RES, f"{name}.json")
        return json.load(open(p))["aggregate"] if os.path.isfile(p) else None

    v7 = agg("v7_servo_safe")
    if v7 is None:
        check("6c", "gate battery present", "MISSING v7_servo_safe.json", False)
        return
    check("6c", "gait valid >= 5/6", f"{v7['gait_valid_conditions']}/6",
          v7["gait_valid_conditions"] >= 5)
    check("6c", "open-field falls <= 1.0 %", f"{v7['fall_rate_pct']:.3f} %",
          v7["fall_rate_pct"] <= 1.0)

    ctl = agg("v6_robust_grid6")
    if ctl:
        d = abs(v7["reference_tracking_rms_deg"] - ctl["reference_tracking_rms_deg"])
        check("6c", "ref RMS within 1.0 deg of v6_robust",
              f"{v7['reference_tracking_rms_deg']:.3f} vs {ctl['reference_tracking_rms_deg']:.3f}", d <= 1.0)

    BARS = [("pusheval_v4def", "push, v4 rule", 1.0),
            ("pusheval_v5def", "push, v5 rule", 1.0),
            ("wrencheval",     "sustained wrench", 80.0),
            ("obstacleeval",   "obstacle graze", 12.0)]
    for suf, label, bar in BARS:
        a = agg(f"v7_servo_safe_{suf}")
        if a is None:
            check("6c", f"{label} <= {bar} %", "MISSING", False); continue
        got = a["fall_rate_pct"]
        check("6c", f"{label} <= {bar} % (vs v6d)", f"{got:.3f} %", got <= bar)
        c = agg(f"v6_robust_{suf}")
        if c:
            check("6c", f"{label} also <= v6_robust {c['fall_rate_pct']:.3f} %",
                  f"{got:.3f} %", got <= c["fall_rate_pct"])


def main() -> int:
    torque(f"{ART}_torque.npz")
    head(f"{ART}_head_straight.npz", "straight", True)
    head(f"{ART}_head_turn.npz", "turn", False)
    gates()

    w = max(len(b) for _, b, _, _, _ in rows)
    cur = None
    for sec, bar, got, ok, hard in rows:
        if sec != cur:
            print(f"\n--- §{sec} ---"); cur = sec
        print(f"  [{'PASS' if ok else 'FAIL'}] {bar:<{w}}  {got}")
    failed = [b for _, b, _, ok, hard in rows if not ok and hard]
    print()
    if failed:
        print(f"VERDICT: FAIL — {len(failed)} hard bar(s): {failed}")
        return 1
    print(f"VERDICT: PASS — all {len(rows)} bars")
    print("NOTE: G-R3 (video) is a MANUAL frame-by-frame audit and is not scored here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
