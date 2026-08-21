#!/usr/bin/env python3
"""Derive the robot's electrical load from a measured torque trace.

Every number here is re-derived from primary sources at run time. The only
literals are the pack topology and the datasheet constants, each cited.

    python3 scripts/power_budget.py --trace v7_torque.npz
    python3 scripts/power_budget.py --trace v6d_torque_baseline.npz --markdown

WHY A SCRIPT AND NOT A TABLE IN A DOC
-------------------------------------
The torque trace changes with every retrain, and a stale wattage table is worse
than none — it looks authoritative. Point this at whichever `*_torque.npz` is
current and the budget follows.

RESISTANCE — 2.86 Ω IS REJECTED (2026-08-21)
--------------------------------------------
Datasheet ST-3250-C001 A/0 prints terminal resistance 1.2 Ω (5-10) AND stall
current 4.2 A at 12 V (5-5). 12/1.2 = 10 A, so 4.2 A is a driver clamp, not V/R.
An earlier revision therefore carried 12/4.2 = 2.86 Ω as a competing "total path"
figure. **That is rejected by reductio:** it implies 1.66 Ω of driver resistance
dissipating 1.66 x 4.2^2 = 29 W in the MOSFETs at stall, inside a 74.5 g servo
whose entire board idles at 0.232 W.

Use **R = 1.4 Ω** (1.2 measured, plus a hot-copper correction), band 1.2-1.6.
The 2.86 Ω column is retained ONLY as an audit upper bound -- never quote it as
a live estimate. See bench_results/sustained_torque.md §1.
"""

from __future__ import annotations

import argparse
import json

# --- datasheet ST-3250-C001 A/0, cited by line -----------------------------
KT_KGCM_PER_A = 11.0          # 5-11
KGCM_TO_NM = 9.80665 / 100.0  # 1 kg·cm = 0.0980665 N·m
KT_NM_PER_A = KT_KGCM_PER_A * KGCM_TO_NM
R_WINDING_OHM = 1.2           # 5-10 terminal resistance
STALL_CURRENT_A = 4.2         # 5-5
NOMINAL_V = 12.0              # 1-1 rated voltage, used only to derive R_total
R_TOTAL_OHM = NOMINAL_V / STALL_CURRENT_A   # 2.857 Ω — REJECTED, kept for audit
R_HOT_OHM = 1.6            # 1.2 Ω cold + hot-copper correction; the working upper
R_CENTRAL_OHM = 1.4        # use this

# --- measured on the bench 2026-08-21 ---------------------------------------
IDLE_CURRENT_A = 0.021        # per servo, torque disabled, 11.1 V

# --- pack + platform ---------------------------------------------------------
PACK_NOMINAL_V = 11.1         # 3S2P 18650, AGENTS.md:219
DCDC_EFFICIENCY = 0.90        # 11.1 V -> 19 V boost for the Jetson barrel jack
JETSON_MODES_W = (7, 15, 25)  # AGENTS.md:233

# Joints that are NOT STS3250. The antennas are micro servos on PWM.
NON_STS3250 = ("left_antenna", "right_antenna", "antenna_left", "antenna_right")


def winding_current_a(torque_nm: float) -> float:
    """Torque at the output shaft -> motor current. Kt already includes gearing:
    datasheet rated torque 16 kg·cm / rated current 1400 mA = 11.4 kg·cm/A,
    self-consistent with the 11 kg·cm/A of 5-11, so no 1/345 factor applies."""
    return torque_nm / KT_NM_PER_A


def dissipation_w(torque_nm: float, resistance_ohm: float) -> float:
    """At a static hold the mechanical output is zero, so all input is heat."""
    return winding_current_a(torque_nm) ** 2 * resistance_ohm


def load_trace(path: str) -> list[tuple[str, float, float]]:
    import numpy as np

    data = np.load(path)
    return [(str(n), float(r), float(p))
            for n, r, p in zip(data["names"], data["rms"], data["peak"])]


def budget(joints: list[tuple[str, float, float]]) -> dict:
    servos = [(n, r, p) for n, r, p in joints if n not in NON_STS3250]
    rows = []
    for name, rms, peak in servos:
        rows.append({
            "joint": name,
            "rms_nm": rms,
            "peak_nm": peak,
            "winding_a": winding_current_a(rms),
            "watts_r_winding": dissipation_w(rms, R_WINDING_OHM),
            "watts_r_total": dissipation_w(rms, R_TOTAL_OHM),
        })
    rows.sort(key=lambda r: -r["watts_r_total"])

    w_lo = sum(r["watts_r_winding"] for r in rows)
    w_hi = sum(r["watts_r_total"] for r in rows)
    quiescent_a = len(rows) * IDLE_CURRENT_A

    out = {
        "n_servos": len(rows),
        "joints": rows,
        "servo_watts": (w_lo, w_hi),
        "servo_amps": (w_lo / PACK_NOMINAL_V + quiescent_a,
                       w_hi / PACK_NOMINAL_V + quiescent_a),
        "quiescent_a": quiescent_a,
        "pack": {},
    }
    for mode_w in JETSON_MODES_W:
        jetson_a = mode_w / DCDC_EFFICIENCY / PACK_NOMINAL_V
        out["pack"][mode_w] = (out["servo_amps"][0] + jetson_a,
                               out["servo_amps"][1] + jetson_a)
    # Concentration: how much of the load sits in the top 6 joints.
    top6 = sum(r["watts_r_total"] for r in rows[:6])
    out["top6_share"] = top6 / w_hi if w_hi else 0.0
    return out


def report(b: dict, markdown: bool = False) -> str:
    lines = []
    bar = "|" if markdown else " "
    if markdown:
        lines.append("| joint | RMS N·m | peak N·m | I_w A | W @1.2Ω | W @2.86Ω |")
        lines.append("|---|---|---|---|---|---|")
    else:
        lines.append(f"{'joint':<18}{'RMS':>8}{'peak':>8}{'I_w A':>8}"
                     f"{'W@1.2':>9}{'W@2.86':>9}")
        lines.append("-" * 60)
    for r in b["joints"]:
        if markdown:
            lines.append(f"| `{r['joint']}` | {r['rms_nm']:.3f} | {r['peak_nm']:.3f} "
                         f"| {r['winding_a']:.3f} | {r['watts_r_winding']:.2f} "
                         f"| {r['watts_r_total']:.2f} |")
        else:
            lines.append(f"{r['joint']:<18}{r['rms_nm']:>8.3f}{r['peak_nm']:>8.3f}"
                         f"{r['winding_a']:>8.3f}{r['watts_r_winding']:>9.2f}"
                         f"{r['watts_r_total']:>9.2f}")
    lo, hi = b["servo_watts"]
    alo, ahi = b["servo_amps"]
    lines.append("")
    lines.append(f"{b['n_servos']} STS3250 servos")
    lines.append(f"  winding dissipation : {lo:6.2f} W  ..  {hi:6.2f} W")
    lines.append(f"  + quiescent          : {b['quiescent_a']:6.3f} A "
                 f"({IDLE_CURRENT_A*1000:.0f} mA each, measured)")
    lines.append(f"  SERVO BUS CURRENT    : {alo:6.3f} A  ..  {ahi:6.3f} A")
    lines.append(f"  top-6 joints are {b['top6_share']*100:.0f}% of servo power")
    lines.append("")
    for mode_w, (plo, phi) in b["pack"].items():
        lines.append(f"  + Jetson {mode_w:>2} W  => PACK {plo:6.3f} A .. {phi:6.3f} A")
    lines.append("")
    lines.append(f"  Kt = {KT_NM_PER_A:.4f} N·m/A")
    lines.append(f"  R  = {R_WINDING_OHM}-{R_HOT_OHM} Ω, central {R_CENTRAL_OHM} Ω. "
                 f"The @{R_TOTAL_OHM:.2f} Ω column is a REJECTED audit bound "
                 "(implies 29 W in the driver at stall) — do not quote it.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="v7_torque.npz")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json", help="write the budget here")
    args = ap.parse_args(argv)

    b = budget(load_trace(args.trace))
    print(f"# source: {args.trace}")
    print(report(b, args.markdown))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(b, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
