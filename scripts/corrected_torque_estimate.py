#!/usr/bin/env python3
"""Estimate v7's REAL torque demand once the measured armature replaces 0.040.

    python3 scripts/corrected_torque_estimate.py

WHY AN ESTIMATE AND NOT A RE-SIMULATION
---------------------------------------
The definitive number requires re-running the policy against a corrected plant
(PLANT-11) and re-gating. This bounds the answer for free, so the decision to
spend that GPU time can be made on arithmetic rather than hope.

THE METHOD
----------
Joint torque splits into an inertial term and everything else (gravity,
Coriolis, contact):

    tau = M_jj * qdd  +  tau_other

Only M_jj changes. From `mj_fullM` on the shipped model:

    M_jj = armature + I_link        (armature is 87-99.7 % of it)

so the inertial term scales by

    ratio = (armature_measured + I_link) / (armature_configured + I_link)

`tau_other` is unchanged. Two bounds follow, and the truth is between them:

  LOWER  every N.m is inertial          -> tau_new = ratio * tau_old
  UPPER  a floor of tau_floor is NOT    -> tau_new = sqrt(tau_floor^2
         inertial and survives intact          + (ratio * tau_dyn)^2)
         where tau_dyn = sqrt(tau_old^2 - tau_floor^2)

The upper bound uses RMS addition, which assumes the gravity and inertial
components are uncorrelated over the gait. They are not exactly, so treat the
upper bound as approximate — it is deliberately the conservative side.
"""

from __future__ import annotations

import argparse
import math

ARMATURE_CONFIGURED = 0.040       # robot_cfg.py:120,135
ARMATURE_MEASURED = 0.00843       # bench, 2026-08-22, 12 trials, IQR 0.0005

# M_jj diagonal from mj_fullM on robot_motors.xml, minus armature -> I_link.
I_LINK = {
    "hip_yaw": 0.001669, "hip_roll": 0.006000, "hip_pitch": 0.004651,
    "knee": 0.001155, "ankle": 0.000104,
}

# servo_torque_budget.md section 4: the irreducible gravity/support load at the
# worst leg joint. Independently reproduced by floating-base inverse dynamics.
GAIT_FLOOR_NM = 0.69

# Datasheet ST-3250-C001 A/0
KT_NM_PER_A = 1.0787              # 5-11
RATED_CURRENT_A = 1.400           # 5-9
RATED_TORQUE_NM = 1.569           # 5-8
WARRANTY_TORQUE_NM = 0.981        # 8-1, 1/5 stall, 100k cycles at 33 % duty

# bench_results/sustained_torque.md
THERMAL_SUSTAINED_NM = 1.1
THERMAL_BAND_NM = (0.85, 1.5)


def link_inertia(joint: str) -> float | None:
    for key, value in I_LINK.items():
        if key in joint:
            return value
    return None


def inertial_ratio(joint: str) -> float | None:
    link = link_inertia(joint)
    if link is None:
        return None
    return (ARMATURE_MEASURED + link) / (ARMATURE_CONFIGURED + link)


def corrected(tau_old: float, ratio: float,
              floor_nm: float = GAIT_FLOOR_NM) -> tuple[float, float]:
    """Return (lower, upper) bounds on the corrected RMS torque."""
    lower = ratio * tau_old
    if tau_old <= floor_nm:
        return lower, tau_old                    # already at or below the floor
    tau_dyn = math.sqrt(tau_old ** 2 - floor_nm ** 2)
    upper = math.sqrt(floor_nm ** 2 + (ratio * tau_dyn) ** 2)
    return lower, max(lower, upper)


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="v7_torque.npz")
    args = ap.parse_args(argv)

    data = np.load(args.trace)
    rows = []
    for name, rms in zip(data["names"], data["rms"]):
        ratio = inertial_ratio(str(name))
        if ratio is None:
            continue                              # head/neck/antenna: not leg joints
        lo, hi = corrected(float(rms), ratio)
        rows.append((str(name), float(rms), ratio, lo, hi))
    rows.sort(key=lambda r: -r[4])

    print(f"armature: configured {ARMATURE_CONFIGURED} -> measured "
          f"{ARMATURE_MEASURED}  ({ARMATURE_CONFIGURED/ARMATURE_MEASURED:.2f}x)\n")
    print(f"{'joint':<18}{'v7 RMS':>9}{'ratio':>8}{'corrected lo':>14}{'corrected hi':>14}")
    print("-" * 63)
    for name, rms, ratio, lo, hi in rows:
        print(f"{name:<18}{rms:>9.3f}{ratio:>8.3f}{lo:>14.3f}{hi:>14.3f}")

    worst_lo, worst_hi = rows[0][3], rows[0][4]
    print(f"\nWORST LEG JOINT: {rows[0][0]}  ->  {worst_lo:.3f} – {worst_hi:.3f} N·m"
          f"  (was {rows[0][1]:.3f})")
    print()
    checks = [
        ("thermal sustained limit", THERMAL_SUSTAINED_NM,
         f"band {THERMAL_BAND_NM[0]}–{THERMAL_BAND_NM[1]}"),
        ("Feetech warranty load (8-1)", WARRANTY_TORQUE_NM, "100k cycles @ 33 % duty"),
        ("nameplate rated torque (5-8)", RATED_TORQUE_NM, "continuous rating"),
    ]
    for label, limit, note in checks:
        verdict = "PASS" if worst_hi <= limit else (
            "MARGINAL" if worst_lo <= limit else "FAIL")
        print(f"  {verdict:<9}{label:<32}{limit:>6.3f} N·m   ({note})")

    amps_lo, amps_hi = worst_lo / KT_NM_PER_A, worst_hi / KT_NM_PER_A
    print(f"\n  worst-joint RMS current: {amps_lo:.3f}–{amps_hi:.3f} A "
          f"= {amps_lo/RATED_CURRENT_A*100:.0f}–{amps_hi/RATED_CURRENT_A*100:.0f} % "
          f"of the {RATED_CURRENT_A} A continuous rating")
    print(f"  (before the correction: 1.910 A = 136 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
