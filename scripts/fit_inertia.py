#!/usr/bin/env python3
"""Identify reflected rotor inertia from a frequency sweep. Settles PLANT-11.

    python3 scripts/fit_inertia.py docs/jetson-mod/bench_results/inertia_sweep.json

THE QUESTION
------------
`robot_cfg.py:120,135` sets `armature = 0.040`, which is 87-99.7 % of the
simulated joint inertia -- 386x the real link at the ankle. If it is 2-5x high,
as the rotor geometry suggests, then the v7 policy's 2.060 N.m worst-joint RMS
is largely an artifact of accelerating a phantom flywheel, and the entire
servo-torque problem may not exist. See known_issues.md PLANT-11.

THE METHOD
----------
Drive the unloaded output shaft with q = A*sin(w t). Then:

    q_dot     = A w cos(w t)        RMS = A w / sqrt(2)
    q_dot_dot = -A w^2 sin(w t)     RMS = A w^2 / sqrt(2)

and the torque is tau = J*q_dot_dot + b*q_dot + (Coulomb + offset). Because
sin and cos are orthogonal over whole cycles, the mean squares add:

    tau_rms^2 = (J * A w^2 / sqrt2)^2 + (b * A w / sqrt2)^2 + c

which is LINEAR in (J^2, b^2, c). Fit by least squares across the sweep, where
w spans 3.1-14.0 rad/s so the inertial term (w^2) grows ~20x while the viscous
term (w) grows only ~4.5x -- that spread is what separates them.

Torque comes from the servo's own current register:
    tau = Kt * I,  Kt = 1.0787 N.m/A,  I = addr69 * 12.258 mA/count

WHAT MAKES A FIT TRUSTWORTHY
----------------------------
- >= 10 samples per cycle, or the RMS of a sinusoid is aliased
- >= 6 (amplitude, period) conditions, or (J^2, b^2, c) is underdetermined
- J^2 and b^2 both non-negative; a negative fit means the model is wrong here
- R^2 >= 0.9
All four are checked and reported; the script does not hand back a number it
cannot defend.
"""

from __future__ import annotations

import argparse
import json
import math

KT_NM_PER_A = 1.0787          # datasheet 5-11, at the OUTPUT shaft
CURRENT_LSB_A = 0.012258      # resolved: addr28=310 <-> the documented 3.80 A trip
COUNTS_PER_REV = 4096
COUNTS_PER_RAD = COUNTS_PER_REV / (2 * math.pi)

# robot_cfg.py:120,135
CONFIGURED_ARMATURE = 0.040
# 0.040 / 345^2 = 3.36e-7 kg.m^2 rotor; a 3-5 g coreless cup at r=5-6 mm gives
# 0.75-1.8e-7, i.e. an armature of 0.009-0.021.
PLAUSIBLE_ARMATURE = (0.009, 0.021)

MIN_SAMPLES_PER_CYCLE = 10
MIN_CONDITIONS = 6
MIN_R2 = 0.90


def group_conditions(rows: list[dict]) -> list[dict]:
    """One entry per (amplitude, period), with RMS current over whole cycles."""
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault((r["amplitude"], r["period"]), []).append(r)

    out = []
    for (amplitude, period), samples in sorted(buckets.items()):
        samples.sort(key=lambda s: s["t"])
        span = samples[-1]["t"] - samples[0]["t"]
        if span <= 0:
            continue
        rate = len(samples) / span
        # Truncate to whole cycles: a partial cycle biases the RMS.
        whole = math.floor(span / period)
        if whole < 1:
            continue
        cutoff = samples[0]["t"] + whole * period
        used = [s for s in samples if s["t"] <= cutoff]
        currents = [abs(s["current_raw"]) * CURRENT_LSB_A for s in used
                    if s.get("current_raw") is not None]
        if not currents:
            continue
        i_rms = math.sqrt(sum(c * c for c in currents) / len(currents))
        omega = 2 * math.pi / period
        amp_rad = amplitude / COUNTS_PER_RAD
        out.append({
            "amplitude_counts": amplitude,
            "period_s": period,
            "omega": omega,
            "amp_rad": amp_rad,
            "n": len(used),
            "rate_hz": rate,
            "samples_per_cycle": rate * period,
            "cycles_used": whole,
            "i_rms_a": i_rms,
            "tau_rms_nm": KT_NM_PER_A * i_rms,
            # regressors
            "x_inertial": (amp_rad * omega ** 2 / math.sqrt(2)) ** 2,
            "x_viscous": (amp_rad * omega / math.sqrt(2)) ** 2,
        })
    return out


def fit(conditions: list[dict]) -> dict:
    """Least squares for (J^2, b^2, c) against tau_rms^2."""
    import numpy as np

    usable = [c for c in conditions
              if c["samples_per_cycle"] >= MIN_SAMPLES_PER_CYCLE]
    dropped = [c for c in conditions if c not in usable]

    result = {
        "n_conditions": len(conditions),
        "n_usable": len(usable),
        "dropped_for_aliasing": [
            {"amplitude_counts": c["amplitude_counts"], "period_s": c["period_s"],
             "samples_per_cycle": round(c["samples_per_cycle"], 1)} for c in dropped
        ],
    }
    if len(usable) < MIN_CONDITIONS:
        result["verdict"] = (
            f"INSUFFICIENT: {len(usable)} usable conditions, need {MIN_CONDITIONS}. "
            "Three unknowns cannot be identified from fewer."
        )
        return result

    A = np.array([[c["x_inertial"], c["x_viscous"], 1.0] for c in usable])
    y = np.array([c["tau_rms_nm"] ** 2 for c in usable])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    j_sq, b_sq, c0 = coeffs

    pred = A @ coeffs
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result.update({
        "j_squared": float(j_sq), "b_squared": float(b_sq), "offset": float(c0),
        "r2": r2,
        "conditions": usable,
    })

    if j_sq < 0 or b_sq < 0:
        result["verdict"] = (
            f"MODEL REJECTED: fit gave J^2={j_sq:.4g}, b^2={b_sq:.4g}. A negative "
            "square means this model does not describe the data — do not read an "
            "inertia out of it."
        )
        return result
    if r2 < MIN_R2:
        result["verdict"] = f"POOR FIT: R^2 = {r2:.3f} < {MIN_R2}. Not trustworthy."
        return result

    j = math.sqrt(j_sq)
    result["armature_measured"] = j
    result["viscous_measured"] = math.sqrt(b_sq)
    result["ratio_to_configured"] = CONFIGURED_ARMATURE / j if j > 0 else float("inf")
    result["verdict"] = "OK"
    return result


def interpret(fit_result: dict) -> list[str]:
    """The branch table from known_issues.md PLANT-11."""
    j = fit_result.get("armature_measured")
    if j is None:
        return [f"NO NUMBER: {fit_result.get('verdict')}"]

    lines = [
        f"measured armature      = {j:.5f} kg·m²",
        f"configured             = {CONFIGURED_ARMATURE} "
        f"({fit_result['ratio_to_configured']:.2f}x the measurement)",
        f"geometry-plausible band= {PLAUSIBLE_ARMATURE[0]}–{PLAUSIBLE_ARMATURE[1]}",
        f"viscous b              = {fit_result['viscous_measured']:.4f} N·m·s/rad",
        f"R²                     = {fit_result['r2']:.4f}",
        "",
    ]
    if j >= 0.032:
        lines += [
            "VERDICT: the configured armature is broadly RIGHT.",
            "  PLANT-11 does not explain the torque figures. The 2.060 N·m stands,",
            "  the servo problem is real, and the next lever is the reference gait",
            "  -- not the reward weight.",
        ]
    elif j <= 0.014:
        lines += [
            "VERDICT: the configured armature is BADLY high.",
            "  The v7 policy is likely already near 0.75–0.90 N·m on real hardware.",
            "  Fix the plant, RE-GATE, and do not retrain for thermal reasons.",
        ]
    else:
        lines += [
            "VERDICT: high, but not enough to close the whole gap on its own.",
            "  Fix the plant, re-gate, and expect a retrain aimed at ~1.2 N·m",
            "  rather than an open-ended torque-reduction campaign.",
        ]
    lines += [
        "",
        "In every branch the plant change invalidates the policy's calibration and",
        "every stored gate number, so a re-gate is required regardless.",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_json")
    ap.add_argument("--out", help="write the fit result here")
    args = ap.parse_args(argv)

    with open(args.sweep_json) as fh:
        payload = json.load(fh)
    rows = payload["samples"] if isinstance(payload, dict) else payload

    conditions = group_conditions(rows)
    print(f"{'A cnt':>6}{'T s':>6}{'w':>7}{'n':>6}{'Hz':>7}{'/cyc':>7}"
          f"{'I_rms A':>9}{'tau_rms':>9}")
    print("-" * 57)
    for c in conditions:
        print(f"{c['amplitude_counts']:>6}{c['period_s']:>6.2f}{c['omega']:>7.2f}"
              f"{c['n']:>6}{c['rate_hz']:>7.0f}{c['samples_per_cycle']:>7.1f}"
              f"{c['i_rms_a']:>9.3f}{c['tau_rms_nm']:>9.3f}")
    print()

    result = fit(conditions)
    for line in interpret(result):
        print(line)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0 if result.get("verdict") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
