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
# The regressor x_inertial spans ~10x across a good sweep. If the measured
# torque does NOT vary by a comparable factor, the current signal is not
# tracking acceleration and there is nothing to fit. Measured 2026-08-22: a
# sweep came back with tau_rms in 0.037-0.057 (1.55x) while x_inertial spanned
# 10x, and still scored R^2 = 0.97 off a single outlier. R^2 alone cannot catch
# this, because a nearly-constant y has almost no variance to explain.
MIN_TAU_DYNAMIC_RANGE = 3.0
# addr69 quantises at 12.258 mA/count, so an RMS of a few counts is floor noise.
MIN_PEAK_I_RMS_COUNTS = 20   # at least one condition must be clear of it


def _fitted_amplitude_rad(samples: list[dict], omega: float) -> float | None:
    """Least-squares fit of position(t) = c0 + a*sin(wt) + b*cos(wt); return |A|.

    Fitting a sinusoid at the KNOWN drive frequency is far more robust than
    numerically differentiating a quantised encoder: one count over a 1.3 ms
    sample is ~770 counts/s of velocity noise, and the second derivative is
    worse still.
    """
    import numpy as np

    ts = np.array([s["t"] for s in samples], dtype=float)
    ys = np.array([s["position"] for s in samples], dtype=float)
    if len(ts) < 8:
        return None
    A = np.column_stack([np.sin(omega * ts), np.cos(omega * ts), np.ones_like(ts)])
    coeffs, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(math.hypot(coeffs[0], coeffs[1]) / COUNTS_PER_RAD)


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

        # Use the amplitude the servo ACTUALLY executed, not the one commanded.
        # Measured 2026-08-22: the closed-loop position controller attenuates
        # the command badly at high frequency (position span / goal span falls
        # from 1.00 at T=2.0 s to 0.26 at T=0.45 s). tau = J*qdd + b*qd holds for
        # the executed trajectory; fitting against the COMMAND measures the
        # servo's bandwidth instead of its inertia, and produces a negative J^2.
        amp_rad = _fitted_amplitude_rad(used, omega)
        if amp_rad is None or amp_rad <= 0:
            continue
        out.append({
            "amplitude_counts": amplitude,
            "amplitude_executed_counts": amp_rad * COUNTS_PER_RAD,
            "tracking_ratio": amp_rad * COUNTS_PER_RAD / amplitude,
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

    taus = [c["tau_rms_nm"] for c in usable]
    tau_range = max(taus) / min(taus) if min(taus) > 0 else float("inf")
    x_range = (max(c["x_inertial"] for c in usable)
               / min(c["x_inertial"] for c in usable))
    peak_counts = max(c["i_rms_a"] for c in usable) / CURRENT_LSB_A
    min_counts = min(c["i_rms_a"] for c in usable) / CURRENT_LSB_A
    result["tau_dynamic_range"] = tau_range
    result["x_inertial_range"] = x_range
    result["min_i_rms_counts"] = min_counts
    result["peak_i_rms_counts"] = peak_counts

    if peak_counts < MIN_PEAK_I_RMS_COUNTS:
        result["verdict"] = (
            f"SIGNAL TOO SMALL: the strongest condition reads only "
            f"{peak_counts:.1f} counts of I_rms (need >= {MIN_PEAK_I_RMS_COUNTS}). "
            "addr69 quantises at 12.258 mA, so this is floor noise, not a "
            "measurement. Drive LARGER amplitudes."
        )
        return result
    if tau_range < MIN_TAU_DYNAMIC_RANGE:
        result["verdict"] = (
            f"NO DYNAMIC RANGE: measured torque varies only {tau_range:.2f}x while "
            f"the inertial regressor varies {x_range:.1f}x. The current is not "
            "tracking acceleration, so there is no inertia to extract. R^2 is "
            "meaningless here — a nearly-constant y has no variance to explain. "
            "Drive LARGER amplitudes so the inertial term dominates friction."
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


# --- step-response identification -------------------------------------------
# The frequency sweep could not work: a smooth sinusoid keeps the position error
# small, so the proportional controller commands little duty and addr69 sat at
# ~3 counts (the 12.258 mA quantisation floor) across every condition. A large
# step saturates torque (load = 1000) and peaks near 180 counts = 2.2 A.
#
# During the launch transient the servo starts from rest at constant saturated
# torque, so q(t) = q0 + 0.5*a*t^2 and J = (Kt*I - tau_friction) / a.
STEP_WINDOW_S = 0.025      # fixed for EVERY trial, so none can include cruise
STEP_MIN_SAMPLES = 12
STEP_MIN_R2 = 0.98
STEP_FRICTION_NM = 0.15    # bench-measured 0.1-0.2; applied as a correction


def fit_step_trials(rows: list[dict], friction_nm: float = STEP_FRICTION_NM) -> dict:
    """Identify J from step responses. Returns per-trial results and a summary."""
    import numpy as np

    trials = []
    for trial_id in sorted({r["trial"] for r in rows}):
        samples = sorted([r for r in rows if r["trial"] == trial_id],
                         key=lambda r: r["t"])
        t = np.array([r["t"] for r in samples], dtype=float)
        disp = np.abs(np.array([r["position"] for r in samples], dtype=float)
                      - samples[0]["position"])
        current = np.array([abs(r["current_raw"]) for r in samples], dtype=float)

        moving = np.where(disp > 2)[0]
        if len(moving) == 0:
            continue
        lo = int(moving[0])
        # A FIXED window from motion onset. Selecting it per-trial by velocity
        # let cruise contaminate the long steps: measured 2026-08-22, windows of
        # 130 ms returned J = 0.044-0.050 while 18-37 ms windows on the same
        # servo returned 0.007-0.010.
        hi = int(np.searchsorted(t, t[lo] + STEP_WINDOW_S))
        if hi - lo < STEP_MIN_SAMPLES or hi > len(t):
            continue

        tw = t[lo:hi] - t[lo]
        dw = disp[lo:hi] - disp[lo]
        A = np.column_stack([0.5 * tw ** 2, tw, np.ones_like(tw)])
        coeffs, *_ = np.linalg.lstsq(A, dw, rcond=None)
        accel = coeffs[0] / COUNTS_PER_RAD
        pred = A @ coeffs
        ss_tot = float(((dw - dw.mean()) ** 2).sum())
        r2 = 1 - float(((dw - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0

        i_mean = float(current[lo:hi].mean()) * CURRENT_LSB_A
        tau_gross = KT_NM_PER_A * i_mean
        tau_net = tau_gross - friction_nm
        trials.append({
            "trial": trial_id, "size": samples[0].get("size"),
            "window_ms": float(tw[-1] * 1000), "n": hi - lo,
            "accel_rad_s2": float(accel), "i_mean_a": i_mean,
            "tau_gross_nm": tau_gross, "tau_net_nm": tau_net,
            "j": tau_net / accel if accel > 0 else None,
            "r2": r2,
            "usable": bool(r2 >= STEP_MIN_R2 and accel > 0 and tau_net > 0),
        })

    good = [x["j"] for x in trials if x["usable"] and x["j"] is not None]
    result = {"method": "step", "trials": trials, "n_usable": len(good),
              "friction_nm": friction_nm, "window_s": STEP_WINDOW_S}
    if len(good) < 4:
        result["verdict"] = (f"INSUFFICIENT: {len(good)} usable trials, need 4.")
        return result
    arr = np.sort(np.array(good))
    result.update({
        "armature_measured": float(np.median(arr)),
        "spread_min": float(arr[0]), "spread_max": float(arr[-1]),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "ratio_to_configured": CONFIGURED_ARMATURE / float(np.median(arr)),
        "viscous_measured": float("nan"),
        "r2": float(np.mean([x["r2"] for x in trials if x["usable"]])),
        "verdict": "OK",
    })
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_json")
    ap.add_argument("--out", help="write the fit result here")
    args = ap.parse_args(argv)

    with open(args.sweep_json) as fh:
        payload = json.load(fh)
    rows = payload["samples"] if isinstance(payload, dict) else payload

    if rows and "trial" in rows[0]:                    # step-response log
        result = fit_step_trials(rows)
        print(f"{'trial':>6}{'size':>6}{'win ms':>8}{'n':>5}{'a rad/s2':>10}"
              f"{'I A':>7}{'tau net':>9}{'J':>9}{'R2':>7}{'':>3}")
        for x in result["trials"]:
            flag = "" if x["usable"] else "  drop"
            j = f"{x['j']:.4f}" if x["j"] is not None else "--"
            print(f"{x['trial']:>6}{x['size']:>6}{x['window_ms']:>8.1f}{x['n']:>5}"
                  f"{x['accel_rad_s2']:>10.1f}{x['i_mean_a']:>7.3f}"
                  f"{x['tau_net_nm']:>9.3f}{j:>9}{x['r2']:>7.3f}{flag}")
        print()
        if result.get("armature_measured"):
            print(f"  n usable = {result['n_usable']},  spread "
                  f"{result['spread_min']:.4f}–{result['spread_max']:.4f},  "
                  f"IQR {result['iqr']:.4f}")
            print(f"  friction correction applied: {result['friction_nm']} N·m\n")
        for line in interpret(result):
            print(line)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(result, fh, indent=2)
            print(f"\nwrote {args.out}")
        return 0 if result.get("verdict") == "OK" else 1

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
