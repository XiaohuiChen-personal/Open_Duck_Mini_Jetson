#!/usr/bin/env python3
# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the v5 reference gait library by filling the missing grid rows.

WHY (Task 2.8 root cause 5, docs/jetson-mod/task_plan.md:1513-1519)
--------------------------------------------------------------------
The frozen library ``polynomial_coefficients.pkl`` is a full 6x4x10 Cartesian
grid over (vx, vy, wz) with **no vy = 0 and no wz = 0 cells**:

    vx: -0.148 -0.074  0.000  0.074  0.148  0.222
    vy: -0.111 -0.037         0.037  0.111          <- no 0
    wz: -1.111 -0.852 -0.593 -0.333 -0.074  0.185 0.444 0.704 0.963 1.222   <- no 0

``ImitationReward`` snaps each command to the nearest cell by unweighted L2,
so the two most common deployment commands land on gaits that do not match:

    turn in place (0, 0, 0.5)  ->  (0.0, -0.037, 0.444)   side-steps, turns 11% slow
    straight walk (0.2, 0, 0)  ->  (0.222, -0.037, -0.074) curves right

This script writes ``polynomial_coefficients_v2.pkl``, extending the grid to

    vy: -0.111 -0.037  0.000  0.037  0.111                     (5)
    wz: -1.111 -0.852 -0.593 -0.500 -0.333 -0.074  0.000
         0.185  0.444  0.500  0.704  0.963  1.222              (13)

= 6 x 5 x 13 = 390 cells, minus 2 quarantined (below) = 388: 240 original
carried over with bit-exact values, 148 synthesized. wz = +/-0.5 is added
because it is the deployment rotation hull limit (duck-embody
``COMMAND_HULL``) and the command 5 of 10 benchmark falls died under.

The output stores plain Python floats rather than the source's
``numpy.float64`` (see ``to_plain`` — re-pickling numpy scalars under numpy
2.x makes the file unreadable inside Isaac Sim's numpy 1.x). Values are
unchanged; both types are IEEE-754 doubles.

METHOD — bilinear coefficient interpolation
-------------------------------------------
Polynomial evaluation is linear in the coefficients and every cell in the
library shares the same period (0.54 s), fps (50) and nb_steps_in_period (27),
so a convex combination of coefficient tensors is itself a valid gait whose
evaluated trajectory is the same convex combination of the parents'
trajectories (hence automatically inside their envelope).

Synthesized cells are bilinear in (vy, wz) over the bracketing original cells;
vx is never interpolated (every needed vx already exists). Original cells are
copied verbatim, never re-derived.

VALIDATION (``--check``, and run automatically after generation)
---------------------------------------------------------------
The load-bearing evidence is leave-one-out: rebuild an EXISTING cell from its
two neighbours and measure the error against the real one. Measured on the
frozen library (2026-07-28):

    wz interpolation: 0.04 - 0.41 deg RMS over the 10 leg joints
    vy interpolation: 0.24 - 0.38 deg RMS
    (for scale: v4_robust's own reference-tracking RMS is 4.49 deg)

i.e. interpolation error is an order of magnitude below the policy's tracking
error, on cells the library itself provides as ground truth.

Four regions of the source grid are NOT smooth (LOO RMS 16-57 deg), all at
extreme forward+lateral+turn corners where the upstream walk engine changes
gait qualitatively, and all outside the +/-0.5 wz command hull:

    wz-axis  (0.222,  0.111,  0.963)   56.6 deg
    wz-axis  (0.222,  0.111,  0.704)   28.4 deg
    vy-axis  (0.222, -0.037, -1.111)   16.0 deg
    wz-axis  (0.222, -0.111, -0.852)   15.9 deg

They are DETECTED by ``find_irregular()`` (both axes; the vy scan is what
catches the fourth, since the wz scan cannot test an axis endpoint), and the
two synthesized cells that would have been built across one — 0.222_0.0_-1.111
and 0.222_0.111_0.5 — are skipped instead. A hole in the grid is safe: the
nearest-neighbour lookup simply picks the next cell. Silently interpolating
across a gait discontinuity would not be.

Note on what CANNOT be checked here: no polynomial dimension usefully encodes
yaw rate (dims 37-39 are world angular velocity, but the wz = 1.222 cell
averages only +0.041 there), so this script cannot verify that the synthesized
(0, 0, 0.5) cell truly rotates at 0.5 rad/s. That is confirmed downstream by
the frame-by-frame audit of the trained policy (plan section 8), not here.

Dim layout (40): joints_pos 0-15 | joints_vel 16-31 | foot_contacts 32-33 |
world_linear_vel 34-36 | world_angular_vel 37-39.  Playground joint order.

USAGE
-----
    python3 scripts/patch_reference_library.py            # generate + validate
    python3 scripts/patch_reference_library.py --check    # validate only
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "isaac_lab_env" / "open_duck_mini_v2" / "data"
SRC_PKL = DATA_DIR / "polynomial_coefficients.pkl"
DST_PKL = DATA_DIR / "polynomial_coefficients_v2.pkl"

# Provenance pin of the immutable source dataset (data/README.md).
SRC_MD5 = "7a5515dce7610094bc8da28cfb690a07"

N_DIMS = 40
N_COEFFS = 16

# Playground-order indices of the 10 leg joints (the only ones the imitation
# reward tracks; see imitation_reward.py LEG_JOINT_NAMES).
LEG_DIMS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]

# Grid additions.
NEW_VY = [0.0]
NEW_WZ = [-0.5, 0.0, 0.5]

# Regions of the source grid where the underlying walk engine is NOT smooth:
# leave-one-out rebuilds there miss by 16-57 deg, against <= 0.42 deg
# everywhere else. All sit at extreme forward+lateral+turn corners, and all lie
# outside the trained command hull (|wz| >= 0.70 or the vy extremes, vs a
# +/-0.5 wz command range). These are DETECTED at run time by
# `find_irregular()`, not trusted from this list; the list is the expected
# result and a mismatch is a hard failure. Synthesized cells that would descend
# from one are skipped rather than silently built on bad data.
EXPECTED_IRREGULAR = {
    ("wz", 0.222, -0.111, -0.852),
    ("wz", 0.222, 0.111, 0.704),
    ("wz", 0.222, 0.111, 0.963),
    ("vy", 0.222, -0.037, -1.111),
}
LOO_ANOMALY_DEG = 2.0

# Above this |wz| the reference's world-frame linear-velocity dims (34-36) stop
# approximating the commanded BODY-frame velocity, because the recording frame
# does not co-rotate with the robot: at vx=0.222 the cycle-mean swings from
# (+0.232, -0.046) m/s at wz ~ 0 to (+0.122, +0.154) at wz = 1.222 while |v|
# stays ~0.2. That is a property of the frozen library which v3/v4 already
# trained against (ImitationReward compares body-frame root_lin_vel_b to these
# dims, imitation_reward.py:253-257) and which this patch inherits unchanged;
# it only limits where a "vy = 0 means zero lateral velocity" assertion is
# meaningful.
BODY_FRAME_VALID_WZ = 0.2


def fmt(v: float) -> str:
    """Library key component formatting (matches the frozen keys exactly)."""
    return str(round(float(v), 3))


def key_of(vx: float, vy: float, wz: float) -> str:
    return f"{fmt(vx)}_{fmt(vy)}_{fmt(wz)}"


def coeff_matrix(entry: dict) -> np.ndarray:
    """(40, 16) coefficient tensor, constant term first."""
    return np.array(
        [entry["coefficients"][f"dim_{i}"] for i in range(N_DIMS)], dtype=np.float64
    )


def coeff_dict(mat: np.ndarray) -> dict:
    return {f"dim_{i}": mat[i].tolist() for i in range(N_DIMS)}


def to_plain(entry: dict) -> dict:
    """Strip every numpy scalar out of a library entry.

    The frozen library stores its coefficients as lists of ``numpy.float64``.
    Copying those objects into a new pickle written under numpy 2.x embeds
    ``numpy._core`` module references, which the numpy 1.x inside Isaac Sim's
    kit python cannot import — the training run dies at reward-manager
    construction with ``ModuleNotFoundError: No module named 'numpy._core'``.
    (Found exactly that way, by stepping the env rather than by parsing it.)

    Converting to built-in floats is exact — ``numpy.float64`` and Python
    ``float`` are both IEEE-754 doubles — and makes the artifact readable by
    any numpy, which a long-lived dataset should be anyway.
    """
    out = dict(entry)
    out["coefficients"] = {
        k: [float(c) for c in v] for k, v in entry["coefficients"].items()
    }
    for key, value in list(out.items()):
        if key == "coefficients":
            continue
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, dict):
            out[key] = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in value.items()}
    return out


def evaluate(mat: np.ndarray, n_steps: int = 27) -> np.ndarray:
    """Evaluate all dims over one gait cycle at normalized phase t in [0, 1).

    Mirrors ImitationReward's Horner evaluation and its phase convention
    t = (i % nb_steps) / nb_steps (imitation_reward.py:213-216).
    """
    ts = np.arange(n_steps) / float(n_steps)
    out = np.repeat(mat[:, N_COEFFS - 1 : N_COEFFS], n_steps, axis=1)
    for i in range(N_COEFFS - 2, -1, -1):
        out = out * ts[None, :] + mat[:, i : i + 1]
    return out  # (40, n_steps)


def bracket(value: float, axis: list[float]) -> tuple[float, float, float]:
    """Return (lo, hi, weight_on_lo) bracketing `value` on `axis`."""
    if value in axis:
        return value, value, 1.0
    lo = max((a for a in axis if a < value), default=None)
    hi = min((a for a in axis if a > value), default=None)
    if lo is None or hi is None:
        raise ValueError(f"{value} is outside the library hull {axis[0]}..{axis[-1]}")
    return lo, hi, (hi - value) / (hi - lo)


def load_source() -> tuple[dict, list[float], list[float], list[float]]:
    raw = pickle.loads(SRC_PKL.read_bytes())
    md5 = hashlib.md5(SRC_PKL.read_bytes()).hexdigest()
    if md5 != SRC_MD5:
        raise SystemExit(
            f"REFUSING: source library md5 {md5} != pinned {SRC_MD5} (data/README.md).\n"
            "The frozen library is an immutable dataset; investigate before proceeding."
        )
    vx = sorted({float(k.split("_")[0]) for k in raw})
    vy = sorted({float(k.split("_")[1]) for k in raw})
    wz = sorted({float(k.split("_")[2]) for k in raw})
    # Every key must round-trip through our formatter, or synthesized keys
    # would silently not match the parse in ImitationReward.
    for a in vx:
        for b in vy:
            for c in wz:
                if key_of(a, b, c) not in raw:
                    raise SystemExit(f"source grid is not the full product: missing {key_of(a, b, c)}")
    return raw, vx, vy, wz


def loo_error_deg(raw: dict, target: tuple, p_lo: tuple, p_hi: tuple, w_lo: float) -> float:
    """Rebuild an existing cell from two real neighbours; leg-joint RMS in degrees."""
    pred = w_lo * coeff_matrix(raw[key_of(*p_lo)]) + (1.0 - w_lo) * coeff_matrix(raw[key_of(*p_hi)])
    return float(np.degrees(np.sqrt(np.mean(
        (evaluate(coeff_matrix(raw[key_of(*target)]))[LEG_DIMS] - evaluate(pred)[LEG_DIMS]) ** 2))))


def find_irregular(raw: dict, vx_axis, vy_axis, wz_axis) -> tuple[set, dict, dict]:
    """Detect grid regions where linear interpolation demonstrably fails.

    Scans both interpolatable axes. Returns (flagged set, wz errors, vy errors)
    where the flagged set holds ("wz"|"vy", vx, vy, wz) tuples naming the cell
    the rebuild missed.
    """
    flagged, errs_wz, errs_vy = set(), {}, {}

    for vx in vx_axis:
        for vy in vy_axis:
            for i in range(1, len(wz_axis) - 1):
                lo, mid, hi = wz_axis[i - 1], wz_axis[i], wz_axis[i + 1]
                e = loo_error_deg(raw, (vx, vy, mid), (vx, vy, lo), (vx, vy, hi),
                                  (hi - mid) / (hi - lo))
                errs_wz[(vx, vy, mid)] = e
                if e > LOO_ANOMALY_DEG:
                    flagged.add(("wz", vx, vy, mid))

    # The vy axis has only four values, so the sole possible rebuild spans the
    # full 0.148 width (the real vy=0 blend spans half that) — a conservative
    # over-estimate of the error, which is what we want for a safety scan.
    w = (0.037 - (-0.037)) / (0.037 - (-0.111))
    for vx in vx_axis:
        for wz in wz_axis:
            e = loo_error_deg(raw, (vx, -0.037, wz), (vx, -0.111, wz), (vx, 0.037, wz), w)
            errs_vy[(vx, -0.037, wz)] = e
            if e > LOO_ANOMALY_DEG:
                flagged.add(("vy", vx, -0.037, wz))

    return flagged, errs_wz, errs_vy


def uniform_metadata(raw: dict) -> dict:
    """Assert the per-entry metadata is uniform and return it."""
    fields = ("period", "fps", "frame_offsets", "start_offset", "nb_steps_in_period",
              "startend_double_support_ratio")
    first = next(iter(raw.values()))
    meta = {f: first[f] for f in fields}
    for k, e in raw.items():
        for f in fields:
            if e[f] != meta[f]:
                raise SystemExit(
                    f"metadata field {f!r} is not uniform across the library "
                    f"({k}: {e[f]!r} vs {meta[f]!r}) — bilinear blending assumes it is."
                )
    return meta


def synthesize(raw, vx_axis, vy_axis, wz_axis, meta, flagged) -> tuple[dict, list, list]:
    """Build the extended library.

    Returns (library, synthesis records, skipped records). A cell is skipped
    when any parent it would be built from sits in a region `find_irregular`
    flagged — better a hole in the grid (nearest-neighbour lookup copes) than a
    plausible-looking gait interpolated across a discontinuity.
    """
    out: dict = {}
    records, skipped = [], []

    bad_coords = {(round(vx, 3), round(vy, 3), round(wz, 3)) for _, vx, vy, wz in flagged}

    new_vy_axis = sorted(set(vy_axis) | set(NEW_VY))
    new_wz_axis = sorted(set(wz_axis) | set(NEW_WZ))

    for vx in vx_axis:
        for vy in new_vy_axis:
            for wz in new_wz_axis:
                key = key_of(vx, vy, wz)
                if key in raw:
                    # Values copied exactly; only the numeric TYPE is normalised
                    # to built-in float (see to_plain) so the artifact carries no
                    # numpy version dependency.
                    out[key] = to_plain(raw[key])
                    continue

                vy_lo, vy_hi, w_vy = bracket(vy, vy_axis)
                wz_lo, wz_hi, w_wz = bracket(wz, wz_axis)

                parents, weights, coords = [], [], []
                for b, wb in ((vy_lo, w_vy), (vy_hi, 1.0 - w_vy)):
                    if wb == 0.0:
                        continue
                    for c, wc in ((wz_lo, w_wz), (wz_hi, 1.0 - w_wz)):
                        if wc == 0.0:
                            continue
                        parents.append(key_of(vx, b, c))
                        weights.append(wb * wc)
                        coords.append((round(vx, 3), round(b, 3), round(c, 3)))

                contaminated = [c for c in coords if c in bad_coords]
                if contaminated:
                    skipped.append({"key": key, "because": contaminated})
                    continue

                mats = [coeff_matrix(raw[p]) for p in parents]
                blend = sum(w * m for w, m in zip(weights, mats))

                traj = [evaluate(m)[LEG_DIMS] for m in mats]
                spread = np.degrees(
                    np.sqrt(np.mean((np.max(traj, axis=0) - np.min(traj, axis=0)) ** 2))
                )

                out[key] = to_plain({"coefficients": coeff_dict(blend), **meta})
                records.append(
                    {"key": key, "parents": parents, "parent_coords": coords,
                     "weights": [round(w, 5) for w in weights],
                     "parent_spread_deg": round(float(spread), 3)}
                )

    return out, records, skipped


def report_and_validate(lib, raw, records, skipped, flagged, errs_wz, errs_vy) -> bool:
    """Print the validation report. Returns True if every check passed."""
    ok = True

    def check(passed: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))

    print(f"\nLibrary: {len(raw)} original + {len(records)} synthesized = {len(lib)} cells"
          + (f"  ({len(skipped)} skipped)" if skipped else ""))

    print("\n1. Original cells preserved exactly (values, as plain floats)")
    def same(a, b):
        if set(a["coefficients"]) != set(b["coefficients"]):
            return False
        for k in a["coefficients"]:
            xs, ys = a["coefficients"][k], b["coefficients"][k]
            if len(xs) != len(ys) or any(float(x) != float(y) for x, y in zip(xs, ys)):
                return False
        return all(a[f] == b[f] for f in a if f != "coefficients")
    identical = all(same(lib[k], raw[k]) for k in raw)
    check(identical, "all 240 original entries bit-exact after float() normalisation")
    numpy_free = not any(
        isinstance(c, np.generic)
        for e in lib.values() for v in e["coefficients"].values() for c in v[:1]
    )
    check(numpy_free, "artifact contains no numpy scalars (loads under any numpy)")

    print("\n2. Leave-one-out interpolation error on EXISTING cells (the method's evidence)")
    vx_axis = sorted({float(k.split("_")[0]) for k in raw})
    vy_axis = sorted({float(k.split("_")[1]) for k in raw})
    wz_axis = sorted({float(k.split("_")[2]) for k in raw})

    good_wz = [e for c, e in errs_wz.items() if ("wz",) + c not in flagged]
    good_vy = [e for c, e in errs_vy.items() if ("vy",) + c not in flagged]
    print(f"       wz-axis LOO RMS: max {max(good_wz):.2f} deg, mean {np.mean(good_wz):.2f} deg "
          f"(n={len(good_wz)})")
    print(f"       vy-axis LOO RMS: max {max(good_vy):.2f} deg, mean {np.mean(good_vy):.2f} deg "
          f"(n={len(good_vy)}; conservative 2x-span proxy — the real vy=0 blend spans half this)")
    print("       for scale: v4_robust's own reference-tracking RMS is 4.49 deg")
    check(max(good_wz) < 1.0 and max(good_vy) < 1.0,
          "interpolation error is an order of magnitude below policy tracking error")

    print("\n3. Non-smooth source regions detected and quarantined")
    check(flagged == EXPECTED_IRREGULAR,
          f"the flagged set is exactly the {len(EXPECTED_IRREGULAR)} documented regions",
          "" if flagged == EXPECTED_IRREGULAR
          else f"detected {sorted(flagged)} vs expected {sorted(EXPECTED_IRREGULAR)}")
    for axis, vx, vy, wz in sorted(flagged, key=lambda f: -(errs_wz if f[0] == "wz" else errs_vy)[f[1:]]):
        e = (errs_wz if axis == "wz" else errs_vy)[(vx, vy, wz)]
        print(f"         {axis}-axis at ({vx:+.3f}, {vy:+.3f}, {wz:+.3f})  LOO RMS {e:.1f} deg "
              f"— outside the +/-0.5 wz command hull")
    if skipped:
        print(f"       synthesized cells skipped rather than built on them: {len(skipped)}")
        for s in skipped:
            print(f"         {s['key']}  (parent {s['because'][0]})")
    else:
        print("       no synthesized cell would have depended on them")
    contaminated = [r["key"] for r in records
                    if any(c in {(vx, vy, wz) for _, vx, vy, wz in flagged}
                           for c in r["parent_coords"])]
    check(not contaminated, "no cell in the output descends from a flagged region",
          "" if not contaminated else f"{len(contaminated)}, e.g. {contaminated[:3]}")
    spreads = [r["parent_spread_deg"] for r in records]
    print(f"       parent spread across synthesized cells: median {np.median(spreads):.2f} deg, "
          f"max {max(spreads):.2f} deg (informational — the grid's natural gait gradient, "
          f"not interpolation error)")

    print("\n4. Numerical sanity of every synthesized cell")
    bad_finite, duty_bad, envelope_bad = [], [], []
    for rec in records:
        m = evaluate(coeff_matrix(lib[rec["key"]]))
        if not np.isfinite(m).all():
            bad_finite.append(rec["key"])
        duty = [(m[32] > 0.5).mean() * 100, (m[33] > 0.5).mean() * 100]
        if not all(40.0 <= dd <= 90.0 for dd in duty):
            duty_bad.append((rec["key"], duty))
        pt = [evaluate(coeff_matrix(lib[p]))[LEG_DIMS] for p in rec["parents"]]
        v = m[LEG_DIMS]
        if (v > np.max(pt, axis=0) + 1e-9).any() or (v < np.min(pt, axis=0) - 1e-9).any():
            envelope_bad.append(rec["key"])
    check(not bad_finite, "all values finite")
    check(not envelope_bad, "every trajectory inside its parents' envelope (convexity)")
    check(not duty_bad, f"stance duty inside the [40, 90]% gait gate band",
          "" if not duty_bad else f"{len(duty_bad)} cells out of band, e.g. {duty_bad[0]}")

    print("\n5. The synthesized cells mean what their labels say")
    # wz = 0 rows must have ~zero mean hip yaw; the library's hip-yaw mean is
    # linear in wz (verified: -0.049 rad at wz=-1.111 -> +0.056 rad at +1.222).
    hy = [abs(evaluate(coeff_matrix(lib[key_of(vx, vy, 0.0)]))[[0, 11]].mean())
          for vx in vx_axis for vy in sorted(set(vy_axis) | {0.0})]
    check(max(hy) < 0.006, f"wz=0 rows have near-zero mean hip yaw (max {max(hy):.4f} rad)")
    # The (vx, 0, 0) row is the "true straight walk" Task 2.8 asks for: no
    # side-step AND no turn. It is the only place a zero-lateral-velocity
    # assertion is well posed, because the recording frame does not co-rotate —
    # as soon as wz != 0 a straight-going gait legitimately shows world-frame
    # lateral velocity (at vx=0.222, wz=+0.185 BOTH parents read positive:
    # +0.004 and +0.080 m/s, so their vy=0 blend reads +0.042 and should).
    lat = [abs(evaluate(coeff_matrix(lib[key_of(vx, 0.0, 0.0)]))[35].mean()) for vx in vx_axis]
    yaw = [abs(evaluate(coeff_matrix(lib[key_of(vx, 0.0, 0.0)]))[[0, 11]].mean()) for vx in vx_axis]
    check(max(lat) < 0.01 and max(yaw) < 0.005,
          f"the (vx, 0, 0) straight-walk row neither side-steps nor turns "
          f"(max |v_y| {max(lat):.4f} m/s, max |hip_yaw| {max(yaw):.4f} rad)")
    curved = max(abs(evaluate(coeff_matrix(lib[key_of(vx, 0.0, wz)]))[35].mean())
                 for vx in vx_axis for wz in (0.185, -0.333, 1.222))
    print(f"       (on curved rows the same quantity reaches {curved:.3f} m/s — the recording's "
          f"world frame does not co-rotate; a property of the frozen library, inherited unchanged)")
    # The turn-in-place pair that motivated the patch.
    for wz in (0.5, -0.5):
        k = key_of(0.0, 0.0, wz)
        m = evaluate(coeff_matrix(lib[k]))
        print(f"       {k}: mean hip_yaw L/R {m[0].mean():+.4f}/{m[11].mean():+.4f} rad, "
              f"mean v_xy ({m[34].mean():+.4f}, {m[35].mean():+.4f}) m/s, "
              f"duty L/R {(m[32] > 0.5).mean() * 100:.0f}/{(m[33] > 0.5).mean() * 100:.0f}%")

    print("\n6. Command snapping now resolves as intended")
    print("       (a command need not sit ON the grid — what matters is that it no longer")
    print("        snaps to a gait that side-steps or curves when it was not asked to)")
    cells = np.array([[float(x) for x in k.split("_")] for k in lib])
    names = list(lib)
    old_cells = np.array([[float(x) for x in k.split("_")] for k in raw])
    old_names = list(raw)
    for cmd, label in (((0.2, 0.0, 0.0), "straight walk"),
                       ((0.0, 0.0, 0.5), "turn in place (the killer command)"),
                       ((0.0, 0.0, -0.5), "turn in place, other way"),
                       ((0.222, 0.0, 0.0), "max-speed straight walk"),
                       ((0.15, 0.05, 0.2), "the eval 'mixed' condition")):
        c = np.array(cmd)
        new = names[int(np.argmin(((cells - c) ** 2).sum(axis=1)))]
        old = old_names[int(np.argmin(((old_cells - c) ** 2).sum(axis=1)))]
        nvy, nwz = float(new.split("_")[1]), float(new.split("_")[2])
        # The property that matters: a command with vy=0 / wz=0 must not snap to
        # a gait that side-steps / turns.
        good = (abs(nvy) < 1e-9 or abs(cmd[1]) > 1e-9) and (abs(nwz) < 1e-9 or abs(cmd[2]) > 1e-9)
        print(f"       {label:36s} {str(cmd):20s} {old}  ->  {new}")
        check(good, f"{label}: snapped gait has no unrequested side-step or turn")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="validate the existing v2 library without regenerating it")
    args = ap.parse_args()

    raw, vx_axis, vy_axis, wz_axis = load_source()
    meta = uniform_metadata(raw)
    print(f"source: {SRC_PKL.name}  ({len(raw)} cells, md5 {SRC_MD5[:8]}… verified)")
    print(f"        period {meta['period']}s, fps {meta['fps']}, "
          f"nb_steps_in_period {meta['nb_steps_in_period']} — uniform across the library")

    flagged, errs_wz, errs_vy = find_irregular(raw, vx_axis, vy_axis, wz_axis)
    lib, records, skipped = synthesize(raw, vx_axis, vy_axis, wz_axis, meta, flagged)

    if args.check:
        if not DST_PKL.exists():
            print(f"\n{DST_PKL.name} does not exist — run without --check to generate it.")
            return 1
        on_disk = pickle.loads(DST_PKL.read_bytes())
        if set(on_disk) != set(lib):
            print(f"\nFAIL: {DST_PKL.name} on disk has {len(on_disk)} cells, "
                  f"regeneration yields {len(lib)} — the file is stale.")
            return 1
        lib = on_disk

    ok = report_and_validate(lib, raw, records, skipped, flagged, errs_wz, errs_vy)

    if not ok:
        print("\nVALIDATION FAILED — not writing.")
        return 1

    if not args.check:
        DST_PKL.write_bytes(pickle.dumps(lib, protocol=pickle.HIGHEST_PROTOCOL))
        md5 = hashlib.md5(DST_PKL.read_bytes()).hexdigest()
        print(f"\nwrote {DST_PKL}")
        print(f"      {DST_PKL.stat().st_size:,} bytes, md5 {md5}")
    else:
        print("\nvalidation only — file unchanged.")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
