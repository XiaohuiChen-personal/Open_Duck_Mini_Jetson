"""Tests for scripts/fit_inertia.py — the measurement that settles PLANT-11.

The headline test is `test_recovers_a_known_inertia`: synthesise a sweep from a
KNOWN J and b, and require the fitter to recover them. Without that, a number
coming out of this script is just a number.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_inertia", REPO / "scripts" / "fit_inertia.py"
)
fi = importlib.util.module_from_spec(SPEC)
sys.modules["fit_inertia"] = fi
SPEC.loader.exec_module(fi)


# --------------------------------------------------------------- synthesis --


def synth(j_true, b_true, amplitudes=(114, 171, 227),
          periods=(2.0, 1.2, 0.8, 0.6, 0.45), rate_hz=120.0, cycles=16,
          offset_nm=0.0):
    """Generate a sweep from known dynamics: tau = J*qdd + b*qd (+ offset)."""
    rows = []
    for amplitude in amplitudes:
        amp_rad = amplitude / fi.COUNTS_PER_RAD
        for period in periods:
            omega = 2 * math.pi / period
            n = int(rate_hz * cycles * period)
            for k in range(n):
                t = k / rate_hz
                qd = amp_rad * omega * math.cos(omega * t)
                qdd = -amp_rad * omega ** 2 * math.sin(omega * t)
                tau = j_true * qdd + b_true * qd + offset_nm
                counts = int(round(abs(tau / fi.KT_NM_PER_A) / fi.CURRENT_LSB_A))
                rows.append({"amplitude": amplitude, "period": period, "t": t,
                             "current_raw": counts})
    return rows


# ------------------------------------------------------- the headline test --


@pytest.mark.parametrize("j_true", [0.040, 0.020, 0.012, 0.009])
def test_recovers_a_known_inertia(j_true):
    """If this fails, no number this script produces can be trusted."""
    rows = synth(j_true, b_true=0.05)
    result = fi.fit(fi.group_conditions(rows))
    assert result["verdict"] == "OK", result["verdict"]
    assert result["armature_measured"] == pytest.approx(j_true, rel=0.10)


def test_recovers_the_viscous_term_too():
    result = fi.fit(fi.group_conditions(synth(0.012, b_true=0.08)))
    assert result["verdict"] == "OK"
    assert result["viscous_measured"] == pytest.approx(0.08, rel=0.25)


def test_discriminates_the_two_hypotheses_that_matter():
    """PLANT-11 turns on 0.040 vs ~0.012. The fitter must not confuse them."""
    high = fi.fit(fi.group_conditions(synth(0.040, 0.05)))["armature_measured"]
    low = fi.fit(fi.group_conditions(synth(0.012, 0.05)))["armature_measured"]
    assert high / low == pytest.approx(0.040 / 0.012, rel=0.15)


def test_a_constant_offset_does_not_leak_into_inertia():
    """Coulomb friction and current-sense bias are absorbed by `c`, not by J."""
    result = fi.fit(fi.group_conditions(synth(0.012, 0.05, offset_nm=0.15)))
    assert result["verdict"] == "OK"
    assert result["armature_measured"] == pytest.approx(0.012, rel=0.15)
    assert result["offset"] > 0


# ------------------------------------------------------------- refusals ----


def test_aliased_conditions_are_dropped_not_fitted():
    """Below ~10 samples/cycle the RMS of a sinusoid is aliased. Fitting it
    silently biases J, which is the whole quantity of interest."""
    rows = synth(0.012, 0.05, rate_hz=12.0)   # 0.45 s period -> 5.4 samples/cycle
    result = fi.fit(fi.group_conditions(rows))
    dropped = {(d["amplitude_counts"], d["period_s"])
               for d in result["dropped_for_aliasing"]}
    assert any(p in (0.45, 0.6) for _, p in dropped)


def test_too_few_conditions_is_refused_not_extrapolated():
    rows = synth(0.012, 0.05, amplitudes=(171,), periods=(2.0, 1.2))
    result = fi.fit(fi.group_conditions(rows))
    assert "INSUFFICIENT" in result["verdict"]
    assert "armature_measured" not in result


def test_a_negative_square_rejects_the_model():
    """A negative J^2 means the model does not describe the data. Taking sqrt of
    it, or clamping to zero, would manufacture a false answer."""
    conditions = fi.group_conditions(synth(0.012, 0.05))
    for c in conditions:            # invert the trend the model expects
        c["tau_rms_nm"] = 1.0 / (1.0 + c["omega"])
    result = fi.fit(conditions)
    assert "REJECTED" in result["verdict"] or "POOR FIT" in result["verdict"]
    assert "armature_measured" not in result


def test_poor_fit_is_refused():
    conditions = fi.group_conditions(synth(0.012, 0.05))
    for i, c in enumerate(conditions):
        c["tau_rms_nm"] = 0.5 if i % 2 else 3.0      # pure noise
    result = fi.fit(conditions)
    assert "armature_measured" not in result


# ------------------------------------------------------------- grouping ----


def test_partial_cycles_are_truncated():
    """A partial final cycle biases the RMS of a sinusoid."""
    rows = synth(0.012, 0.05, amplitudes=(171,), periods=(1.0,), cycles=16)
    rows += [{"amplitude": 171, "period": 1.0, "t": 16.3, "current_raw": 900}]
    conditions = fi.group_conditions(rows)
    assert conditions[0]["cycles_used"] == 16


def test_conditions_carry_their_sample_rate():
    conditions = fi.group_conditions(synth(0.012, 0.05, rate_hz=100.0))
    for c in conditions:
        assert c["rate_hz"] == pytest.approx(100.0, rel=0.1)


# ------------------------------------------------------------ constants ----


def test_current_lsb_is_the_resolved_value():
    """310 counts x 12.258 mA = the datasheet's documented 3.80 A trip."""
    assert 310 * fi.CURRENT_LSB_A == pytest.approx(3.80, abs=0.01)


def test_kt_is_the_output_shaft_constant():
    assert fi.KT_NM_PER_A == pytest.approx(1.0787, abs=1e-4)


def test_configured_armature_matches_the_repo():
    src = (REPO / "isaac_lab_env" / "open_duck_mini_v2" / "robot_cfg.py").read_text()
    assert f"armature={fi.CONFIGURED_ARMATURE}" in src, \
        "robot_cfg.py changed; update CONFIGURED_ARMATURE and re-read PLANT-11"


def test_the_plausible_band_excludes_the_configured_value():
    """The whole premise of PLANT-11: 0.040 is outside what geometry allows."""
    assert fi.CONFIGURED_ARMATURE > fi.PLAUSIBLE_ARMATURE[1]


# ------------------------------------------------------- the branch table --


@pytest.mark.parametrize("j,expect", [
    (0.040, "broadly RIGHT"),
    (0.035, "broadly RIGHT"),
    (0.020, "not enough to close the whole gap"),
    (0.012, "BADLY high"),
    (0.009, "BADLY high"),
])
def test_branch_table_matches_plant_11(j, expect):
    lines = "\n".join(fi.interpret({
        "armature_measured": j, "ratio_to_configured": 0.040 / j,
        "viscous_measured": 0.05, "r2": 0.99,
    }))
    assert expect in lines


def test_every_branch_still_demands_a_regate():
    for j in (0.009, 0.020, 0.040):
        lines = "\n".join(fi.interpret({
            "armature_measured": j, "ratio_to_configured": 0.040 / j,
            "viscous_measured": 0.05, "r2": 0.99,
        }))
        assert "re-gate" in lines.lower()


def test_no_number_is_reported_when_the_fit_failed():
    lines = fi.interpret({"verdict": "POOR FIT: R^2 = 0.4"})
    assert len(lines) == 1 and "NO NUMBER" in lines[0]


# ---------------------------------------------------------------- end-to-end --


def test_cli_round_trip(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"outcome": "completed", "torque_limit": 1000,
                                 "samples": synth(0.012, 0.05)}))
    out = tmp_path / "fit.json"
    assert fi.main([str(sweep), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["armature_measured"] == pytest.approx(0.012, rel=0.10)
