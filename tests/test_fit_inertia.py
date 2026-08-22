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


def synth(j_true, b_true, amplitudes=(400, 700, 1000),
          periods=(2.0, 1.2, 0.8, 0.6, 0.45), rate_hz=120.0, cycles=16,
          offset_nm=0.0, tracking=None):
    """Generate a sweep from known dynamics: tau = J*qdd + b*qd (+ offset).

    `tracking` optionally models the servo's closed-loop attenuation: a callable
    (period) -> ratio in (0,1]. The EXECUTED amplitude is what generates both the
    torque and the logged position, exactly as on real hardware.
    """
    rows = []
    for amplitude in amplitudes:
        for period in periods:
            ratio = tracking(period) if tracking else 1.0
            executed = amplitude * ratio
            amp_rad = executed / fi.COUNTS_PER_RAD
            omega = 2 * math.pi / period
            n = int(rate_hz * cycles * period)
            for k in range(n):
                t = k / rate_hz
                q = amp_rad * math.sin(omega * t)
                qd = amp_rad * omega * math.cos(omega * t)
                qdd = -amp_rad * omega ** 2 * math.sin(omega * t)
                tau = j_true * qdd + b_true * qd + offset_nm
                counts = int(round(abs(tau / fi.KT_NM_PER_A) / fi.CURRENT_LSB_A))
                rows.append({"amplitude": amplitude, "period": period, "t": t,
                             "position": 2048 + q * fi.COUNTS_PER_RAD,
                             "current_raw": counts})
    return rows


def _real_tracking(period):
    """The attenuation actually measured on the STS3250, 2026-08-22."""
    return {2.0: 1.00, 1.2: 0.79, 0.8: 0.57, 0.6: 0.44, 0.45: 0.34}[period]


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


def test_the_plant_correction_has_been_applied():
    """PLANT-11 fix landed 2026-08-22. This guards against a revert: the
    measured 0.00843 must be what the simulation uses, not BAM's 0.040."""
    src = (REPO / "isaac_lab_env" / "open_duck_mini_v2" / "robot_cfg.py").read_text()
    assert "STS3250_ARMATURE = 0.00843" in src
    assert "armature=0.040" not in src, "BAM's inflated armature is back"


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


# ------------------------------ the failure that broke the first real run ----


@pytest.mark.parametrize("j_true", [0.040, 0.020, 0.012])
def test_recovers_j_despite_closed_loop_attenuation(j_true):
    """MEASURED 2026-08-22: the servo executes only 26-48% of the commanded
    amplitude above ~1 Hz, because its internal position controller rolls off.
    Fitting against the COMMANDED amplitude measures that controller instead of
    the inertia, and on the first real sweep produced J^2 = -6.8e-7 -- rejected.
    The executed trajectory is what generates the torque, so the fit must use
    the amplitude derived from logged POSITION."""
    rows = synth(j_true, b_true=0.05, tracking=_real_tracking)
    result = fi.fit(fi.group_conditions(rows))
    assert result["verdict"] == "OK", result["verdict"]
    assert result["armature_measured"] == pytest.approx(j_true, rel=0.10)


def test_fitting_the_commanded_amplitude_would_have_failed():
    """Proves the above test is not vacuous: the OLD method really is broken by
    attenuation, so the new one is doing real work."""
    conditions = fi.group_conditions(synth(0.012, 0.05, tracking=_real_tracking))
    for c in conditions:                      # revert to the commanded amplitude
        amp_rad = c["amplitude_counts"] / fi.COUNTS_PER_RAD
        c["x_inertial"] = (amp_rad * c["omega"] ** 2 / math.sqrt(2)) ** 2
        c["x_viscous"] = (amp_rad * c["omega"] / math.sqrt(2)) ** 2
    bad = fi.fit(conditions)
    assert "armature_measured" not in bad or \
        abs(bad["armature_measured"] - 0.012) / 0.012 > 0.25


def test_tracking_ratio_is_reported():
    """The operator must be able to see the attenuation, not just its effect."""
    conditions = fi.group_conditions(synth(0.012, 0.05, tracking=_real_tracking))
    slow = [c for c in conditions if c["period_s"] == 2.0][0]
    fast = [c for c in conditions if c["period_s"] == 0.45][0]
    assert slow["tracking_ratio"] == pytest.approx(1.00, abs=0.05)
    assert fast["tracking_ratio"] == pytest.approx(0.34, abs=0.05)


def test_the_old_small_amplitudes_are_refused_as_floor_noise():
    """REGRESSION, 2026-08-22. The first real sweep used 10-20 deg and returned
    I_rms of 2.8-4.3 counts against addr69's 12.258 mA quantisation. The fitter
    still produced J = 0.0012 at R^2 = 0.97, driven entirely by one outlier --
    a number 7.5x below even the geometry-plausible floor. R^2 cannot catch
    this: a nearly-constant y has almost no variance to explain."""
    rows = synth(0.012, 0.05, amplitudes=(114, 171, 227), tracking=_real_tracking)
    result = fi.fit(fi.group_conditions(rows))
    assert "armature_measured" not in result
    assert "TOO SMALL" in result["verdict"] or "DYNAMIC RANGE" in result["verdict"]


# --------------------------------------------------- step-response method ----


def synth_steps(j_true, friction_nm=0.15, tau_sat=1.95, rate_hz=1100.0,
                sizes=(400, 800, 1200), repeats=4, cruise_rad_s=6.0):
    """Synthesise step responses: saturated launch, then a velocity plateau.

    The cruise phase is modelled deliberately — contaminating the fit with it is
    the exact error that produced J = 0.044-0.050 on real data (2026-08-22).
    """
    rows, trial = [], 0
    accel = (tau_sat - friction_nm) / j_true
    for size in sizes:
        for _ in range(repeats):
            trial += 1
            t_cruise = cruise_rad_s / accel
            for k in range(int(rate_hz * 0.5)):
                t = k / rate_hz
                if t < t_cruise:
                    q = 0.5 * accel * t ** 2
                    cur = tau_sat / fi.KT_NM_PER_A
                else:
                    q = (0.5 * accel * t_cruise ** 2
                         + cruise_rad_s * (t - t_cruise))
                    cur = friction_nm / fi.KT_NM_PER_A
                counts = q * fi.COUNTS_PER_RAD
                if counts >= size:
                    break
                rows.append({"trial": trial, "size": size, "t": t,
                             "position": 2048 + counts,
                             "current_raw": int(round(cur / fi.CURRENT_LSB_A))})
    return rows


@pytest.mark.parametrize("j_true", [0.040, 0.020, 0.0085])
def test_step_method_recovers_a_known_inertia(j_true):
    result = fi.fit_step_trials(synth_steps(j_true))
    assert result["verdict"] == "OK", result["verdict"]
    assert result["armature_measured"] == pytest.approx(j_true, rel=0.12)


def test_step_result_is_independent_of_step_size():
    """The property that distinguishes a real measurement from a windowing
    artifact. On real data 2026-08-22 the fixed window gave 0.0076-0.0098 across
    400/800/1200-count steps; a velocity-selected window gave 0.007-0.050."""
    rows = synth_steps(0.0085)
    per_size = {}
    for size in (400, 800, 1200):
        subset = [r for r in rows if r["size"] == size]
        per_size[size] = fi.fit_step_trials(subset)["trials"][0]["j"]
    assert max(per_size.values()) / min(per_size.values()) < 1.15, per_size


def test_cruise_contamination_would_inflate_j():
    """Proves the fixed window is doing real work: widening it to include the
    velocity plateau biases J upward, which is what happened on real data."""
    rows = synth_steps(0.0085)
    narrow = fi.fit_step_trials(rows)["armature_measured"]
    old_window = fi.STEP_WINDOW_S
    try:
        fi.STEP_WINDOW_S = 0.15          # long enough to swallow the plateau
        wide = fi.fit_step_trials(rows)["armature_measured"]
    finally:
        fi.STEP_WINDOW_S = old_window
    assert wide > narrow * 1.5, (narrow, wide)


def test_friction_correction_lowers_the_estimate():
    rows = synth_steps(0.0085)
    none = fi.fit_step_trials(rows, friction_nm=0.0)["armature_measured"]
    with_f = fi.fit_step_trials(rows, friction_nm=0.15)["armature_measured"]
    assert with_f < none


def test_too_few_usable_trials_is_refused():
    rows = [r for r in synth_steps(0.0085) if r["trial"] <= 2]
    result = fi.fit_step_trials(rows)
    assert "INSUFFICIENT" in result["verdict"]
    assert "armature_measured" not in result
