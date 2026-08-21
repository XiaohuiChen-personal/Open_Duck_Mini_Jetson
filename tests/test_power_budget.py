"""Tests for scripts/power_budget.py.

The budget decides whether the battery and the servos can sustain a gait, so the
constants it rests on are pinned here — particularly the ones whose provenance is
a datasheet line rather than a measurement.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "power_budget", REPO / "scripts" / "power_budget.py"
)
pb = importlib.util.module_from_spec(SPEC)
sys.modules["power_budget"] = pb
SPEC.loader.exec_module(pb)


# ------------------------------------------------------------- constants ----


def test_kt_matches_the_datasheet_in_si():
    """5-11 gives Kt = 11 kg·cm/A. Everything downstream uses N·m."""
    assert pb.KT_KGCM_PER_A == 11.0
    assert pb.KT_NM_PER_A == pytest.approx(1.0787, abs=1e-4)


def test_kt_is_self_consistent_with_the_rated_pair():
    """The datasheet's own rated torque (16 kg·cm, 5-8) over rated current
    (1400 mA, 5-9) gives 11.4 kg·cm/A. That it agrees with 5-11 to within 4% is
    what proves Kt is quoted at the OUTPUT shaft, gearing included. If it were a
    motor-side constant, the two would differ by the 1/345 ratio."""
    implied = 16.0 / 1.4
    assert implied == pytest.approx(pb.KT_KGCM_PER_A, rel=0.05)


def test_the_two_resistances_really_do_disagree():
    """This is a datasheet contradiction, not a modelling choice. If a future
    edit 'tidies' one of them away, the 2.4x uncertainty silently vanishes."""
    assert pb.R_TOTAL_OHM == pytest.approx(2.857, abs=1e-3)
    assert pb.R_TOTAL_OHM / pb.R_WINDING_OHM == pytest.approx(2.38, abs=0.05)


def test_idle_current_is_the_measured_value_not_the_datasheet_one():
    """Datasheet 5-6 says 24 mA; the bench measured 21 mA on 2026-08-21.
    The budget uses the measurement."""
    assert pb.IDLE_CURRENT_A == 0.021


def test_antennas_are_excluded():
    """They are micro servos on PWM, not STS3250, and counting them as such
    would inflate the servo count from 14 to 16."""
    assert "left_antenna" in pb.NON_STS3250
    assert "right_antenna" in pb.NON_STS3250


# --------------------------------------------------------------- physics ----


def test_winding_current_at_rated_torque_matches_rated_current():
    assert pb.winding_current_a(1.569) == pytest.approx(1.454, abs=0.01)


def test_dissipation_scales_with_the_square_of_torque():
    a = pb.dissipation_w(1.0, pb.R_WINDING_OHM)
    b = pb.dissipation_w(2.0, pb.R_WINDING_OHM)
    assert b == pytest.approx(4 * a)


def test_dissipation_scales_linearly_with_resistance():
    a = pb.dissipation_w(1.0, 1.2)
    b = pb.dissipation_w(1.0, 2.4)
    assert b == pytest.approx(2 * a)


def test_zero_torque_costs_nothing():
    assert pb.dissipation_w(0.0, pb.R_WINDING_OHM) == 0.0


# ---------------------------------------------------------------- budget ----


def _joints(*pairs):
    return [(name, rms, rms * 2) for name, rms in pairs]


def test_budget_excludes_non_sts3250_joints_from_the_count():
    b = pb.budget(_joints(("left_knee", 1.0), ("left_antenna", 0.003)))
    assert b["n_servos"] == 1
    assert all(r["joint"] != "left_antenna" for r in b["joints"])


def test_budget_reports_both_resistance_bounds():
    b = pb.budget(_joints(("left_knee", 1.0)))
    lo, hi = b["servo_watts"]
    assert hi > lo
    assert hi / lo == pytest.approx(pb.R_TOTAL_OHM / pb.R_WINDING_OHM, rel=1e-6)


def test_quiescent_current_scales_with_servo_count():
    b = pb.budget(_joints(("a", 1.0), ("b", 1.0), ("c", 1.0)))
    assert b["quiescent_a"] == pytest.approx(3 * pb.IDLE_CURRENT_A)


def test_joints_are_ranked_by_dissipation():
    b = pb.budget(_joints(("small", 0.2), ("big", 2.0), ("mid", 1.0)))
    assert [r["joint"] for r in b["joints"]] == ["big", "mid", "small"]


def test_jetson_modes_are_all_costed():
    b = pb.budget(_joints(("left_knee", 1.0)))
    assert set(b["pack"]) == set(pb.JETSON_MODES_W)
    lo7, _ = b["pack"][7]
    lo25, _ = b["pack"][25]
    assert lo25 > lo7


def test_dcdc_losses_are_charged_to_the_pack():
    """25 W at the Jetson costs more than 25 W from the pack."""
    b = pb.budget(_joints(("left_knee", 0.0)))
    pack_lo, _ = b["pack"][25]
    naive = 25 / pb.PACK_NOMINAL_V
    assert pack_lo > naive


# ------------------------------------------------- the real v7 trace ---------


@pytest.mark.skipif(not (REPO / "v7_torque.npz").exists(), reason="trace absent")
def test_v7_trace_has_exactly_14_sts3250_joints():
    b = pb.budget(pb.load_trace(str(REPO / "v7_torque.npz")))
    assert b["n_servos"] == 14, "the robot has 14 STS3250 servos plus 2 antennas"


@pytest.mark.skipif(not (REPO / "v7_torque.npz").exists(), reason="trace absent")
def test_leg_joints_dominate_the_power_budget():
    """MEASURED: the six leg joints are ~91% of servo power, so head/neck/hip-yaw
    are thermally irrelevant. Any future 'reduce torque everywhere' effort that
    ignores this is optimising the 9%."""
    b = pb.budget(pb.load_trace(str(REPO / "v7_torque.npz")))
    assert b["top6_share"] > 0.85
    top6 = {r["joint"] for r in b["joints"][:6]}
    assert top6 == {"left_hip_pitch", "right_hip_pitch", "left_knee",
                    "right_knee", "left_ankle", "right_ankle"}
