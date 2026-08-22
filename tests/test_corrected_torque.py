"""Tests for scripts/corrected_torque_estimate.py.

This script produces the number the go/no-go decision rests on, so its bounds
must actually bound and its constants must match their measured sources.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cte", REPO / "scripts" / "corrected_torque_estimate.py")
cte = importlib.util.module_from_spec(SPEC)
sys.modules["cte"] = cte
SPEC.loader.exec_module(cte)


def test_armature_constants_match_the_bench_measurement():
    assert cte.ARMATURE_CONFIGURED == 0.040
    assert cte.ARMATURE_MEASURED == pytest.approx(0.00843, abs=1e-5)


def test_the_plant_correction_has_been_applied():
    """PLANT-11 fix landed 2026-08-22. This guards against a revert: the
    measured 0.00843 must be what the simulation uses, not BAM's 0.040."""
    src = (REPO / "isaac_lab_env" / "open_duck_mini_v2" / "robot_cfg.py").read_text()
    assert "STS3250_ARMATURE = 0.00843" in src
    assert "armature=0.040" not in src, "BAM's inflated armature is back"


def test_ratio_is_smallest_where_the_link_is_lightest():
    """The ankle link is 0.000104 kg.m^2, so armature is 99.7% of its M_jj and
    the correction bites hardest there."""
    assert cte.inertial_ratio("left_ankle") < cte.inertial_ratio("left_hip_roll")


def test_ratio_is_between_zero_and_one():
    for joint in ("left_ankle", "left_knee", "left_hip_pitch", "left_hip_roll"):
        assert 0 < cte.inertial_ratio(joint) < 1


def test_non_leg_joints_are_not_scaled():
    for joint in ("head_yaw", "neck_pitch", "left_antenna"):
        assert cte.inertial_ratio(joint) is None


def test_lower_bound_is_never_above_the_upper():
    for tau in (0.2, 0.69, 1.0, 2.12, 4.0):
        lo, hi = cte.corrected(tau, 0.293)
        assert lo <= hi


def test_correction_always_reduces_the_torque():
    for tau in (0.8, 1.5, 2.12):
        lo, hi = cte.corrected(tau, 0.293)
        assert hi <= tau


def test_a_torque_already_at_the_floor_cannot_be_reduced_below_it():
    """The gravity/support load is irreducible by any plant correction."""
    lo, hi = cte.corrected(0.5, 0.293)
    assert hi == pytest.approx(0.5)


def test_upper_bound_preserves_the_gait_floor():
    lo, hi = cte.corrected(2.12, 0.293)
    assert hi >= cte.GAIT_FLOOR_NM


def test_a_ratio_of_one_changes_nothing():
    lo, hi = cte.corrected(2.12, 1.0)
    assert lo == pytest.approx(2.12) and hi == pytest.approx(2.12)


def test_the_decision_thresholds_match_their_sources():
    assert cte.WARRANTY_TORQUE_NM == pytest.approx(0.981, abs=1e-3)  # 8-1, 1/5 stall
    assert cte.RATED_TORQUE_NM == pytest.approx(1.569, abs=1e-3)     # 5-8
    assert cte.RATED_CURRENT_A == 1.400                              # 5-9
    assert cte.KT_NM_PER_A == pytest.approx(1.0787, abs=1e-4)        # 5-11


@pytest.mark.skipif(not (REPO / "v7_torque.npz").exists(), reason="trace absent")
def test_the_corrected_worst_joint_clears_the_warranty_load():
    """THE DECISION. If this ever fails, sim-to-real on this servo is off."""
    import numpy as np
    data = np.load(REPO / "v7_torque.npz")
    worst = 0.0
    for name, rms in zip(data["names"], data["rms"]):
        ratio = cte.inertial_ratio(str(name))
        if ratio is None:
            continue
        worst = max(worst, cte.corrected(float(rms), ratio)[1])
    assert worst <= cte.WARRANTY_TORQUE_NM, (
        f"worst corrected joint {worst:.3f} N.m exceeds the 8-1 warranty load")
    assert worst / cte.KT_NM_PER_A <= cte.RATED_CURRENT_A
