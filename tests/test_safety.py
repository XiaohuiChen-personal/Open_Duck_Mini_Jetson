"""Task S.10 — the safety layer.

This is the code that stands between a software bug and a destroyed servo, so
the tests assert behaviour under fault, not just that the functions run.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from jetson_runtime.safety import (DERATE, HOLD, OK, TORQUE_OFF,  # noqa: E402
                                   SafetyMonitor, torque_off_sequence)

CONTRACT = json.load(open(os.path.join(REPO, "jetson_runtime", "policy_contract.json")))
N = len(CONTRACT["joint_order"])


def mon():
    return SafetyMonitor(CONTRACT)


def healthy(**over):
    b = {"gravity_b": np.array([0.0, 0.0, -1.0]), "temps_c": np.full(N, 32.0),
         "currents_a": np.full(N, 0.4), "voltage_v": 11.1,
         "foot_contacts": (True, True), "step_elapsed_s": 0.008}
    b.update(over)
    return b


# ------------------------------------------------------------ no literals ---


def test_every_threshold_comes_from_the_contract():
    """The thresholds moved once already -- the firmware cutout is 80 C, not the
    70 C every document assumed until it was read off the servo. Hard-coding
    them is how that happens again."""
    m = mon()
    assert m.temp_warn == CONTRACT["servo_temp_warn_c"]
    assert m.temp_stop == CONTRACT["servo_temp_stop_c"]
    assert m.current_limit == CONTRACT["servo_current_limit_a"]
    assert m.firmware_cutout == CONTRACT["servo_firmware_temp_cutout_c"]
    assert m.min_voltage == CONTRACT["servo_min_input_voltage_v"]


def test_stop_temperature_is_below_the_firmware_cutout():
    """If the firmware wins, the leg goes limp with no warning and no log line."""
    m = mon()
    assert m.temp_stop < m.firmware_cutout
    assert m.temp_warn < m.temp_stop


def test_current_window_is_half_the_firmware_window():
    """Deliberate: the runtime must act before the firmware does."""
    m = mon()
    assert m.current_window_s == CONTRACT["servo_current_limit_window_s"] / 2.0


# --------------------------------------------------------------- verdicts ---


def test_healthy_is_ok():
    assert mon().check(**healthy()).verdict == OK


def test_tilt_past_sixty_degrees_cuts_torque():
    """60 deg is the simulator's own bad_orientation termination. Beyond it the
    policy is outside every state it was trained in."""
    assert mon().check(**healthy(gravity_b=np.array([0.0, -0.9, -0.2]))).verdict \
        == TORQUE_OFF


def test_tilt_just_inside_the_limit_is_allowed():
    import math
    a = math.radians(55.0)
    g = np.array([math.sin(a), 0.0, -math.cos(a)])
    assert mon().check(**healthy(gravity_b=g)).verdict == OK


def test_over_temperature_stops_and_names_the_joint():
    st = mon().check(**healthy(temps_c=np.array([32.0] * (N - 1) + [70.0])))
    assert st.verdict == TORQUE_OFF
    assert CONTRACT["joint_order"][-1] in ";".join(st.reasons)


def test_warm_servo_derates_rather_than_stopping():
    st = mon().check(**healthy(temps_c=np.full(N, 57.0)))
    assert st.verdict == DERATE
    assert st.command_scale < 1.0


def test_under_voltage_uses_the_measured_six_volts():
    """addr15 = 6.0 V, NOT the datasheet text's 4 V. A 3S2P pack sags under
    load and the servo cuts torque there."""
    assert mon().check(**healthy(voltage_v=5.5)).verdict == TORQUE_OFF
    assert mon().check(**healthy(voltage_v=9.0)).verdict == OK


# ------------------------------------------------------ duration-based ------


def test_brief_over_current_does_not_trip():
    """Every Feetech protection is duration-based; a spike is not a fault."""
    m = mon()
    for _ in range(5):
        st = m.check(**healthy(currents_a=np.full(N, 4.5)))
    assert st.verdict != TORQUE_OFF


def test_sustained_over_current_trips():
    m = mon()
    for _ in range(60):
        st = m.check(**healthy(currents_a=np.full(N, 4.5)))
    assert st.verdict == TORQUE_OFF


def test_fall_proxy_needs_both_feet_off_and_tilt_sustained():
    """The substitute for root_height_below_minimum, which cannot be measured."""
    m = mon()
    b = healthy(foot_contacts=(False, False), gravity_b=np.array([0.6, 0.0, -0.5]))
    for i in range(4):
        assert m.check(**b).verdict != TORQUE_OFF, f"tripped early at step {i+1}"
    assert m.check(**b).verdict == TORQUE_OFF


def test_one_foot_down_is_not_a_fall():
    m = mon()
    b = healthy(foot_contacts=(True, False), gravity_b=np.array([0.6, 0.0, -0.5]))
    for _ in range(10):
        assert m.check(**b).verdict != TORQUE_OFF


def test_single_overrun_holds_but_three_cut_torque():
    m = mon()
    assert m.check(**healthy(step_elapsed_s=0.05)).verdict == HOLD
    m.check(**healthy(step_elapsed_s=0.05))
    assert m.check(**healthy(step_elapsed_s=0.05)).verdict == TORQUE_OFF


def test_nan_observation_holds_then_cuts():
    m = mon()
    bad = healthy(obs=np.full(CONTRACT["obs_dim"], np.nan))
    for _ in range(3):
        assert m.check(**bad).verdict == HOLD
    assert m.check(**bad).verdict == TORQUE_OFF


# ---------------------------------------------------------------- latching --


def test_torque_off_latches_and_only_rearm_clears_it():
    """A monitor that un-trips itself trips repeatedly while the robot destroys
    itself."""
    m = mon()
    m.check(**healthy(gravity_b=np.array([0.0, -0.9, -0.2])))
    assert m.state.latched
    for _ in range(5):
        assert m.check(**healthy()).verdict == TORQUE_OFF, "un-latched on its own"
    m.rearm()
    assert m.check(**healthy()).verdict == OK


def test_trip_history_survives_rearm():
    m = mon()
    m.check(**healthy(voltage_v=5.0))
    m.rearm()
    assert len(m.state.trip_history) == 1


# ---------------------------------------------------------- command clamp ---


def test_turn_clamps_inside_the_trained_hull():
    """wz clamps to +/-0.3 while the hull is +/-0.5 -- deliberately conservative
    (DEPLOY-5). Widening it is a decision to record, not a default."""
    m = mon()
    assert m.clamp_command(0, 0, 0.9)[2] == pytest.approx(0.3)
    assert CONTRACT["cmd_hull_trained"]["wz"][1] == 0.5
    assert CONTRACT["cmd_clamp_deployment"]["wz"][1] == 0.3


def test_clamps_are_counted_not_silent():
    m = mon()
    m.clamp_command(9.0, 9.0, 9.0)
    assert m.state.clamped_commands == 3


def test_in_range_command_passes_unchanged():
    m = mon()
    assert m.clamp_command(0.1, 0.0, 0.1) == (0.1, 0.0, 0.1)
    assert m.state.clamped_commands == 0


# ------------------------------------------------------------ torque off ----


def test_torque_off_ramps_to_the_measured_pose():
    """Commanding the last target while the servo is elsewhere makes it fight;
    ramping to where it actually is makes the disable a release, not a drop."""
    measured = np.full(N, 0.5, dtype=np.float32)
    target = np.zeros(N, dtype=np.float32)
    seq = torque_off_sequence(measured, target, steps=2)
    assert len(seq) == 2
    assert np.allclose(seq[-1], measured)
    assert np.all(np.abs(seq[0] - target) < np.abs(seq[1] - target))


# --------------------------------------------------- the injection harness --


def test_fault_injection_harness_passes():
    """Runs all nine fault cases AND the clean trace. Zero false trips required:
    a monitor that fires on healthy data teaches the operator to ignore it."""
    from jetson_runtime.tools import fault_injection
    import glob
    if not glob.glob(os.path.join(REPO, "docs", "jetson-mod", "sim2real",
                                  "reference_trace_v7_*.npz")):
        pytest.skip("S.2 traces absent")
    assert fault_injection.main() == 0
