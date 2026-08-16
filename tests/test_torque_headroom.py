"""Task S.8 — the servo limits carried in the deployment contract.

The runtime's safety layer (S.10) is built on these numbers, so they must be
present, self-consistent, and traceable to a source rather than to memory.
"""

import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(REPO, "jetson_runtime", "policy_contract.json")
ENVELOPE = os.path.join(REPO, "docs", "jetson-mod", "sim2real", "S8_torque_envelope.md")


@pytest.fixture(scope="module")
def c():
    if not os.path.isfile(CONTRACT):
        pytest.skip("policy contract not generated (Task S.1)")
    return json.load(open(CONTRACT))


@pytest.mark.phase4
def test_servo_limits_are_present_and_ordered(c):
    """Rated < overload trip < stall. If that ordering ever breaks, a limiter
    built on these numbers would let the servo into a region it cannot hold."""
    rated = c["servo_continuous_torque_nm"]
    overload = c["servo_overload_trip_nm"]
    stall = c["servo_stall_torque_nm"]
    assert rated < overload < stall, f"limits out of order: {rated}, {overload}, {stall}"
    assert abs(overload - 0.8 * stall) < 1e-6, "the overload trip should be 80 % of stall"


@pytest.mark.phase4
def test_sim_effort_limit_matches_robot_cfg(c):
    """The contract must not carry a stale copy of the sim's effort limit. Task
    M0 changed it once (PLANT-5: BAM's electrical 8.716 -> datasheet 4.903);
    this is what stops it going stale again."""
    import re
    src = open(os.path.join(REPO, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py")).read()
    m = re.search(r"STS3250_EFFORT_LIMIT_NM\s*=\s*([0-9.]+)", src)
    assert m, "STS3250_EFFORT_LIMIT_NM not found"
    assert abs(c["sim_effort_limit_nm"] - float(m.group(1))) < 1e-9, (
        f"contract says {c['sim_effort_limit_nm']}, robot_cfg.py says {m.group(1)}")


@pytest.mark.phase4
def test_thermal_stops_are_below_the_firmware_cutout(c):
    """The whole point of the runtime's thermal limits: stop the robot in a
    controlled way BEFORE the firmware drops torque without warning. A silent
    torque-off mid-stride is a fall."""
    assert c["servo_temp_warn_c"] < c["servo_temp_stop_c"] < c["servo_firmware_temp_cutout_c"], (
        f"warn {c['servo_temp_warn_c']} / stop {c['servo_temp_stop_c']} / "
        f"cutout {c['servo_firmware_temp_cutout_c']} are not strictly ordered")


@pytest.mark.phase4
def test_limits_name_a_source_and_admit_what_is_missing(c):
    """A number without provenance is how '16 kg.cm' propagated through eight
    files in this repo with no citation. The source must also be honest that
    Feetech publishes no thermal data."""
    src = c["servo_limits_source"]
    assert "Feetech" in src and "2024-01-16" in src, "no datasheet citation"
    assert "thermal" in src.lower() and "S.8b" in src, (
        "the source string must state that thermal behaviour is NOT established "
        "by the datasheet and point at the bench task")
    assert "ENGINEERING CHOICE" in c["servo_temp_thresholds_are"]


@pytest.mark.phase4
def test_envelope_document_exists_and_states_a_decision():
    if not os.path.isfile(ENVELOPE):
        pytest.skip("S8_torque_envelope.md not generated yet")
    text = open(ENVELOPE).read()
    assert "## Decision" in text
    assert ("ACCEPT" in text) or ("DO NOT ACCEPT" in text), "the document states no decision"
    assert "S.8b" in text, "must point at the bench task for the thermal question"
