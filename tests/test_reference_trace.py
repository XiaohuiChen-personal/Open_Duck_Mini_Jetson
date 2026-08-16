"""Task S.2 — the recorded Isaac traces are the ground truth S.4 will be built against.

Everything from S.4 onward reimplements what Isaac Lab does between the sensors
and the servos. Without recorded ground truth you can only test that
reimplementation against your own reading of the code — which is exactly how the
4x action-scale error in DEPLOY-1 would have survived.

Every test skips cleanly if the traces are absent, so the suite is green before
S.2 and meaningful after.
"""

import json
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM2REAL = os.path.join(REPO, "docs", "jetson-mod", "sim2real")
CONTRACT = os.path.join(REPO, "jetson_runtime", "policy_contract.json")
TRACES = ("fwd", "turn", "stand")


def _load(name):
    p = os.path.join(SIM2REAL, f"reference_trace_v7_{name}.npz")
    if not os.path.isfile(p):
        pytest.skip(f"trace {name} not recorded yet (Task S.2)")
    return np.load(p, allow_pickle=True)


@pytest.fixture(scope="module")
def c():
    if not os.path.isfile(CONTRACT):
        pytest.skip("policy contract not generated (Task S.1)")
    return json.load(open(CONTRACT))


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_trace_shapes(name, c):
    d = _load(name)
    assert d["obs"].shape == (1500, c["obs_dim"])
    assert d["action"].shape == (1500, c["action_dim"])


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_no_episode_boundary(name):
    """A fall or timeout resets the gait index and last_action, which would make
    every indexing assertion below meaningless."""
    d = _load(name)
    assert not d["terminated"].any(), f"episode ended at step {int(d['terminated'].argmax())}"
    assert not d["truncated"].any()


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_obs_joint_block_is_relative_at_the_recorded_lag(name, c):
    """**The most valuable assertion in the phase.** DEPLOY-2's trap, proven from
    data: the observation's joint block is `q - q_default`, NOT raw encoder
    angles. Feeding absolute angles puts the knees 13-14 sigma outside the
    training distribution.

    The comparison is made at the recorded lag, which is 1 (the recorder samples
    joint_pos AFTER env.step and obs BEFORE) plus the PLANT-7 observation delay
    of 0-2 control steps. Asserting at lag 0 would fail for a reason that has
    nothing to do with relativeness.
    """
    d = _load(name)
    m = d["meta"].item()
    lag = m["obs_joint_block_total_lag"]
    assert lag is not None, "the recorder could not identify the lag"
    a, b = c["obs_slices"]["joint_pos"]
    q = np.array(c["q_default_rad"])
    lhs = d["obs"][lag:, a:b]
    rhs = (d["joint_pos"] - q)[: -lag or None]
    assert np.abs(lhs - rhs).max() < 1e-4, (
        "the observation joint block is not q - q_default at the recorded lag")


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_last_action_block_is_the_previous_action(name, c):
    """Pins the indexing convention the whole phase depends on: row i of obs is
    the input that produced row i of action."""
    d = _load(name)
    a, b = c["obs_slices"]["actions"]
    assert np.abs(d["obs"][1:, a:b] - d["action"][:-1]).max() < 1e-6
    assert np.abs(d["obs"][0, a:b]).max() == 0.0, "obs[0] last_action must be zeros"


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_gait_phase_is_a_unit_circle_of_the_contract_period(name, c):
    """The gait clock is written down nowhere else. It only advances because the
    imitation REWARD term is evaluated, so a zeroed phase silently means the
    reward was not wired in."""
    d = _load(name)
    a, b = c["obs_slices"]["gait_phase"]
    cos, sin = d["obs"][:, a], d["obs"][:, b - 1]
    assert np.abs(cos**2 + sin**2 - 1.0).max() < 1e-5, "gait phase is not on the unit circle"
    assert not np.allclose(sin, 0.0), "gait phase never advances — the reward term is not wired"
    # the phase must complete a whole number of turns at the contract's period
    n = c["gait_nb_steps"]
    ang = np.unwrap(np.arctan2(sin, cos))
    per_step = np.diff(ang).mean()
    assert abs(abs(per_step) - 2 * np.pi / n) < 1e-3, (
        f"phase advances {per_step:.5f} rad/step, expected {2*np.pi/n:.5f} for a {n}-step gait")


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_command_block_matches_the_request_including_yaw(name, c):
    """PLANT-8: the Play cfg leaves heading_command true, so
    UniformVelocityCommand overwrites the YAW channel every step unless it is
    disabled. Checking the yaw channel specifically is what proves the command
    pin actually took."""
    d = _load(name)
    m = d["meta"].item()
    a, b = c["obs_slices"]["velocity_commands"]
    want = np.array(m["command"], dtype=float)
    got = d["obs"][:, a:b]
    assert np.abs(got - want).max() < 1e-5, (
        f"commanded {want} but the observation carried {got.min(0)}..{got.max(0)} — "
        "the heading servo was not disabled")


@pytest.mark.phase4
@pytest.mark.parametrize("name", TRACES)
def test_provenance_matches_the_shipped_policy(name):
    d = _load(name)
    m = d["meta"].item()
    dep = os.path.join(REPO, "exported_policies", "v7_servo_safe_ppo", "deployment_contract.json")
    if not os.path.isfile(dep):
        pytest.skip("no exported policy")
    assert m["checkpoint_md5"] == json.load(open(dep))["checkpoint_md5"], (
        "the trace was recorded from a different checkpoint than the one shipped")
    assert m["enable_corruption"] is False, "the trace may contain a training noise draw"
