"""Task S.4 — prove the runtime core against the S.2 traces, with no robot.

The whole point of S.2 is that this layer can be shown correct BEFORE any
hardware exists. Otherwise its first test is the robot falling over.

Note the dims: **53/14, not the 59/16 in task_plan_v2.md's S.4 text.** Task M0b
removed the two antenna joints (open-loop SG90s with no position feedback, so
four dims could never be measured on hardware). The contract is authoritative.
"""

import glob
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from jetson_runtime.action_decoder import ActionDecoder      # noqa: E402
from jetson_runtime.gait_phase import GaitPhase              # noqa: E402
from jetson_runtime.imu import quat_to_projected_gravity     # noqa: E402
from jetson_runtime.obs_builder import ObsBuilder            # noqa: E402

CONTRACT = json.load(open(os.path.join(REPO, "jetson_runtime", "policy_contract.json")))
TRACES = sorted(glob.glob(os.path.join(
    REPO, "docs", "jetson-mod", "sim2real", "reference_trace_v7_*.npz")))
ONNX = os.path.join(REPO, "exported_policies", "v7_servo_safe_ppo", "policy.onnx")

needs_traces = pytest.mark.skipif(not TRACES, reason="S.2 traces absent")


def load(path):
    return np.load(path, allow_pickle=True)


def trace_ids():
    return [os.path.basename(p).replace("reference_trace_v7_", "").replace(".npz", "")
            for p in TRACES]


# ----------------------------------------------------------------- shapes ---


def test_contract_is_the_post_m0b_shape():
    """M0b: 59/16 -> 53/14. If this regresses, every slice below is wrong."""
    assert CONTRACT["obs_dim"] == 53
    assert CONTRACT["action_dim"] == 14


def test_obs_slices_tile_the_vector_exactly():
    bounds = sorted(CONTRACT["obs_slices"].values())
    assert bounds[0][0] == 0
    assert bounds[-1][1] == CONTRACT["obs_dim"]
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start, "obs_slices must be contiguous with no gaps"


# ------------------------------------------------------------ obs builder ---


# The traces are NOT time-aligned, and that is a property of the recording plus
# PLANT-7, not a defect. Measured 2026-08-22, each block matches at its own lag
# with EXACTLY zero error:
#     base_ang_vel 1, projected_gravity 1, joint_vel 2, joint_pos 3
# One step is the recorder's own offset; the joint block carries PLANT-7's
# delayed_joint_pos_rel / delayed_joint_vel_rel on top of it.
#
# The runtime itself adds NO delay -- the contract's obs_latency_note is explicit
# that real bus latency is what puts hardware inside the training distribution.
# So ObsBuilder stays lag-free and the TEST does the aligning.
BLOCK_LAG = {"base_ang_vel": 1, "projected_gravity": 1,
             "joint_vel": 2, "joint_pos": 3,
             "velocity_commands": 0, "actions": 0, "gait_phase": 0}


def measure_lag(obs_block, source, max_lag=5):
    """Re-derive a block's lag from the data rather than trusting BLOCK_LAG."""
    best, best_err = None, float("inf")
    for lag in range(max_lag):
        a = obs_block[lag:]
        b = source[:len(source) - lag] if lag else source
        n = min(len(a), len(b))
        err = float(np.abs(a[:n] - b[:n]).max())
        if err < best_err:
            best, best_err = lag, err
    return best, best_err


@needs_traces
@pytest.mark.parametrize("path", TRACES, ids=trace_ids())
def test_trace_block_lags_are_what_we_think(path):
    """Guards BLOCK_LAG against a change in the recorder or the latency model."""
    t = load(path)
    q_default = np.asarray(CONTRACT["q_default_rad"], dtype=np.float32)
    sources = {"base_ang_vel": t["root_ang_vel_b"],
               "projected_gravity": t["projected_gravity_b"],
               "joint_pos": t["joint_pos"] - q_default,
               "joint_vel": t["joint_vel"]}
    for name, src in sources.items():
        lo, hi = CONTRACT["obs_slices"][name]
        lag, err = measure_lag(t["obs"][:, lo:hi], src)
        assert lag == BLOCK_LAG[name], f"{name}: measured lag {lag}, expected {BLOCK_LAG[name]}"
        assert err < 1e-5, f"{name}: best-lag error {err}"


@needs_traces
@pytest.mark.parametrize("path", TRACES, ids=trace_ids())
def test_obs_builder_reproduces_isaac(path):
    """The load-bearing test: rebuild every recorded observation from its
    recorded inputs, each block fed at its own lag, and require 1e-5."""
    t = load(path)
    b = ObsBuilder(CONTRACT)
    obs_rec = t["obs"]
    n = obs_rec.shape[0]
    skip = max(BLOCK_LAG.values())
    worst = np.zeros(CONTRACT["obs_dim"], dtype=np.float64)

    for i in range(skip, n):
        la = BLOCK_LAG["actions"]
        j = i - la
        last_action = t["action"][j - 1] if j > 0 else np.zeros(14, dtype=np.float32)
        built = b.build(
            gyro_b=t["root_ang_vel_b"][i - BLOCK_LAG["base_ang_vel"]],
            gravity_b=t["projected_gravity_b"][i - BLOCK_LAG["projected_gravity"]],
            q_meas=t["joint_pos"][i - BLOCK_LAG["joint_pos"]],
            qd_meas=t["joint_vel"][i - BLOCK_LAG["joint_vel"]],
            command=t["command"][i - BLOCK_LAG["velocity_commands"]],
            last_action=last_action,
            gait_cos_sin=t["gait_phase"][i - BLOCK_LAG["gait_phase"]])
        worst = np.maximum(worst, np.abs(built - obs_rec[i]))

    bad = {k: float(worst[a:b_].max())
           for k, (a, b_) in CONTRACT["obs_slices"].items() if worst[a:b_].max() >= 1e-5}
    assert worst.max() < 1e-5, f"per-term max error: {bad}"


@needs_traces
@pytest.mark.parametrize("path", TRACES, ids=trace_ids())
def test_projected_gravity_matches_the_quaternion(path):
    """The recorded gravity vector and the one recomputed from root_quat_w must
    agree — otherwise the quaternion convention in imu.py is wrong."""
    t = load(path)
    err = max(float(np.abs(quat_to_projected_gravity(q) - g).max())
              for q, g in zip(t["root_quat_w"][::25], t["projected_gravity_b"][::25]))
    assert err < 1e-5, f"max |recomputed - recorded| = {err}"


@needs_traces
def test_obs_builder_rejects_nan():
    """A NaN reaching the policy shows up as a convulsing robot, so it must
    fail loudly at the boundary instead."""
    b = ObsBuilder(CONTRACT)
    with pytest.raises(ValueError, match="non-finite"):
        b.build(gyro_b=[np.nan, 0, 0], gravity_b=[0, 0, -1],
                q_meas=np.zeros(14), qd_meas=np.zeros(14), command=np.zeros(3),
                last_action=np.zeros(14), gait_cos_sin=(1.0, 0.0))


def test_obs_builder_rejects_wrong_widths():
    b = ObsBuilder(CONTRACT)
    with pytest.raises(ValueError):
        b.build(gyro_b=[0, 0], gravity_b=[0, 0, -1], q_meas=np.zeros(14),
                qd_meas=np.zeros(14), command=np.zeros(3),
                last_action=np.zeros(14), gait_cos_sin=(1.0, 0.0))


# --------------------------------------------------------- action decoder ---


@needs_traces
@pytest.mark.parametrize("path", TRACES, ids=trace_ids())
def test_action_decoder_reproduces_joint_targets(path):
    """DEPLOY-1: action_scale and q_default are NOT in the ONNX. Getting either
    wrong fails here by ~1 rad, not by rounding. Uses raw_target because Isaac
    does not clamp."""
    t = load(path)
    d = ActionDecoder(CONTRACT)
    err = max(float(np.abs(d.raw_target(a) - tgt).max())
              for a, tgt in zip(t["action"], t["joint_pos_target"]))
    assert err < 1e-5, f"max |raw_target - recorded joint_pos_target| = {err}"


@needs_traces
@pytest.mark.parametrize("path", TRACES, ids=trace_ids())
def test_clamp_count_is_reported_not_assumed(path):
    """The policy is unbounded and Isaac never clamped it, so a nonzero count is
    a fact about the policy rather than a bug here. Assert only that the
    reported count matches an independent computation."""
    t = load(path)
    d = ActionDecoder(CONTRACT)
    total_reported = 0
    total_independent = 0
    for a in t["action"]:
        raw = d.raw_target(a)
        _, n = d.decode(a)
        total_reported += n
        total_independent += int(np.count_nonzero(raw < d.soft_low)
                                 + np.count_nonzero(raw > d.soft_high))
    assert total_reported == total_independent


def test_hard_clamp_is_never_looser_than_soft():
    d = ActionDecoder(CONTRACT)
    assert np.all(d.hard_low <= d.soft_low + 1e-9)
    assert np.all(d.hard_high >= d.soft_high - 1e-9)


# ------------------------------------------------------------- gait phase ---


@needs_traces
@pytest.mark.parametrize("path", TRACES, ids=trace_ids())
def test_gait_phase_matches_trace(path):
    t = load(path)
    g = GaitPhase(CONTRACT["gait_nb_steps"])
    g.reset(int(t["step_idx"][0]))
    err = 0.0
    for i in range(t["gait_phase"].shape[0]):
        err = max(err, float(np.abs(np.array(g.step()) - t["gait_phase"][i]).max()))
    assert err < 1e-6, f"max phase error {err}"


def test_gait_phase_wraps_and_does_not_drift():
    g = GaitPhase(27)
    for _ in range(27 * 10):
        g.step()
    assert g.k == 0


def test_nb_steps_uniform():
    """The mod-27 shortcut is only valid while EVERY library entry shares a
    period. Re-derived here so a future reference library cannot break it
    silently."""
    import pickle
    found = {}
    for name in ("polynomial_coefficients.pkl", "polynomial_coefficients_v2.pkl"):
        hits = glob.glob(os.path.join(REPO, "**", name), recursive=True)
        if not hits:
            continue
        with open(hits[0], "rb") as fh:
            lib = pickle.load(fh)
        periods = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "nb_steps_in_period":
                        periods.add(int(v))
                    else:
                        walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)

        walk(lib)
        if periods:
            found[name] = periods
    if not found:
        pytest.skip("no reference library on disk")
    for name, periods in found.items():
        assert periods == {CONTRACT["gait_nb_steps"]}, \
            f"{name} has periods {periods}; the mod-N shortcut assumes one value"


# ------------------------------------------------------------------ onnx ----


@needs_traces
@pytest.mark.skipif(not os.path.isfile(ONNX), reason="ART-1: policy.onnx absent")
@pytest.mark.parametrize("path", TRACES[:1], ids=trace_ids()[:1])
def test_onnx_reproduces_recorded_actions(path):
    from jetson_runtime.policy import make_policy
    t = load(path)
    p = make_policy("onnx", ONNX, CONTRACT["obs_dim"], CONTRACT["action_dim"])
    err = max(float(np.abs(p.infer(t["obs"][i]) - t["action"][i]).max())
              for i in range(0, t["obs"].shape[0], 10))
    assert err < 1e-4, f"max |onnx - recorded| = {err}"


# ------------------------------------------------------------------ loop ----


@needs_traces
def test_loop_runs_end_to_end_on_a_trace():
    from jetson_runtime.loop import ControlLoop, NullActuatorSink, ReplaySensorSource
    from jetson_runtime.policy import make_policy
    if not os.path.isfile(ONNX):
        pytest.skip("ART-1: policy.onnx absent")
    t = load(TRACES[0])
    loop = ControlLoop(
        contract=CONTRACT, obs_builder=ObsBuilder(CONTRACT),
        decoder=ActionDecoder(CONTRACT),
        policy=make_policy("onnx", ONNX, CONTRACT["obs_dim"], CONTRACT["action_dim"]),
        gait=GaitPhase(CONTRACT["gait_nb_steps"]),
        sensors=ReplaySensorSource({k: t[k] for k in t.files if k != "meta"}),
        actuators=NullActuatorSink())
    out = loop.run(max_steps=100)
    assert len(out) == 100
    assert loop.stats["steps"] == 100
    assert all(np.all(np.isfinite(r["q_target"])) for r in out)


def test_replay_source_refuses_to_run_past_its_data():
    """Silently looping would let a test 'pass' on data it never validated."""
    from jetson_runtime.loop import ReplaySensorSource
    n = 3
    src = ReplaySensorSource({
        "root_ang_vel_b": np.zeros((n, 3)), "projected_gravity_b": np.zeros((n, 3)),
        "joint_pos": np.zeros((n, 14)), "joint_vel": np.zeros((n, 14)),
        "command": np.zeros((n, 3))})
    for _ in range(n):
        src.read()
    with pytest.raises(StopIteration):
        src.read()
