"""Task S.1 — the generated deployment contract must not drift from its sources.

`known_issues.md` DEPLOY-1 exists because the deployment constants lived only in
prose, and prose drifts: `AGENTS.md` once listed a right-leg-first joint order
that was the legacy 15-joint BDX table. The contract is generated from primary
sources precisely so that class of error becomes impossible.

These tests skip cleanly if the contract has not been generated yet, so the
suite is green before S.1 and meaningful after.
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(REPO, "jetson_runtime", "policy_contract.json")
POLICY_DIR = os.path.join(REPO, "exported_policies", "v7_servo_safe_ppo")


@pytest.fixture(scope="module")
def c():
    if not os.path.isfile(CONTRACT):
        pytest.skip("policy_contract.json not generated yet (Task S.1)")
    return json.load(open(CONTRACT))


@pytest.mark.phase4
def test_generator_check_exits_zero():
    """The anti-drift guard. If a primary source changes and the contract is not
    regenerated, this fails — which is the entire point of the task."""
    if not os.path.isfile(CONTRACT):
        pytest.skip("not generated yet")
    p = subprocess.run([sys.executable, "scripts/generate_policy_contract.py", "--check"],
                       cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, f"contract has drifted from its sources:\n{p.stdout}\n{p.stderr}"


@pytest.mark.phase4
def test_obs_slices_tile_the_vector_exactly(c):
    """Every byte of the observation must be accounted for, with no gaps and no
    overlaps — a silently mis-sliced observation is the failure this prevents."""
    spans = sorted(c["obs_slices"].values())
    assert spans[0][0] == 0, f"observation does not start at 0: {spans[0]}"
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert b1 == a2, f"gap or overlap between {(a1,b1)} and {(a2,b2)}"
    assert spans[-1][1] == c["obs_dim"], f"slices end at {spans[-1][1]}, obs_dim is {c['obs_dim']}"
    assert sum(t["dim"] for t in c["obs_terms"]) == c["obs_dim"]


@pytest.mark.phase4
def test_matches_the_shipped_onnx(c):
    """The contract must describe the policy that is actually exported."""
    dep = os.path.join(POLICY_DIR, "deployment_contract.json")
    if not os.path.isfile(dep):
        pytest.skip("no exported policy")
    d = json.load(open(dep))
    assert c["obs_dim"] == d["obs_dim"]
    assert c["action_dim"] == d["action_dim"]
    assert c["joint_order"] == d["joint_order"]
    assert c["onnx_md5"] == d["onnx_md5"], "contract describes a different ONNX"


@pytest.mark.phase4
def test_every_action_has_a_servo_id(c):
    """Since M0b the 14 policy outputs map 1:1 onto the 14 bus servos. If that
    ever stops being true the runtime cannot address a joint it is commanding."""
    missing = [j for j in c["joint_order"] if j not in c["servo_ids"]]
    assert not missing, f"policy commands joints with no servo id: {missing}"
    assert len(c["servo_ids"]) == c["action_dim"], (
        f"{len(c['servo_ids'])} servo ids for {c['action_dim']} actions")
    assert not [j for j in c["joint_order"] if "antenna" in j], (
        "an antenna is in the action space — it has no bus id and no encoder")


@pytest.mark.phase4
def test_deployment_clamp_is_inside_the_trained_hull(c):
    """The clamp is a CONSERVATIVE choice. A clamp wider than the hull would let
    the runtime command velocities the policy never saw."""
    hull, clamp = c["cmd_hull_trained"], c["cmd_clamp_deployment"]
    for k in ("vx", "vy", "wz"):
        assert clamp[k][0] >= hull[k][0] - 1e-9, f"{k} clamp low {clamp[k][0]} outside hull {hull[k]}"
        assert clamp[k][1] <= hull[k][1] + 1e-9, f"{k} clamp high {clamp[k][1]} outside hull {hull[k]}"
    assert clamp["wz"][1] < hull["wz"][1], (
        "the turn clamp is supposed to be deliberately TIGHTER than trained")


@pytest.mark.phase4
def test_soft_limits_are_inside_hard_limits(c):
    for j in c["joint_order"]:
        (hl, hh), (sl, sh) = c["hard_limits_rad"][j], c["soft_limits_rad"][j]
        assert hl <= sl < sh <= hh, f"{j}: soft {sl,sh} not inside hard {hl,hh}"


@pytest.mark.phase4
def test_q_default_is_inside_the_limits(c):
    """A standing pose outside its own joint limits would be clamped on the first
    control step, before the policy does anything."""
    for j, q in zip(c["joint_order"], c["q_default_rad"]):
        lo, hi = c["hard_limits_rad"][j]
        assert lo <= q <= hi, f"{j}: q_default {q} outside hard limits {lo, hi}"


@pytest.mark.phase4
def test_contract_py_has_no_literals():
    """The loader must read everything. A number typed here is a number that can
    disagree with the policy."""
    src = open(os.path.join(REPO, "jetson_runtime", "contract.py")).read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    for bad in ("0.25", "1.368", "left_hip_yaw", "0.02"):
        assert bad not in code, f"contract.py contains the literal {bad!r}"
