"""Task R3 — the sidecar that stops a Jetson runtime from being silently wrong.

`known_issues.md` DEPLOY-1: `action_scale = 0.25` and `q_default` are **not in
the ONNX graph**. `q_target = q_default + 0.25 * a` lives entirely inside Isaac
Lab's `JointPositionAction`, so a runtime that commands the ONNX output directly
is wrong by a **4x gain and a standing-pose offset up to 1.379 rad** — total,
silent failure. The sidecar carries what the graph does not.

Every test skips when the contract is absent, so the suite is green before R3
and meaningful after. These parse JSON only — no ONNX import — so they run under
plain `python3`. The graph-level checks live in
`scripts/verify_deployment_contract.py`, which needs the Isaac interpreter.
"""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO_ROOT, "exported_policies", "v7_servo_safe_ppo")
CONTRACT = os.path.join(POLICY_DIR, "deployment_contract.json")
INIT_POS = os.path.join(REPO_ROOT, "scripts", "duck_init_pos.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "docs", "jetson-mod", "eval_results_rebuild")


@pytest.fixture(scope="module")
def contract():
    if not os.path.isfile(CONTRACT):
        pytest.skip("deployment_contract.json not produced yet (Task R3)")
    return json.load(open(CONTRACT))


@pytest.fixture(scope="module")
def init_pos():
    return json.load(open(INIT_POS))


@pytest.mark.phase4
def test_contract_matches_duck_init_pos(contract, init_pos):
    """The runtime builds `q_target` from these two fields. If they disagree
    with the repo's own record, every commanded angle lands on the wrong joint
    or at the wrong offset."""
    order = contract["joint_order"]
    n = contract["action_dim"]
    candidates = [init_pos.get("action_joint_order"), init_pos.get("joint_order")]
    expected = next((c for c in candidates if c and len(c) == n), None)
    assert expected is not None, (
        f"duck_init_pos.json has no joint order of length {n}")
    assert order == expected, f"contract order {order} != {expected}"

    qd = contract["q_default_rad"]
    assert len(qd) == len(order)
    for name, v in zip(order, qd):
        assert abs(v - init_pos["init_pos_rad"][name]) < 1e-12, (
            f"q_default_rad[{name}] = {v} != "
            f"{init_pos['init_pos_rad'][name]}")


@pytest.mark.phase4
def test_action_scale_matches_env_cfg(contract):
    """0.25 matches Open Duck Playground and is the single most load-bearing
    scalar in the whole deployment path."""
    assert abs(contract["action_scale"] - 0.25) < 1e-12
    env_yaml = os.path.join(POLICY_DIR, "env.yaml")
    if not os.path.isfile(env_yaml):
        pytest.skip("archived env.yaml not present")
    yaml = pytest.importorskip("yaml")
    # safe_load FAILS here: 81 !!python/object and !!python/tuple tags. Do not
    # "fix" the yaml — it is an archived artefact.
    y = yaml.unsafe_load(open(env_yaml))
    assert abs(float(y["actions"]["joint_pos"]["scale"])
               - contract["action_scale"]) < 1e-12


@pytest.mark.phase4
def test_dims_are_self_consistent(contract):
    assert contract["action_dim"] == len(contract["joint_order"])
    assert contract["action_dim"] == len(contract["q_default_rad"])
    # The layout string must add up to obs_dim.
    n = contract["action_dim"]
    expected_obs = 3 + 3 + 3 + n + n + n + 2
    assert contract["obs_dim"] == expected_obs, (
        f"obs_dim {contract['obs_dim']} != 3+3+3+{n}+{n}+{n}+2 = {expected_obs}. "
        "Note 53 (not 55) for n=14: the `actions` term is last_action with "
        "action_name=None, so it returns the whole action tensor and shrinks "
        "with it.")


@pytest.mark.phase4
def test_dims_match_the_measured_plant(contract):
    """The contract's dims must equal what a live env actually produced, which
    the eval JSON's `plant` block records."""
    j = os.path.join(RESULTS_DIR, "v7_servo_safe.json")
    if not os.path.isfile(j):
        pytest.skip("v6d eval JSON not produced yet")
    plant = json.load(open(j))["plant"]
    assert contract["obs_dim"] == plant["obs_dim"]
    assert contract["action_dim"] == plant["action_dim"]
    assert contract.get("usd_asset_hash") == plant.get("usd_asset_hash")


@pytest.mark.phase4
def test_plant_mass_matches_the_results_dir(contract):
    """The row whose absence made PLANT-1 invisible for three policy
    generations."""
    import re
    readme = os.path.join(RESULTS_DIR, "README.md")
    if not os.path.isfile(readme):
        pytest.skip("eval_results_rebuild/README.md missing")
    m = re.search(r"`simulated_total_mass_kg`\s*\|\s*`([0-9.]+)`", open(readme).read())
    assert m, "the results README does not declare a plant mass"
    assert abs(contract["plant_mass_kg"] - float(m.group(1))) < 1e-3


@pytest.mark.phase4
def test_deploy3_is_addressed(contract):
    """One of two things must be true, and silence is a failure.

    Either the antennas are still in the interface, in which case the contract
    must name the four unmeasurable observation dims and two action outputs so
    a runtime author cannot miss them; or M0b removed them, in which case the
    field must be empty AND say so. A contract that simply omits the question
    is how DEPLOY-3 stayed open for three policy generations.
    """
    assert "unmeasurable_obs_dims" in contract, (
        "the contract is silent on DEPLOY-3")
    u = contract["unmeasurable_obs_dims"]
    n = contract["action_dim"]
    if n == 14:
        real = {k: v for k, v in u.items() if not k.startswith("note")}
        assert not real, (
            f"action_dim is 14, so M0b removed the antennas, but the contract "
            f"still lists unmeasurable dims: {real}")
        note = " ".join(str(v) for k, v in u.items() if k.startswith("note"))
        assert "DEPLOY-3" in note and "M0b" in note, (
            "an empty unmeasurable_obs_dims must carry a note saying DEPLOY-3 "
            "was closed by M0b, or a reader cannot tell it from an oversight")
    else:
        assert u.get("joint_pos_rel") == [22, 23]
        assert u.get("joint_vel_rel") == [38, 39]
        assert u.get("action_outputs") == [13, 14]


@pytest.mark.phase4
def test_contract_records_its_provenance(contract):
    """An ONNX that cannot be traced to a checkpoint is not deployable.
    `*.onnx` is gitignored (ART-1), so the md5 here is the ONLY provenance."""
    for key in ("checkpoint_md5", "onnx_md5", "usd_asset_hash", "plant_mass_kg"):
        assert contract.get(key), f"contract has no {key}"
    assert len(contract["checkpoint_md5"]) == 32
    assert len(contract["onnx_md5"]) == 32


@pytest.mark.phase4
def test_relative_joint_obs_is_flagged(contract):
    """DEPLOY-2: feeding absolute encoder angles puts the knees 13-14 sigma
    outside the training distribution."""
    assert contract.get("joint_pos_obs_is_relative") is True
    assert "q_default" in contract["action_formula"]
