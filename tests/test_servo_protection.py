"""SERVO-1 / SERVO-2 — the reward terms that keep the servos inside their envelope.

Both defects were measured, not alleged:

* **SERVO-1** — `neck_pitch` sat on its lower mechanical stop for **100 %** of
  forward-walking steps with **0.003°** of travel, while the policy commanded
  **4° past** it, costing **3.28 N·m RMS** into a hard stop. Cause: the reward
  term commented *"Joint limits: protect servos"* listed only ankles and knees.
* **SERVO-2** — worst leg joint **2.735 N·m RMS = 174 %** of the STS3250's
  16 kg·cm rated torque, with four joints at the 4.903 N·m clip. Cause:
  `dof_torques_l2` and `dof_acc_l2` were both `None` — nothing priced torque.

These parse the **AST**, not the source text. `known_issues.md` **TEST-1**
records that grepped literals in this repo are non-unique, and a string match on
`"neck_pitch"` would pass on the `joint_deviation_head` term that was already
there and never caught the bug.
"""

import ast
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_CFG = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py")
ROBOT_CFG = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py")
MJCF = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml")

HEAD_JOINTS = {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}


def _assignments(path):
    """Every `name = <expr>` at class scope, keyed by name, last one wins."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value
    return out


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _joint_names_of(term):
    """Pull the joint_names list out of a RewTerm(params={"asset_cfg": SceneEntityCfg(...)})."""
    params = _kwarg(term, "params")
    assert params is not None, "term has no params"
    for k, v in zip(params.keys, params.values):
        if isinstance(k, ast.Constant) and k.value == "asset_cfg":
            jn = _kwarg(v, "joint_names")
            return [e.value for e in jn.elts]
    raise AssertionError("no asset_cfg in params")


@pytest.fixture(scope="module")
def terms():
    return _assignments(ENV_CFG)


@pytest.mark.phase2
def test_head_joints_are_limit_protected(terms):
    """SERVO-1. The term that exists to protect servos must cover the joint that
    was actually being destroyed."""
    names = set(_joint_names_of(terms["joint_pos_limits"]))
    missing = HEAD_JOINTS - names
    assert not missing, (
        f"joint_pos_limits does not protect {sorted(missing)} — this is the exact "
        f"omission that let the policy park neck_pitch on its end stop for 100 % "
        f"of forward-walking steps at 3.28 N·m")


@pytest.mark.phase2
def test_torque_penalty_is_enabled(terms):
    """SERVO-2. `dof_torques_l2` was None; nothing priced torque."""
    t = terms["dof_torques_l2"]
    assert not (isinstance(t, ast.Constant) and t.value is None), (
        "dof_torques_l2 is None — nothing in the reward function prices torque")
    w = _kwarg(t, "weight")
    val = ast.literal_eval(w)
    assert val < 0, f"torque penalty must be negative, got {val}"
    # Band supported by the derivation in torque_rewards.py: measured
    # mean sum(tau^2) = 51.84, so -1e-2 costs 0.518/step ~ 4.7 % of the budget.
    # -1e-5 (the legged_gym value) would cost 0.0005 — 90x below the smallest
    # penalty already present, i.e. inert.
    assert -5e-2 <= val <= -1e-3, (
        f"weight {val} is outside the band the measured sum(tau^2)=51.84 "
        f"supports; see torque_rewards.joint_torques_commanded_l2")


@pytest.mark.phase2
def test_torque_penalty_uses_preclip_torque(terms):
    """The stock mdp.joint_torques_l2 squares POST-clip torque. Four leg joints
    already sit at the clip, so there it is a constant with no gradient — the
    exact region SERVO-2 is about. Guard against a future 'simplification'."""
    t = terms["dof_torques_l2"]
    # Fail, do not error, when the term is absent — an AttributeError from
    # ast.Constant is a much harder failure to read than an assertion.
    assert not (isinstance(t, ast.Constant) and t.value is None), (
        "dof_torques_l2 is None, so there is no term to check the tensor of")
    func = _kwarg(t, "func")
    src = ast.unparse(func)
    assert "joint_torques_commanded_l2" in src, (
        f"torque penalty uses {src!r}; it must use the PRE-CLIP term "
        f"torque_rewards.joint_torques_commanded_l2 or it has no gradient "
        f"where the policy is saturated")
    assert not src.startswith("mdp."), "must not revert to the stock post-clip term"


@pytest.mark.phase2
def test_soft_limit_actually_binds():
    """The fix can only work if the resting angle is OUTSIDE the soft limit that
    `mdp.joint_pos_limits` penalises. Computed from the MJCF range and the
    configured factor — no literals."""
    import xml.etree.ElementTree as ET

    root = ET.parse(MJCF).getroot()
    rng = None
    for j in root.iter("joint"):
        if j.get("name") == "neck_pitch":
            rng = [float(v) for v in j.get("range").split()]
    assert rng, "neck_pitch has no range in the MJCF"

    m = re.search(r"soft_joint_pos_limit_factor\s*=\s*([0-9.]+)", open(ROBOT_CFG).read())
    assert m, "soft_joint_pos_limit_factor not found"
    factor = float(m.group(1))

    lo, hi = rng
    mid, span = (lo + hi) / 2.0, hi - lo
    soft_lo = mid - 0.5 * span * factor
    MEASURED_REST = -0.349066  # rad; where v6d parks it = the hard lower stop
    assert MEASURED_REST < soft_lo, (
        f"the measured resting angle {MEASURED_REST:.6f} rad is INSIDE the soft "
        f"limit {soft_lo:.6f}, so joint_pos_limits would apply zero penalty and "
        f"the SERVO-1 fix cannot work")


@pytest.mark.phase2
def test_live_weight_matches_the_shipped_policy():
    """HEAD must reproduce what shipped.

    Caught for real on 2026-08-16: the live config still carried iteration 2's
    -4e-2 — the weight that FAILED three bars — while the shipped policy was
    trained at -1e-2. Anyone retraining from HEAD would have reproduced the
    failing policy and had no way to know.
    """
    import json
    yaml = pytest.importorskip("yaml")
    archived = os.path.join(REPO_ROOT, "exported_policies", "v7_servo_safe_ppo", "env.yaml")
    if not os.path.isfile(archived):
        pytest.skip("no shipped archive to compare against")
    # unsafe_load: the archive carries !!python/object tags. Do not "fix" it.
    shipped = float(yaml.unsafe_load(open(archived))["rewards"]["dof_torques_l2"]["weight"])
    live = float(ast.literal_eval(_kwarg(_assignments(ENV_CFG)["dof_torques_l2"], "weight")))
    assert abs(live - shipped) < 1e-12, (
        f"env_cfg.py has weight {live} but the SHIPPED policy "
        f"(exported_policies/v7_servo_safe_ppo) was trained at {shipped}. "
        f"Retraining from HEAD would not reproduce what is deployed.")


@pytest.mark.phase2
def test_effort_limit_unchanged():
    """Design decision D2: the plant must stay byte-identical to v6d's so its ten
    existing eval JSONs remain a valid control. `evaluate_policies.py` does NOT
    stamp effort_limit_sim, so regate_report.py's exit-2 mixing guard would not
    catch a change here — this test is the only guard."""
    m = re.search(r"STS3250_EFFORT_LIMIT_NM\s*=\s*([0-9.]+)", open(ROBOT_CFG).read())
    assert m, "STS3250_EFFORT_LIMIT_NM not found"
    assert abs(float(m.group(1)) - 4.903) < 1e-9, (
        "effort_limit changed — v6d's eval JSONs are no longer a valid control "
        "and the plan's D2 saving is void; see v7_servo_fix_plan.md §8")
