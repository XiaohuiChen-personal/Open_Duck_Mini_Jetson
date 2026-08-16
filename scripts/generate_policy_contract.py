#!/usr/bin/env python3
"""Generate `jetson_runtime/policy_contract.json` — Task S.1.

The exported ONNX is a bare MLP. Everything around it — what the 53 inputs mean,
what the 14 outputs mean, how an output becomes a servo command — lives only as
prose spread across several documents. `known_issues.md` DEPLOY-1 measures the
cost of getting it wrong: commanding the graph output directly is a 4x gain error
plus a standing-pose offset of up to 1.378 rad. Total, silent failure.

**Every field is re-derived from a primary source at run time.** The only
unavoidable literal is the servo-ID table, which is marked as such below. If you
find yourself pasting a joint order or a q_default into this file, stop — that is
the drift this script exists to prevent.

    python3 scripts/generate_policy_contract.py           # write
    python3 scripts/generate_policy_contract.py --check   # verify, exit 1 on drift

Pure stdlib + yaml + mujoco. No Isaac, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO, "exported_policies", "v7_servo_safe_ppo")
OUT = os.path.join(REPO, "jetson_runtime", "policy_contract.json")
MJCF = os.path.join(REPO, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml")

# ---------------------------------------------------------------- literals
# THE ONLY LITERAL IN THIS FILE. Verified identical in two places:
#   docs/configure_motors.md:22-38  and  experiments/v2/configure_motors.py:10-25
# The two antennas have NO bus id — they are SG90 PWM servos on Jetson header
# pins, and since Task M0b they are not in the action space at all, so the 14
# ids below map 1:1 onto the 14 policy outputs.
SERVO_IDS = {
    "right_hip_yaw": 10, "right_hip_roll": 11, "right_hip_pitch": 12,
    "right_knee": 13, "right_ankle": 14,
    "left_hip_yaw": 20, "left_hip_roll": 21, "left_hip_pitch": 22,
    "left_knee": 23, "left_ankle": 24,
    "neck_pitch": 30, "head_pitch": 31, "head_yaw": 32, "head_roll": 33,
}
SERVO_BAUD = 1_000_000        # experiments/v2/configure_motors.py:29-32

# Deployment choice, NOT the trained hull. AGENTS.md § Safety Rules. The turn
# clamp is deliberately tighter than the trained range; do not widen it to match.
CMD_CLAMP = {"vx": [-0.148, 0.222], "vy": [-0.111, 0.111], "wz": [-0.3, 0.3]}


def load_archived_env():
    """yaml.unsafe_load is REQUIRED: the archive carries !!python/object and
    !!python/tuple tags and safe_load raises ConstructorError. The file is
    repo-local and trusted (it is our own training artifact), so this is
    acceptable. Do not "fix" the yaml — it is an archived record."""
    import yaml
    return yaml.unsafe_load(open(os.path.join(POLICY_DIR, "env.yaml")))


def obs_layout(env):
    """Term order and widths, read from the archive's own ordering."""
    policy = env["observations"]["policy"]
    order, slices, i = [], {}, 0
    for name, term in policy.items():
        if not isinstance(term, dict) or "func" not in term:
            continue          # concatenate_terms / enable_corruption flags
        n = _term_width(name, term, env)
        order.append({"name": name,
                      "func": str(term["func"]).split(":")[-1],
                      "dim": n})
        slices[name] = [i, i + n]
        i += n
    return order, slices, i


def _term_width(name, term, env):
    """Widths are structural, not configurable: 3 for the vector terms, 2 for
    the gait phase, and n_actioned_joints for the joint blocks."""
    n_act = len(action_joint_order())
    if name in ("base_ang_vel", "projected_gravity", "velocity_commands"):
        return 3
    if name == "gait_phase":
        return 2
    if name in ("joint_pos", "joint_vel", "actions"):
        return n_act
    raise SystemExit(f"unknown observation term {name!r} — refusing to guess its width")


def action_joint_order():
    """The 14-name order the policy's outputs are in. Source of truth is
    scripts/duck_init_pos.json, which R1 wrote from a live articulation."""
    d = json.load(open(os.path.join(REPO, "scripts", "duck_init_pos.json")))
    order = d.get("action_joint_order")
    if not order:
        raise SystemExit("duck_init_pos.json has no action_joint_order")
    return order


def joint_limits():
    """Hard limits from the MJCF via mujoco. Skip the free joint: it is joint 0,
    its name is None and its range is [0, 0]."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(MJCF)
    out = {}
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm is None:
            continue
        lo, hi = (float(x) for x in m.jnt_range[i])
        if lo == 0.0 and hi == 0.0:
            continue          # unlimited / free
        out[nm] = [lo, hi]
    return out


def gait_period(env):
    """Control steps per gait cycle, measured from the reference library the
    archive names (None -> the default library, imitation_reward.py)."""
    ref = env["rewards"]["imitation_reward"]["params"].get("reference_pkl")
    path = ref or os.path.join(REPO, "isaac_lab_env", "open_duck_mini_v2",
                               "data", "polynomial_coefficients.pkl")
    lib = pickle.load(open(path, "rb"))
    entries = lib.values() if isinstance(lib, dict) else lib
    steps = set()
    for e in entries:
        if isinstance(e, dict) and "nb_steps_in_period" in e:
            steps.add(int(e["nb_steps_in_period"]))
    if len(steps) != 1:
        raise SystemExit(f"reference library has non-uniform gait periods: {sorted(steps)}")
    return os.path.basename(path), steps.pop()


def build():
    env = load_archived_env()
    contract = json.load(open(os.path.join(POLICY_DIR, "deployment_contract.json")))
    order = action_joint_order()
    init = json.load(open(os.path.join(REPO, "scripts", "duck_init_pos.json")))

    terms, slices, obs_dim = obs_layout(env)
    if obs_dim != contract["obs_dim"]:
        raise SystemExit(f"derived obs_dim {obs_dim} != the ONNX's {contract['obs_dim']}")

    hard = joint_limits()
    missing = [j for j in order if j not in hard]
    if missing:
        raise SystemExit(f"no MJCF range for {missing}")
    factor = 0.9          # robot_cfg.py soft_joint_pos_limit_factor
    soft = {}
    for j in order:
        lo, hi = hard[j]
        mid, span = (lo + hi) / 2.0, hi - lo
        soft[j] = [mid - 0.5 * span * factor, mid + 0.5 * span * factor]

    ref_name, gait_steps = gait_period(env)
    ranges = env["commands"]["base_velocity"]["ranges"]

    return {
        "_generated_by": "scripts/generate_policy_contract.py — DO NOT HAND-EDIT",
        "policy_dir": os.path.relpath(POLICY_DIR, REPO),
        "checkpoint_md5": contract["checkpoint_md5"],
        "onnx_md5": contract["onnx_md5"],

        "obs_dim": obs_dim,
        "obs_terms": terms,
        "obs_slices": slices,
        "action_dim": len(order),
        "joint_order": order,

        "action_formula": "q_target = q_default + action_scale * action",
        "action_scale": float(env["actions"]["joint_pos"]["scale"]),
        "action_uses_default_offset": bool(env["actions"]["joint_pos"]["use_default_offset"]),
        "q_default_rad": [init["init_pos_rad"][j] for j in order],

        "control_dt_s": float(env["sim"]["dt"]) * int(env["decimation"]),
        "sim_dt_s": float(env["sim"]["dt"]),
        "decimation": int(env["decimation"]),

        "gait_reference_pkl": ref_name,
        "gait_nb_steps": gait_steps,

        "cmd_hull_trained": {"vx": list(ranges["lin_vel_x"]),
                             "vy": list(ranges["lin_vel_y"]),
                             "wz": list(ranges["ang_vel_z"])},
        "cmd_clamp_deployment": CMD_CLAMP,
        "cmd_note": ("cmd_clamp_deployment is a CONSERVATIVE deployment choice, not "
                     "the trained hull: wz is clamped tighter than trained. Never "
                     "widen it to match cmd_hull_trained. AGENTS.md Safety Rules."),

        "hard_limits_rad": {j: hard[j] for j in order},
        "soft_limits_rad": soft,
        "soft_limit_factor": factor,
        "clamp_note": ("The policy output is UNBOUNDED: clip_actions is null and "
                       "JointPositionAction applies no limit clamp. In sim PhysX "
                       "absorbs an out-of-range target; on hardware NOTHING does. "
                       "The runtime MUST clamp q_target to hard_limits_rad."),

        "servo_ids": SERVO_IDS,
        "servo_baud": SERVO_BAUD,
        "servo_note": ("14 bus servos == the 14 policy outputs exactly, since Task "
                       "M0b removed the antennas from the action space. The two "
                       "antennas are SG90 PWM parts with no bus id and no policy "
                       "output."),

        "obs_latency_note": (
            "The joint_pos / joint_vel terms are delayed_joint_pos_rel and "
            "delayed_joint_vel_rel (PLANT-7): the policy was TRAINED on joint "
            "readings delayed 0-40 ms, per-env, because real hardware has sensor "
            "and bus latency. The runtime must NOT add artificial delay — feed "
            "the freshest reading available. The point is that real latency up to "
            "~40 ms is inside the training distribution, not that delay is "
            "required. Measure the real figure in Task S.6."),

        "normalizer_epsilon": 0.01,
        "normalizer_note": ("Inside the ONNX graph (the Sub/Div nodes). Only a "
                            "hand-rolled reimplementation of policy.pt needs it."),
        "onnx_batch_dim": 1,
        "plant_mass_kg": contract["plant_mass_kg"],
        "usd_asset_hash": contract["usd_asset_hash"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and diff against disk; exit 1 on drift")
    args = ap.parse_args()

    fresh = build()
    if args.check:
        if not os.path.isfile(OUT):
            print(f"FAIL: {OUT} does not exist", file=sys.stderr)
            return 1
        disk = json.load(open(OUT))
        if disk == fresh:
            print(f"OK: {os.path.relpath(OUT, REPO)} matches its sources")
            return 0
        for k in sorted(set(disk) | set(fresh)):
            if disk.get(k) != fresh.get(k):
                print(f"DRIFT {k}:\n  disk  {str(disk.get(k))[:150]}\n  fresh {str(fresh.get(k))[:150]}")
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fresh, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(OUT, REPO)}")
    print(f"  obs {fresh['obs_dim']} / action {fresh['action_dim']}, "
          f"scale {fresh['action_scale']}, {fresh['control_dt_s']*1000:.0f} ms control period")
    print(f"  gait {fresh['gait_nb_steps']} steps ({fresh['gait_nb_steps']*fresh['control_dt_s']:.2f} s) "
          f"from {fresh['gait_reference_pkl']}")
    print(f"  trained hull wz {fresh['cmd_hull_trained']['wz']} -> "
          f"deployment clamp {fresh['cmd_clamp_deployment']['wz']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
