#!/usr/bin/env python3
"""Generate `deployment_contract.json` for an exported policy.

Task R3. The plan is explicit that this file must be **generated, not
hand-typed** — every field is copied from a source of truth that some other
check already pins, so the contract cannot drift from the policy it describes:

| field | source of truth |
|---|---|
| `joint_order`, `q_default_rad` | `scripts/duck_init_pos.json` (M0/R1) |
| `action_scale` | the run's archived `params/env.yaml` |
| `obs_dim`, `action_dim`, `usd_asset_hash`, `plant_mass_kg` | the eval JSON's `plant` block — i.e. what a **live env actually produced**, never a literal |
| `checkpoint_md5`, `onnx_md5` | the bytes on disk |

Why the sidecar exists at all (`known_issues.md` DEPLOY-1): `action_scale` and
`q_default` are not in the ONNX graph. `q_target = q_default + 0.25 * a` lives
inside Isaac Lab's `JointPositionAction`, so a runtime that commands the graph
output directly is wrong by a 4x gain and a standing-pose offset. This file is
the only thing that carries those two numbers to a Jetson.

    ~/IsaacLab/_isaac_sim/python.sh scripts/make_deployment_contract.py \
        --policy_dir exported_policies/v6d_contact_wrench_ppo \
        --eval_json  docs/jetson-mod/eval_results_rebuild/v6d_contact_wrench.json

Runs under plain `python3` too — it imports json/yaml/hashlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT_POS = os.path.join(REPO, "scripts", "duck_init_pos.json")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_dir", required=True)
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint basename inside policy_dir "
                         "(default: the single model_*.pt found there)")
    args = ap.parse_args()

    d = os.path.abspath(args.policy_dir)
    plant = json.load(open(args.eval_json))["plant"]
    obs_dim, action_dim = int(plant["obs_dim"]), int(plant["action_dim"])

    # --- joint order: the one whose LENGTH matches the action width. ---
    # M0b filtered the antennas out of the actions but left them in the
    # articulation, so duck_init_pos.json carries two orders (16 and 14) and
    # picking the wrong one silently maps every command to the wrong joint.
    ip = json.load(open(INIT_POS))
    cands = [ip.get("action_joint_order"), ip.get("joint_order")]
    order = next((c for c in cands if c and len(c) == action_dim), None)
    if order is None:
        print(f"FATAL: no joint order of length {action_dim} in duck_init_pos.json "
              f"(have {[len(c) for c in cands if c]})", file=sys.stderr)
        return 1
    q_default = [ip["init_pos_rad"][j] for j in order]

    # --- action_scale: parsed from the ARCHIVED env.yaml, not from env_cfg.py.
    # The archive is what this policy was trained under; env_cfg.py is what HEAD
    # says today, and those are allowed to diverge.
    env_yaml = os.path.join(d, "env.yaml")
    if not os.path.isfile(env_yaml):
        print(f"FATAL: {env_yaml} missing — copy params/env.yaml in first",
              file=sys.stderr)
        return 1
    import yaml
    # unsafe_load is REQUIRED: the archive carries 81 !!python/object and
    # !!python/tuple tags and safe_load raises ConstructorError. Do not "fix"
    # the yaml — it is an artefact.
    action_scale = float(yaml.unsafe_load(open(env_yaml))["actions"]["joint_pos"]["scale"])

    ckpt = args.checkpoint
    if ckpt is None:
        pts = sorted(f for f in os.listdir(d) if f.startswith("model_") and f.endswith(".pt"))
        if len(pts) != 1:
            print(f"FATAL: expected exactly one model_*.pt in {d}, found {pts}",
                  file=sys.stderr)
            return 1
        ckpt = pts[0]

    # --- DEPLOY-3: silence is a failure. Either name the unmeasurable dims, or
    # state that M0b removed them. A contract that just omits the question is
    # how the issue stayed open for three policy generations.
    if action_dim == 14:
        unmeasurable = {
            "note": "EMPTY BY CONSTRUCTION. DEPLOY-3 was closed by Task M0b, "
                    "which removed the two antenna joints from the action and "
                    "observation spaces (16->14 actions, 59->53 obs). Every "
                    "remaining dim is measurable on hardware. This is not an "
                    "oversight.",
        }
    else:
        unmeasurable = {
            "note": "antenna joints — see known_issues.md DEPLOY-3",
            "joint_pos_rel": [22, 23],
            "joint_vel_rel": [38, 39],
            "action_outputs": [13, 14],
        }

    n = action_dim
    contract = {
        "obs_dim": obs_dim,
        "obs_layout": [
            "base_ang_vel(3)", "projected_gravity(3)", "velocity_commands(3)",
            f"joint_pos_rel({n})", f"joint_vel_rel({n})", f"actions({n})",
            "gait_phase(2)",
        ],
        "action_dim": action_dim,
        "action_scale": action_scale,
        "action_formula": "q_target = q_default + action_scale * action",
        "joint_order": order,
        "q_default_rad": q_default,
        "joint_pos_obs_is_relative": True,
        "normalizer_epsilon": 0.01,
        "onnx_batch_dim": 1,
        "plant_mass_kg": float(plant["simulated_total_mass_kg"]),
        "usd_asset_hash": plant["usd_asset_hash"],
        "mjcf_sha256": plant.get("mjcf_sha256"),
        "checkpoint": ckpt,
        "checkpoint_md5": md5(os.path.join(d, ckpt)),
        "onnx_md5": md5(os.path.join(d, "policy.onnx")),
        "unmeasurable_obs_dims": unmeasurable,
    }

    out = os.path.join(d, "deployment_contract.json")
    with open(out, "w") as f:
        json.dump(contract, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
    print(f"  obs {obs_dim} / action {action_dim}, scale {action_scale}, "
          f"plant {contract['plant_mass_kg']:.6f} kg")
    print(f"  checkpoint {ckpt} md5 {contract['checkpoint_md5']}")
    print(f"  onnx md5 {contract['onnx_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
