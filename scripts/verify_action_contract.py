#!/usr/bin/env python3
"""Prove that scripts/duck_init_pos.json still describes the loaded articulation.

Why this exists: every exported policy's action vector is interpreted as

    q_target[i] = init_pos_rad[joint_order[i]] + 0.25 * action[i]

and `scripts/duck_init_pos.json` is the only written record of that mapping.
The joint order in it is the *USD articulation* order produced by the
MJCF-to-USD conversion — breadth-first and interleaved, matching neither the
MJCF order nor the Playground order (see AGENTS.md "Joint Orders").

The PLANT-1 fix on 2026-08-11 removed a body from the MJCF (`base` merged into
`trunk_assembly`). That commit verified the obs/action *dimensions* survive at
59/16. It did **not** verify that the 16 actuators still come out of the USD in
the same order. If the order moved, `duck_init_pos.json` is silently wrong,
every deployed action lands on the wrong joint, and every downstream evaluation
number is garbage — with nothing anywhere raising an error.

So: run this before spending GPU-hours, not after.

    cd ~/IsaacLab && ./isaaclab.sh -p \
      ~/Projects/Open_Duck_Mini_Jetson/scripts/verify_action_contract.py --headless

Exit status is the gate: **0** everything matches, **1** something moved.
If it exits 1, stop the phase and report — do not "adapt" anything.

The expected dimensions are CLI-overridable because Task M0b deliberately
changes them: dropping the antennas takes the policy observation 59 -> 53 and
the action space 16 -> 14. After M0b, run with
`--expect_obs_dim 53 --expect_action_dim 14`. (53, not 55: the `actions`
observation term is `last_action` with `action_name=None`, so it returns the
whole action tensor and shrinks with it.)
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT_POS_PATH = os.path.join(REPO_ROOT, "scripts", "duck_init_pos.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0",
    help="Gym task id to build (default: the v5d contact-wrench play task).",
)
parser.add_argument(
    "--num_envs", type=int, default=2,
    help="Environments to build; 2 is enough and starts faster than 64.",
)
parser.add_argument(
    "--expect_obs_dim", type=int, default=59,
    help="Expected policy observation width (53 after Task M0b).",
)
parser.add_argument(
    "--expect_action_dim", type=int, default=16,
    help="Expected action width (14 after Task M0b).",
)
parser.add_argument(
    "--tolerance", type=float, default=1e-6,
    help=("Max allowed |default_joint_pos - init_pos_rad| in radians. The "
          "float32 round-trip error on these values is < 1e-7."),
)

# AppLauncher must register its args and construct the app before any
# isaaclab/torch import, per the Isaac Lab launcher contract.
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, REPO_ROOT)
import isaac_lab_env.open_duck_mini_v2  # noqa: F401,E402  (registers the tasks)


def main() -> int:
    with open(INIT_POS_PATH) as f:
        data = json.load(f)
    expected_order = list(data["joint_order"])
    init_pos = data["init_pos_rad"]

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0",
                            num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    actual_order = list(robot.data.joint_names)
    failures = []

    print()
    print(f"task:      {args_cli.task}")
    print(f"contract:  {INIT_POS_PATH}")
    print()

    # ---- 1. joint order --------------------------------------------------
    if actual_order == expected_order:
        print(f"joint order MATCHES duck_init_pos.json "
              f"({len(actual_order)} joints)")
    else:
        failures.append("joint order differs from duck_init_pos.json")
        print("JOINT ORDER MISMATCH — the action mapping is wrong.")
        width = max(len(n) for n in actual_order + expected_order)
        print(f"  {'idx':>3}  {'duck_init_pos.json':<{width}}  "
              f"{'articulation':<{width}}")
        for i in range(max(len(actual_order), len(expected_order))):
            want = expected_order[i] if i < len(expected_order) else "-"
            got = actual_order[i] if i < len(actual_order) else "-"
            flag = "" if want == got else "   <-- DIFFERS"
            print(f"  {i:>3}  {want:<{width}}  {got:<{width}}{flag}")

    # ---- 2. default joint positions -------------------------------------
    # Checked by NAME against the articulation's own index, so this stays
    # meaningful even when check 1 failed.
    print()
    worst_name, worst_delta = None, 0.0
    missing = [n for n in actual_order if n not in init_pos]
    if missing:
        failures.append(f"init_pos_rad has no entry for {missing}")
        print(f"init_pos_rad is MISSING entries for: {missing}")
    for i, name in enumerate(actual_order):
        if name not in init_pos:
            continue
        delta = abs(float(robot.data.default_joint_pos[0, i]) - init_pos[name])
        if delta > worst_delta:
            worst_name, worst_delta = name, delta
        if delta > args_cli.tolerance:
            failures.append(f"default_joint_pos[{name}] off by {delta:.3e} rad")
            print(f"  DEFAULT POS MISMATCH {name}: "
                  f"articulation {float(robot.data.default_joint_pos[0, i]):+.6f} "
                  f"vs contract {init_pos[name]:+.6f} (delta {delta:.3e})")
    print(f"default joint positions: worst delta {worst_delta:.3e} rad "
          f"on '{worst_name}' (tolerance {args_cli.tolerance:g})")

    # ---- 3. interface dimensions ----------------------------------------
    obs_dim = int(env.observation_manager.group_obs_dim["policy"][0])
    action_dim = int(env.action_manager.total_action_dim)
    print()
    print(f"observation dim: {obs_dim} (expected {args_cli.expect_obs_dim})")
    print(f"action dim:      {action_dim} (expected {args_cli.expect_action_dim})")
    if obs_dim != args_cli.expect_obs_dim:
        failures.append(f"obs dim {obs_dim} != {args_cli.expect_obs_dim}")
    if action_dim != args_cli.expect_action_dim:
        failures.append(
            f"action dim {action_dim} != {args_cli.expect_action_dim}")
    if action_dim != len(expected_order):
        failures.append(
            f"action dim {action_dim} != {len(expected_order)} contract joints")

    # ---- 4. plant identity, for the record -------------------------------
    body_names = list(robot.data.body_names)
    print()
    print(f"articulation root: {body_names[0]}")
    print(f"rigid bodies:      {len(body_names)}")
    print(f"simulated mass:    {float(robot.data.default_mass[0].sum()):.6f} kg")

    print()
    if failures:
        print(f"VERDICT: FAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("VERDICT: PASS — the action contract is intact.")
    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        # Same two Isaac Lab hazards audit_plant_mass.py documents, verbatim:
        # simulation_app.close() never returns and pins the exit status to 0,
        # and Kit leaves stdout block-buffered on redirect so the report
        # vanishes without an explicit flush before os._exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
