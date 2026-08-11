#!/usr/bin/env python3
"""Report the mass/inertia PhysX actually simulates, versus the MJCF and CAD.

Why this exists: the MJCF root body ``base`` is a pure kinematic frame — it
carries the ``<freejoint>`` and declares no ``<inertial>`` and no ``<geom>``.
MuJoCo gives such a body mass 0. PhysX instead fills the unauthored value with
its own default, so the simulated plant is heavier than the robot. Nothing on
disk is wrong; the divergence appears only at load time, which is why it went
unnoticed. See docs/jetson-mod/known_issues.md (PLANT-1).

Run it after ANY change to robot_motors.xml, the STLs, or the USD conversion:

    cd ~/IsaacLab && ./isaaclab.sh -p \
      ~/Projects/Open_Duck_Mini_Jetson/scripts/audit_plant_mass.py --headless

Exits 1 if the simulated total differs from the MJCF total by more than
--tolerance kg, so it can be used as a gate.

PhysX also announces this itself on every run that loads the robot, buried in
Kit's startup output:

    [Warning] [omni.physx.plugin] The rigid body at
    /World/envs/env_0/Robot/base/base has a possibly invalid inertia tensor of
    {1.0, 1.0, 1.0} and a negative mass, small sphere approximated inertia was
    used. Either specify correct values in the mass properties, or add
    collider(s) to any shape(s) that you wish to automatically compute mass
    properties for.

"Negative mass" is USD's -1.0 unauthored sentinel; "{1.0, 1.0, 1.0}" is the
identity inertia the MJCF converter writes when a body has no <inertial>.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(
    REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--tolerance",
    type=float,
    default=1e-3,
    help="max allowed |simulated - MJCF| total mass, in kg (default: 1e-3)",
)

# AppLauncher must register its args and construct the app before any
# isaaclab/torch import, per the Isaac Lab launcher contract.
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, REPO_ROOT)
import isaac_lab_env.open_duck_mini_v2  # noqa: F401,E402  (registers the tasks)

TASK = "Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0"


def mjcf_masses():
    """Per-body mass as authored in the MJCF. None = no <inertial> tag."""
    root = ET.parse(MJCF_PATH).getroot()
    out = {}
    for body in root.iter("body"):
        inertial = body.find("inertial")
        out[body.get("name")] = (
            float(inertial.get("mass")) if inertial is not None else None
        )
    return out


def main() -> int:
    authored = mjcf_masses()
    mjcf_total = sum(m for m in authored.values() if m is not None)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    names = list(robot.data.body_names)
    # default_mass/default_inertia are the values PhysX loaded, BEFORE any
    # randomization event mutates them -- exactly the asset-side question.
    sim_mass = robot.data.default_mass[0].cpu().numpy()
    sim_inertia = robot.data.default_inertia[0].cpu().numpy()

    print()
    print(f"MJCF: {MJCF_PATH}")
    print(f"task: {TASK}")
    print()
    header = f"{'body':<30}{'MJCF (kg)':>14}{'PhysX (kg)':>14}{'delta':>12}   note"
    print(header)
    print("-" * len(header))

    unauthored = []
    for i, name in enumerate(names):
        declared = authored.get(name)
        simulated = float(sim_mass[i])
        if declared is None:
            note = "NO <inertial> IN MJCF -> PhysX default"
            unauthored.append((name, simulated, sim_inertia[i]))
            shown, delta = "none", f"{simulated:+.6f}"
        else:
            # PhysX stores mass as float32, so an exact compare against the
            # float64 parsed from XML flags every body. Compare at float32
            # resolution instead, or this cries wolf on all 21 real bodies.
            same = math.isclose(simulated, declared, rel_tol=1e-6, abs_tol=1e-9)
            note = "" if same else "MISMATCH"
            shown, delta = f"{declared:.6f}", f"{simulated - declared:+.6f}"
        print(f"{name:<30}{shown:>14}{simulated:>14.6f}{delta:>12}   {note}")

    sim_total = float(sim_mass.sum())
    excess = sim_total - mjcf_total
    print("-" * len(header))
    print(f"{'TOTAL':<30}{mjcf_total:>14.6f}{sim_total:>14.6f}{excess:>+12.6f}")
    print()

    for name, mass, inertia in unauthored:
        # Isaac Lab flattens the 3x3 inertia tensor row-major into 9 values.
        diag = (inertia[0], inertia[4], inertia[8])
        print(
            f"unauthored body '{name}': PhysX assigned mass={mass:.6f} kg, "
            f"diagonal inertia=({diag[0]:.3e}, {diag[1]:.3e}, {diag[2]:.3e}) kg*m^2"
        )
    if unauthored:
        print(
            "  -> These are PhysX defaults, not data from the MJCF. "
            "MjcfConverterCfg.import_inertia_tensor documents this path: "
            '"If the inertial tag is missing, then it is imported as an identity."'
        )
    print()

    ok = abs(excess) <= args_cli.tolerance
    print(
        f"VERDICT: {'PASS' if ok else 'FAIL'} — simulated total is "
        f"{excess:+.6f} kg ({100.0 * excess / mjcf_total:+.1f}%) versus the MJCF"
        f" (tolerance {args_cli.tolerance} kg)"
    )
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        # Two Isaac Lab hazards, both hit while writing this script; measured,
        # not assumed.
        #
        # 1. simulation_app.close() NEVER RETURNS -- it terminates the process
        #    with status 0. A print or sys.exit() placed after it is dead code,
        #    which silently pins this script's exit status to 0 and would break
        #    the gate the docstring promises. So do not call it here: env.close()
        #    above already released the environment, and os._exit ends Kit the
        #    same way close() would, but with our status. (isaaclab.sh collapses
        #    any nonzero status to 1, which is all a gate needs.)
        # 2. Kit leaves stdout block-buffered when it is not a tty, so the whole
        #    report vanishes on redirect unless it is flushed first. os._exit
        #    skips atexit handlers, making the explicit flush mandatory.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
