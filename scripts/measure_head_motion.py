#!/usr/bin/env python3
"""Measure how much each joint is COMMANDED to move versus how much it actually
moves, and the torque that costs.

Written 2026-08-15 to answer a specific question about SERVO-1: is the neck
being pinned to a single angle, or is the policy free to move the head while
walking? A rigid head is both a thermal problem (the servo fights body motion)
and a behavioural one (it looks unnatural).

    cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/measure_head_motion.py \
        --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 \
        --checkpoint <repo>/exported_policies/v6d_contact_wrench_ppo/model_5998.pt \
        --num_envs 4 --steps 1500 --vx 0.2 --headless

Reports per joint, in degrees: the default pose, the mean/sd/peak-to-peak of the
COMMANDED angle, the sd/peak-to-peak of the ACTUAL angle, the RMS tracking error
between them, and the RMS torque. Torque = stiffness * error, so the error
column is what actually explains the torque column.

Legs are included as a reference: they are joints nobody disputes should move.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feetech STS3250, from AGENTS.md and the datasheet.
STALL_NM = 4.903          # 50 kg.cm, peak
CONTINUOUS_NM = 1.569     # 16 kg.cm, continuous

# Measured whole-set printed mass by process (scripts/measure_print_mass.py,
# 2026-08-12, 52 pieces / 1571.94 cm3, TPU sole exempt). The two *-shelled rows
# are the voxel-erosion estimate of a 3 mm wall, not a measured shell.
PROCESS_SET_MASS_G = {
    "fdm-abs": 977.15,
    "fdm-asa": 1004.29,
    "fdm-pla": 1158.18,
    "fdm-petg": 1185.35,
    "sls-pa12-shelled(est)": 1380.0,
    "mjf-pa12-shelled(est)": 1435.0,
    "sls-pa12": 1536.58,
    "mjf-pa12": 1597.57,
}
# The plant as modelled today embodies roughly the FDM-PLA printed mass: a
# bottom-up rebuild from BOM point masses plus measured slicer mass reproduced
# the shipped MJCF to 2630.20 g against 2657.07 declared.
BASELINE_PROCESS = "fdm-pla"

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task",
                    default="Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1500,
                    help="control steps at 20 ms each (1500 = 30 s)")
parser.add_argument("--vx", type=float, default=0.2)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--out", default=None,
                    help="where to write the .npz; defaults to REPO/head_kinematics.npz.\n                         Give distinct paths when measuring several conditions,\n                         or each run overwrites the last.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import (  # noqa: E402
    load_cfg_from_registry,
    parse_env_cfg,
)
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from importlib import metadata  # noqa: E402

RSL_RL_VERSION = metadata.version("rsl-rl-lib")

sys.path.insert(0, REPO_ROOT)
import isaac_lab_env.open_duck_mini_v2  # noqa: F401,E402


def _agent_cfg(task: str):
    """Same shim evaluate_policies.py uses. It is REQUIRED, not optional:
    rsl-rl-lib 5.x renamed config keys and OnPolicyRunner dies with
    KeyError: 'class_name' without it. Do not wrap this in a try/except --
    swallowing the import turns a clear failure into that same KeyError one
    frame later."""
    cfg = load_cfg_from_registry(task.split(":")[-1], "rsl_rl_cfg_entry_point")
    return handle_deprecated_rsl_rl_cfg(cfg, RSL_RL_VERSION)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0",
                            num_envs=args_cli.num_envs)
    # Pin the command so this measures one condition, not a random walk.
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x = (args_cli.vx, args_cli.vx)
    r.lin_vel_y = (args_cli.vy, args_cli.vy)
    r.ang_vel_z = (args_cli.wz, args_cli.wz)

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base = gym_env.unwrapped

    agent_cfg = _agent_cfg(args_cli.task)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    print(f"[INFO] Loading RSL-RL checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base.device)

    robot = base.scene["robot"]
    names = list(robot.data.joint_names)
    dev = base.device
    HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    LEG  = ["left_hip_pitch", "left_knee", "left_ankle"]
    idx = {j: names.index(j) for j in HEAD + LEG if j in names}

    q_hist, tgt_hist, tau_hist = [], [], []
    obs, _ = env.reset()
    with torch.inference_mode():
        for i in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            q_hist.append(robot.data.joint_pos[0].detach().clone())
            tgt_hist.append(robot.data.joint_pos_target[0].detach().clone())
            tau_hist.append(robot.data.applied_torque[0].detach().clone())
            if i and i % 500 == 0:
                print(f"[INFO] step {i}/{args_cli.steps}")

    import numpy as np
    q  = torch.stack(q_hist).cpu().numpy()
    tg = torch.stack(tgt_hist).cpu().numpy()
    tq = torch.stack(tau_hist).cpu().numpy()
    dq = robot.data.default_joint_pos[0].cpu().numpy()

    # drop the first 2 s of settle
    s = 100
    q, tg, tq = q[s:], tg[s:], tq[s:]
    print()
    print(f"env 0, {q.shape[0]} control steps ({q.shape[0]*0.02:.1f} s) after a 2 s settle")
    print(f"command (vx={args_cli.vx}, vy={args_cli.vy}, wz={args_cli.wz})")
    print()
    # SERVO-1 acceptance needs the fraction of steps spent ON a mechanical
    # stop, which is the difference between "held at an angle" and "jammed".
    # Ranges come from the MJCF, the same source PhysX was built from.
    import xml.etree.ElementTree as ET
    _r = ET.parse(os.path.join(
        REPO_ROOT, "mini_bdx/robots/open_duck_mini_v2/robot_motors.xml")).getroot()
    lim = {jt.get("name"): [float(v) for v in jt.get("range").split()]
           for jt in _r.iter("joint") if jt.get("name") and jt.get("range")}
    TOL = np.radians(0.5)

    print("DOES THE HEAD MOVE?  angles in degrees, relative to the trunk")
    print(f"{'joint':<16}{'default':>9}{'cmd mean':>10}{'cmd sd':>8}{'cmd p2p':>9}"
          f"{'act sd':>8}{'act p2p':>9}{'err rms':>9}{'tau rms':>9}{'% on stop':>11}")
    for j in HEAD + LEG:
        k = idx[j]
        deg = np.degrees
        cmd, act = tg[:, k], q[:, k]
        err = cmd - act
        if j in lim:
            lo, hi = lim[j]
            on_stop = ((act <= lo + TOL) | (act >= hi - TOL)).mean() * 100.0
        else:
            on_stop = float("nan")
        print(f"{j:<16}{deg(dq[k]):9.2f}{deg(cmd.mean()):10.2f}{deg(cmd.std()):8.3f}"
              f"{deg(cmd.max()-cmd.min()):9.3f}{deg(act.std()):8.3f}"
              f"{deg(act.max()-act.min()):9.3f}{deg(np.sqrt((err**2).mean())):9.3f}"
              f"{np.sqrt((tq[:, k]**2).mean()):9.3f}{on_stop:10.1f}%")
    print()
    print("Reading: 'cmd sd/p2p' = how much the POLICY asks the joint to move.")
    print("         'act sd/p2p' = how much the joint ACTUALLY moves.")
    print("         'err rms'    = lag between the two; torque = 45.53 * err.")
    out_path = args_cli.out or os.path.join(REPO_ROOT, "head_kinematics.npz")
    np.savez(out_path,
             q=q, tgt=tg, tau=tq, names=np.array(names), default=dq)
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
