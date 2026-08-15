#!/usr/bin/env python3
"""Measure the joint torque a trained policy actually commands, per joint.

Task M1 needs this and nothing in the repo produces it. The print-process
decision is a mass decision, and the binding hardware constraint is servo
torque -- but the number everyone quotes for the Feetech STS3250 is its
**peak** stall, 50 kg.cm = 4.903 N.m. Its **continuous** rating is
16 kg.cm = 1.569 N.m, a factor of 3.1 lower. Choosing a process that adds
~430 g is only defensible if sustained torque stays under the continuous
rating with margin.

`evaluate_policies.py` records `energy_proxy_w` (mean of sum |tau*qdot|) but no
per-joint peak or RMS, so the question cannot be answered from existing results.

    cd ~/IsaacLab && ./isaaclab.sh -p \
      ~/Projects/Open_Duck_Mini_Jetson/scripts/measure_joint_torque.py \
      --checkpoint ~/Projects/Open_Duck_Mini_Jetson/exported_policies/v6d_contact_wrench_ppo/model_5998.pt \
      --headless

Reports per joint: peak |tau|, p99, RMS, and the fraction of steps above the
continuous rating. Then scales those to each candidate print process by the
printed-mass delta.

**Two honest limits on the scaling, stated here so they are not forgotten.**
1. It is first-order. Gravity and inertial terms both scale ~linearly with mass
   for a fixed trajectory, which is fine for a go/no-go but not for a final
   margin.
2. `applied_torque` is what the actuator delivered AFTER clipping to
   `effort_limit_sim`, which PLANT-5 records as 8.716 N.m -- 1.78x the datasheet
   stall. A peak at or near that ceiling means the policy is leaning on torque
   the hardware cannot deliver, independent of the mass question, and the
   scaling below understates the problem because the clip hides it.

Exit code is always 0: this is a measurement, not a gate.
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
    n = len(names)
    dev = base.device
    peak = torch.zeros(n, device=dev)
    sq = torch.zeros(n, device=dev)
    over = torch.zeros(n, device=dev)
    samples = 0
    keep = []

    obs, _ = env.reset()
    with torch.inference_mode():
        for i in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            tau = robot.data.applied_torque.abs()
            peak = torch.maximum(peak, tau.max(dim=0).values)
            sq += (tau ** 2).sum(dim=0)
            over += (tau > CONTINUOUS_NM).sum(dim=0).float()
            samples += tau.shape[0]
            if i % 10 == 0:
                keep.append(tau.detach().clone())
            if i and i % 250 == 0:
                print(f"[INFO] step {i}/{args_cli.steps}")

    rms = torch.sqrt(sq / samples)
    frac_over = over / samples
    p99 = torch.quantile(torch.cat(keep, dim=0).float(), 0.99, dim=0)
    plant_kg = float(robot.data.default_mass[0].sum())

    print()
    print(f"task    {args_cli.task}")
    print(f"command (vx={args_cli.vx}, vy={args_cli.vy}, wz={args_cli.wz})")
    print(f"plant   {plant_kg:.6f} kg   "
          f"{args_cli.num_envs} envs x {args_cli.steps} steps "
          f"({args_cli.steps * 0.02:.0f} s)")
    print(f"STS3250 continuous {CONTINUOUS_NM:.3f} N.m (16 kg.cm) | "
          f"peak stall {STALL_NM:.3f} N.m (50 kg.cm)")
    print()
    w = max(len(x) for x in names)
    print(f"{'joint':<{w}}{'peak':>9}{'p99':>9}{'rms':>9}{'% over cont':>13}")
    print("-" * (w + 40))
    for i, nm in enumerate(names):
        print(f"{nm:<{w}}{peak[i]:>9.3f}{p99[i]:>9.3f}{rms[i]:>9.3f}"
              f"{100 * frac_over[i]:>12.2f}%")

    # Legs are what carry the robot; the head/antenna joints are not the
    # constraint and would flatter the summary if averaged in.
    leg = [i for i, nm in enumerate(names)
           if any(k in nm for k in ("hip", "knee", "ankle"))]
    worst = max(leg, key=lambda i: float(peak[i]))
    worst_rms = max(leg, key=lambda i: float(rms[i]))
    print()
    print(f"worst leg joint by peak: {names[worst]}  {float(peak[worst]):.3f} N.m "
          f"({100 * float(peak[worst]) / STALL_NM:.0f}% of stall, "
          f"{100 * float(peak[worst]) / CONTINUOUS_NM:.0f}% of continuous)")
    print(f"worst leg joint by rms : {names[worst_rms]}  "
          f"{float(rms[worst_rms]):.3f} N.m "
          f"({100 * float(rms[worst_rms]) / CONTINUOUS_NM:.0f}% of continuous)")

    base_g = PROCESS_SET_MASS_G[BASELINE_PROCESS]
    print()
    print("Linear mass scaling to each candidate print process")
    print(f"(printed-mass delta against {BASELINE_PROCESS}; first-order, "
          f"go/no-go only):")
    print(f"{'process':<24}{'plant kg':>10}{'peak N.m':>11}{'rms N.m':>10}"
          f"{'rms vs cont':>13}")
    for proc, g in sorted(PROCESS_SET_MASS_G.items(), key=lambda kv: kv[1]):
        new_kg = plant_kg + (g - base_g) / 1000.0
        s = new_kg / plant_kg
        print(f"{proc:<24}{new_kg:>10.3f}{float(peak[worst]) * s:>11.3f}"
              f"{float(rms[worst_rms]) * s:>10.3f}"
              f"{100 * float(rms[worst_rms]) * s / CONTINUOUS_NM:>12.0f}%")

    print()
    print("CAVEAT: applied_torque is post-clip. PLANT-5 records "
          f"effort_limit_sim = 8.716 N.m, 1.78x the {STALL_NM:.3f} N.m datasheet "
          "stall, so any joint sitting at that ceiling is commanding torque the "
          "hardware cannot deliver and this table understates it.")
    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        # os._exit() in the finally block below runs BEFORE the interpreter
        # prints a traceback, so without this an exception here surfaces only
        # as isaaclab.sh's "There was an error running python".
        import traceback
        traceback.print_exc()
    finally:
        # audit_plant_mass.py documents both hazards: simulation_app.close()
        # never returns and pins the exit status to 0, and Kit leaves stdout
        # block-buffered on redirect.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
