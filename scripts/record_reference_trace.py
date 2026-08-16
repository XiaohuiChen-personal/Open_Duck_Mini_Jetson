#!/usr/bin/env python3
"""Record an Isaac reference trace as offline ground truth — Task S.2.

Everything from S.4 onward reimplements what Isaac Lab does between the sensors
and the servos. Without recorded ground truth you can only test that
reimplementation against your own reading of the code — which is exactly how the
4x action-scale error in DEPLOY-1 would survive. This produces frozen .npz files
that later tasks assert against, on any machine, with no GPU.

    cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/record_reference_trace.py \
        --checkpoint <repo>/exported_policies/v7_servo_safe_ppo/model_8997.pt \
        --num_envs 1 --steps 1500 --vx 0.2 --vy 0 --wz 0 \
        --out <repo>/docs/jetson-mod/sim2real/reference_trace_v7_fwd.npz --headless

INDEXING CONVENTION — every later test depends on this:
    row i of `obs` is the input that PRODUCED row i of `action`.
    Therefore obs[i][actions_slice] == action[i-1], and obs[0][actions_slice] is
    all zeros.

TWO CAVEATS RECORDED IN THE META, because both would otherwise mislead:

1. `applied_torque` is an APPROXIMATION, not a PhysX measurement. The robot uses
   ImplicitActuatorCfg; ImplicitActuator.compute() calculates
   kp*(target-q) + kd*(0-qd) in Python and clips it to effort_limit_sim. PhysX's
   real internal torque is not exposed. It is also recomputed once per PHYSICS
   step inside the decimation loop, so a per-control-step read samples only the
   last of the four substeps. Both `computed_torque` (pre-clip) and
   `applied_torque` (post-clip) are recorded.

2. The observation's joint blocks are DELAYED (PLANT-7,
   latency.delayed_joint_pos_rel). Each env draws a fixed delay of 0-2 control
   steps at reset, so obs[i][joint_pos_slice] equals
   (joint_pos[i-d] - q_default) for that env's constant d, NOT d=0. The
   recorded `latency_steps_drawn` field reports d where it can be recovered.
"""


import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feetech STS3250, from the manufacturer's own 8-page product specification
# (Edition A/0, 2024-01-16, feetechrc.com): row 5-4 stall 50 kg.cm +/-10%,
# row 5-8 "Rated Torgue" [sic] 16 kg.cm, row 5-9 rated current 1400 mA.
# NOTE Feetech says "Rated", never "continuous", and publishes NO duty-cycle or
# thermal-derating curve -- see docs/jetson-mod/servo_torque_budget.md, which
# derives a ~1.0 N.m sustained target for an enclosed chassis.
# (The previous comment here credited "AGENTS.md and the datasheet"; AGENTS.md
# carries the stall figure only, never 16 kg.cm.)
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
parser.add_argument("--out", required=True, help="output .npz path")
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
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
    import hashlib
    import json
    import subprocess

    import numpy as np

    REPO_ROOT_ = REPO_ROOT
    contract = json.load(open(os.path.join(REPO_ROOT_, "jetson_runtime", "policy_contract.json")))

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # Protocol: the trace must not contain a training noise draw. The Play cfg
    # already sets this; set it again and assert, so it cannot silently regress.
    env_cfg.observations.policy.enable_corruption = False
    assert env_cfg.observations.policy.enable_corruption is False

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base = gym_env.unwrapped

    agent_cfg = _agent_cfg(args_cli.task)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading RSL-RL checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base.device)

    # --- pin the command. Do NOT poke vel_command_b: heading_command is true
    # with rel_heading_envs 1.0, so UniformVelocityCommand._update_command
    # overwrites the yaw channel EVERY step (PLANT-8) and any poked value dies.
    # Mutate the cfg and reset, exactly as evaluate_policies.apply_condition does.
    cond = (args_cli.vx, args_cli.vy, args_cli.wz)
    term = base.command_manager.get_term("base_velocity")
    term.cfg.heading_command = False
    term.cfg.rel_standing_envs = 0.0
    term.cfg.ranges.lin_vel_x = (cond[0], cond[0])
    term.cfg.ranges.lin_vel_y = (cond[1], cond[1])
    term.cfg.ranges.ang_vel_z = (cond[2], cond[2])

    robot = base.scene["robot"]
    names = list(robot.data.joint_names)
    act_order = contract["joint_order"]
    act_idx = [names.index(j) for j in act_order]

    rec = {k: [] for k in ("obs", "action", "joint_pos", "joint_vel", "joint_pos_target",
                           "root_quat_w", "root_ang_vel_b", "root_lin_vel_b",
                           "projected_gravity_b", "command", "gait_phase", "step_idx",
                           "applied_torque", "computed_torque", "terminated", "truncated")}

    obs, _ = env.reset()          # resample so the command pin takes effect
    gp = contract["obs_slices"]["gait_phase"]
    with torch.inference_mode():
        for i in range(args_cli.steps):
            # obs is a TensorDict of observation GROUPS (policy, critic);
            # the actor sees the "policy" group. Indexing the TensorDict
            # directly yields a dict, not the tensor.
            o = obs["policy"][0].detach().cpu().numpy().copy()
            a = policy(obs)
            rec["obs"].append(o)
            rec["action"].append(a[0].detach().cpu().numpy().copy())
            obs, _, dones, extras = env.step(a)
            d = robot.data
            rec["joint_pos"].append(d.joint_pos[0, act_idx].cpu().numpy().copy())
            rec["joint_vel"].append(d.joint_vel[0, act_idx].cpu().numpy().copy())
            rec["joint_pos_target"].append(d.joint_pos_target[0, act_idx].cpu().numpy().copy())
            rec["applied_torque"].append(d.applied_torque[0, act_idx].cpu().numpy().copy())
            rec["computed_torque"].append(d.computed_torque[0, act_idx].cpu().numpy().copy())
            rec["root_quat_w"].append(d.root_quat_w[0].cpu().numpy().copy())
            rec["root_ang_vel_b"].append(d.root_ang_vel_b[0].cpu().numpy().copy())
            rec["root_lin_vel_b"].append(d.root_lin_vel_b[0].cpu().numpy().copy())
            rec["projected_gravity_b"].append(d.projected_gravity_b[0].cpu().numpy().copy())
            rec["command"].append(
                base.command_manager.get_command("base_velocity")[0].cpu().numpy().copy())
            rec["gait_phase"].append(o[gp[0]:gp[1]].copy())
            rec["step_idx"].append(i)
            term_f = bool(dones[0]) if dones.ndim else bool(dones)
            rec["terminated"].append(term_f)
            rec["truncated"].append(False)
            if i and i % 250 == 0:
                print(f"[INFO] step {i}/{args_cli.steps}")

    arr = {k: np.asarray(v) for k, v in rec.items()}

    # An episode boundary destroys the trace: ImitationReward.reset zeroes the
    # gait step index and last_action resets. Fail rather than ship a broken one.
    if arr["terminated"].any():
        print(f"FATAL: episode boundary at step {int(arr['terminated'].argmax())} "
              f"— the trace is not usable", file=sys.stderr)
        return 1

    # Recover the observation delay (PLANT-7) by finding the lag at which the
    # obs joint block matches the recorded absolute positions.
    q_def = np.array(contract["q_default_rad"])
    jp = contract["obs_slices"]["joint_pos"]
    # The total lag has TWO components and both must be accounted for:
    #   +1  recording offset: joint_pos is sampled AFTER env.step(), obs BEFORE
    #   +d  the PLANT-7 latency draw, a fixed 0-2 control steps for this env
    # so the match lands at 1..3. Searching 0..2 (as the first version did)
    # misses it whenever d >= 2, which is a third of all draws.
    lag_found, lag_err = None, {}
    for lag in range(0, 5):
        if lag == 0:
            lhs, rhs = arr["obs"][:, jp[0]:jp[1]], arr["joint_pos"] - q_def
        else:
            lhs, rhs = arr["obs"][lag:, jp[0]:jp[1]], (arr["joint_pos"] - q_def)[:-lag]
        lag_err[lag] = float(np.abs(lhs - rhs).max())
        if lag_err[lag] < 1e-4 and lag_found is None:
            lag_found = lag
    if lag_found is None:
        print(f"FATAL: the observation joint block does not match joint_pos at any "
              f"lag 0..4 (errors {lag_err}). Either the observation is not "
              f"relative, or the latency model changed. Refusing to write a trace "
              f"whose ground truth is not understood.", file=sys.stderr)
        return 1

    md5 = hashlib.md5(open(args_cli.checkpoint, "rb").read()).hexdigest()
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT_,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"

    meta = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "checkpoint_md5": md5,
        "seed": args_cli.seed,
        "command": list(cond),
        "steps": args_cli.steps,
        "num_envs": args_cli.num_envs,
        "joint_order": act_order,
        "obs_slices": contract["obs_slices"],
        "enable_corruption": False,
        "git_sha": sha,
        "plant_mass_kg": float(robot.data.default_mass[0].sum()),
        "obs_joint_block_total_lag": lag_found,
        "obs_joint_block_lag_errors": lag_err,
        "latency_steps_drawn": (lag_found - 1) if lag_found else None,
        "indexing": "row i of obs produced row i of action; obs[i][actions] == action[i-1]",
        "torque_caveat": ("applied_torque/computed_torque are Isaac Lab's Python "
                          "approximation from ImplicitActuator.compute, clipped "
                          "(applied) or not (computed) at effort_limit_sim; PhysX's "
                          "internal torque is not exposed, and the value is the last "
                          "of the 4 decimation substeps."),
        "latency_caveat": ("obs joint blocks are delayed_joint_pos_rel / "
                           "delayed_joint_vel_rel (PLANT-7): a fixed per-env draw of "
                           "0-2 control steps, so they lag the recorded absolute "
                           "joint_pos by latency_steps_drawn."),
    }
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez_compressed(args_cli.out, meta=np.array(meta, dtype=object), **arr)

    print()
    print(f"wrote {args_cli.out}")
    print(f"  obs {arr['obs'].shape}  action {arr['action'].shape}")
    print(f"  command {cond}  plant {meta['plant_mass_kg']:.6f} kg")
    print(f"  obs joint block matches joint_pos at lag {lag_found} "
          f"(= 1 recording offset + {lag_found - 1} PLANT-7 delay), "
          f"max err {lag_err[lag_found]:.2e}")
    print(f"  size {os.path.getsize(args_cli.out)/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
