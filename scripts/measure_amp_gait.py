"""Measure gait metrics of a trained skrl AMP policy (Gate G2/G3 check).

Rolls out a checkpoint headless and reports forward/lateral velocity,
stance-duty proxy, leg ROM symmetry, and survival rate. Used at every
AMP gate; results are journaled in docs/jetson-mod/experiment_journal.md.

Usage:
    cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/measure_amp_gait.py \
        --checkpoint <run>/checkpoints/best_agent.pt --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure gait metrics of an AMP policy rollout (Gate G2).")
parser.add_argument("--task", default="Isaac-OpenDuck-AMP-PureStyle-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out", default="/tmp/g2_results.txt")
parser.add_argument("--ref_fwd", type=float, default=0.148, help="clip forward velocity for comparison")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import sys
sys.path.insert(0, "/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson")
import gymnasium as gym
import numpy as np
import torch
import isaac_lab_env  # noqa: F401 — registers duck tasks

from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

TASK = args.task
CKPT = args.checkpoint

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
agent_cfg = load_cfg_from_registry(TASK, "skrl_amp_cfg_entry_point")
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
env = SkrlVecEnvWrapper(env, ml_framework="torch")
runner = Runner(env, agent_cfg)
runner.agent.load(CKPT)
runner.agent.set_running_mode("eval")

base_env = env.unwrapped
robot = base_env.scene["robot"]
joint_names = list(robot.data.joint_names)
li = [joint_names.index(n) for n in ["left_hip_pitch", "left_knee", "left_ankle"]]
ri = [joint_names.index(n) for n in ["right_hip_pitch", "right_knee", "right_ankle"]]
ref_body = base_env.ref_body_index
feet = base_env.key_body_indexes

T = 1000
N = args.num_envs
contacts = np.zeros((T, N, 2), bool)
vel_xy = np.zeros((T, N, 2), np.float32)
legL = np.zeros((T, N, 3), np.float32)
legR = np.zeros((T, N, 3), np.float32)
alive = np.ones((T, N), bool)
dead = torch.zeros(N, dtype=torch.bool, device=base_env.device)

obs, _ = env.reset()
with torch.no_grad():
    for t in range(T):
        out = runner.agent.act(obs, timestep=0, timesteps=1)
        actions = out[-1].get("mean_actions", out[0])
        obs, _, terminated, truncated, _ = env.step(actions)
        dead |= terminated.view(-1).bool()
        alive[t] = (~dead).cpu().numpy()
        # contact via env's own buffers: feet positions vs force not exposed;
        # use net contact forces from articulation external? -> use body z-vel
        # Simpler: contact from foot height proxy is noisy; use the env's amp
        # buffer? Use root velocity + joints for the core metrics:
        vel = robot.data.body_lin_vel_w[:, ref_body, :2]
        # rotate into heading frame
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat
        q = robot.data.body_quat_w[:, ref_body]
        v3 = torch.zeros((N, 3), device=vel.device)
        v3[:, :2] = vel
        vel_h = quat_apply_inverse(yaw_quat(q), v3)
        vel_xy[t] = vel_h[:, :2].cpu().numpy()
        legL[t] = robot.data.joint_pos[:, li].cpu().numpy()
        legR[t] = robot.data.joint_pos[:, ri].cpu().numpy()
        # foot contact: feet body z position < threshold as proxy
        fz = robot.data.body_pos_w[:, feet, 2].cpu().numpy()
        contacts[t] = fz < 0.035

m = alive
fwd = float(vel_xy[..., 0][m].mean())
lat = float(vel_xy[..., 1][m].mean())
dutyL = float(contacts[..., 0][m].mean() * 100)
dutyR = float(contacts[..., 1][m].mean() * 100)
surv = float(alive.mean() * 100)
ever_dead = float(dead.float().mean().item() * 100)
romL = np.degrees(np.percentile(legL[:, :, 0][m], 95) - np.percentile(legL[:, :, 0][m], 5))
romR = np.degrees(np.percentile(legR[:, :, 0][m], 95) - np.percentile(legR[:, :, 0][m], 5))
ref_fwd = args.ref_fwd

with open(args.out, "w") as f:
    f.write(f"forward velocity (heading frame): {fwd:+.3f} m/s (clip: {ref_fwd})\n")
    f.write(f"lateral velocity: {lat:+.3f} m/s (clip: -0.037)\n")
    f.write(f"stance duty L/R (z<3.5cm proxy): {dutyL:.1f}% / {dutyR:.1f}% (asym {abs(dutyL-dutyR):.1f}pp)\n")
    f.write(f"hip-pitch ROM L/R: {romL:.1f} / {romR:.1f} deg (ratio {romL/max(romR,1e-6):.2f})\n")
    f.write(f"alive fraction over 20s x 64 envs: {surv:.1f}%; envs that ever fell: {ever_dead:.1f}%\n")

env.close()
app.close()
