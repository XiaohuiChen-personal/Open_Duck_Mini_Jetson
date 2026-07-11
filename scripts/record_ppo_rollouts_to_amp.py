#!/usr/bin/env python3
# Copyright (c) 2025-2026, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Record trained-PPO rollouts as Isaac Lab AMP reference motions.

Runs the deterministic (mean-action) PPO v3 policy in the manager-based
velocity-tracking Play env, one pinned velocity command at a time, and dumps
the ACTUAL simulated body/joint trajectory of one clean environment into the
``MotionLoader`` ``.npz`` schema. These replace the analytically synthesized
gait clips (``scripts/convert_gait_library_to_amp.py``), 23 of whose 51 AMP
observation dimensions are exact constants (root roll/pitch zeroed, constant
root height, degenerate foot offsets) — a real/fake separator the AMP
discriminator can key on regardless of the policy's gait. A rollout recorded
from the physics sim has all 51 dims alive by construction, because the
reference (this file) and the live AMP rollout are then produced by the same
tensors (``body_pos_w`` / ``body_quat_w`` / ``body_*_vel_w``).

Output npz keys (MotionLoader contract, world frame, m/rad/s, float32):
    fps                      int64 scalar (= 50, the policy step rate)
    dof_names                (D,)      str  (robot.data.joint_names order)
    body_names               (B,)      str  (base, foot_assembly, foot_assembly_2)
    dof_positions            (N, D)    float32
    dof_velocities           (N, D)    float32
    body_positions           (N, B, 3) float32   (env-origin subtracted)
    body_rotations           (N, B, 4) float32   WXYZ
    body_linear_velocities   (N, B, 3) float32
    body_angular_velocities  (N, B, 3) float32

Each clip is ONE continuous timeline (no periodicity/tiling): the AMP env
samples arbitrary times, and MultiMotionLoader concatenates clips, so a raw
recorded window is exactly what the discriminator needs. All clips share the
same skeleton and dt=0.02 s (fps=50), matching the AMP env's assertion that
motion dt == policy step dt (duck_amp_env.py:82-85).

Validation (--validate, on by default) runs after recording and needs NO
Isaac Sim: it loads every clip through the vendored MotionLoader /
MultiMotionLoader, reconstructs the 51-dim AMP observation the discriminator
sees (a numpy/torch replica of duck_amp_env.compute_obs), and FAILS if any
dim is dead (std < 1e-6) EXCEPT dim 34 (heading-localized tangent_y), which is
identically zero by construction for every clip and the live rollout alike —
so it is reported but not exploitable, and not a failure (see
STRUCTURAL_ZERO_AMP_DIMS). Use --validate_only to re-validate already-written
clips on a GPU-free machine.

Usage (GPU required to record — do NOT run while a training run owns the GPU):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/record_ppo_rollouts_to_amp.py \
        --checkpoint ~/Projects/Open_Duck_Mini_Jetson/exported_policies/v3_bdx_imitation_ppo/model_2999.pt \
        --headless

GPU-free re-validation of already-written clips (plain python3, torch+numpy):
    python3 scripts/record_ppo_rollouts_to_amp.py --validate_only \
        --output_dir isaac_lab_env/open_duck_mini_v2/amp/motions_ppo/

The script inserts the repo root on sys.path itself, so no PYTHONPATH is
needed. Do not import this module — run it as a script.
"""

# ======================================================================
# Pure-python section — safe without Isaac Sim or a GPU (torch imported
# lazily inside the validator so --validate_only never launches the app).
# ======================================================================

import argparse
import glob
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TASK = "Isaac-Velocity-Rough-OpenDuck-Play-v0"
AMP_DIR = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "amp")
DEFAULT_OUTPUT_DIR = os.path.join(AMP_DIR, "motions_ppo")
EXISTING_MOTIONS_GLOB = os.path.join(AMP_DIR, "motions", "*.npz")
MOTION_LOADER_PATH = os.path.join(AMP_DIR, "motion_loader.py")
DEFAULT_CHECKPOINT = os.path.join(
    REPO_ROOT, "exported_policies", "v3_bdx_imitation_ppo", "model_2999.pt"
)

# Bodies stored per clip. The AMP env remaps by name and consumes only the
# reference body ("base") + the two key bodies (feet); trunk_assembly is not
# needed, so it is dropped (unlike the synthetic clips' 4-body set).
REFERENCE_BODY = "base"
KEY_BODY_NAMES = ["foot_assembly", "foot_assembly_2"]
RECORD_BODY_NAMES = [REFERENCE_BODY] + KEY_BODY_NAMES

AMP_FPS = 50
AMP_DT = 1.0 / AMP_FPS
NUM_AMP_OBS_DIMS = 51
DEAD_DIM_STD = 1e-6

# AMP obs dim 34 is the y-component of the heading-localized forward (tangent)
# vector. Heading localization rotates the forward axis onto +x, so its
# y-component is IDENTICALLY 0 in every AMP frame — reference clips, recorded
# clips and the live rollout alike (verified over random rotations). It is
# therefore structurally constant, not an exploitable real/fake separator, so
# it is excluded from the dead-dim failure (still reported). All OTHER dims are
# expected alive; a constant among them is the synthetic-clip pathology this
# recorder exists to fix.
STRUCTURAL_ZERO_AMP_DIMS = {34}


# ----------------------------------------------------------------------
# Command derivation
# ----------------------------------------------------------------------


def format_command_key(vx: float, vy: float, wz: float) -> str:
    """Format a (vx, vy, wz) triplet the way the gait-library keys read.

    e.g. (0.2, 0.0, 0.0) -> '0.2_0.0_0.0', (0.148, -0.037, -0.074) ->
    '0.148_-0.037_-0.074'. Trailing zeros are trimmed to one decimal so the
    filenames mirror the existing ``duck_gait_*`` names.
    """
    def one(v: float) -> str:
        if v == 0:
            v = 0.0  # avoid a '-0.0' string from negative zero
        s = f"{v:.3f}".rstrip("0")
        return s + "0" if s.endswith(".") else s

    return f"{one(vx)}_{one(vy)}_{one(wz)}"


def derive_commands_from_motions(motions_glob: str) -> list[tuple[str, tuple[float, float, float]]]:
    """Parse the (key, command) list from existing ``duck_gait_*.npz`` names.

    Reusing the source filename substring as the output key makes each
    ``duck_ppo_<key>.npz`` a 1:1 mirror of its ``duck_gait_<key>.npz`` twin.
    """
    prefix, suffix = "duck_gait_", ".npz"
    out: list[tuple[str, tuple[float, float, float]]] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(motions_glob)):
        name = os.path.basename(path)
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        key = name[len(prefix): -len(suffix)]
        parts = key.split("_")
        if len(parts) != 3 or key in seen:
            continue
        seen.add(key)
        out.append((key, (float(parts[0]), float(parts[1]), float(parts[2]))))
    return out


def parse_commands_override(commands_str: str) -> list[tuple[str, tuple[float, float, float]]]:
    """Parse ``vx,vy,wz;vx,vy,wz;...`` into (key, command) pairs."""
    out: list[tuple[str, tuple[float, float, float]]] = []
    for chunk in commands_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(v) for v in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError(f"Command '{chunk}' must be vx,vy,wz")
        cond = (vals[0], vals[1], vals[2])
        out.append((format_command_key(*cond), cond))
    if not out:
        raise ValueError("No commands parsed from --commands")
    return out


# ----------------------------------------------------------------------
# AMP-observation reconstruction (numpy/torch replica of compute_obs) — used
# only by the validator, so it needs no isaaclab.
# ----------------------------------------------------------------------


def _quat_apply(quat, vec):
    """Rotate vec by WXYZ quat: v + 2w(u x v) + 2u x (u x v). Shapes (...,4),(...,3)."""
    import torch

    w = quat[..., 0:1]
    u = quat[..., 1:4]
    t = 2.0 * torch.cross(u, vec, dim=-1)
    return vec + w * t + torch.cross(u, t, dim=-1)


def _quat_conj(quat):
    import torch

    return quat * torch.tensor([1.0, -1.0, -1.0, -1.0], device=quat.device)


def _quat_apply_inverse(quat, vec):
    return _quat_apply(_quat_conj(quat), vec)


def _quat_mul(a, b):
    """Hamilton product of WXYZ quaternions. Shape (..., 4)."""
    import torch

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _yaw_quat(quat):
    """Yaw-only WXYZ quaternion (isaaclab convention). Shape (..., 4)."""
    import torch

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = torch.zeros_like(quat)
    out[..., 0] = torch.cos(0.5 * yaw)
    out[..., 3] = torch.sin(0.5 * yaw)
    return out


def _quat_to_tangent_and_normal(quat):
    import torch

    ref_t = torch.zeros(quat.shape[:-1] + (3,), device=quat.device)
    ref_n = torch.zeros(quat.shape[:-1] + (3,), device=quat.device)
    ref_t[..., 0] = 1.0
    ref_n[..., 2] = 1.0
    return torch.cat([_quat_apply(quat, ref_t), _quat_apply(quat, ref_n)], dim=-1)


def reconstruct_amp_obs(dof_pos, dof_vel, root_pos, root_rot, root_lin, root_ang,
                        key_pos, heading_localize=True):
    """Replica of duck_amp_env.compute_obs (51 dims), heading-localized.

    Mirrors the env exactly so the per-dim std here equals the discriminator's
    input variance. All args are torch tensors; key_pos is (N, K, 3).
    """
    import torch

    key_offsets = key_pos - root_pos.unsqueeze(-2)
    if heading_localize:
        heading = _yaw_quat(root_rot)
        root_rot = _quat_mul(_quat_conj(heading), root_rot)
        root_lin = _quat_apply_inverse(heading, root_lin)
        root_ang = _quat_apply_inverse(heading, root_ang)
        k = key_offsets.shape[1]
        key_offsets = _quat_apply_inverse(
            heading.repeat_interleave(k, dim=0), key_offsets.reshape(-1, 3)
        ).reshape(-1, k, 3)
    return torch.cat(
        [
            dof_pos,
            dof_vel,
            root_pos[:, 2:3],
            _quat_to_tangent_and_normal(root_rot),
            root_lin,
            root_ang,
            key_offsets.reshape(key_offsets.shape[0], -1),
        ],
        dim=-1,
    )


def _amp_dim_label(i: int) -> str:
    if i < 16:
        return f"dof_pos[{i}]"
    if i < 32:
        return f"dof_vel[{i - 16}]"
    if i == 32:
        return "root_z"
    if i < 39:
        return f"orient_tn[{i - 33}]"
    if i < 42:
        return f"root_lin_vel[{i - 39}]"
    if i < 45:
        return f"root_ang_vel[{i - 42}]"
    return f"foot_offset[{i - 45}]"


def _load_motion_loader_module():
    """Import motion_loader.py by file path (its package __init__ pulls in
    isaaclab; the module itself needs only numpy + torch)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("duck_motion_loader", MOTION_LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# Validation (no Isaac Sim required)
# ----------------------------------------------------------------------


def run_validation(output_dir: str) -> int:
    """Load, cross-check and reconstruct AMP obs for every recorded clip.

    Returns a process exit code (0 = all checks pass). Writes the report to
    stdout and to ``<output_dir>/VALIDATION.md``.
    """
    import torch

    ml = _load_motion_loader_module()
    device = torch.device("cpu")

    clip_paths = sorted(glob.glob(os.path.join(output_dir, "duck_ppo_*.npz")))
    lines: list[str] = ["# PPO-rollout AMP clip validation", ""]
    if not clip_paths:
        msg = f"No duck_ppo_*.npz clips found in {output_dir}"
        print(msg)
        lines.append(msg)
        _write_validation_md(output_dir, lines)
        return 1

    # MultiMotionLoader builds one MotionLoader per clip and asserts they share
    # dof_names, body_names and dt — exactly the discriminator's requirement.
    multi = ml.MultiMotionLoader(clip_paths, device)
    loaders = multi.loaders
    failures: list[str] = []

    lines += [f"Clips: {len(clip_paths)} in `{output_dir}`", ""]
    lines += [f"dof_names ({len(multi.dof_names)}): {multi.dof_names}",
              f"body_names ({len(multi.body_names)}): {multi.body_names}",
              f"shared dt: {multi.dt:.6f} s ({1.0 / multi.dt:.1f} fps)", ""]

    # dt check
    for path, loader in zip(clip_paths, loaders):
        if not math.isclose(loader.dt, AMP_DT, rel_tol=1e-4):
            failures.append(f"dt {loader.dt} != {AMP_DT} in {os.path.basename(path)}")

    # skeleton indices
    body_names = list(multi.body_names)
    if REFERENCE_BODY not in body_names or any(n not in body_names for n in KEY_BODY_NAMES):
        failures.append(f"body_names {body_names} missing base/feet")
        _finish_validation(lines, failures, output_dir)
        return 1
    base_idx = body_names.index(REFERENCE_BODY)
    foot_idx = [body_names.index(n) for n in KEY_BODY_NAMES]

    # Per-clip stats + AMP obs reconstruction across ALL frames of ALL clips.
    per_clip = ["## Per-clip summary", "",
                "| clip | frames | root_z min/mean/max (m) | max |dq|/step (rad) | min foot-body z (m) |",
                "|---|---|---|---|---|"]
    obs_chunks = []
    root_z_all = []
    for path, loader in zip(clip_paths, loaders):
        root_pos = loader.body_positions[:, base_idx]
        obs = reconstruct_amp_obs(
            loader.dof_positions,
            loader.dof_velocities,
            root_pos,
            loader.body_rotations[:, base_idx],
            loader.body_linear_velocities[:, base_idx],
            loader.body_angular_velocities[:, base_idx],
            loader.body_positions[:, foot_idx],
            heading_localize=True,
        )
        obs_chunks.append(obs)
        rz = root_pos[:, 2]
        root_z_all.append(rz)
        dq = torch.abs(torch.diff(loader.dof_positions, dim=0))
        max_dq = float(dq.max()) if dq.numel() else 0.0
        foot_z = float(loader.body_positions[:, foot_idx, 2].min())
        per_clip.append(
            f"| {os.path.basename(path)} | {loader.num_frames} "
            f"| {float(rz.min()):.4f}/{float(rz.mean()):.4f}/{float(rz.max()):.4f} "
            f"| {max_dq:.4f} | {foot_z:.4f} |"
        )

    obs_all = torch.cat(obs_chunks, dim=0)
    stds = obs_all.std(dim=0).cpu().numpy()
    dead = [int(i) for i in np.where(stds < DEAD_DIM_STD)[0]]
    structural = [i for i in dead if i in STRUCTURAL_ZERO_AMP_DIMS]
    data_dead = [i for i in dead if i not in STRUCTURAL_ZERO_AMP_DIMS]

    lines += ["## AMP observation liveness (51 dims, all clips concatenated)", "",
              f"Frames: {obs_all.shape[0]}   Dims: {obs_all.shape[1]}",
              f"Data-dead dims (std < {DEAD_DIM_STD:g}, EXPLOITABLE): "
              + ("NONE" if not data_dead else ", ".join(
                  f"{i} ({_amp_dim_label(i)})" for i in data_dead)),
              "Structurally-zero dims (expected, excluded from failure): "
              + (", ".join(f"{i} ({_amp_dim_label(i)})" for i in structural)
                 or "none observed"), ""]
    order = np.argsort(stds)
    lines.append("Five lowest-variance dims:")
    for i in order[:5]:
        tag = " [structural]" if int(i) in STRUCTURAL_ZERO_AMP_DIMS else ""
        lines.append(f"  dim {int(i):2d} {_amp_dim_label(int(i)):16s} std={stds[i]:.3e}{tag}")
    lines.append("")
    if data_dead:
        failures.append(f"{len(data_dead)} data-dead AMP dim(s): "
                        + ", ".join(f"{i}:{_amp_dim_label(i)}" for i in data_dead))

    root_z = torch.cat(root_z_all)
    lines += ["## Root height (base z, all frames)",
              f"min {float(root_z.min()):.4f}  mean {float(root_z.mean()):.4f}  "
              f"max {float(root_z.max()):.4f}  std {float(root_z.std()):.4f} m",
              "(foot z above is the foot-body ORIGIN, not the sole, so ground "
              "penetration is not directly derivable here.)", ""]
    lines += per_clip + [""]

    _finish_validation(lines, failures, output_dir)
    return 0 if not failures else 1


def _finish_validation(lines: list[str], failures: list[str], output_dir: str) -> None:
    verdict = "PASS" if not failures else "FAIL"
    header = [f"Result: {verdict}", ""]
    if failures:
        header.append("Failures:")
        header += [f"  - {f}" for f in failures]
        header.append("")
    lines = lines[:2] + header + lines[2:]
    report = "\n".join(lines)
    print(report)
    _write_validation_md(output_dir, lines)


def _write_validation_md(output_dir: str, lines: list[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "VALIDATION.md"), "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record PPO rollouts as AMP MotionLoader .npz reference clips.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Manager-based velocity Play task to roll out.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="RSL-RL .pt checkpoint (PPO v3 model_2999.pt).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write duck_ppo_*.npz clips into.")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Parallel envs per command (a clean one is selected).")
    parser.add_argument("--settle_s", type=float, default=2.0,
                        help="Seconds discarded after reset (gait transient).")
    parser.add_argument("--record_s", type=float, default=10.0,
                        help="Seconds recorded per clip (500 frames at 50 Hz).")
    parser.add_argument("--commands", type=str, default=None,
                        help="Override commands 'vx,vy,wz;...'; default derives "
                             "the same triplets as the existing duck_gait_* clips.")
    parser.add_argument("--command_name", type=str, default="base_velocity",
                        help="Velocity command term name in the command manager.")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument("--skip_validate", action="store_true",
                        help="Do not run the post-recording validation.")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate existing clips in --output_dir and exit "
                             "(no Isaac Sim / GPU).")
    return parser


# --validate_only short-circuits before any Isaac import (like
# evaluate_policies.py's --self-test / --report-only paths).
if __name__ == "__main__" and "--validate_only" in sys.argv:
    _args, _unknown = build_arg_parser().parse_known_args()
    if _unknown:
        print(f"[WARN] Ignoring unrecognized arguments: {_unknown}")
    sys.exit(run_validation(os.path.abspath(_args.output_dir)))


# ======================================================================
# Isaac Sim section — everything below requires the simulator. AppLauncher
# must run before importing torch / isaaclab (pattern from play_policy.py
# and evaluate_policies.py).
# ======================================================================

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import isaac_lab_env  # noqa: F401, E402 — triggers gym.register()

from isaaclab.app import AppLauncher  # noqa: E402

parser = build_arg_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown_args = parser.parse_known_args()
if _unknown_args:
    print(f"[WARN] Ignoring unrecognized arguments: {_unknown_args}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

RSL_RL_VERSION = metadata.version("rsl-rl-lib")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402


class RslRlPolicy:
    """Deterministic RSL-RL inference adapter (trimmed from evaluate_policies.py).

    handle_deprecated_rsl_rl_cfg is REQUIRED with rsl-rl-lib >= 5.0 (loading
    the runner cfg without it raises KeyError 'class_name'); get_inference_policy
    yields mean actions with obs normalization applied.
    """

    def __init__(self, task_id: str, checkpoint: str, gym_env):
        self.base_env = gym_env.unwrapped
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)
        self.env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        self.runner = OnPolicyRunner(
            self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        resume_path = retrieve_file_path(checkpoint)
        print(f"[INFO] Loading RSL-RL checkpoint: {resume_path}")
        self.runner.load(resume_path)
        self.policy = self.runner.get_inference_policy(device=self.base_env.device)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def act(self, obs) -> torch.Tensor:
        return self.policy(obs)

    def step(self, actions: torch.Tensor):
        obs, _, dones, extras = self.env.step(actions)
        if hasattr(self.base_env, "termination_manager"):
            terminated = self.base_env.termination_manager.terminated
            truncated = self.base_env.termination_manager.time_outs
        else:
            truncated = extras.get("time_outs", torch.zeros_like(dones)).bool()
            terminated = dones.bool() & ~truncated
        if hasattr(self.policy, "reset"):
            self.policy.reset(dones)
        return (
            obs,
            terminated.view(-1).cpu().numpy().astype(bool),
            truncated.view(-1).cpu().numpy().astype(bool),
        )


def build_record_env_cfg(task_id: str):
    """Parse the task cfg with the recording overrides (no corruption/pushes,
    episode long enough that a clean env never truncates mid-window)."""
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = float(args_cli.settle_s + args_cli.record_s + 5.0)
    env_cfg.seed = args_cli.seed
    try:
        env_cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass
    if hasattr(env_cfg, "events"):
        for event_name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
    return env_cfg


def apply_condition(base_env, command_name: str, cond: tuple[float, float, float]):
    """Pin a fixed velocity command via degenerate command ranges.

    The command term reads cfg.ranges / heading_command / rel_standing_envs on
    every resample, and the subsequent reset resamples all envs — so every env
    holds exactly ``cond`` for the whole window.
    """
    if not hasattr(base_env, "command_manager"):
        raise NotImplementedError(
            f"Recording expects a command manager with a '{command_name}' term."
        )
    term = base_env.command_manager.get_term(command_name)
    term.cfg.heading_command = False  # don't let heading control overwrite wz
    term.cfg.rel_standing_envs = 0.0  # no standing envs while recording
    term.cfg.ranges.lin_vel_x = (cond[0], cond[0])
    term.cfg.ranges.lin_vel_y = (cond[1], cond[1])
    term.cfg.ranges.ang_vel_z = (cond[2], cond[2])


def resolve_body_ids(robot) -> list[int]:
    body_names = list(robot.data.body_names)
    missing = [n for n in RECORD_BODY_NAMES if n not in body_names]
    if missing:
        raise RuntimeError(f"Robot missing expected bodies {missing} (has {body_names})")
    return [body_names.index(n) for n in RECORD_BODY_NAMES]


def record_command(adapter, cond, body_ids, dof_names, fps, settle_steps, record_steps):
    """Roll out one pinned command and return npz arrays for one clean env.

    Steps ``settle_steps`` (discarded) then ``record_steps`` (stored) for all
    envs, tracking which env ever terminated/truncated; the returned clip is
    the first env that stayed alive across the whole window.
    """
    base_env = adapter.base_env
    robot = base_env.scene["robot"]
    n = base_env.num_envs
    env_origins = base_env.scene.env_origins  # (n, 3)
    num_bodies = len(body_ids)

    apply_condition(base_env, args_cli.command_name, cond)
    obs = adapter.reset()

    bad_ever = np.zeros(n, dtype=bool)
    for _ in range(settle_steps):
        with torch.no_grad():
            obs, terminated, truncated = adapter.step(adapter.act(obs))
        bad_ever |= terminated | truncated

    dof_pos = np.zeros((record_steps, n, len(dof_names)), dtype=np.float32)
    dof_vel = np.zeros_like(dof_pos)
    body_pos = np.zeros((record_steps, n, num_bodies, 3), dtype=np.float32)
    body_rot = np.zeros((record_steps, n, num_bodies, 4), dtype=np.float32)
    body_lin = np.zeros((record_steps, n, num_bodies, 3), dtype=np.float32)
    body_ang = np.zeros((record_steps, n, num_bodies, 3), dtype=np.float32)

    for t in range(record_steps):
        with torch.no_grad():
            obs, terminated, truncated = adapter.step(adapter.act(obs))
            data = robot.data
            dof_pos[t] = data.joint_pos.cpu().numpy()
            dof_vel[t] = data.joint_vel.cpu().numpy()
            # World frame; subtract the per-env origin so clips are env-local.
            body_pos[t] = (data.body_pos_w[:, body_ids]
                           - env_origins[:, None, :]).cpu().numpy()
            body_rot[t] = data.body_quat_w[:, body_ids].cpu().numpy()  # WXYZ
            body_lin[t] = data.body_lin_vel_w[:, body_ids].cpu().numpy()
            body_ang[t] = data.body_ang_vel_w[:, body_ids].cpu().numpy()
        bad_ever |= terminated | truncated

    clean = np.where(~bad_ever)[0]
    if clean.size == 0:
        raise RuntimeError(
            f"No env survived settle+record for command {cond}; "
            f"raise --num_envs or shorten --record_s."
        )
    e = int(clean[0])

    arrays = {
        "fps": np.int64(fps),
        "dof_names": np.array(dof_names),
        "body_names": np.array(RECORD_BODY_NAMES),
        "dof_positions": dof_pos[:, e],
        "dof_velocities": dof_vel[:, e],
        "body_positions": body_pos[:, e],
        "body_rotations": body_rot[:, e],
        "body_linear_velocities": body_lin[:, e],
        "body_angular_velocities": body_ang[:, e],
    }
    summary = {"env": e, "n_bad": int(np.count_nonzero(bad_ever)),
               "root_z_mean": float(body_pos[:, e, 0, 2].mean())}
    return arrays, summary


def main() -> None:
    if args_cli.commands:
        commands = parse_commands_override(args_cli.commands)
    else:
        commands = derive_commands_from_motions(EXISTING_MOTIONS_GLOB)
    if not commands:
        raise RuntimeError(
            f"No commands derived from {EXISTING_MOTIONS_GLOB}; pass --commands."
        )

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(args_cli.seed)

    env_cfg = build_record_env_cfg(args_cli.task)
    print(f"[INFO] Creating env '{args_cli.task}' ({args_cli.num_envs} envs)")
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        adapter = RslRlPolicy(args_cli.task, args_cli.checkpoint, gym_env)
        base_env = adapter.base_env
        robot = base_env.scene["robot"]
        body_ids = resolve_body_ids(robot)
        dof_names = list(robot.data.joint_names)

        step_dt = float(base_env.step_dt)
        fps = int(round(1.0 / step_dt))
        if not math.isclose(step_dt, AMP_DT, rel_tol=1e-4):
            raise RuntimeError(
                f"Policy step dt {step_dt:.5f}s (={fps} fps) != AMP {AMP_DT}s "
                f"(50 fps). The AMP env asserts motion dt == step dt; recording "
                f"at another rate would make the clips unloadable."
            )
        settle_steps = int(round(args_cli.settle_s * fps))
        record_steps = int(round(args_cli.record_s * fps))
        print(f"[INFO] {len(commands)} commands x ({settle_steps} settle + "
              f"{record_steps} record) steps @ {fps} fps; bodies {RECORD_BODY_NAMES}")

        for key, cond in commands:
            arrays, summary = record_command(
                adapter, cond, body_ids, dof_names, fps, settle_steps, record_steps
            )
            out_path = os.path.join(output_dir, f"duck_ppo_{key}.npz")
            np.savez(out_path, **arrays)
            print(f"[{key}] {record_steps} frames @ {fps} fps from env "
                  f"{summary['env']} ({summary['n_bad']}/{args_cli.num_envs} envs "
                  f"terminated) root_z~{summary['root_z_mean']:.3f} m "
                  f"-> {os.path.basename(out_path)}")
    finally:
        gym_env.close()

    if not args_cli.skip_validate:
        print("\n[INFO] ===== Validating recorded clips =====")
        code = run_validation(output_dir)
        if code != 0:
            print("[WARN] Validation FAILED — see VALIDATION.md")


if __name__ == "__main__":
    main()
    simulation_app.close()
