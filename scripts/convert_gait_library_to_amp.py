#!/usr/bin/env python3
# Copyright (c) 2025-2026, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline converter: Open Duck polynomial gait library -> Isaac Lab AMP motions.

Converts entries of the Open Duck Playground polynomial gait library
(``isaac_lab_env/open_duck_mini_v2/data/polynomial_coefficients.pkl``, 240
parametric walking gaits as degree-15 polynomials over a 0.54 s period) into
the ``.npz`` motion format consumed by Isaac Lab's AMP ``MotionLoader``
(``isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py``).

Output npz keys (MotionLoader contract):
    fps                      int64 scalar
    dof_names                (D,)     str
    body_names               (B,)     str
    dof_positions            (N, D)   float32
    dof_velocities           (N, D)   float32
    body_positions           (N, B, 3) float32
    body_rotations           (N, B, 4) float32, WXYZ quaternions
    body_linear_velocities   (N, B, 3) float32
    body_angular_velocities  (N, B, 3) float32

Design intent / verified facts about the gait library:

- Each motion is keyed ``"dx_dy_dtheta"`` (m/s, m/s, rad/s). Per entry,
  ``coefficients`` holds dims 0..39 as 16 floats each, CONSTANT-TERM-FIRST
  (``np.polynomial.polynomial.polyval`` convention), fit over NORMALIZED
  phase t in [0, 1) — NOT seconds (see the phase-bug fix note in
  ``imitation_reward.py``).
- Dims 0-15: joint positions in Playground order, which is IDENTICAL to the
  MJCF joint file order of ``robot_motors.xml`` — so they map 1:1 onto
  MuJoCo ``qpos[7:23]`` with no reordering. The AMP env remaps BY NAME, so
  ``dof_names`` are simply the 16 MJCF joint names.
- Dims 16-31 (joint velocities) are noisy independent fits — NOT used.
  Velocities are instead the analytic polynomial derivative divided by the
  period (chain rule: d/dt = d/dphase / T).
- Dims 32-33: L/R foot contacts (not needed by MotionLoader).
- Dims 34-36: base linear velocity (m/s). Stored world-frame, but the gait
  was generated heading-aligned, so when tiling cycles the velocity is
  rotated by the accumulated yaw before integration.
- Dims 37-39 are BROKEN (wy identically 0, wz ~30x too small) — ignored.
  The yaw rate is reconstructed from the key's dtheta instead.

Root pose reconstruction:

- yaw(t) = dtheta * t_sec, roll = pitch = 0 (placo walk_trunk_pitch=0).
- World xy from trapezoidal integration of the yaw-rotated heading-frame
  (vx, vy); xy and yaw ACCUMULATE across tiled cycles (no teleports).
- Root z is constant, chosen via MuJoCo forward kinematics so the lower
  stance-foot sole sits at z ~ 0: per frame the robot's lowest point (exact
  mesh vertices, not conservative AABBs) is the stance sole; root z is the
  negated MEAN of that height over one cycle, so the stance sole rides at
  z ~ 0 with only mm-level deviation (cosmetic for AMP discriminator
  features; the reference is never physically simulated).

Joint positions are clamped to soft limits = the MJCF ranges scaled by 0.9
about their midpoints, matching the training env's
``soft_joint_pos_limit_factor=0.9`` (the library's knee swing peaks at
1.78/1.98 rad, beyond the +/-1.5708 rad hard limit).

Sampling grid: one cycle at the native grid t_k = k/27, k = 0..26, i.e.
fps = 50 — exactly the policy control rate, which matters downstream.

This is a pure-offline tool: it needs only numpy + mujoco (no Isaac Sim,
no CUDA). Safe to run while the GPU is busy training.

Usage:
    python3 scripts/convert_gait_library_to_amp.py --preset forward
    python3 scripts/convert_gait_library_to_amp.py --preset curated24
    python3 scripts/convert_gait_library_to_amp.py --keys 0.148_-0.037_-0.074
    python3 scripts/convert_gait_library_to_amp.py --all --cycles 10
"""

from __future__ import annotations

import argparse
import os
import pickle

import mujoco
import numpy as np

# --- Paths ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(
    REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "data",
    "polynomial_coefficients.pkl",
)
MJCF_PATH = os.path.join(
    REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml"
)
DEFAULT_OUT_DIR = os.path.join(
    REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "amp", "motions"
)

# Near-forward gait (smallest |dy| and |dtheta| at dx=0.148 in the library).
FORWARD_KEY = "0.148_-0.037_-0.074"

# Bodies exported for the AMP discriminator features. MuJoCo body names ==
# USD/Isaac Lab body names (same MJCF source); the AMP env remaps by name.
BODY_NAMES = ["base", "trunk_assembly", "foot_assembly", "foot_assembly_2"]

# Matches the training env (robot_cfg.py / Isaac Lab default articulation
# config): soft limits scaled about the range midpoint.
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

# Seam-continuity gate: max frame-to-frame joint jump at a cycle seam (rad).
SEAM_DISCONTINUITY_LIMIT = 0.15

NUM_JOINTS = 16
DIM_FOOT_CONTACTS = (32, 34)  # unused here, documented for completeness
DIM_BASE_LIN_VEL = (34, 37)


# ---------------------------------------------------------------------------
# Quaternion helpers (WXYZ convention, matching MuJoCo xquat and MotionLoader)
# ---------------------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two batches of WXYZ quaternions. Shape (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of WXYZ quaternions. Shape (..., 4)."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    """Pure-yaw WXYZ quaternions from yaw angles. Shape (..., ) -> (..., 4)."""
    half = 0.5 * np.asarray(yaw)
    q = np.zeros(np.shape(half) + (4,))
    q[..., 0] = np.cos(half)
    q[..., 3] = np.sin(half)
    return q


def align_quat_signs(quats: np.ndarray) -> np.ndarray:
    """Flip quaternion signs along the time axis so consecutive frames stay in
    the same hemisphere (q and -q are the same rotation; finite differences
    need temporal sign continuity). Shape (N, 4) -> (N, 4)."""
    out = quats.copy()
    for k in range(1, out.shape[0]):
        if np.dot(out[k], out[k - 1]) < 0.0:
            out[k] = -out[k]
    return out


def angular_velocity_from_quats(quats: np.ndarray, dt: float) -> np.ndarray:
    """Body angular velocities (world frame) from a WXYZ quaternion trajectory.

    Standard formula: omega = 2 * (dq/dt) * conj(q), vector part. dq/dt is a
    central finite difference (one-sided at the endpoints).

    Args:
        quats: Quaternion trajectory, shape (N, 4), WXYZ.
        dt: Frame time step in seconds.

    Returns:
        Angular velocities, shape (N, 3).
    """
    q = align_quat_signs(quats)
    dq = np.empty_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    dq[0] = (q[1] - q[0]) / dt
    dq[-1] = (q[-1] - q[-2]) / dt
    omega_quat = 2.0 * quat_mul(dq, quat_conj(q))
    return omega_quat[..., 1:4]


def central_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference along axis 0 (one-sided at the endpoints)."""
    out = np.empty_like(values)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


# ---------------------------------------------------------------------------
# Gait library access
# ---------------------------------------------------------------------------

def parse_key(key: str) -> tuple[float, float, float]:
    """Parse a library key "dx_dy_dtheta" into (dx m/s, dy m/s, dtheta rad/s)."""
    parts = key.split("_")
    return float(parts[0]), float(parts[1]), float(parts[2])


def load_library(path: str) -> dict:
    """Load the polynomial gait library pickle (240 motions)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def select_curated24(keys: list[str]) -> list[str]:
    """Programmatically pick a <=24-motion spread covering the command box.

    Selection (then dedupe, preserving order):
    - For each of the 6 unique dx values: the key with dy/dtheta nearest zero
      (the forward/backward speed sweep).
    - The 4 extreme (dx, dy) corners, dtheta nearest zero.
    - Pure-turn extremes: dtheta min/max with dx/dy nearest zero.
    - Pure-lateral extremes: dy min/max with dx/dtheta nearest zero.
    - The 4 extreme (dx, dtheta) corners, dy nearest zero.
    - The 4 extreme (dy, dtheta) corners, dx nearest zero.
    """
    vels = np.array([parse_key(k) for k in keys])  # (M, 3)

    def pick(mask: np.ndarray, cost: np.ndarray) -> str:
        idx = np.flatnonzero(mask)
        return keys[idx[np.argmin(cost[idx])]]

    dx, dy, dth = vels[:, 0], vels[:, 1], vels[:, 2]
    dx_vals = np.unique(dx)
    dy_ext = (dy.min(), dy.max())
    dth_ext = (dth.min(), dth.max())
    dx_ext = (dx.min(), dx.max())

    selected: list[str] = []
    # Speed sweep: dy/dtheta nearest zero at each dx.
    for v in dx_vals:
        selected.append(pick(np.isclose(dx, v), dy**2 + dth**2))
    # (dx, dy) corners, |dtheta| minimal.
    for vx in dx_ext:
        for vy in dy_ext:
            selected.append(
                pick(np.isclose(dx, vx) & np.isclose(dy, vy), np.abs(dth))
            )
    # Pure-turn extremes.
    for vt in dth_ext:
        selected.append(pick(np.isclose(dth, vt), dx**2 + dy**2))
    # Pure-lateral extremes.
    for vy in dy_ext:
        selected.append(pick(np.isclose(dy, vy), dx**2 + dth**2))
    # (dx, dtheta) corners, |dy| minimal.
    for vx in dx_ext:
        for vt in dth_ext:
            selected.append(
                pick(np.isclose(dx, vx) & np.isclose(dth, vt), np.abs(dy))
            )
    # (dy, dtheta) corners, |dx| minimal.
    for vy in dy_ext:
        for vt in dth_ext:
            selected.append(
                pick(np.isclose(dy, vy) & np.isclose(dth, vt), np.abs(dx))
            )

    deduped = list(dict.fromkeys(selected))
    assert len(deduped) <= 24, f"curated24 produced {len(deduped)} keys"
    return deduped


# ---------------------------------------------------------------------------
# Conversion core
# ---------------------------------------------------------------------------

def soft_joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Soft joint limits = MJCF ranges scaled by 0.9 about their midpoints.

    Matches Isaac Lab's ``soft_joint_pos_limit_factor``:
        mid = (lo + hi) / 2;  soft = mid -/+ factor * (hi - lo) / 2

    Returns:
        (lower, upper) arrays of shape (16,) in MJCF joint file order
        (joint ids 1..16; id 0 is the free joint).
    """
    ranges = model.jnt_range[1 : 1 + NUM_JOINTS]  # (16, 2)
    mid = 0.5 * (ranges[:, 0] + ranges[:, 1])
    half = 0.5 * (ranges[:, 1] - ranges[:, 0]) * SOFT_JOINT_POS_LIMIT_FACTOR
    return mid - half, mid + half


def evaluate_polynomials(
    entry: dict, phases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the library polynomials at normalized phases in [0, 1).

    Args:
        entry: One library entry (period, fps, coefficients dim_0..dim_39).
        phases: Normalized phases, shape (N,).

    Returns:
        joint_pos: (N, 16) joint positions, dims 0-15 (unclamped).
        joint_vel: (N, 16) joint velocities — analytic polynomial derivative
            divided by the period (NOT the noisy dims 16-31 fits).
        base_vel: (N, 3) heading-frame base linear velocity, dims 34-36.
    """
    period = float(entry["period"])
    coeffs = {
        d: np.asarray(entry["coefficients"][f"dim_{d}"], dtype=np.float64)
        for d in range(40)
    }

    polyval = np.polynomial.polynomial.polyval
    polyder = np.polynomial.polynomial.polyder

    joint_pos = np.stack(
        [polyval(phases, coeffs[j]) for j in range(NUM_JOINTS)], axis=-1
    )
    # Chain rule: the polynomial domain is normalized phase, so
    # d/dt = (d/dphase) / period.
    joint_vel = np.stack(
        [polyval(phases, polyder(coeffs[j])) / period for j in range(NUM_JOINTS)],
        axis=-1,
    )
    base_vel = np.stack(
        [
            polyval(phases, coeffs[d])
            for d in range(DIM_BASE_LIN_VEL[0], DIM_BASE_LIN_VEL[1])
        ],
        axis=-1,
    )
    return joint_pos, joint_vel, base_vel


def build_geom_vertex_cache(model: mujoco.MjModel) -> list[np.ndarray | None]:
    """Per-geom mesh vertices (geom frame) for exact lowest-point queries.

    MuJoCo's per-geom AABBs are conservative for rotated meshes (a rotated
    box corner dips ~1 cm below the true foot sole here), so mesh geoms use
    their actual vertices. Non-mesh geoms (None entries) fall back to the
    transformed AABB, which is exact for boxes/spheres along z.
    """
    cache: list[np.ndarray | None] = []
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[g]
            adr = model.mesh_vertadr[mesh_id]
            num = model.mesh_vertnum[mesh_id]
            cache.append(model.mesh_vert[adr : adr + num].astype(np.float64))
        else:
            cache.append(None)
    return cache


def geom_min_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    vertex_cache: list[np.ndarray | None],
) -> float:
    """Lowest world-z point over all robot geoms at the current FK state.

    During walking the lowest robot point is always a stance-foot sole, so
    this is the sole height. Mesh geoms use exact vertices; primitives use
    the transformed local AABB (``model.geom_aabb``: center + half-size).
    """
    best = np.inf
    xmats = data.geom_xmat.reshape(-1, 3, 3)
    for g in range(model.ngeom):
        row_z = xmats[g, 2, :]  # world-z row of the geom rotation
        verts = vertex_cache[g]
        if verts is not None:
            z = float(np.min(verts @ row_z)) + data.geom_xpos[g, 2]
        else:
            center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            z = data.geom_xpos[g, 2] + row_z @ center - np.abs(row_z) @ half
        if z < best:
            best = z
    return best


def convert_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    entry: dict,
    key: str,
    cycles: int,
    dof_names: list[str],
    body_ids: list[int],
    vertex_cache: list[np.ndarray | None],
) -> tuple[dict, dict]:
    """Convert one gait-library entry into MotionLoader npz arrays.

    Args:
        model: MuJoCo model of robot_motors.xml (floating "base" root,
            qpos[0:7] = xyz + WXYZ quat, qpos[7:23] = 16 joints in file
            order, which equals Playground polynomial order).
        data: Reusable MjData scratch buffer.
        entry: Gait library entry for ``key``.
        key: Library key "dx_dy_dtheta".
        cycles: Number of gait cycles to tile (root xy/yaw accumulate).
        dof_names: The 16 MJCF joint names (npz dof order).
        body_ids: MuJoCo body ids of BODY_NAMES.
        vertex_cache: Per-geom mesh vertices from build_geom_vertex_cache().

    Returns:
        (npz_arrays, diagnostics) — arrays ready for np.savez, plus a dict of
        per-motion validation numbers.
    """
    dx_cmd, dy_cmd, wz_cmd = parse_key(key)
    nb_steps = int(entry["nb_steps_in_period"])  # 27
    fps = int(entry["fps"])  # 50 == policy control rate (exactness matters)
    period = float(entry["period"])  # 0.54
    dt = 1.0 / fps
    num_frames = nb_steps * cycles

    # 1. Native sampling grid: t_k = (k mod 27) / 27, normalized phase.
    frame_idx = np.arange(num_frames)
    phases = (frame_idx % nb_steps) / nb_steps
    t_sec = frame_idx * dt

    joint_pos, joint_vel, base_vel = evaluate_polynomials(entry, phases)

    # 2. Clamp joint positions to the 0.9x soft limits (the knee swing peaks
    # exceed the hard +/-1.5708 rad range and are unreachable in the env).
    soft_lo, soft_hi = soft_joint_limits(model)
    clamped_mask = (joint_pos < soft_lo) | (joint_pos > soft_hi)
    max_clamp = float(
        np.max(np.maximum(soft_lo - joint_pos, joint_pos - soft_hi).clip(min=0.0))
    )
    joint_pos = np.clip(joint_pos, soft_lo, soft_hi)

    # 3. Joint velocities: differentiate the CLAMPED position trajectory
    # (periodic central difference over the tiled cycle) instead of keeping
    # the analytic unclamped polynomial derivative. Where a position sits
    # pinned at a limit the velocity must read ~0 — otherwise the reference
    # contains physically impossible (position-at-limit, velocity-nonzero)
    # states that the policy can never produce, handing the AMP discriminator
    # a trivial real/fake separator. Phase A mode-collapsed to standing with
    # discriminator loss ~0.04 before this fix (Gate G2 failure, 2026-06-12).
    # The tiled sequence repeats the cycle exactly, so np.roll wraparound is
    # periodic-correct; the small seam jump shows up consistently in both
    # positions and velocities, which is exactly what consistency requires.
    joint_vel = (np.roll(joint_pos, -1, axis=0) - np.roll(joint_pos, 1, axis=0)) / (2.0 * dt)

    # 4./5. Root pose, accumulating across tiled cycles. The pkl base
    # velocity is heading-aligned at generation time, so rotate it by the
    # accumulated yaw and integrate (trapezoid) to get world xy.
    yaw = wz_cmd * t_sec
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    v_world = np.stack(
        [
            cos_y * base_vel[:, 0] - sin_y * base_vel[:, 1],
            sin_y * base_vel[:, 0] + cos_y * base_vel[:, 1],
        ],
        axis=-1,
    )
    root_xy = np.zeros((num_frames, 2))
    root_xy[1:] = np.cumsum(0.5 * (v_world[:-1] + v_world[1:]) * dt, axis=0)
    root_quat = yaw_quat(yaw)  # roll = pitch = 0 (placo walk_trunk_pitch=0)

    # 6. Forward kinematics per frame with root z = 0; the constant root z
    # offset is applied afterwards (z is invariant to root xy/yaw, so one FK
    # pass suffices). The sole height is cycle-periodic, so it is only
    # evaluated over the first cycle.
    num_bodies = len(body_ids)
    body_pos = np.zeros((num_frames, num_bodies, 3))
    body_rot = np.zeros((num_frames, num_bodies, 4))
    sole_z = np.zeros(nb_steps)
    for k in range(num_frames):
        data.qpos[0:2] = root_xy[k]
        data.qpos[2] = 0.0
        data.qpos[3:7] = root_quat[k]
        data.qpos[7 : 7 + NUM_JOINTS] = joint_pos[k]
        mujoco.mj_forward(model, data)
        body_pos[k] = data.xpos[body_ids]
        body_rot[k] = data.xquat[body_ids]  # MuJoCo xquat is already WXYZ
        if k < nb_steps:
            sole_z[k] = geom_min_z(model, data, vertex_cache)

    # Constant root z so the lower stance-foot sole rides at z ~ 0: negated
    # mean of the per-frame lowest sole point over one cycle (the duck gait
    # has mm-level foot clearance, so the deviation is ~+/-5 mm).
    root_z = -float(np.mean(sole_z))
    body_pos[:, :, 2] += root_z

    # Body velocities by finite differences.
    body_lin_vel = np.stack(
        [central_difference(body_pos[:, b], dt) for b in range(num_bodies)],
        axis=1,
    )
    body_ang_vel = np.stack(
        [angular_velocity_from_quats(body_rot[:, b], dt) for b in range(num_bodies)],
        axis=1,
    )

    npz_arrays = {
        "fps": np.int64(fps),
        "dof_names": np.array(dof_names),
        "body_names": np.array(BODY_NAMES),
        "dof_positions": joint_pos.astype(np.float32),
        "dof_velocities": joint_vel.astype(np.float32),
        "body_positions": body_pos.astype(np.float32),
        "body_rotations": body_rot.astype(np.float32),
        "body_linear_velocities": body_lin_vel.astype(np.float32),
        "body_angular_velocities": body_ang_vel.astype(np.float32),
    }

    # --- Diagnostics ---
    # Seam continuity: joint deltas across every cycle boundary.
    seam_frames = np.arange(nb_steps, num_frames, nb_steps)
    if seam_frames.size:
        seam_jump = float(
            np.max(np.abs(joint_pos[seam_frames] - joint_pos[seam_frames - 1]))
        )
    else:
        seam_jump = 0.0
    # Total root displacement vs commanded velocity * duration. For a turning
    # gait the straight-line displacement is a chord, not the arc length:
    # chord = arc * |2 sin(theta/2) / theta|.
    duration = (num_frames - 1) * dt
    speed_cmd = float(np.hypot(dx_cmd, dy_cmd))
    theta = wz_cmd * duration
    chord_factor = abs(2.0 * np.sin(theta / 2.0) / theta) if abs(theta) > 1e-9 else 1.0
    expected_disp = speed_cmd * duration * chord_factor
    actual_disp = float(np.linalg.norm(root_xy[-1] - root_xy[0]))

    lf, rf = BODY_NAMES.index("foot_assembly"), BODY_NAMES.index("foot_assembly_2")
    diagnostics = {
        "num_frames": num_frames,
        "fps": fps,
        "period": period,
        "root_z": root_z,
        "feet_lateral_spacing_t0": float(abs(body_pos[0, lf, 1] - body_pos[0, rf, 1])),
        "seam_jump": seam_jump,
        "clamped_samples": int(np.count_nonzero(clamped_mask)),
        "max_clamp": max_clamp,
        "actual_displacement": actual_disp,
        "expected_displacement": expected_disp,
        "has_nan": any(
            bool(np.isnan(v).any()) for v in npz_arrays.values() if v.dtype.kind == "f"
        ),
    }
    return npz_arrays, diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Open Duck polynomial gaits to Isaac Lab AMP "
        "MotionLoader .npz files (offline; no Isaac Sim / CUDA needed)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--keys", nargs="+", metavar="KEY",
        help='Library keys to convert, e.g. "0.148_-0.037_-0.074"',
    )
    group.add_argument(
        "--all", action="store_true", help="Convert all 240 library motions."
    )
    group.add_argument(
        "--preset", choices=["forward", "curated24"],
        help='"forward": the near-forward key; "curated24": <=24-motion '
        "spread covering the command-box corners and cardinals.",
    )
    parser.add_argument(
        "--cycles", type=int, default=10,
        help="Gait cycles to tile per motion (default 10 = 270 frames).",
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    library = load_library(PKL_PATH)
    all_keys = sorted(library.keys())

    if args.all:
        keys = all_keys
    elif args.preset == "forward":
        keys = [FORWARD_KEY]
    elif args.preset == "curated24":
        keys = select_curated24(all_keys)
    else:
        keys = args.keys
    missing = [k for k in keys if k not in library]
    if missing:
        parser.error(f"Keys not in library: {missing}")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    # MJCF joint file order (== Playground order, verified). Joint id 0 is
    # the free root joint, ids 1..16 the actuated hinges.
    dof_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(1, 1 + NUM_JOINTS)
    ]
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in BODY_NAMES
    ]
    vertex_cache = build_geom_vertex_cache(model)

    os.makedirs(args.out, exist_ok=True)
    print(f"Converting {len(keys)} motion(s) -> {args.out}")
    print(f"dof_names: {dof_names}")
    print(f"body_names: {BODY_NAMES}")

    warnings = 0
    for key in keys:
        arrays, diag = convert_motion(
            model, data, library[key], key, args.cycles, dof_names, body_ids,
            vertex_cache,
        )
        out_path = os.path.join(args.out, f"duck_gait_{key}.npz")
        np.savez(out_path, **arrays)

        disp_err = (
            100.0
            * abs(diag["actual_displacement"] - diag["expected_displacement"])
            / max(diag["expected_displacement"], 1e-9)
        )
        print(
            f"[{key}] {diag['num_frames']} frames @ {diag['fps']} fps | "
            f"root_z={diag['root_z']:.4f} m | "
            f"feet_dy(t=0)={diag['feet_lateral_spacing_t0']:.4f} m | "
            f"seam_jump={diag['seam_jump']:.4f} rad | "
            f"clamped={diag['clamped_samples']} (max {diag['max_clamp']:.3f} rad) | "
            f"disp={diag['actual_displacement']:.3f} m "
            f"(cmd*T={diag['expected_displacement']:.3f} m, err {disp_err:.1f}%)"
            f" -> {os.path.basename(out_path)}"
        )
        if diag["has_nan"]:
            warnings += 1
            print(f"  WARNING [{key}]: NaN values in output arrays!")
        if diag["seam_jump"] >= SEAM_DISCONTINUITY_LIMIT:
            warnings += 1
            print(
                f"  WARNING [{key}]: seam discontinuity "
                f"{diag['seam_jump']:.4f} rad >= {SEAM_DISCONTINUITY_LIMIT} rad"
            )

    print(f"Done: {len(keys)} motion(s), {warnings} warning(s).")


if __name__ == "__main__":
    main()
