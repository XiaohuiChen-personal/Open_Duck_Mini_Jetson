# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""BDX-style composite imitation reward for Open Duck Mini v2.

v3 reward design aligned with the Disney BDX paper ("Design and Control of a
Bipedal Robotic Character", Jan 2025) and Open Duck Playground.

Uses polynomial gait library (240 motions, degree-15, 0.54s period) to track:
- Joint positions (dims 0-15): raw quadratic penalty, weight 15.0
- Joint velocities (dims 16-31): raw quadratic penalty, weight 0.001
- Base linear velocity (dims 34-36): exp(-8*error), weight 1.0
- Foot contacts (dims 32-33): binary match reward, weight 1.0

Key difference from v1: raw quadratic `-||q - q_ref||^2 * 15.0` instead of
exponential `exp(-2*error) * 10.0`. The raw quadratic has linear gradient
(no saturation), demanding precise joint tracking.

v3 fixes (vs the v2 run archived in exported_policies/v2_bdx_imitation_ppo):
1. PHASE BUG FIX: the polynomials are fit over NORMALIZED phase t in [0, 1]
   (generator fit_poly.py: `X = np.linspace(0, 1, ...)`), but v2 evaluated
   them at phase in SECONDS in [0, 0.54) — replaying only the first 54% of
   the gait cycle with a discontinuous reference jump at every wrap, which
   produced an asymmetric limping reference (left stance 78% vs right 52%).
   Now uses the upstream Playground convention: an integer control-step
   counter with t = (i % nb_steps_in_period) / nb_steps_in_period.
2. Reference joint positions are clamped to the robot's soft joint limits:
   the library's knee swing peaks (1.78/1.98 rad) exceed the model's
   +/-1.5708 rad limit, so the unclamped reference is physically unreachable.
3. The composite reward is gated to zero for near-zero velocity commands
   (||cmd|| <= 0.01), matching upstream — the library has no standing gait,
   so standing envs must not be rewarded for marching in place.
"""

import math
import os
import pickle

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase

from isaac_lab_env.open_duck_mini_v2 import disturbance_state as _disturbance_state

# Module-level storage for sharing phase state with observation function.
_instances: dict[int, "ImitationReward"] = {}

# Playground polynomial joint order (Open Duck Mini v2, 16 joints).
PLAYGROUND_JOINT_ORDER = [
    "left_hip_yaw",     # 0
    "left_hip_roll",    # 1
    "left_hip_pitch",   # 2
    "left_knee",        # 3
    "left_ankle",       # 4
    "neck_pitch",       # 5
    "head_pitch",       # 6
    "head_yaw",         # 7
    "head_roll",        # 8
    "left_antenna",     # 9
    "right_antenna",    # 10
    "right_hip_yaw",    # 11
    "right_hip_roll",   # 12
    "right_hip_pitch",  # 13
    "right_knee",       # 14
    "right_ankle",      # 15
]

LEG_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

# Contact body names for left/right foot (matching polynomial dims 32, 33)
FOOT_BODY_NAMES = ["foot_assembly", "foot_assembly_2"]


class ImitationReward(ManagerTermBase):
    """BDX-style composite imitation reward using polynomial gait library.

    Returns a single scalar that combines:
    - Joint position tracking (raw quadratic, BDX weight 15.0)
    - Joint velocity tracking (raw quadratic, BDX weight 0.001)
    - Base velocity tracking (exponential, BDX weight 1.0)
    - Contact matching (binary, BDX weight 1.0)

    The weights are baked into the composite — set RewTerm weight=1.0.
    """

    # BDX paper weights (Table I)
    W_JOINT_POS = 15.0
    W_JOINT_VEL = 0.001
    W_BASE_VEL = 1.0
    W_CONTACTS = 1.0

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Load polynomial coefficients. The library is selectable so the v5
        # track can train against the grid-completed variant
        # (polynomial_coefficients_v2.pkl, see scripts/patch_reference_library.py)
        # while every v3/v4 task and the eval protocol keep the frozen original
        # — the dataset all journaled gate metrics were measured against.
        reference_pkl = cfg.params.get("reference_pkl") or "polynomial_coefficients.pkl"
        data_path = os.path.join(os.path.dirname(__file__), "data", reference_pkl)
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)

        # Parse reference motions — now extract ALL 40 dims, not just 16
        velocities = []
        all_coeffs = []  # (num_motions, 40_dims, 16_poly_coeffs)
        periods = []
        nb_steps = []

        for key in sorted(raw_data.keys()):
            entry = raw_data[key]
            parts = key.split("_")
            velocities.append([float(parts[0]), float(parts[1]), float(parts[2])])
            periods.append(entry["period"])
            # Control steps per gait cycle (27 = 0.54 s at 50 Hz). The
            # polynomial domain is normalized phase t = i / nb_steps in [0, 1).
            nb_steps.append(int(entry["nb_steps_in_period"]))

            motion_coeffs = []
            for dim_idx in range(40):
                motion_coeffs.append(entry["coefficients"][f"dim_{dim_idx}"])
            all_coeffs.append(motion_coeffs)

        self._velocities = torch.tensor(
            velocities, dtype=torch.float32, device=env.device
        )  # (240, 3)
        self._coefficients = torch.tensor(
            all_coeffs, dtype=torch.float32, device=env.device
        )  # (240, 40, 16)
        self._periods = torch.tensor(
            periods, dtype=torch.float32, device=env.device
        )  # (240,)
        self._nb_steps = torch.tensor(
            nb_steps, dtype=torch.long, device=env.device
        )  # (240,)

        # Per-env state: integer control-step counter within the gait cycle,
        # matching upstream Playground's `imitation_i` (joystick.py).
        self._step_idx = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._current_motion_idx = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

        # Per-axis spans of the command grid, used only when the nearest-motion
        # match is span-normalized (see `normalized_match` in __call__): the
        # raw L2 mixes m/s with rad/s, so wz — whose span is ~10x that of vy —
        # dominates which gait a command snaps to.
        self._velocity_spans = torch.clamp(
            self._velocities.max(dim=0).values - self._velocities.min(dim=0).values,
            min=1e-6,
        )

        # Joint order mapping — built lazily
        self._pg_leg_indices: torch.Tensor | None = None
        self._il_leg_indices: torch.Tensor | None = None

        # Contact sensor body indices — built lazily
        self._foot_body_ids: list[int] | None = None

        # Register for phase observation
        _instances[id(env)] = self

    def _build_joint_mapping(self, env: ManagerBasedRLEnv):
        """Build Playground→Isaac Lab joint index mapping from runtime joint names."""
        isaac_joint_names = list(env.scene["robot"].data.joint_names)

        pg_leg_indices = []
        il_leg_indices = []
        for name in LEG_JOINT_NAMES:
            pg_leg_indices.append(PLAYGROUND_JOINT_ORDER.index(name))
            il_leg_indices.append(isaac_joint_names.index(name))

        self._pg_leg_indices = torch.tensor(
            pg_leg_indices, dtype=torch.long, device=self.device
        )
        self._il_leg_indices = torch.tensor(
            il_leg_indices, dtype=torch.long, device=self.device
        )

        # Build contact body indices using sensor's find_bodies API
        contact_sensor = env.scene["contact_forces"]
        foot_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES)
        self._foot_body_ids = foot_ids

        # Soft joint limits for the leg joints (identical across envs) — the
        # reference is clamped to these so it never demands unreachable poses.
        leg_limits = env.scene["robot"].data.soft_joint_pos_limits[
            0, self._il_leg_indices, :
        ]
        self._leg_pos_lower = leg_limits[:, 0].clone()  # (10,)
        self._leg_pos_upper = leg_limits[:, 1].clone()  # (10,)

    def reset(self, env_ids=None):
        """Reset gait phase on episode termination."""
        if env_ids is None:
            self._step_idx.zero_()
        else:
            self._step_idx[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "base_velocity",
        reference_pkl: str | None = None,
        normalized_match: bool = False,
        disturbance_gate_scale: float | None = None,
    ) -> torch.Tensor:
        """Composite imitation reward.

        Args:
            reference_pkl: consumed in ``__init__``; accepted here so the term
                config can carry it.
            normalized_match: divide each command axis by its grid span before
                the nearest-motion L2. Off by default — v3/v4 policies were
                trained with the raw mixed-unit distance and their journaled
                metrics depend on it.
            disturbance_gate_scale: when set, the whole composite is scaled
                toward this value while the environment's disturbance gate is
                up (v5). Recovering from a shove should not be scored against
                a nominal gait; see ``gated_rewards``.
        """
        if self._il_leg_indices is None:
            self._build_joint_mapping(env)

        # 1. Match velocity command to closest reference motion
        vel_cmd = env.command_manager.get_command(command_name)[:, :3]
        diffs = vel_cmd.unsqueeze(1) - self._velocities.unsqueeze(0)
        if normalized_match:
            diffs = diffs / self._velocity_spans.view(1, 1, 3)
        dists = torch.sum(diffs**2, dim=-1)
        self._current_motion_idx = torch.argmin(dists, dim=-1)

        # 2. Evaluate ALL 40 polynomial dimensions at the current NORMALIZED
        # phase t = (i % nb_steps) / nb_steps in [0, 1) — the domain the
        # polynomials were fit over (upstream poly_reference_motion.py).
        nb_steps = self._nb_steps[self._current_motion_idx]  # (N,)
        coeffs = self._coefficients[self._current_motion_idx]  # (N, 40, 16)

        t = (self._step_idx % nb_steps).float() / nb_steps.float()
        result = coeffs[:, :, 15]  # (N, 40)
        for i in range(14, -1, -1):
            result = result * t.unsqueeze(-1) + coeffs[:, :, i]
        # result: (N, 40) — all reference dimensions

        # 3. Extract reference components
        ref_joint_pos = result[:, :16]   # dims 0-15: joint positions
        ref_joint_vel = result[:, 16:32]  # dims 16-31: joint velocities
        ref_contacts = result[:, 32:34]   # dims 32-33: foot contacts
        ref_base_vel = result[:, 34:37]   # dims 34-36: base linear velocity

        # ================================================================
        # Term 1: Joint position tracking (BDX weight 15.0)
        # Raw quadratic: -||q_leg - q_ref_leg||^2
        # ================================================================
        # Clamp the reference to the robot's soft joint limits — the library's
        # knee swing peaks exceed the model's limits and would otherwise demand
        # physically unreachable poses (a permanent error floor).
        ref_leg_pos = ref_joint_pos[:, self._pg_leg_indices]
        ref_leg_pos = torch.clamp(
            ref_leg_pos, self._leg_pos_lower, self._leg_pos_upper
        )
        actual_leg_pos = env.scene["robot"].data.joint_pos[:, self._il_leg_indices]
        joint_pos_error = torch.sum((actual_leg_pos - ref_leg_pos) ** 2, dim=-1)
        r_joint_pos = -joint_pos_error  # negative, minimized by tracking

        # ================================================================
        # Term 2: Joint velocity tracking (BDX weight 0.001)
        # Raw quadratic: -||dq_leg - dq_ref_leg||^2
        # ================================================================
        ref_leg_vel = ref_joint_vel[:, self._pg_leg_indices]
        actual_leg_vel = env.scene["robot"].data.joint_vel[:, self._il_leg_indices]
        joint_vel_error = torch.sum((actual_leg_vel - ref_leg_vel) ** 2, dim=-1)
        r_joint_vel = -joint_vel_error

        # ================================================================
        # Term 3: Base velocity tracking (BDX weight 1.0)
        # Exponential: exp(-8 * ||v_xy - v_ref_xy||^2)
        # ================================================================
        actual_base_vel = env.scene["robot"].data.root_lin_vel_b[:, :3]
        base_vel_error = torch.sum(
            (actual_base_vel[:, :2] - ref_base_vel[:, :2]) ** 2, dim=-1
        )
        r_base_vel = torch.exp(-8.0 * base_vel_error)

        # ================================================================
        # Term 4: Contact matching (BDX weight 1.0)
        # Binary: reward when actual contact matches reference
        # ================================================================
        contact_forces = env.scene["contact_forces"].data.net_forces_w_history
        # Use current step (index 0), take norm of force vector
        foot_forces = torch.stack([
            torch.norm(contact_forces[:, 0, self._foot_body_ids[0], :], dim=-1),
            torch.norm(contact_forces[:, 0, self._foot_body_ids[1], :], dim=-1),
        ], dim=-1)  # (N, 2)
        actual_contacts = (foot_forces > 1.0).float()  # threshold at 1N
        ref_contacts_binary = (ref_contacts > 0.5).float()
        contact_match = (actual_contacts == ref_contacts_binary).float()
        r_contacts = torch.mean(contact_match, dim=-1)  # 0 to 1

        # ================================================================
        # Composite reward (BDX-style weighted sum)
        # ================================================================
        reward = (
            self.W_JOINT_POS * r_joint_pos
            + self.W_JOINT_VEL * r_joint_vel
            + self.W_BASE_VEL * r_base_vel
            + self.W_CONTACTS * r_contacts
        )

        # Gate to zero for near-zero commands (upstream parity): the library
        # has no standing gait, so standing envs get no imitation signal.
        cmd_active = (torch.norm(vel_cmd, dim=-1) > 0.01).float()
        reward = reward * cmd_active

        # Fade the imitation prior out while the robot is being disturbed: the
        # reference library contains only nominal gaits, so demanding precise
        # joint tracking mid-recovery penalises exactly the compliance v5 is
        # trying to learn (Hartmann et al., ETH CRL 2024).
        if disturbance_gate_scale is not None:
            state = _disturbance_state.peek_state(env)
            if state is not None:
                reward = reward * (1.0 - (1.0 - disturbance_gate_scale) * state.gate)

        # Advance the gait-cycle step counter (one increment per control step)
        self._step_idx = (self._step_idx + 1) % nb_steps

        return reward


def gait_phase_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation term: [cos(phase), sin(phase)] of the gait cycle."""
    instance = _instances.get(id(env))
    if instance is None:
        return torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)

    nb_steps = instance._nb_steps[instance._current_motion_idx].float()
    phase_norm = (
        2.0 * math.pi * instance._step_idx.float() / nb_steps.clamp(min=1.0)
    )

    return torch.stack(
        [torch.cos(phase_norm), torch.sin(phase_norm)], dim=-1
    )
