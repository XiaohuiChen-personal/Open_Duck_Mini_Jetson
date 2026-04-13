# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""BDX-style composite imitation reward for Open Duck Mini v2.

v2 reward design aligned with the Disney BDX paper ("Design and Control of a
Bipedal Robotic Character", Jan 2025) and Open Duck Playground.

Uses polynomial gait library (240 motions, degree-15, 0.54s period) to track:
- Joint positions (dims 0-15): raw quadratic penalty, weight 15.0
- Joint velocities (dims 16-31): raw quadratic penalty, weight 0.001
- Base linear velocity (dims 34-36): exp(-8*error), weight 1.0
- Foot contacts (dims 32-33): binary match reward, weight 1.0

Key difference from v1: raw quadratic `-||q - q_ref||^2 * 15.0` instead of
exponential `exp(-2*error) * 10.0`. The raw quadratic has linear gradient
(no saturation), demanding precise joint tracking.
"""

import math
import os
import pickle

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase

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

        # Load polynomial coefficients
        data_path = os.path.join(
            os.path.dirname(__file__), "data", "polynomial_coefficients.pkl"
        )
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)

        # Parse reference motions — now extract ALL 40 dims, not just 16
        velocities = []
        all_coeffs = []  # (num_motions, 40_dims, 16_poly_coeffs)
        periods = []

        for key in sorted(raw_data.keys()):
            entry = raw_data[key]
            parts = key.split("_")
            velocities.append([float(parts[0]), float(parts[1]), float(parts[2])])
            periods.append(entry["period"])

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

        # Per-env state
        self._phase = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._current_motion_idx = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

        self._dt = env.step_dt

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

    def reset(self, env_ids=None):
        """Reset gait phase on episode termination."""
        if env_ids is None:
            self._phase.zero_()
        else:
            self._phase[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "base_velocity",
    ) -> torch.Tensor:
        if self._il_leg_indices is None:
            self._build_joint_mapping(env)

        # 1. Match velocity command to closest reference motion
        vel_cmd = env.command_manager.get_command(command_name)[:, :3]
        diffs = vel_cmd.unsqueeze(1) - self._velocities.unsqueeze(0)
        dists = torch.sum(diffs**2, dim=-1)
        self._current_motion_idx = torch.argmin(dists, dim=-1)

        # 2. Evaluate ALL 40 polynomial dimensions at current phase
        periods = self._periods[self._current_motion_idx]
        coeffs = self._coefficients[self._current_motion_idx]  # (N, 40, 16)

        t = self._phase
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
        ref_leg_pos = ref_joint_pos[:, self._pg_leg_indices]
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

        # Advance phase
        self._phase.add_(self._dt)
        over = self._phase >= periods
        if over.any():
            self._phase[over] -= periods[over]

        return reward


def gait_phase_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation term: [cos(phase), sin(phase)] of the gait cycle."""
    instance = _instances.get(id(env))
    if instance is None:
        return torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)

    phase = instance._phase
    periods = instance._periods[instance._current_motion_idx]
    phase_norm = 2.0 * math.pi * phase / periods.clamp(min=1e-6)

    return torch.stack(
        [torch.cos(phase_norm), torch.sin(phase_norm)], dim=-1
    )
