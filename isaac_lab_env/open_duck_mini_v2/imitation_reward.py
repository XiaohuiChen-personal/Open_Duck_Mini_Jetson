# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Reference motion imitation reward for Open Duck Mini v2.

Uses polynomial gait library from the Open Duck Playground to reward the policy
for matching reference leg joint positions. The reference motion is selected
based on the current velocity command (nearest-neighbor in the 240-entry library).

Only leg joints (10 DOFs) are tracked — head/antenna joints are left free for
other reward terms (joint_deviation_head).

Phase advances each policy step (0.02 s at 50 Hz) and wraps at the gait period
(0.54 s = 27 steps). The polynomial coefficients are degree-15, evaluated using
Horner's method for numerical stability.
"""

import math
import os
import pickle

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase

# Module-level storage for sharing phase state with observation function.
# Keyed by env id, value is the ImitationReward instance.
_instances: dict[int, "ImitationReward"] = {}

# Playground polynomial joint order (Open Duck Mini v2, 16 joints).
# This is the order used in polynomial_coefficients.pkl dimensions 0-15.
# Source: Open Duck Playground IsaacGym order + head_roll added in v2.
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

# Leg joint names tracked by the imitation reward (10 of 16 DOFs).
LEG_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


class ImitationReward(ManagerTermBase):
    """Reference motion imitation reward using polynomial gait library.

    Loads polynomial walking gaits from the Open Duck Playground and rewards
    the policy for matching reference leg joint positions. Only leg joints
    (10 DOFs) are tracked — head joints are left free for other reward terms.

    Phase advances based on policy timestep and wraps at the gait period.
    The closest reference motion is selected based on current velocity command.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Load polynomial coefficients
        data_path = os.path.join(
            os.path.dirname(__file__), "data", "polynomial_coefficients.pkl"
        )
        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)

        # Parse reference motions into tensors
        velocities = []
        all_coeffs = []  # (num_motions, 16_joints, 16_poly_coeffs)
        periods = []

        for key in sorted(raw_data.keys()):
            entry = raw_data[key]
            parts = key.split("_")
            velocities.append([float(parts[0]), float(parts[1]), float(parts[2])])
            periods.append(entry["period"])

            # Extract joint position polynomial coefficients (dims 0-15)
            motion_coeffs = []
            for dim_idx in range(16):
                motion_coeffs.append(entry["coefficients"][f"dim_{dim_idx}"])
            all_coeffs.append(motion_coeffs)

        self._velocities = torch.tensor(
            velocities, dtype=torch.float32, device=env.device
        )  # (240, 3)
        self._coefficients = torch.tensor(
            all_coeffs, dtype=torch.float32, device=env.device
        )  # (240, 16, 16)
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

        # Policy timestep (physics_dt * decimation)
        self._dt = env.step_dt

        # Joint order mapping — built lazily on first __call__ when robot data
        # is available from the simulation.
        self._pg_leg_indices: torch.Tensor | None = None
        self._il_leg_indices: torch.Tensor | None = None

        # Register instance for phase observation access
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
        # Lazy-init joint mapping (needs robot data from simulation)
        if self._il_leg_indices is None:
            self._build_joint_mapping(env)

        # 1. Match velocity command to closest reference motion
        vel_cmd = env.command_manager.get_command(command_name)[:, :3]  # (N, 3)
        diffs = vel_cmd.unsqueeze(1) - self._velocities.unsqueeze(0)  # (N, M, 3)
        dists = torch.sum(diffs**2, dim=-1)  # (N, M)
        self._current_motion_idx = torch.argmin(dists, dim=-1)  # (N,)

        # 2. Get polynomial coefficients for matched motions
        periods = self._periods[self._current_motion_idx]  # (N,)
        coeffs = self._coefficients[self._current_motion_idx]  # (N, 16, 16)

        # 3. Evaluate polynomial at current phase (Horner's method, ascending order)
        #    P(t) = c[0] + c[1]*t + c[2]*t^2 + ... + c[15]*t^15
        #    Horner: result = c[15]; for i in 14..0: result = result*t + c[i]
        t = self._phase  # (N,)
        result = coeffs[:, :, 15]  # (N, 16)
        for i in range(14, -1, -1):
            result = result * t.unsqueeze(-1) + coeffs[:, :, i]
        # result: (N, 16) — reference joint positions in Playground order

        # 4. Extract leg joints from reference and actual positions
        ref_leg_pos = result[:, self._pg_leg_indices]  # (N, 10)
        actual_leg_pos = env.scene["robot"].data.joint_pos[
            :, self._il_leg_indices
        ]  # (N, 10)

        # 5. Squared L2 error → exponential reward
        error = torch.sum((actual_leg_pos - ref_leg_pos) ** 2, dim=-1)  # (N,)
        reward = torch.exp(-2.0 * error)

        # 6. Advance phase for next step (in-place for observation access)
        self._phase.add_(self._dt)
        over = self._phase >= periods
        if over.any():
            self._phase[over] -= periods[over]

        return reward


def gait_phase_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation term: [cos(phase), sin(phase)] of the gait cycle.

    Returns (num_envs, 2). Returns zeros if ImitationReward is not active.
    """
    instance = _instances.get(id(env))
    if instance is None:
        return torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)

    phase = instance._phase
    periods = instance._periods[instance._current_motion_idx]
    phase_norm = 2.0 * math.pi * phase / periods.clamp(min=1e-6)

    return torch.stack(
        [torch.cos(phase_norm), torch.sin(phase_norm)], dim=-1
    )
