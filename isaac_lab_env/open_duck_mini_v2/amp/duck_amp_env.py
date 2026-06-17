# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Open Duck Mini Jetson Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) environment for the Open Duck Mini v2.

Ported from the Isaac Lab template at
``isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env.py`` with these
deliberate deltas (see ``duck_amp_env_cfg.py`` for the config-side rationale):

1. **Action application** — ``target = default_joint_pos + 0.25 * action``,
   the duck/playground convention shared with the PPO pipeline and on-robot
   deployment. The template's ``mid + range * action`` formula centers actions
   on the middle of the joint range, which for the duck is NOT the standing
   pose (e.g. knees at 1.37 rad vs a symmetric ±1.57 range) — zero-action
   would buckle the robot.
2. **Velocity commands (optional)** — when ``cfg.include_command_obs`` is set,
   a 3-dim (vx, vy, wz) command is appended to the POLICY observation only
   and a velocity-tracking task reward is returned. The AMP observation never
   sees the command: the discriminator judges *style*, not *intent*. The
   task/style mixing happens in the skrl agent config
   (task_reward_weight/style_reward_weight), not here.
3. **Heading localization** — root velocities and orientation are rotated
   into the yaw-heading frame before assembling the AMP frame, so the
   discriminator cannot key on the world heading of the (+x-walking)
   reference clips.
4. **Multi-clip reference data** — reference state initialization (RSI) and
   discriminator dataset collection sample from a :class:`MultiMotionLoader`
   weighted by clip duration.
5. **Reset root lift +0.02 m** (template: +0.15 m) — the duck's reference
   clips already place feet at ground level; a 2 cm lift avoids initial
   penetration without a visible drop at 0.17 m standing height.
"""

from __future__ import annotations

import glob
import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat

from .duck_amp_env_cfg import DuckAmpEnvCfgBase
from .motion_loader import MultiMotionLoader


class DuckAmpEnv(DirectRLEnv):
    cfg: DuckAmpEnvCfgBase

    def __init__(self, cfg: DuckAmpEnvCfgBase, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action application: default standing pose + scaled action
        # (duck/playground convention — NOT the template's mid+range formula).
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        # load motions: expand glob patterns, deduplicate while keeping order
        motion_files: list[str] = []
        for pattern in self.cfg.motion_files:
            motion_files.extend(sorted(glob.glob(pattern)))
        motion_files = list(dict.fromkeys(motion_files))
        assert len(motion_files) > 0, (
            f"No motion files found for patterns: {self.cfg.motion_files}. "
            "Generate reference clips (50 fps .npz) into amp/motions/ first."
        )
        self._motion_loader = MultiMotionLoader(motion_files=motion_files, device=self.device)

        # The AMP observation history is spaced at the motion frame dt. The
        # policy step (sim.dt * decimation) must match it, otherwise the
        # discriminator sees differently-spaced frame pairs from the sim vs
        # the dataset (the template's history-spacing mismatch).
        assert math.isclose(self._motion_loader.dt, self.step_dt, rel_tol=1e-4), (
            f"Motion frame dt ({self._motion_loader.dt}) must match the policy step dt "
            f"({self.step_dt}). Re-export the motion clips at {1.0 / self.step_dt:.0f} fps."
        )

        # DOF and key body indexes
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in self.cfg.key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(self.cfg.key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # velocity commands (policy obs + task reward only — never in AMP obs)
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.include_command_obs:
            self._resample_commands(self.robot._ALL_INDICES)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # contact sensor — inert for training; gives evaluate_policies.py the
        # same contact-based gait metrics as the manager-based PPO envs
        self.scene.sensors["contact_forces"] = ContactSensor(self.cfg.contact_sensor)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Bound raw actions BEFORE storing them (run-12 fix for run-11's
        # divergence). Root cause: _apply_action clamps PD targets to soft
        # limits, which decouples raw actions from robot motion — an unbounded
        # GaussianMixin mean could emit ever-larger thrashing actions that all
        # clamp to the same standing pose, with no physical feedback (no fall)
        # to correct them, and the soft action-rate penalty alone could not
        # contain the positive-feedback blowup (reward -> -6766). Clipping the
        # raw action to +/-action_clip restores a bounded action space:
        # +/-5.0 * scale 0.25 = +/-1.25 rad reach, which still covers the full
        # reference gait ROM (knee/ankle swings ~0.6-0.7 rad) while killing the
        # divergence channel entirely.
        actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        self._prev_actions = self.actions.clone() if hasattr(self, "actions") else actions.clone()
        self.actions = actions.clone()
        # resample velocity commands on a fixed wall-clock schedule
        if self.cfg.include_command_obs:
            self._command_time_left -= self.step_dt
            expired_ids = (self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
            if len(expired_ids) > 0:
                self._resample_commands(expired_ids)

    def _apply_action(self):
        # duck/playground convention: offset by the standing pose, scale by 0.25.
        # Targets are clamped to the soft joint limits — real servos clamp too
        # (deployment parity), and unbounded targets enabled the run-6
        # action-dithering exploit (|action| spikes of ~26/step that the PD
        # loop low-pass filtered into a standing posture).
        target = self.default_joint_pos + self.cfg.action_scale * self.actions
        limits = self.robot.data.soft_joint_pos_limits
        target = torch.clamp(target, limits[..., 0], limits[..., 1])
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build AMP observation frame (51 dims)
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
            self.cfg.heading_localize_amp_obs,
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        # the policy additionally observes the velocity command (the AMP
        # observation/discriminator never does)
        if self.cfg.include_command_obs:
            obs = torch.cat([obs, self._commands], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Pure style task: constant reward — learning is driven entirely by the
        # discriminator (skrl agent: task_reward_weight=0.0).
        if not self.cfg.include_command_obs:
            return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

        # Command task: velocity tracking in the yaw-heading frame, mixed with
        # the style reward by the skrl agent (task/style weights live in the
        # agent yaml, not here).
        #
        # Kernel widths (run-7 fix): std 0.1 m/s linear / 0.25 rad/s angular.
        # Run 6 used exp(-8*e2)/exp(-2*e2), under which STANDING STILL already
        # earned ~0.87/1.0 for duck-scale commands (|cmd| <= 0.26 m/s) — the
        # measured stand-and-dither exploit. With std=0.1, ignoring a 0.2 m/s
        # command earns ~0.02: walking is now the only way to get paid.
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_vel = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        lin_vel_heading = quat_apply_inverse(yaw_quat(root_quat), root_lin_vel)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - lin_vel_heading[:, :2]), dim=-1)
        ang_vel_error = torch.square(self._commands[:, 2] - root_ang_vel[:, 2])
        r_track = 0.5 * torch.exp(-lin_vel_error / 0.01) + 0.5 * torch.exp(-ang_vel_error / 0.0625)

        # Action-rate penalty: the AMP template has no smoothness term — the
        # style reward is supposed to enforce naturalness, but a saturated
        # discriminator provides no gradient, leaving dithering free (run 6:
        # mean |da| = 26/step). Normalized per joint; weight keeps the penalty
        # ~0.01-0.05 for smooth gaits and ruinous for dithering.
        action_rate = torch.mean(torch.square(self.actions - self._prev_actions), dim=-1)
        return r_track - 0.05 * action_rate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # fresh velocity command for the new episode
        if self.cfg.include_command_obs:
            self._resample_commands(env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random clips and motion times (or time zero if start is True)
        num_samples = env_ids.shape[0]
        loader_ids = self._motion_loader.sample_loader_ids(num_samples)
        if start:
            times = np.zeros(num_samples)
        else:
            _, times = self._motion_loader.sample_times(num_samples, loader_ids)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, loader_ids=loader_ids, times=times)

        # get root transforms (the duck base)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.02  # lift the duck slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, loader_ids, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(
        self,
        num_samples: int,
        current_loader_ids: np.ndarray | None = None,
        current_times: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Sample AMP observation frames from the reference motion library.

        Called by the skrl AMP agent (with ``num_samples`` only) to fill the
        discriminator's motion dataset, and by :meth:`_reset_strategy_random`
        (with explicit clip/time assignments) for reference state init.
        """
        # sample random (clip, time) pairs, or use the specified ones
        if current_loader_ids is None or current_times is None:
            current_loader_ids, current_times = self._motion_loader.sample_times(num_samples)
        # history spacing = motion frame dt (= policy step dt by construction)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        loader_ids = np.repeat(current_loader_ids, self.cfg.num_amp_observations)
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, loader_ids=loader_ids, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
            self.cfg.heading_localize_amp_obs,
        )
        return amp_observation.view(-1, self.amp_observation_size)

    # helpers

    def _resample_commands(self, env_ids: torch.Tensor):
        """Draw new (vx, vy, wz) commands uniformly within the configured ranges."""
        commands = torch.zeros((len(env_ids), 3), device=self.device)
        commands[:, 0].uniform_(*self.cfg.command_vx_range)
        commands[:, 1].uniform_(*self.cfg.command_vy_range)
        commands[:, 2].uniform_(*self.cfg.command_wz_range)
        self._commands[env_ids] = commands
        self._command_time_left[env_ids] = self.cfg.command_resampling_time_s


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    heading_localize: bool,
) -> torch.Tensor:
    # Heading localization: express root velocities, orientation AND the key
    # body offsets in the yaw-heading frame so the AMP frame is fully
    # invariant to world heading. Without rotating the feet offsets too, the
    # discriminator could still key on heading via the world-frame feet
    # positions (the reference clips' yaw drifts as turning gaits accumulate).
    key_body_offsets = key_body_positions - root_positions.unsqueeze(-2)
    if heading_localize:
        heading = yaw_quat(root_rotations)
        root_rotations = quat_mul(quat_inv(heading), root_rotations)
        root_linear_velocities = quat_apply_inverse(heading, root_linear_velocities)
        root_angular_velocities = quat_apply_inverse(heading, root_angular_velocities)
        num_key_bodies = key_body_offsets.shape[1]
        key_body_offsets = quat_apply_inverse(
            heading.repeat_interleave(num_key_bodies, dim=0),
            key_body_offsets.reshape(-1, 3),
        ).reshape(-1, num_key_bodies, 3)
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            # key body positions relative to the root (heading-local frame)
            key_body_offsets.reshape(key_body_offsets.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
