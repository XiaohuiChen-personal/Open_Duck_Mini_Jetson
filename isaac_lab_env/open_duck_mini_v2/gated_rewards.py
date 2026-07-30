# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Disturbance-gated and compliance-friendly reward terms for the v5 track.

The v4 reward asks the policy to track its velocity command at every instant.
That is the right objective while walking and the wrong one while being shoved:
a policy penalised for tracking error during a recovery learns to *resist* the
disturbance — stiffen, plant, refuse to yield — which is precisely the
behaviour that tips a 2.66 kg biped over when it meets furniture. Hartmann et
al. ("Deep Compliant Control for Legged Robots", ETH CRL 2024) freeze the
tracking objective while a disturbance is active and restore it afterwards, so
compliance is free during the event but the gait still has to come back.

Three terms implement that here:

* ``GatedTrackLinVel`` / ``GatedTrackAngVel`` — the stock exponential tracking
  rewards, except that while the gate is up they return a frozen running
  average of their own recent (undisturbed) value. A constant carries no
  gradient with respect to the action, so during a disturbance the policy is
  simply not steered by tracking at all; it is steered by staying alive and
  upright. When the gate drops, the live reward returns and the frozen value
  starts updating again.

* ``flat_orientation_deadzone`` — ``flat_orientation_l2`` with a dead zone.
  Leaning into a sustained load is how a legged robot survives one; a
  quadratic penalty from the very first degree of tilt fights that. Inside the
  dead zone leaning is free, outside it the penalty is unchanged in shape.

* ``ground_contact_penalty`` — head/lower-leg contact penalised **only in
  environments with no obstacle in play**. This is deliberate and is the whole
  reason it is not ``mdp.undesired_contacts``: the training obstacle is 0.7 m
  tall against a 0.42 m robot, so head and shin contact with it is routine and
  desirable in the press-and-slide regime. Penalising contact per se would
  train obstacle *avoidance* — re-creating the "contact = death" failure mode
  Task 2.8 exists to remove (task_plan.md:1550-1556). In an obstacle-free env
  the only thing those bodies can touch is the ground, which is a genuine
  fault.

Why the gate is event-driven only
---------------------------------
An earlier draft also raised the gate on |base angular velocity| above a
threshold. Adversarial review killed it: that trigger is self-inducible, and
inducing it is *profitable* — the imitation composite is signed and goes
strongly negative under hard tracking, so freezing tracking at a favourable
constant while collecting the alive bonus beats tracking honestly. One torso
flick per second would have sustained the gate indefinitely. The gate is now
raised only by an applied wrench, an impulse push, or a measured trunk contact
force (see ``disturbance_state``), none of which the policy can manufacture on
an empty plane. ``Regime/gate_duty`` is logged every step so that any residual
exploitation is visible in TensorBoard rather than inferred from the videos.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaac_lab_env.open_duck_mini_v2 import disturbance_state as ds

# Smoothing for the frozen tracking value. At 50 Hz, 0.01 is roughly a 2 s
# window — long enough to average over several gait cycles, short enough to
# reflect the current command.
_EMA_ALPHA = 0.01


class _GatedTrackingReward(ManagerTermBase):
    """Shared machinery: live reward while ungated, frozen average while gated."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._state = ds.get_state(env)
        # Seeded at the exponential kernel's midpoint rather than 0 so the
        # first disturbance of an episode does not freeze a meaningless value.
        self._ema = torch.full((env.num_envs,), 0.5, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ema[env_ids] = 0.5

    def _blend(self, live: torch.Tensor) -> torch.Tensor:
        gate = self._state.gate
        # Update the running average only from undisturbed steps, so the frozen
        # value describes normal walking rather than the disturbance itself.
        self._ema = torch.where(
            gate > 0.0, self._ema, (1.0 - _EMA_ALPHA) * self._ema + _EMA_ALPHA * live.detach()
        )
        return torch.where(gate > 0.0, self._ema, live)


class GatedTrackLinVel(_GatedTrackingReward):
    """``mdp.track_lin_vel_xy_exp``, frozen while disturbed."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        vel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
        error = torch.sum(
            torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_b[:, :2]), dim=1
        )
        return self._blend(torch.exp(-error / std**2))


class GatedTrackAngVel(_GatedTrackingReward):
    """``mdp.track_ang_vel_z_world_exp``, frozen while disturbed."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        error = torch.square(
            env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
        )
        return self._blend(torch.exp(-error / std**2))


def flat_orientation_deadzone(
    env: ManagerBasedRLEnv,
    deadzone: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise non-flat orientation, but only beyond `deadzone` radians.

    ``mdp.flat_orientation_l2`` returns ``|g_xy|^2`` with no free region, so
    every degree of lean costs reward from the first instant. Task 2.8 asks for
    a ~0.1 rad dead zone so that leaning into a sustained load is not fought
    while the waddle stays intact (task_plan.md:1557-1558).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return torch.square(torch.clamp(tilt - deadzone, min=0.0))


def ground_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise head/lower-leg contact in environments with no obstacle in play.

    Scoping matters more than the weight here — see the module docstring. In an
    obstacle env this returns zero, so sliding along a wall on a shin or a head
    is free; in an obstacle-free env the same contact can only be the ground,
    which is a real fault worth a penalty.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = (forces.norm(dim=-1).amax(dim=1) > threshold).float().sum(dim=1)

    state = ds.peek_state(env)
    if state is None:
        return in_contact
    return in_contact * (~state.obstacle_active).float()


def gate_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The disturbance gate as a (num_envs, 1) signal, for logging/audit traces.

    Not an observation for the policy — the 59-dim actor contract is frozen by
    the deployment interface. This exists so ``audit_rollout.py`` can put the
    gate state in its per-step trace, which the review required in order to
    make spurious or exploited gating visible frame by frame.
    """
    state = ds.peek_state(env)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return state.gate.unsqueeze(-1)
