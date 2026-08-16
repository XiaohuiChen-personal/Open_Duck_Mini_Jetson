"""Torque-cost reward terms — SERVO-2.

`known_issues.md` **SERVO-2**: the shipped `v6d_contact_wrench` policy runs its
worst leg joint at **2.735 N·m RMS = 174 %** of the STS3250's 16 kg·cm rated
torque, with four leg joints reaching the 4.903 N·m clip *and their p99 also at
the clip* — saturation is the operating point, not a heel-strike transient.

The cause was verified, not guessed: `dof_torques_l2` and `dof_acc_l2` were both
`None` in `env_cfg.py`. **Nothing in the reward function priced torque at all.**

Why this module exists rather than Isaac Lab's stock `mdp.joint_torques_l2`
-------------------------------------------------------------------------
The stock term squares ``asset.data.applied_torque``, which is **post-clip**.
`actuators/actuator_pd.py`, `ImplicitActuator.compute()`:

    self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + ...
    self.applied_effort  = self._clip_effort(self.computed_effort)

Because four leg joints already sit *at* the clip, over those steps
``applied_torque`` is a **constant that no action can lower** — the policy would
be charged for torque *delivered* and receive no gradient for asking for less
until it fell below 4.903 N·m. That is precisely the region SERVO-2 is about.

``computed_torque`` is what the actuator was **asked** for and stays
differentiable above the clip, so the penalty has a gradient exactly where the
problem lives.

Note both quantities are Isaac Lab's own approximation: for an implicit
actuator PhysX runs the PD internally and "does not compute this quantity
explicitly" (its docstring). Measurement (`measure_joint_torque.py`) reads
``applied_torque`` — what the hardware would actually deliver — while this
penalty reads ``computed_torque`` — what the policy demanded. Penalise the
demand; measure the delivery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_torques_commanded_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of squared **commanded** (pre-clip) joint torque, per environment.

    Returns shape ``(num_envs,)``. Pair with a negative weight.

    Measured on `v6d_contact_wrench` over the 14 actioned joints during a
    forward walk, mean ``sum(tau^2)`` = **51.84**, so a weight of ``-1e-2``
    costs ``0.518`` per step against an ``alive_bonus`` of ``+9.37`` — about
    4.7 % of the reward budget. Mainstream weights (legged_gym ``-1e-5``) do
    **not** transfer: they come from robots whose torques are 10-50x larger and
    this term is squared, so the same weight would deliver 100-2500x less
    penalty here.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.computed_torque[:, asset_cfg.joint_ids]), dim=1
    )
