# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared per-environment disturbance state for the v5 contact-rich track.

The event that *creates* disturbances (``contact_events.ContactRegimeEvent``)
and the rewards that must *react* to them (``gated_rewards``) live in different
managers and never see each other. This module is the one place they meet — the
same module-global registry pattern ``imitation_reward`` already uses to share
gait phase between a reward term and an observation term
(``imitation_reward._instances``).

Why a gate exists at all: while the robot is being shoved or pressed, tracking
the commanded velocity is the wrong objective — a policy punished for tracking
error during a recovery learns to resist rather than comply, which is the
failure mode Task 2.8 is trying to remove. Hartmann et al. ("Deep Compliant
Control for Legged Robots", ETH CRL 2024) freeze the tracking objective for the
duration of a disturbance plus a short hold, then restore it so the gait must
resume. ``gate`` is that switch, in [0, 1] per environment.

DESIGN CONSTRAINT (from the plan's adversarial review): the gate must only ever
be raised by **externally caused** events — an applied wrench, an impulse push,
a measured contact force. It must NOT be raised by anything the policy can
produce on its own (an angular-rate threshold, for instance): freezing the
tracking reward at a favourable value is worth more than tracking under a hard
command, so a self-inducible trigger is a reward exploit, not a safety feature.
"""

from __future__ import annotations

import torch

# Keyed by id(env) — one state object per live environment instance.
_registry: dict[int, "DisturbanceState"] = {}


class DisturbanceState:
    """Per-environment disturbance flags shared between events and rewards."""

    def __init__(self, num_envs: int, device: str) -> None:
        self.num_envs = num_envs
        self.device = device
        # 1.0 while a disturbance is active or still inside its hold window.
        self.gate = torch.zeros(num_envs, device=device)
        # Seconds of hold remaining after the disturbance itself ends.
        self.hold_left = torch.zeros(num_envs, device=device)
        # Environments whose obstacle box is currently in play (not parked).
        self.obstacle_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Environments currently carrying an applied wrench.
        self.wrench_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def mark(self, env_ids: torch.Tensor | slice | None, hold_s: float) -> None:
        """Raise the gate for `env_ids` and (re)start their hold window."""
        if env_ids is None:
            env_ids = slice(None)
        self.gate[env_ids] = 1.0
        self.hold_left[env_ids] = torch.clamp(self.hold_left[env_ids], min=hold_s)

    def tick(self, dt: float) -> None:
        """Advance hold timers once per policy step and drop expired gates."""
        self.hold_left -= dt
        expired = self.hold_left <= 0.0
        self.hold_left[expired] = 0.0
        # An environment stays gated while its wrench is still applied, even if
        # the hold window has run out.
        self.gate[expired & ~self.wrench_active] = 0.0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.gate[env_ids] = 0.0
        self.hold_left[env_ids] = 0.0
        self.obstacle_active[env_ids] = False
        self.wrench_active[env_ids] = False


def get_state(env) -> DisturbanceState:
    """Fetch (creating on first use) the disturbance state for `env`."""
    state = _registry.get(id(env))
    if state is None:
        state = DisturbanceState(env.num_envs, env.device)
        _registry[id(env)] = state
    return state


def peek_state(env) -> DisturbanceState | None:
    """Fetch the state without creating it.

    Reward terms use this so that an environment configured WITHOUT the
    contact-regime event (every v3/v4 task) degrades to "never gated" instead
    of silently allocating state nobody updates.
    """
    return _registry.get(id(env))
