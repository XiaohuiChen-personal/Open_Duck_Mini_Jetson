# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command with deliberate exposure to sustained fast rotation (v5).

Why this exists
---------------
Five of the ten Duck Embody benchmark falls happened at |wz| = 0.5 rad/s
exactly — the deployment rotation hull limit — and four of those were during a
``turn_to_heading`` macro that holds that rate for up to 8 s. v4_robust had
effectively never experienced it, for two compounding reasons:

1. Every env used the heading servo (``rel_heading_envs = 1.0``), which sets
   wz = clip(0.5 * heading_error, +/-0.5). That *decays* as the robot aligns,
   so |wz| = 0.5 appears only in transient bursts, never held.
2. The command range was exactly +/-0.5, so 0.5 sat on the boundary of the
   training distribution rather than inside it.

Isaac Lab already solves (1) natively: ``rel_heading_envs`` is a per-env split,
re-drawn at every resample, and non-heading envs hold a directly sampled wz for
the whole command interval (velocity_command.py:139-159). That alone is not
enough, though — a plain uniform draw over the range puts only a small
minority of envs anywhere near the rate that kills. This class adds the two
missing pieces:

* **Banded sampling.** A configurable share of direct-wz draws is forced into
  the high-|wz| band instead of spread uniformly, so the killer rate is
  actually trained rather than merely reachable.
* **Asymmetric clipping.** The sampling range is wider than deployment
  (+/-0.7) so that the deployment limit of 0.5 becomes an interior point with
  margin, while the heading servo stays clipped at the deployment-exact +/-0.5
  so heading-hold behaviour is not silently changed. Upstream Open Duck
  Playground trains this robot at +/-1.0 and the reference library spans
  [-1.111, 1.222], so +/-0.7 is comfortably inside what the gait library can
  represent.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


class BandedWzVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with a high-|wz| band and a separate heading clip."""

    cfg: "BandedWzVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)

        if self.cfg.wz_band_frac <= 0.0:
            return

        # Only envs that will follow the directly-sampled wz are worth banding;
        # heading envs have their wz overwritten every step anyway.
        direct = ~self.is_heading_env[env_ids] if self.cfg.heading_command else None
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        lo, hi = self.cfg.wz_band_abs
        banded = r.uniform_(lo, hi) * torch.where(
            torch.rand(n, device=self.device) < 0.5, -1.0, 1.0
        )
        take_band = torch.rand(n, device=self.device) < self.cfg.wz_band_frac
        if direct is not None:
            take_band &= direct

        self.vel_command_b[env_ids, 2] = torch.where(
            take_band, banded, self.vel_command_b[env_ids, 2]
        )

    def _update_command(self) -> None:
        """Same as upstream, except the heading servo has its own clip.

        Reimplemented rather than delegated because the parent clips the
        heading-derived rate to ``ranges.ang_vel_z``, which v5 widens past the
        deployment limit for *sampling* purposes only.
        """
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            limit = self.cfg.heading_wz_limit
            if limit is None:
                lo, hi = self.cfg.ranges.ang_vel_z
            else:
                lo, hi = -limit, limit
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error, min=lo, max=hi
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class BandedWzVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`BandedWzVelocityCommand`."""

    class_type: type = BandedWzVelocityCommand

    wz_band_frac: float = 0.4
    """Share of direct-wz resamples forced into :attr:`wz_band_abs`.

    Set to 0.0 to recover exactly the stock uniform behaviour.
    """

    wz_band_abs: tuple[float, float] = (0.35, 0.7)
    """|wz| band (rad/s) that banded draws land in, sign chosen at random."""

    heading_wz_limit: float | None = 0.5
    """Clip applied to the heading servo's output, independent of the sampling
    range. ``None`` falls back to ``ranges.ang_vel_z`` (stock behaviour).

    Deployment's ``turn_to_heading`` slews at +/-0.5, so heading-driven
    rotation is kept deployment-exact even though sampling reaches further.
    """
