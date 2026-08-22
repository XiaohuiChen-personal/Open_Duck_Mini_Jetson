"""Free-running gait-phase clock — Task S.4.

`obs[51:53]` is a gait phase, and it is documented nowhere outside this file.
In training, `ImitationReward` keeps a per-env integer `_step_idx` that
increments once per control step and wraps at the matched motion's
`nb_steps_in_period` (`imitation_reward.py:246, :332`).

**The matched motion turns out not to matter.** `nb_steps_in_period == 27` for
all 240 entries of `polynomial_coefficients.pkl` and all 388 of
`polynomial_coefficients_v2.pkl`, so the runtime needs only a free-running
mod-27 counter at 50 Hz. `tests/test_runtime_core.py::test_nb_steps_uniform`
re-derives that from both pickles rather than trusting this comment — a future
reference library could break it.

    phase = 2*pi * (k mod 27) / 27
    obs[51:53] = [cos(phase), sin(phase)]

**Reset policy.** In sim `_step_idx` resets to 0 on episode termination
(`imitation_reward.py:200-205`). On hardware there are no episodes. This class
resets only when `reset()` is called explicitly — i.e. when the policy is
started or restarted — and **deliberately does NOT reset on a stumble.**

Rationale: a stumble is a disturbance to recover from, not a new gait. Resetting
mid-stride would discontinuously jump the phase the policy is tracking and fight
the recovery it is already attempting. If that is ever revisited, change it here
and record it in `deployment_contract.md`; do not let callers reset ad hoc.
"""

from __future__ import annotations

import math


class GaitPhase:
    """Mod-N step counter yielding (cos, sin) of the gait phase."""

    def __init__(self, nb_steps: int) -> None:
        if nb_steps <= 0:
            raise ValueError(f"nb_steps must be positive, got {nb_steps}")
        self.nb_steps = int(nb_steps)
        self._k = 0

    @property
    def k(self) -> int:
        """Current step index, always in [0, nb_steps)."""
        return self._k

    def reset(self, k: int = 0) -> None:
        """Restart the clock. `k` allows resuming a recorded trace mid-stream."""
        self._k = int(k) % self.nb_steps

    def peek(self) -> tuple[float, float]:
        """(cos, sin) for the CURRENT k, without advancing."""
        phase = 2.0 * math.pi * self._k / self.nb_steps
        return math.cos(phase), math.sin(phase)

    def step(self) -> tuple[float, float]:
        """Return (cos, sin) for the current step, THEN advance.

        Order matters: the observation at control step k must carry phase k, and
        the counter must be ready for k+1 before the next call.
        """
        out = self.peek()
        self._k = (self._k + 1) % self.nb_steps
        return out
