"""Turn raw ONNX output into joint targets — Task S.4.

    q_target = q_default + action_scale * action        (action_scale = 0.25)

DEPLOY-1: neither `action_scale` nor `q_default` is in the ONNX graph. Both come
from the contract, and getting either wrong is a ~1 rad error, not a rounding
error — which is why `tests/test_runtime_core.py` asserts this against the S.2
traces rather than trusting the arithmetic.

**Isaac does not clamp.** `clip_actions` is null, `clip` is null, and
`JointPositionAction` applies no clamp of any kind — PhysX clamps to HARD limits,
not soft. So a recorded `joint_pos_target` may legitimately sit outside the soft
limits, and `raw_target()` must be compared against it *before* any clamping.

`decode()` adds the clamp the runtime does need, and RETURNS THE COUNT. A nonzero
count is a fact about an unbounded policy, not a bug in this module — treat it as
a signal to watch, not an assertion to satisfy.
"""

from __future__ import annotations

import numpy as np


class ActionDecoder:
    """Affine decode, optional clamp, and servo-unit conversion."""

    def __init__(self, contract: dict, calibration: dict | None = None) -> None:
        self.action_dim = int(contract["action_dim"])
        self.action_scale = float(contract["action_scale"])
        self.joint_order = list(contract["joint_order"])
        self.q_default = np.asarray(contract["q_default_rad"], dtype=np.float32)

        soft = contract["soft_limits_rad"]
        hard = contract["hard_limits_rad"]
        self.soft_low = np.array([soft[j][0] for j in self.joint_order], dtype=np.float32)
        self.soft_high = np.array([soft[j][1] for j in self.joint_order], dtype=np.float32)
        self.hard_low = np.array([hard[j][0] for j in self.joint_order], dtype=np.float32)
        self.hard_high = np.array([hard[j][1] for j in self.joint_order], dtype=np.float32)

        # S.9 supplies real zeros/signs. Identity defaults keep this module
        # usable before calibration exists, rather than blocking S.4 on S.9.
        self.calibration = calibration
        self.servo_ids = list(contract["servo_ids"].values()) \
            if isinstance(contract.get("servo_ids"), dict) else None

    # -- the pure affine, which is what the trace test compares against ----

    def raw_target(self, action: np.ndarray) -> np.ndarray:
        """`q_default + action_scale * action`. NO clamping. Matches Isaac."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"action has {action.shape[0]}, expected {self.action_dim}")
        return self.q_default + self.action_scale * action

    # -- what the runtime actually sends ----------------------------------

    def decode(self, action: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (clamped target, number of joints clamped this step)."""
        raw = self.raw_target(action)
        clamped = np.clip(raw, self.soft_low, self.soft_high)
        n_clamped = int(np.count_nonzero(raw != clamped))
        return clamped, n_clamped

    def decode_hard(self, action: np.ndarray) -> tuple[np.ndarray, int]:
        """Clamp to HARD limits — the mandatory floor even if soft is bypassed.

        Exceeding a hard limit is what PhysX itself refuses in sim; on hardware
        nothing refuses it, so this is the last line before the servo.
        """
        raw = self.raw_target(action)
        clamped = np.clip(raw, self.hard_low, self.hard_high)
        return clamped, int(np.count_nonzero(raw != clamped))

    def to_servo_units(self, q_target: np.ndarray) -> np.ndarray:
        """Radians -> firmware ticks. Identity calibration until S.9 lands.

        The STS3250 encoder is 12-bit over 360 deg, so 4096/(2*pi) counts/rad,
        with a per-joint zero offset and sign from calibration.
        """
        q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
        counts_per_rad = 4096.0 / (2.0 * np.pi)
        if self.calibration is None:
            offsets = np.zeros(self.action_dim, dtype=np.float32)
            signs = np.ones(self.action_dim, dtype=np.float32)
            centre = 2048.0
        else:
            offsets = np.asarray(self.calibration["offsets_rad"], dtype=np.float32)
            signs = np.asarray(self.calibration["signs"], dtype=np.float32)
            centre = float(self.calibration.get("centre_counts", 2048.0))
        ticks = centre + signs * (q_target - offsets) * counts_per_rad
        return np.rint(ticks).astype(np.int32)
