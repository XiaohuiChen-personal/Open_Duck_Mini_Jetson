"""Control-loop skeleton with pluggable I/O — Task S.4.

**No hardware imports at module scope**, so the whole package imports on a
laptop. Real sensors and servos arrive as objects satisfying the two protocols
below; `ReplaySensorSource` feeds a recorded S.2 trace instead, which is what
makes the loop testable before a robot exists.

    sensors -> ObsBuilder -> policy.infer -> ActionDecoder -> actuators

BUS ARCHITECTURE CONSTRAINT (measured 2026-08-21, not in task_plan_v2.md)

`addr 8 response_level = 1` on the physical servo, so an individual WRITE
generates a status reply — a full round trip each. With 14 servos that is ~14x
the cost, and `addr 8` is EEPROM with `addr 55` locked, so it cannot be
configured away. **A conforming ActuatorSink MUST use SYNC_WRITE** (broadcast
0xFE, which never replies) rather than per-servo writes, or the 20 ms budget is
gone before inference starts. Same argument for SYNC_READ on the sensor side.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class SensorSource(Protocol):
    """Whatever supplies a control step's measurements."""

    def read(self) -> dict:
        """Return gyro_b, gravity_b, q_meas, qd_meas (contract joint order)."""
        ...


class ActuatorSink(Protocol):
    """Whatever consumes joint targets. MUST use SYNC_WRITE — see module docs."""

    def write(self, q_target: np.ndarray) -> None: ...


class ReplaySensorSource:
    """Feed a recorded S.2 trace, one control step per `read()`.

    This is what lets the loop be exercised end to end with no robot. It
    deliberately raises at the end rather than looping, so a test cannot
    silently run past the data it is validating against.
    """

    def __init__(self, trace: dict) -> None:
        self.gyro = np.asarray(trace["root_ang_vel_b"], dtype=np.float32)
        self.gravity = np.asarray(trace["projected_gravity_b"], dtype=np.float32)
        self.q = np.asarray(trace["joint_pos"], dtype=np.float32)
        self.qd = np.asarray(trace["joint_vel"], dtype=np.float32)
        self.command = np.asarray(trace["command"], dtype=np.float32)
        self.n = self.q.shape[0]
        self.i = 0

    def read(self) -> dict:
        if self.i >= self.n:
            raise StopIteration(f"trace exhausted after {self.n} steps")
        out = {"gyro_b": self.gyro[self.i], "gravity_b": self.gravity[self.i],
               "q_meas": self.q[self.i], "qd_meas": self.qd[self.i],
               "command": self.command[self.i]}
        self.i += 1
        return out


class NullActuatorSink:
    """Records targets instead of moving anything."""

    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []

    def write(self, q_target: np.ndarray) -> None:
        self.targets.append(np.asarray(q_target, dtype=np.float32).copy())


class ControlLoop:
    """One policy step per `tick()`; `run()` drives it to completion."""

    def __init__(self, contract, obs_builder, decoder, policy, gait,
                 sensors, actuators, clamp_command=None) -> None:
        self.contract = contract
        self.obs_builder = obs_builder
        self.decoder = decoder
        self.policy = policy
        self.gait = gait
        self.sensors = sensors
        self.actuators = actuators
        self.clamp_command = clamp_command
        self.dt = float(contract["control_dt_s"])
        self.action_dim = int(contract["action_dim"])

        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.stats = {"steps": 0, "clamped_joints": 0, "clamped_steps": 0,
                      "max_step_s": 0.0}

    def reset(self) -> None:
        """Start (or restart) the policy. This is the ONLY gait-phase reset."""
        self.gait.reset()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

    def tick(self) -> dict:
        t0 = time.perf_counter()
        s = self.sensors.read()
        command = s["command"]
        if self.clamp_command is not None:
            command = np.asarray(self.clamp_command(*command), dtype=np.float32)

        obs = self.obs_builder.build(
            gyro_b=s["gyro_b"], gravity_b=s["gravity_b"],
            q_meas=s["q_meas"], qd_meas=s["qd_meas"], command=command,
            last_action=self.last_action, gait_cos_sin=self.gait.step())

        action = self.policy.infer(obs)
        q_target, n_clamped = self.decoder.decode(action)
        self.actuators.write(q_target)

        # last_action is the RAW output, not the target. Order matters: it is
        # consumed by the NEXT observation.
        self.last_action = action

        elapsed = time.perf_counter() - t0
        self.stats["steps"] += 1
        self.stats["clamped_joints"] += n_clamped
        self.stats["clamped_steps"] += int(n_clamped > 0)
        self.stats["max_step_s"] = max(self.stats["max_step_s"], elapsed)
        return {"obs": obs, "action": action, "q_target": q_target,
                "n_clamped": n_clamped, "elapsed_s": elapsed}

    def run(self, max_steps: int | None = None) -> list[dict]:
        self.reset()
        out = []
        while max_steps is None or len(out) < max_steps:
            try:
                out.append(self.tick())
            except StopIteration:
                break
        return out

    def report(self) -> str:
        s = self.stats
        pct = 100.0 * s["clamped_steps"] / s["steps"] if s["steps"] else 0.0
        return (f"steps={s['steps']}  clamped_steps={s['clamped_steps']} ({pct:.2f} %)"
                f"  clamped_joints={s['clamped_joints']}"
                f"  max_step={s['max_step_s'] * 1000:.2f} ms"
                f"  budget={self.dt * 1000:.0f} ms")
