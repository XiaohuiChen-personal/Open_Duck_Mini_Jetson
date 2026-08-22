"""Safety layer — Task S.10.

The robot is 2.729 kg with 14 servos that each pull amps and reach 80 C. It WILL
fall. This decides what happens when it does.

**Every threshold is read from the contract. There are no literals here** — that
is the point, because the thresholds moved once already (the firmware cutout is
80 C, not the 70 C every document assumed until it was read off the servo).

VERDICTS, in increasing severity

    OK          proceed
    DERATE      proceed, but scale the velocity command down
    HOLD        repeat the last joint target; do not trust this step's sensors
    TORQUE_OFF  ramp to the measured pose, disable torque, LATCH

TORQUE_OFF LATCHES. Recovery needs an explicit operator `rearm()`, never
automatic — a monitor that un-trips itself is a monitor that trips repeatedly
while the robot destroys itself.

**On a fall, torque OFF; do not hold the pose.** A fallen robot holding a
standing pose stalls its servos against the floor, which trips the stall and
overcurrent protections and heats the coils. The firmware then drops torque
anyway — silently, unrepeatably, and with no log line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OK = "OK"
DERATE = "DERATE"
HOLD = "HOLD"
TORQUE_OFF = "TORQUE_OFF"

_SEVERITY = {OK: 0, DERATE: 1, HOLD: 2, TORQUE_OFF: 3}

# The simulator's own bad_orientation termination (env_cfg.py:592, :701).
# Beyond it the policy is outside every state it was ever trained in, so its
# output is not merely degraded, it is meaningless.
TILT_LIMIT_RAD = math.radians(60.0)

# root_height_below_minimum (env_cfg.py:595, minimum_height = 0.09 m) CANNOT be
# evaluated on hardware -- nothing measures trunk height. This is the substitute:
# both feet off the ground AND tilted past ~37 deg, held for 5 steps.
FALL_PROXY_GRAVITY_Z = 0.8
FALL_PROXY_STEPS = 5

DERATE_CURRENT_FRACTION = 0.7
DERATE_COMMAND_SCALE = 0.5
HOLD_MAX_STEPS = 3
OVERRUN_FACTOR = 2.0
OVERRUN_MAX_CONSECUTIVE = 3


@dataclass
class SafetyState:
    verdict: str = OK
    reasons: list[str] = field(default_factory=list)
    command_scale: float = 1.0
    latched: bool = False
    clamped_commands: int = 0
    trip_history: list[tuple[int, str]] = field(default_factory=list)


class SafetyMonitor:
    """Called once per control step with the current sensor bundle."""

    def __init__(self, contract: dict, calibration: dict | None = None) -> None:
        self.contract = contract
        self.calibration = calibration
        self.joint_order = list(contract["joint_order"])
        self.dt = float(contract["control_dt_s"])

        self.temp_warn = float(contract["servo_temp_warn_c"])
        self.temp_stop = float(contract["servo_temp_stop_c"])
        self.firmware_cutout = float(contract["servo_firmware_temp_cutout_c"])
        self.current_limit = float(contract["servo_current_limit_a"])
        # Deliberately HALF the firmware's window, so the runtime acts first: if
        # the firmware wins, the leg goes limp with no warning and no log line.
        self.current_window_s = float(contract["servo_current_limit_window_s"]) / 2.0
        self.min_voltage = float(contract.get("servo_min_input_voltage_v", 0.0))
        self.cmd_clamp = contract["cmd_clamp_deployment"]

        self._steps = 0
        self._fall_steps = 0
        self._hold_steps = 0
        self._overruns = 0
        self._over_current_s = np.zeros(len(self.joint_order))
        self._over_derate_s = np.zeros(len(self.joint_order))
        self.state = SafetyState()

    # -- command clamping -------------------------------------------------

    def clamp_command(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        """Clip to the DEPLOYMENT hull and count every clip.

        vx/vy clamp at the trained hull; **wz clamps to +/-0.3, deliberately
        INSIDE the trained +/-0.5** (DEPLOY-5 confirms 0.3 is inside). Widening
        it to the full hull is a decision to record, not a default.
        """
        out = []
        for name, v in (("vx", vx), ("vy", vy), ("wz", wz)):
            lo, hi = self.cmd_clamp[name]
            c = min(max(float(v), lo), hi)
            if c != float(v):
                self.state.clamped_commands += 1
            out.append(c)
        return tuple(out)

    # -- the per-step check -----------------------------------------------

    def check(self, gravity_b, temps_c=None, currents_a=None, voltage_v=None,
              foot_contacts=None, step_elapsed_s=None, obs=None) -> SafetyState:
        """Evaluate one control step. Returns the (latching) SafetyState."""
        self._steps += 1
        verdicts: list[tuple[str, str]] = []

        if self.state.latched:
            self.state.verdict = TORQUE_OFF
            return self.state

        # --- sensor sanity first: everything below trusts these numbers ---
        bad_sensor = False
        if gravity_b is None or not np.all(np.isfinite(np.asarray(gravity_b, float))):
            bad_sensor = True
        if obs is not None and not np.all(np.isfinite(np.asarray(obs, float))):
            bad_sensor = True
        if bad_sensor:
            self._hold_steps += 1
            if self._hold_steps > HOLD_MAX_STEPS:
                verdicts.append((TORQUE_OFF, f"sensor_fault_{self._hold_steps}_steps"))
            else:
                verdicts.append((HOLD, "sensor_fault"))
        else:
            self._hold_steps = 0

        if not bad_sensor:
            g = np.asarray(gravity_b, dtype=float)
            n = np.linalg.norm(g)
            if n > 1e-6:
                # angle between measured gravity and straight down
                cos_t = float(np.clip(np.dot(g / n, (0.0, 0.0, -1.0)), -1.0, 1.0))
                tilt = math.acos(cos_t)
                if tilt > TILT_LIMIT_RAD:
                    verdicts.append((TORQUE_OFF, f"tilt_{math.degrees(tilt):.0f}deg"))

                # fall proxy: the height termination we cannot measure
                airborne = (foot_contacts is not None
                            and not any(bool(c) for c in foot_contacts))
                if airborne and abs(g[2] / n) < FALL_PROXY_GRAVITY_Z:
                    self._fall_steps += 1
                    if self._fall_steps >= FALL_PROXY_STEPS:
                        verdicts.append((TORQUE_OFF, "fall_proxy"))
                else:
                    self._fall_steps = 0

        # --- per-servo temperature ---
        if temps_c is not None:
            for i, t in enumerate(np.asarray(temps_c, dtype=float)):
                if not math.isfinite(t):
                    continue
                joint = self.joint_order[i] if i < len(self.joint_order) else f"j{i}"
                if t >= self.temp_stop:
                    verdicts.append((TORQUE_OFF, f"temp_stop_{joint}_{t:.0f}C"))
                elif t >= self.temp_warn:
                    verdicts.append((DERATE, f"temp_warn_{joint}_{t:.0f}C"))

        # --- per-servo current, duration-based like the firmware ---
        if currents_a is not None:
            cur = np.asarray(currents_a, dtype=float)
            over = cur >= self.current_limit
            derate = cur >= DERATE_CURRENT_FRACTION * self.current_limit
            self._over_current_s = np.where(over, self._over_current_s + self.dt, 0.0)
            self._over_derate_s = np.where(derate, self._over_derate_s + self.dt, 0.0)
            for i in np.where(self._over_current_s > self.current_window_s)[0]:
                joint = self.joint_order[i] if i < len(self.joint_order) else f"j{i}"
                verdicts.append((TORQUE_OFF, f"over_current_{joint}_{cur[i]:.2f}A"))
            for i in np.where(self._over_derate_s > self.current_window_s)[0]:
                joint = self.joint_order[i] if i < len(self.joint_order) else f"j{i}"
                verdicts.append((DERATE, f"current_derate_{joint}_{cur[i]:.2f}A"))

        # --- pack under-voltage: MEASURED addr15 = 6.0 V, not the datasheet 4 V
        if voltage_v is not None and math.isfinite(float(voltage_v)):
            if float(voltage_v) < self.min_voltage:
                verdicts.append((TORQUE_OFF, f"under_voltage_{voltage_v:.1f}V"))

        # --- loop overrun ---
        if step_elapsed_s is not None:
            if float(step_elapsed_s) > OVERRUN_FACTOR * self.dt:
                self._overruns += 1
                if self._overruns >= OVERRUN_MAX_CONSECUTIVE:
                    verdicts.append((TORQUE_OFF, f"overrun_x{self._overruns}"))
                else:
                    verdicts.append((HOLD, f"overrun_{step_elapsed_s * 1000:.1f}ms"))
            else:
                self._overruns = 0

        # --- resolve: the most severe verdict wins ---
        if verdicts:
            worst = max(verdicts, key=lambda v: _SEVERITY[v[0]])
            self.state.verdict = worst[0]
            self.state.reasons = [r for _, r in verdicts]
        else:
            self.state.verdict = OK
            self.state.reasons = []

        self.state.command_scale = (DERATE_COMMAND_SCALE
                                    if self.state.verdict == DERATE else 1.0)
        if self.state.verdict == TORQUE_OFF:
            self.state.latched = True
            self.state.trip_history.append((self._steps, ";".join(self.state.reasons)))
        return self.state

    # -- recovery ---------------------------------------------------------

    def rearm(self) -> None:
        """Explicit operator action. Never call this automatically."""
        self.state = SafetyState(clamped_commands=self.state.clamped_commands,
                                 trip_history=list(self.state.trip_history))
        self._fall_steps = self._hold_steps = self._overruns = 0
        self._over_current_s[:] = 0.0
        self._over_derate_s[:] = 0.0


def torque_off_sequence(q_measured, q_last_target, steps: int = 2):
    """Targets that ramp to the MEASURED pose before torque is disabled.

    Commanding the last target while the servo is already elsewhere makes it
    fight; ramping to where it actually is means the disable is a release rather
    than a drop.
    """
    q_measured = np.asarray(q_measured, dtype=np.float32)
    q_last_target = np.asarray(q_last_target, dtype=np.float32)
    return [q_last_target + (q_measured - q_last_target) * ((i + 1) / steps)
            for i in range(steps)]
