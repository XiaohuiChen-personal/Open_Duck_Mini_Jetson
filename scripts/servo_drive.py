#!/usr/bin/env python3
"""Write-capable bench driver for ONE Feetech STS3250. Deliberately narrow.

This is a SEPARATE file from `servo_probe.py` on purpose. The probe's
"no write path exists" guarantee is tested and worth keeping; this tool is where
the risk lives, so the risk is concentrated and bounded here.

THE SAFETY MODEL
----------------
There is exactly one servo. Every rule below exists because breaking it is
either unrecoverable or can injure someone.

1. **SRAM-only allowlist.** Only addresses 40, 41, 42, 46, 48 can be written.
   EEPROM (0-39) is refused outright: those writes persist, and a bad ID or baud
   write loses the servo permanently.
2. **Writing 128 to addr 40 is refused unconditionally.** Datasheet 7-14: that
   is the centering command and it RE-ZEROES the servo midpoint. It would
   silently invalidate every joint zero downstream.
3. **`torque_limit` is a DUTY clamp, not a torque cap.** CORRECTED 2026-08-21:
   addr48 limits PWM duty in per-mille, and at low speed back-EMF is negligible,
   so 20 % duty into a 1.2 ohm winding still draws 0.20*11.1/1.2 = 1.85 A =
   **2.0 N.m** -- not the 0.98 N.m that "20 % of stall" implies. The real
   backstops are the driver's 4.2 A clamp and the PSU current limit. Keep the
   default low anyway (it does bound duty, and therefore speed and heating), but
   **do not rely on it to bound torque.** The servo has no mechanical end stop.
4. **Torque is disabled on every exit path** — normal return, exception,
   Ctrl-C, or abort. If this process dies, the servo goes limp rather than
   holding or fighting.
5. **No lurch on enable.** Goal position is set to the *present* position before
   torque is enabled, so enabling never commands a move.
6. **Telemetry is logged with every step**, including addr65 status and addr60
   load, because a latched protection trip otherwise looks exactly like success:
   torque goes to zero, current collapses, temperature plateaus.

BEFORE RUNNING
--------------
Raise the PSU current limit to **1.0 A**. Datasheet 5-3 gives no-load running
current as 280 mA; at a 0.30 A limit the supply will drop into CC on the first
move and brown the servo out mid-write.

Usage
-----
    python3 scripts/servo_drive.py --hold --seconds 20
    python3 scripts/servo_drive.py --sweep --amplitude 200 --period 4 --seconds 60
    python3 scripts/servo_drive.py --release
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from servo_probe import (  # noqa: E402  - local sibling module
    BAUD_TABLE, HEADER, INST_PING, INST_READ, checksum, decode_status,
    decode_load, decode_status_bits, to_int, to_signed_magnitude,
)

INST_WRITE = 0x03

# ---------------------------------------------------------------- safety ---

EEPROM_LAST_ADDR = 39

# addr -> (width, human name). SRAM only. Nothing here persists across a
# power cycle, so every mistake is recoverable by switching the PSU off.
WRITABLE: dict[int, tuple[int, str]] = {
    40: (1, "torque_enable"),
    41: (1, "acceleration"),
    42: (2, "goal_position"),
    46: (2, "goal_speed"),
    48: (2, "torque_limit"),
}

CENTERING_MAGIC = 128          # addr40 <- 128 re-zeroes the midpoint (7-14)
POSITION_MAX = 4095            # 12-bit encoder
TORQUE_LIMIT_MAX = 1000        # full scale
DEFAULT_TORQUE_LIMIT = 200     # 20 % DUTY -- ~2.0 N.m at stall, NOT 0.98
DEFAULT_GOAL_SPEED = 300       # conservative; units unverified for this model

# Abort thresholds, in real units. addr13 on this unit is 80 C.
ABORT_TEMP_C = 60
ABORT_STATUS_NONZERO = True

# A single sample must never abort a run. Measured 2026-08-21: one sample in a
# 198-sample sweep read 49 C while every neighbour read 33-34 C. That is a comms
# glitch, and a 1-sample abort rule turns it into a spurious stop -- or, worse,
# trains the operator to ignore aborts.
ABORT_CONSECUTIVE = 3

# Characterisation staircase: (amplitude counts, period s, goal_speed).
# Acceleration demand scales as amplitude/period^2, so this spans roughly two
# decades of mechanical duty while staying inside the torque cap. The point is
# to produce a CURVE -- current and temperature against duty -- so that loads we
# cannot apply on this bench can be extrapolated rather than guessed.
STAIRCASE = [
    (200, 4.0, 300),
    (400, 3.0, 500),
    (600, 2.0, 1000),
    (800, 1.5, 1500),
    (1000, 1.0, 2000),
    (1200, 0.8, 3000),
    (1500, 0.6, 4000),
]


class UnsafeWrite(Exception):
    """Raised instead of transmitting anything the safety model forbids."""


def build_write_packet(servo_id: int, address: int, value: int, width: int) -> bytes:
    """Build a WRITE. Every refusal below is a rule from the module docstring."""
    if not 0 <= servo_id <= 0xFD:
        raise UnsafeWrite(f"servo id out of range (0xFE broadcast refused): {servo_id}")
    if address <= EEPROM_LAST_ADDR:
        raise UnsafeWrite(
            f"addr {address} is EEPROM (<= {EEPROM_LAST_ADDR}); writes persist and "
            "can brick ID/baud. Refused."
        )
    if address not in WRITABLE:
        raise UnsafeWrite(f"addr {address} is not on the SRAM allowlist {sorted(WRITABLE)}")
    expected_width, name = WRITABLE[address]
    if width != expected_width:
        raise UnsafeWrite(f"addr {address} ({name}) is {expected_width}-byte, got {width}")
    if address == 40 and value == CENTERING_MAGIC:
        raise UnsafeWrite(
            "addr40 <- 128 is the CENTERING command (datasheet 7-14). It re-zeroes "
            "the servo midpoint and would invalidate every joint zero. Refused."
        )
    if value < 0 or value >= (1 << (8 * width)):
        raise UnsafeWrite(f"value {value} does not fit {width} byte(s)")
    if address == 42 and value > POSITION_MAX:
        raise UnsafeWrite(f"goal_position {value} exceeds {POSITION_MAX}")
    if address == 48 and value > TORQUE_LIMIT_MAX:
        raise UnsafeWrite(f"torque_limit {value} exceeds {TORQUE_LIMIT_MAX}")

    params = bytes([address]) + value.to_bytes(width, "little")
    body = bytes([servo_id, len(params) + 2, INST_WRITE]) + params
    return HEADER + body + bytes([checksum(body)])


# ------------------------------------------------------------------ driver ---


class Driver:
    def __init__(self, port: str, baud: int, servo_id: int, torque_limit: int) -> None:
        import serial

        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.reply_timeout = 0.010      # per-transaction ceiling, not a fixed cost
        self.id = servo_id
        self.torque_limit = min(torque_limit, TORQUE_LIMIT_MAX)
        self.samples: list[dict] = []
        self._fault_run = 0
        self._armed = False

    # -- transport ---------------------------------------------------------

    def _txrx(self, packet: bytes, expect_params: int) -> bytes | None:
        """Write, then read only as long as bytes are actually still arriving.

        pyserial's read(n) blocks until it has n bytes OR the timeout expires.
        Requesting a generous window therefore costs the FULL timeout on every
        transaction: with timeout=0.05 and two transactions per control step the
        loop pins at 10 Hz, which aliases every period below ~1 s. Measured
        2026-08-22. Poll in_waiting instead and return the instant a complete
        frame has landed.
        """
        self.ser.reset_input_buffer()
        self.ser.write(packet)

        want = expect_params + 6
        buf = b""
        deadline = time.perf_counter() + self.reply_timeout
        while time.perf_counter() < deadline:
            pending = self.ser.in_waiting
            if pending:
                buf += self.ser.read(pending)
                body = buf[len(packet):] if buf.startswith(packet) else buf
                start = body.find(HEADER)
                if start >= 0 and len(body) - start >= want:
                    break
            else:
                time.sleep(0.0002)
        if not buf:
            return None
        if buf.startswith(packet):          # half-duplex echo
            buf = buf[len(packet):]
        return self._parse(buf, expect_params)

    def _parse(self, window: bytes, expect_params: int) -> bytes | None:
        """Find a frame that is OURS, not merely one that checksums.

        0xFF 0xFF occurs inside payload data, so syncing on the first header and
        trusting the checksum admits false locks. Measured 2026-08-22: at ~700 Hz
        that corrupted 0.1 % of samples, one of which read 150 C between
        neighbours of 35 C and aborted a 4-minute sweep. Require the ID and the
        length to match what we asked for, and keep searching if they do not.
        """
        want_len = expect_params + 2
        start = 0
        while True:
            start = window.find(HEADER, start)
            if start < 0:
                return None
            frame = window[start:]
            if len(frame) < expect_params + 6:
                return None
            if frame[2] == self.id and frame[3] == want_len:
                try:
                    _, _, params = decode_status(frame[:want_len + 4])
                except ValueError:
                    start += 2
                    continue
                if len(params) == expect_params:
                    return params
            start += 2

    def read(self, address: int, width: int) -> int | None:
        body = bytes([self.id, 4, INST_READ, address, width])
        packet = HEADER + body + bytes([checksum(body)])
        got = self._txrx(packet, width)
        return to_int(got) if got is not None and len(got) == width else None

    def write(self, address: int, value: int) -> bool:
        width, _ = WRITABLE[address]
        packet = build_write_packet(self.id, address, value, width)
        return self._txrx(packet, 0) is not None

    # -- telemetry ---------------------------------------------------------

    def telemetry_fast(self) -> dict | None:
        """All of addr 56..70 in ONE round trip.

        The per-register path costs six serial transactions per sample, which
        caps the loop near 30-60 Hz and cannot resolve a 0.45 s period. It also
        smears a sample across ~10 ms, so position and current describe
        different instants -- fatal when fitting inertia against acceleration.
        """
        raw = self.read_block(56, 15)
        if raw is None:
            return None
        return {
            "position": to_int(raw[0:2]),
            "speed": to_signed_magnitude(to_int(raw[2:4])),
            "load": decode_load(to_int(raw[4:6])),
            "voltage": raw[6] / 10,
            "temperature": raw[7],
            "status": raw[9],
            "moving": raw[10],
            "current_raw": to_signed_magnitude(to_int(raw[13:15])),
        }

    def read_block(self, address: int, width: int) -> bytes | None:
        body = bytes([self.id, 4, INST_READ, address, width])
        packet = HEADER + body + bytes([checksum(body)])
        got = self._txrx(packet, width)
        return got if got is not None and len(got) == width else None

    def telemetry(self) -> dict:
        raw_load = self.read(60, 2)
        raw_curr = self.read(69, 2)
        return {
            "position": self.read(56, 2),
            "load": decode_load(raw_load) if raw_load is not None else None,
            "voltage": (lambda v: v / 10 if v is not None else None)(self.read(62, 1)),
            "temperature": self.read(63, 1),
            "current_raw": to_signed_magnitude(raw_curr) if raw_curr is not None else None,
            "status": self.read(65, 1),
        }

    @staticmethod
    def _abort_reason(t: dict) -> str | None:
        """Per-sample fault test. Debouncing is the caller's job."""
        if ABORT_STATUS_NONZERO and t.get("status"):
            return f"status byte {t['status']:#04x} {decode_status_bits(t['status'])}"
        if t.get("temperature") is not None and t["temperature"] >= ABORT_TEMP_C:
            return f"temperature {t['temperature']} C >= {ABORT_TEMP_C} C"
        if t.get("voltage") is not None and t["voltage"] < 9.0:
            return f"supply sagged to {t['voltage']} V — PSU is in CC"
        return None

    def check_abort(self, t: dict) -> str | None:
        """Abort only after ABORT_CONSECUTIVE faulty samples in a row."""
        reason = self._abort_reason(t)
        if reason is None:
            self._fault_run = 0
            return None
        self._fault_run = getattr(self, "_fault_run", 0) + 1
        if self._fault_run >= ABORT_CONSECUTIVE:
            return f"{reason} (for {self._fault_run} consecutive samples)"
        return None

    # -- lifecycle ---------------------------------------------------------

    def arm(self) -> int:
        """Enable torque WITHOUT commanding a move. Returns the held position."""
        here = self.read(56, 2)
        if here is None:
            raise RuntimeError("cannot read present position; refusing to arm")
        self.write(48, self.torque_limit)
        self.write(46, DEFAULT_GOAL_SPEED)
        self.write(42, here)          # goal := present, so enabling cannot lurch
        self.write(40, 1)
        self._armed = True
        return here

    def release(self) -> None:
        try:
            self.write(40, 0)
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        self._armed = False

    def close(self) -> None:
        self.release()
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- commands ---


def _log(driver: Driver, t: dict, elapsed: float, goal: int | None) -> None:
    row = {"t": round(elapsed, 2), "goal": goal, **t}
    driver.samples.append(row)
    print(
        f"{elapsed:6.1f}  goal={goal if goal is not None else '--':>5} "
        f"pos={row['position'] if row['position'] is not None else '--':>5} "
        f"load={row['load'] if row['load'] is not None else '--':>5} "
        f"{row['voltage'] if row['voltage'] is not None else '--':>5}V "
        f"{row['temperature'] if row['temperature'] is not None else '--':>3}C "
        f"curr={row['current_raw'] if row['current_raw'] is not None else '--':>4} "
        f"{decode_status_bits(row['status']) if row['status'] else ''}",
        flush=True,
    )


def run_hold(driver: Driver, seconds: float, interval: float) -> str:
    here = driver.arm()
    print(f"# armed, holding position {here}, torque_limit={driver.torque_limit}/1000")
    start = time.time()
    while time.time() - start < seconds:
        t = driver.telemetry()
        _log(driver, t, time.time() - start, here)
        reason = driver.check_abort(t)
        if reason:
            return f"ABORT: {reason}"
        time.sleep(interval)
    return "completed"


def run_sweep(driver: Driver, amplitude: int, period: float,
              seconds: float, interval: float) -> str:
    import math

    centre = driver.arm()
    lo, hi = max(0, centre - amplitude), min(POSITION_MAX, centre + amplitude)
    print(f"# armed at {centre}; sweeping {lo}..{hi} (+/-{amplitude} counts, "
          f"{amplitude * 360 / 4096:.1f} deg) period {period}s")
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= seconds:
            return "completed"
        phase_ = 2 * math.pi * elapsed / period
        goal = int(centre + amplitude * math.sin(phase_))
        goal = max(lo, min(hi, goal))
        driver.write(42, goal)
        t = driver.telemetry()
        _log(driver, t, elapsed, goal)
        reason = driver.check_abort(t)
        if reason:
            return f"ABORT: {reason}"
        time.sleep(interval)


def run_goto(driver: Driver, target: int, speed: int, timeout: float,
             interval: float) -> str:
    """Move to an absolute position slowly, then report the settled error."""
    here = driver.arm()
    driver.write(46, speed)
    driver.write(42, max(0, min(POSITION_MAX, target)))
    print(f"# armed at {here}; moving to {target} at goal_speed={speed}")
    start = time.time()
    while time.time() - start < timeout:
        t = driver.telemetry()
        elapsed = time.time() - start
        _log(driver, t, elapsed, target)
        reason = driver.check_abort(t)
        if reason:
            return f"ABORT: {reason}"
        if t["position"] is not None and abs(t["position"] - target) <= 8:
            return "completed"
        time.sleep(interval)
    return "completed (timeout, may not have settled)"


def run_staircase(driver: Driver, seconds_per_level: float, interval: float) -> str:
    """Step through STAIRCASE, logging every sample tagged with its level."""
    import math

    centre = driver.arm()
    print(f"# armed at {centre}; {len(STAIRCASE)} levels x {seconds_per_level}s")
    for index, (amplitude, period, speed) in enumerate(STAIRCASE, start=1):
        lo = max(0, centre - amplitude)
        hi = min(POSITION_MAX, centre + amplitude)
        if hi - lo < amplitude:
            print(f"# level {index}: SKIPPED — +/-{amplitude} does not fit at {centre}")
            continue
        driver.write(46, speed)
        print(f"# --- level {index}: +/-{amplitude} counts "
              f"({amplitude * 360 / 4096:.1f} deg), period {period}s, speed {speed}, "
              f"accel proxy {amplitude / period ** 2:.0f}")
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= seconds_per_level:
                break
            goal = int(centre + amplitude * math.sin(2 * math.pi * elapsed / period))
            goal = max(lo, min(hi, goal))
            driver.write(42, goal)
            t = driver.telemetry()
            t["level"] = index
            t["amplitude"] = amplitude
            t["period"] = period
            t["goal_speed"] = speed
            _log(driver, t, elapsed, goal)
            reason = driver.check_abort(t)
            if reason:
                return f"ABORT at level {index}: {reason}"
            time.sleep(interval)
    return "completed"


# Amplitudes (counts) x periods (s) for the inertia identification. Chosen so
# that J*A*w^2 and b*A*w separate: w spans 3.1..14.0 rad/s, so the inertial term
# (w^2) grows ~20x across the sweep while the viscous term (w) grows ~4.5x.
FREQ_AMPLITUDES = (400, 700, 1000)         # 35, 61, 88 degrees
# Sized 2026-08-22 after a 10-20 deg sweep returned I_rms of 2.8-4.3
# counts -- addr69's 12.258 mA quantisation floor. Inertial torque
# scales with amplitude, so 4x the swing lifts the signal clear of it.
FREQ_PERIODS = (2.0, 1.2, 0.8, 0.6, 0.45)  # seconds
FREQ_CYCLES = 16
# Commanding goal_position faster than this makes the servo's internal profile
# generator restart before it can accelerate: measured 2026-08-22, writing at
# ~700 Hz pinned every condition to ~405 counts/s regardless of amplitude, while
# the servo's actual capability is ~4000 counts/s. 50 Hz is also the robot's
# real control rate, so this characterises the servo as it will be driven.
FREQ_COMMAND_HZ = 50.0


def run_freq_sweep(driver: Driver, out_rows: list, settle_s: float = 0.4) -> str:
    """Amplitude x period sweep for inertia identification.

    MUST run at torque_limit = 1000. Every earlier bench log used 200, which is
    a DUTY clamp -- the servo was speed-limited, not torque-limited, so none of
    that data can identify inertia. See known_issues.md PLANT-11.
    """
    import math

    if driver.torque_limit < TORQUE_LIMIT_MAX:
        return (f"REFUSED: torque_limit is {driver.torque_limit}, must be "
                f"{TORQUE_LIMIT_MAX}. A duty clamp makes the fit meaningless.")

    centre = driver.arm()
    driver.write(41, 0)                     # acceleration limit off
    driver.write(46, 0)                     # goal_speed 0 = unlimited
    print(f"# armed at {centre}, torque_limit={driver.torque_limit}")

    for amplitude in FREQ_AMPLITUDES:
        if centre - amplitude < 0 or centre + amplitude > POSITION_MAX:
            print(f"# amplitude {amplitude}: SKIPPED, does not fit at {centre}")
            continue
        for period in FREQ_PERIODS:
            duration = FREQ_CYCLES * period
            print(f"# --- A={amplitude} counts ({amplitude*360/4096:.1f} deg), "
                  f"T={period}s, w={2*math.pi/period:.2f} rad/s, {duration:.1f}s")
            start = time.time()
            n, first, next_cmd, goal = 0, None, 0.0, centre
            while True:
                elapsed = time.time() - start
                if elapsed >= duration:
                    break
                if elapsed >= next_cmd:      # command at FREQ_COMMAND_HZ...
                    goal = int(centre + amplitude
                               * math.sin(2 * math.pi * elapsed / period))
                    driver.write(42, max(0, min(POSITION_MAX, goal)))
                    next_cmd = elapsed + 1.0 / FREQ_COMMAND_HZ
                t = driver.telemetry_fast()  # ...but sample as fast as we can
                if t is None:
                    continue
                if first is None:
                    first = elapsed
                n += 1
                row = {"amplitude": amplitude, "period": period, "t": elapsed,
                       "goal": goal, **t}
                out_rows.append(row)
                reason = driver.check_abort(t)
                if reason:
                    return f"ABORT at A={amplitude} T={period}: {reason}"
            rate = n / max(elapsed - (first or 0), 1e-6)
            samples_per_cycle = rate * period
            flag = "" if samples_per_cycle >= 10 else "  <-- TOO FEW, drop this period"
            print(f"#     {n} samples, {rate:.0f} Hz, "
                  f"{samples_per_cycle:.1f} per cycle{flag}")
            # let the servo settle between conditions so runs stay independent
            driver.write(42, centre)
            settle_until = time.time() + settle_s
            while time.time() < settle_until:
                driver.telemetry_fast()
    return "completed"


STEP_SIZES = (400, 800, 1200)     # counts
STEP_REPEATS = 4
STEP_WINDOW_S = 0.5


def run_step_test(driver: Driver, out_rows: list) -> str:
    """Step-response identification. The method the frequency sweep could not be.

    A smooth sinusoid keeps the position error small, so the proportional
    controller commands little duty: measured 2026-08-22, addr69 sat at ~3 counts
    (0.038 A) across every sweep condition, which is the quantisation floor. A
    large step maximises the error, the servo commits FULL duty (load = 1000),
    and current peaks at ~180 counts (2.2 A) -- 60x the signal.

    During the launch transient the servo is torque-saturated and starting from
    rest, so tau ~= Kt*I is constant and q(t) = q0 + 0.5*a*t^2. Fitting `a` from
    position and averaging I over the same window gives J = (Kt*I - tau_f)/a.
    """
    if driver.torque_limit < TORQUE_LIMIT_MAX:
        return (f"REFUSED: torque_limit is {driver.torque_limit}, must be "
                f"{TORQUE_LIMIT_MAX}. The step must saturate torque to be usable.")

    centre = driver.arm()
    driver.write(41, 0)
    driver.write(46, 0)
    trial = 0
    for size in STEP_SIZES:
        for rep in range(STEP_REPEATS):
            here = driver.read(56, 2)
            if here is None:
                return "ABORT: lost position feedback"
            direction = 1 if here < POSITION_MAX / 2 else -1
            target = here + direction * size
            if not 0 <= target <= POSITION_MAX:
                direction = -direction
                target = here + direction * size
            driver.write(42, here)
            driver.write(40, 1)
            settle = time.time() + 0.35
            while time.time() < settle:
                driver.telemetry_fast()

            trial += 1
            print(f"# --- step {trial}: {size} counts, {here} -> {target}")
            driver.write(42, target)
            t0 = time.perf_counter()
            n = 0
            while time.perf_counter() - t0 < STEP_WINDOW_S:
                t = driver.telemetry_fast()
                if t is None:
                    continue
                n += 1
                out_rows.append({"trial": trial, "size": size, "repeat": rep,
                                 "direction": direction, "start": here,
                                 "target": target,
                                 "t": time.perf_counter() - t0, **t})
                reason = driver.check_abort(t)
                if reason:
                    return f"ABORT on step {trial}: {reason}"
            print(f"#     {n} samples, {n / STEP_WINDOW_S:.0f} Hz")
            driver.write(40, 0)
            rest = time.time() + 0.5
            while time.time() < rest:
                driver.telemetry_fast()
    return "completed"


# -------------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--torque-limit", type=int, default=DEFAULT_TORQUE_LIMIT,
                    help=f"0..{TORQUE_LIMIT_MAX}; default {DEFAULT_TORQUE_LIMIT} (20%%)")
    ap.add_argument("--interval", type=float, default=0.2, help="log/step period, s")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hold", action="store_true", help="hold present position")
    ap.add_argument("--sweep", action="store_true", help="sinusoidal position sweep")
    ap.add_argument("--amplitude", type=int, default=200, help="sweep half-range, counts")
    ap.add_argument("--period", type=float, default=4.0, help="sweep period, s")
    ap.add_argument("--goto", type=int, metavar="POS",
                    help="move to an absolute position (0..4095) and settle")
    ap.add_argument("--goto-speed", type=int, default=200)
    ap.add_argument("--staircase", action="store_true",
                    help="step through the characterisation duty levels")
    ap.add_argument("--level-seconds", type=float, default=45.0)
    ap.add_argument("--freq-sweep", action="store_true",
                    help="amplitude x period sweep for inertia ID (PLANT-11)")
    ap.add_argument("--step-test", action="store_true",
                    help="step-response inertia identification (PLANT-11)")
    ap.add_argument("--release", action="store_true", help="torque off and exit")
    ap.add_argument("--out", help="write the sample log here as JSON")
    args = ap.parse_args(argv)

    if args.baud not in BAUD_TABLE.values():
        ap.error(f"baud {args.baud} not in {sorted(BAUD_TABLE.values())}")

    driver = Driver(args.port, args.baud, args.id, args.torque_limit)

    def _bail(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _bail)
    signal.signal(signal.SIGTERM, _bail)

    outcome = "not started"
    try:
        if args.release:
            driver.release()
            outcome = "released"
        elif args.hold:
            outcome = run_hold(driver, args.seconds, args.interval)
        elif args.goto is not None:
            outcome = run_goto(driver, args.goto, args.goto_speed,
                               args.seconds, args.interval)
        elif args.step_test:
            outcome = run_step_test(driver, driver.samples)
        elif args.freq_sweep:
            outcome = run_freq_sweep(driver, driver.samples)
        elif args.staircase:
            outcome = run_staircase(driver, args.level_seconds, args.interval)
        elif args.sweep:
            outcome = run_sweep(driver, args.amplitude, args.period,
                                args.seconds, args.interval)
        else:
            ap.error("pick one of --hold / --sweep / --goto / --staircase / --freq-sweep / --release")
    except KeyboardInterrupt:
        outcome = "interrupted"
    except UnsafeWrite as exc:
        outcome = f"REFUSED: {exc}"
    finally:
        driver.close()          # torque OFF on every path
        print(f"# torque disabled. outcome: {outcome}")

    if args.out and driver.samples:
        with open(args.out, "w") as fh:
            json.dump({"outcome": outcome, "torque_limit": driver.torque_limit,
                       "samples": driver.samples}, fh, indent=2)
        print(f"# wrote {args.out} ({len(driver.samples)} samples)")

    return 0 if outcome in ("completed", "released") else 1


if __name__ == "__main__":
    raise SystemExit(main())
