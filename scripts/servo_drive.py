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
        self.id = servo_id
        self.torque_limit = min(torque_limit, TORQUE_LIMIT_MAX)
        self.samples: list[dict] = []
        self._fault_run = 0
        self._armed = False

    # -- transport ---------------------------------------------------------

    def _txrx(self, packet: bytes, expect_params: int) -> bytes | None:
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        window = self.ser.read(len(packet) + expect_params + 6)
        if not window:
            return None
        if window.startswith(packet):          # half-duplex echo
            window = window[len(packet):]
        start = window.find(HEADER)
        if start < 0 or len(window) - start < 6:
            return None
        frame = window[start:]
        total = frame[3] + 4
        if len(frame) < total:
            return None
        try:
            _, _, params = decode_status(frame[:total])
        except ValueError:
            return None
        return params

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
        elif args.staircase:
            outcome = run_staircase(driver, args.level_seconds, args.interval)
        elif args.sweep:
            outcome = run_sweep(driver, args.amplitude, args.period,
                                args.seconds, args.interval)
        else:
            ap.error("pick one of --hold / --sweep / --goto / --staircase / --release")
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
