#!/usr/bin/env python3
"""Read-only Feetech STS bus probe for the S.8b bench test.

WHY THIS EXISTS INSTEAD OF pypot
--------------------------------
`experiments/v2/` uses `pypot.feetech.FeetechSTS3215IO`, but pypot is not
installed and needs a git fork (`apirrone/pypot@support_4B_registers`) — see
`docs/jetson-mod/task_plan_v2.md:5434`. The bench test should not be blocked on
that decision, and `pyserial` is already present.

READ-ONLY BY CONSTRUCTION
-------------------------
Only PING (0x01) and READ (0x02) are implemented. There is deliberately **no
write path in this file at all** — not a disabled one, not a guarded one. With
exactly one servo on hand, a stray WRITE to an EEPROM register (ID, baud, or
the address-40 centering command) is unrecoverable, so the capability is simply
absent. See `docs/jetson-mod/bench_test_servo.md` Stage B.

Usage
-----
    python3 scripts/servo_probe.py --scan                 # find ID + baud
    python3 scripts/servo_probe.py --dump --id 1          # full register dump
    python3 scripts/servo_probe.py --watch --id 1         # live telemetry
    python3 scripts/servo_probe.py --link-test --id 1     # 100x reply check
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# ---------------------------------------------------------------- protocol ---

HEADER = b"\xff\xff"
INST_PING = 0x01
INST_READ = 0x02

# STS/SMS baud codes (register 6). SCS uses a different table.
BAUD_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 76_800,
    6: 57_600,
    7: 38_400,
}
SCAN_BAUDS = list(BAUD_TABLE.values())

# Datasheet ST-3250-C001 A/0 line 7-11 says over-hot torque-off above 70 C.
# Real units ship addr13 = 80. The REGISTER is what the firmware enforces.
DATASHEET_OVER_HOT_C = 70

# addr69 present_current unit is NOT established for this model. Feetech STS
# docs commonly state 6.5 mA/LSB; a 10 mA/LSB reading is also plausible. The
# two differ by 1.54x, which propagates into every torque number, so Stage C
# must resolve it by comparing addr69 against the bench ammeter at a known
# load rather than assuming either.
CURRENT_LSB_CANDIDATES_MA = (6.5, 10.0)

# (address, width, name). Widths are bytes; 2-byte values are LITTLE-endian on
# STS/SMS (big-endian on SCS — getting this backwards silently garbles values).
REGISTERS: list[tuple[int, int, str]] = [
    # ---- EEPROM ----
    (0, 1, "firmware_major"),
    (1, 1, "firmware_minor"),
    (3, 1, "servo_major"),
    (4, 1, "servo_minor"),
    (5, 1, "id"),
    (6, 1, "baud_code"),
    (7, 1, "return_delay"),
    (8, 1, "response_level"),
    (9, 2, "min_angle_limit"),
    (11, 2, "max_angle_limit"),
    (13, 1, "max_temperature_limit"),
    (14, 1, "max_input_voltage"),
    (15, 1, "min_input_voltage"),
    (16, 2, "max_torque_limit"),
    (18, 1, "phase"),
    (19, 1, "unloading_condition"),
    (20, 1, "led_alarm_condition"),
    (21, 1, "pos_p"),
    (22, 1, "pos_d"),
    (23, 1, "pos_i"),
    (24, 2, "min_startup_force"),
    (26, 1, "cw_dead_band"),
    (27, 1, "ccw_dead_band"),
    (28, 2, "protection_current"),
    (30, 1, "angular_resolution"),
    (31, 2, "position_offset"),
    (33, 1, "operation_mode"),
    (34, 1, "protective_torque"),
    (35, 1, "protection_time"),
    (36, 1, "overload_torque"),
    (37, 1, "speed_p"),
    (38, 1, "overcurrent_time"),
    (39, 1, "speed_i"),
    # ---- SRAM ----
    (40, 1, "torque_enable"),
    (41, 1, "acceleration"),
    (42, 2, "goal_position"),
    (44, 2, "goal_time"),
    (46, 2, "goal_speed"),
    (48, 2, "torque_limit"),
    (55, 1, "lock"),
    (56, 2, "present_position"),
    (58, 2, "present_speed"),
    (60, 2, "present_load"),
    (62, 1, "present_voltage"),
    (63, 1, "present_temperature"),
    (64, 1, "async_write_flag"),
    (65, 1, "servo_status"),
    (66, 1, "moving"),
    (69, 2, "present_current"),
]

# Bits of register 65 (servo_status), per the STS error byte.
STATUS_BITS = [
    (0x01, "VOLTAGE"),
    (0x02, "SENSOR"),
    (0x04, "TEMPERATURE"),
    (0x08, "CURRENT"),
    (0x10, "ANGLE"),
    (0x20, "OVERLOAD"),
]


def checksum(payload: bytes) -> int:
    """Feetech checksum: bitwise NOT of the sum of everything after the header."""
    return (~sum(payload)) & 0xFF


def build_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    """Build an instruction packet. Only PING/READ are ever passed here."""
    if instruction not in (INST_PING, INST_READ):
        raise ValueError(
            f"this module is read-only; instruction 0x{instruction:02x} refused"
        )
    if not 0 <= servo_id <= 0xFD:
        raise ValueError(f"servo id out of range: {servo_id}")
    length = len(params) + 2
    body = bytes([servo_id, length, instruction]) + params
    return HEADER + body + bytes([checksum(body)])


def decode_status(packet: bytes) -> tuple[int, int, bytes]:
    """Parse a status packet -> (id, error_byte, params). Raises on malformed."""
    if len(packet) < 6:
        raise ValueError(f"status packet too short: {len(packet)} bytes")
    if packet[:2] != HEADER:
        raise ValueError(f"bad header: {packet[:2].hex()}")
    servo_id, length, error = packet[2], packet[3], packet[4]
    expected_total = length + 4
    if len(packet) != expected_total:
        raise ValueError(f"length mismatch: says {expected_total}, got {len(packet)}")
    body, got = packet[2:-1], packet[-1]
    if checksum(body) != got:
        raise ValueError(f"checksum mismatch: computed {checksum(body):#04x}, got {got:#04x}")
    return servo_id, error, packet[5:-1]


def to_int(raw: bytes) -> int:
    """STS/SMS multi-byte registers are little-endian."""
    return int.from_bytes(raw, "little")


def to_signed_magnitude(value: int, bits: int = 15) -> int:
    """Load and current use sign-magnitude: the top bit is direction."""
    sign_bit = 1 << bits
    return -(value & (sign_bit - 1)) if value & sign_bit else value


def decode_status_bits(error: int) -> list[str]:
    return [name for mask, name in STATUS_BITS if error & mask]


# ------------------------------------------------------------------- bus io ---


class Bus:
    """Thin read-only transport. Import-safe: pyserial is only needed here."""

    def __init__(self, port: str, baud: int, timeout: float = 0.05) -> None:
        import serial  # local import so the pure-logic functions stay testable

        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self) -> None:
        self.ser.close()

    def _txrx(self, packet: bytes, expect_params: int) -> bytes | None:
        """Write an instruction and return the status params, or None.

        The STS bus is HALF-DUPLEX: TX and RX share one wire, so many adapters
        echo the outgoing bytes straight back. An echoed PING is a checksum-valid
        frame (its instruction byte 0x01 lands where the error byte belongs), so
        naive parsing reports a servo that is not there. We read a generous
        window and strip the echo before parsing.
        """
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        window = self.ser.read(len(packet) + expect_params + 6)
        if not window:
            return None
        if window.startswith(packet):  # adapter echoed us; the reply follows
            window = window[len(packet):]
        if not window:
            return None
        return self._parse(window)

    def _parse(self, window: bytes) -> bytes | None:
        # Resync on the header rather than assuming the frame starts at byte 0.
        start = window.find(HEADER)
        if start < 0:
            return None
        frame = window[start:]
        if len(frame) < 6:
            return None
        total = frame[3] + 4
        if len(frame) < total:
            return None
        try:
            _, error, params = decode_status(frame[:total])
        except ValueError:
            return None
        self.last_error = error
        return params

    def ping(self, servo_id: int) -> bool:
        return self._txrx(build_packet(servo_id, INST_PING), 0) is not None

    def read(self, servo_id: int, address: int, width: int) -> int | None:
        params = bytes([address, width])
        got = self._txrx(build_packet(servo_id, INST_READ, params), width)
        if got is None or len(got) != width:
            return None
        return to_int(got)


# ---------------------------------------------------------------- commands ---


def cmd_scan(port: str, bauds: list[int], id_max: int) -> list[dict]:
    found = []
    for baud in bauds:
        try:
            bus = Bus(port, baud)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  baud {baud}: cannot open port: {exc}", file=sys.stderr)
            continue
        print(f"  scanning ids 0..{id_max} at {baud} baud ...", flush=True)
        for servo_id in range(id_max + 1):
            if bus.ping(servo_id):
                print(f"    FOUND id={servo_id} baud={baud}")
                found.append({"id": servo_id, "baud": baud})
        bus.close()
    return found


def cmd_dump(port: str, baud: int, servo_id: int) -> dict:
    bus = Bus(port, baud)
    out: dict[str, object] = {"id": servo_id, "baud": baud, "registers": {}}
    for address, width, name in REGISTERS:
        value = bus.read(servo_id, address, width)
        out["registers"][name] = {"addr": address, "width": width, "value": value}
    bus.close()
    return out


def annotate(dump: dict, expect_volts: float | None = None) -> list[str]:
    """Turn the raw dump into the checks Stage B actually gates on."""
    reg = {k: v["value"] for k, v in dump["registers"].items()}
    notes: list[str] = []

    # --- Validate the register map itself -------------------------------------
    # An earlier version asserted addr13 == 70 (the datasheet's over-hot figure)
    # and declared the whole map void otherwise. That was wrong: a real STS3250
    # ships addr13 = 80, and the assertion produced a false STOP on a good dump.
    # Validate instead on signals that are independently checkable.
    checks_passed, checks = 0, []

    volts = reg.get("present_voltage")
    if volts is not None:
        notes.append(f"     present_voltage = {volts/10:.1f} V (addr62 raw {volts})")
        if expect_volts is not None:
            if abs(volts / 10 - expect_volts) <= 0.3:
                checks_passed += 1
                checks.append(f"addr62 {volts/10:.1f} V matches the supply {expect_volts:.1f} V")
            else:
                notes.append(
                    f"STOP addr62 reads {volts/10:.1f} V but the supply is "
                    f"{expect_volts:.1f} V. Either the map is wrong or the harness "
                    "is dropping volts. Resolve before trusting any other address."
                )
        elif 40 <= volts <= 160:
            checks_passed += 1
            checks.append(f"addr62 {volts/10:.1f} V is a plausible pack voltage")

    if reg.get("id") is not None and 0 <= reg["id"] <= 253:
        checks_passed += 1
        checks.append(f"addr5 id={reg['id']} is in range")
    if reg.get("baud_code") in BAUD_TABLE:
        checks_passed += 1
        checks.append(f"addr6 baud_code {reg['baud_code']} -> {BAUD_TABLE[reg['baud_code']]}")
    pos = reg.get("present_position")
    if pos is not None and 0 <= pos <= 4095:
        checks_passed += 1
        checks.append(f"addr56 position {pos} is inside the 12-bit range")
    temp = reg.get("present_temperature")
    if temp is not None and 0 < temp < 100:
        checks_passed += 1
        checks.append(f"addr63 {temp} C is a plausible case temperature")

    if checks_passed >= 4:
        notes.append(f"OK   register map VALIDATED by {checks_passed} independent checks:")
        notes.extend(f"       - {c}" for c in checks)
    else:
        notes.append(
            f"STOP only {checks_passed} independent checks passed (need 4). "
            "Treat every address in this dump as unverified."
        )

    # --- Over-temperature: report the delta, do not assume the datasheet wins --
    temp_limit = reg.get("max_temperature_limit")
    if temp_limit is not None:
        if temp_limit == DATASHEET_OVER_HOT_C:
            notes.append(f"     addr13 over-temperature cutoff = {temp_limit} C "
                         "(matches datasheet 7-11)")
        else:
            notes.append(
                f"FIND addr13 over-temperature cutoff = {temp_limit} C, but the "
                f"datasheet's 7-11 over-hot text says {DATASHEET_OVER_HOT_C} C. "
                "THE REGISTER IS WHAT THE FIRMWARE ACTUALLY ENFORCES. Use "
                f"{temp_limit} C as the real cutoff and correct any doc citing "
                f"{DATASHEET_OVER_HOT_C} C."
            )

    temp = reg.get("present_temperature")
    if temp is not None:
        notes.append(f"     present_temperature = {temp} C")

    for key, addr in (("max_torque_limit", 16), ("torque_limit", 48)):
        value = reg.get(key)
        if value is not None and value < 1000:
            notes.append(
                f"WARN addr{addr} {key} = {value} (< 1000). Stage C torque would be "
                "silently clamped and the run is VOID until this is understood."
            )

    if reg.get("lock"):
        notes.append(
            "     addr55 lock = 1 (factory default). EEPROM writes fail SILENTLY "
            "until it is cleared — which is protective for a read-only probe, but "
            "means any future config change must read back what it wrote."
        )

    min_v = reg.get("min_input_voltage")
    if min_v is not None and min_v > 90:
        notes.append(
            f"WARN addr15 min_input_voltage = {min_v/10:.1f} V — the servo will trip "
            "at exactly the low-battery condition most worth characterising."
        )

    status = reg.get("servo_status")
    if status:
        notes.append(f"WARN addr65 servo_status = {status:#04x} {decode_status_bits(status)}")

    prot = reg.get("protection_current")
    if prot is not None:
        lo, hi = (prot * c / 1000 for c in CURRENT_LSB_CANDIDATES_MA)
        notes.append(
            f"FIND addr28 protection_current = {prot} raw -> {lo:.2f} A @6.5 mA/LSB "
            f"or {hi:.2f} A @10 mA/LSB. The LSB is UNVERIFIED for this model and the "
            "two differ by 1.54x. Resolve in Stage C against the bench ammeter."
        )

    for key, addr, unit in (("protective_torque", 34, "%"),
                            ("overload_torque", 36, "% of stall"),
                            ("protection_time", 35, "x10 ms"),
                            ("overcurrent_time", 38, "x10 ms")):
        v = reg.get(key)
        if v is not None:
            notes.append(f"     addr{addr} {key} = {v} {unit}")

    return notes


def cmd_link_test(port: str, baud: int, servo_id: int, count: int) -> dict:
    """Stage B gate: require 100/100 clean, self-consistent replies."""
    bus = Bus(port, baud)
    ok, values = 0, []
    for _ in range(count):
        value = bus.read(servo_id, 62, 1)
        if value is not None:
            ok += 1
            values.append(value)
    bus.close()
    spread = (max(values) - min(values)) if values else None
    return {"attempts": count, "ok": ok, "spread_raw": spread,
            "pass": ok == count and (spread is not None and spread <= 2)}


def cmd_watch(port: str, baud: int, servo_id: int, seconds: float) -> None:
    bus = Bus(port, baud)
    print(f"{'t':>6}  {'pos':>6} {'load':>6} {'volt':>5} {'temp':>4} {'curr':>6} status")
    start = time.time()
    try:
        while time.time() - start < seconds:
            pos = bus.read(servo_id, 56, 2)
            load = bus.read(servo_id, 60, 2)
            volt = bus.read(servo_id, 62, 1)
            temp = bus.read(servo_id, 63, 1)
            status = bus.read(servo_id, 65, 1)
            curr = bus.read(servo_id, 69, 2)
            elapsed = time.time() - start
            print(
                f"{elapsed:6.1f}  {pos if pos is not None else '--':>6} "
                f"{to_signed_magnitude(load) if load is not None else '--':>6} "
                f"{volt/10 if volt is not None else '--':>5} "
                f"{temp if temp is not None else '--':>4} "
                f"{to_signed_magnitude(curr) if curr is not None else '--':>6} "
                f"{decode_status_bits(status) if status else ''}",
                flush=True,
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


# -------------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, help="servo id (omit with --scan)")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--scan", action="store_true", help="sweep all ids x all bauds")
    ap.add_argument("--id-max", type=int, default=253)
    ap.add_argument("--dump", action="store_true", help="full read-only register dump")
    ap.add_argument("--link-test", action="store_true", help="100x reply reliability gate")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--watch", action="store_true", help="1 Hz telemetry")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--expect-volts", type=float,
                    help="supply voltage, used to validate the register map")
    ap.add_argument("--out", help="write JSON result here")
    args = ap.parse_args(argv)

    result: object = None

    if args.scan:
        print(f"Scanning {args.port} (READ-ONLY: ping + read only)")
        result = cmd_scan(args.port, SCAN_BAUDS, args.id_max)
        if not result:
            print("\nNo servo answered on any baud rate.")
            return 1
        print(f"\n{len(result)} servo(s) found.")
    elif args.dump:
        if args.id is None:
            ap.error("--dump requires --id")
        result = cmd_dump(args.port, args.baud, args.id)
        width = max(len(n) for _, _, n in REGISTERS)
        for name, info in result["registers"].items():
            value = info["value"]
            shown = "NO REPLY" if value is None else value
            print(f"  [{info['addr']:>3}] {name:<{width}}  {shown}")
        print()
        for note in annotate(result, args.expect_volts):
            print(note)
    elif args.link_test:
        if args.id is None:
            ap.error("--link-test requires --id")
        result = cmd_link_test(args.port, args.baud, args.id, args.count)
        print(f"  {result['ok']}/{result['attempts']} clean replies, "
              f"raw spread {result['spread_raw']}")
        print("  PASS" if result["pass"] else "  FAIL — do not log a run on this link")
    elif args.watch:
        if args.id is None:
            ap.error("--watch requires --id")
        cmd_watch(args.port, args.baud, args.id, args.seconds)
    else:
        ap.error("pick one of --scan / --dump / --link-test / --watch")

    if args.out and result is not None:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
