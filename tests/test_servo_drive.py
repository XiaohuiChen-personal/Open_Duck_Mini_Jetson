"""Safety tests for scripts/servo_drive.py.

This is the only tool in the repo that can command the project's single servo.
Every test here pins a refusal that, if it stopped working, would either be
unrecoverable (EEPROM, re-zeroing the midpoint) or dangerous (uncapped torque on
a joint with no mechanical end stop).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "servo_drive", REPO / "scripts" / "servo_drive.py"
)
drive = importlib.util.module_from_spec(SPEC)
sys.modules["servo_drive"] = drive
SPEC.loader.exec_module(drive)


# ------------------------------------------------------------ the refusals --


def test_the_centering_command_is_refused():
    """Datasheet 7-14: addr40 <- 128 re-zeroes the servo midpoint. It would
    silently invalidate every joint zero downstream, and there is no undo."""
    with pytest.raises(drive.UnsafeWrite, match="CENTERING"):
        drive.build_write_packet(1, 40, 128, 1)


def test_torque_enable_still_accepts_0_and_1():
    """Refusing 128 must not break the register's actual purpose."""
    for value in (0, 1):
        assert drive.build_write_packet(1, 40, value, 1)


@pytest.mark.parametrize("address", [0, 5, 6, 13, 14, 15, 16, 28, 33, 36, 39])
def test_every_eeprom_address_is_refused(address):
    """EEPROM writes persist. A bad addr5/addr6 write loses the servo for good."""
    with pytest.raises(drive.UnsafeWrite, match="EEPROM"):
        drive.build_write_packet(1, address, 1, 1)


def test_the_eeprom_lock_register_is_refused():
    with pytest.raises(drive.UnsafeWrite):
        drive.build_write_packet(1, 55, 0, 1)


def test_addresses_off_the_allowlist_are_refused():
    for address in (56, 60, 62, 63, 65, 69, 100):
        with pytest.raises(drive.UnsafeWrite):
            drive.build_write_packet(1, address, 1, 1)


def test_broadcast_id_is_refused():
    """A broadcast write hits every servo on the bus at once, unacknowledged."""
    with pytest.raises(drive.UnsafeWrite, match="broadcast"):
        drive.build_write_packet(0xFE, 42, 100, 2)


def test_goal_position_beyond_the_encoder_range_is_refused():
    assert drive.build_write_packet(1, 42, 4095, 2)
    with pytest.raises(drive.UnsafeWrite, match="goal_position"):
        drive.build_write_packet(1, 42, 4096, 2)


def test_torque_limit_above_full_scale_is_refused():
    assert drive.build_write_packet(1, 48, 1000, 2)
    with pytest.raises(drive.UnsafeWrite, match="torque_limit"):
        drive.build_write_packet(1, 48, 1001, 2)


def test_wrong_width_for_an_address_is_refused():
    with pytest.raises(drive.UnsafeWrite, match="2-byte"):
        drive.build_write_packet(1, 42, 100, 1)
    with pytest.raises(drive.UnsafeWrite, match="1-byte"):
        drive.build_write_packet(1, 40, 1, 2)


def test_value_that_does_not_fit_is_refused():
    with pytest.raises(drive.UnsafeWrite, match="does not fit"):
        drive.build_write_packet(1, 40, 256, 1)


# ------------------------------------------------------- allowlist contents --


def test_allowlist_is_sram_only():
    assert min(drive.WRITABLE) > drive.EEPROM_LAST_ADDR
    assert set(drive.WRITABLE) == {40, 41, 42, 46, 48}


def test_default_torque_limit_is_capped_well_below_stall():
    """The servo has no mechanical end stop and 4.9 N.m of stall torque."""
    assert drive.DEFAULT_TORQUE_LIMIT <= 200
    assert drive.DEFAULT_TORQUE_LIMIT / drive.TORQUE_LIMIT_MAX <= 0.2


def test_abort_temperature_is_below_the_firmware_cutoff():
    """addr13 on this unit reads 80 C. Abort must fire with real margin."""
    assert drive.ABORT_TEMP_C <= 65


# ------------------------------------------------------------ packet format --


def test_write_packet_matches_the_documented_frame():
    # WRITE torque_enable=1 to id 1: FF FF 01 04 03 28 01 <chk>
    packet = drive.build_write_packet(1, 40, 1, 1)
    assert packet[:7] == bytes([0xFF, 0xFF, 0x01, 0x04, 0x03, 0x28, 0x01])
    assert packet[7] == drive.checksum(bytes([0x01, 0x04, 0x03, 0x28, 0x01]))


def test_two_byte_values_are_little_endian():
    # Use goal_speed (46): goal_position is range-capped at 4095, so 0x1234
    # is correctly refused there — see the test below.
    packet = drive.build_write_packet(1, 46, 0x1234, 2)
    assert packet[5:8] == bytes([46, 0x34, 0x12])


def test_position_cap_fires_even_on_a_well_formed_two_byte_value():
    """Regression: 0x1234 is a valid 2-byte value but an invalid position."""
    with pytest.raises(drive.UnsafeWrite, match="goal_position"):
        drive.build_write_packet(1, 42, 0x1234, 2)


# ---------------------------------------------------------------- lifecycle --


class _FakeSerial:
    def __init__(self):
        self.writes = []

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.writes.append(data)

    def read(self, n):
        return b""          # never acknowledge; exercises the failure path

    def close(self):
        pass


def _driver():
    d = drive.Driver.__new__(drive.Driver)
    d.ser = _FakeSerial()
    d.id, d.torque_limit, d.samples, d._armed = 1, 200, [], False
    return d


def test_release_writes_torque_enable_zero():
    d = _driver()
    d.release()
    assert d.ser.writes, "release must transmit something"
    last = d.ser.writes[-1]
    assert last[5] == 40 and last[6] == 0


def test_close_always_releases_torque():
    """If this process dies, the servo must go limp rather than keep driving."""
    d = _driver()
    d.close()
    assert any(w[5] == 40 and w[6] == 0 for w in d.ser.writes)


def test_arm_refuses_when_position_cannot_be_read():
    """Arming blind would enable torque against an unknown goal — i.e. a lurch."""
    d = _driver()
    with pytest.raises(RuntimeError, match="present position"):
        d.arm()


# ------------------------------------------------------------------ aborts --


def test_nonzero_status_byte_aborts():
    d = _driver()
    assert d.check_abort({"status": 0x04, "temperature": 30, "voltage": 11.1})


def test_overtemperature_aborts_before_the_firmware_does():
    d = _driver()
    assert d.check_abort({"status": 0, "temperature": drive.ABORT_TEMP_C, "voltage": 11.1})


def test_supply_sag_aborts():
    """A sagging rail means the PSU has gone into CC — the run is invalid and a
    brownout mid-write is how EEPROM gets corrupted."""
    d = _driver()
    assert d.check_abort({"status": 0, "temperature": 30, "voltage": 8.5})


def test_healthy_telemetry_does_not_abort():
    d = _driver()
    assert d.check_abort({"status": 0, "temperature": 34, "voltage": 11.1}) is None


def test_probe_module_remains_write_free():
    """servo_drive is where the risk is concentrated; the probe must stay clean."""
    src = (REPO / "scripts" / "servo_probe.py").read_text()
    assert "INST_WRITE" not in src
    assert "0x03" not in src.replace("0x03,", "")  # no write opcode smuggled in
