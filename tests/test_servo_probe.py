"""Protocol tests for scripts/servo_probe.py.

The probe talks to a servo we own exactly one of, over a protocol where a
malformed packet can be interpreted as a WRITE. These tests pin the wire format
and the read-only guarantee without any hardware attached.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "servo_probe", REPO / "scripts" / "servo_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
sys.modules["servo_probe"] = probe
SPEC.loader.exec_module(probe)


# ------------------------------------------------------------------ checksum --


def test_checksum_is_bitwise_not_of_sum():
    # Feetech: checksum = ~(ID + Length + Instruction + params) & 0xFF
    body = bytes([0x01, 0x02, 0x01])
    assert probe.checksum(body) == (~0x04) & 0xFF == 0xFB


def test_checksum_wraps_past_one_byte():
    assert probe.checksum(bytes([0xFF, 0xFF, 0xFF])) == (~0x2FD) & 0xFF


# -------------------------------------------------------------- ping packet --


def test_ping_packet_matches_known_good_frame():
    # The canonical Feetech PING for ID 1: FF FF 01 02 01 FB
    assert probe.build_packet(1, probe.INST_PING) == bytes(
        [0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]
    )


def test_read_packet_matches_known_good_frame():
    # READ 2 bytes from address 56 (present position) on ID 1:
    # FF FF 01 04 02 38 02 <chk>
    packet = probe.build_packet(1, probe.INST_READ, bytes([56, 2]))
    assert packet[:7] == bytes([0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02])
    assert packet[7] == probe.checksum(bytes([0x01, 0x04, 0x02, 0x38, 0x02]))


def test_length_field_counts_params_plus_two():
    for params in (b"", bytes([56, 2]), bytes([0, 10])):
        packet = probe.build_packet(1, probe.INST_READ, params)
        assert packet[3] == len(params) + 2


# ------------------------------------------------------- read-only guarantee --


@pytest.mark.parametrize("instruction", [0x03, 0x04, 0x05, 0x83, 0x00, 0xFF])
def test_write_class_instructions_are_refused(instruction):
    """The whole safety argument of this module. A stray WRITE bricks the servo."""
    with pytest.raises(ValueError, match="read-only"):
        probe.build_packet(1, instruction)


def test_module_source_contains_no_write_instruction_constant():
    source = (REPO / "scripts" / "servo_probe.py").read_text()
    assert "INST_WRITE" not in source
    assert "INST_SYNC_WRITE" not in source


def test_broadcast_and_out_of_range_ids_rejected():
    with pytest.raises(ValueError, match="out of range"):
        probe.build_packet(0xFE, probe.INST_PING)  # broadcast
    with pytest.raises(ValueError, match="out of range"):
        probe.build_packet(-1, probe.INST_PING)


# ------------------------------------------------------------ status parsing --


def _status(servo_id: int, error: int, params: bytes) -> bytes:
    body = bytes([servo_id, len(params) + 2, error]) + params
    return probe.HEADER + body + bytes([probe.checksum(body)])


def test_decode_status_roundtrip():
    packet = _status(1, 0, bytes([0x34, 0x12]))
    servo_id, error, params = probe.decode_status(packet)
    assert (servo_id, error, params) == (1, 0, bytes([0x34, 0x12]))


def test_decode_status_rejects_bad_checksum():
    packet = bytearray(_status(1, 0, bytes([0x00])))
    packet[-1] ^= 0xFF
    with pytest.raises(ValueError, match="checksum"):
        probe.decode_status(bytes(packet))


def test_decode_status_rejects_bad_header():
    packet = bytearray(_status(1, 0, b""))
    packet[0] = 0x00
    with pytest.raises(ValueError, match="header"):
        probe.decode_status(bytes(packet))


def test_decode_status_rejects_truncated_packet():
    with pytest.raises(ValueError, match="too short"):
        probe.decode_status(bytes([0xFF, 0xFF, 0x01]))


def test_decode_status_rejects_length_mismatch():
    packet = bytearray(_status(1, 0, bytes([0x00, 0x00])))
    packet[3] = 9  # claim more than is present
    with pytest.raises(ValueError, match="length mismatch"):
        probe.decode_status(bytes(packet))


# ---------------------------------------------------------------- endianness --


def test_multibyte_registers_are_little_endian():
    """STS/SMS are little-endian; SCS is big-endian. Reversing this silently
    garbles every position, load and current reading."""
    assert probe.to_int(bytes([0x34, 0x12])) == 0x1234


def test_sign_magnitude_decoding():
    # Load/current use bit 15 as direction, not two's complement.
    assert probe.to_signed_magnitude(0x0064) == 100
    assert probe.to_signed_magnitude(0x8064) == -100
    assert probe.to_signed_magnitude(0x0000) == 0


# ------------------------------------------------------------ register table --


def test_register_addresses_are_unique_and_non_overlapping():
    spans = []
    for address, width, name in probe.REGISTERS:
        spans.append((address, address + width, name))
    spans.sort()
    for (start_a, end_a, name_a), (start_b, _, name_b) in zip(spans, spans[1:]):
        assert end_a <= start_b, f"{name_a} overlaps {name_b}"


def test_stage_b_gate_registers_are_present():
    """Stage B gates on these specific addresses; losing one silently weakens it."""
    by_addr = {addr: name for addr, _, name in probe.REGISTERS}
    for addr, expected in [
        (5, "id"), (6, "baud_code"), (13, "max_temperature_limit"),
        (15, "min_input_voltage"), (16, "max_torque_limit"), (40, "torque_enable"),
        (48, "torque_limit"), (55, "lock"), (56, "present_position"),
        (60, "present_load"), (62, "present_voltage"),
        (63, "present_temperature"), (65, "servo_status"), (69, "present_current"),
    ]:
        assert by_addr.get(addr) == expected, f"addr {addr} should be {expected}"


def test_baud_table_covers_the_scan_list():
    assert set(probe.SCAN_BAUDS) == set(probe.BAUD_TABLE.values())
    assert probe.BAUD_TABLE[0] == 1_000_000, "STS factory default is 1 Mbps"


# ---------------------------------------------------------------- annotation --


def _dump(**overrides):
    registers = {name: {"addr": addr, "width": width, "value": None}
                 for addr, width, name in probe.REGISTERS}
    registers["max_temperature_limit"]["value"] = 70
    for key, value in overrides.items():
        registers[key]["value"] = value
    return {"id": 1, "baud": 1_000_000, "registers": registers}


def test_register_map_validated_when_addr13_is_70():
    notes = probe.annotate(_dump())
    assert any("VALIDATED" in n for n in notes)


def test_register_map_rejected_when_addr13_is_not_70():
    """The datasheet's over-hot cutoff is 70 C. If addr13 disagrees, the map is
    for a different model and every other address in the dump is meaningless."""
    dump = _dump()
    dump["registers"]["max_temperature_limit"]["value"] = 80
    notes = probe.annotate(dump)
    assert any(n.startswith("STOP") for n in notes)


def test_clamped_torque_limit_is_flagged_as_void():
    notes = probe.annotate(_dump(max_torque_limit=500))
    assert any("VOID" in n for n in notes)


def test_full_torque_limit_is_not_flagged():
    notes = probe.annotate(_dump(max_torque_limit=1000, torque_limit=1000))
    assert not any("VOID" in n for n in notes)


def test_eeprom_lock_is_flagged():
    assert any("lock is SET" in n for n in probe.annotate(_dump(lock=1)))


def test_raised_min_voltage_is_flagged():
    notes = probe.annotate(_dump(min_input_voltage=100))  # 10.0 V
    assert any("low-battery" in n for n in notes)


def test_status_error_bits_decode():
    assert probe.decode_status_bits(0x24) == ["TEMPERATURE", "OVERLOAD"]
    assert probe.decode_status_bits(0x00) == []


def test_link_test_requires_all_replies():
    """Stage B requires 100/100. A single-digit failure rate invalidates a run."""
    assert probe.cmd_link_test.__doc__ and "100/100" in probe.cmd_link_test.__doc__


# ---------------------------------------------------- half-duplex echo guard --


class _FakeSerial:
    """Stands in for pyserial so transport logic is testable with no hardware."""

    def __init__(self, response: bytes):
        self.response = response
        self.written = b""

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written += data

    def read(self, n):
        out, self.response = self.response[:n], self.response[n:]
        return out

    def close(self):
        pass


def _bus_with(response: bytes):
    bus = probe.Bus.__new__(probe.Bus)  # bypass __init__ so pyserial is not needed
    bus.ser = _FakeSerial(response)
    return bus


def test_echoed_ping_is_not_reported_as_a_servo():
    """The bug a no-hardware negative control caught: on a half-duplex bus an
    echoed PING is checksum-valid, because its instruction byte 0x01 lands where
    the error byte belongs. Naive parsing invents a servo at every ID."""
    ping = probe.build_packet(1, probe.INST_PING)
    # Prove the echo really would decode as a valid status frame...
    servo_id, error, params = probe.decode_status(ping)
    assert (servo_id, error, params) == (1, 0x01, b"")
    # ...and that the transport refuses it anyway.
    assert _bus_with(ping).ping(1) is False


@pytest.mark.parametrize("servo_id", [0, 1, 2, 3, 17, 253])
def test_echo_rejected_for_every_id(servo_id):
    ping = probe.build_packet(servo_id, probe.INST_PING)
    assert _bus_with(ping).ping(servo_id) is False


def test_real_reply_after_an_echo_is_still_found():
    ping = probe.build_packet(1, probe.INST_PING)
    reply = _status(1, 0, b"")
    assert _bus_with(ping + reply).ping(1) is True


def test_real_reply_without_echo_is_found():
    assert _bus_with(_status(1, 0, b"")).ping(1) is True


def test_read_returns_value_after_echo():
    request = probe.build_packet(1, probe.INST_READ, bytes([56, 2]))
    reply = _status(1, 0, bytes([0x34, 0x12]))
    assert _bus_with(request + reply).read(1, 56, 2) == 0x1234


def test_silence_is_not_a_servo():
    assert _bus_with(b"").ping(1) is False


def test_leading_garbage_is_resynced_past():
    reply = _status(1, 0, b"")
    assert _bus_with(b"\x00\x13" + reply).ping(1) is True
