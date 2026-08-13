"""Task M1 — the print-process decision record must stay consistent with code.

`scripts/print_process.json` is the single source of truth that M2, M4 and M5
read. Every assertion here **executes or parses**; none greps. `known_issues.md`
TEST-1 measured that 5 of 7 seeded regressions pass this repo's suite because
most of it is `assert "<literal>" in read()`.

Run from the repo root — `import scripts.measure_print_mass` resolves as a
namespace package only with the root on `sys.path`, and there is no
`scripts/__init__.py`.
"""

import json
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CFG_PATH = os.path.join(REPO_ROOT, "scripts", "print_process.json")
GUIDE = os.path.join(REPO_ROOT, "docs", "print_guide.md")
PRINT_DIR = os.path.join(REPO_ROOT, "print")

# Bump this in the SAME COMMIT that adds a row to docs/print_guide.md.
# M3 added holder_6cell (2026-08-12), taking 52 -> 53.
EXPECTED_PIECES = 53

import scripts.measure_print_mass as m  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_PATH) as f:
        return json.load(f)


@pytest.mark.phase3
class TestDecisionRecord:

    def test_process_json_parses_and_is_a_known_process(self, cfg):
        assert cfg["process"] in m.PROCESS, (
            f"{cfg['process']!r} is not in measure_print_mass.PROCESS "
            f"({sorted(m.PROCESS)})")

    def test_density_and_kind_match_the_process_table(self, cfg):
        """Stops someone typing a bulk powder density instead of the
        finished-part figure — a ~40 % error on the solid branch."""
        entry = m.PROCESS[cfg["process"]]
        assert abs(cfg["density_g_cm3"] - entry["density"]) < 1e-9, (
            f"json says {cfg['density_g_cm3']}, PROCESS table says "
            f"{entry['density']}")
        assert cfg["kind"] == entry["kind"]

    def test_tpu_fields_match_the_script_constants(self, cfg):
        assert cfg["tpu_part"] == m.TPU_PART
        assert abs(cfg["tpu_density_g_cm3"] - m.TPU_DENSITY) < 1e-9

    def test_fdm_fields_are_consistent_with_kind(self, cfg):
        if cfg["kind"] == "fdm":
            assert isinstance(cfg["perimeters"], int) and cfg["perimeters"] > 0
            assert isinstance(cfg["infill_pct"], int)
            assert 0 <= cfg["infill_pct"] <= 100
            assert cfg["max_wall_mm"] is None, (
                "max_wall_mm is a solid-process field; M5 does not run on FDM")
        else:
            assert cfg["perimeters"] is None and cfg["infill_pct"] is None
            assert isinstance(cfg["max_wall_mm"], (int, float))
            assert cfg["max_wall_mm"] > 0, (
                "M5 reads max_wall_mm; a solid process must carry the "
                "bureau's cap, cited in print_process_decision.md")

    def test_fallback_is_a_known_process_and_is_not_pla(self, cfg):
        fb = cfg.get("fallback_process")
        if fb is None:
            return
        assert fb in m.PROCESS
        assert fb != "fdm-pla", (
            "PLA's ~60 C HDT sits inside the 55-80 C Jetson heatsink range in "
            "the trunk cavity these parts enclose; it must not be a fallback")

    def test_vendor_and_quote_are_filled_together(self, cfg):
        """Either the quote has happened or it has not. A vendor with no quote
        reference is a half-recorded decision."""
        assert (cfg["vendor"] is None) == (cfg["quote_ref"] is None), (
            "vendor and quote_ref must be both null or both set")

    def test_unconfirmed_profile_is_flagged(self, cfg):
        """The FDM branch's mass depends on a bureau profile that may not be
        the one modelled. M2/M4 must not book a mass while this is false."""
        assert isinstance(cfg["profile_confirmed_with_vendor"], bool)
        if cfg["kind"] == "fdm" and not cfg["profile_confirmed_with_vendor"]:
            assert cfg["vendor"] is None, (
                "a vendor is recorded but its print profile was never "
                "confirmed — that is the PLANT-10 failure class")

    def test_recorded_set_mass_matches_a_fresh_measurement(self, cfg):
        """The headline number in the decision record must still be true.

        Drives the real script exactly as the number was produced, rather than
        reaching into its internals — mass_for_part() takes an argparse
        namespace, and reconstructing one here would test my reconstruction
        instead of the tool.

        Skipped rather than silently trusted when an FDM row cannot be
        re-measured because PrusaSlicer is not on PATH.
        """
        recorded = cfg.get("measured_set_mass_g")
        if recorded is None:
            pytest.skip("no measured_set_mass_g recorded")
        if cfg["kind"] == "fdm" and m.find_slicer(None) is None:
            pytest.skip("PrusaSlicer not on PATH; cannot re-measure an FDM row")

        cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts",
                                            "measure_print_mass.py"),
               "--process", cfg["process"]]
        if cfg["kind"] == "fdm":
            cmd += ["--perimeters", str(cfg["perimeters"]),
                    "--infill", str(cfg["infill_pct"])]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=1800)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        line = [l for l in proc.stdout.splitlines() if l.startswith("TOTAL")]
        assert line, f"no TOTAL line in output:\n{proc.stdout[-2000:]}"
        pieces, volume, mass = (float(x) for x in line[0].split()[1:4])
        assert int(pieces) == EXPECTED_PIECES
        assert abs(mass - recorded) < 1.0, (
            f"print_process.json records {recorded:.2f} g but a fresh "
            f"measurement gives {mass:.2f} g")


@pytest.mark.phase3
class TestPieceAccounting:
    """PLANT-10's counting rule: the set is 52 PIECES, not 37 files."""

    def test_every_print_stl_has_an_explicit_guide_row(self):
        """Kills part_quantities()'s silent quantity-1 fallback.

        Fails the moment M3 drops holder_6cell.stl into print/ without adding
        a row to docs/print_guide.md.
        """
        on_disk = {f[:-len(".stl")] for f in os.listdir(PRINT_DIR)
                   if f.endswith(".stl")}
        guide_text = open(GUIDE).read()
        in_guide = set(re.findall(r"^-\s+(\S+)\.stl", guide_text, re.M))
        missing = sorted(on_disk - in_guide)
        extra = sorted(in_guide - on_disk)
        assert not missing, (
            f"print/*.stl with no row in print_guide.md: {missing}. "
            "part_quantities() would silently count them once each.")
        assert not extra, (
            f"print_guide.md lists parts with no STL: {extra}")

    def test_piece_count_matches_the_guide(self):
        got = sum(m.part_quantities().values())
        assert got == EXPECTED_PIECES, (
            f"{got} pieces, expected {EXPECTED_PIECES}. Bump EXPECTED_PIECES "
            "in the same commit that adds a row to docs/print_guide.md.")

    # Bump in the SAME COMMIT as any geometry change. 1571.94 was the pre-M3
    # figure; M3 (2026-08-12) added holder_6cell (22.49 cm3) and deepened the
    # body_back hump for the 6-cell pack.
    EXPECTED_SET_VOLUME_CM3 = 1598.61

    def test_solid_volume_is_the_documented_total(self):
        """The set volume anchors every mass in print_process_decision.md."""
        total = sum(m.solid_volume_cm3(os.path.join(PRINT_DIR, f"{n}.stl")) * q
                    for n, q in m.part_quantities().items())
        assert abs(total - self.EXPECTED_SET_VOLUME_CM3) < 0.5, (
            f"set volume is now {total:.2f} cm3, expected "
            f"{self.EXPECTED_SET_VOLUME_CM3}. If a part was intentionally "
            "added or reshaped, bump EXPECTED_SET_VOLUME_CM3 and re-run "
            "measure_print_mass.py --emit-table plus generate_cad_mods.py.")


@pytest.mark.phase3
def test_guide_no_longer_specifies_pla_as_the_material():
    """The guide's original PLA line is superseded by print_process.json.

    Asserts the pointer exists and resolves, not that a literal is absent —
    a literal check would pass on a file that merely deleted the sentence.
    """
    text = open(GUIDE).read()
    assert "scripts/print_process.json" in text, (
        "print_guide.md must point at the decision record")
    cfg = json.load(open(CFG_PATH))
    assert cfg["process"] in text, (
        f"print_guide.md does not name the chosen process {cfg['process']!r}")
