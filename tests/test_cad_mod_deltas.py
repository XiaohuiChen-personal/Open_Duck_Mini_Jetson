"""Task M2 — the booked CAD-mod delta must provably be the measured delta.

This is the assertion PLANT-10 was missing. `generate_cad_mods.py` used to book
the Part-2 mesh deltas as one assumed density times a volume difference: it
booked −88.48 g where the true figure is −13.95 g at the chosen process, so
`trunk_assembly` was 74.53 g light and every policy trained on it.

Nothing here greps. The delta test recomputes the booked net from the JSON and
compares it against the measured table; the density test imports the module and
checks an attribute, so it cannot be defeated by moving a constant into a
string.
"""

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DELTAS = os.path.join(REPO_ROOT, "scripts", "cad_mod_deltas.json")
TABLE = os.path.join(REPO_ROOT, "scripts", "part_mass_table.json")
PROCESS = os.path.join(REPO_ROOT, "scripts", "print_process.json")

CAD_MOD_PARTS = ("body_front", "body_middle_bottom", "trunk_bottom", "body_back")


def _load(path):
    if not os.path.isfile(path):
        pytest.skip(f"{os.path.basename(path)} not generated yet")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def deltas():
    return _load(DELTAS)


@pytest.fixture(scope="module")
def table():
    return _load(TABLE)


@pytest.fixture(scope="module")
def process():
    return _load(PROCESS)


@pytest.mark.phase3
class TestBookedEqualsMeasured:

    def test_deltas_json_matches_the_measured_table(self, deltas, table):
        """Per part, the sum of its term masses == measured current − baseline.

        This is the whole point of M2: the booked number is the measurement, by
        construction, and this test is what keeps it that way.
        """
        by_part = {}
        for t in deltas["terms"]:
            part = t["name"].rsplit("_", 1)[0]
            by_part.setdefault(part, 0.0)
            by_part[part] += t["mass"]

        for p in CAD_MOD_PARTS:
            assert p in by_part, f"no terms booked for {p}"
            expected = (table["parts"][p]["mass_g"]
                        - table["baseline"][p]["mass_g"]) / 1000.0
            assert abs(by_part[p] - expected) < 1e-6, (
                f"{p}: booked {by_part[p] * 1000:.4f} g, measured "
                f"{expected * 1000:.4f} g")

    def test_net_delta_matches_the_table(self, deltas, table):
        net = sum(t["mass"] for t in deltas["terms"]) * 1000.0
        assert abs(net - table["cad_mod_net_delta_g"]) < 1e-3, (
            f"deltas sum to {net:.4f} g, table says "
            f"{table['cad_mod_net_delta_g']:.4f} g")

    def test_no_assumed_density_constant_remains(self):
        """An attribute check, not a text grep — moving the constant into a
        string literal must not defeat it. Importing is safe: main() is
        guarded by `if __name__ == "__main__"`."""
        import scripts.generate_cad_mods as g
        assert not hasattr(g, "PLA_EFFECTIVE_DENSITY"), (
            "the assumed-density constant is back; M2 removed it because one "
            "density cannot describe both sides of a diff")
        assert hasattr(g, "load_part_masses")
        assert hasattr(g, "whole_part_terms")

    def test_deltas_json_records_its_process(self, deltas, process):
        """A delta booked against a different process than the one chosen is
        the PLANT-10 error wearing a new hat."""
        assert deltas.get("process") == process["process"], (
            f"cad_mod_deltas.json booked against {deltas.get('process')!r} but "
            f"print_process.json says {process['process']!r}")
        if process["kind"] == "fdm":
            assert deltas.get("perimeters") == process["perimeters"]
            assert deltas.get("infill_pct") == process["infill_pct"]
        assert "density_kg_m3" not in deltas, (
            "density_kg_m3 is the pre-M2 schema; mass is measured now")


@pytest.mark.phase3
class TestTermsArePhysical:

    def test_positive_mass_terms_have_valid_inertia(self, deltas):
        """Positive-mass terms are real bodies: PSD and triangle-inequality.

        Deliberately NOT applied to negative-mass terms — those are bookkeeping
        subtractions, not bodies, and demanding physical validity of them would
        be a category error.
        """
        checked = 0
        for t in deltas["terms"]:
            if t["mass"] <= 0:
                continue
            checked += 1
            I = np.array(t["tensor"], dtype=float)
            assert I.shape == (3, 3)
            assert np.allclose(I, I.T, atol=1e-12), f"{t['name']}: not symmetric"
            eig = np.linalg.eigvalsh(I)
            assert (eig > 0).all(), f"{t['name']}: not positive definite {eig}"
            a, b, c = sorted(eig)
            assert a + b >= c - 1e-12, (
                f"{t['name']}: violates the triangle inequality {eig}")
        assert checked >= len(CAD_MOD_PARTS), (
            f"only {checked} positive-mass terms; expected one per modified part")

    def test_each_part_has_a_baseline_and_a_current_term(self, deltas):
        names = {t["name"] for t in deltas["terms"]}
        for p in CAD_MOD_PARTS:
            assert f"{p}_baseline" in names, f"missing {p}_baseline"
            assert f"{p}_current" in names, f"missing {p}_current"
        for t in deltas["terms"]:
            if t["name"].endswith("_baseline"):
                assert t["mass"] < 0, f"{t['name']} must carry negative mass"
            elif t["name"].endswith("_current"):
                assert t["mass"] > 0, f"{t['name']} must carry positive mass"

    def test_tensors_are_emitted_positive(self, deltas):
        """The consumer negates when mass < 0. A pre-negated tensor would be
        double-negated, which is silent and wrong."""
        for t in deltas["terms"]:
            I = np.array(t["tensor"], dtype=float)
            assert np.trace(I) > 0, (
                f"{t['name']}: tensor trace is not positive — it looks "
                "pre-negated, which _load_shell_deltas would negate again")


@pytest.mark.phase3
class TestEffectiveDensity:

    def test_effective_density_matches_the_process(self, table, process):
        """Catches a bulk-vs-finished density mix-up in either direction.

        `foot_bottom_tpu` is exempt on every branch — mass_for_part() already
        substitutes the TPU density for it.
        """
        import scripts.measure_print_mass as m
        kind = process["kind"]
        rho_proc = process["density_g_cm3"]
        for section in ("parts", "baseline"):
            for name, rec in table.get(section, {}).items():
                rho = rec["effective_density_g_cm3"]
                if name == m.TPU_PART:
                    if kind == "solid":
                        assert abs(rho - m.TPU_DENSITY) < 1e-3, (
                            f"{name}: {rho} should be the TPU density "
                            f"{m.TPU_DENSITY}")
                    continue
                if kind == "solid":
                    assert abs(rho - rho_proc) < 1e-6, (
                        f"{name}: solid parts must sit exactly at the process "
                        f"density; got {rho} vs {rho_proc}")
                else:
                    # A thin part CAN exceed nominal. Measured here: five parts
                    # do, worst 1.1021 on knee_to_ankle_right_sheet (a ~2 mm
                    # sheet). At 3 perimeters that asks ~2.7 mm of wall per
                    # side into a 2 mm thickness, so the slicer packs solid and
                    # the perimeters overlap — and its mass comes from extruded
                    # filament length, which counts the overlap. 1.15x is the
                    # measured ceiling plus headroom; it is NOT permission for
                    # a solid figure, which the aggregate test below catches.
                    assert 0 < rho < rho_proc * 1.15, (
                        f"{name}: FDM effective density {rho} is more than "
                        f"1.15x the filament density {rho_proc}")

    def test_the_set_is_not_secretly_solid(self, table, process):
        """The check the per-part bound above is too loose to make.

        If a solid-process density leaked into an FDM table, every part would
        sit at or near the filament density. Volume-weighting is what makes
        this bite: the big shells dominate the robot's mass, and they are the
        ones infill actually hollows.
        """
        if process["kind"] != "fdm":
            pytest.skip("solid process: parts are solid by definition")
        vol = sum(r["volume_cm3"] * r["qty"] for r in table["parts"].values())
        mass = sum(r["mass_g"] * r["qty"] for r in table["parts"].values())
        mean_rho = mass / vol
        assert mean_rho < 0.90 * process["density_g_cm3"], (
            f"volume-weighted mean effective density is {mean_rho:.4f} g/cm3 "
            f"against a filament density of {process['density_g_cm3']} — that "
            "is a solid part set, so infill was not applied")

    def test_the_spread_is_wide_enough_to_justify_m2(self, table):
        """The reason a single constant was wrong, asserted rather than
        asserted-in-prose: the parts do not share one density."""
        rhos = [r["effective_density_g_cm3"] for r in table["parts"].values()]
        assert max(rhos) / min(rhos) > 1.5, (
            f"effective density spread is only {max(rhos)/min(rhos):.2f}x "
            f"({min(rhos):.3f}–{max(rhos):.3f}); if the parts really do share "
            "a density, M2's whole-part scheme deserves re-examination")

    def test_table_covers_every_printed_part(self, table):
        import scripts.measure_print_mass as m
        expected = set(m.part_quantities())
        assert set(table["parts"]) == expected, (
            f"missing {sorted(expected - set(table['parts']))}, "
            f"extra {sorted(set(table['parts']) - expected)}")
        assert set(table["baseline"]) == set(CAD_MOD_PARTS)
