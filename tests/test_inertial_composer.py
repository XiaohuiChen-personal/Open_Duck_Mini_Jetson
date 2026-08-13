"""Task M4 — the bottom-up inertial composer.

Every body's mass and inertia is now derived from parts and densities rather
than inherited from an upstream export at densities nobody recorded. These
tests pin the mechanics that make that derivation trustworthy, and encode the
traps as executable facts rather than comments.
"""

import importlib.util
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SIM = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")
MJCF = os.path.join(SIM, "robot_motors.xml")
DENSITIES = os.path.join(REPO_ROOT, "scripts", "part_densities.json")

# Bump this literal when a part's mass genuinely becomes known. It exists so
# that silently adding a new guess fails the suite.
EXPECTED_ASSUMED_SOURCES = 9


@pytest.fixture(scope="module")
def cbi():
    spec = importlib.util.spec_from_file_location(
        "cbi", os.path.join(REPO_ROOT, "scripts", "compose_body_inertials.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def densities():
    return json.load(open(DENSITIES))["parts"]


@pytest.mark.phase1
class TestComposerMechanics:

    def test_composer_reproduces_the_pure_printed_anchors(self, cbi):
        """The load-bearing test.

        `head_pitch_to_yaw` and the two antenna holders contain no bought part,
        so their declared inertia is exactly uniform-density inertia over their
        meshes. Any error in the parallel-axis summation, the quaternion
        convention, the diaginertia-vs-fullinertia handling or the
        de-duplication shows up here directly.
        """
        rows, _, _ = cbi.run(golden=True)
        by = {b: (d, c) for b, d, c in rows}
        for body in cbi.ANCHORS_MASS_AND_INERTIA:
            d, c = by[body]
            assert abs(c["mass"] / d["mass"] - 1.0) < 0.005, (
                f"{body} mass ratio {c['mass']/d['mass']:.4f}")
            pd_, pc = cbi.principal(d["I"]), cbi.principal(c["I"])
            for k in range(3):
                assert abs(pc[k] / pd_[k] - 1.0) < 0.005, (
                    f"{body} principal[{k}] ratio {pc[k]/pd_[k]:.4f}")

    def test_pure_printed_anchors_imply_1250_kg_m3(self, cbi):
        """The printed convention the upstream export used for THESE bodies."""
        # The FROZEN upstream export, not the live MJCF: Task M4 rewrote the
        # live values to the composed ones, so reading them here would just
        # confirm the composer against itself.
        decl = cbi.upstream_inertials()
        insts = cbi.body_instances()
        for body in ("head_pitch_to_yaw", "left_antenna_holder"):
            vol = 0.0
            for i in insts[body]:
                vol += float(cbi.mesh_for(i["mesh"]).volume)
            rho = decl[body]["mass"] / vol
            assert abs(rho - 1250.0) / 1250.0 < 0.01, (
                f"{body} implies {rho:.1f} kg/m3, not ~1250")

    def test_servo_anchor_matches_on_mass_but_not_inertia(self, cbi):
        """Encodes a measured limitation rather than hiding it.

        `neck_yaw_assembly` composes to 0.02 % of declared MASS from printed
        meshes at 1250 plus one 74.5 g servo, which validates the servo figure.
        Its principal moments come out at 0.90-0.98 because this composer
        splits a servo's mass across its five case meshes BY VOLUME, and a real
        STS3250 concentrates its motor and gearbox. If someone later builds a
        servo mass-distribution model, this test should start failing and be
        upgraded -- that is the point.
        """
        rows, _, _ = cbi.run(golden=True)
        d, c = {b: (x, y) for b, x, y in rows}["neck_yaw_assembly"]
        assert abs(c["mass"] / d["mass"] - 1.0) < 0.005
        worst = max(abs(pc / pd_ - 1.0) for pc, pd_ in
                    zip(cbi.principal(c["I"]), cbi.principal(d["I"])))
        assert 0.02 < worst < 0.20, (
            f"servo-inertia mismatch is {worst:.3f}; it was 0.094 when measured "
            "2026-08-12. If it collapsed to ~0, the volume-split was replaced "
            "by something better and this test should be tightened.")


@pytest.mark.phase1
class TestDeduplicationTraps:

    def test_mesh_instances_are_deduplicated(self, cbi):
        """Three counts, all measured, encoding the trap as a fact."""
        root = ET.parse(MJCF).getroot()
        raw = [g for g in root.iter("geom") if g.get("type") == "mesh"]
        with_body = set()
        without_body = set()
        for b in root.iter("body"):
            for g in b.findall("geom"):
                if g.get("type") != "mesh":
                    continue
                key = (g.get("mesh"), g.get("pos") or "", g.get("quat") or "")
                with_body.add((b.get("name"),) + key)
                without_body.add(key)

        insts = cbi.body_instances()
        composed = sum(len(v) for v in insts.values())
        assert composed == len(with_body) == 137, (
            f"composer sees {composed}, unique-with-body is {len(with_body)}; "
            "both should be 137 after Task M3")
        assert len(raw) == 271, (
            f"{len(raw)} raw mesh geoms; summing these would double every mass")
        assert len(without_body) < len(with_body), (
            "keying without the body name must collapse the mirrored legs — "
            f"got {len(without_body)} vs {len(with_body)}")

    def test_one_servo_is_one_mass(self, cbi):
        """An STS3250 is five meshes that do NOT share a pos.

        `passive_palonier` carries a different pos in every servo, so grouping
        by (pos, quat) splits it into a phantom sixth part. (body, quat) yields
        exactly 14 groups of exactly 5.
        """
        groups = cbi.servo_groups(cbi.body_instances())
        assert len(groups) == 14, f"{len(groups)} servo groups, expected 14"
        for key, meshes in groups.items():
            assert len(meshes) == 5, (
                f"servo {key} has {len(meshes)} meshes, expected 5")
        total = 14 * cbi.SERVO_MASS_KG
        assert abs(total - 1.043) < 1e-9, (
            f"14 servos total {total*1000:.1f} g, expected 1043 g")


@pytest.mark.phase1
class TestDensityTable:

    def test_every_asset_mesh_has_a_density_entry(self, densities):
        root = ET.parse(MJCF).getroot()
        assets = {m.get("name") for m in root.iter("mesh")}
        missing = assets - set(densities)
        extra = set(densities) - assets
        assert not missing, f"no density entry for {sorted(missing)}"
        assert not extra, f"density entries with no asset mesh: {sorted(extra)}"

    def test_every_density_entry_cites_a_source(self, densities):
        for name, e in densities.items():
            assert e.get("source"), f"{name} has no source"
        n = sum(1 for e in densities.values()
                if str(e["source"]).startswith("ASSUMED"))
        assert n == EXPECTED_ASSUMED_SOURCES, (
            f"{n} entries carry an ASSUMED source, expected "
            f"{EXPECTED_ASSUMED_SOURCES}. Adding a guess must be deliberate: "
            "bump the literal and say why in the commit.")

    def test_printed_parts_resolve_to_a_print_name(self, densities):
        table = json.load(open(os.path.join(REPO_ROOT, "scripts",
                                            "part_mass_table.json")))["parts"]
        for name, e in densities.items():
            if e["class"] != "printed":
                continue
            assert "print_name" in e, f"{name}: printed entries need print_name"
            pn = e["print_name"]
            if pn is None:
                assert str(e["source"]).startswith("ASSUMED"), (
                    f"{name} declares print_name null but its source does not "
                    "explain why it is priced from mesh volume")
            else:
                assert pn in table, (
                    f"{name} maps to print name {pn!r}, which is not in "
                    "part_mass_table.json. The MJCF/print name mapping is NOT "
                    "the identity.")

    def test_antenna_is_printed_not_bought(self, densities):
        """left_antenna_holder's declared inertia only reproduces when BOTH of
        its meshes sit at ~1250 kg/m3. Classing `antenna` as bought breaks the
        anchor."""
        assert densities["antenna"]["class"] == "printed"


@pytest.mark.phase1
class TestWrittenModel:

    def test_all_bodies_satisfy_triangle_inequality(self):
        """Duplicates an assertion in test_mass_inertia.py on purpose: the
        composer is the new thing that can break it."""
        import mujoco
        m = mujoco.MjModel.from_xml_path(os.path.join(SIM, "scene.xml"))
        real = 0
        for i in range(m.nbody):
            mass = m.body_mass[i]
            if mass < 1e-6:
                continue
            real += 1
            a, b, c = sorted(m.body_inertia[i])
            assert a > 0, f"body {i} has a non-positive principal moment"
            assert a + b >= c - 1e-12, (
                f"body {i} violates the triangle inequality: {(a, b, c)}")
        assert real == 17, f"{real} real-inertial bodies, expected 17"

    def test_marker_frames_are_untouched(self):
        """verify_known_issues.py PLANT-1b asserts these keep their 1e-9 kg
        placeholder; the composer must skip them."""
        root = ET.parse(MJCF).getroot()
        for b in root.iter("body"):
            if b.get("name") not in ("trunk", "left_foot", "head", "right_foot"):
                continue
            it = b.find("inertial")
            assert it is not None
            assert abs(float(it.get("mass")) - 1e-09) < 1e-15, (
                f"{b.get('name')} is a marker frame and must stay at 1e-9 kg")

    def test_fixtures_match_the_model(self):
        import mujoco
        fx = json.load(open(os.path.join(REPO_ROOT, "tests", "fixtures",
                                         "expected_values.json")))
        m = mujoco.MjModel.from_xml_path(os.path.join(SIM, "scene.xml"))
        assert abs(sum(m.body_mass) - fx["total_mass_kg"]) < 1e-6
        for body, key in (("trunk_assembly", "trunk_assembly_mass_kg"),
                          ("head_assembly", "head_assembly_mass_kg")):
            assert abs(m.body_mass[m.body(body).id] - fx[key]) < 1e-6
