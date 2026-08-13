"""Task M3 — six 18650 cells must be real geometry, and they must fit.

The robot is specified as 6x 18650 in 3S2P and `compute_trunk_inertial.py` has
always booked 180 g for the four "extra" cells, but only TWO existed as
geometry. So 180 g of the densest payload in the machine sat at a coordinate no
geometry occupied, and every clearance number was computed against a bay four
cells emptier than the build.

These tests execute booleans and parse XML. The fit test in particular is an
exact boolean, not a docstring claim: `tests/test_cad_dimensions.py` compares
AABBs and has to whitelist a cell sitting in its bore as legitimate nesting,
which means it structurally cannot tell containment from collision. This file
is where that gets proved.
"""

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SIM = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")
MJCF = os.path.join(SIM, "robot_motors.xml")
ROBOT_XML = os.path.join(SIM, "robot.xml")
URDF = os.path.join(SIM, "robot.urdf")

# Visual-only singletons: mesh geoms in robot_motors.xml with no collision twin.
VISUAL_ONLY = {"jetson_orin_nano", "thermal_partition", "dcdc_converter"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gbp():
    return _load("gbp", os.path.join(REPO_ROOT, "scripts",
                                     "generate_battery_pack.py"))


@pytest.fixture(scope="module")
def mbb():
    return _load("mbb", os.path.join(REPO_ROOT, "scripts",
                                     "measure_battery_bay.py"))


@pytest.mark.phase3
class TestCellsAreModelled:

    def test_six_cells_are_modelled(self):
        root = ET.parse(MJCF).getroot()
        cells = [g for g in root.iter("geom") if g.get("mesh") == "cell"]
        assert len(cells) == 12, (
            f"{len(cells)} `cell` geoms in robot_motors.xml, expected 12 "
            "(six poses x collision+visual). Adding only six would make the "
            "new cells collision-only.")
        r2 = ET.parse(ROBOT_XML).getroot()
        assert len([g for g in r2.iter("geom") if g.get("mesh") == "cell"]) == 6, (
            "robot.xml carries ONE geom per pose, not twins")

    def test_holder_6cell_is_declared_and_placed(self):
        root = ET.parse(MJCF).getroot()
        meshes = [m.get("name") for m in root.iter("mesh")]
        assert "holder_6cell" in meshes
        assert len(meshes) == 48, f"{len(meshes)} asset meshes, expected 48"
        trays = [g for g in root.iter("geom") if g.get("mesh") == "holder_6cell"]
        assert len(trays) == 2, "tray needs a collision geom and a visual twin"

    def test_urdf_mirrors_the_mjcf(self):
        txt = open(URDF).read()
        assert txt.count('<mesh filename="cell.stl"/>') == 12, (
            "URDF needs six <visual> + six <collision> cell references")
        assert txt.count('<mesh filename="holder_6cell.stl"/>') == 2

    def test_every_mesh_geom_has_a_visual_twin(self):
        """Pose-level twinning, keyed on the BODY as well as the mesh.

        De-duplicating on (mesh, pos, quat) without the body name collapses the
        mirrored legs together and would hide a whole leg's worth of geometry.
        """
        root = ET.parse(MJCF).getroot()
        groups = {}
        for body in root.iter("body"):
            for g in body.findall("geom"):
                if g.get("type") != "mesh":
                    continue
                key = (body.get("name"), g.get("mesh"),
                       g.get("pos") or "", g.get("quat") or "")
                groups.setdefault(key, []).append(g)

        singles = {k: v for k, v in groups.items() if len(v) != 2}
        for (body, mesh, _, _), v in singles.items():
            assert mesh in VISUAL_ONLY, (
                f"{mesh} on {body} has {len(v)} geom(s) at one pose; expected a "
                "collision geom plus a visual twin")
        for k, v in groups.items():
            if len(v) != 2:
                continue
            flags = [g.get("contype") == "0" for g in v]
            assert sum(flags) == 1, (
                f"{k[1]} on {k[0]}: exactly one of the pair must be visual-only")

        # Derived, not assumed. 133 unique (body, mesh, pose) instances before
        # M3; this task removed the 2 old `cell` poses and the old `holder`
        # pose and added 6 `cell` poses plus 1 `holder_6cell`:
        #     133 - 3 + 7 = 137
        # The task plan predicted 140 on the assumption that the old 2-cell
        # `holder` geom would be RETAINED alongside the new tray. It is not:
        # a robot has one battery tray, and keeping both would add a phantom
        # 7.5 cm^3 part to the collision model. The `holder` mesh asset and
        # holder.stl both remain, for the 2-cell configuration.
        assert len(groups) == 137, (
            f"{len(groups)} unique (body, mesh, pose) instances on the robot, "
            "expected 137. Recompute from the edit rather than trusting this "
            "literal if the model changed.")


@pytest.mark.phase3
class TestPackFits:

    def test_pack_does_not_intersect_the_shells(self, mbb):
        """Exact boolean against every shell and payload part, as authored."""
        geoms = mbb.trunk_geoms()
        cells = [g for g in geoms if g["mesh"] == "cell"]
        assert len(cells) == 6
        pack = [mbb.posed("cell", g["pos"], g["quat"]) for g in cells]
        pack += [mbb.posed(g["mesh"], g["pos"], g["quat"]) for g in geoms
                 if g["mesh"].startswith("holder")]
        worst = mbb.fit_check(pack, geoms, verbose=False)
        assert worst < 5.0, (
            f"worst interference {worst:.2f} mm^3. Re-run "
            "`python3 scripts/measure_battery_bay.py --from-mjcf` to see which "
            "part.")

    def test_pack_is_laterally_symmetric(self, gbp):
        """A biped's hard axis is frontal-plane balance. The pack is 270 g of
        the densest payload on the robot; an asymmetric one biases it forever.
        """
        ys = sorted(c[1] for c in gbp.cell_centres_mm())
        assert abs(sum(ys)) < 1e-9, f"cell y-centres do not cancel: {ys}"
        assert abs(gbp.PACK_ORIGIN_M[1]) < 1e-12


@pytest.mark.phase3
class TestHolderGeometry:

    def test_holder_6cell_is_manifold(self):
        m = trimesh.load(os.path.join(SIM, "holder_6cell.stl"), force="mesh")
        assert m.is_watertight
        assert len(m.split(only_watertight=False)) == 1

    def test_holder_6cell_is_binary_stl(self):
        """tests/test_cad_dimensions.py::_load_tris asserts
        len(data) == 84 + 50*n and fails loudly on ASCII."""
        import struct
        for path in (os.path.join(SIM, "holder_6cell.stl"),
                     os.path.join(REPO_ROOT, "print", "holder_6cell.stl")):
            data = open(path, "rb").read()
            n = struct.unpack("<I", data[80:84])[0]
            assert len(data) == 84 + 50 * n, f"{path} is not binary STL"

    def test_holder_accepts_six_cells(self, gbp):
        """A Ø18.0 probe at each nominal bore centre must pass through the
        tray without touching it. Ø18.0 is the modelled cell; the bore is
        Ø18.6 because a real wrapped 18650 is 18.3-18.6 mm."""
        tray = trimesh.load(os.path.join(REPO_ROOT, "print",
                                         "holder_6cell.stl"), force="mesh")
        for i in range(gbp.ROWS):
            for j in range(gbp.COLS):
                probe = trimesh.creation.cylinder(
                    radius=gbp.CELL_DIA_MM / 2.0, height=60.0, sections=64)
                probe.apply_translation([
                    (i - (gbp.ROWS - 1) / 2) * gbp.PITCH_MM,
                    (j - (gbp.COLS - 1) / 2) * gbp.PITCH_MM,
                    gbp.FLOOR_MM + 30.0])
                inter = trimesh.boolean.boolean_manifold([probe, tray],
                                                         "intersection")
                vol = 0.0 if inter.is_empty else abs(inter.volume)
                assert vol < 1.0, (
                    f"bore ({i},{j}) fouls a Ø{gbp.CELL_DIA_MM} cell by "
                    f"{vol:.3f} mm^3")

    def test_webs_are_printable(self, gbp):
        """At Ø18.6 a 19.0 mm pitch leaves a 0.4 mm web, which does not print."""
        web = gbp.PITCH_MM - gbp.BORE_DIA_MM
        assert web >= 2 * gbp.MIN_WALL_MM, (
            f"web is {web:.2f} mm, below 2 x {gbp.MIN_WALL_MM} mm")

    def test_generator_is_deterministic(self, gbp):
        """Re-running must reproduce byte-identical STLs, or the mass table and
        the USD mesh manifest churn on every run."""
        import hashlib
        before = hashlib.md5(
            open(os.path.join(REPO_ROOT, "print", "holder_6cell.stl"), "rb")
            .read()).hexdigest()
        tray = gbp.build_tray()
        import io
        data = tray.export(file_type="stl")
        assert hashlib.md5(data).hexdigest() == before, (
            "build_tray() no longer reproduces the committed STL")


@pytest.mark.phase3
def test_cell_mass_is_booked_exactly_once():
    """The 180 g lump and six real cells must not both count.

    Until M4's composer takes cell mass from geometry, the lump is the only
    booking and must carry a TODO(M4). After M4 it must be gone. Either state
    is fine; both at once is a 180 g double-count.
    """
    src = open(os.path.join(REPO_ROOT, "scripts",
                            "compute_trunk_inertial.py")).read()
    n = src.count('"cells_extra_4x45g"')
    assert n <= 1, f"cells_extra_4x45g appears {n} times"
    if n == 1:
        idx = src.index('"cells_extra_4x45g"')
        window = src[max(0, idx - 1200):idx]
        assert "TODO(M4)" in window, (
            "the 180 g cell lump survives but carries no TODO(M4). Six cells "
            "are real geometry now; without the marker M4 will double-book "
            "them.")
