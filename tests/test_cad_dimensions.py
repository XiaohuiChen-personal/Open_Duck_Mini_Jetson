"""Phase 3 CAD layout regression tests.

Verifies the component layout from docs/jetson-mod/component_layout_v2.md:
no component-vs-component interference, required clearances, chassis-mesh
compatibility, and position consistency across the three model files.

Geometry is measured from the actual meshes + MJCF poses (never hardcoded
from docs), so any future layout edit is re-validated mechanically.
"""
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_DIR = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")

# Pairs allowed to overlap: containers and their contents (the holder/lid
# physically enclose the cells and BMS), plus hump-wall mounted parts.
NESTING_WHITELIST = {
    frozenset(p)
    for p in [
        ("battery_pack_lid", "bms"),
        ("battery_pack_lid", "holder"),
        ("bms", "holder"),
        ("usb_c_charger", "holder"),
        ("usb_c_charger", "bms"),
        ("power_switch", "holder"),
    ]
    # Task M3 (2026-08-12): the pack went 2 -> 6 cells, so the cell names
    # `_trunk_geoms()` derives from robot.xml are now cell .. cell_6, and the
    # tray they nest in is `holder_6cell`.
    #
    # A cell sitting in its bore IS an AABB overlap -- 7.13 cm^3 each, measured
    # -- and that is containment, not collision. This test compares AABBs and
    # cannot tell the two apart, which is the whole reason this whitelist
    # exists. The exact-boolean proof is `scripts/measure_battery_bay.py
    # --from-mjcf`, which reports **0.00 mm^3** for this layout and is run by
    # tests/test_battery_pack.py.
} | {
    frozenset(p)
    for p in (
        [(f"cell{'' if i == 1 else f'_{i}'}", "holder_6cell") for i in range(1, 7)]
        + [(f"cell{'' if i == 1 else f'_{i}'}", other)
           for i in range(1, 7)
           for other in ("battery_pack_lid", "bms", "usb_c_charger")]
        + [(f"cell{'' if i == 1 else f'_{i}'}", f"cell_{j}")
           for i in range(1, 7) for j in range(2, 7) if i < j]
        + [("holder_6cell", other) for other in
           ("bms", "usb_c_charger", "power_switch", "battery_pack_lid", "holder")]
    )
}

# The robot's own structure: excluded from the component pairwise check
# (chassis/shells checked separately at mesh level; servo cases, bearings and
# horns are the original mechanism, not payload).
STRUCTURE_PREFIXES = (
    "body_",
    "trunk_",
    "roll_bearing",
    "wj-",
    "drive_palonier",
    "passive_palonier",
)

# Declared Part-2 trunk_bottom spine cut (body frame, m): the only region
# where the Jetson envelope may still intersect chassis material until the
# mesh cut lands. Tightened to zero by the Part-2 mesh update.
SPINE_CUT_BOX = (np.array([-0.044, -0.016, -0.0136]), np.array([0.021, 0.017, 0.051]))

SHELL_OFFSET = np.array([-0.019, 0.0, 0.0648909])


def _load_tris(mesh_name):
    path = os.path.join(ROBOT_DIR, f"{mesh_name}.stl")
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    assert len(data) == 84 + 50 * n, f"{mesh_name}.stl is not binary STL"
    arr = np.frombuffer(data[84:], dtype=np.uint8).reshape(n, 50)
    return arr[:, :48].copy().view("<f4").reshape(n, 4, 3)[:, 1:, :].astype(np.float64)


def _quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _trunk_geoms():
    """name -> (mesh, pos, quat) for every mesh geom on trunk_assembly."""
    tree = ET.parse(os.path.join(ROBOT_DIR, "robot.xml"))
    trunk = tree.getroot().find(".//body[@name='trunk_assembly']")
    out, counts = {}, {}
    for geom in trunk.findall("geom"):
        mesh = geom.get("mesh")
        if mesh is None:
            continue
        pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
        quat = np.array([float(v) for v in geom.get("quat", "1 0 0 0").split()])
        counts[mesh] = counts.get(mesh, 0) + 1
        name = mesh if counts[mesh] == 1 else f"{mesh}_{counts[mesh]}"
        out[name] = (mesh, pos, quat)
    return out


def _aabb(mesh, pos, quat):
    tris = _load_tris(mesh)
    verts = tris.reshape(-1, 3)
    corners = np.array(
        [[x, y, z] for x in (verts[:, 0].min(), verts[:, 0].max())
         for y in (verts[:, 1].min(), verts[:, 1].max())
         for z in (verts[:, 2].min(), verts[:, 2].max())]
    )
    world = corners @ _quat_mat(quat).T + pos
    return world.min(axis=0), world.max(axis=0)


def _overlap_vol(a, b):
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    return float(np.prod(np.maximum(hi - lo, 0.0)))


def _z_crossings(tris, x, y):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    a, b, c = v0[:, :2], v1[:, :2], v2[:, :2]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(det) > 1e-14
    safe = np.where(ok, det, 1.0)
    l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / safe
    l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / safe
    l3 = 1 - l1 - l2
    hit = ok & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
    if not hit.any():
        return []
    zs = np.sort(l1[hit] * v0[hit, 2] + l2[hit] * v1[hit, 2] + l3[hit] * v2[hit, 2])
    dedup = [zs[0]]
    for z in zs[1:]:
        if z - dedup[-1] > 1e-7:
            dedup.append(z)
    return dedup


def _mesh_points_inside(mesh_name, points):
    """Subset of points strictly inside the (body-frame) chassis mesh."""
    tris = _load_tris(mesh_name) + SHELL_OFFSET
    inside = []
    for p in points:
        zs = _z_crossings(tris, p[0], p[1])
        n_above = sum(1 for z in zs if z > p[2])
        if n_above % 2 == 1:
            inside.append(p)
    return np.array(inside) if inside else np.empty((0, 3))


@pytest.fixture(scope="module")
def geoms():
    return _trunk_geoms()


@pytest.fixture(scope="module")
def boxes(geoms):
    return {name: _aabb(*g) for name, g in geoms.items()}


@pytest.mark.phase3
class TestComponentLayout:
    def test_jetson_clear_of_all_components(self, boxes):
        """The Jetson envelope must not intersect ANY other trunk geom AABB
        except the chassis meshes handled at mesh level below."""
        jet = boxes["jetson_orin_nano"]
        offenders = []
        for name, bb in boxes.items():
            if name == "jetson_orin_nano" or name.startswith(("body_", "trunk_")):
                continue
            vol = _overlap_vol(jet, bb)
            if vol > 1e-9:
                offenders.append((name, vol * 1e6))
        assert not offenders, f"Jetson envelope overlaps: {offenders} (cm^3)"

    def test_component_pairwise_no_overlap(self, boxes):
        comp = {
            k: v for k, v in boxes.items() if not k.startswith(STRUCTURE_PREFIXES)
        }
        names = sorted(comp)
        offenders = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if frozenset((a, b)) in NESTING_WHITELIST:
                    continue
                vol = _overlap_vol(comp[a], comp[b])
                if vol > 1e-8:
                    offenders.append((a, b, vol * 1e6))
        assert not offenders, f"Component overlaps (cm^3): {offenders}"

    def test_jetson_cavity_clearances(self, boxes):
        lo, hi = boxes["jetson_orin_nano"]
        assert lo[2] >= -0.02388 + 0.002, "Jetson below floor clearance"
        assert lo[1] >= -0.052 + 0.006 and hi[1] <= 0.052 - 0.006, "Jetson side margin < 6 mm"
        assert hi[0] <= 0.036 - 0.005, "Jetson front margin < 5 mm"
        # below the trunk_top plate (fan intake plenum)
        assert hi[2] <= 0.0448 - 0.005, "Jetson top must clear trunk_top plate by >= 5 mm"
        # above the roll bearing housings
        bearing_top = max(boxes["roll_bearing"][1][2], boxes["roll_bearing_2"][1][2])
        assert lo[2] >= bearing_top + 0.0015, "Jetson bottom must clear roll bearings"

    def test_partition_geometry(self, boxes):
        lo, hi = boxes["thermal_partition"]
        jet_lo = boxes["jetson_orin_nano"][0]
        assert hi[0] <= jet_lo[0] - 0.002, "partition must sit >= 2 mm behind the Jetson"
        assert lo[0] >= -0.114 + 0.002, "partition must clear the battery lid"
        assert hi[1] - lo[1] <= 0.104, "partition wider than shell interior (104 mm)"
        assert hi[2] <= 0.0448, "partition must stop at the trunk_top plate underside"
        # Local floor top at the partition plane (x=-0.086) is z=-0.02238
        # (mesh-measured; 1.5 mm higher than the Jetson-zone floor). Seal by
        # near-contact without penetrating the slab.
        assert lo[2] <= -0.0222, "partition must reach the local cavity floor (seal)"
        assert lo[2] >= -0.0226, "partition must not penetrate the floor slab"

    def test_partition_clear_of_relocated_components(self, boxes):
        part = boxes["thermal_partition"]
        for name in ["bno055", "board", "jetson_orin_nano", "dcdc_converter"]:
            assert _overlap_vol(part, boxes[name]) < 1e-9, f"partition intersects {name}"

    def test_imu_in_battery_side_pocket(self, boxes):
        lo, hi = boxes["bno055"]
        part_rear = boxes["thermal_partition"][0][0]
        assert hi[0] <= part_rear - 0.002, "IMU must be behind the partition"
        assert lo[0] >= -0.114, "IMU must be in front of the battery lid"

    def test_dcdc_under_plate_mount(self, boxes):
        lo, hi = boxes["dcdc_converter"]
        jet_hi = boxes["jetson_orin_nano"][1]
        assert lo[2] >= jet_hi[2] + 0.004, "DC-DC must clear the Jetson top by >= 4 mm"
        assert hi[2] <= 0.0448, "DC-DC must fit under the trunk_top plate"


@pytest.mark.phase3
class TestChassisMeshCompatibility:
    GRID_STEP = 0.005

    def _jetson_grid(self, boxes):
        lo, hi = boxes["jetson_orin_nano"]
        xs = np.arange(lo[0] + 1e-4, hi[0], self.GRID_STEP)
        ys = np.arange(lo[1] + 1e-4, hi[1], self.GRID_STEP)
        zs = np.arange(lo[2] + 1e-4, hi[2], self.GRID_STEP)
        return np.array([[x, y, z] for x in xs for y in ys for z in zs])

    def test_trunk_top_untouched_by_jetson(self, boxes):
        pts = self._jetson_grid(boxes)
        # trunk_top solid starts at z=0.0448; only test the top slab of points
        pts = pts[pts[:, 2] > 0.02]
        inside = _mesh_points_inside("trunk_top", pts)
        assert len(inside) == 0, (
            f"{len(inside)} Jetson grid points inside trunk_top — "
            "the low mount must not require any trunk_top cut"
        )

    def test_trunk_bottom_clear_of_jetson(self, boxes):
        """The Part-2 spine cut landed: the Jetson envelope must now have ZERO
        intersection with trunk_bottom (SPINE_CUT_BOX documents the cut that
        made this true; scripts/generate_cad_mods.py applied it)."""
        pts = self._jetson_grid(boxes)
        inside = _mesh_points_inside("trunk_bottom", pts)
        assert len(inside) == 0, (
            f"{len(inside)} Jetson grid points inside trunk_bottom after the "
            f"spine cut: {inside[:5]}"
        )


# Partition bottom corners may clip the body_middle_bottom wall fillets until
# the Part-2 printable part chamfers them (declared allowance, body frame):
# the shell side walls curve inward below z ~ +3 mm.
PARTITION_CHAMFER_BOXES = [
    (np.array([-0.0880, -0.0525, -0.0230]), np.array([-0.0840, -0.0410, 0.0040])),
    (np.array([-0.0880, 0.0410, -0.0230]), np.array([-0.0840, 0.0525, 0.0040])),
]


@pytest.mark.phase3
class TestRelocatedComponentsMeshLevel:
    """Every relocated/added component must be mesh-level clear of the chassis
    and shells (the reviewer found the first-pass board move embedded in solid
    trunk_top — this class prevents that failure mode for all of them)."""

    GRID_STEP = 0.004
    CHASSIS = ["trunk_top", "trunk_bottom", "body_middle_bottom", "body_middle_top"]

    def _box_grid(self, boxes, name):
        lo, hi = boxes[name]
        axes = [np.arange(lo[i] + 5e-4, hi[i], self.GRID_STEP) for i in range(3)]
        axes = [a if len(a) else np.array([(lo[i] + hi[i]) / 2]) for i, a in enumerate(axes)]
        return np.array([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]])

    @pytest.mark.parametrize("component", ["board", "dcdc_converter", "bno055"])
    def test_component_clear_of_chassis(self, boxes, component):
        pts = self._box_grid(boxes, component)
        for mesh in self.CHASSIS:
            inside = _mesh_points_inside(mesh, pts)
            assert len(inside) == 0, (
                f"{component} has {len(inside)} grid points inside {mesh}: "
                f"{inside[:5]}"
            )

    def test_partition_clear_of_chassis_except_chamfer_corners(self, boxes):
        pts = self._box_grid(boxes, "thermal_partition")
        for mesh in self.CHASSIS:
            inside = _mesh_points_inside(mesh, pts)
            outside_allowance = [
                p for p in inside
                if not any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in PARTITION_CHAMFER_BOXES)
            ]
            assert not outside_allowance, (
                f"partition has {len(outside_allowance)} points inside {mesh} outside "
                f"the declared chamfer corners: {np.array(outside_allowance)[:5]}"
            )


@pytest.mark.phase3
class TestModelFileConsistency:
    """The five relocated components must agree across all three model files."""

    COMPONENTS = ["jetson_orin_nano", "thermal_partition", "dcdc_converter", "bno055", "board"]

    def _mjcf_positions(self, fname):
        tree = ET.parse(os.path.join(ROBOT_DIR, fname))
        trunk = tree.getroot().find(".//body[@name='trunk_assembly']")
        pos = {}
        for geom in trunk.findall("geom"):
            mesh = geom.get("mesh")
            if mesh in self.COMPONENTS and mesh not in pos:
                pos[mesh] = np.array([float(v) for v in geom.get("pos").split()])
        return pos

    def _urdf_positions(self):
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.urdf"))
        body = next(l for l in tree.getroot().findall("link") if l.get("name") == "trunk_assembly")
        pos = {}
        for vis in body.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            mesh = os.path.basename(mesh_el.get("filename")).replace(".stl", "")
            if mesh in self.COMPONENTS and mesh not in pos:
                origin = vis.find("origin")
                pos[mesh] = np.array([float(v) for v in origin.get("xyz").split()])
        return pos

    def test_positions_consistent_across_model_files(self):
        a = self._mjcf_positions("robot.xml")
        b = self._mjcf_positions("robot_motors.xml")
        c = self._urdf_positions()
        for comp in self.COMPONENTS:
            assert comp in a and comp in b and comp in c, f"{comp} missing from a model file"
            assert np.allclose(a[comp], b[comp], atol=1e-6), f"{comp}: robot.xml != robot_motors.xml"
            assert np.allclose(a[comp], c[comp], atol=1e-6), f"{comp}: robot.xml != robot.urdf"

    @staticmethod
    def _load_generator():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cti", os.path.join(REPO_ROOT, "scripts", "compute_trunk_inertial.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _xml_inertial(body_name):
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.xml"))
        body = tree.getroot().find(f".//body[@name='{body_name}']")
        inertial = body.find("inertial")
        com = np.array([float(v) for v in inertial.get("pos").split()])
        fi = [float(v) for v in inertial.get("fullinertia").split()]
        tensor = np.array(
            [[fi[0], fi[3], fi[4]], [fi[3], fi[1], fi[5]], [fi[4], fi[5], fi[2]]]
        )
        return float(inertial.get("mass")), com, tensor

    @pytest.mark.parametrize(
        "body,base,components,wiring",
        [
            pytest.param(
                "trunk_assembly", "BASE_TRUNK", "TRUNK_COMPONENTS",
                "TRUNK_WIRING",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "EXPECTED DIVERGENCE, Task M2 -> M4. M2 (2026-08-12) "
                        "fixed PLANT-10: the Part-2 CAD deltas are now "
                        "whole-part MEASURED instead of one assumed density, "
                        "so the generator emits 1.164076 kg where the model "
                        "files still declare 1.089544 kg -- a gap of exactly "
                        "0.074532 kg, the PLANT-10 error. M2 deliberately does "
                        "NOT write the model files: Task M4 rewrites every "
                        "body's inertial from one composer and consumes M2's "
                        "output, and writing them twice would guarantee the "
                        "two disagree. strict=True, so this flips to a FAILURE "
                        "the moment M4 lands and the marker must then be "
                        "deleted. Do not 'fix' this by loosening the "
                        "tolerance."
                    ),
                ),
            ),
            ("head_assembly", "BASE_HEAD", "HEAD_COMPONENTS", "HEAD_WIRING"),
        ],
    )
    def test_inertial_matches_generator(self, body, base, components, wiring):
        """robot.xml inertials must equal the full-tensor generator output.

        The generator's upstream base tensors are themselves frame-proofed at
        import time (URDF full matrix == MJCF quat*diag*quatT self-check), so
        this guards both value drift and the frame-permutation bug class.
        """
        mod = self._load_generator()
        mod.check_base(getattr(mod, base), body)
        total, com, I = mod.compose(
            getattr(mod, base), getattr(mod, components), getattr(mod, wiring)
        )
        xml_mass, xml_com, xml_I = self._xml_inertial(body)
        assert abs(xml_mass - total) < 1e-6
        assert np.allclose(xml_com, com, atol=1e-6), f"{body} CoM drift: {xml_com} vs {com}"
        assert np.allclose(xml_I, I, atol=1e-7), f"{body} tensor drift:\n{xml_I}\nvs\n{I}"

    def test_inertial_principal_moments_match_fixture(self):
        """Independent cross-check: eigenvalues of the XML full tensors must
        equal the fixture's principal moments (which were derived during the
        adversarial review from the upstream URDF, independently of the
        generator script)."""
        import json

        fixture = json.load(
            open(os.path.join(REPO_ROOT, "tests", "fixtures", "expected_values.json"))
        )
        for body, key in [
            ("trunk_assembly", "trunk_assembly_diaginertia"),
            ("head_assembly", "head_assembly_diaginertia"),
        ]:
            _, _, tensor = self._xml_inertial(body)
            eig = np.sort(np.linalg.eigvalsh(tensor))
            expected = np.sort(fixture[key])
            assert np.allclose(eig, expected, atol=1e-7), (
                f"{body} principal moments {eig} != fixture {expected}"
            )
