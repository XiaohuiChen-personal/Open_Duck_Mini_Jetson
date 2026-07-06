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
        ("battery_pack_lid", "cell"),
        ("battery_pack_lid", "cell_2"),
        ("battery_pack_lid", "holder"),
        ("cell", "cell_2"),
        ("cell", "holder"),
        ("cell_2", "holder"),
        ("bms", "holder"),
        ("bms", "cell"),
        ("bms", "cell_2"),
        ("usb_c_charger", "holder"),
        ("usb_c_charger", "bms"),
        ("usb_c_charger", "cell"),
        ("usb_c_charger", "cell_2"),
        ("power_switch", "holder"),
    ]
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
        assert lo[2] <= -0.0230, "partition must reach the cavity floor (seal)"

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

    def test_trunk_bottom_intrusion_within_declared_spine_cut(self, boxes):
        pts = self._jetson_grid(boxes)
        inside = _mesh_points_inside("trunk_bottom", pts)
        lo, hi = SPINE_CUT_BOX
        outside_cut = [
            p for p in inside if not (np.all(p >= lo) and np.all(p <= hi))
        ]
        assert not outside_cut, (
            f"Jetson intersects trunk_bottom OUTSIDE the declared spine-cut box: "
            f"{np.array(outside_cut)}"
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

    def test_trunk_inertial_matches_generator(self):
        """robot.xml trunk inertial must equal scripts/compute_trunk_inertial.py output."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cti", os.path.join(REPO_ROOT, "scripts", "compute_trunk_inertial.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        masses = [mod.BASE["mass"]] + [m for _, m, _, _ in mod.COMPONENTS]
        positions = [mod.BASE["com"]] + [np.asarray(p) for _, _, p, _ in mod.COMPONENTS]
        total = sum(masses) + mod.WIRING_MASS
        com = sum(m * np.asarray(p) for m, p in zip(masses, positions)) / sum(masses)
        inertias = [mod.BASE["inertia"]] + [np.asarray(i) for _, _, _, i in mod.COMPONENTS]
        I = np.zeros(3)
        for m, p, Ic in zip(masses, positions, inertias):
            d = np.asarray(p) - com
            I += Ic + m * np.array(
                [d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2]
            )
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.xml"))
        trunk = tree.getroot().find(".//body[@name='trunk_assembly']")
        inertial = trunk.find("inertial")
        assert abs(float(inertial.get("mass")) - total) < 1e-6
        xml_com = np.array([float(v) for v in inertial.get("pos").split()])
        xml_I = np.array([float(v) for v in inertial.get("diaginertia").split()])
        assert np.allclose(xml_com, com, atol=1e-6), f"CoM drift: {xml_com} vs {com}"
        assert np.allclose(xml_I, I, atol=1e-7), f"inertia drift: {xml_I} vs {I}"
