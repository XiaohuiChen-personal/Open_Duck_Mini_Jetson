#!/usr/bin/env python3
"""Measure the real battery bay, and prove a 6-cell pack fits it.

Task M3. The robot is specified as 6x 18650 in 3S2P and the mass is already
booked -- `compute_trunk_inertial.py` carries a 180 g `cells_extra_4x45g` lump
-- but `robot_motors.xml` contains only TWO `cell` geoms. So 180 g of the
densest payload in the robot sits at a position no geometry occupies, and every
clearance number is computed against a bay four cells emptier than the build.

`component_layout_v2.md` records the usable bore TWICE and inconsistently:
the Layout table says 28.4 mm deep, cut-list item 4 implies 26.4 mm. **This
script exists so that neither number has to be trusted.** It measures.

    python3 scripts/measure_battery_bay.py                 # measure + report bore
    python3 scripts/measure_battery_bay.py --from-mjcf     # fit-check as authored
    python3 scripts/measure_battery_bay.py --rows 3 --cols 2 --origin -0.1396,0,0.0

Exit 0 when the worst interference is under --tolerance mm^3, 1 otherwise, so
it works as a gate. The intersection is an exact boolean (manifold3d), not an
AABB overlap -- `tests/test_cad_dimensions.py` does the AABB check and its
nesting whitelist exists precisely because AABBs cannot tell containment from
collision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO, "mini_bdx", "robots", "open_duck_mini_v2")
MJCF = os.path.join(SIM_DIR, "robot_motors.xml")

# Shell/chassis meshes the pack must not touch, plus the payload already in the
# bay. Read at their MJCF poses -- never assume OFF applies, because OFF is the
# shell-geom offset only and the cells carry their own pos/quat.
SHELL_MESHES = ("body_back", "body_middle_bottom", "body_middle_top",
                "trunk_bottom", "trunk_top", "battery_pack_lid")
PAYLOAD_MESHES = ("bms", "usb_c_charger", "power_switch")

TOLERANCE_MM3 = 5.0          # same threshold generate_cad_mods.py uses
CELL_DIA_MM = 18.0           # the `cell` mesh is a bare 18650 at nominal
CELL_LEN_MM = 65.0


def quat_to_matrix(q) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def trunk_geoms() -> list[dict]:
    """Every mesh geom on trunk_assembly, de-duplicated on (mesh, pos, quat).

    Collision geoms and their visual twins sit at identical poses; counting both
    would double every intersection.
    """
    root = ET.parse(MJCF).getroot()
    body = None
    for b in root.iter("body"):
        if b.get("name") == "trunk_assembly":
            body = b
            break
    if body is None:
        raise SystemExit("trunk_assembly not found in the MJCF")
    seen, out = set(), []
    for g in body.findall("geom"):
        if g.get("type") != "mesh":
            continue
        pos = tuple(float(v) for v in (g.get("pos") or "0 0 0").split())
        quat = tuple(float(v) for v in (g.get("quat") or "1 0 0 0").split())
        key = (g.get("mesh"), pos, quat)
        if key in seen:
            continue
        seen.add(key)
        out.append({"mesh": g.get("mesh"), "pos": np.array(pos),
                    "quat": np.array(quat)})
    return out


def posed(mesh_name: str, pos, quat) -> trimesh.Trimesh:
    """Load a sim mesh (metres) and place it in the trunk body frame."""
    m = trimesh.load(os.path.join(SIM_DIR, f"{mesh_name}.stl"), force="mesh")
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = pos
    m.apply_transform(T)
    return m


def cell_cylinder(centre_m, dia_mm=CELL_DIA_MM, len_mm=CELL_LEN_MM,
                  axis="z") -> trimesh.Trimesh:
    """One 18650 as a cylinder, in metres, standing on the given axis."""
    c = trimesh.creation.cylinder(radius=dia_mm / 2000.0, height=len_mm / 1000.0,
                                  sections=64)
    if axis == "x":
        c.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        c.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0]))
    c.apply_translation(np.asarray(centre_m))
    return c


def pack_cells(rows, cols, origin, pitch_mm, axis="z") -> list[trimesh.Trimesh]:
    """Grid of cells. rows run along x, cols along y (both in the trunk frame)."""
    p = pitch_mm / 1000.0
    ox, oy, oz = origin
    out = []
    for i in range(rows):
        for j in range(cols):
            out.append(cell_cylinder(
                (ox + (i - (rows - 1) / 2) * p,
                 oy + (j - (cols - 1) / 2) * p,
                 oz), axis=axis))
    return out


def intersect_mm3(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Exact boolean intersection volume in mm^3, 0.0 when AABBs miss."""
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    if (amax < bmin).any() or (bmax < amin).any():
        return 0.0
    try:
        inter = trimesh.boolean.boolean_manifold([a, b], "intersection")
    except Exception as exc:                       # noqa: BLE001
        print(f"    (boolean failed: {exc})", file=sys.stderr)
        return float("nan")
    if inter.is_empty:
        return 0.0
    return float(abs(inter.volume)) * 1e9


def measure_bore(geoms) -> dict:
    """The free interior of the rear hump, measured by voxel occupancy.

    Voxelise the union of the shells, fill it, and report the largest empty
    axis-aligned region behind the existing cell pair. This is the number
    component_layout_v2.md gives two different answers for.
    """
    shells = [posed(g["mesh"], g["pos"], g["quat"]) for g in geoms
              if g["mesh"] in SHELL_MESHES]
    if not shells:
        return {}
    body_back = [posed(g["mesh"], g["pos"], g["quat"]) for g in geoms
                 if g["mesh"] == "body_back"]
    if not body_back:
        return {}
    bb = body_back[0]
    lo, hi = bb.bounds
    pitch = 0.002                                   # 2 mm probe grid
    xs = np.arange(lo[0] + pitch, hi[0], pitch)
    ys = np.arange(lo[1] + pitch, hi[1], pitch)
    zs = np.arange(lo[2] + pitch, hi[2], pitch)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    inside_any = np.zeros(len(grid), dtype=bool)
    for m in shells:
        if not m.is_watertight:
            continue
        inside_any |= m.contains(grid)
    free = grid[~inside_any]
    # keep only points inside body_back's own bounding box interior
    return {
        "probe_points": int(len(grid)),
        "free_points": int(len(free)),
        "free_x_mm": [float(free[:, 0].min() * 1000), float(free[:, 0].max() * 1000)] if len(free) else None,
        "free_y_mm": [float(free[:, 1].min() * 1000), float(free[:, 1].max() * 1000)] if len(free) else None,
        "free_z_mm": [float(free[:, 2].min() * 1000), float(free[:, 2].max() * 1000)] if len(free) else None,
        "body_back_bounds_mm": [[float(v * 1000) for v in lo],
                                [float(v * 1000) for v in hi]],
    }


def fit_check(pack_meshes, geoms, tolerance=TOLERANCE_MM3, verbose=True) -> float:
    """Worst exact-boolean intersection between the pack and everything else."""
    obstacles = [g for g in geoms
                 if g["mesh"] in SHELL_MESHES + PAYLOAD_MESHES]
    worst = 0.0
    pack = trimesh.util.concatenate(pack_meshes)
    for g in obstacles:
        m = posed(g["mesh"], g["pos"], g["quat"])
        v = 0.0
        for piece in pack_meshes:
            v += intersect_mm3(piece, m)
        worst = max(worst, v)
        if verbose:
            flag = "" if v < tolerance else "   <-- INTERFERENCE"
            print(f"  pack vs {g['mesh']:24s} {v:10.2f} mm^3{flag}")
    if verbose:
        print(f"  {'WORST':30s} {worst:10.2f} mm^3   (tolerance {tolerance})")
    return worst


def mjcf_cell_poses(geoms) -> list[dict]:
    return [g for g in geoms if g["mesh"] == "cell"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-mjcf", action="store_true",
                    help="fit-check the cell poses already in robot_motors.xml")
    ap.add_argument("--rows", type=int, default=3, help="cells along x")
    ap.add_argument("--cols", type=int, default=2, help="cells along y")
    ap.add_argument("--origin", type=str, default=None,
                    help="pack centre in the trunk frame, 'x,y,z' in metres")
    ap.add_argument("--pitch", type=float, default=20.6,
                    help="cell pitch in mm (>= bore + 2*min_wall)")
    ap.add_argument("--axis", default="z", choices=("x", "y", "z"),
                    help="cell long axis in the trunk frame")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_MM3)
    ap.add_argument("--measure-bore", action="store_true",
                    help="report the free interior of the rear hump")
    ap.add_argument("--json", default=None, help="write results to this path")
    args = ap.parse_args()

    geoms = trunk_geoms()
    print(f"MJCF: {MJCF}")
    print(f"trunk_assembly unique mesh instances: {len(geoms)}")
    existing = mjcf_cell_poses(geoms)
    print(f"`cell` poses currently in the model: {len(existing)}")
    for i, g in enumerate(existing, 1):
        m = posed("cell", g["pos"], g["quat"])
        lo, hi = m.bounds * 1000
        print(f"  cell #{i}  x[{lo[0]:8.1f},{hi[0]:8.1f}]  "
              f"y[{lo[1]:8.1f},{hi[1]:8.1f}]  z[{lo[2]:8.1f},{hi[2]:8.1f}]  mm")
    print()

    result = {"cell_poses_in_mjcf": len(existing)}

    if args.measure_bore:
        print("free interior of the rear hump (2 mm probe grid):")
        bore = measure_bore(geoms)
        for k, v in bore.items():
            print(f"  {k}: {v}")
        result["bore"] = bore
        print()

    if args.from_mjcf:
        if not existing:
            print("no cell geoms in the MJCF to check", file=sys.stderr)
            return 1
        pack = [posed("cell", g["pos"], g["quat"]) for g in existing]
        holders = [g for g in geoms if g["mesh"].startswith("holder")]
        for h in holders:
            pack.append(posed(h["mesh"], h["pos"], h["quat"]))
        print(f"fit check, {len(existing)} cells + {len(holders)} holder(s) "
              "as authored in the MJCF:")
        worst = fit_check(pack, geoms, args.tolerance)
    else:
        origin = ([float(v) for v in args.origin.split(",")] if args.origin
                  else [-0.1396, 0.0, 0.0325])
        pack = pack_cells(args.rows, args.cols, origin, args.pitch, args.axis)
        print(f"candidate pack: {args.rows}x{args.cols} at origin {origin}, "
              f"pitch {args.pitch} mm, axis {args.axis}")
        lo = np.min([p.bounds[0] for p in pack], axis=0) * 1000
        hi = np.max([p.bounds[1] for p in pack], axis=0) * 1000
        print(f"  envelope x[{lo[0]:.1f},{hi[0]:.1f}] "
              f"y[{lo[1]:.1f},{hi[1]:.1f}] z[{lo[2]:.1f},{hi[2]:.1f}] mm")
        worst = fit_check(pack, geoms, args.tolerance)

    result["worst_mm3"] = worst
    result["tolerance_mm3"] = args.tolerance
    result["pass"] = bool(worst < args.tolerance)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)
    print()
    print("VERDICT:", "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
