#!/usr/bin/env python3
"""Author the 6-cell 18650 holder procedurally, and emit its MJCF geom poses.

Task M3. The robot is specified as 6x 18650 in 3S2P and 180 g of cells is
already booked in `compute_trunk_inertial.py`, but only TWO cells exist as
geometry. The stock `holder.stl` is a 76 x 41 x 20 mm two-cell tray: it cannot
be stretched to three-wide, so this authors a NEW part rather than editing it.
`holder.stl` stays, because it is upstream's part and the 2-cell configuration
still references it.

Placement was not guessed. `scripts/measure_battery_bay.py` measured the
pre-M3 bore at 42 (x) x 41 (y) x 69 (z) mm and every candidate layout failed
against it. `generate_cad_mods.py` section 4 then DEEPENED the rear hump
(bore x -167 -> -189 mm), and a 3 (x) x 2 (y) array at 20.6 mm pitch centred on
x = -155 mm measures **0.00 mm^3** of interference against every shell and every
payload part -- better than the shipped 2-cell layout, which interferes with
`battery_pack_lid` by 384.69 mm^3.

Run:
    python3 scripts/generate_battery_pack.py            # write both STLs
    python3 scripts/generate_battery_pack.py --geoms    # print MJCF geom lines

Output is deterministic: re-running produces byte-identical STLs.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO, "mini_bdx", "robots", "open_duck_mini_v2")
PRINT_DIR = os.path.join(REPO, "print")

# ---------------------------------------------------------------- parameters
# A real 18650 is 18.3-18.6 mm across the wrap, not the nominal 18.0 the `cell`
# mesh models, so the POCKET is modelled with clearance and the cell is not.
CELL_DIA_MM = 18.0            # the `cell` mesh, nominal
BORE_DIA_MM = 18.6            # pocket: wrapped-cell max + 0 clearance
MIN_WALL_MM = 1.0             # web between adjacent bores, per side
PITCH_MM = 20.6               # >= BORE_DIA + 2 * MIN_WALL  (18.6 + 2.0)
WALL_MM = 1.5                 # outer wall (see _check_printable)
FLOOR_MM = 2.0                # tray floor under the cells
CELL_LEN_MM = 65.0
TRAY_H_MM = 22.0              # tray height; cells stand proud and the lid caps
ROWS, COLS = 3, 2             # rows along x, columns along y

# Pack centre in the trunk body frame, metres: the centre of the deepened
# bore, x in [-190, -123] mm. measure_battery_bay.py reports 0.00 mm^3 here.
# NOTE the sweep must include the TRAY, not just the cells -- checking cells
# alone gave a 6 mm-wide 'clear' band that the 62.8 mm tray did not share.
PACK_ORIGIN_M = (-0.1565, 0.0, 0.0325)

# The `cell` mesh's local frame: 18 x 18 x 65 mm, axis on z, origin at the
# BOTTOM face (z 0..65), so a geom `pos` places the cell's base, not its centre.
CELL_LOCAL_Z_OFFSET_MM = 0.0


def _check_printable() -> None:
    web = PITCH_MM - BORE_DIA_MM
    assert web >= 2 * MIN_WALL_MM, (
        f"web between bores is {web:.2f} mm, below 2 x min wall "
        f"{2 * MIN_WALL_MM:.2f} mm. At Ø{BORE_DIA_MM} a 19.0 mm pitch would "
        f"leave 0.4 mm, which does not print.")


def cell_centres_mm() -> list[tuple[float, float, float]]:
    """Cell centres in the trunk body frame, millimetres."""
    ox, oy, oz = (v * 1000.0 for v in PACK_ORIGIN_M)
    out = []
    for i in range(ROWS):
        for j in range(COLS):
            out.append((ox + (i - (ROWS - 1) / 2) * PITCH_MM,
                        oy + (j - (COLS - 1) / 2) * PITCH_MM,
                        oz))
    return out


def build_tray() -> trimesh.Trimesh:
    """Tray shell minus six through-bores, in millimetres, centred on origin."""
    _check_printable()
    # Size the tray to the CELLS, not to pitch x count. The naive
    # `ROWS * PITCH + 2 * WALL` adds a full pitch of empty material at each end
    # and gave 65.8 x 45.2 mm, which does not fit a 64 x 44 mm bore.
    outer_x = (ROWS - 1) * PITCH_MM + BORE_DIA_MM + 2 * WALL_MM
    outer_y = (COLS - 1) * PITCH_MM + BORE_DIA_MM + 2 * WALL_MM
    tray = trimesh.creation.box(extents=[outer_x, outer_y, TRAY_H_MM])
    bores = []
    for i in range(ROWS):
        for j in range(COLS):
            c = trimesh.creation.cylinder(radius=BORE_DIA_MM / 2.0,
                                          height=TRAY_H_MM * 2.0, sections=64)
            c.apply_translation([(i - (ROWS - 1) / 2) * PITCH_MM,
                                 (j - (COLS - 1) / 2) * PITCH_MM,
                                 0.0])
            bores.append(c)
    # Keep a floor: shorten the bores so they do not cut through the bottom.
    solid = tray
    for c in bores:
        c.apply_translation([0, 0, FLOOR_MM])
        solid = trimesh.boolean.boolean_manifold([solid, c], "difference")
    return solid


def save(name: str, mesh_mm: trimesh.Trimesh) -> None:
    """Write the metre-scale sim copy and the mm-scale print copy.

    Same contract as generate_cad_mods.py::save -- and the same two assertions,
    because tests/test_cad_dimensions.py::_load_tris requires BINARY STL and
    trimesh's exporter writes binary by default.
    """
    assert mesh_mm.is_watertight, f"{name} not watertight"
    assert len(mesh_mm.split(only_watertight=False)) == 1, (
        f"{name} is not a single component")
    m = mesh_mm.copy()
    m.apply_scale(1.0 / 1000.0)
    m.export(os.path.join(SIM_DIR, f"{name}.stl"))
    mesh_mm.export(os.path.join(PRINT_DIR, f"{name}.stl"))
    lo, hi = mesh_mm.bounds
    print(f"{name}: {mesh_mm.volume / 1000.0:.2f} cm^3, "
          f"{hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm, "
          f"watertight={mesh_mm.is_watertight}")


def geom_lines() -> str:
    """MJCF geom lines for the six cells and the tray, collision + visual.

    Every mesh geom in robot_motors.xml is twinned: a collision geom and a
    visual twin at a byte-identical pose carrying contype="0" conaffinity="0".
    Emitting only one of each would make the new cells collision-only.
    """
    rgba_cell = "0.615686 0.811765 0.929412 1"
    rgba_tray = "0.647059 0.647059 0.647059 1"
    ind = " " * 16
    out = []
    for (cx, cy, cz) in cell_centres_mm():
        # geom pos places the mesh origin, which is the cell's BOTTOM face.
        pos = f"{cx/1000.0:.6g} {cy/1000.0:.6g} {(cz - CELL_LEN_MM/2)/1000.0:.6g}"
        out.append(f'{ind}<geom pos="{pos}" type="mesh" rgba="{rgba_cell}" mesh="cell" />')
        out.append(f'{ind}<geom pos="{pos}" type="mesh" rgba="{rgba_cell}" mesh="cell" '
                   f'contype="0" conaffinity="0" />')
    ox, oy, oz = PACK_ORIGIN_M
    tpos = f"{ox:.6g} {oy:.6g} {oz:.6g}"
    out.append(f'{ind}<geom pos="{tpos}" type="mesh" rgba="{rgba_tray}" mesh="holder_6cell" />')
    out.append(f'{ind}<geom pos="{tpos}" type="mesh" rgba="{rgba_tray}" mesh="holder_6cell" '
               f'contype="0" conaffinity="0" />')
    return "\n".join(out)


def sync_model() -> None:
    """Rewrite the cell / holder_6cell poses in all three model files.

    Kept as code rather than a hand edit because the pack origin was tuned
    against a measured fit and will move again if the bore does. Three files
    must agree: robot_motors.xml (twinned geoms), robot.xml (single geoms) and
    robot.urdf (visual+collision pairs).

    NOTE, a deliberate deviation from the task plan: the old 2-cell `holder`
    GEOM is removed rather than "retained alongside". A robot has one battery
    tray; keeping both would add a phantom 7.5 cm^3 part and double-book its
    mass. The `holder` MESH ASSET stays declared, and holder.stl stays on disk,
    because it is upstream's part and the 2-cell configuration still uses it.
    """
    import xml.etree.ElementTree as ET  # noqa: F401  (parse check only)

    centres = cell_centres_mm()
    ox, oy, oz = PACK_ORIGIN_M

    # ---- robot_motors.xml : collision + visual twin per pose ----------------
    p = os.path.join(SIM_DIR, "robot_motors.xml")
    lines = open(p).read().splitlines(keepends=True)
    keep = [l for l in lines
            if 'mesh="cell"' not in l and 'mesh="holder_6cell"' not in l]
    idx = max(i for i, l in enumerate(keep) if 'mesh="holder"' in l and "<geom" in l) \
        if any('mesh="holder"' in l and "<geom" in l for l in keep) else \
        max(i for i, l in enumerate(keep) if 'mesh="bms"' in l)
    body = [l + "\n" for l in geom_lines().splitlines()]
    open(p, "w").write("".join(keep[:idx + 1] + body + keep[idx + 1:]))

    # ---- robot.xml : one geom per pose --------------------------------------
    p = os.path.join(SIM_DIR, "robot.xml")
    lines = open(p).read().splitlines(keepends=True)
    keep = [l for l in lines
            if 'mesh="cell"' not in l and 'mesh="holder_6cell"' not in l]
    idx = max(i for i, l in enumerate(keep)
              if "<geom" in l and ('mesh="holder"' in l or 'mesh="bms"' in l))
    body = []
    for (cx, cy, cz) in centres:
        pos = f"{cx/1000.0:.6g} {cy/1000.0:.6g} {(cz - CELL_LEN_MM/2)/1000.0:.6g}"
        body.append(f'        <geom pos="{pos}" type="mesh" '
                    f'rgba="0.615686 0.811765 0.929412 1" mesh="cell"/>\n')
    body.append(f'        <geom pos="{ox:.6g} {oy:.6g} {oz:.6g}" type="mesh" '
                f'rgba="0.647059 0.647059 0.647059 1" mesh="holder_6cell"/>\n')
    open(p, "w").write("".join(keep[:idx + 1] + body + keep[idx + 1:]))

    # ---- robot.urdf : visual + collision pair per pose ----------------------
    p = os.path.join(SIM_DIR, "robot.urdf")
    txt = open(p).read()

    def pair(mesh, xyz, matname, rgba):
        o = f'            <origin xyz="{xyz}" rpy="0 0 0" />\n'
        return (f'        <visual>\n{o}            <geometry>\n'
                f'                <mesh filename="{mesh}"/>\n            </geometry>\n'
                f'            <material name="{matname}">\n'
                f'                <color rgba="{rgba}"/>\n            </material>\n'
                f'        </visual>\n'
                f'        <collision>\n{o}            <geometry>\n'
                f'                <mesh filename="{mesh}"/>\n            </geometry>\n'
                f'        </collision>\n')

    CELL_RGBA = ("0.61568627450980395466 0.81176470588235294379 "
                 "0.92941176470588238168 1.0")
    TRAY_RGBA = ("0.64705882352941179736 0.64705882352941179736 "
                 "0.64705882352941179736 1.0")
    block = "".join(
        pair("cell.stl",
             f"{cx/1000.0:.17g} {cy/1000.0:.17g} {(cz - CELL_LEN_MM/2)/1000.0:.17g}",
             "cell_material", CELL_RGBA)
        for (cx, cy, cz) in centres)
    block += pair("holder_6cell.stl", f"{ox:.17g} {oy:.17g} {oz:.17g}",
                  "holder_6cell_material", TRAY_RGBA)

    # Excise every existing cell / holder_6cell visual+collision pair, then
    # reinsert the freshly generated block at the first excision point.
    out, i, first = [], 0, None
    marks = ("cell.stl", "holder_6cell.stl")
    chunks = txt.split("        <visual>\n")
    head = chunks[0]
    rebuilt = [head]
    for c in chunks[1:]:
        piece = "        <visual>\n" + c
        if any(f'<mesh filename="{m}"/>' in piece.split("</collision>")[0]
               for m in marks):
            if first is None:
                first = len(rebuilt)
            continue
        rebuilt.append(piece)
    if first is None:
        raise SystemExit("no cell/holder_6cell block found in robot.urdf")
    rebuilt.insert(first, block)
    open(p, "w").write("".join(rebuilt))
    print(f"synced {len(centres)} cells + 1 holder_6cell into "
          "robot_motors.xml, robot.xml, robot.urdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-model", action="store_true",
                    help="rewrite the poses in the three model files")
    ap.add_argument("--geoms", action="store_true",
                    help="print the MJCF geom lines instead of writing STLs")
    args = ap.parse_args()

    if args.geoms:
        print(geom_lines())
        return 0

    if args.sync_model:
        sync_model()
        return 0

    _check_printable()
    print(f"pack: {ROWS} x {COLS} at pitch {PITCH_MM} mm, bore Ø{BORE_DIA_MM}, "
          f"web {PITCH_MM - BORE_DIA_MM:.1f} mm")
    print(f"origin (trunk frame): {PACK_ORIGIN_M}")
    tray = build_tray()
    tray.apply_translation([0, 0, 0])
    save("holder_6cell", tray)
    print()
    print("cell centres (trunk frame, mm):")
    for i, c in enumerate(cell_centres_mm(), 1):
        print(f"  cell #{i}: ({c[0]:8.2f}, {c[1]:7.2f}, {c[2]:6.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
