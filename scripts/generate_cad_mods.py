#!/usr/bin/env python3
"""Generate the Part-2 CAD modifications (layout v2.1) on the shell/chassis meshes.

Reproducible mesh pipeline (trimesh + manifold3d booleans): applies the cut
list from docs/jetson-mod/component_layout_v2.md to the SIM meshes (meters)
in mini_bdx/robots/open_duck_mini_v2/ and writes matching mm-scale printable
copies to print/. Reports volume deltas + delta centroids for the inertial
bookkeeping in scripts/compute_trunk_inertial.py.

Operations (all coordinates below in the TRUNK BODY frame, mm; meshes are
edited in their local frame, local = body - SHELL_OFFSET):

1. trunk_bottom: remove the central spine (SPINE_CUT_BOX) that blocked the
   Jetson bay. trunk_top mid-span support now flows through the hip-yaw
   towers (span ~45 mm, fine for PLA at these loads).
2. body_middle_bottom:
   - 4x Jetson mount bosses (D8 x to z=-11.5, i.e. 0.1 mm under the Jetson
     base plate) at (-75,+/-38) and (+15,+/-38) on the cavity floor. The
     dev-kit carrier has NO documented hole pattern (P3768 spec gives
     outline only) — bosses provide flat seats + pilot-drill bases for the
     edge/base-plate mount finalized on the physical kit.
   - +y port opening x[-80,-54] z[-15,+24]: the only walled section of the
     Jetson I/O edge (the leg cutout already opens x[-50,+18]).
   - -y exhaust louvers: 3 vertical slots (8 mm wide, z[-8,+20]) at
     x centers -78/-67/-56 (compute zone, between partition and leg cutout).
3. body_front: 2 rows x 3 inlet slots (20 x 6 mm) at plenum height
   (z rows [27,33] and [37,43]) feeding the above-heatsink fan intake.
4. body_back: battery hump rear extension — outer shell to x=-170 (16 mm
   beyond the old outer wall), interior bored to x=-167 within y[-20.5,20.5]
   z[-2,67] so the 6-cell (2 front + 2x2 rear grid) pack fits; y kept narrow
   to spare the USB-C charger mount (y < -25).
5. print/thermal_partition.stl (NEW, mm): 2 mm PLA wall 103.5 x 66.7 with
   45-degree bottom-corner chamfers (shell wall fillets) and a 8 x 5 mm
   bottom-center cable notch. The 1 mm mica sheet is a purchased part; the
   sim placeholder stays the 3 mm assembly envelope.

Run:  python3 scripts/generate_cad_mods.py
"""
import os

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO, "mini_bdx", "robots", "open_duck_mini_v2")
PRINT_DIR = os.path.join(REPO, "print")
OFF = np.array([-0.019, 0.0, 0.0648909])  # body = local + OFF

PLA_EFFECTIVE_DENSITY = 1.116e3  # kg/m^3: 0.9 x solid PLA — perimeter-dominated thin walls


def body_box(lo, hi):
    """Axis-aligned box mesh from body-frame bounds (meters), in LOCAL coords."""
    lo = np.asarray(lo, dtype=float) - OFF
    hi = np.asarray(hi, dtype=float) - OFF
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    return box


def body_cyl(center_xy, z_lo, z_hi, radius):
    cyl = trimesh.creation.cylinder(radius=radius, height=z_hi - z_lo, sections=48)
    c = np.array([center_xy[0], center_xy[1], (z_lo + z_hi) / 2]) - OFF
    cyl.apply_translation(c)
    return cyl


def load(name):
    m = trimesh.load(os.path.join(SIM_DIR, f"{name}.stl"), force="mesh")
    assert m.is_watertight, f"{name} not watertight before ops"
    return m


def save(name, mesh, also_print=True):
    assert mesh.is_watertight, f"{name} not watertight AFTER ops"
    mesh.export(os.path.join(SIM_DIR, f"{name}.stl"))
    if also_print:
        pm = mesh.copy()
        pm.apply_scale(1000.0)
        pm.export(os.path.join(PRINT_DIR, f"{name}.stl"))


def report(name, before, after):
    dv = (after.volume - before.volume) * 1e6
    if abs(dv) > 1e-3:
        # centroid of the delta region: (V2*c2 - V1*c1) / (V2 - V1), in body frame
        c = (after.volume * after.center_mass - before.volume * before.center_mass) / (
            after.volume - before.volume
        ) + OFF
        print(
            f"{name}: {before.volume*1e6:8.2f} -> {after.volume*1e6:8.2f} cm^3 "
            f"(delta {dv:+7.2f} cm^3 = {dv*PLA_EFFECTIVE_DENSITY/1000:+6.1f} g at body ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}))"
        )
    else:
        print(f"{name}: volume unchanged")


def boolean(mesh, other, op):
    out = trimesh.boolean.boolean_manifold([mesh, other], op)
    return out


def main():
    # ------------------------------------------------------------------ 1
    tb = load("trunk_bottom")
    before = tb.copy()
    spine = body_box([-0.044, -0.016, -0.0136], [0.021, 0.017, 0.051])
    tb = boolean(tb, spine, "difference")
    report("trunk_bottom (spine cut)", before, tb)
    save("trunk_bottom", tb)

    # ------------------------------------------------------------------ 2
    bmb = load("body_middle_bottom")
    before = bmb.copy()
    port = body_box([-0.080, 0.045, -0.015], [-0.054, 0.058, 0.024])
    bmb = boolean(bmb, port, "difference")
    for xc in (-0.078, -0.067, -0.056):
        slot = body_box([xc - 0.004, -0.058, -0.008], [xc + 0.004, -0.045, 0.020])
        bmb = boolean(bmb, slot, "difference")
    for x, y in [(-0.075, 0.038), (-0.075, -0.038), (0.015, 0.038), (0.015, -0.038)]:
        boss = body_cyl((x, y), -0.0245, -0.0115, 0.004)
        bmb = boolean(bmb, boss, "union")
    report("body_middle_bottom (port+louvers+bosses)", before, bmb)
    save("body_middle_bottom", bmb)

    # ------------------------------------------------------------------ 3
    bf = load("body_front")
    before = bf.copy()
    for z_lo, z_hi in [(0.027, 0.033), (0.037, 0.043)]:
        for yc in (-0.026, 0.0, 0.026):
            slot = body_box([0.031, yc - 0.010, z_lo], [0.051, yc + 0.010, z_hi])
            bf = boolean(bf, slot, "difference")
    report("body_front (inlet slots)", before, bf)
    save("body_front", bf)

    # ------------------------------------------------------------------ 4
    bb = load("body_back")
    before = bb.copy()
    ext_outer = body_box([-0.170, -0.024, -0.005], [-0.134, 0.024, 0.070])
    ext_inner = body_box([-0.167, -0.0205, -0.002], [-0.125, 0.0205, 0.067])
    bb = boolean(bb, ext_outer, "union")
    bb = boolean(bb, ext_inner, "difference")
    report("body_back (hump extension)", before, bb)
    save("body_back", bb)

    # ------------------------------------------------------------------ 5
    # Printable partition wall (mm, PLA part only — 2 mm of the 3 mm assembly)
    wall = trimesh.creation.box(extents=[2.0, 103.5, 66.7])
    # bottom-corner 45-degree chamfers (12 mm wide x 12 mm tall)
    for sy in (-1, 1):
        ch = trimesh.creation.box(extents=[4.0, 24.0, 24.0])
        ch.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))
        ch.apply_translation([0, sy * 103.5 / 2, -66.7 / 2])
        wall = boolean(wall, ch, "difference")
    # bottom-center cable notch 8 wide x 5 tall
    notch = trimesh.creation.box(extents=[4.0, 8.0, 10.0])
    notch.apply_translation([0, 0, -66.7 / 2])
    wall = boolean(wall, notch, "difference")
    assert wall.is_watertight
    wall.export(os.path.join(PRINT_DIR, "thermal_partition.stl"))
    print(f"print/thermal_partition.stl: {wall.volume/1000:.2f} cm^3 PLA "
          f"({wall.volume/1000*1.24:.1f} g solid)")


if __name__ == "__main__":
    main()
