#!/usr/bin/env python3
"""Generate the Part-2 CAD modifications (layout v2.1) on the shell/chassis meshes.

Reproducible + idempotent mesh pipeline (trimesh + manifold3d booleans):
restores the pristine pre-Part-2 meshes from git (BASELINE_COMMIT), applies
the cut list from docs/jetson-mod/component_layout_v2.md, writes sim meshes
(meters) + mm-scale print/ copies, and emits EXACT signed inertia-delta
terms (mass, centroid, full tensor about own centroid) to
scripts/cad_mod_deltas.json, which scripts/compute_trunk_inertial.py loads.

Review-fix revision (post e7bf790 adversarial review):
- +y port opening shrunk to x[-80,-58] (the leg cutout's true edge is
  x=-54.5, not -50: the old cut to -54 merged with it) and its floor
  dropped to z=-16.5 to remove the tapered 0.2-1.6 mm sill knife-edge.
- -y louvers reduced to 2 (the third merged with the leg cutout).
- Printed partition corners recut to the MEASURED shell fillet profile
  (~24.6 deg slope: half-width 41.0 at body z=-22.3 rising to 52.0 by
  z=+1.7) instead of the too-small 45-deg chamfer, plus a 0.5 mm bottom
  rabbet over the floor ridge at body x[-87,-86.1]; an exact-boolean fit
  assertion now runs in this script.
- Shell deltas booked with exact tensors (was point/box, ~1% Iyy/Izz bias).

Structural note (measured): the removed trunk_bottom spine engaged
trunk_top with only a 2.2 mm^3 contact patch — it was never a meaningful
mid-span support, so no replacement ribs are added; trunk_top continues to
mount via its servo pockets and perimeter as in the original design.

Run:  python3 scripts/generate_cad_mods.py
"""
import json
import os
import subprocess

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO, "mini_bdx", "robots", "open_duck_mini_v2")
PRINT_DIR = os.path.join(REPO, "print")
OFF = np.array([-0.019, 0.0, 0.0648909])  # body = local + OFF

# Pre-Part-2 state of the meshes (Part-1 review-fix commit). All ops below
# are applied to THESE, so re-running the script is idempotent.
BASELINE_COMMIT = "adbc082"

# Effective density for booking printed-PLA volume deltas as mass.
# NOTE (review finding): this is a consistency choice with the unknown
# density baked into the upstream 698.5 g trunk booking, NOT a measured
# value — the spine region is bulky (not thin-wall), so the true delta
# carries a +/-10-30 g band. True-up by weighing old/new prints in Phase 4.
PLA_EFFECTIVE_DENSITY = 1116.0  # kg/m^3 (0.9 x solid)

DELTAS_JSON = os.path.join(REPO, "scripts", "cad_mod_deltas.json")


def baseline_mesh(name):
    blob = subprocess.run(
        ["git", "-C", REPO, "show",
         f"{BASELINE_COMMIT}:mini_bdx/robots/open_duck_mini_v2/{name}.stl"],
        capture_output=True, check=True,
    ).stdout
    m = trimesh.load(trimesh.util.wrap_as_stream(blob), file_type="stl", force="mesh")
    assert m.is_watertight, f"{name} baseline not watertight"
    return m


def body_box(lo, hi):
    lo = np.asarray(lo, dtype=float) - OFF
    hi = np.asarray(hi, dtype=float) - OFF
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    return box


def body_cyl(center_xy, z_lo, z_hi, radius):
    cyl = trimesh.creation.cylinder(radius=radius, height=z_hi - z_lo, sections=48)
    cyl.apply_translation(np.array([center_xy[0], center_xy[1], (z_lo + z_hi) / 2]) - OFF)
    return cyl


def boolean(mesh, other, op):
    return trimesh.boolean.boolean_manifold([mesh, other], op)


def save(name, mesh):
    assert mesh.is_watertight, f"{name} not watertight AFTER ops"
    assert len(mesh.split(only_watertight=False)) == 1, f"{name} not single component"
    mesh.export(os.path.join(SIM_DIR, f"{name}.stl"))
    pm = mesh.copy()
    pm.apply_scale(1000.0)
    pm.export(os.path.join(PRINT_DIR, f"{name}.stl"))


def signed_delta_terms(name, before, after):
    """Exact inertia bookkeeping: one term for removed material (negative
    mass) and one for added material (positive), each with the diff solid's
    exact tensor about its own centroid (body frame)."""
    terms = []
    for label, a, b, sign in [
        (f"{name}_removed", before, after, -1.0),
        (f"{name}_added", after, before, +1.0),
    ]:
        diff = boolean(a, b, "difference")
        if diff.is_empty or abs(diff.volume) < 1e-9:
            continue
        diff.density = PLA_EFFECTIVE_DENSITY
        com_body = diff.center_mass + OFF
        # moment_inertia is about the mesh CoM in mesh (== local == body-
        # aligned) axes; store the full symmetric tensor.
        I = diff.moment_inertia
        terms.append(
            dict(
                name=label,
                mass=sign * diff.mass,
                pos=[float(v) for v in com_body],
                tensor=[[float(I[r][c]) for c in range(3)] for r in range(3)],
                volume_cm3=sign * diff.volume * 1e6,
            )
        )
        print(f"  {label}: {sign*diff.volume*1e6:+.3f} cm^3 = {sign*diff.mass*1000:+.2f} g "
              f"at body ({com_body[0]:.4f}, {com_body[1]:.4f}, {com_body[2]:.4f})")
    return terms


def partition_print_part():
    """2 mm PLA wall, corners cut to the measured shell fillet profile."""
    H, W, T = 66.7, 103.5, 2.0  # mm; local frame: x=thickness, z up, origin center
    wall = trimesh.creation.box(extents=[T, W, H])
    z_bot = -H / 2  # local z of bottom edge = body z -22.35
    # Measured fillet: interior half-width 41.0 mm at body z=-22.3 rising at
    # slope 0.458 mm/mm to 52.0 by body z=+1.7 (local z = body - 11). Cut
    # each corner with a fillet-matched prism (0.3 mm clearance): boundary
    # line from (|y|=40.7, local -33.35) at slope dz/dy = 1/0.458, reaching
    # full half-width 51.75 at local -9.24 (body +1.76, where the interior
    # is already 52.0).
    for sy in (-1, 1):
        pts = []
        for x in (-T / 2 - 1.0, T / 2 + 1.0):
            for y, z in [(40.24, z_bot - 1.0), (53.0, z_bot - 1.0), (53.0, -6.49)]:
                pts.append([x, sy * y, z])
        prism = trimesh.convex.convex_hull(np.array(pts))
        wall = boolean(wall, prism, "difference")
    # 1.0 mm bottom rabbet over the floor ridge at body x[-87.5,-86.1]
    # (ridge top measured up to z=-21.93, spanning y +/-41; PLA occupies
    # body x[-87.5,-85.5]).
    rabbet = trimesh.creation.box(extents=[2.2, 84.0, 2.0])
    rabbet.apply_translation([-0.4, 0, z_bot])
    wall = boolean(wall, rabbet, "difference")
    # 8 x 5 mm bottom-center cable notch
    notch = trimesh.creation.box(extents=[T + 2.0, 8.0, 10.0])
    notch.apply_translation([0, 0, z_bot])
    wall = boolean(wall, notch, "difference")
    assert wall.is_watertight
    return wall


def check_partition_fit(wall_mm, shells):
    """Exact-boolean fit check: PLA part posed in the trunk (body frame).

    The 3 mm assembly spans body x[-87.5,-84.5]; mica (1 mm) faces the
    compute side, so the PLA occupies x[-87.5,-85.5] -> center -86.5 mm.
    """
    part = wall_mm.copy()
    part.apply_scale(1e-3)
    part.apply_translation(np.array([-0.0865, 0.0, 0.011]) - OFF)  # to local/shell frame
    worst = 0.0
    for name, shell in shells.items():
        inter = boolean(part, shell, "intersection")
        vol = 0.0 if inter.is_empty else inter.volume * 1e9  # mm^3
        print(f"  partition-fit vs {name}: {vol:.2f} mm^3")
        worst = max(worst, vol)
    assert worst < 5.0, f"printed partition interferes with shells ({worst:.1f} mm^3)"


def main():
    all_terms = []

    # ------------------------------------------------------------------ 1
    tb0 = baseline_mesh("trunk_bottom")
    tb = boolean(tb0, body_box([-0.044, -0.016, -0.0136], [0.021, 0.017, 0.051]), "difference")
    print("trunk_bottom (spine cut):")
    all_terms += signed_delta_terms("trunk_bottom", tb0, tb)
    save("trunk_bottom", tb)

    # ------------------------------------------------------------------ 2
    bmb0 = baseline_mesh("body_middle_bottom")
    bmb = boolean(bmb0, body_box([-0.080, 0.045, -0.0165], [-0.058, 0.058, 0.024]), "difference")
    for xc in (-0.078, -0.066):
        bmb = boolean(bmb, body_box([xc - 0.004, -0.058, -0.008], [xc + 0.004, -0.045, 0.020]), "difference")
    for x, y in [(-0.075, 0.038), (-0.075, -0.038), (0.015, 0.038), (0.015, -0.038)]:
        bmb = boolean(bmb, body_cyl((x, y), -0.0245, -0.0115, 0.004), "union")
    print("body_middle_bottom (port x[-80,-58] z[-16.5,24] + 2 louvers + 4 bosses):")
    all_terms += signed_delta_terms("body_middle_bottom", bmb0, bmb)
    save("body_middle_bottom", bmb)

    # ------------------------------------------------------------------ 3
    bf0 = baseline_mesh("body_front")
    bf = bf0
    for z_lo, z_hi in [(0.027, 0.033), (0.037, 0.043)]:
        for yc in (-0.026, 0.0, 0.026):
            bf = boolean(bf, body_box([0.031, yc - 0.010, z_lo], [0.051, yc + 0.010, z_hi]), "difference")
    print("body_front (inlet slots):")
    all_terms += signed_delta_terms("body_front", bf0, bf)
    save("body_front", bf)

    # ------------------------------------------------------------------ 4
    bb0 = baseline_mesh("body_back")
    bb = boolean(bb0, body_box([-0.170, -0.024, -0.005], [-0.134, 0.024, 0.070]), "union")
    bb = boolean(bb, body_box([-0.167, -0.0205, -0.002], [-0.125, 0.0205, 0.067]), "difference")
    print("body_back (hump extension):")
    all_terms += signed_delta_terms("body_back", bb0, bb)
    save("body_back", bb)

    # ------------------------------------------------------------------ 5
    wall = partition_print_part()
    wall.export(os.path.join(PRINT_DIR, "thermal_partition.stl"))
    print(f"print/thermal_partition.stl: {wall.volume/1000:.2f} cm^3 PLA")
    print("partition fit check (exact boolean at pose):")
    check_partition_fit(wall, {"body_middle_bottom": bmb, "trunk_bottom": tb,
                               "trunk_top": trimesh.load(os.path.join(SIM_DIR, "trunk_top.stl"), force="mesh")})

    # ------------------------------------------------------------------
    with open(DELTAS_JSON, "w") as f:
        json.dump(
            dict(
                baseline_commit=BASELINE_COMMIT,
                density_kg_m3=PLA_EFFECTIVE_DENSITY,
                terms=all_terms,
            ),
            f,
            indent=1,
        )
    total = sum(t["mass"] for t in all_terms)
    print(f"\nwrote {DELTAS_JSON}: {len(all_terms)} terms, net {total*1000:+.2f} g")


if __name__ == "__main__":
    main()
