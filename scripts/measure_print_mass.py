#!/usr/bin/env python3
"""Measure printed-part mass instead of assuming an effective density.

Why this exists
---------------
`scripts/generate_cad_mods.py` books CAD volume deltas as mass using a single
constant, ``PLA_EFFECTIVE_DENSITY = 1116.0`` (0.9 x solid PLA). Its own comment
is honest that this is "a consistency choice ... NOT a measured value" carrying
a "+/-10-30 g band". It can be measured, and when you measure it the booked
number is wrong by far more than that band:

    booked   -88.48 g
    measured  -6.60 g   (2 perimeters / 15% infill, the profile print_guide.md
                         documents)

The error keeps its sign across every plausible profile (see ``--sweep``), so it
is not an artefact of the settings. The mechanism is that ONE density is applied
to both sides of a diff whose two sides do not print at the same density:
material *removed* came from bulky interiors that are mostly infill (measured
0.28x solid on ``trunk_bottom``), while material *added* is thin vents and
bosses that print nearly solid. Both errors push the same way.

See docs/jetson-mod/known_issues.md#plant-10.

What it measures
----------------
Two different physical questions, because the answer depends on how the parts
are made:

* **FDM** (a printer, or a service printing FDM) -- mass depends on perimeters,
  infill and skins. There is no closed form worth trusting; slice it. Requires
  PrusaSlicer. Infill is NOT a mass fraction: a slicer lays solid perimeters on
  every wall plus solid top/bottom skins, so 15% infill yields an as-printed
  fraction around 0.5-1.0 of solid depending on wall thickness, not 0.15.
* **Powder / resin** (MJF, SLS, SLA -- what an online printing service will
  quote) -- parts are effectively **solid**. Infill is not a parameter you get
  to choose. Mass is volume x as-built density, and no slicer is needed.

Usage
-----
    # solid processes -- no slicer needed
    python3 scripts/measure_print_mass.py --process mjf-pa12
    python3 scripts/measure_print_mass.py --process sls-pa12

    # FDM -- needs PrusaSlicer on PATH or via --slicer
    python3 scripts/measure_print_mass.py --process fdm-pla --perimeters 2 --infill 15
    python3 scripts/measure_print_mass.py --process fdm-pla --sweep

    # the PLANT-10 measurement: slice the pre-CAD-mod geometry against current
    python3 scripts/measure_print_mass.py --cad-delta --perimeters 2 --infill 15

Counting rule
-------------
``docs/print_guide.md`` specifies **x2 and x4 quantities on nine rows**, so the
set is 52 pieces / 1571.94 cm^3, not the 37 distinct STL files / 1388.13 cm^3.
Summing files instead of pieces undercounts the robot by 13%. This script counts
pieces.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRINT_DIR = os.path.join(REPO, "print")
GUIDE = os.path.join(REPO, "docs", "print_guide.md")

# Baseline the CAD mods were cut from; see generate_cad_mods.py BASELINE_COMMIT.
CAD_BASELINE_COMMIT = "adbc082"
CAD_MOD_PARTS = ("body_front", "body_middle_bottom", "trunk_bottom", "body_back")
BOOKED_CAD_DELTA_G = -88.48  # scripts/cad_mod_deltas.json

# As-built densities, g/cm^3. Solid-process figures are finished-part densities
# (including porosity), not powder bulk density -- using bulk density here is a
# classic ~40% error.
PROCESS = {
    "fdm-pla":   dict(kind="fdm",   density=1.24, note="PLA filament, ISO 1183; 5 vendors agree"),
    "fdm-petg":  dict(kind="fdm",   density=1.27, note="PETG filament"),
    "fdm-asa":   dict(kind="fdm",   density=1.07, note="ASA filament"),
    "fdm-abs":   dict(kind="fdm",   density=1.04, note="ABS filament"),
    "mjf-pa12":  dict(kind="solid", density=1.01, note="HP 3D HR PA12, finished part incl. porosity"),
    "sls-pa12":  dict(kind="solid", density=0.97, note="EOS PA2200 finished part; verify with your bureau"),
    "mjf-pa12-gb": dict(kind="solid", density=1.30, note="PA12 glass-bead filled"),
    "sla-tough": dict(kind="solid", density=1.15, note="tough photopolymer, cured"),
}
TPU_PART = "foot_bottom_tpu"
TPU_DENSITY = 1.22          # TPU 95A, ISO 1183
TPU_INFILL_DEFAULT = 40     # print_guide.md


def part_quantities() -> dict[str, int]:
    """Piece count per part, read from docs/print_guide.md (not from the file list)."""
    qty: dict[str, int] = {}
    with open(GUIDE) as fh:
        for line in fh:
            m = re.match(r"- (\S+)\.stl x(\d+)", line.strip())
            if m:
                qty[m.group(1)] = int(m.group(2))
    on_disk = {
        os.path.splitext(f)[0] for f in os.listdir(PRINT_DIR) if f.lower().endswith(".stl")
    }
    for extra in sorted(on_disk - set(qty)):
        qty.setdefault(extra, 1)          # e.g. thermal_partition, a Jetson-mod addition
    return qty


def solid_volume_cm3(stl: str) -> float:
    import trimesh

    return float(trimesh.load(stl, force="mesh").volume) / 1000.0


def find_slicer(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    return shutil.which("prusa-slicer") or shutil.which("prusa-slicer-console")


def slice_mass_g(slicer: str, stl: str, perimeters: int, infill: int,
                 density: float, workdir: str) -> float | None:
    """Grams of extrudate for one part. Returns None if the slicer refuses it.

    Everything that is not the part itself is switched off, so the header's
    ``filament used [g]`` IS the part -- skirt, brim, support, raft and wipe
    tower are all counted by the slicer otherwise (support alone is +24% on
    these geometries).
    """
    out = os.path.join(workdir, os.path.basename(stl) + ".gcode")
    cmd = [
        slicer, "-g",
        "--nozzle-diameter", "0.4",
        "--filament-diameter", "1.75",
        # without --filament-density the [g] line is ABSENT entirely (silent
        # failure mode for a batch script)
        "--filament-density", str(density),
        "--layer-height", "0.2", "--first-layer-height", "0.2",
        "--perimeters", str(perimeters),
        "--fill-density", f"{infill}%", "--fill-pattern", "gyroid",
        "--top-solid-layers", "5", "--bottom-solid-layers", "4",
        # booleans MUST use '='; "--support-material 0" makes PrusaSlicer treat
        # 0 as a filename and silently emit nothing with exit code 0
        "--skirts", "0", "--brim-width", "0",
        "--support-material=0", "--raft-layers", "0", "--wipe-tower=0",
        "--bed-shape", "0x0,250x0,250x210,0x210",
        "--gcode-flavor", "marlin", "--max-print-height", "300",
        "-o", out, stl,
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out):
        return None
    with open(out, errors="ignore") as fh:
        for line in fh:
            if line.startswith("; filament used [g]"):
                return float(line.split("=")[1])
    return None


def _reoriented(stl: str, workdir: str, axis: str) -> str:
    """Lay a part flat on the bed and re-export it.

    Some parts are authored in an assembly pose whose lowest face is a knife
    edge; PrusaSlicer then aborts with "There is an object with no extrusions in
    the first layer". Supports are off, so mass is nearly orientation-invariant
    (~4.6% measured on ``left_cache``, from where the solid top/bottom skins
    land), which makes re-orienting a safe way to get a number.
    """
    import numpy as np
    import trimesh

    m = trimesh.load(stl, force="mesh")
    if axis in ("x", "y"):
        m.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0] if axis == "x" else [0, 1, 0]))
    v = m.vertices.copy()
    v -= v.min(0)
    v[:, 0] += 30.0
    v[:, 1] += 30.0
    out = os.path.join(workdir, f"{os.path.basename(stl)[:-4]}_{axis}.stl")
    trimesh.Trimesh(vertices=v, faces=m.faces, process=False).export(out)
    return out


def mass_for_part(name: str, cfg: dict, args, slicer: str | None,
                  workdir: str) -> tuple[float | None, str]:
    stl = os.path.join(PRINT_DIR, f"{name}.stl")
    is_tpu = name == TPU_PART
    if cfg["kind"] == "solid":
        rho = TPU_DENSITY if is_tpu else cfg["density"]
        return solid_volume_cm3(stl) * rho, "solid"
    if slicer is None:
        return None, "no-slicer"
    rho = TPU_DENSITY if is_tpu else cfg["density"]
    infill = TPU_INFILL_DEFAULT if is_tpu else args.infill
    g = slice_mass_g(slicer, stl, args.perimeters, infill, rho, workdir)
    if g is not None:
        return g, "sliced"
    # knife-edge contact: drop it onto the bed, then try two lay-flat rotations
    for axis, how in (("z", "sliced:on-bed"), ("x", "sliced:rot-x"), ("y", "sliced:rot-y")):
        alt = _reoriented(stl, workdir, axis)
        g = slice_mass_g(slicer, alt, args.perimeters, infill, rho, workdir)
        if g is not None:
            return g, how
    return None, "slice-failed"


def report(args) -> int:
    cfg = PROCESS[args.process]
    qty = part_quantities()
    slicer = find_slicer(args.slicer)
    if cfg["kind"] == "fdm" and slicer is None:
        print("ERROR: FDM needs PrusaSlicer. Install it, put it on PATH, or pass "
              "--slicer /path/to/prusa-slicer.\n"
              "       Solid processes (--process mjf-pa12 / sls-pa12 / sla-tough) "
              "need no slicer.", file=sys.stderr)
        return 2

    workdir = tempfile.mkdtemp(prefix="duckmass-")
    print(f"process : {args.process}  ({cfg['note']})")
    if cfg["kind"] == "fdm":
        print(f"profile : {args.perimeters} perimeters / {args.infill}% infill "
              f"(TPU sole at {TPU_INFILL_DEFAULT}%)")
    else:
        print("profile : solid — powder/resin parts have no infill parameter")
    print()
    print(f"{'part':34s}{'qty':>4s}{'vol cm3':>10s}{'g/piece':>10s}{'g total':>10s}  how")
    total_g = total_v = 0.0
    failed = []
    for name in sorted(qty):
        n = qty[name]
        v = solid_volume_cm3(os.path.join(PRINT_DIR, f"{name}.stl"))
        g, how = mass_for_part(name, cfg, args, slicer, workdir)
        total_v += v * n
        if g is None:
            failed.append(name)
            print(f"{name:34s}{n:4d}{v:10.2f}{'--':>10s}{'--':>10s}  {how}")
            continue
        total_g += g * n
        print(f"{name:34s}{n:4d}{v:10.2f}{g:10.2f}{g * n:10.2f}  {how}")
    print("-" * 78)
    print(f"{'TOTAL':34s}{sum(qty.values()):4d}{total_v:10.2f}{'':>10s}{total_g:10.2f}")
    if failed:
        print(f"\nNOT COUNTED (slicer refused): {failed}")
        print("  a part whose bottom face is a knife edge gives 'no extrusions in the "
              "first layer'; lay it flat first. Mass is orientation-independent here "
              "because supports are off, but the solid top/bottom skins do shift "
              "slightly (~4.6% measured on left_cache).")
    shutil.rmtree(workdir, ignore_errors=True)
    return 1 if failed else 0


def cad_delta(args) -> int:
    """The PLANT-10 measurement: baseline geometry vs current, same settings."""
    cfg = PROCESS[args.process]
    slicer = find_slicer(args.slicer)
    if cfg["kind"] == "fdm" and slicer is None:
        print("ERROR: --cad-delta on an FDM profile needs PrusaSlicer.", file=sys.stderr)
        return 2
    work = tempfile.mkdtemp(prefix="duckdelta-")
    base_dir = os.path.join(work, "baseline")
    os.makedirs(base_dir, exist_ok=True)
    print(f"baseline commit : {CAD_BASELINE_COMMIT}")
    print(f"process         : {args.process}", end="")
    if cfg["kind"] == "fdm":
        print(f", {args.perimeters} perim / {args.infill}% infill")
    else:
        print(" (solid)")
    print()
    print(f"{'part':26s}{'baseline g':>12s}{'current g':>12s}{'delta g':>10s}")
    total = 0.0
    for p in CAD_MOD_PARTS:
        blob = subprocess.run(
            ["git", "-C", REPO, "show", f"{CAD_BASELINE_COMMIT}:print/{p}.stl"],
            capture_output=True,
        )
        if blob.returncode != 0:
            print(f"{p:26s}{'baseline missing at that commit':>34s}")
            continue
        bpath = os.path.join(base_dir, f"{p}.stl")
        with open(bpath, "wb") as fh:
            fh.write(blob.stdout)
        cpath = os.path.join(PRINT_DIR, f"{p}.stl")
        if cfg["kind"] == "solid":
            a = solid_volume_cm3(bpath) * cfg["density"]
            b = solid_volume_cm3(cpath) * cfg["density"]
        else:
            a = slice_mass_g(slicer, bpath, args.perimeters, args.infill, cfg["density"], work)
            b = slice_mass_g(slicer, cpath, args.perimeters, args.infill, cfg["density"], work)
            if a is None or b is None:
                print(f"{p:26s}{'slicer refused one side':>34s}")
                continue
        total += b - a
        print(f"{p:26s}{a:12.2f}{b:12.2f}{b - a:+10.2f}")
    print("-" * 60)
    print(f"{'MEASURED delta':26s}{'':>24s}{total:+10.2f}")
    print(f"{'BOOKED  delta':26s}{'':>24s}{BOOKED_CAD_DELTA_G:+10.2f}"
          f"   (generate_cad_mods.py PLA_EFFECTIVE_DENSITY=1116)")
    print(f"{'error in the model':26s}{'':>24s}{total - BOOKED_CAD_DELTA_G:+10.2f}"
          f"   <- trunk_assembly is this much heavier than booked")
    shutil.rmtree(work, ignore_errors=True)
    return 0


def sweep(args) -> int:
    slicer = find_slicer(args.slicer)
    if slicer is None:
        print("ERROR: --sweep needs PrusaSlicer.", file=sys.stderr)
        return 2
    print("CAD-mod delta across print profiles (sign should be stable):\n")
    print(f"{'profile':28s}{'measured g':>12s}{'booked g':>11s}{'error g':>10s}")
    for perim, infill in ((2, 15), (3, 15), (2, 25), (3, 40)):
        args.perimeters, args.infill = perim, infill
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cad_delta(args)
        m = re.search(r"MEASURED delta\s+([-+\d.]+)", buf.getvalue())
        if not m:
            continue
        val = float(m.group(1))
        print(f"{f'{perim} perim / {infill}% infill':28s}{val:12.2f}"
              f"{BOOKED_CAD_DELTA_G:11.2f}{val - BOOKED_CAD_DELTA_G:+10.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", default="fdm-pla", choices=sorted(PROCESS),
                    help="how the parts are made (default: fdm-pla)")
    ap.add_argument("--perimeters", type=int, default=2,
                    help="FDM only. print_guide.md does NOT specify this and it is "
                         "worth ~150 g on the robot (default: 2)")
    ap.add_argument("--infill", type=int, default=15,
                    help="FDM only, percent (default: 15, per print_guide.md)")
    ap.add_argument("--slicer", default=None, help="path to prusa-slicer")
    ap.add_argument("--cad-delta", action="store_true",
                    help="measure the Part-2 CAD-mod mass delta (PLANT-10)")
    ap.add_argument("--sweep", action="store_true",
                    help="run --cad-delta across several profiles")
    args = ap.parse_args()
    if args.sweep:
        return sweep(args)
    if args.cad_delta:
        return cad_delta(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
