#!/usr/bin/env python3
"""Convert the Open Duck Mini v2 MJCF model to USD for Isaac Sim.

This script uses Isaac Lab's MjcfConverter API to convert robot_motors.xml
(the torque-controlled MuJoCo model) into USD format for use with Isaac Sim
and Isaac Lab.

Requirements:
    - Isaac Sim 5.1.0+ (provides the Omniverse Kit runtime)
    - Isaac Lab 2.3.0+ (provides the MjcfConverter API)

Usage (on DGX Spark or machine with Isaac Sim installed):
    ./isaaclab.sh -p scripts/convert_mjcf_to_usd.py

    Or directly:
    python scripts/convert_mjcf_to_usd.py
"""

import os
import sys

# Ensure the repo root is on the Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Isaac Sim must be running before importing Isaac Lab modules.
# The isaaclab.sh launcher handles this; if running directly, Isaac Sim
# app must be initialised first.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension

# The headless kit file doesn't include the MJCF importer extension.
# Enable it explicitly before using the converter.
enable_extension("isaacsim.asset.importer.mjcf")

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

# --- Paths ---
MJCF_PATH = os.path.join(
    REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml"
)
USD_DIR = os.path.join(
    REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "usd"
)
USD_FILENAME = "open_duck_mini_v2.usd"


def main():
    if not os.path.exists(MJCF_PATH):
        print(f"ERROR: MJCF file not found: {MJCF_PATH}")
        sys.exit(1)

    print(f"Converting MJCF to USD...")
    print(f"  Input:  {MJCF_PATH}")
    print(f"  Output: {os.path.join(USD_DIR, USD_FILENAME)}")

    cfg = MjcfConverterCfg(
        asset_path=MJCF_PATH,
        usd_dir=USD_DIR,
        usd_file_name=USD_FILENAME,
        fix_base=False,  # Free-floating robot, not fixed to world
        make_instanceable=True,  # Reduces memory for parallel envs
        import_inertia_tensor=True,  # Preserve mass properties from Phase 1
        import_sites=True,  # Import MuJoCo sites (e.g., sensor locations)
        force_usd_conversion=True,  # Always regenerate
    )

    converter = MjcfConverter(cfg)
    usd_path = converter.usd_path

    print(f"  Generated USD: {usd_path}")

    # Verify output
    if os.path.exists(usd_path):
        size = os.path.getsize(usd_path)
        print(f"  File size: {size:,} bytes")
        print("  SUCCESS: USD conversion complete.")
    else:
        print("  ERROR: USD file was not created.")
        sys.exit(1)

    write_mesh_manifest()


def write_mesh_manifest():
    """Record md5s of every STL referenced by the MJCF at conversion time.

    Isaac Lab's .asset_hash covers only the MJCF file itself — a mesh
    re-exported under the same filename would leave the USD silently stale.
    tests/test_usd_conversion.py::test_usd_meshes_not_stale compares the
    current STL bytes against this manifest.
    """
    import hashlib
    import json
    import xml.etree.ElementTree as ET

    mjcf_dir = os.path.dirname(MJCF_PATH)
    manifest = {}
    for mesh in ET.parse(MJCF_PATH).getroot().iter("mesh"):
        fname = mesh.get("file")
        if not fname:
            continue
        md5 = hashlib.md5()
        with open(os.path.join(mjcf_dir, fname), "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                md5.update(chunk)
        manifest[fname] = md5.hexdigest()
    manifest_path = os.path.join(USD_DIR, ".mesh_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"  Wrote mesh manifest: {manifest_path} ({len(manifest)} meshes)")

    simulation_app.close()


if __name__ == "__main__":
    main()
