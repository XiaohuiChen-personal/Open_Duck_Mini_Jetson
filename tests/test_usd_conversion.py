"""Tests for Task 2.1 — MJCF to USD conversion."""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PATH = os.path.join(
    REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "usd", "open_duck_mini_v2.usd"
)


@pytest.mark.phase2
class TestUSDConversion:
    """Verify the USD file produced by scripts/convert_mjcf_to_usd.py."""

    def test_usd_file_exists(self):
        """Converted USD file must exist."""
        assert os.path.exists(USD_PATH), (
            f"USD file not found at {USD_PATH}. "
            "Run: ./isaaclab.sh -p scripts/convert_mjcf_to_usd.py"
        )

    def test_usd_file_not_empty(self):
        """USD file must not be suspiciously small."""
        if not os.path.exists(USD_PATH):
            pytest.skip("USD file not yet generated")
        size = os.path.getsize(USD_PATH)
        assert size > 1000, f"USD file suspiciously small: {size} bytes"

    def test_conversion_script_exists(self):
        """The conversion script must exist."""
        script = os.path.join(REPO_ROOT, "scripts", "convert_mjcf_to_usd.py")
        assert os.path.exists(script)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "EXPECTED STALENESS, Phase M -> Task M6. M2 (PLANT-10) and M3 "
            "(six 18650 cells + a deepened body_back hump) changed "
            "robot_motors.xml and its meshes, and M4 will rewrite every "
            "inertial next. The USD is GENERATED, never authored, and "
            "task_plan_v2.md makes regenerating it Task M6, the Phase-M gate, "
            "so that it is rebuilt ONCE at the end rather than after every "
            "edit. strict=True, so this becomes a hard FAILURE the moment M6 "
            "regenerates the USD and the marker must then be deleted. "
            "NOTHING MAY TRAIN BEFORE M6: Isaac Lab loads the USD, not the "
            "MJCF, so a stale USD trains a policy on the old plant silently."
        ),
    )
    def test_usd_not_stale(self):
        """The USD must have been generated from the CURRENT robot_motors.xml.

        Replicates Isaac Lab's asset-hash algorithm
        (isaaclab/sim/converters/asset_converter_base.py::_config_to_hash:
        md5(json(config minus path fields) + asset file bytes)) and compares
        against the stored .asset_hash. Fails whenever robot_motors.xml is
        edited without re-running scripts/convert_mjcf_to_usd.py — the
        failure mode that let policies train on a stale USD would otherwise
        be silent (training loads the USD, not the MJCF).
        """
        import hashlib
        import json

        import yaml

        usd_dir = os.path.dirname(USD_PATH)
        cfg_path = os.path.join(usd_dir, "config.yaml")
        hash_path = os.path.join(usd_dir, ".asset_hash")
        mjcf_path = os.path.join(
            REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml"
        )
        if not (os.path.exists(cfg_path) and os.path.exists(hash_path)):
            pytest.skip("USD conversion metadata not present")
        cfg = yaml.safe_load(open(cfg_path))
        for key in ("asset_path", "usd_dir", "usd_file_name"):
            cfg.pop(key, None)
        md5 = hashlib.md5()
        md5.update(json.dumps(cfg).encode())
        with open(mjcf_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                md5.update(chunk)
        stored = open(hash_path).read().strip()
        assert md5.hexdigest() == stored, (
            "USD is STALE: robot_motors.xml changed since the last conversion. "
            "Re-run: cd ~/IsaacLab && ./isaaclab.sh -p scripts/convert_mjcf_to_usd.py"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "EXPECTED STALENESS, Phase M -> Task M6. M2 (PLANT-10) and M3 "
            "(six 18650 cells + a deepened body_back hump) changed "
            "robot_motors.xml and its meshes, and M4 will rewrite every "
            "inertial next. The USD is GENERATED, never authored, and "
            "task_plan_v2.md makes regenerating it Task M6, the Phase-M gate, "
            "so that it is rebuilt ONCE at the end rather than after every "
            "edit. strict=True, so this becomes a hard FAILURE the moment M6 "
            "regenerates the USD and the marker must then be deleted. "
            "NOTHING MAY TRAIN BEFORE M6: Isaac Lab loads the USD, not the "
            "MJCF, so a stale USD trains a policy on the old plant silently."
        ),
    )
    def test_usd_meshes_not_stale(self):
        """Every STL referenced by robot_motors.xml must match the manifest
        written at conversion time.

        Isaac Lab's .asset_hash covers only the MJCF bytes — a mesh
        re-exported under the same filename (the most likely Phase-3+ drift)
        would otherwise leave the USD silently stale (review finding on
        commit 8f09c31).
        """
        import hashlib
        import json
        import xml.etree.ElementTree as ET

        usd_dir = os.path.dirname(USD_PATH)
        manifest_path = os.path.join(usd_dir, ".mesh_manifest.json")
        assert os.path.exists(manifest_path), (
            "mesh manifest missing — re-run scripts/convert_mjcf_to_usd.py"
        )
        manifest = json.load(open(manifest_path))
        mjcf_path = os.path.join(
            REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "robot_motors.xml"
        )
        mjcf_dir = os.path.dirname(mjcf_path)
        stale = []
        for mesh in ET.parse(mjcf_path).getroot().iter("mesh"):
            fname = mesh.get("file")
            if not fname:
                continue
            md5 = hashlib.md5(open(os.path.join(mjcf_dir, fname), "rb").read())
            if manifest.get(fname) != md5.hexdigest():
                stale.append(fname)
        assert not stale, (
            f"USD is STALE for meshes {stale}: STLs changed since the last "
            "conversion. Re-run scripts/convert_mjcf_to_usd.py"
        )
