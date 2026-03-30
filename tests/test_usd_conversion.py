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
