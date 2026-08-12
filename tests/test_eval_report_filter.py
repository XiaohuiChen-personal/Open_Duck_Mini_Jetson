"""Task R0 — the comparison-table generator must not mix robot models.

These tests **run the real script in a subprocess**. They deliberately do not
grep its source. `known_issues.md` **TEST-1** measured that 5 of 7 seeded
regressions pass this repo's existing suite precisely because most of it is
`assert "<literal>" in open(file).read()`; a source-text assertion here would
pass whether or not `--include` actually filters anything.

`scripts/evaluate_policies.py` must never be imported — everything below its
`--report-only` short-circuit launches Isaac Sim at module scope. The
`--self-test` and `--report-only` paths are dispatched by literal `sys.argv`
checks and `sys.exit()` before the AppLauncher block, which is exactly why
running it as a subprocess is both necessary and sufficient.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "evaluate_policies.py")


def _write_entry(directory, name, **extra):
    """Minimal JSON with the keys the report path actually reads."""
    entry = {
        "name": name,
        "framework": "rsl_rl",
        "aggregate": {},
        "per_condition": {},
    }
    entry.update(extra)
    path = os.path.join(str(directory), f"{name}.json")
    with open(path, "w") as f:
        json.dump(entry, f)
    return path


def _report_only(tmp_path, *extra_args):
    md = tmp_path / "t.md"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--report-only",
         "--output_dir", str(tmp_path),
         "--comparison_md", str(md), *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return proc, md


@pytest.mark.phase2
class TestComparisonTableFilter:
    """EVAL-1: an unfiltered directory scan already corrupted v4_comparison.md."""

    def test_include_filters_the_table(self, tmp_path):
        _write_entry(tmp_path, "keep_me")
        _write_entry(tmp_path, "drop_me")

        proc, md = _report_only(tmp_path, "--include", "keep_me")

        assert proc.returncode == 0, proc.stderr
        text = md.read_text()
        assert "keep_me" in text
        assert "drop_me" not in text, (
            "--include did not filter the table; this is the EVAL-1 defect, "
            "which mixes push-eval rows into a table documented as push-free"
        )

    def test_no_include_keeps_current_behaviour(self, tmp_path):
        """The archived v4/v5 tables must regenerate with every entry."""
        _write_entry(tmp_path, "keep_me")
        _write_entry(tmp_path, "drop_me")

        proc, md = _report_only(tmp_path)

        assert proc.returncode == 0, proc.stderr
        text = md.read_text()
        assert "keep_me" in text and "drop_me" in text

    def test_include_accepts_several_names(self, tmp_path):
        """--include is repeatable and matches exactly, not by prefix."""
        _write_entry(tmp_path, "alpha")
        _write_entry(tmp_path, "beta")
        _write_entry(tmp_path, "alpha_pusheval")

        proc, md = _report_only(tmp_path, "--include", "alpha",
                                "--include", "beta")

        assert proc.returncode == 0, proc.stderr
        text = md.read_text()
        assert "| alpha " in text and "| beta " in text
        assert "alpha_pusheval" not in text


@pytest.mark.phase2
class TestPlantProvenanceLine:
    """EVAL-2: the generated header must describe the run, not the defaults."""

    def test_plant_line_reports_missing_provenance(self, tmp_path):
        """Entries written before the `plant` block existed must not crash."""
        _write_entry(tmp_path, "legacy")

        proc, md = _report_only(tmp_path)

        assert proc.returncode == 0, proc.stderr
        text = md.read_text()
        assert "Plant mass(es) simulated: not recorded" in text

    def test_plant_line_reports_the_measured_plant(self, tmp_path):
        _write_entry(
            tmp_path, "stamped",
            plant={"simulated_total_mass_kg": 2.657067,
                   "obs_dim": 59, "action_dim": 16},
            protocol={"conditions": [[0.2, 0, 0], [0, 0, 0.3]]},
        )

        proc, md = _report_only(tmp_path)

        assert proc.returncode == 0, proc.stderr
        text = md.read_text()
        assert "2.657067" in text
        assert "59/16" in text
        assert "conditions per entry: [2]" in text

    def test_two_plants_are_both_shown(self, tmp_path):
        """A mixed directory must be visibly mixed, not silently averaged."""
        _write_entry(tmp_path, "old",
                     plant={"simulated_total_mass_kg": 3.657067,
                            "obs_dim": 59, "action_dim": 16})
        _write_entry(tmp_path, "new",
                     plant={"simulated_total_mass_kg": 2.657067,
                            "obs_dim": 53, "action_dim": 14})

        proc, md = _report_only(tmp_path)

        assert proc.returncode == 0, proc.stderr
        text = md.read_text()
        assert "2.657067, 3.657067" in text
        assert "53/14" in text and "59/16" in text


@pytest.mark.phase2
def test_self_test_still_passes():
    """The pure-numpy metric suite must survive the R0 edits."""
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--self-test"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Self-test: OK" in proc.stdout
