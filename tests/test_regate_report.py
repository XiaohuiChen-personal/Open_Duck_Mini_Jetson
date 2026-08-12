"""Task R1c — the gate scorer must be mechanical, and must refuse mixed models.

`scripts/regate_report.py` has no Isaac dependency, so it is imported directly
rather than shelled out to. These tests build synthetic result JSONs and drive
the real `main()`; they assert on its exit code and its printed output, never
on its source text (`known_issues.md` TEST-1).
"""

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "regate_report.py")

_spec = importlib.util.spec_from_file_location("regate_report", SCRIPT)
regate_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regate_report)


PLANT = {
    "simulated_total_mass_kg": 2.657067,
    "root_body": "trunk_assembly",
    "num_bodies": 21,
    "obs_dim": 59,
    "action_dim": 16,
    "usd_asset_hash": "10ab887fe4d412b22d3d7c857a9d7f12",
}

BATTERY_SUFFIXES = ["pusheval_v4def", "pusheval_v5def",
                    "wrencheval", "obstacleeval"]


def _write(tmp_path, name, *, falls=0.0, gait=6, rms=4.5, plant=None):
    doc = {
        "name": name,
        "framework": "rsl_rl",
        "plant": dict(PLANT if plant is None else plant),
        "protocol": {"conditions": [[0.2, 0, 0]] * 6},
        "per_condition": {},
        "aggregate": {
            "fall_rate_pct": falls,
            "reference_tracking_rms_deg": rms,
            "gait_valid_conditions": gait,
            "conditions": 6,
            "stance_duty_left_pct": 70.0,
            "stance_duty_right_pct": 68.0,
            "stance_duty_asymmetry_pp": 2.0,
            "ang_vel_z_error_radps": 0.09,
            "energy_proxy_w": 20.0,
            "mean_squared_jerk": 0.08,
        },
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(doc))


def _full_campaign(tmp_path, *, cand_falls=0.0, cand_gait=6, cand_rms=4.5,
                   cand_battery=None, ctrl_battery=None):
    cand_battery = cand_battery or dict.fromkeys(BATTERY_SUFFIXES, 1.0)
    ctrl_battery = ctrl_battery or dict.fromkeys(BATTERY_SUFFIXES, 5.0)
    _write(tmp_path, "v5d_contact_wrench", falls=cand_falls, gait=cand_gait,
           rms=cand_rms)
    _write(tmp_path, "v4_robust_grid6", falls=0.0, gait=6, rms=4.4)
    for s in BATTERY_SUFFIXES:
        _write(tmp_path, f"v5d_contact_wrench_{s}", falls=cand_battery[s])
        _write(tmp_path, f"v4_robust_{s}", falls=ctrl_battery[s])


def _run(tmp_path, *extra):
    return regate_report.main(["--results_dir", str(tmp_path), *extra])


@pytest.mark.phase2
class TestGateScoring:

    def test_gate_passes_on_clean_numbers(self, tmp_path, capsys):
        _full_campaign(tmp_path)
        rc = _run(tmp_path)
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "OVERALL: PASS" in out
        assert "G-R3  MANUAL" in out

    def test_gait_gate_below_bar_fails(self, tmp_path, capsys):
        _full_campaign(tmp_path, cand_gait=4)
        rc = _run(tmp_path)
        out = capsys.readouterr().out
        assert rc == 1
        assert "G-R1  FAIL" in out
        assert "OVERALL: FAIL" in out and "G-R1" in out.split("OVERALL:")[1]

    def test_gait_gate_at_the_bar_passes(self, tmp_path, capsys):
        """5 of 6 is the bar itself, not below it."""
        _full_campaign(tmp_path, cand_gait=5)
        rc = _run(tmp_path)
        assert rc == 0, capsys.readouterr().out

    def test_fall_rate_over_bar_fails(self, tmp_path, capsys):
        _full_campaign(tmp_path, cand_falls=2.5)
        rc = _run(tmp_path)
        out = capsys.readouterr().out
        assert rc == 1 and "G-R2  FAIL" in out

    def test_one_worse_contact_gate_fails(self, tmp_path, capsys):
        battery = dict.fromkeys(BATTERY_SUFFIXES, 1.0)
        battery["wrencheval"] = 60.0
        ctrl = dict.fromkeys(BATTERY_SUFFIXES, 5.0)
        ctrl["wrencheval"] = 55.0
        _full_campaign(tmp_path, cand_battery=battery, ctrl_battery=ctrl)
        rc = _run(tmp_path)
        out = capsys.readouterr().out
        assert rc == 1
        assert "G-R4  FAIL" in out
        assert "WORSE THAN CONTROL" in out

    def test_ref_rms_regression_fails(self, tmp_path, capsys):
        _full_campaign(tmp_path, cand_rms=6.0)   # control is 4.4
        rc = _run(tmp_path)
        out = capsys.readouterr().out
        assert rc == 1 and "G-R5  FAIL" in out


@pytest.mark.phase2
class TestMixedModelGuard:
    """The PLANT-1 class of error, caught at the reporting layer."""

    def test_mixed_plant_refuses(self, tmp_path, capsys):
        _full_campaign(tmp_path)
        stale = dict(PLANT, simulated_total_mass_kg=3.657067)
        _write(tmp_path, "v5d_contact_wrench_wrencheval", falls=47.1, plant=stale)
        rc = _run(tmp_path)
        err = capsys.readouterr().err
        assert rc == 2
        assert "2.657067" in err and "3.657067" in err

    def test_mixed_dims_refuses(self, tmp_path, capsys):
        _full_campaign(tmp_path)
        m0b = dict(PLANT, obs_dim=53, action_dim=14)
        _write(tmp_path, "v5d_contact_wrench_obstacleeval", falls=0.3, plant=m0b)
        rc = _run(tmp_path)
        err = capsys.readouterr().err
        assert rc == 2
        assert "59/16" in err and "53/14" in err

    def test_missing_plant_block_refuses(self, tmp_path, capsys):
        _full_campaign(tmp_path)
        doc = json.loads((tmp_path / "v4_robust_grid6.json").read_text())
        del doc["plant"]
        (tmp_path / "v4_robust_grid6.json").write_text(json.dumps(doc))
        rc = _run(tmp_path)
        err = capsys.readouterr().err
        assert rc == 2 and "v4_robust_grid6.json" in err


@pytest.mark.phase2
class TestMissingFiles:

    def test_missing_control_file_names_it(self, tmp_path, capsys):
        _full_campaign(tmp_path)
        os.remove(tmp_path / "v4_robust_wrencheval.json")
        rc = _run(tmp_path)
        out = capsys.readouterr().out
        assert rc == 1
        assert "v4_robust_wrencheval.json" in out
        assert "G-R4  FAIL" in out

    def test_missing_open_field_control_is_fatal(self, tmp_path, capsys):
        _full_campaign(tmp_path)
        os.remove(tmp_path / "v4_robust_grid6.json")
        rc = _run(tmp_path)
        cap = capsys.readouterr()
        assert rc == 1
        assert "v4_robust_grid6.json" in cap.err

    def test_empty_directory_is_an_error_not_a_pass(self, tmp_path, capsys):
        rc = _run(tmp_path)
        assert rc == 1
        assert "no *.json" in capsys.readouterr().err


@pytest.mark.phase2
def test_markdown_fences_the_report(tmp_path, capsys):
    _full_campaign(tmp_path)
    rc = _run(tmp_path, "--markdown")
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("```") and out.rstrip().endswith("```")
