"""Tasks R1/R1b/R2/R2b — the results directories must each hold one robot.

The failure this file mechanises against is PLANT-1: for three policy
generations every published gate number described a robot 37.6 % heavier than
the one on disk, and nothing anywhere noticed, because no result recorded which
plant it was measured on. Task R0 added a `plant` block to every JSON. These
tests are what make that block load-bearing instead of decorative.

They **parse the produced JSONs**. They grep nothing — `known_issues.md`
TEST-1 measured that 5 of 7 seeded regressions pass this repo's suite because
most of it is `assert "<literal>" in read()`.

Every test skips cleanly while a directory is still empty, so the file is green
before R1 runs and meaningful the moment it finishes.

The expected mass for a directory is read from that directory's own
`README.md`, not hardcoded here. Writing 2.657067 in two places is how the two
copies drift apart.
"""

import glob
import json
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO_ROOT, "docs", "jetson-mod")
INIT_POS_PATH = os.path.join(REPO_ROOT, "scripts", "duck_init_pos.json")

# The 3.657 kg plant. No result measured on it may appear in a Phase-R
# directory; that would be a copy-paste, not a measurement.
STALE_PLANT_MASS_KG = 3.657067

RESULT_DIRS = ["eval_results_m2657", "eval_results_rebuild"]


def _load(dirname):
    """Every result JSON in a Phase-R directory, as (basename, dict) pairs."""
    pattern = os.path.join(DOCS, dirname, "*.json")
    return [(os.path.basename(p), json.load(open(p)))
            for p in sorted(glob.glob(pattern))]


def _require(dirname):
    entries = _load(dirname)
    if not entries:
        pytest.skip(f"docs/jetson-mod/{dirname}/ holds no JSON yet")
    return entries


def _expected_mass(dirname):
    """The plant mass this directory's README declares, in kg.

    Single source of truth per directory: the README table row
    `| `simulated_total_mass_kg` | `2.657067` |`.
    """
    readme = os.path.join(DOCS, dirname, "README.md")
    if not os.path.isfile(readme):
        pytest.skip(f"docs/jetson-mod/{dirname}/README.md does not exist yet")
    text = open(readme).read()
    m = re.search(r"`simulated_total_mass_kg`\s*\|\s*`([0-9.]+)`", text)
    if not m:
        pytest.fail(
            f"{dirname}/README.md does not declare simulated_total_mass_kg; "
            "every results directory must name the plant it belongs to")
    return float(m.group(1))


@pytest.mark.phase2
@pytest.mark.parametrize("dirname", RESULT_DIRS)
class TestPlantProvenance:

    def test_every_json_records_a_plant(self, dirname):
        for name, d in _require(dirname):
            assert "plant" in d, (
                f"{dirname}/{name} has no 'plant' block. It was produced by a "
                "pre-R0 evaluate_policies.py and cannot be attributed to a "
                "robot — re-run it, do not annotate it by hand.")
            for key in ("simulated_total_mass_kg", "root_body", "joint_order",
                        "obs_dim", "action_dim"):
                assert key in d["plant"], f"{dirname}/{name}: plant.{key} missing"

    def test_directory_holds_exactly_one_plant(self, dirname):
        """One results dir + one comparison table per robot model."""
        entries = _require(dirname)
        masses = {round(d["plant"]["simulated_total_mass_kg"], 3)
                  for _, d in entries if "plant" in d}
        dims = {(d["plant"]["obs_dim"], d["plant"]["action_dim"])
                for _, d in entries if "plant" in d}
        assert len(masses) == 1, (
            f"{dirname}/ mixes {len(masses)} plant masses: {sorted(masses)}. "
            "Split it — one directory per robot model.")
        assert len(dims) == 1, (
            f"{dirname}/ mixes obs/action dims: {sorted(dims)}")

    def test_no_stale_plant_leaked_in(self, dirname):
        for name, d in _require(dirname):
            mass = d.get("plant", {}).get("simulated_total_mass_kg")
            if mass is None:
                continue
            assert abs(mass - STALE_PLANT_MASS_KG) > 1e-3, (
                f"{dirname}/{name} was measured on the OLD {STALE_PLANT_MASS_KG} kg "
                "plant. It belongs in eval_results_v5/, not here.")

    def test_mass_matches_the_directory_readme(self, dirname):
        entries = _require(dirname)      # skip first: an empty dir has nothing
        expected = _expected_mass(dirname)   # to disagree with
        for name, d in entries:
            mass = d["plant"]["simulated_total_mass_kg"]
            assert abs(mass - expected) < 1e-3, (
                f"{dirname}/{name} reports {mass:.6f} kg but "
                f"{dirname}/README.md declares {expected:.6f} kg")

    def test_protocol_is_the_frozen_one(self, dirname):
        """The campaign is only a controlled comparison if nothing varied."""
        for name, d in _require(dirname):
            p = d["protocol"]
            assert len(p["conditions"]) == 6, (
                f"{dirname}/{name}: {len(p['conditions'])} conditions, "
                "the frozen protocol is 6")
            assert p["windows_per_condition"] == 10, f"{dirname}/{name}"
            assert p["num_envs"] == 64, f"{dirname}/{name}"
            assert p["episode_length_s"] == 30.0, f"{dirname}/{name}"
            assert p["seed"] == 42, f"{dirname}/{name}"
            assert p["deterministic"] is True, f"{dirname}/{name}"
            assert p["obs_corruption"] is False, f"{dirname}/{name}"


@pytest.mark.phase2
class TestM2657IsTheCorrectedPlant:
    """`eval_results_m2657/` is the PLANT-1-only plant, specifically."""

    DIRNAME = "eval_results_m2657"

    def test_mass_and_root_body(self):
        for name, d in _require(self.DIRNAME):
            p = d["plant"]
            assert abs(p["simulated_total_mass_kg"] - 2.657067) < 1e-3, name
            assert p["root_body"] == "trunk_assembly", (
                f"{name}: root is {p['root_body']!r}. Before the PLANT-1 fix it "
                "was the massless 'base'; a result naming 'base' predates the fix.")
            assert p["num_bodies"] == 21, (
                f"{name}: {p['num_bodies']} rigid bodies, expected 21 "
                "(22 before 'base' was merged into 'trunk_assembly')")

    def test_dims_are_pre_m0b(self):
        """This campaign re-gates the 59/16 checkpoints. M0b's 53/14 belongs
        to `eval_results_rebuild/`."""
        for name, d in _require(self.DIRNAME):
            assert (d["plant"]["obs_dim"], d["plant"]["action_dim"]) == (59, 16), name

    def test_joint_order_matches_duck_init_pos(self):
        """The action contract: q_target = init_pos + 0.25 * action, indexed by
        this exact order. Not asserted for eval_results_rebuild/ — if M0b lands
        the action set is 14 joints and duck_init_pos.json must be regenerated
        by Phase M."""
        expected = json.load(open(INIT_POS_PATH))["joint_order"]
        for name, d in _require(self.DIRNAME):
            assert d["plant"]["joint_order"] == expected, (
                f"{name}: the articulation joint order no longer matches "
                "scripts/duck_init_pos.json — every exported action maps to "
                "the wrong joint. See scripts/verify_action_contract.py.")

    def test_usd_asset_hash_matches_the_directory_readme(self):
        """Every JSON here must name the USD this DIRECTORY belongs to.

        Compared against the hash the README declares, NOT against the live
        repo. This directory is a historical record of the pre-Phase-M plant;
        once Phase M regenerated the USD (Task M6) the live hash legitimately
        moved on, and asserting against it would fail forever for the wrong
        reason. What still needs catching is a JSON from a DIFFERENT model
        landing in here, and the README hash catches exactly that.
        """
        readme = os.path.join(DOCS, self.DIRNAME, "README.md")
        m = re.search(r"`usd_asset_hash`\s*\|\s*`([0-9a-f]+)`", open(readme).read())
        assert m, f"{self.DIRNAME}/README.md does not declare a usd_asset_hash"
        declared = m.group(1)
        for name, d in _require(self.DIRNAME):
            recorded = d["plant"].get("usd_asset_hash")
            assert recorded == declared, (
                f"{name} was measured against USD {recorded!r} but "
                f"{self.DIRNAME}/README.md declares {declared!r} — a result "
                "from another robot model is in this directory.")


# --------------------------------------------- eval naming / fall rules ------

from pathlib import Path  # noqa: E402  - local to this section


def test_gate_labels_describe_what_is_measured_not_a_policy_version():
    """'v4 rule' / 'v5 rule' name POLICY GENERATIONS, not measurements, and a
    reader cannot tell what was counted. Misread once during the 2026-08-22
    re-gate. See docs/jetson-mod/eval_fall_rules.md."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "check_v7_acceptance.py").read_text()
    assert '"push, v4 rule"' not in src
    assert '"push, v5 rule"' not in src
    assert "strict fall (empty arena)" in src
    assert "contact-tolerant fall" in src


def test_the_empty_arena_fact_is_recorded():
    """The v4 push world places NO obstacles, so 'trunk contact > 1 N' can only
    mean trunk-on-ground. A 0.000 % there means it never fell -- NOT that it
    dodged obstacles."""
    doc = (Path(__file__).resolve().parents[1]
           / "docs" / "jetson-mod" / "eval_fall_rules.md")
    assert doc.exists(), "eval_fall_rules.md is the reference for this trap"
    text = doc.read_text()
    assert "no obstacles at all" in text
    assert "obstacle_frac" in text


def test_the_push_world_really_has_no_obstacles():
    """Guards the claim itself: if a future edit adds obstacles to the robust
    track, the doc above becomes wrong and the 0.000 % changes meaning."""
    import re
    src = (Path(__file__).resolve().parents[1] / "isaac_lab_env"
           / "open_duck_mini_v2" / "env_cfg.py").read_text()
    m = re.search(r"class OpenDuckPushEvalEnvCfg\(([^)]*)\):(.*?)(?=\n@configclass)",
                  src, re.S)
    assert m, "OpenDuckPushEvalEnvCfg not found"
    assert "obstacle" not in m.group(2).lower(), \
        "the strict push eval now places obstacles; eval_fall_rules.md is stale"
    assert "Robust" in m.group(1), "it should still inherit the obstacle-free robust cfg"
