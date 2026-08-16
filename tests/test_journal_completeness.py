"""Task R4 — every executed training run must have a journal entry.

`known_issues.md` DOC-2: five executed v5 runs, including the one that produced
the shipped policy, existed only inside a document titled "Execution Plan",
while AGENTS.md makes a journal entry mandatory per run.

This file is unavoidably text-based, but it asserts **structure**, not the
presence of a literal. TEST-1's trap is grepping for a string whose count is > 1
and calling it a semantic check; a heading regex and a set comparison between
the index table and the entries are different things — the second in particular
catches the real failure mode, an entry added without updating the index.
"""

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(REPO_ROOT, "docs", "jetson-mod", "experiment_journal.md")
RESULTS = os.path.join(REPO_ROOT, "docs", "jetson-mod", "v5_contact_results.md")

V5_RUNS = ("v5_smoke", "v5a_gated_ft", "v5b_ungated_ft",
           "v5c_contact_only", "v5d_contact_wrench")
# Written by Tasks R2 and R2b themselves, not by R4.
V6_RUNS = ("v6_robust", "v6d_contact_wrench")
# Phase-R servo fix; R2b-style rule: a finished run owes a journal entry.
V7_RUNS = ("v7_smoke", "v7_servo_safe", "v7b_servo_safe")


@pytest.fixture(scope="module")
def journal():
    return open(JOURNAL).read()


@pytest.mark.phase2
def test_every_v5_run_has_an_entry(journal):
    """A HEADING, not a mention. A run named in passing is not a record."""
    for run in V5_RUNS:
        assert re.search(r"^## Run \d+ — `" + re.escape(run) + r"`", journal, re.M), (
            f"no `## Run N — `{run}`` heading in experiment_journal.md")


@pytest.mark.phase2
def test_run_index_table_covers_every_entry(journal):
    """The real failure mode: an entry added without updating the index."""
    headings = set(re.findall(r"^## Run \d+ — `([^`]+)`", journal, re.M))
    rows = set(re.findall(r"^\| \d+ \| `([^`]+)` \|", journal, re.M))
    missing_rows = headings - rows
    missing_entries = rows - headings
    assert not missing_rows, (
        f"runs with an entry but no index row: {sorted(missing_rows)}")
    assert not missing_entries, (
        f"runs with an index row but no entry: {sorted(missing_entries)}")


@pytest.mark.phase2
def test_v5_entries_carry_a_plant_banner(journal):
    """Every v5 number was measured on the 3.657 kg plant. An entry that does
    not say so invites the reader to treat it as current."""
    for run in V5_RUNS:
        m = re.search(r"^## Run \d+ — `" + re.escape(run) + r"`", journal, re.M)
        nxt = journal.find("\n## Run ", m.end())
        body = journal[m.start():nxt if nxt > 0 else len(journal)]
        assert "3.657" in body, f"{run}'s entry has no plant banner"


@pytest.mark.phase2
def test_no_v5_entry_claims_the_corrected_plant(journal):
    """Prevents the DOC-3 error: a doc asserting a 2.657 kg plant for results
    measured at 3.657. If 2.657 appears at all it must be in a sentence that
    also points at the re-gate document."""
    for run in V5_RUNS:
        m = re.search(r"^## Run \d+ — `" + re.escape(run) + r"`", journal, re.M)
        nxt = journal.find("\n## Run ", m.end())
        body = journal[m.start():nxt if nxt > 0 else len(journal)]
        for line in body.splitlines():
            if "2.657" in line or "2.729" in line:
                assert "m2657_regate.md" in line or "rebuild" in line, (
                    f"{run}: line mentions a corrected-plant mass without "
                    f"pointing at the re-gate: {line.strip()!r}")


@pytest.mark.phase2
def test_v5_contact_results_exists_and_declares_its_plant():
    assert os.path.isfile(RESULTS), "docs/jetson-mod/v5_contact_results.md missing"
    text = open(RESULTS).read()
    first_section = text.index("\n# ")
    head = text[:first_section]
    assert "3.657" in head, (
        "the plant banner must appear ABOVE the first section heading, where a "
        "reader will actually see it")
    assert "m2657_regate.md" in head
    assert "rebuild_results.md" in head


@pytest.mark.phase2
def test_v6_runs_are_journalled_once_they_exist():
    """R2/R2b write their own entries. This asserts the pairing rather than
    the presence: a v6 checkpoint on disk with no journal entry is the DOC-2
    failure repeating."""
    isaaclab = os.path.expanduser("~/IsaacLab/logs/rsl_rl/open_duck_ppo_v6")
    if not os.path.isdir(isaaclab):
        pytest.skip("no v6 runs on disk yet")
    finished = []
    for d in os.listdir(isaaclab):
        run_dir = os.path.join(isaaclab, d)
        if not os.path.isdir(run_dir):
            continue
        # A run in flight writes model_100.pt, model_200.pt ... every
        # save_interval. Only the FINAL checkpoint means the run finished, and
        # only a finished run owes a journal entry.
        if os.path.isfile(os.path.join(run_dir, "model_2999.pt")):
            finished.append(d)
    if not finished:
        pytest.skip("no completed v6 run yet")
    journal = open(JOURNAL).read()
    for run in V6_RUNS:
        if not any(run in d for d in finished):
            continue
        assert re.search(r"^## Run \d+ — `" + re.escape(run) + r"`",
                         journal, re.M), (
            f"{run} has a checkpoint on disk but no journal entry — that is "
            "DOC-2 repeating")
