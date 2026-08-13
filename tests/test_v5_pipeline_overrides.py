"""Task R2b — v5_pipeline.sh must be reusable without editing its constants.

The pipeline hardcoded four things (LOGROOT, RESULTS, the comparison-markdown
path and CONDITIONS) and selected its run directory by mtime rather than by the
run name it was handed — `known_issues.md` **SHELL-1**. The rebuild campaign
needs the same pipeline against a different log root and a different results
directory, and editing the constants inline would destroy the v5 campaign's
reproduction path.

These are shell edits, so they are tested by **executing the script's variable
resolution**, not by grepping it. A grep would pass on a file that merely
mentions `${LOGROOT:-...}` somewhere.

Note the header cannot simply be sourced without positional arguments:
`RUN_NAME="${1:?…}"` aborts the shell if `$1` is unset. They are supplied.
"""

import os
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO_ROOT, "scripts", "v5_pipeline.sh")


def _header(tmp_path, name="header.sh"):
    """The constants block plus the RUNDIR resolution, minus the blocking wait.

    RUNDIR is assigned BELOW the wait-for-training loop, so a naive cut at
    `mkdir -p "$RESULTS"` leaves it unbound and a cut past the loop would block
    for as long as a training run takes. The wait block is excised and the two
    surviving pieces are concatenated, which is exactly the resolution logic
    under test and none of the orchestration.
    """
    src = open(PIPELINE).read()
    consts = src[:src.index('mkdir -p "$RESULTS"')]
    rundir_start = src.index('RUNDIR="${RUNDIR_OVERRIDE:-')
    rundir_end = src.index("CKPT=", rundir_start)
    path = tmp_path / name
    path.write_text(
        consts
        + src[rundir_start:rundir_end]
        + '\necho "$RESULTS|$COMPARISON_MD|$LOGROOT|$RUNDIR"\n')
    return str(path)


def _run(script, *argv, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(["bash", script, *argv], capture_output=True, text=True, env=e)
    return p


@pytest.mark.phase2
def test_syntax_is_valid():
    p = subprocess.run(["bash", "-n", PIPELINE], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.phase2
def test_results_dir_is_overridable(tmp_path):
    script = _header(tmp_path)

    # The header's FATAL guard exits 1 when no run dir exists, so point LOGROOT
    # at a directory that has one. RESULTS/COMPARISON_MD stay unset — those are
    # what this test is about.
    root = tmp_path / "lr"; (root / "2026-01-01_myrun").mkdir(parents=True)
    plain = _run(script, "myrun", "mytask", env={"RUNDIR_OVERRIDE": str(root)})
    assert plain.returncode == 0, plain.stderr
    res, cmp_md, logroot, _ = plain.stdout.strip().split("|")
    assert res.endswith("docs/jetson-mod/eval_results_v5"), res
    assert cmp_md.endswith("docs/jetson-mod/v5_comparison.md"), cmp_md
    assert logroot.endswith("open_duck_ppo_v5"), logroot

    over = _run(script, "myrun", "mytask", env={
        "RESULTS": "/tmp/rebuild_results",
        "COMPARISON_MD": "/tmp/rebuild.md",
        "LOGROOT": "/tmp/v6root",
        "RUNDIR_OVERRIDE": str(root),
    })
    assert over.returncode == 0, over.stderr
    res, cmp_md, logroot, _ = over.stdout.strip().split("|")
    assert res == "/tmp/rebuild_results"
    assert cmp_md == "/tmp/rebuild.md"
    assert logroot == "/tmp/v6root"


@pytest.mark.phase2
def test_rundir_prefers_the_named_run(tmp_path):
    """The SHELL-1 regression.

    Two run directories, and the one that does NOT match the run name is
    NEWER. mtime selection picks the wrong one — which is how a run from a
    different plant could win.
    """
    root = tmp_path / "logroot"
    (root / "2026-01-01_other").mkdir(parents=True)
    (root / "2026-01-01_myrun").mkdir(parents=True)
    # make the non-matching dir strictly newer
    os.utime(root / "2026-01-01_myrun", (1000, 1000))
    os.utime(root / "2026-01-01_other", (2000, 2000))

    script = _header(tmp_path)
    p = _run(script, "myrun", "mytask", env={"LOGROOT": str(root)})
    assert p.returncode == 0, p.stderr
    rundir = p.stdout.strip().split("|")[3]
    assert rundir.rstrip("/").endswith("_myrun"), (
        f"resolved {rundir!r}; mtime selection would have picked _other, which "
        "is SHELL-1 — a run from a different plant could win")


@pytest.mark.phase2
def test_v5_reproduction_path_intact(tmp_path):
    """A bare invocation must still reproduce the v5 campaign exactly."""
    script = _header(tmp_path)
    root = tmp_path / "lr2"; (root / "2026-01-01_v5d_contact_wrench").mkdir(parents=True)
    p = _run(script, "v5d_contact_wrench", "sometask",
             env={"RUNDIR_OVERRIDE": str(root)})
    assert p.returncode == 0, p.stderr
    res, cmp_md, logroot, _ = p.stdout.strip().split("|")
    assert res.endswith("docs/jetson-mod/eval_results_v5")
    assert logroot.endswith("open_duck_ppo_v5")


@pytest.mark.phase2
def test_include_args_default_is_not_empty(tmp_path):
    """EVAL-1 inside the pipeline: without --include, every *.json in the
    results dir is injected into the comparison table."""
    src = open(PIPELINE).read()
    cut = src.index('mkdir -p "$RESULTS"')
    path = tmp_path / "h2.sh"
    path.write_text(src[:cut] + '\necho "$INCLUDE_ARGS"\n')
    p = _run(str(path), "myrun", "mytask")
    assert p.returncode == 0, p.stderr
    assert "--include" in p.stdout, "INCLUDE_ARGS default carries no --include"

    p2 = _run(str(path), "myrun", "mytask",
              env={"INCLUDE_ARGS": "--include a --include b"})
    assert p2.stdout.strip() == "--include a --include b"


@pytest.mark.phase2
def test_run_eval_passes_the_overridable_paths():
    """The call site must use the variables, not the old literals."""
    src = open(PIPELINE).read()
    body = src.split("run_eval () {", 1)[1].split("\n}", 1)[0]
    assert '--comparison_md "$COMPARISON_MD"' in body, (
        "run_eval still hardcodes the v5 comparison path")
    assert "$INCLUDE_ARGS" in body, "run_eval passes no --include"
    assert 'docs/jetson-mod/v5_comparison.md' not in body
