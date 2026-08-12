#!/usr/bin/env python3
"""Last-N-iteration TensorBoard means for one training run.

AGENTS.md "Experiment Journal Protocol" rule 1 makes last-100 means mandatory
in every journal entry and forbids sourcing them from console greps: the
console prints per-iteration values, which are noisy enough that picking one
line can move a reported reward by tens of percent. This is the tool that
produces the numbers the rule demands.

    ~/IsaacLab/_isaac_sim/python.sh scripts/tb_summary.py \
        --run_dir ~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25 \
        --markdown

**Run it under the Isaac interpreter, not the system python3.** Measured
2026-08-11 on this machine: `from tensorboard.backend.event_processing...`
raises `ModuleNotFoundError: No module named 'tensorboard'` under `python3`
and imports cleanly under `~/IsaacLab/_isaac_sim/python.sh`. Nothing else in
this file needs Isaac Sim — it is stdlib plus tensorboard, and it never
launches Kit, so it is safe to run while a GPU job is alive.

Two details that change the numbers if you get them wrong:

* `size_guidance={'scalars': 0}` is load-bearing. EventAccumulator's default
  reservoir-samples scalars down to 1,000 points per tag, so on a 6,000-
  iteration run a "last 100" computed from the default is a last-100 of a
  *subsample* and silently disagrees with the run.
* Wall clock is taken from the first and last event timestamps of the run
  itself. Never estimate it and never reuse another run's figure (rule 2:
  elapsed time is not completion, and a borrowed duration is not a
  measurement).
"""

import argparse
import datetime
import os
import sys

DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Gait/duty_in_band_frac",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run_dir", required=True,
        help="A single rsl_rl run directory (the one holding events.out.*).",
    )
    p.add_argument(
        "--last", type=int, default=100,
        help="Window size in iterations for the trailing mean (default 100).",
    )
    p.add_argument(
        "--tags", action="append", default=None, metavar="TAG",
        help=("Scalar tag to report (repeatable). Default: "
              + ", ".join(DEFAULT_TAGS)
              + ", plus every Episode_Reward/* tag present in the run."),
    )
    p.add_argument(
        "--markdown", action="store_true",
        help="Emit a markdown table ready to paste into experiment_journal.md.",
    )
    return p


def load_accumulator(run_dir: str):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - interpreter-dependent
        sys.stderr.write(
            f"error: {exc}\n"
            "tensorboard is not importable from this interpreter. Use the "
            "Isaac one:\n"
            "  ~/IsaacLab/_isaac_sim/python.sh scripts/tb_summary.py ...\n"
        )
        raise SystemExit(2)
    # scalars: 0 == keep everything. See the module docstring.
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def select_tags(available, requested):
    if requested:
        missing = [t for t in requested if t not in available]
        if missing:
            sys.stderr.write(
                f"warning: tags not present in this run: {missing}\n")
        return [t for t in requested if t in available]
    tags = [t for t in DEFAULT_TAGS if t in available]
    tags += sorted(t for t in available if t.startswith("Episode_Reward/"))
    return tags


def summarize(acc, tag: str, last: int) -> dict:
    events = acc.Scalars(tag)
    values = [e.value for e in events]
    steps = [e.step for e in events]
    window = values[-last:]
    peak_i = max(range(len(values)), key=lambda i: values[i])
    return {
        "tag": tag,
        "n_points": len(values),
        "window": len(window),
        "mean": sum(window) / len(window),
        "final": values[-1],
        "final_step": steps[-1],
        "peak": values[peak_i],
        "peak_step": steps[peak_i],
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if not os.path.isdir(run_dir):
        sys.stderr.write(f"error: no such run directory: {run_dir}\n")
        return 2

    acc = load_accumulator(run_dir)
    available = set(acc.Tags().get("scalars", []))
    if not available:
        sys.stderr.write(
            f"error: no scalar tags found under {run_dir}. Is this the run "
            "directory itself (the one containing events.out.tfevents.*)?\n")
        return 2

    tags = select_tags(available, args.tags)
    if not tags:
        sys.stderr.write("error: no requested tag exists in this run.\n")
        return 2

    rows = [summarize(acc, t, args.last) for t in tags]

    # Wall clock from the run's own event timestamps (rule 2).
    probe = acc.Scalars(tags[0])
    t0, t1 = probe[0].wall_time, probe[-1].wall_time
    elapsed = datetime.timedelta(seconds=int(t1 - t0))
    first_iso = datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds")
    last_iso = datetime.datetime.fromtimestamp(t1).isoformat(timespec="seconds")

    header = (f"| Tag | last-{args.last} mean | final | final step | peak | "
              f"peak step | points |")
    if args.markdown:
        print(f"**Run:** `{run_dir}`  ")
        print(f"**Wall clock:** {elapsed} ({first_iso} -> {last_iso})  ")
        print(f"**Iterations logged:** {rows[0]['final_step']}")
        print()
        print(header)
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| `{r['tag']}` | {r['mean']:.4f} | {r['final']:.4f} | "
                  f"{r['final_step']} | {r['peak']:.4f} | {r['peak_step']} | "
                  f"{r['n_points']} |")
    else:
        print(f"run:        {run_dir}")
        print(f"wall clock: {elapsed}  ({first_iso} -> {last_iso})")
        print(f"iterations: {rows[0]['final_step']}")
        print()
        width = max(len(r["tag"]) for r in rows)
        print(f"{'tag':<{width}}  {f'last-{args.last}':>14}{'final':>14}"
              f"{'peak':>14}{'peak@':>10}{'pts':>7}")
        print("-" * (width + 61))
        for r in rows:
            note = "" if r["window"] == args.last else \
                f"  (only {r['window']} points)"
            print(f"{r['tag']:<{width}}  {r['mean']:>14.4f}{r['final']:>14.4f}"
                  f"{r['peak']:>14.4f}{r['peak_step']:>10}{r['n_points']:>7}"
                  f"{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
