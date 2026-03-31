#!/usr/bin/env python3
"""Monitor PPO training progress by reading TensorBoard logs.

Reads the TensorBoard event files from the training log directory and
extracts key metrics at regular intervals.

Usage:
    python3 scripts/monitor_training.py <log_dir>
"""

import os
import sys
import json
import glob
import time
from datetime import datetime


def read_tensorboard_events(log_dir):
    """Read metrics from TensorBoard event files using tensorboard.backend."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        # Fallback: try to read from the RSL-RL CSV logs
        return read_rsl_rl_logs(log_dir)

    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None

    ea = EventAccumulator(log_dir)
    ea.Reload()

    metrics = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            latest = events[-1]
            metrics[tag] = {"step": latest.step, "value": latest.value}

    return metrics


def read_rsl_rl_logs(log_dir):
    """Fallback: parse RSL-RL console output for metrics."""
    # RSL-RL logs to tensorboard, check for event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    return {"_event_files": len(event_files)} if event_files else None


def find_latest_run(base_dir):
    """Find the most recent training run directory."""
    runs = sorted(glob.glob(os.path.join(base_dir, "2026-*")))
    return runs[-1] if runs else None


def main():
    base_dir = "/home/xiaohui_chen/IsaacLab/logs/rsl_rl/open_duck_ppo"
    output_file = "/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/experiments/v2/ppo_training/training_log.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    log_dir = find_latest_run(base_dir)
    if not log_dir:
        print("No training run found yet")
        return

    print(f"Monitoring: {log_dir}")
    print(f"Logging to: {output_file}")

    metrics = read_tensorboard_events(log_dir)
    if metrics:
        # Write metrics
        entry = {
            "timestamp": datetime.now().isoformat(),
            "log_dir": log_dir,
            "metrics": metrics,
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(metrics, indent=2))
    else:
        # List checkpoint files as progress indicator
        checkpoints = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
        print(f"Checkpoints found: {len(checkpoints)}")
        if checkpoints:
            latest = checkpoints[-1]
            iteration = os.path.basename(latest).replace("model_", "").replace(".pt", "")
            print(f"Latest checkpoint: iteration {iteration}")


if __name__ == "__main__":
    main()
