#!/usr/bin/env bash
# Launch an Isaac Lab training run fully detached from the calling session.
#
# The run gets its own process session (setsid) with no controlling terminal,
# so it survives SSH disconnects, shell exits, and the death of whatever
# process launched it (verified: logind KillUserProcesses=no on this machine).
#
# Usage:
#   launch_training_detached.sh <run_name> [training args passed to train_ppo.py...]
# Example:
#   ./launch_training_detached.sh v3_full \
#       --task Isaac-Velocity-Rough-OpenDuck-v0 --headless --max_iterations 3000 \
#       --video --video_length 200 --video_interval 5000
#
# Monitor:  tail -f .training_runs/<run_name>.log
# Stop:     kill -- -$(cat .training_runs/<run_name>.pid)
set -euo pipefail

RUN_NAME="${1:?usage: launch_training_detached.sh <run_name> [training args...]}"
shift

ISAACLAB_DIR="$HOME/IsaacLab"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$REPO_DIR/.training_runs"
mkdir -p "$RUNS_DIR"
LOG="$RUNS_DIR/${RUN_NAME}.log"
PIDFILE="$RUNS_DIR/${RUN_NAME}.pid"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "ERROR: run '$RUN_NAME' appears to be alive (PID $(cat "$PIDFILE"))." >&2
    echo "Stop it first or pick a different run name." >&2
    exit 1
fi

cd "$ISAACLAB_DIR"
setsid nohup ./isaaclab.sh -p "$REPO_DIR/scripts/train_ppo.py" "$@" \
    < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

echo "Detached training run '$RUN_NAME' started (survives SSH disconnect)."
echo "  PID:     $PID"
echo "  Log:     $LOG"
echo "  Monitor: tail -f $LOG"
echo "  Stop:    kill -- -$PID"
