#!/usr/bin/env bash
# Copyright (c) 2026, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause
#
# Stage 0.7 — sequential experiment queue runner.
#
# Runs a list of training experiments ONE AT A TIME on top of
# launch_training_detached.sh. The GPU fits exactly one Isaac Lab run, so the
# runner launches each queue entry as its own detached session, waits for that
# session's process group to exit, appends a completion record, then moves on
# to the next entry.
#
# Design intent:
#   - The runner itself is detachable: 'start' re-execs this script via
#     setsid + nohup into the background (same survival properties as the
#     launcher — survives SSH disconnects, shell exits, parent death).
#   - 'stop' kills ONLY the runner. The active training run lives in its own
#     session created by launch_training_detached.sh and is NOT killed; stop
#     merely prevents the NEXT queued experiment from starting.
#   - All state lives in .training_runs/ next to the launcher's per-run
#     pid/log files: queue_runner.{pid,log,state} and queue_history.log.
#
# Queue file format (default: .training_runs/queue.txt), one experiment per
# line; '#' comments and blank lines are skipped:
#   <run_name> | <training args passed through launch_training_detached.sh>
#
# Usage:    run_experiment_queue.sh --help
# Monitor:  run_experiment_queue.sh status
#           tail -f .training_runs/queue_runner.log
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_DIR/scripts/$(basename "${BASH_SOURCE[0]}")"
LAUNCHER="$REPO_DIR/scripts/launch_training_detached.sh"
RUNS_DIR="$REPO_DIR/.training_runs"
DEFAULT_QUEUE="$RUNS_DIR/queue.txt"
RUNNER_LOG="$RUNS_DIR/queue_runner.log"
RUNNER_PIDFILE="$RUNS_DIR/queue_runner.pid"
RUNNER_STATE="$RUNS_DIR/queue_runner.state"
HISTORY_LOG="$RUNS_DIR/queue_history.log"

# Seconds between liveness polls of the active run's process group.
POLL_INTERVAL=60

# Any process whose command line matches this ERE counts as "GPU busy".
# Matches both the detached `isaaclab.sh -p .../train_ppo.py ...` wrapper and
# its python child. pgrep never reports itself, and neither this script's nor
# the launcher's command line contains the pattern.
TRAIN_PATTERN='train_(ppo|amp)\.py'

usage() {
    cat <<'EOF'
run_experiment_queue.sh — sequential experiment queue on launch_training_detached.sh

Usage:
  run_experiment_queue.sh start  [queue_file] [--force-wait]
  run_experiment_queue.sh run    [queue_file] [--force-wait]
  run_experiment_queue.sh status
  run_experiment_queue.sh stop
  run_experiment_queue.sh --dry-run [queue_file]
  run_experiment_queue.sh --help

  queue_file defaults to .training_runs/queue.txt

Commands:
  start      Detach the queue runner into its own session (setsid + nohup,
             output to .training_runs/queue_runner.log). Survives SSH
             disconnects and shell exits, like launch_training_detached.sh.
  run        Run the queue in the foreground ('start' re-execs this mode).
  status     Show runner liveness, current queue position, the active
             training run, and a tail of the active run's log.
  stop       Kill the queue runner ONLY. The currently-running training was
             launched in its own detached session and KEEPS RUNNING — stop
             just prevents the next queued experiment from starting.
             To also stop the active training run:
               kill -- -$(cat .training_runs/<run_name>.pid)
  --dry-run  Parse the queue file and print what would run, without
             launching anything.

Options:
  --force-wait  If a train_ppo.py/train_amp.py process is already running
                when an entry is due to start, wait for it to finish instead
                of aborting the queue (the GPU fits exactly one run).

Queue file format (one experiment per line):
  <run_name> | <training args...>
  - lines starting with '#' and blank lines are skipped
  - args are whitespace-split; quoting is NOT supported, so avoid argument
    values that contain spaces
  - see .training_runs/queue.example.txt for a worked example

Per-entry behavior, in order:
  1. Refuse to start (or wait, with --force-wait) while any
     train_ppo.py/train_amp.py process is running.
  2. Launch via launch_training_detached.sh <run_name> <args...>.
  3. Poll the run's pidfile process group every 60 s until it exits.
  4. Append a completion record (name, exit detection time, last 3
     'Mean reward' lines from the run log) to .training_runs/queue_history.log.
EOF
}

# Timestamped log line (goes to queue_runner.log when detached).
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Strip leading/trailing whitespace from $1.
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Print PIDs of any running training process (empty if GPU is free).
training_pids() {
    pgrep -f "$TRAIN_PATTERN" 2>/dev/null || true
}

# Return 0 if a queue runner other than this process is alive.
runner_alive() {
    local pid
    [[ -f "$RUNNER_PIDFILE" ]] || return 1
    pid="$(cat "$RUNNER_PIDFILE")"
    [[ -n "$pid" && "$pid" != "$$" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

# Parse $1 into the QUEUE_NAMES / QUEUE_ARGS parallel arrays.
# Fails loudly on malformed lines so errors surface in --dry-run and in the
# pre-detach validation done by 'start', not hours into a sweep.
parse_queue() {
    local queue_file="$1"
    local line name args lineno=0
    QUEUE_NAMES=()
    QUEUE_ARGS=()
    [[ -f "$queue_file" ]] || die "queue file not found: $queue_file"
    [[ -r "$queue_file" ]] || die "queue file not readable: $queue_file"
    while IFS= read -r line || [[ -n "$line" ]]; do
        lineno=$((lineno + 1))
        line="$(trim "$line")"
        if [[ -z "$line" || "$line" == \#* ]]; then
            continue
        fi
        if [[ "$line" != *"|"* ]]; then
            die "$queue_file:$lineno: missing '|' separator: $line"
        fi
        name="$(trim "${line%%|*}")"
        args="$(trim "${line#*|}")"
        if [[ ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
            die "$queue_file:$lineno: bad run name '$name' (use letters/digits/._- and no leading '-')"
        fi
        if [[ -z "$args" ]]; then
            die "$queue_file:$lineno: no training args for run '$name'"
        fi
        QUEUE_NAMES+=("$name")
        QUEUE_ARGS+=("$args")
    done < "$queue_file"
}

# Persist the runner's position so 'status' can report it from any shell.
# Args: queue_file total index_1based current_run phase
write_state() {
    {
        echo "QUEUE_FILE=$1"
        echo "TOTAL=$2"
        echo "INDEX=$3"
        echo "CURRENT_RUN=$4"
        echo "PHASE=$5"
        echo "UPDATED=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    } > "$RUNNER_STATE"
}

# Print the value for key $1 from the state file (empty if absent).
read_state_value() {
    [[ -f "$RUNNER_STATE" ]] || return 0
    sed -n "s/^$1=//p" "$RUNNER_STATE" | head -n 1
}

# Append a completion record for run $1 (status text $2) to the history log:
# name, exit detection time, and the last 3 'Mean reward' lines from its log.
append_history() {
    local name="$1" status="$2"
    local run_log="$RUNS_DIR/${name}.log"
    local rewards=""
    if [[ -f "$run_log" ]]; then
        rewards="$(grep -F 'Mean reward' "$run_log" | tail -n 3 \
            | sed 's/^[[:space:]]*//' || true)"
    fi
    {
        echo "=== $name ==="
        echo "status:        $status"
        echo "exit_detected: $(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "log:           $run_log"
        echo "mean_reward_last3:"
        if [[ -n "$rewards" ]]; then
            local reward_line
            while IFS= read -r reward_line; do
                echo "    $reward_line"
            done <<< "$rewards"
        else
            echo "    (no 'Mean reward' lines found)"
        fi
        echo
    } >> "$HISTORY_LOG"
}

cmd_dry_run() {
    local queue_file="$1"
    local i total busy
    parse_queue "$queue_file"
    total="${#QUEUE_NAMES[@]}"
    echo "Dry run — queue file: $queue_file"
    echo "Parsed $total experiment(s); nothing will be launched."
    echo
    for (( i = 0; i < total; i++ )); do
        printf '[%d/%d] %s\n' "$((i + 1))" "$total" "${QUEUE_NAMES[$i]}"
        printf '        would run: %s %s %s\n' \
            "$LAUNCHER" "${QUEUE_NAMES[$i]}" "${QUEUE_ARGS[$i]}"
        printf '        then wait for .training_runs/%s.pid process group to exit\n' \
            "${QUEUE_NAMES[$i]}"
    done
    echo
    busy="$(training_pids)"
    if [[ -n "$busy" ]]; then
        echo "NOTE: a training process is currently running (PID(s):" \
            "$(tr '\n' ' ' <<< "$busy")) — the queue would refuse to start" \
            "entry 1 unless --force-wait is given."
    else
        echo "GPU check: no train_ppo.py/train_amp.py process running."
    fi
}

cmd_start() {
    local queue_file="$1" force_wait="$2"
    local pid
    local extra=()
    [[ -x "$LAUNCHER" ]] || die "launcher not found or not executable: $LAUNCHER"
    if runner_alive; then
        die "queue runner already running (PID $(cat "$RUNNER_PIDFILE")) — use 'status' or 'stop'."
    fi
    # Fail fast on queue syntax errors BEFORE detaching, so the user sees
    # them in their terminal rather than buried in queue_runner.log.
    parse_queue "$queue_file"
    mkdir -p "$RUNS_DIR"
    if [[ "$force_wait" == "yes" ]]; then
        extra+=(--force-wait)
    fi
    setsid nohup "$SCRIPT_PATH" run "$queue_file" ${extra[@]+"${extra[@]}"} \
        < /dev/null > "$RUNNER_LOG" 2>&1 &
    pid=$!
    echo "$pid" > "$RUNNER_PIDFILE"
    echo "Detached queue runner started (survives SSH disconnect)."
    echo "  PID:     $pid"
    echo "  Queue:   $queue_file (${#QUEUE_NAMES[@]} experiment(s))"
    echo "  Log:     $RUNNER_LOG"
    echo "  Status:  $SCRIPT_PATH status"
    echo "  Stop:    $SCRIPT_PATH stop   (does NOT kill the active training run)"
}

cmd_run() {
    local queue_file="$1" force_wait="$2"
    local i total name args_str pidfile pid busy
    if runner_alive; then
        die "another queue runner is already running (PID $(cat "$RUNNER_PIDFILE"))."
    fi
    [[ -x "$LAUNCHER" ]] || die "launcher not found or not executable: $LAUNCHER"
    parse_queue "$queue_file"
    total="${#QUEUE_NAMES[@]}"
    mkdir -p "$RUNS_DIR"
    echo "$$" > "$RUNNER_PIDFILE"
    # Remove pid/state on any exit; convert TERM/INT (from 'stop') into a
    # normal exit so the EXIT trap still runs. Launched trainings live in
    # their own sessions and are unaffected.
    trap 'rm -f "$RUNNER_PIDFILE" "$RUNNER_STATE"' EXIT
    trap 'log "queue runner stopped by signal (active training keeps running)."; exit 143' TERM
    trap 'exit 130' INT

    log "queue runner started: $queue_file ($total experiment(s), force_wait=$force_wait)"
    if (( total == 0 )); then
        log "queue is empty — nothing to do."
        return 0
    fi

    for (( i = 0; i < total; i++ )); do
        name="${QUEUE_NAMES[$i]}"
        args_str="${QUEUE_ARGS[$i]}"

        # (1) The GPU fits exactly one run: refuse (or wait) if anything is
        # already training — including runs started outside this queue.
        write_state "$queue_file" "$total" "$((i + 1))" "$name" "checking_gpu"
        busy="$(training_pids)"
        if [[ -n "$busy" ]]; then
            if [[ "$force_wait" == "yes" ]]; then
                log "[$((i + 1))/$total] GPU busy (training PID(s): $(tr '\n' ' ' <<< "$busy")) — waiting (--force-wait)."
                write_state "$queue_file" "$total" "$((i + 1))" "$name" "waiting_for_gpu"
                while [[ -n "$(training_pids)" ]]; do
                    sleep "$POLL_INTERVAL"
                done
                log "[$((i + 1))/$total] GPU is free."
            else
                log "[$((i + 1))/$total] ERROR: a training run is already active (PID(s): $(tr '\n' ' ' <<< "$busy"))."
                log "Aborting queue. Re-run with --force-wait to wait for it instead."
                exit 1
            fi
        fi

        # (2) Launch the experiment as its own detached session.
        local arg_array=()
        read -r -a arg_array <<< "$args_str"
        log "[$((i + 1))/$total] launching '$name': $LAUNCHER $name ${arg_array[*]}"
        write_state "$queue_file" "$total" "$((i + 1))" "$name" "launching"
        if ! "$LAUNCHER" "$name" "${arg_array[@]}"; then
            append_history "$name" "LAUNCH FAILED (launcher exited non-zero)"
            log "[$((i + 1))/$total] ERROR: launcher failed for '$name' — aborting queue."
            exit 1
        fi

        # (3) Wait for the launched process group to exit. The launcher's
        # setsid makes the recorded PID a session/process-group leader, so
        # signal-0 on the negative PID is true while ANY member survives.
        pidfile="$RUNS_DIR/${name}.pid"
        if [[ ! -f "$pidfile" ]]; then
            append_history "$name" "LAUNCH FAILED (no pidfile at $pidfile)"
            log "[$((i + 1))/$total] ERROR: launcher left no pidfile for '$name' — aborting queue."
            exit 1
        fi
        pid="$(cat "$pidfile")"
        log "[$((i + 1))/$total] '$name' running (PID/PGID $pid) — polling every ${POLL_INTERVAL}s."
        write_state "$queue_file" "$total" "$((i + 1))" "$name" "running"
        while kill -0 -- "-$pid" 2>/dev/null; do
            sleep "$POLL_INTERVAL"
        done
        log "[$((i + 1))/$total] '$name' process group exited."

        # (4) Record the outcome before moving on.
        append_history "$name" "completed (process group exited)"
        log "[$((i + 1))/$total] completion record appended to $HISTORY_LOG"
    done

    log "queue finished: all $total experiment(s) processed."
}

cmd_status() {
    local runner_pid="" current run_pidfile run_log busy tp
    if [[ -f "$RUNNER_PIDFILE" ]]; then
        runner_pid="$(cat "$RUNNER_PIDFILE")"
    fi
    if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
        echo "Queue runner: RUNNING (PID $runner_pid)"
    else
        echo "Queue runner: NOT RUNNING"
        if [[ -n "$runner_pid" ]]; then
            echo "  (stale pidfile: $RUNNER_PIDFILE)"
        fi
    fi
    echo "  Runner log:  $RUNNER_LOG"
    if [[ -f "$RUNNER_STATE" ]]; then
        echo "  Queue file:  $(read_state_value QUEUE_FILE)"
        echo "  Position:    $(read_state_value INDEX) / $(read_state_value TOTAL)"
        echo "  Phase:       $(read_state_value PHASE)"
        echo "  Updated:     $(read_state_value UPDATED)"
    fi

    current="$(read_state_value CURRENT_RUN)"
    if [[ -n "$current" && "$(read_state_value PHASE)" != "running" ]]; then
        # Before the launch happens, CURRENT_RUN is the NEXT entry — a stale
        # pidfile from an earlier run with the same name must not be shown
        # as the active training.
        echo "Active run: none ('$current' is queued, not launched yet)"
    elif [[ -n "$current" ]]; then
        run_pidfile="$RUNS_DIR/${current}.pid"
        if [[ -f "$run_pidfile" ]] && kill -0 -- "-$(cat "$run_pidfile")" 2>/dev/null; then
            echo "Active run: $current (PID/PGID $(cat "$run_pidfile"), alive)"
        else
            echo "Active run: $current (process group not alive)"
        fi
        run_log="$RUNS_DIR/${current}.log"
        if [[ -f "$run_log" ]]; then
            echo "--- tail -n 15 $run_log ---"
            tail -n 15 "$run_log"
        fi
    else
        echo "Active run: none"
    fi

    # Independent of runner state: report any training process on the GPU
    # (catches runs launched outside the queue, e.g. v3_full).
    busy="$(training_pids)"
    if [[ -n "$busy" ]]; then
        echo "Training processes matching $TRAIN_PATTERN:"
        while IFS= read -r tp; do
            [[ -n "$tp" ]] || continue
            ps -o pid=,pgid=,etime=,args= -p "$tp" 2>/dev/null || true
        done <<< "$busy"
    else
        echo "Training processes matching $TRAIN_PATTERN: none"
    fi
}

cmd_stop() {
    local pid current
    if ! runner_alive; then
        echo "Queue runner is not running."
        if [[ -f "$RUNNER_PIDFILE" ]]; then
            echo "Removing stale pidfile: $RUNNER_PIDFILE"
            rm -f "$RUNNER_PIDFILE" "$RUNNER_STATE"
        fi
        return 0
    fi
    pid="$(cat "$RUNNER_PIDFILE")"
    current="$(read_state_value CURRENT_RUN)"
    echo "Stopping queue runner (PID $pid)..."
    # The detached runner is a session/group leader; kill its whole group
    # (runner + its sleep). Fall back to the single PID for foreground runs.
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "WARNING: runner still alive — escalate manually with: kill -9 -- -$pid"
    else
        echo "Queue runner stopped. No further queued experiments will start."
        rm -f "$RUNNER_PIDFILE" "$RUNNER_STATE"
    fi
    echo "NOTE: the currently-running training is NOT killed by 'stop'."
    if [[ -n "$current" ]]; then
        echo "  Active run '$current' keeps training. To stop it too:"
        echo "    kill -- -\$(cat $RUNS_DIR/$current.pid)"
    fi
}

main() {
    local cmd="${1:-}"
    if [[ -z "$cmd" ]]; then
        usage >&2
        exit 1
    fi
    shift

    local queue_file="$DEFAULT_QUEUE" force_wait="no" arg
    for arg in "$@"; do
        case "$arg" in
            --force-wait) force_wait="yes" ;;
            --help|-h)    usage; exit 0 ;;
            -*)           die "unknown option '$arg' (see --help)" ;;
            *)            queue_file="$arg" ;;
        esac
    done

    case "$cmd" in
        start)             cmd_start "$queue_file" "$force_wait" ;;
        run)               cmd_run "$queue_file" "$force_wait" ;;
        status)            cmd_status ;;
        stop)              cmd_stop ;;
        --dry-run|dry-run) cmd_dry_run "$queue_file" ;;
        --help|-h|help)    usage ;;
        *)                 die "unknown command '$cmd' (see --help)" ;;
    esac
}

main "$@"
