#!/usr/bin/env bash
# Autonomous chain: v4 control battery -> train next arm -> evaluate it.
#
# The GPU takes one Isaac job at a time (AGENTS.md rule 7), so the only way to
# make progress unattended is to serialise everything into a single background
# process. This is that process.
#
# Step 1 fills a real gap: v5c's contact numbers (0.29% pushed falls, 100%
# wrench falls, 6.22% obstacle falls) have no v4 counterpart measured under the
# same protocol, and "better than v4" is the acceptance rule. v4's only contact
# number on record is 6.84% pushed falls from a 5-condition run on a different
# task. Without the control, none of v5c's battery can be scored.
#
# Usage: scripts/v5_chain.sh <next_run_name> <train_task> <play_task>

set -uo pipefail

RUN_NAME="${1:?usage: v5_chain.sh <run_name> <train_task> <play_task>}"
TRAIN_TASK="${2:?}"
PLAY_TASK="${3:?}"

REPO=/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
ISAACLAB=/home/xiaohui_chen/IsaacLab
RESULTS=$REPO/docs/jetson-mod/eval_results_v5
STATE=$REPO/.training_runs
V4CKPT=$ISAACLAB/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt
CONDITIONS="0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2;0,0,0.5"

exec > >(tee -a "$STATE/chain_${RUN_NAME}.log") 2>&1
echo "############ chain start $(date '+%F %T') -> $RUN_NAME ############"

wait_for_gpu () {
    local pat="train""_ppo.py"
    while pgrep -f "$pat" >/dev/null 2>&1 || pgrep -f "evaluate""_policies" >/dev/null 2>&1; do
        sleep 60
    done
    sleep 20
}

eval_ckpt () {  # name, task, ckpt, extra...
    local name="$1"; local task="$2"; local ckpt="$3"; shift 3
    if [ -f "$RESULTS/$name.json" ]; then echo "[chain] SKIP $name (exists)"; return 0; fi
    echo "[chain] --- eval $name ---"
    wait_for_gpu
    ( cd "$ISAACLAB" && ./isaaclab.sh -p "$REPO/scripts/evaluate_policies.py" \
        --policies "$name=$task:rsl_rl:$ckpt" \
        --conditions "$CONDITIONS" \
        --output_dir "$RESULTS" \
        --comparison_md "$REPO/docs/jetson-mod/v5_comparison.md" \
        --headless "$@" ) >> "$STATE/chain_${name}.log" 2>&1
    echo "[chain] eval $name exit=$?"
    sleep 20
}

# ----------------------------------------------------------------------
# 1. v4_robust control battery — the baseline every v5 contact number is
#    scored against. Same tasks, same 6 conditions, same seed as the v5 runs.
# ----------------------------------------------------------------------
echo "[chain] === step 1: v4_robust contact controls ==="
eval_ckpt "v4_robust_pusheval_v4def"  "Isaac-Velocity-Rough-OpenDuck-PushEval-v0"        "$V4CKPT" --keep-pushes
eval_ckpt "v4_robust_pusheval_v5def"  "Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0" "$V4CKPT" --keep-pushes
eval_ckpt "v4_robust_wrencheval"      "Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0"      "$V4CKPT"
eval_ckpt "v4_robust_obstacleeval"    "Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0"    "$V4CKPT"

# ----------------------------------------------------------------------
# 2. Train the next arm.
# ----------------------------------------------------------------------
echo "[chain] === step 2: train $RUN_NAME ==="
wait_for_gpu
rm -f "$STATE/${RUN_NAME}.pid"
( cd "$REPO" && ./scripts/launch_training_detached.sh "$RUN_NAME" \
    --task "$TRAIN_TASK" --headless \
    --max_iterations 3000 --resume \
    --load_run 0000-00-00_v4robust_seed --checkpoint model_2999.pt \
    --video --video_length 200 --video_interval 5000 )
sleep 120

# Early-abort watchdog: the gait canary reveals a doomed run in minutes.
# v5a burned 2.22 h converging to a standing policy that every training signal
# called healthy; duty_in_band_frac would have shown it almost immediately.
( sleep 1800
  LOGF="$STATE/${RUN_NAME}.log"
  BAND=$(grep -E "Gait/duty_in_band_frac" "$LOGF" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
  if [ -n "$BAND" ]; then
      BAD=$(python3 -c "print(1 if float('$BAND') < 0.30 else 0)" 2>/dev/null)
      if [ "$BAD" = "1" ]; then
          echo "[chain] WATCHDOG: duty_in_band_frac=$BAND < 0.30 after 30 min — aborting $RUN_NAME"
          P=$(cat "$STATE/${RUN_NAME}.pid" 2>/dev/null); [ -n "$P" ] && kill -9 -- -"$P" 2>/dev/null
      else
          echo "[chain] WATCHDOG: duty_in_band_frac=$BAND — healthy, letting it run"
      fi
  fi ) &

# ----------------------------------------------------------------------
# 3. Evaluate it (the pipeline waits for training and gates on the gait gate).
# ----------------------------------------------------------------------
echo "[chain] === step 3: evaluate $RUN_NAME ==="
"$REPO/scripts/v5_pipeline.sh" "$RUN_NAME" "$PLAY_TASK"

echo "############ chain done $(date '+%F %T') ############"
