#!/usr/bin/env bash
# End-to-end v5 pipeline: wait for a training run to finish, then run the full
# evaluation battery, gating the expensive parts on the cheap ones.
#
# WHY THIS EXISTS
# ---------------
# Training runs take ~2.2 h and evaluations ~30 min each. Driving that by hand
# means a human (or an agent) has to keep checking back, and the checking is
# what gets dropped. This script owns the whole mechanical sequence so the only
# thing left to a person is the judgement call at the end.
#
# It encodes one decision rule directly: the gait-validity gate is measured
# FIRST, and the rest of the battery only runs if the candidate passes it.
# Run v5a_gated_ft scored 0/6 on that gate (a standing policy) — running push,
# wrench and obstacle evals on a policy that cannot walk would have burned an
# hour of GPU to characterise something already disqualified.
#
# USAGE
#   scripts/v5_pipeline.sh <run_name> <play_task_id>
# e.g.
#   scripts/v5_pipeline.sh v5b_ungated_ft \
#       Isaac-Velocity-Rough-OpenDuck-ContactUngated-Play-v0
#
# Safe to start while the training is still running: it waits for it.

set -uo pipefail

RUN_NAME="${1:?usage: v5_pipeline.sh <run_name> <play_task_id>}"
PLAY_TASK="${2:?usage: v5_pipeline.sh <run_name> <play_task_id>}"

REPO=/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson
ISAACLAB=/home/xiaohui_chen/IsaacLab
LOGROOT=$ISAACLAB/logs/rsl_rl/open_duck_ppo_v5
RESULTS=$REPO/docs/jetson-mod/eval_results_v5
STATE=$REPO/.training_runs
PIPELOG=$STATE/${RUN_NAME}_pipeline.log

CONDITIONS="0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2;0,0,0.5"

mkdir -p "$RESULTS"
exec > >(tee -a "$PIPELOG") 2>&1
echo "=============================================================="
echo "[pipeline] $RUN_NAME  ($(date '+%F %T'))"
echo "=============================================================="

# ----------------------------------------------------------------------
# 1. Wait for training to finish.
#
# AGENTS.md rule 6: elapsed time is not completion. Require BOTH the process
# group to be dead AND a final checkpoint to exist. Note the checkpoint index:
# a resumed run restores its iteration counter, so a 3000-iteration fine-tune
# from model_2999.pt ends at model_5998.pt, not model_5999.pt.
# ----------------------------------------------------------------------
PIDFILE=$STATE/${RUN_NAME}.pid
if [ -f "$PIDFILE" ]; then
    PGID=$(cat "$PIDFILE")
    echo "[pipeline] waiting for training pgid $PGID ..."
    while kill -0 -- -"$PGID" 2>/dev/null; do sleep 60; done
    echo "[pipeline] training process group exited"
else
    echo "[pipeline] no pidfile for $RUN_NAME; assuming training already done"
fi

# Newest run dir that is not the seeded v4 checkpoint copy.
RUNDIR=$(ls -td "$LOGROOT"/*/ 2>/dev/null | grep -v v4robust_seed | head -1)
if [ -z "$RUNDIR" ]; then echo "[pipeline] FATAL: no run dir under $LOGROOT"; exit 1; fi
CKPT=$(ls "$RUNDIR"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$CKPT" ]; then echo "[pipeline] FATAL: no checkpoint in $RUNDIR"; exit 1; fi
echo "[pipeline] run dir    : $RUNDIR"
echo "[pipeline] checkpoint : $CKPT"

# Give the GPU a moment to drain before the first Isaac job (rule 7: one job
# at a time; two Isaac processes collide during kit startup).
sleep 30

run_eval () {  # name, task, extra args...
    local name="$1"; local task="$2"; shift 2
    if [ -f "$RESULTS/$name.json" ]; then
        echo "[pipeline] SKIP $name (already present)"; return 0
    fi
    echo "[pipeline] --- eval: $name on $task ---"
    ( cd "$ISAACLAB" && ./isaaclab.sh -p "$REPO/scripts/evaluate_policies.py" \
        --policies "$name=$task:rsl_rl:$CKPT" \
        --conditions "$CONDITIONS" \
        --output_dir "$RESULTS" \
        --comparison_md "$REPO/docs/jetson-mod/v5_comparison.md" \
        --headless "$@" ) >> "$STATE/${RUN_NAME}_${name}.log" 2>&1
    local rc=$?
    echo "[pipeline] eval $name exit=$rc"
    sleep 20
    return $rc
}

# ----------------------------------------------------------------------
# 2. The gate first — cheapest disqualifier.
# ----------------------------------------------------------------------
if ! run_eval "$RUN_NAME" "$PLAY_TASK"; then
    echo "[pipeline] FATAL: the gate eval itself failed. Aborting."; exit 1
fi

GATE=$(python3 - "$RESULTS/$RUN_NAME.json" <<'PYG'
import json, sys
# Errors go to STDERR and exit non-zero. The earlier version printed "ERR ..."
# to STDOUT, so $GATE was non-empty, PASSED became "ERR", and the numeric test
# `[ "ERR" -lt 5 ]` errored -> status 2 -> `if` read it as FALSE -> the script
# announced "gait gate passed" on a MISSING eval file. The gate was inverted
# exactly where it mattered.
try:
    a = json.load(open(sys.argv[1]))["aggregate"]
except Exception as e:
    print(f"cannot read gate result: {e}", file=sys.stderr); sys.exit(1)
print(f"{a['gait_valid_conditions']} {a['conditions']} {a['fall_rate_pct']:.3f} "
      f"{a['reference_tracking_rms_deg']:.3f} {a['stance_duty_left_pct']:.2f} "
      f"{a['stance_duty_right_pct']:.2f}")
PYG
)
GATE_RC=$?
if [ "$GATE_RC" -ne 0 ] || [ -z "$GATE" ]; then
    echo "[pipeline] FATAL: could not read the gate result (rc=$GATE_RC). Aborting"
    echo "[pipeline] rather than assuming a pass."
    exit 1
fi
set -- $GATE
PASSED=$1; TOTAL=$2; FALLS=$3; RMS=$4; DUTYL=$5; DUTYR=$6
case "$PASSED" in
  ''|*[!0-9]*) echo "[pipeline] FATAL: non-numeric gate count '$PASSED'"; exit 1 ;;
esac
echo "[pipeline] GATE $PASSED/$TOTAL | falls ${FALLS}% | RMS ${RMS} deg | duty ${DUTYL}/${DUTYR}"
echo "[pipeline] baseline v4_robust: 6/6 | 0.000% | 4.48 deg | 68.7/63.7"

if [ "$PASSED" -lt 5 ]; then
    echo "[pipeline] VERDICT: FAIL — gait gate $PASSED/$TOTAL (< 5). Contact battery SKIPPED."
    echo "[pipeline] A policy that cannot walk does not need its contact robustness characterised."
    echo "[pipeline] DONE $(date '+%F %T')"
    exit 0
fi

echo "[pipeline] gait gate passed — running the contact battery"

# ----------------------------------------------------------------------
# 3. Contact battery (only reached by a policy that walks).
# ----------------------------------------------------------------------
run_eval "${RUN_NAME}_pusheval_v4def"  "Isaac-Velocity-Rough-OpenDuck-PushEval-v0"        --keep-pushes
run_eval "${RUN_NAME}_pusheval_v5def"  "Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0" --keep-pushes
run_eval "${RUN_NAME}_wrencheval"      "Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0"
run_eval "${RUN_NAME}_obstacleeval"    "Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0"

# ----------------------------------------------------------------------
# 4. Videos for the mandatory frame-by-frame audit.
# Protocol requires >= 2 conditions; the v5 failure modes need the turn and
# the obstacle graze specifically, so render four.
# ----------------------------------------------------------------------
play () {  # tag, task, hydra overrides...
    local tag="$1"; local task="$2"; shift 2
    echo "[pipeline] --- video: $tag ---"
    ( cd "$ISAACLAB" && ./isaaclab.sh -p "$REPO/scripts/play_policy.py" \
        --task "$task" --num_envs 2 --checkpoint "$CKPT" \
        --headless --video --video_length 1000 "$@" ) \
        >> "$STATE/${RUN_NAME}_video_${tag}.log" 2>&1
    echo "[pipeline] video $tag exit=$?"
    local vdir="$RUNDIR/videos/play"
    local newest
    newest=$(ls -t "$vdir"/*.mp4 2>/dev/null | head -1)
    [ -n "$newest" ] && mv "$newest" "$vdir/${RUN_NAME}_${tag}.mp4" 2>/dev/null
    sleep 20
}

play forward_vx02  "$PLAY_TASK"
play turn_wz05     "$PLAY_TASK" 'env.commands.base_velocity.ranges.ang_vel_z=[0.5,0.5]'
play press_wrench  "Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0"
play obstacle_graze "Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0"

echo "[pipeline] DONE $(date '+%F %T')"
echo "[pipeline] artifacts: $RESULTS  and  $RUNDIR/videos/play"
