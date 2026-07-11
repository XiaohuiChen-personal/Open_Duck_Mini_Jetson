#!/usr/bin/env python3
"""Render evaluation videos of a trained skrl AMP policy (Open Duck Mini v2).

Thin wrapper (mirrors play_policy.py): registers the duck AMP tasks —
including ``Isaac-OpenDuck-AMP-Video-v0``, the command env with a
robot-tracking camera and a PINNED velocity command (DuckAmpVideoEnvCfg) —
then delegates to Isaac Lab's skrl play.py, which runs deterministic mean
actions and records one mp4 via gym RecordVideo.

The video lands in ``<log_dir>/videos/play/rl-video-step-0.mp4`` where
log_dir = two directory levels above the checkpoint file (skrl play.py
convention) — i.e. the training run dir when given
``.../<run>/checkpoints/agent_72000.pt``.

Protocol (CLAUDE.md "Locomotion Policy Evaluation Protocol"): render TWO
~20 s conditions per policy — forward vx=0.2 (the DuckAmpVideoEnvCfg
default) and turn wz=0.3. RecordVideo always names the file
rl-video-step-0.mp4, so RENAME the first video before rendering the second.

Usage (from ~/IsaacLab):
    # forward condition (default pinned command vx=0.2), 1000 steps = 20 s:
    ./isaaclab.sh -p <repo>/scripts/play_amp.py \
        --task Isaac-OpenDuck-AMP-Video-v0 --algorithm AMP \
        --checkpoint <run_dir>/checkpoints/agent_72000.pt \
        --num_envs 1 --headless --video --video_length 1000

    # turn condition:
    ... 'env.command_vx_range=[0.0,0.0]' 'env.command_wz_range=[0.3,0.3]'

    # run-8-era checkpoints (2-frame discriminator, trained before the
    # action clip existed — disable it so playback matches training):
    ... env.num_amp_observations=2 env.action_clip=1000.0
"""

import os
import sys

# Add the repo root to Python path so isaac_lab_env is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import our environment package BEFORE AppLauncher parses args. This only
# registers entry-point STRINGS (lazy) — no isaaclab modules load pre-app.
import isaac_lab_env  # noqa: F401 — triggers gym.register()

# Add the skrl scripts directory to sys.path (mirrors train_amp.py)
ISAACLAB_ROOT = os.path.expanduser("~/IsaacLab")
SKRL_DIR = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "skrl")
if SKRL_DIR not in sys.path:
    sys.path.insert(0, SKRL_DIR)

# Exec the standard play script — it sees our registered Video task
play_script = os.path.join(SKRL_DIR, "play.py")
exec(compile(open(play_script).read(), play_script, "exec"))
