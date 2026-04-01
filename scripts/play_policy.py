#!/usr/bin/env python3
"""Evaluate a trained PPO policy for the Open Duck Mini v2.

Loads a checkpoint and runs the policy in the PLAY environment
(fewer envs, no randomization, fixed velocity command) to generate
evaluation videos.

Usage:
    cd /home/xiaohui_chen/IsaacLab
    PYTHONPATH="/path/to/Open_Duck_Mini_Jetson:$PYTHONPATH" \
    ./isaaclab.sh -p /path/to/play_policy.py \
        --task Isaac-Velocity-Rough-OpenDuck-Play-v0 \
        --num_envs 4 \
        --checkpoint <path_to_model.pt> \
        --headless --video --video_length 500
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import isaac_lab_env  # noqa: F401

ISAACLAB_ROOT = os.path.expanduser("~/IsaacLab")
RSL_RL_DIR = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

train_script = os.path.join(RSL_RL_DIR, "play.py")
exec(compile(open(train_script).read(), train_script, "exec"))
