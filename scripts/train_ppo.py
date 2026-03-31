#!/usr/bin/env python3
"""Train PPO locomotion policy for the Open Duck Mini v2.

Wrapper that sets up PYTHONPATH for isaac_lab_env discovery, then
delegates to Isaac Lab's RSL-RL training script.

Usage:
    cd /home/xiaohui_chen/IsaacLab
    ./isaaclab.sh -p /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/scripts/train_ppo.py \
        --task Isaac-Velocity-Rough-OpenDuck-v0 \
        --headless --video --video_length 200 --video_interval 5000
"""

import os
import sys

# Add the repo root to Python path so isaac_lab_env is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import our environment package BEFORE AppLauncher parses args.
# This registers our Gymnasium environments so --task can find them.
import isaac_lab_env  # noqa: F401 — triggers gym.register()

# Add the RSL-RL scripts directory to sys.path so `import cli_args` works
ISAACLAB_ROOT = os.path.expanduser("~/IsaacLab")
RSL_RL_DIR = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

# Now exec the standard training script — it gets our registered environments
train_script = os.path.join(RSL_RL_DIR, "train.py")
exec(compile(open(train_script).read(), train_script, "exec"))
