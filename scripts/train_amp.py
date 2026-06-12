#!/usr/bin/env python3
"""Train AMP locomotion policy for the Open Duck Mini v2 via skrl.

Wrapper that sets up PYTHONPATH for isaac_lab_env discovery, then
delegates to Isaac Lab's skrl training script. With ``--algorithm AMP``
the skrl script resolves the ``skrl_amp_cfg_entry_point`` registered by
``isaac_lab_env/open_duck_mini_v2/amp/__init__.py``.

Usage:
    cd /home/xiaohui_chen/IsaacLab
    ./isaaclab.sh -p /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/scripts/train_amp.py \
        --task Isaac-OpenDuck-AMP-PureStyle-v0 \
        --algorithm AMP --headless --num_envs 4096 \
        --video --video_length 200 --video_interval 5000

Tasks:
    Isaac-OpenDuck-AMP-PureStyle-v0 : Pure style imitation (sanity check)
    Isaac-OpenDuck-AMP-v0           : Command-conditioned locomotion
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

# Add the skrl scripts directory to sys.path (mirrors the RSL-RL wrapper)
ISAACLAB_ROOT = os.path.expanduser("~/IsaacLab")
SKRL_DIR = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "skrl")
if SKRL_DIR not in sys.path:
    sys.path.insert(0, SKRL_DIR)

# Now exec the standard training script — it gets our registered environments
train_script = os.path.join(SKRL_DIR, "train.py")
exec(compile(open(train_script).read(), train_script, "exec"))
