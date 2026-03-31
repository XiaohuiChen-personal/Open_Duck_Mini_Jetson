#!/usr/bin/env python3
"""Debug contact sensor to find correct body names for feet_air_time reward."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

from isaac_lab_env.open_duck_mini_v2.robot_cfg import OPEN_DUCK_MINI_V2_CFG

OUTPUT = []


def log(msg=""):
    OUTPUT.append(msg)


sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
sim = SimulationContext(sim_cfg)

robot_cfg = OPEN_DUCK_MINI_V2_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Create contact sensor matching our env config
contact_cfg = ContactSensorCfg(
    prim_path="/World/Robot/base/.*",
    history_length=3,
    track_air_time=True,
)
contact_sensor = ContactSensor(contact_cfg)

sim_utils.spawn_ground_plane("/World/GroundPlane", sim_utils.GroundPlaneCfg())

sim.reset()
robot.update(sim.cfg.dt)
contact_sensor.update(sim.cfg.dt)

log("=" * 60)
log("CONTACT SENSOR DEBUG")
log("=" * 60)

# List all body names the contact sensor covers
log(f"\nContact sensor prim path: {contact_cfg.prim_path}")
log(f"Number of bodies in sensor: {contact_sensor.data.net_forces_w.shape}")
log(f"Body names in sensor: {contact_sensor.body_names}")

# Check which bodies have "foot" in the name
log("\n--- Bodies containing 'foot' ---")
for i, name in enumerate(contact_sensor.body_names):
    if "foot" in name.lower():
        log(f"  [{i}] {name}")

# Simulate a few steps and check contacts
log("\n--- Simulating 100 steps, checking contact forces ---")
for step in range(100):
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.cfg.dt)
    contact_sensor.update(sim.cfg.dt)

# Print contact forces for all bodies
log("\n--- Contact forces after 100 steps ---")
forces = contact_sensor.data.net_forces_w[0]  # first env
for i, name in enumerate(contact_sensor.body_names):
    force_mag = torch.norm(forces[i]).item()
    if force_mag > 0.01:
        log(f"  [{i}] {name}: force={force_mag:.3f} N")

# Check air time data
log(f"\n--- Air time data shape: {contact_sensor.data.current_air_time.shape}")
log(f"--- Air time values (first env): {contact_sensor.data.current_air_time[0]}")

# Try to find the body indices for "left_foot" and "right_foot"
log("\n--- Looking for 'left_foot' and 'right_foot' ---")
body_names_list = list(contact_sensor.body_names)
for target in ["left_foot", "right_foot", "foot_assembly", "foot_assembly_2"]:
    if target in body_names_list:
        idx = body_names_list.index(target)
        force = torch.norm(forces[idx]).item()
        air_time = contact_sensor.data.current_air_time[0, idx].item()
        log(f"  FOUND: '{target}' at index {idx}, force={force:.3f} N, air_time={air_time:.3f} s")
    else:
        log(f"  NOT FOUND: '{target}'")

# Write output
with open("/tmp/contact_debug.txt", "w") as f:
    f.write("\n".join(OUTPUT))

simulation_app.close()
