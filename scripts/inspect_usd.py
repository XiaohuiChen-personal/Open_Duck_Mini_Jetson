#!/usr/bin/env python3
"""Inspect the USD structure to find the correct articulation root."""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

usd_path = os.path.join(
    REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "usd", "open_duck_mini_v2.usd"
)

stage = Usd.Stage.Open(usd_path)
lines = []
lines.append(f"USD structure for: {usd_path}\n")

for prim in stage.Traverse():
    indent = "  " * (len(prim.GetPath().pathString.split("/")) - 1)
    type_name = prim.GetTypeName()

    is_artic = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    is_joint = prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)
    is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)

    extras = []
    if is_artic:
        extras.append("ARTICULATION_ROOT")
    if is_rigid:
        extras.append("RIGID_BODY")
    if is_joint:
        extras.append("JOINT")

    extra_str = f"  [{', '.join(extras)}]" if extras else ""
    lines.append(f"{indent}{prim.GetPath()} ({type_name}){extra_str}")

output = "\n".join(lines)
with open("/tmp/usd_structure.txt", "w") as f:
    f.write(output)

simulation_app.close()
