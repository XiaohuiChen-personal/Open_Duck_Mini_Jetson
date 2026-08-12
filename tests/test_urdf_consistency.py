"""The URDF and the MJCF must describe the same robot.

Why this exists
---------------
This repo carries two robot descriptions of the same machine:

  * ``robot_motors.xml`` (MJCF) — the source of truth. The Isaac/PhysX USD is
    generated from it by ``scripts/convert_mjcf_to_usd.py``, so every trained
    policy is trained against this file.
  * ``robot.urdf`` — the standard interchange format (RViz / MoveIt / Gazebo).
    It is in no training or evaluation code path.

A second description that silently drifts from the first is worse than having no
second description at all, because a reader cannot tell which one is current.
That failure has already happened twice in this project's documentation, and it
is exactly what ``task_plan_v2.md`` Task M7 was written to prevent: the URDF was
kept but had never validated, and nothing checked it against the MJCF.

These tests parse **both** files and compare the structures. They deliberately
do not grep source text — most of this repo's existing suite does, and 5 of 7
seeded regressions pass it (``known_issues.md`` TEST-1).

What is compared, and what is not
---------------------------------
Compared: movable joint names, joint parent/child topology, the set of body/link
names, and total mass.

Not compared: inertia tensors and joint limits. The URDF's are inherited from
the upstream Onshape export and the mass model is being rebuilt in Phase M, so
requiring them to match today would mean asserting a number that is known to be
provisional. See the design ledger at the top of ``robot.urdf``.

Representation differences that are expected, not drift
-------------------------------------------------------
* MuJoCo spells a 1-DOF rotary joint ``hinge``; URDF spells it ``revolute``.
* The MJCF gives the root body a ``<freejoint>``. URDF has no such element — a
  URDF root is implicitly floating — so the freejoint has no URDF counterpart.
* URDF needs an explicit ``fixed`` joint to attach a frame-only link; MJCF just
  nests a child body with no joint. So the URDF has 4 extra ``fixed`` joints
  that correspond to the 4 MJCF marker bodies, and this test does not require a
  1:1 joint count.
"""

import os
import xml.etree.ElementTree as ET

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_DIR = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")
URDF_PATH = os.path.join(ROBOT_DIR, "robot.urdf")
MJCF_PATH = os.path.join(ROBOT_DIR, "robot_motors.xml")


def _urdf():
    """(links, joints) from the URDF. joints: name -> (type, parent, child)."""
    root = ET.parse(URDF_PATH).getroot()
    links = {l.get("name") for l in root.findall("link")}
    joints = {
        j.get("name"): (
            j.get("type"),
            j.find("parent").get("link"),
            j.find("child").get("link"),
        )
        for j in root.findall("joint")
    }
    return links, joints


def _mjcf():
    """(bodies, joints) from the MJCF. joints: name -> (type, parent, child).

    Walks the body tree so each joint's parent/child is read from the nesting,
    which is how MJCF expresses topology.
    """
    root = ET.parse(MJCF_PATH).getroot()
    world = root.find("worldbody")
    bodies, joints = set(), {}

    def walk(elem, parent_name):
        for body in elem.findall("body"):
            name = body.get("name")
            bodies.add(name)
            for j in body.findall("joint"):
                # a joint on a body connects that body to its parent
                joints[j.get("name")] = (
                    j.get("type", "hinge"),  # MJCF defaults to hinge
                    parent_name,
                    name,
                )
            walk(body, name)

    walk(world, None)
    return bodies, joints


# MuJoCo <-> URDF spelling of the same 1-DOF rotary joint.
_MOVABLE_MJCF = {"hinge"}
_MOVABLE_URDF = {"revolute", "continuous", "prismatic"}


@pytest.mark.phase1
class TestUrdfMjcfConsistency:
    def test_both_descriptions_exist(self):
        assert os.path.exists(URDF_PATH), f"missing {URDF_PATH}"
        assert os.path.exists(MJCF_PATH), f"missing {MJCF_PATH}"

    def test_movable_joint_names_match(self):
        """Every actuated joint must exist in both files under the same name."""
        _, uj = _urdf()
        _, mj = _mjcf()
        urdf_movable = {n for n, (t, _, _) in uj.items() if t in _MOVABLE_URDF}
        mjcf_movable = {n for n, (t, _, _) in mj.items() if t in _MOVABLE_MJCF}
        assert urdf_movable == mjcf_movable, (
            f"movable joints differ.\n"
            f"  only in URDF: {sorted(urdf_movable - mjcf_movable)}\n"
            f"  only in MJCF: {sorted(mjcf_movable - urdf_movable)}"
        )

    def test_movable_joint_topology_matches(self):
        """Each shared joint must connect the same two links in both files."""
        _, uj = _urdf()
        _, mj = _mjcf()
        mismatched = []
        for name, (t, up, uc) in uj.items():
            if t not in _MOVABLE_URDF or name not in mj:
                continue
            _, mp, mc = mj[name]
            if (up, uc) != (mp, mc):
                mismatched.append(f"{name}: URDF {up}->{uc} vs MJCF {mp}->{mc}")
        assert not mismatched, "joint topology differs:\n  " + "\n  ".join(mismatched)

    def test_link_and_body_names_match(self):
        """The two files must describe the same set of rigid bodies."""
        ulinks, _ = _urdf()
        mbodies, _ = _mjcf()
        assert ulinks == mbodies, (
            f"body/link sets differ.\n"
            f"  only in URDF: {sorted(ulinks - mbodies)}\n"
            f"  only in MJCF: {sorted(mbodies - ulinks)}"
        )

    def test_root_link_is_the_same_and_carries_mass(self):
        """Both must be rooted on trunk_assembly, and it must be a real link.

        A massless root is the PLANT-1 defect: PhysX cannot represent one and
        substitutes a 1.000 kg default, which is how every policy through v5d
        came to train on a robot 37.6% too heavy.
        """
        uroot = ET.parse(URDF_PATH).getroot()
        ulinks, ujoints = _urdf()
        children = {c for (_, _, c) in ujoints.values()}
        urdf_roots = ulinks - children
        assert urdf_roots == {"trunk_assembly"}, f"URDF root is {urdf_roots}"

        mroot = ET.parse(MJCF_PATH).getroot()
        first = mroot.find("worldbody").find("body")
        assert first.get("name") == "trunk_assembly", (
            f"MJCF root body is {first.get('name')!r}, expected 'trunk_assembly'"
        )
        inertial = first.find("inertial")
        assert inertial is not None and float(inertial.get("mass")) > 0.1, (
            "the MJCF root body must carry real mass (see known_issues.md PLANT-1)"
        )

    def test_total_mass_agrees(self):
        """Total mass must agree between the two descriptions."""
        u = sum(
            float(i.find("mass").get("value"))
            for i in ET.parse(URDF_PATH).getroot().iter("inertial")
            if i.find("mass") is not None
        )
        m = sum(
            float(i.get("mass"))
            for i in ET.parse(MJCF_PATH).getroot().iter("inertial")
            if i.get("mass") is not None
        )
        assert abs(u - m) < 1e-3, f"total mass differs: URDF {u:.6f} kg vs MJCF {m:.6f} kg"

    def test_every_urdf_mesh_reference_resolves(self):
        """A mesh path that does not resolve makes the URDF unrenderable.

        Every reference used to be ``package:///<name>.stl`` — a package:// URI
        with an empty package name — so none of the 266 resolved and no consumer
        could load the robot.
        """
        root = ET.parse(URDF_PATH).getroot()
        missing = set()
        for mesh in root.iter("mesh"):
            fn = mesh.get("filename")
            assert fn, "a <mesh> has no filename"
            assert not fn.startswith("package:///"), (
                f"{fn!r} uses an empty package name; use a path relative to the URDF"
            )
            if not fn.startswith(("package://", "file://", "/")):
                if not os.path.exists(os.path.join(ROBOT_DIR, fn)):
                    missing.add(fn)
        assert not missing, f"mesh files not found beside the URDF: {sorted(missing)}"
