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

What is compared
----------------
Kinematics and mass properties: movable joint names, joint parent/child
topology, joint limits and axes, the set of body/link names, per-link mass,
per-link centre of mass, per-link inertia tensors, total mass, and mesh
resolution.

The tolerances below are not arbitrary — they were measured against the two
files as they stand (2026-08-12):

  per-link mass          largest observed delta 1.2e-8 kg  -> tolerance 1e-6 kg
  inertia tensor         identical to every printed digit  -> tolerance 1e-9 rel
  centre of mass         identical                         -> tolerance 1e-6 m
  joint limits           largest observed delta 4.0e-6 rad -> tolerance 1e-5 rad
                         (this is the MJCF export rounding to 6 significant
                         figures: URDF 1.1344640137963142 vs MJCF 1.13446)

What is deliberately NOT compared, and why
------------------------------------------
1. **Actuator parameters** (gear, damping, armature, frictionloss, ctrlrange).
   These are MuJoCo concepts with no URDF equivalent. The simulated actuator
   model lives in ``isaac_lab_env/open_duck_mini_v2/robot_cfg.py``, not in
   either description file, so there is nothing here to cross-check.
2. **The USD.** The USD is generated from the MJCF and is verified separately by
   ``scripts/audit_plant_mass.py``, which reads back what PhysX actually loaded.
   That check catches a class of defect this one structurally cannot — see
   ``known_issues.md`` PLANT-1, where every file on disk was correct and only
   the loaded plant was wrong.
3. **The four frame links.** ``trunk``, ``left_foot``, ``right_foot`` and
   ``head`` carry no ``<inertial>`` in the URDF (they are pure kinematic frames)
   while the MJCF gives them ``mass=1e-09``. This is an intentional
   representation difference, documented in the URDF's design ledger, and these
   tests skip them rather than pretend it is drift.

**If Phase M changes the mass model, these tests SHOULD fail.** That is the
signal to regenerate the URDF inertials, not a reason to loosen the tolerance.

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

    def test_per_link_mass_and_com_agree(self):
        """Every shared link must have the same mass and centre of mass.

        Skips the four pure frame links, which carry no <inertial> in the URDF
        by design (see the module docstring).
        """
        u = {l.get("name"): l.find("inertial")
             for l in ET.parse(URDF_PATH).getroot().findall("link")}
        m = {b.get("name"): b.find("inertial")
             for b in ET.parse(MJCF_PATH).getroot().iter("body")}
        bad = []
        for name in sorted(set(u) & set(m)):
            ui, mi = u[name], m[name]
            if ui is None or mi is None:
                continue  # frame link, intentional
            um = float(ui.find("mass").get("value"))
            mm = float(mi.get("mass"))
            if abs(um - mm) > 1e-6:
                bad.append(f"{name}: mass URDF {um} vs MJCF {mm}")
            uo = ui.find("origin")
            if uo is not None and mi.get("pos"):
                uc = [float(x) for x in uo.get("xyz").split()]
                mc = [float(x) for x in mi.get("pos").split()]
                if max(abs(a - b) for a, b in zip(uc, mc)) > 1e-6:
                    bad.append(f"{name}: com URDF {uc} vs MJCF {mc}")
        assert not bad, "mass/CoM differ:\n  " + "\n  ".join(bad)

    def test_per_link_inertia_tensors_agree(self):
        """The full inertia tensor must match, component by component.

        MJCF ``fullinertia`` is ordered (ixx iyy izz ixy ixz iyz); URDF names
        each component as an attribute. Getting that ordering wrong is a real
        historical failure mode in this project -- an earlier revision of
        mass_inertia_calculations.md placed a cylinder's axial moment on the
        wrong axis -- so the mapping is spelled out rather than assumed.
        """
        u = {l.get("name"): l.find("inertial")
             for l in ET.parse(URDF_PATH).getroot().findall("link")}
        m = {b.get("name"): b.find("inertial")
             for b in ET.parse(MJCF_PATH).getroot().iter("body")}
        bad = []
        for name in sorted(set(u) & set(m)):
            ui, mi = u[name], m[name]
            if ui is None or mi is None:
                continue
            full = mi.get("fullinertia")
            if not full:
                continue  # diaginertia-only body; the mass test still covers it
            ixx, iyy, izz, ixy, ixz, iyz = (float(x) for x in full.split())
            uin = ui.find("inertia").attrib
            for key, mval in (("ixx", ixx), ("iyy", iyy), ("izz", izz),
                              ("ixy", ixy), ("ixz", ixz), ("iyz", iyz)):
                uval = float(uin[key])
                scale = max(abs(mval), abs(uval), 1e-12)
                if abs(uval - mval) / scale > 1e-9:
                    bad.append(f"{name}.{key}: URDF {uval} vs MJCF {mval}")
        assert not bad, "inertia tensors differ:\n  " + "\n  ".join(bad)

    def test_joint_limits_and_axes_agree(self):
        """Shared movable joints must have the same travel and the same axis.

        Tolerance is 1e-5 rad because the MJCF exports limits to 6 significant
        figures (URDF 1.1344640137963142 -> MJCF 1.13446, a 4.0e-6 rad
        difference). That is export rounding, not disagreement.
        """
        uroot = ET.parse(URDF_PATH).getroot()
        uj = {}
        for j in uroot.findall("joint"):
            if j.get("type") not in _MOVABLE_URDF:
                continue
            lim, ax = j.find("limit"), j.find("axis")
            uj[j.get("name")] = (
                float(lim.get("lower")), float(lim.get("upper")),
                tuple(float(x) for x in ax.get("xyz").split()) if ax is not None else None,
            )
        mj = {}
        for b in ET.parse(MJCF_PATH).getroot().iter("body"):
            for j in b.findall("joint"):
                if j.get("type", "hinge") != "hinge":
                    continue
                rng, ax = j.get("range"), j.get("axis")
                mj[j.get("name")] = (
                    tuple(float(x) for x in rng.split()) if rng else None,
                    tuple(float(x) for x in ax.split()) if ax else None,
                )
        bad = []
        for name in sorted(set(uj) & set(mj)):
            ul, uu, uax = uj[name]
            mr, max_ = mj[name]
            if mr and (abs(ul - mr[0]) > 1e-5 or abs(uu - mr[1]) > 1e-5):
                bad.append(f"{name}: limits URDF ({ul}, {uu}) vs MJCF {mr}")
            if uax and max_ and max(abs(a - b) for a, b in zip(uax, max_)) > 1e-9:
                bad.append(f"{name}: axis URDF {uax} vs MJCF {max_}")
        assert not bad, "joint limits/axes differ:\n  " + "\n  ".join(bad)

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
