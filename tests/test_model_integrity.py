import pytest
import struct
import os
import numpy as np
import mujoco
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_DIR = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")


def load_stl_vertices(filepath):
    """Load vertices from a binary STL file."""
    with open(filepath, "rb") as f:
        f.read(80)  # header
        n_tri = struct.unpack("<I", f.read(4))[0]
        verts = []
        for _ in range(n_tri):
            f.read(12)  # normal
            for _ in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                verts.append([x, y, z])
            f.read(2)  # attribute
    return np.array(verts)


# --- Task 1.2: STL Mesh Tests ---


@pytest.mark.phase1
class TestJetsonMesh:

    def test_jetson_stl_exists(self):
        """Jetson STL mesh file must exist."""
        assert os.path.exists(os.path.join(ROBOT_DIR, "jetson_orin_nano.stl"))

    def test_jetson_stl_dimensions(self):
        """Jetson STL bounding box must match 103x90.5x34.77mm (in meters)."""
        vertices = load_stl_vertices(os.path.join(ROBOT_DIR, "jetson_orin_nano.stl"))
        dims = vertices.max(axis=0) - vertices.min(axis=0)
        assert abs(dims[0] - 0.103) < 0.0005, f"X dimension wrong: {dims[0]}"
        assert abs(dims[1] - 0.0905) < 0.0005, f"Y dimension wrong: {dims[1]}"
        assert abs(dims[2] - 0.03477) < 0.0005, f"Z dimension wrong: {dims[2]}"

    def test_jetson_stl_units_are_meters(self):
        """All coordinates should be < 0.2 (meters, not mm)."""
        vertices = load_stl_vertices(os.path.join(ROBOT_DIR, "jetson_orin_nano.stl"))
        assert vertices.max() < 0.2, "STL appears to be in millimeters, not meters"
        assert vertices.min() > -0.2, "STL appears to be in millimeters, not meters"

    def test_thermal_partition_stl_exists(self):
        """Thermal partition STL mesh file must exist."""
        assert os.path.exists(os.path.join(ROBOT_DIR, "thermal_partition.stl"))

    def test_thermal_partition_stl_dimensions(self):
        """Thermal partition bounding box must match 3x110x90mm (in meters)."""
        vertices = load_stl_vertices(os.path.join(ROBOT_DIR, "thermal_partition.stl"))
        dims = vertices.max(axis=0) - vertices.min(axis=0)
        assert abs(dims[0] - 0.003) < 0.0005, f"X dimension wrong: {dims[0]}"
        assert abs(dims[1] - 0.110) < 0.0005, f"Y dimension wrong: {dims[1]}"
        assert abs(dims[2] - 0.090) < 0.0005, f"Z dimension wrong: {dims[2]}"

    def test_dcdc_stl_exists(self):
        """DC-DC converter STL mesh file must exist."""
        assert os.path.exists(os.path.join(ROBOT_DIR, "dcdc_converter.stl"))


# --- Task 1.3: robot_motors.xml Tests ---


@pytest.mark.phase1
class TestRobotMotorsXML:

    def test_xml_parses_without_error(self):
        """The modified XML must parse and compile in MuJoCo."""
        model = mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, "scene.xml"))
        assert model is not None

    def test_correct_number_of_actuators(self, model):
        """Must still have exactly 16 actuators."""
        assert model.nu == 16

    def test_correct_number_of_joints(self, model):
        """Must still have all 16 named joints."""
        named_joints = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        ]
        expected_joints = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "neck_pitch", "head_pitch", "head_yaw", "head_roll",
            "left_antenna", "right_antenna",
        ]
        for j in expected_joints:
            assert j in named_joints, f"Missing joint: {j}"

    def test_no_raspberrypi_mesh(self):
        """raspberrypizerow mesh should not exist in the model."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot_motors.xml"))
        root = tree.getroot()
        meshes = [m.get("name") for m in root.iter("mesh")]
        assert "raspberrypizerow" not in meshes

    def test_jetson_mesh_exists(self):
        """jetson_orin_nano mesh must be declared in assets."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot_motors.xml"))
        root = tree.getroot()
        meshes = [m.get("name") for m in root.iter("mesh")]
        assert "jetson_orin_nano" in meshes

    def test_thermal_partition_mesh_exists(self):
        """thermal_partition mesh must be declared in assets."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot_motors.xml"))
        root = tree.getroot()
        meshes = [m.get("name") for m in root.iter("mesh")]
        assert "thermal_partition" in meshes

    def test_jetson_geom_in_trunk(self):
        """A geom referencing jetson_orin_nano must exist inside trunk_assembly body."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot_motors.xml"))
        root = tree.getroot()
        for body in root.iter("body"):
            if body.get("name") == "trunk_assembly":
                geom_meshes = [g.get("mesh") for g in body.findall("geom")]
                assert "jetson_orin_nano" in geom_meshes
                break

    def test_no_pi_geom_in_head(self):
        """No geom referencing raspberrypizerow should exist in head_assembly."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot_motors.xml"))
        root = tree.getroot()
        for body in root.iter("body"):
            if body.get("name") == "head_assembly":
                geom_meshes = [g.get("mesh") for g in body.findall("geom")]
                assert "raspberrypizerow" not in geom_meshes
                break

    def test_trunk_mass_updated(self, model):
        """trunk_assembly mass should be approximately 1.18 kg (Jetson + STS3250 servos)."""
        trunk_id = model.body("trunk_assembly").id
        mass = model.body_mass[trunk_id]
        assert 1.05 < mass < 1.30, f"trunk mass out of expected range: {mass}"

    def test_head_mass_updated(self, model):
        """head_assembly mass should be approximately 0.36 kg (STS3250 servos)."""
        head_id = model.body("head_assembly").id
        mass = model.body_mass[head_id]
        assert 0.30 < mass < 0.40, f"head mass out of expected range: {mass}"

    def test_total_mass_in_range(self, model):
        """Total robot mass should be approximately 2.7 kg (STS3250 servos)."""
        total = sum(model.body_mass)
        assert 2.5 < total < 3.0, f"Total mass out of expected range: {total}"

    def test_simulation_does_not_diverge(self):
        """Stepping the simulation 1000 times should not produce NaN or Inf."""
        model = mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, "scene.xml"))
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), "NaN in qpos after 1000 steps"
        assert not np.any(np.isinf(data.qpos)), "Inf in qpos after 1000 steps"


# --- Task 1.4: robot.xml Tests ---


@pytest.mark.phase1
class TestRobotXML:

    def test_xml_parses_without_error(self):
        """robot.xml must compile when loaded via scene_position.xml."""
        model = mujoco.MjModel.from_xml_path(
            os.path.join(ROBOT_DIR, "scene_position.xml")
        )
        assert model is not None

    def test_mass_matches_robot_motors(self):
        """Total mass in robot.xml must match robot_motors.xml."""
        model_motors = mujoco.MjModel.from_xml_path(
            os.path.join(ROBOT_DIR, "scene.xml")
        )
        model_pos = mujoco.MjModel.from_xml_path(
            os.path.join(ROBOT_DIR, "scene_position.xml")
        )
        assert abs(sum(model_motors.body_mass) - sum(model_pos.body_mass)) < 0.001


# --- Task 1.5: URDF Tests ---


@pytest.mark.phase1
class TestRobotURDF:

    def test_urdf_parses(self):
        """URDF must be valid XML."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.urdf"))
        assert tree.getroot().tag == "robot"

    def test_no_raspberrypi_reference(self):
        """No reference to raspberrypizerow.stl in URDF."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.urdf"))
        content = ET.tostring(tree.getroot(), encoding="unicode")
        assert "raspberrypizerow" not in content

    def test_jetson_reference_exists(self):
        """URDF should reference jetson_orin_nano.stl."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.urdf"))
        content = ET.tostring(tree.getroot(), encoding="unicode")
        assert "jetson_orin_nano" in content

    def test_thermal_partition_reference_exists(self):
        """URDF should reference thermal_partition.stl."""
        tree = ET.parse(os.path.join(ROBOT_DIR, "robot.urdf"))
        content = ET.tostring(tree.getroot(), encoding="unicode")
        assert "thermal_partition" in content
