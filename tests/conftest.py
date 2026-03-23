import pytest
import json
import mujoco
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_DIR = os.path.join(REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2")


@pytest.fixture
def model():
    """Load the torque-controlled MuJoCo model."""
    return mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, "scene.xml"))


@pytest.fixture
def model_position():
    """Load the position-controlled MuJoCo model."""
    return mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, "scene_position.xml"))


@pytest.fixture
def updated_model(model):
    """Alias for clarity in test names."""
    return model


@pytest.fixture
def expected_values():
    """Load expected mass/inertia values."""
    with open(os.path.join(REPO_ROOT, "tests", "fixtures", "expected_values.json")) as f:
        return json.load(f)
