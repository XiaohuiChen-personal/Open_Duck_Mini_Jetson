"""Tests for Tasks 2.2-2.3 — Isaac Lab environment and robot configuration.

These tests verify the Python module structure and configuration correctness
without requiring Isaac Sim (which is only available on DGX Spark).

Tests that require Isaac Sim runtime are marked with @pytest.mark.requires_isaac_sim
and are skipped by default.
"""

import os
import importlib

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.phase2
class TestRobotCfgModule:
    """Verify robot_cfg.py module structure (Task 2.2)."""

    def test_robot_cfg_importable(self):
        """robot_cfg module must be importable without Isaac Sim."""
        # The module imports isaaclab which requires Isaac Sim runtime,
        # so we verify the file exists and has the right structure.
        cfg_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py"
        )
        assert os.path.exists(cfg_path), "robot_cfg.py not found"

    def test_robot_cfg_defines_articulation(self):
        """robot_cfg.py must define OPEN_DUCK_MINI_V2_CFG."""
        cfg_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py"
        )
        with open(cfg_path) as f:
            content = f.read()
        assert "OPEN_DUCK_MINI_V2_CFG" in content
        assert "ArticulationCfg" in content
        assert "ImplicitActuatorCfg" in content

    def test_robot_cfg_has_correct_actuator_params(self):
        """Actuator parameters must match BAM identification values."""
        cfg_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py"
        )
        with open(cfg_path) as f:
            content = f.read()
        # BAM-identified Feetech STS3250 parameters
        assert "stiffness=45.53" in content, "kp (stiffness) should be 45.53"
        assert "damping=1.346" in content, "kd (damping) should be 1.346"
        assert "armature=0.04" in content, "armature should be 0.04"
        assert "friction=0.2" in content, "friction should be 0.2"
        assert "effort_limit_sim=8.716" in content, "effort_limit_sim should be 8.716"

    def test_robot_cfg_has_all_16_joints(self):
        """Initial state must define positions for all 16 joints."""
        cfg_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py"
        )
        with open(cfg_path) as f:
            content = f.read()
        expected_joints = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle",
            "neck_pitch", "head_pitch", "head_yaw", "head_roll",
            "left_antenna", "right_antenna",
        ]
        for joint in expected_joints:
            assert f'"{joint}"' in content, f"Joint {joint} not found in init_state"

    def test_robot_cfg_has_contact_sensors(self):
        """USD spawn config must enable contact sensors for reward functions."""
        cfg_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py"
        )
        with open(cfg_path) as f:
            content = f.read()
        assert "activate_contact_sensors=True" in content


@pytest.mark.phase2
class TestEnvCfgModule:
    """Verify env_cfg.py module structure (Task 2.3)."""

    def test_env_cfg_exists(self):
        """env_cfg.py must exist."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        assert os.path.exists(path)

    def test_env_cfg_defines_required_classes(self):
        """env_cfg.py must define DuckRewards, OpenDuckRoughEnvCfg, and PLAY variant."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "class DuckRewards" in content
        assert "class OpenDuckRoughEnvCfg" in content
        assert "class OpenDuckRoughEnvCfg_PLAY" in content

    def test_env_cfg_inherits_from_base(self):
        """Environment must inherit from LocomotionVelocityRoughEnvCfg."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "LocomotionVelocityRoughEnvCfg" in content

    def test_env_cfg_has_biped_rewards(self):
        """Environment must include biped-specific reward terms."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # Key biped rewards
        assert "feet_air_time_positive_biped" in content
        assert "track_lin_vel_xy_yaw_frame_exp" in content
        assert "track_ang_vel_z_world_exp" in content
        assert "feet_slide" in content
        assert "joint_deviation_l1" in content
        assert "joint_pos_limits" in content
        assert "is_terminated" in content

    def test_env_cfg_duck_body_names(self):
        """Environment must reference correct duck body names."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "left_foot" in content
        assert "right_foot" in content
        assert "trunk_assembly" in content

    def test_env_cfg_simulation_timing(self):
        """Simulation must use 200 Hz physics / 50 Hz policy."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "self.sim.dt = 0.005" in content  # 200 Hz
        assert "self.decimation = 4" in content  # 200/4 = 50 Hz


@pytest.mark.phase2
class TestTrainingCfgModule:
    """Verify PPO training configuration (Task 2.3)."""

    def test_ppo_cfg_exists(self):
        """PPO runner config must exist."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents",
            "rsl_rl_ppo_cfg.py",
        )
        assert os.path.exists(path)

    def test_ppo_cfg_defines_runner(self):
        """PPO config must define OpenDuckPPORunnerCfg."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents",
            "rsl_rl_ppo_cfg.py",
        )
        with open(path) as f:
            content = f.read()
        assert "class OpenDuckPPORunnerCfg" in content
        assert "RslRlOnPolicyRunnerCfg" in content

    def test_ppo_cfg_network_architecture(self):
        """Policy network must use [512, 256, 128] architecture."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents",
            "rsl_rl_ppo_cfg.py",
        )
        with open(path) as f:
            content = f.read()
        assert "[512, 256, 128]" in content


@pytest.mark.phase2
class TestGymRegistration:
    """Verify Gymnasium environment registration (Task 2.3)."""

    def test_registration_file_exists(self):
        """__init__.py with gym.register calls must exist."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "__init__.py"
        )
        assert os.path.exists(path)

    def test_registration_has_correct_ids(self):
        """Registration must use Isaac Lab naming convention."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "__init__.py"
        )
        with open(path) as f:
            content = f.read()
        assert "Isaac-Velocity-Rough-OpenDuck-v0" in content
        assert "Isaac-Velocity-Rough-OpenDuck-Play-v0" in content
        assert "disable_env_checker=True" in content

    def test_registration_has_entry_points(self):
        """Registration must include env and rsl_rl config entry points."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "__init__.py"
        )
        with open(path) as f:
            content = f.read()
        assert "env_cfg_entry_point" in content
        assert "rsl_rl_cfg_entry_point" in content
        assert "OpenDuckRoughEnvCfg" in content
        assert "OpenDuckPPORunnerCfg" in content


@pytest.mark.phase2
class TestDirectoryStructure:
    """Verify the complete isaac_lab_env package structure."""

    def test_package_init(self):
        """Top-level __init__.py must exist."""
        path = os.path.join(REPO_ROOT, "isaac_lab_env", "__init__.py")
        assert os.path.exists(path)

    def test_agents_init(self):
        """Agents __init__.py must exist."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents", "__init__.py"
        )
        assert os.path.exists(path)

    def test_usd_directory_exists(self):
        """USD output directory must exist."""
        path = os.path.join(
            REPO_ROOT, "mini_bdx", "robots", "open_duck_mini_v2", "usd"
        )
        assert os.path.isdir(path)
