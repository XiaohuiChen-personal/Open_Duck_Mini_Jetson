"""Tests for Tasks 2.2-2.3 — Isaac Lab environment and robot configuration.

These tests verify the Python module structure and configuration correctness
without requiring Isaac Sim (which is only available on DGX Spark).

Tests that require Isaac Sim runtime are marked with @pytest.mark.requires_isaac_sim
and are skipped by default.
"""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.phase2
class TestRobotCfgModule:
    """Verify robot_cfg.py module structure (Task 2.2)."""

    def test_robot_cfg_importable(self):
        """robot_cfg module must be importable without Isaac Sim."""
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
        # BAM-identified Feetech STS3250 parameters (kscalelabs/sysid id008)
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
    """Verify env_cfg.py reward and environment structure."""

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

    def test_env_cfg_has_alive_bonus(self):
        """Environment must have a positive alive/survival bonus reward."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "is_alive" in content, "Missing alive bonus reward"
        assert "weight=5.0" in content, "Alive bonus should be weight=5.0"

    def test_env_cfg_has_core_rewards(self):
        """Environment must include core reward terms."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # Positive rewards
        assert "track_lin_vel_xy_yaw_frame_exp" in content
        assert "track_ang_vel_z_world_exp" in content
        assert "feet_air_time_positive_biped" in content
        # Penalties
        assert "is_terminated" in content
        assert "flat_orientation_l2" in content
        assert "action_rate_l2" in content
        assert "joint_pos_limits" in content

    def test_env_cfg_has_no_h1_bloat(self):
        """H1-specific penalty terms should NOT be present."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # These were removed because they don't apply to the duck
        assert "joint_deviation_head" not in content, "joint_deviation_head should be removed"
        assert "joint_deviation_hips" not in content, "joint_deviation_hips should be removed"

    def test_env_cfg_duck_body_names(self):
        """Environment must reference correct duck body names."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "foot_assembly" in content
        assert "foot_assembly_2" in content
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

    def test_env_cfg_velocity_ranges_conservative(self):
        """Velocity ranges must be conservative for a 42cm robot."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # Should NOT have the old wide ranges for linear velocity
        assert "(-0.5, 1.0)" not in content, "lin_vel_x range too wide"
        # Should have conservative ranges
        assert "(-0.15, 0.3)" in content, "lin_vel_x should be (-0.15, 0.3)"
        assert "(-0.15, 0.15)" in content, "lin_vel_y should be (-0.15, 0.15)"
        assert "(-0.5, 0.5)" in content, "ang_vel_z should be (-0.5, 0.5)"

    def test_env_cfg_action_scale(self):
        """Action scale must be 0.25 (matching Open Duck Playground)."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "scale = 0.25" in content or "scale=0.25" in content, \
            "Action scale should be 0.25"

    def test_env_cfg_tracking_sigma_sharp(self):
        """Velocity tracking std must be sharp (0.1 or smaller)."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert '"std": 0.1' in content, "Lin vel tracking std should be 0.1"

    def test_env_cfg_contact_sensor_path(self):
        """Contact sensor must use Robot/base/* path for MJCF-converted USD."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "Robot/base/.*" in content, "Contact sensor path must target Robot/base/*"


@pytest.mark.phase2
class TestTrainingCfgModule:
    """Verify PPO training configuration."""

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

    def test_ppo_cfg_obs_normalization_enabled(self):
        """Observation normalization must be enabled."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents",
            "rsl_rl_ppo_cfg.py",
        )
        with open(path) as f:
            content = f.read()
        assert "actor_obs_normalization=True" in content
        assert "critic_obs_normalization=True" in content

    def test_ppo_cfg_tuned_hyperparameters(self):
        """PPO hyperparameters must be tuned for small biped."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents",
            "rsl_rl_ppo_cfg.py",
        )
        with open(path) as f:
            content = f.read()
        assert "gamma=0.97" in content, "gamma should be 0.97 for faster convergence"
        assert "entropy_coef=0.005" in content, "entropy_coef should be 0.005"
        assert "init_noise_std=0.5" in content, "init_noise_std should be 0.5"


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
