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
        """Actuator parameters, PARSED rather than grepped.

        Task M0 (PLANT-5) changed the effort ceiling and this test pinned the
        old value as a string literal, so it failed the moment the fix landed.
        That is TEST-1 in miniature: a literal check asserts the text, not the
        behaviour, and goes stale silently in the other direction too.

        The BAM-identified stiffness/damping/armature/friction are unchanged --
        they describe the servo's dynamics. The effort LIMIT is different in
        kind: 8.716 was BAM's electrical stall at 12.1 V, 1.78x the 4.903 N.m
        the vendor rates, and the simulator enforcing the electrical figure let
        the policy borrow torque the hardware cannot deliver.
        """
        import ast
        cfg_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "robot_cfg.py"
        )
        tree = ast.parse(open(cfg_path).read())
        consts = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    consts[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass

        content = open(cfg_path).read()
        assert "stiffness=45.53" in content, "kp (stiffness) should be 45.53"
        assert "damping=1.346" in content, "kd (damping) should be 1.346"
        # PLANT-11 2026-08-22: BAM's 0.040 was measured to be 4.74x too high.
        # The bench value supersedes it; see bench_results/GO_NO_GO.md.
        assert "STS3250_ARMATURE = 0.00843" in content, \
            "armature should be the measured 0.00843, not BAM's 0.040"
        assert "armature=STS3250_ARMATURE" in content, \
            "both STS3250 actuator groups should use the measured armature"
        # PLANT-6 2026-08-22: MuJoCo applies one frictionloss both at rest and
        # in motion; Isaac's `friction` is rest-only, so without this the gait
        # ran frictionless.
        assert "dynamic_friction=0.200" in content, \
            "dry friction must act during motion, not only at rest"
        assert "friction=0.200" in content, "friction should be 0.200"

        assert abs(consts["STS3250_EFFORT_LIMIT_NM"] - 4.903) < 1e-6, (
            "effort ceiling must be the 4.903 N.m datasheet stall (50 kg.cm), "
            "not BAM's 8.716 electrical stall — see known_issues.md PLANT-5")
        assert "effort_limit_sim=8.716" not in content, (
            "an actuator group still hard-codes the old electrical stall")

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

    def test_env_cfg_has_imitation_reward(self):
        """Environment must have imitation reward as dominant positive signal."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "ImitationReward" in content, "Missing imitation reward"
        # v3 (BDX-aligned): composite reward with sub-weights baked into the
        # class, registered at RewTerm weight=1.0.
        assert "weight=1.0" in content, "Imitation composite should be weight=1.0"
        # Alive bonus IS present in v2/v3 (+10.0, BDX uses +20.0) — safe with
        # the strong imitation signal preventing the v1-era crouching exploit.
        assert "is_alive" in content, "Alive bonus must be present (BDX-style)"
        assert "weight=10.0" in content, "Alive bonus should be weight=10.0"

    def test_env_cfg_has_core_rewards(self):
        """Environment must include core reward terms."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # Positive rewards (v3: explicit command tracking + yaw tracking;
        # feet_air_time/base_height are handled by the imitation composite)
        assert "track_lin_vel_xy_exp" in content
        assert "track_ang_vel_z_world_exp" in content
        # Penalties
        assert "is_terminated" in content
        assert "flat_orientation_l2" in content
        assert "action_rate_l2" in content
        assert "joint_pos_limits" in content

    def test_env_cfg_has_no_hip_deviation_reward(self):
        """joint_deviation_hips reward term must not be defined (imitation handles hips)."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # The reward term assignment pattern should not exist
        assert "joint_deviation_hips = RewTerm" not in content, \
            "joint_deviation_hips reward should be removed"
        # joint_deviation_head IS intentional — head is 21% of mass, must be stabilized
        assert "joint_deviation_head = RewTerm" in content, \
            "joint_deviation_head reward must be present"

    def test_env_cfg_duck_body_names(self):
        """Environment must reference correct duck body names."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # trunk_assembly: termination contact body in env_cfg
        assert "trunk_assembly" in content
        # Foot bodies moved to imitation_reward.py in v2 (contact matching
        # replaced the feet_air_time term)
        reward_path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "imitation_reward.py"
        )
        with open(reward_path) as f:
            reward_content = f.read()
        assert "foot_assembly" in reward_content
        assert "foot_assembly_2" in reward_content

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
        # v3: ranges clipped to the reference-motion grid hull
        assert "(-0.148, 0.222)" in content, "lin_vel_x should be (-0.148, 0.222)"
        assert "(-0.111, 0.111)" in content, "lin_vel_y should be (-0.111, 0.111)"
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

    def test_env_cfg_tracking_sigma(self):
        """Velocity tracking std must be 0.25 (tuned for duck scale)."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        # v3: explicit track_lin_vel_xy_exp uses the base-class std of 0.5
        # (same value the archived v2 run trained with via inheritance)
        assert '"std": 0.5' in content, "Lin vel tracking std should be 0.5"

    def test_env_cfg_contact_sensor_path(self):
        """Contact sensor must target the articulation root subtree.

        Was Robot/base/* until the PLANT-1 fix merged the massless `base`
        wrapper into `trunk_assembly`, which is now the USD articulation root.
        """
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert (
            "Robot/trunk_assembly/.*" in content
        ), "Contact sensor path must target Robot/trunk_assembly/*"


@pytest.mark.phase2
class TestImitationRewardModule:
    """Verify imitation_reward.py module structure."""

    def test_imitation_reward_exists(self):
        """imitation_reward.py must exist."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "imitation_reward.py"
        )
        assert os.path.exists(path)

    def test_imitation_reward_defines_class(self):
        """Must define ImitationReward class and gait_phase_observation function."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "imitation_reward.py"
        )
        with open(path) as f:
            content = f.read()
        assert "class ImitationReward" in content
        assert "ManagerTermBase" in content
        assert "def gait_phase_observation" in content

    def test_imitation_reward_v3_phase_fix(self):
        """v3 regression guard: the polynomial reference must be evaluated at
        NORMALIZED phase via an integer step counter, never at seconds.

        The polynomials are fit over t in [0, 1] (generator fit_poly.py uses
        np.linspace(0, 1, ...)); evaluating at phase-in-seconds replays only
        54% of the gait cycle (the v2 bug — produced a limping policy).
        """
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "imitation_reward.py"
        )
        with open(path) as f:
            content = f.read()
        # Integer step counter + nb_steps_in_period normalization (upstream
        # Playground convention: t = (i % nb_steps) / nb_steps)
        assert "nb_steps_in_period" in content, \
            "Must read nb_steps_in_period from the gait library"
        assert "_step_idx" in content, \
            "Must use an integer control-step counter for the gait phase"
        # The seconds-based phase accumulator from v2 must be gone
        assert "self._phase.add_(self._dt)" not in content, \
            "v2 seconds-based phase advance must not return (phase bug)"
        # v3 reference clamping to soft joint limits
        assert "soft_joint_pos_limits" in content, \
            "Reference must be clamped to the robot's soft joint limits"
        # v3 zero-command gating (upstream parity)
        assert "cmd_active" in content, \
            "Imitation reward must be gated off for near-zero commands"

    def test_imitation_reward_has_joint_mapping(self):
        """Must define Playground joint order and leg joint names."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "imitation_reward.py"
        )
        with open(path) as f:
            content = f.read()
        assert "PLAYGROUND_JOINT_ORDER" in content
        assert "LEG_JOINT_NAMES" in content
        # Must have all 10 leg joints
        for joint in ["left_hip_yaw", "left_knee", "right_hip_pitch", "right_ankle"]:
            assert joint in content

    def test_imitation_reward_loads_polynomial_data(self):
        """Must load polynomial_coefficients.pkl."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "imitation_reward.py"
        )
        with open(path) as f:
            content = f.read()
        assert "polynomial_coefficients.pkl" in content

    def test_polynomial_data_exists(self):
        """Polynomial coefficient data file must exist."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "data",
            "polynomial_coefficients.pkl",
        )
        assert os.path.exists(path)

    def test_polynomial_data_structure(self):
        """Polynomial data must have 240 entries with correct structure."""
        import pickle
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "data",
            "polynomial_coefficients.pkl",
        )
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert len(data) == 240, f"Expected 240 motions, got {len(data)}"
        entry = data[list(data.keys())[0]]
        assert "coefficients" in entry
        assert "period" in entry
        assert len(entry["coefficients"]) == 40, "Expected 40 dimensions"

    def test_env_cfg_has_phase_observation(self):
        """Environment must add gait_phase observation term."""
        path = os.path.join(
            REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py"
        )
        with open(path) as f:
            content = f.read()
        assert "gait_phase" in content, "Missing gait_phase observation"
        assert "gait_phase_observation" in content


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


@pytest.mark.phase2
class TestV5ContactTrack:
    """Verify the v5 contact-rich track (Task 2.8).

    Source-text assertions in the style of the rest of this file (no Isaac Sim
    required), plus real checks on the generated reference library — that one
    is a data artifact, so it can be validated properly rather than grepped.
    """

    ENV_CFG = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "env_cfg.py")
    INIT = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "__init__.py")
    AGENT_CFG = os.path.join(
        REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "agents", "rsl_rl_ppo_cfg.py"
    )
    DATA_DIR = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "data")

    def test_new_modules_exist(self):
        """The three new runtime modules must be present."""
        for name in ("contact_events.py", "gated_rewards.py", "duck_commands.py",
                     "disturbance_state.py"):
            path = os.path.join(REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", name)
            assert os.path.exists(path), f"{name} not found"

    def test_env_cfg_defines_v5_classes(self):
        content = open(self.ENV_CFG).read()
        for cls in (
            "class DuckContactRewards",
            "class OpenDuckContactEnvCfg",
            "class OpenDuckContactEnvCfg_PLAY",
            "class OpenDuckWrenchEvalEnvCfg",
            "class OpenDuckObstacleEvalEnvCfg",
            "class OpenDuckContactPushEvalEnvCfg",
        ):
            assert cls in content, f"{cls} missing from env_cfg.py"

    def test_v5_tasks_registered(self):
        content = open(self.INIT).read()
        for task in (
            "Isaac-Velocity-Rough-OpenDuck-Contact-v0",
            "Isaac-Velocity-Rough-OpenDuck-Contact-Play-v0",
            "Isaac-Velocity-Rough-OpenDuck-WrenchEval-v0",
            "Isaac-Velocity-Rough-OpenDuck-ObstacleEval-v0",
            "Isaac-Velocity-Rough-OpenDuck-ContactPushEval-v0",
        ):
            assert task in content, f"{task} not registered"

    def test_v5_runner_cfg(self):
        content = open(self.AGENT_CFG).read()
        assert "class OpenDuckContactPPORunnerCfg" in content
        assert 'experiment_name = "open_duck_ppo_v5"' in content

    def test_v5_replaces_contact_death_with_fall_termination(self):
        """The core Task 2.8 change: trunk contact must stop being fatal."""
        content = open(self.ENV_CFG).read()
        assert "self.terminations.base_contact = None" in content
        assert "mdp.bad_orientation" in content
        assert "mdp.root_height_below_minimum" in content

    def test_v5_push_ramp_endpoint_is_researched_value(self):
        """0.7 m/s per axis — see v5_retrain_plan.md section 11.4.

        Guards against a silent revert to Task 2.8's original 1.3 m/s, which
        would demand a 24 cm capture step on a robot whose CoM sits at 17 cm.
        """
        content = open(self.ENV_CFG).read()
        assert "PUSH_START = 0.4" in content
        assert "PUSH_END = 0.7" in content

    def test_v5_wrench_owns_the_composer_alone(self):
        """Two writers to the permanent wrench composer would silently fight."""
        content = open(self.ENV_CFG).read()
        assert "self.events.base_external_force_torque = None" in content

    def test_v4_and_v3_tracks_unchanged(self):
        """v5 must be additive: the v4/v3 recipe keeps contact-death and +/-0.5."""
        content = open(self.ENV_CFG).read()
        assert 'self.terminations.base_contact.params["sensor_cfg"].body_names' in content
        assert "self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)" in content

    def test_reference_library_v2_grid_is_complete(self):
        """The patched library must contain the rows the deployment commands need."""
        import pickle

        path = os.path.join(self.DATA_DIR, "polynomial_coefficients_v2.pkl")
        assert os.path.exists(path), "run scripts/patch_reference_library.py"
        lib = pickle.load(open(path, "rb"))

        keys = set(lib)
        assert "0.0_0.0_0.5" in keys, "turn-in-place at the deployment hull limit missing"
        assert "0.0_0.0_-0.5" in keys
        assert "0.222_0.0_0.0" in keys, "true straight-walk row missing"
        # The rows that did not exist in the frozen library at all.
        assert any(k.split("_")[1] == "0.0" for k in keys), "no vy=0 row"
        assert any(k.split("_")[2] == "0.0" for k in keys), "no wz=0 row"
        assert len(lib) == 388, f"expected 388 cells (390 minus 2 quarantined), got {len(lib)}"

    def test_reference_library_v2_is_numpy_free(self):
        """Must load under Isaac Sim's numpy 1.x, not just the system numpy 2.x."""
        import pickle

        path = os.path.join(self.DATA_DIR, "polynomial_coefficients_v2.pkl")
        lib = pickle.load(open(path, "rb"))
        sample = lib["0.0_0.0_0.5"]["coefficients"]["dim_0"]
        assert all(type(c) is float for c in sample), (
            "coefficients must be built-in floats; numpy scalars re-pickled under "
            "numpy 2.x are unreadable inside Isaac Sim"
        )

    def test_frozen_reference_library_untouched(self):
        """The original library is an immutable dataset (data/README.md)."""
        import hashlib

        path = os.path.join(self.DATA_DIR, "polynomial_coefficients.pkl")
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        assert digest == "7a5515dce7610094bc8da28cfb690a07", (
            "the frozen gait library changed — every journaled gate metric was "
            "measured against it"
        )
