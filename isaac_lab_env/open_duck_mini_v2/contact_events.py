# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-rich disturbance events for the v5 track (Task 2.8).

v4_robust trains in a flat, empty world whose only disturbance is an
instantaneous velocity kick. The Duck Embody benchmark then measured 10 falls
in 12 apartment trials, dominated by two regimes that training never contained:
**rotation while loaded against an obstacle** (7 of 10) and **sustained
pressing** (2 of 10). This module adds both.

Three mechanisms, one class-based term
--------------------------------------
``ContactRegimeEvent`` is a single interval term configured to tick every
policy step (``interval_range_s=(step_dt, step_dt)``, ``is_global_time=True``
-> the event manager calls it once per step with ``env_ids=None``; verified at
isaaclab/managers/event_manager.py:216-223). It owns everything that needs
per-environment timers:

1. **Sustained wrench.** A constant horizontal force held 2-6 s, magnitude a
   fraction of the robot's *measured* weight, applied at a randomized point on
   the trunk with a small yaw torque. Isaac Lab has no duration-limited force
   event (verified: no `duration` anywhere in isaaclab/envs/mdp/events.py), but
   it does not need one — the permanent wrench composer re-applies a stored
   wrench at every physics substep until it is overwritten or the env resets
   (assets/rigid_object/rigid_object.py:113-132), so a term that only manages
   timers is sufficient.
2. **Obstacle placement.** On the first step of an episode the env's kinematic
   box is either placed just ahead of the robot, tangent to where it is about
   to walk, or parked below the ground plane. Doing this in the per-step tick
   rather than a reset event is deliberate: it reads the robot pose *after*
   every reset event has run, so it cannot depend on event declaration order.
3. **The disturbance gate.** Raised for wrench, push and measured trunk contact
   (see ``disturbance_state`` for why it must never be raised by anything the
   policy can self-induce), held 1 s past the end of the disturbance.

It also writes the diagnostics that make the training regime auditable — gate
duty, wrench duty, obstacle contact rate, enforced co-occurrence fraction —
because "the obstacle exists" is not evidence that the robot ever touches it.

Sole ownership of the permanent wrench composer
-----------------------------------------------
``set_forces_and_torques`` REPLACES the composer slice for the bodies and envs
it names, so exactly one term may own it. The v5 env cfg therefore sets
``events.base_external_force_torque = None`` (the stock term writes the same
composer, isaaclab/envs/mdp/events.py:1038-1043).
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaac_lab_env.open_duck_mini_v2 import disturbance_state as ds

# Seconds the gate stays raised after a disturbance ends, so the policy is
# scored on resuming the gait rather than on tracking mid-recovery.
GATE_HOLD_S = 1.0

# Trunk contact force treated as "something is pushing on me" (matches the
# contact thresholds already used for feet and terminations, 1 N).
CONTACT_FORCE_THRESHOLD_N = 1.0


class ContactRegimeEvent(ManagerTermBase):
    """Per-step owner of the sustained wrench, the obstacles and the gate.

    Args (via ``EventTermCfg.params``):
        force_frac_range: wrench magnitude as a fraction of measured body
            weight. Hartmann et al. use 10 N and 20 N on a 117.7 N Go1, i.e.
            8.5% and 17% of body weight; (0.05, 0.30) brackets that.
        duration_range_s: how long each wrench is held.
        torque_z_range: yaw torque applied with the force, in N*m.
        lateral_bias: 0 -> uniform direction, 1 -> purely sideways. Pressing
            happens against walls the robot walks beside, so the prior is
            lateral.
        active_frac: target fraction of environments carrying a wrench.
        cooccurrence_frac: minimum fraction of each activation batch drawn from
            environments that are ALSO commanded to rotate. The benchmark's
            modal fall is rotation-under-load, and no published curriculum
            produces that co-occurrence by chance — it has to be forced.
        rotating_cmd_wz: |wz| above which a command counts as "rotating".
        obstacle_frac: fraction of episodes that spawn their box in play.
        obstacle_ahead_range / obstacle_lateral_range: where the box goes,
            in metres, relative to the robot and its heading. The lateral band
            straddles the measured swept gait half-width of 0.11-0.15 m
            (duck-embody configs/benchmark.yaml:131-139), so the robot grazes
            rather than either missing entirely or walking into a wall.
        obstacle_parked_z: where unused boxes wait, below the ground plane.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene["robot"]
        self._state = ds.get_state(env)

        # The USD root body "base" is massless — every body-targeted operation
        # must name trunk_assembly (same constraint the v4 DR events carry,
        # env_cfg.py:307-308).
        self._trunk_ids, _ = self._robot.find_bodies("trunk_assembly")

        # Measured, not assumed: the plan's force range is a fraction of body
        # weight, and this robot is ~2.66 kg after the Jetson mod — an earlier
        # draft inherited a 1.6 kg assumption from the task plan.
        total_mass = float(self._robot.data.default_mass[0].sum())
        self._weight_n = total_mass * 9.81

        n, dev = env.num_envs, env.device
        self._wrench_time_left = torch.zeros(n, device=dev)
        self._force_b = torch.zeros(n, 3, device=dev)
        self._torque_b = torch.zeros(n, 3, device=dev)
        self._offset_b = torch.zeros(n, 3, device=dev)

        # Trunk half-extents used as the wrench application-point envelope.
        self._trunk_half_extent = torch.tensor([0.04, 0.04, 0.05], device=dev)

        # Diagnostics accumulated over an episode, reported through extras.
        self._gate_steps = torch.zeros(n, device=dev)
        self._wrench_steps = torch.zeros(n, device=dev)
        self._obstacle_contact_steps = torch.zeros(n, device=dev)
        self._foot_contact_steps = torch.zeros(n, 2, device=dev)
        self._episode_steps = torch.zeros(n, device=dev)
        self._activations = 0
        self._activations_while_rotating = 0

        # NOTE: use the typed sub-dicts, NOT `env.scene.get(...)` — InteractiveScene
        # exposes only __getitem__/keys(), so a `.get()` call raises AttributeError.
        # That matters more than it looks: this __init__ runs inside a timeline
        # PLAY callback (the event manager is built before sim.reset(), so class
        # terms are instantiated later, manager_base.py:161-171), and exceptions
        # raised in that callback are swallowed — the term silently stays a class
        # and only surfaces later as "reset() missing 1 required positional
        # argument: 'self'". Keep this constructor defensive.
        self._obstacle: RigidObject | None = env.scene.rigid_objects.get("obstacle")
        self._contact_sensor: ContactSensor | None = env.scene.sensors.get("contact_forces")
        self._trunk_sensor_ids = None
        self._foot_sensor_ids = None
        if self._contact_sensor is not None:
            ids, _ = self._contact_sensor.find_bodies("trunk_assembly")
            self._trunk_sensor_ids = ids
            foot_ids, _ = self._contact_sensor.find_bodies(["foot_assembly", "foot_assembly_2"])
            self._foot_sensor_ids = foot_ids

        # Read every param defensively. `params` carries only what a given cfg
        # chose to override — the minimal arm passes no `force_frac_range` at
        # all — and a KeyError here is not a normal crash: this constructor runs
        # inside a timeline PLAY callback whose exceptions are SWALLOWED, so the
        # term silently stays a class and training runs on without it. Nothing
        # in __init__ may assume an optional param is present.
        frac = cfg.params.get("force_frac_range", (0.05, 0.30))
        print(
            f"[v5] ContactRegimeEvent: measured robot weight {self._weight_n:.2f} N "
            f"({total_mass:.3f} kg) -> wrench "
            f"{self._weight_n * frac[0]:.2f}-{self._weight_n * frac[1]:.2f} N "
            f"(active_frac={cfg.params.get('active_frac', 0.5)}); "
            f"obstacle asset {'present' if self._obstacle is not None else 'ABSENT'}, "
            f"obstacle_frac={cfg.params.get('obstacle_frac', 0.25)}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_ids=None) -> None:
        """Clear per-episode state.

        The wrench itself needs no clearing — ``scene.reset`` zeroes the
        permanent composer for the resetting envs
        (scene/interactive_scene.py:443-455).
        """
        if env_ids is None:
            env_ids = slice(None)
        self._wrench_time_left[env_ids] = 0.0
        self._force_b[env_ids] = 0.0
        self._torque_b[env_ids] = 0.0
        self._state.reset(env_ids)
        self._gate_steps[env_ids] = 0.0
        self._wrench_steps[env_ids] = 0.0
        self._obstacle_contact_steps[env_ids] = 0.0
        self._foot_contact_steps[env_ids] = 0.0
        self._episode_steps[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        force_frac_range: tuple[float, float] = (0.05, 0.30),
        duration_range_s: tuple[float, float] = (2.0, 6.0),
        torque_z_range: tuple[float, float] = (0.05, 0.15),
        lateral_bias: float = 0.7,
        active_frac: float = 0.5,
        cooccurrence_frac: float = 0.3,
        rotating_cmd_wz: float = 0.25,
        obstacle_frac: float = 0.25,
        obstacle_ahead_range: tuple[float, float] = (0.3, 0.9),
        obstacle_lateral_range: tuple[float, float] = (0.10, 0.35),
        obstacle_parked_z: float = -2.0,
        command_name: str = "base_velocity",
        contact_raises_gate: bool = False,
    ) -> None:
        dt = env.step_dt

        self._place_obstacles_for_fresh_episodes(
            env, obstacle_frac, obstacle_ahead_range, obstacle_lateral_range, obstacle_parked_z
        )
        self._expire_wrenches(dt)
        self._activate_wrenches(
            env, force_frac_range, duration_range_s, torque_z_range, lateral_bias,
            active_frac, cooccurrence_frac, rotating_cmd_wz, command_name,
        )
        # Default OFF. Measured in run v5a_gated_ft: with obstacles present the
        # policy can CAUSE trunk contact, so this trigger is self-inducible after
        # all — contact duty reached 50%, gate duty 83%, imitation collapsed to
        # +0.008, and the eval scored 0/6 on the gait gate (99.7% stance duty =
        # standing). Only env-scheduled wrenches and pushes may raise the gate.
        if contact_raises_gate:
            self._gate_on_contact()
        else:
            self._count_trunk_contact()
        self._state.tick(dt)
        self._accumulate_diagnostics(env, dt)

    # ------------------------------------------------------------------
    # Mechanisms
    # ------------------------------------------------------------------

    def _place_obstacles_for_fresh_episodes(
        self, env, obstacle_frac, ahead_range, lateral_range, parked_z
    ) -> None:
        """Position each env's box on the first step of its episode."""
        if self._obstacle is None or obstacle_frac <= 0.0:
            return
        fresh = (env.episode_length_buf <= 1).nonzero().flatten()
        if len(fresh) == 0:
            return

        n = len(fresh)
        dev = env.device
        active = torch.rand(n, device=dev) < obstacle_frac
        self._state.obstacle_active[fresh] = active

        root = self._robot.data.root_state_w[fresh]
        pos, quat = root[:, :3], root[:, 3:7]
        # Robot heading from its yaw; the box is offset along that heading and
        # to one side, so the commanded walk grazes it rather than missing.
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        ahead = _uniform(n, ahead_range, dev)
        lateral = _uniform(n, lateral_range, dev) * _rand_sign(n, dev)
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        left = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)

        new_pos = pos.clone()
        new_pos[:, :2] = pos[:, :2] + fwd * ahead.unsqueeze(-1) + left * lateral.unsqueeze(-1)
        # Box origin sits at half its height so it rests on the plane.
        new_pos[:, 2] = torch.where(
            active, torch.full_like(new_pos[:, 2], 0.35), torch.full_like(new_pos[:, 2], parked_z)
        )

        box_yaw = yaw + _uniform(n, (-0.5, 0.5), dev)
        new_quat = torch.zeros(n, 4, device=dev)
        new_quat[:, 0] = torch.cos(box_yaw * 0.5)
        new_quat[:, 3] = torch.sin(box_yaw * 0.5)

        self._obstacle.write_root_pose_to_sim(
            torch.cat([new_pos, new_quat], dim=-1), env_ids=fresh
        )

    def _expire_wrenches(self, dt: float) -> None:
        """Drop wrenches whose duration has run out."""
        active = self._state.wrench_active
        if not bool(active.any()):
            return
        self._wrench_time_left[active] -= dt
        done = active & (self._wrench_time_left <= 0.0)
        if bool(done.any()):
            ids = done.nonzero().flatten()
            zeros = torch.zeros(len(ids), len(self._trunk_ids), 3, device=self._force_b.device)
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                forces=zeros, torques=zeros, body_ids=self._trunk_ids, env_ids=ids
            )
            self._state.wrench_active[ids] = False
            self._force_b[ids] = 0.0
            self._torque_b[ids] = 0.0

    def _activate_wrenches(
        self, env, force_frac_range, duration_range_s, torque_z_range, lateral_bias,
        active_frac, cooccurrence_frac, rotating_cmd_wz, command_name,
    ) -> None:
        """Start new wrenches, forcing rotation co-occurrence on a share of them."""
        idle = (~self._state.wrench_active).nonzero().flatten()
        if len(idle) == 0:
            return

        # Expected duty of `active_frac` with mean duration D and per-step
        # hazard p over the idle pool: p = active_frac / ((1 - active_frac) * D / dt).
        mean_duration = 0.5 * (duration_range_s[0] + duration_range_s[1])
        hazard = env.step_dt * active_frac / max(1e-6, (1.0 - active_frac) * mean_duration)

        cmd = env.command_manager.get_command(command_name)
        rotating = cmd[:, 2].abs() >= rotating_cmd_wz

        draw = torch.rand(len(idle), device=env.device) < hazard
        chosen = idle[draw]

        # Enforce the co-occurrence floor: if too few of the drawn envs are
        # rotating, top up from idle envs that are. Hoping for the overlap is
        # not enough — this is the regime 7 of 10 falls came from.
        if cooccurrence_frac > 0.0 and len(chosen) > 0:
            rot_chosen = int(rotating[chosen].sum())
            need = int(cooccurrence_frac * len(chosen)) - rot_chosen
            if need > 0:
                pool = idle[rotating[idle]]
                pool = pool[~torch.isin(pool, chosen)]
                if len(pool) > 0:
                    chosen = torch.cat([chosen, pool[torch.randperm(len(pool), device=env.device)[:need]]])

        if len(chosen) == 0:
            return

        n, dev = len(chosen), env.device
        frac = _uniform(n, force_frac_range, dev)
        magnitude = frac * self._weight_n
        # Direction: interpolate between uniform and purely lateral.
        angle = _uniform(n, (-torch.pi, torch.pi), dev)
        lateral_angle = (torch.pi / 2.0) * _rand_sign(n, dev)
        angle = (1.0 - lateral_bias) * angle + lateral_bias * lateral_angle

        force = torch.zeros(n, 3, device=dev)
        force[:, 0] = magnitude * torch.cos(angle)
        force[:, 1] = magnitude * torch.sin(angle)
        torque = torch.zeros(n, 3, device=dev)
        torque[:, 2] = _uniform(n, torque_z_range, dev) * _rand_sign(n, dev)
        offset = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * self._trunk_half_extent

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=force.unsqueeze(1),
            torques=torque.unsqueeze(1),
            positions=offset.unsqueeze(1),
            body_ids=self._trunk_ids,
            env_ids=chosen,
            is_global=False,
        )
        self._wrench_time_left[chosen] = _uniform(n, duration_range_s, dev)
        self._state.wrench_active[chosen] = True
        self._force_b[chosen] = force
        self._torque_b[chosen] = torque
        self._offset_b[chosen] = offset
        self._state.mark(chosen, GATE_HOLD_S)

        self._activations += n
        self._activations_while_rotating += int(rotating[chosen].sum())

    def _accumulate_gait_canary(self, env) -> None:
        """Log per-foot stance duty DURING training — the standing early-warning.

        Run v5a_gated_ft spent 2.22 h converging to a policy whose every
        training signal looked healthy (episode length 977/1000, falls near
        zero, reward climbing) and which the eval then scored 0/6 on the gait
        gate at 99.7% stance duty: it had learned to stand. Nothing in the
        training logs would have revealed that, because stance duty was only
        ever computed by ``scripts/evaluate_policies.py`` AFTER the run.

        The same [40, 90]% band the gate uses (GAIT_DUTY_BAND_PCT) is therefore
        computed online here. It costs one contact-sensor read per step and
        turns a 2.2 h post-mortem into an abort at a few hundred iterations.
        This is a diagnostic only — it feeds no reward and cannot change the
        policy.
        """
        if self._contact_sensor is None or self._foot_sensor_ids is None:
            return
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._foot_sensor_ids, :]
        in_contact = (forces.norm(dim=-1).amax(dim=1) > CONTACT_FORCE_THRESHOLD_N).float()
        self._foot_contact_steps += in_contact

        steps = self._episode_steps.clamp(min=1.0).unsqueeze(-1)
        duty = (self._foot_contact_steps / steps) * 100.0
        log = self._env.extras.setdefault("log", {})
        log["Gait/stance_duty_left_pct"] = duty[:, 0].mean()
        log["Gait/stance_duty_right_pct"] = duty[:, 1].mean()
        # Fraction of envs a run would FAIL the gait gate on right now. Rising
        # toward 1.0 means the run is converging to standing (duty -> 100%) or
        # crawling (duty -> 0%); either way it is not worth its remaining hours.
        in_band = ((duty >= 40.0) & (duty <= 90.0)).all(dim=-1).float()
        log["Gait/duty_in_band_frac"] = in_band.mean()

    def _count_trunk_contact(self) -> None:
        """Record trunk-contact duty without touching the gate."""
        if self._contact_sensor is None or self._trunk_sensor_ids is None:
            return
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._trunk_sensor_ids, :]
        loaded = forces.norm(dim=-1).amax(dim=(1, 2)) > CONTACT_FORCE_THRESHOLD_N
        if bool(loaded.any()):
            self._obstacle_contact_steps[loaded.nonzero().flatten()] += 1.0

    def _gate_on_contact(self) -> None:
        """Raise the gate while the trunk is measurably loaded.

        This is the obstacle-contact analogue of the wrench flag: an external,
        sensor-measured force, so it cannot be manufactured by the policy on an
        empty plane.
        """
        if self._contact_sensor is None or self._trunk_sensor_ids is None:
            return
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._trunk_sensor_ids, :]
        loaded = forces.norm(dim=-1).amax(dim=(1, 2)) > CONTACT_FORCE_THRESHOLD_N
        if bool(loaded.any()):
            ids = loaded.nonzero().flatten()
            self._state.mark(ids, GATE_HOLD_S)
            self._obstacle_contact_steps[ids] += 1.0

    def _accumulate_diagnostics(self, env, dt: float) -> None:
        """Report the training regime, so 'it was configured' != 'it happened'."""
        self._gate_steps += self._state.gate
        self._wrench_steps += self._state.wrench_active.float()
        self._episode_steps += 1.0
        self._accumulate_gait_canary(env)

        steps = self._episode_steps.clamp(min=1.0)
        log = env.extras.setdefault("log", {})
        log["Regime/gate_duty"] = (self._gate_steps / steps).mean()
        log["Regime/wrench_duty"] = (self._wrench_steps / steps).mean()
        log["Regime/trunk_contact_duty"] = (self._obstacle_contact_steps / steps).mean()
        log["Regime/obstacle_envs_frac"] = self._state.obstacle_active.float().mean()
        log["Regime/wrench_force_n"] = self._force_b.norm(dim=-1)[self._state.wrench_active].mean() \
            if bool(self._state.wrench_active.any()) else torch.zeros((), device=env.device)
        if self._activations > 0:
            log["Regime/wrench_rotating_cooccurrence"] = torch.tensor(
                self._activations_while_rotating / self._activations, device=env.device
            )


def push_and_mark(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """``mdp.push_by_setting_velocity`` that also raises the disturbance gate.

    The stock term is used unchanged for the physics (it adds a uniform sample
    per axis to the root velocity and accepts x/y/z/roll/pitch/yaw keys —
    isaaclab/envs/mdp/events.py:1046-1071, which is what makes the v5
    rotational pushes free); this wrapper only records that a disturbance
    happened so the gated rewards can react to it.
    """
    mdp.push_by_setting_velocity(env, env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg)
    state = ds.peek_state(env)
    if state is not None:
        state.mark(env_ids, GATE_HOLD_S)


def _uniform(n: int, rng: tuple[float, float], device) -> torch.Tensor:
    return torch.rand(n, device=device) * (rng[1] - rng[0]) + rng[0]


def _rand_sign(n: int, device) -> torch.Tensor:
    return torch.where(
        torch.rand(n, device=device) < 0.5,
        torch.full((n,), -1.0, device=device),
        torch.full((n,), 1.0, device=device),
    )
