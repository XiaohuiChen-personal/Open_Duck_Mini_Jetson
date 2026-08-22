"""Build the 53-dim observation the policy expects — Task S.4.

This is the layer between the sensors and the ONNX. It is written to be provable
against the S.2 reference traces **without a robot**, because otherwise its first
test is the robot falling over.

LAYOUT (from `policy_contract.json`, never hard-coded here)

    [ 0: 3)  base_ang_vel        body-frame angular velocity of `trunk_assembly`
    [ 3: 6)  projected_gravity   unit vector, body frame, ~(0,0,-1) upright
    [ 6: 9)  velocity_commands   vx, vy, wz — already clamped by the caller
    [ 9:23)  joint_pos           q_meas - q_default
    [23:37)  joint_vel           qd_meas  (see note)
    [37:51)  actions             RAW previous ONNX output, unscaled
    [51:53)  gait_phase          cos, sin

**53, not 59.** Task M0b removed the two antenna joints from the action and
observation spaces (59/16 -> 53/14) because they are open-loop SG90s with no
position feedback, so four dims could never be measured on hardware. The S.4
task text in `task_plan_v2.md` predates that change and still says 59/16 with
gait phase at `obs[57:59]`; **the contract is authoritative and this module
follows it.** There are no antenna dims to fill, so DEPLOY-3 needs no option
here at all.

TRAPS ENCODED HERE

- `joint_vel_rel` is numerically identical to absolute joint velocity, because
  the default joint velocity is zero. It is still named `_rel` upstream. Do not
  "fix" this by subtracting something.
- `base_ang_vel` is `root_ang_vel_b`, the BODY-frame angular velocity of the
  root body, which since the PLANT-1 fix is `trunk_assembly` (`robot_cfg.py:79`).
- `projected_gravity` is `R_world->body @ (0,0,-1)`, a UNIT vector — not an
  accelerometer reading and not scaled by g.
- `last_action` is the RAW, unscaled previous ONNX output — not the joint target.
  On the very first step it is zeros.
"""

from __future__ import annotations

import numpy as np


class ObsBuilder:
    """Assemble the policy observation from measured quantities."""

    def __init__(self, contract: dict) -> None:
        self.obs_dim = int(contract["obs_dim"])
        self.action_dim = int(contract["action_dim"])
        self.slices = {k: (int(a), int(b))
                       for k, (a, b) in contract["obs_slices"].items()}
        self.joint_order = list(contract["joint_order"])
        self.q_default = np.asarray(contract["q_default_rad"], dtype=np.float32)
        if self.q_default.shape != (self.action_dim,):
            raise ValueError(
                f"q_default has {self.q_default.shape}, expected ({self.action_dim},)")

    # -- helpers ---------------------------------------------------------

    def _put(self, obs: np.ndarray, name: str, value: np.ndarray) -> None:
        lo, hi = self.slices[name]
        want = hi - lo
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != want:
            raise ValueError(f"{name}: got {value.shape[0]} values, slice wants {want}")
        obs[lo:hi] = value

    # -- the one public method -------------------------------------------

    def build(self, gyro_b, gravity_b, q_meas, qd_meas, command,
              last_action, gait_cos_sin) -> np.ndarray:
        """Return the float32 observation vector.

        Every argument is in CONTRACT JOINT ORDER where applicable, and the
        command must already be clamped by the caller (`contract.clamp_command`).
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        self._put(obs, "base_ang_vel", gyro_b)
        self._put(obs, "projected_gravity", gravity_b)
        self._put(obs, "velocity_commands", command)

        q_meas = np.asarray(q_meas, dtype=np.float32).reshape(-1)
        if q_meas.shape[0] != self.action_dim:
            raise ValueError(f"q_meas has {q_meas.shape[0]}, expected {self.action_dim}")
        self._put(obs, "joint_pos", q_meas - self.q_default)

        # NOT a bug: the default joint velocity is zero, so `_rel` is absolute.
        self._put(obs, "joint_vel", qd_meas)

        self._put(obs, "actions", last_action)
        self._put(obs, "gait_phase", gait_cos_sin)

        # Cheap, and it is the failure mode that shows up as a convulsing robot.
        if obs.dtype != np.float32:
            raise TypeError(f"observation dtype is {obs.dtype}, must be float32")
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"observation shape is {obs.shape}")
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))[0].tolist()
            raise ValueError(f"observation has non-finite values at dims {bad}")
        return obs
