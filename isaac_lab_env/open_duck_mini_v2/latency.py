"""Sensor -> inference latency, the model the simulator has never had.

PLANT-7. There is no latency model of any kind anywhere in this project's
simulation. On the real robot the loop is:

    read 14 servo positions over a shared serial bus
      -> assemble the observation
      -> ONNX inference on the Jetson
      -> write 14 targets back over the same bus

Every stage costs time, the bus is half-duplex and shared, and the control step
is 20 ms. A policy trained with zero latency learns to react on information it
will not have, and the error shows up on hardware as exactly the kind of
high-gain instability that looks like a plant mismatch.

**The value here is PROVISIONAL and deliberately marked so.** Task S.6 measures
the real sensor->inference->actuation loop on the Jetson and replaces it. Until
then this uses a literature-shaped default: 1 control step (20 ms) nominal,
randomised 0-2 steps per environment at reset, i.e. 0-40 ms. That brackets what
a 1 Mbaud Feetech STS chain with 14 servos plus a small MLP typically costs, and
randomising it is what makes the policy robust to the value being wrong -- which
matters more than the nominal, because the nominal is a guess.

Implemented on the OBSERVATION side rather than the action side. Sensor read is
the dominant term on a serial-bus chain, and delaying the observation is exactly
equivalent to delaying the action by the same number of steps for a feedback
policy, while being far simpler to reason about than a second buffer inside the
actuator model.
"""

from __future__ import annotations

import torch

# Imported lazily. This module is a pure buffer over tensors and nothing in it
# needs Isaac at import time, but isaaclab is only importable under the Isaac
# interpreter -- and tests/test_plant_fixes.py must run in the normal suite.
try:  # pragma: no cover - depends on the interpreter
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg
except ModuleNotFoundError:  # pragma: no cover
    Articulation = object

    class SceneEntityCfg:  # minimal stand-in for the default argument
        def __init__(self, name="robot", joint_names=None, **kw):
            self.name = name
            self.joint_names = joint_names
            self.joint_ids = slice(None)

# Provisional until Task S.6 measures the real loop. Units are CONTROL STEPS;
# at decimation=4 and sim.dt=0.005 one step is 20 ms.
DEFAULT_LATENCY_STEPS = (0, 2)

_BUFFERS: dict[str, torch.Tensor] = {}
_DELAYS: dict[str, torch.Tensor] = {}


def _buffer(env, key: str, width: int, depth: int) -> torch.Tensor:
    buf = _BUFFERS.get(key)
    if buf is None or buf.shape[0] != env.num_envs or buf.shape[2] != width:
        buf = torch.zeros(env.num_envs, depth + 1, width, device=env.device)
        _BUFFERS[key] = buf
    return buf


def _delays(env, key: str, lo: int, hi: int) -> torch.Tensor:
    d = _DELAYS.get(key)
    if d is None or d.shape[0] != env.num_envs:
        d = torch.randint(lo, hi + 1, (env.num_envs,), device=env.device)
        _DELAYS[key] = d
    return d


def randomize_latency(env, env_ids, latency_steps: tuple[int, int] = DEFAULT_LATENCY_STEPS):
    """Event term: redraw each resetting env's latency, in control steps.

    Per-env randomisation is the point. The nominal is a guess; a policy that
    only works at one specific latency will not survive the real number.
    """
    lo, hi = latency_steps
    for key, d in _DELAYS.items():
        d[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=env.device)
    for key, buf in _BUFFERS.items():
        buf[env_ids] = 0.0


def _delayed(env, key: str, current: torch.Tensor,
             latency_steps: tuple[int, int]) -> torch.Tensor:
    lo, hi = latency_steps
    buf = _buffer(env, key, current.shape[-1], hi)
    d = _delays(env, key, lo, hi)
    # Shift the ring one step and write the newest sample at index 0.
    buf[:, 1:] = buf[:, :-1].clone()
    buf[:, 0] = current
    idx = d.view(-1, 1, 1).expand(-1, 1, current.shape[-1])
    return buf.gather(1, idx).squeeze(1)


def delayed_joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                          latency_steps: tuple[int, int] = DEFAULT_LATENCY_STEPS):
    """joint_pos_rel as the policy would actually receive it, N steps late."""
    asset: Articulation = env.scene[asset_cfg.name]
    cur = (asset.data.joint_pos[:, asset_cfg.joint_ids]
           - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    return _delayed(env, "joint_pos", cur, latency_steps)


def delayed_joint_vel_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                          latency_steps: tuple[int, int] = DEFAULT_LATENCY_STEPS):
    """joint_vel_rel as the policy would actually receive it, N steps late."""
    asset: Articulation = env.scene[asset_cfg.name]
    cur = (asset.data.joint_vel[:, asset_cfg.joint_ids]
           - asset.data.default_joint_vel[:, asset_cfg.joint_ids])
    return _delayed(env, "joint_vel", cur, latency_steps)
