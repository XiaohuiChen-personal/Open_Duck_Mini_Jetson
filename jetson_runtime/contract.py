"""Typed access to `policy_contract.json` — Task S.1.

**There is not a single policy number literal in this file.** Everything is read
from the generated JSON, which is itself re-derived from primary sources by
`scripts/generate_policy_contract.py`. That is the whole point: `known_issues.md`
DEPLOY-1 exists because the deployment constants lived in prose, and prose drifts.

    from jetson_runtime import contract as C
    q_target = C.Q_DEFAULT + C.ACTION_SCALE * action     # the formula that matters
    q_target = np.clip(q_target, C.HARD_LIMITS_LOW, C.HARD_LIMITS_HIGH)

`setup.cfg` packages only `mini_bdx`, so this package imports only with the repo
root as cwd or on PYTHONPATH.
"""

from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_PATH = os.path.join(_HERE, "policy_contract.json")

if not os.path.isfile(CONTRACT_PATH):
    raise FileNotFoundError(
        f"{CONTRACT_PATH} missing — run `python3 scripts/generate_policy_contract.py`")

_C = json.load(open(CONTRACT_PATH))

# ---- interface ---------------------------------------------------------
OBS_DIM: int = _C["obs_dim"]
ACTION_DIM: int = _C["action_dim"]
JOINT_ORDER: list[str] = _C["joint_order"]
OBS_SLICES: dict[str, slice] = {k: slice(a, b) for k, (a, b) in _C["obs_slices"].items()}
OBS_TERMS: list[dict] = _C["obs_terms"]

# ---- action decoding ---------------------------------------------------
ACTION_SCALE: float = _C["action_scale"]
Q_DEFAULT: np.ndarray = np.array(_C["q_default_rad"], dtype=np.float64)
ACTION_FORMULA: str = _C["action_formula"]

# ---- timing ------------------------------------------------------------
CONTROL_DT: float = _C["control_dt_s"]
GAIT_NB_STEPS: int = _C["gait_nb_steps"]

# ---- limits ------------------------------------------------------------
HARD_LIMITS_LOW: np.ndarray = np.array([_C["hard_limits_rad"][j][0] for j in JOINT_ORDER])
HARD_LIMITS_HIGH: np.ndarray = np.array([_C["hard_limits_rad"][j][1] for j in JOINT_ORDER])
SOFT_LIMITS_LOW: np.ndarray = np.array([_C["soft_limits_rad"][j][0] for j in JOINT_ORDER])
SOFT_LIMITS_HIGH: np.ndarray = np.array([_C["soft_limits_rad"][j][1] for j in JOINT_ORDER])

# ---- commands ----------------------------------------------------------
CMD_HULL: dict[str, list[float]] = _C["cmd_hull_trained"]
CMD_CLAMP: dict[str, list[float]] = _C["cmd_clamp_deployment"]

# ---- servo bus ---------------------------------------------------------
SERVO_IDS: dict[str, int] = _C["servo_ids"]
SERVO_BAUD: int = _C["servo_baud"]
#: policy-output index -> servo bus id, in JOINT_ORDER order
SERVO_ID_BY_ACTION_INDEX: list[int] = [SERVO_IDS[j] for j in JOINT_ORDER]

# ---- provenance --------------------------------------------------------
CHECKPOINT_MD5: str = _C["checkpoint_md5"]
ONNX_MD5: str = _C["onnx_md5"]
PLANT_MASS_KG: float = _C["plant_mass_kg"]


def decode_action(action: np.ndarray) -> np.ndarray:
    """Policy output -> joint position targets, clamped to the HARD limits.

    The clamp is not optional. `clip_actions` is null and Isaac Lab's
    `JointPositionAction` applies no limit clamp, so the policy output is
    unbounded. In simulation PhysX absorbs an out-of-range target; on hardware
    nothing does.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} actions, got {action.shape[0]}")
    return np.clip(Q_DEFAULT + ACTION_SCALE * action, HARD_LIMITS_LOW, HARD_LIMITS_HIGH)


def clamp_command(vx: float, vy: float, wz: float) -> tuple[float, float, float]:
    """Clamp a velocity command to the DEPLOYMENT limits (tighter than trained)."""
    def _c(v, k):
        lo, hi = CMD_CLAMP[k]
        return float(min(max(v, lo), hi))
    return _c(vx, "vx"), _c(vy, "vy"), _c(wz, "wz")
