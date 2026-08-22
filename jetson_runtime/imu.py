"""IMU math — Task S.4. Pure functions, no I2C (that is S.9/S.10).

QUATERNION CONVENTION IS (w, x, y, z), matching Isaac's `root_quat_w`.

The BNO055 reports its quaternion in a different register order. **Adapt it in
the driver, not here** — a convention mismatch buried in shared math is the kind
of bug that shows up as the robot believing it is upside down.
"""

from __future__ import annotations

import numpy as np

# World gravity DIRECTION (not magnitude). `projected_gravity_b` in Isaac is a
# unit vector: R_world->body @ (0, 0, -1). It is ~(0, 0, -1) when upright.
GRAVITY_DIR_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def quat_to_rotation_matrix(q_wxyz) -> np.ndarray:
    """(w, x, y, z) -> 3x3 R_body->world. Normalises defensively."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quaternion must have 4 components, got {q.shape[0]}")
    n = np.linalg.norm(q)
    if n < 1e-9:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_to_projected_gravity(q_wxyz) -> np.ndarray:
    """(w, x, y, z) -> gravity direction expressed in the BODY frame.

    R_world->body is the transpose of R_body->world, so this is R.T @ (0,0,-1).
    """
    R = quat_to_rotation_matrix(q_wxyz)
    return (R.T @ GRAVITY_DIR_W.astype(np.float64)).astype(np.float32)


def rotate_gyro(gyro_imu, R_imu_to_body) -> np.ndarray:
    """Angular velocity from the IMU's frame into the robot's body frame.

    The IMU is not mounted coincident with `trunk_assembly`; S.9 measures the
    fixed rotation between them and this applies it.
    """
    gyro_imu = np.asarray(gyro_imu, dtype=np.float64).reshape(-1)
    if gyro_imu.shape[0] != 3:
        raise ValueError(f"gyro must have 3 components, got {gyro_imu.shape[0]}")
    R = np.asarray(R_imu_to_body, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R_imu_to_body must be 3x3, got {R.shape}")
    return (R @ gyro_imu).astype(np.float32)
