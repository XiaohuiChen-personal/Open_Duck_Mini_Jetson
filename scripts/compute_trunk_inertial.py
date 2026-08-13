#!/usr/bin/env python3
"""Compute trunk_assembly and head_assembly composite inertials (full-tensor).

Reproducible source for the <inertial> values in robot.xml, robot_motors.xml
and robot.urdf. Composes the original (upstream) body inertials with every
added/removed/moved component at its actual position via the parallel-axis
theorem, using FULL 3x3 tensors throughout.

FRAME CORRECTION (v2 of this script): the upstream MJCF inertials carry a
`quat` — their `diaginertia` is expressed in the PRINCIPAL frame, not the
body frame. The first-pass computation (and the original
mass_inertia_calculations.md) treated those principal moments as body-frame
(Ixx, Iyy, Izz), which for the trunk is a near-axis-permutation error
(+78%/-10%/-27% per axis). This script takes the upstream BODY-FRAME full
tensors from the original robot.urdf (cross-checked against the MJCF
quat·diag·quatT at import time) and emits MJCF `fullinertia` / URDF full
matrices, so no frame information is ever dropped.

Component positions follow docs/jetson-mod/component_layout_v2.md.

Run:  python3 scripts/compute_trunk_inertial.py
"""
import os

import numpy as np

# ---------------------------------------------------------------------------
# Upstream (pre-modification) body inertials — BODY-FRAME full tensors from
# the original robot.urdf @ 49b6d7d. The MJCF equivalents (diaginertia+quat)
# reconstruct these to <4e-9 (verified below).
# ---------------------------------------------------------------------------
BASE_TRUNK = dict(
    mass=0.698526,
    com=np.array([-0.0483259, -9.97823e-05, 0.0384971]),
    tensor=np.array(
        [
            [0.0016770053685612820, -1.3166218023193001e-05, -3.2495310924311613e-05],
            [-1.3166218023193001e-05, 0.0034447879527285744, -2.1558142480798318e-06],
            [-3.2495310924311613e-05, -2.1558142480798318e-06, 0.0029263533437058373],
        ]
    ),
    # MJCF form for the self-check: quat + diaginertia from robot.xml @ 49b6d7d
    mjcf_quat=np.array([0.505499, 0.490695, 0.496207, 0.507413]),
    mjcf_diag=np.array([0.00344489, 0.00292719, 0.00167606]),
)

BASE_HEAD = dict(
    mass=0.352583,
    com=np.array([0.00761779, 0.00018098, 0.0242575]),
    tensor=np.array(
        [
            [0.0020638573779904646, -1.3848221046687924e-06, 9.1037277340193335e-05],
            [-1.3848221046687924e-06, 0.0014412682765872009, -2.9145451672243484e-06],
            [9.1037277340193335e-05, -2.9145451672243484e-06, 0.0009167731481529093],
        ]
    ),
    mjcf_quat=np.array([0.999221, -0.0025776, -0.0393431, -0.00138049]),
    mjcf_diag=np.array([0.00207104, 0.00144128, 0.000909578]),
)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def check_base(base, name):
    """Upstream URDF tensor must equal MJCF quat·diag·quatT (frame proof)."""
    R = quat_to_mat(base["mjcf_quat"])
    reconstructed = R @ np.diag(base["mjcf_diag"]) @ R.T
    err = np.abs(reconstructed - base["tensor"]).max()
    assert err < 5e-9, f"{name}: MJCF/URDF base inertia mismatch ({err:.2e})"


def box_tensor(m, lx, ly, lz):
    return np.diag(
        (m / 12.0) * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])
    )


POINT = np.zeros((3, 3))

# ---------------------------------------------------------------------------
# TRUNK components: (name, mass_delta_kg, position_m, tensor_about_own_com).
# Ledger: 0.698526 + 0.0585 + 0.176 + 0.180 + 0.005 + 0.015 + 0.037 + 0.008
#       = 1.178026 kg pre-Part-2; the Part-2 shell deltas appended from
#       cad_mod_deltas.json bring the trunk to its current total (printed
#       below when run).
# ---------------------------------------------------------------------------
TRUNK_COMPONENTS = [
    # 3x STS3215->STS3250 upgrades, at the servo-case AABB centers (mesh-measured).
    ("servo_hip_yaw_L (+19.5g)", 0.0195, [-0.0315, 0.035, 0.0658], POINT),
    ("servo_hip_yaw_R (+19.5g)", 0.0195, [-0.0315, -0.035, 0.0658], POINT),
    ("servo_neck (+19.5g)", 0.0195, [0.001, 0.0, 0.0775], POINT),
    # Jetson Orin Nano Super dev kit, low mount (z-span [-11.4, 23.4] mm).
    # CoM approximated at the envelope center (module/heatsink bias not modeled).
    ("jetson_dev_kit", 0.176, [-0.03, 0.0, 0.006], box_tensor(0.176, 0.103, 0.0905, 0.03477)),
    # 4 extra 18650 cells (pack goes 2 -> 6 = 2S1P -> 3S2P): 2x2 vertical grid
    # in the rear hump extension (Part-2 body_back change), seated on the
    # bore floor at z=-2 mm (a Phase-3/4 cradle/shim retains them).
    ("cells_extra_4x45g", 0.180, [-0.145, 0.0, 0.0306], box_tensor(0.180, 0.038, 0.038, 0.065)),
    # BMS upgrade delta, at the modeled BMS location in the hump.
    ("bms_delta", 0.005, [-0.1263, -0.0267, 0.0189], POINT),
    # DC-DC boost converter, under-plate mount beside the fan plenum.
    ("dcdc_converter", 0.015, [-0.060, 0.025, 0.0355], box_tensor(0.015, 0.043, 0.021, 0.014)),
    # Thermal partition ASSEMBLY (2mm PLA wall 3x103.5x66.7 + 1mm mica +
    # retention rails/gasket/adhesive). 37 g is the assembly budget from
    # hardware-specs; the bare resized wall+mica is ~26 g, the remainder
    # budgets the Part-2 retention features. Slab tensor as envelope.
    ("thermal_partition", 0.037, [-0.086, 0.0, 0.011], box_tensor(0.037, 0.003, 0.1035, 0.0667)),
    # BNO055 IMU relocation (part of BASE mass): battery-side pocket.
    ("imu_move_out", -0.003, [-0.0871, 0.0, 0.0433], POINT),
    ("imu_move_in", 0.003, [-0.100, 0.0, 0.0433], POINT),
    # NOTE: the servo-driver board is NOT moved (first-pass move reverted —
    # the shortened partition no longer reaches the board's z-band).
]

# Part-2 shell/chassis mesh deltas: loaded from scripts/cad_mod_deltas.json
# (written by scripts/generate_cad_mods.py).
#
# CHANGED BY TASK M2 (2026-08-12). These used to be signed DIFF-SOLID terms
# whose mass came from one assumed density. They are now MEASURED WHOLE-PART
# replacements: per modified part, a `<name>_baseline` term at negative mass and
# a `<name>_current` term at positive mass, each mass taken from
# scripts/part_mass_table.json (sliced at the process in
# scripts/print_process.json) and each tensor computed at that part's own
# measured effective density.
#
# The schema is unchanged -- name / mass / pos / tensor / volume_cm3 -- so
# nothing here needed rewriting. The net is now the measured delta by
# construction rather than a density times a volume difference, and the two
# tensors of a pair cancel over the regions the edit did not touch.
#
# Why it mattered: the old constant booked -88.48 g where the true delta is
# -13.95 g at the chosen profile, so trunk_assembly was 74.53 g light. See
# docs/jetson-mod/known_issues.md PLANT-10.
_DELTAS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cad_mod_deltas.json"
)


def _load_shell_deltas():
    import json

    with open(_DELTAS_JSON) as f:
        data = json.load(f)
    terms = []
    for t in data["terms"]:
        tensor = np.array(t["tensor"])
        # removed-material terms carry negative mass; their tensors subtract
        if t["mass"] < 0:
            tensor = -tensor
        terms.append((t["name"], t["mass"], t["pos"], tensor))
    return terms


TRUNK_COMPONENTS = TRUNK_COMPONENTS + _load_shell_deltas()
TRUNK_WIRING = 0.008  # lumped at the composite CoM

# ---------------------------------------------------------------------------
# HEAD components: remove the Pi Zero 2W, add the head_roll STS3250 delta.
# Pi position from the upstream robot.urdf raspberrypizerow origins.
# Servo delta at the head servo-case cluster center (geoms at
# (0.0255, 0, -0.00945) quat (0.7071,0.7071,0,0); case-stack union-AABB
# center measured from the three wj case meshes under that pose:
# (0.0125, 0.0, -0.0198)).
# ---------------------------------------------------------------------------
HEAD_COMPONENTS = [
    ("pi_zero_removed", -0.010, [0.03205, 0.048, 0.00595], POINT),
    ("servo_head_roll (+19.5g)", 0.0195, [0.0125, 0.0, -0.0198], POINT),
]
HEAD_WIRING = 0.0


def compose(base, components, wiring):
    masses = [base["mass"]] + [m for _, m, _, _ in components]
    positions = [base["com"]] + [np.asarray(p) for _, _, p, _ in components]
    total = sum(masses) + wiring
    com = sum(m * p for m, p in zip(masses, positions)) / sum(masses)
    # wiring at the composite CoM: adds mass, shifts nothing
    tensors = [base["tensor"]] + [t for _, _, _, t in components]
    I = np.zeros((3, 3))
    for m, p, Ic in zip(masses, positions, tensors):
        d = np.asarray(p) - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total, com, I


def report(name, total, com, I):
    eigvals = np.sort(np.linalg.eigvalsh(I))[::-1]
    tri = (
        eigvals[1] + eigvals[2] >= eigvals[0] - 1e-12
        and eigvals[0] + eigvals[2] >= eigvals[1] - 1e-12
        and eigvals[0] + eigvals[1] >= eigvals[2] - 1e-12
    )
    print(f"=== {name} ===")
    print(f"mass          : {total:.6f} kg")
    print(f"CoM           : ({com[0]:.7f}, {com[1]:.7f}, {com[2]:.7f}) m")
    print(f"body tensor   : ixx={I[0,0]:.8f} iyy={I[1,1]:.8f} izz={I[2,2]:.8f}")
    print(f"                ixy={I[0,1]:.3e} ixz={I[0,2]:.3e} iyz={I[1,2]:.3e}")
    print(f"principal     : ({eigvals[0]:.8f}, {eigvals[1]:.8f}, {eigvals[2]:.8f})  triangle: {'PASS' if tri else 'FAIL'}")
    print("MJCF:")
    print(
        f'  <inertial pos="{com[0]:.7f} {com[1]:.7f} {com[2]:.7f}" mass="{total:.6f}" '
        f'fullinertia="{I[0,0]:.8f} {I[1,1]:.8f} {I[2,2]:.8f} {I[0,1]:.8e} {I[0,2]:.8e} {I[1,2]:.8e}"/>'
    )
    print("URDF:")
    print(f'  <origin xyz="{com[0]:.7f} {com[1]:.7f} {com[2]:.7f}" rpy="0 0 0"/>')
    print(f'  <mass value="{total:.6f}"/>')
    print(
        f'  <inertia ixx="{I[0,0]:.8f}" ixy="{I[0,1]:.8e}"  ixz="{I[0,2]:.8e}" '
        f'iyy="{I[1,1]:.8f}" iyz="{I[1,2]:.8e}" izz="{I[2,2]:.8f}" />'
    )
    print()


def main():
    check_base(BASE_TRUNK, "trunk")
    check_base(BASE_HEAD, "head")
    print("base-frame self-check PASS (URDF tensor == MJCF quat*diag*quatT)\n")
    report("trunk_assembly", *compose(BASE_TRUNK, TRUNK_COMPONENTS, TRUNK_WIRING))
    report("head_assembly", *compose(BASE_HEAD, HEAD_COMPONENTS, HEAD_WIRING))


if __name__ == "__main__":
    main()
