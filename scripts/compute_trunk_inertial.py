#!/usr/bin/env python3
"""Compute the trunk_assembly composite inertial for the Jetson modification.

Reproducible source for the <inertial> values in robot.xml, robot_motors.xml
and robot.urdf. Composes the original (upstream) trunk inertial with every
added/moved component at its actual position via the parallel-axis theorem,
replacing the rounded mass-ratio scaling used in the first pass of
docs/jetson-mod/mass_inertia_calculations.md.

Component positions follow docs/jetson-mod/component_layout_v2.md (the
post-audit layout that resolves the Jetson/DC-DC/partition/chassis
interferences). All positions are in the trunk body frame, meters.

Run:  python3 scripts/compute_trunk_inertial.py
"""
import numpy as np

# Original upstream trunk_assembly (pre-modification), from
# docs/jetson-mod/mass_inertia_calculations.md "Existing Bodies" (verified
# against git history: robot.xml @ 49b6d7d).
BASE = dict(
    mass=0.698526,
    com=np.array([-0.0483259, -9.97823e-05, 0.0384971]),
    inertia=np.array([0.00344489, 0.00292719, 0.00167606]),
)


def box_inertia(m, lx, ly, lz):
    return (m / 12.0) * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])


def point_inertia(_m, *_):
    return np.zeros(3)


# name: (mass_delta_kg, position_m, inertia_about_own_com)
# Masses match the audited ledger: 0.698526 + 0.0585 + 0.176 + 0.180
#   + 0.005 + 0.015 + 0.037 + 0.008 = 1.178026 kg (unchanged total).
COMPONENTS = [
    # 3x STS3215->STS3250 upgrades inside the trunk, at the servo case centers
    # (case AABB centers measured from robot.xml geoms).
    ("servo_hip_yaw_L (+19.5g)", 0.0195, [-0.0315, 0.035, 0.0658], np.zeros(3)),
    ("servo_hip_yaw_R (+19.5g)", 0.0195, [-0.0315, -0.035, 0.0658], np.zeros(3)),
    ("servo_neck (+19.5g)", 0.0195, [0.001, 0.0, 0.0775], np.zeros(3)),
    # Jetson Orin Nano Super dev kit, low mount on trunk_bottom posts
    # (z-span [-11.4, 23.4] mm: clears roll bearings, chassis plate, servos).
    ("jetson_dev_kit", 0.176, [-0.03, 0.0, 0.006], box_inertia(0.176, 0.103, 0.0905, 0.03477)),
    # 4 extra 18650 cells (2S2P -> 3S2P went 2 -> 6 cells), battery hump.
    ("cells_extra_4x45g", 0.180, [-0.13, 0.0, 0.0325], box_inertia(0.180, 0.036, 0.042, 0.065)),
    # BMS upgrade delta, at the modeled BMS location in the hump.
    ("bms_delta", 0.005, [-0.1263, -0.0267, 0.0189], np.zeros(3)),
    # DC-DC boost converter, under-plate mount beside the fan plenum.
    ("dcdc_converter", 0.015, [-0.060, 0.025, 0.0355], box_inertia(0.015, 0.043, 0.021, 0.014)),
    # Thermal partition (2mm PLA + 1mm mica), floor-to-plate at x=-0.086.
    ("thermal_partition", 0.037, [-0.086, 0.0, 0.0104], box_inertia(0.037, 0.003, 0.1035, 0.068)),
    # Relocations of components already inside BASE (mass moves, not adds):
    # servo driver board +47.5mm x (out of the partition plane).
    ("board_move_out", -0.012, [-0.0845, 0.0165, 0.0596], np.zeros(3)),
    ("board_move_in", 0.012, [-0.037, 0.0165, 0.0596], np.zeros(3)),
    # BNO055 IMU into the battery-side pocket (cool, EMI-distant).
    ("imu_move_out", -0.003, [-0.0871, 0.0, 0.0433], np.zeros(3)),
    ("imu_move_in", 0.003, [-0.100, 0.0, 0.0433], np.zeros(3)),
]

WIRING_MASS = 0.008  # lumped at the composite CoM, as in the original doc


def main():
    masses = [BASE["mass"]] + [m for _, m, _, _ in COMPONENTS]
    positions = [BASE["com"]] + [np.asarray(p) for _, _, p, _ in COMPONENTS]
    total_m = sum(masses)

    com = sum(m * p for m, p in zip(masses, positions)) / total_m
    # wiring at the composite CoM: shifts nothing, adds mass
    total_with_wiring = total_m + WIRING_MASS
    com_final = (com * total_m + com * WIRING_MASS) / total_with_wiring  # unchanged

    inertias = [BASE["inertia"]] + [np.asarray(i) for _, _, _, i in COMPONENTS]
    I = np.zeros(3)
    for m, p, Ic in zip(masses, positions, inertias):
        d = np.asarray(p) - com_final
        shift = m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
        I += Ic + shift
    # wiring: point mass at CoM -> zero parallel-axis contribution

    print(f"total mass      : {total_with_wiring:.6f} kg")
    print(f"CoM             : ({com_final[0]:.7f}, {com_final[1]:.7f}, {com_final[2]:.7f}) m")
    print(f"diag inertia    : ({I[0]:.8f}, {I[1]:.8f}, {I[2]:.8f}) kg.m^2")
    tri = (I[0] + I[1] >= I[2]) and (I[0] + I[2] >= I[1]) and (I[1] + I[2] >= I[0])
    print(f"triangle ineq   : {'PASS' if tri else 'FAIL'}")
    print("\nMJCF:")
    print(f'  <inertial pos="{com_final[0]:.7f} {com_final[1]:.7f} {com_final[2]:.7f}" mass="{total_with_wiring:.6f}" diaginertia="{I[0]:.8f} {I[1]:.8f} {I[2]:.8f}"/>')
    print("URDF:")
    print(f'  <origin xyz="{com_final[0]:.7f} {com_final[1]:.7f} {com_final[2]:.7f}" rpy="0 0 0"/>')
    print(f'  <mass value="{total_with_wiring:.6f}"/>')
    print(f'  <inertia ixx="{I[0]:.8f}" iyy="{I[1]:.8f}" izz="{I[2]:.8f}" ixy="0" ixz="0" iyz="0"/>')


if __name__ == "__main__":
    main()
