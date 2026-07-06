# Mass and Inertia Calculations — Jetson Orin Nano Modification

## Input Parameters

### Existing Bodies

| Body | Mass (kg) | CoM (x, y, z) m | Diag Inertia (Ixx, Iyy, Izz) kg·m² |
|---|---|---|---|
| trunk_assembly | 0.698526 | (-0.0483259, -9.97823e-05, 0.0384971) | (0.00344489, 0.00292719, 0.00167606) |
| head_assembly | 0.352583 | (0.00761779, 0.00018098, 0.0242575) | (0.00207104, 0.00144128, 0.000909578) |

### Components Added to Trunk

| Component | Mass (kg) | Dimensions (m) | Position in trunk frame (m) | Shape |
|---|---|---|---|---|
| Jetson Orin Nano Dev Kit | 0.176 | 0.103 × 0.0905 × 0.03477 | (-0.03, 0.0, 0.04) | Box |
| 2× extra 18650 cells | 0.045 each | ⌀0.018 × 0.065 | (-0.13, 0.0, 0.03) | Cylinder (Y-axis) |
| DC-DC boost converter | 0.015 | 0.043 × 0.021 × 0.014 | (-0.05, 0.03, 0.035) | Box |
| Thermal partition + mica | 0.037 | 0.003 × 0.110 × 0.090 | (-0.08, 0.0, 0.04) | Box (thin slab) |
| Wiring/cables | 0.008 | — | at trunk CoM | Point mass |

### Component Removed from Head

| Component | Mass (kg) | Dimensions (m) | Position in head frame (m) |
|---|---|---|---|
| Raspberry Pi Zero 2W | 0.010 | 0.065 × 0.030 × 0.005 | (0.03205, 0.048, 0.00595) |

## Inertia Formulas

### Box (rectangular solid) about CoM
```
Ixx = m/12 × (Ly² + Lz²)
Iyy = m/12 × (Lx² + Lz²)
Izz = m/12 × (Lx² + Ly²)
```

### Cylinder about CoM (axis along Y)
```
Iyy = m × r² / 2        (axial)
Ixx = Izz = m/12 × (3r² + h²)  (transverse)
```

### Parallel Axis Theorem
```
I_at_P = I_cm + m × d²
where d² = [dy² + dz², dx² + dz², dx² + dy²] (component-wise for diagonal tensor)
```

## Step-by-Step Calculations

### New Trunk Assembly

**Step 1: Total mass**
```
M_new = 0.698526 + 0.176 + 2×0.045 + 0.015 + 0.037 + 0.008
      = 1.024526 kg
```

**Step 2: New center of mass**
```
CoM_new = Σ(mᵢ × posᵢ) / M_new

Numerator:
  0.698526 × (-0.0483259, -9.97823e-05, 0.0384971)
+ 0.176    × (-0.03, 0.0, 0.04)
+ 0.090    × (-0.13, 0.0, 0.03)
+ 0.015    × (-0.05, 0.03, 0.035)
+ 0.037    × (-0.08, 0.0, 0.04)
+ 0.008    × (-0.0483259, -9.97823e-05, 0.0384971)

CoM_new = (-0.0535209, 0.0003704, 0.0380119) m
```

**Step 3: Component inertias about their own CoM**

| Component | Ixx | Iyy | Izz |
|---|---|---|---|
| Jetson | 1.3785e-04 | 1.7333e-04 | 2.7572e-04 |
| Battery (each) | 1.6755e-05 | 1.8225e-06 | 1.6755e-05 |
| DC-DC | 7.9625e-07 | 2.5563e-06 | 2.8625e-06 |
| Partition+mica | 6.2283e-05 | 2.5003e-05 | 3.7336e-05 |

**Step 4: Shift all to new composite CoM via parallel axis theorem, then sum**
```
I_new_trunk = (0.00369962, 0.00380763, 0.00270784) kg·m²
```

**Step 5: Triangle inequality verification** ✓
- Ixx + Iyy = 0.00750725 ≥ Izz = 0.00270784 ✓
- Ixx + Izz = 0.00640746 ≥ Iyy = 0.00380763 ✓
- Iyy + Izz = 0.00651546 ≥ Ixx = 0.00369962 ✓

### New Head Assembly

**Step 1: Mass after removing Pi**
```
M_new = 0.352583 - 0.010 = 0.342583 kg
```

**Step 2: New CoM**
```
CoM_new = (M_orig × CoM_orig - m_pi × pos_pi) / M_new
        = (0.0069046, -0.0012149, 0.0247919) m
```

**Step 3: Remove Pi inertia contribution and shift to new CoM**
```
I_new_head = (0.00204329, 0.00142815, 0.00087563) kg·m²
```

**Step 4: Triangle inequality verification** ✓
- Ixx + Iyy = 0.00347143 ≥ Izz = 0.00087563 ✓
- Ixx + Izz = 0.00291892 ≥ Iyy = 0.00142815 ✓
- Iyy + Izz = 0.00230377 ≥ Ixx = 0.00204329 ✓

---

## STS3250 Servo Migration

The STS3215 servos (55g each) are replaced with STS3250 servos (74.5g each), adding +19.5g per servo. The battery is upgraded from 2S2P (4x 18650, 7.4V) to 3S2P (6x 18650, 11.1V), adding 2 more cells (+90g) and a larger BMS (+5g).

### Servo Mass Changes

14 STS3215→STS3250 servo upgrades distributed across the robot:

| Body | Servos in Body | Mass Delta (g) |
|---|---|---|
| trunk_assembly | 3 (hip_yaw_L, hip_yaw_R, neck_pitch) | +58.5 |
| hip_roll_assembly (L) | 1 (hip_roll_L) | +19.5 |
| left_roll_to_pitch_assembly | 1 (hip_pitch_L) | +19.5 |
| knee_and_ankle_assembly | 1 (knee_L) | +19.5 |
| knee_and_ankle_assembly_2 | 1 (ankle_L) | +19.5 |
| neck_pitch_assembly | 1 (head_pitch) | +19.5 |
| neck_yaw_assembly | 1 (head_yaw) | +19.5 |
| head_assembly | 1 (head_roll) | +19.5 |
| hip_roll_assembly_2 (R) | 1 (hip_roll_R) | +19.5 |
| right_roll_to_pitch_assembly | 1 (hip_pitch_R) | +19.5 |
| knee_and_ankle_assembly_3 | 1 (knee_R) | +19.5 |
| knee_and_ankle_assembly_4 | 1 (ankle_R) | +19.5 |
| **Total** | **14** | **+273.0** |

### Battery/BMS Changes (in trunk_assembly)

| Component | Delta (g) |
|---|---|
| 2× additional 18650 cells | +90 |
| BMS upgrade (2S→3S, ≥15A) | +5 |
| **Subtotal** | **+95** |

### Trunk Assembly (STS3250 update)

```
M_trunk_sts3250 = 1.024526 + 3×0.0195 + 0.095
                = 1.024526 + 0.0585 + 0.095
                = 1.178026 kg
```

Inertia scaled by mass ratio (1.178026 / 1.024526 = 1.149823…; the model
files use the exact ratio, which this doc previously rounded to 1.1498):
```
I_trunk_sts3250 = (0.00425392, 0.00437811, 0.00311354) kg·m²
```

### Head Assembly (STS3250 update)

```
M_head_sts3250 = 0.342583 + 0.0195
               = 0.362083 kg
```

Inertia scaled by mass ratio (0.362083 / 0.342583 = 1.056918…; exact ratio,
as in the model files):
```
I_head_sts3250 = (0.0021596, 0.00150944, 0.000925471) kg·m²
```

### Other Body Mass Updates

Each non-trunk, non-head servo body gets +19.5g with inertia scaled proportionally. See robot_motors.xml for exact values.

## Final Values for robot_motors.xml

### trunk_assembly
```xml
<inertial
    pos="-0.0535209 0.0003704 0.0380119"
    mass="1.178026"
    diaginertia="0.00425392 0.00437811 0.00311354"
/>
```

### head_assembly
```xml
<inertial
    pos="0.0069046 -0.0012149 0.0247919"
    mass="0.362083"
    diaginertia="0.0021596 0.00150944 0.000925471"
/>
```

### Mass Summary

| Body | Original (g) | After Jetson Mod (g) | After STS3250 (g) | Total Change |
|---|---|---|---|---|
| trunk_assembly | 698.5 | 1024.5 | 1178.0 | +479.5 |
| head_assembly | 352.6 | 342.6 | 362.1 | +9.5 |
| Other bodies (12 servo bodies) | 1010.4 | 1010.4 | 1205.4 | +195.0 |
| **Total** | **2061.5** | **2377.5** | **2745.5** | **+684.0 (+33.2%)** |
