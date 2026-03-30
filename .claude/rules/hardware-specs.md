# Hardware Specifications

## Robot Physical Specs

| Parameter | Value |
|---|---|
| Total height | ~420 mm (legs extended) |
| Total mass (after mod) | ~2,746 g |
| DOFs | 15 joints + 1 head_roll = 16 actuators |
| Servos | 14x Feetech STS3250 (12V, 50 kg.cm stall, 74.5g each) |
| Ear servos | 2x SG90 micro servos (in head) |
| Battery | 6x 18650 Li-ion cells (3S2P, 11.1V) |
| IMU | BNO055 (I2C, mounted in trunk) |

## Jetson Orin Nano Super Developer Kit

| Spec | Value |
|---|---|
| Dimensions | 103 x 90.5 x 34.77 mm (full dev kit incl. heatsink+fan) |
| Weight | 176 g |
| GPU | 1024 CUDA + 32 Tensor cores (Ampere) |
| AI Performance | 67 TOPS |
| RAM | 8 GB LPDDR5 unified (shared CPU+GPU) |
| Memory bandwidth | 102 GB/s |
| CPU | 6-core ARM Cortex-A78AE @ 1.5 GHz |
| Power modes | 7W eco / 15W default / 25W max |
| Ports | 2x CSI camera, 4x USB 3.2, DisplayPort, M.2, 40-pin GPIO, DC barrel jack |
| Location on robot | Trunk cavity (relocated from head) |

## Memory Budget on Jetson (8 GB total)

```
Cosmos Reason2 (W4A16-Edge2):  ~5.8 GB
Locomotion policy (TensorRT):  ~0.1 GB
Camera pipeline:               ~0.3 GB
OS + system:                   ~1.5 GB
Total:                         ~7.7 GB  (fits)
```

## Mass Changes from Modification

| Body | Original (g) | Modified (g) | Change |
|---|---|---|---|
| trunk_assembly | 698.5 | ~1,178 | +176 (Jetson) +58.5 (3 servo upgrades STS3215→STS3250 in trunk: +19.5g each) +90 (batteries) +5 (BMS) +15 (DC-DC) +37 (thermal partition+mica) +8 (wiring) |
| head_assembly | 352.6 | ~342.6 | -10 (Pi removed) |
| All 14 servos | 770 (14x55g) | 1,043 (14x74.5g) | +273 g total (+19.5g each x14, 11 allocated to limbs) |
| Total robot | 2,062 | ~2,746 | +684 g (+33.2%) |

## Raspberry Pi Zero 2W (being removed)

| Spec | Value |
|---|---|
| Dimensions | 65 x 30 x 5 mm |
| Weight | 10 g |
| Location | Head (head_assembly body in MuJoCo model) |
| CPU | Quad-core ARM Cortex-A53 @ 1 GHz |
| RAM | 512 MB |
| AI compute | None |

## Motor Parameters (from BAM identification)

Source: `experiments/v2/params_sts3250_id008.json` (kscalelabs/sysid STS3250 id008)

| Parameter | Value | Used in |
|---|---|---|
| kp (position gain) | 45.53 | Isaac Lab actuator config |
| kd (velocity gain) | 1.346 | Isaac Lab actuator config |
| armature | 0.040 | MuJoCo/Isaac Sim joint model |
| frictionloss | 0.200 | MuJoCo/Isaac Sim joint model |
| torque limit | 8.716 Nm | Actuator effort_limit |
| kt (torque constant) | 1.0006 | BAM motor model |
| R (resistance) | 1.3890 | BAM motor model |
