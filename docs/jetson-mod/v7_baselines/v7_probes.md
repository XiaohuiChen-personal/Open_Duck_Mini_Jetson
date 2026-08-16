# v7_servo_safe — Stage 6a/6b results (probes)

Checkpoint `model_8997.pt`, run `2026-08-15_21-46-17_v7_servo_safe`, weight -1e-2.

## Torque envelope
```
joint               peak      p99      rms  % over cont
-------------------------------------------------------
left_hip_yaw       1.064    0.614    0.261        0.00%
neck_pitch         1.046    0.848    0.434        0.00%
right_hip_yaw      1.120    0.534    0.245        0.00%
left_hip_roll      2.241    1.479    0.713        0.29%
head_pitch         2.730    0.785    0.413        0.38%
right_hip_roll     2.657    1.617    0.705        1.49%
left_hip_pitch     4.776    4.651    2.120       42.44%
head_yaw           1.524    1.476    0.697        0.00%
right_hip_pitch    4.903    4.792    1.979       41.34%
left_knee          3.711    2.879    1.740       49.79%
head_roll          2.589    0.716    0.336        0.26%
right_knee         3.369    3.229    1.737       49.28%
left_ankle         4.821    3.790    1.786       36.35%
left_antenna       0.047    0.013    0.003        0.00%
right_antenna      0.061    0.016    0.005        0.00%
right_ankle        4.348    3.902    1.757       34.26%

worst leg joint by peak: right_hip_pitch  4.903 N.m (100% of stall, 312% of continuous)
worst leg joint by rms : left_hip_pitch  2.120 N.m (135% of continuous)

longest CONTINUOUS run above each threshold (env 0, 1500 steps @ 50 Hz); Feetech trips after 2.0 s
joint            >1.569 rated  >3.923 overload  >4.099 overcur
left_hip_yaw            0.00s            0.00s           0.00s
neck_pitch              0.00s            0.00s           0.00s
right_hip_yaw           0.00s            0.00s           0.00s
left_hip_roll           0.02s            0.00s           0.00s
head_pitch              0.06s            0.00s           0.00s
right_hip_roll          0.08s            0.00s           0.00s
left_hip_pitch          0.10s            0.04s           0.00s
head_yaw                0.00s            0.00s           0.00s
right_hip_pitch         0.10s            0.04s           0.04s
left_knee               0.08s            0.00s           0.00s
head_roll               0.04s            0.00s           0.00s
right_knee              0.08s            0.00s           0.00s
left_ankle              0.10s            0.02s           0.00s
left_antenna            0.00s            0.00s           0.00s
right_antenna           0.00s            0.00s           0.00s
right_ankle             0.12s            0.02s           0.00s
  -> joints that WOULD TRIP the 2 s overload cutout: none
```

## Head motion — straight (vx 0.2)
```
DOES THE HEAD MOVE?  angles in degrees, relative to the trunk
joint             default  cmd mean  cmd sd  cmd p2p  act sd  act p2p  err rms  tau rms  % on stop
neck_pitch           0.00    -10.99   1.087    3.964   1.148    3.775    0.805    0.445       0.0%
head_pitch           0.00     -4.56   1.961    6.045   2.067    6.302    0.801    0.360       0.0%
head_yaw             0.00      4.13   5.937   18.022   6.302   19.163    1.786    0.693       0.0%Loading user config located at: '/home/xiaohui_chen/.cache/packman/chk/kit-kernel/107.3.3+isaac.229672.69cbf6ad.gl.manylinux_2_35_aarch64.release/data/Kit/Isaac-Sim/5.1/user.config.json'
[Info] [carb] Logging to file: /home/xiaohui_chen/.cache/packman/chk/kit-kernel/107.3.3+isaac.229672.69cbf6ad.gl.manylinux_2_35_aarch64.release/logs/Kit/Isaac-Sim/5.1/kit_20260815_235948.log
2026-08-16T04:59:48Z [172ms] [Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
```

## Head motion — turn (wz 0.5)
```
DOES THE HEAD MOVE?  angles in degrees, relative to the trunk
joint             default  cmd mean  cmd sd  cmd p2p  act sd  act p2p  err rms  tau rms  % on stop
neck_pitch           0.00     -9.29   1.955    6.283   2.095    6.488    0.778    0.400       0.0%
head_pitch           0.00      0.11   2.754    8.515   2.891    8.753    0.841    0.349       0.0%
head_yaw             0.00      5.09   5.336   16.025   5.678   16.660    1.593    0.609       0.0%```

## Acceptance (scripts/check_v7_acceptance.py)
```

--- §6a ---
  [FAIL] worst leg RMS <= 1.0 N.m                    2.060 (right_hip_pitch)
  [PASS] no leg joint p99 at the 4.903 clip          none
  [PASS] longest run > 3.923 N.m is < 2.0 s          0.04 s

--- §6b ---
  [PASS] neck_pitch travel >= 2.0 deg                3.775 deg
  [PASS] neck_pitch on-stop <= 10 %                  0.0 %
  [PASS] neck_pitch torque RMS <= 1.5 N.m            0.445 N.m
  [PASS] head_yaw travel >= 10 deg (not frozen)      19.2 deg
  [PASS] [turn] neck travel not degraded (>= 2 deg)  6.488 deg
  [PASS] [turn] head_yaw still moves (>= 5 deg)      16.7 deg

--- §6c ---
  [FAIL] gate battery present                        MISSING v7_servo_safe.json

VERDICT: FAIL — 2 hard bar(s): ['worst leg RMS <= 1.0 N.m', 'gate battery present']
```
