# v7b_servo_safe (weight -4e-2) — iteration 2 probes

Checkpoint `model_11996.pt`, fine-tuned from v7's `model_8997.pt`.

## Torque envelope
```
joint               peak      p99      rms  % over cont
-------------------------------------------------------
left_hip_yaw       0.939    0.749    0.333        0.00%
neck_pitch         0.739    0.407    0.207        0.00%
right_hip_yaw      0.758    0.496    0.213        0.00%
left_hip_roll      1.834    1.227    0.569        0.18%
head_pitch         0.991    0.632    0.281        0.00%
right_hip_roll     1.933    1.197    0.650        0.07%
left_hip_pitch     3.786    3.631    1.585       26.79%
head_yaw           1.143    0.767    0.340        0.00%
right_hip_pitch    4.084    3.691    1.587       29.27%
left_knee          3.073    2.015    1.109       10.94%
head_roll          1.231    0.498    0.214        0.00%
right_knee         2.463    2.267    1.182       22.33%
left_ankle         4.903    2.930    1.474       31.54%
left_antenna       0.050    0.009    0.002        0.00%
right_antenna      0.058    0.006    0.002        0.00%
right_ankle        3.321    2.881    1.493       34.33%

worst leg joint by peak: left_ankle  4.903 N.m (100% of stall, 312% of continuous)
worst leg joint by rms : right_hip_pitch  1.587 N.m (101% of continuous)
  -> joints that WOULD TRIP the 2 s overload cutout: none
```

## Head motion — straight / turn
```
DOES THE HEAD MOVE?  angles in degrees, relative to the trunk
joint             default  cmd mean  cmd sd  cmd p2p  act sd  act p2p  err rms  tau rms  % on stop
neck_pitch           0.00     -0.54   0.629    2.299   0.572    2.115    0.453    0.191       0.0%
head_pitch           0.00     -2.52   1.575    4.831   1.675    5.132    0.597    0.270       0.0%
head_yaw             0.00     -0.22   2.776    8.887   2.959    9.101    0.837    0.318       0.0%Loading user config located at: '/home/xiaohui_chen/.cache/packman/chk/kit-kernel/107.3.3+isaac.229672.69cbf6ad.gl.manylinux_2_35_aarch64.release/data/Kit/Isaac-Sim/5.1/user.config.json'
[Info] [carb] Logging to file: /home/xiaohui_chen/.cache/packman/chk/kit-kernel/107.3.3+isaac.229672.69cbf6ad.gl.manylinux_2_35_aarch64.release/logs/Kit/Isaac-Sim/5.1/kit_20260816_054226.log

DOES THE HEAD MOVE?  angles in degrees, relative to the trunk
joint             default  cmd mean  cmd sd  cmd p2p  act sd  act p2p  err rms  tau rms  % on stop
neck_pitch           0.00     -1.56   1.140    3.750   1.218    3.986    0.374    0.220       0.0%
head_pitch           0.00     -0.97   1.029    3.360   1.081    3.378    0.366    0.185       0.0%
head_yaw             0.00      3.84   2.300    7.624   2.464    7.901    0.750    0.319       0.0%```
