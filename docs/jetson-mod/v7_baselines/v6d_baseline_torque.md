# `v6d_contact_wrench` baseline — `v6d_baseline_torque`

Measured 2026-08-15 with the instrumentation added in Stage 1d of
`v7_servo_fix_plan.md`, so the v7 acceptance comparison rests on
numbers produced by committed code rather than an ad-hoc session.

```
task    Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0
command (vx=0.2, vy=0.0, wz=0.0)
plant   2.729035 kg   32 envs x 1500 steps (30 s)
STS3250 continuous 1.569 N.m (16 kg.cm) | peak stall 4.903 N.m (50 kg.cm)
longest CONTINUOUS run above each threshold (env 0, 1500 steps @ 50 Hz); Feetech trips after 2.0 s
joint            >1.569 rated  >3.923 overload  >4.099 overcur
left_hip_yaw            0.00s            0.00s           0.00s
neck_pitch              1.06s            0.08s           0.06s
right_hip_yaw           0.00s            0.00s           0.00s
left_hip_roll           0.06s            0.00s           0.00s
head_pitch              0.08s            0.00s           0.00s
right_hip_roll          0.06s            0.00s           0.00s
left_hip_pitch          0.10s            0.06s           0.06s
head_yaw                0.06s            0.00s           0.00s
right_hip_pitch         0.22s            0.08s           0.08s
left_knee               0.10s            0.04s           0.04s
head_roll               0.00s            0.00s           0.00s
right_knee              0.08s            0.00s           0.00s
left_ankle              0.12s            0.06s           0.06s
left_antenna            0.00s            0.00s           0.00s
right_antenna           0.00s            0.00s           0.00s
right_ankle             0.10s            0.06s           0.04s
  -> joints that WOULD TRIP the 2 s overload cutout: none
  -> per-step torque dumped to /home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/v6d_torque_baseline.npz
joint               peak      p99      rms  % over cont
-------------------------------------------------------
left_hip_yaw       1.506    0.660    0.242        0.00%
neck_pitch         4.710    4.501    3.238       92.95%
right_hip_yaw      1.305    0.723    0.288        0.00%
left_hip_roll      2.442    2.034    0.914        8.16%
head_pitch         3.602    1.544    0.614        0.95%
right_hip_roll     3.000    1.971    0.820        3.40%
left_hip_pitch     4.903    4.903    2.723       58.59%
head_yaw           2.714    2.545    0.980       10.87%
right_hip_pitch    4.903    4.903    2.579       49.86%
left_knee          4.903    3.678    2.250       60.73%
head_roll          1.589    0.723    0.331        0.01%
right_knee         4.029    3.968    2.302       57.41%
left_ankle         4.903    4.903    2.470       40.73%
left_antenna       0.038    0.010    0.002        0.00%
right_antenna      0.053    0.014    0.003        0.00%
right_ankle        4.903    4.903    2.409       35.54%
worst leg joint by peak: left_hip_pitch  4.903 N.m (100% of stall, 312% of continuous)
worst leg joint by rms : left_hip_pitch  2.723 N.m (174% of continuous)
Linear mass scaling to each candidate print process
```
