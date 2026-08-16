# `v6d_contact_wrench` baseline — `v6d_head_turn`

Measured 2026-08-15 with the instrumentation added in Stage 1d of
`v7_servo_fix_plan.md`, so the v7 acceptance comparison rests on
numbers produced by committed code rather than an ad-hoc session.

```
env 0, 1100 control steps (22.0 s) after a 2 s settle
command (vx=0.0, vy=0.0, wz=0.5)
DOES THE HEAD MOVE?  angles in degrees, relative to the trunk
joint             default  cmd mean  cmd sd  cmd p2p  act sd  act p2p  err rms  tau rms  % on stop
neck_pitch           0.00    -17.10   2.588    8.075   2.347    6.524    0.896    0.464      26.1%
head_pitch           0.00      0.31   0.716    3.132   0.813    2.769    0.619    0.372       0.0%
head_yaw             0.00      3.28   3.705   10.973   3.951   11.228    1.286    0.543       0.0%
```
