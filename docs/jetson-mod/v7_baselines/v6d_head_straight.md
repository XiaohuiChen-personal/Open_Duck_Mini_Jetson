# `v6d_contact_wrench` baseline — `v6d_head_straight`

Measured 2026-08-15 with the instrumentation added in Stage 1d of
`v7_servo_fix_plan.md`, so the v7 acceptance comparison rests on
numbers produced by committed code rather than an ad-hoc session.

```
env 0, 1400 control steps (28.0 s) after a 2 s settle
command (vx=0.2, vy=0.0, wz=0.0)
DOES THE HEAD MOVE?  angles in degrees, relative to the trunk
joint             default  cmd mean  cmd sd  cmd p2p  act sd  act p2p  err rms  tau rms  % on stop
neck_pitch           0.00    -23.88   0.872    4.442   0.000    0.005    3.980    3.164     100.0%
head_pitch           0.00     -5.73   2.379    8.938   2.593    9.387    1.358    0.702       0.0%
```
