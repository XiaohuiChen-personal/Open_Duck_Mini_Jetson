# PPO-rollout AMP clip validation

Result: PASS

Clips: 22 in `/home/xiaohui_chen/Projects/Open_Duck_Mini_Jetson/isaac_lab_env/open_duck_mini_v2/amp/motions_ppo`

dof_names (16): ['left_hip_yaw', 'neck_pitch', 'right_hip_yaw', 'left_hip_roll', 'head_pitch', 'right_hip_roll', 'left_hip_pitch', 'head_yaw', 'right_hip_pitch', 'left_knee', 'head_roll', 'right_knee', 'left_ankle', 'left_antenna', 'right_antenna', 'right_ankle']
body_names (3): ['base', 'foot_assembly', 'foot_assembly_2']
shared dt: 0.020000 s (50.0 fps)

## AMP observation liveness (51 dims, all clips concatenated)

Frames: 11000   Dims: 51
Data-dead dims (std < 1e-06, EXPLOITABLE): NONE
Structurally-zero dims (expected, excluded from failure): 34 (orient_tn[1])

Five lowest-variance dims:
  dim 34 orient_tn[1]     std=1.001e-07 [structural]
  dim 33 orient_tn[0]     std=1.083e-03
  dim 38 orient_tn[5]     std=1.291e-03
  dim 32 root_z           std=2.583e-03
  dim 50 foot_offset[5]   std=6.260e-03

## Root height (base z, all frames)
min 0.1635  mean 0.1722  max 0.1774  std 0.0026 m
(foot z above is the foot-body ORIGIN, not the sole, so ground penetration is not directly derivable here.)

## Per-clip summary

| clip | frames | root_z min/mean/max (m) | max |dq|/step (rad) | min foot-body z (m) |
|---|---|---|---|---|
| duck_ppo_-0.074_-0.037_-0.074.npz | 500 | 0.1701/0.1734/0.1762 | 0.1066 | 0.0354 |
| duck_ppo_-0.148_-0.037_-0.074.npz | 500 | 0.1700/0.1735/0.1761 | 0.1283 | 0.0354 |
| duck_ppo_-0.148_-0.037_-1.111.npz | 500 | 0.1666/0.1722/0.1757 | 0.1088 | 0.0361 |
| duck_ppo_-0.148_-0.037_1.222.npz | 500 | 0.1679/0.1722/0.1763 | 0.1036 | 0.0371 |
| duck_ppo_-0.148_-0.111_-0.074.npz | 500 | 0.1687/0.1734/0.1774 | 0.1303 | 0.0355 |
| duck_ppo_-0.148_0.111_-0.074.npz | 500 | 0.1681/0.1732/0.1772 | 0.1454 | 0.0353 |
| duck_ppo_0.074_-0.037_-0.074.npz | 500 | 0.1686/0.1724/0.1753 | 0.1141 | 0.0357 |
| duck_ppo_0.0_-0.037_-0.074.npz | 500 | 0.1698/0.1732/0.1761 | 0.0919 | 0.0356 |
| duck_ppo_0.0_-0.037_-1.111.npz | 500 | 0.1678/0.1733/0.1760 | 0.1069 | 0.0362 |
| duck_ppo_0.0_-0.037_1.222.npz | 500 | 0.1679/0.1713/0.1758 | 0.0903 | 0.0370 |
| duck_ppo_0.0_-0.111_-0.074.npz | 500 | 0.1684/0.1732/0.1770 | 0.1015 | 0.0356 |
| duck_ppo_0.0_-0.111_-1.111.npz | 500 | 0.1664/0.1733/0.1770 | 0.1312 | 0.0365 |
| duck_ppo_0.0_-0.111_1.222.npz | 500 | 0.1678/0.1716/0.1760 | 0.0977 | 0.0370 |
| duck_ppo_0.0_0.111_-0.074.npz | 500 | 0.1684/0.1734/0.1773 | 0.1037 | 0.0352 |
| duck_ppo_0.0_0.111_-1.111.npz | 500 | 0.1705/0.1730/0.1769 | 0.1107 | 0.0360 |
| duck_ppo_0.0_0.111_1.222.npz | 500 | 0.1659/0.1713/0.1765 | 0.1119 | 0.0366 |
| duck_ppo_0.148_-0.037_-0.074.npz | 500 | 0.1681/0.1720/0.1746 | 0.1480 | 0.0357 |
| duck_ppo_0.222_-0.037_-0.074.npz | 500 | 0.1663/0.1708/0.1738 | 0.1810 | 0.0359 |
| duck_ppo_0.222_-0.037_-1.111.npz | 500 | 0.1642/0.1705/0.1753 | 0.1709 | 0.0366 |
| duck_ppo_0.222_-0.037_1.222.npz | 500 | 0.1635/0.1695/0.1751 | 0.1620 | 0.0369 |
| duck_ppo_0.222_-0.111_-0.074.npz | 500 | 0.1662/0.1712/0.1741 | 0.1853 | 0.0357 |
| duck_ppo_0.222_0.111_-0.074.npz | 500 | 0.1655/0.1706/0.1743 | 0.1813 | 0.0358 |
