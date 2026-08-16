# `v7_smoke` — Stage 4 evidence

100-iteration validation of the SERVO-1 + SERVO-2 config, 2026-08-15.
`.training_runs/` is gitignored, so the load-bearing lines are captured here.

## The two blockers the adversarial review caught
```
[INFO]: Loading model checkpoint from: /home/xiaohui_chen/IsaacLab/logs/rsl_rl/open_duck_ppo_v7/0000-00-00_v6d_seed/model_5998.pt
Learning iteration 5998/6098          <- counter starts at the seed, not 0
final checkpoint: model_6097.pt       <- 5998 + 100 - 1, NOT 6098
          Episode_Reward/dof_torques_l2: -0.2898
                                      <- a -2e-5 weight would log ~ -0.001
```

## Plant unchanged (design decision D2)
```
[v5] ContactRegimeEvent: measured robot weight 26.77 N (2.729 kg) -> wrench 1.34-5.35 N (active_frac=0.5); obstacle asset present, obstacle_frac=0.25
```

## The new terms are active and the gait is undisturbed
```
                            Mean reward: 225.41
        Episode_Reward/joint_pos_limits: -0.0133
                 Gait/duty_in_band_frac: 0.9901
v6d reference: joint_pos_limits -0.0040 | duty 0.9891 | reward 225.95
```
