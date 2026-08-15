# `v6d_contact_wrench` torque envelope — raw measurement

Produced 2026-08-15 for [`locomotion_selection.md`](locomotion_selection.md) §4
and `known_issues.md` **SERVO-1** / **SERVO-2**.

```
$ cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/measure_joint_torque.py \
    --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 \
    --checkpoint <repo>/exported_policies/v6d_contact_wrench_ppo/model_5998.pt --headless

STS3250 continuous 1.569 N.m (16 kg.cm) | peak stall 4.903 N.m (50 kg.cm)

joint               peak      p99      rms  % over cont
-------------------------------------------------------
left_hip_yaw       1.501    0.650    0.248        0.00%
neck_pitch         4.710    4.523    3.241       94.70%
right_hip_yaw      1.306    0.711    0.289        0.00%
left_hip_roll      2.454    2.014    0.905        7.63%
head_pitch         3.601    1.656    0.646        1.35%
right_hip_roll     3.005    2.050    0.823        2.87%
left_hip_pitch     4.903    4.903    2.735       60.59%
head_yaw           2.741    2.560    0.987       10.99%
right_hip_pitch    4.903    4.903    2.591       50.91%
left_knee          4.903    3.734    2.269       62.20%
head_roll          1.595    0.695    0.326        0.01%
right_knee         4.052    3.926    2.298       57.58%
left_ankle         4.903    4.903    2.475       40.77%
left_antenna       0.040    0.006    0.001        0.00%
right_antenna      0.057    0.006    0.002        0.00%
right_ankle        4.903    4.903    2.417       34.85%

worst leg joint by peak: left_hip_pitch  4.903 N.m (100% of stall, 312% of continuous)
worst leg joint by rms : left_hip_pitch  2.735 N.m (174% of continuous)

Linear mass scaling to each candidate print process
(printed-mass delta against fdm-pla; first-order, go/no-go only):
process                   plant kg   peak N.m   rms N.m  rms vs cont
fdm-abs                      2.548      4.578     2.553         163%
fdm-asa                      2.575      4.627     2.580         164%
fdm-pla                      2.729      4.903     2.735         174%
fdm-petg                     2.756      4.952     2.762         176%
```

Stored as `.md` deliberately: `.gitignore:18` blanket-ignores `*.txt`, so
every other evidence dump in this directory (`inertial_rebuild_report.txt`,
`battery_bay_before.txt`, the ten `print_mass_*.txt`) is referenced by a
document but **absent from the repository**. A decision record whose
evidence is not in the repo has the same provenance gap as ART-1.
