# Locomotion selection — `v6d_contact_wrench` is the mainline

**Decided 2026-08-15.** This document names the single locomotion policy the
project uses from here, states the evidence, and — because two of the reasons
first given for it were wrong — records what was actually checked and what was
struck.

## DECISION

**`v6d_contact_wrench` is the locomotion policy.** Archive:
[`exported_policies/v6d_contact_wrench_ppo/`](../../exported_policies/v6d_contact_wrench_ppo/),
`model_5998.pt`, md5 `37da88d08bf5d593a3febf74bce96dbe`, obs **53** / action
**14**, plant **2.729035 kg**, USD asset hash `767f2415d1b3a056d95e9c310434dbbc`.

Every other policy the project has trained is retired. No other checkpoint is to
be quoted, deployed, or benchmarked without re-measuring it on the current
plant first.

## 1. Why v6d and not `v6_robust`

`v6_robust` is the same policy before the contact fine-tune, so this is the only
strictly controlled comparison the project owns — same plant, same seed, same
protocol, 3,840 episodes per cell.

**`v6_robust` wins zero of 30 fall-rate condition cells.** Across open field,
both push rules, the wrench and the obstacle × 6 velocity conditions, v6d is at
least as good in every single one.

| gate | `v6_robust` | **`v6d`** |
|---|---|---|
| open field | 0.000 % · 6/6 gait | 0.000 % · 6/6 gait |
| push, v4 fall rule | 10.990 % | **0.104 %** |
| push, v5 fall rule | 13.359 % | **0.026 %** |
| sustained wrench | **100.000 %** (3.03 s mean episode) | **70.365 %** (17.28 s) |
| obstacle graze | 34.245 % | **7.318 %** |
| lin-vel tracking error | 0.1652 m/s | **0.1495 m/s** |
| energy proxy | 19.969 W | **19.586 W** |

Source: [`rebuild_results.md`](rebuild_results.md), scored by
`scripts/regate_report.py` (exit 0).

`v6_robust` keeps three metrics: jerk (0.0601 vs 0.0749), yaw-rate error
(0.1075 vs 0.1151 rad/s — 0.43 deg/s, immaterial) and stance asymmetry. **The
jerk difference is the one real cost of this choice** and it is quantified in §4.

## 2. Why not `v5d_contact_wrench` — and two reasons that were WRONG

Two justifications were offered informally and are struck here, because a
decision record that carries a wrong reason invites the decision to be reopened
on the right one later.

**STRUCK — "the v1–v5 checkpoints cannot be loaded."** False as stated. That is
a property of the gym *registrations*, not of the checkpoints. The M0b antenna
filter is applied in `OpenDuckRoughEnvCfg.__post_init__`
(`env_cfg.py:308`), so a subclass whose own `__post_init__` runs *after*
`super()` can restore `actions.joint_pos.joint_names = [".*"]` and rebuild the
joint observation terms. Verified by building exactly that in `/tmp` with zero
repo mutation: `BUILT obs=59 act=16 critic=62 mass=2.729035`, then
`v5d CHECKPOINT LOADED OK` and 50 control steps upright at root z ≈ 0.177 **on
the current plant**. Reviving a v5 policy is work, not a wall.

**STRUCK — "v5d's numbers describe a 3.657 kg plant that no longer exists."**
Its *most recent* numbers are the re-gate in
[`m2657_regate.md`](m2657_regate.md) at **2.657067 kg** — 72 g from today's
plant, not 1 kg. On that plant v5d posts a **better** sustained-wrench figure
(43.047 % vs v6d's 70.365 %) and a better obstacle figure (0.885 % vs 7.318 %).
Anyone arguing to revive v5d will cite exactly these two numbers, so they are
printed here rather than omitted.

Those two numbers are **not** evidence v5d is better, because that battery ran
under a materially easier test — see §3.

**The actual disqualifier is DEPLOY-3.** Four of v5d's 59 observation
dimensions are antenna joint positions (indices 22, 23 in `joint_pos_rel` and
38, 39 in `joint_vel_rel`). The antennas are open-loop SG90s with **no position
feedback**. On hardware that observation vector cannot be constructed at all —
not approximated, not degraded: there is no sensor to read. Task M0b removing
the antennas from the interface is precisely what makes a policy buildable on
the real robot, and only the v6 generation has it.

Secondary, and sufficient on its own: v5d commands **6.723 N·m** at
`right_hip_pitch` — **137 % of the STS3250's 4.903 N·m datasheet stall** — because
it was trained against the stale 8.716 N·m ceiling (PLANT-5). It asks for torque
the servo physically cannot produce. See §4.

## 3. Why the favourable v5d wrench/obstacle numbers are not comparable

The m2657 battery is a different and easier test, on four independent axes:

| | v5d @ m2657 | v6d @ rebuild |
|---|---|---|
| wrench yaw torque | **silently discarded** (CFG-1) | applied, 0.05–0.15 N·m |
| actuator ceiling | 8.716 N·m | **4.903 N·m** (1.78× lower) |
| observation latency | none | **0–40 ms** per-env (PLANT-7) |
| obstacle draws / episode | 2, second replaces first (CFG-2) | 1 |
| below-ground resets | **61.7 %** open (PLANT-3) | 0.0 % |

The press in the v6 wrench eval is also marginally stronger (5.35 vs 5.21 N) on
a 72 g heavier robot. A policy scoring 70.365 % on the harder test is not
demonstrably worse than one scoring 43.047 % on the easier one; the two numbers
do not subtract. `rebuild_results.md` §1 carries the full seven-caveat list.

## 4. The cost of this decision, measured

`scripts/measure_joint_torque.py` had only ever been run on v5d. It was run on
**v6d** for this decision, on v6d's own PLAY task, current plant, vx = 0.2,
32 envs × 1500 steps, **all disturbances off** — the friendliest condition that
exists. Full output: [`v6d_torque_measurement.md`](v6d_torque_measurement.md).

| joint | peak N·m | p99 | RMS | % of steps over continuous |
|---|---|---|---|---|
| `left_hip_pitch` | **4.903** | **4.903** | 2.735 | 60.59 % |
| `right_hip_pitch` | **4.903** | **4.903** | 2.591 | 50.91 % |
| `left_ankle` | **4.903** | **4.903** | 2.475 | 40.77 % |
| `right_ankle` | **4.903** | **4.903** | 2.417 | 34.85 % |
| `left_knee` | **4.903** | 3.734 | 2.269 | 62.20 % |
| `right_knee` | 4.052 | 3.926 | 2.298 | 57.58 % |
| **`neck_pitch`** | 4.710 | 4.523 | **3.241** | **94.70 %** |

STS3250: **4.903 N·m** peak stall, **1.569 N·m continuous** (the thermal limit).

Three facts follow, none of which were recorded anywhere before this document:

1. **Actuator saturation is the operating point, not an outlier.** Four leg
   joints reach exactly 4.903 N·m — the clip ceiling — and their **p99 is also
   4.903**. The policy spends a large minority of every second asking for full
   stall torque.
2. **Sustained leg torque is 174 % of the thermal rating.** `left_hip_pitch`
   RMS 2.735 N·m ÷ 1.569 = 1.74. This is on the *unloaded* walk; every contact
   gate is worse.
3. **The most stressed joint is not a leg — it is `neck_pitch`, at 207 % of
   continuous for 94.70 % of steps.** It is in the `head` actuator group and
   carries the same 4.903 N·m STS3250 limit (`robot_cfg.py:118-131`). This is
   the head assembly held against gravity, i.e. a **static** load, so it will
   affect *any* policy on this plant. It is a mechanical-design finding, not a
   policy finding. Filed as **SERVO-1**.

**This does not change the choice, because it is not differential.** v5d is
worse on the same axis — it commands 137 % of stall, torque the hardware cannot
deliver, whereas v6d was trained against the true 4.903 N·m ceiling and
therefore never asks for more than the servo can produce. It asks for all of it,
often. The 4.903 ceiling that PLANT-5 introduced is the corrective, and v6d is
the first policy trained under it.

## 5. What "use this policy" does and does not authorise

**Authorised.** v6d is the policy to build the Phase-S runtime against, to
target with `deployment_contract.json`, and to run the bring-up ladder with. It
is the only export with a machine-checked contract
(`scripts/verify_deployment_contract.py`, 10 checks, exit 0), the only one whose
MJCF sha256 and USD asset hash still match the files on disk, and the only
generation that loads against the registered tasks.

**Not authorised.** Sustained unattended hardware operation. Two blockers:

- **Thermal.** §4: legs at 174 % and neck at 207 % of continuous. Task **S.8**
  (servo torque/current/thermal envelope) must resolve this before any duty
  cycle longer than a bring-up run.
- **No get-up.** The press video shows recovery from a deep *lean* — the trunk
  pitches near the 60° termination at ≈7 s and returns upright within about a
  second ([`eval_results_rebuild/videos/AUDIT.md`](eval_results_rebuild/videos/AUDIT.md)).
  Get-up **from the floor** has never been trained or tested by any policy in
  this project. A completed fall ends the run until a human rights the robot.

Also standing: simulation only, one seed per policy, and the wrench eval is a
single point (`force_frac 0.2`, `active_frac 0.9`) on an unmeasured curve.

## Reproduce

```bash
# the gate table and its verdict
python3 scripts/regate_report.py --results_dir docs/jetson-mod/eval_results_rebuild \
    --candidate v6d_contact_wrench --control v6_robust --markdown

# the deployment contract
~/IsaacLab/_isaac_sim/python.sh scripts/verify_deployment_contract.py \
    --policy_dir exported_policies/v6d_contact_wrench_ppo

# the torque envelope in §4 (needs a free GPU, ~4 min)
cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/measure_joint_torque.py \
    --task Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0 \
    --checkpoint <repo>/exported_policies/v6d_contact_wrench_ppo/model_5998.pt --headless
```
