# v7_servo_safe — deployment archive

The servo-protected locomotion policy. Supersedes `v6d_contact_wrench_ppo`
(2026-08-16). Produced by the SERVO-1 / SERVO-2 fix campaign — see
[`servo_fix_results.md`](../../docs/jetson-mod/servo_fix_results.md).

| Field | Value |
|---|---|
| Checkpoint | `model_8997.pt` |
| MD5 | `2e7dda7a771768052d856612843b100b` |
| ONNX MD5 | `8ea61dcf9034192233675f6d6dbd8f14` |
| Training log | `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v7/2026-08-15_21-46-17_v7_servo_safe/` |
| Train task | `Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0` |
| Play task | `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0` |
| Actor obs | **53**-dim · action **14**-dim |
| Eval JSON | `docs/jetson-mod/eval_results_rebuild/v7_servo_safe.json` |
| Video audit | `docs/jetson-mod/eval_results_v7/videos/AUDIT.md` |
| **Plant** | **2.729035 kg**, USD `767f2415d1b3a056d95e9c310434dbbc` |

Provenance: fine-tuned from `v6d_contact_wrench` (`model_5998.pt`) for 3,000
iterations with two reward changes — the head joints added to
`joint_pos_limits` (SERVO-1) and a **pre-clip** torque penalty enabled at
weight −1e-2 (SERVO-2). The plant is byte-identical to v6d's, so v6d's eval
JSONs remain a valid control.

## Why this policy exists

`v6d` drove `neck_pitch` onto its lower mechanical stop for **100 %** of
forward-walking steps — a servo stalled against a hard stop at 207 % of its
continuous thermal rating — and ran its worst leg joint at 174 % of rated,
because nothing in the reward function priced torque.

## What it measures

Open field, 3,840 episodes: **6/6 gait valid**,
**0.000 % falls**, ref RMS 5.068°.

| | v6d | **v7** |
|---|---|---|
| `neck_pitch` % of steps on its end stop | 100.0 % | **0.0 %** |
| `neck_pitch` torque RMS | 3.164 N·m (207 % of rated) | **0.445 (28 %)** |
| worst leg RMS | 2.735 (174 %) | **2.060 (131 %)** |
| mechanical power | 19.59 W | **12.40 W** |
| jerk | 0.0749 | **0.0407** |
| sustained-wrench falls | 70.365 % | **62.969 %** |
| obstacle falls | 7.318 % | **6.562 %** |

## ⚠ Known limits — read before deploying

- **`deployment_contract.json` is REQUIRED.** `action_scale` and `q_default` are
  not in the ONNX graph (`known_issues.md` DEPLOY-1); commanding the graph
  output directly is wrong by a 4× gain and a 1.378 rad offset.
- **SERVO-2 is mitigated, not closed.** Worst leg RMS is 2.060 N·m = 131 % of
  the 1.569 N·m rated torque. The ≤1.0 N·m thermal target is **not reachable by
  reward weight** — a second iteration at −4e-2 reached 1.587 N·m but broke head
  liveliness and push recovery. The remainder is a mass/gearing problem for Task
  S.8.
- **Get-up from a completed fall is untrained and untested.**

