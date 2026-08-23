# Video audit — `v7_servo_safe` on the CORRECTED plant (gate G-R3)

**Audited 2026-08-23 by the owner**, watching the clips directly. This gate
exists because `amp_command7` (journal Run 12) passed every aggregate metric
while crawling, and only the video caught it. **The video verdict outranks the
metrics.**

Checkpoint `model_8997.pt`, plant **PLANT-11 / PLANT-6 corrected**
(`armature` 0.00843, `dynamic_friction` 0.200), 2.729035 kg, obs/action 53/14.
Same recipe as the 2026-08-16 audit — 2 envs, `--video_length 1000` (20 s) — so
the comparison against the old-plant clips is like-for-like.

## Verdict: **PASS**, with one finding

> **Owner, 2026-08-23:** *"Other than the head tilt issue, everything else looks
> good."*

| condition | verdict |
|---|---|
| `forward_vx02` | **PASS** |
| `turn_wz03` (AGENTS.md-mandated) | **PASS** |
| `turn_wz05` | **PASS** |
| `press_wrench` | **PASS** |
| `obstacle_graze` | **PASS** |

Old-plant counterparts for comparison:
[`../../eval_results_v7/videos/`](../../eval_results_v7/videos/).

## The finding: POSE-1

The audit surfaced **[POSE-1](../../known_issues.md#pose-1)** — the head is held
**rolled ~17° and pitched down ~11°** for the entire gait, and the camera is
mounted in the head.

**This is the whole justification for keeping G-R3 as a human gate.** Every
aggregate metric passed. §6b checks neck *travel* and *on-stop %* — whether the
head can move, never whether it is **level** — so a 17° permanent lean satisfies
every scored bar. It took a person watching a video.

Owner raised it to **HIGH** and accepted a retrain: *"the head tilt makes the
robot not a good candidate for sim-to-real for now."* Tracked as **S.13**.

## What this audit does NOT establish

- **Get-up from a completed fall is still untested and untrained.** No policy in
  this project has ever been trained for it. A completed fall ends the run until
  a human intervenes. (Unchanged from the 2026-08-16 audit.)
- Each clip is **2 envs, one draw**. It shows the behaviour, not the rate. Every
  percentage in [`../../plantfix_regate.md`](../../plantfix_regate.md) comes from
  the 3,840-episode battery, not from these videos.
