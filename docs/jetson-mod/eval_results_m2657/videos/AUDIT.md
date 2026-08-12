# Video audit — v5d on the PLANT-1-corrected plant (gate G-R3)

**Audited 2026-08-12** by reading the filmstrips frame by frame, as Task R1
step 7 requires. This gate exists because `amp_command7` (Run 12) passed every
aggregate metric while crawling and only the video caught it. **Video verdict
outranks metrics.**

Checkpoint `exported_policies/v5d_contact_wrench_ppo/model_5998.pt`
(md5 `0333e68a4cd9ed3817310ed80f6715e4`), plant 2.657067 kg, obs/action 59/16.

Each clip is 1000 control steps (20 s) at `--num_envs 2`.

| file | what it is |
|---|---|
| `*_strip.png` | the whole clip at 1.5 fps, 6x4 — overview |
| `*_gait.png` | **the audit evidence**: 8 fps over 2.5 s, centre-cropped and upscaled, so swing clearance is actually legible. The 1.5 fps overview is too coarse to separate walking from shuffling and must not be used to judge the gait. |

## Checklist (from `AGENTS.md`)

trunk upright near 0.17 m · both feet alternate swing with real ground
clearance · feet loaded during stance (no drag / glide / crawl) · heading
straight · no action dither · turn-in-place is a stepping rotation, not a
pivot-scrape.

## Per-condition verdicts

### `forward_vx02` — **PASS** (AGENTS.md-mandated condition 1)
Clear alternating bipedal gait over ~3 cycles. Trunk vertical in every frame,
no forward pitch and no tipping. Swing foot visibly clears the floor; stance
foot is planted flat and loaded. Deep-knee duck posture throughout, which is
the nominal pose (`left_knee` 1.368 rad) and matches the reference motion — it
is not a crouch-crawl. No dither. Aggregate for this condition: gait-valid,
0.00 % falls, duty 64.2 / 64.2 %.

### `turn_wz03` — **PASS** (AGENTS.md-mandated condition 2)
Body yaw advances steadily across the strip while the trunk stays upright. The
feet are **picked up and replaced** — a stepping rotation, not a pivot-scrape.
No dragging foot. Aggregate: gait-valid, 0.00 % falls, duty 67.9 / 62.8 %.

### `turn_wz05` — **PASS**
Same stepping rotation at the faster yaw rate. Trunk upright, alternating swing
with clearance, no scrape. Aggregate: gait-valid, 0.00 % falls,
duty 66.1 / 65.2 %.

### `press_wrench` — **FAILS, and the failure is clean and informative**
The robot walks upright for roughly the first 12 s under the sustained press,
then topples and **stays down for the remainder of the clip — it does not
self-right.** This is a disturbance-rejection failure, not a gait failure: the
gait is valid 6/6 and the open-field fall rate is 0.000 %. It is consistent
with the measured 43.047 % fall rate on this battery.

**Carry this into Phase S.** v5d has no get-up behaviour. Anything that knocks
the robot over ends the run until a human rights it, so the bring-up ladder
must assume that.

### `obstacle_graze` — one adverse draw, do not read a rate off it
The robot walks, contacts the obstacle slab, goes down, and remains down.
**This is a single 20 s clip of 2 envs and the aggregate over 3,840 episodes is
0.885 %**, so the clip illustrates the failure mode, not its frequency.

One caveat worth recording rather than resolving here: `known_issues.md`
**CFG-2** is still open — the obstacle is placed **twice per episode with two
independent draws** — so a clip in which the obstacle appears to arrive on top
of the robot is exactly what that defect would look like. Phase M Task M0
step 7 fixes it. Treat obstacle numbers from this campaign as provisional in
both directions; the defect is present on both sides of the R1-vs-R1b
comparison, so the *comparison* is still controlled.

## Gate result

**G-R3: PASS.** The bar is a PASS in at least two conditions, and AGENTS.md
mandates that forward `vx=0.2` and turn `wz=0.3` be among them. Both pass, and
`turn_wz05` passes as a third.

The two failing clips are both **disturbance** batteries, not locomotion
conditions, and they are scored by G-R4 against the same-plant control rather
than by this gate.
