# Video audit — `v7_servo_safe` (gate G-R3)

**Audited 2026-08-16** by reading the filmstrips frame by frame. This gate exists
because `amp_command7` (journal Run 12) passed every aggregate metric while
crawling, and only the video caught it. **The video verdict outranks the
metrics.**

Checkpoint `model_8997.pt`, run `2026-08-15_21-46-17_v7_servo_safe`, plant
2.729035 kg, obs/action 53/14 — identical to v6d's plant.

All five conditions were examined, including the AGENTS.md-mandated `turn_wz03`,
which the pipeline does **not** render (it renders `turn_wz05`); it was produced
by an explicit extra step, a gap the plan's adversarial review caught.

## Per-condition verdicts

### `forward_vx02` — **PASS**
Clear alternating bipedal gait, trunk vertical in all 20 frames, visible swing
clearance, stance foot planted flat. **The head now sits level** rather than
pitched down — the SERVO-1 fix is visible in the posture, since `neck_pitch`
rests at −10.99° instead of jammed at the −20° stop.

### `turn_wz03` — **PASS** (AGENTS.md-mandated)
Body yaw advances steadily; trunk upright throughout; feet are picked up and
replaced — a stepping rotation, not a pivot-scrape.

### `turn_wz05` — **PASS**
Same stepping rotation at the faster yaw rate, trunk upright, alternating swing
with clearance, no scrape.

### `press_wrench` — **PASS, and better than v6d**
The robot stays **upright across all 24 frames** of the 16 s overview under the
sustained press — trunk vertical, stepping throughout, no topple. v6d's audit
recorded a deep pitch to near the 60° termination at ~7 s with a recovery; v7
does not reach that state in this draw. Consistent with the measured improvement
**70.365 % → 62.969 %**.

As always: one draw of 2 envs. The clip shows the behaviour, not the rate.

### `obstacle_graze` — **PASS**
Walks alongside the slab, contacts it, continues upright past it through all 24
frames. Consistent with 6.562 % aggregate (better than v6d's 7.318 %).

## What this audit does NOT establish

- **Get-up from a completed fall is still untested and untrained.** No policy in
  this project has ever been trained for it. A completed fall still ends the run
  until a human intervenes.
- The clips are 2 envs each; every rate quoted here comes from the 3,840-episode
  battery, not the video.
