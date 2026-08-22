# S.10 — the safety layer

**Written 2026-08-22.** The software half. The physical kill switch is a human
task and is **not** done.

`jetson_runtime/safety.py` is called once per control step and returns one of
`OK`, `DERATE`, `HOLD`, `TORQUE_OFF` plus reasons.

## Every threshold comes from the contract

There is **no threshold literal in `safety.py`**. That is not style — the
thresholds moved once already. Every document in this repo said the firmware cut
out at **70 °C** until the register was read off the servo and said **80**
([HW-2](../known_issues.md#hw-2)). Hard-coding is how that recurs.

| rule | threshold | source |
|---|---|---|
| tilt → `TORQUE_OFF` | **60°** | the sim's own `bad_orientation` (`env_cfg.py:592`, `:701`) |
| temp warn → `DERATE` | 55 °C | contract |
| temp stop → `TORQUE_OFF` | 65 °C | contract |
| over-current → `TORQUE_OFF` | 3.80 A held **1.0 s** | contract; window is **half** the firmware's 2.0 s |
| current derate | 70 % of limit, 1.0 s | contract |
| **under-voltage → `TORQUE_OFF`** | **6.0 V** | **measured `addr 15`**, not the datasheet's 4 V |
| loop overrun | >2× 20 ms → `HOLD`; ×3 → `TORQUE_OFF` | contract `control_dt_s` |
| sensor fault / NaN | `HOLD` ≤3 steps, then `TORQUE_OFF` | — |

**65 °C now sits 15 K below the real cutout rather than 5 K.** More conservative
than intended, and deliberately not relaxed: `addr 63` is a **board** sensor and
the winding runs far hotter than it reports, so that margin buys real protection.

## The fall proxy

The sim's other termination — `root_height_below_minimum` at 0.09 m
(`env_cfg.py:595`) — **cannot be evaluated on hardware**, because nothing on the
robot measures trunk height. The substitute:

> both foot switches open **AND** `|gravity_z| < 0.8` (past ≈37°), for **5
> consecutive steps** → `TORQUE_OFF`.

One foot down is not a fall. A momentary flight phase is not a fall. Both
conditions must hold together, and persist.

## Decisions recorded, not defaulted

**`wz` clamps to ±0.3 while the trained hull is ±0.5.** DEPLOY-5 confirms ±0.3 is
correctly *inside* the hull. `vx`/`vy` clamp at the hull itself. Widening `wz` to
the full ±0.5 is **a decision for the owner to record**, not a default to drift
into. Every clip is counted.

**On a fall, torque OFF — do not hold the pose.** A fallen robot holding a
standing pose stalls its servos against the floor, tripping the stall and
overcurrent protections and heating the coils. The firmware then drops torque
anyway — silently, unrepeatably, with no log line. Better to release deliberately
and log it.

**`TORQUE_OFF` latches.** Recovery requires an explicit operator `rearm()`. A
monitor that un-trips itself trips repeatedly while the robot destroys itself.

**Torque-off ramps to the MEASURED pose over 2 steps before disabling.**
Commanding the last target while the servo is already elsewhere makes it fight;
ramping to where it actually is makes the disable a release rather than a drop.

## Verification

`jetson_runtime/tools/fault_injection.py` replays a real S.2 trace and injects
each fault in turn. All nine reach the expected verdict within budget:

| case | expect | within | reached at |
|---|---|---|---|
| tilt | `TORQUE_OFF` | 1 | 1 |
| over-temperature | `TORQUE_OFF` | 1 | 1 |
| temp warning | `DERATE` | 1 | 1 |
| over-current | `TORQUE_OFF` | 51 | 50 |
| current derate | `DERATE` | 51 | 50 |
| under-voltage | `TORQUE_OFF` | 1 | 1 |
| fall proxy | `TORQUE_OFF` | 5 | 5 |
| loop overrun | `TORQUE_OFF` | 3 | 3 |
| NaN observation | `TORQUE_OFF` | 4 | 4 |

**Clean trace: 0 false trips.** That half matters as much as the faults — a
monitor that fires on healthy data teaches the operator to ignore it.

Plus 22 unit tests in `tests/test_safety.py`.

## NOT done — human tasks

1. **The physical kill switch**, in series with the **servo power rail**,
   reachable without leaning over the robot, and **not** cutting the Jetson (so
   the logs survive the event).
2. **Foot switches wired** — the fall proxy needs them. GPIO map at
   `AGENTS.md:635`: left = header pin 15, right = pin 13.
3. **Verifying the trips on real hardware.** Everything above is verified in
   replay only.
