# STS3250 bench characterisation — measured constants for future extrapolation

**Measured 2026-08-21.** One servo, ID 1, 11.1 V on the split harness, ambient
26–27 °C. Raw logs: [`staircase.json`](staircase.json), [`sweep_test.json`](sweep_test.json),
[`hold_test.json`](hold_test.json), [`sts3250_eeprom_dump.json`](sts3250_eeprom_dump.json).

The purpose of this file is **extrapolation**. The bench has no lever arm, no
weights and no thermometer, so loaded torque cannot be applied. What it *can*
produce is a set of measured constants and a duty curve, so that assumptions we
must still make are anchored to data rather than invented.

---

## 1. Measured constants — use these instead of guessing

| quantity | measured | how | confidence |
|---|---|---|---|
| **idle current** | **21 mA** @ 11.1 V (0.232 W) | PSU panel, 75 s steady | high — datasheet 5-6 says 24 mA |
| **idle case temperature** | **33–34 °C** at 26–27 °C ambient | addr 63, flat over 75 s | high |
| **R_th (board path, idle)** | **≈ 32 K/W** | ΔT 7.5 K ÷ 0.232 W | **low** — see §4 |
| **`goal_speed` unit** | **counts/s** | commanded 200, measured 200.5 counts/s over 9.2 s | **high** — direct |
| **running friction** | **≈ 40 ‰ of stall ≈ 0.196 N·m** | addr 60, constant velocity, unloaded | medium — includes gearbox drag |
| **`addr 60` load unit** | **per-mille of stall (0–1000)** | pins at exactly `torque_limit`=200 | **high** — see §2 |
| **tracking error, gentle** | mean 3.1°, max 10.9° | ±17.6°, 4 s sinusoid, speed 300 | high |
| **voltage sag under load** | **0.3–0.4 V** (11.2 → 10.8 V) | PSU panel + addr 62 | medium — harness + supply |
| **peak power, unloaded motion** | **0.454 W** (41 mA) | PSU panel, gentle sweep | high |

## 2. 🚨 `addr 60` reports the LIMITER, not delivered torque

**The single most important finding for Stage C.**

With `torque_limit` (addr 48) set to **200**, the load register pinned at exactly
**200** for 78–100 % of samples from level 3 onward. It never exceeded the limit,
and it sat on it precisely.

**So when the torque cap binds, `addr 60` is reporting the clamp, not a
measurement.** Any torque figure read from it under those conditions is
circular — you get back the number you set.

> **Stage C must run with `torque_limit = 1000` (full scale), or every load
> reading in the thermal run is void.** The 20 % cap is correct for exploratory
> motion and wrong for measurement.

## 3. The duty staircase — the servo was torque-limited, never thermally limited

Seven levels, 45 s each, centred at 2048, `torque_limit` 200 (0.98 N·m):

| lvl | swing | period | speed | at cap | tracking err (mean) | temp |
|---|---|---|---|---|---|---|
| 1 | ±17.6° | 4.0 s | 300 | 0 % | 81 counts | 34 → 34 |
| 2 | ±35° | 3.0 s | 500 | 0 % | 259 | 34 → 34 |
| 3 | ±53° | 2.0 s | 1000 | **78 %** | 450 | 34 → 34 |
| 4 | ±70° | 1.5 s | 1500 | **100 %** | 687 | 35 → 35 |
| 5 | ±88° | 1.0 s | 2000 | **100 %** | 822 | 35 → 36 |
| 6 | ±105° | 0.8 s | 3000 | **99 %** | 927 | 36 → 36 |
| 7 | ±132° | 0.6 s | 4000 | 76 % | *12 — artifact* | 36 → 37 |

**Total temperature rise: 34 → 37 °C over 5.25 minutes at near-continuous torque
saturation.** Status byte stayed 0 throughout; nothing tripped.

**Level 7's tiny tracking error is a measurement artifact, not good behaviour.**
At a 0.25 s command interval against a 0.6 s period, only ~2.4 commands are
issued per cycle — the servo receives reachable step targets rather than a smooth
sinusoid, and appears to track them. Do not read level 7 as "it got better".

**Two glitched samples** appeared across 518: one 39 °C at level 2 and one 37 °C
at level 3, each surrounded by 34 °C neighbours. This is why aborts now require
3 consecutive faults.

## 4. What these numbers do and do not support

### Safe to extrapolate

- **Idle and low-duty power.** 21 mA idle and 41 mA under gentle motion are
  direct panel readings and are solid.
- **Friction.** ~40 ‰ at constant velocity, unloaded, is a clean steady-state
  reading and is the best available anchor for **PLANT-6**.
- **`goal_speed` in counts/s** is directly measured and can be relied on.
- **Thermal insensitivity at low load.** 3 K over 5 minutes of aggressive
  unloaded motion says the servo is nowhere near stressed without an external
  load. Useful as a **lower bound**.

### NOT safe to extrapolate

- **R_th ≈ 32 K/W must not be applied to motor heating.** At idle the dissipation
  is control electronics, and `addr 63` is a board sensor. The winding-to-sensor
  path is different and unmeasured. Using 32 K/W to predict the temperature at
  4.4 W would give ~140 K of rise, which is almost certainly wrong in a way that
  would kill the project's thermal conclusions if taken seriously.
- **Nothing about sustained loaded torque.** The staircase saturated the torque
  *command* but the horn is unloaded, so delivered torque and dissipation stayed
  tiny. Saturation of the limiter is not thermal loading.
- **The current LSB.** Still unresolved — see §5.

## 5. Why the current LSB is still unresolved

`addr 69` reached 38 raw counts at peak. Reconciling that against the load
register gives implied LSBs of 24–85 mA, matching **none** of the candidates
(6.5 / 10 / 12.26 mA). Two reasons, both structural:

1. **Load was pinned at the limiter** (§2), so it is not a torque measurement and
   cannot calibrate current.
2. **4 Hz sampling of a dynamic system aliases.** Load and current are read in
   separate transactions milliseconds apart, on a signal changing far faster.

**The resolution requires a steady state with a known torque** — i.e. a lever arm
and a weight, held still, with `torque_limit` at 1000. Until then, **treat every
amp figure derived from `addr 69` or `addr 28` as unverified**, which also leaves
`addr 28 = 310` un-convertible (2.02 / 3.10 / 3.80 A depending on LSB).

## 6. Consequences for Stage C

1. **`torque_limit = 1000`.** Otherwise `addr 60` is circular.
2. **Sample faster than 4 Hz, or measure only at steady state.** Static holds are
   fine at 1 Hz; anything dynamic needs a faster loop or it aliases.
3. **Budget for 0.3–0.4 V of harness sag** and log `V_PSU − addr62` as the error
   channel, per `bench_test_servo.md`.
4. **Aborts need 3 consecutive faults.** Two glitched samples in 518 would
   otherwise have stopped the run twice.
5. **Expect nothing thermal without a load.** 3 K in 5 minutes unloaded means the
   thermal question is entirely about applied torque, and the fixture is
   therefore the whole remaining blocker.

## 7. What one servo can never establish

The datasheet prints **±10 %** on stall torque, stall current and no-load speed,
and is internally inconsistent by 8 % on Kt and 2.4× on terminal resistance. This
is a **single unit**. Any envelope derived here carries roughly **±15 %** before
rig error, and `addr 13` / `addr 15` in particular are per-unit EEPROM values —
see [HW-2](../known_issues.md#hw-2). **Dump those registers from all 14 servos at
build time** rather than assuming this one is representative.
