# Bench test — measure what the STS3250 datasheet does not publish

**Written 2026-08-16.** This task does not exist in `task_plan_v2.md`: **no Phase-S
task measures real servo current or temperature.** That is a gap, because every
thermal number in this project is currently an assumption.

## Why this test exists

Feetech publishes torque and current. They publish **no thermal time constant,
no duty-cycle curve, and no gear material**. So the ≤1.0 N·m target in
[`servo_torque_budget.md`](servo_torque_budget.md) — the one `v7_servo_safe`
missed, and which a second retrain could not reach without breaking other bars —
rests on a *derating calculation over assumed thermal behaviour*.

**The single question this test answers:**

> The shipped policy commands **2.060 N·m RMS** at its worst leg joint.
> Held on a real servo, does it stabilise below the firmware's 70 °C torque-off,
> or does it climb until the joint goes limp?

If it stabilises, **SERVO-2 can be closed and no further retraining is needed.**
If it doesn't, we learn the real sustainable torque and can aim at it instead of
at a derived guess.

## Parts

| item | role |
|---|---|
| Feetech STS3250 (12 V) | the device under test |
| FE-URT-1 | USB ↔ servo TTL bus |
| NICE-POWER 30 V/10 A bench supply | power **and** the current reference |
| banana→alligator/fork leads | PSU to servo |
| multimeter | independent current cross-check |
| ~200 mm rigid arm + weights | applies known torque |

## Pinout — go by COLOUR, never by position

Feetech's [STS3215 datasheet][ds] (same 5264-3P connector as the STS3250) defines:

| pin | function | wire |
|---|---|---|
| 1 | GND | **black** |
| 2 | Vcc | **red** |
| 3 | Signal (TTL) | **white** |

"Left to right" flips when you turn the connector over. **Always identify by wire
colour.** Reversed polarity destroys the servo — Feetech's own docs say so.

[ds]: https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf

## Wiring — the servo is powered from the PSU, NOT through the adapter

Two independent reasons, both verified rather than assumed:

1. The FE-URT-1's external-power screw terminal is **printed `DC6V-9V`**
   ([Feetech][urt], [MakerBotics][mb]). Our pack is **11.1 V** — 23 % over the
   printed rating.
2. The board is USB-powered with **500 mA over-current protection**, and the
   servo stalls at **4.2 A**. Even at a legal voltage the current alone is 8×.

A board that fails shorted can put 11.1 V onto the host's USB 5 V rail. The cost
of avoiding that is one connector operation, so we always avoid it.

[urt]: https://www.feetechrc.com/FE-URT1-C001.html
[mb]: https://makerbotics.com/product/mb-elc-servo-controller-urt1/

```
   PSU (+) ──────────────────►  servo VCC     (red)
   PSU (−) ──┬───────────────►  servo GND     (black)
             └───────────────►  adapter GND      ← common ground, REQUIRED
   adapter signal ──────────►  servo Signal  (white)
   USB ─────────────────────►  PC
```

### The cable is female on BOTH ends — that is correct, and it is the catch

The stock Feetech 3-pin cable mates a servo socket to the adapter's header, so
both ends are female housings. There is no bare end to land in the PSU's binding
posts, and the servo's two sockets are **internally paralleled** — so any cable
reaching the adapter also carries the 11.1 V rail to it. Plugging the cable in
whole is exactly the failure mode above.

The fix is a standard, fully reversible connector operation: **extract the red
and black contacts from the ADAPTER end of the cable.** Lift the small latch in
the housing window with a sewing needle and the crimped contact slides straight
out; pushing it back in re-latches it. That leaves:

- **white only** in the housing → adapter (signal, USB-powered, no rail)
- **red and black** dangling as bare crimp tabs → alligator clips from the PSU

Ground the adapter separately (PSU − to its terminal GND pin, which is the one
nearest the 3-pin sockets). Signal return is milliamps; the servo's 4.2 A never
touches the board.

*If you would rather not touch the connector:* male-to-female Dupont jumpers let
you build the same split harness with no disassembly. Order them now — Step 0a
below does not need them, so nothing is blocked while they ship.

## Safety

- The servo **will get hot enough to burn** — 70 °C is the firmware cutoff, and
  the case reaches it. Do not hold it; use a non-flammable surface.
- Weights fall. Keep the arm low over a padded surface, and nothing fragile (or
  a foot) underneath.
- Set the PSU current limit **before** every step, never after.

---

## Step 0a — power only. No adapter, no USB, no PC. (10 min)

Deliberately the smallest possible blast radius: if something is wrong with the
supply, the polarity, or the servo, this step finds it with **nothing else
connected that can be damaged**. It also needs no connector surgery — plug one
cable end into the servo and use the free end as a breakout.

**Set the supply BEFORE anything is connected:**

1. Output **OFF**. Nothing in the binding posts.
2. Set voltage to **11.1 V** — the robot's real 3S2P pack voltage, *not* 12 V.
   (Momentarily enabling the output with no load to read the display is fine;
   switch it back off.)
3. Turn the **current knob fully counter-clockwise** (to zero).

**Then connect:**

4. Insert a male pin into the free connector end's **red** socket and another
   into its **black** socket — a male Dupont jumper, a stripped 22 AWG solid
   wire, or a straightened paperclip all work. Keep them splayed apart; they
   must not touch each other.
5. Alligator-clip **PSU (+) → red** and **PSU (−) → black**. Check the colours
   twice. Leave **white unconnected** — never put the supply on the signal wire.

**Then ramp:**

6. Output **ON**. Because current is dialled to zero, the supply starts in
   constant-current mode and the voltage display will read near 0 V.
7. Turn the current knob **slowly clockwise** until the voltage display rises to
   and holds **11.1 V**. Read the current at that point — that is idle draw.

This ramp is self-protecting: a short or reversed connection means the voltage
*never* climbs no matter how far you turn. That is the whole point of doing it
this way rather than presetting a limit.

| observation | meaning |
|---|---|
| Voltage reaches 11.1 V at **~0.05–0.1 A** | Healthy. Proceed. |
| Voltage stays low, current climbs past ~0.5 A | Short or reversed polarity. **Output OFF immediately.** |
| Voltage reaches 11.1 V at **0.00 A** | No connection — pins not seated. |
| Any smell, heat, or noise | **Output OFF.** Stop and report. |

8. Leave it powered ~60 s. The case must stay at room temperature — an idle
   servo does no work and must not warm up at all.
9. Output **OFF** before touching anything.

**Record:** idle current, the voltage the supply actually held, and case
temperature after 60 s.

> The servo will not move and will make no sound. That is a **pass**, not a
> failure — it holds no position until it is commanded over the bus in Step 0b.

## Step 0b — add the adapter (after 0a passes)

Only now does the FE-URT-1 join the circuit, using the split harness from the
wiring section above. Re-verify with the multimeter, **before plugging in USB**,
that there is **no continuity between the 11.1 V rail and the adapter's Vcc pin**.
Then raise the current limit to **5.0 A** for the loaded steps that follow.

## Step 1 — talk to the servo (10 min)

Use Feetech's `FD.exe` debug software, or the Python SDK
(`pip install feetech-servo-sdk`, or Feetech's `STservo_sdk`). Baud **1000000**
for STS series.

1. Scan for the servo ID (default is usually 1).
2. Read back: present **position, voltage, current, temperature**.
3. Confirm reported voltage matches the PSU display within ~0.2 V.

**Record:** servo ID, firmware version, reported vs actual voltage.

> If reported voltage is wrong, **every later reading from this servo is suspect**
> — that is what this check is for.

## Step 2 — is the servo's self-reporting honest? (15 min)

The whole test relies on the servo's own current and temperature telemetry.
Verify it once against instruments.

1. Put the **multimeter in series** on the positive lead (10 A jack).
2. Command the servo to hold position with no load; log current from **all three**
   sources: multimeter, PSU display, servo telemetry.
3. Repeat holding a 0.5 kg weight at 200 mm (≈0.98 N·m).

**Record:** a 3-column table. **Acceptance: the three agree within ~10 %.**
If the servo's self-report is off, use the PSU/meter figure for everything after
and note the offset.

## Step 3 — verify the torque constant (20 min)

The datasheet claims **Kt = 11 kg·cm/A**. Everything in
`servo_torque_budget.md` that converts current to torque depends on it.

Mount the arm horizontally and hang each weight in turn. Command the servo to
**hold horizontal** (worst case: full gravitational torque, zero motion).

| torque | what it is | @100 mm | @150 mm | @200 mm |
|---|---|---|---|---|
| 0.981 N·m | Feetech's 100k-cycle life load (1/5 stall) | 1.00 kg | 0.67 kg | 0.50 kg |
| 1.569 N·m | **Feetech RATED torque** | 1.60 kg | 1.07 kg | 0.80 kg |
| 2.060 N·m | **v7 worst-leg RMS — what we ship** | 2.10 kg | 1.40 kg | 1.05 kg |
| 2.735 N·m | v6d worst-leg RMS (before the fix) | 2.79 kg | 1.86 kg | 1.39 kg |

*1 L of water = 1.00 kg. A bottle on a string is a fine weight.*

For each: hold ~10 s, record steady current. **Plot torque vs current — the slope
is the real Kt.** Compare with 11 kg·cm/A (= 1.079 N·m/A).

**Record:** torque/current pairs, fitted Kt, deviation from datasheet.

## Step 4 — THE THERMAL TEST (the reason for all of this)

For each torque level below, from a **cold start** (case at room temperature —
allow ~15 min between runs):

1. Set the load, command hold-horizontal.
2. Log **temperature, current, voltage every 1 s**.
3. Stop at **70 °C**, or at **20 minutes**, whichever first.
4. Note the ambient temperature.

Run in this order, and **stop the sequence at the first level that reaches 70 °C**:

| # | torque | question it answers |
|---|---|---|
| 1 | **0.981 N·m** | is Feetech's own life-rating load thermally free? |
| 2 | **1.569 N·m** | is the nameplate actually sustainable indefinitely? |
| 3 | **2.060 N·m** | **can the shipped policy's worst joint run forever?** |
| 4 | 2.735 N·m | how bad was it before the fix? |

**Record per level:** time-to-70 °C (or "stable at X °C"), the steady-state
temperature if it plateaus, and the temperature curve.

**The thermal time constant** is the time to reach ~63 % of the final rise. That
single number replaces the biggest assumption in `servo_torque_budget.md`.

## Step 5 — replay the real gait profile (30 min, optional but best evidence)

Steps 3–4 are *static holds*, which are the worst case. Walking is not static.

`v7_torque.npz` holds the shipped policy's **actual per-step torque, 1500 steps
at 50 Hz**, for every joint. Replay the `left_hip_pitch` trace as a position
command on the bench servo, loop it for 10 minutes, and log temperature.

This is the closest thing to "what the robot will actually do to this servo".
If temperature plateaus safely here even where Step 4 says the static hold does
not, that is a meaningful and favourable result — and it is the number Phase S
should design the duty cycle around.

---

## What the outcomes mean

| result | conclusion | action |
|---|---|---|
| 2.060 N·m plateaus well under 70 °C | **SERVO-2 closed.** The derated ≤1.0 target was too conservative. | Ship v7 as-is. Retire the SERVO-2 check. |
| 2.060 N·m reaches 70 °C but only after many minutes | Duty-cycle limited, not blocked | Set a run-time budget in the S.10 safety layer |
| 2.060 N·m reaches 70 °C in ~1–2 min | **SERVO-2 is real and v7 is not enough** | Mass reduction or gearing (S.8), not another retrain — two retrains already showed weight alone cannot get there |
| measured Kt differs a lot from 11 kg·cm/A | every current↔torque conversion in the project shifts | Re-derive `servo_torque_budget.md` §3 |

## Write the results back into

- `docs/jetson-mod/bench_results_servo.md` — raw logs and plots
- `servo_torque_budget.md` — replace the derating assumptions with measurements
- `known_issues.md` **SERVO-2** — close it, or restate its real target
- `task_plan_v2.md` **S.8** — its "UNVERIFIED" thermal thresholds become cited

## Honest limits of this test

- **One servo, one sample.** Feetech prints ±10 % on stall torque; unit spread is
  real. This measures *your* servo.
- **Open air, not the chassis.** The robot seals its servos in a printed body
  with a Jetson inside. Real in-robot temperatures will be **worse**. Treat these
  numbers as the optimistic bound and re-measure after assembly.
- **The 70 °C cutoff is the firmware's, not a damage threshold.** Gears and
  bearings may degrade below it over many cycles; this test says nothing about
  100k-cycle wear.
