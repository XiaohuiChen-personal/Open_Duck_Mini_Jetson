# Bench test — measure what the STS3250 datasheet does not publish

**Written 2026-08-16. Rewritten 2026-08-20** after the hardware arrived and an
adversarial review overturned the wiring plan. This task does not exist in
`task_plan_v2.md`: **no Phase-S task measures real servo current or
temperature.** That is a gap, because every thermal number in this project is
currently an assumption.

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

## Primary source

All servo numbers below are quoted from the **official Feetech STS3250 product
specification, `ST-3250-C001.pdf`, EDITION A/0, 2024-01-16**, cited by its own
line numbers. Where this document previously relied on a third-party table for a
*different model*, those numbers are now marked as superseded.

| line | spec | value |
|---|---|---|
| 1-2 | operating temperature | **−20 … +60 °C** |
| 5-3 | no-load running current | 280 mA |
| 5-4 | stall torque ±10 % | 50 kg·cm (4.903 N·m) |
| 5-5 | stall current ±10 % | 4.2 A |
| 5-6 | **idle current (stopped)** | **24 mA** |
| 5-8 | rated torque | 16 kg·cm (1.569 N·m) |
| 5-9 | rated current | 1400 mA |
| 5-10 | **motor terminal resistance** | **1.2 Ω** |
| 5-11 | **Kt** | **11 kg·cm/A** = 1.0787 N·m/A |
| 7-11 | over-hot protection | torque off above **70 °C** |

> **A real datasheet inconsistency.** 12 V ÷ 1.2 Ω = 10 A, but 5-5 says stall is
> 4.2 A. So **4.2 A is a driver current clamp, not V/R.** Whether the effective
> circuit resistance is 1.2 Ω or ~2.86 Ω changes predicted heat by 2.4×, so the
> bench run must resolve it. Both bounds are carried through every table below.

## Parts

| item | role | status |
|---|---|---|
| Feetech STS3250 (12 V) | the device under test | have |
| **FE-URT-2** | USB-C ↔ servo TTL bus (**not** the URT-1) | have |
| NICE-POWER 30 V/10 A bench supply | power **and** the current reference | have |
| banana→alligator leads | PSU to the wire stubs | have |
| multimeter | independent cross-check | have |
| ~1 m 24 AWG solid wire | fallback only — clips grip the screw heads fine | salvage Cat5e |
| **non-contact IR thermometer** | 70 °C aluminium case is a contact burn | **buy** |
| second 5264-3P cable | split-harness fallback + spare | **buy** |
| ~200 mm rigid arm + weights | applies known torque | build |
| foam pad | the weight *will* drop | have |

## Pinout — go by COLOUR, never by position

Feetech's [STS3215 datasheet][ds] (same 5264-3P connector as the STS3250) defines:

| pin | function | wire |
|---|---|---|
| 1 | GND | **black** |
| 2 | Vcc | **red** |
| 3 | Signal (TTL) | **white** |

"Left to right" flips when you turn the connector over. **Always identify by wire
colour.** Reversed polarity destroys the servo.

[ds]: https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf

## The adapter is an FE-URT-2

An earlier revision planned around an **FE-URT-1**, whose external-power terminal
is printed `DC6V-9V` ([Feetech][urt], [MakerBotics][mb]). The URT-2 is a
different board — effectively **two independent bus drivers on one PCB**, each
with its own pair of daisy-chain sockets and its own power rail:

| | **TTL bus — ours** | RS485 bus |
|---|---|---|
| socket | **3-pin**, smaller | 4-pin, larger |
| silkscreen | **`G V1 S`** | (`V2` rail) |
| power terminal | **`G V1`**, `DC4.8-12V` | `G V2`, `DC12-24V` |
| servo family | SCS / **STS** | SMS |
| default baud | **1,000,000** | 115,200 |

[urt]: https://www.feetechrc.com/FE-URT1-C001.html
[mb]: https://makerbotics.com/product/mb-elc-servo-controller-urt1/

### ⚠️ The rating is NOT the question

**`V1` being rated 4.8–12 V does not license powering the servo through the
board.** A voltage rating says nothing about whether the servo rail and USB VBUS
share a node — and the URT-2 is advertised as able to power a servo *from USB 5 V*
for parameter tuning, which means **a path between those rails exists by design**.

If that path is a plain connection rather than a blocking diode, putting 11.1 V
on `V1` back-feeds **11.1 V into the host PC's USB port**.

No official FE-URT-2 electrical document exists — Feetech's own URT-2 page serves
URT-1 body text, and vendor listings contradict each other. So the question is
settled **by meter, in pre-flight P3 and P5**, not by any datasheet.

### ✅ DECISION: the split harness is the DEFAULT, not the fallback

Adversarial review 2026-08-21 concluded that proving the board safe is not worth
what it costs, because **an air gap cannot fail short and a semiconductor can.**

```
   PSU (+) ──────────────────►  servo Vcc   (red, servo side of the cut)
   PSU (−) ──┬───────────────►  servo GND   (black, fat wire — star point)
             └──[thin]───────►  board socket G   (signal reference only)
   board 3-pin socket ──[black + white only]──►  servo
   USB-C ────────────────────►  PC
   screw terminal V1 ────────►  PERMANENTLY EMPTY
```

**Break the red (Vcc) conductor between servo and board, feed 11.1 V into the
servo side of the break, and leave the `V1` screw terminal empty forever; the
board runs on USB alone.** The bus still works — `S` is a half-duplex signal
referenced to the common ground, and red is only power pass-through.

**Method — cut. (Extraction was tried first and abandoned: on this cable the
5264 retention tabs are not reachable, 2026-08-21.)**

Cut **red and black** mid-cable. Leave **white uncut**. That yields four ends:

| end | goes to |
|---|---|
| **servo-side red** | PSU (+) |
| **servo-side black** | PSU (−) |
| **board-side black** | PSU (−) — *same clip*, this is the star point |
| **board-side red** | **nothing. Insulate it completely.** |

Both black ends land in the **same PSU(−) clip**, which *is* the star ground — no
extra wire needed, and servo return current never crosses the board's ground
plane. White runs servo↔board untouched and carries the signal.

**Board-side red is now the dangerous end.** It connects to the board's `V1` pin,
which measures 4.94 V off USB VBUS. Tape it off so it cannot touch anything.

> **Why not simply cut red and leave black intact?** Then the servo's only return
> is servo → black → board socket `G` → the board's ground plane → `G` screw →
> PSU(−). If that screw ever loosens, the servo's return current hunts for
> another path — and the only one left is the white signal wire into the board's
> transceiver and out through **USB ground into the host**. Cutting black removes
> that failure mode entirely.

### Harness acceptance — run these before every power-on

Meter in resistance mode, unpowered, cable plugged into **both** board and servo.
**Touch the probes together first and confirm 0.0 Ω** — a broken lead reads `OL`,
and `OL` is the pass condition on check 1.

| # | measure | expect | proves |
|---|---|---|---|
| 1 | board socket `V1` pin ↔ servo-side red | **open** | the break works — the entire point |
| 2 | servo-side red ↔ servo Vcc pin | connected | power still reaches the servo |
| 3 | board socket `G` ↔ servo-side black | connected | shared ground reference intact |
| 4 | servo-side red ↔ the black junction | **open** | the supply is not shorted |

**BUILT AND VERIFIED 2026-08-21 — all four pass.**

If check 1 ever shows continuity, the harness is defeated and the host is exposed
again. **Re-run all four after any re-clip, re-tape or re-route.**

> **Why the servo's spare socket does NOT work as a power inlet.** The servo's
> two sockets are internally paralleled, so 11.1 V injected at socket B appears
> on socket A's V pin, runs down the red wire into the board's socket `V1` pin —
> which this file now records as tied to USB VBUS through a 0.06 V drop. That
> route puts 11.1 V on the host. **The red conductor must be broken. There is no
> wiring arrangement that avoids it.**

This single change removes, by construction rather than by inference:

| risk | why it disappears |
|---|---|
| 11.1 V back-feeding the host USB port | the board never sees 11.1 V |
| a load switch that is off cold and on when live | irrelevant — nothing to switch onto |
| the board's *real* part ratings being URT-1's `DC6V-9V` | the rail never reaches the board |
| servo return current through the board's ground plane | star ground |
| a blocking element that fails short later | there is no element, only air |

Cost: one cut conductor, ten minutes. **Everything below about powering through
the board is retained only for the case where someone deliberately chooses that
path, and it is not the recommended one.**

> **If you must power through the board anyway**, the ordering matters more than
> the current limit: see "mate cold, then ramp" in Stage B. And note that
> hot-plugging USB into a live 11.1 V rail is the *more* dangerous ordering, not
> the less — the PSU's output capacitor dumps into the host's VBUS node in
> microseconds, while the constant-current loop needs 100 µs to milliseconds to
> respond. **The current limit is not yet in the circuit when the damage happens.**

```
   powering through the board (NOT recommended):
   PSU (+) ──►  screw terminal  V1     ← only after P3b and P5 pass
   PSU (−) ──►  screw terminal  G      (the one beside V1)
   3-pin socket  ──[stock cable]──►  servo
   USB-C ────────────────────────►  PC
```

### Two hazards created by there being four sockets

1. **A 3-pin housing can be forced into the 4-pin socket, offset by one
   position.** It feels like it fits, and it lands 11.1 V on the wrong pin.
   Count pins; if a socket needs force, it is the wrong socket.
2. **`V2` is the 12–24 V rail.** Tape over both its screws for the whole campaign.

### Bench-verified wiring (2026-08-20)

Vendor pages disagreed, so the wiring was proven by meter, unpowered, cable in a
3-pin socket, far end free:

| # | from | to | result | establishes |
|---|---|---|---|---|
| 1 | `V1` terminal | cable **red** | continuity | `V1` feeds the socket; red is Vcc |
| 2 | `G` beside `V1` | cable **black** | continuity | ground path; black is GND |
| 3 | `V1` terminal | cable **white** | open | signal isolated from power |
| 4 | `V1` terminal | `G` beside it | open | no short across the rail |
| 5 | `V2` terminal | cable **red** | open | `V2` is not on our bus |
| 6 | `G` (V1 side) | `G` (V2 side) | **continuity** | the two grounds are **common** |

**These six are necessary but NOT sufficient.** They prove the PSU→servo path and
that nothing is shorted. **They do not test the USB rail**, which is the one that
can damage the host. P3/P5 below close that gap, and supersede test 6 as a
decision input — a common ground cannot distinguish the two terminals.

**Use resistance (Ω) mode, not the beeper, for all re-tests.** A continuity
beeper is threshold-triggered and will chirp through a capacitor.

---

## Standing rules — every moment, no exceptions

- **Output OFF before any connector is mated or unmated.** The 5264 has no
  sequenced contacts; hot-plugging pits and welds them.
- **Never turn the voltage knob while the output is on and loaded.** On a 30 V
  supply a knob slip reaches the servo's over-voltage trip in a quarter turn.
- **Never connect anything to the 5-pin 2.54 mm header.** Its `5V` pin is an
  *input* expecting exactly 5.0 V; 11.1 V destroys the board. Meter probes only,
  never a clip. Tape it over after pre-flight.
- **Tape over both `V2` screws** for the entire campaign.
- **Do not bond the PSU's negative post to its green earth post.** Leave the
  output floating; that is what makes a lost ground an open circuit rather than a
  return path through the PC.
- **If the CC light comes on in Stage A or B: OUTPUT OFF.** Do not turn the
  current knob up to make the light go away.
- **Never write EEPROM while the PSU is in CC.** A brownout mid-write bricks the
  servo's ID/baud, and there is exactly one servo.
- **Connect ground first, disconnect it last.**
- **Touch the two test leads together and confirm 0.0 Ω before every session.**
  A broken lead reads `OL` — and `OL` is the PASS criterion on every isolation
  test here. The two are indistinguishable, so an unverified lead can turn a
  dangerous board into a clean bill of health.
- **Resistance and diode mode are unpowered-only.** On a live circuit the meter
  injects its own test current: the reading is meaningless and some meters are
  damaged. Once anything is powered, **volts only**.
- **Ohms mode alone can never clear a semiconductor path.** A junction's forward
  drop falls ~110 mV per decade of current, so at the 200 kΩ range's ~1 µA test
  current a real diode drops ~0.32 V and the meter computes **320 kΩ** — which
  reads as "megohms, therefore isolated". **Diode mode, both polarities, is
  mandatory** for any isolation claim.
- The servo **will get hot enough to burn** — the case is aluminium (6-3), far
  worse than the plastic-cased STS3215. Never test temperature with a finger.

## Landing the PSU on the screw terminal

**Bench finding 2026-08-20: alligator clips DO grip these screw heads directly,
and hold.** A review had asserted this was impossible (clip jaws 8–20 mm vs
5.08 mm screw spacing) and called it blocking. Measured on the actual board, it
is not. The assertion was wrong; the procedure below reflects what works.

Clipping to the screw heads is fine, but it introduces one failure mode that a
landed wire does not: **two clips 5.08 mm apart can touch each other, or slip and
land somewhere else.** Either is a dead short across the supply. So:

- **Orient the clips pointing away from each other**, insulating boots slid fully
  forward, and support the banana leads so their stiffness cannot lever a clip off.
- **Tape over both `V2` screws and the 2.54 mm header** *before* clipping, so a
  slipped clip lands on insulation instead of the 12–24 V rail or the 5 V input.
- **Then verify, meter in Ω mode, with both clips attached and the PSU off:
  resistance across the two clips must read OPEN.** A low reading means the jaws
  are touching or bridging. **Re-run this check after every re-clip** — it is the
  whole safety argument for this method, and it takes five seconds.
- **Tug-test each clip** before switching the output on.

*Fallback if a clip ever proves unreliable under load:* a dead Cat5e patch cable
gives eight 24 AWG solid conductors (0.51 mm), ideal for a 5.08 mm clamp and
leaving no stray strands. Cut two stubs at **different lengths — 30 mm and
60 mm** — so they cannot be confused, strip 8 mm at the terminal end and ~25 mm
at the clip end, and bend the free ends into a V. For Stage C, twist two
conductors together per polarity.

---

## Pre-flight — all zero-risk, none of it can damage anything

### P1 — USB only. No PSU on the bench. No servo.

1. Set the logic switch firmly to **5 V** (a detent, not between positions).
2. Plug into the PC with a **USB-A-to-C** cable — C-to-C often fails to enumerate
   on boards that omit the CC pull-downs.
3. `dmesg | tail` and `ls /dev/ttyACM*`. Expect `/dev/ttyACM0`, no driver needed.
4. Confirm the software opens the port. `FD.exe` is Windows-only; on Linux use
   `feetech-servo-sdk` / `scservo_sdk`. **Decide this now, not at 11.1 V.**
5. Meter on DC volts, black on header `GND`, red on header **`TXD`**. USB must be
   connected. A UART transmit line idles **high**, so this reads the logic level
   the switch actually selects: **~5 V or ~3.3 V** depending on position. Return
   the switch to **5 V** and leave it.

   > **Do not probe the `5V` pin to test the switch.** That pin is a fixed
   > USB-derived power output for driving an external MCU; the switch acts on the
   > *data* lines, not on it. Measured 2026-08-20: it reads **4.71 V in both
   > switch positions**, which is correct behaviour, not a fault. (4.7 V rather
   > than 5.0 V is the usual protection-diode/polyfuse drop off VBUS.)
   >
   > That reading is still useful for a different reason: it confirms the header
   > `5V` pin is live off USB, which is exactly the rail **P3 and P5** test for
   > isolation against the servo supply.

6. Unplug USB.

### P2 — Ohm-map the board. Unpowered, USB out, nothing plugged in.

Probe each of the four terminal screws against each pin of the TTL socket.

| expected | if not |
|---|---|
| **Exactly one** screw < 1 Ω to the socket's `V` pin — that is the TTL supply | Two screws low to the same pin → the positive rails are shared → **STOP, split harness** |
| **Both `G` screws** < 1 Ω to the socket `G` pin | Expected — grounds are common |
| The `V2` screw open to every TTL socket pin | Otherwise the rails are not separate → **STOP** |
| The socket `S` pin open to all four screws | Otherwise signal is tied to a power rail → **STOP** |
| Any reading of a few hundred ohms | Reading through active circuitry → **STOP** |

**Measured 2026-08-20** (`0.1` = connected at ~0.1 Ω, `—` = open):

|  | socket `G` | socket `V1` | socket `S` |
|---|---|---|---|
| screw `G` (V1 side) | **0.1** | — | — |
| screw `V1` | — | **0.1** | — |
| screw `G` (V2 side) | **0.1** | — | — |
| screw `V2` | — | — | — |

**PASS.** Exactly one screw feeds the TTL supply pin, so the two terminals'
positive rails are **not** shared. `V2` reaches nothing on our bus. Signal is
isolated from both rails. No mid-range readings, so nothing was measured through
active circuitry.

> **What this does NOT establish.** P2 compares screws to socket pins only. It
> says nothing about the **USB-derived rail**, which is the path that can reach
> the host. That is P3's job, and P2 passing is not evidence for it.

### P3 — Isolation, passive half. **This is the test that protects your PC.**

Probe **TTL socket `V` pin ↔ header `5V` pin**: resistance both ways, diode mode
both ways, in **both** switch positions.

| reading | meaning |
|---|---|
| **OL / megohms**, all four ways | Rails look separated → continue |
| **< 10 Ω** | Hard-tied. **11.1 V will land on USB VBUS** → split harness |
| 0.3–0.7 V drop, **socket→header** | Diode-OR blocking the wrong way → treat as hard-tied |
| 0.3–0.7 V drop, **header→socket** | Blocking correctly → continue, still do P5 |

**Measured 2026-08-20: all four readings OPEN** — resistance both directions,
diode mode both directions.

**PASS, and it rules out more than a short.** A P-channel MOSFET used to OR two
rails has a **body diode that must conduct in one direction**, and a Schottky
OR-diode likewise. Neither appears, so the two most likely passive/semi-passive
coupling topologies are eliminated, not merely "not detected".

**What survives this result:** a load-switch IC with back-to-back FETs (no
exposed body-diode path), or any element whose off-state impedance is beyond the
meter's range. Those turn **on** when the board is energised, which is exactly
what P3b and P5 exist to catch. **A clean P3 is not permission to connect USB at
11.1 V.**

### P3b — USB connected, VOLTAGE ONLY. The direct test.

> **Never use resistance or diode mode on a powered circuit.** The meter injects
> its own test current, so readings are meaningless, and some meters are damaged
> by it. Ω and diode mode are unpowered-only; once USB is in, volts only.

P3 is a *passive* test, and a passive test cannot see a path that only conducts
once the board is energised — a load switch, an ideal-diode controller or a
P-FET ORing USB onto the servo rail all read open unpowered and turn on live.
With USB connected that element is **on**, so one voltage reading settles it.

USB connected. No external supply. **Nothing in any socket** (so the servo is not
loading the rail).

| # | black probe | red probe | what it answers |
|---|---|---|---|
| 1 | socket `G` pin | socket `V1` pin | **does USB 5 V reach the servo supply rail?** |
| 2 | header `GND` | header `5V` | baseline — known good at 4.71 V |

| reading on #1 | verdict |
|---|---|
| **~0.0 V** | USB does not feed the servo rail with nothing plugged in |
| **~4.3–4.8 V** | **USB 5 V DOES reach the servo bus — the rails are coupled.** The vendor's "power the servo from USB for parameter tuning" feature is real and always-on → the through-board 11.1 V plan needs the split harness unless a blocking element is proven |

This is more decisive than P3's resistance readings, and it costs one measurement
at zero risk — nothing here exceeds 5 V.

### 🚨 MEASURED 2026-08-21 — COUPLED. The through-board plan is DEAD.

| # | measurement | reading |
|---|---|---|
| 1 | socket `G` → socket `V1`, USB only | **4.94 V** |
| 2 | header `GND` → header `5V`, USB only | 4.69 V |

**USB VBUS feeds the servo supply rail.** The vendor's "power the servo from USB
for parameter tuning" feature is real and **always-on**, with nothing plugged in
and no software running.

**The drop size identifies the element.** VBUS ≈ 5.0 V → `V1` at 4.94 V is a
**0.06 V** drop: a MOSFET or a bare trace. A silicon diode costs 0.3–0.7 V and a
Schottky 0.2–0.4 V, so **there is no blocking element in that path**. The header
`5V` pin's larger 0.31 V drop is what a Schottky actually looks like — so the two
nodes reach VBUS through *different* elements.

**That is why P3 passed and was still wrong.** `V1` ↔ header `5V` traverses two
back-to-back elements and reads open, while each is independently tied to VBUS.
**P3 measured the wrong node pair.** The host-facing node is USB VBUS itself, not
the header `5V` pin, which was only ever *presumed* to be equivalent.

**Consequence:** 11.1 V on the `V1` terminal drives current straight into the
host's VBUS through a ~0.06 V-drop path. A MOSFET conducts in **both** directions
once its channel is enhanced — only its body diode is one-way — so "it might
block backwards" is not available as a hope.

> **This also resolves the ID-1 anomaly.** A bus scan with no external supply
> reported a servo at ID 1, and echo was ruled out. With `V1` live at 4.94 V off
> USB, a plugged-in servo is powered by USB alone and that reply was genuine.

**→ Use the split harness. `V1` stays permanently empty.**

**It also explains the open question from the bus scan.** A scan with no external
supply reported a servo at ID 1. If reading #1 shows ~4.7 V, then a servo plugged
into the socket is powered from USB alone, and that reply was genuine.

### P4 — Set the supply properly, with no load

Ramping the current knob up until the voltage holds **is not a way to set a
current limit** — with no load the knob does nothing observable, and you end up
with a limit barely above idle, which browns out under load.

1. **Masking tape under each knob, labelled V-C, V-F, A-C, A-F.**
2. Output OFF. **A-COARSE fully CCW.**
3. **Clip the two PSU leads to each other** — a deliberate short. This is safe and
   is what the supply is for.
4. Output ON. The supply enters **CC**; the V display drops near zero.
   **This is the only state in which the A display shows your actual limit.**
5. Dial until the A display reads **0.10 A**. Output OFF. Remove the short.
6. Output ON, leads open. Set **5.00 V**, and **verify with the multimeter** —
   the panel is a ±(0.5 % + 2 digit) affair.
7. **Verify polarity:** DMM red probe on the red clip. It must read **+5.00 V,
   not −5.00 V.** Output OFF.

### P5 — Isolation, active half, at 5 V. **The gate.**

1. Stubs in the TTL terminal (identified electrically in P2, not by silkscreen).
   USB **unplugged**, servo **unplugged**, nothing in any socket.
2. PSU **5.00 V / 0.10 A**. Output ON.
3. Meter DC volts: header `5V` → header `GND`.

| reading | verdict |
|---|---|
| **~0.00 V** | **Rails isolated. Through-board plan is GO.** |
| **~4.3–5.0 V** | **Coupled. STOP.** → Appendix A split harness |

### P6 — Polarity and cable straight-through, at 5 V, servo still unplugged

1. Output OFF. Cable into the **3-pin TTL socket only**, far end dangling.
2. Output ON at 5.00 V / 0.10 A, USB still unplugged.
3. Probe the dangling end (push a single Cat5e strand into each recessed hole):
   - black probe → **black**, red probe → **red**: must read **+5.00 V**
   - **−5.00 V** → terminals reversed; swap the stubs and re-measure
   - **0 V** → wrong terminal, wrong socket, or a bad clamp
   - black → **white**: ~0 V or floating. **Never 5 V.**
4. Output OFF.

This one measurement verifies terminal polarity, terminal→socket mapping, correct
socket, and cable continuity at once, with nothing destructible in circuit.

### P7 — Repeat at 11.1 V, still no servo and no USB

1. Recalibrate the limit by the shorting method to **1.0 A**. Set **11.1 V**,
   verify with the DMM, then **do not touch the voltage knob again.**
2. Output ON. Re-measure header `5V` → `GND`. **Must still be ~0.00 V.**
   Anything above ~0.5 V → OFF, split harness.
3. Re-measure at the dangling connector: **+11.1 V** black-to-red. Output OFF.
4. Tape over the 2.54 mm header and both `V2` screws.

---

## Stage A — first servo power-up

**Preconditions: P1–P7 passed. Limit 1.0 A. 11.1 V verified. USB unplugged.**

1. **Servo clamped to the bench. Bare horn. No arm, no weight. Hands clear.**
   The datasheet lists no mechanical limit angle; 4.9 N·m through steel gears will
   not stop for a finger.
2. Output OFF. Plug the cable into the servo, leaving a service loop.
3. Output ON.

| observation | meaning |
|---|---|
| **~24 mA**, holding 11.1 V, CV light | **Healthy** (5-6) |
| **~280 mA**, holding | Powered up torque-enabled and hunting (5-3). Unexpected, not a fault |
| **Voltage plateaus below 5 V** with current rising | **OUTPUT OFF.** Reversed polarity presents as a *diode*, not a short — H-bridge body diodes conduct at 0.7–1.4 V. **Watch the voltage display, not the current display.** |
| CC light on at all | **OUTPUT OFF** |
| 11.1 V at 0.00 A | Connector not seated |
| Any smell, heat, buzz | **OUTPUT OFF** |

The old "0.05–0.1 A" pass band matched **neither** datasheet figure and is deleted.

4. Hold 60 s. Case must stay at room temperature. Output OFF.

**Record:** idle current, held voltage, case temperature after 60 s.

> The servo will not move and will make no sound. That is a **pass** — it holds
> no position until commanded over the bus.

## Stage B — comms and the EEPROM dump

**Gated on P5. Limit 1.0 A. Still no arm, no weight.**

1. Output ON at 11.1 V. Confirm **CV**, not CC. **Then** plug in USB-C.
   A **powered USB hub** between PC and board is cheap insurance.
2. **Scan IDs 0–253 across all baud rates**, not just 1 Mbps — factory default is
   ID 1 @ 1 Mbps, but a servo that went through `configure_motor.py` may be at
   ID 10–14 / 20–24 / 30–33.
3. **Full read-only dump. Change nothing.** Addresses 5, 6, 13, 14, 15, 16, 19,
   20, 28, 34, 35, 36, 40, 48, 55, 56, 60, 62, 63, 69.
4. **Self-validate the register map.** These addresses come from a community
   STS3215 reference. Check **addr 13 reads 70** (matching 7-11's 70 °C) and
   **addr 62 reads ~111** (11.1 V at 0.1 V/LSB). **If addr 13 is not 70, the map
   is wrong for this model and every register number above is void.**
5. **Commit the raw dump.** Commit `b253142` currently cites the 3.8 A / 2 s /
   80 % / 70 °C thresholds from a third-party table for a *different model*. This
   dump upgrades them from vendor claim to bench fact.
6. **Flags to act on:** addr 16 or 48 below 1000 → every Stage C torque is
   silently clamped and the run is void. addr 55 (lock) set → EEPROM writes fail
   silently. addr 15 raised → the servo trips at exactly the low-battery
   condition most worth characterising.
7. **Sanity gate:** read addr 62 one hundred times. Require **100/100** clean
   replies before logging anything.
8. **Never write 128 to address 40** — 7-14, that re-zeroes the servo's midpoint.

## Stage C — loaded torque and thermal

**Limit raised to 3.0 A by the shorting method** — above the ~1.9 A clamped-stall
bus transient, at/below the connector rating. **Not 5 A.**

### What the ammeter will actually read

At a static hold the mechanical output is zero, so all electrical input becomes
heat, and `I_bus = I_winding² × R / V`. **Bus current is not winding current** —
they differ by the PWM duty and will legitimately disagree by 2–10×.

| torque | winding current (addr 69) | **PSU ammeter** | **heat** |
|---|---|---|---|
| 0.981 N·m | 0.91 A | 0.09 … 0.21 A | 1.0 – 2.4 W |
| 1.569 N·m *(rated)* | 1.45 A | 0.23 … 0.54 A | 2.5 – 6.0 W |
| **2.060 N·m** *(target)* | **1.91 A** | **0.39 … 0.94 A** | **4.4 – 10.4 W** |
| 2.735 N·m | 2.54 A | 0.70 … 1.66 A | 7.7 – 18.4 W |

*(ranges span R = 1.2 Ω to 2.86 Ω. If you see ~2 A on the bus, something is wrong.)*

> **The old acceptance criterion "the three current sources agree within ~10 %"
> was wrong and is deleted.** The servo reports winding current; the PSU and DMM
> report bus current. Following it would have made you discard the servo
> telemetry, halving every inferred torque and understating heat by ~4×.

Your 2.060 N·m target is **136 % of rated current** (5-9). The honest prior is
"probably not sustainable indefinitely" — this test quantifies duty-cycle
headroom, it does not produce a yes/no.

### Mechanical setup

- Servo **bolted or clamped down**. 2 N·m will flip a 74.5 g servo off the bench.
- **Metal horn**, single M3×6. This is the weak link at 2+ N·m, and spline slip is
  silent — logging addr 56 catches it, nothing prevents it.
- 200 mm arm: **weigh it and find its centre of mass.** A 100 g arm with CoM at
  100 mm adds 0.098 N·m — **4.8 % of target** — and is not in the weight table.
  Subtract it from the hung mass.
- Weight on a **string**, arm **horizontal** (verify with a phone level; torque
  scales as cos θ, and 30 ° off is a 13 % error), hanging **≤50 mm above foam**.
- **IR thermometer or taped thermocouple.** Never a finger.
- **Eye protection.** **Emergency stop = finger on the OUTPUT button**, never yank
  a live clip.

### Per-level sequence

1. Cold start — case within 2 °C of ambient. **Record ambient.** ~15 min between.
2. Torque **off**. Hold the arm horizontal by hand.
3. **Read addr 56, write goal position = that exact value, *then* enable torque.**
   This eliminates the torque-enable lurch, the only realistic path to a stall
   transient in this test.
4. Release the arm gently.
5. **Superimpose a dither: ±3 ° at the output, 0.5 Hz.** Through the 1/345 gearbox
   that is ±2.9 motor revolutions — enough to spread commutation across a
   *coreless* motor's brushes, while changing mean torque by cos(3°) = **0.14 %**.
   A 20-minute static hold parks the whole current on one or two commutator
   segments; that is a known coreless failure mode and it is not what walking does.
6. **Log at 1 Hz:** addr 56, 60, 62, 63, 65, 69 + PSU volts, PSU amps, ambient,
   wall clock.
7. Stop at **plateau** (dT/dt < 0.2 K/min for 5 min), **70 °C trip**, or 20 min.

**Order: 0.981 → 1.569 → 2.060 → 2.735 N·m. Stop at the first level that trips.**

### The trip detector — check first on every sample

7-11 says over-load and over-current are cleared by **re-sending a position
command** — and the dither *is* a position command every 2 s. So:

- Any sample where **addr 60 or 69 drops >50 % below the run's median** with no
  command change is a trip. **Truncate the run there.**
- Any **step change in addr 56** with no command change is a trip or horn slip.
- **Without the dither and the load channel, a trip latches silently**: torque
  goes to zero, current collapses to 24 mA, and temperature **plateaus** — which
  is this test's own pass condition. You would close SERVO-2 on a servo that had
  gone limp.
- At 2.060 N·m you sit at **1.91 A vs the 3.8 A over-current limit** and well
  under the 80 % overload limit, so **only over-temperature can legitimately
  trip.** Any other trip means the rig is wrong, not the servo.

### Abort conditions

- PSU goes **CC** → OFF.
- PSU voltage reads **above** 11.1 V → regeneration into a supply that cannot
  sink → OFF.
- **addr 62 more than 0.3 V below the PSU display** → harness drop is corrupting
  the experiment → stop and fix.
- Any 5264 housing too warm to hold → OFF.

### What to derive — model-free, and the real payoff

- **P_servo = (addr 62 × 0.1 V) × I_PSU** — measured, exact.
- **Harness drop = V_PSU − (addr 62 × 0.1)**, logged at every level. This is also
  a hard requirement the robot's real harness must beat.
- **R_th = ΔT / P_servo [K/W]** — **this is the transferable number**, not the
  absolute temperature.
- **τ_thermal** = time to 63 % of ΔT. Expect ~2–3 min, so the answer arrives in
  ~10 minutes, not 20.
- **R_effective = P_servo / (addr 69)²** — resolves the 1.2 Ω vs 2.86 Ω
  inconsistency. **If it lands near 2.86 Ω, add a 9.0 V repeat of the worst
  passing level; if near 1.2 Ω, skip it.**
- **Kt_measured** = torque / addr 69, **recorded with case temperature at each
  point** — copper resistance rises +0.393 %/K, so a sweep mixing cold early
  points with hot late points fits drift, not Kt.

### The pass bar — state it before running

"Plateaus below 70 °C on the bench" is the wrong bar, and it fails favourably,
which is the worst direction. The real bar is:

> **ΔT_bench × k ≤ 70 °C − T_internal**

where T_internal is the trunk bay ambient (the Jetson is in there — see
[`project_thermal_safety`](thermal_safety.md)) and k is the enclosure factor.
With T_internal = 45 °C and k = 1.5, that is **ΔT_bench ≤ 16.7 K — a bench
plateau near 42 °C, nothing like 70.** Note 1-2 also caps rated operating ambient
at **60 °C**, so a bay above 60 °C is out of spec regardless of the cutoff.

**Measure k, don't guess it:** repeat one level (1.569 N·m) with the servo inside
the actual printed leg shell. `k = ΔT_enclosed / ΔT_open` then survives into
`servo_torque_budget.md` as a measured number.

**Instrument note:** use the series DMM (10 A jack) for **one** cross-check at one
operating point, record the PSU ammeter's offset, then **remove it and pull the
lead out of the A jack.** Most 10 A jacks are unfused or rated for seconds, not
20 minutes — and a lead left in the A jack with the dial on V is a dead short.

---

## Appendix A — split-harness fallback (only if P3, P5 or P7 fails)

The board must never see the 11.1 V rail. Because the servo's two sockets are
**internally paralleled**, feeding power into the spare socket does **not** help —
it appears on the first socket's V pin and travels down the cable to the board.
A second cable alone buys zero isolation.

1. On **cable A**, at the **board end**, lift the latch in the housing window with
   a sewing needle and slide the **red (Vcc)** contact out. It re-latches when
   pushed back — fully reversible. The board end now carries **white + black only**.
2. Cut **cable B** in half, strip red and black, land them on the PSU stubs, and
   plug its surviving housing into the servo's **second** socket.
3. PSU (+) → servo Vcc; PSU (−) → servo GND **and** the board's `G` screw as a
   **signal reference only** (milliamps). The board keeps its own 5 V from USB.
4. **Re-run P6 on cable A** before connecting the servo, to confirm the right
   contact was extracted.

---

## Honest limits

1. **There is no official FE-URT-2 electrical document.** Every URT-2-specific
   claim rests on vendor listings that contradict each other. The meter tests
   close most of this, but cannot prove the absence of a path that only conducts
   above some threshold. You are characterising an undocumented board.
2. **One servo, one sample.** The datasheet prints ±10 % on stall torque, stall
   current and no-load speed, and is internally inconsistent by 8 % on Kt and
   2.4× on terminal resistance. **Any envelope carries ~±15 % before rig error.
   Do not close SERVO-2 on a smaller margin.**
3. **addr 63 is a board sensor, not a winding sensor.** It is the *correct* metric
   for "when does the firmware cut out" and the *wrong* one for "is the winding
   safe." A coreless winding hotspot is invisible to it.
4. **Open air ≠ sealed printed leg next to a Jetson.** R_th generalises; absolute
   temperature does not.
5. **Over-temperature recovery is unspecified.** 7-11 documents how over-load,
   over-current and over-voltage clear, and says nothing about over-hot. Measure
   it: after a 70 °C cutoff, log how long until addr 63 falls below 60 °C. That
   recovery time is the real duty-cycle constraint on the robot.
6. **A thermal pass licenses nothing about gear life.** 8-1 warrants >100,000
   cycles at **1/5 stall torque (0.98 N·m)** on a **33 % motion duty cycle**. A
   continuous hold at 2.060 N·m is 2.1× that torque at 3× the duty. State
   explicitly that a thermal pass means *only* "will not hit the 70 °C cutoff".
7. **The weight falls the moment the servo trips — and reaching 70 °C is the
   experiment's success condition.** Padding reduces the consequence, not the event.

## What the outcomes mean

| result | conclusion | action |
|---|---|---|
| 2.060 N·m meets the ΔT bar | **SERVO-2 closed.** The derated ≤1.0 target was too conservative | Ship v7 as-is; retire the SERVO-2 check |
| plateaus, but above the ΔT bar | Duty-cycle limited, not blocked | Set a run-time budget in the S.10 safety layer |
| trips at 70 °C in ~1–2 min | **SERVO-2 is real and v7 is not enough** | Mass reduction or gearing (S.8) — two retrains already showed reward weight alone cannot get there |
| measured Kt far from 11 kg·cm/A | every current↔torque conversion shifts | Re-derive `servo_torque_budget.md` §3 |

## Write results back into

- `docs/jetson-mod/bench_results_servo.md` — raw logs, the EEPROM dump, plots
- `servo_torque_budget.md` — derating assumptions → measured R_th, τ, R_effective
- `known_issues.md` **SERVO-2** — close it, or restate its real target
- `task_plan_v2.md` **S.8** — "UNVERIFIED" thermal thresholds become cited
