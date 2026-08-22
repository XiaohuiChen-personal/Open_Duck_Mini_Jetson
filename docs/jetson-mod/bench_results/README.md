# Bench results — STS3250, 2026-08-21

**The reference for every servo number measured rather than assumed.** When a
document elsewhere in this repo cites a servo figure, it should either point here
or to the datasheet — and this file says which of those two wins where they
disagree.

## The rig, and why each part of it matters

| | |
|---|---|
| device under test | ONE Feetech **STS3250**, ID 1, 1,000,000 baud, firmware 3.10 |
| supply | bench PSU, **11.1 V** (the 3S2P pack's real nominal, *not* 12 V) |
| adapter | **FE-URT-2**, USB-C, `/dev/ttyACM0`, CH343 (`1a86:55d3`), cdc_acm |
| wiring | **split harness** — see [HW-1](../known_issues.md#hw-1) |
| ambient | **26–27 °C** |
| instruments | PSU panel (V/A/W), one DMM |
| **absent** | lever arm, calibrated weights, thermometer, oscilloscope |

> **The split harness is not optional.** The URT-2 ties its servo supply rail to
> USB VBUS through a 0.06 V drop, so powering the servo through the board would
> put 11.1 V on the host's USB port. Red and black are cut in the 3-pin cable;
> the `V1` screw terminal stays permanently empty. Verified under load: the
> board's socket `V1` pin reads **0.097 V** with 11.1 V live on the servo.

## Files

| file | what it holds |
|---|---|
| [`sts3250_eeprom_dump.json`](sts3250_eeprom_dump.json) | full read-only register dump |
| [`stage_b_register_findings.md`](stage_b_register_findings.md) | what the registers mean, and where they contradict the datasheet |
| [`servo_characterisation.md`](servo_characterisation.md) | duty staircase, measured constants, extrapolation limits |
| [`hold_test.json`](hold_test.json) | 20 s torque-enabled hold |
| [`sweep_test.json`](sweep_test.json) | 2 min gentle sweep, 198 samples |
| [`staircase.json`](staircase.json) | 7-level duty staircase, 518 samples |
| [`power_budget_v7.json`](power_budget_v7.json) | pack load derived from the v7 torque trace |
| [`sustained_torque.md`](sustained_torque.md) | the sustained-torque limit and the retrain analysis |
| [**`GO_NO_GO.md`**](GO_NO_GO.md) | **⭐ should the project continue? YES — read this first** |
| [`step_test.json`](step_test.json) | step responses that measured the armature |
| [`inertia_fit.json`](inertia_fit.json) | the armature fit: 0.00843 kg·m² |

Reproduce with `scripts/servo_probe.py` (read-only), `scripts/servo_drive.py`
(write-capable, SRAM allowlist), `scripts/power_budget.py`.

---

## 1. Measured values — prefer these over the datasheet

| quantity | measured | datasheet | confidence |
|---|---|---|---|
| idle current | **21 mA** @ 11.1 V | 24 mA (5-6) | high |
| idle power | **0.232 W** | — | high |
| idle case temperature | **33–34 °C** @ 26–27 °C ambient | — | high |
| **over-temperature cutoff** | **80 °C** (addr 13) | **70 °C** (7-11 text) | high — see §3 |
| **max input voltage** | **16.0 V** (addr 14) | 14 V (7-11) | high |
| **min input voltage** | **6.0 V** (addr 15) | 4 V (7-11) | high |
| overload threshold | 80 % stall (addr 36) | 80 % (7-11) | ✅ agree |
| overload duration | 2.0 s (addr 35 = 200) | 2 s (7-11) | ✅ agree |
| protection current | **310 raw = 3.80 A** @ 12.258 mA/count | 3.8 A (7-11) | high — **resolved**, matches exactly |
| torque limits | 1000 / 1000, full scale | — | high |
| `goal_speed` unit | **counts/s** | undocumented | high — direct |
| `addr 60` load unit | **per-mille PWM DUTY**, not torque | undocumented | high |
| `addr 69` current | **winding, not bus** | undocumented | high — see §4 |
| running friction | **0.1–0.2 N·m** (readings do not close) | undocumented | low |
| voltage sag under load | **0.3–0.4 V** | — | medium |
| encoder range | 0–4095, 12-bit | — | high |
| tracking error, gentle | mean 3.1°, max 10.9° | — | high |

## 2. Identity

`ID 1`, `1,000,000 baud` (code 0), firmware `3.10`, servo version `9.11`,
operating mode `0` (position servo), **EEPROM locked** (`addr 55 = 1`).

Factory default — never through `configure_motors.py`, which assigns 10–14 /
20–24 / 30–33. The scan swept **41 IDs × 8 baud rates** and only ID 1 answered,
which also rules out bus echo (an echo answers at *every* ID).

## 3. 🚨 The cited firmware thresholds are configurable defaults

Datasheet §7-11 says of the protections: **`可自定义设定`** — user configurable.
They are EEPROM registers, and this unit does not ship at the documented values.
**The register is what the firmware enforces.** Full detail and the list of
affected files: [HW-2](../known_issues.md#hw-2).

The under-voltage delta is the one that reaches the robot: **6.0 V, not 4 V.** A
3S2P pack sags under load, and this servo cuts torque at 6.0 V.

## 4. Two register-decoding facts that fail silently

**`addr 60` load signs on bit 10, not bit 15.** A bit-15 decode turns a raw 1080
into a large *positive* load when it means **−76 ‰**. Measured: 63 of 198 sweep
samples were affected. Pinned by `tests/test_servo_drive.py`.

**`addr 60` reports the LIMITER when `torque_limit` binds.** With the limit at
200 it pinned at exactly 200 for 78–100 % of samples. Any torque read under those
conditions is circular. **Stage C must run at `torque_limit = 1000`.**

**`addr 69` is winding current, not bus current.** With torque disabled it reads
**0–1 raw** while the bus draws a measured 21 mA — at any candidate LSB, bus
current would show as 2–3 counts. So the register excludes the board's quiescent
draw.

## 5. Thermal — what was and was not established

| | |
|---|---|
| idle, steady | 33–34 °C at 26–27 °C ambient, 0.232 W → **R_th ≈ 32 K/W** |
| 2 min gentle motion | 33 → 34 °C |
| 5.25 min duty staircase | 34 → 37 °C, monotonic, **not plateaued** |

> **RETRACTED 2026-08-21 — both the R_th figure and the time constant.**
>
> **32 K/W is a board→case SPREADING resistance, not the servo's.** At idle with
> torque disabled, the heat source (regulator, MCU, bus transceiver) and the
> sensor (`addr 63`) are the same object. Roughly 29 of the 32 K/W never appears
> in the winding path at all. Name the path whenever it is quoted.
>
> **"τ ≈ 26 min" is withdrawn, and it was self-refuting.** There are three time
> constants — winding 20–60 s, board 25–30 s, bulk 8–15 min — and the **fast** one
> governs the failure mode: at 4.4 W the winding is 90 % of the way to its steady
> rise in **~70 seconds**. A 26-minute figure invites the belief that brief high
> torque is thermally free. It is not. And if τ really were 26 min, the 75 s idle
> test reached 4.7 % of steady state and measured nothing — so the estimate
> refutes its own premise.
>
> **The staircase proves nothing thermal.** No PSU current was logged, so the 3 K
> rise is one observation against two unknowns and fits τ from 5 to 60 min.

**No loaded thermal data exists.** The staircase was **speed**-saturated, not
torque-saturated, and the horn was unloaded, so delivered torque stayed tiny.
**Saturating a limiter is not thermal loading.**

For the model that replaces this, and the sustained-torque answer, see
[`sustained_torque.md`](sustained_torque.md).

## 6. Electrical load of the whole robot

From the v7 torque trace via `scripts/power_budget.py`.

> **Use the @1.2 Ω column.** The @2.86 Ω column is a **rejected** audit bound,
> retained only so the rejection is auditable: 12 V / 4.2 A back-calculates from
> a *driver clamp*, and would imply 1.66 Ω of driver resistance dissipating
> **29 W in the MOSFETs at stall** inside a 74.5 g servo whose whole board idles
> at 0.232 W. Working value is **R = 1.4 Ω** (band 1.2–1.6).

| | @1.2 Ω | @2.86 Ω |
|---|---|---|
| 14 servos, winding dissipation | 23.5 W | 56.0 W |
| servo bus current (incl. quiescent) | **2.41 A** | **5.34 A** |
| + Jetson 7 W | 3.12 A | 6.04 A |
| + Jetson 15 W | 3.92 A | 6.84 A |
| + Jetson 25 W | 4.92 A | 7.85 A |

**Six leg joints are 91 % of servo power:**

| joint | RMS N·m | W @1.2 Ω | W @2.86 Ω |
|---|---|---|---|
| `left_hip_pitch` | 2.120 | 4.63 | 11.03 |
| `right_hip_pitch` | 1.979 | 4.04 | 9.62 |
| `left_ankle` | 1.786 | 3.29 | 7.83 |
| `right_ankle` | 1.757 | 3.18 | 7.58 |
| `left_knee` | 1.740 | 3.12 | 7.44 |
| `right_knee` | 1.737 | 3.11 | 7.41 |
| *8 others combined* | ≤0.713 each | 2.16 | 5.14 |

**Head, neck and hip-yaw are thermally irrelevant.** Any future effort to reduce
torque should target the six, not spread across fourteen.

**Pack verdict: current is not the constraint.** A 2P 18650 arrangement supplies
10–20 A and the BMS is spec'd ≥15 A, against a worst case of 7.85 A. Runtime at
5–7 Ah is **40–130 min**, which is materially shorter than `AGENTS.md:790`'s
"~1–2 hours at 7 W" — that figure appears to be compute-only and omits 2.4–5.3 A
of servos.

## 7. Open — and what would close each

| question | why it is open | what closes it |
|---|---|---|
| ~~current LSB~~ | **RESOLVED: 12.258 mA/count** — 310 × 12.258 = 3.80 A, the documented trip | — |
| ~~`addr 28` in amps~~ | **RESOLVED: 3.80 A** | — |
| **effective resistance** (1.2–1.6 Ω; 2.86 Ω **rejected**) | hot-copper correction unmeasured | locked-rotor duty sweep, ~2 min |
| **bay ambient** | never measured; **dominates the sustained-torque answer** | thermocouple, ~30 min |
| **motor-path R_th** | only the board path was measured | loaded thermal run to plateau |
| **sustained torque** | no load could be applied | lever arm + weights |
| **unit-to-unit spread** | **one** servo | dump `addr 13` / `addr 15` from all 14 at build time |

All six need the same thing: **a lever arm and calibrated weights.**

## 8. What one servo can never establish

The datasheet prints **±10 %** on stall torque, stall current and no-load speed,
and is internally inconsistent by 8 % on Kt and 2.4× on terminal resistance. Any
envelope derived here carries roughly **±15 %** before rig error. `addr 13` and
`addr 15` in particular are per-unit EEPROM values — do not assume the other 13
servos match this one.

## 9. Corrections this bench made to the repo

| what was believed | what is true | filed as |
|---|---|---|
| the adapter can safely carry 11.1 V | it ties servo V to USB VBUS | [HW-1](../known_issues.md#hw-1) |
| firmware trips at 70 °C | **80 °C** on this unit | [HW-2](../known_issues.md#hw-2) |
| under-voltage trips at 4 V | **6.0 V** | [HW-2](../known_issues.md#hw-2) |
| over-current is 3.8 A | `addr 28 = 310`, units unresolved | [HW-2](../known_issues.md#hw-2) |
| "three current sources should agree within 10 %" | winding ≠ bus current; they differ 2–10× | `bench_test_servo.md` |
| idle draw 0.05–0.1 A | **21 mA** | this file |
