# Wiring diagram — read this before trusting the PNG

## ⚠️ `jetson_wiring_diagram.png` is STALE as of 2026-08-22

**`jetson_wiring_diagram.drawio` is the source of truth.** The PNG was exported
before the HW-1 fix and still shows **11.1 V going into the adapter**, which is
the arrangement that puts 11.1 V on the Jetson's USB port.

The `.drawio` has been corrected but no drawio CLI is available on this machine
to re-export. **Re-export the PNG from the `.drawio` before using it for a build.**

## What changed (HW-1)

| element | was | now |
|---|---|---|
| `mcb` | "Motor Control Board" | **FE-URT-2 — SIGNAL + GND ONLY** |
| `mcb_dc` | "11.1 V DC in" | **V1: LEAVE EMPTY** |
| `P7` edge | power switch → adapter's V1 | power switch → **servo power rail** |
| — | — | new: **servo power rail 11.1 V, direct from pack** |
| — | — | new: HW-1 warning block |

## Why

Measured on the physical FE-URT-2, 2026-08-21: **`V1` sits at 4.94 V from USB
alone** — a 0.06 V drop from VBUS, meaning **no blocking element** in that path.
Feeding the pack's 11.1 V to `V1` drives current straight into the host's USB
rail. The board was confirmed by the owner (2026-08-22) as the part in this
diagram, so this is direct, not by analogy.

Exposure is asymmetric: VBUS on a PD-capable port has over-voltage protection
and a replaceable eFuse, but **D+/D− has an absolute maximum near 3.6 V, no
external clamp is possible, and on a USB4-class port it runs through a
non-replaceable retimer.** A blown VBUS fuse is survivable; a blown mux is a
motherboard.

## The rule for the build

**The bus harness carries V from the pack to the servos. The adapter taps S and
GND only.**

```
  pack 11.1 V ──┬──► servo 1 V ──┬──► servo 2 V ── … ── servo 14 V
                └───────────────────  (bus V wire, adapter NOT on it)

  common GND ───┴──► all servos ──┬──► adapter GND
  bus signal (S) ─────────────────┴──► adapter S
  Jetson USB-C ───────────────────────► adapter (its ONLY power source)
```

Break the red conductor between servo and adapter, and star-ground at the servo
connector so servo return current never crosses the adapter's ground plane.

**Do not** feed power through a servo's spare socket — the two sockets are
internally paralleled, so it arrives at the adapter anyway.

Full detail: [`known_issues.md` HW-1](known_issues.md#hw-1),
[`bench_test_servo.md`](bench_test_servo.md).
