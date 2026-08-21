# Stage B — what the servo's own registers say

**Measured 2026-08-21** from the physical STS3250, read-only, at 11.1 V on the
split harness. Raw dump: [`sts3250_eeprom_dump.json`](sts3250_eeprom_dump.json).

Tool: `scripts/servo_probe.py --dump --id 1 --expect-volts 11.1`. Read-only by
construction — only PING and READ exist in that file.

## Identity

| field | value |
|---|---|
| ID | **1** (factory default — never through `configure_motors.py`) |
| baud | **1,000,000** (code 0) |
| firmware | 3.10 |
| servo version | 9.11 |
| operating mode | 0 — position servo |
| EEPROM lock (addr 55) | **1 = locked** |

The scan swept **41 IDs × 8 baud rates** and only ID 1 answered, which also
confirms the reply was genuine rather than bus echo (an echo answers at *every*
ID — see the regression in `tests/test_servo_probe.py`).

## Register map validated — 5 independent checks

`addr62 present_voltage = 111` → **11.1 V, exactly the supply setting.** Plus ID,
baud code, position in the 12-bit range, and a plausible case temperature. The
map is confirmed, so every value below is real.

## 🚨 The project's cited firmware thresholds are the datasheet's DEFAULTS, not this unit's settings

Datasheet §7-11 says of these protections: **`可自定义设定`** — "user
configurable". They are EEPROM values, and this unit does not ship at the
documented defaults.

| protection | datasheet §7-11 | **this unit** | delta |
|---|---|---|---|
| over-temperature | 70 °C | **80 °C** (addr 13) | **+10 °C of headroom** |
| over-voltage | > 14 V | **16.0 V** (addr 14) | +2 V |
| under-voltage | < 4 V | **6.0 V** (addr 15) | **trips 2 V EARLIER** |
| overload | 80 % stall / 2 s | **80** (addr 36), **200** (addr 35) | ✅ matches |
| over-current | 3.8 A / 2 s | **310** raw (addr 28), 250 (addr 38) | **units unresolved** |

### The 80 °C finding changes the S.8 pass bar

Every thermal calculation in this project is written against a **70 °C** cutoff.
The firmware on this servo actually enforces **80 °C**. That is 10 K of headroom
nobody had counted, and it moves the Stage C stop condition.

It does **not** make the test easier in the way it first appears: the binding
constraint was never the firmware cutoff but the in-chassis temperature rise
(`ΔT_bench × k ≤ cutoff − T_internal`). Raising the cutoff by 10 K raises the
allowed ΔT_bench by 10/k — with k = 1.5, about **6.7 K more**. Real, but modest.

### The under-voltage finding is the one that bites on the robot

**addr 15 = 6.0 V, not the datasheet's 4 V.** A 3S2P pack sags under load, and
this servo cuts torque at 6.0 V — during a walking gait, on a low battery, with
the robot standing. Worth measuring pack sag under load before trusting it.

### addr 28 — do not convert this to amps yet

`protection_current = 310` raw. The LSB is **not established** for this model:

| assumed LSB | implied trip |
|---|---|
| 6.5 mA (common Feetech figure) | 2.02 A |
| 10 mA | 3.10 A |
| 12.26 mA (the value that would make it equal the datasheet's 3.8 A) | 3.80 A |

The candidates span **1.9×**, and this number propagates into every
current↔torque conversion. **Resolve it in Stage C** by commanding a known torque
and comparing `addr 69` against the bench ammeter. Until then, treat any amp
figure derived from addr 28 as unverified.

## Confirmed healthy — no alarm

| register | value | meaning |
|---|---|---|
| addr 65 servo_status | **0** | **no error bits set at all** |
| addr 40 torque_enable | **0** | boots with torque OFF |
| addr 60 / 69 load, current | 0, ~0 | consistent with torque off and 21 mA idle |
| addr 63 present_temperature | 33 °C | cool |
| addr 16 / 48 torque limits | **1000 / 1000** | full scale — Stage C will not be silently clamped |

**This settles the steady-red-LED question by measurement.** `servo_status = 0`
means no alarm of any kind, and `addr 14 = 16.0 V` means 11.1 V cannot trigger
over-voltage. The steady LED is power-on indication. The earlier community claim
about flashing-red over-voltage remains plausible but is now irrelevant here.

**`addr 40 = 0` also refutes the predicted "may boot torque-enabled and lurch"
risk** for this unit — it explains the silence and the 21 mA directly.

## Consequences

1. **`servo_torque_budget.md`, `S8_torque_envelope.md`, `known_issues.md` and
   `task_plan_v2.md` all cite 70 °C.** That is the datasheet default, not this
   servo. → **HW-2**.
2. **`S8_torque_envelope.md` claims the firmware thresholds are "CITED … no
   longer UNVERIFIED".** Half true: overload matches, over-temperature does not,
   over-current has an unresolved unit. Citing a *configurable* default as if it
   were the enforced value is the actual defect.
3. **`generate_policy_contract.py:239` hard-codes `servo_current_limit_a: 3.8`.**
   That is the datasheet default. This unit's addr 28 may not correspond to it.
4. **The EEPROM is locked (addr 55 = 1).** Protective for us — nothing can be
   altered accidentally — but any future configuration change must unlock, write,
   and **read back**, because writes fail silently while locked.

## Open

- **addr 28 / addr 69 current LSB.** Stage C, against the bench ammeter.
- **Whether all 14 servos share this configuration.** This is one unit. The
  robot's servos may ship differently, and `addr 13`/`addr 15` in particular are
  worth dumping from every one of them at build time.
