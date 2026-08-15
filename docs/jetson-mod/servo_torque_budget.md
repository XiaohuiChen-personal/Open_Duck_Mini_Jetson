# Leg servo torque budget — what to actually target

**Researched 2026-08-15**, to answer a direct question: *is 1.569 N·m the right
sustained target for the legs, given that brief excursions above it are fine?*

**Short answer: 1.569 N·m is the right *nameplate*, but the wrong *target*.
Aim for ≈1.0 N·m RMS per leg joint, and separately lower the simulator's hard
ceiling from 4.903 N·m to ≈3.9 N·m — because the servo's own firmware switches
the output off above that.**

## 1. The 1.569 N·m figure is real — the citation inside this repo was not

`STS3250_CONTINUOUS_NM = 1.569` appears in eight places in this repo and, until
now, **not one of them cited a source**. `measure_joint_torque.py` attributed it
to "AGENTS.md and the datasheet"; `grep -c "16 kg.cm\|1.569" AGENTS.md` returns
**0**.

The number itself survives verification. Feetech's own 8-page product
specification for the STS3250 (Edition A/0, 2024-01-16, hosted on
feetechrc.com) states:

```
5-8   额定负载 Rated Torgue [sic]      16kg.cm     -> 1.5691 N·m
5-9   额定电流 Rated Current           1400mA
5-4   堵转扭力±10% Stall Torque         50kg.cm     -> 4.9033 N·m
5-5   堵转电流±10% Stall Current        4.2A
5-11  Kt 常数                          11kg.cm/A
```

The sheet is internally consistent — Kt × rated current = 15.4 kg·cm ≈ 16, and
Kt × stall current = 46.2 kg·cm ≈ 50, both inside the stated ±10 % — which is
what you would expect from measured factory figures rather than marketing.

**But two glosses this repo applies are not Feetech's.** Feetech calls it
*Rated Torque* (額定負載), not "continuous", and nowhere calls it a thermal
limit. The 8-page sheet contains **no duty-cycle spec, no thermal derating
curve, and no time-at-rated-torque figure**. Treating 16 kg·cm as indefinitely
sustainable is our inference, made at Feetech's 25 °C ±5 °C bench conditions
with no enclosure and no mounting qualification.

## 2. The finding that changes the ceiling

The servo protects itself, and **both of its trip points sit BELOW the
simulator's effort limit**:

| | torque | source |
|---|---|---|
| Feetech **over-current** trip: >3.8 A held 2 s → output OFF | **4.099 N·m** | datasheet 7-11, via Kt = 11 kg·cm/A |
| Feetech **overload** trip: >80 % of stall held 2 s → protection | **3.923 N·m** | datasheet 7-11 |
| **simulator `effort_limit_sim`** | **4.903 N·m** | `robot_cfg.py:34` |

Measured, `v6d_contact_wrench` reaches **exactly 4.903 N·m on four leg joints,
with p99 also at 4.903** ([`v6d_torque_measurement.md`](v6d_torque_measurement.md)).

**So the policy has been trained to live in a torque band the real servo's
firmware will refuse to hold.** On hardware it would trip into protection and
drop the joint. Both thresholds are user-configurable, but raising them does not
create torque the motor can produce — it only removes the protection.

This is the single most actionable result of this research, and it is
independent of every thermal argument below.

## 3. Two independent routes land on ≈1.0 N·m

**Route A — thermal derating.** Winding heat is I²R and torque is Kt·I, so heat
scales with **τ²**, which makes **RMS torque** the correct metric (not mean, not
peak). Feetech's rating is validated at 25 °C bench with free air. This robot
seals its servos in a printed chassis **with a Jetson Orin Nano inside**. Using
the servo's own 70 °C over-temperature cutoff as the ceiling, a 40 °C internal
ambient costs √((70−40)/(70−25)) = **0.82** of the rating, and a sealed printed
mount costs perhaps another 0.8 → **≈1.03 N·m**.

**Route B — Feetech's own durability spec.** The >100,000-cycle endurance figure
is defined at a load of **1/5 of stall = 10 kg·cm = 0.981 N·m**, at a **33 %
motion duty cycle**. Robotis publishes the same 1/5-of-stall design rule for
Dynamixel. Walking is ~100 % duty, so this is if anything generous.

Two unrelated derivations — one thermal, one mechanical-life — converging within
5 % of each other is a stronger signal than either alone.

For reference, today's measured worst leg joint is **2.735 N·m RMS**: 1.74× the
nameplate, **2.7× the ≈1.0 N·m target, and 2.79× the load at which Feetech
specifies its cycle life** — at three times the duty cycle that life assumes.

## 4. Is ≈1.0 N·m even reachable?

Yes, with headroom. The irreducible floor is set by holding the robot up:
single-support gravity at `hip_roll` needs ≈1.04 N·m at the instant of peak
lever, and a whole-gait RMS floor is ≈0.69 N·m including the model's Coulomb
friction. A 1.0 N·m RMS target clears that floor by ≈1.45×.

It is reachable because **most of the present torque is not load**. Verified in
this repo:

```
dof_torques_l2   DISABLED (None)     <- env_cfg.py:209
dof_acc_l2       DISABLED (None)     <- env_cfg.py:210
```

**There is no torque penalty and no acceleration penalty in the shipped reward
function at all.** Nothing ever told the policy that torque costs anything, so
it spent freely up to the clip. Compounding it, `armature = 0.040 kg·m²` is
applied to every leg and head joint — 1,056× the principal inertia of
`hip_roll_assembly_2` and 5,635× that of `head_pitch_to_yaw` — so a large share
of commanded torque is accelerating simulated rotor inertia, which nothing
penalises either.

Upstream Open Duck Playground carries `torques = -1.0e-3`; mainstream Isaac Lab
and legged_gym configs use `dof_torques_l2` between −1e-5 and −1.5e-7 against
tracking weights of 1.0–2.5. The one controlled ablation found (Spectral
Normalization, Table II) cut torque-difference 33 % with task return
statistically unchanged — the penalty is cheap.

## 5. Recommendation

| | value | rationale |
|---|---|---|
| **Hard ceiling** — `effort_limit_sim` | **3.9 N·m** | the firmware overload trip; the policy must not learn to command torque the servo will refuse |
| **Sustained target** — worst-joint RMS | **≤ 1.0 N·m** | thermal derating and Feetech's own cycle-life load, agreeing independently |
| **Excursions** | up to 3.9 N·m, **never > 2 s** | both Feetech trips are defined as "exceeded for 2 s" |
| **How to measure** | RMS over a **5 s** and a **10 min** window, per joint | the winding time constant is seconds, the case constant tens of minutes; a single pooled RMS hides both |

**Why not 1.569 N·m:** it is a bench figure at 25 °C in free air that Feetech
never qualified for duty cycle, and this robot runs its servos sealed in a
printed chassis next to a Jetson. Targeting the nameplate leaves nothing for
ambient rise, mounting, or the ±10 % unit spread Feetech itself prints.

## 6. What would change this, and what must still be checked

- **No hardware data exists.** Everything here is datasheet plus simulation.
  Task **S.8** must measure real current draw and case temperature on a bench.
- **The measured 2.735 N·m is censored.** Four joints sit exactly on the clip,
  so the true unconstrained demand is unknown and 2.735 is a *lower* bound.
- **PLANT-6 is open** — dry friction is inactive during motion and BAM's viscous
  term is dropped, so simulated torque is not a faithful reproduction of motor
  output torque.
- **`armature = 0.040 kg·m²` is itself unverified.** Reflected rotor inertia is
  N²·J_rotor; for a plausible gear ratio and rotor this looks high. If it is too
  high, the simulator overstates torque demand and the real target is easier.
- **±10 % unit spread** on stall torque is printed on the datasheet; a
  worst-case unit is 45 kg·cm, not 50.

## 7. Concrete next change

Two edits, one retrain, one re-gate:

1. `robot_cfg.py`: `STS3250_EFFORT_LIMIT_NM` 4.903 → **3.9** (name it for the
   firmware trip, not the stall).
2. `env_cfg.py`: enable `dof_torques_l2` (start ≈ −2e-5 and tune to hit the
   1.0 N·m target), and add the head joints to `joint_pos_limits` — which
   currently lists only `right_ankle, left_ankle, right_knee, left_knee` and is
   why `neck_pitch` sits on its end stop (**SERVO-1**).
3. Retrain, then re-run `measure_joint_torque.py` and the full gate battery.
   The acceptance test is worst-joint RMS ≤ 1.0 N·m **with the gait gate still
   6/6 and the contact gates still beating the control.**
