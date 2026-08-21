# Sustained torque — the answer, and how much to trust it

**Question:** what torque can the STS3250 hold indefinitely, and does the shipped
policy's **2.060 N·m RMS** at the worst leg joint exceed it?

**Answered 2026-08-21** from bench data plus adversarial review. This is a
**modelled** number, not a measured one — no bench point exceeded ~11 K of rise
or ~0.47 A of motor current, so the operating point is an 8–15× extrapolation in
power. Read §4 before quoting anything here.

---

## 1. The number

> **≈ 1.1 N·m RMS sustained at the worst leg joint, in the robot, assuming a
> 45 °C bay. Honest band 0.85 – 1.5 N·m. Use 1.0 N·m as a planning figure.**

Two-node thermal model — only copper heats the winding, everything else heats the
case:

```
T_case = T_amb + (P_copper + P_mech + P_elec) · R_case→amb
T_wind = T_case + P_copper · R_wind→case
```

| term | value | basis |
|---|---|---|
| R_case→amb, open bench | 10 K/W | 65 cm² body, h ≈ 13 W/m²K |
| R_case→amb, in printed leg | 13.5 K/W (9–18) | enclosure factor ≈ 1.35 |
| R_wind→case | 20 K/W (14–30) | comparable coreless motors + press-fit |
| R_effective | **1.4 Ω** (1.2–1.6) | datasheet 5-10 + hot-copper correction |
| P_mech | ≈ 0.7 · τ | from the trace, η_fwd 0.7 / η_back 0.57 |
| P_elec | 0.232 W | **measured** |
| T_winding limit | 110 °C (100–125) | epoxy-bonded coreless rotor |

Case constraint gives 1.14 N·m, winding constraint 1.13 N·m. They bind at nearly
the same place — coincidence, not design.

**The bay ambient dominates everything.** 30 °C → 1.27 N·m. 55 °C → 0.91 N·m.
**That single unmeasured number moves the answer more than the entire thermal
dispute**, and it is currently a guess.

## 2. Is 2.060 N·m over it? **YES — high confidence**

1.9× over in torque, **~3.5× over in copper power**.

### The strongest evidence needs no thermal model at all

> **I_rms = 2.060 / 1.0787 = 1.910 A, against a nameplate rated continuous
> current of 1.400 A (datasheet 5-9). That is 136 % of rated, indefinitely, in an
> ambient hotter than the rating point.** On the forward-walk trace the worst
> joint reaches 2.2075 N·m → 2.046 A → **146 % of rated**.

This uses only **Kt**, which is verified three independent ways: the rated pair
(1.4 A × 1.0787 = 1.510 vs 1.569 stated, +3.9 %), the stall pair (4.531 vs 4.903,
+8.2 %) — both inside the printed ±10 % — and the motor-referred reading being
absurd by 233×. **Every thermal parameter in dispute could be wrong and this
number does not move.**

### The backstop, also model-free

For 2.060 N·m to survive a 45 °C bay you would need 4.38 W (best case: cold
copper, zero mechanical loss, zero winding gradient) to produce ≤35 K of case
rise — i.e. **R_case→amb ≤ 8.0 K/W**. The still-air floor for a 65 cm² object is
~11–12 K/W, and the servo is *sealed inside a printed leg*, which moves it the
wrong way. **The most favourable corner of the honest parameter space is
1.66 N·m. The band never reaches 2.060.**

### Do not read the clean bench runs as reassurance

Peak motor current across the entire campaign was **0.466 A — 24 %** of what the
policy demands. Nothing tripped because nothing was tested.

### And the firmware will not save you

`addr 28 = 310` at 12.258 mA/count = a **3.80 A** over-current trip, matching
datasheet 7-11 exactly. The longest contiguous run above that in the v7 trace is
**0.08 s** against a **2 s** protection window. Only 1.6 % of samples clip at all.

**The policy is executable. It is just not survivable.** And the 80 °C `addr 13`
cutoff is a *board-sensor* limit: at 5 W of copper the winding sits ~100 K above
what `addr 63` reads. The firmware will let the policy cook the rotor with status
byte 0 throughout.

## 3. Retrain — but not yet, and not at the old target

**The warranty argument is the stronger one for the binary; the thermal argument
is the stronger one for the number.** They do different jobs.

Datasheet **8-1** warrants >100,000 cycles at **1/5 stall (0.981 N·m)** on a
**33 % motion duty cycle**. The shipped policy is **2.10× the warranted load at
~3× the warranted duty**. That argument shares **zero inputs** with the thermal
chain — no Kt, no R, no R_th, no ambient, no cutoff register, no sensor
placement. Perturb the bay ambient and every thermal number moves while 8-1 does
not.

### `servo_torque_budget.md` is NOT independent corroboration

The earlier ≤1.0 N·m target and this estimate agree to ~4 %, and that is
**coincidence**. Both are I²R heating with √(ΔT) scaling; Route A merely cancels
R_th and substitutes a *chosen* 0.8 enclosure factor. Both encode the same
guessed bay ambient and move together. As derating factors on 1.569 N·m: Route A
gives 0.653 (two guesses multiplied), this chain gives 0.678 (one measurement).
**Agreement between a guess and a measurement of the same quantity is not
corroboration.** There are **two** legs here, not three: {thermal, fused} and
{8-1}.

### Why the previous two retrains failed

Both aimed at ≤1.0–1.569 N·m — targets sitting only **1.0–1.45×** above the
0.69 N·m gait floor. That corridor is very narrow, which is a sufficient
explanation for breaking other acceptance bars. A ~1.1–1.2 N·m ceiling gives
**1.6–1.7×** headroom.

> **Recommended target: 1.2 N·m worst-joint RMS** — thermally safe against the
> central estimate, accepting gears as a wear item.
> **1.0 N·m** if you want to stay inside Feetech's warranted envelope.

Note v7b's 1.599 N·m ("101 % of nameplate") aimed at the wrong constant —
**nameplate is a 25 °C free-air figure, not a target.**

### Do not spend the third 5 GPU-hours yet

Two cheaper things can move the target more than any reward weight:

1. **Measure the bay ambient.** The shared guessed input under both thermal
   routes. 30 °C → 1.27 N·m; 55 °C → 0.91 N·m and *cooling* becomes the binding
   work item rather than retraining.
2. **Close PLANT-6 and check `armature = 0.040`** (`robot_cfg.py:120,135`).
   Reflected inertia implies J_rotor = 3.36e-7; a 3–5 g coreless cup at r = 5–6 mm
   gives 0.75–1.8e-7, so armature looks **2–4× high**. Retraining against a torque
   figure the repo itself flags as unfaithful is the real risk in retrain #3.
   Also: `applied_torque` is censored at `effort_limit_sim = 4.903` with peaks at
   4.71–4.86, so the RMS is a **lower** bound.

## 4. What NOT to claim from this

- **Do not call 30 K/W "the servo's thermal resistance."** It is a **board→case
  spreading** resistance measured with **zero watts in the winding**. Name the
  path every time it is quoted.
- **Do not claim τ = 26 min**, or that the staircase corroborates it. There are
  three time constants — winding ~20–60 s, board ~25–30 s, bulk 8–15 min — and
  the **fast** one governs the failure mode. At 4.4 W the winding is 90 % of the
  way to its steady rise in **~70 seconds**. A "26-minute" figure invites the
  belief that brief high torque is thermally free. It is not.
- **The original chain contradicted itself:** if τ were 26 min, a 75 s idle test
  reached 4.7 % of steady state and step 1 measured nothing. Step 2 at face value
  refutes step 1.
- **Do not quote "steady rise 131 K or 313 K."** The arithmetic is right and the
  claim is meaningless — the servo destroys itself first. Quote **"required power
  exceeds allowance by ~3.5×"**, a ratio that survives R_th being nonlinear.
- **Do not treat R_th as ΔT-independent.** h ∝ ΔT^0.25, so a 7 K-anchored figure
  overstates predicted temperature at a 35–53 K rise by ~40 %. This is the
  chain's largest *conservative* bias, and it partly cancels the omitted winding
  path — **which is why the original estimate landed near the right answer for
  the wrong reasons.**
- **Do not present 0.7–1.3 N·m as validated.** It was right by cancelling errors:
  the invalid 2.86 Ω branch pushed the low end down while omitted mechanical and
  quiescent heat pushed the high end down. Right answer, wrong derivation.
- **Do not claim the 80 °C cutoff protects the motor.** It protects the board.
- **Do not claim `torque_limit` bounds torque.** It clamps **duty**; at low speed
  a 20 % clamp still permits ~2.0 N·m.

## 5. What would settle it — ~$30–40 and 2.5 hours

Two K-type thermocouples and a 2-channel logger. Everything else is owned.

1. **Bay ambient** (~30 min). The single most decision-relevant unmeasured
   number, free once the probe exists.
2. **Locked-rotor duty sweep** (~2 min). Clamp the horn, command duty 0.05 / 0.10
   / 0.20, log PSU volts and amps. ω = 0 kills back-EMF, so **R = d²·V_bus /
   I_bus exactly**, and slope-vs-intercept separates I²R from brush drop. Closes
   the R dispute in two minutes.
3. **The loaded hold to plateau** (~90 min) — the run that has never been done.
   `bench_test_servo.md` Stage C, with four mandatory changes:
   - **Run 45–90 min to a genuine plateau**, not the 20 min cap (that is only
     72–92 % of final). Fit the exponential; do not read the endpoint.
   - **Thermocouple on the case over the motor can, and one at the board end**,
     logged against `addr 63`. `T_external − addr63` measures the sensor offset
     and internal gradient — the term this model sets to zero, and the term that
     decides whether the 80 °C trip is protective or decorative.
   - **Take power from PSU volts × amps**, not from Kt, R or `addr 69`. At a
     static hold with ±3° dither the mechanical output is 0.14 % of input, so PSU
     power *is* the heat. This removes the R dispute from the extrapolation
     entirely.
   - **Extrapolate by scaling, not constants:** P(2.060) = P_measured ×
     (2.060/1.569)² = **× 1.723**. No datasheet constants remain in the chain.

**Falsifiable prediction, so run it as a test:** at 1.569 N·m on an open 26.5 °C
bench, if R ≈ 1.2 Ω then P ≈ 2.5 W, the case plateaus at **51–52 °C with
τ ≈ 9 min**, and `addr 63` reads **3–8 K below** the external probe. If R were
~2.9 Ω the case heads for ~86 °C and trips in 12–15 min. **Either outcome already
fails the doc's own ΔT_bench ≤ 16.7 K bar at nameplate torque**, let alone at
2.060.

This collapses R_case→amb, R_board→case, sensor offset, C, τ and R from modelled
to measured — band from **±30 % to roughly ±12 %**. The one term it cannot close
is **R_wind→case**; that stays modelled at 14–30 K/W, and no instrument you can
attach to a sealed servo will measure it.

> **Do not run an instrumented walk on the real robot before (2) and (3).** The
> firmware cutoff will not protect the winding, so the failure mode of that test
> is a destroyed servo.

**One-unit caveat with teeth:** unit-to-unit R_th spread for identical
construction is ±10–15 %, but **mounting** spread across a printed chassis is
**±30–50 %**. A hip joint buried in the trunk beside a Jetson is not an ankle in
open air. One unit on an open bench measures the **best** case.
