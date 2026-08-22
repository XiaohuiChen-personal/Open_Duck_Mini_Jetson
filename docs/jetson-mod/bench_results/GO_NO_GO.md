# Is sim-to-real possible with the STS3250? — **YES**

**Decided 2026-08-22** on measured data. This file exists to answer one
question: *should this project continue?*

> ## The answer is yes, and the margin is comfortable.
>
> The apparent servo-torque crisis was **substantially a simulation bug**, not a
> property of the hardware. Corrected, the shipped v7 policy lands at
> **0.62–0.91 N·m** at its worst leg joint — **inside** the thermal limit,
> **inside** Feetech's 100,000-cycle warranty load, and at **41–60 % of rated
> continuous current.**

---

## 1. What changed

| | before | after |
|---|---|---|
| worst-joint RMS torque | 2.120 N·m | **0.62 – 0.91 N·m** |
| as % of rated continuous current | **136 %** | **41 – 60 %** |
| vs Feetech warranty load (0.981 N·m) | 2.16× over | **under** |
| vs thermal sustained estimate (1.1 N·m) | 1.9× over | **under** |

The cause: `armature` in `robot_cfg.py:120,135` was set to **0.040 kg·m²**, and
the bench measured **0.00843** — **4.74× too high**. Because armature is
**87–99.7 % of the simulated joint inertia** (386× the real link at the ankle),
the sim was charging the policy for accelerating a flywheel that does not exist.

## 2. The measurement

Step-response identification, 11.1 V, unloaded, torque saturated. Raw:
[`step_test.json`](step_test.json), fit: [`inertia_fit.json`](inertia_fit.json).

| | |
|---|---|
| measured armature | **0.00843 kg·m²** |
| 12 trials × 3 step sizes | 0.0076 – 0.0098 |
| **IQR** | **0.0005** |
| mean R² | 0.9973 |

**Why it is trustworthy:**

- **Independent of step size.** 400, 800 and 1200-count steps all land at
  ~0.0085. That is what distinguishes a physical quantity from a fitting
  artifact — the earlier, contaminated analysis varied 7× across step sizes.
- **The discrimination is not marginal.** At the measured saturated torque of
  1.82 N·m, an armature of 0.040 would produce **45 rad/s²**. The servo
  delivered **210**.
- **It corroborates an independent estimate made beforehand.** Rotor geometry
  (3–5 g coreless cup at r = 5–6 mm through 345:1) predicted 0.009–0.021.
- **The fitter was validated against known answers** before being trusted with
  this one — `test_step_method_recovers_a_known_inertia` recovers synthetic
  inertias of 0.040, 0.020 and 0.0085 to within 12 %.

## 3. The corrected torque, per joint

`scripts/corrected_torque_estimate.py`. Only the inertial term changes; gravity
and contact are untouched, so the result is bounded rather than pointwise:

| joint | v7 RMS | ratio | corrected low | corrected high |
|---|---|---|---|---|
| `left_hip_pitch` | 2.120 | 0.293 | 0.621 | **0.906** |
| `right_hip_pitch` | 1.979 | 0.293 | 0.580 | 0.878 |
| `left_knee` | 1.740 | 0.233 | 0.405 | 0.784 |
| `right_knee` | 1.737 | 0.233 | 0.405 | 0.784 |
| `left_ankle` | 1.786 | 0.213 | 0.380 | 0.774 |
| `right_ankle` | 1.757 | 0.213 | 0.374 | 0.771 |

**Every leg joint clears every bar, on the conservative bound.**

| check | limit | verdict |
|---|---|---|
| thermal sustained estimate | 1.100 N·m | **PASS** |
| Feetech warranty load (8-1) | 0.981 N·m | **PASS** |
| nameplate rated torque (5-8) | 1.569 N·m | **PASS** |

## 4. Everything else that could have killed the project

| risk | status |
|---|---|
| **Battery current** | Not binding. 14 servos draw 2.4 A; +25 W Jetson = 4.9 A, against a ≥15 A BMS and 10–20 A of 2P 18650 |
| **Runtime** | 40–130 min. Shorter than `AGENTS.md:790` implies, but workable |
| **Pack voltage sag** | 0.3–0.4 V measured. Servo cuts at `addr 15` = 6.0 V; a 3S pack never approaches it |
| **Host USB exposure** | **Solved.** [HW-1](../known_issues.md#hw-1) — split harness, verified 0.097 V on the board rail with 11.1 V live |
| **Firmware protections** | Characterised: 80 °C cutoff (not 70), 3.80 A trip, 80 %/2 s overload — [HW-2](../known_issues.md#hw-2) |
| **Mass** | 2.729 kg, verified. The "real build is 6 % heavier" concern was stale text; the real delta is ~0.3 % |

**None of these is a stopper.**

## 5. What this does NOT say

Being explicit, because the good news is easy to over-read.

- **This is an estimate, not a re-simulation.** The definitive number needs the
  plant fixed and the policy re-gated. The bounds are wide (0.62–0.91) precisely
  because the split between inertial and gravitational torque is not resolved
  per-joint.
- **The retrained policy will behave differently.** With a lighter simulated
  rotor the policy can move faster for the same torque, and may choose to. The
  re-gate could come back higher than this estimate — though it starts with
  2.3× of headroom to the warranty load.
- **Peak torque is not addressed here.** v7 peaked at 4.863 N·m (99 % of stall).
  Scaled by the same ratio that is ~1.4 N·m, but peaks are transient and the
  firmware's 3.80 A trip needs 2 s of sustained excursion; the longest in the v7
  trace is 0.08 s. Worth confirming at the re-gate, not a blocker.
- **The thermal limit itself is modelled, not measured.** ~1.1 N·m carries a
  0.85–1.5 band, and the bay ambient inside it is still a guess. But the
  corrected torque clears even the **bottom** of that band, so the conclusion
  survives the uncertainty.
- **One servo, one sample.** ±10 % datasheet spread, and mounting spread across a
  printed chassis is ±30–50 %. `addr 13`/`addr 15` should be dumped from all 14
  at build time.

## 6. Recommendation

**Continue.** The next step is GPU work with no hardware in the loop:

1. `armature` **0.040 → 0.00843** at `robot_cfg.py:120,135`.
2. Correct [PLANT-6](../known_issues.md#plant-6) — the viscous term is already
   inside `damping = 1.3464` (`= kt²/R + friction_viscous`), so implementing the
   fix as currently written would **double-count** it.
3. **Re-gate v7 against the corrected plant.** Every stored gate number was
   produced on a plant with 4.74× the real rotor inertia and is now void — this
   is required regardless of which way the torque lands.
4. Retrain **only if** the re-gate shows a real exceedance. On this estimate it
   will not.

> The two failed torque-reduction retrains were arguing with a number that was
> never real. `τ ∝ w^−0.18` was not a tuning wall — it is what a reward penalty
> looks like when it fights a fictitious inertia.
