> # ⚠ EVERY NUMBER ON THIS PAGE WAS MEASURED ON THE 3.657 kg PLANT
>
> PhysX substituted a phantom **1.000 kg** on the massless MJCF root `base` for
> the whole v5 campaign, so the robot these results describe is **37.6 % heavier**
> than the one on disk. See
> [`known_issues.md#plant-1`](known_issues.md#plant-1).
>
> - **Corrected-plant re-gate (2.657 kg):** [`m2657_regate.md`](m2657_regate.md).
>   v5d passes every gate there; `v4_robust` collapses to **24.167 %** open-field
>   falls.
> - **Post-Phase-M rebuild (2.729 kg, obs 53 / action 14):**
>   [`rebuild_results.md`](rebuild_results.md).
>
> These numbers are kept as the historical record of the campaign that shipped
> `v5d_contact_wrench`. **Do not quote them as hardware numbers.**

# v5 contact-rich retrain — results

**Written 2026-08-13 by Task R4.** `known_issues.md` **DOC-2** recorded that five
executed training runs, including the one that produced the shipped policy,
existed only inside a document titled "Execution Plan". This is the results
document `v5_retrain_plan.md` mandated and the companion to
[`v4_retrain_results.md`](v4_retrain_results.md). Per-run detail, with
last-100-iteration TensorBoard means, is in
[`experiment_journal.md`](experiment_journal.md) Runs 17–21.

## 1. What v5 changed versus `v4_robust`

`v4_robust` is excellent in the world it was trained for — flat, empty, and
disturbed only by instantaneous velocity kicks. The Duck Embody benchmark then
put it in a furnished apartment and measured **10 falls in 12 trials**, split 7
rotation-under-contact / 1 free-space rotation / 2 other. The contact-rich track
exists to close that gap. Four levers:

| lever | what it does |
|---|---|
| **fall-only terminations** | v4 terminated on *any* trunk contact above 1 N with a −200 penalty, so 3,000 iterations taught "contact = death" and never taught recovery. v5 uses the deployment's own definition: tilt > 60° or root height < 0.09 m |
| **obstacles** | a box the robot can meet while walking |
| **sustained wrench** | a persistent force at 0.2 × body weight, not an instantaneous kick |
| **disturbance-gated rewards** | tracking terms suppressed while a disturbance is active — tried in v5a, **removed after it failed** |

## 2. The four arms, one lever each

| run | arm | lever versus the previous arm |
|---|---|---|
| 17 | `v5_smoke` | 100-iteration validation of the track |
| 18 | `v5a_gated_ft` | disturbance-**gated** rewards |
| 19 | `v5b_ungated_ft` | the gate **removed** |
| 20 | `v5c_contact_only` | obstacles, **no** sustained wrench |
| 21 | `v5d_contact_wrench` | **sustained wrench added** |

## 3. Open-field grid

6 conditions × 10 windows × 64 envs × 30 s = **3,840 episodes** per entry,
seed 42, deterministic, no observation corruption, pushes off.

| policy | gait valid | fall rate | ref RMS (°) |
|---|---|---|---|
| `v5a_gated_ft` | **0 / 6** | 0.000 % | 6.183 |
| `v5b_ungated_ft` | **3 / 6** | 0.000 % | 4.790 |
| `v5c_contact_only` | 6 / 6 | 0.000 % | 4.486 |
| **`v5d_contact_wrench`** | **6 / 6** | **0.000 %** | 4.669 |
| `v4_robust` (control) | 6 / 6 | 0.000 % | 4.478 |

**Zero falls with zero valid gait is not a good result — it is a standing
policy.** That is v5a, and it is why the gait gate is applied *before* any
quality ranking.

## 4. Contact battery

Fall rate (%). The gate is **relative**: each candidate against the `v4_robust`
control under the same protocol on the same plant.

| battery | `v4_robust` | `v5c` | **`v5d`** |
|---|---|---|---|
| push, v4 fall rule | 6.354 | 0.286 | **1.068** |
| push, v5 fall rule | 11.120 | 3.698 | **0.000** |
| **sustained wrench** | 100.000 | **100.000** | **47.109** |
| obstacle graze | 32.604 | 6.224 | **0.312** |

**v5d beats the control on all four.** The decisive line is the wrench row:
`v5c` — obstacles but no sustained wrench — fell in **every single episode**,
identically to the control. Adding the wrench took it to 47.109 %.

**The curriculum has to contain the disturbance it is meant to survive.** That
is the campaign's finding, and it is the reason v5d shipped rather than v5c.

## 5. Video audit

Rendered at `vx = 0.2` and `wz = 0.5` (the v5 campaign's tag; AGENTS.md
specifies `wz = 0.3`, and that difference is `known_issues.md` **EVAL-2**).
v5d showed alternating bipedal gait with real swing clearance and a stepping
turn-in-place. The audit that exists frame-by-frame is the corrected-plant one
in [`eval_results_m2657/videos/AUDIT.md`](eval_results_m2657/videos/AUDIT.md),
which also records that **v5d has no get-up behaviour** — under sustained press
it topples and stays down.

## 6. Verdict

**`v5d_contact_wrench` shipped**, to
`exported_policies/v5d_contact_wrench_ppo/` (`model_5998.pt`, md5
`0333e68a4cd9ed3817310ed80f6715e4`).

It survived the plant correction too: on the 2.657 kg plant it holds 6/6 gait
and 0.000 % open-field falls and beats the same-plant control on all four
contact gates, while `v4_robust` falls **24.167 %** of the time simply walking
([`m2657_regate.md`](m2657_regate.md)). Neither policy had mass DR beyond
`add_base_mass (−0.10, +0.15) kg`, so **that robustness came from the contact
curriculum, not from randomisation** — which is the strongest argument for
carrying these levers into the rebuild.

## 7. What was wrong with this campaign, in hindsight

Recorded because the numbers above are still cited elsewhere.

- **CFG-1 — the wrench applied no yaw torque.** `torque_z_range` was configured
  and then silently discarded: the warp kernel behind
  `set_forces_and_torques_at_position` assigns the composed torque and then
  assigns it again from the position-induced moment. **v5d's headline lever was
  weaker than its own description**, for the whole run. Fixed by Task M0.
- **CFG-3 — the disturbance gate is dead in v5c/v5d.** It is maintained every
  step and read by nothing in that arm, so "gated rewards" describes v5a only.
- **CFG-2 — the obstacle was placed twice per episode** with two independent
  draws, the second silently replacing the first. Every obstacle number above is
  affected. Fixed by Task M0.
- **CFG-5 — `Gait/duty_in_band_frac` reads a 15 ms ContactSensor history**, not
  60 ms, so the in-training canary is systematically optimistic relative to
  `evaluate_policies.py`. Use it as a watchdog, never as a gate number.
- **EVAL-2 — the protocol that ran is not the documented one.** Six conditions
  against AGENTS.md's five, and the turn video at `wz = 0.5` against the
  specified 0.3.
- **PLANT-1, above all.** The plant was 37.6 % heavier than the model on disk.

None of these invalidate the *relative* comparison, which is what the campaign
was designed to make: every arm and the control ran under the same defects on
the same plant. They do mean **no absolute number here is a hardware
prediction.**
