# Re-gate on the corrected plant — **PASS, all 15 bars**

**2026-08-22.** v7 (`model_8997`) re-scored against the PLANT-11 / PLANT-6 plant.
No retrain. Results: [`eval_results_plantfix/`](eval_results_plantfix/).

## Why this had to happen

Every stored gate number was produced on a plant whose leg `armature` was
**4.74× the measured value** — 87–99.7 % of the simulated joint inertia, 386× the
real link at the ankle. Those numbers were void regardless of which way the
re-gate landed. See [PLANT-11](known_issues.md#plant-11) and
[`bench_results/GO_NO_GO.md`](bench_results/GO_NO_GO.md).

## Verdict

```
VERDICT: PASS — all 15 bars
```

### §6a — torque, the bar v7 had always failed

| bar | result |
|---|---|
| worst leg RMS ≤ 1.0 N·m | **0.754** (`left_ankle`) ✅ |
| no leg joint p99 at the 4.903 clip | none ✅ |
| longest run > 3.923 N·m under 2.0 s | 0.00 s ✅ |

**2.120 → 0.754 N·m.** On the old plant this bar failed at 2.060; it is the only
bar v7 ever failed, and it was measuring a phantom flywheel.

*Two figures appear in this campaign and both are right:* **0.720 N·m** is the
RMS aggregated over all 32 envs, **0.754** is the single recorded env that the
gate scores. The gate uses the more conservative one.

### §6b — SERVO-1, head and neck

| bar | result |
|---|---|
| `neck_pitch` travel ≥ 2.0° | 3.139° ✅ |
| `neck_pitch` on-stop ≤ 10 % | 0.0 % ✅ |
| `neck_pitch` torque RMS ≤ 1.5 N·m | 0.309 ✅ |
| `head_yaw` travel ≥ 10° | 16.8° ✅ |
| [turn] neck travel ≥ 2° | 3.931° ✅ |
| [turn] `head_yaw` ≥ 5° | 12.8° ✅ |

### §6c — stability, and this is the surprise

| bar | old plant | new plant | limit | |
|---|---|---|---|---|
| gait valid | 6/6 | **6/6** | ≥5/6 | ✅ |
| open-field falls | 0.000 % | **0.000 %** | 1.0 % | ✅ |
| push, strict fall (empty arena) | 0.417 % | **0.000 %** | 1.0 % | ✅ |
| push, contact-tolerant fall | 0.234 % | **0.000 %** | 1.0 % | ✅ |
| **sustained wrench** | 62.969 % | **36.250 %** | 80.0 % | ✅ |
| obstacle graze | 6.562 % | **6.042 %** | 12.0 % | ✅ |

**Four of six improved and none regressed.**

I predicted the opposite. Reduced rotor inertia means less passive resistance to
being shoved, so push recovery and sustained wrench were the two places a
regression seemed likely. Both improved instead — wrench by **42 %**. The reading
that fits: the policy performs better when its joints respond the way it was
trained to expect, rather than dragging 4.74× the real inertia. Its corrections
land sooner and it catches itself earlier.

Corroborating, from the main eval: **energy proxy fell 70 %** (12.400 → 3.748) and
lin-vel tracking error improved (0.158 → 0.135). Stance duty rose ~6 pp, i.e. more
double support — a stability-increasing direction.

## Method note

The **primary evidence is a self-comparison**: same policy, same protocol, same
seed, only the plant differs. That isolates the change with no confound.

The relative "also ≤ v6_robust" clauses compare against baselines measured on the
**old** plant and are therefore not strictly valid. It does not affect the
verdict — every bar carries an absolute threshold, and all fifteen pass on those.
If any bar had gone marginal, the correct response was to re-measure the
baseline, not condemn the policy.

**Statistical power:** 3840 episodes per eval. Zero falls gives a 95 % upper
bound of 3/3840 = **0.078 %** by the rule of three. A true 1 % rate would have
produced ~38 falls.

## One operational note

The first `head_turn` run **hung for an hour at 0 % GPU** after the batteries
completed, producing no output. Killed and re-run standalone: it finished in
**~90 seconds**. Cause not diagnosed. If it recurs, kill and retry rather than
waiting — the symptom is CPU spinning near 100 % with GPU idle.

## Not scored here

**G-R3, the video audit, is a manual frame-by-frame review and remains
outstanding.** It is the owner's sign-off (`task_plan_v2.md` S.0).

---

## Isaac shutdown hang — operational, recurring, and it cost hours

**`record_reference_trace.py` and `measure_head_motion.py` write their output and
then hang instead of exiting.** Observed three times on 2026-08-22.

The symptom is easy to misread as a slow run:

- **~100 % CPU with the GPU at 0 %**
- no further log output
- **the artifact is already on disk and complete**

The turn trace wrote its `.npz` in **50 seconds** and then hung; an earlier
`head_turn` sat like that for **an hour** before being killed, and its standalone
re-run finished in ~90 s. Nothing is being computed during the hang.

**Wait on the artifact, not on the process:**

```bash
./isaaclab.sh -p script.py --out "$OUT" ... &
for i in $(seq 1 150); do
  [ -s "$OUT" ] && { sleep 3; pkill -9 -f script.py; break; }
  sleep 10
done
```

**Related, and self-inflicted:** launching a second Isaac job while one is still
hanging produces
`Disabling key-value database because another kit process is locking it`
and the second job dies early. That is what truncated the first trace re-record
after only `fwd`. **One Isaac job at a time, and confirm the previous one is
actually gone — not merely finished.**
