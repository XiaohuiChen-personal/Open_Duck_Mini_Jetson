# The two fall rules — what "v4 rule" and "v5 rule" actually mean

**Written 2026-08-22** because the names are actively misleading. **"v4" and
"v5" refer to POLICY GENERATIONS, not to what the evaluation measures.** A
report line reading `push, v4 rule <= 1.0 %` tells the reader nothing about what
was counted, and invites exactly the wrong inference.

## The one-line version

| eval | world | a "fall" is | descriptive name |
|---|---|---|---|
| `pusheval_v4def` | **flat and EMPTY** | trunk touches the **ground** | **push recovery, strict fall** |
| `pusheval_v5def` | contact-rich | a **genuine fall** — trunk contact is survivable | **push recovery, contact-tolerant fall** |
| `obstacleeval` | **obstacles present** | brush or hit an **obstacle** | **obstacle graze** |
| `wrencheval` | sustained external force | loss of control under continuous push | **sustained wrench** |

**Both push evals apply the identical disturbance**: a velocity kick of
**±0.3 m/s in x and y, every 4–7 s** (`env_cfg.py:556`, `:1123`). Only the fall
*definition* differs.

## 🚨 The finding that prompted this file

> **The v4 push-eval world contains no obstacles at all.**

The inheritance chain is

```
OpenDuckPushEvalEnvCfg          (env_cfg.py:556)
  └── OpenDuckRobustEnvCfg_PLAY (:534)
        └── OpenDuckRobustEnvCfg (:440)
              └── OpenDuckRoughEnvCfg
```

and **none of them place obstacles**. The obstacle machinery — `obstacle_frac` —
appears only in the *Contact* environments (`env_cfg.py:775`, `:873`), which are
the v5 track. The `ObstacleEval` task runs at `obstacle_frac = 1.0`; the robust
track sets it to `0.0` (`:951`, `:972`, `:1058`).

**So in the v4 push eval there is nothing to collide with except the floor.**

That changes how the rule should be read. "Any trunk contact above 1 N" sounds
like a collision detector; in an empty arena it is a **strict fall detector**,
and a good one — it catches partial falls and stumbles that a trunk-height
threshold would miss.

**Consequence for reading results:** a `0.000 %` on the v4 push eval means *the
robot never put its trunk on the floor across the whole run* — **not** that it
avoided obstacles, because there were none.

This is easy to get wrong. It was misread once during the 2026-08-22 re-gate,
and the misreading survived until someone asked "aren't there walls?".

## Why two rules exist at all

From `env_cfg.py:1126`:

> *"The v4 `PushEval` task counts any trunk contact above 1 N as a fall, which v5
> is explicitly trained to treat as survivable. Both numbers get reported: the
> v4-comparable one (for continuity with the 6.84 % baseline) and this one, so a
> gate miss can be attributed to the definition rather than to genuine falls."*

The history matters. **v4_robust was excellent in the world it was trained for** —
5/5 gait gate, 0.00 % unpushed falls, 6.84 % pushed falls, on flat empty ground
disturbed only by instantaneous velocity kicks.

Then the Duck Embody benchmark put it in a **furnished apartment**: **10 falls in
12 trials** (1.58 per policy-minute), split 7 rotation-under-contact, 1
free-space rotation, 2 sustained press. **None of those three regimes existed
anywhere in v4 training.** The v5 track was built to add them — and a policy
trained to lean on furniture cannot be scored by a rule that calls leaning a
fall.

So the two numbers answer different questions:

- **strict** — did it stay on its feet in clean conditions? (comparable to the
  historical 6.84 % baseline)
- **contact-tolerant** — did it actually fall, in a world where touching things
  is normal?

By construction the contact-tolerant number should be **≤** the strict one, since
the rule is strictly more permissive. If it ever comes back *higher*, something
is wrong with the measurement, not the policy.

## Naming

The on-disk artifact suffixes remain `pusheval_v4def` / `pusheval_v5def`, and the
gym task IDs remain `Isaac-Velocity-Rough-OpenDuck-PushEval-v0` /
`-ContactPushEval-v0`. **Those are deliberately not renamed** — they are
registered identifiers referenced by existing logs, checkpoints and result files,
and renaming them would break reproducibility against every historical run for a
cosmetic gain.

**What was renamed is every human-facing label**, in
`scripts/check_v7_acceptance.py`:

| was | now |
|---|---|
| `push, v4 rule` | `push recovery, strict fall (empty arena)` |
| `push, v5 rule` | `push recovery, contact-tolerant fall` |
| `obstacle graze` | `obstacle graze (obstacles present)` |

The rule for anything new: **name an evaluation after what it measures, never
after the policy generation that introduced it.**
