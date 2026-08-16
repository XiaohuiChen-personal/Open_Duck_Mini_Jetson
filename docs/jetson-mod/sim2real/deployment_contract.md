# Deployment contract — every field, with its source

**Generated artifact.** `jetson_runtime/policy_contract.json` is written by
`scripts/generate_policy_contract.py`, which re-derives every field from a
primary source at run time. **Do not hand-edit it.**
`generate_policy_contract.py --check` regenerates in memory and exits 1 on any
difference; `tests/test_policy_contract.py` runs that check.

Why it exists: `known_issues.md` **DEPLOY-1** — the ONNX is a bare MLP, and
everything around it lived only as prose. Prose drifts: `AGENTS.md` once listed a
right-leg-first joint order that was the legacy 15-joint BDX table.

Describes **`exported_policies/v7_servo_safe_ppo`**, ONNX md5 `8ea61dcf9034192233675f6d6dbd8f14`.

## Fields and where each comes from

| field | value | source |
|---|---|---|
| `obs_dim` | 53 | derived by summing the term widths; asserted equal to the ONNX input |
| `action_dim` | 14 | `len(joint_order)`; asserted equal to the ONNX output |
| `joint_order` | 14 names | `scripts/duck_init_pos.json` → `action_joint_order`, written by R1 from a live articulation |
| `obs_terms` / `obs_slices` | see below | the archived `env.yaml`'s own `observations.policy` ordering |
| `action_scale` | 0.25 | archived `env.yaml` → `actions.joint_pos.scale` |
| `q_default_rad` | 14 floats | `scripts/duck_init_pos.json` → `init_pos_rad`, in `joint_order` |
| `control_dt_s` | 0.02 | `sim.dt` 0.005 × `decimation` 4 |
| `gait_nb_steps` | 27 (0.54 s) | measured from `polynomial_coefficients.pkl`; asserted uniform across all entries |
| `cmd_hull_trained` | wz [-0.5, 0.5] | archived `env.yaml` → `commands.base_velocity.ranges` |
| `cmd_clamp_deployment` | wz [-0.3, 0.3] | `AGENTS.md` § Safety Rules — a **choice**, not the hull |
| `hard_limits_rad` | 14 pairs | `robot_motors.xml` via `mujoco.MjModel`, free joint skipped |
| `soft_limits_rad` | 14 pairs | `mean ± 0.5·range·0.9` (`robot_cfg.py` soft factor) |
| `servo_ids` | 14 ids | **the one unavoidable literal**, verified identical in `docs/configure_motors.md` and `experiments/v2/configure_motors.py` |
| `servo_baud` | 1,000,000 | `experiments/v2/configure_motors.py` |
| `normalizer_epsilon` | 0.01 | inside the ONNX graph; only a reimplementation needs it |
| `onnx_batch_dim` | 1 | hard-fixed; batch 2 raises |
| `plant_mass_kg` | 2.729035 | the eval JSON's `plant` block — what a live env produced |

## Observation layout

| slice | term | function | dim |
|---|---|---|---|
| `[0:3]` | `base_ang_vel` | `base_ang_vel` | 3 |
| `[3:6]` | `projected_gravity` | `projected_gravity` | 3 |
| `[6:9]` | `velocity_commands` | `generated_commands` | 3 |
| `[9:23]` | `joint_pos` | `delayed_joint_pos_rel` | 14 |
| `[23:37]` | `joint_vel` | `delayed_joint_vel_rel` | 14 |
| `[37:51]` | `actions` | `last_action` | 14 |
| `[51:53]` | `gait_phase` | `gait_phase_observation` | 2 |

## Four things a runtime author must not get wrong

1. **`q_target = q_default + 0.25 × action`.** Neither number is in
   the ONNX graph. Commanding the graph output directly is a 4× gain error plus a
   standing-pose offset (DEPLOY-1).
2. **The output is UNBOUNDED and you must clamp it.** `clip_actions` is null and
   `JointPositionAction` applies no limit clamp. In sim PhysX absorbs an
   out-of-range target; on hardware nothing does.
3. **The joint blocks are RELATIVE** (`q − q_default`), not raw encoder angles
   (DEPLOY-2).
4. **`delayed_joint_pos_rel` — the policy was trained on joint readings
   delayed 0–40 ms** (PLANT-7), because real hardware has latency. Do **not** add
   artificial delay; feed the freshest reading. The point is that real latency is
   inside the training distribution. Measure the real figure in Task S.6.

## `ticks_per_rad` is deliberately absent

The contract stops at joint angles in **radians**. Converting radians to servo
encoder ticks is a property of the servo and its configuration, not of the
policy, and it belongs to the runtime's servo driver. Putting it here would
imply the policy knows something about the bus that it does not.

## Reproduce

```bash
python3 scripts/generate_policy_contract.py --check   # exits 0 if in sync
python3 -m pytest tests/test_policy_contract.py -q
```

