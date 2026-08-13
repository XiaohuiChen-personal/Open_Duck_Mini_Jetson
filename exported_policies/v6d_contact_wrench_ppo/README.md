# v6d_contact_wrench — deployment archive

Pinned export of the **v6d contact-wrench** PPO policy, the first policy trained
on the post-Phase-M plant. Produced by Task R3 of `docs/jetson-mod/task_plan_v2.md`.

| Field | Value |
|---|---|
| Checkpoint | `model_5998.pt` |
| MD5 | `37da88d08bf5d593a3febf74bce96dbe` |
| ONNX MD5 | `976c09f31ecb41642c8c9c10007a3e1e` |
| Training log | `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v6/2026-08-13_03-50-54_v6d_contact_wrench/` |
| Train task | `Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0` |
| Play task | `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0` |
| Actor obs | **53**-dim · action **14**-dim |
| Runner | `OpenDuckContactPPORunnerCfg` (`obs_groups` actor←policy, critic←critic) |
| Eval JSON | `docs/jetson-mod/eval_results_rebuild/v6d_contact_wrench.json` |
| Results | `docs/jetson-mod/rebuild_results.md` |
| Video audit | `docs/jetson-mod/eval_results_rebuild/videos/AUDIT.md` |
| **Plant** | **2.729035 kg**, USD asset hash `767f2415d1b3a056d95e9c310434dbbc`, MJCF sha256 `c444493a651b4745…` |

The **Plant** row is not decoration. Its absence is why PLANT-1 — PhysX
substituting a phantom 1.000 kg on a massless MJCF root — stayed invisible
across three policy generations while every published gate number described a
robot 37.6 % heavier than the one on disk.

Provenance: fine-tuned from `v6_robust` (`model_2999.pt`), itself trained
**from scratch** on the post-Phase-M plant because the v4/v5 seeds were trained
at obs 59 / action 16 and no longer load after Task M0b. rsl-rl resume yields
final iter 5998. The PLAY twin disables pushes, mass/CoM DR, contact wrenches
(`active_frac=0`) and obstacles (`obstacle_frac=0`).

## Gate result

Open-field grid, 3,840 episodes, seed 42, deterministic:
**6/6 gait valid**, **0.000 % falls**,
ref RMS 4.719°. The four contact gates and the
same-plant control comparison are in `docs/jetson-mod/rebuild_results.md`.

## ⚠ `deployment_contract.json` is REQUIRED to deploy this policy

`known_issues.md` **DEPLOY-1**, re-confirmed by
`scripts/verify_deployment_contract.py` against this very export:

- **0.25 is not in the ONNX graph.** 0 of 193,784 initializer
  scalars equal it.
- **`q_default` is not in the graph.** The closest action-width tensor
  (`mlp.6.bias`) differs by **1.374409 rad**.

`q_target = q_default + 0.25 * action` lives entirely inside Isaac
Lab's `JointPositionAction`. **A runtime that commands the ONNX output directly
is wrong by a 4x gain and a standing-pose offset of up to 1.374 rad — total,
silent failure.** The sidecar carries both, plus:

- `joint_pos_obs_is_relative: true` — the observation's joint block is
  `q − q_default`, **not** raw encoder angles (DEPLOY-2). Feeding absolute
  angles puts the knees 13–14 σ outside the training distribution.
- `normalizer_epsilon: 0.01` — not in the checkpoint; it is a plain Python
  attribute in `rsl_rl/modules/normalization.py`. rsl-rl's own loader
  reconstructs it, so `policy.pt`/`policy.onnx` are correct; only a hand-rolled
  **reimplementation** is at risk, and that is exactly what a Jetson runtime is.
  Dividing by `_std` alone is off by 28.0 % on one channel.
- `onnx_batch_dim: 1` — hard-fixed (`dynamic_axes={}`). Batch 2/4 raise
  `InvalidArgument`. This is recorded, not a defect to "fix".
- `unmeasurable_obs_dims: {}` — **empty by construction.** Task M0b removed the
  two antenna joints from the action and observation spaces (16→14, 59→53), so
  every remaining dim is measurable on hardware. DEPLOY-3 is closed.

Verify any change with:

```bash
~/IsaacLab/_isaac_sim/python.sh scripts/verify_deployment_contract.py \
    --policy_dir exported_policies/v6d_contact_wrench_ppo
```

10 checks, exit 0. `*.onnx` is gitignored (`known_issues.md` ART-1), so the md5
above is the **only** provenance for the graph; `policy.onnx.data` is tracked.

