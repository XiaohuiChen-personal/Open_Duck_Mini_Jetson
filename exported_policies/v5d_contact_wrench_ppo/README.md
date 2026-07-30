# v5d_contact_wrench — course-project / deployment archive

Pinned export of the **v5d contact-wrench** PPO policy for reproducible
load-and-step (EN.535.782 haptic teleop baseline switch, 2026-07-30).

| Field | Value |
|---|---|
| Checkpoint | `model_5998.pt` |
| MD5 | `0333e68a4cd9ed3817310ed80f6715e4` |
| Training log | `~/IsaacLab/logs/rsl_rl/open_duck_ppo_v5/2026-07-29_08-59-25/` |
| Train task | `Isaac-Velocity-Rough-OpenDuck-ContactWrench-v0` |
| Play task | `Isaac-Velocity-Rough-OpenDuck-ContactWrench-Play-v0` |
| Actor obs | 59-dim (no `base_lin_vel`; same contract as v4_robust) |
| Runner | `OpenDuckContactPPORunnerCfg` (`obs_groups` actor←policy, critic←critic) |
| Eval JSON | `docs/jetson-mod/eval_results_v5/v5d_contact_wrench.json` |
| Comparison | `docs/jetson-mod/v5_comparison.md` |

Provenance: fine-tune from `v4_robust` (`model_2999.pt`) under the contact-wrench
curriculum; rsl-rl resume yields final iter **5998**. PLAY twin disables pushes,
mass/CoM DR, contact wrenches (`active_frac=0`), and obstacles (`obstacle_frac=0`).

ONNX sidecar (`policy.onnx` + `.data`) and `policy.pt` were copied from the run's
`exported/` folder for optional Jetson/TensorRT work; the course teleop path loads
the RSL-RL `.pt` via `OnPolicyRunner`.
