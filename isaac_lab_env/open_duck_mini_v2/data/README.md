# Reference Gait Data

## polynomial_coefficients.pkl

Frozen polynomial gait library: a dict of 240 reference walking gaits
(command-grid keyed), each fitted as degree-15 polynomials over one gait
cycle in the Playground joint order. Originates from the Open Duck
Playground pipeline (placo walk engine → gait recording over the command
grid → polynomial fitting).

**Provenance pin:** md5 `7a5515dce7610094bc8da28cfb690a07` (3,107,886 bytes).
Treat this file as an immutable dataset, NOT derivable code — all journaled
PPO gate metrics (e.g. v3's G1 PASS: ref RMS 4.59°) are measured against
this exact artifact, and regenerating it from upstream would not reproduce
those numbers bit-for-bit.

Consumers:
- `isaac_lab_env/open_duck_mini_v2/imitation_reward.py` (ImitationReward —
  loads it at init, no fallback; required for v3-reward training)
- `scripts/evaluate_policies.py` (G1 gate metrics vs the reference)
- `scripts/convert_gait_library_to_amp.py` (source of the 22 AMP clips in
  `../amp/motions/`)

Not needed on the Jetson at runtime — the deployed policy only requires the
gait-phase clock, not the reference library.

The library is kinematic (joint trajectories), so mass/CoM changes (e.g. the
Phase 3 CAD redesign) do not invalidate it. It only needs regeneration if
the robot's kinematics change (link lengths, joint placement).

This file is committed via a `.gitignore` exception to the global `*.pkl`
rule so fresh clones and worktrees are self-contained.
