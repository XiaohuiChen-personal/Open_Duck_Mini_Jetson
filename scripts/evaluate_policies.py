# Copyright (c) 2025, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Task 2.5 — Standardized evaluation protocol for all trained policies.

Applies ONE fixed protocol uniformly to every candidate policy (PPO v2,
PPO v3, AMP variants, ...) so the numbers in the course paper's results
table are directly comparable. Per policy, per velocity condition:

    --episodes rollout windows of --episode_length seconds with --num_envs
    parallel envs, deterministic (mean) actions, observation corruption and
    external pushes disabled. Each window contributes num_envs env-episodes.

Metrics (computed in pure numpy from per-step CPU recordings):
    1. Fall rate           — % of episodes terminated early (terminated,
                             not time-out truncated).
    2. Mean episode length — seconds survived (survivors count full window).
    3. Reference RMS (deg) — RMS error over the 10 leg joints vs the FIXED
                             normalized-phase polynomial reference of the
                             library motion nearest to the commanded
                             velocity. Polynomials are constant-term-first
                             and are evaluated at t = (i % nb_steps) /
                             nb_steps, the CORRECTED phase convention from
                             imitation_reward.py (the v2 phase bug evaluated
                             them in seconds — do not regress this).
    4. Action smoothness   — mean squared jerk sum((a_t - 2*a_{t-1} +
                             a_{t-2})^2) over action dims, plus action std.
    5. Stance duty L/R     — % of steps with foot contact force > 1 N per
                             foot, and |L - R| asymmetry in percentage
                             points (the bug-sensitive gait-symmetry metric:
                             the v2 phase bug produced 78%/52% L/R duty).
    6. Leg ROM symmetry    — per joint-pair ratio of the p5-p95 joint range
                             of motion, left / right.
    7. Velocity tracking   — mean ||v_xy_base - cmd_xy|| and |wz - cmd_wz|.
    8. Energy proxy        — mean sum_j |tau_j * qdot_j| from the applied
                             joint torques (W).

Policy adapters own BOTH the checkpoint format and WHICH gym task to build:
RSL-RL policies (62-dim manager-env observations) and skrl AMP policies
(potentially 51/54-dim direct-env observations) are not interchangeable
across envs, so each --policies entry carries its own task id.

Outputs:
    docs/jetson-mod/eval_results/<name>.json   (one file per policy)
    docs/jetson-mod/algorithm_comparison.md    (regenerated comparison table
                                                from ALL JSONs in the results
                                                dir — separate invocations
                                                accumulate)

Usage (GPU required — do NOT run while a training run owns the GPU):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/Projects/Open_Duck_Mini_Jetson/scripts/evaluate_policies.py \
        --policies ppo_v3=Isaac-Velocity-Rough-OpenDuck-Play-v0:rsl_rl:/path/model.pt \
        --policies amp_v1=Isaac-Velocity-OpenDuck-AMP-v0:skrl_amp:/path/agent.pt:/path/skrl_amp_cfg.yaml \
        --headless

GPU-free unit tests of the pure-numpy metric functions (plain python3):
    python3 scripts/evaluate_policies.py --self-test

File layout note: the pure-python/numpy section (constants, metric
functions, writers, self-test) comes FIRST so that --self-test can
short-circuit before any Isaac Sim / torch import. The AppLauncher
boilerplate (copied from IsaacLab rsl_rl/play.py) follows immediately
after and still precedes every isaaclab/torch import, as required.
Do not import this module — run it as a script.
"""

# ======================================================================
# Pure-python / numpy section — safe without Isaac Sim, torch, or a GPU.
# ======================================================================

import argparse
import datetime
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Playground polynomial joint order — MUST match imitation_reward.py
# (duplicated here so --self-test runs without importing isaaclab).
PLAYGROUND_JOINT_ORDER = [
    "left_hip_yaw",     # 0
    "left_hip_roll",    # 1
    "left_hip_pitch",   # 2
    "left_knee",        # 3
    "left_ankle",       # 4
    "neck_pitch",       # 5
    "head_pitch",       # 6
    "head_yaw",         # 7
    "head_roll",        # 8
    "left_antenna",     # 9
    "right_antenna",    # 10
    "right_hip_yaw",    # 11
    "right_hip_roll",   # 12
    "right_hip_pitch",  # 13
    "right_knee",       # 14
    "right_ankle",      # 15
]

# Left legs first, right legs second — the L/R pairing below relies on this.
LEG_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
LEG_PAIR_NAMES = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]

# Index of each leg joint within the 40-dim polynomial reference (dims 0-15
# are joint positions in PLAYGROUND_JOINT_ORDER).
PLAYGROUND_LEG_INDICES = [PLAYGROUND_JOINT_ORDER.index(n) for n in LEG_JOINT_NAMES]

# Contact body names, left then right — matches imitation_reward.py
# FOOT_BODY_NAMES (polynomial contact dims 32=left, 33=right).
FOOT_BODY_NAMES = ["foot_assembly", "foot_assembly_2"]

DEFAULT_CONDITIONS_STR = "0.2,0,0;-0.1,0,0;0,0.1,0;0,0,0.3;0.15,0.05,0.2"
DEFAULT_REFERENCE_PKL = os.path.join(
    REPO_ROOT, "isaac_lab_env", "open_duck_mini_v2", "data",
    "polynomial_coefficients.pkl",
)
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "jetson-mod", "eval_results")
DEFAULT_COMPARISON_MD = os.path.join(
    REPO_ROOT, "docs", "jetson-mod", "algorithm_comparison.md"
)

# Frameworks understood by the policy adapters (values are canonical names).
FRAMEWORK_ALIASES = {
    "rsl_rl": "rsl_rl",
    "rsl-rl": "rsl_rl",
    "rslrl": "rsl_rl",
    "ppo": "rsl_rl",
    "skrl_amp": "skrl_amp",
    "skrl-amp": "skrl_amp",
    "skrl": "skrl_amp",
    "amp": "skrl_amp",
}

# Scalar metric keys averaged across conditions for the aggregate row.
SCALAR_METRIC_KEYS = [
    "fall_rate_pct",
    "mean_episode_length_s",
    "reference_tracking_rms_deg",
    "mean_squared_jerk",
    "action_std",
    "stance_duty_left_pct",
    "stance_duty_right_pct",
    "stance_duty_asymmetry_pp",
    "rom_ratio_mean",
    "lin_vel_xy_error_mps",
    "ang_vel_z_error_radps",
    "energy_proxy_w",
]


@dataclass
class PolicySpec:
    """One policy to evaluate.

    Each adapter owns WHICH gym task id to build: rsl-rl policies expect the
    62-dim manager-env observations while skrl AMP policies may expect
    51/54-dim direct-env observations — the two are not interchangeable.
    """

    name: str
    task_id: str
    framework: str  # canonical: "rsl_rl" | "skrl_amp"
    checkpoint: str
    agent_cfg: str | None = None  # entry-point name or yaml path (skrl), entry-point (rsl_rl)


def parse_policy_spec(spec: str) -> PolicySpec:
    """Parse one ``--policies`` entry.

    Format: ``name=task_id:framework:checkpoint[:agent_cfg]``
    (paths must not contain ':').
    """
    if "=" not in spec:
        raise ValueError(f"Policy spec '{spec}' missing 'name=' prefix")
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Policy spec '{spec}' must be name=task_id:framework:checkpoint[:agent_cfg]"
        )
    framework_raw = parts[1].strip().lower()
    if framework_raw not in FRAMEWORK_ALIASES:
        raise ValueError(
            f"Unknown framework '{parts[1]}' in policy spec '{spec}' "
            f"(known: {sorted(set(FRAMEWORK_ALIASES))})"
        )
    return PolicySpec(
        name=name.strip(),
        task_id=parts[0].strip(),
        framework=FRAMEWORK_ALIASES[framework_raw],
        checkpoint=parts[2].strip(),
        agent_cfg=parts[3].strip() if len(parts) == 4 else None,
    )


def parse_conditions(conditions_str: str) -> list[tuple[float, float, float]]:
    """Parse ``vx,vy,wz;vx,vy,wz;...`` into a list of velocity commands."""
    conditions = []
    for chunk in conditions_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(v) for v in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError(f"Condition '{chunk}' must be vx,vy,wz")
        conditions.append(tuple(vals))
    if not conditions:
        raise ValueError("No conditions parsed")
    return conditions


def condition_key(cond: tuple[float, float, float]) -> str:
    """Stable string key for a velocity condition, used in JSON/markdown."""
    return f"vx{cond[0]:+.2f}_vy{cond[1]:+.2f}_wz{cond[2]:+.2f}"


# ----------------------------------------------------------------------
# Reference motion library (pure numpy mirror of imitation_reward.py)
# ----------------------------------------------------------------------


def load_reference_library(pkl_path: str) -> dict:
    """Load the Playground polynomial gait library (leg joints only).

    Returns dict with:
        keys             : list[str], motion keys "vx_vy_wz"
        velocities       : (M, 3) float64 command velocities
        leg_coefficients : (M, 10, 16) float64 — degree-15 polynomial
                           coefficients, CONSTANT TERM FIRST, for the 10
                           leg joints in LEG_JOINT_NAMES order
        nb_steps         : (M,) int — control steps per gait cycle (27 at
                           50 Hz for the 0.54 s period)
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    keys = sorted(raw.keys())
    velocities, leg_coefficients, nb_steps = [], [], []
    for key in keys:
        entry = raw[key]
        parts = key.split("_")
        velocities.append([float(parts[0]), float(parts[1]), float(parts[2])])
        nb_steps.append(int(entry["nb_steps_in_period"]))
        leg_coefficients.append(
            [entry["coefficients"][f"dim_{i}"] for i in PLAYGROUND_LEG_INDICES]
        )
    return {
        "keys": keys,
        "velocities": np.asarray(velocities, dtype=np.float64),
        "leg_coefficients": np.asarray(leg_coefficients, dtype=np.float64),
        "nb_steps": np.asarray(nb_steps, dtype=np.int64),
    }


def nearest_motion_index(velocities: np.ndarray, cmd: np.ndarray) -> int:
    """Index of the library motion with the closest (vx, vy, wz) command."""
    dists = np.sum((velocities - np.asarray(cmd, dtype=np.float64)) ** 2, axis=-1)
    return int(np.argmin(dists))


def evaluate_polynomial_reference(coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate degree-15 polynomials at normalized phase t in [0, 1).

    Horner's scheme with CONSTANT-TERM-FIRST coefficients — identical to
    ImitationReward.__call__ (imitation_reward.py): result = c[15]; then
    result = result * t + c[i] for i = 14..0.

    Args:
        coeffs: (D, 16) coefficients, constant term first.
        t: (T,) normalized phase values, t = (i % nb_steps) / nb_steps.

    Returns:
        (T, D) reference values.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    result = np.broadcast_to(coeffs[:, 15], (t.shape[0], coeffs.shape[0])).copy()
    for i in range(14, -1, -1):
        result = result * t[:, None] + coeffs[None, :, i]
    return result


# ----------------------------------------------------------------------
# Metric functions (pure numpy — unit-tested by --self-test)
#
# Shapes: T = steps per window, N = episodes (envs x windows concatenated).
# ``mask`` is (T, N) bool — True where the post-step state of env n at step
# t belongs to its FIRST episode in the window (False from the step the env
# first terminated/truncated onward, so auto-reset garbage is excluded).
# ----------------------------------------------------------------------


def fall_metrics(fall_flags: np.ndarray, episode_len_steps: np.ndarray,
                 step_dt: float) -> dict:
    """Metric 1+2: fall rate (%) and mean episode length (s)."""
    episodes = int(fall_flags.shape[0])
    falls = int(np.count_nonzero(fall_flags))
    return {
        "episodes": episodes,
        "falls": falls,
        "fall_rate_pct": 100.0 * falls / max(episodes, 1),
        "mean_episode_length_s": float(np.mean(episode_len_steps) * step_dt),
    }


def reference_tracking_rms_deg(actual_leg_pos: np.ndarray, ref_leg_pos: np.ndarray,
                               mask: np.ndarray) -> float:
    """Metric 3: RMS error (deg) over the 10 leg joints vs the reference.

    Args:
        actual_leg_pos: (T, N, 10) measured joint positions (rad), in
            LEG_JOINT_NAMES order.
        ref_leg_pos: (T, 10) fixed normalized-phase reference (rad).
        mask: (T, N) valid-sample mask.
    """
    err_sq = (actual_leg_pos - ref_leg_pos[:, None, :]) ** 2  # (T, N, 10)
    valid = err_sq[mask]  # (K, 10)
    if valid.size == 0:
        return float("nan")
    return float(math.degrees(math.sqrt(float(np.mean(valid)))))


def action_smoothness_metrics(actions: np.ndarray, mask: np.ndarray) -> dict:
    """Metric 4: mean squared jerk and action std.

    mean_squared_jerk: mean over valid steps of
        sum_dims (a_t - 2*a_{t-1} + a_{t-2})^2.
    action_std: per-env std over time per action dim, averaged over dims
        and envs.
    """
    T, N, _ = actions.shape
    if T >= 3:
        jerk = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]  # (T-2, N, A)
        sq_jerk = np.sum(jerk**2, axis=-1)  # (T-2, N)
        valid3 = mask[2:] & mask[1:-1] & mask[:-2]
        mean_sq_jerk = float(np.mean(sq_jerk[valid3])) if valid3.any() else float("nan")
    else:
        mean_sq_jerk = float("nan")

    stds = []
    for n in range(N):
        v = mask[:, n]
        if np.count_nonzero(v) >= 2:
            stds.append(float(np.mean(np.std(actions[v, n, :], axis=0))))
    action_std = float(np.mean(stds)) if stds else float("nan")
    return {"mean_squared_jerk": mean_sq_jerk, "action_std": action_std}


def stance_duty_metrics(foot_forces: np.ndarray, mask: np.ndarray,
                        threshold: float = 1.0) -> dict:
    """Metric 5: stance duty L/R (%) and asymmetry (pp).

    Args:
        foot_forces: (T, N, 2) contact force norms [left, right] in N.
        mask: (T, N) valid-sample mask.
        threshold: contact force threshold in N (protocol: 1 N).
    """
    if not mask.any():
        nan = float("nan")
        return {
            "stance_duty_left_pct": nan,
            "stance_duty_right_pct": nan,
            "stance_duty_asymmetry_pp": nan,
        }
    contact = foot_forces > threshold  # (T, N, 2)
    duty_left = 100.0 * float(np.mean(contact[..., 0][mask]))
    duty_right = 100.0 * float(np.mean(contact[..., 1][mask]))
    return {
        "stance_duty_left_pct": duty_left,
        "stance_duty_right_pct": duty_right,
        "stance_duty_asymmetry_pp": abs(duty_left - duty_right),
    }


def rom_symmetry_metrics(leg_pos: np.ndarray, mask: np.ndarray,
                         min_steps: int = 50) -> tuple[dict, float]:
    """Metric 6: per joint-pair p5-p95 ROM ratio, left / right.

    Computes the p95 - p5 range of motion per episode per joint (episodes
    shorter than ``min_steps`` valid samples are skipped), averages the ROM
    over episodes, then forms the L/R ratio per joint pair.

    Args:
        leg_pos: (T, N, 10) joint positions in LEG_JOINT_NAMES order
            (left joints 0-4, right joints 5-9).
        mask: (T, N) valid-sample mask.

    Returns:
        (per_pair ratios dict keyed by LEG_PAIR_NAMES, mean ratio).
    """
    _, N, _ = leg_pos.shape
    roms = []
    for n in range(N):
        v = mask[:, n]
        if np.count_nonzero(v) < min_steps:
            continue
        p5, p95 = np.percentile(leg_pos[v, n, :], [5.0, 95.0], axis=0)
        roms.append(p95 - p5)  # (10,)
    if not roms:
        nan = float("nan")
        return {name: nan for name in LEG_PAIR_NAMES}, nan

    mean_rom = np.mean(np.asarray(roms), axis=0)  # (10,)
    left, right = mean_rom[:5], mean_rom[5:]
    ratios = left / np.maximum(right, 1e-9)
    per_pair = {name: float(r) for name, r in zip(LEG_PAIR_NAMES, ratios)}
    return per_pair, float(np.mean(ratios))


def velocity_tracking_metrics(lin_vel_xy: np.ndarray, ang_vel_z: np.ndarray,
                              cmd: np.ndarray, mask: np.ndarray) -> dict:
    """Metric 7: mean ||v_xy_achieved - cmd_xy|| and mean |wz - cmd_wz|.

    Args:
        lin_vel_xy: (T, N, 2) base-frame linear velocity (matches the
            command frame used by track_lin_vel_xy_exp).
        ang_vel_z: (T, N) world-frame yaw rate (matches
            track_ang_vel_z_world_exp).
        cmd: (3,) commanded (vx, vy, wz).
        mask: (T, N) valid-sample mask.
    """
    if not mask.any():
        return {
            "lin_vel_xy_error_mps": float("nan"),
            "ang_vel_z_error_radps": float("nan"),
        }
    cmd = np.asarray(cmd, dtype=np.float64)
    err_xy = np.linalg.norm(lin_vel_xy - cmd[:2], axis=-1)  # (T, N)
    err_wz = np.abs(ang_vel_z - cmd[2])  # (T, N)
    return {
        "lin_vel_xy_error_mps": float(np.mean(err_xy[mask])),
        "ang_vel_z_error_radps": float(np.mean(err_wz[mask])),
    }


def energy_proxy_metric(applied_torque: np.ndarray, joint_vel: np.ndarray,
                        mask: np.ndarray) -> float:
    """Metric 8: mean over valid steps of sum_j |tau_j * qdot_j| (W)."""
    if not mask.any():
        return float("nan")
    power = np.sum(np.abs(applied_torque * joint_vel), axis=-1)  # (T, N)
    return float(np.mean(power[mask]))


def compute_condition_metrics(rec: dict, ref_leg_pos: np.ndarray, cmd: np.ndarray,
                              step_dt: float, contact_threshold: float) -> dict:
    """Assemble all protocol metrics for one velocity condition.

    Args:
        rec: recording dict with keys actions (T,N,A), leg_pos (T,N,10),
            joint_vel (T,N,J), torque (T,N,J), foot_force (T,N,2),
            lin_vel_xy (T,N,2), ang_vel_z (T,N), mask (T,N) bool,
            fall (N,) bool, ep_len (N,) int.
        ref_leg_pos: (T, 10) fixed normalized-phase reference (rad).
        cmd: (3,) commanded velocity for this condition.
    """
    mask = rec["mask"]
    metrics = fall_metrics(rec["fall"], rec["ep_len"], step_dt)
    metrics["reference_tracking_rms_deg"] = reference_tracking_rms_deg(
        rec["leg_pos"], ref_leg_pos, mask
    )
    metrics.update(action_smoothness_metrics(rec["actions"], mask))
    metrics.update(stance_duty_metrics(rec["foot_force"], mask, contact_threshold))
    per_pair, rom_mean = rom_symmetry_metrics(rec["leg_pos"], mask)
    metrics["rom_ratio_per_pair"] = per_pair
    metrics["rom_ratio_mean"] = rom_mean
    metrics.update(
        velocity_tracking_metrics(rec["lin_vel_xy"], rec["ang_vel_z"], cmd, mask)
    )
    metrics["energy_proxy_w"] = energy_proxy_metric(
        rec["torque"], rec["joint_vel"], mask
    )
    metrics["command"] = [float(c) for c in cmd]
    return metrics


def aggregate_metrics(per_condition: dict) -> dict:
    """Average the scalar metrics across conditions (nan-tolerant)."""
    aggregate: dict = {}
    for key in SCALAR_METRIC_KEYS:
        vals = [m[key] for m in per_condition.values() if key in m]
        aggregate[key] = float(np.nanmean(vals)) if vals else float("nan")
    aggregate["episodes"] = int(
        sum(m.get("episodes", 0) for m in per_condition.values())
    )
    aggregate["falls"] = int(sum(m.get("falls", 0) for m in per_condition.values()))
    return aggregate


# ----------------------------------------------------------------------
# Writers (JSON per policy + markdown comparison table)
# ----------------------------------------------------------------------

AUTO_BEGIN = "<!-- BEGIN AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->"
AUTO_END = "<!-- END AUTO-GENERATED RESULTS (scripts/evaluate_policies.py) -->"

PROTOCOL_HEADER = """# Algorithm Comparison — Open Duck Mini v2 (Task 2.5)

Standardized evaluation protocol applied uniformly to all trained policies
(PPO v2, PPO v3, AMP variants). Produced by `scripts/evaluate_policies.py`;
per-policy raw numbers live in `docs/jetson-mod/eval_results/<name>.json`
(each JSON records the exact protocol parameters used for that run).

## Protocol (defaults — all CLI-overridable)

- **Velocity conditions** (vx m/s, vy m/s, wz rad/s):
  (0.2, 0, 0), (-0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.3), (0.15, 0.05, 0.2)
- **Per condition:** 10 rollout windows x 30 s, 64 parallel envs
  (each window yields 64 env-episodes; only each env's FIRST episode per
  window is scored — post-fall auto-reset data is discarded)
- **Policy:** deterministic (mean) actions, no observation corruption,
  no external pushes, fixed command (degenerate command ranges)
- **Contact threshold:** 1 N on the foot contact force norm

## Metric definitions

| Metric | Definition |
|---|---|
| Fall rate (%) | episodes terminated early (termination, not time-out) |
| Ep len (s) | mean episode length; survivors count the full window |
| Ref RMS (deg) | RMS over 10 leg joints vs the nearest library motion's polynomial reference, evaluated at normalized phase t = (i % nb_steps) / nb_steps, coefficients constant-term-first (the corrected v3 phase convention) |
| Jerk | mean over steps of sum over action dims of (a_t - 2a_(t-1) + a_(t-2))^2 |
| Action std | per-env std of actions over time, averaged over dims and envs |
| Duty L / R (%) | share of steps with foot contact force > 1 N, per foot |
| Duty asym (pp) | abs(duty_L - duty_R) — gait-symmetry indicator (the v2 phase bug showed 78/52) |
| ROM ratio L/R | per joint-pair ratio of the p5-p95 range of motion, left/right (1.0 = symmetric) |
| v_xy err (m/s) | mean L2 error between achieved base-frame planar velocity and command |
| wz err (rad/s) | mean abs error between world-frame yaw rate and command |
| Energy (W) | mean of sum_j abs(tau_j * qdot_j) over applied joint torques |

Aggregate values below are means over the five conditions.

"""


def _fmt(value, spec: str = "{:.2f}") -> str:
    """Format a metric value for markdown, mapping None/NaN to 'n/a'."""
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "n/a"
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def render_results_section(entries: list[dict]) -> str:
    """Render the comparison tables (pure function of the JSON entries)."""
    lines = [
        "## Results",
        "",
        f"_Last regenerated: {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "### Aggregate over all conditions",
        "",
        "| Policy | Framework | Fall rate (%) | Ep len (s) | Ref RMS (deg) | Jerk | "
        "Action std | Duty L (%) | Duty R (%) | Duty asym (pp) | ROM ratio L/R | "
        "v_xy err (m/s) | wz err (rad/s) | Energy (W) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        a = e.get("aggregate", {})
        lines.append(
            f"| {e.get('name', '?')} "
            f"| {e.get('framework', '?')} "
            f"| {_fmt(a.get('fall_rate_pct'), '{:.1f}')} "
            f"| {_fmt(a.get('mean_episode_length_s'), '{:.1f}')} "
            f"| {_fmt(a.get('reference_tracking_rms_deg'), '{:.2f}')} "
            f"| {_fmt(a.get('mean_squared_jerk'), '{:.4f}')} "
            f"| {_fmt(a.get('action_std'), '{:.3f}')} "
            f"| {_fmt(a.get('stance_duty_left_pct'), '{:.1f}')} "
            f"| {_fmt(a.get('stance_duty_right_pct'), '{:.1f}')} "
            f"| {_fmt(a.get('stance_duty_asymmetry_pp'), '{:.1f}')} "
            f"| {_fmt(a.get('rom_ratio_mean'), '{:.2f}')} "
            f"| {_fmt(a.get('lin_vel_xy_error_mps'), '{:.3f}')} "
            f"| {_fmt(a.get('ang_vel_z_error_radps'), '{:.3f}')} "
            f"| {_fmt(a.get('energy_proxy_w'), '{:.2f}')} |"
        )

    # Per-condition fall-rate matrix (condition keys in first-seen order).
    cond_keys: list[str] = []
    for e in entries:
        for k in e.get("per_condition", {}):
            if k not in cond_keys:
                cond_keys.append(k)
    if cond_keys:
        lines += [
            "",
            "### Fall rate (%) per condition",
            "",
            "| Policy | " + " | ".join(cond_keys) + " |",
            "|---|" + "---|" * len(cond_keys),
        ]
        for e in entries:
            pc = e.get("per_condition", {})
            cells = [
                _fmt(pc.get(k, {}).get("fall_rate_pct"), "{:.1f}") for k in cond_keys
            ]
            lines.append(f"| {e.get('name', '?')} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_policy_json(output_dir: str, result: dict) -> str:
    """Write one policy's evaluation results to <output_dir>/<name>.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result['name']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def write_comparison_markdown(md_path: str, results_dir: str) -> str:
    """(Re)generate the comparison table from ALL JSONs in results_dir.

    The table lives between AUTO_BEGIN/AUTO_END markers so hand-written
    analysis around it survives regeneration; separate per-policy script
    invocations accumulate into the same table.
    """
    entries = []
    if os.path.isdir(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(results_dir, fname)) as f:
                    entries.append(json.load(f))

    section = f"{AUTO_BEGIN}\n{render_results_section(entries)}{AUTO_END}\n"

    if os.path.isfile(md_path):
        with open(md_path) as f:
            content = f.read()
        if AUTO_BEGIN in content and AUTO_END in content:
            head = content.split(AUTO_BEGIN)[0]
            tail = content.split(AUTO_END, 1)[1]
            content = head + section + tail
        else:
            content = content.rstrip("\n") + "\n\n" + section
    else:
        content = PROTOCOL_HEADER + section

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write(content)
    return md_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Protocol arguments (AppLauncher args are appended later)."""
    parser = argparse.ArgumentParser(
        description="Task 2.5 standardized evaluation protocol for trained policies."
    )
    parser.add_argument(
        "--policies",
        action="append",
        default=None,
        metavar="NAME=TASK_ID:FRAMEWORK:CHECKPOINT[:AGENT_CFG]",
        help=(
            "Policy to evaluate (repeatable). FRAMEWORK is rsl_rl or skrl_amp. "
            "AGENT_CFG is an agent cfg entry-point name (default "
            "rsl_rl_cfg_entry_point / skrl_amp_cfg_entry_point) or, for skrl, "
            "a path to the agent yaml."
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Rollout windows per condition (each window yields num_envs episodes).",
    )
    parser.add_argument(
        "--episode_length", type=float, default=30.0,
        help="Episode window length in seconds.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=64,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--conditions", type=str, default=DEFAULT_CONDITIONS_STR,
        help="Velocity conditions 'vx,vy,wz;vx,vy,wz;...' (m/s, m/s, rad/s).",
    )
    parser.add_argument(
        "--command_name", type=str, default="base_velocity",
        help="Name of the velocity command term in the command manager.",
    )
    parser.add_argument(
        "--contact_threshold", type=float, default=1.0,
        help="Foot contact force threshold in N for stance detection.",
    )
    parser.add_argument(
        "--reference_pkl", type=str, default=DEFAULT_REFERENCE_PKL,
        help="Path to the Playground polynomial_coefficients.pkl gait library.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-policy JSON results.",
    )
    parser.add_argument(
        "--comparison_md", type=str, default=DEFAULT_COMPARISON_MD,
        help="Markdown comparison table to (re)generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Environment seed.",
    )
    parser.add_argument(
        "--self-test", action="store_true", dest="self_test",
        help="Run the pure-numpy metric unit tests and exit (no Isaac Sim).",
    )
    return parser


# ----------------------------------------------------------------------
# Self-test (runs WITHOUT Isaac Sim / torch — plain python3 + numpy)
# ----------------------------------------------------------------------


def run_self_test() -> int:
    """Unit-test the pure-numpy metric functions with synthetic arrays."""
    failures = 0

    def check(name: str, ok: bool):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures += 1

    # --- CLI parsing -------------------------------------------------
    conds = parse_conditions(DEFAULT_CONDITIONS_STR)
    check(
        "parse_conditions default protocol",
        conds == [(0.2, 0, 0), (-0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.3),
                  (0.15, 0.05, 0.2)],
    )

    spec = parse_policy_spec(
        "ppo_v3=Isaac-Velocity-Rough-OpenDuck-Play-v0:rsl_rl:/tmp/model.pt"
    )
    check(
        "parse_policy_spec rsl_rl (3 fields)",
        spec == PolicySpec("ppo_v3", "Isaac-Velocity-Rough-OpenDuck-Play-v0",
                           "rsl_rl", "/tmp/model.pt", None),
    )
    spec = parse_policy_spec("amp=Task-v0:amp:/tmp/agent.pt:/tmp/amp_cfg.yaml")
    check(
        "parse_policy_spec skrl_amp alias + agent yaml (4 fields)",
        spec == PolicySpec("amp", "Task-v0", "skrl_amp", "/tmp/agent.pt",
                           "/tmp/amp_cfg.yaml"),
    )
    try:
        parse_policy_spec("x=Task-v0:sb3:/tmp/model.zip")
        check("parse_policy_spec rejects unknown framework", False)
    except ValueError:
        check("parse_policy_spec rejects unknown framework", True)

    # --- Polynomial evaluation (constant-term-first Horner) ----------
    coeffs = np.zeros((1, 16))
    coeffs[0, :3] = [2.0, 3.0, 1.0]  # p(t) = 2 + 3t + t^2
    vals = evaluate_polynomial_reference(coeffs, np.array([0.0, 0.5, 1.0]))
    check(
        "polynomial constant-term-first: p(t)=2+3t+t^2 at t=[0,.5,1]",
        np.allclose(vals[:, 0], [2.0, 3.75, 6.0]),
    )

    rng = np.random.default_rng(0)
    rand_coeffs = rng.normal(size=(4, 16))
    t = np.linspace(0.0, 1.0, 13)
    ours = evaluate_polynomial_reference(rand_coeffs, t)
    numpy_ref = np.stack(
        [np.polyval(rand_coeffs[d, ::-1], t) for d in range(4)], axis=-1
    )
    check("polynomial matches np.polyval(c[::-1]) on random degree-15",
          np.allclose(ours, numpy_ref))

    # --- Reference tracking RMS --------------------------------------
    T, N = 27, 4
    nb = 27
    phase = (np.arange(T) % nb) / nb
    ref = evaluate_polynomial_reference(rng.normal(size=(10, 16)), phase)  # (T,10)
    actual = np.repeat(ref[:, None, :], N, axis=1)
    mask = np.ones((T, N), dtype=bool)
    check("ref RMS zero for perfect tracking",
          reference_tracking_rms_deg(actual, ref, mask) == 0.0)
    rms = reference_tracking_rms_deg(actual + 0.1, ref, mask)
    check("ref RMS = degrees(0.1) for +0.1 rad offset",
          np.isclose(rms, math.degrees(0.1)))
    # Mask invariance: corrupt masked-out env with garbage.
    mask2 = mask.copy()
    mask2[:, 3] = False
    actual2 = actual + 0.1
    actual2[:, 3, :] = 1e9
    check("ref RMS ignores masked-out samples",
          np.isclose(reference_tracking_rms_deg(actual2, ref, mask2),
                     math.degrees(0.1)))

    # --- Fall metrics -------------------------------------------------
    fm = fall_metrics(np.array([True, False, False, False]),
                      np.array([500, 1500, 1500, 1500]), 0.02)
    check("fall rate 25% / mean episode length 25 s",
          np.isclose(fm["fall_rate_pct"], 25.0)
          and np.isclose(fm["mean_episode_length_s"], 25.0)
          and fm["episodes"] == 4 and fm["falls"] == 1)

    # --- Action smoothness ---------------------------------------------
    T = 10
    tt = np.arange(T, dtype=np.float64)
    acts = np.zeros((T, 1, 2))
    acts[:, 0, 0] = tt**2   # second difference = 2 -> squared jerk 4
    acts[:, 0, 1] = 3 * tt  # linear ramp -> second difference 0
    sm = action_smoothness_metrics(acts, np.ones((T, 1), dtype=bool))
    check("mean squared jerk: quadratic+ramp -> 4.0",
          np.isclose(sm["mean_squared_jerk"], 4.0))
    sm_const = action_smoothness_metrics(np.ones((T, 2, 3)),
                                         np.ones((T, 2), dtype=bool))
    check("constant actions -> zero jerk and zero std",
          sm_const["mean_squared_jerk"] == 0.0 and sm_const["action_std"] == 0.0)

    # --- Stance duty ----------------------------------------------------
    forces = np.zeros((10, 1, 2))
    forces[:6, 0, 0] = 5.0  # left in contact 60%
    forces[:4, 0, 1] = 5.0  # right in contact 40%
    sd = stance_duty_metrics(forces, np.ones((10, 1), dtype=bool), threshold=1.0)
    check("stance duty 60/40, asymmetry 20 pp",
          np.isclose(sd["stance_duty_left_pct"], 60.0)
          and np.isclose(sd["stance_duty_right_pct"], 40.0)
          and np.isclose(sd["stance_duty_asymmetry_pp"], 20.0))

    # --- ROM symmetry ---------------------------------------------------
    T = 1000
    wave = np.sin(2 * np.pi * np.arange(T) / 100.0)
    pos = np.zeros((T, 2, 10))
    pos[:, :, :5] = wave[:, None, None] * 1.0   # left joints, amplitude 1.0
    pos[:, :, 5:] = wave[:, None, None] * 0.5   # right joints, amplitude 0.5
    per_pair, rom_mean = rom_symmetry_metrics(pos, np.ones((T, 2), dtype=bool))
    check("ROM ratio L/R = 2.0 for 2x left amplitude",
          np.allclose(list(per_pair.values()), 2.0) and np.isclose(rom_mean, 2.0))
    check("ROM pair names", list(per_pair.keys()) == LEG_PAIR_NAMES)

    # --- Velocity tracking ----------------------------------------------
    cmd = np.array([0.2, 0.0, 0.3])
    lin = np.zeros((5, 3, 2))
    lin[..., 0] = 0.2 + 0.1  # +0.1 m/s error in x
    ang = np.full((5, 3), 0.3 - 0.2)  # -0.2 rad/s error
    vt = velocity_tracking_metrics(lin, ang, cmd, np.ones((5, 3), dtype=bool))
    check("velocity tracking errors 0.1 m/s and 0.2 rad/s",
          np.isclose(vt["lin_vel_xy_error_mps"], 0.1)
          and np.isclose(vt["ang_vel_z_error_radps"], 0.2))

    # --- Energy proxy ------------------------------------------------------
    tau = np.full((5, 2, 16), 2.0)
    qd = np.full((5, 2, 16), -3.0)
    check("energy proxy sum |tau*qdot| = 96 W",
          np.isclose(energy_proxy_metric(tau, qd, np.ones((5, 2), dtype=bool)),
                     96.0))

    # --- Nearest motion + condition keys -----------------------------------
    vels = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    check("nearest motion index",
          nearest_motion_index(vels, np.array([0.19, 0.0, 0.0])) == 1)
    check("condition key format",
          condition_key((0.15, 0.05, 0.2)) == "vx+0.15_vy+0.05_wz+0.20")

    # --- compute_condition_metrics end-to-end on synthetic record ----------
    T, N, A, J = 60, 3, 16, 16
    nb = 27
    phase = (np.arange(T) % nb) / nb
    ref = evaluate_polynomial_reference(rand_coeffs[:1].repeat(10, axis=0), phase)
    rec = {
        "actions": rng.normal(size=(T, N, A)) * 0.1,
        "leg_pos": np.repeat(ref[:, None, :], N, axis=1) + 0.05,
        "joint_vel": rng.normal(size=(T, N, J)),
        "torque": rng.normal(size=(T, N, J)),
        "foot_force": np.abs(rng.normal(size=(T, N, 2))) * 3.0,
        "lin_vel_xy": np.tile(np.array([0.2, 0.0]), (T, N, 1)),
        "ang_vel_z": np.zeros((T, N)),
        "mask": np.ones((T, N), dtype=bool),
        "fall": np.array([False, True, False]),
        "ep_len": np.array([T, 30, T]),
    }
    rec["mask"][30:, 1] = False  # env 1 fell at step 30
    m = compute_condition_metrics(rec, ref, np.array([0.2, 0.0, 0.0]), 0.02, 1.0)
    expected_keys = set(SCALAR_METRIC_KEYS) | {
        "episodes", "falls", "rom_ratio_per_pair", "command",
    }
    check("compute_condition_metrics returns all protocol metrics",
          expected_keys.issubset(m.keys()))
    check("compute_condition_metrics values finite",
          all(np.isfinite(m[k]) for k in SCALAR_METRIC_KEYS))
    check("compute_condition_metrics ref RMS ~ degrees(0.05)",
          np.isclose(m["reference_tracking_rms_deg"], math.degrees(0.05)))

    agg = aggregate_metrics({"c1": m, "c2": m})
    check("aggregate_metrics averages scalars and sums episode counts",
          np.isclose(agg["fall_rate_pct"], m["fall_rate_pct"])
          and agg["episodes"] == 2 * m["episodes"])

    # --- Markdown rendering ---------------------------------------------
    entries = [
        {"name": "ppo_v3", "framework": "rsl_rl", "aggregate": agg,
         "per_condition": {"vx+0.20_vy+0.00_wz+0.00": m}},
        {"name": "amp_v1", "framework": "skrl_amp", "aggregate": {},
         "per_condition": {}},
    ]
    md = render_results_section(entries)
    check("markdown table lists policies and condition columns",
          "| ppo_v3 " in md and "| amp_v1 " in md
          and "vx+0.20_vy+0.00_wz+0.00" in md)
    check("markdown maps missing metrics to n/a", "n/a" in md)

    # --- Real gait library (skipped gracefully if the pkl is absent) -----
    if os.path.isfile(DEFAULT_REFERENCE_PKL):
        lib = load_reference_library(DEFAULT_REFERENCE_PKL)
        check("library: 240 motions x 10 leg joints x 16 coefficients",
              lib["velocities"].shape == (240, 3)
              and lib["leg_coefficients"].shape == (240, 10, 16)
              and lib["nb_steps"].shape == (240,))
        midx = nearest_motion_index(lib["velocities"], np.array([0.2, 0.0, 0.0]))
        nb = int(lib["nb_steps"][midx])
        phase = (np.arange(2 * nb) % nb) / nb  # two full gait cycles
        ref = evaluate_polynomial_reference(lib["leg_coefficients"][midx], phase)
        step_delta = np.max(np.abs(np.diff(ref[:nb], axis=0)))
        wrap_delta = np.max(np.abs(ref[nb] - ref[0]))
        check(
            "library reference finite, bounded, continuous over the cycle "
            f"(nearest to (0.2,0,0): '{lib['keys'][midx]}', nb_steps={nb}, "
            f"max step delta {step_delta:.3f} rad, wrap delta {wrap_delta:.3f})",
            bool(np.all(np.isfinite(ref)))
            and float(np.max(np.abs(ref))) < 3.0
            and step_delta < 0.5
            and wrap_delta == 0.0,
        )
    else:
        print(f"[SKIP] gait library not found at {DEFAULT_REFERENCE_PKL}")

    print(f"\nSelf-test: {'OK' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__" and "--self-test" in sys.argv:
    sys.exit(run_self_test())


# ======================================================================
# Isaac Sim section — everything below requires the simulator.
#
# AppLauncher boilerplate copied from IsaacLab
# scripts/reinforcement_learning/rsl_rl/play.py: the app MUST be launched
# before importing torch / isaaclab.envs / isaaclab_rl / isaaclab_tasks.
# ======================================================================

"""Launch Isaac Sim Simulator first."""

# Register the Open Duck tasks BEFORE gym.make (string entry points only —
# no heavy imports happen here; pattern from scripts/play_policy.py).
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import isaac_lab_env  # noqa: F401, E402 — triggers gym.register()

from isaaclab.app import AppLauncher  # noqa: E402

parser = build_arg_parser()
# append AppLauncher cli args (--headless, --device, ...)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown_args = parser.parse_known_args()
if _unknown_args:
    print(f"[WARN] Ignoring unrecognized arguments: {_unknown_args}")
if not args_cli.policies:
    parser.error(
        "at least one --policies NAME=TASK_ID:FRAMEWORK:CHECKPOINT[:AGENT_CFG] "
        "is required (or use --self-test)"
    )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata  # noqa: E402

RSL_RL_VERSION = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from skrl.utils.runner.torch import Runner as SkrlRunner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import (  # noqa: E402
    load_cfg_from_registry,
    parse_env_cfg,
)


# ----------------------------------------------------------------------
# Policy adapters
# ----------------------------------------------------------------------


class RslRlPolicy:
    """Adapter for RSL-RL checkpoints (.pt) loaded via OnPolicyRunner.

    Mirrors IsaacLab rsl_rl/play.py: the handle_deprecated_rsl_rl_cfg shim
    is REQUIRED with rsl-rl-lib 5.0.1 — loading the runner config without
    it crashes with KeyError 'class_name'.
    """

    def __init__(self, spec: PolicySpec, gym_env):
        self.spec = spec
        self.base_env = gym_env.unwrapped

        entry_point = spec.agent_cfg or "rsl_rl_cfg_entry_point"
        agent_cfg = load_cfg_from_registry(spec.task_id.split(":")[-1], entry_point)
        # handle deprecated configurations (rsl-rl-lib >= 5.0 renamed keys;
        # without this shim OnPolicyRunner raises KeyError: 'class_name')
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)

        self.env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        self.runner = OnPolicyRunner(
            self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        resume_path = retrieve_file_path(spec.checkpoint)
        print(f"[INFO] Loading RSL-RL checkpoint: {resume_path}")
        self.runner.load(resume_path)
        # deterministic inference policy (mean actions + obs normalization)
        self.policy = self.runner.get_inference_policy(device=self.base_env.device)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def act(self, obs) -> torch.Tensor:
        return self.policy(obs)

    def step(self, actions: torch.Tensor):
        obs, _, dones, extras = self.env.step(actions)
        if hasattr(self.base_env, "termination_manager"):
            terminated = self.base_env.termination_manager.terminated
            truncated = self.base_env.termination_manager.time_outs
        else:
            truncated = extras.get("time_outs", torch.zeros_like(dones)).bool()
            terminated = dones.bool() & ~truncated
        # reset recurrent states for episodes that have terminated (no-op for MLP)
        if hasattr(self.policy, "reset"):
            self.policy.reset(dones)
        return (
            obs,
            terminated.view(-1).cpu().numpy().astype(bool),
            truncated.view(-1).cpu().numpy().astype(bool),
        )


class SkrlAmpPolicy:
    """Adapter for skrl AMP checkpoints loaded via the skrl Runner.

    Mirrors IsaacLab skrl/play.py: the agent config comes from either a yaml
    file path or a gym-registry entry point, the Runner builds the agent,
    and deterministic evaluation uses the policy's mean actions.
    """

    def __init__(self, spec: PolicySpec, gym_env):
        self.spec = spec
        self.base_env = gym_env.unwrapped

        agent_cfg_spec = spec.agent_cfg or "skrl_amp_cfg_entry_point"
        if agent_cfg_spec.endswith((".yaml", ".yml")) and os.path.isfile(agent_cfg_spec):
            experiment_cfg = SkrlRunner.load_cfg_from_yaml(agent_cfg_spec)
            if not experiment_cfg:
                raise RuntimeError(f"Failed to load skrl agent yaml: {agent_cfg_spec}")
        else:
            experiment_cfg = load_cfg_from_registry(
                spec.task_id.split(":")[-1], agent_cfg_spec
            )

        # evaluation mode: no env auto-close, no logging, no checkpoints
        experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
        agent_experiment = experiment_cfg.setdefault("agent", {}).setdefault(
            "experiment", {}
        )
        agent_experiment["write_interval"] = 0
        agent_experiment["checkpoint_interval"] = 0

        self.env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")
        self.runner = SkrlRunner(self.env, experiment_cfg)
        resume_path = os.path.abspath(spec.checkpoint)
        print(f"[INFO] Loading skrl checkpoint: {resume_path}")
        self.runner.agent.load(resume_path)
        self.runner.agent.set_running_mode("eval")

    def reset(self):
        # skrl's IsaacLabWrapper.reset() only resets the sim ONCE and then
        # returns cached observations — force a true reset for every
        # evaluation window by re-arming its private flag.
        if hasattr(self.env, "_reset_once"):
            self.env._reset_once = True
        obs, _ = self.env.reset()
        return obs

    def act(self, obs) -> torch.Tensor:
        outputs = self.runner.agent.act(obs, timestep=0, timesteps=0)
        # deterministic: mean actions when the model exposes them
        return outputs[-1].get("mean_actions", outputs[0])

    def step(self, actions: torch.Tensor):
        obs, _, terminated, truncated, _ = self.env.step(actions)
        return (
            obs,
            terminated.view(-1).cpu().numpy().astype(bool),
            truncated.view(-1).cpu().numpy().astype(bool),
        )


ADAPTER_CLASSES = {"rsl_rl": RslRlPolicy, "skrl_amp": SkrlAmpPolicy}


# ----------------------------------------------------------------------
# Environment setup helpers
# ----------------------------------------------------------------------


@dataclass
class EnvMappings:
    """Runtime index mappings resolved from the live environment."""

    leg_joint_idx: list[int] = field(default_factory=list)  # isaac idx of LEG_JOINT_NAMES
    foot_body_ids: list[int] = field(default_factory=list)  # [left, right] in contact sensor
    num_joints: int = 0


def build_eval_env_cfg(task_id: str):
    """Parse the task's env cfg and apply the evaluation protocol overrides."""
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = float(args_cli.episode_length)
    env_cfg.seed = args_cli.seed
    # Protocol: no observation corruption (manager-based envs only).
    try:
        env_cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass
    # Protocol: no external pushes during evaluation.
    if hasattr(env_cfg, "events"):
        for event_name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
    return env_cfg


def build_env_mappings(base_env) -> EnvMappings:
    """Resolve joint and contact-body indices from the live environment."""
    robot = base_env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    missing = [n for n in LEG_JOINT_NAMES if n not in joint_names]
    if missing:
        raise RuntimeError(f"Robot is missing expected leg joints: {missing}")
    leg_joint_idx = [joint_names.index(n) for n in LEG_JOINT_NAMES]

    contact_sensor = base_env.scene["contact_forces"]
    try:
        foot_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    except TypeError:  # older find_bodies signature without preserve_order
        foot_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES)
    if len(foot_ids) != 2:
        raise RuntimeError(f"Expected 2 foot bodies {FOOT_BODY_NAMES}, got {foot_ids}")

    return EnvMappings(
        leg_joint_idx=leg_joint_idx,
        foot_body_ids=list(foot_ids),
        num_joints=int(robot.num_joints),
    )


def apply_condition(base_env, command_name: str, cond: tuple[float, float, float]):
    """Force a fixed velocity command via degenerate command ranges.

    Mutating the live command term's cfg is safe: UniformVelocityCommand
    reads cfg.ranges / cfg.heading_command / cfg.rel_standing_envs
    dynamically on every resample, and the subsequent env reset resamples
    all envs — so every env gets exactly ``cond`` for the whole window.
    """
    if not hasattr(base_env, "command_manager"):
        # Direct-workflow hook (DuckAmpEnv): commands live in env._commands
        # and are resampled from cfg.command_*_range. Pin the ranges to the
        # condition and disable resampling churn; the env's _reset_idx /
        # _resample_commands then deal exactly `cond` to every env.
        if hasattr(base_env, "_commands") and hasattr(base_env.cfg, "command_vx_range"):
            base_env.cfg.command_vx_range = (cond[0], cond[0])
            base_env.cfg.command_vy_range = (cond[1], cond[1])
            base_env.cfg.command_wz_range = (cond[2], cond[2])
            base_env._commands[:, 0] = cond[0]
            base_env._commands[:, 1] = cond[1]
            base_env._commands[:, 2] = cond[2]
            return
        raise NotImplementedError(
            "Evaluation requires a command manager with a velocity command term "
            f"named '{command_name}' (direct-workflow envs need their own hook)."
        )
    term = base_env.command_manager.get_term(command_name)
    term.cfg.heading_command = False  # don't let heading control overwrite wz
    term.cfg.rel_standing_envs = 0.0  # no standing envs during evaluation
    term.cfg.ranges.lin_vel_x = (cond[0], cond[0])
    term.cfg.ranges.lin_vel_y = (cond[1], cond[1])
    term.cfg.ranges.ang_vel_z = (cond[2], cond[2])


# ----------------------------------------------------------------------
# Rollout recording
# ----------------------------------------------------------------------


def run_window(adapter, T: int, maps: EnvMappings) -> dict:
    """Run one episode window of T steps; record per-step data to CPU numpy.

    Each env contributes exactly one episode: from the step an env first
    terminates (fall) or truncates, its mask goes False permanently so
    auto-reset data never contaminates the metrics. The post-step state at
    a done step is already the RESET state, hence it is masked out too.
    """
    base_env = adapter.base_env
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene["contact_forces"]
    N = base_env.num_envs
    J = maps.num_joints

    bufs: dict | None = None
    mask = np.zeros((T, N), dtype=bool)
    fall = np.zeros(N, dtype=bool)
    ep_len = np.full(N, T, dtype=np.int64)
    done_seen = np.zeros(N, dtype=bool)

    obs = adapter.reset()
    for t in range(T):
        # no_grad, NOT inference_mode: stepping the env inside inference_mode
        # marks lazily-created sim-state tensors as inference tensors, and the
        # next out-of-scope env.reset() then fails on its in-place writes
        # ("Inplace update to inference tensor outside InferenceMode").
        with torch.no_grad():
            actions = adapter.act(obs)
            act_np = actions.detach().view(N, -1).cpu().numpy()
            if bufs is None:
                A = act_np.shape[1]
                bufs = {
                    "actions": np.zeros((T, N, A), dtype=np.float32),
                    "leg_pos": np.zeros((T, N, len(LEG_JOINT_NAMES)), dtype=np.float32),
                    "joint_vel": np.zeros((T, N, J), dtype=np.float32),
                    "torque": np.zeros((T, N, J), dtype=np.float32),
                    "foot_force": np.zeros((T, N, 2), dtype=np.float32),
                    "lin_vel_xy": np.zeros((T, N, 2), dtype=np.float32),
                    "ang_vel_z": np.zeros((T, N), dtype=np.float32),
                }
            bufs["actions"][t] = act_np

            obs, terminated, truncated = adapter.step(actions)

            # Post-step robot state (done envs hold reset state — masked out).
            data = robot.data
            bufs["leg_pos"][t] = data.joint_pos[:, maps.leg_joint_idx].cpu().numpy()
            bufs["joint_vel"][t] = data.joint_vel.cpu().numpy()
            bufs["torque"][t] = data.applied_torque.cpu().numpy()
            bufs["lin_vel_xy"][t] = data.root_lin_vel_b[:, :2].cpu().numpy()
            bufs["ang_vel_z"][t] = data.root_ang_vel_w[:, 2].cpu().numpy()
            foot_forces = contact_sensor.data.net_forces_w[:, maps.foot_body_ids, :]
            bufs["foot_force"][t] = torch.norm(foot_forces, dim=-1).cpu().numpy()

        done = terminated | truncated
        fall |= terminated & ~done_seen  # fall = first done was a termination
        newly_done = done & ~done_seen
        ep_len[newly_done] = t + 1
        done_seen |= done
        mask[t] = ~done_seen

    bufs["mask"] = mask
    bufs["fall"] = fall
    bufs["ep_len"] = ep_len
    return bufs


def concat_windows(windows: list[dict]) -> dict:
    """Concatenate window recordings along the episode axis."""
    out = {}
    for key in windows[0]:
        axis = 0 if windows[0][key].ndim == 1 else 1
        out[key] = np.concatenate([w[key] for w in windows], axis=axis)
    return out


# ----------------------------------------------------------------------
# Evaluation driver
# ----------------------------------------------------------------------


def evaluate_policy(adapter, conditions: list[tuple[float, float, float]],
                    library: dict) -> dict:
    """Apply the full protocol to one policy adapter."""
    base_env = adapter.base_env
    step_dt = float(base_env.step_dt)
    T = int(round(args_cli.episode_length / step_dt))
    maps = build_env_mappings(base_env)

    per_condition = {}
    for cond in conditions:
        key = condition_key(cond)
        motion_idx = nearest_motion_index(library["velocities"], np.asarray(cond))
        nb_steps = int(library["nb_steps"][motion_idx])
        # CORRECTED normalized phase convention: t = (i % nb_steps) / nb_steps.
        # The per-step index i matches ImitationReward._step_idx, which is 0
        # at reset and increments once per control step.
        phase = (np.arange(T) % nb_steps) / float(nb_steps)
        ref_leg_pos = evaluate_polynomial_reference(
            library["leg_coefficients"][motion_idx], phase
        )

        print(
            f"[INFO] [{adapter.spec.name}] condition {key}: "
            f"{args_cli.episodes} windows x {T} steps x {base_env.num_envs} envs "
            f"(reference motion '{library['keys'][motion_idx]}')"
        )
        windows = []
        for w in range(args_cli.episodes):
            apply_condition(base_env, args_cli.command_name, cond)
            windows.append(run_window(adapter, T, maps))
            n_falls = int(np.count_nonzero(windows[-1]["fall"]))
            print(f"[INFO]   window {w + 1}/{args_cli.episodes}: {n_falls} falls")

        rec = concat_windows(windows)
        metrics = compute_condition_metrics(
            rec, ref_leg_pos, np.asarray(cond), step_dt, args_cli.contact_threshold
        )
        metrics["reference_motion"] = {
            "key": library["keys"][motion_idx],
            "velocity": library["velocities"][motion_idx].tolist(),
            "nb_steps_in_period": nb_steps,
        }
        per_condition[key] = metrics

    return {
        "name": adapter.spec.name,
        "task_id": adapter.spec.task_id,
        "framework": adapter.spec.framework,
        "checkpoint": adapter.spec.checkpoint,
        "agent_cfg": adapter.spec.agent_cfg,
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "windows_per_condition": args_cli.episodes,
            "env_episodes_per_condition": args_cli.episodes * base_env.num_envs,
            "episode_length_s": args_cli.episode_length,
            "num_envs": base_env.num_envs,
            "step_dt": step_dt,
            "contact_force_threshold_n": args_cli.contact_threshold,
            "deterministic": True,
            "obs_corruption": False,
            "seed": args_cli.seed,
            "conditions": [list(c) for c in conditions],
            "reference_pkl": os.path.abspath(args_cli.reference_pkl),
        },
        "per_condition": per_condition,
        "aggregate": aggregate_metrics(per_condition),
    }


def main():
    """Evaluate every requested policy and write JSON + markdown outputs."""
    conditions = parse_conditions(args_cli.conditions)
    specs = [parse_policy_spec(s) for s in args_cli.policies]
    library = load_reference_library(args_cli.reference_pkl)
    torch.manual_seed(args_cli.seed)

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Group policies by task id: each obs-space family gets its own env.
    task_ids = list(dict.fromkeys(s.task_id for s in specs))
    for task_id in task_ids:
        env_cfg = build_eval_env_cfg(task_id)
        print(f"[INFO] Creating environment '{task_id}' "
              f"({args_cli.num_envs} envs, {args_cli.episode_length} s episodes)")
        gym_env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        try:
            for spec in (s for s in specs if s.task_id == task_id):
                print(f"\n[INFO] ===== Evaluating '{spec.name}' "
                      f"({spec.framework}, task '{task_id}') =====")
                adapter = ADAPTER_CLASSES[spec.framework](spec, gym_env)
                result = evaluate_policy(adapter, conditions, library)
                json_path = write_policy_json(output_dir, result)
                print(f"[INFO] Wrote {json_path}")
        finally:
            gym_env.close()

    md_path = write_comparison_markdown(args_cli.comparison_md, output_dir)
    print(f"[INFO] Wrote comparison table to {md_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
