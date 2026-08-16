#!/usr/bin/env python3
"""Summarise the S.2 reference traces — action rate and target excursion.

Task S.11 rung 2's abort criterion is defined against the SIM action-rate p95:
the bring-up ladder aborts if the real robot's action rate exceeds what the
policy does in simulation. That number therefore has to exist, and be citable,
before S.11 can be written at all.

    python3 scripts/reference_trace_stats.py

Writes docs/jetson-mod/sim2real/reference_trace_stats.json. Pure numpy.
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM2REAL = os.path.join(REPO, "docs", "jetson-mod", "sim2real")
CONTRACT = os.path.join(REPO, "jetson_runtime", "policy_contract.json")


def main() -> int:
    c = json.load(open(CONTRACT))
    order, q = c["joint_order"], np.array(c["q_default_rad"])
    out = {"_note": ("Action rate is |a_t - a_{t-1}| on the RAW policy output. "
                     "Multiply by action_scale to get radians of joint target. "
                     "S.11 rung 2 aborts if the real robot exceeds the p95 here."),
           "action_scale": c["action_scale"], "joint_order": order, "traces": {}}

    for name in ("fwd", "turn", "stand"):
        p = os.path.join(SIM2REAL, f"reference_trace_v7_{name}.npz")
        if not os.path.isfile(p):
            print(f"  skip {name}: not recorded")
            continue
        d = np.load(p, allow_pickle=True)
        m = d["meta"].item()
        a = d["action"]
        rate = np.abs(np.diff(a, axis=0))                    # (steps-1, 14)
        tgt_dev = np.abs(d["joint_pos_target"] - q)          # radians from default

        out["traces"][name] = {
            "command": m["command"],
            "steps": int(a.shape[0]),
            "obs_joint_block_total_lag": m.get("obs_joint_block_total_lag"),
            "plant7_delay_steps": m.get("latency_steps_drawn"),
            "action_rate": {
                "p50": float(np.percentile(rate, 50)),
                "p95": float(np.percentile(rate, 95)),
                "max": float(rate.max()),
                "p95_per_joint": {j: float(np.percentile(rate[:, i], 95))
                                  for i, j in enumerate(order)},
            },
            "joint_target_dev_from_default_rad": {
                "max": float(tgt_dev.max()),
                "max_per_joint": {j: float(tgt_dev[:, i].max()) for i, j in enumerate(order)},
            },
            "applied_torque_rms_per_joint": {
                j: float(np.sqrt((d["applied_torque"][:, i] ** 2).mean()))
                for i, j in enumerate(order)},
        }
        r = out["traces"][name]["action_rate"]
        print(f"  {name:6s} action-rate p50 {r['p50']:.4f}  p95 {r['p95']:.4f}  "
              f"max {r['max']:.4f}   (x{c['action_scale']} = "
              f"{r['p95']*c['action_scale']:.4f} rad/step at p95)")

    if not out["traces"]:
        print("no traces found", flush=True)
        return 1
    dst = os.path.join(SIM2REAL, "reference_trace_stats.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.relpath(dst, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
