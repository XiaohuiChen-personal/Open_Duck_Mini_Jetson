"""Replay a trace through the safety layer, injecting faults — Task S.10.

This is how the safety layer gets tested without endangering the robot. It runs
each fault condition in turn and asserts the monitor reaches the expected verdict
within the expected number of steps.

**It also runs the CLEAN trace and counts false trips**, which is the half that
is easy to skip: a monitor that trips on healthy data is worse than no monitor,
because it teaches the operator to ignore it.

    python3 -m jetson_runtime.tools.fault_injection
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from jetson_runtime.safety import (DERATE, HOLD, OK, TORQUE_OFF, SafetyMonitor)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACT = json.load(open(os.path.join(REPO, "jetson_runtime", "policy_contract.json")))
N_JOINTS = len(CONTRACT["joint_order"])


def _healthy(i, trace):
    """A nominal sensor bundle for step i."""
    return {"gravity_b": trace["projected_gravity_b"][i],
            "temps_c": np.full(N_JOINTS, 32.0),
            "currents_a": np.full(N_JOINTS, 0.4),
            "voltage_v": 11.1,
            "foot_contacts": (True, True),
            "step_elapsed_s": 0.008,
            "obs": trace["obs"][i]}


# Each case mutates the healthy bundle and states what must happen, and by when.
CASES = {
    "tilt": (lambda b: {**b, "gravity_b": np.array([0.0, -0.9, -0.2])},
             TORQUE_OFF, 1),
    "over_temperature": (lambda b: {**b, "temps_c": np.full(N_JOINTS, 70.0)},
                         TORQUE_OFF, 1),
    "temp_warning": (lambda b: {**b, "temps_c": np.full(N_JOINTS, 57.0)},
                     DERATE, 1),
    "over_current": (lambda b: {**b, "currents_a": np.full(N_JOINTS, 4.5)},
                     TORQUE_OFF, 51),      # 1.0 s window at 20 ms steps
    "current_derate": (lambda b: {**b, "currents_a": np.full(N_JOINTS, 3.0)},
                       DERATE, 51),
    "under_voltage": (lambda b: {**b, "voltage_v": 5.5}, TORQUE_OFF, 1),
    "fall_proxy": (lambda b: {**b, "foot_contacts": (False, False),
                              "gravity_b": np.array([0.6, 0.0, -0.5])},
                   TORQUE_OFF, 5),
    "loop_overrun": (lambda b: {**b, "step_elapsed_s": 0.05}, TORQUE_OFF, 3),
    "nan_observation": (lambda b: {**b, "obs": np.full(CONTRACT["obs_dim"], np.nan)},
                        TORQUE_OFF, 4),
}


def run_case(trace, mutate, expect, within):
    mon = SafetyMonitor(CONTRACT)
    for i in range(min(within + 5, trace["obs"].shape[0])):
        st = mon.check(**mutate(_healthy(i, trace)))
        if st.verdict == expect:
            return True, i + 1, st.reasons
    return False, None, mon.state.reasons


def run_clean(trace, limit=1500):
    """False-trip count on healthy data. Must be zero."""
    mon = SafetyMonitor(CONTRACT)
    worst, trips = OK, 0
    for i in range(min(limit, trace["obs"].shape[0])):
        st = mon.check(**_healthy(i, trace))
        if st.verdict != OK:
            trips += 1
            worst = st.verdict
    return trips, worst


def main() -> int:
    traces = sorted(glob.glob(os.path.join(
        REPO, "docs", "jetson-mod", "sim2real", "reference_trace_v7_*.npz")))
    if not traces:
        print("no S.2 traces on disk")
        return 1
    trace = np.load(traces[0], allow_pickle=True)

    print(f"# fault injection over {os.path.basename(traces[0])}\n")
    failures = 0
    print(f"{'case':<20}{'expect':<12}{'within':>7}{'got at':>8}  reasons")
    print("-" * 76)
    for name, (mutate, expect, within) in CASES.items():
        ok, at, reasons = run_case(trace, mutate, expect, within)
        if not ok or at > within:
            failures += 1
        print(f"{name:<20}{expect:<12}{within:>7}{str(at):>8}  "
              f"{';'.join(reasons)[:34]}{'' if ok and at <= within else '   <-- FAIL'}")

    trips, worst = run_clean(trace)
    print(f"\nclean trace: {trips} false trips (worst verdict {worst})")
    if trips:
        failures += 1
        print("  A monitor that trips on healthy data teaches the operator to ignore it.")
    print(f"\n{'PASS' if not failures else f'{failures} FAILURE(S)'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
