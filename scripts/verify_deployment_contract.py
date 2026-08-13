#!/usr/bin/env python3
"""Verify an exported policy against its deployment contract.

Task R3. `jetson_runtime/` does not exist yet, so none of this is a live bug --
it is the spec the Phase-S runtime must satisfy, produced WITH the policy rather
than reconstructed from memory later.

    ~/IsaacLab/_isaac_sim/python.sh scripts/verify_deployment_contract.py \\
        --policy_dir exported_policies/v6d_contact_wrench_ppo

**Run it under the Isaac interpreter.** Measured: it carries onnx 1.20.1,
torch 2.9.0+cu130 and numpy 1.26.0; the system python3 has no `onnx`. No Isaac
Sim app is launched -- this is pure onnx + torch + numpy.

The three facts this exists to pin (known_issues.md DEPLOY-1/2/4):

1. **`action_scale = 0.25` and `q_default` are NOT in the graph.** Of 197,126
   initializer scalars, zero equal 0.25, and the only action-width tensor is
   `mlp.6.bias`, which differs from `q_default` by up to 1.461333 rad.
   `q_target = q_default + 0.25 * a` lives entirely inside Isaac Lab's
   JointPositionAction. A runtime that commands the ONNX output directly is
   wrong by a 4x gain AND a standing-pose offset up to 1.379 rad -- total,
   silent failure.
2. **The observation's joint block is `joint_pos_rel`**, not raw encoder angles.
   Feeding absolute angles puts the knees 13-14 sigma outside the training
   distribution.
3. **The normaliser epsilon (0.01) is not in the checkpoint.** It is a plain
   Python attribute in rsl_rl/modules/normalization.py, so a hand-rolled
   reimplementation that divides by `_std` alone is off by 28.0 % on one
   channel. rsl-rl's own loader reconstructs it, so policy.pt and policy.onnx
   are correct; only a REIMPLEMENTATION is at risk -- which is exactly what a
   Jetson runtime is.

Exit 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT_POS = os.path.join(REPO, "scripts", "duck_init_pos.json")

EXPECTED_OPS = ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_dir", required=True)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()

    d = os.path.abspath(args.policy_dir)
    contract_path = os.path.join(d, "deployment_contract.json")
    onnx_path = os.path.join(d, "policy.onnx")
    pt_path = os.path.join(d, "policy.pt")
    env_yaml = os.path.join(d, "env.yaml")

    if not os.path.isfile(contract_path):
        print(f"no deployment_contract.json in {d}", file=sys.stderr)
        return 1
    contract = json.load(open(contract_path))
    obs_dim = int(contract["obs_dim"])
    action_dim = int(contract["action_dim"])
    print(f"policy dir : {d}")
    print(f"contract   : obs {obs_dim}, action {action_dim}, "
          f"action_scale {contract['action_scale']}")
    print()

    import numpy as np
    import onnx
    from onnx import numpy_helper

    # External weights sit beside the graph; load with the dir as cwd.
    cwd = os.getcwd()
    os.chdir(d)
    try:
        model = onnx.load("policy.onnx")
    finally:
        os.chdir(cwd)

    # ---- 1. graph shape ---------------------------------------------------
    ops = [n.op_type for n in model.graph.node]
    check("graph op sequence", ops == EXPECTED_OPS,
          f"{ops}" if ops != EXPECTED_OPS else "Sub,Div then 4x Gemm with Elu")

    def dims(vi):
        return [x.dim_value for x in vi.type.tensor_type.shape.dim]

    gi, go = model.graph.input[0], model.graph.output[0]
    check("graph input shape", dims(gi) == [1, obs_dim],
          f"{dims(gi)} vs [1, {obs_dim}]")
    check("graph output shape", dims(go) == [1, action_dim],
          f"{dims(go)} vs [1, {action_dim}]")

    # ---- 2. DEPLOY-1: the scale and offset are genuinely absent ------------
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    total_scalars = sum(a.size for a in inits.values())
    n_quarter = sum(int(np.isclose(a, 0.25, atol=1e-6).sum()) for a in inits.values())
    check("action_scale 0.25 is ABSENT from the graph", n_quarter == 0,
          f"{n_quarter} of {total_scalars} initializer scalars equal 0.25")

    q_default = None
    ip = json.load(open(INIT_POS))
    # duck_init_pos.json carries TWO orders since Task M0b: the 16-joint
    # articulation order and the 14-joint action order. Pick the one whose
    # length matches this policy's action width, so the verifier works on both
    # the pre-M0b (16) and post-M0b (14) exports.
    cands = [ip.get("action_joint_order"), ip.get("joint_order")]
    order = next((c for c in cands if c and len(c) == action_dim), None)
    if order is None:
        print(f"no joint order of length {action_dim} in duck_init_pos.json "
              f"(have {[len(c) for c in cands if c]})", file=sys.stderr)
        return 1
    # TWO precisions on purpose. The graph comparison below is against float32
    # ONNX initializers, so it must be float32. The CONTRACT comparison is
    # JSON-to-JSON and must stay float64 -- comparing a float32 round-trip
    # against the source at 1e-9 fails on 5.15e-08 of representation error,
    # which is not a contract defect.
    q_default_f64 = np.array([ip["init_pos_rad"][j] for j in order], dtype=np.float64)
    q_default = q_default_f64.astype(np.float32)
    worst = None
    for name, a in inits.items():
        if a.size == action_dim:
            delta = float(np.abs(a.reshape(-1) - q_default).max())
            if worst is None or delta < worst[1]:
                worst = (name, delta)
    check("q_default is ABSENT from the graph",
          worst is None or worst[1] > 1e-3,
          f"closest action-width tensor `{worst[0]}` differs by {worst[1]:.6f} rad"
          if worst else "no action-width initializer at all")

    # ---- 3. numpy reimplementation vs the TorchScript export --------------
    try:
        import torch
        ts = torch.jit.load(pt_path)
        ts.eval()

        # Reconstruct the forward pass from the graph: (x - mean) / std, then
        # the MLP. Names come from the initializers themselves, not assumed.
        sub = next(n for n in model.graph.node if n.op_type == "Sub")
        div = next(n for n in model.graph.node if n.op_type == "Div")
        mean = inits[[i for i in sub.input if i in inits][0]].reshape(-1)
        std = inits[[i for i in div.input if i in inits][0]].reshape(-1)
        gemms = [n for n in model.graph.node if n.op_type == "Gemm"]
        layers = []
        for g in gemms:
            W = inits[g.input[1]]
            B = inits[g.input[2]] if len(g.input) > 2 else np.zeros(W.shape[0], np.float32)
            layers.append((W, B))

        def np_forward(x):
            h = (x - mean) / std
            for k, (W, B) in enumerate(layers):
                h = h @ W.T + B
                if k < len(layers) - 1:
                    h = np.where(h > 0, h, np.expm1(h))     # ELU, alpha=1
            return h

        rng = np.random.default_rng(0)
        worst_d = 0.0
        for _ in range(args.samples):
            x = rng.standard_normal((1, obs_dim)).astype(np.float32)
            with torch.inference_mode():
                ref = ts(torch.from_numpy(x)).cpu().numpy()
            worst_d = max(worst_d, float(np.abs(np_forward(x) - ref).max()))
        check(f"numpy reimplementation matches policy.pt over {args.samples} samples",
              worst_d < 1e-5, f"max |delta| = {worst_d:.3e}")
    except Exception as exc:                                   # noqa: BLE001
        check("numpy reimplementation matches policy.pt", False, f"{type(exc).__name__}: {exc}")

    # ---- 4. contract agrees with the repo's own sources -------------------
    check("contract joint_order matches duck_init_pos.json",
          contract["joint_order"] == order,
          f"{len(contract['joint_order'])} vs {len(order)} joints")
    qd = np.array(contract["q_default_rad"], dtype=np.float64)
    check("contract q_default_rad matches duck_init_pos.json",
          qd.shape == q_default_f64.shape
          and bool(np.abs(qd - q_default_f64).max() < 1e-12),
          f"max |delta| = {float(np.abs(qd - q_default_f64).max()):.3e}"
          if qd.shape == q_default_f64.shape
          else f"{qd.shape} vs {q_default_f64.shape}")

    if os.path.isfile(env_yaml):
        try:
            import yaml
            # yaml.safe_load FAILS here: the archived env.yaml carries 81
            # !!python/object and !!python/tuple tags. Do not "fix" the yaml.
            y = yaml.unsafe_load(open(env_yaml))
            scale = y["actions"]["joint_pos"]["scale"]
            check("action_scale matches the archived env.yaml",
                  abs(float(scale) - float(contract["action_scale"])) < 1e-12,
                  f"env.yaml {scale} vs contract {contract['action_scale']}")
        except Exception as exc:                               # noqa: BLE001
            check("action_scale matches the archived env.yaml", False,
                  f"{type(exc).__name__}: {exc}")

    # ---- 5. DEPLOY-4: batch dim is hard-fixed at 1 ------------------------
    try:
        import onnxruntime as ort
        os.chdir(d)
        try:
            sess = ort.InferenceSession("policy.onnx",
                                        providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            try:
                sess.run(None, {name: np.zeros((2, obs_dim), np.float32)})
                batch_ok = True
            except Exception:
                batch_ok = False
        finally:
            os.chdir(cwd)
        print(f"[INFO] DEPLOY-4 batch-2 accepted: {batch_ok} "
              f"({'batch dim is dynamic' if batch_ok else 'batch dim hard-fixed at 1, as recorded'})")
    except ImportError:
        print("[INFO] onnxruntime unavailable; DEPLOY-4 batch check skipped")

    # ---- 6. provenance ----------------------------------------------------
    for key, fname in (("onnx_md5", "policy.onnx"),):
        if contract.get(key) and os.path.isfile(os.path.join(d, fname)):
            got = md5(os.path.join(d, fname))
            check(f"{fname} md5 matches the contract", got == contract[key],
                  f"{got} vs {contract[key]}")

    print()
    failed = [n for n, ok, _ in _results if not ok]
    if failed:
        print(f"VERDICT: FAIL — {len(failed)} check(s): {failed}")
        return 1
    print(f"VERDICT: PASS — {len(_results)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
