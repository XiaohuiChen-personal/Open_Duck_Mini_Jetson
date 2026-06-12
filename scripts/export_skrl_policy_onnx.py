#!/usr/bin/env python3
# Copyright (c) 2026, Open Duck Mini Jetson Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained skrl AMP policy checkpoint to ONNX for Jetson deployment.

Design intent
-------------
skrl whole-agent checkpoints (``agent_*.pt`` / ``best_agent.pt``, written by
``skrl.agents.torch.base.Agent.save``) are a plain dict of state_dicts::

    {"policy": ..., "value": ..., "discriminator": ..., "optimizer": ...,
     "state_preprocessor": ..., "value_preprocessor": ...,
     "amp_state_preprocessor": ...}

For deployment we only need the deterministic policy path:

    obs -> RunningStandardScaler (frozen mean/var) -> policy MLP -> mean action

This script reconstructs that path as a plain ``nn.Module`` with **no skrl
dependency** (skrl is not installed on the Jetson), loads the trained weights
from the checkpoint, and exports with the **legacy TorchScript ONNX exporter**
(``dynamo=False``) — the dynamo exporter requires ``onnxscript``, which may be
unavailable on aarch64. The resulting .onnx is the direct input to
``trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16`` on the Jetson.

Reconstruction notes (skrl 1.4.3 model_instantiator naming):

* Separate Gaussian policy (our AMP cfg, ``models.separate: True``): a single
  ``nn.Sequential`` named ``net_container`` whose last Linear is the action
  head (the instantiator embeds the output layer), plus ``log_std_parameter``.
  The log_std is dropped — deployment uses the distribution mean.
* Shared model fallback: trunk ``net_container`` (ends with an activation)
  followed by a ``policy_layer`` Linear head.
* ``RunningStandardScaler`` buffers are ``running_mean`` / ``running_variance``
  / ``current_count`` (float64); its eval-mode forward is
  ``clamp((x - mean.float()) / (sqrt(var.float()) + eps), -clip, clip)``,
  which is reproduced here exactly with frozen float32 buffers.

With ``--bake-action-postproc`` the Isaac Lab JointPositionAction
post-processing (``target = init_pos + action_scale * action``) is baked into
the graph so the Jetson runtime can feed servo targets directly. ``init_pos``
comes from a JSON sidecar (see ``scripts/duck_init_pos.json``) listing the 16
joints in Isaac Lab joint order.

After export the graph is verified in-process: N random observations are run
through both the torch module and the ONNX model (onnxruntime if available,
else onnx's pure-python ReferenceEvaluator) and the max abs difference must be
< 1e-5 with no NaN. A sidecar ``<out>.meta.json`` documents the observation
layout, joint order, post-processing status and verification result.

Usage::

    /home/xiaohui_chen/IsaacLab/isaaclab.sh -p \\
        scripts/export_skrl_policy_onnx.py \\
        --checkpoint logs/skrl/<run>/checkpoints/best_agent.pt \\
        --obs-dim 62 --act-dim 16 \\
        --out exported_policies/open_duck_amp.onnx \\
        --bake-action-postproc

No GPU is touched: everything runs on CPU (safe while training is running).
"""

import argparse
from collections import OrderedDict
import datetime
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# --- Constants -------------------------------------------------------------

# RunningStandardScaler defaults (skrl 1.4.3); the AMP cfg passes
# state_preprocessor_kwargs: null, so the defaults are what training used.
SCALER_EPSILON = 1e-8
SCALER_CLIP_THRESHOLD = 5.0

# Isaac Lab / USD articulation joint order (from the MJCF->USD conversion,
# interleaved). This is the order of the 16 action outputs and of the
# joint_pos / joint_vel observation blocks.
ISAAC_LAB_JOINT_ORDER: List[str] = [
    "left_hip_yaw", "neck_pitch", "right_hip_yaw", "left_hip_roll",
    "head_pitch", "right_hip_roll", "left_hip_pitch", "head_yaw",
    "right_hip_pitch", "left_knee", "head_roll", "right_knee",
    "left_ankle", "left_antenna", "right_antenna", "right_ankle",
]

# Policy observation layout for the OpenDuck velocity env (env_cfg.py:
# LocomotionVelocityRoughEnvCfg terms, height_scan removed, gait_phase
# appended). Only emitted in the meta sidecar when obs_dim matches.
OBS_LAYOUT_62: List[Dict[str, Union[str, int]]] = [
    {"name": "base_lin_vel", "start": 0, "size": 3, "doc": "base linear velocity, body frame (m/s)"},
    {"name": "base_ang_vel", "start": 3, "size": 3, "doc": "base angular velocity, body frame (rad/s)"},
    {"name": "projected_gravity", "start": 6, "size": 3, "doc": "gravity direction in body frame (unit)"},
    {"name": "velocity_commands", "start": 9, "size": 3, "doc": "commanded (vx, vy, yaw_rate)"},
    {"name": "joint_pos", "start": 12, "size": 16, "doc": "joint positions relative to default, Isaac Lab joint order (rad)"},
    {"name": "joint_vel", "start": 28, "size": 16, "doc": "joint velocities, Isaac Lab joint order (rad/s)"},
    {"name": "actions", "start": 44, "size": 16, "doc": "previous policy action (unscaled)"},
    {"name": "gait_phase", "start": 60, "size": 2, "doc": "[cos(phase), sin(phase)] of the 0.54 s gait cycle"},
]

# Activation names supported by skrl's model_instantiator yaml schema.
ACTIVATIONS: Dict[str, type] = {
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
}


# --- Deployable module -----------------------------------------------------


class FrozenObservationScaler(nn.Module):
    """Inference-time replica of skrl's RunningStandardScaler.

    The running statistics are frozen into float32 buffers at export time.
    The op order matches skrl's eval path exactly (float-cast *before* sqrt,
    epsilon added to the std, symmetric clamp) so torch/ONNX outputs are
    bit-comparable with the training-time preprocessor.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, clip_threshold: float) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.clip_threshold = float(clip_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.mean) / self.std,
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )


class PolicyMLP(nn.Module):
    """Container whose attribute names mirror skrl's instantiated policy.

    Keeping the names (``net_container``, optional ``policy_layer``) lets us
    load the checkpoint's policy state_dict without any key remapping; the
    ``log_std_parameter`` is intentionally absent (mean action only).
    """

    def __init__(self, net_container: nn.Sequential, policy_layer: Optional[nn.Linear]) -> None:
        super().__init__()
        self.net_container = net_container
        self.policy_layer = policy_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net_container(x)
        if self.policy_layer is not None:
            x = self.policy_layer(x)
        return x


class DeployablePolicy(nn.Module):
    """obs -> frozen scaler -> policy MLP -> mean action [-> servo target].

    When ``init_pos`` is given, the Isaac Lab JointPositionAction
    post-processing ``target = init_pos + action_scale * action`` is part of
    the graph and the ONNX output is a joint position target in radians
    (Isaac Lab joint order).
    """

    def __init__(
        self,
        scaler: Optional[FrozenObservationScaler],
        mlp: PolicyMLP,
        init_pos: Optional[torch.Tensor] = None,
        action_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.scaler = scaler
        self.mlp = mlp
        self.bake_postproc = init_pos is not None
        self.action_scale = float(action_scale)
        if init_pos is not None:
            self.register_buffer("init_pos", init_pos)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.scaler(obs) if self.scaler is not None else obs
        action = self.mlp(x)
        if self.bake_postproc:
            return self.init_pos + self.action_scale * action
        return action


# --- Checkpoint reconstruction ---------------------------------------------


def load_skrl_checkpoint(path: str) -> Dict[str, dict]:
    """Load a skrl whole-agent checkpoint (dict of module state_dicts)."""
    # weights_only=False: checkpoint also contains the optimizer state dict
    # (plain python containers); the file comes from our own training runs.
    modules = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(modules, dict) or "policy" not in modules:
        raise SystemExit(
            f"[ERROR] {path} is not a skrl whole-agent checkpoint "
            f"(expected a dict with a 'policy' entry, got keys: "
            f"{list(modules.keys()) if isinstance(modules, dict) else type(modules)})"
        )
    return modules


def build_policy_mlp(
    policy_state_dict: Dict[str, torch.Tensor],
    obs_dim: int,
    act_dim: int,
    activation: str,
) -> PolicyMLP:
    """Rebuild the skrl model_instantiator policy network from its state_dict.

    The Linear layer positions inside ``net_container`` are recovered from the
    state_dict key indices; the gaps between them are the (parameter-free)
    activations, whose type cannot be recovered from weights and must match
    the training yaml (``--activation``).
    """
    if activation not in ACTIVATIONS:
        raise SystemExit(f"[ERROR] Unsupported activation '{activation}'. Choose from {sorted(ACTIVATIONS)}")
    act_cls = ACTIVATIONS[activation]

    linear_indices = sorted(
        int(key.split(".")[1])
        for key in policy_state_dict
        if key.startswith("net_container.") and key.endswith(".weight")
    )
    if not linear_indices:
        raise SystemExit(
            "[ERROR] No 'net_container.*' keys in the policy state_dict. "
            f"Got keys: {list(policy_state_dict.keys())}"
        )

    has_policy_layer = "policy_layer.weight" in policy_state_dict

    modules: "OrderedDict[str, nn.Module]" = OrderedDict()
    for idx in range(max(linear_indices) + 1):
        if idx in linear_indices:
            weight = policy_state_dict[f"net_container.{idx}.weight"]
            modules[str(idx)] = nn.Linear(weight.shape[1], weight.shape[0])
        else:
            modules[str(idx)] = act_cls()
    if has_policy_layer:
        # Shared-model trunk ends with an activation (invisible in the
        # state_dict because it has no parameters) before the policy head.
        modules[str(max(linear_indices) + 1)] = act_cls()

    net_container = nn.Sequential(modules)
    policy_layer: Optional[nn.Linear] = None
    if has_policy_layer:
        weight = policy_state_dict["policy_layer.weight"]
        policy_layer = nn.Linear(weight.shape[1], weight.shape[0])

    mlp = PolicyMLP(net_container, policy_layer)
    result = mlp.load_state_dict(policy_state_dict, strict=False)

    # The only tolerated unexpected keys are the dropped Gaussian log_std and
    # (for shared models) the value head. Anything else means the rebuilt
    # architecture does not match the checkpoint.
    tolerated = {"log_std_parameter"}
    bad_unexpected = [
        k for k in result.unexpected_keys if k not in tolerated and not k.startswith("value_layer.")
    ]
    if result.missing_keys or bad_unexpected:
        raise SystemExit(
            f"[ERROR] Rebuilt policy does not match checkpoint. "
            f"Missing: {result.missing_keys}, unexpected: {bad_unexpected}"
        )

    # Sanity-check I/O dimensions against the CLI contract.
    first_linear = policy_state_dict[f"net_container.{linear_indices[0]}.weight"]
    if first_linear.shape[1] != obs_dim:
        raise SystemExit(
            f"[ERROR] Checkpoint expects obs_dim={first_linear.shape[1]}, but --obs-dim={obs_dim}"
        )
    out_features = (
        policy_layer.out_features if policy_layer is not None
        else policy_state_dict[f"net_container.{max(linear_indices)}.weight"].shape[0]
    )
    if out_features != act_dim:
        raise SystemExit(
            f"[ERROR] Checkpoint policy outputs {out_features} actions, but --act-dim={act_dim}"
        )
    return mlp


def build_observation_scaler(
    preprocessor_state_dict: Optional[Dict[str, torch.Tensor]], obs_dim: int
) -> Optional[FrozenObservationScaler]:
    """Freeze a RunningStandardScaler state_dict into an inference module."""
    if preprocessor_state_dict is None:
        return None
    mean64 = preprocessor_state_dict["running_mean"]
    var64 = preprocessor_state_dict["running_variance"]
    if mean64.numel() != obs_dim:
        raise SystemExit(
            f"[ERROR] state_preprocessor size {mean64.numel()} != --obs-dim {obs_dim}"
        )
    # Match skrl exactly: cast to float32 first, then sqrt, then add epsilon.
    mean = mean64.detach().float().reshape(-1)
    std = var64.detach().float().reshape(-1).sqrt() + SCALER_EPSILON
    return FrozenObservationScaler(mean, std, SCALER_CLIP_THRESHOLD)


def load_init_pos(json_path: str, act_dim: int) -> Tuple[torch.Tensor, List[str]]:
    """Load joint_name->radians JSON and return (init_pos vector, joint order).

    Accepts either the documented format with explicit ``joint_order`` +
    ``init_pos_rad`` keys, or a flat name->radians mapping whose key insertion
    order is taken as the joint order.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "init_pos_rad" in data:
        joint_order = list(data.get("joint_order", data["init_pos_rad"].keys()))
        mapping = data["init_pos_rad"]
    else:
        mapping = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        joint_order = list(mapping.keys())
    if len(joint_order) != act_dim:
        raise SystemExit(
            f"[ERROR] {json_path} lists {len(joint_order)} joints, but --act-dim={act_dim}"
        )
    missing = [name for name in joint_order if name not in mapping]
    if missing:
        raise SystemExit(f"[ERROR] {json_path}: joints in joint_order without values: {missing}")
    init_pos = torch.tensor([float(mapping[name]) for name in joint_order], dtype=torch.float32)
    return init_pos, joint_order


# --- Export and verification ------------------------------------------------


def export_onnx(model: DeployablePolicy, obs_dim: int, out_path: str, opset: int) -> None:
    """Export with the legacy TorchScript exporter (dynamo exporter needs
    onnxscript, which may be missing on aarch64)."""
    model.eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    kwargs = dict(
        input_names=["obs"],
        output_names=["joint_target" if model.bake_postproc else "action"],
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, (dummy,), out_path, dynamo=False, **kwargs)
    except TypeError:
        # Older torch without the `dynamo` kwarg: legacy exporter is default.
        torch.onnx.export(model, (dummy,), out_path, **kwargs)


def verify_onnx(
    onnx_path: str, model: DeployablePolicy, obs_dim: int, num_samples: int
) -> Tuple[float, str, Tuple[int, ...]]:
    """Run random observations through torch and ONNX; return max abs diff.

    Prefers onnxruntime; falls back to onnx's pure-python ReferenceEvaluator
    (numerically equivalent, just slower) when onnxruntime is not installed.
    onnx.checker runs in both cases.
    """
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))

    torch.manual_seed(0)
    obs = torch.randn(num_samples, obs_dim, dtype=torch.float32) * 2.0
    model.eval()
    with torch.no_grad():
        ref = model(obs).cpu().numpy()

    rows = []
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        backend = f"onnxruntime {onnxruntime.__version__}"
        for i in range(num_samples):
            rows.append(session.run(None, {"obs": obs[i : i + 1].numpy()})[0])
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        evaluator = ReferenceEvaluator(onnx_path)
        backend = f"onnx.reference.ReferenceEvaluator (onnx {onnx.__version__}; onnxruntime not installed)"
        for i in range(num_samples):
            rows.append(evaluator.run(None, {"obs": obs[i : i + 1].numpy()})[0])

    import numpy as np

    out = np.concatenate(rows, axis=0)
    if np.isnan(out).any() or np.isnan(ref).any():
        raise SystemExit("[ERROR] NaN detected in torch or ONNX outputs")
    max_abs_diff = float(np.abs(out - ref).max())
    if max_abs_diff >= 1e-5:
        raise SystemExit(f"[ERROR] torch vs ONNX max abs diff {max_abs_diff:.3e} >= 1e-5")
    return max_abs_diff, backend, tuple(rows[0].shape)


def write_meta(
    out_path: str,
    args: argparse.Namespace,
    joint_order: List[str],
    has_scaler: bool,
    init_pos: Optional[torch.Tensor],
    max_abs_diff: float,
    backend: str,
) -> str:
    """Write the <out>.meta.json sidecar describing the deployed interface."""
    checkpoint_mtime = datetime.datetime.fromtimestamp(
        os.path.getmtime(args.checkpoint)
    ).isoformat(timespec="seconds")
    if args.obs_dim == 62:
        obs_layout: Union[List[Dict[str, Union[str, int]]], str] = OBS_LAYOUT_62
    else:
        obs_layout = (
            f"obs_dim={args.obs_dim} does not match the default OpenDuck velocity-env "
            "layout (62); consult the training env_cfg.py observation group for the "
            "term order."
        )
    meta = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_mtime": checkpoint_mtime,
        "obs_dim": args.obs_dim,
        "act_dim": args.act_dim,
        "input": {"name": "obs", "shape": [1, args.obs_dim], "dtype": "float32"},
        "output": {
            "name": "joint_target" if args.bake_action_postproc else "action",
            "shape": [1, args.act_dim],
            "dtype": "float32",
            "doc": (
                "joint position targets in radians, Isaac Lab joint order"
                if args.bake_action_postproc
                else "raw mean action (unscaled); apply target = init_pos + "
                f"{args.action_scale} * action externally"
            ),
        },
        "obs_layout": obs_layout,
        "joint_order": joint_order,
        "observation_normalization": {
            "baked": has_scaler,
            "type": "skrl RunningStandardScaler (frozen running_mean/running_variance)",
            "epsilon": SCALER_EPSILON,
            "clip_threshold": SCALER_CLIP_THRESHOLD,
        },
        "action_postproc": {
            "baked": bool(args.bake_action_postproc),
            "formula": f"target = init_pos + {args.action_scale} * action",
            "action_scale": args.action_scale,
            "init_pos_rad": (
                {name: float(value) for name, value in zip(joint_order, init_pos.tolist())}
                if init_pos is not None
                else None
            ),
        },
        "policy": {
            "type": "skrl GaussianMixin mean (log_std dropped, deterministic)",
            "activation": args.activation,
        },
        "exporter": {
            "torch_version": torch.__version__,
            "exporter": "legacy TorchScript (torch.onnx.export dynamo=False)",
            "opset": args.opset,
        },
        "verification": {
            "backend": backend,
            "num_samples": args.num_verify,
            "max_abs_diff": max_abs_diff,
        },
    }
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta_path


# --- CLI ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a skrl AMP policy checkpoint to ONNX for TensorRT deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="skrl agent_*.pt or best_agent.pt")
    parser.add_argument("--obs-dim", type=int, required=True, help="Policy observation dimension")
    parser.add_argument("--act-dim", type=int, default=16, help="Action dimension")
    parser.add_argument("--out", required=True, help="Output .onnx path")
    parser.add_argument(
        "--bake-action-postproc",
        action="store_true",
        help="Bake target = init_pos + action_scale * action into the graph",
    )
    parser.add_argument(
        "--init-pos-json",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "duck_init_pos.json"),
        help="JSON with joint_order + init_pos_rad (joint_name -> radians)",
    )
    parser.add_argument("--action-scale", type=float, default=0.25, help="Isaac Lab action scale")
    parser.add_argument(
        "--activation",
        default="relu",
        choices=sorted(ACTIVATIONS),
        help="Hidden activation used in the training yaml (not recoverable from weights)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--num-verify", type=int, default=100, help="Random samples for verification")
    args = parser.parse_args(argv)

    modules = load_skrl_checkpoint(args.checkpoint)
    print(f"[INFO] Loaded checkpoint {args.checkpoint} with modules: {sorted(modules.keys())}")

    mlp = build_policy_mlp(modules["policy"], args.obs_dim, args.act_dim, args.activation)
    print(f"[INFO] Rebuilt policy MLP:\n{mlp}")

    scaler = build_observation_scaler(modules.get("state_preprocessor"), args.obs_dim)
    if scaler is None:
        print("[WARN] No 'state_preprocessor' in checkpoint — exporting WITHOUT obs normalization")
    else:
        print("[INFO] Frozen RunningStandardScaler stats baked into the graph")

    init_pos: Optional[torch.Tensor] = None
    joint_order = list(ISAAC_LAB_JOINT_ORDER)
    if args.bake_action_postproc:
        init_pos, joint_order = load_init_pos(args.init_pos_json, args.act_dim)
        print(
            f"[INFO] Baking action postproc: target = init_pos + {args.action_scale} * action "
            f"(init_pos from {args.init_pos_json})"
        )

    model = DeployablePolicy(scaler, mlp, init_pos=init_pos, action_scale=args.action_scale)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    export_onnx(model, args.obs_dim, args.out, args.opset)
    print(f"[INFO] Exported ONNX to {args.out} (opset {args.opset}, legacy TorchScript exporter)")

    max_abs_diff, backend, out_shape = verify_onnx(args.out, model, args.obs_dim, args.num_verify)
    print(
        f"[PASS] Verified {args.num_verify} samples via {backend}: "
        f"output shape {list(out_shape)}, max abs diff {max_abs_diff:.3e} < 1e-5, no NaN"
    )

    meta_path = write_meta(args.out, args, joint_order, scaler is not None, init_pos, max_abs_diff, backend)
    print(f"[INFO] Wrote sidecar {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
