"""Inference wrapper with swappable backends — Task S.4.

Two backends, chosen at construction:

  ``onnxruntime``  works on any machine; used for every offline test
  ``tensorrt``     the Jetson path (Task S.5), not implemented here

DEPLOY-4: the exported graph has a **hard-fixed batch dimension of 1**. Batch 2
raises `onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument`. So `infer`
reshapes to (1, obs_dim) and there is no batched path — if one is ever wanted,
the graph must be re-exported with a dynamic axis.

ART-1: `policy.onnx` is gitignored (`.gitignore:19 *.onnx`) while its weight
sidecar `policy.onnx.data` IS tracked. A fresh clone therefore gets an orphan
`.data` and no graph, and onnxruntime fails with a confusing external-data
error. `OnnxPolicy` checks for both up front and raises something that names the
real problem instead.
"""

from __future__ import annotations

import os

import numpy as np


class OnnxPolicy:
    """CPU/GPU-agnostic ONNX Runtime backend."""

    def __init__(self, onnx_path: str, obs_dim: int, action_dim: int) -> None:
        import onnxruntime as ort

        if not os.path.isfile(onnx_path):
            sidecar = onnx_path + ".data"
            hint = ""
            if os.path.isfile(sidecar):
                hint = (f"\n  Its weight sidecar {os.path.basename(sidecar)} IS present. "
                        "This is ART-1: *.onnx is gitignored while the .data is tracked, "
                        "so a fresh clone gets an orphan sidecar. Re-export the graph "
                        "from the checkpoint, or fix the ignore rule.")
            raise FileNotFoundError(f"ONNX graph not found: {onnx_path}{hint}")

        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        shape = self.session.get_inputs()[0].shape
        if len(shape) == 2 and isinstance(shape[1], int) and shape[1] != self.obs_dim:
            raise ValueError(
                f"graph expects obs dim {shape[1]}, contract says {self.obs_dim}")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """(obs_dim,) -> (action_dim,) float32. Batch is fixed at 1 (DEPLOY-4)."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs has {obs.shape[0]} dims, expected {self.obs_dim}")
        out = self.session.run([self.output_name],
                               {self.input_name: obs.reshape(1, self.obs_dim)})[0]
        action = np.asarray(out, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"graph returned {action.shape[0]} actions, expected {self.action_dim}")
        return action


class TensorRTPolicy:
    """Jetson backend — Task S.5. Deliberately not implemented in S.4."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "TensorRT backend is Task S.5. S.4 is hardware-free by design; use "
            "OnnxPolicy for offline verification.")


def make_policy(backend: str, onnx_path: str, obs_dim: int, action_dim: int):
    backend = backend.lower()
    if backend in ("onnx", "onnxruntime"):
        return OnnxPolicy(onnx_path, obs_dim, action_dim)
    if backend in ("trt", "tensorrt"):
        return TensorRTPolicy(onnx_path, obs_dim, action_dim)
    raise ValueError(f"unknown backend {backend!r}; expected onnxruntime or tensorrt")
