---
paths:
  - "jetson_runtime/**"
  - "exported_policies/**"
---

# Jetson Deployment

Rules and context for code that runs on the Jetson Orin Nano Super.

## Two-Thread Architecture

The robot runs two concurrent loops on the Jetson GPU:

```
Thread 1: Cosmos Reason2 (2-3 Hz) — high-level reasoning via vLLM API
Thread 2: Locomotion policy (50 Hz) — low-level walking via TensorRT
```

They share the GPU. Locomotion takes <1 ms, Cosmos takes ~300-500 ms. They time-share without conflict.

## GPIO Pin Mapping (Pi → Jetson)

| Function | Pi GPIO | Jetson GPIO | Pin |
|---|---|---|---|
| Left Eye LED | 23 | 23 | 16 |
| Right Eye LED | 24 | 24 | 18 |
| Projector LED | 25 | 25 | 22 |
| Left Antenna PWM | 12 | 12 | 32 |
| Right Antenna PWM | 13 | 13 | 33 |
| Left Foot Switch | 22 | 22 | 15 |
| Right Foot Switch | 27 | 27 | 13 |
| IMU SDA | 2 | 2 | 3 |
| IMU SCL | 3 | 3 | 5 |

**Important:** Verify against the actual Jetson Orin Nano dev kit carrier board pinout before wiring. Use `Jetson.GPIO` library (not `RPi.GPIO`).

## TensorRT Policy Inference

```python
# Convert ONNX to TensorRT on the Jetson itself:
trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16

# Inference wrapper: jetson_runtime/trt_infer.py
# Input: 56-dim float32 observation vector
# Output: 16-dim float32 action vector
# Latency: <1 ms
```

## Cosmos Reason2 Deployment

```bash
# Start vLLM server:
docker run --rm -it --runtime=nvidia --network host --shm-size=4g \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve "embedl/Cosmos-Reason2-2B-W4A16-Edge2" \
    --max-model-len 2048 --gpu-memory-utilization 0.70 --max-num-seqs 1

# Query via HTTP API from Python:
# POST http://localhost:8000/v1/chat/completions
# Send camera frame as base64 image + text prompt
# Parse JSON velocity command from text response
```

## Safety Rules

- All velocity commands from Cosmos MUST be clamped: forward [-0.2, 0.3], lateral [-0.2, 0.2], turn [-0.3, 0.3]
- On any Cosmos inference failure, fall back to last known good command (not zero — that could cause mid-stride fall)
- IMU-based emergency stop: if tilt > 60 degrees, cut all motors
- Servo current monitoring: if any servo exceeds safe current, reduce velocity

## Power

- Battery: 6x 18650 Li-ion cells (3S2P, 11.1V) → DC-DC boost (11.1V → 19V) for Jetson barrel jack
- Run Jetson at 7W eco mode for maximum battery life (67 TOPS still available)
- Estimated battery life: ~1-2 hours at 7W, ~30 min at 25W
