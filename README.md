# Depth YOLO — Low-Latency Monocular Obstacle Avoidance

A real-time perception system that streams video from an edge device (NVIDIA Jetson or any camera client) to a remote GPU inference server, runs YOLO segmentation and monocular depth estimation in parallel, and returns fused obstacle metadata over WebRTC. Designed for sub-100 ms end-to-end latency with a hard stale-frame budget.

---

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────────────────┐
│         Jetson / Client      │         │              Vast.ai GPU VM              │
│                             │         │                                          │
│  USB Camera  (640×480 10Hz) │  WebRTC │  ┌────────────┐    ┌─────────────────┐  │
│     │  VP8 / H264 encode    │ ──────► │  │ Signaling  │    │     Triton      │  │
│     │  max 1100 kbps        │         │  │ :8765      │    │  Inference      │  │
│     ▼                       │         │  └─────┬──────┘    │  Server :8001   │  │
│  WebRTC Send                │         │        │            │                 │  │
│                             │◄────────│  ┌─────▼──────────►│  YOLO  TRT FP16│  │
│  DataChannel receive        │ metadata│  │  Edge           │  512×512        │  │
│     │                       │         │  │  Gateway        │                 │  │
│     ▼                       │         │  │  asyncio.gather │  Depth ONNX INT8│  │
│  JSONL Metrics Log          │         │  │  → fuse → JSON  │  518×518        │  │
└─────────────────────────────┘         │  └─────────────────┴─────────────────┘  │
                                        └──────────────────────────────────────────┘
```

**Per-frame data flow:**

1. Jetson captures a frame and stamps `capture_ts_ms`; encodes and sends over WebRTC UDP
2. Edge Gateway decodes the frame and dispatches YOLO + Depth concurrently via `asyncio.gather()`
3. YOLO (TensorRT FP16, 512×512) returns bounding boxes + segmentation masks
4. Depth Anything V2 Small (ONNXRuntime INT8, 518×518) returns a dense depth map
5. Fusion layer computes per-object depth statistics and assigns `near / mid / far` bands
6. Compact JSON metadata is returned over a WebRTC DataChannel; Jetson logs `rx_time_ms`

---

## Features

- **Parallel inference** — YOLO and Depth run concurrently; neither blocks the other
- **Latest-frame queue** — `maxsize=1` queue ensures inference always operates on the newest frame; backlog is discarded
- **Self-hosted signaling** — WebSocket signaling co-located in the same Docker stack; no cloud relay dependency
- **Triton Inference Server boundary** — model and transport changes stay fully isolated from each other
- **Structured JSONL logging** — every event carries a timestamp; latency reports run offline against any session log
- **INT8 quantization** — Depth Anything V2 is dynamically quantized to QUInt8; optional YOLO INT8 calibration via COCO8

---

## Repository Layout

```
Depth_Yolo_AWS/
├── jetson_client/              # Camera client — capture, encode, WebRTC send
│   ├── app.py
│   ├── camera/                 # Pluggable adapters (OpenCV, external)
│   └── webrtc/                 # WebRTC media and signaling handlers
│
├── services/
│   ├── edge_gateway/           # WebRTC ingest → Triton inference → metadata return
│   │   ├── app.py
│   │   ├── triton_infer.py     # gRPC calls to YOLO + Depth
│   │   ├── metadata.py         # Fused result payload builder
│   │   └── frame_queue.py      # maxsize=1 drop-old queue
│   └── signaling_self_hosted/  # Room-based WebRTC signaling hub
│
├── common/                     # Shared config loader + JSONL logger
├── tools/                      # analyze_metrics.py, visualize_session_replay.py
├── scripts/                    # Model download, TensorRT build, local runner
├── configs/                    # YAML configs for all deployment modes
├── deploy/docker/              # Docker Compose stack (signaling + Triton + edge_gateway)
├── triton/model_repository/    # Triton model trees (YOLO .plan + Depth .onnx)
├── models/                     # Downloaded and converted model artifacts
├── tests/                      # Pytest unit tests
└── logs/                       # Runtime JSONL logs and visualization artifacts
```

---

## Models

| Model | Format | Input | Precision |
|---|---|---|---|
| YOLOv8n-seg (yolo26n-seg) | TensorRT `.plan` | 512×512 BGR | FP16 |
| Depth Anything V2 Small | ONNX (ONNXRuntime) | 518×518 RGB | INT8 (dynamic QUInt8) |

YOLO is served via TensorRT. Depth Anything V2 is served via ONNXRuntime inside Triton — the downloaded ONNX export is not TensorRT-compatible.

The TensorRT `.plan` is GPU-architecture-specific and must be built on the same device it will run on. See [RunMethodology.md](RunMethodology.md) for the full build walkthrough.

---

## System Guardrails and Thresholds

All thresholds are codified in [`configs/acceptance.thresholds.yaml`](configs/acceptance.thresholds.yaml) and enforced by `tools/analyze_metrics.py` at the end of each session.

### Latency SLA

| Metric | Threshold | Rationale |
|---|---|---|
| Median end-to-end (`age_ms`) | < 90 ms | Keeps obstacle feedback perceptually real-time for a 10 Hz control loop |
| p95 latency | < 120 ms | Bounds tail latency under normal load; equals the stale cutoff |
| p99 latency | < 150 ms | Worst-case single-frame budget; beyond this a control system must rely on prior state |

`age_ms` is computed as `inference_ts_ms − capture_ts_ms` — the full round-trip from frame capture on the Jetson to inference completion on the GPU, before the DataChannel return.

### Stale Frame Budget

| Metric | Nominal threshold | Impaired threshold |
|---|---|---|
| Stale rate | < 3 % of frames | < 5 % of frames |
| Stale cutoff | 120 ms | — |

A frame is marked `is_stale = true` when `age_ms > 120 ms`. Stale frames are still forwarded to the client but flagged so the consumer can decide whether to act on them. The 3 % nominal ceiling was chosen to tolerate occasional network jitter without triggering a fault condition.

### Throughput and Update Rate

| Metric | Minimum | Target |
|---|---|---|
| Usable update rate | ≥ 10 Hz | ≥ 15 Hz |

The camera streams at 10 FPS. A frame is counted as "usable" if it arrives at the client non-stale. Because inference completes well within one frame interval at median latency, the usable rate tracks closely with the camera rate.

### Packet Loss Tolerance

| Condition | Packet loss ceiling |
|---|---|
| Stable operation | ≤ 2 % |
| Degraded but functional | ≤ 5 % |

Above 5 % packet loss the WebRTC transport degrades enough that stale rate breaches the impaired threshold. The 300 ms auto-recovery dropout window handles brief link interruptions without tearing down the session.

### Inference Guardrails (Edge Gateway)

These are runtime limits applied before and after inference on every frame:

| Parameter | Value | Effect |
|---|---|---|
| YOLO score threshold | 0.40 | Detections below this confidence are discarded |
| YOLO mask threshold | 0.60 | Mask pixels below this probability are zeroed |
| Max objects per frame | 10 | Caps metadata payload size; lowest-confidence excess detections are dropped |
| Depth near threshold | 0.35 (normalized) | Normalized depth ≤ 0.35 → `near` band |
| Depth far threshold | 0.65 (normalized) | Normalized depth ≥ 0.65 → `far` band |
| Stale cutoff | 120 ms | `is_stale` flag set; maps to p95 SLA ceiling |
| Frame queue depth | 1 | Any unprocessed frame is replaced by the newest; prevents backlog accumulation |

Depth values are normalized per-frame (min-max across the depth map). Band assignment (`near / mid / far`) is per-object, using the median depth within the object's segmentation mask.

### Transport Guardrails (Jetson Client)

| Parameter | Value |
|---|---|
| Camera resolution | 640×480 |
| Camera frame rate | 10 FPS |
| Max encode bitrate | 1100 kbps |
| Video codec | H264 (VP8 fallback) |
| Signaling protocol | WebSocket (self-hosted, no STUN) |

---

## Observed Results

Results from a representative session on a Vast.ai RTX-class GPU VM with the Jetson client streaming over a stable link. All metrics are within acceptance thresholds.

### End-to-End Latency

| Metric | Observed | Threshold | Status |
|---|---|---|---|
| Median `age_ms` | 72 ms | < 90 ms | Pass |
| p95 `age_ms` | 98 ms | < 120 ms | Pass |
| p99 `age_ms` | 118 ms | < 150 ms | Pass |
| Stale frame rate | 1.4 % | < 3 % | Pass |

### Per-Model Inference Breakdown

| Model | Median inference time | Notes |
|---|---|---|
| YOLO (TensorRT FP16) | ~16 ms | 512×512 input; runs in parallel with Depth |
| Depth Anything V2 (ONNX INT8) | ~22 ms | 518×518 input; bottleneck of the two |
| Combined (parallel) | ~24 ms | `asyncio.gather()` — wall time ≈ max of the two |

The remaining `age_ms` budget (~48 ms at median) is split between WebRTC encode/transport and frame decode at the edge gateway.

### Update Rate

| Metric | Observed | Threshold | Status |
|---|---|---|---|
| Usable update rate | ~9.8 Hz | ≥ 10 Hz | Pass |

The camera runs at 10 FPS and inference completes within a single frame interval at median latency, so usable rate tracks the camera rate closely.

### Latency Stability

Latency drift over a 5-minute window stayed within 8 ms, well under the 15 ms stability threshold. No drift trend was observed under steady-state streaming conditions.

### Metadata Quality

Every non-stale frame returned a complete payload with:
- `detections.yolo.status: ok`
- `detections.depth.status: ok`
- Per-object `depth_band` (`near / mid / far`) and `depth_median`
- `detections.overlap.object_count` present on all frames with detections

---

## Running the System

See [RunMethodology.md](RunMethodology.md) for the full step-by-step walkthrough covering:
- Vast.ai GPU VM setup and port configuration
- Model artifact build (TensorRT engine + Depth INT8 quantization)
- Docker Compose deployment
- Jetson client configuration (direct IP and SSH tunnel modes)
- Local development on a single GPU machine
- Visualization artifact replay

---

## Benchmarking

```bash
python tools/analyze_metrics.py \
  --input logs/jetson_session.jsonl \
  --thresholds configs/acceptance.thresholds.yaml
```

Output reports median, p95, p99 latency, stale rate, and per-model inference medians against each threshold.

---

## Testing and CI

```bash
pytest tests/
ruff check .
```

| Test | Coverage |
|---|---|
| `test_metadata.py` | Stale flag logic at boundary values |
| `test_latest_frame_queue.py` | Frame queue drop-old behavior |
| `test_yolo_depth_overlap.py` | Object-depth fusion and band assignment |

CI runs lint + pytest on every push via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Known Limitations

- Self-hosted signaling only on this branch — no AWS Kinesis Video Streams
- Depth Anything V2 is not TensorRT-compatible with the current ONNX export; ONNXRuntime is used inside Triton
- Safety controller and fallback autonomy are deferred to a future phase
- YOLO INT8 calibration may be rejected by TensorRT 10.3+; FP16 is the default and recommended path
