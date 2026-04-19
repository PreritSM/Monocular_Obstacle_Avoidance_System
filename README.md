# Depth + YOLO WebRTC Perception Pipeline

Low-latency, Triton-first perception stack for obstacle-awareness experiments.

The pipeline ingests camera video over WebRTC, runs parallel YOLO segmentation + Depth Anything inference on Triton, fuses outputs into compact depth-aware metadata, and returns it over a WebRTC DataChannel.

## What This Implements

- Jetson/client-side camera capture and WebRTC sender
- Self-hosted signaling server for WebRTC negotiation
- Edge gateway that receives frames and performs dual-model inference
- Triton model serving for:
   - YOLO segmentation (TensorRT plan)
   - Depth Anything V2 (ONNXRuntime model)
- Structured JSONL logs on both edge and client
- Latency instrumentation with per-stage timing decomposition
- Offline metrics analysis against acceptance thresholds

## End-to-End Flow

1. Client publishes video track via WebRTC to edge gateway.
2. Edge gateway stores only the latest frame (queue depth defaults to 1).
3. For each frame, edge runs YOLO + Depth inference in parallel threads.
4. Edge decodes model outputs and computes depth overlap per YOLO object.
5. Edge emits metadata with age and timing breakdown on DataChannel.
6. Client logs metadata and timing snapshots to JSONL for post-run analysis.

## Core Components

- `jetson_client/`
   - Camera adapters (`opencv`, external adapter)
   - WebRTC sender and metadata receiver
   - JSONL telemetry logging

- `services/signaling_self_hosted/`
   - Self-hosted signaling server used by both client and edge

- `services/edge_gateway/`
   - WebRTC receiver
   - Latest-frame queueing strategy
   - Triton model clients and decoding/fusion logic
   - Metadata generation and logging

- `triton/model_repository/`
   - `yolo26n_seg` model repo (TensorRT plan)
   - `depth_anything_v2_small` model repo (ONNX)

- `tools/`
   - `analyze_metrics.py` for latency summary and threshold checks

## Implementation Details

### 1) Frame Queueing Strategy

The edge gateway uses a latest-frame queue (`queue_depth: 1` by default), so old frames are dropped when the consumer lags. This protects freshness and prevents unbounded queue growth.

### 2) Parallel Inference

For each dequeued frame, edge dispatches YOLO and Depth inference concurrently using `asyncio.to_thread` and waits with `asyncio.gather`.

### 3) Decoding and Fusion

- YOLO segmentation decode extracts objects, masks, confidence, class labels, and bounding boxes.
- Depth decode computes frame-level depth percentiles and depth map stats.
- Overlap module maps YOLO masks into depth space and assigns object depth bands (`near`, `mid`, `far`).

### 4) Metadata Contract

Each `inference_done` payload includes:

- `trace_id`, `capture_ts_ms`, `edge_rx_ts_ms`, `inference_ts_ms`
- `age_ms`, `is_stale`, `stale_threshold_ms`
- `detections.yolo`, `detections.depth`, `detections.overlap`
- `timings_ms` breakdown:
   - `yolo.inference_ms`, `yolo.decode_ms`
   - `depth.inference_ms`, `depth.decode_ms`
   - `fusion_ms`, `parallel_window_ms`
   - `capture_to_edge_rx_ms`, `edge_rx_to_inference_done_ms`
   - `age_ms_reconstructed`

This enables direct verification that latency accounting is consistent:

- `age_ms ≈ age_ms_reconstructed`

### 5) Client Telemetry Logging

Client logs incoming metadata as `metadata_rx` events and writes extracted timing snapshots as `inference_timing_rx` entries.

## Model Artifacts and Build

Configured source artifacts:

- YOLO: `models/yolo26n-seg.pt`
- Depth: `models/depth_anything_v2_vits.onnx`

Build pipeline (`scripts/build_triton_engine.sh`):

1. YOLO ONNX is converted to TensorRT plan.
2. Depth ONNX is quantized using `onnxruntime.quantization` (dynamic QUInt8).
3. Quantized depth ONNX is copied into Triton model repository.

Generated artifacts:

- `triton/model_repository/yolo26n_seg/1/model.plan`
- `models/depth_anything_v2_vits.int8.onnx`
- `triton/model_repository/depth_anything_v2_small/1/model.onnx`

## Configuration Summary

### Edge

`configs/edge_gateway.self_hosted.yaml` and `configs/edge_gateway.vast_ai.yaml` define:

- Signaling endpoint
- YOLO Triton endpoint + I/O names + input size
- Depth Triton endpoint + I/O names + input size
- Runtime knobs:
   - `queue_depth`
   - `stale_threshold_ms`
   - `yolo_score_threshold`
   - `yolo_mask_threshold`
   - `max_objects_per_frame`
   - `depth_near_threshold`, `depth_far_threshold`

### Client

`configs/jetson.self_hosted.yaml` defines:

- Camera device/resolution/FPS
- WebRTC signaling URL and ICE config
- Metadata channel label
- Local stale threshold reference

## Local Development Quick Start

1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir -p logs
```

2. Prepare model repository

```bash
bash scripts/download_model.sh
python -c "from ultralytics import YOLO; model = YOLO('models/yolo26n-seg.pt'); model.export(format='onnx')"
bash scripts/build_triton_engine.sh
bash scripts/prepare_triton_repo.sh
```

3. Start services

```bash
python -m services.signaling_self_hosted.server
python -m services.edge_gateway.app --config configs/edge_gateway.self_hosted.yaml
python -m jetson_client.app --config configs/jetson.self_hosted.yaml
```

Alternative local orchestration:

```bash
bash scripts/run_self_hosted_local.sh
```

## Vast.ai Deployment

Use Docker Compose from `deploy/docker/`:

```bash
cd deploy/docker
docker compose up --build
```

Required ports:

- `8765` signaling
- `8001` Triton gRPC
- `8000` Triton HTTP (optional health checks)

## Metrics and Validation

Acceptance thresholds are in `configs/acceptance.thresholds.yaml`.

Run analysis:

```bash
python tools/analyze_metrics.py --input logs/jetson_session.jsonl --thresholds configs/acceptance.thresholds.yaml
```

The tool reports:

- latency median/p95/p99
- stale rate
- YOLO and depth median inference times
- reconstructed age consistency residual

## Log Files

- Edge: `logs/edge_session.jsonl`
   - Includes `frame_rx`, `inference_done`, connection events

- Client: `logs/jetson_session.jsonl`
   - Includes `metadata_rx`, `inference_timing_rx`, connection events

## Notes and Scope

- This branch currently supports self-hosted signaling mode.
- Triton is the inference boundary by design.
- The repo is focused on perception and latency instrumentation; higher-level safety/autonomy logic is outside current scope.

## Detailed Run Guide

For a full step-by-step runbook (Vast.ai and local), see:

- `RunMethodology.md`
