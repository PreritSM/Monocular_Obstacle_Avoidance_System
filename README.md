# Depth YOLO - Vast.ai GPU VM

This branch focuses on a low-latency perception loop hosted on rented GPU VMs:

- USB camera on Jetson or any upstream client
- WebRTC media uplink over UDP
- Vast.ai GPU VM for signaling, edge inference, and Triton
- Triton Inference Server for YOLO-seg TensorRT execution
- Compact metadata return over WebRTC DataChannel
- Structured logging and benchmark utilities

## Repository layout

- `jetson_client/`: camera capture and WebRTC sender (modular camera adapters)
- `services/edge_gateway/`: WebRTC ingest + Triton inference + metadata return
- `services/signaling_self_hosted/`: self-hosted signaling server
- `deploy/docker/`: Docker Compose stack for Vast.ai and local GPU VMs
- `tools/`: latency and metrics analysis helpers
- `.github/workflows/`: CI checks

## Model

Configured model artifact target:

- `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt`

Phase 1 keeps the runtime path Triton-first. TensorRT engine build is intentionally externalized to a script hook and model repository layout.

## Quick start (local dev)

1. Create virtual environment and install deps:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements-dev.txt`
2. Start self-hosted signaling:
   - `python -m services.signaling_self_hosted.server`
3. Start edge gateway:
   - `python -m services.edge_gateway.app --config configs/edge_gateway.self_hosted.yaml`
4. Start Jetson sender (on Jetson host):
   - `python -m jetson_client.app --config configs/jetson.self_hosted.yaml`

## Benchmark and acceptance metrics

Acceptance thresholds are codified in `configs/acceptance.thresholds.yaml` and evaluated with:

- `python -m tools.analyze_metrics --input logs/session.jsonl --thresholds configs/acceptance.thresholds.yaml`

## Vast.ai deployment

1. Start the host-networked stack on the rented GPU VM:
   - `docker compose -f deploy/docker/docker-compose.yml up -d --build`
2. Ensure the Vast.ai instance exposes the WebRTC and signaling ports you expect the client to reach.
3. Keep the TensorRT model repository mounted read-only so cold starts only pay container startup, not model export time.
4. Use the self-hosted configs in `configs/edge_gateway.vast_ai.yaml` and `configs/jetson.self_hosted.yaml` unless you need a different signaling endpoint.

## Notes

- Self-hosted signaling is the only supported signaling path on this branch.
- Triton remains the inference boundary so model and transport changes stay isolated.
- Safety controller and fallback autonomy are deferred to the next phase by design.
