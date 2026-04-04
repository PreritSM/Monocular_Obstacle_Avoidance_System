# Depth YOLO AWS - Phase 1

Phase 1 implements a low-latency cloud-assisted perception loop for drone experiments:

- USB camera on Jetson (480p30 target)
- WebRTC media uplink over UDP
- EC2 G6 in us-east-1 Local Zone (single instance)
- Triton Inference Server (YOLO-seg TensorRT model path)
- Compact metadata return over WebRTC DataChannel
- Structured logging and benchmark utilities

## Repository layout

- `jetson_client/`: camera capture and WebRTC sender (modular camera adapters)
- `services/edge_gateway/`: WebRTC ingest + Triton inference + metadata return
- `services/signaling_self_hosted/`: self-hosted signaling server
- `services/signaling_kvs/`: AWS Kinesis signaling adapter skeleton
- `deploy/terraform/`: AWS infrastructure (VPC, subnet, SG, IAM, EC2)
- `deploy/docker/`: Docker Compose stack for EC2
- `tools/`: local zone candidate benchmark + metrics analysis
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

## AWS deployment

1. Select Local Zone candidates under `us-east-1`:
   - `python -m tools.select_local_zone --region us-east-1 --instance-type g6.xlarge --city "Henrietta, NY"`
2. Apply Terraform:
   - `cd deploy/terraform && terraform init && terraform apply`
3. Deploy Docker stack on EC2 with generated `.env` values.

## Notes

- Self-hosted signaling path is fully implemented.
- AWS Kinesis signaling path is scaffolded behind an interface for Phase 1 experimentation and can be completed with account-specific channel setup.
- Safety controller and fallback autonomy are deferred to the next phase by design.
