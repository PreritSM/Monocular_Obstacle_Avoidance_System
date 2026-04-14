# Run Methodology

## Overview

The system has two sides:
- **Vast.ai GPU VM** — runs Triton (inference), signaling server, and edge_gateway (all via Docker)
- **Jetson / your machine** — runs the camera client

---

## Part 1: Set Up the Vast.ai GPU VM

### Step 1 — Rent a GPU instance on Vast.ai
- Pick any instance with an NVIDIA GPU and Docker + NVIDIA drivers pre-installed
- Expose these TCP ports in Vast.ai instance settings: `8765` (signaling), `8001` (Triton gRPC), `8000` (Triton HTTP, optional health checks)
- Note the public IP — you'll need it for the Jetson config

### Step 2 — SSH into the VM and clone the repo
```bash
git clone <your-repo-url>
cd Depth_Yolo_AWS
```

### Step 3 — Build model artifacts for Triton (YOLO + Depth)

You now need two models in Triton:
- YOLO segmentation (TensorRT plan)
- Depth Anything V2 Small (ONNX)

YOLO must be converted to a TensorRT `.plan` file **on the same GPU** it will run on:

```bash
# Install ultralytics for YOLO export
pip install ultralytics

# Download model artifacts for both YOLO and Depth
bash scripts/download_model.sh

# Export YOLO PyTorch model to ONNX
python -c "from ultralytics import YOLO; model = YOLO('models/yolo26n-seg.pt'); model.export(format='onnx')"

# Export the YOLO ONNX to TensorRT engine directly into the Triton repo
bash scripts/build_triton_engine.sh

# Verify both models are in place
bash scripts/prepare_triton_repo.sh
```

This process:
1. Downloads `models/yolo26n-seg.pt` and `models/depth_anything_v2_vits.onnx`
2. Exports YOLO to `models/yolo26n-seg.onnx`
3. Builds TensorRT engine `triton/model_repository/yolo26n_seg/1/model.plan`
4. Copies Depth model to `triton/model_repository/depth_anything_v2_small/1/model.onnx`

Expected interface is documented in:

`triton/model_repository/depth_anything_v2_small/1/README.md`

### Step 4 — Start all GPU-side services with Docker Compose
```bash
cd deploy/docker
docker compose up --build
```

This starts 3 containers (all on host network):
- **signaling** on port `8765` — WebRTC signaling server
- **tritonserver** on port `8001` — YOLO (TensorRT) + Depth (ONNXRuntime) inference
- **edge_gateway** — runs parallel YOLO + depth requests per frame and sends fused compact metadata

The edge_gateway automatically uses `configs/edge_gateway.vast_ai.yaml` (configured in docker-compose.yml).

Confirm they're up:
```bash
docker compose ps
docker compose logs -f triton     # should show "Started GRPCInferenceService at 0.0.0.0:8001" and model status READY
docker compose logs -f edge_gateway   # should say "waiting for offer..."
```

Optional preflight checks from your local/Jetson machine:
```bash
nc -vz <VAST_AI_PUBLIC_IP> 8765
curl --max-time 5 http://<VAST_AI_PUBLIC_IP>:8000/v2/health/ready
```

Expected:
- `8765` should be reachable (no timeout)
- Triton health endpoint should return `OK` once Triton is ready

---

## Part 2: Set Up the Jetson Client

### Step 1 — Clone the repo and install dependencies
```bash
git clone <your-repo-url>
cd Depth_Yolo_AWS

python -m venv .venv
source .venv/bin/activate
pip install -r jetson_client/requirements.txt
```

### Step 2 — Edit the Jetson config to point at your Vast.ai VM

Open `configs/jetson.self_hosted.yaml` and update `signaling_url` with the Vast.ai public IP:

```yaml
webrtc:
  signaling_url: ws://<VAST_AI_PUBLIC_IP>:8765/ws   # <-- change this
```

If direct public access to port `8765` is blocked, use an SSH tunnel instead and keep `signaling_url` on localhost:

```bash
ssh -N \
  -L 8765:127.0.0.1:8765 \
  -L 8000:127.0.0.1:8000 \
  -L 8001:127.0.0.1:8001 \
  root@<VAST_AI_PUBLIC_IP>
```

Then set:

```yaml
webrtc:
  signaling_url: ws://127.0.0.1:8765/ws
```

Everything else can stay as-is unless you need a different camera device index.

### Step 3 — Create the logs directory
```bash
mkdir -p logs
```

### Step 4 — Run the client
```bash
python -m jetson_client.app --config configs/jetson.self_hosted.yaml
```

You should see it connect to the signaling server, complete WebRTC negotiation, and start streaming. Inference metadata will be logged to `logs/jetson_session.jsonl`.

If you get `TimeoutError` or `ConnectionRefusedError` during `websockets.connect(...)`, it usually means the signaling endpoint is not reachable yet. Re-check:
1. If using the public IP, confirm port `8765` is reachable from your Jetson.
2. If using localhost, confirm the SSH tunnel is running and `signaling_url` is `ws://127.0.0.1:8765/ws`.
3. `docker compose ps` on the Vast VM shows `signaling` is running.
4. `docker compose logs -f signaling` shows no startup errors.

---

## Part 3: Verify It's Working

On the Jetson, check the log for `metadata_rx` events:
```bash
tail -f logs/jetson_session.jsonl | grep metadata_rx
```

You should see entries like:
```json
{"event": "metadata_rx", "rx_time_ms": 1234567890, "message": "{\"age_ms\": 72, \"is_stale\": false, ...}"}
```

To analyze full latency stats after a session, use the Jetson `metadata_rx` payloads from the log:
```bash
python tools/analyze_metrics.py --input logs/jetson_session.jsonl --thresholds configs/acceptance.thresholds.yaml
```

---

## Local Development (Everything on One Machine)

Use this path to test the full pipeline locally — no Vast.ai account needed. You need a local NVIDIA GPU and Docker with the NVIDIA container toolkit installed.

### Step 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir -p logs
```

### Step 2 — Build model artifacts for local Triton

Same as the Vast.ai flow — YOLO engine must be built on the local GPU:

```bash
bash scripts/download_model.sh    # saves YOLO + depth source artifacts in models/

# Export YOLO PyTorch model to ONNX (required before TensorRT conversion)
python -c "from ultralytics import YOLO; model = YOLO('models/yolo26n-seg.pt'); model.export(format='onnx')"

bash scripts/build_triton_engine.sh
bash scripts/prepare_triton_repo.sh
```

The build script writes YOLO as TensorRT and copies Depth Anything V2 Small as ONNXRuntime input:

- `triton/model_repository/yolo26n_seg/1/model.plan`
- `triton/model_repository/depth_anything_v2_small/1/model.onnx`

### Step 3 — Install Triton Inference Server locally (via Docker)

Triton is distributed as a Docker image — no pip install. The NVIDIA container toolkit must be installed so Docker can access the GPU.

**Install NVIDIA container toolkit (if not already installed):**
```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

<!-- **Build the engine locally:**
```bash
docker run --rm --gpus all --network host \
  -v $(pwd)/models:/workspace/models:ro \
  -v $(pwd)/triton/model_repository:/models \
  --shm-size=1g \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/yolo26n-seg.onnx --saveEngine=/models/yolo26n_seg/1/model.plan --fp16
``` -->

Engine generation is complete when the command exits successfully.

Then start Triton itself:

```bash
docker run --rm --gpus all --network host \
  -v $(pwd)/triton/model_repository:/models:ro \
  --shm-size=1g \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --strict-model-config=false --exit-on-error=false --log-verbose=0
```

Triton is ready when you see:
```
Started GRPCInferenceService at 0.0.0.0:8001
```

### Step 4 — Confirm self-hosted model endpoints

`configs/edge_gateway.self_hosted.yaml` should point both model clients to local Triton:

```yaml
yolo_inference:
  triton_url: 127.0.0.1:8001

depth_inference:
  triton_url: 127.0.0.1:8001
  input_name: l_x_
  output_names: ["select_36"]
```

Also verify model names match Triton repository folders:

- `yolo_inference.model_name: yolo26n_seg`
- `depth_inference.model_name: depth_anything_v2_small`

Optional: if your YOLO model is not COCO-80, set explicit class labels:

```yaml
yolo_inference:
  class_names: ["class0", "class1", "class2"]
```

### Step 5 — Start the signaling server and edge_gateway

```bash
# Terminal 1
bash scripts/run_self_hosted_local.sh
```

This starts both the signaling server (port `8765`) and the edge_gateway in the background, using `configs/edge_gateway.self_hosted.yaml`.

### Step 6 — Run the camera client

```bash
# Terminal 2
python -m jetson_client.app --config configs/jetson.self_hosted.yaml
```

`configs/jetson.self_hosted.yaml` already points `signaling_url` to `ws://127.0.0.1:8765/ws`, so no changes needed.

### Step 7 — Verify

```bash
tail -f logs/jetson_session.jsonl | grep metadata_rx
```

You should see inference metadata arriving with `age_ms` values. Run the latency report from the Jetson log:

```bash
python tools/analyze_metrics.py --input logs/jetson_session.jsonl --thresholds configs/acceptance.thresholds.yaml
```

For phase-1 validation, inspect fused payloads and verify both model statuses per frame:

```bash
tail -f logs/jetson_session.jsonl | grep -E 'metadata_rx|overlap|depth_band|depth_median'
```

You should see metadata where:
- `detections.yolo.status` is `ok`
- `detections.depth.status` is `ok`
- `detections.depth.depth_percentiles` is present
- `detections.yolo.objects[*].class_name` is present
- `detections.overlap.object_count` is present
- `detections.overlap.objects[*].class_name` is present
- `detections.overlap.objects[*].depth_median` is present
- `detections.overlap.objects[*].depth_spread_p90_p10` is present
- `detections.overlap.objects[*].depth_band` is one of `near|mid|far`

---

## Key Config Files

| File | Used by | Purpose |
|---|---|---|
| `configs/jetson.self_hosted.yaml` | Jetson client | Camera settings, signaling URL (localhost for local dev, Vast.ai IP for remote) |
| `configs/edge_gateway.self_hosted.yaml` | Local edge gateway (direct Python execution) | YOLO + depth model endpoints (localhost Triton), runtime thresholds |
| `configs/edge_gateway.vast_ai.yaml` | Docker edge_gateway container (Vast.ai) | YOLO + depth model endpoints (localhost Triton in Docker), runtime thresholds |

**Note:** Both `self_hosted` and `vast_ai` configs use localhost (`127.0.0.1:8001`) because Docker containers on host network communicate locally. The `self_hosted` suffix refers to GPU provider, not network location.
| `deploy/docker/docker-compose.yml` | Vast.ai VM | Defines all 3 server-side services |
