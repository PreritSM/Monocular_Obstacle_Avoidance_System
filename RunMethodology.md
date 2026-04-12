# Run Methodology

## Overview

The system has two sides:
- **Vast.ai GPU VM** — runs Triton (inference), signaling server, and edge_gateway (all via Docker)
- **Jetson / your machine** — runs the camera client

---

## Part 1: Set Up the Vast.ai GPU VM

### Step 1 — Rent a GPU instance on Vast.ai
- Pick any instance with an NVIDIA GPU and Docker + NVIDIA drivers pre-installed
- Note the public IP — you'll need it for the Jetson config

### Step 2 — SSH into the VM and clone the repo
```bash
git clone <your-repo-url>
cd Depth_Yolo_AWS
```

### Step 3 — Build the TensorRT engine (model.plan)

The model must be converted to a TensorRT `.plan` file **on the same GPU** it will run on:

```bash
# Install ultralytics to export
pip install ultralytics

# Download the YOLO weights
bash scripts/download_model.sh    # saves to models/yolo26n-seg.pt

# Export to TensorRT engine and write it directly into the Triton repo
bash scripts/build_triton_engine.sh

# Verify
bash scripts/prepare_triton_repo.sh
```

### Step 4 — Start all GPU-side services with Docker Compose
```bash
cd deploy/docker
docker compose up --build
```

This starts 3 containers (all on host network):
- **signaling** on port `8765`
- **tritonserver** on port `8001`
- **edge_gateway** — connects to both, waits for a WebRTC offer

Confirm they're up:
```bash
docker compose ps
docker compose logs -f edge_gateway   # should say "waiting for offer..."
```

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

### Step 2 — Build the TensorRT engine

Same as the Vast.ai flow — the engine must be built on the local GPU:

```bash
bash scripts/download_model.sh    # saves models/yolo26n-seg.pt
bash scripts/build_triton_engine.sh
bash scripts/prepare_triton_repo.sh
```

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

**Build the engine locally:**
```bash
docker run --rm --gpus all --network host \
  -v $(pwd)/models:/workspace/models:ro \
  -v $(pwd)/triton/model_repository:/models \
  --shm-size=1g \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/yolo26n-seg.onnx --saveEngine=/models/yolo26n_seg/1/model.plan --fp16
```

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

### Step 4 — Fix the self-hosted config to use localhost Triton

`configs/edge_gateway.self_hosted.yaml` has `triton_url: triton:8001` (Docker DNS intended for Compose). Change it to `127.0.0.1:8001` for local use:

```yaml
inference:
  triton_url: 127.0.0.1:8001   # <-- change from triton:8001
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

---

## Key Config Files

| File | Used by | Purpose |
|---|---|---|
| `configs/jetson.self_hosted.yaml` | Jetson client | Camera settings, signaling URL |
| `configs/edge_gateway.vast_ai.yaml` | Docker (edge_gateway) | Triton URL, signaling URL (localhost since host network) |
| `deploy/docker/docker-compose.yml` | Vast.ai VM | Defines all 3 server-side services |
