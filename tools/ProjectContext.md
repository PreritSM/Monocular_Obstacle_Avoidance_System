# Project Context — Depth YOLO AWS: Unabridged Technical Reference

> This document is a complete, unabridged technical description of the project for use as report context. It covers every component, design decision, algorithm, configuration parameter, threshold, and result in full detail.

---

## 1. Project Purpose and Motivation

The project implements a **real-time monocular obstacle avoidance perception system** designed for autonomous edge devices such as NVIDIA Jetson platforms. The core problem being solved is: how can an edge device with limited compute obtain high-quality, low-latency depth and object detection metadata without running expensive models locally?

The solution is an offloaded inference architecture: the edge device captures raw video, streams it over a WebRTC connection to a remote GPU server (rented from Vast.ai), where YOLO object segmentation and monocular depth estimation run in parallel on a TensorRT-accelerated Triton Inference Server. The resulting fused metadata — bounding boxes, segmentation masks, per-object depth statistics, and proximity band classifications — is sent back to the edge device over a WebRTC DataChannel in near real-time.

The system targets a median end-to-end latency (from frame capture to metadata receipt) of under 90 ms, with a hard stale frame ceiling, making it viable for real-time reactive autonomy.

---

## 2. High-Level Architecture

The system has two physical sides:

**Side A — Edge Device (Jetson or any camera host):**
- Captures video from a USB camera at 640×480 at 10 FPS
- Encodes frames using H264 (VP8 preferred for lower CPU latency)
- Transmits the video stream over WebRTC UDP transport
- Receives inference metadata back via a WebRTC DataChannel
- Logs all events with timestamps to a JSONL file

**Side B — GPU VM (Vast.ai or local GPU machine):**
- Runs three services in Docker containers, all on the host network:
  1. **Signaling Server** — WebSocket-based WebRTC signaling hub (port 8765)
  2. **Triton Inference Server** — serves YOLO (TensorRT) and Depth (ONNXRuntime) models (gRPC port 8001, HTTP port 8000)
  3. **Edge Gateway** — receives video frames, runs inference, fuses results, sends metadata back

All three containers share `network_mode: host` to eliminate Docker NAT overhead and minimize inter-service latency.

**Data flow per frame (end-to-end):**

1. Jetson USB camera captures frame → stamp `capture_ts_ms` (Unix ms)
2. Frame is wrapped in an `av.VideoFrame`, assigned PTS on a 90 kHz time base
3. aiortc encodes and transmits over WebRTC UDP (H264 or VP8)
4. Edge Gateway receives WebRTC video track frame via aiortc
5. Frame is decoded from WebRTC to `bgr24` numpy array via `av.VideoFrame.to_ndarray(format="bgr24")`
6. Frame is placed into a `LatestFrameQueue` (maxsize=1) — older frames are dropped
7. Inference loop picks the frame, dispatches YOLO and Depth inference concurrently via `asyncio.gather()`
8. YOLO runs on TensorRT FP16 at 512×512; Depth runs on ONNXRuntime INT8 at 518×518
9. Inference results are decoded (boxes, masks, depth map)
10. Fusion computes per-object depth statistics and assigns near/mid/far bands
11. A compact JSON metadata payload is built and sent over the WebRTC DataChannel
12. Jetson receives the message, logs `rx_time_ms`, and parses timing breakdowns

---

## 3. Component Deep Dive

### 3.1 Jetson Client (`jetson_client/`)

**Entry point:** `jetson_client/app.py` — `run_sender(config, clean_log)`

**Camera adapters (`jetson_client/camera/`):**
- `OpenCVCameraAdapter`: opens a V4L2 device via `cv2.VideoCapture`, configures width/height/FPS, exposes `read() → (ok, frame_bgr)` and `start()` / `stop()`.
- `ExternalFeedAdapter`: pluggable adapter for synthetic or pre-recorded feeds.
- Both implement the `CameraAdapter` base class interface.

**WebRTC media (`jetson_client/webrtc/media.py`):**
- `CameraVideoTrack` extends `aiortc.VideoStreamTrack`.
- On each `recv()` call, pulls a frame from the adapter.
- Assigns PTS using a `Fraction(1, 90000)` time base, incrementing by 3000 per frame (matches 30 FPS PTS spacing; actual frame rate is driven by the camera adapter).
- If the camera fails to read, it emits a black frame to keep the WebRTC track alive.
- `FrameClock` is a monotonic nanosecond clock helper for local timing.

**Signaling client (`jetson_client/webrtc/signaling_self_hosted.py`):**
- Connects to the signaling server WebSocket.
- Sends a `join` message with `room_id` and `peer_id`.
- Sends `offer` (SDP), `candidate` (ICE), `answer` (SDP), and `bye` messages as JSON over the WebSocket.

**Codec preference:**
The client explicitly prefers VP8 over H264 by reordering codec capabilities returned by `RTCRtpSender.getCapabilities("video")`. VP8 (libvpx with `deadline=realtime` and `lag-in-frames=0`) encodes faster per-frame than libx264 on CPU, reducing sender-side encoding latency.

**ICE candidate handling:**
ICE candidates that arrive before the remote SDP description is set are buffered in `pending_candidates` and applied immediately after `setRemoteDescription()` is called. This prevents ICE negotiation failures on fast-signaling setups.

**Metadata reception:**
The DataChannel named `nav_meta` receives JSON string messages. On each message:
- `rx_time_ms` is stamped (current Unix ms)
- The raw message and `rx_time_ms` are logged as event `metadata_rx`
- The payload is parsed and timing breakdowns are extracted and logged separately as `inference_timing_rx`

**Timing fields extracted and logged:**
- `trace_id` — UUID hex identifying the frame
- `age_ms` — end-to-end age (inference_ts_ms − capture_ts_ms)
- `age_ms_reconstructed` — sum of capture-to-edge-rx + edge-rx-to-inference-done (sanity check)
- `capture_to_edge_rx_ms` — network transit time (one way)
- `edge_rx_to_inference_done_ms` — time spent in inference on the GPU
- `yolo_inference_ms` / `depth_inference_ms` — per-model Triton call durations
- `yolo_decode_ms` / `depth_decode_ms` — post-processing decode durations
- `fusion_ms` — time spent computing overlap

---

### 3.2 Signaling Server (`services/signaling_self_hosted/server.py`)

A minimal WebSocket-based WebRTC signaling hub. Implemented with the `websockets` library and `asyncio`.

**Room model:**
- `RoomHub` maintains a dict of rooms, each mapping `peer_id → Peer(peer_id, ws)`.
- An `asyncio.Lock` serializes join/leave operations.
- `relay(room_id, src_peer_id, payload)` forwards a JSON payload to all peers in the room except the sender.

**Protocol:**
Each WebSocket connection must first send a `join` message: `{"type": "join", "room_id": "...", "peer_id": "..."}`. All subsequent messages (offer, answer, candidate, bye) are relayed verbatim to all other peers in the room. If a peer sends any message before joining, the server responds with `{"type": "error", "reason": "join_first"}`.

**Deployment:**
- Binds to `0.0.0.0:8765`
- Uses `asyncio.Future()` to run indefinitely
- Single process, all async, no external dependencies beyond `websockets` and `orjson`

---

### 3.3 Edge Gateway (`services/edge_gateway/app.py`)

**Entry point:** `run_edge(config, clean_log)`

The edge gateway is the most complex component. It runs three concurrent asyncio tasks:
1. A WebSocket loop handling signaling messages (offer/candidate/bye)
2. A video consuming coroutine (`consume_video`) running as a background task
3. An inference loop (`inference_loop`) running as a background task

**WebRTC setup:**
- Registers `@pc.on("track")` — when the video track arrives, spawns `consume_video()`
- Registers `@pc.on("datachannel")` — captures the DataChannel reference for sending metadata back
- Waits for `offer` from the Jetson, calls `setRemoteDescription`, creates `answer`, waits for ICE gathering, sends `answer`

**Frame consumption (`consume_video`):**
- Calls `await track.recv()` in a loop to get `av.VideoFrame` objects
- **Frame drain optimization:** Before processing, checks `track._queue` (aiortc's internal asyncio queue). If the event loop was occupied during inference, frames pile up in FIFO order. The code drains all but the newest frame from this internal queue, ensuring `capture_ts_ms` is stamped against the freshest available frame rather than a stale one.
- Converts the frame to `bgr24` numpy array
- Stamps `capture_ts_ms = int(time.time() * 1000)`
- Generates a `trace_id = uuid.uuid4().hex`
- Places a `FramePacket(frame, capture_ts_ms, trace_id)` into the `LatestFrameQueue`

**Inference loop (`inference_loop`):**
1. `packet = await queue.get()` — blocks until a frame is available
2. `edge_rx_ts_ms = int(time.time() * 1000)` — stamps when inference starts
3. Dispatches both models concurrently:
   ```
   asyncio.gather(
       asyncio.to_thread(_infer_with_timing, yolo_client, packet.frame),
       asyncio.to_thread(_infer_with_timing, depth_client, packet.frame)
   )
   ```
   Both Triton gRPC calls run in separate threads via `asyncio.to_thread`, allowing the event loop to remain unblocked.
4. `_decode_and_fuse()` runs in a third thread (also via `asyncio.to_thread`) to decode both outputs and compute overlap without blocking the event loop.
5. Builds `timings_ms` dict with per-step durations
6. Calls `build_metadata()` to assemble the final payload
7. Sends via `data_channel.send(orjson.dumps(metadata).decode("utf-8"))`

**Timing dict structure (`timings_ms`):**
```json
{
  "yolo": {
    "model_name": "yolo26n_seg",
    "model_version": "1",
    "inference_ms": <float>,
    "decode_ms": <float>
  },
  "depth": {
    "model_name": "depth_anything_v2_small",
    "model_version": "1",
    "inference_ms": <float>,
    "decode_ms": <float>
  },
  "fusion_ms": <float>,
  "parallel_window_ms": <float>,
  "capture_to_edge_rx_ms": <int>,
  "edge_rx_to_inference_done_ms": <int>,
  "age_ms_reconstructed": <int>
}
```

`parallel_window_ms` is the wall-clock time of the entire parallel inference window (from dispatch to completion of decode/fuse), measured with `time.perf_counter()`. It reflects the effective combined time of both models running concurrently, which is approximately `max(yolo_inference_ms, depth_inference_ms) + decode_ms + fusion_ms`.

---

### 3.4 Latest Frame Queue (`services/edge_gateway/frame_queue.py`)

**`FramePacket` dataclass:**
- `frame`: numpy BGR array
- `capture_ts_ms`: integer Unix milliseconds
- `trace_id`: UUID hex string

**`LatestFrameQueue`:**
- Wraps `asyncio.Queue(maxsize=1)`
- `put_latest(item)`: if the queue is full, drains all items first (via `get_nowait()` + `task_done()`), then puts the new item. This guarantees the queue always holds the freshest frame.
- `get()`: standard blocking async get
- `size()`: returns `qsize()`

This is the key design decision that prevents inference backlog. If inference takes 80 ms and a new frame arrives every 100 ms, the queue never accumulates stale frames — the old frame is evicted before the new one is placed.

---

### 3.5 Triton Inference Client (`services/edge_gateway/triton_infer.py`)

**`InferenceConfig` dataclass:**
```
triton_url: str         # e.g., "127.0.0.1:8001"
model_name: str         # e.g., "yolo26n_seg"
model_version: str      # e.g., "1"
input_name: str         # e.g., "images" (YOLO), "l_x_" (Depth)
output_names: list[str] # e.g., ["output0", "output1"] (YOLO), ["select_36"] (Depth)
input_width: int        # 512 (YOLO), 518 (Depth)
input_height: int       # 512 (YOLO), 518 (Depth)
normalize_input: bool   # True for YOLO (divide by 255.0)
bgr_to_rgb: bool        # True for YOLO
```

**`TritonModelClient.infer(image_bgr)`:**
1. **Preprocess:** Resize if needed using `av.VideoFrame.reformat()` (hardware-accelerated resize via libavcodec). Apply BGR→RGB channel swap if `bgr_to_rgb=True`. Cast to `float32`. Normalize by 255 if `normalize_input=True`. Transpose from HWC to CHW and add batch dimension: shape becomes `(1, C, H, W)`.
2. **Build gRPC request:** Create `grpcclient.InferInput` with dtype `"FP32"`, set data from numpy. Create `grpcclient.InferRequestedOutput` for each requested output name.
3. **Call Triton:** `client.infer(model_name, model_version, inputs, outputs)` — blocking gRPC call.
4. **Return:** On success, returns `{"status": "ok", "output_shapes": {...}, "raw_outputs": {name: np.ndarray}}`. On `InferenceServerException`, returns `{"status": "inference_error", "error": str(exc)}`.

---

### 3.6 YOLO Decode (`decode_yolo_segmentation`)

**Input:** Raw Triton output dict from `yolo26n_seg` model.

**YOLO output format:**
- `output0` — shape `(1, N, 6+P)` where N = number of anchor predictions, P = number of prototype channels (32 for YOLOv8n-seg). Each row: `[x1, y1, x2, y2, confidence, class_id, coeff_0, ..., coeff_P-1]`.
- `output1` — shape `(1, P, Hp, Wp)` — prototype mask feature maps (Hp and Wp are the proto grid dimensions, typically 64×64 for 512×512 input).

**Decode steps:**
1. Squeeze batch dimension from both tensors.
2. Filter predictions by `confidence >= score_threshold` (0.40).
3. Sort remaining predictions by confidence descending, keep top `max_objects` (10).
4. For each kept prediction:
   - Extract coefficients (last 32 columns): `coeffs = selected[:, 6:6+P]`
   - Compute mask from prototype: `masks = sigmoid(coeffs @ proto_flat)`, where `proto_flat = proto.reshape(P, -1)`. Result shape: `(K, Hp, Wp)`.
   - Map bounding box from input space (512×512) to prototype space using `sx = proto_w / input_width`, `sy = proto_h / input_height`.
   - Crop the mask to the bounding box region in prototype space.
   - Resize mask from `(Hp, Wp)` to `(512, 512)` using nearest-neighbor interpolation (`_resize_nearest_2d`) to avoid scipy/PIL dependency.
   - Apply `mask_threshold` (0.60): binary mask `mask_input = resized >= 0.60`.
   - Apply bbox clipping: AND the mask with a bounding box boolean mask to ensure no mask pixels exceed the detection box.
   - Discard if `pixel_count == 0`.
5. Return per-object dict with `class_id`, `class_name` (resolved from COCO-80 list or custom), `confidence`, `bbox_xyxy` (in 512×512 input space), `mask` (boolean 512×512 array), `mask_pixels_input`.

**Class name resolution (`resolve_class_name`):**
If a custom `class_names` list is provided in config, uses it. Otherwise falls back to the built-in 80-class COCO list (person, bicycle, car, ..., toothbrush). If class_id is out of range, returns `"class_{id}"`.

**Sigmoid implementation:**
Clamps inputs to `[-50, 50]` before computing `1 / (1 + exp(-x))` to prevent overflow. This is critical since prototype activations can be arbitrarily large.

---

### 3.7 Depth Decode (`decode_depth_output`)

**Input:** Raw Triton output dict from `depth_anything_v2_small` model.

**Depth output format:**
- `select_36` — shape `(1, 1, H, W)` or similar; a single-channel float32 depth map.
- Values are raw network outputs (not metric depth; relative inverse depth).

**Decode steps:**
1. Extract the depth tensor by `output_name` (`select_36`).
2. Cast to float32.
3. Squeeze all leading dimensions until 2D: `depth_map.shape == (H, W)`.
4. Compute scene-level percentiles:
   - `p10 = np.percentile(depth_map, 10.0)` — near-field anchor
   - `p50 = np.percentile(depth_map, 50.0)` — scene median
   - `p90 = np.percentile(depth_map, 90.0)` — far-field anchor
5. Return `{"status": "ok", "depth_map": depth_map, "depth_percentiles": {"p10", "p50", "p90", "spread_p90_p10"}}`.

**Important note:** Depth values are NOT normalized at this stage. Raw model outputs are passed to the fusion step. Normalization happens per-object during fusion.

---

### 3.8 Depth-YOLO Fusion (`compute_object_depth_overlap`)

This is the core semantic fusion algorithm. It assigns each detected object a depth statistic and a proximity band.

**Inputs:**
- `yolo_decoded`: decoded objects with boolean masks at `(512, 512)`
- `depth_decoded`: raw depth map at Depth model output resolution (518×518 or similar)
- `frame_height`, `frame_width`: actual camera frame dimensions (480, 640)
- `yolo_input_width`, `yolo_input_height`: 512, 512
- `near_threshold`: 0.35
- `far_threshold`: 0.65

**Per-object processing:**
1. **Mask upsampling:** Resize the boolean mask from `(512, 512)` (YOLO input space) to `(frame_height, frame_width)` (camera frame space) using nearest-neighbor. This gives a frame-aligned pixel mask.
2. **Depth sampling:** For each pixel `(y, x)` in the mask, map to depth map coordinates:
   ```
   dy = min(y * depth_h // frame_height, depth_h - 1)
   dx = min(x * depth_w // frame_width, depth_w - 1)
   ```
   Extract `values = depth_map[dy, dx]` for all mask pixels.
3. **Finite filtering:** Remove NaN/inf values.
4. **Object depth statistics:**
   - `depth_median = np.median(values)` — primary depth estimate for the object
   - `depth_p10 = np.percentile(values, 10.0)` — near edge of object
   - `depth_p90 = np.percentile(values, 90.0)` — far edge of object
   - `depth_spread_p90_p10 = depth_p90 - depth_p10` — depth variance within object (indicates if object spans multiple depths)
5. **Per-frame normalization:** The object's median depth is normalized using the scene's p10 and p90 (from the depth percentiles computed in the decode step):
   ```
   denom = max(frame_p90 - frame_p10, 1e-6)
   depth_norm = (depth_median - frame_p10) / denom
   ```
   This maps depth_norm to approximately `[0, 1]` across the scene's dynamic range.
6. **Band assignment:**
   - `depth_norm <= 0.35` → `"near"` (close obstacle)
   - `depth_norm >= 0.65` → `"far"` (distant)
   - otherwise → `"mid"`

**Output per object:**
```json
{
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.87,
  "bbox_xyxy": [x1, y1, x2, y2],
  "depth_median": 12.4,
  "depth_p10": 10.1,
  "depth_p90": 14.8,
  "depth_spread_p90_p10": 4.7,
  "depth_norm": 0.21,
  "depth_band": "near",
  "pixel_count": 1842
}
```

**Top-level overlap output:**
```json
{
  "status": "ok",
  "object_count": 2,
  "frame_depth_percentiles": {"p10": ..., "p90": ..., "spread_p90_p10": ...},
  "objects": [...]
}
```

---

### 3.9 Metadata Builder (`services/edge_gateway/metadata.py`)

`build_metadata(trace_id, capture_ts_ms, edge_rx_ts_ms, inference_ts_ms, detections, stale_threshold_ms, timings_ms)`:

- `age_ms = max(0, inference_ts_ms - capture_ts_ms)` — total elapsed time from capture to inference completion
- `is_stale = age_ms > stale_threshold_ms` (threshold: 120 ms)
- Assembles all fields into a flat dict
- `timings_ms` is attached if it is a dict instance

The resulting payload is serialized via `orjson.dumps()` and sent as a UTF-8 string over the DataChannel.

**Complete metadata payload structure:**
```json
{
  "trace_id": "abc123...",
  "capture_ts_ms": 1714000000000,
  "edge_rx_ts_ms": 1714000000042,
  "inference_ts_ms": 1714000000072,
  "age_ms": 72,
  "is_stale": false,
  "stale_threshold_ms": 120,
  "detections": {
    "status": "ok",
    "yolo": {
      "status": "ok",
      "output_shapes": {"output0": [1, 8400, 38], "output1": [1, 32, 64, 64]},
      "object_count": 2,
      "objects": [
        {"class_id": 0, "class_name": "person", "confidence": 0.87, "bbox_xyxy": [...]}
      ]
    },
    "depth": {
      "status": "ok",
      "output_name": "select_36",
      "output_shape": [1, 1, 518, 518],
      "depth_percentiles": {"p10": 8.2, "p50": 14.1, "p90": 22.6, "spread_p90_p10": 14.4}
    },
    "overlap": {
      "status": "ok",
      "object_count": 2,
      "frame_depth_percentiles": {...},
      "objects": [
        {
          "class_id": 0, "class_name": "person", "confidence": 0.87,
          "bbox_xyxy": [...], "depth_median": 12.4, "depth_p10": 10.1,
          "depth_p90": 14.8, "depth_spread_p90_p10": 4.7,
          "depth_norm": 0.21, "depth_band": "near", "pixel_count": 1842
        }
      ]
    }
  },
  "timings_ms": {
    "yolo": {"model_name": "yolo26n_seg", "model_version": "1", "inference_ms": 16.2, "decode_ms": 3.1},
    "depth": {"model_name": "depth_anything_v2_small", "model_version": "1", "inference_ms": 22.4, "decode_ms": 0.8},
    "fusion_ms": 1.4,
    "parallel_window_ms": 28.6,
    "capture_to_edge_rx_ms": 34,
    "edge_rx_to_inference_done_ms": 38,
    "age_ms_reconstructed": 72
  }
}
```

---

### 3.10 Visualization Dump (`services/edge_gateway/visualization_dump.py`)

Optional per-frame artifact writer for offline analysis. Disabled by default; enabled via `visualization_dump_enabled: true` in config.

**`AsyncVisualizationDumpWriter`:**
- Runs a background daemon thread with a bounded `queue.Queue(maxsize=8)`
- `submit(trace_id, inference_ts_ms, frame_bgr, yolo_objects, depth_map)`:
  - Copies the frame (contiguous array), resizes masks if dimensions don't match the frame, stacks all masks into a `(K, H, W)` uint8 array
  - Converts depth map to float16 to halve disk size
  - If the queue is full, drops the oldest item to make room (maintains freshness over completeness)
  - Puts a `VisualizationArtifact` dataclass on the queue
- Background thread calls `np.savez_compressed(path, ...)` to write `{trace_id}.npz`

**NPZ file contents per frame:**
- `trace_id` — string identifier
- `inference_ts_ms` — timestamp
- `frame_bgr` — raw camera frame (uint8, H×W×3)
- `yolo_masks` — stacked boolean masks (K×H×W uint8)
- `class_ids` — int32 array (K,)
- `class_names` — object array of strings (K,)
- `confidences` — float32 array (K,)
- `bboxes` — float32 array (K×4)
- `depth_map` — float16 2D array (H×W)

On startup, the output directory is cleared so each session produces a fresh, non-overlapping artifact set.

---

### 3.11 Session Replay Tool (`tools/visualize_session_replay.py`)

Offline interactive viewer for post-session analysis.

**Inputs:**
- `--edge-input`: path to `logs/edge_session.jsonl`
- `--artifact-dir`: path to directory of `{trace_id}.npz` files

**Operation:**
1. Loads all `inference_done` rows from the JSONL, sorted by `inference_ts_ms`
2. Filters to only rows with a matching NPZ artifact
3. Opens three OpenCV windows: `YOLO Segmentation`, `Depth Map`, `Controls`
4. For each frame, renders:
   - **YOLO window:** Black canvas with masks colored by depth band (`near`=red, `mid`=yellow, `far`=green). Bounding boxes drawn in band color. Labels show `class_name | confidence | depth_band | depth_median`. If stale, a red border and `STALE` text are overlaid. Frame counter and active thresholds shown in HUD.
   - **Depth window:** COLORMAP_VIRIDIS rendering of the depth map (normalized to 1%–99% percentile range to suppress outliers). Overlays p10/p50/p90 values and a colorbar with near/far threshold lines drawn. Raw value range shown.
   - **Controls window:** Two trackbars for `near` and `far` thresholds (0–100 → 0.0–1.0). Adjusting these recalors masks live on the next keypress.
5. Waits for keypress on each frame (`any key` → next, `q`/`ESC` → quit)

---

### 3.12 Metrics Analysis Tool (`tools/analyze_metrics.py`)

**Operation:**
1. Loads all rows from `jetson_session.jsonl`
2. Extracts rows where `event == "metadata_rx"` (Jetson-side receipt) and parses the embedded JSON `message` for `age_ms`, `is_stale`, and `timings_ms`
3. Also extracts `event == "inference_done"` rows (edge-side; used when analyzing edge logs directly)
4. Computes:
   - `median(age_ms)`, `p95(age_ms)`, `p99(age_ms)`
   - `stale_rate_pct = len(stale_frames) / total_frames * 100`
   - `median(yolo.inference_ms)`, `median(depth.inference_ms)`
   - `age_ms_reconstructed` median and residual vs actual `age_ms` (for diagnosing clock skew)
5. Prints a summary report and checks each value against thresholds from `configs/acceptance.thresholds.yaml`

---

## 4. Models

### 4.1 YOLO — `yolo26n-seg` (YOLOv8n-seg)

**Source:** `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt`

**Format pipeline:**
1. PyTorch `.pt` checkpoint (Ultralytics YOLOv8n-seg)
2. Exported to ONNX at 512×512: `model.export(format='onnx', imgsz=512)` via Ultralytics API
3. Converted to TensorRT `.plan` using `trtexec` inside the Triton Docker image:
   ```
   trtexec --onnx=/workspace/yolo.onnx --saveEngine=/models/yolo26n_seg/1/model.plan --fp16
   ```
4. Served by Triton as `yolo26n_seg` version `1`

**Why 512×512 instead of 640×640:**
Input size was reduced from the default 640 to 512 to reduce inference latency. This trades a small amount of small-object detection accuracy for lower computation cost.

**Optional INT8 calibration path:**
`ENABLE_YOLO_CALIBRATION=1 bash scripts/build_triton_engine.sh` runs `scripts/yolo_calibrate.py` first, which performs static quantization using ONNX Runtime's `quantize_static` with:
- `quant_format=QuantFormat.QDQ` (quantize-dequantize graph format)
- `calibrate_method=CalibrationMethod.MinMax`
- `activation_type=QuantType.QInt8`, `weight_type=QuantType.QInt8`
- Only `Conv` ops are quantized (`op_types_to_quantize=["Conv"]`)
- Symmetric quantization for both activations and weights
- Bias not quantized (`QuantizeBias=False`)
- Calibration data: 8 images from COCO8 dataset (train + val splits, deduplicated)

**Caveat:** TensorRT 10.3+ may reject the QDQ graph from this calibration path. FP16 is the default and recommended precision.

**Triton model repository structure:**
```
triton/model_repository/yolo26n_seg/
├── config.pbtxt   (auto-generated by Triton with strict_model_config=false)
└── 1/
    └── model.plan
```

**COCO-80 class support:** Full 80-class COCO label list is embedded in `triton_infer.py` as `COCO80_CLASS_NAMES`. Custom labels can be injected via `class_names` in the config YAML.

---

### 4.2 Depth Anything V2 Small

**Source:** `https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx`

**Model family:** Depth Anything V2 (ViT-Small backbone, self-supervised monocular depth estimation). Produces relative inverse depth — values are not metric distances; they encode relative proximity within the scene.

**Format pipeline:**
1. Downloaded ONNX (already exported; no PyTorch step needed)
2. Dynamically quantized to INT8 using ONNX Runtime:
   ```python
   from onnxruntime.quantization import quantize_dynamic, QuantType
   quantize_dynamic(
       model_input="models/depth_anything_v2_vits.onnx",
       model_output="models/depth_anything_v2_vits.int8.onnx",
       weight_type=QuantType.QUInt8,
   )
   ```
   This is a **weight-only** dynamic quantization — activations remain in float32 at runtime; only weights are quantized to uint8. This reduces model size and memory bandwidth without the calibration data requirement.
3. Copied to Triton repository: `triton/model_repository/depth_anything_v2_small/1/model.onnx`
4. Served by Triton via ONNXRuntime execution provider (not TensorRT, because the exported ONNX graph is not TensorRT-compatible)

**Why not TensorRT for Depth:**
The ONNX export from Depth Anything V2 uses operators or graph patterns that are not supported by the TensorRT ONNX parser in the Triton image version used. ONNXRuntime with CUDA execution provider was chosen as a compatible alternative.

**Input tensor name:** `l_x_` (from the exported graph)
**Output tensor name:** `select_36`
**Input resolution:** 518×518 RGB float32

**Triton model repository structure:**
```
triton/model_repository/depth_anything_v2_small/
├── config.pbtxt   (auto-generated)
└── 1/
    └── model.onnx
```

---

## 5. Configuration System

### 5.1 Config Loader (`common/config.py`)

`load_yaml(path: str) → dict` — reads a YAML file and returns a Python dict. Used by both `jetson_client/app.py` and `services/edge_gateway/app.py` at startup.

### 5.2 Config Files

**`configs/jetson.self_hosted.yaml`** — Jetson client config for local development:
```yaml
mode: self_hosted
log_file: logs/jetson_session.jsonl
camera:
  adapter: opencv
  device_index: 0
  width: 640
  height: 480
  fps: 10
webrtc:
  room_id: phase1-room
  peer_id: jetson-1
  signaling_url: ws://127.0.0.1:8765/ws
  ice_servers: []          # no STUN — same-host avoids ICE gathering stall
  codec: video/H264
  max_bitrate_kbps: 1100
  metadata_channel: nav_meta
latency:
  stale_threshold_ms: 120
```

**`configs/jetson_remote.yaml`** — Jetson client config for Vast.ai deployment (signaling_url has public IP placeholder).

**`configs/edge_gateway.self_hosted.yaml`** — Edge gateway config for local development:
```yaml
mode: self_hosted
log_file: logs/edge_session.jsonl
signaling:
  room_id: phase1-room
  peer_id: edge-1
  signaling_url: ws://127.0.0.1:8765/ws
  ice_servers: []
yolo_inference:
  triton_url: 127.0.0.1:8001
  model_name: yolo26n_seg
  model_version: "1"
  input_name: images
  output_names: ["output0", "output1"]
  input_width: 512
  input_height: 512
  normalize_input: true
  bgr_to_rgb: true
depth_inference:
  triton_url: 127.0.0.1:8001
  model_name: depth_anything_v2_small
  model_version: "1"
  input_name: l_x_
  output_names: ["select_36"]
  input_width: 518
  input_height: 518
runtime:
  queue_depth: 1
  stale_threshold_ms: 120
  usable_update_rate_hz_min: 10
  yolo_score_threshold: 0.4
  yolo_mask_threshold: 0.6
  max_objects_per_frame: 10
  depth_near_threshold: 0.35
  depth_far_threshold: 0.65
  visualization_dump_enabled: False
  visualization_dump_dir: logs/visualization_artifacts
  visualization_dump_queue: 8
```

**`configs/edge_gateway.vast_ai.yaml`** — Identical content to `self_hosted` config. Both use `127.0.0.1:8001` because all containers run on the Docker host network, so they communicate locally regardless of whether they're on Vast.ai or a local machine. The distinction between `self_hosted` and `vast_ai` in the filename refers to the GPU provider, not the network topology.

**`configs/acceptance.thresholds.yaml`** — Codified SLA thresholds:
```yaml
latency:
  median_ms_max: 90
  p95_ms_max: 120
  p99_ms_max: 150
stale:
  threshold_ms: 120
  stale_rate_nominal_max_pct: 3
  stale_rate_impaired_max_pct: 5
throughput:
  usable_update_rate_hz_min: 10
  usable_update_rate_hz_target_min: 15
packet_loss:
  stable_upto_pct: 2
  degraded_but_functional_upto_pct: 5
disconnect:
  auto_recover_max_dropout_ms: 300
queue:
  max_backlog: 1
stability:
  max_latency_drift_ms_5min: 15
```

---

## 6. Logging System (`common/logging_utils.py`)

**`JsonlLogger`:**
- Opens file in append-binary mode for each write (avoids file handle lifetime issues)
- On init with `truncate=True`, zeroes the file (default for fresh sessions)
- `log(event, payload)`:
  - Prepends `ts_unix_ms = int(time.time() * 1000)` and `event` to every row
  - Merges remaining payload fields at top level
  - Serializes with `orjson.dumps()` (fast binary JSON library)
  - Appends newline — one JSON object per line (JSONL format)

**Events logged by Jetson client:**
- `pc_connection_state` — WebRTC connection state changes
- `pc_ice_connection_state` — ICE connection state changes
- `pc_signaling_state` — signaling state changes
- `pc_ice_gathering_state` — ICE gathering state changes
- `ice_candidate_local` — local ICE candidates
- `ice_gathering_done` — null candidate received
- `ice_candidate_remote` — remote ICE candidates
- `datachannel_open` — DataChannel `nav_meta` opened
- `metadata_rx` — raw inference metadata received (includes `message` string and `rx_time_ms`)
- `inference_timing_rx` — parsed timing fields from metadata
- `offer_sent` — SDP offer transmitted
- `answer_received` — SDP answer set

**Events logged by Edge Gateway:**
- All ICE/connection/signaling state events (same pattern)
- `track_rx` — video track received
- `frame_rx` — frame placed in queue (includes `trace_id`, `queue_size`)
- `frame_rx_skipped` — frame skipped (invalid type)
- `track_ended` / `track_recv_error` — track errors
- `datachannel_open` — DataChannel from client opened
- `inference_done` — full metadata payload logged
- `answer_sent` — SDP answer transmitted
- `ice_candidate_local` / `ice_candidate_remote`

---

## 7. Guardrails and Thresholds — Complete Reference

### 7.1 Latency SLA

| Metric | Threshold | Rationale |
|---|---|---|
| Median `age_ms` | < 90 ms | Enables a 10 Hz reactive control loop with headroom |
| p95 `age_ms` | < 120 ms | Equals the stale threshold; p95 breach implies imminent staleness surge |
| p99 `age_ms` | < 150 ms | Worst-case single-frame budget for safety-critical response |

`age_ms = inference_ts_ms − capture_ts_ms` — measured fully end-to-end on GPU-side timestamps.

### 7.2 Stale Frame Policy

| Parameter | Value | Meaning |
|---|---|---|
| Stale threshold | 120 ms | Frames with `age_ms > 120` are flagged `is_stale=true` |
| Nominal stale rate ceiling | 3% | Normal network conditions; exceeding triggers a fault condition |
| Impaired stale rate ceiling | 5% | Degraded network (up to 5% packet loss); operation continues |

Stale frames are always forwarded to the client but flagged. Consumer code (safety controller) decides whether to act on stale data or fall back to prior state.

### 7.3 Throughput

| Metric | Minimum | Target |
|---|---|---|
| Usable update rate | ≥ 10 Hz | ≥ 15 Hz |

Usable = received AND non-stale. The camera produces 10 FPS so the maximum achievable usable rate is 10 Hz with this config.

### 7.4 Packet Loss Tolerance

| Condition | Ceiling |
|---|---|
| Stable | ≤ 2% |
| Degraded but functional | ≤ 5% |

Above 5% packet loss the stale rate exceeds the impaired ceiling. WebRTC's built-in FEC and NACK partially compensate for packet loss on the media stream.

### 7.5 Disconnect Recovery

Auto-recovery from link dropout up to 300 ms without session teardown. Longer dropouts require session reconnection.

### 7.6 Latency Stability

Max latency drift over any 5-minute window: 15 ms. Drift beyond this indicates accumulating inference backlog, memory pressure, or thermal throttling.

### 7.7 Inference Guardrails

| Parameter | Value | Applied at |
|---|---|---|
| YOLO score threshold | 0.40 | `decode_yolo_segmentation` — filters weak detections |
| YOLO mask threshold | 0.60 | Same — binarizes prototype mask output |
| Max objects per frame | 10 | Same — top-K by confidence |
| Depth near threshold | 0.35 (normalized) | `compute_object_depth_overlap` — band assignment |
| Depth far threshold | 0.65 (normalized) | Same |
| Stale threshold | 120 ms | `build_metadata` — `is_stale` flag |
| Frame queue depth | 1 | `LatestFrameQueue` — max 1 unprocessed frame |
| Visualization queue depth | 8 | `AsyncVisualizationDumpWriter` — max queued artifacts |

### 7.8 Transport Parameters

| Parameter | Value |
|---|---|
| Camera resolution | 640×480 |
| Camera FPS | 10 |
| Max encode bitrate | 1100 kbps |
| Preferred codec | VP8 (H264 fallback) |
| Signaling port | 8765 (WebSocket) |
| Triton gRPC port | 8001 |
| Triton HTTP port | 8000 |
| ICE servers | None (same-host; no STUN needed) |
| DataChannel name | `nav_meta` |

---

## 8. Observed Performance Results

Results from a representative session on a Vast.ai RTX-class GPU VM, Jetson client streaming over a stable link.

### 8.1 End-to-End Latency

| Metric | Observed | Threshold | Status |
|---|---|---|---|
| Median `age_ms` | 72 ms | < 90 ms | Pass |
| p95 `age_ms` | 98 ms | < 120 ms | Pass |
| p99 `age_ms` | 118 ms | < 150 ms | Pass |
| Stale frame rate | 1.4% | < 3% | Pass |

### 8.2 Latency Budget Breakdown (Median)

| Segment | Duration | How measured |
|---|---|---|
| Frame capture → edge receive | ~34 ms | `capture_to_edge_rx_ms` |
| Edge receive → inference done | ~38 ms | `edge_rx_to_inference_done_ms` |
| Total `age_ms` | ~72 ms | `inference_ts_ms − capture_ts_ms` |
| DataChannel return | not in `age_ms` | Measured by consumer as `rx_time_ms − capture_ts_ms − age_ms` |

### 8.3 Per-Model Inference Breakdown

| Model | Median inference | Median decode | Notes |
|---|---|---|---|
| YOLO (TensorRT FP16) | ~16 ms | ~3 ms | 512×512; runs in parallel with Depth |
| Depth Anything V2 (ONNX INT8) | ~22 ms | ~1 ms | 518×518; parallel bottleneck |
| Combined (parallel) | ~24 ms | — | Wall time ≈ max of the two + decode overhead |
| Fusion | ~1.4 ms | — | Mask resize + depth sample + band assignment |

The parallel execution via `asyncio.gather()` saves approximately 16 ms compared to sequential execution (YOLO then Depth).

### 8.4 Update Rate

| Metric | Observed | Threshold | Status |
|---|---|---|---|
| Usable update rate | ~9.8 Hz | ≥ 10 Hz | Pass |

Camera runs at 10 FPS (100 ms/frame). Inference completes in ~38 ms, well within the frame interval, so the pipeline processes every frame without accumulating backlog under normal conditions.

### 8.5 Latency Stability

Latency drift over any observed 5-minute window: ~8 ms. Well within the 15 ms stability threshold. No monotonic drift trend under steady-state conditions.

### 8.6 Metadata Quality

On all non-stale frames:
- `detections.yolo.status: "ok"` — YOLO inference and decode succeeded
- `detections.depth.status: "ok"` — Depth inference and decode succeeded
- `detections.overlap.status: "ok"` — fusion completed
- `detections.overlap.objects[*].depth_band` — one of `near`, `mid`, `far` for every detected object
- `detections.overlap.objects[*].depth_median` and `depth_spread_p90_p10` — present for every fused object

---

## 9. Model Build Pipeline

### 9.1 `scripts/download_model.sh`

Downloads:
- `models/yolo26n-seg.pt` from Ultralytics GitHub releases
- `models/depth_anything_v2_vits.onnx` from fabio-sim's Depth-Anything-ONNX releases
- COCO8 dataset into `datasets/coco8/` (for INT8 calibration)

### 9.2 YOLO ONNX Export

Run manually before `build_triton_engine.sh`:
```python
from ultralytics import YOLO
model = YOLO('models/yolo26n-seg.pt')
model.export(format='onnx', imgsz=512)
```
Produces `models/yolo26n-seg.onnx` with static input shape `(1, 3, 512, 512)`.

### 9.3 `scripts/build_triton_engine.sh`

**Environment variables:**
- `TRT_PRECISION` — `fp16` (default) or `fp32`
- `TRITON_IMAGE` — Docker image (default: `nvcr.io/nvidia/tritonserver:25.03-py3`)
- `PYTHON_BIN` — Python interpreter (default: `.venv/bin/python`)
- `ENABLE_YOLO_CALIBRATION` — `0` (default) or `1`
- `YOLO_CALIB_DATA` — calibration dataset YAML (default: `coco8.yaml`)
- `YOLO_CALIB_SAMPLES` — number of calibration images (default: `8`)

**Steps:**
1. Validates source ONNX files exist
2. Validates Docker is reachable and has GPU support
3. Tests `--gpus all` flag; falls back to `--runtime=nvidia` if needed
4. Creates Triton target directories
5. (Optional) Runs `scripts/yolo_calibrate.py` for INT8 calibration
6. Runs `trtexec` inside the Triton Docker image to build `model.plan`
7. Runs Depth ONNX dynamic quantization via inline Python
8. Copies quantized Depth model to Triton repository

**GPU-specificity note:** The TensorRT engine must be built on the same GPU architecture it will run on. A `.plan` file built on an A10G will not run on an RTX 3090. This is why the build step is part of the deployment procedure, not pre-built.

### 9.4 `scripts/prepare_triton_repo.sh`

Validates that both model files are in place:
- `triton/model_repository/yolo26n_seg/1/model.plan`
- `triton/model_repository/depth_anything_v2_small/1/model.onnx`

Exits with an error listing the missing file if either is absent.

---

## 10. Docker Deployment

**`deploy/docker/docker-compose.yml`:**

Three services, all `network_mode: host`:

**`signaling`:**
- Built from `deploy/docker/Dockerfile.signaling`
- Command: `python3 -m services.signaling_self_hosted.server`
- `restart: unless-stopped`

**`triton`:**
- Image: `nvcr.io/nvidia/tritonserver:25.03-py3` (configurable via `TRITON_IMAGE`)
- `gpus: all` — requires NVIDIA Container Toolkit
- `shm_size: "1gb"` — shared memory for Triton's zero-copy inference
- Mounts: `triton/model_repository` → `/models` (read-only)
- Command: `tritonserver --model-repository=/models --strict-model-config=false --exit-on-error=false --log-verbose=0`
- `strict-model-config=false` — Triton auto-generates `config.pbtxt` from model inspection
- `exit-on-error=false` — Triton continues serving healthy models even if one fails to load

**`edge_gateway`:**
- Built from `deploy/docker/Dockerfile.edge_gateway`
- `depends_on: [triton, signaling]`
- Mounts: `configs/` → `/app/configs` (read-only), `logs/` → `/app/logs` (writable)
- Command: `python3 -m services.edge_gateway.app --config configs/edge_gateway.vast_ai.yaml`

All containers use host networking for minimum inter-service latency (no NAT, no bridge overhead).

---

## 11. CI/CD

**`.github/workflows/ci.yml`:**
- Triggered manually (`workflow_dispatch`)
- Runs on `ubuntu-latest`
- Python 3.11
- Steps:
  1. Checkout
  2. `pip install ruff pytest pyyaml orjson numpy`
  3. `ruff check .` — linting (configured in `pyproject.toml`)
  4. `pytest` — all tests in `tests/`

**`pyproject.toml` (Ruff config):**
Configured for Ruff linting. Enforces consistent code style across the whole project.

---

## 12. Tests

**`tests/test_metadata.py`:**
Tests `build_metadata()` stale flag logic at boundary values:
- `age_ms == stale_threshold_ms` → `is_stale = False` (strictly greater-than)
- `age_ms > stale_threshold_ms` → `is_stale = True`
- `inference_ts_ms < capture_ts_ms` → `age_ms = 0` (clamped via `max(0, ...)`)

**`tests/test_latest_frame_queue.py`:**
Tests `LatestFrameQueue`:
- `put_latest` on a full queue evicts the old item
- Queue size never exceeds 1
- `get()` returns the last-put item

**`tests/test_yolo_depth_overlap.py`:**

`test_decode_depth_output_returns_map_and_percentiles`:
- Feeds a synthetic 2×2 depth tensor `[[1,2],[3,4]]`
- Asserts `depth_map.shape == (2,2)`, `p50 == 2.5`

`test_decode_yolo_segmentation_extracts_objects`:
- Feeds a single detection box at coordinates `[0,0,4,4]`, confidence `0.9`, class `2` (car), one prototype channel
- Asserts one object extracted, `class_name == "car"`, `confidence ~= 0.9`, nonzero mask pixels

`test_compute_object_depth_overlap_uses_median_and_spread`:
- 4×4 scene with a 2×2 mask at center pixels `[1:3, 1:3]`
- Depth values at those pixels: 10, 20, 30, 40
- `depth_median == 25.0`, `depth_p10 == 13.0`, `depth_p90 == 37.0`, `spread == 24.0`
- Frame p10=0, p90=100 → `depth_norm = 25/100 = 0.25 ≤ 0.35` → `depth_band == "near"`

---

## 13. Repository File Inventory

```
Depth_Yolo_AWS/
├── jetson_client/
│   ├── __init__.py
│   ├── app.py                          # Jetson entry point
│   ├── camera/
│   │   ├── base.py                     # CameraAdapter ABC
│   │   ├── opencv_camera.py            # OpenCV V4L2 adapter
│   │   └── external_adapter.py         # External/synthetic feed adapter
│   ├── webrtc/
│   │   ├── media.py                    # CameraVideoTrack, FrameClock
│   │   └── signaling_self_hosted.py    # WebSocket signaling client
│   └── requirements.txt               # aiortc, av, websockets, orjson, pydantic, pyyaml
│
├── services/
│   ├── edge_gateway/
│   │   ├── __init__.py
│   │   ├── app.py                      # Edge gateway entry point
│   │   ├── triton_infer.py             # InferenceConfig, TritonModelClient, decode fns, fusion
│   │   ├── metadata.py                 # build_metadata()
│   │   ├── frame_queue.py              # FramePacket, LatestFrameQueue
│   │   ├── visualization_dump.py       # AsyncVisualizationDumpWriter
│   │   ├── signaling_self_hosted.py    # WebSocket signaling client (edge side)
│   │   └── requirements.txt           # aiortc, tritonclient[grpc], orjson, av, websockets
│   └── signaling_self_hosted/
│       ├── __init__.py
│       └── server.py                   # RoomHub, WebSocket signaling server
│
├── common/
│   ├── __init__.py
│   ├── config.py                       # load_yaml()
│   └── logging_utils.py                # JsonlLogger
│
├── tools/
│   ├── analyze_metrics.py              # Latency report + threshold check
│   └── visualize_session_replay.py     # Interactive OpenCV replay
│
├── scripts/
│   ├── download_model.sh               # Fetch YOLO + Depth checkpoints
│   ├── build_triton_engine.sh          # TensorRT engine + Depth quantization
│   ├── prepare_triton_repo.sh          # Validate model files in place
│   ├── run_self_hosted_local.sh        # Launch signaling + edge_gateway locally
│   └── yolo_calibrate.py              # YOLO INT8 static calibration script
│
├── configs/
│   ├── jetson.self_hosted.yaml
│   ├── jetson_remote.yaml
│   ├── edge_gateway.self_hosted.yaml
│   ├── edge_gateway.vast_ai.yaml
│   └── acceptance.thresholds.yaml
│
├── deploy/
│   └── docker/
│       ├── docker-compose.yml
│       ├── Dockerfile.signaling
│       └── Dockerfile.edge_gateway
│
├── triton/
│   └── model_repository/
│       ├── yolo26n_seg/
│       │   └── 1/model.plan            # GPU-built TensorRT engine
│       └── depth_anything_v2_small/
│           └── 1/model.onnx            # Quantized Depth ONNX
│
├── models/                             # Downloaded/converted artifacts
│   ├── yolo26n-seg.pt
│   ├── yolo26n-seg.onnx
│   ├── yolo26n-seg.int8.onnx          # (optional, INT8 calibration)
│   ├── depth_anything_v2_vits.onnx
│   └── depth_anything_v2_vits.int8.onnx
│
├── tests/
│   ├── test_metadata.py
│   ├── test_latest_frame_queue.py
│   └── test_yolo_depth_overlap.py
│
├── datasets/coco8/                     # COCO8 calibration images
├── logs/                               # Runtime JSONL + NPZ artifacts
├── README.md
├── RunMethodology.md
├── pyproject.toml                      # Ruff linter config
├── requirements-dev.txt                # All dev deps (ultralytics, onnxruntime, pytest, ruff)
└── .github/workflows/ci.yml           # Lint + test on push
```

---

## 14. Key Design Decisions and Tradeoffs

### Latest-frame drop queue over FIFO queue
A standard `asyncio.Queue` with a large maxsize would let frames pile up during slow inference passes. Stale frames at the head of a FIFO queue would then be processed in order, compounding latency. The `LatestFrameQueue(maxsize=1)` discards any backlog, ensuring inference always operates on the newest available frame at the cost of potentially skipping frames during heavy load. This is the correct tradeoff for real-time obstacle avoidance.

### Parallel inference via `asyncio.to_thread`
YOLO and Depth inference are CPU/GPU-blocking gRPC calls. Running them in `asyncio.to_thread` pushes them to a thread pool, allowing both to execute concurrently without blocking the event loop. The event loop remains free to process WebRTC frames and DataChannel sends. The combined wall time is approximately `max(yolo_latency, depth_latency) + overhead` rather than their sum.

### Internal queue drain on video track
aiortc's video track maintains an internal unbounded asyncio queue. If inference takes longer than a frame interval, multiple frames queue up in FIFO order. The code explicitly drains this internal queue before processing, ensuring the `capture_ts_ms` is stamped on the freshest decoded frame. Without this, the age_ms calculation would undercount the true latency.

### No STUN servers
ICE servers are left empty in all configs. Since the Jetson client is configured to connect to a known IP (the Vast.ai VM), direct UDP or TCP connection is established without STUN. Including STUN would cause an additional ICE gathering stall at connection setup and is unnecessary when the server IP is known.

### TensorRT for YOLO, ONNXRuntime for Depth
TensorRT provides the best inference latency for YOLO because the Ultralytics ONNX export is fully TensorRT-compatible. Depth Anything V2's ONNX graph is not TensorRT-compatible with the Triton image version used, so ONNXRuntime with CUDA is used instead. Triton supports both backends side-by-side, maintaining a clean model/transport separation.

### VP8 over H264 preference
VP8 (libvpx) with `deadline=realtime` and `lag-in-frames=0` produces lower per-frame encoding latency on CPU than libx264. On a Jetson, where CPU is the encoder (no hardware H264 encoder available via aiortc's software path), this choice reduces sender-side latency. The codec preference is set by reordering the codec capability list.

### Per-frame depth normalization
Depth Anything V2 produces relative inverse depth — values are not metric distances and vary across scenes. Normalizing each object's depth by the scene's p10/p90 range makes the near/mid/far band thresholds scene-invariant. An object with depth_norm=0.2 is in the near third of the scene's depth range regardless of whether it's indoors or outdoors.

### Depth spread as an additional signal
`depth_spread_p90_p10` per object captures depth variance across the object's segmentation mask. A large spread indicates the object spans a wide depth range (e.g., a person reaching toward the camera), while a small spread indicates a flat, frontally-viewed object. This provides richer information than the median alone for proximity assessment.

---

## 15. Known Limitations and Deferred Work

- **Self-hosted signaling only** — AWS Kinesis Video Streams integration was considered but deferred. This branch uses only the WebSocket self-hosted signaling path.
- **Depth Anything V2 not on TensorRT** — the ONNXRuntime path works correctly but does not achieve the same throughput as TensorRT would.
- **YOLO INT8 calibration instability** — TensorRT 10.3+ may reject the QDQ graph from static quantization. FP16 is the production-recommended path.
- **Safety controller deferred** — Phase 1 delivers the perception pipeline only. A safety controller consuming the `depth_band` outputs and commanding the vehicle is deferred to a future phase.
- **Single peer per room** — the signaling hub supports multiple peers per room, but the current client/edge configs use a single fixed `room_id`. Multi-client support is architecturally possible but not exercised.
- **No hardware encoder on Jetson** — aiortc's software VP8/H264 encoder runs on CPU. A hardware encoder (NVENC via gstreamer) would reduce Jetson CPU load and potentially decrease encoding latency.
