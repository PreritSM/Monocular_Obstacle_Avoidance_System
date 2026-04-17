from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import av
import numpy as np
import orjson
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from common.config import load_yaml
from common.logging_utils import JsonlLogger
from services.edge_gateway.frame_queue import FramePacket, LatestFrameQueue
from services.edge_gateway.metadata import build_metadata
from services.edge_gateway.signaling_self_hosted import SelfHostedSignalingClient
from services.edge_gateway.triton_infer import (
    compute_object_depth_overlap,
    decode_depth_output,
    decode_yolo_segmentation,
    InferenceConfig,
    TritonModelClient,
)
from services.edge_gateway.visualization_dump import AsyncVisualizationDumpWriter


async def _wait_for_ice_gathering_complete(pc: RTCPeerConnection, timeout_s: float = 3.0) -> None:
    start = time.monotonic()
    while pc.iceGatheringState != "complete":
        if (time.monotonic() - start) >= timeout_s:
            break
        await asyncio.sleep(0.05)


async def run_edge(config: dict[str, Any], clean_log: bool = True) -> None:
    logger = JsonlLogger(config["log_file"], truncate=clean_log)

    mode = config.get("mode", "self_hosted")
    if mode != "self_hosted":
        raise ValueError("Only self_hosted signaling is supported on this branch")

    scfg = config["signaling"]
    signaling = SelfHostedSignalingClient(
        url=scfg["signaling_url"], room_id=scfg["room_id"], peer_id=scfg["peer_id"]
    )

    await signaling.connect()

    ice_servers = [RTCIceServer(urls=item["urls"]) for item in scfg.get("ice_servers", [])]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    pending_candidates: list[dict[str, Any]] = []

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:  # type: ignore[no-redef]
        logger.log("pc_connection_state", {"state": pc.connectionState})

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:  # type: ignore[no-redef]
        logger.log("pc_ice_connection_state", {"state": pc.iceConnectionState})

    @pc.on("signalingstatechange")
    async def on_signalingstatechange() -> None:  # type: ignore[no-redef]
        logger.log("pc_signaling_state", {"state": pc.signalingState})

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange() -> None:
        logger.log("pc_ice_gathering_state", {"state": pc.iceGatheringState})


    yolo_cfg_dict = config.get("yolo_inference") or config.get("inference")
    depth_cfg_dict = config.get("depth_inference")
    if yolo_cfg_dict is None:
        raise ValueError("Missing yolo_inference/inference configuration")

    yolo_cfg_data = dict(yolo_cfg_dict)
    class_names_raw = yolo_cfg_data.pop("class_names", None)
    yolo_class_names: list[str] | None = None
    if isinstance(class_names_raw, list):
        yolo_class_names = [str(item) for item in class_names_raw]

    yolo_cfg = InferenceConfig(**yolo_cfg_data)
    yolo_client = TritonModelClient(yolo_cfg)

    depth_client = None
    depth_output_name = None
    if depth_cfg_dict is not None:
        depth_cfg = InferenceConfig(**depth_cfg_dict)
        depth_client = TritonModelClient(depth_cfg)
        if depth_cfg.output_names:
            depth_output_name = depth_cfg.output_names[0]

    runtime_cfg = config["runtime"]
    queue = LatestFrameQueue(maxsize=runtime_cfg.get("queue_depth", 1))
    stale_threshold_ms = runtime_cfg["stale_threshold_ms"]
    depth_near_threshold = float(runtime_cfg.get("depth_near_threshold", 0.35))
    depth_far_threshold = float(runtime_cfg.get("depth_far_threshold", 0.65))
    yolo_score_threshold = float(runtime_cfg.get("yolo_score_threshold", 0.25))
    yolo_mask_threshold = float(runtime_cfg.get("yolo_mask_threshold", 0.5))
    max_objects_per_frame = int(runtime_cfg.get("max_objects_per_frame", 20))
    visualization_dump_enabled = bool(runtime_cfg.get("visualization_dump_enabled", False))
    visualization_dump_dir = str(runtime_cfg.get("visualization_dump_dir", "logs/visualization_artifacts"))
    visualization_dump_queue = int(runtime_cfg.get("visualization_dump_queue", 8))

    visualization_writer = None
    if visualization_dump_enabled:
        visualization_writer = AsyncVisualizationDumpWriter(
            output_dir=visualization_dump_dir,
            max_queue_size=max(1, visualization_dump_queue),
        )

    data_channel = None

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:  # type: ignore[no-redef]
        nonlocal data_channel
        data_channel = channel

        @channel.on("open")
        def on_open() -> None:
            logger.log("datachannel_open", {"label": channel.label})

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-redef]
        if track.kind != "video":
            return

        logger.log("track_rx", {"kind": track.kind})

        async def consume_video() -> None:
            while True:
                frame: av.VideoFrame = await track.recv()
                frame_np = frame.to_ndarray(format="bgr24")
                now_ms = int(time.time() * 1000)
                trace_id = uuid.uuid4().hex
                await queue.put_latest(
                    FramePacket(frame=frame_np, capture_ts_ms=now_ms, trace_id=trace_id)
                )
                logger.log("frame_rx", {"trace_id": trace_id, "queue_size": queue.size()})

        asyncio.create_task(consume_video())

    @pc.on("icecandidate")
    async def on_icecandidate(candidate) -> None:  # type: ignore[no-redef]
        if candidate is None:
            logger.log("ice_gathering_done", {})
            return
        logger.log("ice_candidate_local", {
            "candidate": candidate.to_sdp(),
            "sdpMid": candidate.sdpMid,
            "sdpMLineIndex": candidate.sdpMLineIndex,
        })
        await signaling.send_candidate(
            candidate=candidate.to_sdp(),
            sdp_mid=candidate.sdpMid,
            sdp_mline_index=candidate.sdpMLineIndex,
        )

    def _infer_with_timing(client: TritonModelClient, frame) -> tuple[dict, float]:
        start = time.perf_counter()
        result = client.infer(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result, elapsed_ms

    async def inference_loop() -> None:
        while True:
            packet = await queue.get()
            edge_rx_ts_ms = int(time.time() * 1000)
            inference_window_start = time.perf_counter()

            yolo_task = asyncio.to_thread(_infer_with_timing, yolo_client, packet.frame)
            if depth_client is not None:
                depth_task = asyncio.to_thread(_infer_with_timing, depth_client, packet.frame)
                (yolo_result, yolo_inference_ms), (depth_result_raw, depth_inference_ms) = await asyncio.gather(
                    yolo_task,
                    depth_task,
                )
                depth_decode_start = time.perf_counter()
                depth_decoded = decode_depth_output(
                    depth_result_raw,
                    output_name=depth_output_name,
                )
                depth_decode_ms = (time.perf_counter() - depth_decode_start) * 1000.0
            else:
                yolo_result, yolo_inference_ms = await yolo_task
                depth_inference_ms = None
                depth_decode_ms = 0.0
                depth_decoded = {
                    "status": "disabled",
                }

            yolo_decode_start = time.perf_counter()

            yolo_decoded = decode_yolo_segmentation(
                yolo_result,
                input_width=yolo_cfg.input_width,
                input_height=yolo_cfg.input_height,
                score_threshold=yolo_score_threshold,
                mask_threshold=yolo_mask_threshold,
                max_objects=max_objects_per_frame,
                class_names=yolo_class_names,
            )
            yolo_decode_ms = (time.perf_counter() - yolo_decode_start) * 1000.0

            overlap_start = time.perf_counter()

            if depth_decoded.get("status") == "ok":
                overlap = compute_object_depth_overlap(
                    yolo_decoded=yolo_decoded,
                    depth_decoded=depth_decoded,
                    frame_height=packet.frame.shape[0],
                    frame_width=packet.frame.shape[1],
                    yolo_input_width=yolo_cfg.input_width,
                    yolo_input_height=yolo_cfg.input_height,
                    near_threshold=depth_near_threshold,
                    far_threshold=depth_far_threshold,
                )
                depth_payload = {
                    "status": "ok",
                    "output_name": depth_decoded.get("output_name"),
                    "output_shape": depth_decoded.get("output_shape"),
                    "depth_percentiles": depth_decoded.get("depth_percentiles", {}),
                }
            else:
                overlap = {
                    "status": "disabled",
                    "object_count": 0,
                    "objects": [],
                }
                depth_payload = {
                    "status": depth_decoded.get("status", "inference_error"),
                    "error": depth_decoded.get("error", "depth_not_available"),
                }
            overlap_ms = (time.perf_counter() - overlap_start) * 1000.0

            yolo_objects_compact: list[dict[str, Any]] = []
            for obj in yolo_decoded.get("objects", []):
                yolo_objects_compact.append(
                    {
                        "class_id": int(obj.get("class_id", -1)),
                        "class_name": str(obj.get("class_name", "")),
                        "confidence": float(obj.get("confidence", 0.0)),
                        "bbox_xyxy": [float(v) for v in obj.get("bbox_xyxy", [0, 0, 0, 0])],
                    }
                )

            fused = {
                "status": "ok",
                "yolo": {
                    "status": yolo_decoded.get("status", "inference_error"),
                    "output_shapes": yolo_result.get("output_shapes", {}),
                    "object_count": yolo_decoded.get("object_count", 0),
                    "objects": yolo_objects_compact,
                },
                "depth": depth_payload,
                "overlap": overlap,
            }

            inference_ts_ms = int(time.time() * 1000)
            edge_to_inference_done_ms = max(0, inference_ts_ms - edge_rx_ts_ms)
            capture_to_edge_rx_ms = max(0, edge_rx_ts_ms - packet.capture_ts_ms)
            parallel_window_ms = (time.perf_counter() - inference_window_start) * 1000.0
            timings_ms = {
                "yolo": {
                    "model_name": yolo_cfg.model_name,
                    "model_version": yolo_cfg.model_version,
                    "inference_ms": float(yolo_inference_ms),
                    "decode_ms": float(yolo_decode_ms),
                },
                "depth": {
                    "model_name": depth_cfg_dict.get("model_name") if isinstance(depth_cfg_dict, dict) else None,
                    "model_version": depth_cfg_dict.get("model_version") if isinstance(depth_cfg_dict, dict) else None,
                    "inference_ms": float(depth_inference_ms) if depth_inference_ms is not None else None,
                    "decode_ms": float(depth_decode_ms),
                },
                "fusion_ms": float(overlap_ms),
                "parallel_window_ms": float(parallel_window_ms),
                "capture_to_edge_rx_ms": int(capture_to_edge_rx_ms),
                "edge_rx_to_inference_done_ms": int(edge_to_inference_done_ms),
                "age_ms_reconstructed": int(capture_to_edge_rx_ms + edge_to_inference_done_ms),
            }

            metadata = build_metadata(
                trace_id=packet.trace_id,
                capture_ts_ms=packet.capture_ts_ms,
                edge_rx_ts_ms=edge_rx_ts_ms,
                inference_ts_ms=inference_ts_ms,
                detections=fused,
                stale_threshold_ms=stale_threshold_ms,
                timings_ms=timings_ms,
            )

            if visualization_writer is not None:
                depth_map = depth_decoded.get("depth_map")
                if not isinstance(depth_map, np.ndarray):
                    depth_map = None
                visualization_writer.submit(
                    trace_id=packet.trace_id,
                    inference_ts_ms=inference_ts_ms,
                    frame_bgr=packet.frame,
                    yolo_objects=yolo_decoded.get("objects", []),
                    depth_map=depth_map,
                )

            logger.log("inference_done", metadata)
            if data_channel is not None and data_channel.readyState == "open":
                data_channel.send(orjson.dumps(metadata).decode("utf-8"))

    asyncio.create_task(inference_loop())

    while True:
        msg = await signaling.recv()
        msg_type = msg.get("type")
        if msg_type == "offer":
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"])
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await _wait_for_ice_gathering_complete(pc)
            local_answer = pc.localDescription
            if local_answer is None:
                raise RuntimeError("Local answer was not set")
            await signaling.send_answer(sdp=local_answer.sdp, sdp_type=local_answer.type)
            while pending_candidates:
                candidate_msg = pending_candidates.pop(0)
                candidate = candidate_from_sdp(candidate_msg["candidate"])
                candidate.sdpMid = candidate_msg.get("sdp_mid")
                candidate.sdpMLineIndex = candidate_msg.get("sdp_mline_index")
                await pc.addIceCandidate(candidate)
            logger.log("answer_sent", {"mode": mode})
        elif msg_type == "candidate":
            logger.log("ice_candidate_remote", {"raw": msg.get("candidate"), "sdp_mid": msg.get("sdp_mid")})
            candidate = candidate_from_sdp(msg["candidate"])
            candidate.sdpMid = msg.get("sdp_mid")
            candidate.sdpMLineIndex = msg.get("sdp_mline_index")
            if pc.remoteDescription is None:
                pending_candidates.append(msg)
            else:
                await pc.addIceCandidate(candidate)
        elif msg_type == "bye":
            break

    await pc.close()
    await signaling.close()
    if visualization_writer is not None:
        visualization_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--clean-log",
        dest="clean_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the log file before starting (default: enabled).",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    asyncio.run(run_edge(config, clean_log=args.clean_log))


if __name__ == "__main__":
    main()
