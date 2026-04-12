from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any

import av
import orjson
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from common.config import load_yaml
from common.logging_utils import JsonlLogger
from services.edge_gateway.frame_queue import FramePacket, LatestFrameQueue
from services.edge_gateway.metadata import build_metadata
from services.edge_gateway.signaling_self_hosted import SelfHostedSignalingClient
from services.edge_gateway.triton_infer import InferenceConfig, TritonYoloClient


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


    inf_cfg = InferenceConfig(**config["inference"])
    infer_client = TritonYoloClient(inf_cfg)

    runtime_cfg = config["runtime"]
    queue = LatestFrameQueue(maxsize=runtime_cfg.get("queue_depth", 1))
    stale_threshold_ms = runtime_cfg["stale_threshold_ms"]

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

    async def inference_loop() -> None:
        while True:
            packet = await queue.get()
            edge_rx_ts_ms = int(time.time() * 1000)
            result = infer_client.infer(packet.frame)
            inference_ts_ms = int(time.time() * 1000)
            metadata = build_metadata(
                trace_id=packet.trace_id,
                capture_ts_ms=packet.capture_ts_ms,
                edge_rx_ts_ms=edge_rx_ts_ms,
                inference_ts_ms=inference_ts_ms,
                detections=result,
                stale_threshold_ms=stale_threshold_ms,
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
