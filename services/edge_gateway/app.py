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


async def run_edge(config: dict[str, Any]) -> None:
    logger = JsonlLogger(config["log_file"])

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

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-redef]
        if track.kind != "video":
            return

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
            return
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
            await signaling.send_answer(sdp=answer.sdp, sdp_type=answer.type)
            logger.log("answer_sent", {"mode": mode})
        elif msg_type == "candidate":
            candidate = candidate_from_sdp(msg["candidate"])
            candidate.sdpMid = msg.get("sdp_mid")
            candidate.sdpMLineIndex = msg.get("sdp_mline_index")
            await pc.addIceCandidate(candidate)
        elif msg_type == "bye":
            break

    await pc.close()
    await signaling.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    asyncio.run(run_edge(config))


if __name__ == "__main__":
    main()
