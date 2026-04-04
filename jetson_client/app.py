from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from common.config import load_yaml
from common.logging_utils import JsonlLogger
from jetson_client.camera.external_adapter import ExternalFeedAdapter
from jetson_client.camera.opencv_camera import OpenCVCameraAdapter
from jetson_client.webrtc.media import CameraVideoTrack
from jetson_client.webrtc.signaling_aws_kvs import AwsKvsSignalingClient
from jetson_client.webrtc.signaling_self_hosted import SelfHostedSignalingClient


async def run_sender(config: dict[str, Any]) -> None:
    logger = JsonlLogger(config["log_file"])

    cam_cfg = config["camera"]
    adapter_name = cam_cfg.get("adapter", "opencv")
    if adapter_name == "opencv":
        adapter = OpenCVCameraAdapter(
            device_index=cam_cfg["device_index"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
        )
    else:
        adapter = ExternalFeedAdapter()

    adapter.start()

    mode = config["mode"]
    if mode == "self_hosted":
        wcfg = config["webrtc"]
        signaling = SelfHostedSignalingClient(
            url=wcfg["signaling_url"], room_id=wcfg["room_id"], peer_id=wcfg["peer_id"]
        )
    else:
        wcfg = config["webrtc"]
        signaling = AwsKvsSignalingClient(
            channel_name=wcfg["channel_name"], region=wcfg["region"], client_id=wcfg["client_id"]
        )

    await signaling.connect()

    ice_servers = [RTCIceServer(urls=item["urls"]) for item in wcfg.get("ice_servers", [])]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))

    @pc.on("icecandidate")
    async def on_icecandidate(candidate) -> None:  # type: ignore[no-redef]
        if candidate is None:
            return
        await signaling.send_candidate(
            candidate=candidate.to_sdp(),
            sdp_mid=candidate.sdpMid,
            sdp_mline_index=candidate.sdpMLineIndex,
        )

    metadata_channel = pc.createDataChannel(wcfg.get("metadata_channel", "nav_meta"))

    @metadata_channel.on("message")
    def on_message(message: str) -> None:
        now_ms = int(time.time() * 1000)
        logger.log("metadata_rx", {"message": message, "rx_time_ms": now_ms})

    pc.addTrack(CameraVideoTrack(adapter))

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await signaling.send_offer(sdp=offer.sdp, sdp_type=offer.type)

    logger.log("offer_sent", {"mode": mode})

    while True:
        msg = await signaling.recv()
        msg_type = msg.get("type")
        if msg_type == "answer":
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"])
            )
            logger.log("answer_received", {})
        elif msg_type == "candidate":
            candidate = candidate_from_sdp(msg["candidate"])
            candidate.sdpMid = msg.get("sdp_mid")
            candidate.sdpMLineIndex = msg.get("sdp_mline_index")
            await pc.addIceCandidate(candidate)
        elif msg_type == "bye":
            break

    await pc.close()
    await signaling.close()
    adapter.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    asyncio.run(run_sender(config))


if __name__ == "__main__":
    main()
