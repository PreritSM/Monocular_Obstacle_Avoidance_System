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
from jetson_client.webrtc.signaling_self_hosted import SelfHostedSignalingClient


async def _wait_for_ice_gathering_complete(pc: RTCPeerConnection, timeout_s: float = 3.0) -> None:
    start = time.monotonic()
    while pc.iceGatheringState != "complete":
        if (time.monotonic() - start) >= timeout_s:
            break
        await asyncio.sleep(0.05)


async def run_sender(config: dict[str, Any], clean_log: bool = True) -> None:
    logger = JsonlLogger(config["log_file"], truncate=clean_log)

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

    mode = config.get("mode", "self_hosted")
    if mode != "self_hosted":
        raise ValueError("Only self_hosted signaling is supported on this branch")

    wcfg = config["webrtc"]
    signaling = SelfHostedSignalingClient(
        url=wcfg["signaling_url"], room_id=wcfg["room_id"], peer_id=wcfg["peer_id"]
    )

    await signaling.connect()

    ice_servers = [RTCIceServer(urls=item["urls"]) for item in wcfg.get("ice_servers", [])]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    pending_candidates: list[Any] = []

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

    metadata_channel = pc.createDataChannel(wcfg.get("metadata_channel", "nav_meta"))

    @metadata_channel.on("open")
    def on_open() -> None:
        logger.log("datachannel_open", {"label": metadata_channel.label})

    @metadata_channel.on("message")
    def on_message(message: str) -> None:
        now_ms = int(time.time() * 1000)
        logger.log("metadata_rx", {"message": message, "rx_time_ms": now_ms})

    pc.addTrack(CameraVideoTrack(adapter))

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _wait_for_ice_gathering_complete(pc)
    local_offer = pc.localDescription
    if local_offer is None:
        raise RuntimeError("Local offer was not set")
    await signaling.send_offer(sdp=local_offer.sdp, sdp_type=local_offer.type)

    logger.log("offer_sent", {"mode": mode})

    while True:
        msg = await signaling.recv()
        msg_type = msg.get("type")
        if msg_type == "answer":
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"])
            )
            while pending_candidates:
                candidate_msg = pending_candidates.pop(0)
                candidate = candidate_from_sdp(candidate_msg["candidate"])
                candidate.sdpMid = candidate_msg.get("sdp_mid")
                candidate.sdpMLineIndex = candidate_msg.get("sdp_mline_index")
                await pc.addIceCandidate(candidate)
            logger.log("answer_received", {})
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
    adapter.stop()


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
    asyncio.run(run_sender(config, clean_log=args.clean_log))


if __name__ == "__main__":
    main()
