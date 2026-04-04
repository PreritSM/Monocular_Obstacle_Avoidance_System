from __future__ import annotations

import asyncio
from typing import Any

import orjson
import websockets


class SelfHostedSignalingClient:
    def __init__(self, url: str, room_id: str, peer_id: str) -> None:
        self._url = url
        self._room_id = room_id
        self._peer_id = peer_id
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._rx: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def connect(self) -> None:
        self._ws = await websockets.connect(self._url, ping_interval=10, ping_timeout=10)
        await self._send({"type": "join", "room_id": self._room_id, "peer_id": self._peer_id})
        asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            msg = orjson.loads(raw)
            await self._rx.put(msg)

    async def _send(self, message: dict[str, Any]) -> None:
        assert self._ws is not None
        await self._ws.send(orjson.dumps(message).decode("utf-8"))

    async def send_offer(self, sdp: str, sdp_type: str) -> None:
        await self._send({"type": "offer", "room_id": self._room_id, "peer_id": self._peer_id, "sdp": sdp, "sdp_type": sdp_type})

    async def send_answer(self, sdp: str, sdp_type: str) -> None:
        await self._send({"type": "answer", "room_id": self._room_id, "peer_id": self._peer_id, "sdp": sdp, "sdp_type": sdp_type})

    async def send_candidate(self, candidate: str, sdp_mid: str, sdp_mline_index: int) -> None:
        await self._send({
            "type": "candidate",
            "room_id": self._room_id,
            "peer_id": self._peer_id,
            "candidate": candidate,
            "sdp_mid": sdp_mid,
            "sdp_mline_index": sdp_mline_index,
        })

    async def recv(self) -> dict[str, Any]:
        return await self._rx.get()

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
