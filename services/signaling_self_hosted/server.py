from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass

import orjson
import websockets


@dataclass
class Peer:
    peer_id: str
    ws: websockets.WebSocketServerProtocol


class RoomHub:
    def __init__(self) -> None:
        self.rooms: dict[str, dict[str, Peer]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def join(self, room_id: str, peer_id: str, ws: websockets.WebSocketServerProtocol) -> None:
        async with self._lock:
            self.rooms[room_id][peer_id] = Peer(peer_id=peer_id, ws=ws)

    async def leave(self, room_id: str, peer_id: str) -> None:
        async with self._lock:
            if room_id in self.rooms and peer_id in self.rooms[room_id]:
                del self.rooms[room_id][peer_id]
            if room_id in self.rooms and not self.rooms[room_id]:
                del self.rooms[room_id]

    async def relay(self, room_id: str, src_peer_id: str, payload: dict) -> None:
        peers = self.rooms.get(room_id, {})
        data = orjson.dumps(payload).decode("utf-8")
        for peer_id, peer in peers.items():
            if peer_id == src_peer_id:
                continue
            await peer.ws.send(data)


hub = RoomHub()


async def handler(ws: websockets.WebSocketServerProtocol) -> None:
    room_id = ""
    peer_id = ""
    try:
        async for raw in ws:
            msg = orjson.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "join":
                room_id = msg["room_id"]
                peer_id = msg["peer_id"]
                await hub.join(room_id, peer_id, ws)
                continue

            if not room_id or not peer_id:
                await ws.send(orjson.dumps({"type": "error", "reason": "join_first"}).decode("utf-8"))
                continue

            await hub.relay(room_id=room_id, src_peer_id=peer_id, payload=msg)
    finally:
        if room_id and peer_id:
            await hub.leave(room_id, peer_id)


async def run(host: str = "0.0.0.0", port: int = 8765) -> None:
    async with websockets.serve(handler, host, port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(run())
