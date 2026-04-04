from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class FramePacket:
    frame: Any
    capture_ts_ms: int
    trace_id: str


class LatestFrameQueue:
    def __init__(self, maxsize: int = 1) -> None:
        self._queue: asyncio.Queue[FramePacket] = asyncio.Queue(maxsize=maxsize)

    async def put_latest(self, item: FramePacket) -> None:
        while self._queue.full():
            _ = self._queue.get_nowait()
            self._queue.task_done()
        await self._queue.put(item)

    async def get(self) -> FramePacket:
        return await self._queue.get()

    def size(self) -> int:
        return self._queue.qsize()
