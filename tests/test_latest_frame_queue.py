from __future__ import annotations

import asyncio

from services.edge_gateway.frame_queue import FramePacket, LatestFrameQueue


async def _run() -> None:
    q = LatestFrameQueue(maxsize=1)
    await q.put_latest(FramePacket(frame=None, capture_ts_ms=1, trace_id="a"))
    await q.put_latest(FramePacket(frame=None, capture_ts_ms=2, trace_id="b"))
    item = await q.get()
    assert item.trace_id == "b"


def test_latest_wins() -> None:
    asyncio.run(_run())
