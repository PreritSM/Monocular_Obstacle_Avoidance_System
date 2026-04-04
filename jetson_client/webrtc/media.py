from __future__ import annotations

import time
from fractions import Fraction

import av
from aiortc import VideoStreamTrack

from jetson_client.camera.base import CameraAdapter


class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, adapter: CameraAdapter) -> None:
        super().__init__()
        self._adapter = adapter
        self._pts = 0
        self._time_base = Fraction(1, 90000)

    async def recv(self) -> av.VideoFrame:
        ok, frame_bgr = self._adapter.read()
        if not ok:
            await self.next_timestamp()
            black = av.VideoFrame(width=640, height=480, format="bgr24")
            black.pts = self._pts
            black.time_base = self._time_base
            self._pts += 3000
            return black

        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 3000
        await self.next_timestamp()
        return frame


class FrameClock:
    def __init__(self) -> None:
        self._start_ns = time.time_ns()

    def now_ms(self) -> int:
        return int((time.time_ns() - self._start_ns) / 1_000_000)
