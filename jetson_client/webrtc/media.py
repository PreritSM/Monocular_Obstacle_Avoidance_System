from __future__ import annotations

import time
from fractions import Fraction

import av
import cv2
from aiortc import VideoStreamTrack

from jetson_client.camera.base import CameraAdapter


class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, adapter: CameraAdapter, width: int | None = None, height: int | None = None) -> None:
        super().__init__()
        self._adapter = adapter
        self._width = width
        self._height = height
        self._pts = 0
        self._time_base = Fraction(1, 90000)

    def _resize_if_needed(self, frame_bgr):
        if self._width is None or self._height is None:
            return frame_bgr
        if frame_bgr.shape[1] == self._width and frame_bgr.shape[0] == self._height:
            return frame_bgr
        return cv2.resize(frame_bgr, (self._width, self._height), interpolation=cv2.INTER_AREA)

    async def recv(self) -> av.VideoFrame:
        ok, frame_bgr = self._adapter.read()
        if not ok:
            await self.next_timestamp()
            black_width = self._width if self._width is not None else 640
            black_height = self._height if self._height is not None else 480
            black = av.VideoFrame(width=black_width, height=black_height, format="bgr24")
            black.pts = self._pts
            black.time_base = self._time_base
            self._pts += 3000
            return black

        frame_bgr = self._resize_if_needed(frame_bgr)
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
