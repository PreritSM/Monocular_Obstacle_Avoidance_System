from __future__ import annotations

from collections import deque

import numpy as np

from jetson_client.camera.base import CameraAdapter


class ExternalFeedAdapter(CameraAdapter):
    """Drop-in adapter for integrating an existing camera/feed pipeline."""

    def __init__(self, max_queue: int = 2) -> None:
        self._queue: deque[np.ndarray] = deque(maxlen=max_queue)

    def start(self) -> None:
        return

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        self._queue.append(frame_bgr)

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._queue:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, self._queue.pop()

    def stop(self) -> None:
        self._queue.clear()
