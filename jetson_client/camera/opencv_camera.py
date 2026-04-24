from __future__ import annotations

import cv2
import numpy as np

from jetson_client.camera.base import CameraAdapter


class OpenCVCameraAdapter(CameraAdapter):
    def __init__(self, device_index: int, width: int, height: int, fps: int) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read(self) -> tuple[bool, np.ndarray]:
        if self._cap is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        ok, frame = self._cap.read()
        if not ok:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, frame

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
