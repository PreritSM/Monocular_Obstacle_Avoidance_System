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

    def _open_camera(self, device_index: int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None

        return cap

    def start(self) -> None:
        cap = self._open_camera(self._device_index)
        if cap is None:
            for idx in range(8):
                if idx == self._device_index:
                    continue
                cap = self._open_camera(idx)
                if cap is not None:
                    self._device_index = idx
                    break

        if cap is None:
            raise RuntimeError(
                f"Unable to open any camera device (requested index {self._device_index}, tried 0-7)"
            )

        self._cap = cap

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
