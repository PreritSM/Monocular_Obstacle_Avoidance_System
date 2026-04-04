from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraAdapter(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
