from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


@dataclass
class InferenceConfig:
    triton_url: str
    model_name: str
    model_version: str
    input_name: str
    output_names: list[str]


class TritonYoloClient:
    def __init__(self, cfg: InferenceConfig) -> None:
        self._cfg = cfg
        self._client = grpcclient.InferenceServerClient(url=cfg.triton_url, verbose=False)

    def infer(self, image_bgr: np.ndarray) -> dict:
        img = image_bgr.astype(np.float32)
        # NHWC->NCHW and batch=1 for expected TensorRT/Triton input layout.
        img = np.transpose(img, (2, 0, 1))[None, ...]

        infer_input = grpcclient.InferInput(self._cfg.input_name, img.shape, "FP32")
        infer_input.set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput(name) for name in self._cfg.output_names]

        try:
            result = self._client.infer(
                model_name=self._cfg.model_name,
                model_version=self._cfg.model_version,
                inputs=[infer_input],
                outputs=outputs,
            )
        except InferenceServerException:
            return {
                "detections": [],
                "status": "inference_error",
            }

        raw = {}
        for name in self._cfg.output_names:
            value = result.as_numpy(name)
            raw[name] = value.tolist() if value is not None else []

        return {
            "detections": raw,
            "status": "ok",
        }
