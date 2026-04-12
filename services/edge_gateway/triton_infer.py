from __future__ import annotations

from dataclasses import dataclass

import av
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
        # TensorRT model is configured for [1, 3, 640, 640].
        if image_bgr.shape[0] != 640 or image_bgr.shape[1] != 640:
            frame = av.VideoFrame.from_ndarray(image_bgr, format="bgr24")
            frame = frame.reformat(width=640, height=640, format="bgr24")
            image_bgr = frame.to_ndarray(format="bgr24")

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
        except InferenceServerException as exc:
            return {
                "status": "inference_error",
                "error": str(exc),
            }

        output_shapes: dict[str, list[int]] = {}
        for name in self._cfg.output_names:
            value = result.as_numpy(name)
            output_shapes[name] = list(value.shape) if value is not None else []

        return {
            "status": "ok",
            "output_shapes": output_shapes,
        }
