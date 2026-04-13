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
    input_width: int = 640
    input_height: int = 640


class TritonModelClient:
    def __init__(self, cfg: InferenceConfig) -> None:
        self._cfg = cfg
        self._client = grpcclient.InferenceServerClient(url=cfg.triton_url, verbose=False)

    def _preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if (
            image_bgr.shape[0] != self._cfg.input_height
            or image_bgr.shape[1] != self._cfg.input_width
        ):
            frame = av.VideoFrame.from_ndarray(image_bgr, format="bgr24")
            frame = frame.reformat(
                width=self._cfg.input_width,
                height=self._cfg.input_height,
                format="bgr24",
            )
            image_bgr = frame.to_ndarray(format="bgr24")

        img = image_bgr.astype(np.float32)
        # NHWC->NCHW and batch=1 for expected Triton input layout.
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def infer(self, image_bgr: np.ndarray) -> dict:
        img = self._preprocess_image(image_bgr)

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
        raw_outputs: dict[str, np.ndarray] = {}
        for name in self._cfg.output_names:
            value = result.as_numpy(name)
            output_shapes[name] = list(value.shape) if value is not None else []
            if value is not None:
                raw_outputs[name] = value

        return {
            "status": "ok",
            "output_shapes": output_shapes,
            "raw_outputs": raw_outputs,
        }


class TritonYoloClient(TritonModelClient):
    pass


def summarize_depth_output(result: dict, output_name: str | None = None) -> dict:
    if result.get("status") != "ok":
        return {
            "status": result.get("status", "inference_error"),
            "error": result.get("error", "unknown_error"),
        }

    raw = result.get("raw_outputs")
    if not isinstance(raw, dict) or not raw:
        return {
            "status": "inference_error",
            "error": "missing_depth_output",
        }

    chosen_name = output_name
    if chosen_name is None:
        chosen_name = next(iter(raw.keys()))

    depth = raw.get(chosen_name)
    if not isinstance(depth, np.ndarray):
        return {
            "status": "inference_error",
            "error": f"depth_output_not_found:{chosen_name}",
        }

    depth_map = depth.astype(np.float32)
    while depth_map.ndim > 2:
        depth_map = depth_map[0]

    p10 = float(np.percentile(depth_map, 10.0))
    p50 = float(np.percentile(depth_map, 50.0))
    p90 = float(np.percentile(depth_map, 90.0))

    return {
        "status": "ok",
        "output_name": chosen_name,
        "output_shape": list(depth.shape),
        "depth_percentiles": {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "spread_p90_p10": p90 - p10,
        },
    }
