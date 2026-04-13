from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import av
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


COCO80_CLASS_NAMES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _resize_nearest_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = arr.shape[:2]
    y_idx = np.minimum((np.arange(target_h) * src_h // target_h), src_h - 1)
    x_idx = np.minimum((np.arange(target_w) * src_w // target_w), src_w - 1)
    return arr[y_idx[:, None], x_idx[None, :]]


def resolve_class_name(class_id: int, class_names: list[str] | None = None) -> str:
    if class_names is not None and 0 <= class_id < len(class_names):
        return class_names[class_id]
    if 0 <= class_id < len(COCO80_CLASS_NAMES):
        return COCO80_CLASS_NAMES[class_id]
    return f"class_{class_id}"


def decode_yolo_segmentation(
    result: dict,
    *,
    boxes_output_name: str = "output0",
    proto_output_name: str = "output1",
    input_width: int = 640,
    input_height: int = 640,
    score_threshold: float = 0.25,
    mask_threshold: float = 0.5,
    max_objects: int = 20,
    class_names: list[str] | None = None,
) -> dict:
    if result.get("status") != "ok":
        return {
            "status": result.get("status", "inference_error"),
            "error": result.get("error", "unknown_error"),
            "object_count": 0,
            "objects": [],
        }

    raw = result.get("raw_outputs")
    if not isinstance(raw, dict):
        return {
            "status": "inference_error",
            "error": "missing_raw_outputs",
            "object_count": 0,
            "objects": [],
        }

    boxes = raw.get(boxes_output_name)
    proto = raw.get(proto_output_name)
    if not isinstance(boxes, np.ndarray) or not isinstance(proto, np.ndarray):
        return {
            "status": "inference_error",
            "error": "missing_yolo_outputs",
            "object_count": 0,
            "objects": [],
        }

    boxes_arr = boxes.astype(np.float32)
    proto_arr = proto.astype(np.float32)

    if boxes_arr.ndim == 3 and boxes_arr.shape[0] == 1:
        boxes_arr = boxes_arr[0]
    if boxes_arr.ndim != 2:
        return {
            "status": "inference_error",
            "error": "invalid_boxes_shape",
            "object_count": 0,
            "objects": [],
        }

    if proto_arr.ndim == 4 and proto_arr.shape[0] == 1:
        proto_arr = proto_arr[0]
    if proto_arr.ndim != 3:
        return {
            "status": "inference_error",
            "error": "invalid_proto_shape",
            "object_count": 0,
            "objects": [],
        }

    proto_channels, proto_h, proto_w = proto_arr.shape
    expected_features = 6 + proto_channels
    if boxes_arr.shape[1] < expected_features:
        return {
            "status": "inference_error",
            "error": (
                f"unexpected_box_feature_count:{boxes_arr.shape[1]}"
                f"<required:{expected_features}"
            ),
            "object_count": 0,
            "objects": [],
        }

    max_objects = max(1, int(max_objects))
    conf = boxes_arr[:, 4]
    valid_idx = np.where(np.isfinite(conf) & (conf >= float(score_threshold)))[0]
    if valid_idx.size == 0:
        return {
            "status": "ok",
            "object_count": 0,
            "objects": [],
        }

    sorted_idx = valid_idx[np.argsort(conf[valid_idx])[::-1]]
    sorted_idx = sorted_idx[:max_objects]
    selected = boxes_arr[sorted_idx]

    coeffs = selected[:, 6 : 6 + proto_channels]
    proto_flat = proto_arr.reshape(proto_channels, -1)
    masks = _sigmoid(coeffs @ proto_flat).reshape(-1, proto_h, proto_w)

    sx = float(proto_w) / float(input_width)
    sy = float(proto_h) / float(input_height)

    objects: list[dict[str, Any]] = []
    for i in range(selected.shape[0]):
        box = selected[i]
        x1 = float(np.clip(box[0], 0, input_width - 1))
        y1 = float(np.clip(box[1], 0, input_height - 1))
        x2 = float(np.clip(box[2], 0, input_width))
        y2 = float(np.clip(box[3], 0, input_height))
        if x2 <= x1 or y2 <= y1:
            continue

        x1p = int(np.clip(np.floor(x1 * sx), 0, max(0, proto_w - 1)))
        y1p = int(np.clip(np.floor(y1 * sy), 0, max(0, proto_h - 1)))
        x2p = int(np.clip(np.ceil(x2 * sx), 1, proto_w))
        y2p = int(np.clip(np.ceil(y2 * sy), 1, proto_h))
        if x2p <= x1p or y2p <= y1p:
            continue

        mask_proto = np.zeros((proto_h, proto_w), dtype=np.float32)
        mask_proto[y1p:y2p, x1p:x2p] = masks[i, y1p:y2p, x1p:x2p]

        mask_input = _resize_nearest_2d(mask_proto, input_height, input_width) >= mask_threshold

        bx1 = int(np.clip(np.floor(x1), 0, max(0, input_width - 1)))
        by1 = int(np.clip(np.floor(y1), 0, max(0, input_height - 1)))
        bx2 = int(np.clip(np.ceil(x2), 1, input_width))
        by2 = int(np.clip(np.ceil(y2), 1, input_height))
        bbox_clip = np.zeros((input_height, input_width), dtype=bool)
        bbox_clip[by1:by2, bx1:bx2] = True
        mask_input &= bbox_clip

        pixel_count = int(mask_input.sum())
        if pixel_count == 0:
            continue

        objects.append(
            {
                "class_id": int(box[5]),
                "class_name": resolve_class_name(int(box[5]), class_names),
                "confidence": float(box[4]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "mask": mask_input,
                "mask_pixels_input": pixel_count,
            }
        )

    return {
        "status": "ok",
        "object_count": len(objects),
        "objects": objects,
    }


def decode_depth_output(result: dict, output_name: str | None = None) -> dict:
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
        "depth_map": depth_map,
        "depth_percentiles": {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "spread_p90_p10": p90 - p10,
        },
    }


def summarize_depth_output(result: dict, output_name: str | None = None) -> dict:
    decoded = decode_depth_output(result, output_name=output_name)
    decoded.pop("depth_map", None)
    return decoded


def compute_object_depth_overlap(
    *,
    yolo_decoded: dict,
    depth_decoded: dict,
    frame_height: int,
    frame_width: int,
    yolo_input_width: int,
    yolo_input_height: int,
    near_threshold: float = 0.35,
    far_threshold: float = 0.65,
) -> dict:
    if yolo_decoded.get("status") != "ok":
        return {
            "status": "inference_error",
            "error": "yolo_decode_failed",
            "object_count": 0,
            "objects": [],
        }

    if depth_decoded.get("status") != "ok":
        return {
            "status": "inference_error",
            "error": "depth_decode_failed",
            "object_count": 0,
            "objects": [],
        }

    depth_map = depth_decoded.get("depth_map")
    if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
        return {
            "status": "inference_error",
            "error": "invalid_depth_map",
            "object_count": 0,
            "objects": [],
        }

    depth_h, depth_w = depth_map.shape
    frame_percentiles = depth_decoded.get("depth_percentiles", {})
    frame_p10 = float(frame_percentiles.get("p10", np.percentile(depth_map, 10.0)))
    frame_p90 = float(frame_percentiles.get("p90", np.percentile(depth_map, 90.0)))

    objects_out: list[dict[str, Any]] = []
    for item in yolo_decoded.get("objects", []):
        mask_input = item.get("mask")
        if not isinstance(mask_input, np.ndarray):
            continue

        if mask_input.shape != (yolo_input_height, yolo_input_width):
            continue

        mask_frame = _resize_nearest_2d(mask_input, frame_height, frame_width).astype(bool)
        ys, xs = np.where(mask_frame)
        if ys.size == 0:
            continue

        dy = np.minimum((ys.astype(np.int64) * depth_h // frame_height), depth_h - 1)
        dx = np.minimum((xs.astype(np.int64) * depth_w // frame_width), depth_w - 1)

        values = depth_map[dy, dx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        d_obj = float(np.median(values))
        p10 = float(np.percentile(values, 10.0))
        p90 = float(np.percentile(values, 90.0))
        spread = p90 - p10

        denom = max(frame_p90 - frame_p10, 1e-6)
        depth_norm = (d_obj - frame_p10) / denom
        if depth_norm <= near_threshold:
            depth_band = "near"
        elif depth_norm >= far_threshold:
            depth_band = "far"
        else:
            depth_band = "mid"

        objects_out.append(
            {
                "class_id": int(item.get("class_id", -1)),
                "class_name": str(
                    item.get(
                        "class_name",
                        resolve_class_name(int(item.get("class_id", -1))),
                    )
                ),
                "confidence": float(item.get("confidence", 0.0)),
                "bbox_xyxy": [float(v) for v in item.get("bbox_xyxy", [0, 0, 0, 0])],
                "depth_median": d_obj,
                "depth_p10": p10,
                "depth_p90": p90,
                "depth_spread_p90_p10": spread,
                "depth_norm": float(depth_norm),
                "depth_band": depth_band,
                "pixel_count": int(values.size),
            }
        )

    return {
        "status": "ok",
        "object_count": len(objects_out),
        "frame_depth_percentiles": {
            "p10": frame_p10,
            "p90": frame_p90,
            "spread_p90_p10": frame_p90 - frame_p10,
        },
        "objects": objects_out,
    }
