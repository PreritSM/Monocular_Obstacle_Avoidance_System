#!/usr/bin/env python3
"""YOLO INT8 static quantization with COCO calibration data.

NOTE: TensorRT 10.3 may reject the QDQ graph produced by this script.
Use only via ENABLE_YOLO_CALIBRATION=1 in build_triton_engine.sh.

Usage:
    python scripts/yolo_calibrate.py \
        --model-input  models/yolo26n-seg.onnx \
        --model-output models/yolo26n-seg.int8.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics.data.utils import check_det_dataset


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        image_paths: list[Path],
        input_name: str,
        input_width: int,
        input_height: int,
    ) -> None:
        self._image_paths = image_paths
        self._input_name = input_name
        self._input_width = input_width
        self._input_height = input_height
        self._index = 0

    def get_next(self):
        while self._index < len(self._image_paths):
            image_path = self._image_paths[self._index]
            self._index += 1
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            if image_bgr.shape[0] != self._input_height or image_bgr.shape[1] != self._input_width:
                image_bgr = cv2.resize(
                    image_bgr,
                    (self._input_width, self._input_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))[None, ...]
            return {self._input_name: image}
        return None


def _collect_coco_images(dataset_name: str, sample_limit: int) -> list[Path]:
    dataset = check_det_dataset(dataset_name, autodownload=True)
    image_paths: list[Path] = []
    for split_name in ("train", "val"):
        split_value = dataset.get(split_name)
        if not split_value:
            continue
        split_paths = split_value if isinstance(split_value, list) else [split_value]
        for split_path in split_paths:
            candidate = Path(split_path)
            if candidate.is_dir():
                for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    image_paths.extend(sorted(candidate.glob(suffix)))
            elif candidate.is_file():
                image_paths.append(candidate)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in image_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    if not unique_paths:
        raise RuntimeError(f"No calibration images found for dataset: {dataset_name}")
    return unique_paths[:sample_limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO INT8 static quantization with COCO calibration")
    parser.add_argument("--model-input", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--dataset", default="coco8.yaml")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--input-name", default="images")
    args = parser.parse_args()

    image_paths = _collect_coco_images(args.dataset, args.samples)
    reader = YoloCalibrationDataReader(
        image_paths=image_paths,
        input_name=args.input_name,
        input_width=args.image_size,
        input_height=args.image_size,
    )

    quantize_static(
        model_input=str(args.model_input),
        model_output=str(args.model_output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False},
    )
    print(f"Wrote calibrated YOLO model: {args.model_output}")


if __name__ == "__main__":
    main()
