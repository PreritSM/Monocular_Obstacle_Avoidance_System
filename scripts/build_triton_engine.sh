#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_SOURCE_MODEL="${ROOT_DIR}/models/yolo26n-seg.onnx"
YOLO_QUANT_MODEL="${ROOT_DIR}/models/yolo26n-seg.int8.onnx"
DEPTH_SOURCE_MODEL="${ROOT_DIR}/models/depth_anything_v2_vits.onnx"
DEPTH_QUANT_MODEL="${ROOT_DIR}/models/depth_anything_v2_vits.int8.onnx"
TRITON_REPO="${ROOT_DIR}/triton/model_repository"
YOLO_TARGET_ENGINE_DIR="${TRITON_REPO}/yolo26n_seg/1"
DEPTH_TARGET_ENGINE_DIR="${TRITON_REPO}/depth_anything_v2_small/1"
TRT_PRECISION="${TRT_PRECISION:-fp16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
YOLO_CALIB_DATA="${YOLO_CALIB_DATA:-coco8.yaml}"
YOLO_CALIB_SAMPLES="${YOLO_CALIB_SAMPLES:-8}"
YOLO_CALIB_IMAGE_SIZE="${YOLO_CALIB_IMAGE_SIZE:-640}"

if [[ "${TRT_PRECISION}" != "fp16" && "${TRT_PRECISION}" != "fp32" ]]; then
  echo "Unsupported TRT_PRECISION=${TRT_PRECISION}. Use fp16 or fp32."
  exit 1
fi

if [[ ! -f "${YOLO_SOURCE_MODEL}" ]]; then
  echo "YOLO source ONNX not found: ${YOLO_SOURCE_MODEL}"
  exit 1
fi

if [[ ! -f "${DEPTH_SOURCE_MODEL}" ]]; then
  echo "Depth source ONNX not found: ${DEPTH_SOURCE_MODEL}"
  echo "Run: bash scripts/download_model.sh"
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}"
  echo "Set PYTHON_BIN to a valid Python 3 executable, e.g.:"
  echo "  PYTHON_BIN=python3 bash scripts/build_triton_engine.sh"
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
from ultralytics.data.utils import check_det_dataset
import cv2
import numpy as np
PY
then
  echo "Missing dependency: onnxruntime quantization tools, Ultralytics dataset helpers, and OpenCV are required."
  echo "Install them with:"
  echo "  python3 -m pip install onnxruntime ultralytics opencv-python"
  echo "Then re-run: bash scripts/build_triton_engine.sh"
  exit 1
fi

TRT_FLAGS=""
if [[ "${TRT_PRECISION}" == "fp16" ]]; then
  TRT_FLAGS="--fp16"
fi

mkdir -p "${YOLO_TARGET_ENGINE_DIR}" "${DEPTH_TARGET_ENGINE_DIR}"

echo "Quantizing YOLO ONNX with COCO calibration (${YOLO_CALIB_DATA})..."
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import os

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from ultralytics.data.utils import check_det_dataset


class YoloCalibrationDataReader(CalibrationDataReader):
  def __init__(self, image_paths: list[Path], input_name: str, input_width: int, input_height: int) -> None:
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

  unique_paths = []
  seen = set()
  for path in image_paths:
    resolved = path.resolve()
    if resolved in seen:
      continue
    seen.add(resolved)
    unique_paths.append(resolved)

  if not unique_paths:
    raise RuntimeError(f"No calibration images found for dataset: {dataset_name}")
  return unique_paths[:sample_limit]


calibration_dataset = os.environ.get("YOLO_CALIB_DATA", "${YOLO_CALIB_DATA}")
calibration_samples = int(os.environ.get("YOLO_CALIB_SAMPLES", "${YOLO_CALIB_SAMPLES}"))
calibration_image_size = int(os.environ.get("YOLO_CALIB_IMAGE_SIZE", "${YOLO_CALIB_IMAGE_SIZE}"))

image_paths = _collect_coco_images(calibration_dataset, calibration_samples)
reader = YoloCalibrationDataReader(
  image_paths=image_paths,
  input_name="images",
  input_width=calibration_image_size,
  input_height=calibration_image_size,
)

quantize_static(
  model_input="${YOLO_SOURCE_MODEL}",
  model_output="${YOLO_QUANT_MODEL}",
  calibration_data_reader=reader,
  quant_format=QuantFormat.QDQ,
  calibrate_method=CalibrationMethod.MinMax,
  activation_type=QuantType.QUInt8,
  weight_type=QuantType.QInt8,
)
print(f"Wrote calibrated YOLO model: ${YOLO_QUANT_MODEL}")
PY

echo "Building YOLO TensorRT engine (${TRT_PRECISION})..."
docker run --rm --gpus all --network host \
  -v "${YOLO_QUANT_MODEL}:/workspace/yolo.onnx:ro" \
  -v "${TRITON_REPO}:/models" \
  --shm-size=1g \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/yolo.onnx --saveEngine=/models/yolo26n_seg/1/model.plan ${TRT_FLAGS}

echo "Quantizing Depth Anything V2 Small ONNX (dynamic QUInt8)..."
"${PYTHON_BIN}" - <<PY
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="${DEPTH_SOURCE_MODEL}",
    model_output="${DEPTH_QUANT_MODEL}",
    weight_type=QuantType.QUInt8,
)
print("Wrote quantized model: ${DEPTH_QUANT_MODEL}")
PY

echo "Depth Anything V2 Small is served via Triton ONNXRuntime; copying quantized model artifact..."
cp "${DEPTH_QUANT_MODEL}" "${DEPTH_TARGET_ENGINE_DIR}/model.onnx"

echo "Engine written to ${YOLO_TARGET_ENGINE_DIR}/model.plan"
echo "Quantized depth model written to ${DEPTH_TARGET_ENGINE_DIR}/model.onnx"
