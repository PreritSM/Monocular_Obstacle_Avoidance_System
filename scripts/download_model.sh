#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"
DATASETS_DIR="${ROOT_DIR}/datasets"
COCO8_DIR="${DATASETS_DIR}/coco8"

YOLO_PT_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt"
DEPTH_ONNX_URL="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
COCO8_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"

mkdir -p "${MODELS_DIR}"
mkdir -p "${DATASETS_DIR}"

echo "Downloading YOLO checkpoint..."
curl -L "${YOLO_PT_URL}" -o "${MODELS_DIR}/yolo26n-seg.pt"

echo "Downloading Depth Anything V2 Small ONNX..."
curl -L "${DEPTH_ONNX_URL}" -o "${MODELS_DIR}/depth_anything_v2_vits.onnx"

if [[ -d "${COCO8_DIR}/images/train" && -d "${COCO8_DIR}/images/val" ]]; then
	echo "COCO8 calibration dataset already present at ${COCO8_DIR}; skipping download."
else
	echo "Downloading COCO8 calibration dataset..."
	tmp_zip="$(mktemp "${DATASETS_DIR}/coco8.XXXXXX.zip")"
	curl -L "${COCO8_URL}" -o "${tmp_zip}"
	if command -v unzip >/dev/null 2>&1; then
		unzip -q -o "${tmp_zip}" -d "${DATASETS_DIR}"
	else
		python3 - <<PY
from pathlib import Path
import zipfile

zip_path = Path("${tmp_zip}")
target_dir = Path("${DATASETS_DIR}")
with zipfile.ZipFile(zip_path) as zf:
		zf.extractall(target_dir)
PY
	fi
	rm -f "${tmp_zip}"
fi

echo "Downloaded: ${MODELS_DIR}/yolo26n-seg.pt"
echo "Downloaded: ${MODELS_DIR}/depth_anything_v2_vits.onnx"
echo "Calibration dataset ready: ${COCO8_DIR}"
