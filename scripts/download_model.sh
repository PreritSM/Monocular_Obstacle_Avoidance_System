#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"

YOLO_PT_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt"
DEPTH_ONNX_URL="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"

mkdir -p "${MODELS_DIR}"

echo "Downloading YOLO checkpoint..."
curl -L "${YOLO_PT_URL}" -o "${MODELS_DIR}/yolo26n-seg.pt"

echo "Downloading Depth Anything V2 Small ONNX..."
curl -L "${DEPTH_ONNX_URL}" -o "${MODELS_DIR}/depth_anything_v2_vits.onnx"

echo "Downloaded: ${MODELS_DIR}/yolo26n-seg.pt"
echo "Downloaded: ${MODELS_DIR}/depth_anything_v2_vits.onnx"
