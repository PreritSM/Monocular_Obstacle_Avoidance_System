#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_MODEL="${ROOT_DIR}/models/yolo26n-seg.onnx"
TRITON_REPO="${ROOT_DIR}/triton/model_repository"
TARGET_ENGINE_DIR="${TRITON_REPO}/yolo26n_seg/1"

if [[ ! -f "${SOURCE_MODEL}" ]]; then
  echo "Source ONNX not found: ${SOURCE_MODEL}"
  exit 1
fi

mkdir -p "${TARGET_ENGINE_DIR}"

docker run --rm --gpus all --network host \
  -v "${SOURCE_MODEL}:/workspace/model.onnx:ro" \
  -v "${TRITON_REPO}:/models" \
  --shm-size=1g \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/model.onnx --saveEngine=/models/yolo26n_seg/1/model.plan --fp16

echo "Engine written to ${TARGET_ENGINE_DIR}/model.plan"
