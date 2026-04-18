#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_SOURCE_MODEL="${ROOT_DIR}/models/yolo26n-seg.onnx"
DEPTH_SOURCE_MODEL="${ROOT_DIR}/models/depth_anything_v2_vits.onnx"
TRITON_REPO="${ROOT_DIR}/triton/model_repository"
YOLO_TARGET_ENGINE_DIR="${TRITON_REPO}/yolo26n_seg/1"
DEPTH_TARGET_ENGINE_DIR="${TRITON_REPO}/depth_anything_v2_small/1"
TRT_PRECISION="${TRT_PRECISION:-fp16}"

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

TRT_FLAGS=""
if [[ "${TRT_PRECISION}" == "fp16" ]]; then
  TRT_FLAGS="--fp16"
fi

mkdir -p "${YOLO_TARGET_ENGINE_DIR}" "${DEPTH_TARGET_ENGINE_DIR}"

echo "Building YOLO TensorRT engine (${TRT_PRECISION})..."
docker run --rm --gpus all --network host \
  -v "${YOLO_SOURCE_MODEL}:/workspace/yolo.onnx:ro" \
  -v "${TRITON_REPO}:/models" \
  --shm-size=1g \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/yolo.onnx --saveEngine=/models/yolo26n_seg/1/model.plan ${TRT_FLAGS}

echo "Depth Anything V2 Small is served via Triton ONNXRuntime for now; copying model artifact..."
cp "${DEPTH_SOURCE_MODEL}" "${DEPTH_TARGET_ENGINE_DIR}/model.onnx"

echo "Engine written to ${YOLO_TARGET_ENGINE_DIR}/model.plan"
echo "Depth model written to ${DEPTH_TARGET_ENGINE_DIR}/model.onnx"
