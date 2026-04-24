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
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.03-py3}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
ENABLE_YOLO_CALIBRATION="${ENABLE_YOLO_CALIBRATION:-0}"
YOLO_CALIB_DATA="${YOLO_CALIB_DATA:-coco8.yaml}"
YOLO_CALIB_SAMPLES="${YOLO_CALIB_SAMPLES:-8}"
YOLO_CALIB_IMAGE_SIZE="${YOLO_CALIB_IMAGE_SIZE:-512}"

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

if ! "${PYTHON_BIN}" --version >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}"
  echo "Set PYTHON_BIN to a valid Python 3 executable, e.g.:"
  echo "  PYTHON_BIN=python3 bash scripts/build_triton_engine.sh"
  exit 1
fi

TRT_FLAGS=""
if [[ "${TRT_PRECISION}" == "fp16" ]]; then
  TRT_FLAGS="--fp16"
fi

DOCKER_GPU_ARGS=(--gpus all)
DOCKER_GPU_ENV=()

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable from this environment."
  echo "Start Docker on the Vast.ai host or run this on a machine with Docker access."
  exit 1
fi

if ! docker info 2>/dev/null | grep -Eq 'Runtimes:.*nvidia|nvidia'; then
  echo "Docker GPU support is not enabled on this host."
  echo "This build step needs NVIDIA container runtime support so trtexec can access the GPU."
  echo ""
  echo "Fix options:"
  echo "  1. Use a Vast.ai instance/image with NVIDIA Container Toolkit already installed."
  echo "  2. Install and configure the toolkit on the host, then restart Docker."
  echo "  3. If you already have a GPU-ready container runtime, verify that 'docker run --gpus all ...' works before rerunning this script."
  echo ""
  echo "Common verification commands:"
  echo "  docker info | grep -i runtimes"
  echo "  nvidia-smi"
  exit 1
fi

if ! docker run --rm --gpus all --pull=never nvidia/cuda:12.4.1-base-ubuntu22.04 true >/dev/null 2>&1; then
  echo "Docker's --gpus path is not working on this host; falling back to --runtime=nvidia."
  DOCKER_GPU_ARGS=(--runtime=nvidia)
  DOCKER_GPU_ENV=(-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi

mkdir -p "${YOLO_TARGET_ENGINE_DIR}" "${DEPTH_TARGET_ENGINE_DIR}"

# --- YOLO TensorRT engine ---
YOLO_ENGINE_INPUT="${YOLO_SOURCE_MODEL}"

if [[ "${ENABLE_YOLO_CALIBRATION}" == "1" ]]; then
  echo "Running YOLO INT8 calibration (dataset=${YOLO_CALIB_DATA}, samples=${YOLO_CALIB_SAMPLES})..."
  if ! "${PYTHON_BIN}" "${ROOT_DIR}/scripts/yolo_calibrate.py" \
      --model-input "${YOLO_SOURCE_MODEL}" \
      --model-output "${YOLO_QUANT_MODEL}" \
      --dataset "${YOLO_CALIB_DATA}" \
      --samples "${YOLO_CALIB_SAMPLES}" \
      --image-size "${YOLO_CALIB_IMAGE_SIZE}"; then
    echo ""
    echo "ERROR: YOLO INT8 calibration failed."
    echo "TensorRT 10.3 may not support this QDQ graph. Falling back to fp16 is recommended:"
    echo "  bash scripts/build_triton_engine.sh  (without ENABLE_YOLO_CALIBRATION=1)"
    exit 1
  fi
  YOLO_ENGINE_INPUT="${YOLO_QUANT_MODEL}"
  TRT_FLAGS="${TRT_FLAGS} --int8"
fi

echo "Building YOLO TensorRT engine (${TRT_PRECISION})..."
docker run --rm "${DOCKER_GPU_ARGS[@]}" --network host "${DOCKER_GPU_ENV[@]}" \
  -v "${YOLO_ENGINE_INPUT}:/workspace/yolo.onnx:ro" \
  -v "${TRITON_REPO}:/models" \
  --shm-size=1g \
  "${TRITON_IMAGE}" \
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/yolo.onnx --saveEngine=/models/yolo26n_seg/1/model.plan ${TRT_FLAGS}

# --- Depth dynamic quantization (unchanged) ---
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

echo "Copying quantized depth model to Triton repo..."
cp "${DEPTH_QUANT_MODEL}" "${DEPTH_TARGET_ENGINE_DIR}/model.onnx"

echo "Engine written to ${YOLO_TARGET_ENGINE_DIR}/model.plan"
echo "Quantized depth model written to ${DEPTH_TARGET_ENGINE_DIR}/model.onnx"
