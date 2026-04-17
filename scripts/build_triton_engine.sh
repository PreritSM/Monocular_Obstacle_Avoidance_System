#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_SOURCE_MODEL="${ROOT_DIR}/models/yolo26n-seg.onnx"
DEPTH_SOURCE_MODEL="${ROOT_DIR}/models/depth_anything_v2_vits.onnx"
DEPTH_QUANT_MODEL="${ROOT_DIR}/models/depth_anything_v2_vits.int8.onnx"
TRITON_REPO="${ROOT_DIR}/triton/model_repository"
YOLO_TARGET_ENGINE_DIR="${TRITON_REPO}/yolo26n_seg/1"
DEPTH_TARGET_ENGINE_DIR="${TRITON_REPO}/depth_anything_v2_small/1"
TRT_PRECISION="${TRT_PRECISION:-fp16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
from onnxruntime.quantization import quantize_dynamic, QuantType
PY
then
  echo "Missing dependency: onnxruntime quantization tools are required."
  echo "Install them with:"
  echo "  python3 -m pip install onnxruntime"
  echo "Then re-run: bash scripts/build_triton_engine.sh"
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
