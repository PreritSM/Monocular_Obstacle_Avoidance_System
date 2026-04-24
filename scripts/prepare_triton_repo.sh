#!/bin/bash
set -euo pipefail

YOLO_TARGET_DIR="triton/model_repository/yolo26n_seg/1"
DEPTH_TARGET_DIR="triton/model_repository/depth_anything_v2_small/1"

mkdir -p "$YOLO_TARGET_DIR" "$DEPTH_TARGET_DIR"

if [[ ! -f "$YOLO_TARGET_DIR/model.plan" ]]; then
  echo "YOLO model.plan not found in $YOLO_TARGET_DIR"
  echo "Expected file: $YOLO_TARGET_DIR/model.plan"
  exit 1
fi

if [[ ! -f "$DEPTH_TARGET_DIR/model.onnx" ]]; then
  echo "Depth model.onnx not found in $DEPTH_TARGET_DIR"
  echo "Expected file: $DEPTH_TARGET_DIR/model.onnx"
  exit 1
fi

echo "Triton model repository looks ready for YOLO TensorRT + Depth ONNXRuntime."
