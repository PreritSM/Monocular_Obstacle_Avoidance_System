#!/bin/bash
set -euo pipefail

TARGET_DIR="triton/model_repository/yolo26n_seg/1"
mkdir -p "$TARGET_DIR"

if [[ ! -f "$TARGET_DIR/model.plan" ]]; then
  echo "model.plan not found in $TARGET_DIR"
  echo "Place your TensorRT engine at: $TARGET_DIR/model.plan"
  exit 1
fi

echo "Triton model repository looks ready."
