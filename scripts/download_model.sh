#!/bin/bash
set -euo pipefail

mkdir -p models
curl -L "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt" -o models/yolo26n-seg.pt

echo "Downloaded models/yolo26n-seg.pt"
