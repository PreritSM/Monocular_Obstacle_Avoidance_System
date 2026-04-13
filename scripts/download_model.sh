#!/bin/bash
set -euo pipefail

# 1. Setup paths (Matching your workspace)
MODELS_DIR="/workspace"
mkdir -p "${MODELS_DIR}"

YOLO_PT_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt"
DEPTH_ONNX_URL="https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"

# 2. Install required tools
echo "Checking dependencies..."
pip install -U onnx onnx-simplifier

# 3. Download Models
echo "Downloading YOLO..."
curl -L "${YOLO_PT_URL}" -o "${MODELS_DIR}/yolo26n-seg.pt"

echo "Downloading Depth Anything V2..."
curl -L "${DEPTH_ONNX_URL}" -o "${MODELS_DIR}/depth_raw.onnx"

# 4. FORCE INLINE LOCAL FUNCTIONS
# This step is CRITICAL. It converts the "pkg.depth_anything_v2" 
# functions into a flat graph that TensorRT can actually read.
echo "Inlining ONNX functions..."
python3 -c "
import onnx
try:
    from onnx import inliner
    model = onnx.load('${MODELS_DIR}/depth_raw.onnx')
    inlined = inliner.inline(model)
    onnx.save(inlined, '${MODELS_DIR}/depth_inlined.onnx')
    print('Successfully inlined local functions.')
except Exception as e:
    print(f'Inlining failed: {e}')
"

# 5. SIMPLIFY THE INLINED MODEL
echo "Running onnx-simplifier..."
python3 -m onnxsim "${MODELS_DIR}/depth_inlined.onnx" "${MODELS_DIR}/depth.onnx"

# Clean up temporary files
rm "${MODELS_DIR}/depth_raw.onnx" "${MODELS_DIR}/depth_inlined.onnx"

echo "------------------------------------------------"
echo "Done! The file is ready: ${MODELS_DIR}/depth.onnx"
echo "Run your trtexec command now."