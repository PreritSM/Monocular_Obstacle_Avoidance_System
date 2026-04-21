#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/PreritSM/Monocular_Obstacle_Avoidance_System.git}"
REPO_DIR="${REPO_DIR:-Monocular_Obstacle_Avoidance_System}"
BRANCH="${BRANCH:-vast-ai-multithread}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
FOLLOW_LOGS="${FOLLOW_LOGS:-0}"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.03-py3}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  python3 -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  log "ERROR: uv is not available in PATH after installation attempt."
  exit 1
fi

log "Ensuring Python ${PYTHON_VERSION} is available via uv"
uv python install "${PYTHON_VERSION}"

log "Using Triton image: ${TRITON_IMAGE}"

if [[ -d "${VENV_DIR}" ]]; then
  log "Removing existing ${VENV_DIR} to recreate with Python ${PYTHON_VERSION}"
  rm -rf "${VENV_DIR}"
fi

log "Creating/updating repo-local venv at ${VENV_DIR} using uv + Python ${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

VENV_ROOT="${PWD}/${VENV_DIR}"
VENV_PY="${VENV_ROOT}/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  log "ERROR: virtual environment python not found in ${VENV_DIR}/bin"
  exit 1
fi

ACTIVE_PY="$("${VENV_PY}" -c 'import sys; print(sys.executable)')"
ACTIVE_PREFIX="$("${VENV_PY}" -c 'import sys; print(sys.prefix)')"
ACTIVE_VER="$("${VENV_PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${ACTIVE_PREFIX}" != "${VENV_ROOT}" ]]; then
  log "ERROR: Python is not using repo venv. Expected prefix ${VENV_ROOT}, got ${ACTIVE_PREFIX}"
  exit 1
fi
if [[ "${ACTIVE_VER}" != "${PYTHON_VERSION}" ]]; then
  log "ERROR: Venv Python version mismatch. Expected ${PYTHON_VERSION}, got ${ACTIVE_VER}"
  exit 1
fi

log "Ensuring pip is present in the venv"
"${VENV_PY}" -m ensurepip --upgrade

if ! "${VENV_PY}" -m pip --version >/dev/null 2>&1; then
  log "ERROR: pip module is not available in venv after ensurepip"
  exit 1
fi

ACTIVE_PIP="$("${VENV_PY}" -m pip --version | awk '{print $4}')"
if [[ "${ACTIVE_PIP}" != "${VENV_ROOT}"/* ]]; then
  log "ERROR: Pip is not using repo venv. Expected path under ${VENV_ROOT}, got ${ACTIVE_PIP}"
  exit 1
fi

log "Using venv python: ${ACTIVE_PY}"
log "Using venv pip: ${ACTIVE_PIP}"
log "Using venv python version: ${ACTIVE_VER}"

log "Upgrading pip in venv"
"${VENV_PY}" -m pip install --upgrade pip

log "Installing dev requirements"
"${VENV_PY}" -m pip install -r requirements-dev.txt

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  log "Installing NVIDIA container toolkit"
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

log "Configuring Docker runtime for NVIDIA"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

log "Quick GPU runtime checks"
nvidia-smi
docker info | grep -i runtimes || true

log "Downloading model artifacts"
bash scripts/download_model.sh

log "Exporting YOLO ONNX at 512x512"
"${VENV_PY}" -c "from ultralytics import YOLO; model = YOLO('models/yolo26n-seg.pt'); model.export(format='onnx', imgsz=512)"

log "Building Triton engine and quantized depth model"
TRITON_IMAGE="${TRITON_IMAGE}" PYTHON_BIN="${VENV_PY}" bash scripts/build_triton_engine.sh

log "Verifying Triton model repository"
bash scripts/prepare_triton_repo.sh

log "Building and starting Docker services"
pushd deploy/docker >/dev/null
TRITON_IMAGE="${TRITON_IMAGE}" docker compose up -d --build

log "Compose status"
docker compose ps

log "Recent Triton logs"
docker compose logs --tail=120 triton

log "Recent edge gateway logs"
docker compose logs --tail=120 edge_gateway

if [[ "${FOLLOW_LOGS}" == "1" ]]; then
  log "Following logs (Ctrl+C to stop)"
  docker compose logs -f triton edge_gateway signaling
fi

popd >/dev/null
log "Done"
