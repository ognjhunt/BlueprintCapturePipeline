#!/usr/bin/env bash
# =============================================================================
# Install the supported single-VM site-world pipeline runtime.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WITH_SAM3=false
WITH_DA3=false
WITH_LOCAL_QWEN=false
WITH_DEEPPRIVACY2=false
SKIP_PREWARM=false

HF_CACHE_DIR="${HF_HOME:-/opt/hf}"
SAM3_DIR="${SAM3_DIR:-/opt/sam3}"
SAM3_WEIGHTS_PATH="${SAM3_WEIGHTS_PATH:-/opt/sam3_weights/sam3.pt}"
DA3_DIR="${DA3_DIR:-/opt/da3}"
DA3_WEIGHTS_DIR="${DA3_MODEL_PATH:-/opt/da3/weights/metric_large}"
DA3_MODEL_ID="${DA3_MODEL_ID:-depth-anything/DA3Metric-Large}"
DA3_MODEL_NAME="${DA3_MODEL_NAME:-da3metric-large}"
QWEN_EDIT_DIR="${QWEN_IMAGE_EDIT_MODEL_PATH:-/opt/qwen-image-edit}"
DEEPPRIVACY2_DIR="${DEEPPRIVACY2_DIR:-/opt/deepprivacy2}"
DEEPPRIVACY2_MODEL_PATH="${DEEPPRIVACY2_MODEL_PATH:-/opt/deepprivacy2/weights}"
NVIDIA_PYPI_INDEX="${NVIDIA_PYPI_INDEX:-https://pypi.nvidia.com}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
YOLO_WORLD_MODEL="${YOLO_WORLD_MODEL:-yolov8s-worldv2.pt}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

log() {
  echo "[install-ml-stack] $*"
}

die() {
  echo "[install-ml-stack] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  install_ml_stack.sh [options]

Default install:
  - CUDA PyTorch
  - BlueprintCapturePipeline package + runtime deps
  - ffmpeg and base system tooling
  - YOLO-World cache prewarm

Optional installs:
  --with-sam3         Install optional SAM3 runtime from source
  --with-da3          Install optional Depth Anything 3 runtime + weights
  --with-local-qwen   Install optional local Qwen image-edit weights
  --with-deepprivacy2 Install optional DeepPrivacy2 runtime from source
  --skip-prewarm      Skip model cache prewarm checks
  -h, --help          Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with-sam3)
      WITH_SAM3=true
      shift
      ;;
    --with-da3)
      WITH_DA3=true
      shift
      ;;
    --with-local-qwen)
      WITH_LOCAL_QWEN=true
      shift
      ;;
    --with-deepprivacy2)
      WITH_DEEPPRIVACY2=true
      shift
      ;;
    --skip-prewarm)
      SKIP_PREWARM=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

clone_or_update_repo() {
  local repo_url="$1"
  local dst="$2"
  local ref="$3"

  mkdir -p "$(dirname "$dst")"
  if [ ! -d "$dst/.git" ]; then
    git clone "$repo_url" "$dst"
  fi
  git -C "$dst" fetch --tags origin
  git -C "$dst" checkout "$ref"
  if [ "$ref" = "main" ] || [ "$ref" = "master" ]; then
    git -C "$dst" pull --ff-only origin "$ref"
  fi
}

ensure_repo_root() {
  [ -f "${REPO_ROOT}/pyproject.toml" ] || die "Missing pyproject.toml at ${REPO_ROOT}"
  [ -d "${REPO_ROOT}/src/blueprint_pipeline" ] || die "Missing src/blueprint_pipeline at ${REPO_ROOT}"
}

install_system_dependencies() {
  log "Installing base system dependencies..."
  $SUDO apt-get update -o Acquire::Retries=5 --fix-missing
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-pip \
    python3-venv \
    rsync
}

install_python_runtime() {
  log "Installing Python runtime dependencies from repo checkout..."
  python3 -m pip install --upgrade pip setuptools wheel
  python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"
  python3 -m pip install --no-cache-dir "huggingface_hub[cli]"
  python3 -m pip install --no-cache-dir -e "${REPO_ROOT}[runtime]"
}

prewarm_yolo_world() {
  log "Prewarming YOLO-World model cache..."
  HF_HOME="${HF_CACHE_DIR}" python3 - <<PY
from ultralytics import YOLOWorld

model = YOLOWorld("${YOLO_WORLD_MODEL}")
print(f"YOLO_WORLD_READY {model.model_name}")
PY
}

install_sam3() {
  log "Installing optional SAM3 runtime..."
  clone_or_update_repo "https://github.com/facebookresearch/sam3.git" "${SAM3_DIR}" "main"
  python3 -m pip install --no-cache-dir -e "${SAM3_DIR}"

  if [ -f "${SAM3_WEIGHTS_PATH}" ]; then
    log "Found SAM3 weights at ${SAM3_WEIGHTS_PATH}"
  else
    log "WARNING: SAM3 installed but weights are missing at ${SAM3_WEIGHTS_PATH}"
    log "SAM3 backend will skip until SAM3_WEIGHTS_PATH points to a valid checkpoint."
  fi
}

install_da3() {
  log "Installing optional Depth Anything 3 runtime..."
  clone_or_update_repo "https://github.com/ByteDance-Seed/Depth-Anything-3.git" "${DA3_DIR}" "main"
  python3 -m pip install --no-cache-dir -e "${DA3_DIR}"
  mkdir -p "${DA3_WEIGHTS_DIR}"
  if [ ! -f "${DA3_WEIGHTS_DIR}/config.json" ]; then
    HF_HOME="${HF_CACHE_DIR}" huggingface-cli download "${DA3_MODEL_ID}" --local-dir "${DA3_WEIGHTS_DIR}"
  fi
}

install_local_qwen() {
  log "Installing optional Qwen image-edit runtime..."
  python3 -m pip install --no-cache-dir diffusers accelerate sentencepiece protobuf transformers
  mkdir -p "${QWEN_EDIT_DIR}"
  if [ ! -f "${QWEN_EDIT_DIR}/model_index.json" ]; then
    HF_HOME="${HF_CACHE_DIR}" huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir "${QWEN_EDIT_DIR}"
  fi
}

install_deepprivacy2() {
  log "Installing optional DeepPrivacy2 runtime..."
  clone_or_update_repo "https://github.com/hukkelas/deep_privacy2.git" "${DEEPPRIVACY2_DIR}" "main"
  python3 -m pip install --no-cache-dir -e "${DEEPPRIVACY2_DIR}"
  mkdir -p "${DEEPPRIVACY2_MODEL_PATH}"
  if [ -z "${DEEPPRIVACY2_COMMAND:-}" ]; then
    log "WARNING: DEEPPRIVACY2_COMMAND is not configured. Privacy fallback will fail closed until it is set."
  fi
}

prewarm_optional_models() {
  if [ "${WITH_SAM3}" = true ] && [ -f "${SAM3_WEIGHTS_PATH}" ]; then
    log "Validating SAM3 optional runtime..."
    HF_HOME="${HF_CACHE_DIR}" SAM3_WEIGHTS_PATH="${SAM3_WEIGHTS_PATH}" python3 - <<'PY'
import importlib.util
from pathlib import Path

weights = Path(Path(__import__("os").environ["SAM3_WEIGHTS_PATH"]))
if importlib.util.find_spec("sam3") is None:
    raise SystemExit("SAM3 package was requested but is not importable")
print(f"SAM3_READY {weights}")
PY
  fi

  if [ "${WITH_DA3}" = true ]; then
    log "Validating DA3 optional runtime..."
    HF_HOME="${HF_CACHE_DIR}" python3 - <<PY
from pathlib import Path
from depth_anything_3.api import DepthAnything3

DepthAnything3.from_pretrained(str(Path("${DA3_WEIGHTS_DIR}")), model_name="${DA3_MODEL_NAME}")
print("DA3_READY")
PY
  fi

  if [ "${WITH_LOCAL_QWEN}" = true ]; then
    log "Validating Qwen optional runtime..."
    HF_HOME="${HF_CACHE_DIR}" python3 - <<PY
import torch
from diffusers import QwenImageEditPlusPipeline

pipe = QwenImageEditPlusPipeline.from_pretrained("${QWEN_EDIT_DIR}", torch_dtype=torch.bfloat16)
print(f"QWEN_READY {type(pipe).__name__}")
PY
  fi
}

write_environment_profile() {
  log "Writing environment profile..."
  cat <<EOF | $SUDO tee /etc/profile.d/blueprint_capture_ml.sh >/dev/null
export HF_HOME=${HF_CACHE_DIR}
export YOLO_WORLD_MODEL=${YOLO_WORLD_MODEL}
export SAM3_DIR=${SAM3_DIR}
export SAM3_WEIGHTS_PATH=${SAM3_WEIGHTS_PATH}
export DA3_MODEL_PATH=${DA3_WEIGHTS_DIR}
export DA3_MODEL_NAME=${DA3_MODEL_NAME}
export QWEN_IMAGE_EDIT_MODEL_PATH=${QWEN_EDIT_DIR}
export DEEPPRIVACY2_DIR=${DEEPPRIVACY2_DIR}
export DEEPPRIVACY2_MODEL_PATH=${DEEPPRIVACY2_MODEL_PATH}
export CROP_CLEANUP_PROVIDER=skip
EOF
}

verify_runtime() {
  log "Verifying installed runtime..."
  python3 - <<'PY'
import importlib.util
import os
import shutil

required_modules = {
    "torch": importlib.util.find_spec("torch") is not None,
    "trimesh": importlib.util.find_spec("trimesh") is not None,
    "ultralytics": importlib.util.find_spec("ultralytics") is not None,
    "blueprint_pipeline": importlib.util.find_spec("blueprint_pipeline") is not None,
}
missing = [name for name, present in required_modules.items() if not present]
if missing:
    raise SystemExit(f"Missing required Python modules: {', '.join(missing)}")
if shutil.which("ffmpeg") is None:
    raise SystemExit("ffmpeg is not installed")
print("RUNTIME_OK")
PY
}

ensure_repo_root
mkdir -p "${HF_CACHE_DIR}"

install_system_dependencies
install_python_runtime

if [ "${WITH_SAM3}" = true ]; then
  install_sam3
fi
if [ "${WITH_DA3}" = true ]; then
  install_da3
fi
if [ "${WITH_LOCAL_QWEN}" = true ]; then
  install_local_qwen
fi
if [ "${WITH_DEEPPRIVACY2}" = true ]; then
  install_deepprivacy2
fi

if [ "${SKIP_PREWARM}" = false ]; then
  prewarm_yolo_world
  prewarm_optional_models
else
  log "Skipping model prewarm."
fi

write_environment_profile
verify_runtime

log "ML stack install complete."
