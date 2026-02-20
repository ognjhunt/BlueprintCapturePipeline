#!/usr/bin/env bash
# =============================================================================
# Install full local ML stack for nurec_shim (CUDA COLMAP + SAM3 + DA3 + 3DGRUT)
# =============================================================================
# This script is designed for VM/bootstrap/snapshot use.
# It pre-installs dependencies and pre-warms model caches so runtime does not
# need network downloads.
# =============================================================================

set -euo pipefail

WITH_FIXER=false
WITH_LOCAL_QWEN=false
SKIP_PREWARM=false
COLMAP_REF="${COLMAP_REF:-main}"
COLMAP_CUDA_ARCHS="${COLMAP_CUDA_ARCHS:-89}"

THREEDGRUT_DIR="${THREEDGRUT_DIR:-/opt/3dgrut}"
SAM3_DIR="${SAM3_DIR:-/opt/sam3}"
DA3_DIR="${DA3_DIR:-/opt/da3}"
DA3_WEIGHTS_DIR="${DA3_MODEL_PATH:-/opt/da3/weights/metric_large}"
DA3_MODEL_ID="${DA3_MODEL_ID:-depth-anything/DA3Metric-Large}"
DA3_MODEL_NAME="${DA3_MODEL_NAME:-da3metric-large}"
FIXER_DIR="${FIXER_DIR:-/opt/Fixer}"
FIXER_WEIGHTS_DIR="${FIXER_WEIGHTS_DIR:-/opt/fixer_weights}"
QWEN_EDIT_DIR="${QWEN_IMAGE_EDIT_MODEL_PATH:-/opt/qwen-image-edit}"
HF_CACHE_DIR="${HF_HOME:-/opt/hf}"

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

Options:
  --with-fixer           Install Fixer + weights (optional heavy stage)
  --with-local-qwen      Install local Qwen-Image-Edit weights (large download)
  --skip-prewarm         Skip model cache prewarm/offline checks
  --colmap-ref REF       COLMAP git ref (default: main)
  -h, --help             Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with-fixer)
      WITH_FIXER=true
      shift
      ;;
    --with-local-qwen)
      WITH_LOCAL_QWEN=true
      shift
      ;;
    --skip-prewarm)
      SKIP_PREWARM=true
      shift
      ;;
    --colmap-ref)
      COLMAP_REF="$2"
      shift 2
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
  python3 \
  python3-pip \
  python3-venv \
  rsync

# Best-effort texrecon install for textured visual mesh generation.
log "Installing texrecon (best-effort)..."
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends texrecon 2>/dev/null \
  || DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends mvs-texturing 2>/dev/null \
  || log "WARNING: texrecon/mvs-texturing not available, textured mesh will fall back to vertex-colored"

mkdir -p "$HF_CACHE_DIR"

log "Installing Python base dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install --no-cache-dir "huggingface_hub[cli]"

if [ -f /app/pyproject.toml ]; then
  log "Installing BlueprintCapturePipeline package..."
  python3 -m pip install --no-cache-dir -e /app
fi
if [ -f /app/requirements.txt ]; then
  python3 -m pip install --no-cache-dir -r /app/requirements.txt
fi

if [ -x /app/scripts/install_colmap_cuda.sh ]; then
  log "Installing CUDA-enabled COLMAP..."
  COLMAP_CUDA_ARCHS="$COLMAP_CUDA_ARCHS" /app/scripts/install_colmap_cuda.sh --ref "$COLMAP_REF"
else
  die "/app/scripts/install_colmap_cuda.sh not found. Sync repo first."
fi

log "Installing 3DGRUT..."
clone_or_update_repo "https://github.com/nv-tlabs/3DGRUT.git" "$THREEDGRUT_DIR" "main"
python3 -m pip install --no-cache-dir -e "$THREEDGRUT_DIR"

log "Installing SAM3..."
clone_or_update_repo "https://github.com/facebookresearch/sam3.git" "$SAM3_DIR" "main"
python3 -m pip install --no-cache-dir -e "$SAM3_DIR"

log "Installing Depth Anything V3..."
clone_or_update_repo "https://github.com/ByteDance-Seed/Depth-Anything-3.git" "$DA3_DIR" "main"
python3 -m pip install --no-cache-dir -e "$DA3_DIR"

log "Ensuring DA3 metric weights are present..."
mkdir -p "$DA3_WEIGHTS_DIR"
if [ ! -f "$DA3_WEIGHTS_DIR/config.json" ]; then
  HF_HOME="$HF_CACHE_DIR" huggingface-cli download depth-anything/DA3Metric-Large --local-dir "$DA3_WEIGHTS_DIR"
fi

if [ "$WITH_FIXER" = true ]; then
  log "Installing Fixer..."
  clone_or_update_repo "https://github.com/nv-tlabs/Fixer.git" "$FIXER_DIR" "main"
  # Keep the existing CUDA/PyTorch stack and install Fixer runtime deps explicitly.
  python3 -m pip install --no-cache-dir --no-deps "cosmos-predict2==1.0.9"
  python3 -m pip install --no-cache-dir \
    "accelerate==1.7.0" \
    clean-fid \
    datasets \
    "facexlib==0.3.0" \
    fire \
    imageio-ffmpeg \
    lpips \
    natsort \
    numpy \
    peft \
    "torchmetrics[image]" \
    wandb
  python3 -m pip install --no-cache-dir "git+https://github.com/openai/CLIP.git"
  mkdir -p "$FIXER_WEIGHTS_DIR"
  if [ ! -f "$FIXER_WEIGHTS_DIR/pretrained/pretrained_fixer.pkl" ]; then
    HF_HOME="$HF_CACHE_DIR" huggingface-cli download nvidia/Fixer --local-dir "$FIXER_WEIGHTS_DIR"
  fi
fi

if [ "$WITH_LOCAL_QWEN" = true ]; then
  log "Installing Qwen-Image-Edit dependencies..."
  python3 -m pip install --no-cache-dir diffusers accelerate sentencepiece protobuf transformers

  log "Downloading Qwen-Image-Edit-2511 weights..."
  mkdir -p "$QWEN_EDIT_DIR"
  if [ ! -f "$QWEN_EDIT_DIR/model_index.json" ]; then
    HF_HOME="$HF_CACHE_DIR" huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir "$QWEN_EDIT_DIR"
  fi
else
  log "Skipping local Qwen-Image-Edit install (use Together provider by default)"
fi

if [ "$SKIP_PREWARM" = false ]; then
  log "Prewarming SAM3 + DA3 model caches..."
  HF_HOME="$HF_CACHE_DIR" python3 - <<PY
from pathlib import Path

from sam3 import build_sam3_image_model
from depth_anything_3.api import DepthAnything3

build_sam3_image_model()
da3_path = Path("${DA3_WEIGHTS_DIR}")
DepthAnything3.from_pretrained(
    str(da3_path) if da3_path.exists() else "${DA3_MODEL_ID}",
    model_name="${DA3_MODEL_NAME}",
)
print("PREWARM_OK")
PY

  log "Validating offline model load (no network)..."
  HF_HOME="$HF_CACHE_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 - <<PY
from pathlib import Path

from sam3 import build_sam3_image_model
from depth_anything_3.api import DepthAnything3

build_sam3_image_model()
DepthAnything3.from_pretrained(
    str(Path("${DA3_WEIGHTS_DIR}")),
    model_name="${DA3_MODEL_NAME}",
)
print("OFFLINE_LOAD_OK")
PY
  if [ "$WITH_LOCAL_QWEN" = true ]; then
    log "Prewarming Qwen-Image-Edit pipeline..."
    HF_HOME="$HF_CACHE_DIR" QWEN_IMAGE_EDIT_MODEL_PATH="$QWEN_EDIT_DIR" python3 - <<PY
import torch
from diffusers import QwenImageEditPlusPipeline
pipe = QwenImageEditPlusPipeline.from_pretrained("${QWEN_EDIT_DIR}", torch_dtype=torch.bfloat16)
print("QWEN_IMAGE_EDIT_PREWARM_OK")
PY

    log "Validating offline Qwen-Image-Edit load..."
    HF_HOME="$HF_CACHE_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      QWEN_IMAGE_EDIT_MODEL_PATH="$QWEN_EDIT_DIR" python3 - <<PY
import torch
from diffusers import QwenImageEditPlusPipeline
pipe = QwenImageEditPlusPipeline.from_pretrained("${QWEN_EDIT_DIR}", torch_dtype=torch.bfloat16)
print("QWEN_IMAGE_EDIT_OFFLINE_OK")
PY
  fi
fi

log "Writing environment profile..."
cat <<EOF | $SUDO tee /etc/profile.d/blueprint_capture_ml.sh >/dev/null
export THREEDGRUT_DIR=${THREEDGRUT_DIR}
export FIXER_DIR=${FIXER_DIR}
export FIXER_WEIGHTS_DIR=${FIXER_WEIGHTS_DIR}
export DA3_MODEL_PATH=${DA3_WEIGHTS_DIR}
export DA3_MODEL_NAME=${DA3_MODEL_NAME}
export QWEN_IMAGE_EDIT_MODEL_PATH=${QWEN_EDIT_DIR}
export HF_HOME=${HF_CACHE_DIR}
export CROP_CLEANUP_PROVIDER=skip
EOF

log "Verifying installed stack..."
colmap help 2>&1 | head -n 5
python3 - <<'PY'
import torch
print(f"TORCH={torch.__version__} CUDA={torch.version.cuda} GPU={torch.cuda.is_available()}")
PY

log "ML stack install complete."
