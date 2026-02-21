#!/usr/bin/env bash
# Install GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting
# https://github.com/GSFix3D/GSFix3D
#
# Requirements: Python 3.11, CUDA 12.1+, ~10GB disk (code + SD-v2 weights)
# Tested on: Ubuntu 22.04, RTX 4090/4500 Ada
set -euo pipefail

GSFIX3D_DIR="${GSFIX3D_DIR:-/opt/GSFix3D}"
GSFIX3D_WEIGHTS_DIR="${GSFIX3D_WEIGHTS_DIR:-/opt/gsfix3d_weights}"
PYTHON="${GSFIX3D_PYTHON:-python3.11}"

log() { echo "[install-gsfix3d] $*"; }

# ---------------------------------------------------------------------------
# Clone repo
# ---------------------------------------------------------------------------
if [ -d "$GSFIX3D_DIR" ] && [ -f "$GSFIX3D_DIR/scripts/gsfixer/inference.py" ]; then
    log "GSFix3D already installed at $GSFIX3D_DIR"
else
    log "Cloning GSFix3D..."
    git clone --recursive https://github.com/GSFix3D/GSFix3D.git "$GSFIX3D_DIR"
fi

cd "$GSFIX3D_DIR"

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
log "Installing Python dependencies..."
$PYTHON -m pip install --no-cache-dir -r requirements.txt 2>/dev/null || {
    log "WARNING: requirements.txt install had errors; installing core deps individually..."
    $PYTHON -m pip install --no-cache-dir \
        torch torchvision \
        diffusers transformers accelerate \
        open3d plyfile trimesh \
        pillow numpy scipy \
        lpips safetensors einops omegaconf
}

# ---------------------------------------------------------------------------
# Build diff-gaussian-rasterization (custom CUDA extension)
# ---------------------------------------------------------------------------
if [ -d "$GSFIX3D_DIR/diff-gaussian-rasterization" ]; then
    log "Building diff-gaussian-rasterization CUDA extension..."
    cd "$GSFIX3D_DIR/diff-gaussian-rasterization"
    $PYTHON -m pip install . --no-build-isolation 2>/dev/null || {
        log "WARNING: diff-gaussian-rasterization build failed; may need manual CUDA setup"
    }
    cd "$GSFIX3D_DIR"
fi

# ---------------------------------------------------------------------------
# Download base weights (Stable Diffusion v2 + GSFixer pretrained)
# ---------------------------------------------------------------------------
mkdir -p "$GSFIX3D_WEIGHTS_DIR"

# GSFixer pretrained checkpoints from HuggingFace
if [ ! -d "$GSFIX3D_WEIGHTS_DIR/gsfixer-full" ]; then
    log "Downloading GSFixer pretrained weights from HuggingFace..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download goldoak1421/gsfixer-full \
            --local-dir "$GSFIX3D_WEIGHTS_DIR/gsfixer-full" || {
            log "WARNING: Failed to download gsfixer-full; will need manual download"
            log "  Visit: https://huggingface.co/collections/goldoak1421/gsfix3d"
        }
    else
        log "huggingface-cli not found; installing..."
        $PYTHON -m pip install --no-cache-dir huggingface_hub[cli]
        huggingface-cli download goldoak1421/gsfixer-full \
            --local-dir "$GSFIX3D_WEIGHTS_DIR/gsfixer-full" || {
            log "WARNING: Failed to download gsfixer-full weights"
        }
    fi
fi

# Stable Diffusion v2 base (needed for training from scratch; optional for inference)
if [ ! -d "$GSFIX3D_WEIGHTS_DIR/stable-diffusion-2" ]; then
    log "NOTE: Stable Diffusion v2 base not downloaded."
    log "  For fine-tuning, download from: stabilityai/stable-diffusion-2"
    log "  For inference-only, GSFixer pretrained weights are sufficient."
fi

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
log "Verifying installation..."
VERIFY=$($PYTHON -c "
import sys
checks = []
try:
    import torch; checks.append(f'torch={torch.__version__}')
except: checks.append('torch=MISSING')
try:
    import diffusers; checks.append(f'diffusers={diffusers.__version__}')
except: checks.append('diffusers=MISSING')
try:
    import open3d; checks.append(f'open3d={open3d.__version__}')
except: checks.append('open3d=MISSING')
try:
    import diff_gaussian_rasterization; checks.append('rasterizer=OK')
except: checks.append('rasterizer=MISSING')
print(' | '.join(checks))
" 2>/dev/null || echo "verify_failed")

log "Install check: $VERIFY"
log "GSFix3D installed at: $GSFIX3D_DIR"
log "Weights at: $GSFIX3D_WEIGHTS_DIR"

# Set env vars for pipeline integration
log ""
log "Add to your environment:"
log "  export GSFIX3D_DIR=$GSFIX3D_DIR"
log "  export GSFIX3D_WEIGHTS_DIR=$GSFIX3D_WEIGHTS_DIR"
log "  export BASE_CKPT_DIR=$GSFIX3D_WEIGHTS_DIR"
