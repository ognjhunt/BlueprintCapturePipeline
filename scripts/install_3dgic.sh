#!/usr/bin/env bash
# Install 3DGIC: 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency
# https://github.com/peterjohnsonhuang/3dgic
# (CVPR 2025, NVIDIA Research)
#
# Purpose: Object removal + inpainting (NOT gap-filling for unseen regions).
# Use for: removing specific objects from a trained 3DGS scene with multi-view consistency.
#
# Requirements: Python 3.10+, CUDA 12.x, ~5GB disk
# Tested by authors on: RTX 5090 with nvidia-driver-570
set -euo pipefail

DGIC_DIR="${DGIC_DIR:-/opt/3dgic}"
PYTHON="${DGIC_PYTHON:-python3.10}"

log() { echo "[install-3dgic] $*"; }

# ---------------------------------------------------------------------------
# Clone repo
# ---------------------------------------------------------------------------
if [ -d "$DGIC_DIR" ] && [ -f "$DGIC_DIR/train.py" ]; then
    log "3DGIC already installed at $DGIC_DIR"
else
    log "Cloning 3DGIC..."
    git clone --recursive https://github.com/peterjohnsonhuang/3dgic.git "$DGIC_DIR"
fi

cd "$DGIC_DIR"

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
log "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    $PYTHON -m pip install --no-cache-dir -r requirements.txt 2>/dev/null || {
        log "WARNING: requirements.txt had errors; installing core deps..."
    }
fi

# Build CUDA extensions
for subdir in diff-gaussian-rasterization-depth simple-knn; do
    if [ -d "$DGIC_DIR/$subdir" ]; then
        log "Building $subdir CUDA extension..."
        cd "$DGIC_DIR/$subdir"
        $PYTHON -m pip install . --no-build-isolation 2>/dev/null || {
            log "WARNING: $subdir build failed"
        }
        cd "$DGIC_DIR"
    fi
done

# Install nvdiffrast if available
if [ -d "$DGIC_DIR/nvdiffrast" ]; then
    log "Building nvdiffrast..."
    cd "$DGIC_DIR/nvdiffrast"
    $PYTHON -m pip install . 2>/dev/null || log "WARNING: nvdiffrast build failed"
    cd "$DGIC_DIR"
fi

$PYTHON -m pip install --no-cache-dir lpips plyfile 2>/dev/null || true

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
log "3DGIC installed at: $DGIC_DIR"
log ""
log "NOTE: 3DGIC is a 'suboptimal version' per the authors."
log "For object removal workflow:"
log "  1. Train 3DGS model: bash script/run_bear.sh"
log "  2. Run object removal"
log "  3. Get depth-guided masks"
log "  4. 2D inpainting (external: LAMA or SDXL-inpaint)"
log "  5. 3D inpainting via final script"
log ""
log "Add to your environment:"
log "  export DGIC_DIR=$DGIC_DIR"
