#!/usr/bin/env bash
# =============================================================================
# Install Inpaint360GS for scene cleaning (object removal + background inpainting)
# =============================================================================
# This script installs Inpaint360GS and its dependencies for use with the
# BlueprintCapturePipeline's Stage 9.5 scene cleaning.
#
# Inpaint360GS: https://github.com/dfki-av/Inpaint360GS (WACV 2026)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

INSTALL_DIR="${INPAINT360GS_DIR:-/opt/Inpaint360GS}"
INPAINT360GS_PYTHON="${INPAINT360GS_PYTHON:-python3.10}"
# Pin to a known-good commit for reproducibility
INPAINT360GS_REF="${INPAINT360GS_REF:-main}"
SKIP_CUDA_KERNELS="${SKIP_CUDA_KERNELS:-false}"
SKIP_LAMA="${SKIP_LAMA:-false}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

log() {
  echo "[install-inpaint360gs] $*"
}

die() {
  echo "[install-inpaint360gs] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  install_inpaint360gs.sh [options]

Options:
  --install-dir DIR       Installation directory (default: /opt/Inpaint360GS)
  --ref REF               Git ref to checkout (default: main)
  --skip-cuda-kernels     Skip building CUDA rasterization kernels
  --skip-lama             Skip LaMa dependency installation
  -h, --help              Show this help

Environment variables:
  INPAINT360GS_DIR        Same as --install-dir
  INPAINT360GS_PYTHON     Python binary to use (default: python3.10)
  INPAINT360GS_REF        Same as --ref
  CUDA_HOME               CUDA toolkit directory (auto-detected if unset)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --ref)
      INPAINT360GS_REF="$2"
      shift 2
      ;;
    --skip-cuda-kernels)
      SKIP_CUDA_KERNELS=true
      shift
      ;;
    --skip-lama)
      SKIP_LAMA=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

# ── Validate prerequisites ──────────────────────────────────────────────────

command -v git >/dev/null || die "git is required but not found"
command -v "$INPAINT360GS_PYTHON" >/dev/null || die "$INPAINT360GS_PYTHON is required but not found"

PYTHON_VERSION="$("$INPAINT360GS_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "Using Python: $INPAINT360GS_PYTHON (v${PYTHON_VERSION})"

if [ "$SKIP_CUDA_KERNELS" = "false" ]; then
  if [ -z "${CUDA_HOME:-}" ]; then
    # Auto-detect CUDA
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12; do
      if [ -d "$cuda_dir" ]; then
        export CUDA_HOME="$cuda_dir"
        break
      fi
    done
  fi
  [ -n "${CUDA_HOME:-}" ] || die "CUDA_HOME not set and no CUDA installation found. Use --skip-cuda-kernels to skip."
  log "CUDA_HOME: $CUDA_HOME"
fi

# ── Clone repository ────────────────────────────────────────────────────────

if [ -d "$INSTALL_DIR/.git" ]; then
  log "Inpaint360GS already cloned at $INSTALL_DIR"
  cd "$INSTALL_DIR"
  git fetch origin
else
  log "Cloning Inpaint360GS to $INSTALL_DIR..."
  $SUDO mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --depth 1 https://github.com/dfki-av/Inpaint360GS.git "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi

if [ "$INPAINT360GS_REF" != "main" ]; then
  log "Checking out ref: $INPAINT360GS_REF"
  git fetch origin "$INPAINT360GS_REF" --depth 1
  git checkout "$INPAINT360GS_REF"
fi

log "Commit: $(git rev-parse --short HEAD)"

# ── Install Python dependencies ─────────────────────────────────────────────

log "Installing Inpaint360GS Python dependencies..."

# Core 3DGS dependencies (plyfile, tqdm, etc.)
"$INPAINT360GS_PYTHON" -m pip install --no-cache-dir \
  plyfile \
  tqdm \
  lpips \
  open3d \
  trimesh \
  Pillow \
  2>&1 | tail -5

# ── Build CUDA kernels ──────────────────────────────────────────────────────

if [ "$SKIP_CUDA_KERNELS" = "false" ]; then
  log "Building diff-gaussian-rasterization CUDA kernel..."
  if [ -d "$INSTALL_DIR/submodules/diff-gaussian-rasterization" ]; then
    "$INPAINT360GS_PYTHON" -m pip install --no-cache-dir --no-build-isolation \
      "$INSTALL_DIR/submodules/diff-gaussian-rasterization" \
      2>&1 | tail -5
    log "  diff-gaussian-rasterization installed"
  else
    log "  WARNING: submodules/diff-gaussian-rasterization not found, skipping"
  fi

  log "Building simple-knn CUDA kernel..."
  if [ -d "$INSTALL_DIR/submodules/simple-knn" ]; then
    "$INPAINT360GS_PYTHON" -m pip install --no-cache-dir --no-build-isolation \
      "$INSTALL_DIR/submodules/simple-knn" \
      2>&1 | tail -5
    log "  simple-knn installed"
  else
    log "  WARNING: submodules/simple-knn not found, skipping"
  fi
else
  log "Skipping CUDA kernel builds (--skip-cuda-kernels)"
fi

# ── Install LaMa (inpainting model) ─────────────────────────────────────────

if [ "$SKIP_LAMA" = "false" ]; then
  log "Installing LaMa inpainting dependencies..."
  LAMA_REQ="$INSTALL_DIR/LaMa/requirements.txt"
  if [ -f "$LAMA_REQ" ]; then
    "$INPAINT360GS_PYTHON" -m pip install --no-cache-dir -r "$LAMA_REQ" \
      2>&1 | tail -5
    log "  LaMa dependencies installed"
  else
    log "  WARNING: LaMa/requirements.txt not found, skipping"
  fi

  # Download LaMa pretrained weights if not present
  LAMA_WEIGHTS_DIR="$INSTALL_DIR/LaMa/big-lama"
  if [ ! -d "$LAMA_WEIGHTS_DIR" ]; then
    log "  LaMa weights not found at $LAMA_WEIGHTS_DIR"
    log "  They may be downloaded automatically on first run"
  else
    log "  LaMa weights found at $LAMA_WEIGHTS_DIR"
  fi
else
  log "Skipping LaMa installation (--skip-lama)"
fi

# ── Verify installation ─────────────────────────────────────────────────────

log "Verifying installation..."

VERIFY_RESULT="$("$INPAINT360GS_PYTHON" -c "
import sys
checks = []

# Check core scripts exist
from pathlib import Path
install_dir = Path('${INSTALL_DIR}')
required_scripts = ['train.py', 'train_finetune.py', 'edit_object_removal.py', 'edit_object_inpaint.py']
for script in required_scripts:
    if (install_dir / script).is_file():
        checks.append(f'  {script}: OK')
    else:
        checks.append(f'  {script}: MISSING')

# Check Python imports
for mod in ('torch', 'numpy', 'PIL', 'plyfile', 'tqdm'):
    try:
        __import__(mod)
        checks.append(f'  import {mod}: OK')
    except ImportError:
        checks.append(f'  import {mod}: MISSING')

print('\n'.join(checks))
" 2>&1)"

echo "$VERIFY_RESULT"

RUNNER_SCRIPT="${REPO_ROOT}/scripts/inpaint360gs_runner.py"
if [ -f "$RUNNER_SCRIPT" ]; then
  log "Running inpaint360gs_runner probe..."
  "$INPAINT360GS_PYTHON" "$RUNNER_SCRIPT" --probe || die "inpaint360gs_runner probe failed"
fi

log ""
log "============================================================"
log "Inpaint360GS installation complete"
log "============================================================"
log "  Install dir:  $INSTALL_DIR"
log "  Python:       $INPAINT360GS_PYTHON (v${PYTHON_VERSION})"
log "  Git commit:   $(cd "$INSTALL_DIR" && git rev-parse --short HEAD)"
log ""
log "Set these env vars for the pipeline:"
log "  export INPAINT360GS_DIR=$INSTALL_DIR"
log "  export INPAINT360GS_PYTHON=$INPAINT360GS_PYTHON"
log ""
log "To test, run:  python3 -m pytest tests/test_inpaint360gs_runner.py -v"
