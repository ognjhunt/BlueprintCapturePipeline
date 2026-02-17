#!/usr/bin/env bash
# =============================================================================
# Install CUDA-enabled COLMAP (headless-friendly) on Ubuntu.
# =============================================================================
# Example:
#   ./scripts/install_colmap_cuda.sh --ref main
#
# Optional env:
#   COLMAP_REF=main
#   COLMAP_SRC_DIR=/opt/src/colmap
#   COLMAP_INSTALL_PREFIX=/usr/local
# =============================================================================

set -euo pipefail

COLMAP_REF="${COLMAP_REF:-main}"
COLMAP_SRC_DIR="${COLMAP_SRC_DIR:-/opt/src/colmap}"
COLMAP_BUILD_DIR="${COLMAP_BUILD_DIR:-${COLMAP_SRC_DIR}/build}"
COLMAP_INSTALL_PREFIX="${COLMAP_INSTALL_PREFIX:-/usr/local}"
COLMAP_CUDA_ARCHS="${COLMAP_CUDA_ARCHS:-89}"
INSTALL_DEPS_ONLY=false

log() {
  echo "[install-colmap-cuda] $*"
}

die() {
  echo "[install-colmap-cuda] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  install_colmap_cuda.sh [options]

Options:
  --ref REF            Git ref/tag/branch to build (default: main)
  --src-dir DIR        Source checkout dir (default: /opt/src/colmap)
  --build-dir DIR      Build dir (default: <src-dir>/build)
  --install-prefix DIR Install prefix (default: /usr/local)
  --deps-only          Install build deps only, skip build/install
  -h, --help           Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --ref)
      COLMAP_REF="$2"
      shift 2
      ;;
    --src-dir)
      COLMAP_SRC_DIR="$2"
      shift 2
      ;;
    --build-dir)
      COLMAP_BUILD_DIR="$2"
      shift 2
      ;;
    --install-prefix)
      COLMAP_INSTALL_PREFIX="$2"
      shift 2
      ;;
    --deps-only)
      INSTALL_DEPS_ONLY=true
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

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "WARNING: nvidia-smi not found. Continuing (common in container build contexts)."
fi

if ! command -v nvcc >/dev/null 2>&1; then
  log "WARNING: nvcc not found in PATH. CUDA toolkit may be missing."
  log "Install CUDA toolkit first (matching your driver), then rerun."
fi

log "Installing build dependencies..."
$SUDO apt-get update -o Acquire::Retries=5 --fix-missing
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --fix-missing -o Acquire::Retries=5 --no-install-recommends \
  build-essential \
  ccache \
  cmake \
  git \
  ninja-build \
  pkg-config \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-program-options-dev \
  libboost-system-dev \
  libcgal-dev \
  libceres-dev \
  libeigen3-dev \
  libflann-dev \
  libfreeimage-dev \
  libgflags-dev \
  libglew-dev \
  libgoogle-glog-dev \
  libopenimageio-dev \
  libopencv-dev \
  openimageio-tools \
  libopenexr-dev \
  libmetis-dev \
  libsqlite3-dev \
  libsuitesparse-dev \
  libatlas-base-dev \
  libgl1-mesa-dev

if [ "$INSTALL_DEPS_ONLY" = true ]; then
  log "Dependency install complete (--deps-only)."
  exit 0
fi

log "Preparing COLMAP source at ${COLMAP_SRC_DIR} (ref=${COLMAP_REF})..."
mkdir -p "$(dirname "$COLMAP_SRC_DIR")"
if [ ! -d "${COLMAP_SRC_DIR}/.git" ]; then
  git clone https://github.com/colmap/colmap.git "$COLMAP_SRC_DIR"
fi

git -C "$COLMAP_SRC_DIR" fetch --tags origin
git -C "$COLMAP_SRC_DIR" checkout "$COLMAP_REF"
if [ "$COLMAP_REF" = "main" ] || [ "$COLMAP_REF" = "master" ]; then
  git -C "$COLMAP_SRC_DIR" pull --ff-only origin "$COLMAP_REF"
fi

log "Configuring COLMAP (CUDA on, GUI off)..."
cmake -S "$COLMAP_SRC_DIR" -B "$COLMAP_BUILD_DIR" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$COLMAP_INSTALL_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="$COLMAP_CUDA_ARCHS" \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DBUILD_TESTING=OFF

log "Building COLMAP..."
cmake --build "$COLMAP_BUILD_DIR" -j"$(nproc)"

log "Installing COLMAP into ${COLMAP_INSTALL_PREFIX}..."
$SUDO cmake --install "$COLMAP_BUILD_DIR"
# Ensure runtime can resolve libs installed under /usr/local/lib64 (e.g., ONNX runtime).
if [ -d "${COLMAP_INSTALL_PREFIX}/lib64" ]; then
  echo "${COLMAP_INSTALL_PREFIX}/lib64" | $SUDO tee /etc/ld.so.conf.d/colmap-local-lib64.conf >/dev/null
fi
$SUDO ldconfig || true

COLMAP_BIN="${COLMAP_INSTALL_PREFIX}/bin/colmap"
if [ ! -x "$COLMAP_BIN" ]; then
  COLMAP_BIN="$(command -v colmap || true)"
fi
[ -n "$COLMAP_BIN" ] || die "COLMAP binary not found after install"

if ! HELP_OUT="$("$COLMAP_BIN" help 2>&1)"; then
  echo "$HELP_OUT"
  die "COLMAP failed to execute after install (likely missing runtime libs)."
fi

log "Installed COLMAP help banner:"
echo "$HELP_OUT" | head -n 8

if echo "$HELP_OUT" | grep -qi "without cuda"; then
  die "Installed COLMAP still reports 'without CUDA'. Check your CUDA toolkit and CMake configure output."
fi

log "Success: CUDA-enabled COLMAP installed at ${COLMAP_BIN}"
