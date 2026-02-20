#!/usr/bin/env bash
# =============================================================================
# Build and push the GPU snapshot image (preloaded ML stack, no runtime pulls)
# Works with Vast.ai, RunPod, Lambda, or any CUDA-capable host.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_REPO="${IMAGE_REPO:-nijelhunt/blueprint-capture-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-cuda-snapshot}"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
PUSH=true

usage() {
  cat <<'EOF'
Usage:
  build_vast_snapshot.sh [options]

Options:
  --tag TAG          Image tag (default: cuda-snapshot)
  --repo REPO        Image repository (default: nijelhunt/blueprint-capture-pipeline)
  --platform VALUE   Target platform (default: linux/amd64)
  --no-push          Build only, do not push
  -h, --help         Show help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --repo)
      IMAGE_REPO="$2"
      shift 2
      ;;
    --platform)
      TARGET_PLATFORM="$2"
      shift 2
      ;;
    --no-push)
      PUSH=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"

echo "[build-snapshot] Building ${IMAGE}..."
echo "[build-snapshot] Local Qwen-Image-Edit weights are intentionally excluded from snapshot builds."
docker build \
  --platform "$TARGET_PLATFORM" \
  -f "$PROJECT_ROOT/Dockerfile.snapshot" \
  -t "$IMAGE" \
  "$PROJECT_ROOT"

if [ "$PUSH" = true ]; then
  echo "[build-snapshot] Pushing ${IMAGE}..."
  docker push "$IMAGE"
fi

echo "[build-snapshot] Done: ${IMAGE}"
