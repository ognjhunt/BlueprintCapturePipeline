#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="$repo_root/deploy/docker/robot_eval_worker/isaac/Dockerfile"
image_ref="${BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF:-${BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF:-}}"
platform="${BLUEPRINT_ISAAC_WORKER_PLATFORM:-linux/amd64}"
base_image="${BLUEPRINT_ISAAC_SIM_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
allow_push="${BLUEPRINT_ALLOW_ISAAC_WORKER_IMAGE_PUSH:-false}"

if [[ -z "$image_ref" ]]; then
  echo "missing BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF" >&2
  exit 2
fi

if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]; then
  echo "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF must be versioned: $image_ref" >&2
  exit 2
fi

case "$image_ref" in
  *:latest|*:local|*:dev|*:test)
    echo "refuses non-launch worker image tag: $image_ref" >&2
    exit 2
    ;;
esac

docker info >/dev/null

if [[ "$allow_push" == "true" ]]; then
  docker buildx build \
    --platform "$platform" \
    --build-arg "ISAAC_SIM_BASE_IMAGE=$base_image" \
    -f "$dockerfile" \
    -t "$image_ref" \
    --push \
    "$repo_root"
else
  docker buildx build \
    --platform "$platform" \
    --build-arg "ISAAC_SIM_BASE_IMAGE=$base_image" \
    -f "$dockerfile" \
    -t "$image_ref" \
    --load \
    "$repo_root"
fi
