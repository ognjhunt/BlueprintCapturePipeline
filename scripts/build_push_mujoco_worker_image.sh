#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="$repo_root/deploy/docker/robot_eval_worker/mujoco/Dockerfile"
image_ref="${BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF:-}"
platform="${BLUEPRINT_MUJOCO_WORKER_PLATFORM:-linux/amd64}"
allow_push="${BLUEPRINT_ALLOW_MUJOCO_WORKER_IMAGE_PUSH:-false}"

if [[ -z "$image_ref" ]]; then
  echo "missing BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF" >&2
  exit 2
fi

if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]; then
  echo "BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF must be versioned: $image_ref" >&2
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
    -f "$dockerfile" \
    -t "$image_ref" \
    --push \
    "$repo_root"
else
  docker buildx build \
    --platform "$platform" \
    -f "$dockerfile" \
    -t "$image_ref" \
    --load \
    "$repo_root"
fi
