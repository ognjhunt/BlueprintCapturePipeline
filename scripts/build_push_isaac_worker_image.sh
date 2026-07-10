#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="$repo_root/deploy/docker/robot_eval_worker/isaac/Dockerfile"
image_ref="${BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF:-${BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF:-}}"
platform="${BLUEPRINT_ISAAC_WORKER_PLATFORM:-linux/amd64}"
base_image="${BLUEPRINT_ISAAC_SIM_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d}"
allow_push="${BLUEPRINT_ALLOW_ISAAC_WORKER_IMAGE_PUSH:-false}"
manifest_output="${BLUEPRINT_ISAAC_WORKER_IMAGE_MANIFEST_OUTPUT:-$repo_root/output/isaac_worker_image_manifest_diagnostic.json}"

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

# Registry manifest diagnostic v2 (layer layout, sizes, recommended startup
# timeout) via the shared module — the same shape every launcher consumes.
digest_output="${BLUEPRINT_ISAAC_WORKER_RESOLVED_DIGEST_OUTPUT:-$repo_root/output/isaac_worker_image_resolved_digest.txt}"
if command -v python3 >/dev/null; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m blueprint_pipeline.isaac_worker_image_manifest \
      --image "$image_ref" \
      --output "$manifest_output"
  # A mutable tag is never final evidence: resolve and record the immutable
  # digest reference, and fail the push path when it cannot be resolved.
  resolved_digest_ref="$(
    BLUEPRINT_IMAGE_MANIFEST_OUTPUT="$manifest_output" python3 - <<'PY'
import json, os
payload = json.load(open(os.environ["BLUEPRINT_IMAGE_MANIFEST_OUTPUT"], encoding="utf-8"))
print(payload.get("resolved_digest_ref") or "")
PY
  )"
  if [[ "$allow_push" == "true" ]]; then
    if [[ "$resolved_digest_ref" != *@sha256:* ]]; then
      echo "pushed image digest could not be resolved; mutable tag is not final evidence" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$digest_output")"
    printf '%s\n' "$resolved_digest_ref" > "$digest_output"
    echo "resolved immutable digest: $resolved_digest_ref"
    echo "wrote resolved digest ref: $digest_output"
  fi
else
  echo "python3 not found; skipped Isaac worker image manifest diagnostic" >&2
fi

cat >&2 <<'EOF'
NOTE: build/publish completion is NOT worker readiness evidence. Before any
success claim, run BOTH canaries against the new immutable digest:
  1. fast startup canary (isaac_worker_runtime_preflight via the parity job
     --image-startup-canary lane), and
  2. review renderer canary (blueprint_pipeline.isaac_review_renderer_canary).
Registry inspection alone proves layer layout, never startup or rendering.
EOF
