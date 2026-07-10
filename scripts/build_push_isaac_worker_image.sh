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

if command -v python3 >/dev/null; then
  BLUEPRINT_IMAGE_REF="$image_ref" \
  BLUEPRINT_IMAGE_MANIFEST_OUTPUT="$manifest_output" \
  python3 - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

image_ref = os.environ["BLUEPRINT_IMAGE_REF"]
output_path = Path(os.environ["BLUEPRINT_IMAGE_MANIFEST_OUTPUT"]).expanduser()
output_path.parent.mkdir(parents=True, exist_ok=True)
result = subprocess.run(
    ["docker", "buildx", "imagetools", "inspect", "--raw", image_ref],
    capture_output=True,
    text=True,
    timeout=120,
    check=False,
)
payload = {
    "schema_version": "isaac_worker_image_manifest_diagnostic.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "image_ref": image_ref,
    "command": "docker buildx imagetools inspect --raw <image-ref>",
    "exit_code": result.returncode,
    "raw_secret_values_recorded": False,
}
if result.returncode != 0:
    payload.update(
        {
            "status": "blocked",
            "blockers": ["worker_image_manifest_inspection_failed"],
            "stderr_tail": result.stderr[-1000:],
            "proof_boundary": (
                "Image manifest inspection failed. This does not prove provider "
                "startup or Isaac Sim execution."
            ),
        }
    )
else:
    manifest = json.loads(result.stdout)
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        layers = manifest.get("manifests") if isinstance(manifest.get("manifests"), list) else []
    normalized_layers = []
    sizes = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        size = layer.get("size")
        size_value = int(size) if isinstance(size, (int, float)) and size >= 0 else None
        if size_value is not None:
            sizes.append(size_value)
        normalized_layers.append(
            {
                "mediaType": layer.get("mediaType"),
                "digest": layer.get("digest"),
                "size": size_value,
                "platform": layer.get("platform"),
            }
        )
    total_size = sum(sizes) if sizes else None
    largest_layer = max(sizes) if sizes else None
    payload.update(
        {
            "status": "completed",
            "mediaType": manifest.get("mediaType"),
            "layer_count": len(sizes),
            "total_compressed_size_bytes": total_size,
            "largest_layer_size_bytes": largest_layer,
            "large_image_pull_risk": bool(
                (total_size is not None and total_size >= 8_000_000_000)
                or (largest_layer is not None and largest_layer >= 3_000_000_000)
            ),
            "layers": normalized_layers,
            "proof_boundary": (
                "Registry manifest size metadata only. This does not prove "
                "container startup or Isaac Sim execution."
            ),
        }
    )
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote Isaac worker image manifest diagnostic: {output_path}")
PY
else
  echo "python3 not found; skipped Isaac worker image manifest diagnostic" >&2
fi
