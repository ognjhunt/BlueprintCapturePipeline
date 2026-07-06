#!/usr/bin/env bash
# Build + optionally push the sealed blueprint-groot-oscar-eval worker image
# (GR00T N1.7 + SONIC x OSCAR-2B closed-loop lane). Reproducible clean-build path.
# The PRIMARY build path is scripts/snapshot_groot_oscar_eval_pod.sh (crane
# snapshot of an already-provisioned pod); this Dockerfile build is the
# from-the-pinned-base fallback + source of truth.
#
# Usage:
#   BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF=docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64 \
#   BLUEPRINT_ALLOW_GROOT_OSCAR_CLOSED_LOOP_IMAGE_PUSH=true \
#   ./scripts/build_push_groot_oscar_closed_loop_image.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker_dir="$repo_root/deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
dockerfile="$docker_dir/Dockerfile"
image_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF:-}"
platform="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_PLATFORM:-linux/amd64}"
allow_push="${BLUEPRINT_ALLOW_GROOT_OSCAR_CLOSED_LOOP_IMAGE_PUSH:-false}"
base_image="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BASE_IMAGE:-docker.io/nijelhunt/blueprint-oscar-wam@sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3}"
groot_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_GROOT_SOURCE_REF:-e5749287857afd97b78f1147166137de29746392}"
prefetch_checkpoints="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_PREFETCH_CHECKPOINTS:-true}"
hf_token_file="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_HF_TOKEN_FILE:-${HF_TOKEN_FILE:-$HOME/.blueprint-secrets/hf_token}}"
manifest_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_MANIFEST_OUTPUT:-$repo_root/output/groot_oscar_closed_loop_image_manifest.json}"
min_free_gib="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_MIN_FREE_GIB:-120}"
disk_check_free_kib=""
disk_check_required_kib=""

write_manifest() {
  local status="$1" blockers_json="$2"
  python3 - "$manifest_output" "$status" "$blockers_json" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoints" "$min_free_gib" "${disk_check_free_kib:-}" "${disk_check_required_kib:-}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

def _int_or_none(v):
    try:
        return int(v) if v else None
    except ValueError:
        return None

out = Path(sys.argv[1]).expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
free_kib = _int_or_none(sys.argv[10])
required_kib = _int_or_none(sys.argv[11])
payload = {
    "schema_version": "groot_oscar_closed_loop_image_build_manifest.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "blockers": json.loads(sys.argv[3]),
    "image_ref": sys.argv[4] or None,
    "platform": sys.argv[5],
    "base_image": sys.argv[6],
    "groot_source_ref": sys.argv[7],
    "prefetch_checkpoints": sys.argv[8].lower() == "true",
    "dockerfile": "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile",
    "local_disk_check": {
        "min_free_gib": _int_or_none(sys.argv[9]),
        "available_free_gib": round(free_kib / 1024 / 1024, 3) if free_kib is not None else None,
        "required_free_gib": round(required_kib / 1024 / 1024, 3) if required_kib is not None else None,
    },
    "raw_secret_values_recorded": False,
    "claim_boundary": {
        "image_build_is_not_provider_startup": True,
        "image_build_is_not_policy_inference": True,
        "image_build_is_not_task_success": True,
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
}

if [[ -z "$image_ref" ]]; then
  write_manifest "blocked" '["missing_BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF"]' >/dev/null
  echo "missing BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF" >&2
  exit 2
fi
if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]; then
  write_manifest "blocked" '["groot_oscar_closed_loop_image_ref_must_be_versioned"]' >/dev/null
  echo "image ref must be versioned: $image_ref" >&2
  exit 2
fi
case "$image_ref" in
  *:latest|*:local|*:dev|*:test)
    write_manifest "blocked" '["groot_oscar_closed_loop_image_ref_refuses_unstable_tag"]' >/dev/null
    echo "refuses non-launch image tag: $image_ref" >&2
    exit 2
    ;;
esac

docker info >/dev/null

if [[ "${BLUEPRINT_SKIP_GROOT_OSCAR_CLOSED_LOOP_DISK_CHECK:-false}" != "true" ]]; then
  disk_check_free_kib="$(df -Pk "$repo_root" | awk 'NR==2 {print $4}')"
  disk_check_required_kib=$((min_free_gib * 1024 * 1024))
  if [[ "${disk_check_free_kib:-0}" -lt "$disk_check_required_kib" ]]; then
    write_manifest "blocked" '["insufficient_local_disk_for_groot_oscar_closed_loop_image_build"]' >/dev/null
    echo "insufficient local disk: need ${min_free_gib}GiB free (bakes both checkpoints)" >&2
    exit 2
  fi
fi

secret_args=()
if [[ "$prefetch_checkpoints" == "true" && -f "$hf_token_file" ]]; then
  secret_args=(--secret "id=hf_token,src=$hf_token_file")
fi

build_args=(
  docker buildx build
  --platform "$platform"
  --progress plain
  --build-arg "BASE_IMAGE=$base_image"
  --build-arg "GROOT_SOURCE_REF=$groot_ref"
  --build-arg "PREFETCH_CHECKPOINTS=$prefetch_checkpoints"
  -f "$dockerfile"
  -t "$image_ref"
)
if [[ "$allow_push" == "true" ]]; then
  build_args+=(--push)
else
  build_args+=(--load)
fi
build_args+=("${secret_args[@]}" "$repo_root")

"${build_args[@]}"

python3 - "$manifest_output" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoints" "$allow_push" <<'PY'
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1]).expanduser()
image_ref = sys.argv[2]
inspect = subprocess.run(
    ["docker", "buildx", "imagetools", "inspect", "--raw", image_ref],
    capture_output=True, text=True, timeout=180, check=False,
)
payload = {
    "schema_version": "groot_oscar_closed_loop_image_build_manifest.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "completed" if inspect.returncode == 0 else "built_manifest_inspection_blocked",
    "image_ref": image_ref,
    "platform": sys.argv[3],
    "base_image": sys.argv[4],
    "groot_source_ref": sys.argv[5],
    "prefetch_checkpoints": sys.argv[6].lower() == "true",
    "pushed": sys.argv[7].lower() == "true",
    "dockerfile": "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile",
    "blockers": [] if inspect.returncode == 0 else ["groot_oscar_closed_loop_image_manifest_inspection_failed"],
    "raw_secret_values_recorded": False,
    "claim_boundary": {
        "image_build_is_not_provider_startup": True,
        "image_build_is_not_policy_inference": True,
        "image_build_is_not_task_success": True,
    },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
