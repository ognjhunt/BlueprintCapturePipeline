#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/artifact_storage.sh"
artifact_cache_root="$(blueprint_artifact_cache_root)"
docker_dir="$repo_root/deploy/docker/robot_eval_worker/unitree_groot_sonic_wam"
dockerfile="$docker_dir/Dockerfile"
image_ref="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_REF:-}"
platform="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_PLATFORM:-linux/amd64}"
allow_push="${BLUEPRINT_ALLOW_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_PUSH:-false}"
base_image="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_BASE_IMAGE:-docker.io/nijelhunt/blueprint-oscar-wam@sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3}"
groot_ref="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_GROOT_SOURCE_REF:-e5749287857afd97b78f1147166137de29746392}"
prefetch_checkpoint="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_PREFETCH_CHECKPOINT:-true}"
hf_token_file="${BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE:-${HF_TOKEN_FILE:-$HOME/.blueprint-secrets/hf_token}}"
manifest_output="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_MANIFEST_OUTPUT:-$artifact_cache_root/unitree_groot_sonic_wam_image_manifest.json}"
min_free_gib="${BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_MIN_FREE_GIB:-80}"
disk_check_enabled="true"
disk_check_free_kib=""
disk_check_required_kib=""

write_manifest() {
  local status="$1"
  local blockers_json="$2"
  python3 - "$manifest_output" "$status" "$blockers_json" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoint" "$disk_check_enabled" "$min_free_gib" "${disk_check_free_kib:-}" "${disk_check_required_kib:-}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _int_or_none(value: str):
    try:
        return int(value) if value else None
    except ValueError:
        return None


out = Path(sys.argv[1]).expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
free_kib = _int_or_none(sys.argv[11])
required_kib = _int_or_none(sys.argv[12])
free_gib = round(free_kib / 1024 / 1024, 3) if free_kib is not None else None
required_gib = round(required_kib / 1024 / 1024, 3) if required_kib is not None else None
payload = {
    "schema_version": "unitree_groot_sonic_wam_image_build_manifest.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "blockers": json.loads(sys.argv[3]),
    "image_ref": sys.argv[4] or None,
    "platform": sys.argv[5],
    "base_image": sys.argv[6],
    "groot_source_ref": sys.argv[7],
    "prefetch_checkpoint": sys.argv[8].lower() == "true",
    "dockerfile": "deploy/docker/robot_eval_worker/unitree_groot_sonic_wam/Dockerfile",
    "local_disk_check": {
        "enabled": sys.argv[9].lower() == "true",
        "min_free_gib": _int_or_none(sys.argv[10]),
        "free_kib": free_kib,
        "required_kib": required_kib,
        "available_free_gib": free_gib,
        "required_free_gib": required_gib,
    },
    "raw_secret_values_recorded": False,
    "claim_boundary": {
        "image_build_manifest_is_not_provider_startup": True,
        "image_build_manifest_is_not_policy_inference": True,
        "image_build_manifest_is_not_task_success": True,
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
}

if [[ -z "$image_ref" ]]; then
  write_manifest "blocked" '["missing_BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_REF"]' >/dev/null
  echo "missing BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_REF" >&2
  exit 2
fi

if [[ "$image_ref" != *:* && "$image_ref" != *@sha256:* ]]; then
  write_manifest "blocked" '["unitree_groot_sonic_wam_image_ref_must_be_versioned"]' >/dev/null
  echo "BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_REF must be versioned: $image_ref" >&2
  exit 2
fi

case "$image_ref" in
  *:latest|*:local|*:dev|*:test)
    write_manifest "blocked" '["unitree_groot_sonic_wam_image_ref_refuses_unstable_tag"]' >/dev/null
    echo "refuses non-launch image tag: $image_ref" >&2
    exit 2
    ;;
esac

docker info >/dev/null

if [[ "${BLUEPRINT_SKIP_UNITREE_GROOT_SONIC_IMAGE_DISK_CHECK:-false}" != "true" ]]; then
  disk_check_free_kib="$(df -Pk "$repo_root" | awk 'NR==2 {print $4}')"
  disk_check_required_kib=$((min_free_gib * 1024 * 1024))
  if [[ "${disk_check_free_kib:-0}" -lt "$disk_check_required_kib" ]]; then
    write_manifest "blocked" '["insufficient_local_disk_for_unitree_groot_sonic_wam_image_build"]' >/dev/null
    echo "insufficient local disk for sealed GR00T/SONIC image build: need ${min_free_gib}GiB free" >&2
    exit 2
  fi
else
  disk_check_enabled="false"
fi

secret_args=()
if [[ "$prefetch_checkpoint" == "true" && -f "$hf_token_file" ]]; then
  secret_args=(--secret "id=hf_token,src=$hf_token_file")
fi

build_args=(
  docker buildx build
  --platform "$platform"
  --progress plain
  --build-arg "BASE_IMAGE=$base_image"
  --build-arg "GROOT_SOURCE_REF=$groot_ref"
  --build-arg "PREFETCH_GROOT_CHECKPOINT=$prefetch_checkpoint"
  -f "$dockerfile"
  -t "$image_ref"
)

if [[ "$allow_push" == "true" ]]; then
  build_args+=(--push)
else
  build_args+=(--load)
fi
build_args+=("${secret_args[@]}" "$docker_dir")

"${build_args[@]}"

python3 - "$manifest_output" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoint" "$allow_push" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1]).expanduser()
image_ref = sys.argv[2]
inspect_result = subprocess.run(
    ["docker", "buildx", "imagetools", "inspect", "--raw", image_ref],
    capture_output=True,
    text=True,
    timeout=180,
    check=False,
)
payload = {
    "schema_version": "unitree_groot_sonic_wam_image_build_manifest.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "completed" if inspect_result.returncode == 0 else "built_manifest_inspection_blocked",
    "image_ref": image_ref,
    "platform": sys.argv[3],
    "base_image": sys.argv[4],
    "groot_source_ref": sys.argv[5],
    "prefetch_checkpoint": sys.argv[6].lower() == "true",
    "pushed": sys.argv[7].lower() == "true",
    "dockerfile": "deploy/docker/robot_eval_worker/unitree_groot_sonic_wam/Dockerfile",
    "blockers": [] if inspect_result.returncode == 0 else ["unitree_groot_sonic_wam_image_manifest_inspection_failed"],
    "raw_secret_values_recorded": False,
    "claim_boundary": {
        "image_build_manifest_is_not_provider_startup": True,
        "image_build_manifest_is_not_policy_inference": True,
        "image_build_manifest_is_not_task_success": True,
    },
}
if inspect_result.returncode == 0:
    try:
        manifest = json.loads(inspect_result.stdout)
    except Exception:
        manifest = {}
    payload["registry_manifest_media_type"] = manifest.get("mediaType")
    payload["registry_manifest_raw_available"] = bool(manifest)
else:
    payload["stderr_tail"] = inspect_result.stderr[-1000:]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
