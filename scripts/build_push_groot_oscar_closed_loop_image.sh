#!/usr/bin/env bash
# Build + optionally push the sealed blueprint-groot-oscar-eval worker image
# (GR00T N1.7 + SONIC x OSCAR-2B closed-loop lane). Reproducible clean-build path.
# Strict G1 kitchen images must use this pinned Isaac-Sim Dockerfile build.
# Historical crane snapshots of the OSCAR-only carrier are not eligible.
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
base_image="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d}"
groot_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_GROOT_SOURCE_REF:-e5749287857afd97b78f1147166137de29746392}"
prefetch_checkpoints="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_PREFETCH_CHECKPOINTS:-true}"
hf_token_file="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_HF_TOKEN_FILE:-${HF_TOKEN_FILE:-$HOME/.blueprint-secrets/hf_token}}"
manifest_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_MANIFEST_OUTPUT:-$repo_root/output/groot_oscar_closed_loop_image_manifest.json}"
registry_manifest_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_REGISTRY_MANIFEST_OUTPUT:-$repo_root/output/groot_oscar_closed_loop_registry_manifest_diagnostic.json}"
min_free_gib="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_MIN_FREE_GIB:-120}"
allow_dirty_release_build="${BLUEPRINT_ALLOW_DIRTY_GROOT_OSCAR_CLOSED_LOOP_RELEASE_BUILD:-false}"
source_identity_json="$(
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
    'import json, sys; from blueprint_pipeline.g1_kitchen_bundle_compatibility import build_source_tree_identity; print(json.dumps(build_source_tree_identity(sys.argv[1]), sort_keys=True))' \
    "$repo_root"
)"
source_identity_gate_json="$(
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
    'import json, sys; from blueprint_pipeline.g1_kitchen_bundle_compatibility import evaluate_release_image_source_identity; print(json.dumps(evaluate_release_image_source_identity(json.loads(sys.argv[1]), push_requested=sys.argv[2].lower() == "true", allow_dirty_release_build=sys.argv[3].lower() == "true"), sort_keys=True))' \
    "$source_identity_json" "$allow_push" "$allow_dirty_release_build"
)"
source_commit="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["source_commit"])' "$source_identity_json")"
source_dirty_patch_sha256="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["source_dirty_patch_sha256"])' "$source_identity_json")"
disk_check_free_kib=""
disk_check_required_kib=""
build_context_dir=""
build_metadata_file=""

cleanup_build_context() {
  if [[ -n "$build_context_dir" && -d "$build_context_dir" ]]; then
    rm -rf "$build_context_dir"
  fi
}
trap cleanup_build_context EXIT

write_manifest() {
  local status="$1" blockers_json="$2"
  python3 - "$manifest_output" "$status" "$blockers_json" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoints" "$min_free_gib" "${disk_check_free_kib:-}" "${disk_check_required_kib:-}" "$source_identity_json" "$source_identity_gate_json" <<'PY'
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
identity = json.loads(sys.argv[12])
identity_gate = json.loads(sys.argv[13])
payload = {
    "schema_version": "groot_oscar_closed_loop_image_build_manifest.v2",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "blockers": json.loads(sys.argv[3]),
    "image_ref": sys.argv[4] or None,
    "platform": sys.argv[5],
    "base_image": sys.argv[6],
    "groot_source_ref": sys.argv[7],
    "source_commit": identity["source_commit"],
    "source_dirty_patch_sha256": identity["source_dirty_patch_sha256"],
    "source_worktree_dirty": bool(identity["dirty"]),
    "untracked_file_count": identity["untracked_file_count"],
    "identity_includes_staged_unstaged_and_untracked": bool(
        identity["identity_includes_staged_unstaged_and_untracked"]
    ),
    "canonical_clean_patch_sha256": identity["canonical_clean_patch_sha256"],
    "source_identity_release_gate": identity_gate,
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
image_name="${image_ref##*/}"
if [[ "$image_ref" != *@sha256:* ]]; then
  # A colon in the registry authority (for example registry.example:5000)
  # is not an image tag. Require a non-empty tag on the final path component.
  if [[ "$image_name" != *:* || -z "${image_name##*:}" ]]; then
    write_manifest "blocked" '["groot_oscar_closed_loop_image_ref_must_be_versioned"]' >/dev/null
    echo "image ref must be versioned: $image_ref" >&2
    exit 2
  fi
fi
case "$image_name" in
  *:latest|*:local|*:dev|*:test)
    write_manifest "blocked" '["groot_oscar_closed_loop_image_ref_refuses_unstable_tag"]' >/dev/null
    echo "refuses non-launch image tag: $image_ref" >&2
    exit 2
    ;;
esac

# FABLE-008 release evidence must bind the Isaac base by immutable digest.
# Keep mutable base overrides available for local --load investigation, but
# never publish a release image whose recorded base can move after the build.
if [[ "$allow_push" == "true" && ! "$base_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  write_manifest "blocked" '["groot_oscar_closed_loop_base_image_must_be_digest_pinned"]' >/dev/null
  echo "release image push requires a digest-pinned base image: $base_image" >&2
  exit 2
fi

# The immutable git-archive context below contains only committed bytes. Refuse
# dirty local, staged, or untracked inputs even for --load/debug builds; otherwise
# the recorded patch hash would describe bytes that were not placed in the image.
if [[ "$(python3 -c 'import json,sys;print(str(bool(json.loads(sys.argv[1])["dirty"])).lower())' "$source_identity_json")" == "true" ]]; then
  write_manifest "blocked" '["groot_oscar_closed_loop_image_build_requires_clean_source_worktree"]' >/dev/null
  echo "groot+oscar image builds require a clean source worktree" >&2
  exit 2
fi

if [[ "$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["status"])' "$source_identity_gate_json")" != "passed" ]]; then
  write_manifest "blocked" "$(python3 -c 'import json,sys;print(json.dumps(json.loads(sys.argv[1])["blockers"]))' "$source_identity_gate_json")" >/dev/null
  echo "release image push requires a clean source worktree (see source_identity_release_gate in the manifest)" >&2
  exit 2
fi

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

# A release build consumes an immutable archive of the clean source commit,
# not the mutable worktree that happened to pass the identity check earlier.
build_context_dir="$(mktemp -d "${TMPDIR:-/tmp}/blueprint-groot-oscar-context.XXXXXX")"
build_metadata_file="$build_context_dir/buildx-metadata.json"
git -C "$repo_root" archive --format=tar "$source_commit" | tar -xf - -C "$build_context_dir"
dockerfile="$build_context_dir/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
test -f "$dockerfile"

build_args=(
  docker buildx build
  --platform "$platform"
  --progress plain
  --metadata-file "$build_metadata_file"
  --build-arg "ISAAC_SIM_BASE_IMAGE=$base_image"
  --build-arg "GROOT_SOURCE_REF=$groot_ref"
  --build-arg "BLUEPRINT_SOURCE_COMMIT=$source_commit"
  --build-arg "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=$source_dirty_patch_sha256"
  --build-arg "PREFETCH_CHECKPOINTS=$prefetch_checkpoints"
  -f "$dockerfile"
  -t "$image_ref"
)
if [[ "$allow_push" == "true" ]]; then
  build_args+=(--push)
else
  build_args+=(--load)
fi
build_args+=("${secret_args[@]}" "$build_context_dir")

"${build_args[@]}"

registry_diagnostic_exit=0
if [[ "$allow_push" == "true" ]]; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m blueprint_pipeline.isaac_worker_image_manifest \
      --image "$image_ref" --output "$registry_manifest_output" \
    || registry_diagnostic_exit=$?
fi
source_identity_after_json="$(
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
    'import json, sys; from blueprint_pipeline.g1_kitchen_bundle_compatibility import build_source_tree_identity; print(json.dumps(build_source_tree_identity(sys.argv[1]), sort_keys=True))' \
    "$repo_root"
)"

python3 - "$manifest_output" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoints" "$allow_push" "$source_identity_json" "$source_identity_gate_json" "$registry_manifest_output" "$registry_diagnostic_exit" "$build_metadata_file" "$source_identity_after_json" <<'PY'
import hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1]).expanduser()
image_ref = sys.argv[2]
identity = json.loads(sys.argv[8])
identity_gate = json.loads(sys.argv[9])
registry_path = Path(sys.argv[10]).expanduser().resolve()
registry_diagnostic_exit = int(sys.argv[11])
metadata_path = Path(sys.argv[12]).expanduser().resolve()
identity_after = json.loads(sys.argv[13])
pushed = sys.argv[7].lower() == "true"
try:
    metadata_bytes = metadata_path.read_bytes()
    build_metadata = json.loads(metadata_bytes.decode("utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    metadata_bytes = b""
    build_metadata = {}
build_digest = str(build_metadata.get("containerimage.digest") or "")
if not re.fullmatch(r"sha256:[0-9a-f]{64}", build_digest):
    descriptor = build_metadata.get("containerimage.descriptor")
    descriptor = descriptor if isinstance(descriptor, dict) else {}
    build_digest = str(descriptor.get("digest") or "")
try:
    registry_bytes = registry_path.read_bytes() if pushed else b""
    registry = json.loads(registry_bytes.decode("utf-8")) if pushed else {}
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    registry_bytes = b""
    registry = {}
exact_inspect_ref = str(registry.get("resolved_digest_ref") or "")
inspect = (
    subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", exact_inspect_ref],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if pushed and exact_inspect_ref
    else None
)
registry_complete = bool(
    pushed
    and
    isinstance(registry, dict)
    and registry.get("status") == "completed"
    and registry.get("resolved_digest_ref")
    and registry.get("runnable_platform") == "linux/amd64"
    and isinstance(registry.get("layer_count"), int)
    and registry.get("layer_count", 0) > 0
    and isinstance(registry.get("total_compressed_size_bytes"), int)
    and registry.get("total_compressed_size_bytes", 0) > 0
    and isinstance(registry.get("largest_layer_size_bytes"), int)
    and registry.get("largest_layer_size_bytes", 0) > 0
    and registry.get("history_layer_count_matches") is True
    and registry.get("checkpoint_ownership_copyup_detected") is False
    and registry.get("resolved_digest") == build_digest
)
blockers = []
if identity_after != identity:
    blockers.append("groot_oscar_closed_loop_source_identity_changed_during_build")
if pushed and not re.fullmatch(r"sha256:[0-9a-f]{64}", build_digest):
    blockers.append("groot_oscar_closed_loop_buildx_digest_missing")
if pushed and (inspect is None or inspect.returncode != 0):
    blockers.append("groot_oscar_closed_loop_image_manifest_inspection_failed")
if pushed and not registry_complete:
    blockers.append("groot_oscar_closed_loop_registry_diagnostic_incomplete")
if pushed and registry.get("resolved_digest") != build_digest:
    blockers.append("groot_oscar_closed_loop_buildx_registry_digest_mismatch")
if pushed and registry_diagnostic_exit != 0:
    blockers.append("groot_oscar_closed_loop_registry_diagnostic_command_failed")
payload = {
    "schema_version": "groot_oscar_closed_loop_image_build_manifest.v2",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": (
        "completed" if pushed and not blockers
        else "local_build_completed" if not pushed and not blockers
        else "built_manifest_inspection_blocked"
    ),
    "image_ref": image_ref,
    "platform": sys.argv[3],
    "base_image": sys.argv[4],
    "groot_source_ref": sys.argv[5],
    "source_commit": identity["source_commit"],
    "source_dirty_patch_sha256": identity["source_dirty_patch_sha256"],
    "source_worktree_dirty": bool(identity["dirty"]),
    "untracked_file_count": identity["untracked_file_count"],
    "identity_includes_staged_unstaged_and_untracked": bool(
        identity["identity_includes_staged_unstaged_and_untracked"]
    ),
    "canonical_clean_patch_sha256": identity["canonical_clean_patch_sha256"],
    "source_identity_release_gate": identity_gate,
    "prefetch_checkpoints": sys.argv[6].lower() == "true",
    "pushed": pushed,
    "dockerfile": "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile",
    "build_context": {
        "mode": "git_archive",
        "source_commit": identity["source_commit"],
        "worktree_identity_unchanged_through_build": identity_after == identity,
    },
    "buildx_metadata": {
        "sha256": hashlib.sha256(metadata_bytes).hexdigest() if metadata_bytes else None,
        "bytes": len(metadata_bytes),
        "containerimage_digest": build_digest or None,
    },
    "resolved_digest": registry.get("resolved_digest"),
    "resolved_digest_ref": registry.get("resolved_digest_ref"),
    "runnable_platform": registry.get("runnable_platform"),
    "runnable_child_digest": registry.get("runnable_child_digest"),
    "layer_count": registry.get("layer_count"),
    "total_compressed_size_bytes": registry.get("total_compressed_size_bytes"),
    "largest_layer_size_bytes": registry.get("largest_layer_size_bytes"),
    "history_layer_count_matches": registry.get("history_layer_count_matches"),
    "checkpoint_ownership_copyup_detected": registry.get(
        "checkpoint_ownership_copyup_detected"
    ),
    "checkpoint_ownership_copyup_layers": registry.get(
        "checkpoint_ownership_copyup_layers"
    ),
    "recommended_startup_no_runtime_timeout_seconds": registry.get(
        "recommended_startup_no_runtime_timeout_seconds"
    ),
    "registry_manifest_diagnostic": {
        "path": str(registry_path),
        "sha256": hashlib.sha256(registry_bytes).hexdigest() if registry_bytes else None,
        "bytes": len(registry_bytes),
        "status": registry.get("status"),
    },
    "build_time_healthcheck": {
        "status": "passed",
        "command": (
            "/opt/oscar-venv/bin/python "
            "/opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time"
        ),
    },
    "blockers": blockers,
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

final_status="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$manifest_output")"
if [[ "$final_status" != "completed" && "$final_status" != "local_build_completed" ]]; then
  echo "groot+oscar image build evidence is blocked; see $manifest_output" >&2
  exit 2
fi
