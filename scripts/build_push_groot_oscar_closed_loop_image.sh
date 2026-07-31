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

echo "legacy build path disabled; use paid_resource_allocator cpu-build" >&2
exit 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/artifact_storage.sh"
artifact_cache_root="$(blueprint_artifact_cache_root)"
docker_dir="$repo_root/deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
dockerfile="$docker_dir/Dockerfile"
image_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF:-}"
platform="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_PLATFORM:-linux/amd64}"
allow_push="${BLUEPRINT_ALLOW_GROOT_OSCAR_CLOSED_LOOP_IMAGE_PUSH:-false}"
base_image="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d}"
groot_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_GROOT_SOURCE_REF:-e5749287857afd97b78f1147166137de29746392}"
wbc_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_WBC_SOURCE_REF:-6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b}"
gear_checkpoint_revision="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_GEAR_CHECKPOINT_REVISION:-5e22ddc69abcea2a9aafc40536b14c232d3f9d7f}"
oscar_source_ref="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_OSCAR_SOURCE_REF:-4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb}"
oscar_checkpoint_revision="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_OSCAR_CHECKPOINT_REVISION:-c9781ffa7dd8556d862d7d9f338a2ea008a58ca6}"
prefetch_checkpoints="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_PREFETCH_CHECKPOINTS:-true}"
hf_token_file="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_HF_TOKEN_FILE:-${HF_TOKEN_FILE:-$HOME/.blueprint-secrets/hf_token}}"
manifest_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_MANIFEST_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_image_manifest.json}"
registry_manifest_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_REGISTRY_MANIFEST_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_registry_manifest_diagnostic.json}"
runtime_smoke_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_RUNTIME_SMOKE_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_runtime_smoke.json}"
sbom_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SBOM_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_sbom.spdx.json}"
provenance_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_PROVENANCE_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_provenance.json}"
layer_report_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_LAYER_REPORT_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_layer_report.json}"
buildkit_sbom_attestation_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BUILDKIT_SBOM_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_buildkit_sbom_attestation.json}"
buildkit_provenance_attestation_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BUILDKIT_PROVENANCE_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_buildkit_provenance_attestation.json}"
buildkit_attestation_index_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_BUILDKIT_ATTESTATION_INDEX_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_buildkit_attestation_index.json}"
min_free_gib="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_MIN_FREE_GIB:-120}"
expected_compressed_gib="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_EXPECTED_COMPRESSED_GIB:-46}"
expected_unpacked_gib="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_EXPECTED_UNPACKED_GIB:-176}"
disk_admission_output="${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_DISK_ADMISSION_OUTPUT:-$artifact_cache_root/groot_oscar_closed_loop_disk_admission.json}"
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
local_build_metadata_file=""
publish_build_metadata_file=""
publish_staging_ref=""

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
if [[ "$allow_push" == "true" && "$image_ref" == *@sha256:* ]]; then
  write_manifest "blocked" '["groot_oscar_closed_loop_release_push_requires_final_tag"]' >/dev/null
  echo "release image push requires a versioned final tag, not a digest reference" >&2
  exit 2
fi

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
  docker_root_dir="$(docker info --format '{{.DockerRootDir}}')"
  build_temp_root="${TMPDIR:-/tmp}"
  read -r disk_check_free_kib disk_check_required_kib < <(\
    PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$disk_admission_output" "$expected_compressed_gib" "$expected_unpacked_gib" \
    "$repo_root" "$docker_root_dir" "$build_temp_root" <<'PY'
import json, shutil, sys
from pathlib import Path
from blueprint_pipeline.groot_oscar_release_hardening import DiskAdmission

gib = 1024 ** 3
storage_paths = []
for role, raw_path in zip(
    ("source_and_evidence", "docker_buildkit", "build_and_scan_temp"),
    sys.argv[4:7],
    strict=True,
):
    path = Path(raw_path).expanduser().resolve()
    available = shutil.disk_usage(path).free
    storage_paths.append(
        {"role": role, "path": str(path), "available_bytes": available}
    )
limiting = min(storage_paths, key=lambda item: item["available_bytes"])
evidence = DiskAdmission(
    available_bytes=int(limiting["available_bytes"]),
    image_compressed_bytes=int(float(sys.argv[2]) * gib),
    image_unpacked_bytes=int(float(sys.argv[3]) * gib),
).evidence()
evidence["storage_paths"] = storage_paths
evidence["limiting_storage_role"] = limiting["role"]
evidence["limiting_storage_path"] = limiting["path"]
path = Path(sys.argv[1]).expanduser().resolve()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
print(
    int(limiting["available_bytes"]) // 1024,
    (int(evidence["required_bytes"]) + 1023) // 1024,
)
PY
  )
  legacy_required_kib=$((min_free_gib * 1024 * 1024))
  if [[ "$legacy_required_kib" -gt "$disk_check_required_kib" ]]; then
    disk_check_required_kib="$legacy_required_kib"
  fi
  if [[ "${disk_check_free_kib:-0}" -lt "$disk_check_required_kib" ]]; then
    write_manifest "blocked" '["insufficient_local_disk_for_groot_oscar_closed_loop_image_build"]' >/dev/null
    echo "insufficient local disk for image build plus explicit registry scan; see $disk_admission_output" >&2
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
local_build_metadata_file="$build_context_dir/buildx-local-metadata.json"
publish_build_metadata_file="$build_context_dir/buildx-publish-metadata.json"
build_metadata_file="$local_build_metadata_file"
git -C "$repo_root" archive --format=tar "$source_commit" | tar -xf - -C "$build_context_dir"
dockerfile="$build_context_dir/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
test -f "$dockerfile"

build_args=(
  docker buildx build
  --platform "$platform"
  --progress plain
  --build-arg "ISAAC_SIM_BASE_IMAGE=$base_image"
  --build-arg "GROOT_SOURCE_REF=$groot_ref"
  --build-arg "WBC_SOURCE_REF=$wbc_ref"
  --build-arg "GEAR_SONIC_CHECKPOINT_REVISION=$gear_checkpoint_revision"
  --build-arg "OSCAR_SOURCE_REF=$oscar_source_ref"
  --build-arg "OSCAR_CHECKPOINT_REVISION=$oscar_checkpoint_revision"
  --build-arg "BLUEPRINT_SOURCE_COMMIT=$source_commit"
  --build-arg "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=$source_dirty_patch_sha256"
  --build-arg "PREFETCH_CHECKPOINTS=$prefetch_checkpoints"
  -f "$dockerfile"
)
# Always build into the local content store first. The release tag must not be
# registry-visible until this exact local runtime closure passes the sealed OCI
# smoke below. The later publish export reuses this build cache and is bound
# back to the smoked image-config digest.
local_build_args=(
  "${build_args[@]}"
  -t "$image_ref"
  --metadata-file "$local_build_metadata_file"
  --load
  "${secret_args[@]}"
  "$build_context_dir"
)
"${local_build_args[@]}"

runtime_image_ref="$image_ref"
smoked_local_image_id="$(docker image inspect --format '{{.Id}}' "$runtime_image_ref")"
if [[ ! "$smoked_local_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  write_manifest "blocked" '["groot_oscar_closed_loop_local_image_id_invalid"]' >/dev/null
  echo "local image config digest is invalid; refusing runtime smoke" >&2
  exit 2
fi

runtime_smoke_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
runtime_smoke_stdout="$(mktemp "${TMPDIR:-/tmp}/blueprint-groot-oscar-runtime-smoke.XXXXXX")"
runtime_smoke_stderr="${runtime_smoke_stdout}.stderr"
runtime_smoke_exit=0
docker run --rm --entrypoint /bin/bash "$runtime_image_ref" -lc '
  set -euo pipefail
  test "$(id -un)" = blueprint
  id -nG | tr " " "\n" | grep -Fx isaac-sim
  for interpreter in \
    /isaac-sim/python.sh \
    /opt/oscar-venv/bin/python \
    /opt/gr00t-venv/bin/python; do
    test -x "$interpreter"
    "$interpreter" -c "import os; assert os.geteuid() == 10001"
  done
  /opt/oscar-venv/bin/python \
    /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --build-time
' >"$runtime_smoke_stdout" 2>"$runtime_smoke_stderr" || runtime_smoke_exit=$?

python3 - "$runtime_smoke_output" "$runtime_image_ref" "$runtime_smoke_started_at" \
  "$runtime_smoke_exit" "$runtime_smoke_stdout" "$runtime_smoke_stderr" \
  "$smoked_local_image_id" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1]).expanduser()
stdout = Path(sys.argv[5]).read_bytes()
stderr = Path(sys.argv[6]).read_bytes()
exit_code = int(sys.argv[4])
payload = {
    "schema_version": "groot_oscar_closed_loop_runtime_smoke.v2",
    "started_at": sys.argv[3],
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed" if exit_code == 0 else "failed",
    "image_ref": sys.argv[2],
    "smoked_local_image_id": sys.argv[7],
    "published_digest_ref": None,
    "published_runnable_config_digest": None,
    "published_runtime_identity_matches_smoked_local_image": None,
    "exit_code": exit_code,
    "checks": [
        "oci_configured_user_is_blueprint",
        "oci_runtime_resolves_isaac_sim_supplementary_group",
        "isaac_interpreter_executes_as_uid_10001",
        "oscar_interpreter_executes_as_uid_10001",
        "groot_interpreter_executes_as_uid_10001",
        "sealed_worker_build_time_healthcheck_as_oci_runtime_user",
    ],
    "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
    "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    "stdout_bytes": len(stdout),
    "stderr_bytes": len(stderr),
    "raw_secret_values_recorded": False,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
rm -f "$runtime_smoke_stdout" "$runtime_smoke_stderr"

if [[ "$runtime_smoke_exit" -ne 0 ]]; then
  write_manifest "blocked" '["groot_oscar_closed_loop_oci_runtime_smoke_failed"]' >/dev/null
  echo "finished image failed OCI runtime-user smoke; release tag was not published" >&2
  exit 2
fi

if [[ "$allow_push" == "true" ]]; then
  # Export to a non-release staging tag after runtime smoke. The final release
  # tag is promoted only after the registry's runnable config digest is proven
  # byte-identical to the local image that passed the sealed OCI smoke.
  publish_staging_ref="${image_ref}-candidate-${source_commit:0:12}"
  publish_build_args=(
    "${build_args[@]}"
    -t "$publish_staging_ref"
    --metadata-file "$publish_build_metadata_file"
    --push
    --attest type=sbom
    --provenance mode=max
    "${secret_args[@]}"
    "$build_context_dir"
  )
  "${publish_build_args[@]}"
  build_metadata_file="$publish_build_metadata_file"
  build_digest="$(python3 - "$build_metadata_file" <<'PY'
import json, re, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
digest = str(payload.get("containerimage.digest") or "")
if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    descriptor = payload.get("containerimage.descriptor")
    digest = str(descriptor.get("digest") or "") if isinstance(descriptor, dict) else ""
if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise SystemExit("buildx metadata did not contain an immutable image digest")
print(digest)
PY
)"
  runtime_image_ref="${publish_staging_ref%%@*}"
  runtime_image_ref="${runtime_image_ref%:*}@${build_digest}"
  published_config_digest="$(python3 - "$runtime_image_ref" <<'PY'
import json, re, subprocess, sys

exact_ref = sys.argv[1]
repository = exact_ref.rsplit("@", 1)[0]

def inspect_raw(ref):
    completed = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", ref],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return json.loads(completed.stdout)

manifest = inspect_raw(exact_ref)
if isinstance(manifest.get("manifests"), list):
    candidates = [
        item.get("digest")
        for item in manifest["manifests"]
        if isinstance(item, dict)
        and isinstance(item.get("platform"), dict)
        and item["platform"].get("os") == "linux"
        and item["platform"].get("architecture") == "amd64"
    ]
    if len(candidates) != 1:
        raise SystemExit("published index does not have exactly one linux/amd64 runtime")
    manifest = inspect_raw(f"{repository}@{candidates[0]}")
config = manifest.get("config")
digest = str(config.get("digest") or "") if isinstance(config, dict) else ""
if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise SystemExit("published runnable config digest missing")
print(digest)
PY
)"
  if [[ "$published_config_digest" != "$smoked_local_image_id" ]]; then
    write_manifest "blocked" '["published_runtime_identity_differs_from_smoked_local_image"]' >/dev/null
    echo "published runtime config does not match the smoke-tested local image" >&2
    exit 2
  fi
  python3 - "$runtime_smoke_output" "$runtime_image_ref" \
    "$published_config_digest" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
payload["published_digest_ref"] = sys.argv[2]
payload["published_runnable_config_digest"] = sys.argv[3]
payload["published_runtime_identity_matches_smoked_local_image"] = (
    payload.get("smoked_local_image_id") == sys.argv[3]
)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

registry_diagnostic_exit=0
if [[ "$allow_push" == "true" ]]; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m blueprint_pipeline.isaac_worker_image_manifest \
      --image "$runtime_image_ref" --output "$registry_manifest_output" \
    || registry_diagnostic_exit=$?
fi

supply_chain_exit=0
if [[ "$allow_push" == "true" && "$registry_diagnostic_exit" -eq 0 ]]; then
  exact_digest_ref="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["resolved_digest_ref"])' "$registry_manifest_output")"
  docker buildx imagetools inspect --format '{{json .SBOM}}' "$exact_digest_ref" \
    > "$buildkit_sbom_attestation_output" || supply_chain_exit=$?
  docker buildx imagetools inspect --format '{{json .Provenance}}' "$exact_digest_ref" \
    > "$buildkit_provenance_attestation_output" || supply_chain_exit=$?
  docker buildx imagetools inspect --raw "$exact_digest_ref" \
    > "$buildkit_attestation_index_output" || supply_chain_exit=$?
  # The explicit registry: source is a safety property. Never let Syft infer
  # the Docker daemon and export a second copy of this very large image.
  syft "registry:${exact_digest_ref}" -o "spdx-json=${sbom_output}" || supply_chain_exit=$?
  if [[ "$supply_chain_exit" -eq 0 ]]; then
    PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 - \
      "$sbom_output" "$provenance_output" "$layer_report_output" \
      "$registry_manifest_output" "$build_metadata_file" "$exact_digest_ref" \
      "$buildkit_sbom_attestation_output" "$buildkit_provenance_attestation_output" \
      "$buildkit_attestation_index_output" <<'PY' \
      || supply_chain_exit=$?
import json, sys
from pathlib import Path
from blueprint_pipeline.groot_oscar_release_hardening import (
    build_layer_report,
    validate_buildkit_provenance_binding,
    validate_spdx_document,
)

sbom_path, provenance_path, layer_path, registry_path, metadata_path = map(
    lambda value: Path(value).expanduser().resolve(), sys.argv[1:6]
)
digest_ref = sys.argv[6]
buildkit_sbom_path = Path(sys.argv[7]).expanduser().resolve()
buildkit_provenance_path = Path(sys.argv[8]).expanduser().resolve()
buildkit_index_path = Path(sys.argv[9]).expanduser().resolve()
expected_digest = digest_ref.rsplit("@", 1)[-1]
sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
registry = json.loads(registry_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
blockers = validate_spdx_document(sbom)
attestations = {}
for label, path in (("sbom", buildkit_sbom_path), ("provenance", buildkit_provenance_path)):
    try:
        attestation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        attestation = None
    if not attestation:
        blockers.append(f"buildkit_{label}_attestation_missing")
    attestations[label] = attestation
try:
    attestation_index = json.loads(buildkit_index_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    attestation_index = None
if not isinstance(attestation_index, dict):
    blockers.append("buildkit_attestation_index_missing")
else:
    blockers.extend(
        validate_buildkit_provenance_binding(
            attestations.get("provenance") or {},
            attestation_index,
            str(registry.get("runnable_child_digest") or ""),
        )
    )
provenance = {
    "schema_version": "groot_oscar_buildkit_provenance.v1",
    "status": "passed",
    "subject_digest": expected_digest,
    "subject_digest_ref": digest_ref,
    "buildkit_provenance_attestation": {
        "enabled": True,
        "mode": "max",
        "predicate_validated": not any(
            blocker.startswith("buildkit_provenance_") for blocker in blockers
        ),
        "oci_subject_binding_validated": (
            "buildkit_attestation_subject_binding_missing" not in blockers
        ),
    },
    "buildkit_sbom_attestation": {"enabled": True},
    "buildx_metadata": metadata,
    "raw_secret_values_recorded": False,
}
provenance["blockers"] = sorted(set(blockers))
provenance["status"] = "passed" if not blockers else "blocked"
provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
history = registry.get("layer_history") or []
created_by = {
    str(row.get("digest") or ""): str(row.get("created_by") or "")
    for row in history if isinstance(row, dict)
}
layers = [
    {
        "digest": row.get("digest"),
        "size_bytes": row.get("size"),
        "created_by": created_by.get(str(row.get("digest") or ""), ""),
    }
    for row in registry.get("layers") or [] if isinstance(row, dict)
]
layer_report = build_layer_report(layers)
layer_path.write_text(json.dumps(layer_report, indent=2, sort_keys=True) + "\n")
blockers.extend(layer_report["blockers"])
if blockers:
    raise SystemExit(2)
PY
  fi
fi

source_identity_after_json="$(
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
    'import json, sys; from blueprint_pipeline.g1_kitchen_bundle_compatibility import build_source_tree_identity; print(json.dumps(build_source_tree_identity(sys.argv[1]), sort_keys=True))' \
    "$repo_root"
)"

# Only now is the release tag allowed to become visible. The immutable staging
# digest has passed local runtime identity binding, registry inspection,
# BuildKit SBOM/provenance validation, explicit registry-source Syft scanning,
# and layer admission.
if [[ "$allow_push" == "true" && "$registry_diagnostic_exit" -eq 0 && "$supply_chain_exit" -eq 0 && "$source_identity_after_json" == "$source_identity_json" ]]; then
  if ! docker buildx imagetools create --tag "$image_ref" "$runtime_image_ref"; then
    write_manifest "blocked" '["groot_oscar_closed_loop_final_tag_promotion_failed"]' >/dev/null
    echo "validated candidate digest could not be promoted to the final release tag" >&2
    exit 2
  fi
  promoted_digest="$(docker buildx imagetools inspect --format '{{json .}}' "$image_ref" | python3 -c '
import json, re, sys
payload = json.load(sys.stdin)
manifest = (
    payload.get("manifest") or payload.get("Manifest")
    if isinstance(payload, dict)
    else None
)
digest = str(manifest.get("digest") or "") if isinstance(manifest, dict) else ""
if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise SystemExit("promoted release tag digest missing")
print(digest)
')"
  if [[ "$promoted_digest" != "$build_digest" ]]; then
    write_manifest "blocked" '["groot_oscar_closed_loop_final_tag_digest_mismatch"]' >/dev/null
    echo "final release tag does not resolve to the validated candidate digest" >&2
    exit 2
  fi
fi

python3 - "$manifest_output" "$image_ref" "$platform" "$base_image" "$groot_ref" "$prefetch_checkpoints" "$allow_push" "$source_identity_json" "$source_identity_gate_json" "$registry_manifest_output" "$registry_diagnostic_exit" "$build_metadata_file" "$source_identity_after_json" "$runtime_smoke_output" "$sbom_output" "$provenance_output" "$layer_report_output" "$supply_chain_exit" "$wbc_ref" "$gear_checkpoint_revision" "$oscar_source_ref" "$oscar_checkpoint_revision" <<'PY'
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
runtime_smoke_path = Path(sys.argv[14]).expanduser().resolve()
sbom_path = Path(sys.argv[15]).expanduser().resolve()
provenance_path = Path(sys.argv[16]).expanduser().resolve()
layer_report_path = Path(sys.argv[17]).expanduser().resolve()
supply_chain_exit = int(sys.argv[18])
wbc_ref = sys.argv[19]
gear_checkpoint_revision = sys.argv[20]
oscar_source_ref = sys.argv[21]
oscar_checkpoint_revision = sys.argv[22]
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
try:
    runtime_smoke_bytes = runtime_smoke_path.read_bytes()
    runtime_smoke = json.loads(runtime_smoke_bytes.decode("utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    runtime_smoke_bytes = b""
    runtime_smoke = {}
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
if pushed and supply_chain_exit != 0:
    blockers.append("groot_oscar_closed_loop_supply_chain_evidence_failed")
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
    "wbc_source_ref": wbc_ref,
    "gear_sonic_checkpoint_revision": gear_checkpoint_revision,
    "oscar_source_ref": oscar_source_ref,
    "oscar_checkpoint_revision": oscar_checkpoint_revision,
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
    "oci_runtime_smoke": {
        "path": str(runtime_smoke_path),
        "sha256": hashlib.sha256(runtime_smoke_bytes).hexdigest()
        if runtime_smoke_bytes else None,
        "bytes": len(runtime_smoke_bytes),
        "status": runtime_smoke.get("status"),
        "checks": runtime_smoke.get("checks", []),
    },
    "buildkit_attestations": {
        "sbom_enabled": pushed,
        "provenance_enabled": pushed,
        "provenance_mode": "max" if pushed else None,
    },
    "registry_sbom": {"path": str(sbom_path), "source": "registry_digest"},
    "provenance": {"path": str(provenance_path)},
    "layer_report": {"path": str(layer_report_path)},
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
