#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/artifact_storage.sh"
artifact_cache_root="$(blueprint_artifact_cache_root)"
base_image="${BLUEPRINT_ISAAC_WORKER_OVERLAY_BASE_IMAGE_DIGEST:-docker.io/nijelhunt/blueprint-isaac-eval-worker@sha256:865633c0bb99058ce15dd1d977876bf931b73f6b04057c94f5235dc701ea0a91}"
target_image="${BLUEPRINT_ISAAC_WORKER_OVERLAY_TARGET_IMAGE_REF:-}"
output_dir="${BLUEPRINT_ISAAC_WORKER_OVERLAY_OUTPUT_DIR:-$artifact_cache_root/isaac_worker_source_overlay}"
crane_bin="${BLUEPRINT_CRANE_BIN:-crane}"
allow_push="${BLUEPRINT_ALLOW_ISAAC_WORKER_IMAGE_PUSH:-false}"
prepare_only="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image) base_image="$2"; shift 2 ;;
    --target-image) target_image="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --prepare-only) prepare_only="true"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$target_image" ]]; then
  echo "missing --target-image or BLUEPRINT_ISAAC_WORKER_OVERLAY_TARGET_IMAGE_REF" >&2
  exit 2
fi
if ! command -v "$crane_bin" >/dev/null 2>&1 && [[ ! -x "$crane_bin" ]]; then
  echo "crane not found: $crane_bin" >&2
  exit 2
fi

mkdir -p "$output_dir"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m blueprint_pipeline.isaac_worker_source_overlay prepare \
    --repo-root "$repo_root" \
    --output-dir "$output_dir" \
    --base-image "$base_image" \
    --target-image "$target_image" \
    --source-commit "$source_commit" \
    > "$output_dir/prepare.stdout.json"

if [[ "$prepare_only" == "true" ]]; then
  echo "prepared exact-source overlay without registry mutation: $output_dir"
  exit 0
fi
if [[ "$allow_push" != "true" ]]; then
  echo "registry mutation refused; set BLUEPRINT_ALLOW_ISAAC_WORKER_IMAGE_PUSH=true" >&2
  exit 2
fi

username_file="${BLUEPRINT_DOCKER_USERNAME_FILE:-$HOME/.blueprint-secrets/docker_username}"
password_file="${BLUEPRINT_DOCKER_PASSWORD_FILE:-$HOME/.blueprint-secrets/docker_pat}"
test -f "$username_file"
test -f "$password_file"
crane_config_dir="$(mktemp -d)"
export DOCKER_CONFIG="$crane_config_dir"

cleanup() {
  "$crane_bin" auth logout docker.io >/dev/null 2>&1 || true
  find "$crane_config_dir" -type f -delete >/dev/null 2>&1 || true
  find "$crane_config_dir" -depth -type d -empty -delete >/dev/null 2>&1 || true
}
trap cleanup EXIT
"$crane_bin" auth login docker.io -u "$(cat "$username_file")" --password-stdin < "$password_file"

plan="$output_dir/isaac_worker_source_overlay.v1.json"
layer="$output_dir/isaac_worker_source_overlay.tar.gz"
source_manifest="$(python3 - "$plan" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_manifest_sha256"])
PY
)"
clean_patch="$(python3 - "$plan" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_dirty_patch_sha256"])
PY
)"

"$crane_bin" manifest "$base_image" > "$output_dir/base_manifest.json"
"$crane_bin" mutate "$base_image" \
  --append "$layer" \
  --env "BLUEPRINT_SOURCE_COMMIT=$source_commit" \
  --env "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=$clean_patch" \
  --env "BLUEPRINT_WORKER_IMAGE_BUILD_METHOD=crane_exact_source_overlay_experimental" \
  --env "BLUEPRINT_WORKER_BASE_IMAGE_DIGEST=$base_image" \
  --env "BLUEPRINT_WORKER_SOURCE_MANIFEST_SHA256=$source_manifest" \
  --label "org.opencontainers.image.revision=$source_commit" \
  --label "io.blueprint.base-image=$base_image" \
  --label "io.blueprint.source-manifest=$source_manifest" \
  --tag "$target_image"

stage_digest="$("$crane_bin" digest "$target_image")"
repository="${target_image%:*}"
stage_digest_ref="$repository@$stage_digest"
"$crane_bin" manifest "$stage_digest_ref" > "$output_dir/stage_manifest.json"
source_layer_digest="$(python3 - "$output_dir/stage_manifest.json" <<'PY'
import json,re,sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
layers=payload.get("layers") or []
value=str((layers[-1] if layers else {}).get("digest") or "")
print(value)
raise SystemExit(0 if re.fullmatch(r"sha256:[0-9a-f]{64}", value) else 2)
PY
)"

"$crane_bin" mutate "$stage_digest_ref" \
  --env "BLUEPRINT_WORKER_SOURCE_LAYER_DIGEST=$source_layer_digest" \
  --tag "$target_image"
final_digest="$("$crane_bin" digest "$target_image")"
final_digest_ref="$repository@$final_digest"
"$crane_bin" manifest "$final_digest_ref" > "$output_dir/final_manifest.json"
"$crane_bin" config "$final_digest_ref" > "$output_dir/final_config.json"

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m blueprint_pipeline.isaac_worker_source_overlay verify-registry \
    --plan "$plan" \
    --base-manifest "$output_dir/base_manifest.json" \
    --final-manifest "$output_dir/final_manifest.json" \
    --final-config "$output_dir/final_config.json" \
    --resolved-digest "$final_digest" \
    --output "$output_dir/isaac_worker_source_overlay_registry_result.v1.json" \
    > "$output_dir/verify.stdout.json"

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m blueprint_pipeline.isaac_worker_source_overlay write-image-manifest \
    --image-ref "$final_digest_ref" \
    --final-manifest "$output_dir/final_manifest.json" \
    --final-config "$output_dir/final_config.json" \
    --resolved-digest "$final_digest" \
    --output "$output_dir/isaac_worker_image_manifest_diagnostic.json" \
    > "$output_dir/image_manifest.stdout.json"
printf '%s\n' "$final_digest_ref" > "$output_dir/resolved_image_digest.txt"
echo "pushed and verified exact-source Isaac worker overlay: $final_digest_ref"
