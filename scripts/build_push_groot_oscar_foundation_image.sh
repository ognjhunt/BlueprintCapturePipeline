#!/usr/bin/env bash
# Slow-changing Isaac/CUDA/robot-runtime foundation.  Contains no checkpoints.
set -euo pipefail

[[ "${BLUEPRINT_CANONICAL_CPU_BUILD_CONTEXT:-false}" == true ]] || {
  echo "legacy build path disabled; use paid_resource_allocator cpu-build" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_ref="${BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF:-}"
allow_push="${BLUEPRINT_ALLOW_GROOT_OSCAR_FOUNDATION_IMAGE_PUSH:-false}"
manifest="${BLUEPRINT_GROOT_OSCAR_FOUNDATION_MANIFEST_OUTPUT:-$repo_root/output/groot_oscar_foundation_image_manifest.json}"
[[ -n "$image_ref" ]] || { echo "missing BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF" >&2; exit 2; }
[[ "${image_ref##*/}" == *:* ]] || { echo "foundation image ref must be versioned" >&2; exit 2; }
[[ "${image_ref##*:}" != latest ]] || { echo "foundation image refuses latest" >&2; exit 2; }
git -C "$repo_root" diff --quiet && git -C "$repo_root" diff --cached --quiet \
  || { echo "foundation build requires a clean source worktree" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" ls-files --others --exclude-standard)" ]] \
  || { echo "foundation build refuses untracked source" >&2; exit 2; }
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
metadata="$(mktemp "${TMPDIR:-/tmp}/blueprint-foundation-metadata.XXXXXX")"
diagnostic="$(mktemp "${TMPDIR:-/tmp}/blueprint-foundation-diagnostic.XXXXXX")"
trap 'rm -f "$metadata" "$diagnostic"' EXIT

args=(docker buildx build --platform linux/amd64 --progress plain --metadata-file "$metadata"
  -f "$repo_root/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile"
  -t "$image_ref")
if [[ "$allow_push" == true ]]; then args+=(--push); else args+=(--load); fi
args+=("$repo_root")
"${args[@]}"

if [[ "$allow_push" == true ]]; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m blueprint_pipeline.isaac_worker_image_manifest \
      --image "$image_ref" --output "$diagnostic"
fi
python3 - "$manifest" "$image_ref" "$source_commit" "$allow_push" "$diagnostic" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
diagnostic={}
try: diagnostic=json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
except Exception: pass
payload={
 "schema_version":"groot_oscar_foundation_build.v1",
 "generated_at":datetime.now(timezone.utc).isoformat(),
 "status":"completed" if sys.argv[4] == "true" else "local_build_completed",
 "image_ref":sys.argv[2], "source_commit":sys.argv[3],
 "resolved_image_ref":diagnostic.get("resolved_digest_ref"),
 "total_compressed_size_bytes":diagnostic.get("total_compressed_size_bytes"),
 "largest_layer_size_bytes":diagnostic.get("largest_layer_size_bytes"),
 "models_embedded":False, "blueprint_release_code_embedded":False,
 "runtime_environment":"shared_oscar_groot_robot_venv",
 "wbc_runtime_only":True,
 "claim_boundary":{"foundation_build_is_not_model_cache_verification":True,
                    "foundation_build_is_not_provider_startup":True},
}
out=Path(sys.argv[1]); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(out)
PY
