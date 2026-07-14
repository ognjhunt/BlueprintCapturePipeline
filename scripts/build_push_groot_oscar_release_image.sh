#!/usr/bin/env bash
# Thin, frequently pulled Blueprint release built on an immutable foundation.
set -euo pipefail

echo "legacy build path disabled; use paid_resource_allocator cpu-build" >&2
exit 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
foundation="${BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF:-}"
image_ref="${BLUEPRINT_GROOT_OSCAR_RELEASE_IMAGE_REF:-${BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF:-}}"
allow_push="${BLUEPRINT_ALLOW_GROOT_OSCAR_RELEASE_IMAGE_PUSH:-false}"
manifest="${BLUEPRINT_GROOT_OSCAR_RELEASE_MANIFEST_OUTPUT:-$repo_root/output/groot_oscar_release_image_manifest.json}"
max_release_bytes="${BLUEPRINT_GROOT_OSCAR_RELEASE_MAX_COMPRESSED_BYTES:-2147483648}"

die() { echo "$1" >&2; exit 2; }
[[ "$foundation" =~ @sha256:[0-9a-f]{64}$ ]] || die "foundation image must be digest pinned"
[[ -n "$image_ref" ]] || die "missing BLUEPRINT_GROOT_OSCAR_RELEASE_IMAGE_REF"
[[ "${image_ref##*/}" == *:* || "$image_ref" == *@sha256:* ]] || die "release image ref must be versioned"
case "${image_ref##*/}" in *:latest|*:local|*:dev|*:test) die "release image refuses unstable tag";; esac
git -C "$repo_root" diff --quiet && git -C "$repo_root" diff --cached --quiet \
  || die "thin release build requires a clean source worktree"
[[ -z "$(git -C "$repo_root" ls-files --others --exclude-standard)" ]] \
  || die "thin release build refuses untracked source"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
clean_patch_sha="$(python3 -c 'import hashlib;print(hashlib.sha256(b"").hexdigest())')"
metadata="$(mktemp "${TMPDIR:-/tmp}/blueprint-release-metadata.XXXXXX")"
trap 'rm -f "$metadata"' EXIT

args=(docker buildx build --platform linux/amd64 --progress plain --metadata-file "$metadata"
  --build-arg "FOUNDATION_IMAGE=$foundation"
  --build-arg "BLUEPRINT_SOURCE_COMMIT=$source_commit"
  --build-arg "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256=$clean_patch_sha"
  -f "$repo_root/deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile"
  -t "$image_ref")
if [[ "$allow_push" == true ]]; then args+=(--push); else args+=(--load); fi
args+=("$repo_root")
"${args[@]}"

resolved_ref="$image_ref"
if [[ "$allow_push" == true ]]; then
  digest="$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("containerimage.digest") or p.get("containerimage.descriptor",{}).get("digest") or "")' "$metadata")"
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die "release build digest missing"
  resolved_ref="$(python3 -c 'import sys; ref=sys.argv[1].split("@",1)[0]; leaf=ref.rsplit("/",1)[-1]; print((ref.rsplit(":",1)[0] if ":" in leaf else ref)+"@"+sys.argv[2])' "$image_ref" "$digest")"
  diagnostic="$(mktemp "${TMPDIR:-/tmp}/blueprint-release-diagnostic.XXXXXX")"
  foundation_diagnostic="$(mktemp "${TMPDIR:-/tmp}/blueprint-foundation-diagnostic.XXXXXX")"
  thin_contract="$(mktemp "${TMPDIR:-/tmp}/blueprint-thin-contract.XXXXXX")"
  trap 'rm -f "$metadata" "$diagnostic" "$foundation_diagnostic" "$thin_contract"' EXIT
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m blueprint_pipeline.isaac_worker_image_manifest \
      --image "$resolved_ref" --output "$diagnostic"
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m blueprint_pipeline.isaac_worker_image_manifest \
      --image "$foundation" --output "$foundation_diagnostic"
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 - "$diagnostic" "$foundation_diagnostic" "$thin_contract" "$max_release_bytes" <<'PY'
import json, sys
from pathlib import Path
from blueprint_pipeline.thin_release_image_contract import build_thin_release_contract
release=json.load(open(sys.argv[1])); foundation=json.load(open(sys.argv[2]))
result=build_thin_release_contract(release,foundation,max_release_bytes=int(sys.argv[4]))
Path(sys.argv[3]).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if result["status"] == "passed" else 2)
PY
  total="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["release_delta_compressed_size_bytes"])' "$thin_contract")"
  [[ "$total" -le "$max_release_bytes" ]] || die "thin release exceeds compressed budget: $total > $max_release_bytes"
else
  total=""
fi

python3 - "$manifest" "$image_ref" "$resolved_ref" "$foundation" "$source_commit" "$total" "$max_release_bytes" "${diagnostic:-}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out=Path(sys.argv[1]); out.parent.mkdir(parents=True,exist_ok=True)
total=int(sys.argv[6]) if sys.argv[6] else None
diagnostic=json.load(open(sys.argv[8])) if sys.argv[8] else {}
payload={
 "schema_version":"groot_oscar_thin_release_build.v1",
 "generated_at":datetime.now(timezone.utc).isoformat(), "status":"completed",
 "image_ref":sys.argv[2], "resolved_image_ref":sys.argv[3],
 "foundation_image_ref":sys.argv[4], "source_commit":sys.argv[5],
 "runnable_platform":"linux/amd64",
 "required_cuda_version":diagnostic.get("required_cuda_version"),
 "required_cuda_version_source":diagnostic.get("required_cuda_version_source"),
 "models_embedded":False, "model_cache_mount":"/models/blueprint-groot-oscar-v1",
 "total_compressed_size_bytes":total, "release_budget_bytes":int(sys.argv[7]),
 "release_budget_passed":None if total is None else total <= int(sys.argv[7]),
 "claim_boundary":{"image_build_is_not_model_cache_verification":True,
                    "image_build_is_not_warm_worker_readiness":True,
                    "image_build_is_not_task_success":True},
}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(out)
PY
