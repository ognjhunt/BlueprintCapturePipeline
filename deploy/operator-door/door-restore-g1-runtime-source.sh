#!/bin/bash
# Plan, or with DOOR_APPLY=1 perform, the active release's restore of the shared G1
# runtime source packet and its receipt from the retained object-store copy
# (native_g1_runtime_source_restore). Every path and digest is pinned in that
# module; the request carries only apply. The restored files belong to the service
# user, so the module runs as that user with a private umask; this root script
# keeps only the door's log, result and outcome. It starts no compute.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request restore-g1-runtime-source
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_SERVICE_USER:?}"
case "${DOOR_APPLY:-}" in "" | 1) ;; *) door_fail g1_runtime_source_apply_invalid ;; esac
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail g1_runtime_source_user_invalid

# Read only BLUEPRINT_* names as KEY=VALUE data; never evaluate the file.
env_file="${DOOR_CONTROL_PLANE_ENV_FILE:-/etc/blueprint/pipeline-control-plane.env}"
if [ -r "$env_file" ]; then
  assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  double_quoted='^"(.*)"$'
  single_quoted="^'(.*)'\$"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ $assignment ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "$key" in BLUEPRINT_*) ;; *) continue ;; esac
    if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    export "$key=$value"
  done <"$env_file"
fi
# The WAM provider object store, as the launch dispatcher binds it.
: "${BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE:=/etc/blueprint/provider-secrets/digitalocean_spaces_access_key_id}"
: "${BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE:=/etc/blueprint/provider-secrets/digitalocean_spaces_secret_access_key}"
: "${BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE:=/etc/blueprint/provider-secrets/digitalocean_spaces_endpoint_url}"
: "${BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE:=/etc/blueprint/provider-secrets/digitalocean_spaces_bucket}"
: "${BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE:=/etc/blueprint/provider-secrets/digitalocean_spaces_region}"
: "${BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT:=/var/lib/blueprint/pipeline-control-plane/disk-reservations}"
export BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE \
  BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE \
  BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE \
  BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE \
  BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE \
  BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
# An active release older than the module cannot run it: say so instead of failing opaquely.
[ -f src/blueprint_pipeline/native_g1_runtime_source_restore.py ] || door_fail g1_runtime_source_release_lacks_module
# The service user cannot create a directory at the root-owned top of the work mount, so on apply this
# root script creates the one pinned target directory, owned by the service user (0750). It never follows
# a link, and an existing directory is left to the module, which locks and verifies it.
target_dir="${DOOR_G1_RUNTIME_SOURCE_DIR:-/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926}"
if [ "${DOOR_APPLY:-}" = 1 ]; then
  if [ -L "$target_dir" ]; then door_fail g1_runtime_source_target_conflict; fi
  if [ ! -e "$target_dir" ]; then
    install -d -o "$DOOR_SERVICE_USER" -g "$DOOR_SERVICE_USER" -m 0750 "$target_dir" \
      || door_fail g1_runtime_source_target_dir_unavailable
  fi
  if [ ! -d "$target_dir" ] || [ -L "$target_dir" ]; then door_fail g1_runtime_source_target_conflict; fi
fi
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.g1-runtime-source.json"
args=()
if [ "${DOOR_APPLY:-}" = 1 ]; then args+=(--apply); fi
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.native_g1_runtime_source_restore \
  ${args[@]+"${args[@]}"})"
rc=$?
set -e
printf '%s\n' "$summary" >"$result.tmp"
chmod 0644 "$result.tmp"
mv -f "$result.tmp" "$result"
verdict="$(python3 - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        row = json.load(stream)
except (OSError, ValueError):
    row = {}
row = row if isinstance(row, dict) else {}
outcome = row.get("outcome")
outcome = outcome if outcome in {"planned", "restored", "already_restored", "refused"} else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200] or "-"
source = row.get("runtime_source") if isinstance(row.get("runtime_source"), dict) else {}
sha = source.get("sha256")
sha = sha if isinstance(sha, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", sha) else "-"
size = source.get("size_bytes")
size = str(size) if type(size) is int and size > 0 else "-"
print(outcome, code, sha, size)
PY
)"
read -r outcome code receipt_sha256 receipt_size_bytes <<<"$verdict"
[ "$code" != - ] || code=""
if [ "$rc" -eq 0 ] && { [ "$outcome" = planned ] || [ "$outcome" = restored ] || [ "$outcome" = already_restored ]; }; then
  if [ "$receipt_sha256" != - ] && [ "$receipt_size_bytes" != - ]; then
    # The install request's runtime_source reference; the result has it with an integer size.
    door_outcome "$outcome" "" 0 result "$result" receipt_sha256 "$receipt_sha256" \
      receipt_size_bytes "$receipt_size_bytes"
  else
    door_outcome "$outcome" "" 0 result "$result"
  fi
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
if [ "$outcome" = refused ]; then
  door_outcome refused "${code:-g1_runtime_source_restore_failed}" "$rc" result "$result"
else
  door_outcome failed g1_runtime_source_restore_failed "$rc" result "$result"
fi
exit "$rc"
