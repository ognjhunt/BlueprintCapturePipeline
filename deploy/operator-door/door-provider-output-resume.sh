#!/bin/bash
# Resume one streamed policy-canary attempt's provider output with the active release:
# promotion to B2, the gated cleanup, the staged-object absence proof and, with
# DOOR_INGEST=1, the needed members' ingestion. The attempt tree belongs to the
# service user, so the module runs as that user with a private umask (review I5);
# this root script keeps only the door's log, result and outcome.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request provider-output-resume
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_CANARY_ROOT:?}" "${DOOR_RUN:?}" \
  "${DOOR_ATTEMPT:?}" "${DOOR_SERVICE_USER:?}"
[[ "$DOOR_CANARY_ROOT" = /* ]] || door_fail provider_output_resume_root_invalid
if ! [[ "$DOOR_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || [[ "$DOOR_RUN" == . || "$DOOR_RUN" == .. ]]; then
  door_fail provider_output_resume_run_invalid
fi
[[ "$DOOR_ATTEMPT" =~ ^[1-9][0-9]{0,2}$ ]] || door_fail provider_output_resume_attempt_invalid
case "${DOOR_INGEST:-}" in "" | 1) ;; *) door_fail provider_output_resume_ingest_invalid ;; esac
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail provider_output_resume_user_invalid
attempt="$DOOR_CANARY_ROOT/$DOOR_RUN/allocator/attempts/$(printf 'attempt_%03d' "$DOOR_ATTEMPT")"
if [ -L "$attempt" ] || [ ! -d "$attempt" ]; then
  door_fail provider_output_resume_attempt_missing
fi
# Every component must resolve inside the canary root: no linked run or attempts directory.
canary_root="$(cd -P "$DOOR_CANARY_ROOT" && pwd)" || door_fail provider_output_resume_root_invalid
attempt="$(cd -P "$attempt" && pwd)" || door_fail provider_output_resume_attempt_missing
[[ "$attempt" == "$canary_root"/* ]] || door_fail provider_output_resume_attempt_outside_root

# Read only expected environment names as KEY=VALUE data; never evaluate the file.
env_file="${DOOR_CONTROL_PLANE_ENV_FILE:-/etc/blueprint/pipeline-control-plane.env}"
if [ -r "$env_file" ]; then
  assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  double_quoted='^"(.*)"$'
  single_quoted="^'(.*)'\$"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ $assignment ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "$key" in BLUEPRINT_* | GOOGLE_* | GCLOUD_PROJECT) ;; *) continue ;; esac
    if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    export "$key=$value"
  done <"$env_file"
fi
# The dedicated B2 store, as the canary dispatcher binds it: promotion refuses without it.
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_key_id}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_application_key}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_bucket}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_s3_endpoint_url}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_region}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET:=blueprint-task-evaluation-artifacts-prod}"
: "${BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT:=/var/lib/blueprint/pipeline-control-plane/disk-reservations}"
export BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET \
  BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.provider-output-resume.json"
args=(resume --attempt-root "$attempt")
if [ "${DOOR_INGEST:-}" = 1 ]; then args+=(--ingest); fi
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.provider_output_promotion "${args[@]}")"
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
status = row.get("status") if row.get("status") in {"completed", "blocked"} else "failed"
blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(blockers[0]))[:200] if blockers else ""
print(status, code)
PY
)"
status="${verdict%% *}"
code="${verdict#* }"
if [ "$rc" -eq 0 ] && [ "$status" = completed ]; then
  door_outcome completed "" 0 result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
door_outcome "$status" "${code:-provider_output_resume_failed}" "$rc" result "$result"
exit "$rc"
