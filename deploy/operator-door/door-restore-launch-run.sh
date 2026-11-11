#!/bin/bash
# Restore one stored job's offloaded configured-scene launch run with the active release, after reserving
# its disk peak in the shared disk ledger. It deletes nothing, starts no compute and records a receipt.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request restore-launch-run
: "${DOOR_SITE_ID:?}" "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}"
[[ "$DOOR_SITE_ID" =~ ^site-[A-Za-z0-9][A-Za-z0-9._-]{0,122}$ ]] || door_fail site_id_invalid

# Read only expected environment names as KEY=VALUE data; never evaluate the files.
for env_file in /etc/blueprint/pipeline-control-plane.env /etc/blueprint/beta-policy-screening.env; do
  if [ -r "$env_file" ]; then
    assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
    double_quoted='^"(.*)"$'; single_quoted="^'(.*)'\$"
    while IFS= read -r line || [ -n "$line" ]; do
      [[ "$line" =~ $assignment ]] || continue
      key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
      case "$key" in BLUEPRINT_* | GOOGLE_* | GCLOUD_PROJECT) ;; *) continue ;; esac
      if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then value="${BASH_REMATCH[1]}"; fi
      export "$key=$value"
    done <"$env_file"
  fi
done
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_key_id}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_application_key}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_bucket}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_s3_endpoint_url}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE:=/etc/blueprint/provider-secrets/backblaze_b2_region}"
: "${BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET:=blueprint-task-evaluation-artifacts-prod}"
export BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE \
  BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.launch-run-restore.json"
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.screening_launch_run_restore \
  --site-id "$DOOR_SITE_ID" --door-request-id "$DOOR_REQUEST_ID" --result-out "$result"
rc=$?
set -e
verdict="$(python3 - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        document = json.load(stream)
except (OSError, ValueError):
    document = {}
document = document if isinstance(document, dict) else {}
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(document.get("reason") or ""))[:200]
print(document.get("status") or "failed", code)
PY
)"
status="${verdict%% *}"
code="${verdict#* }"
if [ "$rc" -eq 0 ] && { [ "$status" = restored ] || [ "$status" = already_on_disk ]; }; then
  door_outcome "$status" "" 0 site_id "$DOOR_SITE_ID" result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
if [ "$status" = refused ]; then
  door_outcome refused "${code:-launch_run_restore_refused}" "$rc" site_id "$DOOR_SITE_ID" result "$result"
else
  door_outcome failed launch_run_restore_failed "$rc" site_id "$DOOR_SITE_ID" result "$result"
fi
exit "$rc"
