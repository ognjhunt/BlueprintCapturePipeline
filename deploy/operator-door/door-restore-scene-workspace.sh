#!/bin/bash
# Replay a verified scene retirement receipt to its canonical workspace path.
set -euo pipefail
umask 022
. "$(dirname "$0")/door-common.sh"

door_init_request restore-scene-workspace
: "${DOOR_SCENE_ID:?}" "${DOOR_BUCKET:?}" "${DOOR_VENV_PYTHON:?}"
if ! [[ "$DOOR_SCENE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || [[ "$DOOR_SCENE_ID" == . || "$DOOR_SCENE_ID" == .. ]]; then
  door_fail scene_id_invalid
fi
if ! [[ "$DOOR_BUCKET" =~ ^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$ ]] || [[ "$DOOR_BUCKET" == *..* ]]; then
  door_fail bucket_invalid
fi

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

storage="${BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT:-/var/lib/blueprint/pubsub-handoffs}"
receipt="$storage/$DOOR_BUCKET/scenes/$DOOR_SCENE_ID.retired.v1.json"
destination="$storage/$DOOR_BUCKET/scenes/$DOOR_SCENE_ID"
cd -P "${DOOR_CONTROL_PLANE_REPO:-/opt/blueprint/task-evaluation-control-plane}" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.restore.json"
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.website_scene_workspace_retention restore \
  --receipt "$receipt" --destination "$destination" --result-out "$result"
rc=$?
set -e
status="$(python3 - "$result" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        document = json.load(stream)
except (OSError, ValueError):
    document = {}
print(document.get("status") if isinstance(document, dict) else "")
PY
)"
if [ "$rc" -eq 0 ] && [ "$status" = restored ]; then
  door_outcome restored "" 0 scene_id "$DOOR_SCENE_ID" result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
door_outcome failed restore_failed "$rc" scene_id "$DOOR_SCENE_ID" result "$result"
exit "$rc"
