#!/bin/bash
# Plan or apply one website scene workspace retirement (operator door, transient root unit).
#
# Runs the active release's own blueprint_pipeline.website_scene_workspace_retention for one
# scene; no new code is fetched. Without DOOR_APPLY it only plans ("planned", or "retained" with
# the first reason). With it, the module re-proves every check under the listener's ledger locks,
# archives what Firebase Storage does not hold, writes a replayable receipt and only then removes
# the workspace, or keeps it and says why. This script maps the module's status to the outcome.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request retire-scene-workspace
: "${DOOR_SCENE_ID:?}" "${DOOR_VENV_PYTHON:?}"
# The runner validated these already; check them again before they reach a command line.
if ! [[ "$DOOR_SCENE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  door_fail scene_id_invalid
fi
if [ -n "${DOOR_BUCKET:-}" ] && { ! [[ "$DOOR_BUCKET" =~ ^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$ ]] || [[ "$DOOR_BUCKET" == *..* ]]; }; then
  door_fail bucket_invalid
fi
case "${DOOR_APPLY:-}" in
  "" | 1) ;;
  *) door_fail apply_invalid ;;
esac
echo "[$(date -u +%FT%TZ)] retire-scene-workspace ${DOOR_SCENE_ID} apply=${DOOR_APPLY:-0} (request ${DOOR_REQUEST_ID})"

# The control-plane environment (Firebase Storage credentials among it), never echoed. It is
# systemd's KEY=VALUE format, not shell, so it is read as data: one KEY=VALUE per line, matching
# surrounding quotes stripped, nothing expanded or run. Only the module's own settings
# (BLUEPRINT_*, GOOGLE_*, GCLOUD_PROJECT) are exported, so the file cannot steer this script.
env_file="${DOOR_CONTROL_PLANE_ENV_FILE:-/etc/blueprint/pipeline-control-plane.env}"
if [ -r "$env_file" ]; then
  assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  double_quoted='^"(.*)"$'
  single_quoted="^'(.*)'\$"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ $assignment ]] || continue  # comments, blank and indented lines never match
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "$key" in
      BLUEPRINT_* | GOOGLE_* | GCLOUD_PROJECT) ;;
      *) continue ;;
    esac
    if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    export "$key=$value"
  done <"$env_file"
fi
# Archive to the same private artifact store the reclaim timer uses (its unit sets these), so a
# receipt restores the same way whichever path retired the scene.
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

cd -P "${DOOR_CONTROL_PLANE_REPO:-/opt/blueprint/task-evaluation-control-plane}" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.retirement.json"
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.website_scene_workspace_retention retire \
  --scene-id "$DOOR_SCENE_ID" ${DOOR_BUCKET:+--bucket "$DOOR_BUCKET"} \
  ${DOOR_APPLY:+--apply --ack retire-scene-workspace} \
  --result-out "$result"
rc=$?
set -e

# "<status> <code>": planned | retained <first reason> | retired | failed <code>. The code is
# reduced to a typed token, whatever a reason names.
outcome="$(python3 - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        document = json.load(stream)
except (OSError, ValueError):
    document = None
document = document if isinstance(document, dict) else {}
status = document.get("status")
if status not in ("planned", "retained", "retired", "failed"):
    status = "failed"
code = ""
if status == "retained":
    reasons = document.get("reasons")
    code = str(reasons[0]) if isinstance(reasons, list) and reasons else "retained"
elif status == "failed":
    code = str(document.get("code") or "")
print(status + " " + re.sub(r"[^A-Za-z0-9._:/@+=-]", "_", code)[:200])
PY
)"
status="${outcome%% *}"
code="${outcome#* }"
case "$status" in
  planned | retained | retired)
    door_outcome "$status" "$code" 0 scene_id "$DOOR_SCENE_ID" result "$result"
    exit 0
    ;;
esac
[ -n "$code" ] || code="retention_exit_$rc"
[ "$rc" -ne 0 ] || rc=1
door_outcome failed "$code" "$rc" scene_id "$DOOR_SCENE_ID" result "$result"
exit "$rc"
