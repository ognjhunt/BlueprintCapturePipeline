#!/bin/bash
# One exact original envelope; no subscription/timer/grant changes.
set -euo pipefail
umask 077
. "$(dirname "$0")/door-common.sh"
door_init_request selected-handoff
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_SELECTED_HANDOFF_REQUEST:?}" "${DOOR_SERVICE_USER:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail selected_handoff_user_invalid
cd "$DOOR_CONTROL_PLANE_REPO"
# Read existing service configuration as data, never as shell instructions.
for env_file in /etc/blueprint/pipeline-control-plane.env \
  /etc/blueprint/task-evaluation-scene-progression.env \
  /etc/blueprint/task-evaluation-scene-configuration-release.env; do
if [ -r "$env_file" ]; then
  assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  double_quoted='^"(.*)"$'; single_quoted="^'(.*)'\$"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ $assignment ]] || continue
    key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
    case "$key" in BLUEPRINT_* | GOOGLE_* | GCLOUD_PROJECT | PIPELINE_* | PRIVACY_* | RETRIEVAL_* | OPENAI_* | GEMINI_*) ;; *) continue ;; esac
    if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then value="${BASH_REMATCH[1]}"; fi
    export "$key=$value"
  done <"$env_file"
fi
done
export BLUEPRINT_PIPELINE_REPO="$PWD"
# Existing native dispatch keeps its production environment fence. Inspection
# does not activate workers or run the guard's unrelated reconciliation.
mode="$(python3 -c 'import json,os; print(json.loads(os.environ["DOOR_SELECTED_HANDOFF_REQUEST"])["mode"])')"
if [ "$mode" = dispatch ]; then
  setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all -- env PYTHONPATH=src \
    "$DOOR_VENV_PYTHON" -m blueprint_pipeline.production_runtime_env_guard > /dev/null \
    || door_fail selected_handoff_runtime_admission_failed
fi
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.selected-handoff.json"
set +e
setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all -- env PYTHONPATH=src \
  "$DOOR_VENV_PYTHON" -m blueprint_pipeline.selected_handoff_recovery \
  --request-json "$DOOR_SELECTED_HANDOFF_REQUEST" > "$result.tmp"
rc=$?
set -e
chgrp blueprint-door "$result.tmp"; chmod 0640 "$result.tmp"; mv "$result.tmp" "$result"
if [ "$rc" -eq 0 ]; then door_outcome completed "" 0 result "$result"; else door_outcome failed selected_handoff_blocked "$rc" result "$result"; fi
exit "$rc"
