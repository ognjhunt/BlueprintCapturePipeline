#!/bin/bash
# Preview, or record, one owner confirmation that a published website job's
# success thresholds, exactly as built, are the owner's, with the active
# release. The module runs as the service user that owns the intents; this root
# script keeps only the door's log, result and outcome. It changes no threshold
# and moves no money.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request confirm-owner-success-contract
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_OWNER_SUCCESS_REQUEST:?}" \
  "${DOOR_SERVICE_USER:?}" "${DOOR_REQUESTED_BY:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail owner_success_user_invalid
[[ "$DOOR_REQUESTED_BY" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || door_fail owner_success_caller_invalid

# Read only expected environment names as KEY=VALUE data; never evaluate the files.
for env_file in /etc/blueprint/pipeline-control-plane.env \
  /etc/blueprint/task-evaluation-scene-progression.env; do
  if [ -r "$env_file" ]; then
    assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
    double_quoted='^"(.*)"$'; single_quoted="^'(.*)'\$"
    while IFS= read -r line || [ -n "$line" ]; do
      [[ "$line" =~ $assignment ]] || continue
      key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
      case "$key" in BLUEPRINT_* | PIPELINE_*) ;; *) continue ;; esac
      if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then value="${BASH_REMATCH[1]}"; fi
      export "$key=$value"
    done <"$env_file"
  fi
done
: "${BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT:=/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents}"
export BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.owner-success.json"
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.task_evaluation_owner_success_confirmation_door \
  --request-json "$DOOR_OWNER_SUCCESS_REQUEST" --door-caller "$DOOR_REQUESTED_BY" --door-request-id "$DOOR_REQUEST_ID")"
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
outcome = row.get("door_outcome")
outcome = outcome if outcome in {"planned", "confirmed", "already_confirmed", "refused"} else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200]
print(outcome, code)
PY
)"
outcome="${verdict%% *}"
code="${verdict#* }"
if [ "$rc" -eq 0 ] && { [ "$outcome" = planned ] || [ "$outcome" = confirmed ] || [ "$outcome" = already_confirmed ]; }; then
  door_outcome "$outcome" "" 0 result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
case "$outcome" in
  refused) door_outcome refused "${code:-owner_success_refused}" "$rc" result "$result" ;;
  *) door_outcome failed owner_success_failed "$rc" result "$result" ;;
esac
exit "$rc"
