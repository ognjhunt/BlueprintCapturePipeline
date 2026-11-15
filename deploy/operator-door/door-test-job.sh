#!/bin/bash
# Start one internal test job through the website's signed internal route, with the
# active release (blueprint_pipeline.internal_test_job_door) as the service user.
# This root script keeps only the door's log, result and outcome. No cap, grant or
# provider is an input; the job's spend passes every paid admission.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request test-job
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_TEST_JOB_REQUEST:?}" "${DOOR_SERVICE_USER:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail test_job_user_invalid

# Read only expected environment names as KEY=VALUE data; never evaluate the file.
env_file=/etc/blueprint/pipeline-control-plane.env
if [ -r "$env_file" ]; then
  assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  double_quoted='^"(.*)"$'; single_quoted="^'(.*)'\$"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ $assignment ]] || continue
    key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
    # Only what the call and the month read need: the website URL, the token's file, the month's limit config.
    case "$key" in PIPELINE_SYNC_WEBAPP_URL | PIPELINE_SYNC_TOKEN_FILE | BLUEPRINT_ACCOUNT_MONTHLY_SPEND_* \
      | BLUEPRINT_LAUNCH_PROOF_MODE) ;; *) continue ;; esac
    if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then value="${BASH_REMATCH[1]}"; fi
    export "$key=$value"
  done <"$env_file"
fi

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.test-job.json"
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.internal_test_job_door \
  --request-json "$DOOR_TEST_JOB_REQUEST" --door-request-id "$DOOR_REQUEST_ID")"
rc=$?
set -e
printf '%s\n' "$summary" >"$result.tmp"
chmod 0644 "$result.tmp"
mv -f "$result.tmp" "$result"
case "$rc" in
  0) door_outcome submitted "" 0 result "$result" ;;
  2) door_outcome refused test_job_refused 2 result "$result" ;;
  *) door_outcome failed test_job_failed "$rc" result "$result" ;;
esac
exit "$rc"
