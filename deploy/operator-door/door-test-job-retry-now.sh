#!/bin/bash
# Bring one internal test job's next attempt forward to now, with the active release
# (blueprint_pipeline.internal_test_job_retry_now) as the service user that owns the records, with no network.
# The module re-validates the door's request, finds the job in the door's own test-job records, and sets only when
# its scheduled Atlas preview and preparation retries are due, each under the lock its tick or listener takes, at
# most once per release. This root script keeps only the door's log, result and outcome. No count, wall, grant,
# spend or admission changes.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request test-job-retry-now
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_TEST_JOB_RETRY_NOW_REQUEST:?}" \
  "${DOOR_SERVICE_USER:?}" "${DOOR_REQUESTED_BY:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail test_job_retry_now_user_invalid
[[ "$DOOR_REQUESTED_BY" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || door_fail test_job_retry_now_caller_invalid

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
# An active release older than the module cannot run it: say so instead of failing opaquely.
[ -f src/blueprint_pipeline/internal_test_job_retry_now.py ] || door_fail test_job_retry_now_release_lacks_module
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.test-job-retry-now.json"
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.internal_test_job_retry_now \
  --request-json "$DOOR_TEST_JOB_RETRY_NOW_REQUEST" --door-request-id "$DOOR_REQUEST_ID" \
  --requested-by "$DOOR_REQUESTED_BY" --door-results "$DOOR_RESULTS_DIR")"
rc=$?
set -e
printf '%s\n' "$summary" >"$result.tmp"
chmod 0644 "$result.tmp"
mv -f "$result.tmp" "$result"
# The module's own refusal code, for the outcome; the result file keeps the whole answer.
code="$(python3 - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        row = json.load(stream)
except (OSError, ValueError):
    row = {}
print(re.sub(r"[^A-Za-z0-9_:.-]", "_", str((row if isinstance(row, dict) else {}).get("code") or ""))[:200])
PY
)"
case "$rc" in
  0) door_outcome brought_forward "" 0 result "$result" ;;
  3) door_outcome unchanged "" 0 result "$result"; rc=0 ;;
  2) door_outcome refused "${code:-test_job_retry_now_refused}" 2 result "$result" ;;
  *) door_outcome failed "${code:-test_job_retry_now_failed}" "$rc" result "$result" ;;
esac
exit "$rc"
