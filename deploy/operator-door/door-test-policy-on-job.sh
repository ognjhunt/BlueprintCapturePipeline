#!/bin/bash
# Plan and enqueue one limited screening campaign (named candidates on one stored job) with the active
# release, as the service user that owns the screening queue. It only enqueues: it reserves, launches and
# pays for nothing, and every money check stays where the coordinator and the launch keep it.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request test-policy-on-job
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_JOB_TEST_REQUEST:?}" "${DOOR_SERVICE_USER:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail job_test_user_invalid

# Read only expected environment names as KEY=VALUE data; never evaluate the files.
for env_file in /etc/blueprint/pipeline-control-plane.env /etc/blueprint/beta-policy-screening.env; do
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

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.job-test.json"
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD" \
  "$DOOR_VENV_PYTHON" -m blueprint_pipeline.beta_policy_screening_on_job \
  --request-json "$DOOR_JOB_TEST_REQUEST" --door-request-id "$DOOR_REQUEST_ID")"
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
outcome = outcome if outcome in {"enqueued", "refused"} else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200]
print(outcome, code)
PY
)"
outcome="${verdict%% *}"
code="${verdict#* }"
if [ "$rc" -eq 0 ] && [ "$outcome" = enqueued ]; then
  door_outcome enqueued "" 0 result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
case "$outcome" in
  refused) door_outcome refused "${code:-job_test_refused}" "$rc" result "$result" ;;
  *) door_outcome failed job_test_failed "$rc" result "$result" ;;
esac
exit "$rc"
