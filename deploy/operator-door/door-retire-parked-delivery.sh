#!/bin/bash
# Plan, or with the request's apply perform, the active release's owner-approved retirement of
# one parked website preparation delivery (website_preparation_delivery_retirement). The module
# re-validates the door's request, holds the capture's ledger lock, and moves that delivery's
# record to its terminal "retired" state with the owner's approval on it; the delivery index
# drops its hint. It runs as the service user that owns the records, with no network; this root
# script keeps only the door's log, result and outcome. Nothing is deleted.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request retire-parked-delivery
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_PARKED_DELIVERY_REQUEST:?}" \
  "${DOOR_SERVICE_USER:?}" "${DOOR_REQUESTED_BY:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail parked_delivery_user_invalid
[[ "$DOOR_REQUESTED_BY" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || door_fail parked_delivery_caller_invalid

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
# The handoff storage the listener and the delivery index use.
: "${BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT:=/var/lib/blueprint/pubsub-handoffs}"
export BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
# An active release older than the module cannot run it: say so instead of failing opaquely.
[ -f src/blueprint_pipeline/website_preparation_delivery_retirement.py ] || door_fail parked_delivery_release_lacks_module
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.parked-delivery.json"
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.website_preparation_delivery_retirement \
  --request-json "$DOOR_PARKED_DELIVERY_REQUEST" --door-caller "$DOOR_REQUESTED_BY" --door-request-id "$DOOR_REQUEST_ID")"
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
outcome = outcome if outcome in {"planned", "retired", "already_retired", "refused"} else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200] or "-"
print(outcome, code)
PY
)"
read -r outcome code <<<"$verdict"
[ "$code" != - ] || code=""
if [ "$rc" -eq 0 ] && { [ "$outcome" = planned ] || [ "$outcome" = retired ] || [ "$outcome" = already_retired ]; }; then
  door_outcome "$outcome" "" 0 result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
if [ "$outcome" = refused ]; then
  door_outcome refused "${code:-parked_delivery_retirement_failed}" "$rc" result "$result"
else
  door_outcome failed parked_delivery_retirement_failed "$rc" result "$result"
fi
exit "$rc"
