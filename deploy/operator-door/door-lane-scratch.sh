#!/bin/bash
# Manage one sealed scratch lease through the active control-plane release.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request lane-scratch
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_SCRATCH_ROOT:?}" \
  "${DOOR_SCRATCH_ACTION:?}" "${DOOR_SCRATCH_LANE:?}"
[[ "$DOOR_SCRATCH_ROOT" = /* ]] || door_fail lane_scratch_root_invalid
[[ "$DOOR_SCRATCH_LANE" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || door_fail lane_scratch_lane_invalid
case "$DOOR_SCRATCH_ACTION" in
  ls)
    [[ "${DOOR_SCRATCH_LIMIT:-}" =~ ^[0-9]{1,3}$ ]] || door_fail lane_scratch_limit_invalid
    [[ "${DOOR_SCRATCH_OFFSET:-}" =~ ^[0-9]{1,5}$ ]] || door_fail lane_scratch_offset_invalid
    (( DOOR_SCRATCH_LIMIT >= 1 && DOOR_SCRATCH_LIMIT <= 100 )) || door_fail lane_scratch_limit_invalid
    (( DOOR_SCRATCH_OFFSET <= 10000 )) || door_fail lane_scratch_offset_invalid
    ;;
  renew | release)
    [[ "${DOOR_SCRATCH_NAME:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || door_fail lane_scratch_name_invalid
    [[ "${DOOR_SCRATCH_OWNER:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || door_fail lane_scratch_owner_invalid
    [[ "${DOOR_SCRATCH_EXPECTED_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]] || door_fail lane_scratch_digest_invalid
    if [ "$DOOR_SCRATCH_ACTION" = renew ]; then
      [[ "${DOOR_SCRATCH_TTL_SECONDS:-}" =~ ^[0-9]{1,7}$ ]] || door_fail lane_scratch_ttl_invalid
      (( DOOR_SCRATCH_TTL_SECONDS >= 1 && DOOR_SCRATCH_TTL_SECONDS <= 1209600 )) || door_fail lane_scratch_ttl_invalid
    fi
    ;;
  *) door_fail lane_scratch_action_invalid ;;
esac

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.lane-scratch.json"
args=("$DOOR_SCRATCH_ACTION" --root "$DOOR_SCRATCH_ROOT" --lane "$DOOR_SCRATCH_LANE" --result-out "$result")
if [ "$DOOR_SCRATCH_ACTION" = ls ]; then
  args+=(--limit "$DOOR_SCRATCH_LIMIT" --offset "$DOOR_SCRATCH_OFFSET")
else
  args+=(--name "$DOOR_SCRATCH_NAME" --owner "$DOOR_SCRATCH_OWNER" \
    --expected-digest "$DOOR_SCRATCH_EXPECTED_DIGEST")
  if [ "$DOOR_SCRATCH_ACTION" = renew ]; then args+=(--ttl-seconds "$DOOR_SCRATCH_TTL_SECONDS"); fi
fi
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.control_plane_lane_scratch_door "${args[@]}"
rc=$?
set -e
status="$(python3 - "$result" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        row = json.load(stream)
except (OSError, ValueError):
    row = {}
print(row.get("status") if row.get("status") in {"listed", "renewed", "released", "failed"} else "failed")
PY
)"
if [ "$rc" -eq 0 ] && [ "$status" != failed ]; then
  door_outcome "$status" "" 0 result "$result"
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
door_outcome failed lane_scratch_command_failed "$rc" result "$result"
exit "$rc"
