#!/bin/bash
# Fixed report-only historical owner census. No packet, approval, or action mode.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"
door_init_request legacy-owner-census
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_CONFIG_PATH:?}"
[ "$DOOR_CONFIG_PATH" = /etc/blueprint-operator-door/door.json ] || door_fail legacy_owner_options_invalid
cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail legacy_owner_installed_bridge_invalid
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" \
  -m blueprint_pipeline.control_plane_lane_legacy_owner_door report \
  --door-config "$DOOR_CONFIG_PATH" --results-dir "$DOOR_RESULTS_DIR" \
  --request-id "$DOOR_REQUEST_ID"
rc=$?
set -e
if [ "$rc" -eq 0 ] && [ -f "$DOOR_OUTCOME" ] && [ ! -L "$DOOR_OUTCOME" ]; then
  DOOR_OUTCOME_WRITTEN=1
  exit 0
fi
door_outcome refused legacy_owner_census_incomplete 1
exit 1
