#!/bin/bash
# Observe protected consent with the active release; write public report metadata only.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"
door_init_request owner-census-decision
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_CONFIG_PATH:?}" \
  "${DOOR_CONSENT_ID:?}" "${DOOR_CONSENT_SHA256:?}" "${DOOR_CONSENT_SIZE_BYTES:?}"
[[ "$DOOR_CONSENT_ID" =~ ^[0-9a-f]{32}$ ]] || door_fail owner_consent_options_invalid
[[ "$DOOR_CONSENT_SHA256" =~ ^sha256:[0-9a-f]{64}$ ]] || door_fail owner_consent_options_invalid
[[ "$DOOR_CONSENT_SIZE_BYTES" =~ ^[0-9]{1,6}$ ]] || door_fail owner_consent_options_invalid
(( 10#$DOOR_CONSENT_SIZE_BYTES >= 1 && 10#$DOOR_CONSENT_SIZE_BYTES <= 524288 )) || door_fail owner_consent_options_invalid
[ "$DOOR_CONFIG_PATH" = /etc/blueprint-operator-door/door.json ] || door_fail owner_consent_options_invalid
cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail owner_consent_installed_bridge_invalid
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.control_plane_lane_owner_consents report \
  --consent-id "$DOOR_CONSENT_ID" --expected-sha256 "$DOOR_CONSENT_SHA256" \
  --expected-size-bytes "$DOOR_CONSENT_SIZE_BYTES" --door-config "$DOOR_CONFIG_PATH" \
  --results-dir "$DOOR_RESULTS_DIR" --request-id "$DOOR_REQUEST_ID"
rc=$?
set -e
if [ "$rc" -eq 0 ] && [ -f "$DOOR_OUTCOME" ] && [ ! -L "$DOOR_OUTCOME" ]; then
  DOOR_OUTCOME_WRITTEN=1
  exit 0
fi
door_outcome refused owner_consent_report_refused 1
exit 1
