#!/bin/bash
set -euo pipefail
umask 077
. "$(dirname "$0")/door-common.sh"
door_init_request unit
: "${DOOR_INSTALL_ROOT:?}" "${DOOR_EXPECTED_POSTCHECK_SHA256:?}" "${DOOR_EXPECTED_SOURCE_COMMIT:?}"
[[ "$DOOR_EXPECTED_POSTCHECK_SHA256" =~ ^sha256:[0-9a-f]{64}$ ]] || door_fail expected_identity_invalid
[[ "$DOOR_EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || door_fail expected_identity_invalid
receipt="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.notifier-binding.json"
set +e
PYTHONPATH="$DOOR_INSTALL_ROOT" python3 -m operator_door.notifier_repair \
  --expected-sha256 "$DOOR_EXPECTED_POSTCHECK_SHA256" \
  --expected-commit "$DOOR_EXPECTED_SOURCE_COMMIT" --receipt-out "$receipt"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  door_outcome repaired "" "$rc" receipt "$receipt"
else
  door_outcome refused notifier_binding_not_repaired "$rc"
fi
exit "$rc"
