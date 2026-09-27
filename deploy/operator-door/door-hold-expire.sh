#!/bin/bash
# A transient systemd timer runs this after an owned hold's deadline.
set -euo pipefail
source "$(dirname "$0")/door-common.sh"
: "${DOOR_HOLDS_DIR:?}" "${DOOR_HOLD_UNIT:?}" "${DOOR_HOLD_REQUEST_ID:?}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(dirname "$0")" python3 -m operator_door.holds \
  --holds-dir "$DOOR_HOLDS_DIR" --unit "$DOOR_HOLD_UNIT" --request-id "$DOOR_HOLD_REQUEST_ID"
