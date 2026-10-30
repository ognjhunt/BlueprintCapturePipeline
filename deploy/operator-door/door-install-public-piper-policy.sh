#!/bin/bash
# Build as blueprint, then use only independently installed stdlib code for CAS.
set -euo pipefail
umask 022
. "$(dirname "$0")/door-common.sh"
operation="${DOOR_PUBLIC_POLICY_OPERATION:-install-public-piper-policy}"
case "$operation" in
  install-public-piper-policy|install-public-yam-policy|install-public-franka-policy) ;;
  *) echo "unsupported fixed public policy operation" >&2; exit 2 ;;
esac
door_init_request "$operation"
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_PUBLIC_POLICY_REQUEST:?}" "${DOOR_SERVICE_USER:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail public_policy_service_user_invalid
door_code="$(cd "$(dirname "$0")" && pwd)"
cd -P "$DOOR_CONTROL_PLANE_REPO"
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.$operation.json"
set +e
(
  umask 077
  setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all -- env PYTHONPATH=src \
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD" \
    "$DOOR_VENV_PYTHON" -m blueprint_pipeline.openwam_piper_registration \
    --request-json "$DOOR_PUBLIC_POLICY_REQUEST" > "$result.prepare.tmp"
)
rc=$?
if [ "$rc" -eq 0 ]; then
  python3 -I -S "$door_code/operator_door/public_policy_registration.py" \
    --request-json "$DOOR_PUBLIC_POLICY_REQUEST" > "$result.tmp"
  rc=$?
else
  cp "$result.prepare.tmp" "$result.tmp"
fi
set -e
chgrp blueprint-door "$result.tmp" "$result.prepare.tmp"
chmod 0640 "$result.tmp" "$result.prepare.tmp"
mv "$result.tmp" "$result"
if [ "$rc" -eq 0 ]; then
  door_outcome completed "" 0 result "$result" disposition public_policy_registered
else
  door_outcome failed public_policy_registration_failed "$rc" result "$result"
fi
exit "$rc"
