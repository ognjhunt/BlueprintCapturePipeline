#!/bin/bash
# Enqueue one sealed grant. The installed .path activates the allocator-backed
# service; this transient has no network and reads no provider credentials.
set -euo pipefail
umask 022
. "$(dirname "$0")/door-common.sh"
door_init_request public-embodiment-preparation
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_EMBODIMENT_PREPARATION_REQUEST:?}" "${DOOR_SERVICE_USER:?}"
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail embodiment_preparation_user_invalid
cd -P "$DOOR_CONTROL_PLANE_REPO"
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.public-embodiment-preparation.json"
set +e
(
  umask 077
  setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all -- env PYTHONPATH=src \
    GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD" \
    "$DOOR_VENV_PYTHON" -m blueprint_pipeline.deployment_embodiment_preparation_controller \
    enqueue --request-json "$DOOR_EMBODIMENT_PREPARATION_REQUEST" > "$result.tmp"
)
rc=$?
set -e
chgrp blueprint-door "$result.tmp"; chmod 0640 "$result.tmp"; mv "$result.tmp" "$result"
if [ "$rc" -eq 0 ]; then
  door_outcome completed "" 0 result "$result" disposition grant_queued
else
  door_outcome failed embodiment_preparation_enqueue_failed "$rc" result "$result"
fi
exit "$rc"
