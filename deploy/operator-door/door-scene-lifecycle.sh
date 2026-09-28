#!/bin/bash
# One protected-consent action through the active release's same GC/door engine.
set -euo pipefail
umask 022
. "$(dirname "$0")/door-common.sh"
: "${DOOR_SCENE_ACTION:?}"
case "$DOOR_SCENE_ACTION" in
  retire | restore) ;;
  *) exit 2 ;;
esac
door_init_request "${DOOR_SCENE_ACTION}-scene"
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_SCENE_INTENT_ID:?}" \
  "${DOOR_SCENE_CONSENT_ID:?}" "${DOOR_SCENE_CONSENT_SHA256:?}" "${DOOR_SCENE_CONSENT_SIZE_BYTES:?}"
[[ "$DOOR_SCENE_INTENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || door_fail scene_lifecycle_options_invalid
[[ "$DOOR_SCENE_CONSENT_ID" =~ ^[0-9a-f]{32}$ ]] || door_fail scene_lifecycle_options_invalid
[[ "$DOOR_SCENE_CONSENT_SHA256" =~ ^sha256:[0-9a-f]{64}$ ]] || door_fail scene_lifecycle_options_invalid
[[ "$DOOR_SCENE_CONSENT_SIZE_BYTES" =~ ^[0-9]{1,6}$ ]] || door_fail scene_lifecycle_options_invalid
(( 10#$DOOR_SCENE_CONSENT_SIZE_BYTES >= 1 && 10#$DOOR_SCENE_CONSENT_SIZE_BYTES <= 524288 )) || door_fail scene_lifecycle_options_invalid
case "${DOOR_SCENE_APPLY:-0}" in
  0 | 1) ;;
  *) door_fail scene_lifecycle_options_invalid ;;
esac
[ "$DOOR_SCENE_ACTION" != restore ] || [ "$DOOR_SCENE_APPLY" = 1 ] || door_fail scene_lifecycle_options_invalid
cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail scene_lifecycle_release_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.scene-lifecycle.json"
apply=()
[ "$DOOR_SCENE_APPLY" != 1 ] || apply=(--apply)
set +e
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.task_evaluation_scene_retirement_cli \
  "$DOOR_SCENE_ACTION" --intent-id "$DOOR_SCENE_INTENT_ID" --consent-id "$DOOR_SCENE_CONSENT_ID" \
  --expected-sha256 "$DOOR_SCENE_CONSENT_SHA256" --expected-size-bytes "$DOOR_SCENE_CONSENT_SIZE_BYTES" \
  ${apply[@]+"${apply[@]}"} >"$result"
rc=$?
set -e
# The module publishes only secret-free counters/typed refusal, never consent.
status="$(python3 - "$result" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as source:
        value=json.load(source)
    status=value.get('status')
except (OSError,ValueError,AttributeError):
    status=None
print(status if status in {'planned','kept','retired','restored','incomplete'} else 'failed')
PY
)"
if [ "$rc" = 0 ]; then
  case "$status" in
    planned | retired | restored) door_outcome "$status" "" 0 intent_id "$DOOR_SCENE_INTENT_ID" result "$result"; exit 0 ;;
    kept) door_outcome retained scene_lifecycle_kept 0 intent_id "$DOOR_SCENE_INTENT_ID" result "$result"; exit 0 ;;
  esac
fi
door_outcome failed scene_lifecycle_incomplete 1 intent_id "$DOOR_SCENE_INTENT_ID" result "$result"
exit 1
