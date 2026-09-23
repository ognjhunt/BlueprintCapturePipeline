#!/bin/bash
# Replay one retained stage against a pushed candidate commit (operator door).
#
# This is the repo's sanctioned loop for a failed production stage:
# task_evaluation_stage_replay --isolate runs the stage handler from the
# candidate tree on the retained inputs, as the service user, with no network,
# no GPU and no model call. Paid phases are never replayed from here.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init stage-replay
child="${DOOR_CHILD:-}"
parent="${DOOR_PARENT:-}"
if [ -n "$child" ] && [ -z "$parent" ] && [[ "$child" =~ ^sam31-[a-f0-9]{8,64}$ ]]; then
  target=(--child "$child")
elif [ -n "$parent" ] && [ -z "$child" ] && [[ "$parent" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{3,159}$ ]]; then
  target=(--parent "$parent")
else
  door_fail replay_target_invalid
fi

door_prepare_source
door_require_pushed
door_add_tool

report="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.replay.json"
set +e
(cd "$DOOR_TOOL" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DOOR_TOOL/src" \
  "$DOOR_VENV_PYTHON" -m blueprint_pipeline.task_evaluation_stage_replay \
  "${target[@]}" --isolate --json-out "$report")
rc=$?
set -e

# A refusing stage is a finding, not a door failure: the report names the predicate.
if [ -s "$report" ]; then
  door_outcome replayed "" "$rc" report "$report"
else
  door_outcome failed "replay_exit_$rc" "$rc"
fi
exit 0
