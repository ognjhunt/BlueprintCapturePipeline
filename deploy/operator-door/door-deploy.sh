#!/bin/bash
# Deploy one commit on origin/main to the control plane (operator door, transient root unit).
#
# Mirrors the deploy the agent lanes have been running by hand over SSH: the
# target commit's own scripts/deploy_control_plane_commit.py with --iteration
# and --preserve-configured-controls-state (the owner's controls pause survives
# the deploy; the repo's wrapper scripts do not pass it, so they are not used).
# Only merged code is deployed: the commit must be an ancestor of origin/main.
# The deploy tool's own guards still apply: it holds the paid-launch locks for
# the whole deploy, refuses dirty or unpushed sources, and proves every surface
# moved.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init deploy
: "${DOOR_STATE_ROOT:?}"

if [ "${DOOR_WAIT_FOR_IDLE:-1}" = "1" ]; then
  deadline=$((SECONDS + ${DOOR_IDLE_WAIT_SECONDS:-1800}))
  IFS=, read -r -a idle_units <<<"${DOOR_IDLE_UNITS:-}"
  while :; do
    busy=""
    for unit in ${idle_units[@]+"${idle_units[@]}"}; do
      state="$(systemctl is-active "$unit" 2>/dev/null || true)"
      case "$state" in active|activating|deactivating|reloading) busy="$unit" ;; esac
    done
    [ -z "$busy" ] && break
    [ "$SECONDS" -ge "$deadline" ] && door_fail "idle_wait_timeout:$busy"
    echo "waiting for $busy to go idle"
    sleep "${DOOR_IDLE_POLL_SECONDS:-20}"
  done
fi

door_prepare_source
door_require_on_main
door_add_tool

receipt="$DOOR_STATE_ROOT/deploy-receipts/iteration_${DOOR_COMMIT:0:12}_door.json"
set +e
PYTHONDONTWRITEBYTECODE=1 "$DOOR_VENV_PYTHON" "$DOOR_TOOL/scripts/deploy_control_plane_commit.py" \
  --source-repo "$DOOR_SOURCE_CLONE" \
  --source-commit "$DOOR_COMMIT" \
  --release-root /opt/blueprint/task-evaluation-control-plane-releases \
  --state-root "$DOOR_STATE_ROOT" \
  --active-link /opt/blueprint/task-evaluation-control-plane \
  --iteration --preserve-configured-controls-state \
  --receipt-out "$receipt"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  door_outcome deployed "" "$rc" receipt "$receipt"
else
  door_outcome failed "deploy_tool_exit_$rc" "$rc" receipt "$receipt"
fi
exit "$rc"
