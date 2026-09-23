#!/bin/bash
# Deploy one pushed commit to the control plane (operator door, transient root unit).
#
# Mirrors the deploy the agent lanes have been running by hand over SSH: the
# target commit's own scripts/deploy_control_plane_commit.py with --iteration
# and --preserve-configured-controls-state (the owner's controls pause survives
# the deploy; the repo's wrapper scripts do not pass it, so they are not used).
# main mode requires an ancestor of origin/main; canary mode (--canary) allows
# any pushed ref. The deploy tool's own guards still apply: it refuses while a
# paid launch holds a Vast lock, refuses dirty or unpushed sources, and proves
# every surface moved.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init deploy
: "${DOOR_MODE:?}" "${DOOR_STATE_ROOT:?}"
case "$DOOR_MODE" in main|canary) ;; *) door_fail mode_invalid ;; esac

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
if [ "$DOOR_MODE" = "main" ]; then
  door_require_on_main
  prefix=iteration
  flags=(--iteration --preserve-configured-controls-state)
else
  door_require_pushed
  prefix=canary
  flags=(--iteration --canary --preserve-configured-controls-state)
fi
door_add_tool

receipt="$DOOR_STATE_ROOT/deploy-receipts/${prefix}_${DOOR_COMMIT:0:12}_door.json"
set +e
PYTHONDONTWRITEBYTECODE=1 "$DOOR_VENV_PYTHON" "$DOOR_TOOL/scripts/deploy_control_plane_commit.py" \
  --source-repo "$DOOR_SOURCE_CLONE" \
  --source-commit "$DOOR_COMMIT" \
  --release-root /opt/blueprint/task-evaluation-control-plane-releases \
  --state-root "$DOOR_STATE_ROOT" \
  --active-link /opt/blueprint/task-evaluation-control-plane \
  "${flags[@]}" \
  --receipt-out "$receipt"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  door_outcome deployed "" "$rc" receipt "$receipt" mode "$DOOR_MODE"
else
  door_outcome failed "deploy_tool_exit_$rc" "$rc" receipt "$receipt" mode "$DOOR_MODE"
fi
exit "$rc"
