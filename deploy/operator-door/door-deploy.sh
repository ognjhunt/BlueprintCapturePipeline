#!/bin/bash
# Deploy one commit on origin/main to the control plane (operator door, transient root unit).
#
# Mirrors the deploy the agent lanes have been running by hand over SSH: the
# target commit's own scripts/deploy_control_plane_commit.py. Without provenance
# this retains --iteration. Exact official provenance staged by the root runner
# opts into the existing canonical --release-provenance validation instead.
# Both modes use --preserve-configured-controls-state (the owner's pause survives
# the deploy; the repo's wrapper scripts do not pass it, so they are not used).
# Only merged code is deployed: the commit must be an ancestor of origin/main.
# The deploy tool's own guards still apply: it holds the paid-launch locks for
# the whole deploy, refuses dirty or unpushed sources, and proves every surface
# moved.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request deploy
door_init_git
: "${DOOR_STATE_ROOT:?}"

mode=iteration
deploy_options=(--iteration)
if [ -n "${DOOR_RELEASE_PROVENANCE_FILE+x}" ] || [ -n "${DOOR_RELEASE_PROVENANCE_SHA256+x}" ]; then
  expected="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.release-provenance.json"
  [ "${DOOR_RELEASE_PROVENANCE_FILE:-}" = "$expected" ] || door_fail release_provenance_transport_invalid
  [[ "${DOOR_RELEASE_PROVENANCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || door_fail release_provenance_transport_invalid
  # The caller supplies bytes, never a host path. Recheck the root runner's fixed
  # file before fetching source; the deploy tool owns schema/source validation.
  if ! python3 -I -S - "$expected" "$DOOR_RELEASE_PROVENANCE_SHA256" <<'PY'
import hashlib, os, stat, sys
try:
    fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or not 0 < info.st_size <= 16 * 1024):
            raise ValueError()
        payload = stream.read(16 * 1024 + 1)
    if len(payload) > 16 * 1024 or hashlib.sha256(payload).hexdigest() != sys.argv[2]:
        raise ValueError()
except (OSError, ValueError):
    sys.exit(2)
PY
  then
    door_fail release_provenance_transport_invalid
  fi
  mode=production
  deploy_options=(--release-provenance "$expected")
fi

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

receipt="$DOOR_STATE_ROOT/deploy-receipts/${mode}_${DOOR_COMMIT:0:12}_door.json"
set +e
PYTHONDONTWRITEBYTECODE=1 "$DOOR_VENV_PYTHON" "$DOOR_TOOL/scripts/deploy_control_plane_commit.py" \
  --source-repo "$DOOR_SOURCE_CLONE" \
  --source-commit "$DOOR_COMMIT" \
  --release-root /opt/blueprint/task-evaluation-control-plane-releases \
  --state-root "$DOOR_STATE_ROOT" \
  --active-link /opt/blueprint/task-evaluation-control-plane \
  "${deploy_options[@]}" --preserve-configured-controls-state \
  --receipt-out "$receipt"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  door_outcome deployed "" "$rc" receipt "$receipt" mode "$mode"
else
  door_outcome failed "deploy_tool_exit_$rc" "$rc" receipt "$receipt"
fi
exit "$rc"
