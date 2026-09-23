#!/bin/bash
# Reinstall the operator door from a commit on main (operator door, transient root unit).
#
# Runs the target commit's install.sh --upgrade, which keeps the previous door
# and rolls back if the new one fails its health check, so a bad upgrade cannot
# lock cloud sessions out for longer than the upgrade itself.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init door-upgrade
door_prepare_source
door_require_on_main
door_add_tool

set +e
bash "$DOOR_TOOL/deploy/operator-door/install.sh" --upgrade
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
  door_outcome upgraded "" "$rc"
else
  door_outcome failed "install_exit_$rc" "$rc"
fi
exit "$rc"
