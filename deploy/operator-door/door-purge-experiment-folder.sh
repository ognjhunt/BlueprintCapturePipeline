#!/bin/bash
# Plan, or with the request's apply perform, the owner-approved purge of one old experiment
# folder (operator_door/experiment_purge.py, installed with the door). The module builds the
# path from its own allowlist, proves no process, mount or recent write touches the folder, and
# writes a receipt before and after deleting its contents. It needs no release and no network;
# this root script keeps only the door's log, result and outcome. Nothing can be restored.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request purge-experiment-folder
: "${DOOR_INSTALL_ROOT:?}" "${DOOR_PURGE_FOLDER:?}" "${DOOR_PURGE_APPLY:?}"
case "$DOOR_PURGE_FOLDER" in
  atlas-beta-dishwasher-20261021 | flora-episode-benchmarks) ;;
  *) door_fail purge_folder_not_approved ;;
esac
args=(--folder "$DOOR_PURGE_FOLDER" --request-id "$DOOR_REQUEST_ID")
case "$DOOR_PURGE_APPLY" in
  0) ;;
  1) args+=(--apply) ;;
  *) door_fail apply_invalid ;;
esac
module="$DOOR_INSTALL_ROOT/operator_door/experiment_purge.py"
[ -f "$module" ] || door_fail purge_module_missing
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.purge-experiment-folder.json"
cd /
set +e
python3 -I -S "$module" "${args[@]}" --result-out "$result"
rc=$?
set -e
verdict="$(python3 -I -S - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        row = json.load(stream)
except (OSError, ValueError):
    row = {}
row = row if isinstance(row, dict) else {}
outcome = row.get("outcome")
known = {"planned", "emptied", "already_empty", "absent", "refused", "partly_emptied", "emptying"}
outcome = outcome if outcome in known else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200] or "-"
print(outcome, code)
PY
)"
read -r outcome code <<<"$verdict"
[ "$code" != - ] || code=""
case "$outcome" in
  planned | emptied | already_empty | absent)
    if [ "$rc" -eq 0 ]; then
      door_outcome "$outcome" "" 0 result "$result"
      exit 0
    fi
    ;;
esac
[ "$rc" -ne 0 ] || rc=1
case "$outcome" in
  refused) door_outcome refused "${code:-purge_refused}" "$rc" result "$result" ;;
  partly_emptied | emptying) door_outcome failed "${code:-purge_incomplete}" "$rc" result "$result" ;;
  *) door_outcome failed purge_failed "$rc" result "$result" ;;
esac
exit "$rc"
