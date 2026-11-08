#!/bin/bash
# Plan, or with DOOR_APPLY=1 perform, the active release's deletion of the internal folders the
# owner approved on this host (owner_approved_folder_reclaim). The folders are listed only in the
# release's reviewed allowlist, config/owner_approved_reclaim_folders.v1.json; the request carries
# only apply. The module runs as root with only CAP_DAC_OVERRIDE (the unit's bounding set), as
# blueprint-owner-approved-deletion.service does, because a folder may hold files another account
# owns. It reads no secret, uses no network, starts no compute and moves no money.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request reclaim-approved-folders
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}"
case "${DOOR_APPLY:-}" in "" | 1) ;; *) door_fail reclaim_approved_folders_apply_invalid ;; esac

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
# An active release older than the module or its allowlist cannot run it: say so instead of failing opaquely.
{ [ -f src/blueprint_pipeline/owner_approved_folder_reclaim.py ] && [ -f config/owner_approved_reclaim_folders.v1.json ]; } \
  || door_fail reclaim_approved_folders_release_lacks_module
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.reclaim-approved-folders.json"
args=()
if [ "${DOOR_APPLY:-}" = 1 ]; then args+=(--apply); fi
set +e
summary="$(umask 0077 && exec env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.owner_approved_folder_reclaim \
  ${args[@]+"${args[@]}"})"
rc=$?
set -e
printf '%s\n' "$summary" >"$result.tmp"
chmod 0644 "$result.tmp"
mv -f "$result.tmp" "$result"
verdict="$(python3 - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        row = json.load(stream)
except (OSError, ValueError):
    row = {}
row = row if isinstance(row, dict) else {}
outcome = row.get("outcome")
outcome = outcome if outcome in {"planned", "reclaimed", "already_reclaimed", "refused"} else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200] or "-"
receipt = row.get("receipt_path")
receipt = receipt if isinstance(receipt, str) and re.fullmatch(r"/[A-Za-z0-9_./-]{1,1024}\.json", receipt) else "-"
print(outcome, code, receipt)
PY
)"
read -r outcome code receipt <<<"$verdict"
[ "$code" != - ] || code=""
if [ "$rc" -eq 0 ] && { [ "$outcome" = planned ] || [ "$outcome" = reclaimed ] || [ "$outcome" = already_reclaimed ]; }; then
  if [ "$receipt" != - ]; then
    # The receipt kept under storage-gc/owner-approved-reclaims.
    door_outcome "$outcome" "" 0 result "$result" receipt "$receipt"
  else
    door_outcome "$outcome" "" 0 result "$result"
  fi
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
if [ "$outcome" = refused ]; then
  if [ "$receipt" != - ]; then
    # A deletion that stopped part way still kept its receipt of what it removed.
    door_outcome refused "${code:-reclaim_approved_folders_failed}" "$rc" result "$result" receipt "$receipt"
  else
    door_outcome refused "${code:-reclaim_approved_folders_failed}" "$rc" result "$result"
  fi
else
  door_outcome failed reclaim_approved_folders_failed "$rc" result "$result"
fi
exit "$rc"
