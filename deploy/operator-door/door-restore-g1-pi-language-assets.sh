#!/bin/bash
# Plan, or with DOOR_APPLY=1 perform, the active release's restore of the pi0.5 G1
# language files from a retained G1 provider bundle on this host
# (native_g1_pi_language_asset_restore). The folder, the search root and the inventory
# are pinned in that module; the request carries only apply. The module runs as the
# service user that owns the folder, with a private umask; this root script keeps only
# the door's log, result and outcome. It reads no secret, uses no network, starts no
# compute and moves no money.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request restore-g1-pi-language-assets
: "${DOOR_VENV_PYTHON:?}" "${DOOR_CONTROL_PLANE_REPO:?}" "${DOOR_SERVICE_USER:?}"
case "${DOOR_APPLY:-}" in "" | 1) ;; *) door_fail g1_pi_language_assets_apply_invalid ;; esac
[ "$DOOR_SERVICE_USER" = blueprint ] || door_fail g1_pi_language_assets_user_invalid

# Read only BLUEPRINT_* names as KEY=VALUE data; never evaluate the file. The disk
# budget's floor settings live there, so the hold is admitted as the service admits it.
env_file="${DOOR_CONTROL_PLANE_ENV_FILE:-/etc/blueprint/pipeline-control-plane.env}"
if [ -r "$env_file" ]; then
  assignment='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  double_quoted='^"(.*)"$'
  single_quoted="^'(.*)'\$"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ $assignment ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "$key" in BLUEPRINT_*) ;; *) continue ;; esac
    if [[ "$value" =~ $double_quoted ]] || [[ "$value" =~ $single_quoted ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    export "$key=$value"
  done <"$env_file"
fi
: "${BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT:=/var/lib/blueprint/pipeline-control-plane/disk-reservations}"
export BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT

cd -P "$DOOR_CONTROL_PLANE_REPO" || door_fail control_plane_release_missing
# An active release older than the module cannot run it: say so instead of failing opaquely.
[ -f src/blueprint_pipeline/native_g1_pi_language_asset_restore.py ] \
  || door_fail g1_pi_language_assets_release_lacks_module
# No directory is made here: the module writes only beside the folder, which the service user owns.
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.g1-pi-language-assets.json"
args=()
if [ "${DOOR_APPLY:-}" = 1 ]; then args+=(--apply); fi
set +e
summary="$(umask 0077 && exec setpriv --reuid="$DOOR_SERVICE_USER" --regid="$DOOR_SERVICE_USER" --init-groups \
  --inh-caps=-all -- env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.native_g1_pi_language_asset_restore \
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
outcome = outcome if outcome in {"planned", "restored", "already_restored", "refused"} else "failed"
code = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(row.get("code") or ""))[:200] or "-"
receipt = row.get("receipt_path")
receipt = receipt if isinstance(receipt, str) and re.fullmatch(r"/[A-Za-z0-9_./-]{1,1024}\.json", receipt) else "-"
print(outcome, code, receipt)
PY
)"
read -r outcome code receipt <<<"$verdict"
[ "$code" != - ] || code=""
if [ "$rc" -eq 0 ] && { [ "$outcome" = planned ] || [ "$outcome" = restored ] || [ "$outcome" = already_restored ]; }; then
  if [ "$receipt" != - ]; then
    # The receipt kept beside the folder; its name avoids the word the door refuses to read.
    door_outcome "$outcome" "" 0 result "$result" receipt "$receipt"
  else
    door_outcome "$outcome" "" 0 result "$result"
  fi
  exit 0
fi
[ "$rc" -ne 0 ] || rc=1
if [ "$outcome" = refused ]; then
  door_outcome refused "${code:-g1_pi_language_assets_restore_failed}" "$rc" result "$result"
else
  door_outcome failed g1_pi_language_assets_restore_failed "$rc" result "$result"
fi
exit "$rc"
