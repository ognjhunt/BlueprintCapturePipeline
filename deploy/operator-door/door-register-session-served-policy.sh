#!/bin/bash
# Add or remove one session-served policy entry in the root registry with only independently installed
# stdlib code: a compare-and-swap against the request's expected registry digest, with a kept receipt.
set -euo pipefail
umask 022
# shellcheck source-path=SCRIPTDIR source=door-common.sh
. "$(dirname "$0")/door-common.sh"

door_init_request register-session-served-policy
: "${DOOR_SESSION_SERVED_REQUEST:?}" "${DOOR_CONTROL_PLANE_REPO:?}"
door_code="$(cd "$(dirname "$0")" && pwd)"
catalog="$(cd -P "$DOOR_CONTROL_PLANE_REPO" && pwd)/config/beta_policy_candidates.v1.json"
result="$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.session-served-policy.json"
set +e
python3 -I -S "$door_code/operator_door/session_served_registration.py" \
  --request-json "$DOOR_SESSION_SERVED_REQUEST" --catalog "$catalog" > "$result.tmp"
rc=$?
set -e
chmod 0640 "$result.tmp"
chgrp blueprint-door "$result.tmp" 2>/dev/null || true
mv -f "$result.tmp" "$result"
if [ "$rc" -eq 0 ]; then
  door_outcome completed "" 0 result "$result" disposition session_served_policy_registered
else
  code="$(python3 - "$result" <<'PY'
import json, re, sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        document = json.load(stream)
except (OSError, ValueError):
    document = {}
print(re.sub(r"[^A-Za-z0-9_.:-]", "_", str((document if isinstance(document, dict) else {}).get("code") or ""))[:200])
PY
)"
  door_outcome refused "${code:-session_served_registration_failed}" "$rc" result "$result"
fi
exit "$rc"
