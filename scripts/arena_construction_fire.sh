#!/bin/bash
# Arm the execute gate and submit an arena run through the website.
#   CUR=r12 bash scripts/arena_construction_fire.sh
# The submit secret is pulled from Render into a mode-600 temp file and removed
# on exit; it is never printed. Ordering is load-bearing: the execute gate must
# be armed BEFORE submission, because the dispatcher fires on queue-write and
# terminal-blocks at retry-0 without a matching EXECUTE_ID.
set -euo pipefail
: "${CUR:?set CUR=<attempt tag, e.g. r12>}"
CP=/opt/blueprint/task-evaluation-control-plane
PY=/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python
E=/var/lib/blueprint/task-evaluation-inputs
A=$E/arena-launch-$CUR
ENVF=/etc/blueprint/pipeline-control-plane.env
PROFILE_JSON=$A/arena_construction_live_profile.v1.json
test -f "$PROFILE_JSON" || { echo "no profile at $PROFILE_JSON -- run arena_construction_launch_chain.sh first"; exit 1; }
PROFILE_ID=$($PY -c "import json;print(json.load(open('$PROFILE_JSON'))['profile_id'])")
PROFILE_DIGEST=$($PY -c "import json,hashlib,sys;
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
print(json.load(open('$PROFILE_JSON')).get('profile_digest') or '')" 2>/dev/null || echo "")
if [ -z "$PROFILE_DIGEST" ]; then
  PROFILE_DIGEST=$($PY -c "
import json,hashlib
raw=open('/etc/blueprint/task-evaluation-launch-profiles/${PROFILE_ID}.json','rb').read()
print('sha256:'+hashlib.sha256(raw).hexdigest())")
fi
SHORT=$(git -C $CP rev-parse --short=8 HEAD)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUFFIX=$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')
LID=adp-arena-construction-840920-task-a-${SHORT}-${CUR}-api-${STAMP}-${SUFFIX}
EXPIRES=$(date -u -d "+3 hours" +%Y-%m-%dT%H:%M:%S+00:00)
echo "profile_id: $PROFILE_ID"
echo "launch_id:  $LID"

echo "== arm execute gate BEFORE submission =="
cp -a $ENVF ${ENVF}.bak-${CUR}-${STAMP}
sed -i "s|^BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID=.*|BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID=${LID}|" $ENVF
grep -E "^BLUEPRINT_TASK_EVALUATION_LAUNCH_(EXECUTE|EXECUTE_ID)=" $ENVF

echo "== author the WebApp-shaped request =="
REQ=$A/request.webapp.${CUR}.json
$PY - <<PYEOF
import json, pathlib
pathlib.Path("$REQ").write_text(json.dumps({
 "confirm_execution": True,
 "launch_id": "$LID", "run_id": "$LID",
 "profile_id": "$PROFILE_ID", "profile_digest": "$PROFILE_DIGEST",
 "rights": {"evidence": {
   "digest": "sha256:980298d20acfc1cccab19f16c1b20ff0ce272a38bde1cb6c96ab4215477f5775",
   "uri": "gs://blueprint-8c1ca-scenes/task-evaluation/immutable-rights-authorities/sha256/98/980298d20acfc1cccab19f16c1b20ff0ce272a38bde1cb6c96ab4215477f5775.json"},
  "scope": "internal_noncommercial_research_only"},
 "spend": {"expires_at": "$EXPIRES", "max_spend_usd": 2.0},
}, indent=1) + "\n", encoding="utf-8")
print("wrote $REQ")
PYEOF
chown blueprint:blueprint $REQ

echo "== submit through the website =="
umask 077
SF=$(mktemp); trap 'rm -f $SF' EXIT
RKEY=$(cat /etc/blueprint/provider-secrets/render_api_key)
curl -sS -H "Authorization: Bearer $RKEY" -H "Accept: application/json" \
  "https://api.render.com/v1/services/srv-d4vnmk3e5dus73aiohk0/env-vars?limit=100" \
| $PY -c "
import json,sys
for e in json.load(sys.stdin):
    v=e['envVar']
    if v['key']=='BLUEPRINT_TASK_EVALUATION_LAUNCH_SUBMIT_SECRET':
        sys.stdout.write(v['value']); break
else:
    raise SystemExit('submit secret not found')" > $SF
chmod 600 $SF; chown blueprint:blueprint $SF
sudo -u blueprint env PYTHONPATH=$CP/src $PY $CP/scripts/submit_task_evaluation_launch_via_webapp.py \
  --request $REQ --secret-file $SF \
  --receipt-out $A/webapp_submit_receipt.${CUR}.json 2>&1 | tail -4
echo "LAUNCH_ID=$LID"
