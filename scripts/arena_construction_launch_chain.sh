#!/bin/bash
# Parameterized arena construction launch.
#   PREV=r10 CUR=r11 bash scripts/arena_construction_launch_chain.sh
# Every step is idempotent: single-write receipts are skipped when present, so
# the script is safe to re-run after any failure.
set -euo pipefail
: "${PREV:?set PREV=<predecessor tag, e.g. r10>}"
: "${CUR:?set CUR=<this attempt tag, e.g. r11>}"
CP=/opt/blueprint/task-evaluation-control-plane
PY=/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python
RUN="sudo -u blueprint env PYTHONPATH=$CP/src $PY"
E=/var/lib/blueprint/task-evaluation-inputs
A=$E/arena-launch-$CUR
P=$E/arena-launch-$PREV
LR=/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs
TASK=task_a_washer_door_open

# newest launch-run directory for the predecessor tag (no pipe: pipefail+head
# SIGPIPEs ls and exits 141)
mapfile -t _RUNS < <(ls -dt ${LR}/adp-arena-construction-840920-task-a-*-${PREV}-api-*/)
RPREV=${_RUNS[0]}
JOBPREV=$RPREV/allocator/arena-construction-job
mapfile -t _AUDS < <(ls -dt /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/billing-audit/*/)
AUD=${_AUDS[0]}
COMMIT=$(git -C $CP rev-parse HEAD)

echo "prev run: $RPREV"
echo "commit:   $COMMIT"
echo "audit:    $AUD"
sudo -u blueprint mkdir -p $A
cd $CP

echo "== 0. predecessor provider zero"
[ -f $P/construction_provider_zero.v1.json ] && echo '  exists, skipping' || \
  $RUN scripts/seal_native_task_arena_provider_zero.py \
    --authority $P/native_task_arena_paid_attempt_authority.v1.json \
    --result $JOBPREV/adp_arena_vast_result.json \
    --output $P/construction_provider_zero.v1.json | tail -1

echo "== 1. lane-local spend reconciliation"
# Reconciliation is about PROVIDER SPEND, and it binds a real positive
# instance id (`prior_provider_instance_id_invalid`). A run that ended before
# allocating -- no offer met the lane's constraints, admission refused -- has
# `vast_instance_ids: []`, no billing row, and nothing to reconcile. It is not
# a prior paid attempt, so walk back to the most recent run that actually
# allocated. Step 0 above still seals the immediate predecessor, so a
# non-allocating run is proven empty rather than skipped.
# Newest-first across ALL attempt tags, not just $PREV, so the walk can reach
# past a run that allocated nothing.
mapfile -t _ALLRUNS < <(ls -dt ${LR}/adp-arena-construction-840920-task-a-*-api-*/)
JOBSPEND=""
ZEROSPEND=""
for _run in "${_ALLRUNS[@]}"; do
  _job="$_run/allocator/arena-construction-job"
  _td="$_job/attempts/attempt_001/vast_provider_run/vast_teardown_manifest.json"
  [ -f "$_td" ] || continue
  INST=$($PY -c "
import json
ids=json.load(open('$_td')).get('vast_instance_ids') or []
print(ids[0] if ids else '')")
  _tag=$(basename $_run | sed -E 's/.*-(r[0-9]+)-api-.*/\1/')
  if [ -z "$INST" ]; then echo "  skipping $_tag: allocated nothing"; continue; fi
  # terminal result, teardown, and provider zero must all describe the SAME
  # attempt, so take the zero from that attempt's own input directory.
  _zero=$E/arena-launch-${_tag}/construction_provider_zero.v1.json
  [ -f "$_zero" ] || { echo "  skipping $_tag: no sealed provider zero yet"; continue; }
  JOBSPEND="$_job"; ZEROSPEND="$_zero"; break
done
[ -n "$JOBSPEND" ] || { echo "no predecessor run allocated an instance -- nothing to reconcile against"; exit 1; }
echo "  predecessor instance: $INST (from $_tag)"
mapfile -t _RESPS < <(grep -l "$INST" ${AUD}response-00*-vast.json)
VASTRESP=${_RESPS[0]}
[ -f $A/prior_spend_reconciliation.v1.json ] && echo '  exists, skipping' || \
  $RUN scripts/materialize_same_goal_spend_reconciliation.py \
    --lane native_task_arena \
    --terminal-result $JOBSPEND/adp_arena_vast_result.json \
    --teardown-manifest $JOBSPEND/attempts/attempt_001/vast_provider_run/vast_teardown_manifest.json \
    --provider-zero $ZEROSPEND \
    --official-billing-response "$VASTRESP" \
    --provider-billing-source-receipt ${AUD}provider_billing_source_receipt.json \
    --output $A/prior_spend_reconciliation.v1.json | tail -1

echo "== 2. staged packet"
if [ -d $A/arena_packet/$TASK ]; then
  echo '  exists, skipping'
else
  # hardlink the predecessor's sealed packet: identical inodes are the strongest
  # possible statement that the staged bytes did not change, and it costs no disk
  sudo -u blueprint mkdir -p $A/arena_packet
  sudo -u blueprint cp -al $P/arena_packet/$TASK $A/arena_packet/$TASK
  echo "  hardlinked from $PREV"
fi

echo "== 3. construction bundle at the deployed commit"
[ -f $A/arena_construction_job/native_task_arena_provider_bundle_receipt.v1.json ] && echo '  exists, skipping' || \
$RUN - <<EOF | tail -2
from blueprint_pipeline.native_task_arena_construction_bundle import build_native_task_arena_construction_bundle
r = build_native_task_arena_construction_bundle(
    job_dir="$A/arena_construction_job",
    packet_dir="$A/arena_packet/$TASK",
    runtime_source_packet_receipt="$E/native-task-runtime-source-c3e8b79a/native_task_runtime_source_packet.v1.json",
    implementation_commit="$COMMIT")
print("bundle:", r.get("status"), r.get("execution_mode"))
EOF

echo "== 4. attempt authority chained off $PREV"
[ -f $A/native_task_arena_paid_attempt_authority.v1.json ] && echo '  exists, skipping' || \
  $RUN scripts/issue_native_task_arena_paid_attempt_authority.py \
    --bundle-receipt $A/arena_construction_job/native_task_arena_provider_bundle_receipt.v1.json \
    --prior-authority $P/native_task_arena_paid_attempt_authority.v1.json \
    --prior-result $JOBPREV/adp_arena_vast_result.json \
    --prior-provider-zero $P/construction_provider_zero.v1.json \
    --prior-spend-reconciliation $A/prior_spend_reconciliation.v1.json \
    --authority-reference active_goal_scene840920_arena_construction_authorization_20260818 \
    --authorized-by nijelhunt_1 \
    --authorized-on "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --blueprint-commit $COMMIT \
    --max-hourly-rate-usd 1.0 --hard-cap-usd 2.0 --hard-ttl-seconds 7200 \
    --output $A/native_task_arena_paid_attempt_authority.v1.json | tail -1

echo "== 5. publish bundle receipt"
[ -f $A/bundle_manifest_publication_receipt.json ] && echo '  exists, skipping' || \
sudo -u blueprint bash -c "set -a; source /etc/blueprint/pipeline-control-plane.env; set +a; PYTHONPATH=$CP/src $PY scripts/publish_task_evaluation_immutable_manifest.py \
  --manifest $A/arena_construction_job/native_task_arena_provider_bundle_receipt.v1.json \
  --profile-builder build_native_task_arena_live_profile.py \
  --destination-prefix r2://blueprint/task-evaluation/immutable-manifests \
  --output $A/bundle_manifest_publication_receipt.json" | tail -1

echo "== 6. live profile"
[ -f $A/arena_construction_live_profile.v1.json ] && echo '  exists, skipping' || \
  $RUN scripts/build_native_task_arena_live_profile.py construction \
    --packet-dir $A/arena_packet/$TASK \
    --bundle-receipt $A/arena_construction_job/native_task_arena_provider_bundle_receipt.v1.json \
    --attempt-authority $A/native_task_arena_paid_attempt_authority.v1.json \
    --runtime-source-packet $E/native-task-runtime-source-c3e8b79a/native_task_runtime_source_packet.v1.json \
    --source-commit $COMMIT --scene-id 840920 --task-id $TASK \
    --revision $CUR \
    --raw-manifest-uri $A/bundle_manifest_publication_receipt.json \
    --machine-avoidlist $E/arena-launch-r5/machine_avoidlist.json \
    --max-hourly-rate-usd 1.0 --max-spend-usd 2.0 --hard-ttl-seconds 7200 \
    --output $A/arena_construction_live_profile.v1.json | tail -1

echo "== 7. terminal contract rehearsal (no spend)"
$RUN scripts/rehearse_lane_terminal_contract.py \
  --profile $A/arena_construction_live_profile.v1.json \
  --lane-module adp_isaac_lab_arena_vast.py \
  --lane native_task_arena_construction \
  --receipt-out $A/arena_construction_terminal_rehearsal.v1.json 2>&1 | grep -E '"status"' | tail -1

PROFILE_ID=$($PY -c "import json;print(json.load(open('$A/arena_construction_live_profile.v1.json'))['profile_id'])")
echo "profile_id: $PROFILE_ID"

echo "== 8. publish profile + standing authorization"
env PYTHONPATH=$CP/src $PY scripts/publish_task_evaluation_launch_profiles.py \
  --profile $A/arena_construction_live_profile.v1.json \
  --profile-dir /etc/blueprint/task-evaluation-launch-profiles \
  --webapp-catalog-out /var/lib/blueprint/task-evaluation-webapp/launch-catalog.json \
  --service-account blueprint | tail -1
env PYTHONPATH=$CP/src $PY scripts/materialize_task_evaluation_standing_launch_authorization.py \
  --profile /etc/blueprint/task-evaluation-launch-profiles/${PROFILE_ID}.json \
  --output-dir /var/lib/blueprint/pipeline-control-plane/standing-authorizations \
  --authorized-by nijelhunt_1 \
  --authorization-reference active_goal_scene840920_arena_construction_authorization_20260818 \
  --issued-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --expires-at "$(date -u -d "+1 day" +%Y-%m-%dT%H:%M:%SZ)" \
  --max-launches 2 --max-total-spend-usd 4.0 --service-account blueprint | tail -1
echo "READY-TO-SUBMIT profile_id=$PROFILE_ID"
