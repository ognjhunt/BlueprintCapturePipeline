#!/bin/bash
# Deploy a pushed commit from ANY origin ref to the control plane, without
# waiting for a merge or the Full Test Lane, then leave it ready to fire.
#
#   SHA=<40-char pushed commit> bash scripts/deploy_control_plane_canary.sh
#
# Why this exists: fix-and-fire debugging on a development_only lane spends
# 5-15 minutes per attempt merging (review, CI, rebase churn against a
# fast-moving main) for evidence the canary run cannot use anyway. What is NOT
# traded away is immutability: the deploy tool refuses any commit that is not
# reachable from a pushed origin ref, so the running bytes are always publicly
# recorded. The release is stamped status=canary, promotion_eligible=false,
# and every paid run from it records evidence_grade_ceiling=development_only.
#
# Promote with the normal lane-verified deploy before sealing evidence, and
# land the debugged fix on main the same session -- a canary ref is a debugging
# vehicle, never a place for changes to live.
set -euo pipefail
: "${SHA:?set SHA=<40-char pushed commit>}"
CP=/opt/blueprint/BlueprintCapturePipeline
PY=$CP/.venv/bin/python
cd $CP
# Fetch all refs so freshly pushed canary branches resolve. Branch creation
# under .git/refs/remotes can hit service-account permissions on unrelated
# refs; tolerate partial failures and let the deploy tool's own reachability
# check decide.
sudo -u blueprint git fetch -q origin || true
$PY scripts/deploy_control_plane_commit.py \
  --source-repo $CP \
  --source-commit "$SHA" \
  --release-root /opt/blueprint/task-evaluation-control-plane-releases \
  --state-root /var/lib/blueprint/pipeline-control-plane \
  --active-link /opt/blueprint/task-evaluation-control-plane \
  --iteration --canary \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/deploy-receipts/canary_${SHA:0:8}.json 2>&1 | tail -6
echo "=== active release ==="
ls -la /opt/blueprint/task-evaluation-control-plane | tail -1
echo "NOTE: status=canary, promotion_eligible=false -- evidence is development_only"
