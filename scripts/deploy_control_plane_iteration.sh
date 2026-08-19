#!/bin/bash
# Deploy a pushed commit to the control plane WITHOUT waiting for the Full
# Test Lane, then leave it ready to fire.
#
#   SHA=<40-char pushed commit> bash scripts/deploy_control_plane_iteration.sh
#
# Why this exists: the lane takes ~15 minutes, so a fix-and-fire loop that
# blocks on it spends ~18 minutes per attempt -- tens of hours across a
# campaign of GPU runs. What is NOT traded away is knowing which bytes ran:
# the release is built from a real pushed commit, so the running code and
# main cannot silently diverge the way an edit-in-place hotfix would.
#
# The release is stamped promotion_eligible=false and every paid run from it
# records evidence_grade_ceiling=development_only. Promote with the normal
# lane-verified deploy (scripts/deploy_control_plane_commit.py with
# --release-provenance) before sealing evidence.
set -euo pipefail
: "${SHA:?set SHA=<40-char pushed commit>}"
CP=/opt/blueprint/BlueprintCapturePipeline
PY=$CP/.venv/bin/python
cd $CP
# Fetch main specifically. A bare `git fetch origin` tries to create a ref lock
# for every remote branch, and the service account cannot write new ones under
# .git/refs/remotes -- it fails with "Permission denied" on branches that have
# nothing to do with this deploy.
sudo -u blueprint git fetch -q origin main
# Fail closed if the commit is not on main: an unmerged SHA is exactly the
# local-only drift this mode exists to prevent. The deploy tool enforces this
# too, so bypassing this wrapper does not bypass the guard.
sudo -u blueprint git merge-base --is-ancestor "$SHA" origin/main 2>/dev/null \
  || { echo "refusing: $SHA is not on origin/main -- merge it first"; exit 1; }
$PY scripts/deploy_control_plane_commit.py \
  --source-repo $CP \
  --source-commit "$SHA" \
  --release-root /opt/blueprint/task-evaluation-control-plane-releases \
  --state-root /var/lib/blueprint/pipeline-control-plane \
  --active-link /opt/blueprint/task-evaluation-control-plane \
  --iteration \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/deploy-receipts/iteration_${SHA:0:8}.json 2>&1 | tail -6
echo "=== active release ==="
ls -la /opt/blueprint/task-evaluation-control-plane | tail -1
echo "NOTE: promotion_eligible=false -- evidence from runs on this release is development_only"
