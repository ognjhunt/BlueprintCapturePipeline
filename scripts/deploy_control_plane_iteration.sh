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
SCRIPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# Provider-runtime-only canary fixes default to a signed overlay.  The active
# release stays byte-exact; the next canary bundle receives only the tested
# changed modules and is capped at development_only evidence.  Unsupported
# changes continue through the normal exact-main deployment below.
active_sha="$(git -C /opt/blueprint/task-evaluation-control-plane rev-parse HEAD)"
route_json="$(PYTHONPATH="$SCRIPT_REPO/src" "$PY" -m blueprint_pipeline.task_evaluation_canary_hotfix_overlay route \
  --repo-root "$SCRIPT_REPO" --base-commit "$active_sha" --patch-commit "$SHA")"
strategy="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["strategy"])' <<<"$route_json")"
if [[ "$strategy" == "signed_hotfix_overlay" ]]; then
  : "${BLUEPRINT_CANARY_HOTFIX_EXACT_FAILURE_INPUT:?eligible canary hotfix requires the exact failed input path}"
  : "${BLUEPRINT_CANARY_HOTFIX_TEST_COMMAND_JSON:?eligible canary hotfix requires one focused test argv JSON array}"
  overlay_root="${BLUEPRINT_CANARY_HOTFIX_OUTPUT_DIR:-/var/lib/blueprint/pipeline-control-plane/canary-hotfix-overlays/${SHA:0:12}}"
  PYTHONPATH="$SCRIPT_REPO/src" "$PY" -m blueprint_pipeline.task_evaluation_canary_hotfix_overlay prepare \
    --repo-root "$SCRIPT_REPO" \
    --output-dir "$overlay_root" \
    --base-commit "$active_sha" \
    --patch-commit "$SHA" \
    --exact-failure-input "$BLUEPRINT_CANARY_HOTFIX_EXACT_FAILURE_INPUT" \
    --test-command-json "$BLUEPRINT_CANARY_HOTFIX_TEST_COMMAND_JSON"
  PYTHONPATH="$SCRIPT_REPO/src" "$PY" -m blueprint_pipeline.task_evaluation_canary_hotfix_overlay install \
    --plan "$overlay_root/task_evaluation_canary_hotfix_overlay_plan.v1.json" \
    --drop-in /etc/systemd/system/blueprint-task-evaluation-policy-canary-dispatcher.service.d/96-signed-hotfix-overlay.conf \
    --receipt "$overlay_root/task_evaluation_canary_hotfix_installation.v1.json"
  systemctl daemon-reload
  echo "installed signed development-only canary hotfix overlay: $overlay_root"
  echo "normal exact-main deployment remains required for promotion"
  exit 0
fi

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
