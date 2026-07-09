#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
ENV_DIR="${ENV_DIR:-/etc/blueprint}"
ENV_FILE="${ENV_FILE:-${ENV_DIR}/pipeline-control-plane.env}"
STATE_DIR="${STATE_DIR:-/var/lib/blueprint/pipeline-control-plane}"
HANDOFF_DIR="${HANDOFF_DIR:-/var/lib/blueprint/pubsub-handoffs}"
GPU_SPEND_GUARD_DIR="${GPU_SPEND_GUARD_DIR:-/var/lib/blueprint/gpu-spend-guard}"
ENABLE_NOW=false
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: scripts/install_live_pipeline_control_plane.sh [--enable-now] [--dry-run]

Installs the Blueprint live pipeline control-plane systemd service/timer, the
capture handoff Pub/Sub listener service/timer, the GPU spend-guard reaper
service/timer, and the optional authenticated WebApp intake service unit.
The service runs one safe control-plane pass on each timer tick:
read env, audit readiness, optionally consume the robot-eval job inbox, write
manifests, run the proof-boundary audit, and exit. It does not add secrets or
enable live simulator/provider actions by itself.
The GPU spend-guard timer runs scripts/gpu_spend_guard.py --reap on a short
interval, reaping orphaned GPU pods (never-booted duds and, past the hard age
ceiling, booted orphans whose launching host died) and persisting a JSON spend
snapshot as durable teardown evidence.

Environment overrides:
  SYSTEMD_DIR=/etc/systemd/system
  ENV_DIR=/etc/blueprint
  ENV_FILE=/etc/blueprint/pipeline-control-plane.env
  STATE_DIR=/var/lib/blueprint/pipeline-control-plane
  HANDOFF_DIR=/var/lib/blueprint/pubsub-handoffs
  GPU_SPEND_GUARD_DIR=/var/lib/blueprint/gpu-spend-guard
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-now)
      ENABLE_NOW=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "${EUID}" -ne 0 && "${DRY_RUN}" != "true" ]]; then
  echo "Run as root or use --dry-run." >&2
  exit 1
fi

run install -d -m 0755 "${SYSTEMD_DIR}" "${ENV_DIR}" "${HANDOFF_DIR}"
run install -d -m 0750 \
  "${STATE_DIR}" \
  "${STATE_DIR}/robot-eval-job-requests" \
  "${STATE_DIR}/incoming_webapp_job_requests" \
  "${STATE_DIR}/deliveries" \
  "${GPU_SPEND_GUARD_DIR}"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-control-plane.service" \
  "${SYSTEMD_DIR}/blueprint-pipeline-control-plane.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-control-plane.timer" \
  "${SYSTEMD_DIR}/blueprint-pipeline-control-plane.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-intake.service" \
  "${SYSTEMD_DIR}/blueprint-pipeline-intake.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pubsub-handoff-listener.service" \
  "${SYSTEMD_DIR}/blueprint-pubsub-handoff-listener.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pubsub-handoff-listener.timer" \
  "${SYSTEMD_DIR}/blueprint-pubsub-handoff-listener.timer"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-gpu-spend-guard.service" \
  "${SYSTEMD_DIR}/blueprint-gpu-spend-guard.service"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-gpu-spend-guard.timer" \
  "${SYSTEMD_DIR}/blueprint-gpu-spend-guard.timer"

if [[ ! -f "${ENV_FILE}" ]]; then
  run install -m 0600 \
    "${REPO_ROOT}/deploy/systemd/pipeline-control-plane.env.example" \
    "${ENV_FILE}"
  echo "created ${ENV_FILE}; fill secrets before enabling live actions"
else
  echo "kept existing ${ENV_FILE}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

systemctl daemon-reload
if [[ "${ENABLE_NOW}" == "true" ]]; then
  systemctl enable --now blueprint-pipeline-control-plane.timer
  systemctl enable --now blueprint-pubsub-handoff-listener.timer
  systemctl enable --now blueprint-gpu-spend-guard.timer
else
  echo "installed; enable timer with: systemctl enable --now blueprint-pipeline-control-plane.timer"
  echo "enable handoff listener with: systemctl enable --now blueprint-pubsub-handoff-listener.timer"
  echo "enable GPU spend guard with: systemctl enable --now blueprint-gpu-spend-guard.timer"
  echo "start intake service with: systemctl enable --now blueprint-pipeline-intake.service"
fi
