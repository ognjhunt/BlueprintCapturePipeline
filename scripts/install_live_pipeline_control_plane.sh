#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
ENV_DIR="${ENV_DIR:-/etc/blueprint}"
ENV_FILE="${ENV_FILE:-${ENV_DIR}/pipeline-control-plane.env}"
STATE_DIR="${STATE_DIR:-/var/lib/blueprint/pipeline-control-plane}"
HANDOFF_DIR="${HANDOFF_DIR:-/var/lib/blueprint/pubsub-handoffs}"
PROVIDER_SECRETS_DIR="${PROVIDER_SECRETS_DIR:-${ENV_DIR}/provider-secrets}"
SERVICE_USER="${SERVICE_USER:-blueprint}"
SERVICE_GROUP="${SERVICE_GROUP:-blueprint}"
ENABLE_NOW=false
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: scripts/install_live_pipeline_control_plane.sh [--enable-now] [--dry-run]

Installs the Blueprint live pipeline control-plane systemd service/timer, the
capture handoff Pub/Sub listener service/timer, and the optional authenticated
WebApp intake service unit, plus the paid-work spend admission guard service/
timer. The spend guard remains locked until current external billing input and
provider credentials are installed.
The service runs one safe control-plane pass on each timer tick:
read env, audit readiness, optionally consume the robot-eval job inbox, write
manifests, run the proof-boundary audit, and exit. It does not add secrets or
enable live simulator/provider actions by itself.

Environment overrides:
  SYSTEMD_DIR=/etc/systemd/system
  ENV_DIR=/etc/blueprint
  ENV_FILE=/etc/blueprint/pipeline-control-plane.env
  STATE_DIR=/var/lib/blueprint/pipeline-control-plane
  HANDOFF_DIR=/var/lib/blueprint/pubsub-handoffs
  PROVIDER_SECRETS_DIR=/etc/blueprint/provider-secrets
  SERVICE_USER=blueprint
  SERVICE_GROUP=blueprint
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

if ! getent group "${SERVICE_GROUP}" >/dev/null 2>&1; then
  run groupadd --system "${SERVICE_GROUP}"
fi
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  run useradd --system --gid "${SERVICE_GROUP}" --home-dir /nonexistent \
    --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

run install -d -m 0755 "${SYSTEMD_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" "${ENV_DIR}"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${HANDOFF_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" \
  "${PROVIDER_SECRETS_DIR}"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${STATE_DIR}" \
  "${STATE_DIR}/robot-eval-job-requests" \
  "${STATE_DIR}/incoming_webapp_job_requests" \
  "${STATE_DIR}/deliveries" \
  "${STATE_DIR}/gpu_spend_guard"
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
  run install -o root -g "${SERVICE_GROUP}" -m 0640 \
    "${REPO_ROOT}/deploy/systemd/pipeline-control-plane.env.example" \
    "${ENV_FILE}"
  echo "created ${ENV_FILE}; fill secrets before enabling live actions"
else
  run chown root:"${SERVICE_GROUP}" "${ENV_FILE}"
  run chmod 0640 "${ENV_FILE}"
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
  echo "enable spend admission guard with: systemctl enable --now blueprint-gpu-spend-guard.timer"
  echo "start intake service with: systemctl enable --now blueprint-pipeline-intake.service"
fi
