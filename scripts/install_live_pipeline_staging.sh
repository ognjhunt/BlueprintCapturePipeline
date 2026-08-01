#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
ENV_DIR="${ENV_DIR:-/etc/blueprint}"
ENV_FILE="${ENV_FILE:-${ENV_DIR}/pipeline-intake-staging.env}"
STATE_DIR="${STATE_DIR:-/var/lib/blueprint-staging}"
STAGING_REPO="${STAGING_REPO:-/opt/blueprint/BlueprintCapturePipeline-staging}"
SERVICE_USER="${SERVICE_USER:-blueprint}"
SERVICE_GROUP="${SERVICE_GROUP:-blueprint}"
ENABLE_NOW=false
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage: scripts/install_live_pipeline_staging.sh [--enable-now] [--dry-run]

Installs only the isolated staging intake service. It never copies production
secrets, enables Pub/Sub, invokes a provider, or changes the production unit.
The staging checkout must be clean and exactly match its origin/main.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-now) ENABLE_NOW=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
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

if [[ ! -d "${STAGING_REPO}/.git" ]]; then
  echo "staging checkout is missing: ${STAGING_REPO}" >&2
  exit 1
fi
staging_head="$(git -C "${STAGING_REPO}" rev-parse HEAD)"
staging_origin_main="$(git -C "${STAGING_REPO}" rev-parse origin/main)"
if [[ "${staging_head}" != "${staging_origin_main}" ]]; then
  echo "staging checkout HEAD must equal origin/main" >&2
  exit 1
fi
if [[ -n "$(git -C "${STAGING_REPO}" status --porcelain)" ]]; then
  echo "staging checkout must be clean" >&2
  exit 1
fi

run install -d -m 0755 "${SYSTEMD_DIR}"
run install -d -m 0750 -o root -g "${SERVICE_GROUP}" "${ENV_DIR}"
run install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
  "${STATE_DIR}" \
  "${STATE_DIR}/pipeline-intake" \
  "${STATE_DIR}/pipeline-intake/incoming" \
  "${STATE_DIR}/pipeline-intake/nonces" \
  "${STATE_DIR}/capture-intakes"
run install -m 0644 \
  "${REPO_ROOT}/deploy/systemd/blueprint-pipeline-intake-staging.service" \
  "${SYSTEMD_DIR}/blueprint-pipeline-intake-staging.service"

if [[ ! -f "${ENV_FILE}" ]]; then
  run install -o root -g "${SERVICE_GROUP}" -m 0640 \
    "${REPO_ROOT}/deploy/systemd/pipeline-intake-staging.env.example" \
    "${ENV_FILE}"
  echo "created ${ENV_FILE}; set an exact source commit and staging-only secret"
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
  systemctl enable --now blueprint-pipeline-intake-staging.service
else
  echo "installed; enable after filling ${ENV_FILE}:"
  echo "  systemctl enable --now blueprint-pipeline-intake-staging.service"
fi
