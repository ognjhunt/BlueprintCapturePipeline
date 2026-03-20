#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_ROOT}/.env.vast.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.vast.local"
  set +a
fi

COSMOS_OFFICIAL_REPO_URL="${COSMOS_OFFICIAL_REPO_URL:-https://github.com/nvidia-cosmos/cosmos-predict2.git}"
COSMOS_OFFICIAL_REPO_ROOT="${COSMOS_OFFICIAL_REPO_ROOT:-${HOME}/workspace/cosmos-predict2.5}"
COSMOS_OFFICIAL_REPO_REF="${COSMOS_OFFICIAL_REPO_REF:-main}"
COSMOS_OFFICIAL_REPO_UV_EXTRA="${COSMOS_OFFICIAL_REPO_UV_EXTRA:-cu128}"
COSMOS_WORKER_PYTHON_BIN="${COSMOS_WORKER_PYTHON_BIN:-${COSMOS_OFFICIAL_REPO_ROOT}/.venv/bin/python}"

log() {
  echo "[bootstrap-cosmos-repo] $*"
}

die() {
  echo "[bootstrap-cosmos-repo] ERROR: $*" >&2
  exit 1
}

clone_or_update_repo() {
  local repo_url="$1"
  local dst="$2"
  local ref="$3"

  mkdir -p "$(dirname "$dst")"
  if [ ! -d "$dst/.git" ]; then
    git clone "$repo_url" "$dst"
  fi
  git -C "$dst" fetch --tags origin
  git -C "$dst" checkout "$ref"
  if [ "$ref" = "main" ] || [ "$ref" = "master" ]; then
    git -C "$dst" pull --ff-only origin "$ref"
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --no-cache-dir uv
fi

clone_or_update_repo "${COSMOS_OFFICIAL_REPO_URL}" "${COSMOS_OFFICIAL_REPO_ROOT}" "${COSMOS_OFFICIAL_REPO_REF}"

log "Syncing ${COSMOS_OFFICIAL_REPO_ROOT} with uv extra ${COSMOS_OFFICIAL_REPO_UV_EXTRA}..."
(
  cd "${COSMOS_OFFICIAL_REPO_ROOT}"
  uv sync --extra "${COSMOS_OFFICIAL_REPO_UV_EXTRA}"
)

[ -x "${COSMOS_WORKER_PYTHON_BIN}" ] || die "Expected worker python at ${COSMOS_WORKER_PYTHON_BIN}"

log "Validating official Cosmos repo runtime imports..."
PYTHONPATH="${COSMOS_OFFICIAL_REPO_ROOT}:${COSMOS_OFFICIAL_REPO_ROOT}/packages/cosmos-oss${PYTHONPATH:+:${PYTHONPATH}}" \
  "${COSMOS_WORKER_PYTHON_BIN}" - <<'PY'
from cosmos_oss.init import init_environment
from cosmos_predict2.config import SetupArguments
from cosmos_predict2.inference import Inference

print("COSMOS_OFFICIAL_REPO_READY", SetupArguments.__name__, Inference.__name__)
PY

log "Official Cosmos repo worker environment is ready."
