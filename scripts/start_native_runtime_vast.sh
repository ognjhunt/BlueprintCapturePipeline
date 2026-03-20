#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.env.vast.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.vast.local"
  set +a
fi

export GCS_ROOT="${GCS_ROOT:-/root/blueprint-storage}"
export PIPELINE_BUCKET="${PIPELINE_BUCKET:-vast-local}"
export COSMOS_OFFICIAL_REPO_ROOT="${COSMOS_OFFICIAL_REPO_ROOT:-${HOME}/workspace/cosmos-predict2.5}"
export COSMOS_WORKER_PYTHON_BIN="${COSMOS_WORKER_PYTHON_BIN:-${COSMOS_OFFICIAL_REPO_ROOT}/.venv/bin/python}"
export COSMOS_DISABLE_GUARDRAILS="${COSMOS_DISABLE_GUARDRAILS:-1}"
export NATIVE_WORLD_MODEL_SYNTHESIS_MODE="${NATIVE_WORLD_MODEL_SYNTHESIS_MODE:-cosmos_i2w}"
export SITE_WORLD_RUNTIME_SERVICE_PORT="${SITE_WORLD_RUNTIME_SERVICE_PORT:-8791}"
export COSMOS_CHUNK_OVERLAP="${COSMOS_CHUNK_OVERLAP:-4}"
export COSMOS_CHUNK_SIZE="${COSMOS_CHUNK_SIZE:-33}"

if [ ! -x "${COSMOS_WORKER_PYTHON_BIN}" ]; then
  echo "[start-native-runtime-vast] ERROR: expected worker python at ${COSMOS_WORKER_PYTHON_BIN}" >&2
  echo "[start-native-runtime-vast] Run ./scripts/bootstrap_cosmos_official_repo.sh first." >&2
  exit 1
fi

if [ ! -x "${REPO_ROOT}/.venv/bin/python" ]; then
  echo "[start-native-runtime-vast] ERROR: expected repo venv at ${REPO_ROOT}/.venv/bin/python" >&2
  echo "[start-native-runtime-vast] Run ./scripts/install_ml_stack.sh first." >&2
  exit 1
fi

source "${REPO_ROOT}/.venv/bin/activate"
exec python -m blueprint_pipeline.native_runtime_service
