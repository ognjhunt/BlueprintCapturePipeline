#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <pinned-simready-foundation-source-root> <isolated-venv-path> <expected-git-revision>" >&2
  exit 2
fi

source_root="$1"
venv_path="$2"
expected_revision="$3"
requirements_path="${source_root}/nv_core/validator_sample/requirements.txt"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${requirements_path}" ]]; then
  echo "missing pinned SimReady requirements: ${requirements_path}" >&2
  exit 2
fi
if [[ "${venv_path}" == "${repo_root}/.venv" || "${venv_path}" == "${repo_root}/.venv/"* ]]; then
  echo "refusing to install SimReady dependencies into the core repository environment" >&2
  exit 2
fi
actual_revision="$(git -C "${source_root}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${expected_revision}" ]]; then
  echo "source revision mismatch: expected ${expected_revision}, got ${actual_revision}" >&2
  exit 2
fi

python3.12 -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install --requirement "${requirements_path}"
"${venv_path}/bin/simready-validate" --help >/dev/null

echo "isolated SimReady validator environment ready at ${venv_path}"
