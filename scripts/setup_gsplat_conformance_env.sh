#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <usd-convert-gsplat-source-root> <venv-path> <expected-git-revision>" >&2
  exit 2
fi

source_root="$(cd "$1" && pwd)"
oracle_env="$2"
expected_revision="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$oracle_env" == "$repo_root/.venv" || "$oracle_env" == "$repo_root/.venv/"* ]]; then
  echo "the optional converter oracle must not use the core .venv" >&2
  exit 2
fi
actual_revision="$(git -C "$source_root" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "source revision mismatch: expected $expected_revision, got $actual_revision" >&2
  exit 2
fi
if [[ ! -f "$source_root/source/python/pyproject.toml" ]]; then
  echo "usd-convert-gsplat Python package is missing; run the upstream build/subst step" >&2
  exit 2
fi

python3.12 -m venv "$oracle_env"
"$oracle_env/bin/python" -m pip install --upgrade pip
"$oracle_env/bin/python" -m pip install "$source_root/source/python[usd]"
"$oracle_env/bin/python" -m usd_convert_gsplat --help >/dev/null
echo "created pinned usd-convert-gsplat conformance environment at $oracle_env"
