#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Cosmos 3 Edge requires a supported Linux/NVIDIA worker" >&2
  exit 2
fi
if [[ $# -ne 4 ]]; then
  echo "usage: $0 <cosmos-source-root> <venv-path> <expected-git-revision> <cu128-train|cu130-train>" >&2
  exit 2
fi

source_root="$(cd "$1" && pwd)"
edge_env="$2"
expected_revision="$3"
cuda_group="$4"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$edge_env" == "$repo_root/.venv" || "$edge_env" == "$repo_root/.venv/"* ]]; then
  echo "Cosmos dependencies must not use the core .venv" >&2
  exit 2
fi
if [[ "$cuda_group" != "cu128-train" && "$cuda_group" != "cu130-train" ]]; then
  echo "unsupported CUDA dependency group: $cuda_group" >&2
  exit 2
fi
actual_revision="$(git -C "$source_root" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "source revision mismatch: expected $expected_revision, got $actual_revision" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build the isolated Cosmos environment" >&2
  exit 2
fi

uv venv --python 3.12 "$edge_env"
VIRTUAL_ENV="$edge_env" uv sync \
  --project "$source_root" \
  --active \
  --all-extras \
  --group "$cuda_group"
"$edge_env/bin/python" -m cosmos_framework.scripts.inference --help >/dev/null
"$edge_env/bin/python" -m pip install "nvidia-ml-py>=12,<14"
echo "created pinned Cosmos 3 Edge environment at $edge_env"
