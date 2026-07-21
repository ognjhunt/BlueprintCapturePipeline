#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ovrtx and ovphysx canaries require a supported Linux worker" >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <ovrtx-venv-path> <ovphysx-venv-path>" >&2
  exit 2
fi

ovrtx_env="$1"
ovphysx_env="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for candidate in "$ovrtx_env" "$ovphysx_env"; do
  if [[ "$candidate" == "$repo_root/.venv" || "$candidate" == "$repo_root/.venv/"* ]]; then
    echo "Omniverse prerelease dependencies must not use the core .venv" >&2
    exit 2
  fi
done

python3.12 -m venv "$ovrtx_env"
"$ovrtx_env/bin/python" -m pip install --upgrade pip
"$ovrtx_env/bin/python" -m pip install \
  "ovrtx==0.4.0.346409" \
  "ovstage==0.1.0.346039" \
  "numpy>=1.26,<3" \
  "Pillow>=10,<13" \
  "nvidia-ml-py>=12,<14"
"$ovrtx_env/bin/python" -c 'import ovrtx, ovstage; print(ovrtx.__version__)'

python3.12 -m venv "$ovphysx_env"
"$ovphysx_env/bin/python" -m pip install --upgrade pip
"$ovphysx_env/bin/python" -m pip install \
  "ovphysx==0.4.13" \
  "numpy>=1.26,<3" \
  "nvidia-ml-py>=12,<14"
"$ovphysx_env/bin/python" -c 'import ovphysx; print(ovphysx.__version__)'

echo "created isolated ovrtx and ovphysx environments"
