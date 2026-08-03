#!/usr/bin/env bash
# Create the ISOLATED SkyPilot venv for the pilot lane (C4 step b).
#
# SkyPilot is never installed into the Blueprint runtime environment: a
# dry-run install of skypilot[vast]==0.13.0 into it would pull ~90 packages
# and downgrade Click 8.4->8.1, Uvicorn 0.51->0.35, and Pillow 12.3->12.2.
# The pipeline talks to SkyPilot exclusively through the pinned CLI binary
# this script produces (see src/blueprint_pipeline/skypilot_provisioner.py,
# which shells out and never imports the package).
#
# Usage:
#   scripts/setup_skypilot_venv.sh [venv_dir]
# Then export:
#   export BLUEPRINT_SKYPILOT_BIN="$(pwd)/.venvs/skypilot/bin/sky"
#   export SKYPILOT_DISABLE_USAGE_COLLECTION=1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-${REPO_ROOT}/.venvs/skypilot}"
REQUIREMENTS="${REPO_ROOT}/requirements/skypilot-pilot.txt"

if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "missing ${REQUIREMENTS}" >&2
  exit 2
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS}"

INSTALLED="$("${VENV_DIR}/bin/pip" show skypilot | awk '/^Version:/{print $2}')"
if [[ "${INSTALLED}" != "0.13.0" ]]; then
  echo "skypilot version mismatch: expected 0.13.0, got ${INSTALLED}" >&2
  exit 3
fi

"${VENV_DIR}/bin/sky" --version

cat <<EOF

SkyPilot pilot venv ready (isolated; not part of the runtime SBOM).
Export before running the pilot lane:

  export BLUEPRINT_SKYPILOT_BIN="${VENV_DIR}/bin/sky"
  export SKYPILOT_DISABLE_USAGE_COLLECTION=1

Reminder: mutations still require a paid_resource_allocator admission grant
(resource class skypilot_vast_pilot); the CLI alone cannot spend.
EOF
