#!/usr/bin/env bash
# Create the ISOLATED nerfstudio/gsplat venv for one splatfacto bakeoff arm.
#
# Usage:
#   scripts/setup_splatfacto_venv.sh g1   # Splatfacto (nerfstudio==1.1.5, default strategy)
#   scripts/setup_splatfacto_venv.sh g2   # Splatfacto-MCMC (nerfstudio git pin)
#
# The arm venvs are never the Blueprint runtime environment and are absent
# from the runtime SBOM/license policy; pins are reviewed in
# docs/architecture/isolated-component-license-inventory.md. gsplat==1.4.0
# builds CUDA kernels at first use — run on the Linux GPU worker, not on a
# laptop. No paid resources are touched by this script.

set -euo pipefail

ARM="${1:-}"
case "${ARM}" in
  g1|g2) ;;
  *) echo "usage: $0 {g1|g2} [venv_dir]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${2:-${REPO_ROOT}/.venvs/splatfacto-${ARM}}"
REQUIREMENTS="${REPO_ROOT}/requirements/splatfacto-arm-${ARM}.txt"

if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "missing ${REQUIREMENTS}" >&2
  exit 2
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS}"

GSPLAT_VERSION="$("${VENV_DIR}/bin/pip" show gsplat | awk '/^Version:/{print $2}')"
if [[ "${GSPLAT_VERSION}" != "1.4.0" ]]; then
  echo "gsplat version mismatch: expected 1.4.0, got ${GSPLAT_VERSION}" >&2
  exit 3
fi

"${VENV_DIR}/bin/ns-train" --help >/dev/null

cat <<EOF

Splatfacto arm ${ARM} venv ready: ${VENV_DIR}
The worker execution receipt must record: exact argv, 'pip freeze' output,
and durations. Arm intent (strategy, seed, iterations, dataset digest) is
pinned by provider_packets/splatfacto/splatfacto_execution_packet.v1.json.
EOF
