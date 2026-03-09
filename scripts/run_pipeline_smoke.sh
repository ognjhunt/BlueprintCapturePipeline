#!/usr/bin/env bash
#
# Fast wiring smoke test for the wrapper. Uses tiny synthetic artifacts and
# best-effort fallback so it completes in seconds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

SMOKE_ROOT="${1:-/tmp/bcp_smoke_$(date +%Y%m%d_%H%M%S)}"
INPUT_VIDEO="${SMOKE_ROOT}/input_smoke.mov"
NUREC_FIXTURE_DIR="${SMOKE_ROOT}/nurec_fixture"

echo "[smoke] root=${SMOKE_ROOT}"
mkdir -p "${NUREC_FIXTURE_DIR}"
printf 'smoke-video\n' > "${INPUT_VIDEO}"

printf 'usdz' > "${NUREC_FIXTURE_DIR}/export_last.usdz"
printf 'ply' > "${NUREC_FIXTURE_DIR}/nvblox_mesh.ply"
printf 'glb' > "${NUREC_FIXTURE_DIR}/visual_mesh.glb"
printf 'occ' > "${NUREC_FIXTURE_DIR}/occupancy.bin"
cat > "${NUREC_FIXTURE_DIR}/mesh_manifest.json" <<'JSON'
{
  "primary_visual_asset": "export_last.usdz"
}
JSON
cat > "${NUREC_FIXTURE_DIR}/object_point_cloud_index.json" <<'JSON'
{
  "environment": "default",
  "objects": []
}
JSON
printf 'poisson_open3d\n' > "${NUREC_FIXTURE_DIR}/mesh_method.txt"

TEXT_ASSET_GENERATION_PROVIDER_CHAIN="sam3d" \
TEXT_SAM3D_API_HOST="" \
TEXT_SAM3D_API_KEY="" \
bash "${APP_DIR}/scripts/run_full_pipeline.sh" \
  --completion-mode best_effort \
  --skip-nurec \
  --workspace "${SMOKE_ROOT}" \
  --nurec-output-dir "${NUREC_FIXTURE_DIR}" \
  "${INPUT_VIDEO}"

for required in \
  "${SMOKE_ROOT}/full_pipeline/run_summary.json" \
  "${SMOKE_ROOT}/full_pipeline/run_summary.md" \
  "${SMOKE_ROOT}/full_pipeline/log_summary.json" \
  "${SMOKE_ROOT}/full_pipeline/log_summary.md"; do
  if [ ! -f "${required}" ]; then
    echo "[smoke] missing expected artifact: ${required}" >&2
    exit 1
  fi
done

echo "[smoke] success"
echo "[smoke] summary: ${SMOKE_ROOT}/full_pipeline/run_summary.json"
