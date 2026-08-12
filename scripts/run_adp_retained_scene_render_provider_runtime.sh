#!/usr/bin/env bash
# Run one immutable retained-scene GPU render bundle without network installs.
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${BLUEPRINT_ADP_RETAINED_SCENE_RENDER_OUTPUT_DIR:-${BLUEPRINT_ADP_GAUSSIAN_EXCISION_OUTPUT_DIR:-${runtime_dir}/runtime_output}}"
mkdir -p "${output_dir}"

if [[ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-}" == "1" ]]; then
  node "${runtime_dir}/adp_retained_scene_render_provider_runner.mjs" \
    --runtime "${runtime_dir}" \
    --output "${output_dir}" \
    --rehearsal
else
  node "${runtime_dir}/adp_retained_scene_render_provider_runner.mjs" \
    --runtime "${runtime_dir}" \
    --output "${output_dir}"
fi
