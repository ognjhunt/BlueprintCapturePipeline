#!/usr/bin/env bash
set -u

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${RUNTIME_DIR}/FlashSplat"
OUTPUT_DIR="${BLUEPRINT_ADP_GAUSSIAN_EXCISION_OUTPUT_DIR:-${RUNTIME_DIR}/../runtime_output}"
RESULT_PATH="${OUTPUT_DIR}/adp009b_gaussian_excision_result.json"
mkdir -p "${OUTPUT_DIR}"

write_missing_result() {
  local blocker="$1"
  python3 - "${RESULT_PATH}" "${blocker}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({
        "schema_version": "adp009b_gaussian_excision_result.v1",
        "status": "blocked",
        "blockers": [sys.argv[2]],
        "released_code_executed": False,
        "heldout_cameras_accessed_for_classification": False,
        "provider_zero_required_after_return": True,
        "depth_anything_3_used": False,
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
python3 "${RUNTIME_DIR}/provider_archive.py" \
  "${RUNTIME_DIR}/flashsplat_source.zip" "${SOURCE_DIR}" \
  --receipt "${OUTPUT_DIR}/flashsplat_source_extraction.json" || {
  write_missing_result "gaussian_excision_source_archive_extract_failed"; exit 2;
}
if [ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-0}" = "1" ]; then
  python3 - "${OUTPUT_DIR}/provider_bundle_rehearsal.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "provider_bundle_entrypoint_rehearsal.v1",
    "status": "passed",
    "entrypoint": "run_adp_gaussian_excision_provider_runtime.sh",
    "archive_extraction_executed": True,
    "gpu_runtime_started": False,
    "paid_inference_performed": False,
    "provider_mutations_performed": 0,
    "stopped_before": "dependency_install_and_cuda_execution",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit 0
fi
python3 - <<'PY' || { write_missing_result "gaussian_excision_base_image_invalid"; exit 2; }
import sys
import torch
import torchvision

assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split("+")[0] == "2.5.1"
assert torchvision.__version__.split("+")[0] == "0.20.1"
assert torch.version.cuda == "12.4"
PY
python3 -m venv --system-site-packages "${SOURCE_DIR}/.venv" || {
  write_missing_result "gaussian_excision_venv_failed"; exit 2;
}
RUNTIME_PY="${SOURCE_DIR}/.venv/bin/python"
export CUDA_HOME=/usr/local/cuda
export PIP_NO_INDEX=1
"${RUNTIME_PY}" -m pip install --disable-pip-version-check --no-index --no-deps \
  --find-links "${RUNTIME_DIR}/dependency_wheelhouse" \
  setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0 \
  numpy==1.26.4 pillow==10.2.0 plyfile==1.1.3 \
  opencv-python-headless==4.11.0.86 || {
  write_missing_result "gaussian_excision_offline_wheelhouse_install_failed"; exit 2;
}
"${RUNTIME_PY}" -m pip install --disable-pip-version-check --no-index --no-build-isolation \
  "${SOURCE_DIR}/submodules/diff-gaussian-rasterization" \
  "${SOURCE_DIR}/submodules/flashsplat-rasterization" \
  "${SOURCE_DIR}/submodules/simple-knn" || {
  write_missing_result "gaussian_excision_cuda_extensions_failed"; exit 2;
}
"${RUNTIME_PY}" -m pip freeze > "${OUTPUT_DIR}/pip-freeze.txt"
"${RUNTIME_PY}" "${RUNTIME_DIR}/adp_gaussian_excision_provider_runner.py" \
  --runtime-dir "${RUNTIME_DIR}" --source-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}"
runner_rc=$?
if [ ${runner_rc} -ne 0 ] && [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "gaussian_excision_runner_failed_without_runtime_result"
  echo "blocked_gaussian_excision_process_exited_without_result:${runner_rc}"
fi
exit ${runner_rc}
