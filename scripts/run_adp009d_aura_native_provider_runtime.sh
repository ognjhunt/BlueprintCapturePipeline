#!/usr/bin/env bash
set -u

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${RUNTIME_DIR}/AuraFusion360_official"
OUTPUT_DIR="${BLUEPRINT_ADP009D_AURA_NATIVE_OUTPUT_DIR:-${RUNTIME_DIR}/../runtime_output}"
RESULT_PATH="${OUTPUT_DIR}/adp009d_aura_native_live_camera_result.json"
mkdir -p "${OUTPUT_DIR}"

write_missing_result() {
  local blocker="$1"
  python3 - "${RESULT_PATH}" "${blocker}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    p.write_text(json.dumps({
        "schema_version": "adp009d_aura_native_live_camera_result.v1",
        "status": "blocked",
        "blockers": [sys.argv[2]],
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
python3 -m zipfile -e "${RUNTIME_DIR}/aurafusion360_source.zip" "${SOURCE_DIR}" || { write_missing_result "aura_native_source_extract_failed"; exit 2; }
python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7 || { write_missing_result "aura_native_uv_install_failed"; exit 2; }
UV_BIN="$(command -v uv)"
export UV_NATIVE_TLS=true
"${UV_BIN}" python install 3.10 || { write_missing_result "aura_native_python_install_failed"; exit 2; }
"${UV_BIN}" venv "${SOURCE_DIR}/.venv" --python 3.10 || { write_missing_result "aura_native_venv_failed"; exit 2; }
AURA_PY="${SOURCE_DIR}/.venv/bin/python"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda
"${UV_BIN}" pip install --python "${AURA_PY}" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || { write_missing_result "aura_native_torch_install_failed"; exit 2; }
"${UV_BIN}" pip install --python "${AURA_PY}" setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0 || { write_missing_result "aura_native_build_dependencies_failed"; exit 2; }
"${UV_BIN}" pip install --python "${AURA_PY}" --no-build-isolation "${SOURCE_DIR}/submodules/diff-surfel-rasterization" "${SOURCE_DIR}/submodules/simple-knn" || { write_missing_result "aura_native_cuda_extensions_failed"; exit 2; }
"${UV_BIN}" pip install --python "${AURA_PY}" numpy==1.26.4 pillow==10.2.0 plyfile==1.1.3 opencv-python-headless==4.11.0.86 matplotlib==3.8.0 || { write_missing_result "aura_native_runtime_dependencies_failed"; exit 2; }
"${UV_BIN}" pip freeze --python "${AURA_PY}" > "${OUTPUT_DIR}/aura-native-pip-freeze.txt"
"${AURA_PY}" "${RUNTIME_DIR}/adp009d_aura_native_provider_runner.py" --runtime-dir "${RUNTIME_DIR}" --output-dir "${OUTPUT_DIR}"
runner_rc=$?
if [ ${runner_rc} -ne 0 ] && [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "aura_native_runner_failed_without_result"
fi
exit ${runner_rc}
