#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/AuraFusion360_official"
SAM2_SOURCE_DIR="${SCRIPT_DIR}/sam2_source"
OUTPUT_DIR="${BLUEPRINT_ADP_AURA_OUTPUT_DIR:-${SCRIPT_DIR}/../runtime_output}"
RESULT_PATH="${OUTPUT_DIR}/adp_aura_author_smoke_result.json"
mkdir -p "${OUTPUT_DIR}"

write_missing_result() {
  local blocker="$1"
  python3 - "${RESULT_PATH}" "${blocker}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({
        "schema_version": "adp_aura_author_smoke_result.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "blocked",
        "blockers": [sys.argv[2]],
        "inpaint_init_executed": False,
        "author_source_modified": False,
        "published_expected_output_bound": False,
        "depth_anything3_used": False,
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
python3 -m zipfile -e "${SCRIPT_DIR}/aurafusion360_source.zip" "${SOURCE_DIR}"
unzip_rc=$?
if [ "${unzip_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_source_archive_extract_failed"
  exit "${unzip_rc}"
fi
rm -rf "${SAM2_SOURCE_DIR}"
mkdir -p "${SAM2_SOURCE_DIR}"
python3 -m zipfile -e "${SCRIPT_DIR}/sam2_source.zip" "${SAM2_SOURCE_DIR}"
sam2_unzip_rc=$?
if [ "${sam2_unzip_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_sam2_source_archive_extract_failed"
  exit "${sam2_unzip_rc}"
fi

python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7
uv_rc=$?
if [ "${uv_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_uv_install_failed"
  exit "${uv_rc}"
fi
UV_BIN="$(command -v uv)"
"${UV_BIN}" python install 3.10
"${UV_BIN}" venv "${SOURCE_DIR}/.venv" --python 3.10
venv_rc=$?
if [ "${venv_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_python310_venv_failed"
  exit "${venv_rc}"
fi

PYTHON="${SOURCE_DIR}/.venv/bin/python"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_HOME=/usr/local/cuda
"${UV_BIN}" pip install --python "${PYTHON}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
torch_rc=$?
if [ "${torch_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_torch_cuda124_install_failed"
  exit "${torch_rc}"
fi
"${UV_BIN}" pip install --python "${PYTHON}" \
  setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0
build_tools_rc=$?
if [ "${build_tools_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_build_tooling_install_failed"
  exit "${build_tools_rc}"
fi
"${UV_BIN}" pip install --python "${PYTHON}" --no-build-isolation \
  "${SOURCE_DIR}/submodules/diff-surfel-rasterization" \
  "${SOURCE_DIR}/submodules/simple-knn" \
  "${SAM2_SOURCE_DIR}"
native_install_rc=$?
if [ "${native_install_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_native_dependency_install_failed"
  exit "${native_install_rc}"
fi
"${UV_BIN}" pip install --python "${PYTHON}" -r "${SOURCE_DIR}/requirements.txt"
requirements_rc=$?
if [ "${requirements_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_python_requirements_install_failed"
  exit "${requirements_rc}"
fi
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${OUTPUT_DIR}/pip-freeze.txt"

export HF_HOME="${SOURCE_DIR}/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
"${PYTHON}" "${SCRIPT_DIR}/adp_aura_author_smoke_provider_runner.py" --prepare-only
prepare_rc=$?
if [ "${prepare_rc}" -ne 0 ]; then
  write_missing_result "aurafusion360_pinned_inputs_prepare_failed"
  exit "${prepare_rc}"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"${PYTHON}" "${SCRIPT_DIR}/adp_aura_author_smoke_provider_runner.py"
runner_rc=$?

if [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "adp_aura_smoke_runner_failed_without_runtime_result"
  echo "blocked_adp_aura_smoke_process_exited_without_result:${runner_rc}"
fi
exit "${runner_rc}"
