#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/AuraFusion360_official"
SAM2_SOURCE_DIR="${SCRIPT_DIR}/sam2_source"
OUTPUT_DIR="${BLUEPRINT_ADP_AURA_INTERIORGS_OUTPUT_DIR:-${SCRIPT_DIR}/../runtime_output}"
RESULT_PATH="${OUTPUT_DIR}/adp_aura_interiorgs_result.json"
mkdir -p "${OUTPUT_DIR}"

write_missing_result() {
  local blocker="$1"
  python3 - "${RESULT_PATH}" "${blocker}" <<'PY'
import datetime as dt, json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    p.write_text(json.dumps({"schema_version":"adp_aura_interiorgs_result.v1","generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"status":"blocked","blockers":[sys.argv[2]],"inpaint_finetune_executed":False,"retry_cap":0,"raw_secret_values_recorded":False},indent=2,sort_keys=True)+"\n")
PY
}

run_with_progress() {
  local stage="$1"
  shift
  echo "BLUEPRINT_ADP_AURA_INTERIORGS_STAGE_STARTED:${stage}"
  "$@" &
  local child_pid=$!
  (
    while kill -0 "${child_pid}" 2>/dev/null; do
      output_bytes="$(du -sk "${OUTPUT_DIR}" 2>/dev/null | awk '{print $1 * 1024}')"
      echo "BLUEPRINT_ADP_AURA_INTERIORGS_RUNTIME_PROGRESS:${stage}:$(date -u +%Y-%m-%dT%H:%M:%SZ):output_bytes=${output_bytes:-0}"
      sleep 60
    done
  ) &
  local progress_pid=$!
  wait "${child_pid}"
  local child_rc=$?
  kill "${progress_pid}" 2>/dev/null || true
  wait "${progress_pid}" 2>/dev/null || true
  echo "BLUEPRINT_ADP_AURA_INTERIORGS_STAGE_FINISHED:${stage}:returncode=${child_rc}"
  return "${child_rc}"
}

rm -rf "${SOURCE_DIR}" "${SAM2_SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}" "${SAM2_SOURCE_DIR}"
python3 -m zipfile -e "${SCRIPT_DIR}/aurafusion360_source.zip" "${SOURCE_DIR}" || { write_missing_result "aurafusion360_interiorgs_source_extract_failed"; exit 2; }
python3 -m zipfile -e "${SCRIPT_DIR}/sam2_source.zip" "${SAM2_SOURCE_DIR}" || { write_missing_result "aurafusion360_interiorgs_sam2_extract_failed"; exit 2; }

python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7 || { write_missing_result "aurafusion360_interiorgs_uv_install_failed"; exit 2; }
UV_BIN="$(command -v uv)"
export UV_NATIVE_TLS=true
"${UV_BIN}" python install 3.10 3.8
"${UV_BIN}" venv "${SOURCE_DIR}/.venv" --python 3.10
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda
AURA_PY="${SOURCE_DIR}/.venv/bin/python"
"${UV_BIN}" pip install --python "${AURA_PY}" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || { write_missing_result "aurafusion360_interiorgs_torch_install_failed"; exit 2; }
"${UV_BIN}" pip install --python "${AURA_PY}" setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0
"${UV_BIN}" pip install --python "${AURA_PY}" --no-build-isolation "${SOURCE_DIR}/submodules/diff-surfel-rasterization" "${SOURCE_DIR}/submodules/simple-knn" "${SAM2_SOURCE_DIR}" || { write_missing_result "aurafusion360_interiorgs_native_install_failed"; exit 2; }
"${UV_BIN}" pip install --python "${AURA_PY}" -r "${SOURCE_DIR}/requirements.txt" || { write_missing_result "aurafusion360_interiorgs_requirements_failed"; exit 2; }

"${UV_BIN}" pip freeze --python "${AURA_PY}" > "${OUTPUT_DIR}/aura-pip-freeze.txt"

export HF_HOME="${SOURCE_DIR}/.hf_home" HF_HUB_CACHE="${SOURCE_DIR}/.hf_home/hub" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600
run_with_progress prepare "${AURA_PY}" "${SCRIPT_DIR}/adp_aura_interiorgs_provider_runner.py" --prepare-only || { write_missing_result "aurafusion360_interiorgs_prepare_failed"; exit 2; }
"${UV_BIN}" venv "${SOURCE_DIR}/LaMa/.venv" --python 3.8
LAMA_PY="${SOURCE_DIR}/LaMa/.venv/bin/python"
"${UV_BIN}" pip install --python "${LAMA_PY}" torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html || { write_missing_result "aurafusion360_interiorgs_lama_torch_failed"; exit 2; }
"${UV_BIN}" pip install --python "${LAMA_PY}" setuptools==57.5.0 wheel==0.37.1 Cython==0.29.37
"${UV_BIN}" pip install --python "${LAMA_PY}" --no-build-isolation pyyaml==5.4.1
"${UV_BIN}" pip install --python "${LAMA_PY}" numpy==1.21.6 tqdm==4.67.1 easydict==1.9.0 scikit-image==0.19.3 scikit-learn==1.0.2 scipy==1.7.3 opencv-python-headless==4.5.5.64 joblib==1.1.1 matplotlib==3.5.3 pandas==1.3.5 albumentations==0.5.2 hydra-core==1.1.0 pytorch-lightning==1.2.9 tabulate==0.9.0 kornia==0.5.0 webdataset==0.1.103 packaging==24.2 tensorboard==2.11.2 || { write_missing_result "aurafusion360_interiorgs_lama_dependencies_failed"; exit 2; }
"${UV_BIN}" pip uninstall --python "${LAMA_PY}" opencv-python || { write_missing_result "aurafusion360_interiorgs_lama_opencv_cleanup_failed"; exit 2; }
"${UV_BIN}" pip install --python "${LAMA_PY}" --reinstall --no-deps opencv-python-headless==4.5.5.64 || { write_missing_result "aurafusion360_interiorgs_lama_opencv_pin_failed"; exit 2; }
"${LAMA_PY}" - <<'PY' || { write_missing_result "aurafusion360_interiorgs_lama_opencv_validation_failed"; exit 2; }
from importlib import metadata

import cv2
import numpy

assert metadata.version("opencv-python-headless") == "4.5.5.64"
try:
    metadata.version("opencv-python")
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("conflicting opencv-python distribution remains installed")
assert cv2.__version__ == "4.5.5"
assert numpy.__version__ == "1.21.6"
PY
"${UV_BIN}" pip freeze --python "${LAMA_PY}" > "${OUTPUT_DIR}/lama-pip-freeze.txt"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_with_progress execute "${AURA_PY}" "${SCRIPT_DIR}/adp_aura_interiorgs_provider_runner.py"
runner_rc=$?
if [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "adp_aura_interiorgs_runner_failed_without_runtime_result"
  echo "blocked_adp_aura_interiorgs_process_exited_without_result:${runner_rc}"
fi
exit "${runner_rc}"
