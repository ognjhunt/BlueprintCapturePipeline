#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/Inpaint360GS"
PACKET_DIR="${SCRIPT_DIR}/interiorgs_adapter"
OUTPUT_DIR="${BLUEPRINT_ADP_INPAINT360_OUTPUT_DIR:-${SCRIPT_DIR}/../runtime_output}"
RESULT_PATH="${OUTPUT_DIR}/adp_inpaint360_interiorgs_result.json"
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
        "schema_version": "adp_inpaint360_interiorgs_result.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "blocked",
        "blockers": [sys.argv[2]],
        "mask_association_executed": False,
        "virtual_masks_materialized": False,
        "lama_color_executed": False,
        "lama_depth_executed": False,
        "inpaint_3d_executed": False,
        "source_modified": False,
        "rendered_frames_have_no_hidden_background_truth": True,
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

rm -rf "${SOURCE_DIR}" "${PACKET_DIR}"
mkdir -p "${SOURCE_DIR}" "${PACKET_DIR}"
tar -xf "${SCRIPT_DIR}/inpaint360gs_source.tar" -C "${SOURCE_DIR}"
source_rc=$?
python3 -m zipfile -e "${SCRIPT_DIR}/lama_training_data.zip" "${SOURCE_DIR}/LaMa"
lama_source_rc=$?
python3 -m zipfile -e "${SCRIPT_DIR}/interiorgs_adapter.zip" "${PACKET_DIR}"
packet_rc=$?
if [ "${source_rc}" -ne 0 ] || [ "${lama_source_rc}" -ne 0 ] || [ "${packet_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_runtime_archive_extract_failed"
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-pip build-essential git ca-certificates \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
system_deps_rc=$?
if [ "${system_deps_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_system_dependencies_install_failed"
  exit "${system_deps_rc}"
fi
dpkg-query -W > "${OUTPUT_DIR}/dpkg-query.txt"

python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7
uv_rc=$?
if [ "${uv_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_uv_install_failed"
  exit "${uv_rc}"
fi
UV_BIN="$(command -v uv)"
export UV_NATIVE_TLS=true
"${UV_BIN}" python install 3.10 3.8
"${UV_BIN}" venv "${SOURCE_DIR}/.venv" --python 3.10
"${UV_BIN}" venv "${SOURCE_DIR}/LaMa/.venv" --python 3.8
venv_rc=$?
if [ "${venv_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_two_environment_creation_failed"
  exit "${venv_rc}"
fi

MAIN_PY="${SOURCE_DIR}/.venv/bin/python"
LAMA_PY="${SOURCE_DIR}/LaMa/.venv/bin/python"
export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
"${UV_BIN}" pip install --python "${MAIN_PY}" \
  torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
main_torch_rc=$?
if [ "${main_torch_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_main_torch_cu118_install_failed"
  exit "${main_torch_rc}"
fi
"${UV_BIN}" pip install --python "${MAIN_PY}" setuptools==75.8.0 wheel==0.45.1 ninja==1.13.0 packaging==24.2
"${UV_BIN}" pip install --python "${MAIN_PY}" --no-build-isolation \
  "${SOURCE_DIR}/gaussian_splatting/submodules/diff-gaussian-rasterization" \
  "${SOURCE_DIR}/gaussian_splatting/submodules/simple-knn" \
  "${SOURCE_DIR}/submodules/diff-gaussian-rasterization"
native_rc=$?
if [ "${native_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_cuda_extension_install_failed"
  exit "${native_rc}"
fi
"${UV_BIN}" pip install --python "${MAIN_PY}" --no-deps --no-build-isolation "${SOURCE_DIR}"
"${UV_BIN}" pip install --python "${MAIN_PY}" \
  numpy==1.26.4 plyfile==1.1.2 tqdm==4.67.1 scipy==1.13.1 \
  opencv-python-headless==4.10.0.84 scikit-learn==1.4.2 lpips==0.1.4 \
  torchmetrics==1.2.1 wandb==0.19.11 imageio==2.37.0 open3d==0.18.0 \
  timm==1.0.15 pytorch-fid==0.3.0 pillow==11.1.0
main_deps_rc=$?
if [ "${main_deps_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_main_dependencies_install_failed"
  exit "${main_deps_rc}"
fi

"${UV_BIN}" pip install --python "${LAMA_PY}" \
  torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
lama_torch_rc=$?
if [ "${lama_torch_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_lama_torch_install_failed"
  exit "${lama_torch_rc}"
fi
"${UV_BIN}" pip install --python "${LAMA_PY}" \
  setuptools==57.5.0 wheel==0.37.1 Cython==0.29.37
"${UV_BIN}" pip install --python "${LAMA_PY}" --no-build-isolation pyyaml==5.4.1
lama_build_rc=$?
if [ "${lama_build_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_lama_legacy_build_dependencies_failed"
  exit "${lama_build_rc}"
fi
"${UV_BIN}" pip install --python "${LAMA_PY}" \
  numpy==1.21.6 tqdm==4.67.1 easydict==1.9.0 \
  scikit-image==0.19.3 scikit-learn==1.0.2 scipy==1.7.3 \
  opencv-python-headless==4.5.5.64 joblib==1.1.1 matplotlib==3.5.3 \
  pandas==1.3.5 albumentations==0.5.2 hydra-core==1.1.0 \
  pytorch-lightning==1.2.9 tabulate==0.9.0 kornia==0.5.0 \
  webdataset==0.1.103 packaging==24.2 tensorboard==2.11.2
lama_deps_rc=$?
if [ "${lama_deps_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_lama_dependencies_install_failed"
  exit "${lama_deps_rc}"
fi

# albumentations 0.5.2 permits the non-headless OpenCV distribution as a
# transitive dependency.  Keeping both distributions in one environment can
# leave cv2 assembled from incompatible releases, so restore the exact
# headless release after the complete legacy dependency solve.
"${UV_BIN}" pip uninstall --python "${LAMA_PY}" opencv-python
lama_opencv_cleanup_rc=$?
"${UV_BIN}" pip install --python "${LAMA_PY}" --reinstall --no-deps \
  opencv-python-headless==4.5.5.64
lama_opencv_pin_rc=$?
if [ "${lama_opencv_cleanup_rc}" -ne 0 ] || [ "${lama_opencv_pin_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_lama_opencv_conflict_resolution_failed"
  exit 2
fi
"${LAMA_PY}" - <<'PY'
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
lama_opencv_validation_rc=$?
if [ "${lama_opencv_validation_rc}" -ne 0 ]; then
  write_missing_result "inpaint360_lama_opencv_runtime_validation_failed"
  exit "${lama_opencv_validation_rc}"
fi

python3 -m zipfile -e "${SCRIPT_DIR}/big-lama.zip" "${SOURCE_DIR}/LaMa"
lama_model_rc=$?
if [ "${lama_model_rc}" -ne 0 ] || [ ! -f "${SOURCE_DIR}/LaMa/big-lama/config.yaml" ]; then
  write_missing_result "inpaint360_big_lama_extract_failed"
  exit 2
fi
"${UV_BIN}" pip freeze --python "${MAIN_PY}" > "${OUTPUT_DIR}/main-pip-freeze.txt"
"${UV_BIN}" pip freeze --python "${LAMA_PY}" > "${OUTPUT_DIR}/lama-pip-freeze.txt"

"${MAIN_PY}" "${SCRIPT_DIR}/adp_inpaint360_interiorgs_provider_runner.py"
runner_rc=$?
if [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "adp_inpaint360_runner_failed_without_runtime_result"
  echo "blocked_adp_inpaint360_process_exited_without_result:${runner_rc}"
fi
exit "${runner_rc}"
