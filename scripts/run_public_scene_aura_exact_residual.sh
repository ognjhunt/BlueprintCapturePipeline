#!/usr/bin/env bash
# Execute an immutable Aura exact-residual bundle or prove its zero-cost seam.
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${BLUEPRINT_PUBLIC_SCENE_AURA_EXACT_RESIDUAL_OUTPUT_DIR:-${runtime_dir}/../runtime_output}"
mkdir -p "${output_dir}"

if [[ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-}" == "1" ]]; then
  python3 "${runtime_dir}/public_scene_aura_exact_residual_runner.py" \
    --runtime-dir "${runtime_dir}" --output-dir "${output_dir}" --rehearsal
  exit 0
fi

run_with_progress() {
  local stage="$1"
  shift
  echo "BLUEPRINT_PUBLIC_SCENE_AURA_EXACT_RESIDUAL_STAGE_STARTED:${stage}"
  "$@" &
  local child_pid=$!
  (
    while kill -0 "${child_pid}" 2>/dev/null; do
      local output_bytes
      output_bytes="$(du -sk "${output_dir}" 2>/dev/null | awk '{print $1 * 1024}')"
      echo "BLUEPRINT_PUBLIC_SCENE_AURA_EXACT_RESIDUAL_PROGRESS:${stage}:$(date -u +%Y-%m-%dT%H:%M:%SZ):output_bytes=${output_bytes:-0}"
      sleep 60
    done
  ) &
  local progress_pid=$!
  wait "${child_pid}"
  local child_rc=$?
  kill "${progress_pid}" 2>/dev/null || true
  wait "${progress_pid}" 2>/dev/null || true
  echo "BLUEPRINT_PUBLIC_SCENE_AURA_EXACT_RESIDUAL_STAGE_FINISHED:${stage}:returncode=${child_rc}"
  return "${child_rc}"
}

result_path="${output_dir}/public_scene_aura_exact_residual_runtime_result.json"
write_missing_result() {
  local blocker="$1"
  python3 - "${result_path}" "${blocker}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({
        "schema_version": "public_scene_aura_exact_residual_runtime_result.v1",
        "status": "blocked",
        "blockers": [sys.argv[2]],
        "aura_inpainting_executed": False,
        "provider_mutations_performed": 0,
        "learned_policy_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7 \
  || { write_missing_result "aura_exact_residual_uv_install_failed"; exit 2; }
uv_bin="$(command -v uv)"
export UV_NATIVE_TLS=true
"${uv_bin}" python install 3.10 3.8 \
  || { write_missing_result "aura_exact_residual_python_install_failed"; exit 2; }
"${uv_bin}" venv "${runtime_dir}/.aura-venv" --python 3.10 \
  || { write_missing_result "aura_exact_residual_venv_failed"; exit 2; }
aura_python="${runtime_dir}/.aura-venv/bin/python"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda
"${uv_bin}" pip install --python "${aura_python}" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 \
  || { write_missing_result "aura_exact_residual_torch_install_failed"; exit 2; }
"${uv_bin}" pip install --python "${aura_python}" setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 packaging==25.0 \
  || { write_missing_result "aura_exact_residual_build_dependencies_failed"; exit 2; }
"${uv_bin}" pip install --python "${aura_python}" --no-build-isolation \
  "${runtime_dir}/AuraFusion360_official/submodules/diff-surfel-rasterization" \
  "${runtime_dir}/AuraFusion360_official/submodules/simple-knn" \
  || { write_missing_result "aura_exact_residual_cuda_extensions_failed"; exit 2; }
"${uv_bin}" pip install --python "${aura_python}" -r "${runtime_dir}/AuraFusion360_official/requirements.txt" \
  || { write_missing_result "aura_exact_residual_aura_requirements_failed"; exit 2; }
"${uv_bin}" pip freeze --python "${aura_python}" > "${output_dir}/aura-pip-freeze.txt"
"${uv_bin}" venv "${runtime_dir}/.lama-venv" --python 3.8 \
  || { write_missing_result "aura_exact_residual_lama_venv_failed"; exit 2; }
lama_python="${runtime_dir}/.lama-venv/bin/python"
"${uv_bin}" pip install --python "${lama_python}" torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
  || { write_missing_result "aura_exact_residual_lama_torch_install_failed"; exit 2; }
"${uv_bin}" pip install --python "${lama_python}" setuptools==57.5.0 wheel==0.37.1 Cython==0.29.37 \
  || { write_missing_result "aura_exact_residual_lama_build_dependencies_failed"; exit 2; }
"${uv_bin}" pip install --python "${lama_python}" --no-build-isolation pyyaml==5.4.1 \
  || { write_missing_result "aura_exact_residual_lama_yaml_install_failed"; exit 2; }
"${uv_bin}" pip install --python "${lama_python}" numpy==1.21.6 tqdm==4.67.1 easydict==1.9.0 scikit-image==0.19.3 scikit-learn==1.0.2 scipy==1.7.3 opencv-python-headless==4.5.5.64 joblib==1.1.1 matplotlib==3.5.3 pandas==1.3.5 albumentations==0.5.2 hydra-core==1.1.0 pytorch-lightning==1.2.9 tabulate==0.9.0 kornia==0.5.0 webdataset==0.1.103 packaging==24.2 tensorboard==2.11.2 \
  || { write_missing_result "aura_exact_residual_lama_requirements_failed"; exit 2; }
"${uv_bin}" pip uninstall --python "${lama_python}" opencv-python \
  || { write_missing_result "aura_exact_residual_lama_opencv_cleanup_failed"; exit 2; }
"${uv_bin}" pip install --python "${lama_python}" --reinstall --no-deps opencv-python-headless==4.5.5.64 \
  || { write_missing_result "aura_exact_residual_lama_opencv_pin_failed"; exit 2; }
"${uv_bin}" pip freeze --python "${lama_python}" > "${output_dir}/lama-pip-freeze.txt"
export HF_HOME="${runtime_dir}/.hf_home" HF_HUB_CACHE="${runtime_dir}/.hf_home/hub" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600
set +e
run_with_progress execute "${aura_python}" "${runtime_dir}/public_scene_aura_exact_residual_runner.py" \
  --runtime-dir "${runtime_dir}" --output-dir "${output_dir}"
runner_rc=$?
set -e
if [[ ${runner_rc} -ne 0 && ! -f "${result_path}" ]]; then
  write_missing_result "aura_exact_residual_runner_failed_without_result"
fi
exit "${runner_rc}"
