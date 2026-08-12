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
"${uv_bin}" python install 3.10 \
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
"${uv_bin}" venv "${runtime_dir}/.lama-venv" --python 3.10 \
  || { write_missing_result "aura_exact_residual_lama_venv_failed"; exit 2; }
lama_python="${runtime_dir}/.lama-venv/bin/python"
"${uv_bin}" pip install --python "${lama_python}" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 \
  || { write_missing_result "aura_exact_residual_lama_torch_install_failed"; exit 2; }
"${uv_bin}" pip install --python "${lama_python}" -r "${runtime_dir}/LaMa/requirements.txt" \
  || { write_missing_result "aura_exact_residual_lama_requirements_failed"; exit 2; }
"${uv_bin}" pip freeze --python "${lama_python}" > "${output_dir}/lama-pip-freeze.txt"
set +e
"${aura_python}" "${runtime_dir}/public_scene_aura_exact_residual_runner.py" \
  --runtime-dir "${runtime_dir}" --output-dir "${output_dir}"
runner_rc=$?
set -e
if [[ ${runner_rc} -ne 0 && ! -f "${result_path}" ]]; then
  write_missing_result "aura_exact_residual_runner_failed_without_result"
fi
exit "${runner_rc}"
