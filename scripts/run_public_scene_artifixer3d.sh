#!/usr/bin/env bash
# Execute the immutable ArtiFixer/3D/3D+ candidate bundle or rehearse it locally.
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_root="$(cd "${runtime_dir}/.." && pwd)"
output_dir="${BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_OUTPUT_DIR:-${bundle_root}/runtime_output}"
mkdir -p "${output_dir}"

if [[ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-}" == "1" ]]; then
  python3 "${runtime_dir}/public_scene_artifixer3d_runner.py" \
    --bundle-root "${bundle_root}" --output-root "${output_dir}" --rehearsal
  exit 0
fi

result_path="${output_dir}/public_scene_artifixer3d_runtime_result.json"
write_missing_result() {
  local blocker="$1"
  python3 - "${result_path}" "${blocker}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({
        "schema_version": "public_scene_artifixer3d_runtime_result.v1",
        "status": "blocked",
        "tasks": [],
        "model_loaded": False,
        "artifixer_direct_inference_executed": False,
        "artifixer3d_distillation_executed": False,
        "artifixer3d_plus_inference_executed": False,
        "provider_mutations_performed": 1,
        "blockers": [sys.argv[2]],
        "provider_zero_required_after_return": True,
        "physical_or_deployment_evidence": False,
        "claim_boundary": "runtime_setup_failure_only",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_with_progress() {
  local stage="$1"
  shift
  echo "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_STAGE_STARTED:${stage}"
  "$@" &
  local child_pid=$!
  (
    while kill -0 "${child_pid}" 2>/dev/null; do
      local output_bytes
      output_bytes="$(du -sk "${output_dir}" 2>/dev/null | awk '{print $1 * 1024}')"
      echo "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_PROGRESS:${stage}:$(date -u +%Y-%m-%dT%H:%M:%SZ):output_bytes=${output_bytes:-0}"
      sleep 60
    done
  ) &
  local progress_pid=$!
  set +e
  wait "${child_pid}"
  local child_rc=$?
  set -e
  kill "${progress_pid}" 2>/dev/null || true
  wait "${progress_pid}" 2>/dev/null || true
  echo "BLUEPRINT_PUBLIC_SCENE_ARTIFIXER3D_STAGE_FINISHED:${stage}:returncode=${child_rc}"
  return "${child_rc}"
}

python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7 \
  || { write_missing_result "artifixer3d_uv_install_failed"; exit 2; }
uv_bin="$(command -v uv)"
export UV_NATIVE_TLS=true
"${uv_bin}" venv "${runtime_dir}/.artifixer-venv" --python "$(command -v python3)" \
  || { write_missing_result "artifixer3d_venv_failed"; exit 2; }
artifixer_python="${runtime_dir}/.artifixer-venv/bin/python"
export CUDA_HOME=/usr/local/cuda

"${uv_bin}" pip install --python "${artifixer_python}" \
  torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
  || { write_missing_result "artifixer3d_torch_install_failed"; exit 2; }
"${uv_bin}" pip uninstall --python "${artifixer_python}" flash-attn opencv-python || true
"${uv_bin}" pip install --python "${artifixer_python}" \
  accelerate==1.13.0 diffusers==0.37.1 transformers==5.5.0 ftfy \
  'numpy<2.0' einops scipy wandb tqdm Pillow matplotlib opencv-python-headless \
  pyyaml torchmetrics imageio-ffmpeg h5py av torch-fidelity huggingface-hub \
  || { write_missing_result "artifixer3d_python_dependencies_failed"; exit 2; }

submodule_dir="${runtime_dir}/ArtiFixer_official/thirdparty/3DGRUT-ArtiFixer"
if [[ -e "${submodule_dir}" ]]; then
  write_missing_result "artifixer3d_submodule_destination_not_empty"
  exit 2
fi
mkdir -p "$(dirname "${submodule_dir}")"
git init -q "${submodule_dir}" \
  || { write_missing_result "artifixer3d_submodule_init_failed"; exit 2; }
git -C "${submodule_dir}" remote add origin https://github.com/nv-tlabs/3DGRUT-ArtiFixer.git
git -C "${submodule_dir}" fetch --depth 1 origin 62e1038b74b2edc01440fd4ddf5f080109b6faba \
  || { write_missing_result "artifixer3d_submodule_fetch_failed"; exit 2; }
git -C "${submodule_dir}" checkout -q --detach FETCH_HEAD \
  || { write_missing_result "artifixer3d_submodule_checkout_failed"; exit 2; }
[[ "$(git -C "${submodule_dir}" rev-parse HEAD)" == "62e1038b74b2edc01440fd4ddf5f080109b6faba" ]] \
  || { write_missing_result "artifixer3d_submodule_commit_mismatch"; exit 2; }
[[ "$(git -C "${submodule_dir}" rev-parse 'HEAD^{tree}')" == "494ecc2dd0834fcf71bf0124de152940e0c6d845" ]] \
  || { write_missing_result "artifixer3d_submodule_tree_mismatch"; exit 2; }

"${uv_bin}" pip install --python "${artifixer_python}" -r "${submodule_dir}/requirements.txt" \
  || { write_missing_result "artifixer3d_3dgrut_requirements_failed"; exit 2; }
bash "${submodule_dir}/scripts/install_slangc.sh" /usr/local \
  || { write_missing_result "artifixer3d_slangc_install_failed"; exit 2; }
"${uv_bin}" pip install --python "${artifixer_python}" -e "${submodule_dir}" \
  || { write_missing_result "artifixer3d_3dgrut_install_failed"; exit 2; }
"${uv_bin}" pip freeze --python "${artifixer_python}" > "${output_dir}/artifixer3d-pip-freeze.txt"

export HF_HOME="${bundle_root}/.hf_home"
export HF_HUB_CACHE="${bundle_root}/.hf_home/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800
set +e
run_with_progress execute "${artifixer_python}" "${runtime_dir}/public_scene_artifixer3d_runner.py" \
  --bundle-root "${bundle_root}" --output-root "${output_dir}"
runner_rc=$?
set -e
if [[ ${runner_rc} -ne 0 && ! -f "${result_path}" ]]; then
  write_missing_result "artifixer3d_runner_failed_without_result"
fi
exit "${runner_rc}"
