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

mapfile -t runtime_mode < <(
  python3 - "${runtime_dir}/artifixer3d_runtime_request.json" <<'PY'
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(request.get("direct_editor_backend", ""))
print("true" if request.get("semantic_editor_only") is True else "false")
print(request.get("pipeline_mode", ""))
PY
)
direct_editor_backend="${runtime_mode[0]:-}"
semantic_editor_only="${runtime_mode[1]:-}"
pipeline_mode="${runtime_mode[2]:-}"
if [[ "${pipeline_mode}" == "dual_target_artifixer3d_only" \
      || "${pipeline_mode}" == "dual_target_artifixer3d_render_only" ]]; then
  if [[ "${direct_editor_backend}" != "none" \
        || "${semantic_editor_only}" != "false" ]]; then
    write_missing_result "artifixer3d_dual_target_mode_invalid"
    exit 2
  fi
elif [[ "${direct_editor_backend}" != "artifixer" \
      && "${direct_editor_backend}" != "qwen_image_edit_2511" \
      && "${direct_editor_backend}" != "vibe_image_edit" ]]; then
  write_missing_result "artifixer3d_direct_editor_backend_invalid"
  exit 2
fi
if [[ "${direct_editor_backend}" == "vibe_image_edit" \
      && "${semantic_editor_only}" != "true" ]]; then
  write_missing_result "artifixer3d_vibe_requires_semantic_editor_only"
  exit 2
fi

if [[ "${direct_editor_backend}" == "vibe_image_edit" ]]; then
  "${uv_bin}" pip install --python "${artifixer_python}" \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
    || { write_missing_result "artifixer3d_vibe_torch_install_failed"; exit 2; }
  "${uv_bin}" pip install --python "${artifixer_python}" \
    accelerate==1.11.0 annotated-types==0.7.0 click==8.3.1 diffusers==0.33.1 \
    huggingface-hub==0.35.3 loguru==0.7.3 numpy==1.26.4 protobuf==3.20.2 \
    pydantic==2.0.3 pydantic-core==2.3.0 pydantic-settings==2.0.3 \
    python-dotenv==1.2.1 'sentencepiece~=0.1.99' tokenizers==0.22.1 \
    transformers==4.57.1 Pillow \
    || { write_missing_result "artifixer3d_vibe_dependencies_failed"; exit 2; }

  vibe_source_dir="${runtime_dir}/VIBE_source"
  if [[ -e "${vibe_source_dir}" ]]; then
    write_missing_result "artifixer3d_vibe_source_destination_not_empty"
    exit 2
  fi
  git init -q "${vibe_source_dir}" \
    || { write_missing_result "artifixer3d_vibe_source_init_failed"; exit 2; }
  git -C "${vibe_source_dir}" remote add origin https://github.com/ai-forever/VIBE.git
  git -C "${vibe_source_dir}" fetch --depth 1 origin 7f0f01f9a6f66d55aa0fec2bf2562c332bba262b \
    || { write_missing_result "artifixer3d_vibe_source_fetch_failed"; exit 2; }
  git -C "${vibe_source_dir}" checkout -q --detach FETCH_HEAD \
    || { write_missing_result "artifixer3d_vibe_source_checkout_failed"; exit 2; }
  [[ "$(git -C "${vibe_source_dir}" rev-parse HEAD)" == "7f0f01f9a6f66d55aa0fec2bf2562c332bba262b" ]] \
    || { write_missing_result "artifixer3d_vibe_source_commit_mismatch"; exit 2; }
  [[ "$(git -C "${vibe_source_dir}" rev-parse 'HEAD^{tree}')" == "208f31e15a70de8a8b58e20acd6aba465ac1fcbc" ]] \
    || { write_missing_result "artifixer3d_vibe_source_tree_mismatch"; exit 2; }
  [[ "sha256:$(sha256sum "${vibe_source_dir}/LICENSE" | awk '{print $1}')" == \
      "sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4" ]] \
    || { write_missing_result "artifixer3d_vibe_source_license_mismatch"; exit 2; }
  "${uv_bin}" pip install --python "${artifixer_python}" --no-deps -e "${vibe_source_dir}" \
    || { write_missing_result "artifixer3d_vibe_source_install_failed"; exit 2; }
  "${uv_bin}" pip check --python "${artifixer_python}" \
    > "${output_dir}/artifixer3d-pip-check.txt" \
    || { write_missing_result "artifixer3d_vibe_dependency_conflict"; exit 2; }

  export PYTHONPATH="${vibe_source_dir}:${runtime_dir}/ArtiFixer_official${PYTHONPATH:+:${PYTHONPATH}}"
  "${artifixer_python}" - "${output_dir}/artifixer3d-runtime-preflight.json" "${vibe_source_dir}" <<'PY' \
    || { write_missing_result "artifixer3d_vibe_runtime_preflight_failed"; exit 2; }
import json
from pathlib import Path
import sys

import torch
from vibe.editor import ImageEditor  # noqa: F401

receipt_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
receipt = {
    "schema_version": "public_scene_artifixer3d_runtime_preflight.v1",
    "status": "completed",
    "direct_editor_backend": "vibe_image_edit",
    "source_commit": "7f0f01f9a6f66d55aa0fec2bf2562c332bba262b",
    "source_tree": "208f31e15a70de8a8b58e20acd6aba465ac1fcbc",
    "source_root": str(source_root),
    "torch": {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    },
    "blockers": [],
}
if not receipt["torch"]["cuda_available"] or receipt["torch"]["device_count"] != 1:
    receipt["status"] = "blocked"
    receipt["blockers"].append("single_cuda_device_unavailable")
else:
    receipt["torch"]["device_name"] = torch.cuda.get_device_name(0)
    receipt["torch"]["device_capability"] = list(torch.cuda.get_device_capability(0))
    receipt["torch"]["device_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if receipt["blockers"]:
    raise SystemExit(";".join(receipt["blockers"]))
PY
else
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
git -C "${submodule_dir}" submodule update --init --recursive \
  || { write_missing_result "artifixer3d_nested_submodule_fetch_failed"; exit 2; }
if git -C "${submodule_dir}" submodule status --recursive | grep -Eq '^[-+U]'; then
  write_missing_result "artifixer3d_nested_submodule_identity_mismatch"
  exit 2
fi

"${uv_bin}" pip install --python "${artifixer_python}" --no-build-isolation \
  -r "${submodule_dir}/requirements.txt" \
  || { write_missing_result "artifixer3d_3dgrut_requirements_failed"; exit 2; }
bash "${submodule_dir}/scripts/install_slangc.sh" /usr/local \
  || { write_missing_result "artifixer3d_slangc_install_failed"; exit 2; }
"${uv_bin}" pip install --python "${artifixer_python}" -e "${submodule_dir}" \
  || { write_missing_result "artifixer3d_3dgrut_install_failed"; exit 2; }
"${uv_bin}" pip check --python "${artifixer_python}" \
  > "${output_dir}/artifixer3d-pip-check.txt" \
  || { write_missing_result "artifixer3d_python_dependency_conflict"; exit 2; }

# Import the exact released entrypoint graphs before downloading model weights or
# beginning inference.  In particular, importing data_processing.artifixer3d
# forces the 3DGRUT JIT extension to resolve its recursively pinned CUDA headers;
# a shallow parent-only submodule checkout therefore fails here, before a long
# direct-inference pass can hide the packaging defect.
export PYTHONPATH="${runtime_dir}/ArtiFixer_official:${submodule_dir}${PYTHONPATH:+:${PYTHONPATH}}"
"${artifixer_python}" - "${output_dir}/artifixer3d-runtime-preflight.json" "${submodule_dir}" <<'PY' \
  || { write_missing_result "artifixer3d_runtime_preflight_failed"; exit 2; }
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

receipt_path = Path(sys.argv[1])
submodule_root = Path(sys.argv[2])
required_commands = ("git", "gcc", "g++", "cmake", "ninja", "nvcc", "slangc")
required_files = (
    submodule_root / "thirdparty" / "tiny-cuda-nn" / "include" / "tiny-cuda-nn" / "common.h",
    submodule_root
    / "thirdparty"
    / "tiny-cuda-nn"
    / "dependencies"
    / "cutlass"
    / "include"
    / "cutlass"
    / "cutlass.h",
)
entrypoint_modules = (
    "model_eval.run_inference",
    "data_processing.run_artifixer3d",
    "data_processing.render_3dgrut_colmap",
)

receipt = {
    "schema_version": "public_scene_artifixer3d_runtime_preflight.v1",
    "status": "blocked",
    "commands": {},
    "required_files": {},
    "entrypoint_imports": {},
    "nested_submodules": [],
    "torch": {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    },
    "blockers": [],
}
try:
    for command in required_commands:
        resolved = shutil.which(command)
        receipt["commands"][command] = resolved
        if resolved is None:
            receipt["blockers"].append(f"missing_command:{command}")
    for path in required_files:
        present = path.is_file()
        receipt["required_files"][str(path.relative_to(submodule_root))] = present
        if not present:
            receipt["blockers"].append(f"missing_file:{path.relative_to(submodule_root)}")
    nested = subprocess.run(
        ["git", "-C", str(submodule_root), "submodule", "status", "--recursive"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt["nested_submodules"] = [
        line.strip() for line in nested.stdout.splitlines() if line.strip()
    ]
    if any(line.startswith(("-", "+", "U")) for line in receipt["nested_submodules"]):
        receipt["blockers"].append("nested_submodule_identity_mismatch")
    if not receipt["torch"]["cuda_available"] or receipt["torch"]["device_count"] != 1:
        receipt["blockers"].append("single_cuda_device_unavailable")
    else:
        receipt["torch"]["device_name"] = torch.cuda.get_device_name(0)
        receipt["torch"]["device_capability"] = list(torch.cuda.get_device_capability(0))
        receipt["torch"]["device_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    for module in entrypoint_modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            receipt["entrypoint_imports"][module] = False
            receipt["blockers"].append(
                f"entrypoint_import_failed:{module}:{type(exc).__name__}"
            )
            raise
        else:
            receipt["entrypoint_imports"][module] = True
    if not receipt["blockers"]:
        receipt["status"] = "completed"
finally:
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if receipt["blockers"]:
    raise SystemExit(";".join(receipt["blockers"]))
PY
fi
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
