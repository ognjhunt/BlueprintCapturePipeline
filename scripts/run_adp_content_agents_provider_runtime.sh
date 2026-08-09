#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${BLUEPRINT_ADP_CONTENT_AGENTS_OUTPUT_DIR:-${SCRIPT_DIR}/../runtime_output}"
SOURCE_DIR="${SCRIPT_DIR}/content_agents_source"
RESULT_PATH="${OUTPUT_DIR}/adp_content_agents_vast_result.json"
mkdir -p "${OUTPUT_DIR}"

write_missing_result() {
  local blocker="$1"
  python3 - "${RESULT_PATH}" "${blocker}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
blocker = sys.argv[2]
if not path.exists():
    path.write_text(json.dumps({
        "schema_version": "adp_content_agents_vast_result.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "blocked",
        "blockers": [blocker],
        "material_agent_executed": False,
        "texture_agent_executed": False,
        "physics_agent_executed": False,
        "validation_agent_executed": False,
        "joint_agent_inapplicable_single_rigid_body": True,
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
python3 "${SCRIPT_DIR}/provider_archive.py" \
  "${SCRIPT_DIR}/content_agents_source.zip" "${SOURCE_DIR}" \
  --receipt "${OUTPUT_DIR}/content_agents_source_extraction.json"
unzip_rc=$?
if [ "${unzip_rc}" -ne 0 ]; then
  write_missing_result "content_agents_source_archive_extract_failed"
  exit "${unzip_rc}"
fi

if [ "${BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL:-0}" = "1" ]; then
  python3 - "${OUTPUT_DIR}/provider_bundle_rehearsal.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "provider_bundle_entrypoint_rehearsal.v1",
    "status": "passed",
    "entrypoint": "run_adp_content_agents_provider_runtime.sh",
    "archive_extraction_executed": True,
    "gpu_runtime_started": False,
    "paid_inference_performed": False,
    "provider_mutations_performed": 0,
    "stopped_before": "dependency_install_and_agent_execution",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit 0
fi

python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7
uv_rc=$?
if [ "${uv_rc}" -ne 0 ]; then
  write_missing_result "content_agents_uv_install_failed"
  exit "${uv_rc}"
fi
UV_BIN="$(command -v uv)"
"${UV_BIN}" python install 3.12
python_rc=$?
if [ "${python_rc}" -ne 0 ]; then
  write_missing_result "content_agents_python312_install_failed"
  exit "${python_rc}"
fi
"${UV_BIN}" venv "${SOURCE_DIR}/.venv" --python 3.12
venv_rc=$?
if [ "${venv_rc}" -ne 0 ]; then
  write_missing_result "content_agents_python_venv_failed"
  exit "${venv_rc}"
fi

"${UV_BIN}" pip install \
  --python "${SOURCE_DIR}/.venv/bin/python" \
  -e "${SOURCE_DIR}" \
  -e "${SOURCE_DIR}/apps/material_agent" \
  -e "${SOURCE_DIR}/apps/physics_agent" \
  -e "${SOURCE_DIR}/apps/texture_agent" \
  -e "${SOURCE_DIR}/apps/validation_agent"
install_rc=$?
if [ "${install_rc}" -ne 0 ]; then
  write_missing_result "content_agents_dependency_install_failed"
  exit "${install_rc}"
fi

export WU_OVRTX_VENV_DIR="${SOURCE_DIR}/.ovrtx_venv"
export WU_OVRTX_AUTO_PROVISION=0
"${UV_BIN}" venv "${WU_OVRTX_VENV_DIR}" --python "${SOURCE_DIR}/.venv/bin/python"
"${UV_BIN}" pip install \
  --python "${WU_OVRTX_VENV_DIR}/bin/python" \
  -r "${SOURCE_DIR}/world_understanding/functions/graphics/pylock.ovrtx-runtime.toml" \
  --require-hashes --no-deps --no-config --no-sources
ovrtx_rc=$?
if [ "${ovrtx_rc}" -ne 0 ]; then
  write_missing_result "content_agents_ovrtx_provision_failed"
  exit "${ovrtx_rc}"
fi
WU_OVRTX_LOCK_DIR="${SOURCE_DIR}/.ovrtx_locks" \
  "${SOURCE_DIR}/.venv/bin/python" \
  -m world_understanding.functions.graphics.render_ovrtx --provision-only
ovrtx_probe_rc=$?
if [ "${ovrtx_probe_rc}" -ne 0 ]; then
  write_missing_result "content_agents_ovrtx_runtime_probe_failed"
  exit "${ovrtx_probe_rc}"
fi

if [ -d "${SCRIPT_DIR}/native" ]; then
  NATIVE_OVRTX_ENV="${SOURCE_DIR}/.ovrtx_native_venv"
  "${UV_BIN}" venv "${NATIVE_OVRTX_ENV}" --python "${SOURCE_DIR}/.venv/bin/python"
  native_ovrtx_venv_rc=$?
  if [ "${native_ovrtx_venv_rc}" -ne 0 ]; then
    write_missing_result "content_agents_native_ovrtx_venv_failed"
    exit "${native_ovrtx_venv_rc}"
  fi
  "${UV_BIN}" pip install \
    --python "${NATIVE_OVRTX_ENV}/bin/python" \
    --extra-index-url https://pypi.nvidia.com \
    "ovrtx==0.4.0.346409" \
    "ovstage==0.1.0.346039" \
    "numpy>=1.26,<3" \
    "Pillow>=10,<13" \
    "nvidia-ml-py>=12,<14"
  native_ovrtx_rc=$?
  if [ "${native_ovrtx_rc}" -ne 0 ]; then
    write_missing_result "content_agents_native_ovrtx_provision_failed"
    exit "${native_ovrtx_rc}"
  fi
  "${NATIVE_OVRTX_ENV}/bin/python" -c \
    'import importlib.metadata as m; import ovrtx, ovstage; assert m.version("ovrtx") == "0.4.0.346409" and m.version("ovstage") == "0.1.0.346039"'
  native_ovrtx_probe_rc=$?
  if [ "${native_ovrtx_probe_rc}" -ne 0 ]; then
    write_missing_result "content_agents_native_ovrtx_dependency_closure_failed"
    exit "${native_ovrtx_probe_rc}"
  fi

  OVPX_ENV="${SOURCE_DIR}/.ovphysx_venv"
  "${UV_BIN}" venv "${OVPX_ENV}" --python "${SOURCE_DIR}/.venv/bin/python"
  ovphysx_venv_rc=$?
  if [ "${ovphysx_venv_rc}" -ne 0 ]; then
    write_missing_result "content_agents_ovphysx_venv_failed"
    exit "${ovphysx_venv_rc}"
  fi
  "${UV_BIN}" pip install \
    --python "${OVPX_ENV}/bin/python" \
    "ovphysx==0.4.13" \
    "numpy>=1.26,<3" \
    "nvidia-ml-py>=12,<14"
  ovphysx_rc=$?
  if [ "${ovphysx_rc}" -ne 0 ]; then
    write_missing_result "content_agents_ovphysx_provision_failed"
    exit "${ovphysx_rc}"
  fi
  "${OVPX_ENV}/bin/python" -c 'import ovphysx; assert ovphysx.__version__ == "0.4.13"'
  ovphysx_probe_rc=$?
  if [ "${ovphysx_probe_rc}" -ne 0 ]; then
    write_missing_result "content_agents_ovphysx_runtime_probe_failed"
    exit "${ovphysx_probe_rc}"
  fi
fi

export BLUEPRINT_ADP_CONTENT_AGENTS_PYTHON="${SOURCE_DIR}/.venv/bin/python"
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 >"${OUTPUT_DIR}/xvfb.log" 2>&1 &
xvfb_pid=$!
sleep 2

"${SOURCE_DIR}/.venv/bin/python" "${SCRIPT_DIR}/adp_content_agents_provider_runner.py"
runner_rc=$?
kill "${xvfb_pid}" >/dev/null 2>&1 || true
wait "${xvfb_pid}" >/dev/null 2>&1 || true

if [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "adp_content_agents_runner_failed_without_runtime_result"
  echo "blocked_adp_content_agents_process_exited_without_result:${runner_rc}"
fi
if [ "${runner_rc}" -ne 0 ]; then
  echo "BLUEPRINT_ADP_CONTENT_AGENTS_BLOCKED:${runner_rc}"
fi
exit "${runner_rc}"
