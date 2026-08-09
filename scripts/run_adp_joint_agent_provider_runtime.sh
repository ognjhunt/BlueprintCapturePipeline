#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${BLUEPRINT_ADP_JOINT_AGENT_OUTPUT_DIR:-${SCRIPT_DIR}/../runtime_output}"
SOURCE_DIR="${SCRIPT_DIR}/content_agents_source"
RESULT_PATH="${OUTPUT_DIR}/adp_joint_agent_result.json"
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
        "schema_version": "adp_joint_agent_result.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "blocked",
        "blockers": [sys.argv[2]],
        "joint_agent_inference_executed": False,
        "owned_core_publication_executed": False,
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
python3 -m zipfile -e "${SCRIPT_DIR}/content_agents_source.zip" "${SOURCE_DIR}" || {
  write_missing_result "joint_agent_source_archive_extract_failed"; exit 2;
}
python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.10.7 || {
  write_missing_result "joint_agent_uv_install_failed"; exit 2;
}
UV_BIN="$(command -v uv)"
"${UV_BIN}" python install 3.12 || { write_missing_result "joint_agent_python312_install_failed"; exit 2; }
"${UV_BIN}" venv "${SOURCE_DIR}/.venv" --python 3.12 || {
  write_missing_result "joint_agent_python_venv_failed"; exit 2;
}
"${UV_BIN}" pip install --python "${SOURCE_DIR}/.venv/bin/python" \
  -e "${SOURCE_DIR}" \
  -e "${SOURCE_DIR}/apps/joint_agent" \
  -e "${SOURCE_DIR}/apps/ovrtx_rendering_api" || {
  write_missing_result "joint_agent_dependency_install_failed"; exit 2;
}
export WU_OVRTX_VENV_DIR="${SOURCE_DIR}/.ovrtx_venv"
export WU_OVRTX_AUTO_PROVISION=0
"${UV_BIN}" venv "${WU_OVRTX_VENV_DIR}" --python "${SOURCE_DIR}/.venv/bin/python" || {
  write_missing_result "joint_agent_ovrtx_venv_failed"; exit 2;
}
"${UV_BIN}" pip install --python "${WU_OVRTX_VENV_DIR}/bin/python" \
  -r "${SOURCE_DIR}/world_understanding/functions/graphics/pylock.ovrtx-runtime.toml" \
  --require-hashes --no-deps --no-config --no-sources || {
  write_missing_result "joint_agent_ovrtx_provision_failed"; exit 2;
}
WU_OVRTX_LOCK_DIR="${SOURCE_DIR}/.ovrtx_locks" \
  "${SOURCE_DIR}/.venv/bin/python" \
  -m world_understanding.functions.graphics.render_ovrtx --provision-only || {
  write_missing_result "joint_agent_ovrtx_runtime_probe_failed"; exit 2;
}

export PYTHONPATH="${SCRIPT_DIR}/blueprint_src:${SOURCE_DIR}:${SOURCE_DIR}/apps/ovrtx_rendering_api"
export RENDER_ENDPOINT="http://127.0.0.1:8001"
export OVRTX_RENDER_MODE="pt"
export OVRTX_NUM_SENSOR_UPDATES="128"
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 >"${OUTPUT_DIR}/xvfb.log" 2>&1 &
xvfb_pid=$!
"${SOURCE_DIR}/.venv/bin/python" -m uvicorn service.main:app \
  --host 127.0.0.1 --port 8001 >"${OUTPUT_DIR}/ovrtx_service.log" 2>&1 &
renderer_pid=$!
renderer_ready=0
for _ in $(seq 1 180); do
  if curl -fsS "${RENDER_ENDPOINT}/health" | grep -q '"gpu_initialized":true'; then
    renderer_ready=1
    break
  fi
  sleep 5
done
if [ "${renderer_ready}" -ne 1 ]; then
  kill "${renderer_pid}" "${xvfb_pid}" >/dev/null 2>&1 || true
  write_missing_result "joint_agent_local_ovrtx_renderer_not_ready"
  exit 2
fi

cd "${SCRIPT_DIR}" || exit 2
"${SOURCE_DIR}/.venv/bin/python" "${SCRIPT_DIR}/adp_joint_agent_provider_runner.py"
runner_rc=$?
kill "${renderer_pid}" "${xvfb_pid}" >/dev/null 2>&1 || true
wait "${renderer_pid}" >/dev/null 2>&1 || true
wait "${xvfb_pid}" >/dev/null 2>&1 || true
if [ ! -f "${RESULT_PATH}" ]; then
  write_missing_result "adp_joint_agent_runner_failed_without_runtime_result"
  echo "blocked_adp_joint_agent_process_exited_without_result:${runner_rc}"
fi
exit "${runner_rc}"
