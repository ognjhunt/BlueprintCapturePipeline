#!/usr/bin/env bash
set -u

RUNTIME_ROOT=${BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT:-/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime}
OUTPUT_ROOT=${BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT:-/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output}
RESULT_PATH="$OUTPUT_ROOT/task_evaluation_scene_configuration_provider_result.v1.json"
mkdir -p "$OUTPUT_ROOT"

PYTHON_BIN=/isaac-sim/python.sh
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=$(command -v python3 || true)
fi

export PYTHONPATH="$RUNTIME_ROOT"
export BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT="$RUNTIME_ROOT/toolchain"
export BLUEPRINT_SCENE_CONFIGURATION_PROVIDER_RESULT="$RESULT_PATH"

write_blocked_result() {
  blocker="$1"
  "$PYTHON_BIN" - "$RESULT_PATH" "$blocker" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
    "status": "blocked",
    "blockers": [sys.argv[2]],
    "first_stage_started": False,
    "evaluation_episode_executed": False,
    "candidate_policy_queried": False,
    "provider_zero_required_after_return": True,
    "result_digest": "",
}
payload = dict(value)
payload.pop("result_digest", None)
value["result_digest"] = "sha256:" + hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

if [ -d "$RUNTIME_ROOT/toolchain" ] && [ -n "$PYTHON_BIN" ]; then
  # The bundle is unpacked with `python -m zipfile -e`, and CPython's zipfile
  # never restores mode bits -- every file lands 0644 regardless of what the
  # control plane sealed. The toolchain self-check compares each file's execute
  # bit against the `executable` flag recorded in its own manifest, so an
  # unrestored tree fails closed on its first entry, and any component the
  # stages exec would raise PermissionError. Restore the recorded modes from
  # the manifest -- the same document the check reads -- before sealing the
  # tree read-only.
  "$PYTHON_BIN" - "$RUNTIME_ROOT/toolchain" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "task_evaluation_scene_configuration_toolchain.v1.json"
rows = json.loads(manifest.read_text(encoding="utf-8")).get("files") or []
for row in rows:
    relative = str(row.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        continue
    target = root / relative
    if target.is_symlink() or not target.is_file():
        continue
    os.chmod(target, 0o555 if row.get("executable") is True else 0o444)
for directory in [root, *(p for p in root.rglob("*") if p.is_dir())]:
    if not directory.is_symlink():
        os.chmod(directory, 0o555)
PY
  chmod -R a-w "$RUNTIME_ROOT/toolchain"
fi

# Every stage runs its child with capture_output=True, so the container emits
# nothing at all between entrypoint start and exit. The Vast no-progress
# watchdog then sees a byte-identical log tail on every poll and tears the
# instance down mid-run. Emit a payload that genuinely varies -- a tick
# counter, not a clock, because the watchdog strips timestamps before
# comparing.
progress_ticker() {
  tick=0
  while true; do
    tick=$((tick + 1))
    printf 'BLUEPRINT_SCENE_CONFIGURATION_PROGRESS: tick=%s\n' "$tick"
    sleep 60
  done
}
progress_ticker &
ticker_pid=$!
trap 'kill "$ticker_pid" 2>/dev/null || true' EXIT

PYTHON_WHEELHOUSE="$RUNTIME_ROOT/toolchain/components/artifixer3d_observed_object_removal/package/python_wheelhouse"
PYTHON_RUNTIME="$OUTPUT_ROOT/.venv/provider_python_runtime"
mkdir -p "$(dirname "$PYTHON_RUNTIME")"
python_runtime_rc=0
if [ -z "$PYTHON_BIN" ] || [ ! -d "$PYTHON_WHEELHOUSE" ]; then
  python_runtime_rc=86
else
  "$PYTHON_BIN" -m \
  blueprint_pipeline.task_evaluation_scene_configuration_python_runtime \
  --wheelhouse-root "$PYTHON_WHEELHOUSE" \
  --output-root "$PYTHON_RUNTIME" \
  >"$OUTPUT_ROOT/provider_python_runtime_setup.log" 2>&1 \
  || python_runtime_rc=$?
fi

if [ "$python_runtime_rc" -ne 0 ]; then
  if [ -n "$PYTHON_BIN" ]; then
    write_blocked_result "scene_configuration_provider_python_runtime_invalid"
  fi
  runner_rc="$python_runtime_rc"
else
  export PYTHONPATH="$RUNTIME_ROOT:$PYTHON_RUNTIME"
  # Import the actual stage modules before starting the chain. This checks the
  # complete eager closure (including pxr from Isaac and the bundled Agents
  # SDK/Pydantic runtime) before the first expensive render or training step.
  if ! "$PYTHON_BIN" - \
    >"$OUTPUT_ROOT/provider_python_import_preflight.log" 2>&1 <<'PY'
import agents
import numpy
import pydantic
import scipy
import yaml
from PIL import Image
from pxr import Usd
from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_driver
from blueprint_pipeline import task_evaluation_scene_configuration_content_agents_driver
from blueprint_pipeline import task_evaluation_scene_configuration_native_import_driver

assert agents and numpy and pydantic and scipy and yaml and Image and Usd
assert task_evaluation_scene_configuration_artifixer_driver
assert task_evaluation_scene_configuration_content_agents_driver
assert task_evaluation_scene_configuration_native_import_driver
PY
  then
    write_blocked_result "scene_configuration_provider_stage_import_closure_invalid"
    runner_rc=86
  else
  "$PYTHON_BIN" "$RUNTIME_ROOT/task_evaluation_scene_configuration_provider_runner.py"
  runner_rc=$?
  fi
fi

kill "$ticker_pid" 2>/dev/null || true

if [ ! -f "$RESULT_PATH" ] && [ -n "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" - "$RESULT_PATH" "$runner_rc" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "task_evaluation_scene_configuration_provider_result.v1",
    "status": "blocked",
    "blockers": [f"scene_configuration_provider_runner_failed_without_result:{sys.argv[2]}"],
    "first_stage_started": False,
    "evaluation_episode_executed": False,
    "candidate_policy_queried": False,
    "provider_zero_required_after_return": True,
    "result_digest": "",
}
payload = dict(value)
payload.pop("result_digest", None)
value["result_digest"] = "sha256:" + hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
PY
fi

if [ ! -f "$RESULT_PATH" ]; then
  fallback_payload='{"blockers":["scene_configuration_provider_python_runtime_missing"],"candidate_policy_queried":false,"evaluation_episode_executed":false,"first_stage_started":false,"provider_zero_required_after_return":true,"schema_version":"task_evaluation_scene_configuration_provider_result.v1","status":"blocked"}'
  if command -v sha256sum >/dev/null 2>&1; then
    fallback_digest=$(printf '%s' "$fallback_payload" | sha256sum | cut -d' ' -f1)
    printf '%s\n' '{"blockers":["scene_configuration_provider_python_runtime_missing"],"candidate_policy_queried":false,"evaluation_episode_executed":false,"first_stage_started":false,"provider_zero_required_after_return":true,"result_digest":"sha256:'"$fallback_digest"'","schema_version":"task_evaluation_scene_configuration_provider_result.v1","status":"blocked"}' > "$RESULT_PATH"
  else
    printf '%s\n' "$fallback_payload" > "$RESULT_PATH"
  fi
fi

exit "$runner_rc"
