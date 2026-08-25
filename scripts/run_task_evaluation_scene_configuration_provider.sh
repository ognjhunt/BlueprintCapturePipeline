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

if [ -d "$RUNTIME_ROOT/toolchain" ]; then
  chmod -R a-w "$RUNTIME_ROOT/toolchain"
fi

if [ -n "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" "$RUNTIME_ROOT/task_evaluation_scene_configuration_provider_runner.py"
  runner_rc=$?
else
  runner_rc=127
fi

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
