"""Drive the shipped shell entrypoint through Isaac Lab's noisy launcher boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint
from blueprint_pipeline.native_task_arena_policy_canary_session import PROVIDER_RESULT_FILENAME


pytestmark = pytest.mark.slow


def _run_entrypoint(tmp_path: Path, *, required: bool = False, fault: str = ""):
    runtime = tmp_path / "runtime"
    output = tmp_path / "out"
    inputs = runtime / "runtime_inputs"
    inputs.mkdir(parents=True)
    controls = {"mode": "required_per_cell", "control_ids": [
        "zero_action_negative", "deterministic_scripted_positive"]}
    (inputs / "policy_canary_runtime_inputs.json").write_text(json.dumps({
        "task_success_contract": {"criteria": {"controls": controls} if required else {}},
    }))
    for candidate in ("pi05_droid", "groot_n17_droid"):
        (inputs / f"policy_execution_spec.{candidate}.json").write_text("{}")
        (runtime / f"adp009d_policy_provisioning.{candidate}.sh").write_text(
            f'printf "provision:{candidate}\\n" >> "$OUT_DIR/events.txt"\n'
        )
    (runtime / "adp_arena_provider_runner.py").write_text('''import json, os
from pathlib import Path
def preflight_policy_canary_runtime_directory(runtime, *, physical_gpu_output_root):
    return {"status": "blocked" if os.getenv("ENTRYPOINT_FAULT") == "input" else "passed",
            "candidate_policy_queried": False, "blockers": []}
if __name__ == "__main__":
    output = Path(os.environ["OUT_DIR"])
    stage = os.getenv("BLUEPRINT_POLICY_CANARY_MATRIX_STAGE", "worker")
    with (output / "events.txt").open("a") as stream: stream.write(stage + "\\n")
    if stage != "controls":
        (output / "native_task_arena_policy_canary_session_result.v1.json").write_text(
            json.dumps({"status": "fixture_worker_reached", "candidate_policy_queried": False}))
''')
    wrapper = tmp_path / "isaac-python"
    wrapper.write_text(f'''#!{sys.executable}
import os, subprocess, sys
from pathlib import Path
print('[WARNING] isaaclab.sh is deprecated and will be removed in Isaac Lab 3.1.', flush=True)
args = sys.argv[1:]
program = sys.stdin.read() if args and args[0] == '-' else None
if args[:2] == ['-m', 'blueprint_pipeline.native_task_runtime_source_provision']:
    rc = 0
elif program and 'controls_required(inputs[' in program and os.getenv('ENTRYPOINT_FAULT') in {{'missing', 'invalid'}}:
    if os.environ['ENTRYPOINT_FAULT'] == 'invalid':
        Path(os.environ['OUT_DIR'], 'policy_canary_control_mode.txt').write_text('unexpected\\n')
    rc = 0
else:
    rc = subprocess.run([sys.executable, *args], input=program, text=True, check=False).returncode
print('[INFO] Using Python: "' + sys.executable + '"', flush=True)
raise SystemExit(rc)
''')
    wrapper.chmod(0o700)
    entrypoint = runtime / "entrypoint.sh"
    entrypoint.write_text(_entrypoint().replace("/isaac-sim/python.sh", shlex.quote(str(wrapper))))
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(output),
           "PYTHONPATH": os.pathsep.join([str(runtime), str(source_root)]), "ENTRYPOINT_FAULT": fault}
    result = subprocess.run(["bash", str(entrypoint)], env=env, capture_output=True, text=True,
                            check=False, timeout=20)
    return result, output


@pytest.mark.parametrize("required", [False, True])
def test_actual_entrypoint_ignores_launcher_stdout_for_control_mode(tmp_path, required):
    result, output = _run_entrypoint(tmp_path, required=required)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[INFO] Using Python:" in result.stdout
    events = (output / "events.txt").read_text().splitlines()
    assert events == (["controls"] if required else []) + [
        "provision:pi05_droid", "provision:groot_n17_droid", "policies" if required else "worker"]
    assert json.loads((output / PROVIDER_RESULT_FILENAME).read_text())["status"] == "fixture_worker_reached"
    assert json.loads((output / "policy_canary_static_startup_preflight.v1.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("fault", ["missing", "invalid", "input"])
def test_entrypoint_prepolicy_failure_keeps_typed_gap_and_skips_absent_servers(tmp_path, fault):
    result, output = _run_entrypoint(tmp_path, fault=fault)
    assert result.returncode != 0
    assert not (output / "events.txt").exists()
    failure = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    reason = {"input": "policy_canary_input_preflight_failed", "missing": "policy_canary_control_mode_missing",
              "invalid": "policy_canary_control_mode_invalid"}[fault]
    assert failure["blockers"] == [reason]
    assert failure["visual_evidence"]["media_gap"] == {"type": "before_first_observation", "reason": reason}
    assert failure["candidate_policy_queried"] is False
    assert not list(output.glob("adp009d_policy_server_teardown.*"))
