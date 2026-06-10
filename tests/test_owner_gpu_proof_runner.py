from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline.simulation_automation import build_simulation_automation
from blueprint_pipeline.owner_gpu_proof_runner import main, run_owner_gpu_proof


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    return capture_root


def _owner_command_script(path: Path, *, write_traces: bool = True) -> None:
    trace_body = (
        """
import json
import os
from pathlib import Path

def write(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

write(os.environ["BLUEPRINT_SCENE_LOAD_TRACE"], {"status": "loaded", "scene_loaded": True})
write(os.environ["BLUEPRINT_SPAWN_TRACE"], {"status": "validated", "spawn_pose_loaded": True})
write(os.environ["BLUEPRINT_ACTION_OR_POLICY_TRACE"], {"status": "completed", "actions": [{"id": "a1"}]})
write(os.environ["BLUEPRINT_ARTIFACT_MANIFEST"], {"status": "complete", "artifacts": [{"path": "trace.log"}]})
print("owner command completed")
"""
        if write_traces
        else 'print("owner command completed without traces")\n'
    )
    path.write_text(trace_body, encoding="utf-8")


def test_owner_gpu_proof_runner_writes_validated_proof(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    command_script = tmp_path / "owner_command.py"
    _owner_command_script(command_script)

    result = run_owner_gpu_proof(
        capture_root=capture_root,
        command=f"{sys.executable} {command_script}",
        owner_system_id="runpod-test",
        simulator_backend="isaac_sim",
        simulator_version="6.0.0",
        gpu_model="L40S 48GB",
        operator_id="operator-1",
        operator_attestation="I ran this command on the owner GPU VM.",
        timeout_seconds=30,
    )

    proof_path = Path(result["proof_path"])
    validation_path = Path(result["validation_manifest_path"])
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    assert result["owner_gpu_simulator_execution_proven"] is True
    assert result["validation_status"] == "accepted"
    assert proof["schema_version"] == "gpu_owner_system_proof.v1"
    assert proof["exit_code"] == 0
    assert proof["robot_readiness_proven"] is False
    assert validation["owner_gpu_simulator_execution_proven"] is True
    assert validation["robot_readiness_proven"] is False
    assert "owner command completed" in (
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_proof"
        / "owner_simulator_stdout.log"
    ).read_text(encoding="utf-8")


def test_owner_gpu_proof_runner_output_is_ingested_by_simulation_automation(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    command_script = tmp_path / "owner_command.py"
    _owner_command_script(command_script)

    proof_result = run_owner_gpu_proof(
        capture_root=capture_root,
        command=f"{sys.executable} {command_script}",
        owner_system_id="runpod-test",
        simulator_backend="isaac_sim",
        simulator_version="6.0.0",
        gpu_model="L40S 48GB",
        operator_id="operator-1",
        operator_attestation="I ran this command on the owner GPU VM.",
        timeout_seconds=30,
    )
    automation_result = build_simulation_automation(capture_root=capture_root)

    automation_root = capture_root / "pipeline" / "simulation_automation"
    gpu_handoff = json.loads((automation_root / "gpu_handoff_packet.json").read_text(encoding="utf-8"))
    proof_boundary = json.loads((automation_root / "proof_boundary.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(
        (automation_root / "simulation_automation_run_manifest.json").read_text(encoding="utf-8")
    )

    assert proof_result["owner_gpu_simulator_execution_proven"] is True
    assert gpu_handoff["owner_gpu_simulator_execution_proven"] is True
    assert "owner_gpu_simulator_execution_not_run" not in gpu_handoff["blockers"]
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["robot_readiness_proven"] is False
    assert run_manifest["owner_gpu_simulator_execution_proven"] is True
    assert run_manifest["robot_readiness_proven"] is False
    assert automation_result["claim_boundary"]["robot_readiness_proven"] is False


def test_owner_gpu_proof_runner_blocks_when_owner_command_omits_traces(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    command_script = tmp_path / "owner_command.py"
    _owner_command_script(command_script, write_traces=False)

    result = run_owner_gpu_proof(
        capture_root=capture_root,
        command=f"{sys.executable} {command_script}",
        owner_system_id="runpod-test",
        simulator_backend="isaac_sim",
        simulator_version="6.0.0",
        gpu_model="L40S 48GB",
        operator_id="operator-1",
        operator_attestation="I ran this command on the owner GPU VM.",
        timeout_seconds=30,
    )

    assert result["owner_gpu_simulator_execution_proven"] is False
    assert result["validation_status"] == "blocked"
    assert "scene_load_trace_owner_proof_artifact_missing" in result["validation_blockers"]
    assert "owner_gpu_scene_load_trace_not_proven" in result["validation_blockers"]


def test_owner_gpu_proof_runner_cli_returns_nonzero_for_incomplete_proof(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    command_script = tmp_path / "owner_command.py"
    _owner_command_script(command_script, write_traces=False)

    exit_code = main(
        [
            "--capture-root",
            str(capture_root),
            "--command",
            f"{sys.executable} {command_script}",
            "--owner-system-id",
            "runpod-test",
            "--simulator-backend",
            "isaac_sim",
            "--simulator-version",
            "6.0.0",
            "--gpu-model",
            "L40S 48GB",
            "--operator-id",
            "operator-1",
            "--operator-attestation",
            "I ran this command on the owner GPU VM.",
            "--timeout-seconds",
            "30",
        ]
    )

    assert exit_code == 1
