from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import owner_gpu_proof_runner as owner_runner
from blueprint_pipeline.owner_gpu_default_smoke_artifacts import (
    main as smoke_artifact_main,
    write_default_smoke_artifacts,
)
from blueprint_pipeline.owner_gpu_proof_runner import main, run_owner_gpu_proof
from blueprint_pipeline.simulation_automation import build_simulation_automation


pytestmark = [pytest.mark.slow, pytest.mark.integration]


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


def test_owner_gpu_default_smoke_artifact_helper_writes_required_outputs(
    tmp_path: Path,
) -> None:
    policy_trace = tmp_path / "owner_action_or_policy_trace.json"
    sim_pov = tmp_path / "owner_sim_robot_pov_evidence_manifest.json"
    artifact_manifest = tmp_path / "owner_artifact_manifest.json"

    result = write_default_smoke_artifacts(
        policy_trace_path=policy_trace,
        sim_robot_pov_evidence_path=sim_pov,
        artifact_manifest_path=artifact_manifest,
        target="walk_to_loading_dock",
        simulator="isaac_sim",
        sim_pov_frames=["frames/front-rgbd-0001.png"],
    )

    policy_payload = json.loads(policy_trace.read_text(encoding="utf-8"))
    sim_pov_payload = json.loads(sim_pov.read_text(encoding="utf-8"))
    artifact_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert policy_payload["default_policy_executed"] is True
    assert policy_payload["actions"][0]["name"] == "walk_to_target"
    assert policy_payload["actions"][0]["target"] == "walk_to_loading_dock"
    assert sim_pov_payload["sim_robot_pov_captured"] is True
    assert sim_pov_payload["frames"] == [
        {"camera": "front_rgbd", "path": "frames/front-rgbd-0001.png"}
    ]
    assert {"kind": "policy_trace", "path": str(policy_trace), "required": True} in (
        artifact_payload["artifacts"]
    )
    assert {"kind": "sim_robot_pov", "path": str(sim_pov), "required": True} in (
        artifact_payload["artifacts"]
    )


def test_owner_gpu_default_smoke_artifact_helper_cli_requires_pov_evidence(
    tmp_path: Path,
) -> None:
    policy_trace = tmp_path / "owner_action_or_policy_trace.json"
    sim_pov = tmp_path / "owner_sim_robot_pov_evidence_manifest.json"
    artifact_manifest = tmp_path / "owner_artifact_manifest.json"

    with pytest.raises(SystemExit) as exc_info:
        smoke_artifact_main(
            [
                "--policy-trace",
                str(policy_trace),
                "--sim-robot-pov-evidence",
                str(sim_pov),
                "--artifact-manifest",
                str(artifact_manifest),
            ]
        )

    assert exc_info.value.code == 2
    assert not policy_trace.exists()
    assert not sim_pov.exists()


def _owner_command_script(
    path: Path,
    *,
    write_traces: bool = True,
    robot_asset_kind: str = "unitree_g1",
) -> None:
    if robot_asset_kind == "unitree_g1":
        robot_asset_expr = """{
    "name": os.environ["BLUEPRINT_ROBOT_ASSET_NAME"],
    "uri_or_path": os.environ["BLUEPRINT_ROBOT_ASSET_URI_OR_PATH"],
    "source": os.environ["BLUEPRINT_ROBOT_ASSET_SOURCE"],
    "asset_class": os.environ["BLUEPRINT_ROBOT_ASSET_CLASS"],
}"""
    else:
        robot_asset_expr = """{
    "name": "procedural_humanoid_proxy",
    "uri_or_path": "generated/mujoco/procedural_humanoid_proxy.xml",
    "source": "generated_mujoco_proxy",
    "asset_class": "humanoid_proxy",
}"""
    trace_body = (
        """
import json
import os
from pathlib import Path

def write(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

robot_asset = __ROBOT_ASSET_EXPR__
write(os.environ["BLUEPRINT_SCENE_LOAD_TRACE"], {
    "status": "loaded",
    "scene_loaded": True,
    "robot_asset": robot_asset,
})
write(os.environ["BLUEPRINT_SPAWN_TRACE"], {
    "status": "validated",
    "spawn_pose_loaded": True,
    "robot_asset": robot_asset,
})
write(os.environ["BLUEPRINT_POLICY_EXECUTION_TRACE"], {
    "status": "completed",
    "default_policy_executed": True,
    "actions": [{
        "id": "a1",
        "name": "walk_to_target",
        "target": os.environ["BLUEPRINT_DEFAULT_SMOKE_POLICY_TARGET"],
        "status": "attempted"
    }],
})
write(os.environ["BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"], {
    "status": "complete",
    "sim_robot_pov_captured": True,
    "frames": [{"camera": "front_rgbd", "path": "owner-pov-frame-0001.png"}],
})
write(os.environ["BLUEPRINT_ARTIFACT_MANIFEST"], {
    "status": "complete",
    "artifacts": [
        {"kind": "policy_trace", "path": os.environ["BLUEPRINT_POLICY_EXECUTION_TRACE"]},
        {"kind": "sim_robot_pov", "path": os.environ["BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"]},
    ],
})
print("owner command completed")
"""
        if write_traces
        else 'print("owner command completed without traces")\n'
    )
    trace_body = trace_body.replace("__ROBOT_ASSET_EXPR__", robot_asset_expr)
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
    assert proof["robot_asset"]["name"] == "Unitree G1"
    assert proof["robot_asset"]["uri_or_path"] == "Robots/Unitree/G1/g1.usd"
    assert proof["rank_fidelity_result_proven"] is False
    assert validation["owner_gpu_simulator_execution_proven"] is True
    assert validation["isaac_sim_execution_proven"] is True
    assert validation["isaac_robot_asset_execution_proven"] is True
    assert validation["unitree_g1_asset_spawned"] is True
    assert validation["owner_gpu_default_policy_execution_proven"] is True
    assert validation["owner_gpu_sim_robot_pov_evidence_proven"] is True
    assert validation["real_robot_pov_evidence_proven"] is False
    assert validation["rank_fidelity_result_proven"] is False
    assert "owner command completed" in (
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_proof"
        / "owner_simulator_stdout.log"
    ).read_text(encoding="utf-8")


def test_owner_gpu_proof_runner_marks_mujoco_g1_without_isaac_claim(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    command_script = tmp_path / "owner_command.py"
    _owner_command_script(command_script)

    result = run_owner_gpu_proof(
        capture_root=capture_root,
        command=f"{sys.executable} {command_script}",
        owner_system_id="local-mujoco-g1-test",
        simulator_backend="mujoco",
        simulator_version="3.9.0",
        gpu_model="local-cpu-mujoco",
        operator_id="operator-1",
        operator_attestation="I ran this command in a local MuJoCo test harness.",
        timeout_seconds=30,
        robot_asset_name="Unitree G1",
        robot_asset_uri_or_path="output/external_assets/mujoco_menagerie/unitree_g1/g1.xml",
        robot_asset_source="google_deepmind_mujoco_menagerie",
        robot_asset_class="humanoid_mjcf",
    )

    validation = json.loads(Path(result["validation_manifest_path"]).read_text(encoding="utf-8"))
    assert result["owner_gpu_simulator_execution_proven"] is True
    assert validation["status"] == "accepted"
    assert validation["simulator_backend"] == "mujoco"
    assert validation["mujoco_g1_asset_spawned"] is True
    assert validation["mujoco_g1_asset_execution_proven"] is True
    assert validation["isaac_sim_execution_proven"] is False
    assert validation["isaac_robot_asset_execution_proven"] is False
    assert validation["claim_boundary"]["mujoco_g1_asset_execution_proven"] is True
    assert validation["claim_boundary"]["isaac_sim_execution_proven"] is False


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
    assert gpu_handoff["claim_boundary"]["simulator_execution_proven"] is True
    assert gpu_handoff["claim_boundary"]["owner_gpu_simulator_execution_proven"] is True
    assert gpu_handoff["claim_boundary"]["owner_gpu_default_policy_execution_proven"] is True
    assert gpu_handoff["claim_boundary"]["owner_gpu_sim_robot_pov_evidence_proven"] is True
    assert gpu_handoff["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert "simulator_execution_completed" not in gpu_handoff["claim_boundary"]["disallowed_claims"]
    assert "robot_ready" in gpu_handoff["claim_boundary"]["disallowed_claims"]
    assert "simulator load trace" not in gpu_handoff["claim_boundary"]["proof_upgrade_requires"]
    assert "action or policy logs" not in gpu_handoff["claim_boundary"]["proof_upgrade_requires"]
    assert "physics/contact validation logs" in gpu_handoff["claim_boundary"]["proof_upgrade_requires"]
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["isaac_sim_execution_proven"] is True
    assert proof_boundary["isaac_robot_asset_execution_proven"] is True
    assert proof_boundary["owner_gpu_default_policy_execution_proven"] is True
    assert proof_boundary["owner_gpu_sim_robot_pov_evidence_proven"] is True
    assert "simulator_execution_completed" not in proof_boundary["claim_boundary"]["disallowed_claims"]
    assert "robot-team policy/action logs beyond the default smoke policy" in proof_boundary["claim_boundary"]["proof_upgrade_requires"]
    assert run_manifest["simulators_run"] is True
    assert "simulator_execution_completed" not in run_manifest["claim_boundary"]["disallowed_claims"]
    assert proof_boundary["real_robot_pov_evidence_proven"] is False
    assert proof_boundary["rank_fidelity_result_proven"] is False
    assert run_manifest["owner_gpu_simulator_execution_proven"] is True
    assert run_manifest["isaac_sim_execution_proven"] is True
    assert run_manifest["isaac_robot_asset_execution_proven"] is True
    assert run_manifest["owner_gpu_default_policy_execution_proven"] is True
    assert run_manifest["owner_gpu_sim_robot_pov_evidence_proven"] is True
    assert run_manifest["rank_fidelity_result_proven"] is False
    assert automation_result["claim_boundary"]["rank_fidelity_result_proven"] is False


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


def test_owner_gpu_proof_runner_blocks_isaac_proxy_robot_asset(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    command_script = tmp_path / "owner_command.py"
    _owner_command_script(command_script, robot_asset_kind="proxy")

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

    validation = json.loads(Path(result["validation_manifest_path"]).read_text(encoding="utf-8"))
    assert result["owner_gpu_simulator_execution_proven"] is False
    assert result["validation_status"] == "blocked"
    assert "owner_gpu_robot_asset_mismatch" in result["validation_blockers"]
    assert "owner_gpu_unitree_g1_asset_not_spawned" in result["validation_blockers"]
    assert validation["isaac_sim_execution_proven"] is False
    assert validation["isaac_robot_asset_execution_proven"] is False


def test_owner_gpu_proof_runner_records_empty_command_error(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    result = run_owner_gpu_proof(
        capture_root=capture_root,
        command="   ",
        owner_system_id="local-test",
        simulator_backend="mujoco",
        simulator_version="3.9.0",
        gpu_model="local-cpu",
        operator_id="operator-1",
        operator_attestation="I attempted this command in a local test harness.",
        timeout_seconds=30,
    )

    proof = json.loads(Path(result["proof_path"]).read_text(encoding="utf-8"))
    stderr = (
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_proof"
        / "owner_simulator_stderr.log"
    ).read_text(encoding="utf-8")
    assert result["command_exit_code"] == 127
    assert result["owner_gpu_simulator_execution_proven"] is False
    assert proof["pass_fail_criteria"]["execution_error"] == "ValueError"
    assert "--command must not be empty" in stderr


def test_owner_gpu_proof_runner_records_timeout_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)

    def raise_timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(
            cmd=["simulator"],
            timeout=7,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(owner_runner.subprocess, "run", raise_timeout)

    result = run_owner_gpu_proof(
        capture_root=capture_root,
        command=f"{sys.executable} -c pass",
        owner_system_id="local-timeout-test",
        simulator_backend="mujoco",
        simulator_version="3.9.0",
        gpu_model="local-cpu",
        operator_id="operator-1",
        operator_attestation="I attempted this command in a local test harness.",
        timeout_seconds=7,
    )

    proof = json.loads(Path(result["proof_path"]).read_text(encoding="utf-8"))
    proof_dir = capture_root / "pipeline" / "simulation_automation" / "owner_gpu_proof"
    assert result["command_exit_code"] == 124
    assert proof["pass_fail_criteria"]["execution_error"] == "timeout_after_7_seconds"
    assert (proof_dir / "owner_simulator_stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (proof_dir / "owner_simulator_stderr.log").read_text(encoding="utf-8") == "partial stderr"


def test_owner_gpu_proof_runner_keeps_external_proof_paths_absolute(tmp_path: Path) -> None:
    outside_path = tmp_path / "external-proof" / "owner_simulator_stdout.log"
    relative_path = Path("owner_gpu_proof") / "owner_simulator_stdout.log"

    assert owner_runner._relative_or_absolute(outside_path, base=tmp_path / "capture") == str(
        outside_path
    )
    assert owner_runner._relative_or_absolute(
        tmp_path / "capture" / relative_path,
        base=tmp_path / "capture",
    ) == str(relative_path)


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
