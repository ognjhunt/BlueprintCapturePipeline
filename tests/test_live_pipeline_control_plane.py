from __future__ import annotations

import json
import os
from pathlib import Path

from blueprint_pipeline.live_pipeline_control_plane import (
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
    LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION,
    run_live_pipeline_control_plane,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path, *, with_webapp_ids: bool = True) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    descriptor: dict[str, object] = {
        "scene_id": "scene-1",
        "capture_id": "capture-1",
    }
    if with_webapp_ids:
        descriptor.update(
            {
                "site_submission_id": "site-submission-1",
                "request_id": "request-1",
                "buyer_request_id": "buyer-request-1",
                "capture_job_id": "capture-job-1",
            }
        )
    _write_json(
        capture_root / "capture_descriptor.json",
        descriptor,
    )
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def _webapp_queue_envelope(capture_root: Path, *, job_id: str = "webapp-job-1") -> dict[str, object]:
    request = {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": job_id,
        "site_package": {
            "capture_root": str(capture_root),
            "site_id": "site-1",
            "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
        },
        "source": {
            "system": "Blueprint-WebApp",
            "site_submission_id": "site-submission-1",
            "request_id": "request-1",
            "buyer_request_id": "buyer-request-1",
            "capture_job_id": "capture-job-1",
        },
        "policy_package": {
            "policy_api_endpoint": {"endpoint_url": "https://robot-team.example/policy"}
        },
    }
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": request,
    }


def _webapp_site_library_queue_envelope(
    capture_root: Path, *, job_id: str = "webapp-job-1"
) -> dict[str, object]:
    buyer_request_id = "buyer-request-1"
    request = {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": job_id,
        "buyer_request_id": buyer_request_id,
        "site_package": {
            "capture_root": str(capture_root),
            "site_submission_id": "site-submission-1",
            "capture_job_id": "capture-job-1",
            "buyer_request_id": buyer_request_id,
            "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
        },
        "owner_system": {
            "name": "Blueprint-WebApp",
            "request_id": job_id,
            "buyer_request_id": buyer_request_id,
            "site_submission_id": "site-submission-1",
            "capture_job_id": "capture-job-1",
        },
        "source": {
            "system": "Blueprint-WebApp",
            "selection_state": {
                "buyer_request_id": buyer_request_id,
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
            },
        },
        "policy_package": {
            "policy_api_endpoint": {"endpoint_url": "https://robot-team.example/policy"}
        },
    }
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": request,
    }


def test_live_pipeline_control_plane_blocks_without_capture_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-control-plane")
    output_path = tmp_path / "control" / "manifest.json"

    result = run_live_pipeline_control_plane(
        load_local_env=False,
        output_path=output_path,
    )

    assert result["schema_version"] == LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION
    assert result["status"] == "blocked"
    assert result["capture_root"] is None
    assert "missing_capture_root" in result["blockers"]
    assert result["inbox_run"]["blockers"] == ["missing_capture_root"]
    assert result["control_plane_boundary"]["simulator_execution_proven"] is False
    assert result["secrets_leaked"] is False
    assert "secret-openai-control-plane" not in json.dumps(result)
    assert output_path.is_file()
    packet_path = Path(result["external_input_packet"]["path"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["schema_version"] == LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION
    assert packet["status"] == "waiting_for_external_inputs"
    assert packet["secrets_leaked"] is False
    assert "secret-openai-control-plane" not in json.dumps(packet)
    assert {item["id"] for item in packet["required_inputs"]} == {
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcomes",
        "robot_team_policy_package",
    }


def test_live_pipeline_control_plane_processes_empty_inbox_without_live_actions(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    inbox_dir = tmp_path / "webapp-job-inbox"

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=inbox_dir,
        load_local_env=False,
        output_path=tmp_path / "control-plane.json",
    )

    queue_manifest = capture_root / "pipeline" / "robot_eval_job_requests" / "inbox_run_manifest.json"

    assert result["status"] == "waiting_for_jobs"
    assert result["inbox_run"]["status"] == "empty"
    assert result["inbox_run"]["processed"] is True
    assert result["inbox_run"]["processed_count"] == 0
    assert result["execution_config"]["allow_simulator_execution"] is False
    assert result["execution_config"]["allow_rollout_vision_labeling"] is False
    assert result["execution_config"]["allow_delivery_upload"] is False
    assert result["operator_config"]["live_agents_sdk_allowed_by_control_plane"] is False
    assert result["operator_config"]["live_codex_sdk_allowed_by_control_plane"] is False
    assert queue_manifest.is_file()
    packet_info = result["external_input_packet"]
    packet_path = Path(packet_info["path"])
    packet_markdown_path = Path(packet_info["markdown_path"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    enablement_input_ids = {item["id"] for item in packet["enablement_inputs"]}

    assert packet_info["schema_version"] == LIVE_PIPELINE_EXTERNAL_INPUT_PACKET_SCHEMA_VERSION
    assert packet_info["status"] == "waiting_for_external_inputs"
    assert packet_info["required_input_count"] == 4
    assert packet_info["enablement_input_count"] == 4
    assert packet_path.is_file()
    assert packet_markdown_path.is_file()
    assert required_input_ids == {
        "isaac_lab_arena_owner_evidence",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcomes",
        "robot_team_policy_package",
    }
    assert "webapp_upstream_truth" not in required_input_ids
    assert enablement_input_ids == {
        "rollout_vision_labeling",
        "delivery_upload",
        "live_agents_operator",
        "live_codex_operator",
    }
    assert packet["proof_boundary"]["simulator_execution_proven"] is False
    assert (
        packet["example_robot_eval_job_request"]["source"]["capture_job_id"]
        == "REPLACE_WITH_CAPTURE_JOB_ID"
    )
    assert "Live Pipeline External Input Packet" in packet_markdown_path.read_text(
        encoding="utf-8"
    )


def test_live_pipeline_control_plane_accepts_matching_webapp_inbox_truth(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    inbox_dir = tmp_path / "webapp-job-inbox"
    _write_json(inbox_dir / "webapp-job-1.json", _webapp_queue_envelope(capture_root))

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=inbox_dir,
        process_inbox=False,
        load_local_env=False,
        output_path=tmp_path / "control-plane.json",
    )

    packet = json.loads(
        Path(result["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    next_inputs = " ".join(result["next_inputs_needed"])

    assert result["webapp_inbox_truth"]["status"] == "ready"
    assert result["webapp_inbox_truth"]["accepted_request_ids"] == ["webapp-job-1"]
    assert result["effective_webapp_upstream_truth_ready"] is True
    assert packet["webapp_upstream_truth"]["ready"] is True
    assert packet["webapp_upstream_truth"]["job_request_inbox_status"] == "ready"
    assert "webapp_upstream_truth" not in required_input_ids
    assert "WebApp capture root" not in next_inputs
    assert "Isaac Lab-Arena" in next_inputs
    assert "live closure evidence" in next_inputs
    assert "deployment outcome" in next_inputs
    assert "policy package" not in next_inputs


def test_live_pipeline_control_plane_accepts_webapp_site_library_id_locations(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    inbox_dir = tmp_path / "webapp-job-inbox"
    _write_json(
        inbox_dir / "webapp-job-1.json",
        _webapp_site_library_queue_envelope(capture_root),
    )

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=inbox_dir,
        process_inbox=False,
        load_local_env=False,
        output_path=tmp_path / "control-plane.json",
    )

    packet = json.loads(
        Path(result["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    candidate = result["webapp_inbox_truth"]["candidates"][0]

    assert result["webapp_inbox_truth"]["status"] == "ready"
    assert candidate["fields_present"] == {
        "site_submission_id": True,
        "request_id": True,
        "buyer_request_id": True,
        "capture_job_id": True,
    }
    assert candidate["missing_fields"] == []
    assert candidate["policy_package_ready"] is True
    assert candidate["policy_package_ready_modalities"] == ["policy_api_endpoint"]
    assert result["effective_webapp_upstream_truth_ready"] is True
    assert "webapp_upstream_truth" not in required_input_ids
    assert "robot_team_policy_package" not in required_input_ids


def test_live_pipeline_control_plane_keeps_policy_package_input_for_invalid_inline_refs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    inbox_dir = tmp_path / "webapp-job-inbox"
    envelope = _webapp_site_library_queue_envelope(capture_root)
    job_request = envelope["job_request"]
    assert isinstance(job_request, dict)
    job_request["policy_package"] = {
        "docker_container": {"image_ref": "registry.example/policy:latest"}
    }
    _write_json(inbox_dir / "webapp-job-1.json", envelope)

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=inbox_dir,
        process_inbox=False,
        load_local_env=False,
        output_path=tmp_path / "control-plane.json",
    )

    packet = json.loads(
        Path(result["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    candidate = result["webapp_inbox_truth"]["candidates"][0]

    assert result["webapp_inbox_truth"]["status"] == "ready"
    assert candidate["accepted_as_webapp_truth"] is True
    assert candidate["policy_package_ready"] is False
    assert candidate["policy_package_selected_modalities"] == ["docker_container"]
    assert candidate["policy_package_missing_inputs"] == {
        "docker_container": ["policy_package.docker_container.digest"]
    }
    assert "webapp_upstream_truth" not in required_input_ids
    assert "robot_team_policy_package" in required_input_ids


def test_live_pipeline_control_plane_rejects_mismatched_webapp_inbox_truth(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    inbox_dir = tmp_path / "webapp-job-inbox"
    other_capture_root = tmp_path / "other" / "captures" / "capture-2"
    _write_json(inbox_dir / "webapp-job-1.json", _webapp_queue_envelope(other_capture_root))

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=inbox_dir,
        process_inbox=False,
        load_local_env=False,
        output_path=tmp_path / "control-plane.json",
    )

    packet = json.loads(
        Path(result["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}

    assert result["webapp_inbox_truth"]["status"] == "blocked"
    assert result["webapp_inbox_truth"]["accepted_request_count"] == 0
    assert "no_job_request_matches_configured_capture_root" in result["webapp_inbox_truth"][
        "blockers"
    ]
    assert result["effective_webapp_upstream_truth_ready"] is False
    assert "webapp_upstream_truth" in required_input_ids
    assert packet["required_inputs"][0]["webapp_inbox_truth"]["request_count"] == 1


def test_live_pipeline_control_plane_loads_env_paths_and_redacts_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    inbox_dir = tmp_path / "env-configured-inbox"
    output_path = tmp_path / "env-configured-control-plane.json"
    secret_value = "secret-sync-token-control-plane"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"BLUEPRINT_PIPELINE_CAPTURE_ROOT={capture_root}",
                f"BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX={inbox_dir}",
                f"BLUEPRINT_CONTROL_PLANE_OUTPUT_PATH={output_path}",
                "BLUEPRINT_CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING=true",
                "BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND=python -c 'print(1)'",
                f"PIPELINE_SYNC_TOKEN={secret_value}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = run_live_pipeline_control_plane()

    serialized = json.dumps(result)
    assert result["capture_root"] == str(capture_root.resolve())
    assert result["job_request_inbox"] == str(inbox_dir.resolve())
    assert result["output_path"] == str(output_path.resolve())
    assert result["execution_config"]["allow_rollout_vision_labeling"] is True
    assert result["env_files"]["files"]
    assert "PIPELINE_SYNC_TOKEN" in (
        result["env_files"]["loaded_keys"] + result["env_files"]["skipped_existing_keys"]
    )
    assert result["secrets_leaked"] is False
    assert secret_value not in serialized
    assert os.environ.get("PIPELINE_SYNC_TOKEN") != secret_value
    assert output_path.is_file()


def test_live_pipeline_control_plane_next_inputs_follow_ready_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    inbox_dir = tmp_path / "webapp-job-inbox"
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true")

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=inbox_dir,
        vision_labeling_command="python -c 'print(1)'",
        delivery_command="python -c 'print(1)'",
        load_local_env=False,
        output_path=tmp_path / "control-plane.json",
    )

    next_inputs = " ".join(result["next_inputs_needed"])
    assert "capture root" not in next_inputs
    assert "job request inbox" not in next_inputs
    assert "vision-labeling command" not in next_inputs
    assert "delivery command" not in next_inputs
    assert "Isaac Lab-Arena" in next_inputs
    assert "live closure evidence" in next_inputs
    assert "deployment outcome" in next_inputs
    assert "policy package" in next_inputs


def test_live_pipeline_control_plane_records_simulator_command_configuration(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)

    result = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        simulator="isaac_lab_arena",
        allowed_simulators=("isaac_lab_arena",),
        simulator_commands=("isaac_lab_arena=python -c 'print(1)'",),
        simulator_audit_command="python -c 'print(1)'",
        load_local_env=False,
        output_path=tmp_path / "command-control.json",
    )

    assert result["execution_config"]["simulator"] == "isaac_lab_arena"
    assert result["execution_config"]["allowed_simulators"] == ["isaac_lab_arena"]
    assert result["execution_config"]["simulator_commands_configured"] == ["isaac_lab_arena"]
    assert result["execution_config"]["allow_simulator_execution"] is False
    assert result["setup_status"] in {
        "local_ready_live_external_blocked",
        "ready_for_live_external_execution",
        "blocked",
    }
