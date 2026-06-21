from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.live_pipeline_control_plane import (
    LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
    run_live_pipeline_control_plane,
)
from blueprint_pipeline.live_pipeline_input_intake import build_live_pipeline_input_intake
from blueprint_pipeline.live_pipeline_proof_audit import (
    LIVE_PIPELINE_PROOF_AUDIT_SCHEMA_VERSION,
    _artifact,
    _audit_status,
    _blocker_packet_audit,
    _enablement_input_ids,
    _proof_violations,
    _read_mapping,
    _required_input_ids,
    build_live_pipeline_proof_audit,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path, *, with_webapp_ids: bool = True) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    descriptor: dict[str, object] = {"scene_id": "scene-1", "capture_id": "capture-1"}
    if with_webapp_ids:
        descriptor.update(
            {
                "site_submission_id": "site-submission-1",
                "request_id": "request-1",
                "buyer_request_id": "buyer-request-1",
                "capture_job_id": "capture-job-1",
            }
        )
    _write_json(capture_root / "capture_descriptor.json", descriptor)
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def _live_closure_evidence(path: Path, *, job_id: str = "closure-job-1") -> Path:
    _write_json(
        path,
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "job_id": job_id,
            "review_acceptance": {"accepted": True, "reviewer": "owner-reviewer"},
            "delivery": {
                "storage_upload_performed": True,
                "signed_urls": ["https://delivery.example/signed/package-1"],
                "entitlement_verified": True,
            },
            "safety_contact_physics": {
                "physics_contact_validated": True,
                "safety_validated": True,
                "robot_readiness_proven": True,
                "methodology_uri_or_path": "owner://methodology",
                "contact_validation_uri_or_path": "owner://contact",
                "safety_validation_uri_or_path": "owner://safety",
                "operator_attestation": {
                    "attested_by": "safety-owner",
                    "attestation": "Owner accepted contact, physics, and safety evidence.",
                },
            },
        },
    )
    return path


def _deployment_outcomes(
    path: Path,
    *,
    job_id: str = "deployment-job-1",
    include_evidence: bool = True,
    include_prediction_key: bool = True,
) -> Path:
    record: dict[str, object] = {
        "outcome_id": "pilot-outcome-1",
        "task_id": "place_return_in_bin",
        "scenario_id": "scenario_place_return_in_bin_mobile",
        "actual_success": True,
    }
    if include_prediction_key:
        record["scenario_eval_run_id"] = "scenario-eval-run-1"
        record["scenario_variation_instance_id"] = "scenario-variation-1"
    if include_evidence:
        record["evidence_refs"] = {"pilot_log": "owner://pilot/pilot-outcome-1"}
    _write_json(
        path,
        {
            "schema_version": "deployment_outcome_manifest.v1",
            "job_id": job_id,
            "records": [record],
        },
    )
    return path


def _policy_package(path: Path, *, job_id: str = "policy-job-1") -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_team_policy_package.v1",
            "job_id": job_id,
            "policy_package": {
                "policy_api_endpoint": {
                    "endpoint_url": "https://robot-team.example/policy"
                }
            },
        },
    )
    return path


def _blocker_packet(input_id: str) -> dict[str, str]:
    return {
        "id": input_id,
        "owner": "owner",
        "required_input": "input",
        "safe_proof_command": "python -m blueprint_pipeline.noop",
        "retry_condition": "input staged",
        "resume_target": "live_pipeline_control_plane",
        "disallowed_workaround": "do not fabricate evidence",
    }


def test_live_pipeline_proof_audit_private_helpers_cover_malformed_shapes(
    tmp_path: Path,
) -> None:
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        _read_mapping(non_object)

    assert _artifact(None) == {"path": None, "exists": False, "size_bytes": 0}
    assert _required_input_ids({"required_inputs": "not-a-list"}) == []
    assert _enablement_input_ids({"enablement_inputs": "not-a-list"}) == []
    assert _proof_violations(
        {"robot_readiness_proven": True},
        artifact_name="manifest",
    )[0]["field"] == "robot_readiness_proven"
    assert _audit_status(internal_blockers=[], required_input_ids=[], require_live_ready=False) == (
        "passed_live_ready_inputs_present"
    )

    audit = _blocker_packet_audit(
        {
            "required_inputs": [
                "skip",
                {"id": ""},
                {"id": "input-1", "blocker_packet": {"id": "other"}},
            ],
            "enablement_inputs": [{"id": "input-2", "blocker_packet": "missing"}],
        }
    )
    assert audit["status"] == "failed"
    assert audit["invalid_blocker_packet_fields"]["input-1"] == [
        "owner",
        "required_input",
        "safe_proof_command",
        "retry_condition",
        "resume_target",
        "disallowed_workaround",
        "id_matches_input",
    ]
    assert audit["missing_blocker_packet_input_ids"] == ["input-2"]


def test_live_pipeline_proof_audit_reports_malformed_control_plane_edges(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    packet_path = control_dir / "packet.json"
    setup_path = control_dir / "missing_setup.json"
    staged_path = control_dir / "missing_staged.json"
    manifest_path = control_dir / "live_pipeline_control_plane_manifest.json"
    output_path = control_dir / "explicit_audit.json"
    _write_json(
        packet_path,
        {
            "schema_version": "wrong",
            "status": "packet-actual",
            "secrets_leaked": True,
            "required_inputs": [
                {
                    "id": "unexpected_input",
                    "blocker_packet": {
                        **_blocker_packet("unexpected_input"),
                        "id": "wrong",
                    },
                }
            ],
        },
    )
    _write_json(
        manifest_path,
        {
            "schema_version": "wrong",
            "status": "blocked",
            "setup_status": "blocked",
            "secrets_leaked": True,
            "effective_webapp_upstream_truth_ready": False,
            "external_input_packet": {
                "path": str(packet_path),
                "markdown_path": str(control_dir / "missing_packet.md"),
                "status": "packet-expected",
            },
            "setup_manifest_path": str(setup_path),
            "staged_inputs": {"path": str(staged_path), "status": "ready"},
        },
    )

    audit = build_live_pipeline_proof_audit(
        manifest_path=manifest_path,
        output_path=output_path,
    )

    assert audit["output_path"] == str(output_path.resolve())
    assert set(audit["internal_blockers"]) >= {
        "control_plane_manifest_schema_mismatch",
        "control_plane_manifest_reports_secret_leak",
        "external_input_packet_schema_mismatch",
        "external_input_packet_reports_secret_leak",
        "external_input_packet_status_mismatch",
        "external_input_packet_markdown_missing",
        "setup_manifest_missing",
        "staged_inputs_ready_but_missing",
        "external_input_packet_invalid_blocker_packets",
        "unexpected_required_input:unexpected_input",
    }

    missing_packet_manifest = control_dir / "missing_packet_manifest.json"
    _write_json(
        missing_packet_manifest,
        {
            "schema_version": LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
            "external_input_packet": {"path": str(control_dir / "absent.json")},
            "setup_manifest_path": str(setup_path),
            "staged_inputs": {"status": "blocked"},
        },
    )
    missing_packet_audit = build_live_pipeline_proof_audit(
        manifest_path=missing_packet_manifest,
        output_path=control_dir / "missing_packet_audit.json",
    )
    assert "external_input_packet_missing" in missing_packet_audit["internal_blockers"]
    assert "staged_inputs_blocked" in missing_packet_audit["internal_blockers"]


def test_live_pipeline_proof_audit_direct_main_reports_blocker_counts(
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    manifest_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=manifest_path,
    )

    assert main(["--manifest-path", str(manifest_path)]) == 0
    assert "external_blockers=" in capsys.readouterr().out

    packet_path = tmp_path / "control" / "live_pipeline_external_input_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["schema_version"] = "wrong"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    assert (
        main(
            [
                "--manifest-path",
                str(manifest_path),
                "--output-path",
                str(tmp_path / "control" / "direct_main_audit.json"),
                "--require-live-ready",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "internal_blockers=" in output
    assert "external_blockers=" in output


def test_live_pipeline_proof_audit_passes_when_external_inputs_are_blocked(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["schema_version"] == LIVE_PIPELINE_PROOF_AUDIT_SCHEMA_VERSION
    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
        "real_robot_pov_evidence",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcomes",
        "robot_team_policy_package",
    ]
    assert audit["live_readiness"]["webapp_upstream_truth_ready"] is False
    assert audit["live_readiness"]["owner_arena_evidence_ready"] is False
    assert audit["live_readiness"]["live_closure_evidence_ready"] is False
    assert audit["live_readiness"]["deployment_outcomes_ready"] is False
    assert audit["live_readiness"]["policy_package_ready"] is False
    assert audit["proof_boundary"]["simulator_execution_proven"] is False
    assert Path(str(audit["output_path"])).is_file()


def test_live_pipeline_proof_audit_can_require_live_ready_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(
        manifest_path=output_path,
        require_live_ready=True,
    )

    assert audit["status"] == "failed_live_ready_required"
    assert "required_live_inputs_missing" in audit["internal_blockers"]


def test_live_pipeline_proof_audit_fails_on_proof_overclaim(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    packet_path = tmp_path / "control" / "live_pipeline_external_input_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["proof_boundary"]["robot_readiness_proven"] = True
    packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "failed"
    assert "forbidden_proof_boundary_upgrade" in audit["internal_blockers"]
    assert audit["proof_violations"][0]["field"] == "proof_boundary.robot_readiness_proven"


def test_live_pipeline_proof_audit_requires_resumable_blocker_packets(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    packet_path = tmp_path / "control" / "live_pipeline_external_input_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    del packet["required_inputs"][0]["blocker_packet"]
    packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "failed"
    assert "external_input_packet_missing_blocker_packets" in audit["internal_blockers"]
    assert audit["blocker_packet_audit"]["missing_blocker_packet_input_ids"] == [
        packet["required_inputs"][0]["id"]
    ]


def test_live_pipeline_proof_audit_accepts_valid_staged_arena_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    arena_results = tmp_path / "arena-results"
    _write_json(
        arena_results / "rollout_manifest.json",
        {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "scenario_id": "scenario-1",
                    "status": "success",
                    "success": True,
                }
            ]
        },
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    build_live_pipeline_input_intake(
        manifest_path=output_path,
        arena_results_dir=arena_results,
        stage_arena_results=True,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "real_robot_pov_evidence",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcomes",
        "robot_team_policy_package",
    ]
    assert audit["staged_inputs_audit"]["status"] == "ready"
    assert audit["staged_inputs_audit"]["arena_results_ready"] is True
    assert audit["staged_inputs_audit"]["live_closure_evidence_ready"] is False
    assert audit["staged_inputs_audit"]["deployment_outcomes_ready"] is False
    assert audit["staged_inputs_audit"]["policy_package_ready"] is False


def test_live_pipeline_proof_audit_accepts_valid_staged_closure_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    evidence_path = _live_closure_evidence(tmp_path / "incoming" / "closure.json")
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    build_live_pipeline_input_intake(
        manifest_path=output_path,
        live_closure_evidence=evidence_path,
        stage_live_closure_evidence=True,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
        "real_robot_pov_evidence",
        "real_world_deployment_outcomes",
        "robot_team_policy_package",
    ]
    assert audit["staged_inputs_audit"]["live_closure_evidence_ready"] is True
    assert (
        audit["staged_inputs_audit"]["live_closure_evidence_job_id"]
        == "closure-job-1"
    )


def test_live_pipeline_proof_audit_accepts_valid_staged_deployment_outcomes(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    outcomes_path = _deployment_outcomes(tmp_path / "incoming" / "deployment.json")
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    build_live_pipeline_input_intake(
        manifest_path=output_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
        "real_robot_pov_evidence",
        "live_robot_eval_closure_evidence",
        "robot_team_policy_package",
    ]
    assert audit["staged_inputs_audit"]["deployment_outcomes_ready"] is True
    assert audit["staged_inputs_audit"]["deployment_outcomes_owner_evidence_ready"] is True
    assert audit["staged_inputs_audit"]["deployment_outcomes_job_id"] == "deployment-job-1"
    assert audit["staged_inputs_audit"]["deployment_outcome_record_count"] == 1
    assert audit["staged_inputs_audit"]["deployment_outcome_owner_evidence_record_count"] == 1
    assert audit["staged_inputs_audit"]["deployment_outcome_missing_owner_evidence_record_ids"] == []
    assert audit["live_readiness"]["deployment_outcomes_ready"] is True
    assert audit["live_readiness"]["deployment_outcomes_owner_evidence_ready"] is True


def test_live_pipeline_proof_audit_treats_missing_prediction_keys_as_external_blocker(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    outcomes_path = _deployment_outcomes(
        tmp_path / "incoming" / "deployment.json",
        include_prediction_key=False,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    build_live_pipeline_input_intake(
        manifest_path=output_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert "predicted_vs_actual_exact_match_keys" in audit["external_blockers"]
    assert audit["staged_inputs_audit"]["deployment_outcomes_ready"] is True
    assert (
        audit["staged_inputs_audit"]["deployment_outcomes_prediction_match_keys_ready"]
        is False
    )
    assert audit["staged_inputs_audit"]["deployment_outcome_prediction_match_key_record_count"] == 0
    assert audit["staged_inputs_audit"]["deployment_outcome_missing_prediction_match_key_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert audit["live_readiness"]["deployment_outcomes_ready"] is True
    assert audit["live_readiness"]["deployment_outcomes_prediction_match_keys_ready"] is False
    assert (
        audit["goal_requirement_audit"]["predicted_vs_actual_exact_match_keys"]["status"]
        == "external_input_missing"
    )


def test_live_pipeline_proof_audit_keeps_owner_evidence_blocker_for_outcome_records(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    outcomes_path = _deployment_outcomes(
        tmp_path / "incoming" / "deployment.json",
        include_evidence=False,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    build_live_pipeline_input_intake(
        manifest_path=output_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
        "real_robot_pov_evidence",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcome_owner_evidence",
        "robot_team_policy_package",
    ]
    assert audit["staged_inputs_audit"]["deployment_outcomes_ready"] is True
    assert audit["staged_inputs_audit"]["deployment_outcomes_owner_evidence_ready"] is False
    assert audit["staged_inputs_audit"]["deployment_outcome_record_count"] == 1
    assert audit["staged_inputs_audit"]["deployment_outcome_owner_evidence_record_count"] == 0
    assert audit["staged_inputs_audit"]["deployment_outcome_missing_owner_evidence_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert audit["live_readiness"]["deployment_outcomes_ready"] is True
    assert audit["live_readiness"]["deployment_outcomes_owner_evidence_ready"] is False
    assert (
        audit["goal_requirement_audit"]["real_world_deployment_outcomes"]["status"]
        == "ready"
    )
    assert (
        audit["goal_requirement_audit"]["real_world_deployment_outcome_owner_evidence"]["status"]
        == "external_input_missing"
    )


def test_live_pipeline_proof_audit_accepts_valid_staged_policy_package(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    package_path = _policy_package(tmp_path / "incoming" / "policy-package.json")
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    build_live_pipeline_input_intake(
        manifest_path=output_path,
        policy_package=package_path,
        stage_policy_package=True,
    )
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
        "real_robot_pov_evidence",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcomes",
    ]
    assert audit["staged_inputs_audit"]["policy_package_ready"] is True
    assert audit["staged_inputs_audit"]["policy_package_job_id"] == "policy-job-1"
    assert audit["staged_inputs_audit"]["policy_package_selected_modalities"] == [
        "policy_api_endpoint"
    ]


def test_live_pipeline_proof_audit_fails_on_malformed_staged_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    _write_json(
        tmp_path / "control" / "live_pipeline_staged_inputs.json",
        {"schema_version": "wrong"},
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "failed"
    assert "staged_inputs_schema_mismatch" in audit["internal_blockers"]


def test_live_pipeline_proof_audit_module_cli(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    env = os.environ.copy()
    src_root = Path.cwd() / "src"
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_proof_audit",
            "--manifest-path",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=passed_external_inputs_blocked" in completed.stdout
    audit = json.loads(
        (tmp_path / "control" / "live_pipeline_proof_boundary_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "passed_external_inputs_blocked"
