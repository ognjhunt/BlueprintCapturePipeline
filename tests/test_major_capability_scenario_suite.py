from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import major_capability_scenario_suite as major_suite
from blueprint_pipeline.major_capability_scenario_suite import (
    EVALUATION_METHOD_ID,
    MAJOR_CAPABILITY_SCENARIO_SUITE_SCHEMA_VERSION,
    build_major_capability_scenario_suite,
    major_capability_scenario_definitions,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_major_capability_artifacts(capture_root: Path, *, job_id: str) -> None:
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / job_id
    presentation_dir = capture_root / "pipeline" / "presentation_world"
    evaluation_prep_dir = capture_root / "pipeline" / "evaluation_prep"
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    autoresearch_dir = job_dir / "policy_autoresearch"

    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "schema_version": "blueprint_raw_capture_manifest.v1",
            "capture_id": "capture-realistic-001",
            "site_id": "lightwheel-kitchen",
            "workflow": "inspect prep station and move tote",
            "zone": "prep-line-a",
            "success_criteria": [
                "robot reaches the selected waypoint",
                "tote transfer attempts are labeled success or failure",
            ],
            "provenance": {"raw_capture_authoritative": True},
        },
    )
    _write_json(
        robot_eval_dir / "robot_eval_dataset_manifest.json",
        {
            "schema_version": "real_site_robot_eval_dataset_manifest.v1",
            "status": "ready_for_robot_eval_packaging",
            "site_card_count": 1,
            "task_card_count": 2,
            "scenario_card_count": 2,
            "eval_card_count": 2,
            "proof_boundary": {
                "raw_capture_authoritative": True,
                "downstream_artifacts_are_supporting_evidence": True,
                "public_claim_upgrade_allowed": False,
            },
        },
    )
    _write_json(robot_eval_dir / "site_card.json", {"site_id": "lightwheel-kitchen"})
    _write_json(
        robot_eval_dir / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "cards": [
                {"task_id": "navigate_to_station"},
                {"task_id": "move_tote"},
            ],
        },
    )
    _write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "cards": [
                {"scenario_id": "prep_station_navigation"},
                {"scenario_id": "tote_transfer_under_occlusion"},
            ],
        },
    )
    _write_json(
        robot_eval_dir / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "cards": [
                {"eval_id": "navigation_success"},
                {"eval_id": "transfer_failure_labeling"},
            ],
        },
    )
    _write_json(
        robot_eval_dir / "proof_boundaries.json",
        {
            "raw_capture_authoritative": True,
            "generated_artifacts_are_support_only": True,
        },
    )
    _write_json(
        robot_eval_dir / "rights_packet.json",
        {"status": "accepted", "commercialization_allowed": True},
    )

    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "buyer_request_id": "buyer-lightwheel-001",
            "site_package": {"capture_root": str(capture_root)},
        },
    )
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run-navigation-001",
                    "scenario_variation_instance_id": "variation-navigation-001",
                },
                {
                    "scenario_eval_run_id": "run-transfer-001",
                    "scenario_variation_instance_id": "variation-transfer-001",
                },
            ],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "normalized_attempt_trace.v1",
            "attempts": [
                {"attempt_id": "attempt-navigation-001", "success": True},
                {"attempt_id": "attempt-transfer-001", "success": False},
            ],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "failure_labels.v1",
            "labels": [
                {"attempt_id": "attempt-transfer-001", "label": "occluded_handle"},
            ],
        },
    )
    _write_json(
        job_dir / "policy_execution_trace.json",
        {"schema_version": "policy_execution_trace.v1", "events": [{"kind": "step"}]},
    )
    _write_json(
        job_dir / "proof_boundary.json",
        {
            "simulator_execution_proven": True,
            "robot_policy_execution_proven": True,
            "real_world_outcome_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )

    _write_json(
        job_dir / "dataset_card.json",
        {
            "schema_version": "post_training_dataset_card.v1",
            "status": "ready",
            "episode_count": 2,
        },
    )
    _write_json(
        job_dir / "license_manifest.json",
        {"schema_version": "license_manifest.v1", "rights_status": "accepted"},
    )
    _write_json(
        job_dir / "package_index.json",
        {
            "schema_version": "package_index.v1",
            "files": [
                "dataset_card.json",
                "license_manifest.json",
                "post_training_data_package_export_manifest.json",
            ],
        },
    )
    _write_json(
        job_dir / "checksums.json",
        {"schema_version": "checksums.v1", "files": {"dataset_card.json": "sha256:test"}},
    )
    _write_json(
        job_dir / "post_training_data_package_export_manifest.json",
        {
            "schema_version": "post_training_data_package_export.v1",
            "status": "exported",
            "episode_count": 2,
            "export_formats": {"jsonl": {"format_written": True}},
            "claim_boundary": {
                "post_training_package_exported": True,
                "training_completed": False,
                "robot_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )

    _write_json(
        presentation_dir / "presentation_world_manifest.json",
        {
            "schema_version": "presentation_world_manifest.v1",
            "status": "ready",
            "site_world_id": "lightwheel-kitchen-world",
        },
    )
    _write_json(
        presentation_dir / "runtime_demo_manifest.json",
        {
            "schema_version": "runtime_demo_manifest.v1",
            "status": "ready",
            "session_contract_version": "site_world_runtime.v1",
            "hosted_session_artifact": True,
            "model_backend": "replaceable_fixture_backend",
        },
    )
    _write_json(
        evaluation_prep_dir / "site_world_registration.json",
        {
            "schema_version": "site_world_registration.v1",
            "site_world_id": "lightwheel-kitchen-world",
            "runtime_registration_ready": True,
        },
    )
    _write_json(
        evaluation_prep_dir / "site_world_health.json",
        {"schema_version": "site_world_health.v1", "status": "healthy", "model_ready": True},
    )

    _write_json(
        automation_dir / "simulation_automation_run_manifest.json",
        {
            "schema_version": "simulation_automation_run_manifest.v1",
            "status": "ready_for_simulation_support",
            "scenario_variation_count": 2,
            "generated_support_assets": [
                "episode_specs",
                "scenario_variations",
                "simulator_handoff_packet",
            ],
        },
    )
    _write_json(
        automation_dir / "proof_boundary.json",
        {
            "world_model_support_assets_generated": True,
            "simulator_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )
    _write_json(
        capture_root / "pipeline" / "provider_preview_qa_manifest.json",
        {
            "schema_version": "provider_preview_qa_manifest.v1",
            "status": "provider_preview_packet_ready",
            "privacy_safe_input_verified": True,
            "raw_video_bypass_used": False,
        },
    )
    _write_json(
        capture_root / "pipeline" / "production_handoff_readiness_manifest.json",
        {
            "schema_version": "production_handoff_readiness_manifest.v1",
            "status": "ready_except_owner_gpu_simulator_execution",
            "public_claim_upgrade_allowed": False,
        },
    )
    _write_json(
        autoresearch_dir / "policy_autoresearch_report.json",
        {
            "schema_version": "policy_autoresearch_report.v1",
            "status": "completed",
            "heldout_eval": {"success_rate": 0.95, "safety_contact_failures": 0},
            "proof_boundary": {
                "simulator_execution_proven": True,
                "robot_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )


def test_major_capability_scenarios_define_five_scenarios_with_one_method() -> None:
    scenarios = major_capability_scenario_definitions()

    assert len(scenarios) == 5
    assert {scenario["evaluation_method_id"] for scenario in scenarios} == {EVALUATION_METHOD_ID}
    assert {scenario["scenario_id"] for scenario in scenarios} == {
        "capture_to_robot_eval_artifacts",
        "task_evaluation_run_execution",
        "post_training_data_package_export",
        "hosted_runtime_session",
        "support_assets_trust_and_policy_improvement",
    }
    assert all(scenario["success_criteria"] for scenario in scenarios)


def test_major_capability_scenario_suite_passes_and_records_evidence(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-lightwheel-001"
    _seed_major_capability_artifacts(capture_root, job_id=job_id)

    suite = build_major_capability_scenario_suite(capture_root=capture_root, job_id=job_id)

    assert suite["schema_version"] == MAJOR_CAPABILITY_SCENARIO_SUITE_SCHEMA_VERSION
    assert suite["status"] == "passed"
    assert suite["evaluation_method"]["method_id"] == EVALUATION_METHOD_ID  # type: ignore[index]
    assert suite["summary"]["scenario_count"] == 5  # type: ignore[index]
    assert suite["summary"]["passed_count"] == 5  # type: ignore[index]
    assert suite["summary"]["failed_count"] == 0  # type: ignore[index]
    assert suite["conditions"]["same_conditions_applied"] is True  # type: ignore[index]
    assert suite["conditions"]["network_calls_allowed"] is False  # type: ignore[index]
    assert suite["conditions"]["external_provider_calls_allowed"] is False  # type: ignore[index]
    assert suite["conditions"]["job_id"] == job_id  # type: ignore[index]

    for scenario in suite["scenarios"]:  # type: ignore[index]
        assert scenario["status"] == "passed"
        assert scenario["evaluation_method_id"] == EVALUATION_METHOD_ID
        assert scenario["evidence"]
        assert {item["status"] for item in scenario["evidence"]} == {"passed"}

    report_path = Path(suite["artifacts"]["report"])  # type: ignore[index]
    markdown_path = Path(suite["artifacts"]["report_markdown"])  # type: ignore[index]
    assert report_path.is_file()
    assert markdown_path.is_file()

    persisted = _read_json(report_path)
    assert persisted["status"] == "passed"
    assert "Support Assets, Trust, And Policy Improvement" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_major_capability_scenario_suite_fails_closed_on_overclaimed_package(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-lightwheel-001"
    _seed_major_capability_artifacts(capture_root, job_id=job_id)
    export_manifest = (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / job_id
        / "post_training_data_package_export_manifest.json"
    )
    payload = _read_json(export_manifest)
    claim_boundary = dict(payload["claim_boundary"])  # type: ignore[index]
    claim_boundary["public_claim_upgrade_allowed"] = True
    payload["claim_boundary"] = claim_boundary
    _write_json(export_manifest, payload)

    suite = build_major_capability_scenario_suite(capture_root=capture_root, job_id=job_id)

    assert suite["status"] == "failed"
    assert suite["summary"]["passed_count"] == 4  # type: ignore[index]
    assert suite["summary"]["failed_count"] == 1  # type: ignore[index]
    package_scenario = next(
        scenario
        for scenario in suite["scenarios"]  # type: ignore[index]
        if scenario["scenario_id"] == "post_training_data_package_export"
    )
    assert package_scenario["status"] == "failed"
    failed_criteria = [
        item for item in package_scenario["evidence"] if item["status"] == "failed"
    ]
    assert failed_criteria
    assert failed_criteria[0]["criterion_id"] == "package_public_claim_boundary"
    assert failed_criteria[0]["observed"] is True


def test_major_capability_scenario_suite_cli_writes_report(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-lightwheel-001"
    _seed_major_capability_artifacts(capture_root, job_id=job_id)

    exit_code = main(["--capture-root", str(capture_root), "--job-id", job_id])

    assert exit_code == 0
    assert (
        capture_root
        / "pipeline"
        / "major_capability_scenarios"
        / "major_capability_scenario_report.json"
    ).is_file()


def test_major_capability_scenario_helpers_fail_closed_on_missing_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = tmp_path / "capture"
    jobs_root = capture_root / "pipeline" / "robot_eval_jobs"

    assert major_suite._find_primary_job_id(capture_root, None) is None

    (jobs_root / "job-b").mkdir(parents=True)
    assert major_suite._find_primary_job_id(capture_root, None) == "job-b"

    _write_json(jobs_root / "job-a" / "job_request.json", {"schema_version": "robot_eval_job_request.v1"})
    assert major_suite._find_primary_job_id(capture_root, None) == "job-a"
    assert major_suite._field_value({"a": {"b": 3}}, None) == {"a": {"b": 3}}
    assert major_suite._field_value({"a": {}}, "a.missing|other") is major_suite._MISSING
    assert major_suite._expected_for({"check": "unknown"}) is None
    assert major_suite._json_payload_for(tmp_path / "missing.json") == (None, "artifact_missing")

    list_payload = tmp_path / "list.json"
    list_payload.write_text("{}", encoding="utf-8")
    original_optional_read_json = major_suite.optional_read_json

    def read_list_for_payload(path: Path):
        if path == list_payload:
            return []
        return original_optional_read_json(path)

    monkeypatch.setattr(major_suite, "optional_read_json", read_list_for_payload)
    assert major_suite._json_payload_for(list_payload) == (None, "artifact_not_json_object")

    assert major_suite._evaluate_json_criterion(
        criterion={"check": "json_field_present", "field": "missing"},
        payload={},
    ) == (False, None, "field_missing")
    assert major_suite._evaluate_json_criterion(
        criterion={"check": "unknown", "field": "value"},
        payload={"value": "present"},
    ) == (False, "present", "unsupported_check:unknown")
    missing_result = major_suite._evaluate_criterion(
        capture_root=capture_root,
        job_id=None,
        criterion={
            "criterion_id": "missing-json",
            "description": "Missing JSON is fail-closed.",
            "artifact": "missing.json",
            "check": "json_field_equals",
            "field": "status",
            "expected": "ready",
        },
    )
    assert missing_result["status"] == "failed"
    assert missing_result["message"] == "artifact_missing"
