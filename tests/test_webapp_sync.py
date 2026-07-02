from __future__ import annotations

import hashlib
import hmac
import json
from urllib import error as urllib_error

import pytest

from blueprint_pipeline.webapp_sync import (
    WebappSyncError,
    _artifact_uri_checksums,
    _bool_env,
    _buyer_access_check_payload,
    _int_env,
    build_webapp_pipeline_attachment_payload,
    derive_webapp_opportunity_state,
    derive_webapp_qualification_state,
    sync_webapp_pipeline_attachment,
)


def _minimal_payload() -> dict[str, object]:
    return {
        "site_submission_id": "site-1",
        "request_id": "request-1",
        "buyer_request_id": "buyer-request-1",
        "capture_job_id": "capture-job-1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        "qualification_state": "qualified_ready",
        "opportunity_state": "handoff_ready",
        "artifacts": {"qualification_summary_uri": "gs://bucket/path.json"},
    }


def test_sync_webapp_pipeline_attachment_skips_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "skipped"
    assert result["reason"] == "sync_not_configured"
    assert result["attempts"] == 0
    assert result["attachment_payload"]["qualification_state"] == "qualified_ready"
    assert result["attachment_payload"]["placeholder_fallback_allowed"] is False
    assert result["attachment_payload"]["upstream_links_verified"] is True


def test_sync_payload_projects_robot_eval_status_without_provider_details(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)

    result = sync_webapp_pipeline_attachment(
        **_minimal_payload(),
        robot_eval_status_projection={
            "schema_version": "internal_robot_eval_status_projection.v1",
            "job_id": "job-robot-eval-1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "status": "simulator_command_completed",
            "state": "completed",
            "buyer_display_state": "simulator_results_ready_review_required",
            "provider_command": "python private_runner.py --token secret",
            "scenario_batch": {
                "status": "completed",
                "scenario_eval_run_count": 500,
                "target_scenario_eval_run_count": 500,
                "base_scenario_eval_run_count": 11,
                "scenario_eval_batch_expanded": True,
                "target_scenario_eval_run_count_satisfied": True,
                "episode_authoring_contract": {
                    "spawn_target_variation_seed_handling": (
                        "deterministic_frozen_matrix_rows"
                    ),
                    "runtime_spawn_goal_variation_mutation_allowed": False,
                },
                "covered_scenario_eval_run_count": 500,
                "missing_scenario_eval_run_count": 0,
                "scenario_eval_run_coverage_complete": True,
                "scenario_eval_matrix_path": "scenario_eval_matrix.json",
            },
            "batch_closure": {
                "status": "completed_with_robot_team_grade_blockers",
                "batch_execution_status": "completed",
                "machine_trace_package_complete": True,
                "robot_team_grade_package_complete": False,
                "robot_team_grade_blockers": ["visual_video_coverage_not_complete"],
                "batch_closure_manifest_path": (
                    "simulator_command_batch_closure_manifest.json"
                ),
            },
            "digital_twin_fidelity": {
                "status": "blocked",
                "machine_fidelity_audit_complete": True,
                "robot_team_grade_fidelity_passed": False,
                "blockers": ["visual_object_without_matching_physics"],
            },
            "policy_interface": {
                "status": "contract_declared",
                "selected_modalities": ["docker_container"],
                "supported_modalities": ["docker_container", "recorded_action_trace"],
                "observation_schema_id": "blueprint.robot_eval.observation.v1",
                "action_schema_id": "blueprint.robot_eval.action_trace.v1",
                "reproducible_replay_required": True,
                "robot_policy_execution_proven": False,
            },
            "closure_audit": {
                "live_eval_closure_status": "blocked",
                "selected_scenario_coverage_closed": True,
                "machine_trace_package_complete": True,
                "robot_team_grade_package_complete": False,
                "post_training_data_package_status": "export_ready_review_required",
                "no_readiness_claim_upgrade_without_evidence": True,
            },
            "remote_cloud_execution": {
                "status": "ready_for_explicit_provider_runtime",
                "contract_ready_for_remote_runtime": True,
                "remote_cloud_execution_proven": False,
                "clean_shutdown_proven": False,
                "live_provider_calls_performed": False,
                "blockers": ["remote_provider_runtime_not_executed"],
                "closure_manifest_path": "remote_cloud_execution_closure_manifest.json",
                "provider_private_payload": {"token": "should-not-pass"},
            },
            "robot_team_grade_eval_closure": {
                "status": "blocked_robot_team_grade_requirements",
                "sim_only_beta_core_complete": True,
                "robot_team_grade_evaluation_complete": False,
                "evaluation_readiness_complete": False,
                "blocked_requirement_ids": [
                    "remote_cloud_execution_path",
                ],
                "all_blocked_requirement_ids": [
                    "remote_cloud_execution_path",
                    "sim_vs_real_calibration_path",
                ],
                "robot_team_grade_blocked_requirement_ids": [
                    "remote_cloud_execution_path",
                ],
                "evaluation_readiness_blocked_requirement_ids": [
                    "remote_cloud_execution_path",
                    "sim_vs_real_calibration_path",
                ],
                "closure_manifest_path": "robot_team_grade_eval_closure_manifest.json",
                "private_requirement_notes": {"provider": "should-not-pass"},
            },
            "proof_boundary": {
                "simulator_execution_proven": True,
                "robot_policy_execution_proven": False,
                "real_world_outcome_proven": False,
                "physics_contact_validated": False,
                "non_ranking_operational_claim_validated": False,
                "rank_fidelity_result_proven": False,
                "public_claim_upgrade_allowed": False,
            },
            "artifact_paths": {
                "scenario_eval_matrix": "scenario_eval_matrix.json",
                "proof_boundary": "proof_boundary.json",
                "webapp_robot_eval_status_projection": (
                    "webapp_robot_eval_status_projection.json"
                ),
                "remote_cloud_execution_closure_manifest": (
                    "remote_cloud_execution_closure_manifest.json"
                ),
                "robot_team_grade_eval_closure_manifest": (
                    "robot_team_grade_eval_closure_manifest.json"
                ),
                "provider_private_log": "private_provider.log",
            },
        },
    )

    projection = result["attachment_payload"]["robot_eval_status_projection"]
    assert projection["schema_version"] == "webapp_robot_eval_status_projection.v1"
    assert projection["provider_details_exposed"] is False
    assert projection["provider_complexity_hidden"] is True
    assert projection["scenario_batch"]["scenario_eval_run_count"] == 500
    assert projection["scenario_batch"]["target_scenario_eval_run_count"] == 500
    assert projection["scenario_batch"]["base_scenario_eval_run_count"] == 11
    assert projection["scenario_batch"]["scenario_eval_batch_expanded"] is True
    assert projection["scenario_batch"]["target_scenario_eval_run_count_satisfied"] is True
    assert (
        projection["scenario_batch"]["episode_authoring_contract"][
            "spawn_target_variation_seed_handling"
        ]
        == "deterministic_frozen_matrix_rows"
    )
    assert projection["batch_closure"]["robot_team_grade_package_complete"] is False
    assert projection["digital_twin_fidelity"]["robot_team_grade_fidelity_passed"] is False
    assert projection["policy_interface"]["observation_schema_id"] == (
        "blueprint.robot_eval.observation.v1"
    )
    assert projection["closure_audit"]["selected_scenario_coverage_closed"] is True
    assert projection["closure_audit"]["robot_team_grade_package_complete"] is False
    assert (
        projection["closure_audit"]["post_training_data_package_status"]
        == "export_ready_review_required"
    )
    assert projection["remote_cloud_execution"]["contract_ready_for_remote_runtime"] is True
    assert projection["remote_cloud_execution"]["remote_cloud_execution_proven"] is False
    assert projection["remote_cloud_execution"]["clean_shutdown_proven"] is False
    assert "provider_private_payload" not in projection["remote_cloud_execution"]
    assert projection["robot_team_grade_eval_closure"]["sim_only_beta_core_complete"] is True
    assert (
        projection["robot_team_grade_eval_closure"]["robot_team_grade_evaluation_complete"]
        is False
    )
    assert projection["robot_team_grade_eval_closure"]["blocked_requirement_ids"] == [
        "remote_cloud_execution_path"
    ]
    assert projection["robot_team_grade_eval_closure"]["all_blocked_requirement_ids"] == [
        "remote_cloud_execution_path",
        "sim_vs_real_calibration_path",
    ]
    assert projection["robot_team_grade_eval_closure"][
        "evaluation_readiness_blocked_requirement_ids"
    ] == ["remote_cloud_execution_path", "sim_vs_real_calibration_path"]
    assert "private_requirement_notes" not in projection["robot_team_grade_eval_closure"]
    assert projection["proof_boundary"]["public_claim_upgrade_allowed"] is False
    assert projection["buyer_display_guardrails"]["readiness_claim_upgrade_allowed"] is False
    assert projection["buyer_display_guardrails"]["provider_commands_exposed"] is False
    assert "provider_command" not in projection
    assert "provider_private_log" not in projection["artifact_paths"]
    assert projection["artifact_paths"]["webapp_robot_eval_status_projection"] == (
        "webapp_robot_eval_status_projection.json"
    )
    assert projection["artifact_paths"]["remote_cloud_execution_closure_manifest"] == (
        "remote_cloud_execution_closure_manifest.json"
    )
    assert projection["artifact_paths"]["robot_team_grade_eval_closure_manifest"] == (
        "robot_team_grade_eval_closure_manifest.json"
    )


def test_projection_accepts_string_guardrails_and_null_projection(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)

    result = sync_webapp_pipeline_attachment(
        **_minimal_payload(),
        robot_eval_status_projection={
            "job_id": "job-1",
            "buyer_display_guardrails": {
                "must_not_display_as": "rank_fidelity",
            },
        },
    )
    assert result["attachment_payload"]["robot_eval_status_projection"][
        "buyer_display_guardrails"
    ]["must_not_display_as"] == ["rank_fidelity"]

    without_projection = sync_webapp_pipeline_attachment(
        **_minimal_payload(),
        robot_eval_status_projection="not-a-mapping",  # type: ignore[arg-type]
    )
    assert without_projection["attachment_payload"]["robot_eval_status_projection"] is None


def test_env_and_checksum_helpers_cover_invalid_and_falsey_values(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_MAX_ATTEMPTS", "not-an-int")
    monkeypatch.delenv("PIPELINE_SYNC_FLAG", raising=False)
    assert _int_env("PIPELINE_SYNC_MAX_ATTEMPTS", 7) == 7
    assert _bool_env("PIPELINE_SYNC_FLAG", default=True) is True

    monkeypatch.setenv("PIPELINE_SYNC_FLAG", "yes")
    assert _bool_env("PIPELINE_SYNC_FLAG") is True
    assert _artifact_uri_checksums({"kept": "gs://bucket/object", "empty": ""}) == {
        "kept": _artifact_uri_checksums({"kept": "gs://bucket/object"})["kept"]
    }


def test_sync_webapp_pipeline_attachment_raises_when_required(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")

    with pytest.raises(WebappSyncError, match="sync_not_configured"):
        sync_webapp_pipeline_attachment(**_minimal_payload())


@pytest.mark.parametrize(
    "field",
    ["site_submission_id", "request_id", "buyer_request_id", "capture_job_id"],
)
def test_sync_payload_requires_upstream_request_job_and_bootstrap_records(monkeypatch, field: str) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    payload = _minimal_payload()
    payload[field] = ""

    with pytest.raises(ValueError, match=field):
        sync_webapp_pipeline_attachment(**payload)


def test_sync_rejects_generated_capture_ids_as_upstream_links(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    payload = _minimal_payload()
    payload["site_submission_id"] = "scene-1:capture-1"
    payload["request_id"] = "scene-1:capture-1"

    with pytest.raises(ValueError, match="generated capture ids"):
        sync_webapp_pipeline_attachment(**payload)


def test_sync_rejects_placeholder_upstream_links(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    payload = _minimal_payload()
    payload["buyer_request_id"] = "example-buyer-request"

    with pytest.raises(ValueError, match="placeholder upstream ids"):
        sync_webapp_pipeline_attachment(**payload)


def test_sync_without_upstream_links_fails_closed_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)
    payload = _minimal_payload()
    payload["capture_job_id"] = ""

    result = sync_webapp_pipeline_attachment(**payload)

    assert result["status"] == "failed"
    assert result["blocker"] == "missing_upstream_pipeline_records"
    assert result["attempts"] == 0
    assert result["attachment_payload"]["upstream_links_verified"] is False
    assert result["attachment_payload"]["missing_upstream_links"] == ["capture_job_id"]
    assert result["attachment_payload"]["placeholder_fallback_allowed"] is False
    assert result["buyer_access_check"]["blocker"] == "missing_upstream_pipeline_records"


def test_placeholder_request_fallback_requires_explicit_internal_flag(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)
    monkeypatch.setenv("PIPELINE_SYNC_ALLOW_PLACEHOLDER_REQUESTS", "true")

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "skipped"
    assert result["attachment_payload"]["placeholder_fallback_allowed"] is True
    assert result["attachment_payload"]["upstream_links_verified"] is True


def test_sync_webapp_pipeline_attachment_returns_buyer_access_and_checksums(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/pipeline-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.setenv("PIPELINE_BUYER_ACCESS_CHECK_URL", "https://webapp.test/api/buyer-access")

    class _Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        assert timeout > 0
        if request.full_url.endswith("/buyer-access"):
            return _Response({"buyer_accessible": True})
        timestamp = request.get_header("X-blueprint-pipeline-timestamp")
        signature = request.get_header("X-blueprint-pipeline-signature")
        assert timestamp
        assert signature
        expected = hmac.new(
            b"token",
            f"{timestamp}.".encode("utf-8") + request.data,
            hashlib.sha256,
        ).hexdigest()
        assert signature == f"sha256={expected}"
        assert request.get_header("X-blueprint-pipeline-token") is None
        return _Response(
            {
                "attachment_id": "att-1",
                "listing_id": "listing-1",
                "artifact_id": "artifact-1",
            }
        )

    monkeypatch.setattr("blueprint_pipeline.webapp_sync.urllib_request.urlopen", _fake_urlopen)

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "succeeded"
    assert result["webapp_response_ids"]["attachment_id"] == "att-1"
    assert result["webapp_response_ids"]["listing_id"] == "listing-1"
    assert result["artifact_uri_checksums"]["qualification_summary_uri"]
    assert result["buyer_access_check"]["buyer_access_checked"] is True
    assert result["buyer_access_check"]["buyer_accessible"] is True


def test_successful_sync_extracts_nested_ids_and_skips_unconfigured_buyer_access(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/pipeline-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.delenv("PIPELINE_BUYER_ACCESS_CHECK_URL", raising=False)

    class _Response:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return json.dumps({"attachment": {"attachment_id": "nested-att-1"}}).encode(
                "utf-8"
            )

    monkeypatch.setattr(
        "blueprint_pipeline.webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: _Response(),
    )

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "succeeded"
    assert result["webapp_response_ids"] == {"attachment_id": "nested-att-1"}
    assert result["buyer_access_check"]["status"] == "skipped"
    assert result["buyer_access_check"]["reason"] == "buyer_access_check_not_configured"


def test_buyer_access_check_handles_callback_failure_and_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_BUYER_ACCESS_CHECK_URL", "https://webapp.test/api/buyer-access")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.setenv("PIPELINE_BUYER_ACCESS_TIMEOUT_SECONDS", "not-an-int")

    def _raise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise TimeoutError("slow")

    monkeypatch.setattr("blueprint_pipeline.webapp_sync.urllib_request.urlopen", _raise)
    failed = _buyer_access_check_payload({"id": "att-1"})
    assert failed["status"] == "blocked"
    assert failed["blocker"] == "buyer_access_check_failed"

    class _InvalidResponse:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b"{not-json"

    monkeypatch.setattr(
        "blueprint_pipeline.webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: _InvalidResponse(),
    )
    inaccessible = _buyer_access_check_payload({"id": "att-1"})
    assert inaccessible["buyer_access_checked"] is True
    assert inaccessible["buyer_accessible"] is False
    assert inaccessible["response"] == {}


def test_derive_webapp_states_cover_ready_risky_and_incomplete_paths() -> None:
    assert (
        derive_webapp_qualification_state(
            readiness_state="ready",
            completeness_status="sufficient",
        )
        == "qualified_ready"
    )
    assert (
        derive_webapp_qualification_state(
            readiness_state="risky",
            completeness_status="sufficient",
        )
        == "qualified_risky"
    )
    assert (
        derive_webapp_qualification_state(
            readiness_state="ready",
            completeness_status="missing",
        )
        == "needs_more_evidence"
    )
    assert (
        derive_webapp_qualification_state(
            readiness_state="blocked",
            completeness_status="sufficient",
        )
        == "not_ready_yet"
    )
    assert derive_webapp_opportunity_state(qualification_state="qualified_ready") == "handoff_ready"
    assert derive_webapp_opportunity_state(qualification_state="qualified_risky") == "handoff_ready"
    assert derive_webapp_opportunity_state(qualification_state="not_ready_yet") == "not_applicable"


def test_build_payload_requires_site_submission_or_request_id() -> None:
    with pytest.raises(ValueError, match="site_submission_id or request_id is required"):
        build_webapp_pipeline_attachment_payload(
            site_submission_id="",
            request_id="",
            scene_id="scene-1",
            capture_id="capture-1",
            pipeline_prefix="pipeline",
            qualification_state="qualified_ready",
            opportunity_state="handoff_ready",
            artifacts={},
        )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            urllib_error.HTTPError(
                "https://webapp.test/api/pipeline-sync",
                503,
                "unavailable",
                {},
                None,
            ),
            "http_error:503",
        ),
        (urllib_error.URLError("offline"), "url_error:offline"),
    ],
)
def test_sync_failures_report_http_and_url_errors(monkeypatch, exc: Exception, expected: str) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/pipeline-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.setenv("PIPELINE_SYNC_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("PIPELINE_SYNC_RETRY_DELAY_MS", "0")

    def _raise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise exc

    monkeypatch.setattr("blueprint_pipeline.webapp_sync.urllib_request.urlopen", _raise)

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "failed"
    assert result["reason"] == expected
    assert result["attempts"] == 1


def test_sync_retries_then_raises_required_timeout(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/pipeline-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.setenv("PIPELINE_SYNC_REQUIRED", "true")
    monkeypatch.setenv("PIPELINE_SYNC_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("PIPELINE_SYNC_RETRY_DELAY_MS", "1")
    calls: list[int] = []

    def _raise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        raise TimeoutError("slow")

    monkeypatch.setattr("blueprint_pipeline.webapp_sync.urllib_request.urlopen", _raise)

    with pytest.raises(WebappSyncError, match="timeouterror"):
        sync_webapp_pipeline_attachment(**_minimal_payload())

    assert len(calls) == 2


def test_sync_invalid_json_response_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.test/api/pipeline-sync")
    monkeypatch.setenv("PIPELINE_SYNC_TOKEN", "token")
    monkeypatch.setenv("PIPELINE_SYNC_MAX_ATTEMPTS", "1")

    class _InvalidResponse:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b"{not-json"

    monkeypatch.setattr(
        "blueprint_pipeline.webapp_sync.urllib_request.urlopen",
        lambda *_args, **_kwargs: _InvalidResponse(),
    )

    result = sync_webapp_pipeline_attachment(**_minimal_payload())

    assert result["status"] == "failed"
    assert result["reason"] == "invalid_json"


def test_production_sync_is_required_and_disables_placeholder_fallback(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_TOKEN", raising=False)
    monkeypatch.delenv("PIPELINE_SYNC_REQUIRED", raising=False)

    with pytest.raises(WebappSyncError, match="sync_not_configured"):
        sync_webapp_pipeline_attachment(**_minimal_payload())
