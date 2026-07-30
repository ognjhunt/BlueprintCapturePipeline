from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import (
    LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
    arkit_metric_scaffold_method_profile,
)
from blueprint_pipeline.reconstruction_capability import plan_reconstruction_methods
from blueprint_pipeline.task_evaluation_supervisor import (
    ReconstructionExecutionReadinessError,
    ToolRegistry,
    build_capture_reconstruction_route,
    build_reconstruction_execution_readiness,
    load_capture_build_ingress,
    validate_reconstruction_execution_readiness,
)


CAPTURE_DIGEST = "sha256:" + "1" * 64
ENVELOPE_DIGEST = "sha256:" + "2" * 64
QA_DIGEST = "sha256:" + "3" * 64
CONTEXT_DIGEST = "sha256:" + "4" * 64
SOURCE_COMMIT = "5" * 40


def _capture_build(tmp_path: Path) -> dict:
    manifest = tmp_path / "capture_intake_envelope.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "capture_intake_envelope.v1",
                "capture_session_id": "session-iphone",
                "intake_id": "intake-iphone",
                "capture_authority_profile": "iphone_arkit_lidar",
                "capture_digest": CAPTURE_DIGEST,
                "envelope_digest": ENVELOPE_DIGEST,
                "qa_report_digest": QA_DIGEST,
            }
        ),
        encoding="utf-8",
    )
    return load_capture_build_ingress(manifest)


def _plan(*, capture_digest: str = CAPTURE_DIGEST) -> dict:
    return plan_reconstruction_methods(
        intake_id="intake-iphone",
        capture_digest=capture_digest,
        capture_authority_profile="iphone_arkit_lidar",
        claim_ceiling={
            "calibrated_camera_poses": True,
            "decoded_video_pts": True,
            "metric_geometry": True,
        },
        requested_claim_types=["reachability"],
        permitted_provider_identities=["local"],
        method_profiles=[arkit_metric_scaffold_method_profile(execution_authorized=True)],
    )


def _inspection(*, authorized: bool, executed: bool, capture_digest: str = CAPTURE_DIGEST) -> dict:
    plan = _plan(capture_digest=capture_digest)
    authorization = None
    if authorized:
        authorization = {
            "schema_version": "reconstruction_execution_authorization.v1",
            "plan_id": "reconstruction-fixture",
            "reconstruction_plan_digest": plan["reconstruction_plan_digest"],
            "context_digest": CONTEXT_DIGEST,
            "authorized_adapter_references": [LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER],
            "actor": {"role": "operator", "identity": "fixture-operator"},
            "idempotency_key": "fixture-authority",
            "live_provider_execution": False,
            "paid_compute_authorized": False,
            "physical_robot_run_authorized": False,
            "proof_boundary": {
                "authorization_is_method_qualification": False,
                "simulation_is_physical_success": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
        }
        authorization["authorization_digest"] = canonical_digest(
            authorization,
            digest_field="authorization_digest",
        )
    execution = None
    if executed:
        assert authorization is not None
        execution = {
            "schema_version": "reconstruction_control_plane_execution_result.v1",
            "plan_id": "reconstruction-fixture",
            "state": "completed",
            "reconstruction_plan_digest": plan["reconstruction_plan_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "context_digest": CONTEXT_DIGEST,
            "results": [{"method_id": "local_arkit_metric_scaffold"}],
            "errors": [],
            "missing_representations": [],
            "next_cheapest_experiments": [],
            "cost_usd": 0.0,
            "proof_boundary": {
                "execution_was_local_and_explicitly_authorized": True,
                "derived_reconstruction_upgrades_raw_capture": False,
                "physical_task_success_established": False,
                "deployment_or_safety_approved": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
            "already_exists": False,
        }
        execution["execution_result_digest"] = canonical_digest(
            execution,
            digest_field="execution_result_digest",
        )
    return {
        "schema_version": "reconstruction_control_plane_inspection.v1",
        "plan_id": "reconstruction-fixture",
        "state": "completed" if executed else "authorization_required",
        "source_binding": {
            "capture_session_id": "session-iphone",
            "intake_id": "intake-iphone",
            "capture_digest": capture_digest,
            "envelope_digest": ENVELOPE_DIGEST,
            "qa_report_digest": QA_DIGEST,
            "object_manifest_digest": None,
            "context_digest": CONTEXT_DIGEST,
        },
        "reconstruction_plan": plan,
        "execution_authorization": authorization,
        "execution_result": execution,
        "proof_boundary": {
            "inspection_recomputes_scientific_truth": False,
            "physical_task_success_established": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }


def _route_and_bound_tools(capture_build: dict) -> tuple[dict, list[str]]:
    route = build_capture_reconstruction_route(
        capture_build,
        requested_claim_types=["reachability"],
    )
    bound = [
        row["stage_id"]
        for row in route["stages"]
        if row["implementation_status"] == "registered_conditional"
    ]
    return route, bound


def test_readiness_reports_missing_plan_without_claim_or_execution_authority(
    tmp_path: Path,
) -> None:
    capture_build = _capture_build(tmp_path)
    route, bound = _route_and_bound_tools(capture_build)

    readiness = build_reconstruction_execution_readiness(
        capture_build_value=capture_build,
        route_value=route,
        tool_registry_manifest=ToolRegistry.default().manifest(),
        bound_tool_ids=bound,
        source_commit_sha=SOURCE_COMMIT,
        recorded_at="2026-07-30T20:00:00Z",
    )

    assert readiness["status"] == "not_ready"
    assert "control_plane_plan_missing:compile_frozen_frame_dataset" in readiness["blockers"]
    scaffold = next(
        row for row in readiness["stages"] if row["stage_id"] == "compile_arkit_metric_scaffold"
    )
    assert scaffold["runtime_bound"] is True
    assert scaffold["readiness_status"] == "awaiting_control_plane_plan"
    assert readiness["proof_boundary"]["readiness_is_execution_authority"] is False
    assert readiness["proof_boundary"]["readiness_is_reconstruction_evidence"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/task_evaluation_reconstruction_execution_readiness.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(readiness, schema)


def test_readiness_distinguishes_explicit_authority_from_recorded_execution(
    tmp_path: Path,
) -> None:
    capture_build = _capture_build(tmp_path)
    route, bound = _route_and_bound_tools(capture_build)
    common = {
        "capture_build_value": capture_build,
        "route_value": route,
        "tool_registry_manifest": ToolRegistry.default().manifest(),
        "bound_tool_ids": bound,
        "source_commit_sha": SOURCE_COMMIT,
        "recorded_at": "2026-07-30T20:00:00Z",
    }

    waiting = build_reconstruction_execution_readiness(
        **common,
        control_plane_inspection=_inspection(authorized=False, executed=False),
    )
    authorized = build_reconstruction_execution_readiness(
        **common,
        control_plane_inspection=_inspection(authorized=True, executed=False),
    )
    executed = build_reconstruction_execution_readiness(
        **common,
        control_plane_inspection=_inspection(authorized=True, executed=True),
    )

    assert waiting["status"] == "not_ready"
    assert any(
        row["readiness_status"] == "awaiting_control_plane_authority" for row in waiting["stages"]
    )
    assert authorized["status"] == "ready_for_bounded_execution"
    assert not authorized["blockers"]
    assert any(
        row["readiness_status"] == "recorded_support_completed" for row in executed["stages"]
    )
    assert executed["proof_boundary"]["proof_effect"] == "none"


def test_readiness_refuses_source_mismatch_unregistered_binding_and_tampering(
    tmp_path: Path,
) -> None:
    capture_build = _capture_build(tmp_path)
    route, bound = _route_and_bound_tools(capture_build)
    common = {
        "capture_build_value": capture_build,
        "route_value": route,
        "tool_registry_manifest": ToolRegistry.default().manifest(),
        "source_commit_sha": SOURCE_COMMIT,
        "recorded_at": "2026-07-30T20:00:00Z",
    }
    mismatch = build_reconstruction_execution_readiness(
        **common,
        bound_tool_ids=bound,
        control_plane_inspection=_inspection(
            authorized=True,
            executed=False,
            capture_digest="sha256:" + "9" * 64,
        ),
    )
    assert mismatch["status"] == "source_binding_mismatch"
    assert "control_plane_capture_digest_mismatch" in mismatch["blockers"]

    with pytest.raises(
        ReconstructionExecutionReadinessError,
        match="readiness_bound_tool_unregistered",
    ):
        build_reconstruction_execution_readiness(
            **common,
            bound_tool_ids=bound + ["unregistered_shell"],
        )

    stale_plan = _inspection(authorized=False, executed=False)
    stale_plan["reconstruction_plan"]["source_capture"]["capture_digest"] = "sha256:" + "8" * 64
    stale_plan["reconstruction_plan"]["reconstruction_plan_digest"] = canonical_digest(
        stale_plan["reconstruction_plan"],
        digest_field="reconstruction_plan_digest",
    )
    with pytest.raises(
        ReconstructionExecutionReadinessError,
        match="control_plane_source_plan_mismatch",
    ):
        build_reconstruction_execution_readiness(
            **common,
            bound_tool_ids=bound,
            control_plane_inspection=stale_plan,
        )

    unplanned_result = _inspection(authorized=True, executed=True)
    unplanned_result["execution_result"]["results"] = [{"method_id": "fabricated_unplanned_method"}]
    unplanned_result["execution_result"]["execution_result_digest"] = canonical_digest(
        unplanned_result["execution_result"],
        digest_field="execution_result_digest",
    )
    with pytest.raises(
        ReconstructionExecutionReadinessError,
        match="control_plane_execution_unplanned_method",
    ):
        build_reconstruction_execution_readiness(
            **common,
            bound_tool_ids=bound,
            control_plane_inspection=unplanned_result,
        )

    tampered = json.loads(json.dumps(mismatch))
    tampered["proof_boundary"]["readiness_is_execution_authority"] = True
    tampered["reconstruction_execution_readiness_digest"] = canonical_digest(
        tampered,
        digest_field="reconstruction_execution_readiness_digest",
    )
    with pytest.raises(
        ReconstructionExecutionReadinessError,
        match="execution_readiness_contract_invalid",
    ):
        validate_reconstruction_execution_readiness(tampered)
