from __future__ import annotations

import json
from pathlib import Path

from scripts.run_sim_only_beta_release_gate import (
    SCHEMA_VERSION,
    build_release_gate_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _local_gate_report(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "passed",
            "blockers": [],
            "proof_boundary": {
                "local_mujoco_simulator_execution_proven": True,
            },
            "scenario_eval_matrix": {
                "semantic_spawn_target_coverage_complete": True,
                "deterministic_fallback_spawn_target_run_count": 0,
            },
            "batch_closure": {
                "attempt_count": 11,
                "scenario_eval_run_coverage_complete": True,
                "scenario_eval_run_id_coverage_exact": True,
                "metric_coverage_complete": True,
                "machine_trace_package_complete": True,
                "failure_label_coverage_complete": True,
                "visual_review_coverage_complete": True,
                "visual_coverage": {
                    "all_required_runs_have_visual_recording": True,
                    "all_video_files_complete": True,
                },
            },
            "robot_team_grade_closure": {
                "sim_only_beta_core_complete": True,
                "robot_team_grade_evaluation_complete": False,
                "evaluation_readiness_complete": False,
            },
        },
    )


def _forwarding_report(
    path: Path,
    *,
    ready: bool = True,
    warnings: list[str] | None = None,
) -> None:
    _write_json(
        path,
        {
            "status": "ready_for_required_forwarding_with_probe" if ready else "blocked",
            "blockers": [] if ready else ["missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"],
            "warnings": list(warnings or []),
            "forwarding_required": True,
            "endpoint_configured": ready,
            "configured_env": {
                "forward_url": {
                    "configured": ready,
                    "valid": ready,
                    "origin": "https://pipeline.tryblueprint.io",
                    "pathname": "/api/live-pipeline/job-requests",
                },
                "forward_token": {"configured": ready, "redacted": True},
                "forward_timeout_ms": {"configured": True, "value": 10000, "valid": True},
                "capture_root_by_site_json": {
                    "configured": True,
                    "valid": True,
                    "site_count": 1,
                    "site_slugs": ["site-1"],
                },
                "single_capture_root_override": {"configured": False},
            },
            "probe": {
                "requested": ready,
                "attempted": ready,
                "status": "reachable" if ready else "skipped",
                "http_status": 200 if ready else None,
                "audit_status": "staged_for_control_plane" if ready else None,
                "input_blockers_count": 0 if ready else None,
                "webapp_staging_performed": True if ready else None,
            },
            "proof_boundary": {
                "command_is_read_only": True,
                "no_job_queued": True,
                "no_pipeline_mutation_requested": True,
                "no_gpu_allocated": True,
                "no_simulator_execution_proven": True,
                "no_rank_fidelity_result_proven": True,
                "no_public_claim_upgrade_allowed": True,
            },
        },
    )


def _route_proof(path: Path, capture_root: Path) -> None:
    _write_json(
        path,
        {
            "capture_root": str(capture_root),
            "status": "forwarded_to_pipeline_intake",
            "job_request": {
                "site_package": {
                    "capture_root": str(capture_root),
                },
            },
            "pipeline_intake": {
                "accepted": True,
                "status": "staged_for_control_plane",
                "input_blockers": [],
            },
            "proof_boundary": {
                "production_live_webapp_forwarding_proven": True,
                "simulator_execution_proven": False,
                "robot_policy_execution_proven": False,
                "real_robot_pov_evidence_proven": False,
                "non_ranking_operational_claim_validated": False,
                "customer_delivery_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        },
    )


def _deployment_proof(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "passed",
            "production_deployment_proven": True,
            "webapp_health_ready": True,
            "pipeline_intake_health_ready": True,
            "git_parity_proven": True,
            "webapp_url": "https://www.tryblueprint.io",
            "pipeline_intake_url": "https://pipeline.tryblueprint.io/api/live-pipeline",
        },
    )


def test_release_gate_blocks_without_production_forwarding_and_deploy_proof(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    local_gate = capture_root / "local_gate.json"
    forwarding = tmp_path / "forwarding.json"
    _local_gate_report(local_gate)
    _forwarding_report(forwarding, ready=False)

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=None,
        production_deployment_proof_path=None,
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "blocked"
    assert report["ready_for_beta_release"] is False
    assert "production_webapp_to_pipeline_forwarding:forwarding_preflight_not_ready_with_probe" in report[
        "blockers"
    ]
    assert "production_route_forwarding_proof:production_route_forwarding_proof_path_missing" in report[
        "blockers"
    ]
    assert "production_deployment_parity:production_deployment_proof_path_missing" in report[
        "blockers"
    ]


def test_release_gate_passes_with_local_sim_core_and_production_proofs(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    local_gate = capture_root / "local_gate.json"
    forwarding = tmp_path / "forwarding.json"
    route = tmp_path / "route.json"
    deployment = tmp_path / "deployment.json"
    _local_gate_report(local_gate)
    _forwarding_report(forwarding)
    _route_proof(route, capture_root)
    _deployment_proof(deployment)

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=route,
        production_deployment_proof_path=deployment,
    )

    assert report["status"] == "passed"
    assert report["ready_for_beta_release"] is True
    assert report["blockers"] == []
    assert [gate["status"] for gate in report["gates"]] == ["passed", "passed", "passed", "passed"]


def test_release_gate_allows_known_nonblocking_forwarding_warning(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    local_gate = capture_root / "local_gate.json"
    forwarding = tmp_path / "forwarding.json"
    route = tmp_path / "route.json"
    deployment = tmp_path / "deployment.json"
    _local_gate_report(local_gate)
    _forwarding_report(forwarding, warnings=["capture_root_override_not_configured"])
    _route_proof(route, capture_root)
    _deployment_proof(deployment)

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=route,
        production_deployment_proof_path=deployment,
    )

    assert report["status"] == "passed"
    forwarding_gate = next(
        gate
        for gate in report["gates"]
        if gate["id"] == "production_webapp_to_pipeline_forwarding"
    )
    assert forwarding_gate["evidence"]["blocking_warnings"] == []
    assert forwarding_gate["evidence"]["non_blocking_warnings"] == [
        "capture_root_override_not_configured"
    ]


def test_release_gate_blocks_unknown_forwarding_warning(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    local_gate = capture_root / "local_gate.json"
    forwarding = tmp_path / "forwarding.json"
    route = tmp_path / "route.json"
    deployment = tmp_path / "deployment.json"
    _local_gate_report(local_gate)
    _forwarding_report(forwarding, warnings=["unknown_warning"])
    _route_proof(route, capture_root)
    _deployment_proof(deployment)

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=route,
        production_deployment_proof_path=deployment,
    )

    assert report["status"] == "blocked"
    assert "production_webapp_to_pipeline_forwarding:forwarding_preflight_has_warnings" in report[
        "blockers"
    ]


def test_release_gate_accepts_route_proof_with_same_scene_capture_identity(
    tmp_path: Path,
) -> None:
    capture_root = (
        tmp_path
        / "local-blueprint"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
    )
    production_capture_root = Path(
        "/var/lib/blueprint/pipeline-control-plane/captures/local-blueprint/scenes/scene-1/captures/capture-1"
    )
    local_gate = capture_root / "local_gate.json"
    forwarding = tmp_path / "forwarding.json"
    route = tmp_path / "route.json"
    deployment = tmp_path / "deployment.json"
    _local_gate_report(local_gate)
    _forwarding_report(forwarding)
    _route_proof(route, production_capture_root)
    _deployment_proof(deployment)

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=route,
        production_deployment_proof_path=deployment,
    )

    assert report["status"] == "passed"
    assert report["ready_for_beta_release"] is True
    assert report["blockers"] == []


def test_release_gate_blocks_route_proof_for_different_capture_root(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    other_capture_root = tmp_path / "other-capture"
    local_gate = capture_root / "local_gate.json"
    forwarding = tmp_path / "forwarding.json"
    route = tmp_path / "route.json"
    deployment = tmp_path / "deployment.json"
    _local_gate_report(local_gate)
    _forwarding_report(forwarding)
    _route_proof(route, other_capture_root)
    _deployment_proof(deployment)

    report = build_release_gate_report(
        capture_root=capture_root,
        local_gate_report_path=local_gate,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=route,
        production_deployment_proof_path=deployment,
    )

    assert report["status"] == "blocked"
    assert (
        "production_route_forwarding_proof:production_route_capture_root_mismatch"
        in report["blockers"]
    )
    assert (
        "production_route_forwarding_proof:production_route_job_request_capture_root_mismatch"
        in report["blockers"]
    )
