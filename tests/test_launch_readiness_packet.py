from __future__ import annotations

import json
from pathlib import Path

from scripts.build_launch_readiness_packet import build_launch_readiness_packet


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, text: str = "artifact") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_launch_readiness_packet_links_artifacts_and_preserves_live_blockers(tmp_path: Path) -> None:
    pipeline = tmp_path / "BlueprintCapturePipeline"
    webapp = tmp_path / "Blueprint-WebApp"
    contracts = tmp_path / "BlueprintContracts"
    capture = tmp_path / "BlueprintCapture"
    for repo in (pipeline, webapp, contracts, capture):
        repo.mkdir()

    _write_json(
        pipeline / "output" / "paid_marketplace_launch_gate.json",
        {
            "schema_version": "v1",
            "overall_status": "automated_contracts_passed_manual_ops_required",
            "closeout_summary": {
                "remaining_manual_evidence_ids": ["buyer_payment_settlement"],
            },
        },
    )
    _write_text(pipeline / "output" / "paid_marketplace_launch_gate.md")
    _write_json(
        pipeline / "output" / "external_alpha_launch_gate.json",
        {
            "schema_version": "external_alpha_launch_gate.v1",
            "overall_status": "passed_manual_required",
            "checks": [{"id": "android_capture_contract_tests", "status": "manual_required"}],
        },
    )
    _write_text(pipeline / "output" / "external_alpha_launch_gate.md")
    _write_json(
        pipeline
        / "output"
        / "sim_only_beta_local_gate_fixture"
        / "capture"
        / "pipeline"
        / "live_pipeline_control_plane"
        / "sim_only_beta_local_gate"
        / "sim_only_beta_local_gate_report.json",
        {"schema_version": "blueprint.sim_only_beta_local_gate_report.v1", "status": "passed"},
    )
    _write_json(
        pipeline / "output" / "launch_audit_live_pipeline_setup_20260707.json",
        {
            "schema_version": "blueprint_live_pipeline_setup.v1",
            "status": "local_ready_live_external_blocked",
            "blockers": ["delivery_upload:missing_delivery_command"],
        },
    )
    _write_json(
        webapp / "output" / "pipeline" / "robot_eval_job_requests" / "forwarding_preflight.json",
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "blocked",
            "blockers": ["missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"],
        },
    )

    packet = build_launch_readiness_packet(
        pipeline_repo=pipeline,
        webapp_repo=webapp,
        contracts_repo=contracts,
        capture_repo=capture,
        generated_at="2026-07-07T00:00:00+00:00",
    )

    assert packet["status"] == "local_ready_live_external_blocked"
    assert packet["artifact_blockers"] == []
    assert all(artifact["sha256"] for artifact in packet["artifacts"])
    assert packet["readiness_summary"] == {
        "paid_marketplace_launch_gate": "automated_contracts_passed_manual_ops_required",
        "external_alpha_launch_gate": "passed_manual_required",
        "sim_only_beta_local_gate": "passed",
        "live_pipeline_setup": "local_ready_live_external_blocked",
        "webapp_forwarding_preflight": "blocked",
    }
    assert packet["remaining_blockers"]["manual_live_evidence_ids"] == [
        "buyer_payment_settlement"
    ]
    assert packet["remaining_blockers"]["external_alpha_manual_items"] == [
        "android_capture_contract_tests:manual_required"
    ]
    assert packet["remaining_blockers"]["live_pipeline_setup_blockers"] == [
        "delivery_upload:missing_delivery_command"
    ]
    assert packet["remaining_blockers"]["webapp_forwarding_blockers"] == [
        "missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"
    ]
    assert packet["claim_boundary"]["automated_contracts_do_not_prove_real_pubsub_delivery"] is True


def test_launch_readiness_packet_blocks_missing_required_artifacts(tmp_path: Path) -> None:
    pipeline = tmp_path / "BlueprintCapturePipeline"
    webapp = tmp_path / "Blueprint-WebApp"
    contracts = tmp_path / "BlueprintContracts"
    capture = tmp_path / "BlueprintCapture"
    for repo in (pipeline, webapp, contracts, capture):
        repo.mkdir()

    packet = build_launch_readiness_packet(
        pipeline_repo=pipeline,
        webapp_repo=webapp,
        contracts_repo=contracts,
        capture_repo=capture,
        generated_at="2026-07-07T00:00:00+00:00",
    )

    assert packet["status"] == "incomplete_packet"
    assert "missing_artifact:paid_marketplace_launch_gate_json" in packet["artifact_blockers"]
    assert "missing_artifact:webapp_forwarding_preflight" in packet["artifact_blockers"]
