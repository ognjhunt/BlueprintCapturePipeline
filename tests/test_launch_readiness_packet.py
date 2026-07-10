from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.build_launch_readiness_packet import (
    _ci_evidence_blockers,
    _forwarding_packet_blockers,
    build_launch_readiness_packet,
    launch_readiness_exit_code,
)
from tests import test_release_evidence_graph as release_graph_fixture


_PIPELINE_TRUST: dict[Path, release_graph_fixture.TrustContext] = {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, text: str = "artifact") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pipeline_source(head: str) -> dict[str, object]:
    return {
        "repo_name": "BlueprintCapturePipeline",
        "head": head,
    }


def _valid_paid_release_evidence_graph(
    head: str, pipeline_repo: Path
) -> dict[str, object]:
    """Build the same signed v2 graph shape trusted by the production validator."""

    trust = _PIPELINE_TRUST[pipeline_repo.resolve()]
    evidence_dir = pipeline_repo / "output" / "test-release-evidence"
    previous = (
        release_graph_fixture.REPOSITORY_SHA,
        release_graph_fixture.IMAGE_DIGEST,
        release_graph_fixture.NOW,
    )
    release_graph_fixture.REPOSITORY_SHA = head
    release_graph_fixture.IMAGE_DIGEST = f"sha256:{'a' * 64}"
    release_graph_fixture.NOW = datetime(2026, 7, 7, tzinfo=timezone.utc)
    try:
        release_graph_fixture._write_scope(evidence_dir, "PAID", trust=trust)
        graph = release_graph_fixture._evaluate(
            evidence_dir, trust=trust, scope="PAID"
        )
    finally:
        (
            release_graph_fixture.REPOSITORY_SHA,
            release_graph_fixture.IMAGE_DIGEST,
            release_graph_fixture.NOW,
        ) = previous
    assert graph["status"] == "passed"
    return graph


def _beta_data_retention_policy_payload() -> dict[str, object]:
    return {
        "schema_version": "blueprint.beta_data_retention_policy.v1",
        "status": "declared_validator_enforced_operator_signoff_required",
    }


def _beta_data_residency_transfer_policy_payload() -> dict[str, object]:
    return {
        "schema_version": "blueprint.beta_data_residency_transfer_policy.v1",
        "status": "declared_us_only_scope_operator_signoff_required",
    }


def _write_ci_evidence(
    path: Path,
    *,
    evidence_id: str,
    head_sha: str,
    workflow: str,
    test_counts: dict[str, int] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "blueprint.github_actions_evidence.v1",
        "evidence_id": evidence_id,
        "workflow_name": workflow,
        "status": "completed",
        "conclusion": "success",
        "head_sha": head_sha,
        "url": f"https://github.test/actions/runs/{evidence_id}",
    }
    if test_counts is not None:
        payload["test_counts"] = test_counts
    _write_json(path, payload)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _init_repo_with_origin_main(repo: Path, *, dirty: bool = False) -> tuple[str, str]:
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _write_text(repo / "tracked.txt", "main")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "main")
    origin_main = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", origin_main)
    _git(repo, "checkout", "-b", "feature")
    _write_text(repo / "feature.txt", "feature")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature")
    feature_head = _git(repo, "rev-parse", "HEAD")
    if dirty:
        _write_text(repo / "dirty.txt", "dirty")
    return feature_head, origin_main


def _init_clean_repo_at_origin_main(
    repo: Path,
    *,
    tracked_files: dict[str, str] | None = None,
) -> str:
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _write_text(repo / "tracked.txt", "main")
    for relative_path, content in (tracked_files or {}).items():
        _write_text(repo / relative_path, content)
    _git(repo, "add", "tracked.txt")
    for relative_path in tracked_files or {}:
        _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", "main")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    return head


def _init_clean_pipeline_repo_at_origin_main(repo: Path) -> str:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    requirements = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "release_evidence_requirements.json"
        ).read_text(encoding="utf-8")
    )
    fingerprint = hashlib.sha256(public_key).hexdigest()
    for authority in requirements["attestation_authorities"].values():
        authority["public_key_sha256"] = fingerprint
    head = _init_clean_repo_at_origin_main(
        repo,
        tracked_files={
            "docs/release_evidence_requirements.json": json.dumps(requirements),
            "docs/beta_data_retention_policy_2026-07-09.json": json.dumps(
                _beta_data_retention_policy_payload()
            ),
            "docs/BETA_DATA_RETENTION_POLICY_2026-07-09.md": (
                "# Beta Data Retention Policy\n"
                "Status: `declared_validator_enforced_operator_signoff_required`\n"
            ),
            "docs/beta_data_residency_transfer_policy_2026-07-09.json": json.dumps(
                _beta_data_residency_transfer_policy_payload()
            ),
            "docs/BETA_DATA_RESIDENCY_TRANSFER_POLICY_2026-07-09.md": (
                "# Beta Data Residency And Transfer Policy\n"
                "Status: `declared_us_only_scope_operator_signoff_required`\n"
            ),
        },
    )
    _PIPELINE_TRUST[repo.resolve()] = release_graph_fixture.TrustContext(
        requirements_path=repo / "docs" / "release_evidence_requirements.json",
        private_key=private_key,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
    )
    return head


def test_launch_readiness_packet_links_artifacts_and_preserves_live_blockers(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "BlueprintCapturePipeline"
    webapp = tmp_path / "Blueprint-WebApp"
    contracts = tmp_path / "BlueprintContracts"
    capture = tmp_path / "BlueprintCapture"
    heads = {
        "pipeline": _init_clean_pipeline_repo_at_origin_main(pipeline),
        "webapp": _init_clean_repo_at_origin_main(webapp),
        "contracts": _init_clean_repo_at_origin_main(contracts),
        "capture": _init_clean_repo_at_origin_main(capture),
    }
    tracked_forwarding_preflight = (
        webapp / "output" / "pipeline" / "robot_eval_job_requests" / "forwarding_preflight.json"
    )
    _write_json(
        tracked_forwarding_preflight,
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "blocked",
            "blockers": ["placeholder"],
        },
    )
    _git(webapp, "add", str(tracked_forwarding_preflight.relative_to(webapp)))
    _git(webapp, "commit", "-m", "track forwarding preflight")
    _git(webapp, "update-ref", "refs/remotes/origin/main", _git(webapp, "rev-parse", "HEAD"))

    _write_json(
        pipeline / "output" / "paid_marketplace_launch_gate.json",
        {
            "schema_version": "blueprint.paid_marketplace_launch_gate.v2",
            "overall_status": "automated_contracts_passed_manual_ops_required",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "release_evidence_graph": _valid_paid_release_evidence_graph(
                heads["pipeline"], pipeline
            ),
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
            "pipeline_source": _pipeline_source(heads["pipeline"]),
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
        {
            "schema_version": "blueprint.sim_only_beta_local_gate_report.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "status": "passed",
        },
    )
    _write_json(
        pipeline / "output" / "launch_audit_live_pipeline_setup_20260707.json",
        {
            "schema_version": "blueprint_live_pipeline_setup.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "status": "local_ready_live_external_blocked",
            "blockers": ["delivery_upload:missing_delivery_command"],
        },
    )
    _write_json(
        tracked_forwarding_preflight,
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "blocked",
            "blockers": ["missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"],
        },
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_main_ci_evidence.json",
        evidence_id="pipeline_main_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="CI",
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_full_test_lane_ci_evidence.json",
        evidence_id="pipeline_full_test_lane_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="Full Test Lane",
        test_counts={"tests": 3917, "failures": 0, "errors": 0, "skipped": 0},
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_sim_only_local_gate_ci_evidence.json",
        evidence_id="pipeline_sim_only_local_gate_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="Sim-Only Local Gate",
    )
    _write_ci_evidence(
        pipeline / "output" / "webapp_main_ci_evidence.json",
        evidence_id="webapp_main_ci_evidence",
        head_sha=_git(webapp, "rev-parse", "HEAD"),
        workflow="CI",
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
    assert packet["artifact_trust_blockers"] == []
    assert packet["artifact_source_blockers"] == []
    assert all(artifact["sha256"] for artifact in packet["artifacts"])
    assert packet["readiness_summary"] == {
        "paid_marketplace_launch_gate": "automated_contracts_passed_manual_ops_required",
        "external_alpha_launch_gate": "passed_manual_required",
        "sim_only_beta_local_gate": "passed",
        "live_pipeline_setup": "local_ready_live_external_blocked",
        "webapp_forwarding_preflight": "blocked",
        "pipeline_main_ci": "success",
        "pipeline_full_test_lane_ci": "success",
        "pipeline_sim_only_local_gate_ci": "success",
        "webapp_main_ci": "success",
        "beta_data_retention_policy": "declared_validator_enforced_operator_signoff_required",
        "beta_data_residency_transfer_policy": ("declared_us_only_scope_operator_signoff_required"),
        "operator_evidence": "blocked",
    }
    assert packet["remaining_blockers"]["manual_live_evidence_ids"] == ["buyer_payment_settlement"]
    assert packet["remaining_blockers"]["external_alpha_manual_items"] == [
        "android_capture_contract_tests:manual_required"
    ]
    assert packet["remaining_blockers"]["live_pipeline_setup_blockers"] == [
        "delivery_upload:missing_delivery_command"
    ]
    assert packet["remaining_blockers"]["webapp_forwarding_blockers"] == [
        "missing_env_ROBOT_EVAL_JOB_REQUEST_FORWARD_URL"
    ]
    assert packet["remaining_blockers"]["ci_evidence_blockers"] == []
    assert packet["claim_boundary"]["automated_contracts_do_not_prove_real_pubsub_delivery"] is True
    assert (
        packet["claim_boundary"]["beta_data_retention_policy_is_declared_contract_not_signed_dpa"]
        is True
    )
    assert packet["operator_evidence_status"]["evidence_file_present"] is False
    assert packet["operator_evidence_status"]["remaining_ids"] == ["buyer_payment_settlement"]
    assert launch_readiness_exit_code(packet) == 1


def test_launch_readiness_packet_filters_verified_operator_evidence_ids(tmp_path: Path) -> None:
    pipeline = tmp_path / "BlueprintCapturePipeline"
    webapp = tmp_path / "Blueprint-WebApp"
    contracts = tmp_path / "BlueprintContracts"
    capture = tmp_path / "BlueprintCapture"
    heads = {
        "pipeline": _init_clean_pipeline_repo_at_origin_main(pipeline),
        "webapp": _init_clean_repo_at_origin_main(webapp),
        "contracts": _init_clean_repo_at_origin_main(contracts),
        "capture": _init_clean_repo_at_origin_main(capture),
    }

    _write_json(
        pipeline / "output" / "paid_marketplace_launch_gate.json",
        {
            "schema_version": "blueprint.paid_marketplace_launch_gate.v2",
            "overall_status": "automated_contracts_passed_manual_ops_required",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "release_evidence_graph": _valid_paid_release_evidence_graph(
                heads["pipeline"], pipeline
            ),
            "closeout_summary": {
                "remaining_manual_evidence_ids": [
                    "buyer_payment_settlement",
                    "buyer_artifact_access",
                ],
            },
        },
    )
    _write_text(pipeline / "output" / "paid_marketplace_launch_gate.md")
    _write_json(
        pipeline / "output" / "external_alpha_launch_gate.json",
        {
            "schema_version": "external_alpha_launch_gate.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "overall_status": "passed",
            "checks": [],
        },
    )
    _write_text(pipeline / "output" / "external_alpha_launch_gate.md")
    _write_json(
        pipeline / "output" / "sim_only_beta_local_gate_report.json",
        {
            "schema_version": "blueprint.sim_only_beta_local_gate_report.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "status": "passed",
        },
    )
    _write_json(
        pipeline / "output" / "launch_audit_live_pipeline_setup_20260707.json",
        {
            "schema_version": "blueprint_live_pipeline_setup.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "status": "ready",
            "blockers": [],
        },
    )
    _write_json(
        webapp / "output" / "pipeline" / "robot_eval_job_requests" / "forwarding_preflight.json",
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "ready",
            "blockers": [],
        },
    )
    _write_json(
        pipeline / "output" / "operator_launch_evidence.json",
        {
            "schema_version": "operator_launch_evidence.v1",
            "checks": {
                "buyer_payment_settlement": {
                    "status": "verified",
                    "evidence_uri": "gs://blueprint-live/evidence/buyer_payment_settlement.json",
                    "verified_at": "2026-07-07T00:00:00+00:00",
                    "verified_by": "ops-owner",
                    "payment_intent_id": "pi_live_123",
                    "stripe_event_id": "evt_live_123",
                    "stripe_mode": "live",
                }
            },
        },
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_main_ci_evidence.json",
        evidence_id="pipeline_main_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="CI",
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_full_test_lane_ci_evidence.json",
        evidence_id="pipeline_full_test_lane_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="Full Test Lane",
        test_counts={"tests": 3917, "failures": 0, "errors": 0, "skipped": 0},
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_sim_only_local_gate_ci_evidence.json",
        evidence_id="pipeline_sim_only_local_gate_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="Sim-Only Local Gate",
    )
    _write_ci_evidence(
        pipeline / "output" / "webapp_main_ci_evidence.json",
        evidence_id="webapp_main_ci_evidence",
        head_sha=heads["webapp"],
        workflow="CI",
    )

    packet = build_launch_readiness_packet(
        pipeline_repo=pipeline,
        webapp_repo=webapp,
        contracts_repo=contracts,
        capture_repo=capture,
        generated_at="2026-07-07T00:00:00+00:00",
    )

    assert packet["status"] == "local_ready_live_external_blocked"
    assert packet["operator_evidence_status"]["verified_ids"] == ["buyer_payment_settlement"]
    assert packet["operator_evidence_status"]["remaining_ids"] == ["buyer_artifact_access"]
    assert packet["remaining_blockers"]["manual_live_evidence_ids"] == ["buyer_artifact_access"]
    assert packet["readiness_summary"]["operator_evidence"] == "blocked"


def test_launch_readiness_packet_blocks_stale_pipeline_artifact_source_heads(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "BlueprintCapturePipeline"
    webapp = tmp_path / "Blueprint-WebApp"
    contracts = tmp_path / "BlueprintContracts"
    capture = tmp_path / "BlueprintCapture"
    heads = {
        "pipeline": _init_clean_pipeline_repo_at_origin_main(pipeline),
        "webapp": _init_clean_repo_at_origin_main(webapp),
        "contracts": _init_clean_repo_at_origin_main(contracts),
        "capture": _init_clean_repo_at_origin_main(capture),
    }
    stale_head = "0" * 40

    _write_json(
        pipeline / "output" / "paid_marketplace_launch_gate.json",
        {
            "schema_version": "blueprint.paid_marketplace_launch_gate.v2",
            "pipeline_source": _pipeline_source(stale_head),
            "overall_status": "automated_contracts_passed_manual_ops_required",
            "release_evidence_graph": _valid_paid_release_evidence_graph(
                heads["pipeline"], pipeline
            ),
            "closeout_summary": {"remaining_manual_evidence_ids": []},
        },
    )
    _write_text(pipeline / "output" / "paid_marketplace_launch_gate.md")
    _write_json(
        pipeline / "output" / "external_alpha_launch_gate.json",
        {
            "schema_version": "external_alpha_launch_gate.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "overall_status": "passed",
            "checks": [],
        },
    )
    _write_text(pipeline / "output" / "external_alpha_launch_gate.md")
    _write_json(
        pipeline / "output" / "sim_only_beta_local_gate_report.json",
        {
            "schema_version": "blueprint.sim_only_beta_local_gate_report.v1",
            "pipeline_source": _pipeline_source(heads["pipeline"]),
            "status": "passed",
        },
    )
    _write_json(
        pipeline / "output" / "launch_audit_live_pipeline_setup_20260707.json",
        {
            "schema_version": "blueprint_live_pipeline_setup.v1",
            "status": "ready",
            "blockers": [],
        },
    )
    _write_json(
        webapp / "output" / "pipeline" / "robot_eval_job_requests" / "forwarding_preflight.json",
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "ready_for_required_forwarding_with_probe",
            "blockers": [],
            "forwarding_required": True,
            "endpoint_configured": True,
            "probe": {"attempted": True, "status": "reachable"},
        },
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_main_ci_evidence.json",
        evidence_id="pipeline_main_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="CI",
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_full_test_lane_ci_evidence.json",
        evidence_id="pipeline_full_test_lane_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="Full Test Lane",
        test_counts={"tests": 3917, "failures": 0, "errors": 0, "skipped": 0},
    )
    _write_ci_evidence(
        pipeline / "output" / "pipeline_sim_only_local_gate_ci_evidence.json",
        evidence_id="pipeline_sim_only_local_gate_ci_evidence",
        head_sha=heads["pipeline"],
        workflow="Sim-Only Local Gate",
    )
    _write_ci_evidence(
        pipeline / "output" / "webapp_main_ci_evidence.json",
        evidence_id="webapp_main_ci_evidence",
        head_sha=heads["webapp"],
        workflow="CI",
    )

    packet = build_launch_readiness_packet(
        pipeline_repo=pipeline,
        webapp_repo=webapp,
        contracts_repo=contracts,
        capture_repo=capture,
        generated_at="2026-07-07T00:00:00+00:00",
    )

    assert packet["status"] == "incomplete_packet"
    assert packet["artifact_source_blockers"] == [
        f"artifact_source_head_mismatch:paid_marketplace_launch_gate_json:{stale_head}",
        "artifact_source_head_missing:live_pipeline_setup_audit",
    ]
    assert (
        packet["remaining_blockers"]["artifact_source_blockers"]
        == packet["artifact_source_blockers"]
    )


def test_forwarding_packet_blockers_reject_false_calm_without_probe() -> None:
    blockers = _forwarding_packet_blockers(
        {
            "status": "not_configured",
            "blockers": [],
            "forwarding_required": False,
            "endpoint_configured": False,
            "probe": {"attempted": False, "status": "not_requested"},
        }
    )

    assert blockers == [
        "webapp_forwarding_not_required",
        "webapp_forwarding_endpoint_not_configured",
        "webapp_forwarding_status_not_required_probe_ready:not_configured",
        "webapp_forwarding_probe_not_attempted",
        "webapp_forwarding_probe_not_reachable:not_requested",
    ]


def test_launch_readiness_exit_code_is_zero_only_for_ready_packet() -> None:
    assert launch_readiness_exit_code({"status": "ready"}) == 0
    assert launch_readiness_exit_code({"status": "incomplete_packet"}) == 1
    assert launch_readiness_exit_code({"status": "local_ready_live_external_blocked"}) == 1


def test_launch_readiness_blocks_successful_ci_evidence_with_skips() -> None:
    artifact = {
        "id": "pipeline_full_test_lane_ci_evidence",
        "repo_name": "BlueprintCapturePipeline",
        "exists": True,
        "schema_version": "blueprint.github_actions_evidence.v1",
        "conclusion": "success",
        "head_sha": "a" * 40,
        "test_counts": {"tests": 10, "failures": 0, "errors": 0, "skipped": 1},
    }

    blockers = _ci_evidence_blockers([artifact], {"BlueprintCapturePipeline": {"head": "a" * 40}})

    assert blockers == ["ci_evidence_test_skips:pipeline_full_test_lane_ci_evidence:1"]


def test_forwarding_packet_blockers_reject_localhost_probe_as_production_ready() -> None:
    blockers = _forwarding_packet_blockers(
        {
            "status": "ready_for_required_forwarding_with_probe",
            "blockers": [],
            "forwarding_required": True,
            "endpoint_configured": True,
            "configured_env": {
                "forward_url": {
                    "origin": "http://127.0.0.1:50560",
                },
            },
            "probe": {
                "attempted": True,
                "status": "reachable",
                "intake_audit_url": {
                    "origin": "http://127.0.0.1:50560",
                },
            },
        }
    )

    assert blockers == [
        "webapp_forwarding_forward_url_loopback:http://127.0.0.1:50560",
        "webapp_forwarding_probe_url_loopback:http://127.0.0.1:50560",
    ]


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
    assert "missing_artifact:beta_data_retention_policy_json" in packet["artifact_blockers"]
    assert (
        "missing_artifact:beta_data_residency_transfer_policy_json" in packet["artifact_blockers"]
    )
    assert (
        "missing_ci_evidence:pipeline_full_test_lane_ci_evidence" in packet["ci_evidence_blockers"]
    )
    assert (
        "missing_ci_evidence:pipeline_sim_only_local_gate_ci_evidence"
        in packet["ci_evidence_blockers"]
    )


def test_launch_readiness_packet_records_origin_main_when_local_checkout_is_dirty_feature(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "BlueprintCapturePipeline"
    webapp = tmp_path / "Blueprint-WebApp"
    contracts = tmp_path / "BlueprintContracts"
    capture = tmp_path / "BlueprintCapture"
    for repo in (pipeline, contracts, capture):
        repo.mkdir()
    feature_head, origin_main = _init_repo_with_origin_main(webapp, dirty=True)
    _write_json(
        webapp / "output" / "pipeline" / "robot_eval_job_requests" / "forwarding_preflight.json",
        {
            "schema_version": "blueprint.webapp.robot_eval_forwarding_readiness.v1",
            "status": "ready_for_required_forwarding_with_probe",
            "blockers": [],
        },
    )

    packet = build_launch_readiness_packet(
        pipeline_repo=pipeline,
        webapp_repo=webapp,
        contracts_repo=contracts,
        capture_repo=capture,
        generated_at="2026-07-07T00:00:00+00:00",
    )

    webapp_info = packet["repos"]["Blueprint-WebApp"]
    assert packet["status"] == "incomplete_packet"
    assert webapp_info["branch"] == "feature"
    assert webapp_info["head"] == feature_head
    assert webapp_info["origin_main_head"] == origin_main
    assert webapp_info["head_matches_origin_main"] is False
    assert webapp_info["dirty_entry_count"] == 2
    assert "repo_dirty:Blueprint-WebApp:1" in packet["repository_blockers"]
    assert "repo_not_at_origin_main:Blueprint-WebApp" in packet["repository_blockers"]
    assert (
        "untrusted_artifact_repo:webapp_forwarding_preflight:Blueprint-WebApp"
        in packet["artifact_trust_blockers"]
    )
