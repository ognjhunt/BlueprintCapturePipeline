from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.release_evidence_graph import (
    build_release_evidence_source_attestation_statement,
    evaluate_release_evidence_graph,
    load_release_evidence_requirements,
    validate_release_evidence_graph_result,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_REQUIREMENTS = ROOT / "docs" / "release_evidence_requirements.json"
REPOSITORY_SHA = "a" * 40
IMAGE_DIGEST = f"sha256:{'b' * 64}"
SOURCE_DIGEST = f"sha256:{'c' * 64}"
ARTIFACT_DIGEST = f"sha256:{'d' * 64}"
NOW = datetime(2026, 7, 9, 18, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TrustContext:
    requirements_path: Path
    private_key: Ed25519PrivateKey
    public_key_base64: str


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.fixture
def trust(tmp_path: Path) -> TrustContext:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    payload = json.loads(PRODUCTION_REQUIREMENTS.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(public_key).hexdigest()
    for authority in payload["attestation_authorities"].values():
        authority["public_key_sha256"] = fingerprint
    requirements_path = tmp_path / "release_evidence_requirements.json"
    requirements_path.write_text(json.dumps(payload), encoding="utf-8")
    return TrustContext(
        requirements_path=requirements_path,
        private_key=private_key,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
    )


def _github_source(workflow_path: str) -> dict[str, Any]:
    return {
        "ci_provider": "github_actions",
        "repository": "ognjhunt/BlueprintCapturePipeline",
        "workflow_path": workflow_path,
        "head_sha": REPOSITORY_SHA,
        "run_id": "123456789",
        "run_attempt": 1,
        "run_url": "https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/123456789",
        "jobs": [
            {
                "name": "required-release-check",
                "status": "completed",
                "conclusion": "success",
            }
        ],
    }


def _source_payload(
    requirements: dict[str, Any],
    node_id: str,
    *,
    scope: str,
) -> dict[str, Any]:
    requirement = requirements["nodes"][node_id]
    validation = requirements["node_validation"][node_id]
    accepted = requirement["accepted_statuses_by_scope"][scope][0]
    source: dict[str, Any] = {
        "schema_version": validation["source_schema_version"],
        "evidence_id": node_id,
        "evidence_schema_version": requirement["evidence_schema_version"],
        "status": accepted,
        "repository_sha": REPOSITORY_SHA,
        "image_digest": IMAGE_DIGEST,
        "generated_at": (NOW - timedelta(hours=1)).isoformat(),
        "expires_at": (NOW + timedelta(hours=1)).isoformat(),
        "blockers": [],
    }
    if validation["source_status_field"] == "conclusion":
        source["status"] = "completed"
        source["conclusion"] = accepted
    semantic: dict[str, dict[str, Any]] = {
        "pipeline_ci": _github_source(".github/workflows/ci.yml"),
        "full_test_lane_ci": {
            **_github_source(".github/workflows/full-test-lane.yml"),
            "lane_id": "cpu_full",
            "canonical_full_lane": True,
            "collection_filtering_used": False,
            "pytest_args": ["-m", ""],
            "planned_test_count": 4360,
            "executed_test_count": 4360,
            "junit_test_count": 4360,
            "planned_test_ids_sha256": ARTIFACT_DIGEST,
            "executed_test_ids_sha256": ARTIFACT_DIGEST,
            "junit_test_ids_sha256": ARTIFACT_DIGEST,
            "failure_count": 0,
            "error_count": 0,
            "skipped_count": 0,
        },
        "dependency_policy": {
            "known_vulnerability_count": 0,
            "dependencies_audited": 173,
            "uv_lock_sha256": ARTIFACT_DIGEST,
            "pip_audit_version": "2.10.1",
            "claim_boundary": {"runtime_python_dependency_scan_only": True},
        },
        "container_contract": {
            "lane_id": "container_production",
            "executed": True,
            "skipped_count": 0,
            "production_image_built": True,
            "nonroot_user_verified": True,
            "read_only_rootfs_verified": True,
            "healthcheck_passed": True,
            "compose_config_valid": True,
            "artifact_digests": {"production_image": ARTIFACT_DIGEST},
        },
        "sast_policy": {
            "scanner": "bandit",
            "finding_counts": {"high": 0, "medium": 2, "low": 4, "total": 6},
            "triaged_medium_count": 2,
        },
        "supply_chain_contract": {
            "component_count": 173,
            "license_review_count": 173,
            "artifact_subject_count": 2,
            "claim_boundary": {"sbom_and_provenance_generated": True},
        },
        "codeql_analysis": {
            **_github_source(".github/workflows/codeql.yml"),
            "alert_count": 0,
            "analysis_completed": True,
            "database_uploaded": True,
        },
        "sim_only_gate": {
            "simulator_execution_proven": True,
            "sim_only_beta_requirements_satisfied": True,
            "wam_handoff_artifacts_satisfied": True,
        },
        "ptdp_end_to_end": {
            "pipeline_complete": True,
            "archive_valid": True,
            "buyer_load_verified": True,
            "rights_verified": True,
            "privacy_verified": True,
            "provenance_verified": True,
            "training_row_count": 24,
            "package_digest": ARTIFACT_DIGEST,
        },
        "native_lerobot_export": {
            "lane_id": "native_lerobot_export",
            "executed": True,
            "skipped_count": 0,
            "export_file_count": 12,
            "validation_report": {
                "status": "passed",
                "loader": "lerobot_native+hermetic",
                "checks": {"lerobot_native_load": "passed"},
            },
            "artifact_digests": {"native_lerobot_export_tree": ARTIFACT_DIGEST},
        },
        "sc3_inputs": {
            "protocol_defined": True,
            "runtime_ready": True,
            "claim_ready": True,
            "accepted_anchor_count": 37,
            "matched_policy_count": 7,
            "study_digest": ARTIFACT_DIGEST,
        },
        "restore_drill": {
            "restore_verified": True,
            "digest_match_verified": True,
            "source_destroyed_before_restore": True,
            "restored_object_count": 51,
            "restored_tree_digest": ARTIFACT_DIGEST,
        },
        "provider_canary": {
            "lane_id": "gpu_provider_canary",
            "executed": True,
            "skipped_count": 0,
            "result_contract": {
                "heartbeat_completed": True,
                "gpu_sanity_completed": True,
                "provider_bundle_downloaded_and_ran": True,
                "provider_output_upload_ok": True,
                "provider_runtime_output_zip_produced": True,
                "canary_marker_observed": True,
                "continuing_spend_from_this_run": False,
            },
            "artifact_digests": {"vast_teardown_manifest.json": ARTIFACT_DIGEST},
        },
        "pubsub_integration": {
            "lane_id": "pubsub_emulator_integration",
            "executed": True,
            "skipped_count": 0,
            "emulator_loopback_only": True,
            "round_trip_payload_received": True,
            "message_acknowledged": True,
            "cleanup_succeeded": True,
            "artifact_digests": {"round_trip_payload": ARTIFACT_DIGEST},
        },
        "artifact_signature": {
            "proof_artifact_count": 3,
            "source_artifact_digest": ARTIFACT_DIGEST,
            "signature_verified": True,
            "certificate_identity_verified": True,
        },
        "immutable_retention": {
            "object_lock_mode": "COMPLIANCE",
            "readback_verified": True,
            "restore_readback_verified": True,
            "archive_uri": "s3://blueprint-release-evidence/releases/release.tar.gz",
            "version_id": "immutable-version-1",
            "retain_until": (NOW + timedelta(days=30)).isoformat(),
            "bundle_sha256": ARTIFACT_DIGEST,
        },
        "deployment_readback": {
            "service_healthy": True,
            "signature_verified": True,
            "sbom_digest_match": True,
            "commit_readback_verified": True,
            "deployed_repository_sha": REPOSITORY_SHA,
            "deployed_image_digest": IMAGE_DIGEST,
            "deployment_manifest_digest": ARTIFACT_DIGEST,
        },
    }
    source.update(semantic[node_id])
    return source


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_envelope_for_source(
    evidence_dir: Path,
    node_id: str,
    *,
    trust: TrustContext,
    scope: str,
    signing_key: Ed25519PrivateKey | None = None,
    public_key_base64: str | None = None,
) -> None:
    requirements = load_release_evidence_requirements(trust.requirements_path)
    requirement = requirements["nodes"][node_id]
    validation = requirements["node_validation"][node_id]
    source_path = evidence_dir / "sources" / f"{node_id}.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_digest = _sha256(source_path)
    claims_digest = f"sha256:{hashlib.sha256(_canonical(source)).hexdigest()}"
    generated_at = datetime.fromisoformat(source["generated_at"])
    expires_at = datetime.fromisoformat(source["expires_at"])
    statement = build_release_evidence_source_attestation_statement(
        authority_id=validation["trusted_attestation_authority_id"],
        node_id=node_id,
        source_artifact_digest=source_digest,
        source_claims_digest=claims_digest,
        repository_sha=REPOSITORY_SHA,
        image_digest=IMAGE_DIGEST,
        generated_at=generated_at,
        expires_at=expires_at,
    )
    key = signing_key or trust.private_key
    signature = key.sign(_canonical(statement))
    envelope = {
        "schema_version": "blueprint.release_evidence.v2",
        "evidence_id": node_id,
        "evidence_schema_version": requirement["evidence_schema_version"],
        "status": source[validation["source_status_field"]],
        "repository_sha": source["repository_sha"],
        "image_digest": source["image_digest"],
        "generated_at": source["generated_at"],
        "expires_at": source["expires_at"],
        "source_artifact_path": f"sources/{node_id}.json",
        "source_artifact_digest": source_digest,
        "evidence_uri": f"gs://blueprint-release-evidence/{node_id}.json",
        "source_verifier_attestation": {
            "schema_version": "blueprint.release_evidence_source_attestation.v1",
            "algorithm": "ed25519",
            "authority_id": validation["trusted_attestation_authority_id"],
            "public_key_base64": public_key_base64 or trust.public_key_base64,
            "statement": statement,
            "signature_base64": base64.b64encode(signature).decode("ascii"),
        },
    }
    (evidence_dir / f"{node_id}.json").write_text(json.dumps(envelope), encoding="utf-8")


def _write_evidence(
    evidence_dir: Path,
    node_id: str,
    *,
    trust: TrustContext,
    scope: str = "PAID",
    mutate_source: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    requirements = load_release_evidence_requirements(trust.requirements_path)
    source = _source_payload(requirements, node_id, scope=scope)
    if mutate_source is not None:
        mutate_source(source)
    source_dir = evidence_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{node_id}.json").write_text(json.dumps(source), encoding="utf-8")
    _write_envelope_for_source(evidence_dir, node_id, trust=trust, scope=scope)


def _write_scope(evidence_dir: Path, scope: str, *, trust: TrustContext) -> None:
    requirements = load_release_evidence_requirements(trust.requirements_path)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for node_id in requirements["scopes"][scope]:
        _write_evidence(evidence_dir, node_id, trust=trust, scope=scope)


def _mutate_and_resign(
    evidence_dir: Path,
    node_id: str,
    *,
    trust: TrustContext,
    mutate: Callable[[dict[str, Any]], None],
    scope: str = "PAID",
) -> None:
    source_path = evidence_dir / "sources" / f"{node_id}.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mutate(source)
    source_path.write_text(json.dumps(source), encoding="utf-8")
    _write_envelope_for_source(evidence_dir, node_id, trust=trust, scope=scope)


def _evaluate(
    evidence_dir: Path,
    *,
    trust: TrustContext,
    scope: str = "PAID",
) -> dict[str, object]:
    return evaluate_release_evidence_graph(
        scope=scope,
        repository_sha=REPOSITORY_SHA,
        image_digest=IMAGE_DIGEST,
        evidence_dir=evidence_dir,
        requirements_path=trust.requirements_path,
        now=NOW,
    )


def test_paid_release_evidence_graph_requires_bound_signed_native_sources(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)

    graph = _evaluate(evidence_dir, trust=trust)

    assert graph["status"] == "passed"
    assert graph["exit_code"] == 0
    assert graph["blockers"] == []
    assert all(node["outcome"] == "accepted" for node in graph["nodes"])
    assert all(node["source_binding_verified"] is True for node in graph["nodes"])
    assert graph["claim_boundary"]["envelope_or_uri_is_never_source_proof"] is True
    assert graph["claim_boundary"]["every_node_requires_trusted_source_attestation"] is True


def test_forged_uri_only_paid_envelopes_cannot_pass(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    requirements = load_release_evidence_requirements(trust.requirements_path)
    evidence_dir.mkdir()
    for node_id in requirements["scopes"]["PAID"]:
        requirement = requirements["nodes"][node_id]
        payload = {
            "schema_version": "blueprint.release_evidence.v2",
            "evidence_id": node_id,
            "evidence_schema_version": requirement["evidence_schema_version"],
            "status": requirement["accepted_statuses_by_scope"]["PAID"][0],
            "repository_sha": REPOSITORY_SHA,
            "image_digest": IMAGE_DIGEST,
            "generated_at": (NOW - timedelta(hours=1)).isoformat(),
            "expires_at": (NOW + timedelta(hours=1)).isoformat(),
            "source_artifact_digest": SOURCE_DIGEST,
            "evidence_uri": f"gs://blueprint-release-evidence/{node_id}.json",
        }
        (evidence_dir / f"{node_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    graph = _evaluate(evidence_dir, trust=trust)

    assert graph["status"] == "blocked"
    assert graph["exit_code"] == 1
    assert any(blocker.startswith("source_artifact:pipeline_ci:") for blocker in graph["blockers"])
    assert any(
        blocker.startswith("untrusted_attestation:pipeline_ci:") for blocker in graph["blockers"]
    )


def test_trusted_signature_does_not_make_generic_local_source_semantically_valid(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)

    def remove_dependency_proof(source: dict[str, Any]) -> None:
        for field in (
            "known_vulnerability_count",
            "dependencies_audited",
            "uv_lock_sha256",
            "pip_audit_version",
            "claim_boundary",
        ):
            source.pop(field, None)

    _mutate_and_resign(
        evidence_dir,
        "dependency_policy",
        trust=trust,
        mutate=remove_dependency_proof,
    )

    graph = _evaluate(evidence_dir, trust=trust)

    assert graph["status"] == "blocked"
    assert any(
        blocker.startswith("malformed_evidence:dependency_policy:source_semantic:")
        for blocker in graph["blockers"]
    )


def test_full_lane_requires_identical_planned_executed_junit_zero_skip_proof(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    _mutate_and_resign(
        evidence_dir,
        "full_test_lane_ci",
        trust=trust,
        mutate=lambda source: source.update(
            {
                "executed_test_count": source["planned_test_count"] - 1,
                "junit_test_ids_sha256": SOURCE_DIGEST,
                "skipped_count": 1,
            }
        ),
    )

    graph = _evaluate(evidence_dir, trust=trust)

    blockers = graph["blockers"]
    assert (
        "malformed_evidence:full_test_lane_ci:source_semantic:planned_executed_junit_counts"
        in blockers
    )
    assert (
        "malformed_evidence:full_test_lane_ci:source_semantic:planned_executed_junit_id_digests"
        in blockers
    )
    assert "malformed_evidence:full_test_lane_ci:source_semantic:skipped_count" in blockers


def test_cross_node_attestation_replay_and_digest_relabel_block(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    pipeline = json.loads((evidence_dir / "pipeline_ci.json").read_text(encoding="utf-8"))
    codeql_path = evidence_dir / "codeql_analysis.json"
    codeql = json.loads(codeql_path.read_text(encoding="utf-8"))
    codeql["source_verifier_attestation"] = pipeline["source_verifier_attestation"]
    codeql["source_artifact_digest"] = SOURCE_DIGEST
    codeql_path.write_text(json.dumps(codeql), encoding="utf-8")

    graph = _evaluate(evidence_dir, trust=trust)

    assert "source_artifact:codeql_analysis:digest_mismatch" in graph["blockers"]
    assert any(
        blocker.startswith("untrusted_attestation:codeql_analysis:")
        for blocker in graph["blockers"]
    )


def test_source_symlink_escape_and_post_signature_mutation_block(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    source_path = evidence_dir / "sources" / "pipeline_ci.json"
    outside = tmp_path / "outside.json"
    source_path.rename(outside)
    source_path.symlink_to(outside)

    symlinked = _evaluate(evidence_dir, trust=trust)

    assert "source_artifact:pipeline_ci:source_artifact_symlink" in symlinked["blockers"]

    source_path.unlink()
    outside.rename(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["jobs"] = []
    source_path.write_text(json.dumps(source), encoding="utf-8")
    mutated = _evaluate(evidence_dir, trust=trust)
    assert "source_artifact:pipeline_ci:digest_mismatch" in mutated["blockers"]
    assert "malformed_evidence:pipeline_ci:source_semantic:jobs" in mutated["blockers"]

    envelope_path = evidence_dir / "pipeline_ci.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["source_artifact_path"] = "sources/../outside.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    escaped = _evaluate(evidence_dir, trust=trust)
    assert "source_artifact:pipeline_ci:source_artifact_path_invalid" in escaped["blockers"]


def test_self_signed_untrusted_authority_cannot_pass(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    attacker = Ed25519PrivateKey.generate()
    attacker_public = attacker.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    _write_envelope_for_source(
        evidence_dir,
        "provider_canary",
        trust=trust,
        scope="PAID",
        signing_key=attacker,
        public_key_base64=base64.b64encode(attacker_public).decode("ascii"),
    )

    graph = _evaluate(evidence_dir, trust=trust)

    assert "untrusted_attestation:provider_canary:public_key_pin" in graph["blockers"]


@pytest.mark.parametrize(
    ("node_id", "status", "expected_blocker"),
    [
        ("pipeline_ci", "failure", "red_ci:pipeline_ci:failure"),
        ("sc3_inputs", "blocked", "sc3_inputs_blocked:blocked"),
        ("dependency_policy", "failed", "dependency_policy_failed:failed"),
        ("provider_canary", "failed", "provider_canary_failed:failed"),
    ],
)
def test_paid_release_evidence_graph_rejects_source_failure_statuses(
    tmp_path: Path,
    trust: TrustContext,
    node_id: str,
    status: str,
    expected_blocker: str,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    requirements = load_release_evidence_requirements(trust.requirements_path)
    status_field = requirements["node_validation"][node_id]["source_status_field"]
    _mutate_and_resign(
        evidence_dir,
        node_id,
        trust=trust,
        mutate=lambda source: source.update({status_field: status}),
    )

    graph = _evaluate(evidence_dir, trust=trust)

    assert graph["status"] == "blocked"
    assert expected_blocker in graph["blockers"]


def test_missing_restore_and_stale_source_block(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    (evidence_dir / "restore_drill.json").unlink()
    missing = _evaluate(evidence_dir, trust=trust)
    assert "missing_evidence:restore_drill" in missing["blockers"]

    _write_evidence(evidence_dir, "restore_drill", trust=trust)
    _mutate_and_resign(
        evidence_dir,
        "pipeline_ci",
        trust=trust,
        mutate=lambda source: source.update(
            {
                "generated_at": (NOW - timedelta(days=2)).isoformat(),
                "expires_at": (NOW - timedelta(days=1)).isoformat(),
            }
        ),
    )
    stale = _evaluate(evidence_dir, trust=trust)
    assert "stale_evidence:pipeline_ci" in stale["blockers"]


def test_wrong_source_commit_and_image_block_even_when_resigned(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    _mutate_and_resign(
        evidence_dir,
        "dependency_policy",
        trust=trust,
        mutate=lambda source: source.update(
            {"repository_sha": "e" * 40, "image_digest": f"sha256:{'f' * 64}"}
        ),
    )

    graph = _evaluate(evidence_dir, trust=trust)

    assert f"wrong_repository_sha:dependency_policy:{'e' * 40}" in graph["blockers"]
    assert f"wrong_image_digest:dependency_policy:sha256:{'f' * 64}" in graph["blockers"]


def test_sim_scope_stays_scope_bounded_but_still_requires_trusted_sources(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "SIM", trust=trust)

    graph = _evaluate(evidence_dir, trust=trust, scope="SIM")

    assert graph["status"] == "passed"
    assert "sc3_inputs" not in graph["required_node_ids"]
    assert "restore_drill" not in graph["required_node_ids"]
    assert graph["claim_boundary"]["sim_only_does_not_require_physical_robot_evidence"] is True


def test_production_unconfigured_trust_roots_fail_closed(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    graph = evaluate_release_evidence_graph(
        scope="BASE",
        repository_sha=REPOSITORY_SHA,
        image_digest=IMAGE_DIGEST,
        evidence_dir=evidence_dir,
        requirements_path=PRODUCTION_REQUIREMENTS,
        now=NOW,
    )
    assert graph["status"] == "blocked"


def test_persisted_graph_revalidates_signed_claims_and_rejects_tampering(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    graph = _evaluate(evidence_dir, trust=trust)

    assert (
        validate_release_evidence_graph_result(
            graph,
            expected_scope="PAID",
            expected_repository_sha=REPOSITORY_SHA,
            requirements_path=trust.requirements_path,
            expected_image_digest=IMAGE_DIGEST,
            now=NOW,
        )
        == []
    )

    dependency_node = next(node for node in graph["nodes"] if node["id"] == "dependency_policy")
    dependency_node["source_claims"]["known_vulnerability_count"] = 9
    blockers = validate_release_evidence_graph_result(
        graph,
        expected_scope="PAID",
        expected_repository_sha=REPOSITORY_SHA,
        requirements_path=trust.requirements_path,
        expected_image_digest=IMAGE_DIGEST,
        now=NOW,
    )
    assert any(
        blocker.startswith("release_evidence_graph_node_binding_invalid:dependency_policy:")
        for blocker in blockers
    )


def test_nested_graph_cannot_omit_required_paid_node(
    tmp_path: Path,
    trust: TrustContext,
) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_scope(evidence_dir, "PAID", trust=trust)
    graph = _evaluate(evidence_dir, trust=trust)
    graph["required_node_ids"] = graph["required_node_ids"][:-1]
    graph["nodes"] = graph["nodes"][:-1]

    blockers = validate_release_evidence_graph_result(
        graph,
        expected_scope="PAID",
        expected_repository_sha=REPOSITORY_SHA,
        requirements_path=trust.requirements_path,
        now=NOW,
    )

    assert "release_evidence_graph_required_nodes_mismatch" in blockers
