"""Fail-closed, release-bound evidence graph evaluation.

Evidence envelopes are routing metadata, never proof by themselves. Every node
must bind a contained source artifact and a trusted Ed25519 verifier
attestation. The evaluator recomputes the source digest, validates the native
source contract and node-specific semantics, and verifies that the trusted
attestation covers the node, release, source bytes, normalized source claims,
and validity interval.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


REQUIREMENTS_SCHEMA_VERSION = "blueprint.release_evidence_requirements.v2"
EVIDENCE_ENVELOPE_SCHEMA_VERSION = "blueprint.release_evidence.v2"
GRAPH_SCHEMA_VERSION = "blueprint.release_evidence_graph.v2"
SOURCE_ATTESTATION_SCHEMA_VERSION = "blueprint.release_evidence_source_attestation.v1"
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
KEY_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EVIDENCE_URI_SCHEMES = {"gs", "https", "oci", "s3"}
MAX_SOURCE_ARTIFACT_BYTES = 2 * 1024 * 1024

_SEMANTIC_VALIDATORS = {
    "pipeline_ci",
    "full_test_lane_ci",
    "dependency_policy",
    "container_contract",
    "sast_policy",
    "supply_chain_contract",
    "codeql_analysis",
    "sim_only_gate",
    "ptdp_end_to_end",
    "native_lerobot_export",
    "sc3_inputs",
    "restore_drill",
    "provider_canary",
    "pubsub_integration",
    "artifact_signature",
    "immutable_retention",
    "deployment_readback",
}

_BLOCKER_PRIORITY = {
    "automation_failed": 0,
    "untrusted_attestation": 1,
    "source_artifact": 1,
    "malformed_evidence": 2,
    "wrong_repository_sha": 3,
    "wrong_image_digest": 4,
    "stale_evidence": 5,
    "future_evidence": 6,
    "status_not_accepted": 7,
    "missing_evidence": 8,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _load_mapping(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_SOURCE_ARTIFACT_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _mapping_digest(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json_bytes(value)).hexdigest()}"


def _evidence_uri_is_durable(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme.lower() in ALLOWED_EVIDENCE_URI_SCHEMES
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
    )


def _get_path(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _strict_int(value: object) -> int | None:
    return value if type(value) is int else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _source_relative_path(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or "\\" in text or "\x00" in text:
        return None
    pure = PurePosixPath(text)
    if pure.is_absolute() or not pure.parts or pure.parts[0] != "sources":
        return None
    if any(part in {"", ".", ".."} for part in pure.parts):
        return None
    return pure.as_posix()


def _resolve_source_artifact(
    *, evidence_dir: Path, raw_path: object
) -> tuple[Path | None, str | None]:
    relative = _source_relative_path(raw_path)
    if relative is None:
        return None, "source_artifact_path_invalid"
    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        return None, "source_artifact_root_invalid"
    root = evidence_dir.resolve()
    candidate = evidence_dir
    for part in PurePosixPath(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            return None, "source_artifact_symlink"
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None, "source_artifact_missing_or_escape"
    if not candidate.is_file() or candidate.stat().st_size > MAX_SOURCE_ARTIFACT_BYTES:
        return None, "source_artifact_missing_or_oversized"
    return candidate, None


def build_release_evidence_source_attestation_statement(
    *,
    authority_id: str,
    node_id: str,
    source_artifact_digest: str,
    source_claims_digest: str,
    repository_sha: str,
    image_digest: str,
    generated_at: datetime,
    expires_at: datetime,
) -> dict[str, Any]:
    """Return the exact statement a trusted verifier must sign."""

    return {
        "schema_version": SOURCE_ATTESTATION_SCHEMA_VERSION,
        "authority_id": authority_id,
        "node_id": node_id,
        "source_artifact_digest": source_artifact_digest,
        "source_claims_digest": source_claims_digest,
        "repository_sha": repository_sha,
        "image_digest": image_digest,
        "generated_at": _format_time(generated_at),
        "expires_at": _format_time(expires_at),
    }


def load_release_evidence_requirements(path: Path) -> dict[str, Any]:
    """Load and structurally validate the requirements graph and trust pins."""

    payload = _load_mapping(path)
    if payload is None:
        raise ValueError(f"release evidence requirements are unreadable: {path}")
    if payload.get("schema_version") != REQUIREMENTS_SCHEMA_VERSION:
        raise ValueError("release evidence requirements schema mismatch")
    if payload.get("evidence_envelope_schema_version") != EVIDENCE_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("release evidence envelope schema mismatch")
    scopes = payload.get("scopes")
    nodes = payload.get("nodes")
    validations = payload.get("node_validation")
    authorities = payload.get("attestation_authorities")
    if not isinstance(scopes, Mapping):
        raise ValueError("release evidence requirements scopes mapping missing")
    if not isinstance(nodes, Mapping):
        raise ValueError("release evidence requirements nodes mapping missing")
    if not isinstance(validations, Mapping):
        raise ValueError("release evidence requirements validation mapping missing")
    if not isinstance(authorities, Mapping):
        raise ValueError("release evidence requirements authorities mapping missing")
    for authority_id, raw_authority in authorities.items():
        if not isinstance(authority_id, str) or not isinstance(raw_authority, Mapping):
            raise ValueError("release evidence authorities must be named mappings")
        if raw_authority.get("algorithm") != "ed25519":
            raise ValueError(f"release evidence authority {authority_id} algorithm invalid")
        fingerprint = raw_authority.get("public_key_sha256")
        if fingerprint is not None and KEY_FINGERPRINT_PATTERN.fullmatch(str(fingerprint)) is None:
            raise ValueError(f"release evidence authority {authority_id} key pin invalid")
    for scope, required_ids in scopes.items():
        if not isinstance(scope, str) or not isinstance(required_ids, list) or not required_ids:
            raise ValueError("every release scope needs a nonempty node list")
        if len(required_ids) != len(set(required_ids)):
            raise ValueError(f"release scope {scope} repeats evidence nodes")
        unknown = [node_id for node_id in required_ids if node_id not in nodes]
        if unknown:
            raise ValueError(f"release scope {scope} references unknown nodes: {unknown}")
    if set(validations) != set(nodes):
        raise ValueError("every release evidence node needs exactly one validation policy")
    for node_id, raw_requirement in nodes.items():
        if not isinstance(node_id, str) or not isinstance(raw_requirement, Mapping):
            raise ValueError("release evidence nodes must be named mappings")
        accepted_by_scope = raw_requirement.get("accepted_statuses_by_scope")
        max_age = raw_requirement.get("max_age_seconds")
        if not isinstance(raw_requirement.get("evidence_schema_version"), str):
            raise ValueError(f"release evidence node {node_id} has no payload schema")
        if not isinstance(accepted_by_scope, Mapping):
            raise ValueError(f"release evidence node {node_id} has no scope status allowlists")
        for scope, required_ids in scopes.items():
            if node_id not in required_ids:
                continue
            accepted = accepted_by_scope.get(scope)
            if (
                not isinstance(accepted, list)
                or not accepted
                or not all(isinstance(status, str) and status for status in accepted)
            ):
                raise ValueError(
                    f"release evidence node {node_id} has no accepted statuses for {scope}"
                )
        if not isinstance(max_age, int) or max_age <= 0:
            raise ValueError(f"release evidence node {node_id} has invalid max age")
        validation = validations[node_id]
        if not isinstance(validation, Mapping):
            raise ValueError(f"release evidence node {node_id} validation invalid")
        if not isinstance(validation.get("source_schema_version"), str):
            raise ValueError(f"release evidence node {node_id} source schema missing")
        if validation.get("source_status_field") not in {"status", "conclusion"}:
            raise ValueError(f"release evidence node {node_id} status field invalid")
        if validation.get("semantic_validator") not in _SEMANTIC_VALIDATORS:
            raise ValueError(f"release evidence node {node_id} semantic validator invalid")
        authority_id = validation.get("trusted_attestation_authority_id")
        if not isinstance(authority_id, str) or authority_id not in authorities:
            raise ValueError(f"release evidence node {node_id} authority invalid")
    return payload


def _special_status_blocker(node_id: str, status: str) -> str:
    if node_id in {"pipeline_ci", "full_test_lane_ci"}:
        return f"red_ci:{node_id}:{status or 'missing'}"
    if node_id == "dependency_policy":
        return f"dependency_policy_failed:{status or 'missing'}"
    if node_id == "provider_canary":
        return f"provider_canary_failed:{status or 'missing'}"
    if node_id == "sc3_inputs":
        return f"sc3_inputs_blocked:{status or 'missing'}"
    return f"status_not_accepted:{node_id}:{status or 'missing'}"


def _special_missing_blocker(node_id: str) -> str:
    return f"missing_evidence:{node_id}"


def _blocker_class(blocker: str) -> str:
    prefix = blocker.split(":", 1)[0]
    if prefix in {
        "red_ci",
        "dependency_policy_failed",
        "provider_canary_failed",
        "sc3_inputs_blocked",
    }:
        return "status_not_accepted"
    return prefix if prefix in _BLOCKER_PRIORITY else "malformed_evidence"


def _semantic_blocker(node_id: str, field: str) -> str:
    return f"malformed_evidence:{node_id}:source_semantic:{field}"


def _validate_github_source(
    *,
    node_id: str,
    source: Mapping[str, Any],
    repository_sha: str,
    workflow_path: str,
) -> list[str]:
    blockers: list[str] = []
    exact = {
        "ci_provider": "github_actions",
        "repository": "ognjhunt/BlueprintCapturePipeline",
        "workflow_path": workflow_path,
        "head_sha": repository_sha,
    }
    for field, expected in exact.items():
        if source.get(field) != expected:
            blockers.append(_semantic_blocker(node_id, field))
    run_id = str(source.get("run_id") or "")
    run_attempt = _strict_int(source.get("run_attempt"))
    if not run_id.isdigit() or int(run_id) <= 0:
        blockers.append(_semantic_blocker(node_id, "run_id"))
    if run_attempt is None or run_attempt <= 0:
        blockers.append(_semantic_blocker(node_id, "run_attempt"))
    run_url = str(source.get("run_url") or "")
    parsed = urlsplit(run_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or not run_id
        or f"/actions/runs/{run_id}" not in parsed.path
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        blockers.append(_semantic_blocker(node_id, "run_url"))
    jobs = source.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        blockers.append(_semantic_blocker(node_id, "jobs"))
    elif any(
        not isinstance(job, Mapping)
        or job.get("status") != "completed"
        or job.get("conclusion") != "success"
        for job in jobs
    ):
        blockers.append(_semantic_blocker(node_id, "jobs_not_all_success"))
    return blockers


def _validate_full_test_lane_source(
    *, node_id: str, source: Mapping[str, Any], repository_sha: str
) -> list[str]:
    blockers = _validate_github_source(
        node_id=node_id,
        source=source,
        repository_sha=repository_sha,
        workflow_path=".github/workflows/full-test-lane.yml",
    )
    exact = {
        "lane_id": "cpu_full",
        "canonical_full_lane": True,
        "collection_filtering_used": False,
        "pytest_args": ["-m", ""],
    }
    for field, expected in exact.items():
        if source.get(field) != expected:
            blockers.append(_semantic_blocker(node_id, field))
    counts = [
        _strict_int(source.get(field))
        for field in ("planned_test_count", "executed_test_count", "junit_test_count")
    ]
    if any(value is None or value <= 0 for value in counts) or len(set(counts)) != 1:
        blockers.append(_semantic_blocker(node_id, "planned_executed_junit_counts"))
    digests = [
        str(source.get(field) or "")
        for field in (
            "planned_test_ids_sha256",
            "executed_test_ids_sha256",
            "junit_test_ids_sha256",
        )
    ]
    if (
        any(ARTIFACT_DIGEST_PATTERN.fullmatch(value) is None for value in digests)
        or len(set(digests)) != 1
    ):
        blockers.append(_semantic_blocker(node_id, "planned_executed_junit_id_digests"))
    for field in ("failure_count", "error_count", "skipped_count"):
        if _strict_int(source.get(field)) != 0:
            blockers.append(_semantic_blocker(node_id, field))
    return blockers


def _validate_rule_set(
    *,
    node_id: str,
    source: Mapping[str, Any],
    equals: Mapping[str, Any] | None = None,
    true_fields: Sequence[str] = (),
    false_fields: Sequence[str] = (),
    zero_fields: Sequence[str] = (),
    positive_int_fields: Sequence[str] = (),
    digest_fields: Sequence[str] = (),
    nonempty_fields: Sequence[str] = (),
) -> list[str]:
    blockers: list[str] = []
    for field, expected in (equals or {}).items():
        if _get_path(source, field) != expected:
            blockers.append(_semantic_blocker(node_id, field))
    for field in true_fields:
        if _get_path(source, field) is not True:
            blockers.append(_semantic_blocker(node_id, field))
    for field in false_fields:
        if _get_path(source, field) is not False:
            blockers.append(_semantic_blocker(node_id, field))
    for field in zero_fields:
        if _strict_int(_get_path(source, field)) != 0:
            blockers.append(_semantic_blocker(node_id, field))
    for field in positive_int_fields:
        value = _strict_int(_get_path(source, field))
        if value is None or value <= 0:
            blockers.append(_semantic_blocker(node_id, field))
    for field in digest_fields:
        if ARTIFACT_DIGEST_PATTERN.fullmatch(str(_get_path(source, field) or "")) is None:
            blockers.append(_semantic_blocker(node_id, field))
    for field in nonempty_fields:
        value = _get_path(source, field)
        if not isinstance(value, str) or not value.strip():
            blockers.append(_semantic_blocker(node_id, field))
    return blockers


def _validate_semantic_source(
    *,
    node_id: str,
    validator: str,
    source: Mapping[str, Any],
    repository_sha: str,
    image_digest: str,
    expires_at: datetime | None,
) -> list[str]:
    if validator == "pipeline_ci":
        return _validate_github_source(
            node_id=node_id,
            source=source,
            repository_sha=repository_sha,
            workflow_path=".github/workflows/ci.yml",
        )
    if validator == "full_test_lane_ci":
        return _validate_full_test_lane_source(
            node_id=node_id, source=source, repository_sha=repository_sha
        )
    if validator == "dependency_policy":
        return _validate_rule_set(
            node_id=node_id,
            source=source,
            zero_fields=("known_vulnerability_count",),
            positive_int_fields=("dependencies_audited",),
            digest_fields=("uv_lock_sha256",),
            nonempty_fields=("pip_audit_version",),
            true_fields=("claim_boundary.runtime_python_dependency_scan_only",),
        )
    if validator == "container_contract":
        return _validate_rule_set(
            node_id=node_id,
            source=source,
            equals={"lane_id": "container_production"},
            true_fields=(
                "executed",
                "production_image_built",
                "nonroot_user_verified",
                "read_only_rootfs_verified",
                "healthcheck_passed",
                "compose_config_valid",
            ),
            zero_fields=("skipped_count",),
            digest_fields=("artifact_digests.production_image",),
        )
    if validator == "sast_policy":
        blockers = _validate_rule_set(
            node_id=node_id,
            source=source,
            zero_fields=("finding_counts.high",),
            nonempty_fields=("scanner",),
        )
        medium = _strict_int(_get_path(source, "finding_counts.medium"))
        triaged = _strict_int(source.get("triaged_medium_count"))
        total = _strict_int(_get_path(source, "finding_counts.total"))
        if medium is None or medium < 0 or triaged != medium or total is None or total < medium:
            blockers.append(_semantic_blocker(node_id, "finding_counts"))
        return blockers
    if validator == "supply_chain_contract":
        return _validate_rule_set(
            node_id=node_id,
            source=source,
            positive_int_fields=(
                "component_count",
                "license_review_count",
                "artifact_subject_count",
            ),
            true_fields=("claim_boundary.sbom_and_provenance_generated",),
        )
    if validator == "codeql_analysis":
        blockers = _validate_github_source(
            node_id=node_id,
            source=source,
            repository_sha=repository_sha,
            workflow_path=".github/workflows/codeql.yml",
        )
        blockers.extend(
            _validate_rule_set(
                node_id=node_id,
                source=source,
                zero_fields=("alert_count",),
                true_fields=("analysis_completed", "database_uploaded"),
            )
        )
        return blockers
    rules: dict[str, dict[str, Any]] = {
        "sim_only_gate": {
            "true_fields": (
                "simulator_execution_proven",
                "sim_only_beta_requirements_satisfied",
                "wam_handoff_artifacts_satisfied",
            )
        },
        "ptdp_end_to_end": {
            "true_fields": (
                "pipeline_complete",
                "archive_valid",
                "buyer_load_verified",
                "rights_verified",
                "privacy_verified",
                "provenance_verified",
            ),
            "positive_int_fields": ("training_row_count",),
            "digest_fields": ("package_digest",),
        },
        "native_lerobot_export": {
            "equals": {
                "lane_id": "native_lerobot_export",
                "validation_report.status": "passed",
                "validation_report.loader": "lerobot_native+hermetic",
                "validation_report.checks.lerobot_native_load": "passed",
            },
            "true_fields": ("executed",),
            "zero_fields": ("skipped_count",),
            "positive_int_fields": ("export_file_count",),
            "digest_fields": ("artifact_digests.native_lerobot_export_tree",),
        },
        "sc3_inputs": {
            "true_fields": ("protocol_defined", "runtime_ready", "claim_ready"),
            "positive_int_fields": ("accepted_anchor_count", "matched_policy_count"),
            "digest_fields": ("study_digest",),
        },
        "restore_drill": {
            "true_fields": (
                "restore_verified",
                "digest_match_verified",
                "source_destroyed_before_restore",
            ),
            "positive_int_fields": ("restored_object_count",),
            "digest_fields": ("restored_tree_digest",),
        },
        "provider_canary": {
            "equals": {"lane_id": "gpu_provider_canary"},
            "true_fields": ("executed",),
            "zero_fields": ("skipped_count",),
        },
        "pubsub_integration": {
            "equals": {"lane_id": "pubsub_emulator_integration"},
            "true_fields": (
                "executed",
                "emulator_loopback_only",
                "round_trip_payload_received",
                "message_acknowledged",
                "cleanup_succeeded",
            ),
            "zero_fields": ("skipped_count",),
            "digest_fields": ("artifact_digests.round_trip_payload",),
        },
        "artifact_signature": {
            "positive_int_fields": ("proof_artifact_count",),
            "digest_fields": ("source_artifact_digest",),
            "true_fields": ("signature_verified", "certificate_identity_verified"),
        },
        "immutable_retention": {
            "equals": {"object_lock_mode": "COMPLIANCE"},
            "true_fields": ("readback_verified", "restore_readback_verified"),
            "nonempty_fields": ("archive_uri", "version_id", "retain_until"),
            "digest_fields": ("bundle_sha256",),
        },
        "deployment_readback": {
            "true_fields": (
                "service_healthy",
                "signature_verified",
                "sbom_digest_match",
                "commit_readback_verified",
            ),
            "equals": {
                "deployed_repository_sha": repository_sha,
                "deployed_image_digest": image_digest,
            },
            "digest_fields": ("deployment_manifest_digest",),
        },
    }
    rule = rules.get(validator)
    if rule is None:
        return [_semantic_blocker(node_id, "validator_unknown")]
    blockers = _validate_rule_set(
        node_id=node_id,
        source=source,
        equals=rule.get("equals") if isinstance(rule.get("equals"), Mapping) else None,
        true_fields=tuple(rule.get("true_fields") or ()),
        false_fields=tuple(rule.get("false_fields") or ()),
        zero_fields=tuple(rule.get("zero_fields") or ()),
        positive_int_fields=tuple(rule.get("positive_int_fields") or ()),
        digest_fields=tuple(rule.get("digest_fields") or ()),
        nonempty_fields=tuple(rule.get("nonempty_fields") or ()),
    )
    if validator == "sc3_inputs":
        count = _strict_int(source.get("matched_policy_count"))
        if count is None or count < 7:
            blockers.append(_semantic_blocker(node_id, "matched_policy_count_min_7"))
    if validator == "provider_canary":
        contract = source.get("result_contract")
        if (
            not isinstance(contract, Mapping)
            or contract.get("continuing_spend_from_this_run") is not False
        ):
            blockers.append(_semantic_blocker(node_id, "provider_teardown"))
        required_true = {
            "heartbeat_completed",
            "gpu_sanity_completed",
            "provider_bundle_downloaded_and_ran",
            "provider_output_upload_ok",
            "provider_runtime_output_zip_produced",
            "canary_marker_observed",
        }
        if not isinstance(contract, Mapping) or any(
            contract.get(field) is not True for field in required_true
        ):
            blockers.append(_semantic_blocker(node_id, "provider_result_contract"))
        artifacts = source.get("artifact_digests")
        teardown_digest = (
            artifacts.get("vast_teardown_manifest.json") if isinstance(artifacts, Mapping) else None
        )
        if ARTIFACT_DIGEST_PATTERN.fullmatch(str(teardown_digest or "")) is None:
            blockers.append(_semantic_blocker(node_id, "provider_teardown_digest"))
    if validator == "immutable_retention":
        retain_until = _parse_time(source.get("retain_until"))
        if retain_until is None or expires_at is None or retain_until <= expires_at:
            blockers.append(_semantic_blocker(node_id, "retain_until"))
        if not _evidence_uri_is_durable(str(source.get("archive_uri") or "")):
            blockers.append(_semantic_blocker(node_id, "archive_uri"))
    return blockers


def _validate_source_payload(
    *,
    node_id: str,
    scope: str,
    requirement: Mapping[str, Any],
    validation: Mapping[str, Any],
    source: Mapping[str, Any],
    repository_sha: str,
    image_digest: str,
    now: datetime,
) -> tuple[str, datetime | None, datetime | None, list[str]]:
    blockers: list[str] = []
    if source.get("schema_version") != validation["source_schema_version"]:
        blockers.append(_semantic_blocker(node_id, "schema_version"))
    if source.get("evidence_id") != node_id:
        blockers.append(_semantic_blocker(node_id, "evidence_id"))
    if source.get("evidence_schema_version") != requirement["evidence_schema_version"]:
        blockers.append(_semantic_blocker(node_id, "evidence_schema_version"))
    if source.get("blockers") != []:
        blockers.append(_semantic_blocker(node_id, "blockers"))
    status = str(source.get(str(validation["source_status_field"])) or "").strip()
    if status == "automation_failed":
        blockers.append(f"automation_failed:{node_id}")
    accepted_statuses = set(requirement["accepted_statuses_by_scope"][scope])
    if status not in accepted_statuses:
        blockers.append(_special_status_blocker(node_id, status))
    actual_sha = str(source.get("repository_sha") or "").strip().lower()
    if actual_sha != repository_sha:
        blockers.append(f"wrong_repository_sha:{node_id}:{actual_sha or 'missing'}")
    actual_image = str(source.get("image_digest") or "").strip().lower()
    if actual_image != image_digest:
        blockers.append(f"wrong_image_digest:{node_id}:{actual_image or 'missing'}")
    generated_at = _parse_time(source.get("generated_at"))
    expires_at = _parse_time(source.get("expires_at"))
    if generated_at is None or expires_at is None or expires_at <= generated_at:
        blockers.append(f"malformed_evidence:{node_id}:source_validity_interval")
    else:
        max_age = timedelta(seconds=int(requirement["max_age_seconds"]))
        if generated_at > now + timedelta(minutes=5):
            blockers.append(f"future_evidence:{node_id}")
        if expires_at <= now or now - generated_at > max_age:
            blockers.append(f"stale_evidence:{node_id}")
        if expires_at - generated_at > max_age:
            blockers.append(f"malformed_evidence:{node_id}:source_expiry_exceeds_max_age")
    blockers.extend(
        _validate_semantic_source(
            node_id=node_id,
            validator=str(validation["semantic_validator"]),
            source=source,
            repository_sha=repository_sha,
            image_digest=image_digest,
            expires_at=expires_at,
        )
    )
    return status, generated_at, expires_at, blockers


def _verify_source_attestation(
    *,
    node_id: str,
    validation: Mapping[str, Any],
    authorities: Mapping[str, Any],
    attestation: object,
    expected_statement: Mapping[str, Any] | None,
) -> list[str]:
    prefix = f"untrusted_attestation:{node_id}"
    if not isinstance(attestation, Mapping):
        return [f"{prefix}:missing"]
    authority_id = str(validation["trusted_attestation_authority_id"])
    authority = authorities.get(authority_id)
    if not isinstance(authority, Mapping):
        return [f"{prefix}:authority_missing"]
    pin = authority.get("public_key_sha256")
    if not isinstance(pin, str) or KEY_FINGERPRINT_PATTERN.fullmatch(pin) is None:
        return [f"{prefix}:authority_unconfigured"]
    blockers: list[str] = []
    if attestation.get("schema_version") != SOURCE_ATTESTATION_SCHEMA_VERSION:
        blockers.append(f"{prefix}:schema")
    if attestation.get("algorithm") != "ed25519":
        blockers.append(f"{prefix}:algorithm")
    if attestation.get("authority_id") != authority_id:
        blockers.append(f"{prefix}:authority_id")
    if expected_statement is None or attestation.get("statement") != expected_statement:
        blockers.append(f"{prefix}:statement")
    try:
        public_key = base64.b64decode(
            str(attestation.get("public_key_base64") or ""), validate=True
        )
        signature = base64.b64decode(str(attestation.get("signature_base64") or ""), validate=True)
    except (ValueError, TypeError):
        blockers.append(f"{prefix}:encoding")
        return blockers
    if len(public_key) != 32 or len(signature) != 64:
        blockers.append(f"{prefix}:key_or_signature_length")
        return blockers
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if fingerprint != pin:
        blockers.append(f"{prefix}:public_key_pin")
        return blockers
    if expected_statement is not None:
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                signature,
                _canonical_json_bytes(expected_statement),
            )
        except (InvalidSignature, ValueError):
            blockers.append(f"{prefix}:signature")
    return blockers


def _validate_node(
    *,
    node_id: str,
    scope: str,
    requirement: Mapping[str, Any],
    validation: Mapping[str, Any],
    authorities: Mapping[str, Any],
    evidence_path: Path,
    evidence_dir: Path,
    repository_sha: str,
    image_digest: str,
    now: datetime,
) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {
        "id": node_id,
        "path": evidence_path.name,
        "outcome": "blocked",
        "accepted_statuses": list(requirement["accepted_statuses_by_scope"][scope]),
        "expected_evidence_schema_version": requirement["evidence_schema_version"],
        "max_age_seconds": requirement["max_age_seconds"],
    }
    if evidence_path.is_symlink():
        blocker = f"malformed_evidence:{node_id}:symlink"
        result["blockers"] = [blocker]
        return result, [blocker]
    if not evidence_path.is_file():
        blocker = _special_missing_blocker(node_id)
        result["blockers"] = [blocker]
        return result, [blocker]
    envelope = _load_mapping(evidence_path)
    if envelope is None:
        blocker = f"malformed_evidence:{node_id}:invalid_json_or_oversized"
        result["blockers"] = [blocker]
        return result, [blocker]

    blockers: list[str] = []
    if envelope.get("schema_version") != EVIDENCE_ENVELOPE_SCHEMA_VERSION:
        blockers.append(f"malformed_evidence:{node_id}:envelope_schema")
    if envelope.get("evidence_id") != node_id:
        blockers.append(f"malformed_evidence:{node_id}:evidence_id")
    if envelope.get("evidence_schema_version") != requirement["evidence_schema_version"]:
        blockers.append(f"malformed_evidence:{node_id}:payload_schema")
    evidence_uri = str(envelope.get("evidence_uri") or "").strip()
    if not _evidence_uri_is_durable(evidence_uri):
        blockers.append(f"malformed_evidence:{node_id}:evidence_uri")

    source_path, source_path_error = _resolve_source_artifact(
        evidence_dir=evidence_dir,
        raw_path=envelope.get("source_artifact_path"),
    )
    source: dict[str, Any] | None = None
    source_digest = ""
    source_claims_digest = ""
    status = ""
    generated_at: datetime | None = None
    expires_at: datetime | None = None
    if source_path_error is not None or source_path is None:
        blockers.append(f"source_artifact:{node_id}:{source_path_error}")
    else:
        relative = source_path.relative_to(evidence_dir).as_posix()
        result["source_artifact_path"] = relative
        try:
            source_digest = _sha256(source_path)
        except OSError:
            blockers.append(f"source_artifact:{node_id}:unreadable")
        source = _load_mapping(source_path)
        if source is None:
            blockers.append(f"source_artifact:{node_id}:invalid_json_or_oversized")
        else:
            try:
                source_claims_digest = _mapping_digest(source)
            except (TypeError, ValueError):
                blockers.append(f"source_artifact:{node_id}:noncanonical_json_values")
            status, generated_at, expires_at, source_blockers = _validate_source_payload(
                node_id=node_id,
                scope=scope,
                requirement=requirement,
                validation=validation,
                source=source,
                repository_sha=repository_sha,
                image_digest=image_digest,
                now=now,
            )
            blockers.extend(source_blockers)
            result["source_claims"] = source
    result["status"] = status or None
    result["evidence_generated_at"] = _format_time(generated_at) if generated_at else None
    result["evidence_expires_at"] = _format_time(expires_at) if expires_at else None
    result["source_artifact_digest"] = source_digest or None
    result["source_claims_digest"] = source_claims_digest or None
    envelope_digest = str(envelope.get("source_artifact_digest") or "")
    if ARTIFACT_DIGEST_PATTERN.fullmatch(envelope_digest) is None:
        blockers.append(f"malformed_evidence:{node_id}:source_artifact_digest")
    elif not source_digest or envelope_digest != source_digest:
        blockers.append(f"source_artifact:{node_id}:digest_mismatch")

    source_sha = str(source.get("repository_sha") or "").lower() if source else ""
    source_image = str(source.get("image_digest") or "").lower() if source else ""
    if str(envelope.get("repository_sha") or "").lower() != source_sha:
        blockers.append(f"malformed_evidence:{node_id}:envelope_source_sha_mismatch")
    if str(envelope.get("image_digest") or "").lower() != source_image:
        blockers.append(f"malformed_evidence:{node_id}:envelope_source_image_mismatch")
    if str(envelope.get("status") or "") != status:
        blockers.append(f"malformed_evidence:{node_id}:envelope_source_status_mismatch")
    if _parse_time(envelope.get("generated_at")) != generated_at:
        blockers.append(f"malformed_evidence:{node_id}:envelope_source_generated_at_mismatch")
    if _parse_time(envelope.get("expires_at")) != expires_at:
        blockers.append(f"malformed_evidence:{node_id}:envelope_source_expires_at_mismatch")

    expected_statement = None
    authority_id = str(validation["trusted_attestation_authority_id"])
    if (
        source_digest
        and source_claims_digest
        and generated_at is not None
        and expires_at is not None
    ):
        expected_statement = build_release_evidence_source_attestation_statement(
            authority_id=authority_id,
            node_id=node_id,
            source_artifact_digest=source_digest,
            source_claims_digest=source_claims_digest,
            repository_sha=repository_sha,
            image_digest=image_digest,
            generated_at=generated_at,
            expires_at=expires_at,
        )
    attestation = envelope.get("source_verifier_attestation")
    blockers.extend(
        _verify_source_attestation(
            node_id=node_id,
            validation=validation,
            authorities=authorities,
            attestation=attestation,
            expected_statement=expected_statement,
        )
    )
    result["source_verifier_attestation"] = (
        dict(attestation) if isinstance(attestation, Mapping) else None
    )
    blockers = sorted(
        set(blockers),
        key=lambda blocker: (_BLOCKER_PRIORITY[_blocker_class(blocker)], blocker),
    )
    result["blockers"] = blockers
    result["source_binding_verified"] = not blockers
    if not blockers:
        result["outcome"] = "accepted"
    return result, blockers


def evaluate_release_evidence_graph(
    *,
    scope: str,
    repository_sha: str,
    image_digest: str,
    evidence_dir: Path,
    requirements_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate one release scope and return a cryptographically bound graph."""

    evaluated_at = (now or _utc_now()).astimezone(timezone.utc)
    normalized_scope = scope.strip().upper()
    normalized_sha = repository_sha.strip().lower()
    normalized_image = image_digest.strip().lower()
    binding_blockers: list[str] = []
    if GIT_SHA_PATTERN.fullmatch(normalized_sha) is None:
        binding_blockers.append("malformed_evidence:release_binding:repository_sha")
    if IMAGE_DIGEST_PATTERN.fullmatch(normalized_image) is None:
        binding_blockers.append("malformed_evidence:release_binding:image_digest")
    try:
        requirements = load_release_evidence_requirements(requirements_path)
    except ValueError as exc:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "generated_at": _format_time(evaluated_at),
            "scope": normalized_scope,
            "status": "blocked",
            "release_binding": {
                "repository_sha": normalized_sha or None,
                "image_digest": normalized_image or None,
            },
            "nodes": [],
            "blockers": [f"malformed_evidence:requirements:{exc}"],
            "exit_code": 1,
        }
    raw_scopes = requirements["scopes"]
    if normalized_scope not in raw_scopes:
        binding_blockers.append(
            f"malformed_evidence:release_binding:scope:{normalized_scope or 'missing'}"
        )
        required_ids: list[str] = []
    else:
        required_ids = list(raw_scopes[normalized_scope])
    nodes: list[dict[str, Any]] = []
    blockers = list(binding_blockers)
    if not binding_blockers:
        for node_id in required_ids:
            node, node_blockers = _validate_node(
                node_id=node_id,
                scope=normalized_scope,
                requirement=requirements["nodes"][node_id],
                validation=requirements["node_validation"][node_id],
                authorities=requirements["attestation_authorities"],
                evidence_path=evidence_dir / f"{node_id}.json",
                evidence_dir=evidence_dir,
                repository_sha=normalized_sha,
                image_digest=normalized_image,
                now=evaluated_at,
            )
            nodes.append(node)
            blockers.extend(node_blockers)
    blockers = sorted(
        set(blockers),
        key=lambda blocker: (_BLOCKER_PRIORITY[_blocker_class(blocker)], blocker),
    )
    accepted = not blockers and len(nodes) == len(required_ids)
    fresh_until_values = [
        parsed
        for node in nodes
        if node.get("outcome") == "accepted"
        for parsed in [_parse_time(node.get("evidence_expires_at"))]
        if parsed is not None
    ]
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "requirements_schema_version": requirements["schema_version"],
        "generated_at": _format_time(evaluated_at),
        "fresh_until": _format_time(min(fresh_until_values)) if fresh_until_values else None,
        "scope": normalized_scope,
        "status": "passed" if accepted else "blocked",
        "release_binding": {
            "repository_sha": normalized_sha or None,
            "image_digest": normalized_image or None,
        },
        "required_node_ids": required_ids,
        "nodes": nodes,
        "blockers": blockers,
        "exit_code": 0 if accepted else 1,
        "claim_boundary": {
            "manual_closeout_cannot_override_blockers": True,
            "scope_specific_acceptance_only": True,
            "sim_only_does_not_require_physical_robot_evidence": True,
            "evidence_is_release_sha_and_image_digest_bound": True,
            "envelope_or_uri_is_never_source_proof": True,
            "every_node_requires_trusted_source_attestation": True,
            "source_semantics_are_revalidated_by_node_type": True,
        },
    }


def _persisted_node_binding_blockers(
    *,
    node: Mapping[str, Any],
    node_id: str,
    scope: str,
    requirement: Mapping[str, Any],
    validation: Mapping[str, Any],
    authorities: Mapping[str, Any],
    repository_sha: str,
    image_digest: str,
    now: datetime,
) -> list[str]:
    source = node.get("source_claims")
    if not isinstance(source, Mapping):
        return [f"release_evidence_graph_node_binding_invalid:{node_id}:source_claims"]
    status, generated_at, expires_at, source_blockers = _validate_source_payload(
        node_id=node_id,
        scope=scope,
        requirement=requirement,
        validation=validation,
        source=source,
        repository_sha=repository_sha,
        image_digest=image_digest,
        now=now,
    )
    blockers = list(source_blockers)
    if node.get("status") != status:
        blockers.append("persisted_status_mismatch")
    try:
        claims_digest = _mapping_digest(source)
    except (TypeError, ValueError):
        blockers.append("persisted_source_claims_noncanonical")
        claims_digest = ""
    if node.get("source_claims_digest") != claims_digest:
        blockers.append("persisted_source_claims_digest_mismatch")
    source_digest = str(node.get("source_artifact_digest") or "")
    if ARTIFACT_DIGEST_PATTERN.fullmatch(source_digest) is None:
        blockers.append("persisted_source_artifact_digest_invalid")
    expected_statement = None
    if claims_digest and generated_at is not None and expires_at is not None:
        expected_statement = build_release_evidence_source_attestation_statement(
            authority_id=str(validation["trusted_attestation_authority_id"]),
            node_id=node_id,
            source_artifact_digest=source_digest,
            source_claims_digest=claims_digest,
            repository_sha=repository_sha,
            image_digest=image_digest,
            generated_at=generated_at,
            expires_at=expires_at,
        )
    blockers.extend(
        _verify_source_attestation(
            node_id=node_id,
            validation=validation,
            authorities=authorities,
            attestation=node.get("source_verifier_attestation"),
            expected_statement=expected_statement,
        )
    )
    if node.get("source_binding_verified") is not True:
        blockers.append("persisted_source_binding_not_verified")
    return [
        f"release_evidence_graph_node_binding_invalid:{node_id}:{blocker}"
        for blocker in sorted(set(blockers))
    ]


def validate_release_evidence_graph_result(
    graph: Mapping[str, Any] | None,
    *,
    expected_scope: str,
    expected_repository_sha: str,
    requirements_path: Path,
    expected_image_digest: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Revalidate a persisted graph, including every signed source binding."""

    if not isinstance(graph, Mapping):
        return ["release_evidence_graph_missing"]
    blockers: list[str] = []
    if graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
        blockers.append("release_evidence_graph_schema_mismatch")
    normalized_scope = expected_scope.strip().upper()
    if graph.get("scope") != normalized_scope:
        blockers.append("release_evidence_graph_scope_mismatch")
    expected_ids: list[str] = []
    requirements: dict[str, Any] | None = None
    try:
        requirements = load_release_evidence_requirements(requirements_path)
    except ValueError:
        blockers.append("release_evidence_graph_requirements_malformed")
    else:
        if graph.get("requirements_schema_version") != requirements["schema_version"]:
            blockers.append("release_evidence_graph_requirements_schema_mismatch")
        if normalized_scope not in requirements["scopes"]:
            blockers.append("release_evidence_graph_expected_scope_unknown")
        else:
            expected_ids = list(requirements["scopes"][normalized_scope])
    expected_sha = expected_repository_sha.strip().lower()
    binding = graph.get("release_binding")
    actual_image = ""
    if not isinstance(binding, Mapping):
        blockers.append("release_evidence_graph_binding_missing")
    else:
        actual_sha = str(binding.get("repository_sha") or "").lower()
        if actual_sha != expected_sha:
            blockers.append(f"release_evidence_graph_wrong_sha:{actual_sha or 'missing'}")
        actual_image = str(binding.get("image_digest") or "").lower()
        if IMAGE_DIGEST_PATTERN.fullmatch(actual_image) is None:
            blockers.append("release_evidence_graph_image_digest_malformed")
        elif expected_image_digest is not None and actual_image != expected_image_digest.lower():
            blockers.append(f"release_evidence_graph_wrong_image:{actual_image}")
    graph_blockers = graph.get("blockers")
    if not isinstance(graph_blockers, list):
        blockers.append("release_evidence_graph_blockers_malformed")
    elif graph_blockers:
        blockers.extend(f"release_evidence:{item}" for item in graph_blockers)
    if graph.get("status") != "passed" or graph.get("exit_code") != 0:
        blockers.append(f"release_evidence_graph_not_passed:{graph.get('status') or 'missing'}")
    required_ids = graph.get("required_node_ids")
    nodes = graph.get("nodes")
    if not isinstance(required_ids, list) or not isinstance(nodes, list):
        blockers.append("release_evidence_graph_nodes_malformed")
    else:
        if required_ids != expected_ids:
            blockers.append("release_evidence_graph_required_nodes_mismatch")
        node_map = {str(node.get("id")): node for node in nodes if isinstance(node, Mapping)}
        if len(node_map) != len(nodes):
            blockers.append("release_evidence_graph_node_ids_duplicate_or_malformed")
        for node_id in expected_ids:
            node = node_map.get(node_id)
            if not isinstance(node, Mapping) or node.get("outcome") != "accepted":
                blockers.append(f"release_evidence_graph_node_not_accepted:{node_id}")
                continue
            if requirements is not None and actual_image:
                blockers.extend(
                    _persisted_node_binding_blockers(
                        node=node,
                        node_id=node_id,
                        scope=normalized_scope,
                        requirement=requirements["nodes"][node_id],
                        validation=requirements["node_validation"][node_id],
                        authorities=requirements["attestation_authorities"],
                        repository_sha=expected_sha,
                        image_digest=actual_image,
                        now=(now or _utc_now()).astimezone(timezone.utc),
                    )
                )
    fresh_until = _parse_time(graph.get("fresh_until"))
    if fresh_until is None:
        blockers.append("release_evidence_graph_fresh_until_malformed")
    elif fresh_until <= (now or _utc_now()).astimezone(timezone.utc):
        blockers.append("release_evidence_graph_stale")
    return sorted(set(blockers))
