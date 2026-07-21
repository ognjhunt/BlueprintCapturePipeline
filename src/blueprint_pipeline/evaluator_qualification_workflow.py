"""Compose evaluator-bounded site, policy, runtime, ranking, and delivery proof."""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .decision_grade_ranking import build_decision_grade_ranking
from .evaluator_evidence_profiles import required_evaluator_evidence_digest_fields
from .evaluator_runtime_evidence import normalize_evaluator_runtime_evidence
from .policy_evaluation_contracts import validate_policy_evaluation_design
from .site_reference_database import validate_evaluation_site_admission


REQUEST_SCHEMA_VERSION = "evaluator_qualification_workflow_request.v1"
RESULT_SCHEMA_VERSION = "evaluator_qualification_workflow.v1"
ALLOCATION_SCHEMA_VERSION = "qualification_provider_allocation.v1"
MEDIA_SCHEMA_VERSION = "qualification_media_validity.v1"
DELIVERY_SCHEMA_VERSION = "qualification_delivery_evidence.v1"
TEARDOWN_SCHEMA_VERSION = "qualification_teardown_evidence.v1"
MAX_TEARDOWN_EVIDENCE_AGE = timedelta(hours=24)
MAX_CLOCK_SKEW = timedelta(minutes=5)
MINIMUM_QUALIFICATION_SITE_COUNT = 4
MAX_REQUEST_BYTES = 256 * 1024 * 1024
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "api_token",
    "access_token",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "password",
    "private_key",
    "raw_response",
    "secret",
    "signed_url",
    "token",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(row, Mapping) for row in value):
        return [], False
    return [dict(row) for row in value], True


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item).strip() for item in value if isinstance(item, str) and item.strip())


def _normalized_digest(value: Any) -> str:
    digest = str(value or "").strip().lower()
    return digest.removeprefix("sha256:") if _SHA256_RE.fullmatch(digest) else ""


def _digest(value: Any) -> bool:
    return bool(_normalized_digest(value))


def _canonical_sha256(value: Any) -> str:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _finite_nonnegative(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _aware_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _sensitive_paths(value: Any, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{path}.{key}"
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
            if any(marker in normalized_key for marker in _SENSITIVE_KEY_MARKERS):
                paths.append(child)
            else:
                paths.extend(_sensitive_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            paths.extend(_sensitive_paths(item, f"{path}[{index}]"))
    return paths


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int] | None:
    policy_id = str(row.get("policy_id") or "").strip()
    site_id = str(row.get("site_id") or "").strip()
    task_id = str(row.get("task_id") or "").strip()
    condition_id = str(row.get("condition_id") or "").strip()
    seed = row.get("seed")
    if (
        not policy_id
        or not site_id
        or not task_id
        or not condition_id
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        return None
    return policy_id, site_id, task_id, condition_id, seed


def _state(status: str, blockers: Sequence[str], **details: Any) -> dict[str, Any]:
    return {"status": status, "blockers": sorted(set(blockers)), **details}


def _validate_release_identity(value: Any) -> tuple[dict[str, Any], list[str]]:
    identity = _mapping(value)
    blockers: list[str] = []
    commit = str(identity.get("source_commit") or "").strip().lower()
    if not _COMMIT_RE.fullmatch(commit):
        blockers.append("release_source_commit_missing_or_invalid")
    for field in (
        "source_archive_sha256",
        "release_manifest_sha256",
        "container_image_sha256",
        "model_set_manifest_sha256",
        "data_split_manifest_sha256",
    ):
        if not _digest(identity.get(field)):
            blockers.append(f"release_binding_digest_missing_or_invalid:{field}")
    model_members, model_members_valid = _strict_rows(identity.get("model_set_members"))
    if not model_members_valid or not model_members:
        blockers.append("release_model_set_members_missing_or_invalid")
    member_ids: set[tuple[str, str]] = set()
    member_digests: set[str] = set()
    for index, member in enumerate(model_members):
        artifact_kind = str(member.get("artifact_kind") or "").strip()
        artifact_id = str(member.get("artifact_id") or "").strip()
        digest = _normalized_digest(member.get("sha256"))
        identity_key = (artifact_kind, artifact_id)
        if not artifact_kind or not artifact_id or identity_key in member_ids:
            blockers.append(f"release_model_set_member_identity_invalid:{index}")
        member_ids.add(identity_key)
        if not digest or digest in member_digests:
            blockers.append(f"release_model_set_member_digest_invalid_or_duplicate:{index}")
        member_digests.add(digest)
    if _normalized_digest(identity.get("model_set_manifest_sha256")) != _normalized_digest(
        _canonical_sha256(model_members)
    ):
        blockers.append("release_model_set_manifest_digest_mismatch")
    return identity, blockers


def _validate_model_set_binding(
    release_identity: Mapping[str, Any], design: Mapping[str, Any]
) -> list[str]:
    members, _ = _strict_rows(release_identity.get("model_set_members"))
    declared = {_normalized_digest(member.get("sha256")) for member in members}
    policies, _ = _strict_rows(design.get("policies"))
    rows, _ = _strict_rows(design.get("rows"))
    expected = {
        _normalized_digest(policy.get(field))
        for policy in policies
        for field in ("policy_sha256", "checkpoint_sha256")
    }
    expected.update(
        _normalized_digest(_mapping(row.get("evaluator_backend")).get("model_artifact_sha256"))
        for row in rows
    )
    expected.discard("")
    declared.discard("")
    return [] if declared == expected else ["release_model_set_does_not_exactly_bind_run_artifacts"]


def _validate_sites(
    manifests: Any,
    *,
    design_rows: Sequence[Mapping[str, Any]],
    release_split_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, payload_valid = _strict_rows(manifests)
    blockers: list[str] = []
    if not payload_valid:
        blockers.append("site_admissions_payload_invalid")
    validations = [validate_evaluation_site_admission(row) for row in rows]
    for index, validation in enumerate(validations):
        blockers.extend(f"site:{index}:{item}" for item in validation["blockers"])
    registered_sites = [str(item.get("site_id") or "").strip() for item in validations]
    if any(not site_id for site_id in registered_sites):
        blockers.append("site_admission_identity_missing")
    if len(registered_sites) != len(set(registered_sites)):
        blockers.append("duplicate_site_admission")
    expected_sites = {str(row.get("site_id") or "").strip() for row in design_rows}
    if set(registered_sites) != expected_sites:
        blockers.append("site_admissions_do_not_exactly_cover_evaluation_design")
    if len(expected_sites) < MINIMUM_QUALIFICATION_SITE_COUNT:
        blockers.append(f"qualification_site_count_lt_{MINIMUM_QUALIFICATION_SITE_COUNT}")
    tasks_by_site: dict[str, set[str]] = {}
    capture_ids: set[str] = set()
    source_bundle_ids: set[str] = set()
    capture_digests: set[str] = set()
    source_bundle_digests: set[str] = set()
    source_manifest_digests: set[str] = set()
    site_dedup_ids: set[str] = set()
    task_dedup_ids: set[str] = set()
    trajectory_dedup_ids: set[str] = set()
    split_manifest_digests: set[str] = set()
    split_partitions: set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = set()
    held_out_design_sites: set[str] = set()
    for manifest in rows:
        identity = _mapping(manifest.get("immutable_source_identity"))
        site_id = str(identity.get("site_id") or "").strip()
        capture_ids.add(str(identity.get("capture_id") or "").strip())
        source_bundle_ids.add(str(identity.get("source_bundle_id") or "").strip())
        capture_digests.add(_normalized_digest(identity.get("capture_sha256")))
        source_bundle_digests.add(_normalized_digest(identity.get("source_bundle_sha256")))
        source_manifest_digests.add(_normalized_digest(identity.get("manifest_sha256")))
        dedup = _mapping(manifest.get("deduplication"))
        site_dedup_ids.add(str(dedup.get("site_dedup_id") or "").strip())
        task_dedup_ids.add(str(dedup.get("task_dedup_id") or "").strip())
        trajectory_dedup_ids.add(str(dedup.get("trajectory_dedup_id") or "").strip())
        splits = _mapping(manifest.get("frozen_splits"))
        split_manifest_digests.add(_normalized_digest(splits.get("split_manifest_sha256")))
        partition = tuple(
            tuple(sorted(_strings(splits.get(field))))
            for field in ("train_sites", "dev_sites", "held_out_sites")
        )
        split_partitions.add(partition)
        if site_id in set(partition[2]):
            held_out_design_sites.add(site_id)
        task_rows, task_payload_valid = _strict_rows(manifest.get("task_contracts"))
        tasks_by_site[site_id] = {
            str(task.get("task_id") or "").strip() for task in task_rows if task_payload_valid
        }
    for row in design_rows:
        site_id = str(row.get("site_id") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        if task_id not in tasks_by_site.get(site_id, set()):
            blockers.append(f"evaluation_task_not_admitted_for_site:{site_id}:{task_id}")
    for label, values in (
        ("capture_id", capture_ids),
        ("source_bundle_id", source_bundle_ids),
        ("capture_sha256", capture_digests),
        ("source_bundle_sha256", source_bundle_digests),
        ("source_manifest_sha256", source_manifest_digests),
        ("site_dedup_id", site_dedup_ids),
        ("task_dedup_id", task_dedup_ids),
        ("trajectory_dedup_id", trajectory_dedup_ids),
    ):
        if len(values) != len(rows):
            blockers.append(f"independent_site_capture_identity_not_unique:{label}")
    if len(split_manifest_digests) != 1 or len(split_partitions) != 1:
        blockers.append("site_admissions_do_not_share_one_frozen_split")
    if split_manifest_digests != {_normalized_digest(release_split_manifest_sha256)}:
        blockers.append("site_split_manifest_does_not_match_release")
    if not held_out_design_sites:
        blockers.append("evaluation_design_has_no_entire_held_out_site")
    return validations, sorted(set(blockers))


_EVALUATOR_STATE_FIELDS = (
    "evaluator_profile_id",
    "fresh_evaluator_model_execution_proven",
    "fresh_evaluator_model_run_steps",
    "action_control_suite_status",
    "authoritative_manifest_status",
    "infrastructure_status",
    "evaluator_outcome_status",
    "criterion_result_status",
    "evaluator_identity_is_compute_provider",
)


def _validate_runtime_rows(
    envelopes: Any,
    *,
    design_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[str], list[str], bool]:
    rows, payload_valid = _strict_rows(envelopes)
    blockers: list[str] = []
    provider_evaluator_separation_proven = payload_valid and bool(rows)
    if not payload_valid:
        blockers.append("runtime_evidence_requests_payload_invalid")
    design_by_key = {_cell_key(row): dict(row) for row in design_rows if _cell_key(row) is not None}
    observed_keys: set[tuple[str, str, str, str, int]] = set()
    summaries: list[dict[str, Any]] = []
    execution_ids: set[str] = set()
    for index, envelope in enumerate(rows):
        key = _cell_key(envelope)
        if key is None:
            blockers.append(f"runtime_evidence_cell_identity_invalid:{index}")
            provider_evaluator_separation_proven = False
            continue
        if key in observed_keys:
            blockers.append(f"duplicate_runtime_evidence_cell:{index}")
        observed_keys.add(key)
        request = _mapping(envelope.get("normalization_request"))
        normalized = normalize_evaluator_runtime_evidence(request)
        blockers.extend(f"runtime:{index}:{item}" for item in normalized["blockers"])
        candidate = normalized.get("evaluator_row")
        candidate_row = dict(candidate) if isinstance(candidate, Mapping) else {}
        if _cell_key(candidate_row) != key:
            blockers.append(f"runtime_normalized_cell_identity_mismatch:{index}")
        expected = design_by_key.get(key, {})
        expected_backend = _mapping(expected.get("evaluator_backend"))
        actual_backend = _mapping(candidate_row.get("evaluator_backend"))
        receipt = _mapping(request.get("runtime_receipt"))
        provider_execution = _mapping(receipt.get("provider_execution"))
        if any(
            value is not False
            for value in (
                expected.get("evaluator_identity_is_compute_provider"),
                candidate_row.get("evaluator_identity_is_compute_provider"),
                expected_backend.get("backend_is_compute_provider"),
                actual_backend.get("backend_is_compute_provider"),
                receipt.get("backend_is_compute_provider"),
                provider_execution.get("provider_is_evaluator_identity"),
            )
        ):
            provider_evaluator_separation_proven = False
        profile_id = str(expected.get("evaluator_profile_id") or "")
        for field in (
            *required_evaluator_evidence_digest_fields(profile_id),
            *_EVALUATOR_STATE_FIELDS,
        ):
            expected_value = expected.get(field)
            actual_value = candidate_row.get(field)
            if field.endswith("sha256"):
                matches = _normalized_digest(actual_value) == _normalized_digest(expected_value)
            else:
                matches = actual_value == expected_value
            if not matches:
                blockers.append(f"runtime_design_binding_mismatch:{index}:{field}")
        for field in ("backend_id", "model_family", "model_version"):
            if (
                str(actual_backend.get(field) or "").strip()
                != str(expected_backend.get(field) or "").strip()
            ):
                blockers.append(f"runtime_design_backend_binding_mismatch:{index}:{field}")
        execution_id = str(provider_execution.get("execution_id") or "").strip()
        if not execution_id:
            blockers.append(f"runtime_provider_execution_identity_missing:{index}")
        else:
            execution_ids.add(execution_id)
        summaries.append(
            {
                "cell": list(key),
                "status": normalized.get("status"),
                "runtime_id": normalized.get("runtime_id"),
                "provider_id": normalized.get("provider_id"),
                "evaluator_backend_id": normalized.get("evaluator_backend_id"),
                "evaluator_model_family": normalized.get("evaluator_model_family"),
                "model_output_id": normalized.get("model_output_id"),
                "blockers": normalized.get("blockers", []),
            }
        )
    if observed_keys != set(design_by_key):
        blockers.append("runtime_evidence_does_not_exactly_cover_evaluation_design")
        provider_evaluator_separation_proven = False
    return (
        summaries,
        execution_ids,
        sorted(set(blockers)),
        provider_evaluator_separation_proven,
    )


def _validate_allocations(
    value: Any,
    *,
    expected_execution_ids: set[str],
    source_commit: str,
    marketed_provider_ids: set[str],
    container_image_sha256: str,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    rows, payload_valid = _strict_rows(value)
    blockers: list[str] = []
    if not payload_valid:
        blockers.append("provider_allocations_payload_invalid")
    execution_ids: set[str] = set()
    allocation_ids: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("schema_version") != ALLOCATION_SCHEMA_VERSION:
            blockers.append(f"provider_allocation_schema_invalid:{index}")
        allocation_id = str(row.get("allocation_id") or "").strip()
        execution_id = str(row.get("execution_id") or "").strip()
        provider_id = str(row.get("provider_id") or "").strip()
        if not allocation_id or allocation_id in allocation_ids:
            blockers.append(f"provider_allocation_identity_missing_or_duplicate:{index}")
        if not execution_id or execution_id in execution_ids:
            blockers.append(f"provider_execution_identity_missing_or_duplicate:{index}")
        if not provider_id:
            blockers.append(f"provider_allocation_provider_missing:{index}")
        elif provider_id not in marketed_provider_ids:
            blockers.append(f"provider_allocation_not_on_marketed_provider:{index}")
        allocation_ids.add(allocation_id)
        execution_ids.add(execution_id)
        if row.get("status") != "allocated":
            blockers.append(f"provider_allocation_status_not_allocated:{index}")
        if str(row.get("source_commit") or "").strip().lower() != source_commit:
            blockers.append(f"provider_allocation_source_commit_mismatch:{index}")
        for field in (
            "allocation_receipt_sha256",
            "allocation_command_evidence_sha256",
            "budget_admission_sha256",
        ):
            if not _digest(row.get(field)):
                blockers.append(f"provider_allocation_digest_invalid:{index}:{field}")
        if _normalized_digest(row.get("container_image_sha256")) != _normalized_digest(
            container_image_sha256
        ):
            blockers.append(f"provider_allocation_container_image_mismatch:{index}")
        summaries.append(
            {
                "allocation_id": allocation_id or None,
                "execution_id": execution_id or None,
                "provider_id": provider_id or None,
                "status": row.get("status"),
            }
        )
    if execution_ids != expected_execution_ids:
        blockers.append("provider_allocations_do_not_exactly_cover_runtime_executions")
    return summaries, sorted(set(blockers)), allocation_ids


def _validate_media(
    value: Any,
    *,
    design_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows, payload_valid = _strict_rows(value)
    blockers: list[str] = []
    if not payload_valid:
        blockers.append("media_validity_payload_invalid")
    design_by_key = {_cell_key(row): dict(row) for row in design_rows if _cell_key(row) is not None}
    observed: set[tuple[str, str, str, str, int]] = set()
    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        key = _cell_key(row)
        if key is None or key in observed:
            blockers.append(f"media_cell_identity_missing_or_duplicate:{index}")
            continue
        observed.add(key)
        if row.get("schema_version") != MEDIA_SCHEMA_VERSION:
            blockers.append(f"media_schema_invalid:{index}")
        if row.get("status") != "valid":
            blockers.append(f"media_status_not_valid:{index}")
        if row.get("model_derived") is not True:
            blockers.append(f"media_model_derivation_not_proven:{index}")
        for field in ("media_sha256", "validation_report_sha256"):
            if not _digest(row.get(field)):
                blockers.append(f"media_digest_invalid:{index}:{field}")
        expected = design_by_key.get(key, {})
        if _normalized_digest(row.get("model_output_sha256")) != _normalized_digest(
            expected.get("model_output_sha256")
        ):
            blockers.append(f"media_model_output_binding_mismatch:{index}")
        summaries.append({"cell": list(key), "status": row.get("status")})
    if observed != set(design_by_key):
        blockers.append("media_validity_does_not_exactly_cover_evaluation_design")
    return summaries, sorted(set(blockers))


def _validate_episode_and_criteria(
    ranking_request: Mapping[str, Any],
    *,
    design_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    results, payload_valid = _strict_rows(ranking_request.get("episode_results"))
    episode_blockers: list[str] = []
    criterion_blockers: list[str] = []
    if not payload_valid:
        return ["episode_results_payload_invalid"], ["criterion_results_unavailable"]
    expected = {_cell_key(row) for row in design_rows if _cell_key(row) is not None}
    observed: set[tuple[str, str, str, str, int]] = set()
    for index, row in enumerate(results):
        key = _cell_key(row)
        if key is None or key in observed:
            episode_blockers.append(f"episode_cell_identity_missing_or_duplicate:{index}")
            continue
        observed.add(key)
        if row.get("full_ordered_episode_evidence") is not True:
            episode_blockers.append(f"full_ordered_episode_not_proven:{index}")
        if row.get("authoritative_manifest_status") != "completed":
            episode_blockers.append(f"authoritative_manifest_not_completed:{index}")
        if row.get("artifact_freshness_status") != "current":
            episode_blockers.append(f"episode_artifact_not_current:{index}")
        if not _digest(row.get("episode_evidence_sha256")):
            episode_blockers.append(f"episode_evidence_digest_invalid:{index}")
        criteria, criteria_valid = _strict_rows(row.get("criterion_results"))
        if not criteria_valid or not criteria:
            criterion_blockers.append(f"criterion_results_missing_or_invalid:{index}")
        elif _normalized_digest(row.get("criterion_result_sha256")) != _normalized_digest(
            _canonical_sha256(criteria)
        ):
            criterion_blockers.append(f"criterion_result_digest_mismatch:{index}")
    if observed != expected:
        episode_blockers.append("episode_results_do_not_exactly_cover_evaluation_design")
    return sorted(set(episode_blockers)), sorted(set(criterion_blockers))


def _validate_delivery(
    value: Any,
    *,
    qualification_id: str,
    source_commit: str,
    ranking_result_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    row = _mapping(value)
    blockers: list[str] = []
    if row.get("schema_version") != DELIVERY_SCHEMA_VERSION:
        blockers.append("delivery_schema_invalid")
    if row.get("status") != "delivered":
        blockers.append("delivery_status_not_delivered")
    if row.get("authenticated") is not True or row.get("authorized") is not True:
        blockers.append("delivery_authentication_or_authorization_not_proven")
    if row.get("tenant_isolation_verified") is not True:
        blockers.append("delivery_tenant_isolation_not_verified")
    if str(row.get("qualification_id") or "").strip() != qualification_id:
        blockers.append("delivery_qualification_identity_mismatch")
    if str(row.get("source_commit") or "").strip().lower() != source_commit:
        blockers.append("delivery_source_commit_mismatch")
    for field in (
        "buyer_scorecard_sha256",
        "delivery_receipt_sha256",
        "ranking_result_sha256",
    ):
        if not _digest(row.get(field)):
            blockers.append(f"delivery_digest_invalid:{field}")
    if _normalized_digest(row.get("ranking_result_sha256")) != _normalized_digest(
        ranking_result_sha256
    ):
        blockers.append("delivery_ranking_result_digest_mismatch")
    return {
        "status": row.get("status"),
        "qualification_id": row.get("qualification_id"),
    }, sorted(set(blockers))


def _validate_teardown(
    value: Any,
    *,
    qualification_id: str,
    source_commit: str,
    allocation_ids: set[str],
    marketed_provider_ids: set[str],
    evaluated_at: datetime | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    row = _mapping(value)
    blockers: list[str] = []
    billing_blockers: list[str] = []
    if row.get("schema_version") != TEARDOWN_SCHEMA_VERSION:
        blockers.append("teardown_schema_invalid")
    if row.get("status") != "proven_zero":
        blockers.append("teardown_status_not_proven_zero")
    if str(row.get("qualification_id") or "").strip() != qualification_id:
        blockers.append("teardown_qualification_identity_mismatch")
    if str(row.get("source_commit") or "").strip().lower() != source_commit:
        blockers.append("teardown_source_commit_mismatch")
    raw_allocation_ids = row.get("exact_attempt_allocation_ids")
    teardown_allocation_ids = (
        {str(item).strip() for item in raw_allocation_ids}
        if isinstance(raw_allocation_ids, Sequence)
        and not isinstance(raw_allocation_ids, (str, bytes, bytearray))
        and all(isinstance(item, str) and item.strip() for item in raw_allocation_ids)
        else set()
    )
    if teardown_allocation_ids != allocation_ids:
        blockers.append("teardown_does_not_exactly_cover_allocations")
    if row.get("exact_attempt_active_resource_count") != 0:
        blockers.append("exact_attempt_resources_not_zero")
    inventories, inventories_valid = _strict_rows(row.get("global_provider_inventory"))
    if not inventories_valid or not inventories:
        blockers.append("global_provider_inventory_missing_or_invalid")
    provider_ids: set[str] = set()
    for index, inventory in enumerate(inventories):
        provider_id = str(inventory.get("provider_id") or "").strip()
        if not provider_id or provider_id in provider_ids:
            blockers.append(f"global_provider_identity_missing_or_duplicate:{index}")
        provider_ids.add(provider_id)
        if inventory.get("active_resource_count") != 0:
            blockers.append(f"global_provider_resources_not_zero:{index}")
        burn = _finite_nonnegative(inventory.get("hourly_allocation_burn_usd"))
        if burn != 0.0:
            blockers.append(f"global_provider_hourly_burn_not_zero:{index}")
        if not _digest(inventory.get("inventory_report_sha256")):
            blockers.append(f"global_provider_inventory_digest_invalid:{index}")
    if provider_ids != marketed_provider_ids:
        blockers.append("global_provider_inventory_does_not_cover_marketed_providers")
    if not _digest(row.get("teardown_report_sha256")):
        blockers.append("teardown_report_digest_invalid")
    observed_at = _aware_datetime(row.get("observed_at"))
    if observed_at is None or evaluated_at is None:
        blockers.append("teardown_observation_time_missing_or_invalid")
    elif observed_at > evaluated_at + MAX_CLOCK_SKEW:
        blockers.append("teardown_observation_time_in_future")
    elif evaluated_at - observed_at > MAX_TEARDOWN_EVIDENCE_AGE:
        blockers.append("teardown_observation_stale")
    billing = _mapping(row.get("billing_reconciliation"))
    if billing.get("status") != "reconciled":
        billing_blockers.append("billing_export_not_reconciled")
    if not _digest(billing.get("billing_export_sha256")):
        billing_blockers.append("billing_export_digest_invalid")
    if _finite_nonnegative(billing.get("total_spend_usd")) is None:
        billing_blockers.append("billing_total_spend_missing_or_invalid")
    summary = {
        "status": row.get("status"),
        "observed_at": row.get("observed_at"),
        "provider_ids": sorted(provider_ids),
        "exact_attempt_active_resource_count": row.get("exact_attempt_active_resource_count"),
        "hourly_allocation_burn_usd": sum(
            float(inventory.get("hourly_allocation_burn_usd") or 0.0)
            for inventory in inventories
            if _finite_nonnegative(inventory.get("hourly_allocation_burn_usd")) is not None
        ),
        "billing_reconciliation_status": billing.get("status"),
        "total_spend_usd": billing.get("total_spend_usd")
        if billing.get("status") == "reconciled"
        else None,
    }
    return summary, sorted(set(blockers)), sorted(set(billing_blockers))


def build_evaluator_qualification_workflow(request: Mapping[str, Any]) -> dict[str, Any]:
    """Independently derive scientific and public-launch qualification states."""

    request_blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        request_blockers.append("qualification_request_schema_missing_or_unsupported")
    qualification_id = str(request.get("qualification_id") or "").strip()
    if not qualification_id:
        request_blockers.append("qualification_identity_missing")
    evaluated_at = _aware_datetime(request.get("evaluated_at"))
    if evaluated_at is None:
        request_blockers.append("qualification_evaluation_time_missing_or_invalid")
    sensitive_paths = _sensitive_paths(request)
    if sensitive_paths:
        request_blockers.append("qualification_request_contains_sensitive_fields")
    if not _canonical_sha256(request):
        request_blockers.append("qualification_request_not_canonical_json")
    release_identity, release_blockers = _validate_release_identity(request.get("release_identity"))
    request_blockers.extend(release_blockers)
    source_commit = str(release_identity.get("source_commit") or "").strip().lower()
    marketed_provider_rows = _strings(request.get("marketed_provider_ids"))
    marketed_provider_ids = set(marketed_provider_rows)
    if not marketed_provider_rows or len(marketed_provider_rows) != len(marketed_provider_ids):
        request_blockers.append("marketed_provider_ids_missing_or_duplicate")

    design = _mapping(request.get("evaluation_design"))
    design_validation = validate_policy_evaluation_design(design)
    request_blockers.extend(_validate_model_set_binding(release_identity, design))
    design_rows, design_rows_valid = _strict_rows(design.get("rows"))
    if not design_rows_valid:
        design_rows = []
    site_validations, site_blockers = _validate_sites(
        request.get("site_admissions"),
        design_rows=design_rows,
        release_split_manifest_sha256=str(release_identity.get("data_split_manifest_sha256") or ""),
    )
    (
        runtime_summaries,
        execution_ids,
        runtime_blockers,
        provider_evaluator_separation_proven,
    ) = _validate_runtime_rows(
        request.get("runtime_evidence_requests"), design_rows=design_rows
    )
    allocation_summaries, allocation_blockers, allocation_ids = _validate_allocations(
        request.get("provider_allocations"),
        expected_execution_ids=execution_ids,
        source_commit=source_commit,
        marketed_provider_ids=marketed_provider_ids,
        container_image_sha256=str(release_identity.get("container_image_sha256") or ""),
    )
    media_summaries, media_blockers = _validate_media(
        request.get("media_validity"), design_rows=design_rows
    )

    ranking_inputs = _mapping(request.get("ranking_inputs"))
    ranking_request = {**ranking_inputs, "evaluation_design": design}
    episode_blockers, criterion_blockers = _validate_episode_and_criteria(
        ranking_request, design_rows=design_rows
    )
    ranking = build_decision_grade_ranking(ranking_request)
    ranking_result_sha256 = _canonical_sha256(ranking)
    delivery_summary, delivery_blockers = _validate_delivery(
        request.get("delivery_evidence"),
        qualification_id=qualification_id,
        source_commit=source_commit,
        ranking_result_sha256=ranking_result_sha256,
    )
    teardown_summary, teardown_blockers, billing_blockers = _validate_teardown(
        request.get("teardown_evidence"),
        qualification_id=qualification_id,
        source_commit=source_commit,
        allocation_ids=allocation_ids,
        marketed_provider_ids=marketed_provider_ids,
        evaluated_at=evaluated_at,
    )

    policy_blockers = list(design_validation["blockers"])
    ranking_blockers = list(ranking["blockers"])
    lifecycle = {
        "request_acceptance": _state(
            "accepted" if not request_blockers else "blocked", request_blockers
        ),
        "site_admission": _state(
            "evaluation_ready" if not site_blockers else "blocked",
            site_blockers,
            site_count=len(site_validations),
        ),
        "policy_registry": _state(
            "decision_grade" if not policy_blockers else "blocked",
            policy_blockers,
            policy_count=design_validation.get("policy_count", 0),
            independent_checkpoint_count=design_validation.get("independent_checkpoint_count", 0),
        ),
        "provider_allocation": _state(
            "proven" if not allocation_blockers else "blocked",
            allocation_blockers,
            allocation_count=len(allocation_summaries),
        ),
        "model_execution": _state(
            "proven" if not runtime_blockers else "blocked",
            runtime_blockers,
            runtime_row_count=len(runtime_summaries),
        ),
        "episode_artifacts_assembled": _state(
            "proven" if not episode_blockers else "blocked", episode_blockers
        ),
        "media_validity": _state(
            "valid" if not media_blockers else "blocked",
            media_blockers,
            media_row_count=len(media_summaries),
        ),
        "evaluator_validity": _state(
            "validated" if not runtime_blockers else "blocked", runtime_blockers
        ),
        "criterion_result": _state(
            "valid" if not criterion_blockers else "blocked", criterion_blockers
        ),
        "rank_result": _state(
            "decision_grade" if not ranking_blockers else "blocked", ranking_blockers
        ),
        "delivery": _state("delivered" if not delivery_blockers else "blocked", delivery_blockers),
        "teardown": _state(
            "proven_zero" if not teardown_blockers else "blocked", teardown_blockers
        ),
        "billing_reconciliation": _state(
            "reconciled" if not billing_blockers else "blocked", billing_blockers
        ),
    }
    scientific_state_names = (
        "request_acceptance",
        "site_admission",
        "policy_registry",
        "model_execution",
        "episode_artifacts_assembled",
        "media_validity",
        "evaluator_validity",
        "criterion_result",
        "rank_result",
    )
    public_state_names = (
        *scientific_state_names,
        "provider_allocation",
        "delivery",
        "teardown",
        "billing_reconciliation",
    )
    scientific_ready = all(
        lifecycle[name]["status"] != "blocked" for name in scientific_state_names
    )
    public_ready = all(lifecycle[name]["status"] != "blocked" for name in public_state_names)
    all_blockers = sorted(
        {f"{name}:{blocker}" for name, state in lifecycle.items() for blocker in state["blockers"]}
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "public_launch_qualified" if public_ready else "blocked",
        "scientific_qualification_status": "decision_grade" if scientific_ready else "blocked",
        "public_launch_qualification_status": "qualified" if public_ready else "blocked",
        "qualification_id": qualification_id or None,
        "evaluated_at": request.get("evaluated_at"),
        "source_commit": source_commit or None,
        "request_sha256": None if sensitive_paths else _canonical_sha256(request),
        "lifecycle": lifecycle,
        "matrix": {
            "policy_count": design_validation.get("policy_count", 0),
            "site_count": len({str(row.get("site_id") or "") for row in design_rows}),
            "task_count": len({str(row.get("task_id") or "") for row in design_rows}),
            "condition_count": len({str(row.get("condition_id") or "") for row in design_rows}),
            "matched_cell_count_per_policy": design_validation.get(
                "matched_cell_count_per_policy", 0
            ),
            "minimum_matched_replicates_per_policy_condition": design_validation.get(
                "minimum_matched_replicates_per_policy_condition"
            ),
        },
        "model_provider_proof": {
            "runtime_rows": runtime_summaries,
            "provider_allocations": allocation_summaries,
            "providers_are_not_evaluator_identities": provider_evaluator_separation_proven,
        },
        "ranking": ranking,
        "delivery": delivery_summary,
        "teardown": teardown_summary,
        "blockers": all_blockers,
        "sensitive_paths_omitted": len(sensitive_paths),
        "claim_boundary": {
            "simulator_ranking_is_not_physical_robot_performance": True,
            "generated_media_is_not_criterion_or_ranking_success": True,
            "authoritative_manifest_controls_episode_completion": True,
            "provider_allocation_is_not_model_execution": True,
            "model_execution_is_not_evaluator_validity": True,
            "ranking_is_not_delivery_or_teardown": True,
            "allocation_inventory_is_not_billing_reconciliation": True,
            "oscar_sc3_cosmos_and_future_models_are_replaceable_backends": True,
            "paper_metrics_are_not_blueprint_results": True,
            "correlation_status": ranking.get("correlation_status", "correlation_not_measured"),
            "physical_robot_claim_upgrade_required": True,
        },
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("qualification_request_must_be_regular_file")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("qualification_request_exceeds_size_limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("qualification_request_must_be_json_object")
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            with suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_evaluator_qualification_workflow(
        _read_json_object(args.request.expanduser().resolve())
    )
    output = args.output.expanduser().resolve()
    _write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "schema_version": "evaluator_qualification_workflow_cli_result.v1",
                "status": result["status"],
                "scientific_qualification_status": result["scientific_qualification_status"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "public_launch_qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
