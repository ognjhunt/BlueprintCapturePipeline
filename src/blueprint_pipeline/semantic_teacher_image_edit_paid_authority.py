"""Single-use file-backed authority for one semantic-teacher provider bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .semantic_teacher_image_edit_bundle import BUNDLE_RECEIPT_SCHEMA_VERSION
from .spend_authority_consumption_root import prepare_consumption_root


AUTHORITY_SCHEMA_VERSION = "semantic_teacher_image_edit_paid_authority.v1"
CONSUMPTION_SCHEMA_VERSION = "semantic_teacher_image_edit_authority_consumption.v1"
PRIOR_SPEND_RECONCILIATION_SCHEMA_VERSION = "adp_same_goal_spend_reconciliation.v1"
PRIOR_SPEND_ENTRY_SCHEMA_VERSION = "adp_same_goal_spend_entry.v1"
MAX_ATTEMPT_SPEND_USD = 5.0
MAX_TTL_SECONDS = 3_600
_DIGEST_FIELDS = (
    "receipt_digest",
    "result_digest",
    "execution_result_digest",
    "execution_digest",
    "provider_zero_digest",
    "provider_zero_receipt_digest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bound_record(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (float(value) > 0 if positive else float(value) >= 0)
    )


def _validated_digest(value: Mapping[str, Any], field: Any) -> str:
    if field not in _DIGEST_FIELDS:
        raise ValueError("semantic_teacher_prior_spend_digest_field_invalid")
    digest = value.get(str(field))
    if digest != canonical_digest(value, digest_field=str(field)):
        raise ValueError("semantic_teacher_prior_spend_digest_invalid")
    return str(digest)


def _json_path(value: Any, path: Any) -> Any:
    if not isinstance(path, list) or not path:
        raise ValueError("semantic_teacher_prior_spend_binding_path_invalid")
    current = value
    for component in path:
        if isinstance(component, str) and isinstance(current, Mapping):
            if component not in current:
                raise ValueError(
                    "semantic_teacher_prior_spend_binding_path_invalid"
                )
            current = current[component]
        elif (
            isinstance(component, int)
            and not isinstance(component, bool)
            and isinstance(current, list)
            and 0 <= component < len(current)
        ):
            current = current[component]
        else:
            raise ValueError("semantic_teacher_prior_spend_binding_path_invalid")
    return current


def _validate_prior_spend_reconciliation(
    path: Path, *, expected_total_cost_usd: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read(path, code="semantic_teacher_prior_spend_reconciliation_invalid")
    entries = value.get("entries")
    if (
        value.get("schema_version") != PRIOR_SPEND_RECONCILIATION_SCHEMA_VERSION
        or value.get("status")
        != "all_same_goal_paid_attempts_terminal_and_provider_zero"
        or value.get("goal_id") != "arm-decision-proof-v1"
        or not isinstance(entries, list)
        or not entries
        or value.get("entry_count") != len(entries)
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not _finite(value.get("total_cost_usd"))
        or float(value["total_cost_usd"]) != float(expected_total_cost_usd)
    ):
        raise ValueError("semantic_teacher_prior_spend_reconciliation_invalid")
    seen_attempt_ids: set[str] = set()
    costs: list[float] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("semantic_teacher_prior_spend_entry_invalid")
        attempt_id = str(entry.get("attempt_id") or "")
        cost = entry.get("cost_usd")
        historical_gap = entry.get("evidence_kind") == "historical_evidence_gap"
        if (
            entry.get("schema_version") != PRIOR_SPEND_ENTRY_SCHEMA_VERSION
            or entry.get("goal_id") != "arm-decision-proof-v1"
            or not attempt_id
            or attempt_id in seen_attempt_ids
            or not str(entry.get("lane") or "").strip()
            or not _finite(cost)
            or entry.get("continuing_spend_from_this_run") is not False
            or entry.get("provider_zero_confirmed") is not True
            or (
                historical_gap
                and (
                    entry.get("lane") != "content_agents_pan_chera_v8r2"
                    or float(cost or 0) != 0.387839
                    or entry.get("bundle_sha256") is not None
                    or entry.get("bundle_digest_available") is not False
                    or entry.get("typed_abstention")
                    != "historical_bundle_bytes_not_retained"
                )
            )
            or entry.get("entry_digest")
            != canonical_digest(entry, digest_field="entry_digest")
        ):
            raise ValueError("semantic_teacher_prior_spend_entry_invalid")
        seen_attempt_ids.add(attempt_id)
        sources = entry.get("source_receipts")
        bindings = entry.get("bindings")
        if (
            not isinstance(sources, list)
            or not sources
            or not isinstance(bindings, list)
            or not bindings
        ):
            raise ValueError("semantic_teacher_prior_spend_entry_invalid")
        reopened: dict[str, dict[str, Any]] = {}
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("semantic_teacher_prior_spend_source_invalid")
            role = str(source.get("role") or "")
            source_path = _bound_record(
                source.get("record"), code="prior_source_receipt_unbound"
            )
            source_value = _read(source_path, code="prior_source_receipt_invalid")
            digest_field = source.get("digest_field")
            if (
                not role
                or role in reopened
                or source_value.get("schema_version") != source.get("schema_version")
            ):
                raise ValueError("semantic_teacher_prior_spend_source_invalid")
            if digest_field is None:
                if (
                    source.get("legacy_digest_gap")
                    != "exact_source_bytes_sha256_bound_no_canonical_digest"
                ):
                    raise ValueError("semantic_teacher_prior_spend_source_invalid")
            else:
                digest = _validated_digest(source_value, digest_field)
                if digest != (source.get("record") or {}).get("receipt_digest"):
                    raise ValueError("semantic_teacher_prior_spend_source_invalid")
            reopened[role] = source_value
        observed_kinds: set[str] = set()
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            kind = str(binding.get("kind") or "")
            source_value = reopened.get(str(binding.get("source_role") or ""))
            if source_value is None or kind in observed_kinds:
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            observed = _json_path(source_value, binding.get("json_path"))
            expected = binding.get("expected_value")
            if observed != expected:
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            observed_kinds.add(kind)
            if kind == "cost_usd" and (
                not _finite(observed) or float(observed) != float(cost)
            ):
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            if kind == "provider_zero" and observed not in {
                True,
                "PASS",
                "provider_zero",
                "provider_zero_confirmed",
            }:
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            if kind == "continuing_spend" and observed is not False:
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            if kind == "instance_id" and not str(observed or "").strip():
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            if kind == "authority_digest" and observed != entry.get("authority_digest"):
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
            if kind == "bundle_sha256" and observed != entry.get("bundle_sha256"):
                raise ValueError("semantic_teacher_prior_spend_binding_invalid")
        required_kinds = {
            "cost_usd",
            "provider_zero",
            "continuing_spend",
            "instance_id",
            "authority_digest",
            "bundle_sha256",
        }
        if historical_gap:
            required_kinds.remove("bundle_sha256")
        if observed_kinds != required_kinds:
            raise ValueError("semantic_teacher_prior_spend_binding_incomplete")
        costs.append(float(cost))
    if not math.isclose(
        math.fsum(costs),
        float(value["total_cost_usd"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("semantic_teacher_prior_spend_total_invalid")
    return value, {
        **_record(path),
        "receipt_digest": value["receipt_digest"],
        "entry_count": len(entries),
        "total_cost_usd": value["total_cost_usd"],
    }


def materialize_semantic_teacher_image_edit_paid_authority(
    *,
    bundle_path: str | Path,
    bundle_receipt_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    source_commit_sha: str,
    backend_entry_digest: str,
    task_count: int,
    camera_count: int,
    maximum_hourly_rate_usd: float,
    hard_total_spend_cap_usd: float,
    hard_ttl_seconds: int,
    aggregate_goal_spend_before_attempt_usd: float,
    aggregate_goal_spend_cap_usd: float,
    output_path: str | Path,
    prior_spend_reconciliation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal exact bundle bytes and aggregate budget into one allocation grant."""

    bundle = Path(bundle_path).expanduser().resolve()
    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    receipt = _read(receipt_path, code="semantic_teacher_authority_receipt_invalid")
    maximum_cost_per_request_usd = receipt.get("maximum_cost_per_request_usd")
    maximum_editor_request_cost_usd = (
        float(maximum_cost_per_request_usd) * camera_count
        if _finite(maximum_cost_per_request_usd, positive=True)
        else math.inf
    )
    maximum_compute_cost_usd = (
        maximum_hourly_rate_usd * hard_ttl_seconds / 3600
        if _finite(maximum_hourly_rate_usd, positive=True)
        and isinstance(hard_ttl_seconds, int)
        and not isinstance(hard_ttl_seconds, bool)
        else math.inf
    )
    prior_record: dict[str, Any] | None = None
    if prior_spend_reconciliation_path is not None:
        prior_path = Path(prior_spend_reconciliation_path).expanduser().resolve()
        _prior, prior_record = _validate_prior_spend_reconciliation(
            prior_path,
            expected_total_cost_usd=aggregate_goal_spend_before_attempt_usd,
        )
    bundle_digest = _sha256(bundle) if bundle.is_file() else ""
    configuration_valid = (
        receipt.get("schema_version") == BUNDLE_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "completed_no_upload_no_inference"
        and receipt.get("receipt_digest")
        == canonical_digest(receipt, digest_field="receipt_digest")
        and (receipt.get("bundle") or {}).get("sha256") == bundle_digest
        and (receipt.get("bundle") or {}).get("size_bytes") == bundle.stat().st_size
        and receipt.get("source_commit_sha") == source_commit_sha
        and receipt.get("backend_entry_digest") == backend_entry_digest
        and re.fullmatch(
            r"\S+@sha256:[0-9a-f]{64}", str(receipt.get("worker_image_digest") or "")
        )
        is not None
        and receipt.get("runtime_image_identity") == receipt.get("worker_image_digest")
        and str(receipt.get("worker_source_sha256") or "").startswith("sha256:")
        and str(receipt.get("model_snapshot") or "").strip()
        and str(receipt.get("pricing_binding_digest") or "").startswith("sha256:")
        and _finite(maximum_cost_per_request_usd, positive=True)
        and receipt.get("task_count") == task_count
        and receipt.get("camera_count") == camera_count
        and receipt.get("provider_mutations_performed") == 0
        and receipt.get("secret_values_stored") is False
        and receipt.get("raw_nonredistributable_source_bytes_included") is False
        and isinstance(receipt.get("rehearsal"), Mapping)
        and receipt["rehearsal"].get("status") == "passed"
        and receipt["rehearsal"].get("token_lookup_performed") is False
        and receipt["rehearsal"].get("provider_mutations_performed") == 0
        and len(source_commit_sha) == 40
        and backend_entry_digest.startswith("sha256:")
        and len(backend_entry_digest) == 71
        and 1 <= task_count <= 5
        and camera_count >= task_count
        and authorization_reference.strip()
        and authorized_by.strip()
        and authorized_on.strip()
        and _finite(maximum_hourly_rate_usd, positive=True)
        and _finite(hard_total_spend_cap_usd, positive=True)
        and maximum_hourly_rate_usd <= hard_total_spend_cap_usd
        and hard_total_spend_cap_usd <= MAX_ATTEMPT_SPEND_USD
        and isinstance(hard_ttl_seconds, int)
        and not isinstance(hard_ttl_seconds, bool)
        and 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
        and maximum_hourly_rate_usd * hard_ttl_seconds / 3600
        <= hard_total_spend_cap_usd
        and maximum_editor_request_cost_usd + maximum_compute_cost_usd
        <= hard_total_spend_cap_usd
        and _finite(aggregate_goal_spend_before_attempt_usd)
        and _finite(aggregate_goal_spend_cap_usd, positive=True)
        and aggregate_goal_spend_before_attempt_usd + hard_total_spend_cap_usd
        <= aggregate_goal_spend_cap_usd
        and (
            aggregate_goal_spend_before_attempt_usd == 0
            or prior_record is not None
        )
    )
    if not configuration_valid:
        raise ValueError("semantic_teacher_paid_authority_configuration_invalid")
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_semantic_teacher_image_edit",
        "paid_execution_authorized": True,
        "source_commit_sha": source_commit_sha,
        "bundle": _record(bundle),
        "bundle_receipt": _record(receipt_path),
        "bundle_receipt_digest": receipt["receipt_digest"],
        "runtime_request_digest": receipt["runtime_request_digest"],
        "backend_entry_digest": backend_entry_digest,
        "worker_image_digest": receipt["worker_image_digest"],
        "worker_container_image_digest": receipt["runtime_image_identity"],
        "runtime_image_identity": receipt["runtime_image_identity"],
        "worker_source_sha256": receipt["worker_source_sha256"],
        "model_snapshot": receipt["model_snapshot"],
        "adapter_id": receipt["adapter_id"],
        "pricing_binding_digest": receipt["pricing_binding_digest"],
        "maximum_cost_per_request_usd": maximum_cost_per_request_usd,
        "maximum_editor_request_cost_usd": maximum_editor_request_cost_usd,
        "maximum_compute_cost_usd": maximum_compute_cost_usd,
        "hosted_editor_spend_upper_bound_usd": maximum_editor_request_cost_usd,
        "vast_spend_upper_bound_usd": maximum_compute_cost_usd,
        "task_count": task_count,
        "camera_count": camera_count,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
        "hard_total_spend_cap_usd": hard_total_spend_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": aggregate_goal_spend_before_attempt_usd,
        "aggregate_goal_spend_cap_usd": aggregate_goal_spend_cap_usd,
        "prior_spend_reconciliation": prior_record,
        "consumption_root_kind": "host_private_atomic_single_use",
        "raw_nonredistributable_bytes_upload_authorized": False,
        "canonical_interiorgs_mutation_authorized": False,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("semantic_teacher_paid_authority_output_exists")
    ensure_dir(output.parent)
    write_json(output, authority)
    validate_semantic_teacher_image_edit_paid_authority(
        authority,
        bundle_path=bundle,
        bundle_receipt=receipt,
        source_commit_sha=source_commit_sha,
        backend_entry_digest=backend_entry_digest,
        task_count=task_count,
        camera_count=camera_count,
        maximum_hourly_rate_usd=maximum_hourly_rate_usd,
        hard_total_spend_cap_usd=hard_total_spend_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
    )
    return authority


def validate_semantic_teacher_image_edit_paid_authority(
    authority: Mapping[str, Any],
    *,
    bundle_path: str | Path,
    bundle_receipt: Mapping[str, Any],
    source_commit_sha: str,
    backend_entry_digest: str,
    task_count: int,
    camera_count: int,
    maximum_hourly_rate_usd: float,
    hard_total_spend_cap_usd: float,
    hard_ttl_seconds: int,
) -> dict[str, Any]:
    """Reopen every bound file and fail on self-consistent authority tampering."""

    value = dict(authority)
    expected = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "purpose": "one_shot_semantic_teacher_image_edit",
        "paid_execution_authorized": True,
        "source_commit_sha": source_commit_sha,
        "bundle_receipt_digest": bundle_receipt.get("receipt_digest"),
        "runtime_request_digest": bundle_receipt.get("runtime_request_digest"),
        "backend_entry_digest": backend_entry_digest,
        "worker_image_digest": bundle_receipt.get("worker_image_digest"),
        "worker_container_image_digest": bundle_receipt.get("runtime_image_identity"),
        "runtime_image_identity": bundle_receipt.get("runtime_image_identity"),
        "worker_source_sha256": bundle_receipt.get("worker_source_sha256"),
        "model_snapshot": bundle_receipt.get("model_snapshot"),
        "adapter_id": bundle_receipt.get("adapter_id"),
        "pricing_binding_digest": bundle_receipt.get("pricing_binding_digest"),
        "maximum_cost_per_request_usd": bundle_receipt.get(
            "maximum_cost_per_request_usd"
        ),
        "maximum_editor_request_cost_usd": float(
            bundle_receipt.get("maximum_cost_per_request_usd") or 0
        )
        * camera_count,
        "maximum_compute_cost_usd": maximum_hourly_rate_usd
        * hard_ttl_seconds
        / 3600,
        "hosted_editor_spend_upper_bound_usd": float(
            bundle_receipt.get("maximum_cost_per_request_usd") or 0
        )
        * camera_count,
        "vast_spend_upper_bound_usd": maximum_hourly_rate_usd
        * hard_ttl_seconds
        / 3600,
        "task_count": task_count,
        "camera_count": camera_count,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "maximum_hourly_rate_usd": maximum_hourly_rate_usd,
        "hard_total_spend_cap_usd": hard_total_spend_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "consumption_root_kind": "host_private_atomic_single_use",
        "raw_nonredistributable_bytes_upload_authorized": False,
        "canonical_interiorgs_mutation_authorized": False,
    }
    errors = [
        f"{key}_mismatch"
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("authorization_digest_invalid")
    bundle = Path(bundle_path).expanduser().resolve()
    try:
        if _bound_record(value.get("bundle"), code="bundle_unbound") != bundle:
            errors.append("bundle_path_mismatch")
        receipt_path = _bound_record(
            value.get("bundle_receipt"), code="bundle_receipt_unbound"
        )
        if _read(receipt_path, code="bundle_receipt_invalid") != dict(bundle_receipt):
            errors.append("bundle_receipt_changed")
        if (
            bundle_receipt.get("receipt_digest")
            != canonical_digest(bundle_receipt, digest_field="receipt_digest")
            or (bundle_receipt.get("bundle") or {}).get("sha256") != _sha256(bundle)
            or (bundle_receipt.get("bundle") or {}).get("size_bytes")
            != bundle.stat().st_size
        ):
            errors.append("bundle_receipt_invalid")
        prior = value.get("prior_spend_reconciliation")
        before = value.get("aggregate_goal_spend_before_attempt_usd")
        if prior is None:
            if before != 0:
                errors.append("prior_spend_missing")
        else:
            prior_path = _bound_record(prior, code="prior_spend_unbound")
            try:
                _prior_value, expected_record = _validate_prior_spend_reconciliation(
                    prior_path, expected_total_cost_usd=float(before)
                )
            except (TypeError, ValueError):
                errors.append("prior_spend_invalid")
            else:
                if dict(prior) != expected_record:
                    errors.append("prior_spend_record_mismatch")
    except ValueError as exc:
        errors.append(str(exc))
    before = value.get("aggregate_goal_spend_before_attempt_usd")
    aggregate_cap = value.get("aggregate_goal_spend_cap_usd")
    ceilings = (
        value.get("maximum_editor_request_cost_usd"),
        value.get("maximum_compute_cost_usd"),
        value.get("hosted_editor_spend_upper_bound_usd"),
        value.get("vast_spend_upper_bound_usd"),
    )
    if (
        not _finite(before)
        or not _finite(aggregate_cap, positive=True)
        or before + hard_total_spend_cap_usd > aggregate_cap
        or any(not _finite(item) for item in ceilings)
        or sum(float(item) for item in ceilings[:2]) > hard_total_spend_cap_usd
        or sum(float(item) for item in ceilings[2:]) > hard_total_spend_cap_usd
    ):
        errors.append("aggregate_spend_invalid")
    if errors:
        raise ValueError(
            "semantic_teacher_paid_authority_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return value


def consume_semantic_teacher_image_edit_paid_authority_once(
    authority: Mapping[str, Any], *, source_commit_sha: str
) -> dict[str, Any]:
    """Atomically consume one authority before any staging or provider mutation."""

    digest = str(authority.get("authorization_digest") or "")
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or digest
        != canonical_digest(dict(authority), digest_field="authorization_digest")
        or authority.get("source_commit_sha") != source_commit_sha
        or len(source_commit_sha) != 40
    ):
        return {
            "status": "blocked",
            "blockers": ["semantic_teacher_paid_authority_identity_invalid"],
        }
    payload = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "authorization_digest": digest,
        "source_commit_sha": source_commit_sha,
        "bundle_sha256": (authority.get("bundle") or {}).get("sha256"),
        "backend_entry_digest": authority.get("backend_entry_digest"),
        "task_count": authority.get("task_count"),
        "camera_count": authority.get("camera_count"),
        "maximum_provider_allocations": 1,
        "consumed_at": utc_now_iso(),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        root = prepare_consumption_root()
        root_stat = root.stat()
        if root.is_symlink() or root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o077:
            raise OSError("insecure_consumption_root")
        identity = digest.removeprefix("sha256:")
        destination = root / f"semantic-teacher-{identity}.json"
        temporary = root / f".semantic-teacher-{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["semantic_teacher_paid_authority_consumed"],
        }
    except (OSError, ValueError):
        return {
            "status": "blocked",
            "blockers": ["semantic_teacher_paid_authority_consumption_failed"],
        }
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--backend-entry-digest", required=True)
    parser.add_argument("--task-count", required=True, type=int)
    parser.add_argument("--camera-count", required=True, type=int)
    parser.add_argument("--maximum-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-total-spend-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--aggregate-goal-spend-before-usd", required=True, type=float)
    parser.add_argument("--aggregate-goal-spend-cap-usd", required=True, type=float)
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    materialize_semantic_teacher_image_edit_paid_authority(
        bundle_path=args.bundle,
        bundle_receipt_path=args.bundle_receipt,
        authorization_reference=args.authorization_reference,
        authorized_by=args.authorized_by,
        authorized_on=args.authorized_on,
        source_commit_sha=args.source_commit_sha,
        backend_entry_digest=args.backend_entry_digest,
        task_count=args.task_count,
        camera_count=args.camera_count,
        maximum_hourly_rate_usd=args.maximum_hourly_rate_usd,
        hard_total_spend_cap_usd=args.hard_total_spend_cap_usd,
        hard_ttl_seconds=args.hard_ttl_seconds,
        aggregate_goal_spend_before_attempt_usd=args.aggregate_goal_spend_before_usd,
        aggregate_goal_spend_cap_usd=args.aggregate_goal_spend_cap_usd,
        prior_spend_reconciliation_path=args.prior_spend_reconciliation,
        output_path=args.output,
    )
    return 0


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "PRIOR_SPEND_ENTRY_SCHEMA_VERSION",
    "PRIOR_SPEND_RECONCILIATION_SCHEMA_VERSION",
    "consume_semantic_teacher_image_edit_paid_authority_once",
    "materialize_semantic_teacher_image_edit_paid_authority",
    "validate_semantic_teacher_image_edit_paid_authority",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
