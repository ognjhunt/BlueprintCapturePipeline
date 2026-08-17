"""One-shot paid native Isaac import for a bound 1--5 replacement bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import zipfile
from typing import Any

from .adp009d_native_microcheck_bundle import DEFAULT_IMAGE
from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_attempt_authority import (
    active_instance_allowlist_metadata_error,
    flatten_active_instance_allowlist,
    normalize_active_instance_allowlist,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .paired_target_native_import_bundle import (
    RESULT_FILENAME,
    RESULT_SCHEMA_VERSION as RUNTIME_RESULT_SCHEMA_VERSION,
    validate_paired_target_native_import_bundle,
)
from .paired_target_native_import_recovery import (
    validate_paired_target_native_import_recovered_provider_zero,
)
from .public_scene_artifixer3d_vast import validate_artifixer3d_terminal_spend_chain
from .spend_authority_consumption_root import consumption_root
from .task_evaluation_artifact_manifest import (
    seal_lane_terminal_artifacts,
    seal_unallocated_provider_teardown,
)
from .vast_independent_watchdog_control import (
    EVIDENCE_NAME as WATCHDOG_EVIDENCE_NAME,
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-paired-target-native-import"
PROVIDER_BUNDLE_KIND = "paired_target_native_import"
RESULT_SCHEMA_VERSION = "paired_target_native_import_vast_run.v1"
PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "paired_target_native_import_paid_attempt_authority.v1"
)
POST_ATTEMPT_PROVIDER_ZERO_SCHEMA_VERSION = (
    "paired_target_native_import_provider_zero.v1"
)
SUPPLEMENTAL_SPEND_SCHEMA_VERSION = (
    "paired_target_native_import_supplemental_spend_reconciliation.v1"
)
PREALLOCATION_ZERO_SCHEMA_VERSION = (
    "paired_target_native_import_preallocation_provider_zero.v1"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/paired-target-native-import"
INSTANCE_LABEL_PREFIX = "blueprint-adp-paired-native-import-"
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 7_200
MAX_HARD_CAP_USD = 2.0
AGGREGATE_GOAL_SPEND_CAP_USD = 12.0
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound_record(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    path = Path(str(value.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def _validate_content_agents_spend_entry(
    records: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    required = (
        "authority",
        "terminal_result",
        "session_budget",
        "vast_budget_ledger",
        "provider_adapter",
        "object_store_cleanup",
        "teardown",
        "post_teardown_provider_zero",
        "api_zero_guard_snapshot",
        "provider_output_download_manifest",
    )
    paths = {
        key: _bound_record(
            records.get(key), f"paired_target_supplemental_{key}_unbound"
        )[0]
        for key in required
    }
    values = {
        key: _read(path, f"paired_target_supplemental_{key}_unreadable")
        for key, path in paths.items()
    }
    authority = values["authority"]
    result = values["terminal_result"]
    session = values["session_budget"]
    ledger = values["vast_budget_ledger"]
    adapter = values["provider_adapter"]
    cleanup = values["object_store_cleanup"]
    teardown = values["teardown"]
    zero = values["post_teardown_provider_zero"]
    snapshot = values["api_zero_guard_snapshot"]
    download = values["provider_output_download_manifest"]
    cost = result.get("estimated_cost_usd")
    instance_ids = teardown.get("vast_instance_ids")
    attempts = session.get("attempts")
    guard = snapshot.get("guard")
    inventory = guard.get("inventory_results") if isinstance(guard, Mapping) else None
    vast_rows = [
        row
        for row in inventory or []
        if isinstance(row, Mapping) and row.get("provider") == "vast"
    ]
    if (
        authority.get("schema_version")
        != "adp_content_agents_paid_attempt_authority.v1"
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or authority.get("paid_compute_authorized") is not True
        or authority.get("maximum_paid_attempts") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("zero_retry") is not True
        or result.get("schema_version") != "adp_content_agents_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("bundle_sha256") != authority.get("bundle_sha256")
        or result.get("hard_cap_usd") != authority.get("hard_attempt_spend_cap_usd")
        or result.get("hard_ttl_seconds")
        != authority.get("maximum_single_resource_ttl_seconds")
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or result.get("blockers") != []
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or session.get("schema_version") != "vast_session_cost_summary.v4"
        or session.get("status") != "completed"
        or session.get("attempt_count") != 1
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or session.get("estimated_cost_usd") != cost
        or attempts[0].get("estimated_cost_usd") != cost
        or attempts[0].get("continuing_spend_from_this_run") is not False
        or attempts[0].get("vast_instance_ids") != instance_ids
        or ledger.get("schema_version") != "vast_budget_ledger.v1"
        or ledger.get("status") != "completed"
        or ledger.get("estimated_cost_usd") != cost
        or ledger.get("hard_cap_usd") != result.get("hard_cap_usd")
        or ledger.get("continuing_spend_from_this_run") is not False
        or ledger.get("vast_instance_ids") != instance_ids
        or adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("status") != "completed"
        or adapter.get("provider_bundle_kind") != "adp_content_agents"
        or adapter.get("selected_container_image") != authority.get("container_image")
        or adapter.get("provider_create_attempted") is not True
        or adapter.get("estimated_cost_usd") != cost
        or adapter.get("continuing_spend_from_this_run") is not False
        or adapter.get("vast_instance_ids") != instance_ids
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or not isinstance(instance_ids, list)
        or not instance_ids
        or zero.get("schema_version")
        != "task_evaluation_post_teardown_provider_zero.v1"
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("required_providers") != ["vast"]
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("allocator_invoked") is not False
        or zero.get("provider_mutation_performed") is not False
        or zero.get("automatic_retry_performed") is not False
        or zero.get("blockers") != []
        or zero.get("provider_zero_receipt_digest")
        != canonical_digest(zero, digest_field="provider_zero_receipt_digest")
        or zero.get("teardown_manifest", {}).get("digest") != _sha256(paths["teardown"])
        or zero.get("independent_guard_snapshot", {}).get("snapshot_digest")
        != snapshot.get("snapshot_digest")
        or snapshot.get("schema_version")
        != "task_evaluation_provider_zero_guard_snapshot.v1"
        or snapshot.get("snapshot_digest")
        != canonical_digest(snapshot, digest_field="snapshot_digest")
        or not isinstance(guard, Mapping)
        or guard.get("provider_zero_verified") is not True
        or guard.get("live_instance_count") != 0
        or guard.get("total_burn_per_hour_usd") != 0
        or len(vast_rows) != 1
        or vast_rows[0].get("status") != "succeeded"
        or vast_rows[0].get("row_count") != 0
        or download.get("schema_version") != "vast_provider_output_download.v1"
        or download.get("status") != "completed"
        or download.get("provider_upload_marker_seen") is not True
        or download.get("output_zip_present_after_download") is not True
        or not isinstance(download.get("output_zip_size_bytes"), int)
        or download.get("output_zip_size_bytes", 0) <= 0
    ):
        raise ValueError("paired_target_content_agents_supplemental_spend_invalid")
    return {
        "kind": "content_agents_terminal_closeout",
        "authority_digest": authority["authorization_digest"],
        "provider": "vast",
        "instance_ids": list(instance_ids),
        "cost_usd": round(float(cost), 6),
        "independent_watchdog_receipt_present": False,
        "independent_watchdog_typed_gap": (
            "content_agents_lane_did_not_arm_separate_independent_watchdog"
        ),
        "records": {key: _record(path) for key, path in paths.items()},
    }, round(float(cost), 6)


def materialize_paired_target_native_import_supplemental_spend_reconciliation(
    *,
    content_agents_attempts: Sequence[Mapping[str, str | Path]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind terminal same-goal CAD spend without upgrading its scientific claims."""

    entries: list[dict[str, Any]] = []
    required = {
        "authority",
        "terminal_result",
        "session_budget",
        "vast_budget_ledger",
        "provider_adapter",
        "object_store_cleanup",
        "teardown",
        "post_teardown_provider_zero",
        "api_zero_guard_snapshot",
        "provider_output_download_manifest",
    }
    for row in content_agents_attempts:
        if set(row) != required:
            raise ValueError("paired_target_supplemental_spend_entries_invalid")
        records = {
            key: _record(Path(value).expanduser().resolve())
            for key, value in row.items()
        }
        entry, _ = _validate_content_agents_spend_entry(records)
        entries.append(entry)
    identities = [str(row["authority_digest"]) for row in entries]
    if not entries or len(identities) != len(set(identities)):
        raise ValueError("paired_target_supplemental_spend_entries_invalid")
    value: dict[str, Any] = {
        "schema_version": SUPPLEMENTAL_SPEND_SCHEMA_VERSION,
        "status": "all_supplemental_spend_terminal_and_provider_zero",
        "entries": entries,
        "total_cost_usd": round(sum(float(row["cost_usd"]) for row in entries), 6),
        "continuing_spend": False,
        "provider_zero_confirmed_for_every_entry": True,
        "scientific_claims_upgraded": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_supplemental_spend_output_exists")
    ensure_dir(output.parent)
    write_json(output, value)
    return value


def _validate_supplemental_spend_reconciliation(
    path: Path,
) -> tuple[dict[str, Any], float]:
    value = _read(path, "paired_target_supplemental_spend_unreadable")
    entries = value.get("entries")
    if (
        value.get("schema_version") != SUPPLEMENTAL_SPEND_SCHEMA_VERSION
        or value.get("status") != "all_supplemental_spend_terminal_and_provider_zero"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("continuing_spend") is not False
        or value.get("provider_zero_confirmed_for_every_entry") is not True
        or value.get("scientific_claims_upgraded") is not False
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("paired_target_supplemental_spend_invalid")
    total = 0.0
    identities: list[str] = []
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("paired_target_supplemental_spend_invalid")
        expected, cost = _validate_content_agents_spend_entry(row.get("records") or {})
        if dict(row) != expected:
            raise ValueError("paired_target_supplemental_spend_entry_mismatch")
        identities.append(expected["authority_digest"])
        total += cost
    total = round(total, 6)
    if len(identities) != len(set(identities)) or total != value.get("total_cost_usd"):
        raise ValueError("paired_target_supplemental_spend_total_mismatch")
    return value, total


def _validated_preallocation_payload(records: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "attempt_authority",
        "terminal_result",
        "watchdog_handoff",
        "object_store_cleanup",
        "api_provider_zero",
    }
    if set(records) != required:
        raise ValueError("paired_target_preallocation_provider_zero_invalid")
    paths = {
        key: _bound_record(
            records.get(key), "paired_target_preallocation_provider_zero_unbound"
        )[0]
        for key in required
    }
    values = {
        key: _read(path, f"paired_target_preallocation_{key}_unreadable")
        for key, path in paths.items()
    }
    authority = values["attempt_authority"]
    result = values["terminal_result"]
    watchdog = values["watchdog_handoff"]
    cleanup = values["object_store_cleanup"]
    zero = values["api_provider_zero"]
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "blocked"
        or result.get("provider_mutations_performed") != 0
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("all_staged_objects_absent") is not True
        or result.get("blockers") != ["paired_target_native_import_watchdog_not_armed"]
        or watchdog.get("schema_version") != "vast_independent_watchdog_handoff.v1"
        or watchdog.get("status") != "blocked"
        or watchdog.get("watchdog_armed_before_allocation") is not False
        or watchdog.get("provider_mutations_performed") != 0
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or zero.get("schema_version") != "adp_paid_provider_zero.v1"
        or zero.get("provider") != "vast"
        or zero.get("api_confirmed") is not True
        or zero.get("global_live_resource_count") != 0
        or zero.get("provider_zero") is not True
        or zero.get("inventory") != []
        or zero.get("provider_zero_digest")
        != canonical_digest(zero, digest_field="provider_zero_digest")
    ):
        raise ValueError("paired_target_preallocation_provider_zero_invalid")
    return {
        "schema_version": PREALLOCATION_ZERO_SCHEMA_VERSION,
        "status": "blocked_before_provider_allocation_and_provider_zero",
        "attempt_authority_digest": authority["authorization_digest"],
        "provider_mutations_performed": 0,
        "attempt_cost_usd": 0.0,
        "provider_zero_confirmed": True,
        "independent_watchdog_armed": False,
        "records": {key: _record(path) for key, path in paths.items()},
    }


def materialize_paired_target_native_import_preallocation_provider_zero(
    *,
    attempt_authority_path: str | Path,
    result_path: str | Path,
    watchdog_handoff_path: str | Path,
    cleanup_path: str | Path,
    api_provider_zero_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Close one consumed authority that stopped before provider allocation."""

    paths = {
        "attempt_authority": Path(attempt_authority_path).expanduser().resolve(),
        "terminal_result": Path(result_path).expanduser().resolve(),
        "watchdog_handoff": Path(watchdog_handoff_path).expanduser().resolve(),
        "object_store_cleanup": Path(cleanup_path).expanduser().resolve(),
        "api_provider_zero": Path(api_provider_zero_path).expanduser().resolve(),
    }
    records = {key: _record(path) for key, path in paths.items()}
    value: dict[str, Any] = {
        **_validated_preallocation_payload(records),
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_preallocation_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, value)
    return value


def _validate_preallocation_provider_zero(path: Path) -> dict[str, Any]:
    value = _read(path, "paired_target_preallocation_provider_zero_unreadable")
    if (
        value.get("schema_version") != PREALLOCATION_ZERO_SCHEMA_VERSION
        or value.get("status")
        != "blocked_before_provider_allocation_and_provider_zero"
        or value.get("provider_mutations_performed") != 0
        or value.get("attempt_cost_usd") != 0.0
        or value.get("provider_zero_confirmed") is not True
        or value.get("independent_watchdog_armed") is not False
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
    ):
        raise ValueError("paired_target_preallocation_provider_zero_invalid")
    records = value.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("paired_target_preallocation_provider_zero_invalid")
    expected = {
        **_validated_preallocation_payload(records),
        "receipt_digest": value["receipt_digest"],
    }
    if value != expected:
        raise ValueError("paired_target_preallocation_provider_zero_invalid")
    return value


def _validated_prior_paired_attempts(
    paths: Sequence[str | Path],
    *,
    source_request_digest: str,
) -> tuple[list[dict[str, Any]], float, tuple[int, ...]]:
    entries: list[dict[str, Any]] = []
    authority_digests: list[str] = []
    instance_ids: list[int] = []
    total = 0.0
    exclusions: set[int] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        zero = validate_paired_target_native_import_recovered_provider_zero(path)
        records = zero.get("records")
        if not isinstance(records, Mapping):
            raise ValueError("paired_target_prior_attempt_invalid")
        authority_path = _bound_record(
            records.get("attempt_authority"), "paired_target_prior_authority_unbound"
        )[0]
        authority = _read(authority_path, "paired_target_prior_authority_unreadable")
        digest = authority.get("authorization_digest")
        authority_source_request_digest = authority.get("source_request_digest")
        instance_id = zero.get("provider_instance_id")
        official_cost = zero.get("official_cost_usd")
        declared_prior = authority.get("prior_paired_attempts") or []
        declared_digests = [
            row.get("attempt_authority_digest")
            for row in declared_prior
            if isinstance(row, Mapping)
        ]
        recommended = zero.get("recommended_excluded_machine_ids")
        if (
            authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
            or digest != canonical_digest(authority, digest_field="authorization_digest")
            or digest != zero.get("attempt_authority_digest")
            or authority_source_request_digest != source_request_digest
            or declared_digests != authority_digests
            or len(declared_digests) != len(declared_prior)
            or isinstance(instance_id, bool)
            or not isinstance(instance_id, int)
            or instance_id <= 0
            or isinstance(official_cost, bool)
            or not isinstance(official_cost, (int, float))
            or not math.isfinite(float(official_cost))
            or float(official_cost) < 0
            or not isinstance(recommended, list)
            or not recommended
            or any(
                isinstance(machine_id, bool)
                or not isinstance(machine_id, int)
                or machine_id <= 0
                for machine_id in recommended
            )
        ):
            raise ValueError("paired_target_prior_attempt_invalid")
        entry = {
            **_record(path),
            "receipt_digest": zero["receipt_digest"],
            "attempt_authority_digest": digest,
            "source_request_digest": authority_source_request_digest,
            "provider_instance_id": instance_id,
            "official_cost_usd": round(float(official_cost), 6),
            "recommended_excluded_machine_ids": sorted(set(recommended)),
        }
        entries.append(entry)
        authority_digests.append(str(digest))
        instance_ids.append(instance_id)
        total += float(official_cost)
        exclusions.update(int(machine_id) for machine_id in recommended)
    if (
        len(authority_digests) != len(set(authority_digests))
        or len(instance_ids) != len(set(instance_ids))
    ):
        raise ValueError("paired_target_prior_attempt_duplicate")
    return entries, round(total, 6), tuple(sorted(exclusions))


def _claim_paired_lineage_scope(
    *, authority_digest: str, source_request_digest: str
) -> None:
    root = consumption_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError("paired_target_consumption_root_invalid")
    path = root / f"paired-target-lineage-scope-{authority_digest[7:]}.json"
    value = {
        "schema_version": "paired_target_native_import_lineage_scope.v1",
        "authority_digest": authority_digest,
        "source_request_digest": source_request_digest,
        "claim_digest": "",
    }
    value["claim_digest"] = canonical_digest(value, digest_field="claim_digest")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = root / f".{path.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        existing = _read(path, "paired_target_lineage_scope_already_claimed")
        if existing != value:
            raise ValueError("paired_target_lineage_scope_already_claimed")
    finally:
        temporary.unlink(missing_ok=True)


def _consumed_paired_authority_digests(*, source_request_digest: str) -> list[str]:
    root = consumption_root()
    if not root.exists():
        return []
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError("paired_target_consumption_root_invalid")
    rows: list[tuple[str, str]] = []
    for path in root.glob("paired-target-native-import-*.json"):
        value = _read(path, "paired_target_consumption_record_invalid")
        digest = value.get("authorization_digest")
        consumed_at = value.get("consumed_at")
        record_scope = value.get("source_request_digest")
        if (
            value.get("schema_version")
            != "paired_target_native_import_authority_consumption.v1"
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or path.name != f"paired-target-native-import-{digest[7:]}.json"
            or not isinstance(consumed_at, str)
            or not consumed_at
        ):
            raise ValueError("paired_target_consumption_record_invalid")
        if record_scope is None:
            scope_path = root / f"paired-target-lineage-scope-{digest[7:]}.json"
            if not scope_path.is_file() or scope_path.is_symlink():
                raise ValueError("paired_target_unscoped_consumption_requires_recovery")
            scope_claim = _read(scope_path, "paired_target_lineage_scope_invalid")
            if (
                scope_claim.get("schema_version")
                != "paired_target_native_import_lineage_scope.v1"
                or scope_claim.get("authority_digest") != digest
                or scope_claim.get("claim_digest")
                != canonical_digest(scope_claim, digest_field="claim_digest")
            ):
                raise ValueError("paired_target_lineage_scope_invalid")
            record_scope = scope_claim.get("source_request_digest")
        if record_scope != source_request_digest:
            continue
        rows.append((consumed_at, digest))
    rows.sort()
    digests = [digest for _consumed_at, digest in rows]
    if len(digests) != len(set(digests)):
        raise ValueError("paired_target_consumption_record_duplicate")
    return digests


def _claim_paired_lineage_successor(
    *, predecessor_digest: str, successor_digest: str
) -> None:
    root = consumption_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError("paired_target_consumption_root_invalid")
    path = root / f"paired-target-lineage-successor-{predecessor_digest[7:]}.json"
    value = {
        "schema_version": "paired_target_native_import_lineage_successor_claim.v1",
        "predecessor_authority_digest": predecessor_digest,
        "successor_authority_digest": successor_digest,
        "claim_digest": "",
    }
    value["claim_digest"] = canonical_digest(value, digest_field="claim_digest")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = root / f".{path.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        existing = _read(path, "paired_target_lineage_successor_already_claimed")
        if existing != value:
            raise ValueError("paired_target_lineage_successor_already_claimed")
    finally:
        temporary.unlink(missing_ok=True)


def materialize_paired_target_native_import_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    prior_artifixer_authority_path: str | Path,
    prior_artifixer_result_path: str | Path,
    prior_artifixer_cleanup_path: str | Path,
    prior_artifixer_provider_zero_path: str | Path,
    supplemental_prior_spend_reconciliation_path: str | Path | None = None,
    prior_native_preallocation_provider_zero_path: str | Path | None = None,
    prior_paired_attempt_provider_zero_paths: Sequence[str | Path] = (),
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Seal one new authority chained to the exact prior scene spend and zero receipt."""

    bundle = validate_paired_target_native_import_bundle(bundle_receipt_path)
    terminal = validate_artifixer3d_terminal_spend_chain(
        authority_path=prior_artifixer_authority_path,
        result_path=prior_artifixer_result_path,
        cleanup_path=prior_artifixer_cleanup_path,
        provider_zero_path=prior_artifixer_provider_zero_path,
    )
    prior_spend = float(terminal["aggregate_goal_spend_after_attempt_usd"])
    supplemental: dict[str, Any] | None = None
    if supplemental_prior_spend_reconciliation_path is not None:
        supplemental_path = Path(
            supplemental_prior_spend_reconciliation_path
        ).expanduser().resolve()
        supplemental, supplemental_cost = _validate_supplemental_spend_reconciliation(
            supplemental_path
        )
        supplemental = {
            **_record(supplemental_path),
            "receipt_digest": supplemental["receipt_digest"],
            "total_cost_usd": supplemental_cost,
        }
        prior_spend = round(prior_spend + supplemental_cost, 6)
    prior_native: dict[str, Any] | None = None
    if prior_native_preallocation_provider_zero_path is not None:
        native_zero_path = Path(
            prior_native_preallocation_provider_zero_path
        ).expanduser().resolve()
        native_zero = _validate_preallocation_provider_zero(native_zero_path)
        prior_native = {
            **_record(native_zero_path),
            "receipt_digest": native_zero["receipt_digest"],
            "attempt_authority_digest": native_zero["attempt_authority_digest"],
            "attempt_cost_usd": 0.0,
        }
    source_request_digest = str(bundle["source_request_digest"])
    prior_paired, prior_paired_cost, excluded = _validated_prior_paired_attempts(
        prior_paired_attempt_provider_zero_paths,
        source_request_digest=source_request_digest,
    )
    for row in prior_paired:
        _claim_paired_lineage_scope(
            authority_digest=str(row["attempt_authority_digest"]),
            source_request_digest=source_request_digest,
        )
    consumed_digests = _consumed_paired_authority_digests(
        source_request_digest=source_request_digest
    )
    prior_digests = [row["attempt_authority_digest"] for row in prior_paired]
    if prior_digests != consumed_digests:
        raise ValueError("paired_target_prior_attempt_lineage_required")
    prior_spend = round(prior_spend + prior_paired_cost, 6)
    aggregate_cap = min(
        AGGREGATE_GOAL_SPEND_CAP_USD,
        float(terminal["aggregate_goal_spend_cap_usd"]),
    )
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or blueprint_commit != bundle.get("implementation_commit")
        or DEFAULT_IMAGE != bundle.get("container_image")
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
        or prior_spend + hard_cap_usd > aggregate_cap
        or any(value <= 0 for value in allowed)
        or any(value <= 0 for value in excluded)
    ):
        raise ValueError("paired_target_native_import_authority_configuration_invalid")
    receipt = Path(str(bundle["receipt_path"])).resolve()
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_paired_target_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt": _record(receipt),
        "bundle_receipt_digest": bundle["receipt_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "probe_spec_sha256": bundle["probe_spec_sha256"],
        "source_request_digest": bundle["source_request_digest"],
        "replacement_count": bundle["replacement_count"],
        "blueprint_commit": blueprint_commit,
        "container_image": DEFAULT_IMAGE,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": prior_spend,
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "prior_terminal_artifixer": {
            **terminal["records"],
            "authority_digest": terminal["authority_digest"],
            "attempt_cost_usd": terminal["attempt_cost_usd"],
            "lineage_cost_usd": terminal["lineage_cost_usd"],
        },
        "supplemental_prior_spend_reconciliation": supplemental,
        "prior_native_preallocation_attempt": prior_native,
        "prior_paired_attempts": prior_paired,
        "paired_attempt_ordinal": len(prior_paired) + 1,
        "active_instance_allowlist": {
            "external_provider_owned": list(allowed),
            "same_goal_concurrent": [],
        },
        "excluded_vast_machine_ids": list(excluded),
        "native_simulator_import_probe_only": True,
        "candidate_policy_queried": False,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "physical_success_established": False,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_native_import_authority_output_exists")
    ensure_dir(output.parent)
    validate_paired_target_native_import_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed,
        excluded_machine_ids=excluded,
    )
    if consumed_digests:
        _claim_paired_lineage_successor(
            predecessor_digest=consumed_digests[-1],
            successor_digest=authority["authorization_digest"],
        )
    write_json(output, authority)
    return authority


def validate_paired_target_native_import_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
    excluded_machine_ids: Sequence[int] = (),
) -> dict[str, Any]:
    value = dict(authority)
    allowlist = normalize_active_instance_allowlist(value.get("active_instance_allowlist"))
    expected = normalize_active_instance_allowlist(list(allowed_active_instance_ids))
    expected_excluded = tuple(sorted({int(value) for value in excluded_machine_ids}))
    errors: list[str] = []
    if value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION:
        errors.append("schema_invalid")
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("digest_invalid")
    expected_fields = {
        "purpose": "one_shot_paired_target_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt_digest": prepared_bundle.get("receipt_digest"),
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "probe_spec_sha256": prepared_bundle.get("probe_spec_sha256"),
        "source_request_digest": prepared_bundle.get("source_request_digest"),
        "replacement_count": prepared_bundle.get("replacement_count"),
        "blueprint_commit": prepared_bundle.get("implementation_commit"),
        "container_image": DEFAULT_IMAGE,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "native_simulator_import_probe_only": True,
        "candidate_policy_queried": False,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "physical_success_established": False,
        "excluded_vast_machine_ids": list(expected_excluded),
        "paired_attempt_ordinal": len(value.get("prior_paired_attempts") or []) + 1,
    }
    errors.extend(f"{key}_mismatch" for key, expected_value in expected_fields.items() if value.get(key) != expected_value)
    if allowlist is None or expected is None or flatten_active_instance_allowlist(
        allowlist or {"external_provider_owned": (), "same_goal_concurrent": ()}
    ) != flatten_active_instance_allowlist(
        expected or {"external_provider_owned": (), "same_goal_concurrent": ()}
    ):
        errors.append("active_instance_allowlist_mismatch")
    elif active_instance_allowlist_metadata_error(value, allowlist=allowlist) is not None:
        errors.append("active_instance_allowlist_metadata_invalid")
    if any(value <= 0 for value in expected_excluded):
        errors.append("excluded_machine_ids_invalid")
    if (
        not isinstance(value.get("aggregate_goal_spend_before_attempt_usd"), (int, float))
        or isinstance(value.get("aggregate_goal_spend_before_attempt_usd"), bool)
        or value.get("aggregate_goal_spend_before_attempt_usd", 0) + hard_cap_usd
        > value.get("aggregate_goal_spend_cap_usd", 0)
    ):
        errors.append("aggregate_spend_invalid")
    try:
        receipt_path, _ = _bound_record(
            value.get("bundle_receipt"), "paired_target_native_import_bundle_unbound"
        )
        if receipt_path != Path(str(prepared_bundle.get("receipt_path") or "")).resolve():
            errors.append("bundle_receipt_path_mismatch")
        predecessor = value.get("prior_terminal_artifixer")
        if not isinstance(predecessor, Mapping):
            raise ValueError("predecessor_invalid")
        paths = {
            key: _bound_record(predecessor.get(key), "predecessor_unbound")[0]
            for key in (
                "authority",
                "terminal_result",
                "object_store_cleanup",
                "provider_zero",
            )
        }
        terminal = validate_artifixer3d_terminal_spend_chain(
            authority_path=paths["authority"],
            result_path=paths["terminal_result"],
            cleanup_path=paths["object_store_cleanup"],
            provider_zero_path=paths["provider_zero"],
        )
        supplemental_cost = 0.0
        supplemental_record = value.get("supplemental_prior_spend_reconciliation")
        if supplemental_record is not None:
            supplemental_path, supplemental_bound = _bound_record(
                supplemental_record, "supplemental_prior_spend_unbound"
            )
            supplemental, supplemental_cost = _validate_supplemental_spend_reconciliation(
                supplemental_path
            )
            if (
                supplemental_record.get("receipt_digest")
                != supplemental.get("receipt_digest")
                or supplemental_record.get("total_cost_usd") != supplemental_cost
                or supplemental_bound.get("sha256") != _sha256(supplemental_path)
            ):
                errors.append("supplemental_prior_spend_mismatch")
        prior_paired_records = value.get("prior_paired_attempts")
        if not isinstance(prior_paired_records, list):
            raise ValueError("prior_paired_attempts_invalid")
        prior_paired_paths = [
            _bound_record(row, "prior_paired_attempt_unbound")[0]
            for row in prior_paired_records
            if isinstance(row, Mapping)
        ]
        if len(prior_paired_paths) != len(prior_paired_records):
            raise ValueError("prior_paired_attempts_invalid")
        source_request_digest = str(prepared_bundle.get("source_request_digest") or "")
        paired_entries, paired_cost, paired_exclusions = _validated_prior_paired_attempts(
            prior_paired_paths,
            source_request_digest=source_request_digest,
        )
        if prior_paired_records != paired_entries:
            errors.append("prior_paired_attempts_mismatch")
        if expected_excluded != paired_exclusions:
            errors.append("excluded_machine_ids_lineage_mismatch")
        consumed = _consumed_paired_authority_digests(
            source_request_digest=source_request_digest
        )
        declared = [row["attempt_authority_digest"] for row in paired_entries]
        current_digest = value.get("authorization_digest")
        if tuple(consumed) not in {
            tuple(declared),
            tuple([*declared, current_digest]),
        }:
            errors.append("consumed_paired_attempt_lineage_mismatch")
        prior_native_record = value.get("prior_native_preallocation_attempt")
        if prior_native_record is not None:
            native_path, _ = _bound_record(
                prior_native_record, "prior_native_preallocation_attempt_unbound"
            )
            native_zero = _validate_preallocation_provider_zero(native_path)
            if (
                prior_native_record.get("receipt_digest")
                != native_zero.get("receipt_digest")
                or prior_native_record.get("attempt_authority_digest")
                != native_zero.get("attempt_authority_digest")
                or prior_native_record.get("attempt_cost_usd") != 0.0
            ):
                errors.append("prior_native_preallocation_attempt_mismatch")
        if (
            predecessor.get("authority_digest") != terminal["authority_digest"]
            or predecessor.get("attempt_cost_usd") != terminal["attempt_cost_usd"]
            or predecessor.get("lineage_cost_usd") != terminal["lineage_cost_usd"]
            or value.get("aggregate_goal_spend_before_attempt_usd")
            != round(
                terminal["aggregate_goal_spend_after_attempt_usd"]
                + supplemental_cost
                + paired_cost,
                6,
            )
            or value.get("aggregate_goal_spend_cap_usd")
            != terminal["aggregate_goal_spend_cap_usd"]
        ):
            errors.append("prior_terminal_spend_mismatch")
    except ValueError:
        errors.append("prior_terminal_spend_invalid")
    if errors:
        raise ValueError(
            "paired_target_native_import_authority_invalid:" + ",".join(sorted(set(errors)))
        )
    return value


def consume_paired_target_native_import_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["paired_target_native_import_authority_identity_invalid"]}
    root = consumption_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = root.stat()
        if root.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise OSError("insecure_root")
        destination = root / f"paired-target-native-import-{digest[7:]}.json"
        payload = {
            "schema_version": "paired_target_native_import_authority_consumption.v1",
            "authorization_digest": digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "source_request_digest": authority.get("source_request_digest"),
            "blueprint_commit": blueprint_commit,
            "consumed_at": utc_now_iso(),
            "maximum_provider_allocations": 1,
        }
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = root / f".paired-target-native-import-{digest[7:]}.{os.getpid()}.tmp"
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
        return {"status": "blocked", "blockers": ["paired_target_native_import_authority_consumed"]}
    except OSError:
        return {"status": "blocked", "blockers": ["paired_target_native_import_authority_consumption_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


@contextmanager
def _mutation_authority():
    previous = {name: os.environ.get(name) for name in _MUTATION_ENV}
    try:
        for name in _MUTATION_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract_result(source: Path, destination: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if destination.exists():
        shutil.rmtree(destination)
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError("path_traversal")
            archive.extractall(destination)
    except (OSError, ValueError, zipfile.BadZipFile):
        blockers.append("paired_target_native_import_output_zip_invalid")
    result_path = destination / RESULT_FILENAME
    try:
        execution = _read(result_path, "paired_target_native_import_runtime_result_missing")
    except ValueError:
        execution = {}
        blockers.append("paired_target_native_import_runtime_result_missing")
    if execution and (
        execution.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or execution.get("result_digest")
        != canonical_digest(execution, digest_field="result_digest")
    ):
        blockers.append("paired_target_native_import_runtime_result_invalid")
    return execution, blockers


def run_paired_target_native_import_vast(
    *,
    job_dir: str | Path,
    prepared_bundle: Mapping[str, Any],
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    paid_attempt_authority: Mapping[str, Any] | None,
    execute: bool,
    machine_avoidlist_path: str | Path | None = None,
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 1.0,
    hard_ttl_seconds: int = 3_600,
    allowed_active_instance_ids: Sequence[int] = (),
    excluded_machine_ids: Sequence[int] = (),
) -> dict[str, Any]:
    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    result_path = job / "paired_target_native_import_vast_result.v1.json"
    receipt_value = prepared_bundle.get("receipt_path")
    if not receipt_value:
        raise ValueError("paired_target_native_import_prepared_bundle_receipt_missing")
    bundle = validate_paired_target_native_import_bundle(receipt_value)
    if bundle.get("bundle_sha256") != prepared_bundle.get("bundle_sha256"):
        raise ValueError("paired_target_native_import_prepared_bundle_mismatch")
    if paid_attempt_authority is not None:
        authority = validate_paired_target_native_import_paid_attempt_authority(
            paid_attempt_authority,
            prepared_bundle=bundle,
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
            allowed_active_instance_ids=allowed_active_instance_ids,
            excluded_machine_ids=excluded_machine_ids,
        )
    else:
        authority = None
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle_sha256": bundle["bundle_sha256"],
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result
    if paid_resource_admission_grant is None or authority is None:
        raise ValueError("paired_target_native_import_paid_execution_authority_missing")
    consumption = consume_paired_target_native_import_authority_once(
        authority, blueprint_commit=authority["blueprint_commit"]
    )
    if consumption.get("status") != "consumed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
        write_json(result_path, result)
        return result
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=Path(str(bundle["bundle_path"])),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=hard_ttl_seconds + 1_800,
    )
    if staging.get("status") != "completed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "blockers": staging.get("blockers") or ["paired_target_native_import_staging_blocked"],
        }
        write_json(result_path, result)
        return result
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    handoff, handle = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=hard_ttl_seconds // 60,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed,
        pod_name_prefix=INSTANCE_LABEL_PREFIX,
    )
    if handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": handoff,
            "blockers": ["paired_target_native_import_watchdog_not_armed"],
        }
        write_json(result_path, result)
        return result
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any]
    try:
        with _mutation_authority():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=hard_ttl_seconds // 60,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=DEFAULT_IMAGE,
                isaac_image=DEFAULT_IMAGE,
                ngc_image_login_mode="always",
                provider_bundle=bundle["bundle_path"],
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=True,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=True,
                min_cold_isaac_pull_live_minutes=30,
                disk_gb=120,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=1_800,
                session_budget_ledger_path=job / "paired_target_native_import_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=True,
                preferred_gpu_keywords=("L40S", "RTX 4090", "RTX A6000"),
                prefer_isaac_rt=True,
                machine_avoidlist_path=machine_avoidlist_path,
                excluded_machine_ids=excluded_machine_ids,
                allowed_active_instance_ids=allowed,
                vast_launch_lock_file=job.parent / "paired_target_native_import_paid_launch.lock",
                instance_label_prefix=INSTANCE_LABEL_PREFIX,
                started_instance_id_path=handle.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [
                f"paired_target_native_import_adapter_failed:{redacted_failure_detail(exc)}"
            ],
        }
        # The adapter may never have been entered -- resolving a secret or a
        # staged URL raises before it. Record the absence of any allocation so
        # the run can close; the sealer declines whenever the evidence does not
        # support that claim.
        seal_unallocated_provider_teardown(
            provider_run, reason="paired_target_native_import_adapter_failed"
        )
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path, "paired_target_native_import_teardown_missing") if teardown_path.is_file() else {}
    instance_ids = [value for value in teardown.get("vast_instance_ids") or [] if isinstance(value, int) and value > 0]
    watchdog = close_independent_vast_watchdog(
        job_dir=job,
        handle=handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=not instance_ids and adapter.get("provider_create_attempted") is not True,
    )
    execution, blockers = _extract_result(output_zip, job / "immutable_execution")
    if adapter.get("status") != "completed":
        blockers.append("paired_target_native_import_provider_adapter_not_completed")
    if (
        execution.get("status") != "completed"
        or execution.get("native_isaac_executed") is not True
        or execution.get("all_replacements_import_qualified") is not True
        or execution.get("replacement_count") != bundle.get("replacement_count")
        or execution.get("request_digest") != bundle.get("request_digest")
        or execution.get("candidate_policy_queried") is not False
        or execution.get("physical_equivalence_claimed") is not False
    ):
        blockers.append("paired_target_native_import_runtime_not_qualified")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("paired_target_native_import_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        blockers.append("paired_target_native_import_watchdog_not_terminal")
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    final = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "request_digest": bundle["request_digest"],
        "replacement_count": bundle["replacement_count"],
        "native_result_path": str(job / "immutable_execution" / RESULT_FILENAME),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(teardown_path),
        "watchdog_receipt_path": str(watchdog_path),
        "object_store_cleanup_path": str(staging_dir / "wam_provider_object_store_cleanup.json"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_mutations_performed": 1 if adapter.get("provider_create_attempted") is True else 0,
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "excluded_vast_machine_ids": sorted(
            set(int(value) for value in excluded_machine_ids)
        ),
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "authorization_consumption": consumption,
        "independent_watchdog": watchdog,
        "candidate_policy_queried": False,
        "physical_success_established": False,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    final = seal_lane_terminal_artifacts(final, attempt_root=job, lane="paired_target_native_import")
    write_json(result_path, final)
    return final


def materialize_paired_target_native_import_provider_zero(
    *,
    attempt_authority_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    authority_path = Path(attempt_authority_path).expanduser().resolve()
    terminal_path = Path(result_path).expanduser().resolve()
    authority = _read(authority_path, "paired_target_native_import_authority_unreadable")
    result = _read(terminal_path, "paired_target_native_import_result_unreadable")
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = Path(str(result.get("object_store_cleanup_path") or "")).resolve()
    adapter_path = Path(str(result.get("adapter_result_path") or "")).resolve()
    watchdog = _read(watchdog_path, "paired_target_native_import_watchdog_unreadable")
    cleanup = _read(cleanup_path, "paired_target_native_import_cleanup_unreadable")
    adapter = _read(adapter_path, "paired_target_native_import_adapter_unreadable")
    inventory = watchdog.get("final_global_inventory")
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") not in {"completed", "blocked"}
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
        or adapter.get("continuing_spend_from_this_run") is not False
    ):
        raise ValueError("paired_target_native_import_provider_zero_invalid")
    receipt = {
        "schema_version": POST_ATTEMPT_PROVIDER_ZERO_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed",
        "attempt_authority": _record(authority_path),
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(terminal_path),
        "provider_adapter": _record(adapter_path),
        "watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_zero_confirmed": True,
        "inventory": inventory,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("paired_target_native_import_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


__all__ = [
    "PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "SUPPLEMENTAL_SPEND_SCHEMA_VERSION",
    "PREALLOCATION_ZERO_SCHEMA_VERSION",
    "consume_paired_target_native_import_authority_once",
    "materialize_paired_target_native_import_paid_attempt_authority",
    "materialize_paired_target_native_import_provider_zero",
    "materialize_paired_target_native_import_preallocation_provider_zero",
    "materialize_paired_target_native_import_supplemental_spend_reconciliation",
    "run_paired_target_native_import_vast",
    "validate_paired_target_native_import_paid_attempt_authority",
]
