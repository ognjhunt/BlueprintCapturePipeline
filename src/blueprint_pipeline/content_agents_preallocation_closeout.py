"""Seal and reopen a consumed Content Agents attempt that never allocated.

This receipt is deliberately narrow.  It does not turn an allocator failure into
a terminal Content Agents result; it only proves that one single-use authority
was consumed before provider allocation, object-store staging was removed, the
website received the blocked result, and a later authenticated global guard saw
no paid resources.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir, write_json
from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest
from .spend_authority_consumption_root import consumption_root


SCHEMA_VERSION = "content_agents_preallocation_provider_zero.v1"
AUTHORITY_SCHEMA_VERSION = "adp_content_agents_paid_attempt_authority.v1"
RESULT_SCHEMA_VERSION = "adp_content_agents_vast_run.v1"
WATCHDOG_SCHEMA_VERSION = "vast_independent_watchdog_handoff.v1"
CLEANUP_SCHEMA_VERSION = "wam_provider_object_store_cleanup.v1"
WEBAPP_SYNC_SCHEMA_VERSION = "task_evaluation_launch_webapp_sync_result.v1"
_EXPECTED_RESULT_BLOCKERS = ["adp_content_agents_independent_watchdog_not_armed"]
MAX_GLOBAL_GUARD_AGE_SECONDS = 300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not path.is_file() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_record(value: Any, code: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    candidate = Path(str(value.get("path") or "")).expanduser()
    absolute = candidate.absolute()
    path = candidate.resolve()
    if (
        candidate.is_symlink()
        or path != absolute
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def _global_guard_zero(value: Mapping[str, Any]) -> bool:
    rows = value.get("inventory_results")
    required = {
        row.get("provider"): row
        for row in rows or []
        if isinstance(row, Mapping) and row.get("required") is True
    }
    provider_zero = value.get("provider_zero")
    provider_zero = provider_zero if isinstance(provider_zero, Mapping) else {}
    return (
        value.get("schema_version") == "gpu_spend_guard.v1"
        and value.get("status") == "passed"
        and value.get("reap_mode") is True
        and value.get("provider_zero_verified") is True
        and value.get("live_instance_count") == 0
        and value.get("total_burn_per_hour_usd") in (0, 0.0)
        and value.get("reap_candidate_ids") in ([], ())
        and all(
            isinstance(row, Mapping) and row.get("status") == "terminated"
            for row in value.get("reap_results") or []
        )
        and provider_zero.get("status") == "verified"
        and provider_zero.get("global_live_instance_count") == 0
        and provider_zero.get("global_total_burn_per_hour_usd") in (0, 0.0)
        and set(required) == {"runpod", "vast", "digitalocean"}
        and all(
            row.get("status") == "succeeded" and row.get("row_count") == 0
            for row in required.values()
        )
    )


def _time(value: Any, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(code) from exc
    if parsed.tzinfo is None:
        raise ValueError(code)
    return parsed.astimezone(timezone.utc)


def _webapp_identity(value: Mapping[str, Any]) -> tuple[str, str, str]:
    response = value.get("response")
    response = response if isinstance(response, Mapping) else {}
    launch_id = value.get("launch_id")
    run_id = value.get("run_id")
    request_digest = value.get("request_digest")
    sync_digest = value.get("sync_result_digest")
    if (
        value.get("schema_version") != WEBAPP_SYNC_SCHEMA_VERSION
        or value.get("status") != "succeeded"
        or not isinstance(launch_id, str)
        or not launch_id
        or run_id != launch_id
        or not isinstance(request_digest, str)
        or len(request_digest) != 71
        or not request_digest.startswith("sha256:")
        or response.get("launch_id") != launch_id
        or response.get("run_id") != run_id
        or response.get("request_digest") != request_digest
        or response.get("receipt_digest") != value.get("receipt_digest")
        or response.get("schema_version")
        != "task_evaluation_launch_web_sync_receipt.v1"
        or response.get("status") != "blocked"
        or (
            sync_digest is not None
            and sync_digest
            != canonical_digest(value, digest_field="sync_result_digest")
        )
    ):
        raise ValueError("content_agents_preallocation_webapp_sync_invalid")
    return launch_id, run_id, request_digest


def _validated_payload(
    records: Mapping[str, Any], *, materialized_at: datetime
) -> dict[str, Any]:
    required = {
        "attempt_authority",
        "authority_consumption",
        "allocator_result",
        "watchdog_handoff",
        "object_store_cleanup",
        "fresh_global_guard",
        "webapp_sync",
    }
    if set(records) != required:
        raise ValueError("content_agents_preallocation_records_invalid")
    paths = {
        role: _bound_record(
            records.get(role), f"content_agents_preallocation_{role}_unbound"
        )[0]
        for role in required
    }
    values = {
        role: _read(path, f"content_agents_preallocation_{role}_invalid")
        for role, path in paths.items()
    }
    authority = values["attempt_authority"]
    consumption = values["authority_consumption"]
    result = values["allocator_result"]
    watchdog = values["watchdog_handoff"]
    cleanup = values["object_store_cleanup"]
    guard = values["fresh_global_guard"]
    launch_id, run_id, request_digest = _webapp_identity(values["webapp_sync"])
    result_at = _time(
        result.get("generated_at"), "content_agents_preallocation_result_time_invalid"
    )
    consumed_at = _time(
        consumption.get("consumed_at"),
        "content_agents_preallocation_consumption_time_invalid",
    )
    watchdog_at = _time(
        watchdog.get("generated_at"),
        "content_agents_preallocation_watchdog_time_invalid",
    )
    webapp_at = _time(
        values["webapp_sync"].get("attempted_at"),
        "content_agents_preallocation_webapp_time_invalid",
    )
    guard_at = _time(
        guard.get("generated_at"), "content_agents_preallocation_guard_time_invalid"
    )
    result_watchdog = result.get("independent_watchdog")
    result_consumption = result.get("authorization_consumption")
    result_consumption = (
        result_consumption if isinstance(result_consumption, Mapping) else {}
    )
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or consumption.get("schema_version")
        != "adp_content_agents_paid_attempt_consumption.v1"
        or consumption.get("authorization_digest")
        != authority.get("authorization_digest")
        or consumption.get("bundle_sha256") != authority.get("bundle_sha256")
        or consumption.get("maximum_provider_allocations") != 1
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") != "blocked"
        or result.get("provider_mutations_performed") != 0
        or result.get("all_staged_objects_absent") is not True
        or result.get("blockers") != _EXPECTED_RESULT_BLOCKERS
        or (
            result_consumption
            and (
                result_consumption.get("status") != "consumed"
                or result_consumption.get("authorization_digest")
                != authority.get("authorization_digest")
            )
        )
        or result_watchdog != watchdog
        or watchdog.get("schema_version") != WATCHDOG_SCHEMA_VERSION
        or watchdog.get("status") != "blocked"
        or watchdog.get("watchdog_armed_before_allocation") is not False
        or watchdog.get("provider_mutations_performed") != 0
        or cleanup.get("schema_version") != CLEANUP_SCHEMA_VERSION
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or not _global_guard_zero(guard)
        or result.get("instance_id") not in (None, "")
        or result.get("vast_instance_ids") not in (None, [], ())
        or result.get("provider_instance_ids") not in (None, [], ())
        or result.get("provider_create_attempted") not in (None, False)
    ):
        raise ValueError("content_agents_preallocation_evidence_invalid")
    if (
        guard_at < max(consumed_at, result_at, watchdog_at, webapp_at)
        or materialized_at < guard_at
        or (materialized_at - guard_at).total_seconds()
        > MAX_GLOBAL_GUARD_AGE_SECONDS
    ):
        raise ValueError("content_agents_preallocation_guard_not_fresh_after_attempt")
    launch_root = paths["webapp_sync"].parent
    if (
        launch_root.name != launch_id
        or paths["webapp_sync"].name != "webapp_sync_succeeded.json"
        or launch_root not in paths["allocator_result"].parents
        or launch_root not in paths["watchdog_handoff"].parents
        or launch_root not in paths["object_store_cleanup"].parents
    ):
        raise ValueError("content_agents_preallocation_launch_binding_invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_before_provider_allocation_and_provider_zero",
        "launch_id": launch_id,
        "run_id": run_id,
        "request_digest": request_digest,
        "attempt_authority_digest": authority["authorization_digest"],
        "bundle_sha256": authority.get("bundle_sha256"),
        "provider_allocations_performed": 0,
        "provider_mutations_performed": 0,
        "official_cost_usd": 0.0,
        "continuing_spend_from_attempt": False,
        "provider_zero_confirmed": True,
        "scientific_attempt_started": False,
        "content_agents_completed": False,
        "automatic_retry_performed": False,
        "evidence_times": {
            "authority_consumed_at": consumed_at.isoformat(),
            "allocator_result_generated_at": result_at.isoformat(),
            "watchdog_handoff_generated_at": watchdog_at.isoformat(),
            "webapp_sync_attempted_at": webapp_at.isoformat(),
            "fresh_global_guard_generated_at": guard_at.isoformat(),
            "materialized_at": materialized_at.isoformat(),
            "maximum_guard_age_seconds": MAX_GLOBAL_GUARD_AGE_SECONDS,
        },
        "records": {role: _record(path) for role, path in paths.items()},
    }


def materialize_content_agents_preallocation_provider_zero(
    *,
    attempt_authority_path: str | Path,
    allocator_result_path: str | Path,
    watchdog_handoff_path: str | Path,
    object_store_cleanup_path: str | Path,
    fresh_global_guard_path: str | Path,
    webapp_sync_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the immutable, zero-cost closeout for one consumed authority."""

    paths = {
        "attempt_authority": Path(attempt_authority_path).expanduser().resolve(),
        "allocator_result": Path(allocator_result_path).expanduser().resolve(),
        "watchdog_handoff": Path(watchdog_handoff_path).expanduser().resolve(),
        "object_store_cleanup": Path(object_store_cleanup_path).expanduser().resolve(),
        "fresh_global_guard": Path(fresh_global_guard_path).expanduser().resolve(),
        "webapp_sync": Path(webapp_sync_path).expanduser().resolve(),
    }
    authority = _read(
        paths["attempt_authority"], "content_agents_preallocation_attempt_authority_invalid"
    )
    identity = str(authority.get("authorization_digest") or "").removeprefix(
        "sha256:"
    )
    if len(identity) != 64:
        raise ValueError("content_agents_preallocation_attempt_authority_invalid")
    paths["authority_consumption"] = (
        consumption_root() / f"content-agents-{identity}.json"
    ).resolve()
    materialized_at = _time(
        utc_now_iso(), "content_agents_preallocation_observed_time_invalid"
    )
    value = {
        **_validated_payload(
            {role: _record(path) for role, path in paths.items()},
            materialized_at=materialized_at,
        ),
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("content_agents_preallocation_output_exists")
    ensure_dir(output.parent)
    write_json(output, value)
    return value


def validate_content_agents_preallocation_provider_zero(
    path: str | Path,
) -> dict[str, Any]:
    """Reopen every source byte behind one pre-allocation closeout."""

    source = Path(path).expanduser().resolve()
    value = _read(source, "content_agents_preallocation_receipt_invalid")
    records = value.get("records")
    times = value.get("evidence_times")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or not isinstance(records, Mapping)
        or not isinstance(times, Mapping)
    ):
        raise ValueError("content_agents_preallocation_receipt_invalid")
    expected = {
        **_validated_payload(
            records,
            materialized_at=_time(
                times.get("materialized_at"),
                "content_agents_preallocation_observed_time_invalid",
            ),
        ),
        "receipt_digest": value["receipt_digest"],
    }
    if value != expected:
        raise ValueError("content_agents_preallocation_receipt_invalid")
    return value


def bind_prior_content_agents_preallocation_attempts(
    paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """Return one complete linear predecessor chain for the next authority."""

    entries: list[dict[str, Any]] = []
    authority_digests: list[str] = []
    launch_ids: list[str] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        zero = validate_content_agents_preallocation_provider_zero(path)
        authority_path = _bound_record(
            (zero.get("records") or {}).get("attempt_authority"),
            "content_agents_preallocation_authority_unbound",
        )[0]
        authority = _read(
            authority_path, "content_agents_preallocation_authority_invalid"
        )
        declared = authority.get("prior_preallocation_attempts") or []
        declared_digests = [
            row.get("attempt_authority_digest")
            for row in declared
            if isinstance(row, Mapping)
        ]
        if (
            len(declared_digests) != len(declared)
            or declared_digests != authority_digests
        ):
            raise ValueError("content_agents_preallocation_lineage_invalid")
        entry = {
            "receipt": {
                **_record(path),
                "receipt_digest": zero["receipt_digest"],
            },
            "attempt_authority_digest": zero["attempt_authority_digest"],
            "launch_id": zero["launch_id"],
            "run_id": zero["run_id"],
            "request_digest": zero["request_digest"],
            "official_cost_usd": 0.0,
        }
        entries.append(entry)
        authority_digests.append(str(zero["attempt_authority_digest"]))
        launch_ids.append(str(zero["launch_id"]))
    if (
        len(authority_digests) != len(set(authority_digests))
        or len(launch_ids) != len(set(launch_ids))
    ):
        raise ValueError("content_agents_preallocation_lineage_duplicate")
    return entries


def validate_bound_prior_content_agents_preallocation_attempts(
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    declared = authority.get("prior_preallocation_attempts", [])
    if not isinstance(declared, list):
        raise ValueError("content_agents_preallocation_lineage_invalid")
    paths: list[Path] = []
    for row in declared:
        if not isinstance(row, Mapping):
            raise ValueError("content_agents_preallocation_lineage_invalid")
        paths.append(
            _bound_record(
                row.get("receipt"), "content_agents_preallocation_receipt_unbound"
            )[0]
        )
    expected = bind_prior_content_agents_preallocation_attempts(paths)
    if declared != expected:
        raise ValueError("content_agents_preallocation_lineage_invalid")
    ordinal = authority.get("preallocation_attempt_ordinal", 1)
    if ordinal != len(expected) + 1:
        raise ValueError("content_agents_preallocation_ordinal_invalid")
    return expected


def claim_content_agents_preallocation_successor(
    *, predecessor_authority_digest: str, successor_authority_digest: str
) -> None:
    """Prevent two issued successors from branching from the same zero closeout."""

    root = consumption_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stat = root.stat()
    if root.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError("content_agents_preallocation_lineage_root_insecure")
    destination = root / (
        "content-agents-preallocation-successor-"
        f"{predecessor_authority_digest.removeprefix('sha256:')}.json"
    )
    record = {
        "schema_version": "content_agents_preallocation_successor_claim.v1",
        "predecessor_authority_digest": predecessor_authority_digest,
        "successor_authority_digest": successor_authority_digest,
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _read(
                destination, "content_agents_preallocation_successor_claim_invalid"
            )
            if existing != record:
                raise ValueError("content_agents_preallocation_successor_already_claimed")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--allocator-result", required=True)
    parser.add_argument("--watchdog-handoff", required=True)
    parser.add_argument("--object-store-cleanup", required=True)
    parser.add_argument("--fresh-global-guard", required=True)
    parser.add_argument("--webapp-sync", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = materialize_content_agents_preallocation_provider_zero(
            attempt_authority_path=args.attempt_authority,
            allocator_result_path=args.allocator_result,
            watchdog_handoff_path=args.watchdog_handoff,
            object_store_cleanup_path=args.object_store_cleanup,
            fresh_global_guard_path=args.fresh_global_guard,
            webapp_sync_path=args.webapp_sync,
            output_path=args.output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(Path(args.output).expanduser().resolve()),
                "receipt_digest": value["receipt_digest"],
                "provider_allocations_performed": 0,
                "official_cost_usd": 0.0,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "bind_prior_content_agents_preallocation_attempts",
    "claim_content_agents_preallocation_successor",
    "main",
    "materialize_content_agents_preallocation_provider_zero",
    "validate_bound_prior_content_agents_preallocation_attempts",
    "validate_content_agents_preallocation_provider_zero",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
