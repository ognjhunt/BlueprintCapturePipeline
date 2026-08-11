"""Dry-only canonical-allocator preflight for a production launch profile.

This validator exists so a signed website dry run traverses the same immutable
profile and canonical allocator boundary as a future paid run without treating
an older GPU adapter as the ADP-009D policy-evaluation runtime.  It never calls
a provider and it rejects ``execute=True`` unconditionally.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROBE_KIND = "task-evaluation-profile-preflight"
REQUEST_SCHEMA_VERSION = "task_evaluation_allocator_preflight_request.v1"
RELEASE_SCHEMA_VERSION = "task_evaluation_pipeline_release_evidence.v1"
READINESS_SCHEMA_VERSION = "task_evaluation_runtime_readiness.v1"
GUARD_SCHEMA_VERSION = "gpu_spend_guard.v1"
_DIGEST_PREFIX = "sha256:"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"preflight_input_invalid:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"preflight_input_not_object:{path.name}")
    return dict(value)


def _digest(path: Path) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _DIGEST_PREFIX + hashlib.sha256(encoded).hexdigest()


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "digest": _digest(path) if path.is_file() and not path.is_symlink() else None,
        "size_bytes": path.stat().st_size if path.is_file() and not path.is_symlink() else None,
    }


def run_task_evaluation_profile_preflight(
    *,
    request_path: str | Path,
    release_evidence_path: str | Path,
    readiness_receipt_path: str | Path,
    provider_guard_path: str | Path,
    expected_source_commit: str,
    observed_source_commit: str,
    execute: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate immutable launch inputs plus current provider-zero evidence."""

    raw_paths = {
        "request": Path(request_path).expanduser(),
        "release_evidence": Path(release_evidence_path).expanduser(),
        "runtime_readiness": Path(readiness_receipt_path).expanduser(),
        "provider_guard": Path(provider_guard_path).expanduser(),
    }
    paths = {name: path.resolve() for name, path in raw_paths.items()}
    blockers: list[str] = []
    values: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            if raw_paths[name].is_symlink():
                raise ValueError(f"preflight_input_symlink:{name}")
            values[name] = _read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append(f"task_evaluation_preflight_input_invalid:{name}")
            values[name] = {}

    request = values["request"]
    release = values["release_evidence"]
    readiness = values["runtime_readiness"]
    guard = values["provider_guard"]

    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("task_evaluation_preflight_request_schema_invalid")
    if request.get("request_digest") != _canonical_digest(request, "request_digest"):
        blockers.append("task_evaluation_preflight_request_digest_mismatch")
    if request.get("provider") != "vast":
        blockers.append("task_evaluation_preflight_provider_must_be_vast")
    if request.get("retry_cap") != 0:
        blockers.append("task_evaluation_preflight_retry_cap_invalid")
    if request.get("live_execution_authorized") is not False:
        blockers.append("task_evaluation_preflight_must_be_dry_only")

    expected_commit = str(expected_source_commit or "").strip()
    observed_commit = str(observed_source_commit or "").strip()
    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or expected_commit != observed_commit
    ):
        blockers.append("task_evaluation_preflight_source_commit_mismatch")
    if (
        release.get("schema_version") != RELEASE_SCHEMA_VERSION
        or release.get("status") != "passed"
        or release.get("source_commit") != expected_commit
        or release.get("source_ref") != "main"
        or release.get("tracked_state") != "clean"
        or release.get("release_digest")
        != _canonical_digest(release, "release_digest")
    ):
        blockers.append("task_evaluation_preflight_release_evidence_invalid")

    requested_inputs = _mapping(request.get("immutable_inputs"))
    expected_input_paths = {
        "source_bundle_manifest": requested_inputs.get("source_bundle_manifest"),
        "evaluation_run_spec": requested_inputs.get("evaluation_run_spec"),
        "runtime_readiness": requested_inputs.get("runtime_readiness"),
    }
    for name, reference_value in expected_input_paths.items():
        reference = _mapping(reference_value)
        raw_path = str(reference.get("path") or "")
        unresolved_path = Path(raw_path).expanduser() if raw_path else Path("/")
        path = unresolved_path.resolve()
        if (
            not raw_path
            or unresolved_path.is_symlink()
            or not path.is_file()
            or reference.get("digest") != _digest(path)
        ):
            blockers.append(f"task_evaluation_preflight_immutable_input_invalid:{name}")
    readiness_reference = _mapping(expected_input_paths["runtime_readiness"])
    if str(readiness_reference.get("path") or "") != str(paths["runtime_readiness"]):
        blockers.append("task_evaluation_preflight_readiness_path_mismatch")

    readiness_blockers = readiness.get("blockers")
    if (
        readiness.get("schema_version") != READINESS_SCHEMA_VERSION
        or readiness.get("receipt_digest")
        != _canonical_digest(readiness, "receipt_digest")
        or readiness.get("status") not in {"passed", "blocked"}
        or not isinstance(readiness_blockers, list)
        or any(not isinstance(item, str) or not item for item in readiness_blockers)
        or (readiness.get("status") == "passed" and readiness_blockers)
        or (readiness.get("status") == "blocked" and not readiness_blockers)
    ):
        blockers.append("task_evaluation_preflight_runtime_readiness_invalid")
        readiness_blockers = []

    required_providers = request.get("required_provider_zero")
    inventory_results = guard.get("inventory_results")
    inventory_results = inventory_results if isinstance(inventory_results, list) else []
    inventory_by_provider = {
        str(row.get("provider")): row
        for row in inventory_results
        if isinstance(row, Mapping)
    }
    provider_zero_blockers: list[str] = []
    if (
        not isinstance(required_providers, list)
        or not required_providers
        or any(provider not in {"runpod", "vast", "digitalocean"} for provider in required_providers)
    ):
        blockers.append("task_evaluation_preflight_required_providers_invalid")
        required_providers = []
    for provider in required_providers:
        row = _mapping(inventory_by_provider.get(str(provider)))
        if row.get("status") != "succeeded" or row.get("row_count") != 0:
            provider_zero_blockers.append(
                f"task_evaluation_preflight_provider_zero_unverified:{provider}"
            )
    generated_at = _parse_time(guard.get("generated_at"))
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    max_guard_age = request.get("max_guard_age_seconds")
    if (
        guard.get("schema_version") != GUARD_SCHEMA_VERSION
        or not isinstance(max_guard_age, int)
        or isinstance(max_guard_age, bool)
        or max_guard_age <= 0
        or generated_at is None
        or (observed_now - generated_at).total_seconds() > max_guard_age
        or generated_at > observed_now
        or guard.get("live_instance_count") != 0
        or guard.get("total_burn_per_hour_usd") not in {0, 0.0}
    ):
        provider_zero_blockers.append("task_evaluation_preflight_provider_guard_invalid")
    if guard.get("provider_zero_verified") is not True:
        provider_zero_blockers.append(
            "task_evaluation_preflight_provider_zero_not_explicitly_verified"
        )
    blockers.extend(provider_zero_blockers)

    guard_blockers = guard.get("blockers")
    if not isinstance(guard_blockers, list) or any(
        not isinstance(item, str) or not item for item in guard_blockers
    ):
        blockers.append("task_evaluation_preflight_provider_guard_blockers_invalid")
        guard_blockers = []
    live_blockers = sorted(
        set(
            [
                *(str(item) for item in readiness_blockers or []),
                *guard_blockers,
                "task_evaluation_profile_preflight_is_dry_only",
            ]
        )
    )
    if execute:
        blockers.append("task_evaluation_profile_preflight_execute_forbidden")
    result = {
        "schema_version": "task_evaluation_allocator_preflight_result.v1",
        "status": "dry_run_ready" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "profile_id": request.get("profile_id"),
        "source_commit": expected_commit or None,
        "provider": "vast",
        "provider_zero_verified": not provider_zero_blockers,
        "provider_mutation_attempted": False,
        "provider_mutations_performed": 0,
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
        "live_execution_enabled": False,
        "live_execution_blockers": live_blockers,
        "agent_operator_used": False,
        "artifacts": {name: _artifact(path) for name, path in paths.items()},
    }
    return result
