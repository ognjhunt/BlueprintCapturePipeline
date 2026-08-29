"""Single-use paid authority and spend chain for native Arena execution.

The first native Arena attempt follows the zero-closed paired replacement-import
probe.  Every later construction, control, or policy attempt follows the prior
zero-closed native Arena attempt.  Binding only the immediate predecessor keeps
the interface small while its digest-bound authority retains the complete
recursive spend lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .vast_evidence_contracts import valid_vast_provider_zero_api_call
from .native_task_arena_authority_bundle_loader import (
    native_task_arena_bundle_loader as _bundle_loader,
)
from .paid_attempt_authority import (
    bind_lane_prior_spend,
    normalize_active_instance_allowlist,
    validate_bound_lane_prior_spend,
)
from .project_spend_reconciliation import validate_project_spend_reconciliation
from .spend_authority_consumption_root import prepare_consumption_root
from .task_evaluation_immutable_input_resolver import (
    ImmutableInputResolutionError,
    resolve_immutable_input,
)


AUTHORITY_SCHEMA_VERSION = "native_task_arena_paid_attempt_authority.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "native_task_arena_provider_zero.v1"
#: Written when the watchdog was armed but the run ended before allocating.
WATCHDOG_HANDOFF_SCHEMA_VERSION = "vast_independent_watchdog_handoff.v1"
PREALLOCATION_TEARDOWN_SCHEMA_VERSION = (
    "native_task_arena_preallocation_teardown.v1"
)
PREALLOCATION_CLOSEOUT_KIND = "independent_watchdog_not_armed_before_allocation"
PRE_SPEND_CLOSEOUT_KIND = "pre_spend_preflight_blocked_before_allocation"
MAX_PREALLOCATION_API_ZERO_AGE_SECONDS = 300
MAX_INITIAL_PROVIDER_ZERO_AGE_SECONDS = 900
CONSUMPTION_SCHEMA_VERSION = "native_task_arena_authority_consumption.v1"
# Explicitly expanded by the active Task Arena goal owner on 2026-08-24 so the
# controls and two frozen policy candidates can keep using ordinary 24 GB GPU
# offers.  The aggregate ceiling does not weaken the per-attempt contract:
# every authority remains single-use, retry-0, provider-zero-gated, and bounded
# by ``MAX_HARD_CAP_USD`` below.
AGGREGATE_GOAL_SPEND_CAP_USD = 50.0
MAX_HARD_CAP_USD = 2.0
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 14_400


def native_task_arena_attempt_budget_blockers(
    *,
    max_hourly_rate_usd: Any,
    hard_cap_usd: Any,
    hard_ttl_seconds: Any,
) -> tuple[str, ...]:
    """Validate rate, total cap, and TTL without comparing unlike units."""

    blockers: list[str] = []
    numeric_budget = not any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (max_hourly_rate_usd, hard_cap_usd)
    )
    if numeric_budget:
        rate = float(max_hourly_rate_usd)
        cap = float(hard_cap_usd)
        numeric_budget = (
            math.isfinite(rate)
            and math.isfinite(cap)
            and rate > 0
            and 0 < cap <= MAX_HARD_CAP_USD
        )
    if not numeric_budget:
        blockers.append("native_task_arena_budget_invalid")

    ttl_valid = (
        not isinstance(hard_ttl_seconds, bool)
        and isinstance(hard_ttl_seconds, int)
        and MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
    )
    if not ttl_valid:
        blockers.append("native_task_arena_hard_ttl_invalid")

    if numeric_budget and ttl_valid:
        projected_cost = hard_ttl_seconds * rate / 3600
        if not math.isfinite(projected_cost) or projected_cost > cap:
            blockers.append("native_task_arena_runtime_cost_exceeds_hard_cap")
    return tuple(blockers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _lexical_absolute_path(value: Any, code: str) -> Path:
    raw = str(value or "")
    expanded = Path(raw).expanduser()
    if not raw or not expanded.is_absolute():
        raise ValueError(code)
    return Path(os.path.abspath(str(expanded)))


def _recorded_path(record: Mapping[str, Any], code: str) -> Path:
    """Return the source path sealed by one byte-verified record.

    Dispatcher children read digest-named staged snapshots, while closeout
    layout assertions still describe the original attempt tree.  The record
    has already been verified by :func:`_bound_record`, so those assertions
    must use its sealed source identity, not the transient staged filename.
    """

    return _lexical_absolute_path(record.get("path"), code)


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
    try:
        path = resolve_immutable_input(
            str(value.get("path") or ""),
            expected_digest=str(value.get("sha256") or ""),
            expected_size_bytes=value.get("size_bytes"),
        )
    except ImmutableInputResolutionError as exc:
        raise ValueError(code) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path, dict(value)


def _bound_record_matches_observed(
    bound_record: Mapping[str, Any], observed_record: Mapping[str, Any]
) -> bool:
    """Compare a sealed source record with a validator's staged readback.

    Dispatcher children reopen an immutable input through its digest-named
    staged snapshot.  Validators therefore observe the staged filename even
    though the authority must retain the original sealed source identity.
    ``_bound_record`` has already proved that exact source path, size, and
    digest map to the staged bytes, so only the transient readback path may
    differ here; every other field remains exact and fail-closed.
    """

    normalized_observed = dict(observed_record)
    normalized_observed["path"] = bound_record.get("path")
    return dict(bound_record) == normalized_observed


def _finite_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("native_task_arena_terminal_cost_invalid")
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("native_task_arena_terminal_cost_invalid")
    return cost


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one immutable closeout member without replacing existing evidence."""

    ensure_dir(path.parent)
    payload = (json.dumps(dict(value), indent=1, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _validated_api_provider_zero(path: Path) -> dict[str, Any]:
    value = _read(path, "native_task_arena_preallocation_api_zero_invalid")
    if (
        value.get("schema_version") != "adp_paid_provider_zero.v1"
        or value.get("provider") != "vast"
        or value.get("api_confirmed") is not True
        or value.get("global_live_resource_count") != 0
        or value.get("provider_zero") is not True
        or value.get("inventory") != []
        or not valid_vast_provider_zero_api_call(value.get("api_command"))
        or value.get("raw_secret_values_recorded") is not False
        or not isinstance(value.get("stderr_present"), bool)
        or value.get("provider_zero_digest")
        != canonical_digest(value, digest_field="provider_zero_digest")
    ):
        raise ValueError("native_task_arena_preallocation_api_zero_invalid")
    return value


def _aware_time(value: Any, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(code) from exc
    if parsed.tzinfo is None:
        raise ValueError(code)
    return parsed.astimezone(timezone.utc)


def _expected_watchdog_blocker(authority: Mapping[str, Any]) -> str:
    mode = str(authority.get("execution_mode") or "")
    if mode not in {"construction_canary", "controls", "policy", "policy_diagnostic"}:
        raise ValueError("native_task_arena_preallocation_authority_mode_invalid")
    return f"native_task_arena_{mode}_independent_watchdog_not_armed"


def _expected_job_dir(authority: Mapping[str, Any]) -> str:
    return {
        "construction_canary": "arena-construction-job",
        "controls": "arena-controls-job",
        "policy": "arena-policy-job",
        "policy_diagnostic": "arena-policy-diagnostic-job",
    }[str(authority.get("execution_mode") or "")]


def _lower_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _preallocation_cleanup_valid(value: Mapping[str, Any]) -> bool:
    objects = value.get("objects")
    if (
        value.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or value.get("status") != "completed"
        or value.get("blockers") != []
        or value.get("raw_secret_values_recorded") is not False
        or not isinstance(value.get("cleanup_attempts"), int)
        or value.get("cleanup_attempts", 0) < 1
        or not isinstance(objects, list)
        or not objects
        or value.get("exact_object_count") != len(objects)
        or not _lower_hex(value.get("staging_manifest_sha256"), length=64)
        or value.get("all_objects_absent") is not True
        or value.get("signed_url_files_removed") is not True
    ):
        return False
    for row in objects:
        if not isinstance(row, Mapping):
            return False
        absence = row.get("absence")
        if (
            not _lower_hex(row.get("key_sha256"), length=64)
            or not isinstance(absence, Mapping)
            or absence.get("absence_confirmed") is not True
            or absence.get("http_status_code") != 404
            or absence.get("status") != "passed"
            or absence.get("raw_secret_values_recorded") is not False
        ):
            return False
    return True


def _preallocation_attempt_layout_valid(
    *,
    authority: Mapping[str, Any],
    original_path: Path,
    watchdog_path: Path,
    cleanup_path: Path,
    attempt_root_raw: str,
) -> bool:
    if not attempt_root_raw.startswith("/"):
        return False
    attempt_root = Path(attempt_root_raw).resolve()
    suffix = attempt_root.name.removeprefix("attempt_")
    return (
        attempt_root.name.startswith("attempt_")
        and len(suffix) == 3
        and suffix.isdigit()
        and attempt_root.parent.name == "attempts"
        and attempt_root.parent.parent.name == _expected_job_dir(authority)
        and original_path == attempt_root / "adp_arena_vast_result.json"
        and watchdog_path == attempt_root / "vast_independent_watchdog_handoff.json"
        and cleanup_path
        == attempt_root
        / "object_store_staging"
        / "wam_provider_object_store_cleanup.json"
    )


def _validate_preallocation_provider_zero_value(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema_version") != PROVIDER_ZERO_SCHEMA_VERSION
        or value.get("status") != "completed_preallocation_provider_zero"
        or value.get("inventory_scope") != "no_provider_allocation"
        or value.get("provider_zero_confirmed") is not True
        or value.get("estimated_cost_usd") != 0.0
        or value.get("continuing_spend_from_this_run") is not False
        or value.get("all_staged_objects_absent") is not True
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
    ):
        raise ValueError("native_task_arena_preallocation_provider_zero_invalid")
    if not isinstance(value.get("sibling_preallocation_closeouts"), list):
        raise ValueError("native_task_arena_preallocation_provider_zero_invalid")
    return dict(value)


def _validate_pre_spend_closed_chain(
    *,
    authority: Mapping[str, Any],
    result: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> None:
    """Validate a derivative closeout for a block before any provider mutation."""

    _validate_preallocation_provider_zero_value(zero)
    authority_path, _ = _bound_record(
        zero.get("attempt_authority"),
        "native_task_arena_pre_spend_authority_unbound",
    )
    original_path, original_record = _bound_record(
        result.get("original_allocator_result"),
        "native_task_arena_pre_spend_original_result_unbound",
    )
    preflight_path, preflight_record = _bound_record(
        result.get("pre_spend_preflight"),
        "native_task_arena_pre_spend_preflight_unbound",
    )
    consumption_path, consumption_record = _bound_record(
        result.get("authority_consumption_record"),
        "native_task_arena_pre_spend_consumption_unbound",
    )
    api_zero_path, _ = _bound_record(
        zero.get("api_provider_zero"),
        "native_task_arena_pre_spend_api_zero_unbound",
    )
    teardown_path, _ = _bound_record(
        zero.get("teardown"),
        "native_task_arena_pre_spend_teardown_unbound",
    )
    original = _read(original_path, "native_task_arena_pre_spend_result_invalid")
    preflight = _read(preflight_path, "native_task_arena_pre_spend_preflight_invalid")
    consumption = _read(
        consumption_path, "native_task_arena_pre_spend_consumption_invalid"
    )
    api_zero = _validated_api_provider_zero(api_zero_path)
    teardown = _read(teardown_path, "native_task_arena_pre_spend_teardown_invalid")
    recorded_original_path = _recorded_path(
        original_record, "native_task_arena_pre_spend_original_result_unbound"
    )
    recorded_preflight_path = _recorded_path(
        preflight_record, "native_task_arena_pre_spend_preflight_unbound"
    )
    recorded_consumption_path = _recorded_path(
        consumption_record, "native_task_arena_pre_spend_consumption_unbound"
    )
    attempt_root_raw = str(original.get("attempt_root") or "")
    if not attempt_root_raw.startswith("/"):
        raise ValueError("native_task_arena_pre_spend_closeout_invalid")
    attempt_root = Path(attempt_root_raw).resolve()
    expected_blocker = next(
        (
            str(item)
            for item in original.get("blockers") or []
            if str(item).endswith("_pre_spend_preflight_not_passed")
        ),
        "",
    )
    observed_preflight_blockers = [
        str(item) for item in preflight.get("blockers") or [] if str(item)
    ]
    expected_blockers = sorted({expected_blocker, *observed_preflight_blockers})
    reason = (
        observed_preflight_blockers[0]
        if len(observed_preflight_blockers) == 1
        else expected_blocker
    )
    expected_visual = {
        "status": "unavailable_before_first_observation",
        "media_gap": {"type": "before_first_observation", "reason": reason},
    }
    result_at = _aware_time(
        original.get("generated_at"), "native_task_arena_pre_spend_result_time_invalid"
    )
    zero_at = _aware_time(
        api_zero.get("observed_at_utc"),
        "native_task_arena_pre_spend_api_zero_time_invalid",
    )
    closeout_at = _aware_time(
        zero.get("generated_at"),
        "native_task_arena_pre_spend_closeout_time_invalid",
    )
    provider_run = attempt_root / "vast_provider_run"
    if (
        _read(authority_path, "native_task_arena_pre_spend_authority_invalid")
        != dict(authority)
        or result.get("closeout_kind") != PRE_SPEND_CLOSEOUT_KIND
        or result.get("status") != "sealed_blocked_attempt"
        or result.get("estimated_cost_usd") != 0.0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or result.get("candidate_policy_queried") is not False
        or result.get("scientific_attempt_started") is not False
        or result.get("visual_evidence") != expected_visual
        or result.get("receipt_digest")
        != canonical_digest(result, digest_field="receipt_digest")
        or original.get("schema_version") != "native_task_arena_vast_run.v1"
        or original.get("status") != "blocked"
        or original.get("blockers") != expected_blockers
        or original.get("provider_mutations_performed") != 0
        or original.get("estimated_cost_usd") is not None
        or original.get("continuing_spend_from_this_run") is not None
        or original.get("instance_id") not in (None, "")
        or original.get("vast_instance_ids") not in (None, [], ())
        or original.get("provider_instance_ids") not in (None, [], ())
        or original.get("provider_create_attempted") not in (None, False)
        or not expected_blocker
        or preflight.get("schema_version") != "pre_spend_preflight.v1"
        or preflight.get("status") != "FAIL"
        or preflight.get("spend_allowed") is not False
        or not observed_preflight_blockers
        or original.get("pre_spend_preflight") != preflight
        or consumption.get("schema_version") != CONSUMPTION_SCHEMA_VERSION
        or consumption.get("authorization_digest")
        != authority.get("authorization_digest")
        or consumption.get("bundle_sha256") != authority.get("bundle_sha256")
        or consumption.get("blueprint_commit") != authority.get("blueprint_commit")
        or consumption.get("maximum_provider_allocations") != 1
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or recorded_original_path != attempt_root / "adp_arena_vast_result.json"
        or recorded_preflight_path != provider_run / "pre_spend_preflight.json"
        or recorded_consumption_path.name
        != f"native-task-arena-{str(authority.get('authorization_digest'))[7:]}.json"
        or attempt_root.name != "attempt_001"
        or attempt_root.parent.name != "attempts"
        or attempt_root.parent.parent.name != _expected_job_dir(authority)
        or (provider_run / "vast_provider_adapter_result.json").exists()
        or (provider_run / "vast_teardown_manifest.json").exists()
        or (attempt_root / "object_store_staging").exists()
        or (attempt_root / "vast_independent_watchdog_handoff.json").exists()
        or teardown.get("schema_version") != PREALLOCATION_TEARDOWN_SCHEMA_VERSION
        or teardown.get("status") != "not_required_pre_spend_preflight_blocked"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("provider_mutations_performed") != 0
        or teardown.get("vast_instance_ids") != []
        or zero_at < result_at
        or closeout_at < zero_at
        or (closeout_at - zero_at).total_seconds()
        > MAX_PREALLOCATION_API_ZERO_AGE_SECONDS
    ):
        raise ValueError("native_task_arena_pre_spend_closeout_invalid")


def _validate_preallocation_closed_chain(
    *,
    authority: Mapping[str, Any],
    result: Mapping[str, Any],
    zero: Mapping[str, Any],
    allow_siblings: bool = True,
) -> None:
    _validate_preallocation_provider_zero_value(zero)
    zero_authority_path, _zero_authority_record = _bound_record(
        zero.get("attempt_authority"),
        "native_task_arena_preallocation_authority_unbound",
    )
    original_path, original_record = _bound_record(
        result.get("original_allocator_result"),
        "native_task_arena_preallocation_original_result_unbound",
    )
    watchdog_path, watchdog_record = _bound_record(
        result.get("watchdog_handoff"),
        "native_task_arena_preallocation_watchdog_unbound",
    )
    cleanup_path, cleanup_record = _bound_record(
        result.get("object_store_cleanup"),
        "native_task_arena_preallocation_cleanup_unbound",
    )
    teardown_path, teardown_record = _bound_record(
        zero.get("teardown"), "native_task_arena_preallocation_teardown_unbound"
    )
    _zero_result_path, zero_result_record = _bound_record(
        zero.get("terminal_result"),
        "native_task_arena_preallocation_terminal_result_unbound",
    )
    api_zero_path, api_zero_record = _bound_record(
        zero.get("api_provider_zero"),
        "native_task_arena_preallocation_api_zero_unbound",
    )
    original = _read(
        original_path, "native_task_arena_preallocation_original_result_invalid"
    )
    watchdog = _read(
        watchdog_path, "native_task_arena_preallocation_watchdog_invalid"
    )
    cleanup = _read(
        cleanup_path, "native_task_arena_preallocation_cleanup_invalid"
    )
    teardown = _read(
        teardown_path, "native_task_arena_preallocation_teardown_invalid"
    )
    zero_authority = _read(
        zero_authority_path, "native_task_arena_preallocation_authority_invalid"
    )
    api_zero = _validated_api_provider_zero(api_zero_path)
    authorization_digest = authority.get("authorization_digest")
    result_at = _aware_time(
        original.get("generated_at"),
        "native_task_arena_preallocation_result_time_invalid",
    )
    zero_at = _aware_time(
        api_zero.get("observed_at_utc"),
        "native_task_arena_preallocation_api_zero_time_invalid",
    )
    closeout_at = _aware_time(
        zero.get("generated_at"),
        "native_task_arena_preallocation_closeout_time_invalid",
    )
    expected_times = {
        "allocator_result_generated_at": result_at.isoformat(),
        "api_provider_zero_observed_at": zero_at.isoformat(),
        "closeout_generated_at": closeout_at.isoformat(),
        "maximum_api_zero_age_seconds": MAX_PREALLOCATION_API_ZERO_AGE_SECONDS,
    }
    attempt_root_raw = str(original.get("attempt_root") or "")
    sibling_records = zero.get("sibling_preallocation_closeouts")
    recorded_original_path = _recorded_path(
        original_record, "native_task_arena_preallocation_original_result_unbound"
    )
    recorded_watchdog_path = _recorded_path(
        watchdog_record, "native_task_arena_preallocation_watchdog_unbound"
    )
    recorded_cleanup_path = _recorded_path(
        cleanup_record, "native_task_arena_preallocation_cleanup_unbound"
    )
    recorded_teardown_path = _recorded_path(
        teardown_record, "native_task_arena_preallocation_teardown_unbound"
    )
    recorded_zero_result_path = _recorded_path(
        zero_result_record,
        "native_task_arena_preallocation_terminal_result_unbound",
    )
    expected_visual_evidence = {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": _expected_watchdog_blocker(authority),
        },
    }
    if (
        result.get("closeout_kind") != PREALLOCATION_CLOSEOUT_KIND
        or zero_authority != dict(authority)
        or result.get("status") != "sealed_blocked_attempt"
        or result.get("estimated_cost_usd") != 0.0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or result.get("visual_evidence") != expected_visual_evidence
        or result.get("receipt_digest")
        != canonical_digest(result, digest_field="receipt_digest")
        or original.get("schema_version") != "native_task_arena_vast_run.v1"
        or original.get("status") != "blocked"
        or original.get("blockers") != [_expected_watchdog_blocker(authority)]
        or original.get("provider_mutations_performed") != 0
        or original.get("all_staged_objects_absent") is not True
        or original.get("estimated_cost_usd") is not None
        or original.get("continuing_spend_from_this_run") is not None
        or original.get("retry_cap") != 0
        or original.get("authorization_consumption", {}).get("status") != "consumed"
        or original.get("authorization_consumption", {}).get("authorization_digest")
        != authorization_digest
        or original.get("instance_id") not in (None, "")
        or original.get("vast_instance_ids") not in (None, [], ())
        or original.get("provider_instance_ids") not in (None, [], ())
        or original.get("provider_create_attempted") not in (None, False)
        or not _preallocation_attempt_layout_valid(
            authority=authority,
            original_path=recorded_original_path,
            watchdog_path=recorded_watchdog_path,
            cleanup_path=recorded_cleanup_path,
            attempt_root_raw=attempt_root_raw,
        )
        or original.get("independent_watchdog") != watchdog
        or watchdog.get("schema_version") != WATCHDOG_HANDOFF_SCHEMA_VERSION
        or watchdog.get("status") != "blocked"
        or watchdog.get("watchdog_armed_before_allocation") is not False
        or watchdog.get("independent_process") is not False
        or watchdog.get("provider_mutations_performed") != 0
        or not _preallocation_cleanup_valid(cleanup)
        or teardown.get("schema_version") != PREALLOCATION_TEARDOWN_SCHEMA_VERSION
        or teardown.get("status")
        != "not_required_independent_watchdog_not_armed_before_allocation"
        or teardown.get("vast_instance_ids") != []
        or teardown.get("provider_mutations_performed") != 0
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("receipt_digest")
        != canonical_digest(teardown, digest_field="receipt_digest")
        or teardown.get("original_allocator_result") != original_record
        or teardown.get("watchdog_handoff") != watchdog_record
        or _lexical_absolute_path(
            result.get("teardown_manifest_path"),
            "native_task_arena_preallocation_teardown_unbound",
        )
        != recorded_teardown_path
        or zero.get("attempt_authority_digest") != authorization_digest
        or zero.get("evidence_times") != expected_times
        or zero_at < result_at
        or zero_at > closeout_at
        or (closeout_at - zero_at).total_seconds()
        > MAX_PREALLOCATION_API_ZERO_AGE_SECONDS
        or _lexical_absolute_path(
            result.get("closeout_path"),
            "native_task_arena_preallocation_terminal_result_unbound",
        )
        != recorded_zero_result_path
        or zero.get("teardown") != teardown_record
        or zero.get("watchdog") != watchdog_record
        or zero.get("object_store_cleanup") != cleanup_record
        or zero.get("api_provider_zero") != api_zero_record
        or not isinstance(sibling_records, list)
        or (bool(sibling_records) and not allow_siblings)
    ):
        raise ValueError("native_task_arena_preallocation_closeout_invalid")

    sibling_digests: set[str] = set()
    for record in sibling_records:
        sibling_path, _ = _bound_record(
            record, "native_task_arena_preallocation_sibling_unbound"
        )
        sibling_zero = _read(
            sibling_path, "native_task_arena_preallocation_sibling_invalid"
        )
        _validate_preallocation_provider_zero_value(sibling_zero)
        sibling_authority_path, _ = _bound_record(
            sibling_zero.get("attempt_authority"),
            "native_task_arena_preallocation_sibling_authority_unbound",
        )
        sibling_result_path, _ = _bound_record(
            sibling_zero.get("terminal_result"),
            "native_task_arena_preallocation_sibling_result_unbound",
        )
        sibling_authority = _read(
            sibling_authority_path,
            "native_task_arena_preallocation_sibling_authority_invalid",
        )
        sibling_result = _read(
            sibling_result_path,
            "native_task_arena_preallocation_sibling_result_invalid",
        )
        sibling_digest = str(sibling_zero.get("attempt_authority_digest") or "")
        if (
            not sibling_digest
            or sibling_digest == authorization_digest
            or sibling_digest in sibling_digests
        ):
            raise ValueError("native_task_arena_preallocation_sibling_invalid")
        sibling_digests.add(sibling_digest)
        _validate_preallocation_closed_chain(
            authority=sibling_authority,
            result=sibling_result,
            zero=sibling_zero,
            allow_siblings=False,
        )


#: Authority schema of an accepted predecessor -> the terminal result schema
#: that predecessor writes.  Every value here must also be readable by the
#: official-billing extractor: a chained authority requires the predecessor's
#: reconciled posted charges, so a predecessor whose result schema the
#: extractor does not accept can close cleanly and still block every later
#: attempt.  ``tests/test_native_task_arena_paid_authority.py`` pins that.
PREDECESSOR_RESULT_SCHEMAS: Mapping[str, str] = {
    "paired_target_native_import_paid_attempt_authority.v1": (
        "paired_target_native_import_vast_run.v1"
    ),
    AUTHORITY_SCHEMA_VERSION: "native_task_arena_vast_run.v1",
}


#: Provider-zero statuses a chained authority accepts from its predecessor.
#: A recovered zero is a full zero -- it proves the same absence from fresher
#: evidence -- so a lane that had to recover is still chainable. Admitting only
#: ``completed`` here would have made every recovered attempt a dead end,
#: silently, on both this lane and the import lane that has had a recovered
#: seal all along.
ACCEPTED_PREDECESSOR_ZERO_STATUSES = frozenset(
    {
        "completed",
        "completed_recovered_provider_zero",
        "completed_preallocation_provider_zero",
    }
)


def validate_terminal_spend_chain(
    *, authority_path: str | Path, result_path: str | Path, provider_zero_path: str | Path
) -> dict[str, Any]:
    """Validate one immediate terminal predecessor and return cumulative spend."""

    authority_file = Path(authority_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    zero_file = Path(provider_zero_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_predecessor_authority_invalid")
    result = _read(result_file, "native_task_arena_predecessor_result_invalid")
    zero = _read(zero_file, "native_task_arena_predecessor_provider_zero_invalid")
    schema = authority.get("schema_version")
    if schema == "paired_target_native_import_paid_attempt_authority.v1":
        authority_digest_field = "authorization_digest"
        result_schema = PREDECESSOR_RESULT_SCHEMAS[schema]
        zero_schema = "paired_target_native_import_provider_zero.v1"
        zero_digest_field = "receipt_digest"
        consumption = result.get("authorization_consumption") or {}
        result_authority_digest = consumption.get("authorization_digest")
        zero_authority_digest = zero.get("attempt_authority_digest")
        zero_result_record = zero.get("terminal_result")
    elif schema == AUTHORITY_SCHEMA_VERSION:
        authority_digest_field = "authorization_digest"
        result_schema = PREDECESSOR_RESULT_SCHEMAS[schema]
        zero_schema = PROVIDER_ZERO_SCHEMA_VERSION
        zero_digest_field = "receipt_digest"
        consumption = result.get("authorization_consumption") or {}
        result_authority_digest = consumption.get("authorization_digest")
        zero_authority_digest = zero.get("attempt_authority_digest")
        zero_result_record = zero.get("terminal_result")
    else:
        raise ValueError("native_task_arena_predecessor_authority_schema_invalid")
    authorization_digest = authority.get(authority_digest_field)
    try:
        cost = _finite_cost(result.get("estimated_cost_usd"))
    except ValueError:
        # Older pre-allocation adapter results omitted the explicit 0.0 even
        # though their provider-zero receipt had already proved that no
        # provider allocation occurred.  The self-digesting zero and its
        # terminal-result binding are validated below, so only that exact
        # evidence scope may repair the missing historical scalar.
        if (
            result.get("estimated_cost_usd") is None
            and zero.get("inventory_scope") == "no_provider_allocation"
        ):
            cost = 0.0
        else:
            raise
    before = _finite_cost(authority.get("aggregate_goal_spend_before_attempt_usd"))
    cap = _finite_cost(authority.get("aggregate_goal_spend_cap_usd"))
    zero_result_path, zero_result_bound = _bound_record(
        zero_result_record, "native_task_arena_predecessor_zero_result_unbound"
    )
    preallocation_closeout = (
        schema == AUTHORITY_SCHEMA_VERSION
        and zero.get("status") == "completed_preallocation_provider_zero"
    )
    if preallocation_closeout:
        if result.get("closeout_kind") == PRE_SPEND_CLOSEOUT_KIND:
            _validate_pre_spend_closed_chain(
                authority=authority, result=result, zero=zero
            )
        else:
            _validate_preallocation_closed_chain(
                authority=authority, result=result, zero=zero
            )
    if (
        authorization_digest != canonical_digest(authority, digest_field=authority_digest_field)
        or result.get("schema_version") != result_schema
        or result.get("status")
        not in {"completed", "blocked", "sealed_blocked_attempt"}
        or result_authority_digest != authorization_digest
        or result.get("bundle_sha256") != authority.get("bundle_sha256")
        or result.get("hard_cap_usd") != authority.get("hard_attempt_spend_cap_usd")
        or result.get("hard_ttl_seconds")
        != authority.get("maximum_single_resource_ttl_seconds")
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or zero.get("schema_version") != zero_schema
        or zero.get("status") not in ACCEPTED_PREDECESSOR_ZERO_STATUSES
        or zero.get(zero_digest_field)
        != canonical_digest(zero, digest_field=zero_digest_field)
        or zero_authority_digest != authorization_digest
        or zero.get("provider_zero_confirmed") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("all_staged_objects_absent") is not True
        or zero_result_bound.get("sha256") != _sha256(result_file)
        or before + cost > cap
    ):
        raise ValueError("native_task_arena_terminal_spend_chain_invalid")
    return {
        "authority_digest": authorization_digest,
        "attempt_cost_usd": round(cost, 6),
        "aggregate_goal_spend_before_attempt_usd": round(before, 6),
        "aggregate_goal_spend_after_attempt_usd": round(before + cost, 6),
        "aggregate_goal_spend_cap_usd": cap,
        "records": {
            "authority": _record(authority_file),
            # The zero receipt names the canonical retained terminal result.
            # A caller may supply a byte-identical allocator alias, but the
            # successor always binds the path that provider-zero actually
            # attested.
            "terminal_result": _record(zero_result_path),
            "provider_zero": _record(zero_file),
        },
    }


def _native_policy_campaign_binding(
    *,
    campaign_path: str | Path,
    expected_campaign_record: Mapping[str, Any] | None = None,
    campaign_member_id: str,
    prepared_bundle: Mapping[str, Any],
    blueprint_commit: str,
    prior_spend: float,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int],
) -> dict[str, Any]:
    """Load one member without making the ordinary authority path campaign-aware."""

    from .native_task_arena_policy_campaign import (
        campaign_member,
        load_verified_native_task_arena_policy_campaign,
        verify_native_task_arena_policy_campaign_bundles,
    )

    campaign, campaign_record = load_verified_native_task_arena_policy_campaign(
        campaign_path,
        expected_blueprint_commit=blueprint_commit,
    )
    member, sibling = campaign_member(campaign, member_id=campaign_member_id)
    try:
        verify_native_task_arena_policy_campaign_bundles(
            campaign, expected_blueprint_commit=blueprint_commit
        )
    except ValueError as exc:
        raise ValueError(
            "native_task_arena_policy_campaign_member_binding_invalid"
        ) from exc
    controls_ids = tuple(campaign["controls_allowed_active_instance_ids"])
    if (
        prepared_bundle.get("execution_mode") not in {"policy", "policy_diagnostic"}
        or member.get("candidate_id") != prepared_bundle.get("policy_candidate_id")
        or member.get("execution_mode") != prepared_bundle.get("execution_mode")
        or member.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or member.get("bundle_input_digest") != prepared_bundle.get("input_digest")
        or member.get("blueprint_commit") != prepared_bundle.get("implementation_commit")
        or member.get("maximum_hourly_rate_usd") != max_hourly_rate_usd
        or member.get("hard_attempt_spend_cap_usd") != hard_cap_usd
        or member.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds
        or campaign.get("prior_official_spend", {}).get(
            "aggregate_goal_spend_before_campaign_usd"
        )
        != prior_spend
        or controls_ids != tuple(sorted(set(allowed_active_instance_ids)))
    ):
        raise ValueError("native_task_arena_policy_campaign_member_binding_invalid")
    return {
        "campaign": (
            dict(expected_campaign_record)
            if expected_campaign_record is not None
            else campaign_record
        ),
        "campaign_id": campaign["campaign_id"],
        "campaign_digest": campaign["campaign_digest"],
        "member_id": member["member_id"],
        "launch_id": member["launch_id"],
        "resource_name": member["resource_name"],
        "sibling_member_id": sibling["member_id"],
        "sibling_launch_id": sibling["launch_id"],
        "sibling_resource_name": sibling["resource_name"],
    }


def materialize_native_task_arena_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    prior_authority_path: str | Path | None = None,
    prior_result_path: str | Path | None = None,
    prior_provider_zero_path: str | Path | None = None,
    prior_spend_reconciliation_path: str | Path | None = None,
    project_spend_reconciliation_path: str | Path | None = None,
    initial_provider_zero_path: str | Path | None = None,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
    supplemental_prior_result_paths: Sequence[str | Path] = (),
    allowed_active_instance_ids: Sequence[int] = (),
    retain_warm_session: bool = False,
    policy_campaign_path: str | Path | None = None,
    campaign_member_id: str | None = None,
) -> dict[str, Any]:
    """Seal one zero-retry authority for a new or continuing native lane.

    A continuing lane binds one exact terminal predecessor. A brand-new lane
    instead binds a complete project-spend reconciliation plus a fresh global
    provider-zero receipt, so it never borrows another lane's launch authority.
    """

    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    raw_bundle = _read(receipt_file, "native_task_arena_bundle_receipt_invalid")
    mode = str(raw_bundle.get("execution_mode") or "")
    bundle = _bundle_loader(mode)(
        receipt_file,
        expected_implementation_commit=blueprint_commit,
        expected_packet_receipt_digest=raw_bundle.get("packet_receipt_digest"),
        expected_runtime_source_packet_digest=(raw_bundle.get("runtime_source_packet") or {}).get(
            "receipt_digest"
        ),
    )
    terminal_inputs = (
        prior_authority_path,
        prior_result_path,
        prior_provider_zero_path,
        prior_spend_reconciliation_path,
    )
    initial_inputs = (project_spend_reconciliation_path, initial_provider_zero_path)
    terminal_mode = all(value is not None for value in terminal_inputs)
    terminal_inputs_present = any(value is not None for value in terminal_inputs)
    initial_mode = not terminal_inputs_present and all(
        value is not None for value in initial_inputs
    )
    if (
        terminal_mode == initial_mode
        or terminal_inputs_present != terminal_mode
        or (initial_provider_zero_path is not None and not initial_mode)
        or (
            not terminal_mode
            and any(value is not None for value in initial_inputs) != initial_mode
        )
        or (initial_mode and supplemental_prior_result_paths)
    ):
        raise ValueError("native_task_arena_authority_lineage_invalid")
    prior: dict[str, Any] | None = None
    reconciled: dict[str, Any] | None = None
    main_actual_provider_charge: float | None = None
    project_spend_record: dict[str, Any] | None = None
    initial_zero_record: dict[str, Any] | None = None
    if terminal_mode:
        prior = validate_terminal_spend_chain(
            authority_path=str(prior_authority_path),
            result_path=str(prior_result_path),
            provider_zero_path=str(prior_provider_zero_path),
        )
        prior_result_paths = (
            prior["records"]["terminal_result"]["path"],
            *(
                str(Path(item).expanduser().resolve())
                for item in supplemental_prior_result_paths
            ),
        )
        if len(prior_result_paths) != len(set(prior_result_paths)):
            raise ValueError("native_task_arena_prior_result_duplicate")
        reconciled = bind_lane_prior_spend(
            prior_result_paths=prior_result_paths,
            reconciliation_path=prior_spend_reconciliation_path,
            lane="native_task_arena",
        )
        main_result_sha256 = prior["records"]["terminal_result"]["sha256"]
        main_rows = [
            row
            for row in reconciled["prior_terminal_attempts"]
            if (
                row.get("result_sha256")
                or (row.get("result") or {}).get("sha256")
            )
            == main_result_sha256
        ]
        if len(main_rows) != 1:
            raise ValueError("native_task_arena_primary_prior_reconciliation_invalid")
        main_actual_provider_charge = main_rows[0].get(
            "actual_provider_charge_usd"
        )
        if main_actual_provider_charge is None:
            if len(reconciled["prior_terminal_attempts"]) != 1:
                raise ValueError(
                    "native_task_arena_primary_prior_reconciliation_invalid"
                )
            main_actual_provider_charge = reconciled["actual_total_usd"]
        prior_spend = round(
            prior["aggregate_goal_spend_before_attempt_usd"]
            + reconciled["actual_total_usd"],
            6,
        )
        if project_spend_reconciliation_path is not None:
            project_path = Path(
                str(project_spend_reconciliation_path)
            ).expanduser().resolve()
            project_spend, project_spend_record = (
                validate_project_spend_reconciliation(project_path)
            )
            project_total = round(float(project_spend["total_cost_usd"]), 6)
            if project_total < prior_spend:
                raise ValueError("native_task_arena_project_spend_stale")
            prior_spend = project_total
    else:
        project_path = Path(
            str(project_spend_reconciliation_path)
        ).expanduser().resolve()
        project_spend, project_spend_record = validate_project_spend_reconciliation(
            project_path
        )
        zero_path = Path(str(initial_provider_zero_path)).expanduser().resolve()
        initial_zero = _validated_api_provider_zero(zero_path)
        authorized_time = _aware_time(
            authorized_on, "native_task_arena_initial_authorized_time_invalid"
        )
        zero_time = _aware_time(
            initial_zero.get("observed_at_utc"),
            "native_task_arena_initial_provider_zero_time_invalid",
        )
        if (
            zero_time > authorized_time
            or (authorized_time - zero_time).total_seconds()
            > MAX_INITIAL_PROVIDER_ZERO_AGE_SECONDS
            or allowed_active_instance_ids
            or retain_warm_session
            or policy_campaign_path is not None
            or campaign_member_id is not None
        ):
            raise ValueError("native_task_arena_initial_authority_invalid")
        initial_zero_record = {
            **_record(zero_path),
            "provider_zero_digest": initial_zero["provider_zero_digest"],
        }
        prior_spend = round(float(project_spend["total_cost_usd"]), 6)
    allowed = tuple(sorted({int(value) for value in allowed_active_instance_ids}))
    # A newly deployed program ceiling is itself the authority boundary.  The
    # predecessor's lower historical ceiling remains bound in its immutable
    # receipt, but must not make an explicitly raised current ceiling
    # impossible to issue: taking ``min`` here permanently froze the chain at
    # its first value even after the owner expanded the active goal budget.
    aggregate_cap = AGGREGATE_GOAL_SPEND_CAP_USD
    campaign_inputs_complete = (
        policy_campaign_path is not None and campaign_member_id is not None
    )
    campaign_inputs_partial = (policy_campaign_path is None) != (
        campaign_member_id is None
    )
    budget_blockers = native_task_arena_attempt_budget_blockers(
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
    )
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or budget_blockers
        or prior_spend + hard_cap_usd > aggregate_cap
        or any(value <= 0 for value in allowed)
        or (
            retain_warm_session
            and mode not in {"construction_canary", "controls"}
        )
        or campaign_inputs_partial
    ):
        raise ValueError("native_task_arena_authority_configuration_invalid")
    campaign_binding = (
        _native_policy_campaign_binding(
            campaign_path=policy_campaign_path,
            campaign_member_id=str(campaign_member_id),
            prepared_bundle=bundle,
            blueprint_commit=blueprint_commit,
            prior_spend=prior_spend,
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
            allowed_active_instance_ids=allowed,
        )
        if campaign_inputs_complete
        else None
    )
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": "one_shot_native_task_arena_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "retain_warm_session": bool(retain_warm_session),
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_receipt": _record(receipt_file),
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_input_digest": bundle["input_digest"],
        "packet_receipt_digest": bundle["packet_receipt_digest"],
        "runtime_source_packet_receipt_digest": bundle["runtime_source_packet"][
            "receipt_digest"
        ],
        "execution_mode": mode,
        "policy_candidate_id": bundle.get("policy_candidate_id"),
        "blueprint_commit": blueprint_commit,
        "container_image": bundle["container_image"],
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": prior_spend,
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "lineage_kind": (
            "terminal_predecessor" if terminal_mode else "project_spend_genesis"
        ),
        **(
            {
                "prior_terminal_attempt": {
                    **prior["records"],
                    "authority_digest": prior["authority_digest"],
                    "attempt_cost_usd": prior["attempt_cost_usd"],
                    "actual_provider_charge_usd": main_actual_provider_charge,
                },
                "prior_terminal_attempts": reconciled["prior_terminal_attempts"],
                "prior_spend_reconciliation": reconciled["reconciliation"],
                "prior_actual_provider_spend_usd": reconciled["actual_total_usd"],
                **(
                    {"project_spend_reconciliation": project_spend_record}
                    if project_spend_record is not None
                    else {}
                ),
            }
            if terminal_mode
            else {
                "project_spend_reconciliation": project_spend_record,
                "initial_provider_zero": initial_zero_record,
                "prior_terminal_attempts": [],
                "prior_actual_provider_spend_usd": prior_spend,
            }
        ),
        "active_instance_allowlist": {
            "external_provider_owned": list(allowed),
            "same_goal_concurrent": [],
        },
        **(
            {"policy_campaign_binding": campaign_binding}
            if campaign_binding is not None
            else {}
        ),
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("native_task_arena_authority_output_exists")
    ensure_dir(output.parent)
    write_json(output, authority)
    validate_native_task_arena_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowed,
        retain_warm_session=retain_warm_session,
    )
    return authority


def validate_native_task_arena_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int] = (),
    retain_warm_session: bool = False,
) -> dict[str, Any]:
    value = dict(authority)
    expected_allowlist = {
        "external_provider_owned": tuple(sorted(set(allowed_active_instance_ids))),
        "same_goal_concurrent": (),
    }
    observed_allowlist = normalize_active_instance_allowlist(value.get("active_instance_allowlist"))
    errors = list(
        native_task_arena_attempt_budget_blockers(
            max_hourly_rate_usd=max_hourly_rate_usd,
            hard_cap_usd=hard_cap_usd,
            hard_ttl_seconds=hard_ttl_seconds,
        )
    )
    expected = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "purpose": "one_shot_native_task_arena_execution",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "bundle_sha256": prepared_bundle.get("bundle_sha256"),
        "bundle_input_digest": prepared_bundle.get("input_digest"),
        "packet_receipt_digest": prepared_bundle.get("packet_receipt_digest"),
        "runtime_source_packet_receipt_digest": (
            prepared_bundle.get("runtime_source_packet") or {}
        ).get("receipt_digest"),
        "execution_mode": prepared_bundle.get("execution_mode"),
        "policy_candidate_id": prepared_bundle.get("policy_candidate_id"),
        "blueprint_commit": prepared_bundle.get("implementation_commit"),
        "container_image": prepared_bundle.get("container_image"),
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "raw_nonredistributable_bytes_uploaded": False,
        "canonical_interiorgs_uploaded_or_mutated": False,
        "simulator_output_is_not_physical_evidence": True,
    }
    errors.extend(
        f"{key}_mismatch" for key, expected_value in expected.items() if value.get(key) != expected_value
    )
    observed_retain = value.get("retain_warm_session", False)
    if not isinstance(observed_retain, bool) or observed_retain != retain_warm_session:
        errors.append("retain_warm_session_mismatch")
    if value.get("authorization_digest") != canonical_digest(
        value, digest_field="authorization_digest"
    ):
        errors.append("digest_invalid")
    if observed_allowlist != expected_allowlist:
        errors.append("active_instance_allowlist_mismatch")
    actual_after: float | None = None
    try:
        _receipt_path, receipt_record = _bound_record(
            value.get("bundle_receipt"), "bundle_receipt_unbound"
        )
        expected_receipt_path = (
            Path(str(prepared_bundle.get("bundle_path"))).parent
            / "native_task_arena_provider_bundle_receipt.v1.json"
        )
        recorded_receipt_path = Path(
            os.path.abspath(str(Path(str(receipt_record.get("path") or "")).expanduser()))
        )
        if recorded_receipt_path != expected_receipt_path:
            errors.append("bundle_receipt_path_mismatch")
        lineage_kind = value.get("lineage_kind", "terminal_predecessor")
        if lineage_kind == "project_spend_genesis":
            if (
                value.get("prior_terminal_attempt") is not None
                or value.get("prior_spend_reconciliation") is not None
                or value.get("prior_terminal_attempts") != []
                or retain_warm_session
                or allowed_active_instance_ids
                or value.get("policy_campaign_binding") is not None
            ):
                raise ValueError("genesis_shape_invalid")
            project_path, project_record = _bound_record(
                value.get("project_spend_reconciliation"),
                "project_spend_reconciliation_unbound",
            )
            project_spend, observed_project_record = (
                validate_project_spend_reconciliation(project_path)
            )
            zero_path, zero_record = _bound_record(
                value.get("initial_provider_zero"),
                "initial_provider_zero_unbound",
            )
            initial_zero = _validated_api_provider_zero(zero_path)
            authorized_time = _aware_time(
                value.get("authorized_on"),
                "native_task_arena_initial_authorized_time_invalid",
            )
            zero_time = _aware_time(
                initial_zero.get("observed_at_utc"),
                "native_task_arena_initial_provider_zero_time_invalid",
            )
            actual_after = round(float(project_spend["total_cost_usd"]), 6)
            if (
                project_record != observed_project_record
                or zero_record.get("provider_zero_digest")
                != initial_zero.get("provider_zero_digest")
                or zero_time > authorized_time
                or (authorized_time - zero_time).total_seconds()
                > MAX_INITIAL_PROVIDER_ZERO_AGE_SECONDS
                or value.get("prior_actual_provider_spend_usd") != actual_after
                or value.get("aggregate_goal_spend_before_attempt_usd")
                != actual_after
            ):
                errors.append("project_spend_genesis_mismatch")
        elif lineage_kind == "terminal_predecessor":
            predecessor = value.get("prior_terminal_attempt")
            if not isinstance(predecessor, Mapping):
                raise ValueError("predecessor_invalid")
            paths = {
                key: _bound_record(predecessor.get(key), "predecessor_unbound")[0]
                for key in ("authority", "terminal_result", "provider_zero")
            }
            prior = validate_terminal_spend_chain(
                authority_path=paths["authority"],
                result_path=paths["terminal_result"],
                provider_zero_path=paths["provider_zero"],
            )
            reconciled = validate_bound_lane_prior_spend(
                value, lane="native_task_arena"
            )
            main_result_sha256 = prior["records"]["terminal_result"]["sha256"]
            main_rows = [
                row
                for row in reconciled["prior_terminal_attempts"]
                if (
                    row.get("result_sha256")
                    or (row.get("result") or {}).get("sha256")
                )
                == main_result_sha256
            ]
            if len(main_rows) != 1:
                raise ValueError("primary_predecessor_reconciliation_invalid")
            main_actual_provider_charge = main_rows[0].get(
                "actual_provider_charge_usd"
            )
            if main_actual_provider_charge is None:
                if len(reconciled["prior_terminal_attempts"]) != 1:
                    raise ValueError("primary_predecessor_reconciliation_invalid")
                main_actual_provider_charge = reconciled["actual_total_usd"]
            terminal_actual_after = round(
                prior["aggregate_goal_spend_before_attempt_usd"]
                + reconciled["actual_total_usd"],
                6,
            )
            actual_after = terminal_actual_after
            project_record_value = value.get("project_spend_reconciliation")
            if project_record_value is not None:
                project_path, project_record = _bound_record(
                    project_record_value,
                    "project_spend_reconciliation_unbound",
                )
                project_spend, observed_project_record = (
                    validate_project_spend_reconciliation(project_path)
                )
                project_total = round(float(project_spend["total_cost_usd"]), 6)
                if (
                    not _bound_record_matches_observed(
                        project_record, observed_project_record
                    )
                    or project_total < terminal_actual_after
                ):
                    raise ValueError("project_spend_continuation_mismatch")
                actual_after = project_total
            if (
                predecessor.get("authority_digest") != prior["authority_digest"]
                or predecessor.get("attempt_cost_usd") != prior["attempt_cost_usd"]
                or predecessor.get("actual_provider_charge_usd")
                != main_actual_provider_charge
                or value.get("aggregate_goal_spend_before_attempt_usd")
                != actual_after
            ):
                errors.append("prior_terminal_spend_mismatch")
        else:
            raise ValueError("lineage_kind_invalid")
        if (
            value.get("aggregate_goal_spend_cap_usd")
            != AGGREGATE_GOAL_SPEND_CAP_USD
            or value.get("aggregate_goal_spend_before_attempt_usd", 0) + hard_cap_usd
            > value.get("aggregate_goal_spend_cap_usd", 0)
        ):
            errors.append("aggregate_goal_spend_mismatch")
    except ValueError:
        errors.append("prior_terminal_spend_invalid")
    campaign_binding = value.get("policy_campaign_binding")
    if campaign_binding is not None:
        try:
            if not isinstance(campaign_binding, Mapping) or actual_after is None:
                raise ValueError("campaign_binding_invalid")
            campaign_path, campaign_record = _bound_record(
                campaign_binding.get("campaign"),
                "policy_campaign_unbound",
            )
            observed_campaign_binding = _native_policy_campaign_binding(
                campaign_path=campaign_path,
                expected_campaign_record=campaign_record,
                campaign_member_id=str(campaign_binding.get("member_id") or ""),
                prepared_bundle=prepared_bundle,
                blueprint_commit=str(prepared_bundle.get("implementation_commit") or ""),
                prior_spend=actual_after,
                max_hourly_rate_usd=max_hourly_rate_usd,
                hard_cap_usd=hard_cap_usd,
                hard_ttl_seconds=hard_ttl_seconds,
                allowed_active_instance_ids=allowed_active_instance_ids,
            )
            if dict(campaign_binding) != observed_campaign_binding:
                errors.append("policy_campaign_binding_mismatch")
        except ValueError:
            errors.append("policy_campaign_binding_invalid")
    if errors:
        raise ValueError("native_task_arena_authority_invalid:" + ",".join(sorted(set(errors))))
    return value


def consume_native_task_arena_authority_once(authority: Mapping[str, Any]) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return {"status": "blocked", "blockers": ["native_task_arena_authority_identity_invalid"]}
    try:
        # The reconciler may have created this owned directory with a group
        # traversal bit (0710). Tighten it through the shared hardened helper
        # before enforcing the single-use record. Refusing the repairable
        # directory here previously blocked every native Task Arena attempt
        # before provider allocation.
        root = prepare_consumption_root()
    except (OSError, ValueError):
        return {
            "status": "blocked",
            "blockers": ["native_task_arena_authority_consumption_failed"],
        }
    payload = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION,
        "authorization_digest": digest,
        "bundle_sha256": authority.get("bundle_sha256"),
        "blueprint_commit": authority.get("blueprint_commit"),
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 1,
        **(
            {
                "policy_campaign_digest": (
                    authority.get("policy_campaign_binding") or {}
                ).get("campaign_digest"),
                "policy_campaign_member_id": (
                    authority.get("policy_campaign_binding") or {}
                ).get("member_id"),
            }
            if authority.get("policy_campaign_binding") is not None
            else {}
        ),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        stat = root.stat()
        if root.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise OSError("insecure_root")
        destination = root / f"native-task-arena-{digest[7:]}.json"
        temporary = root / f".native-task-arena-{digest[7:]}.{os.getpid()}.tmp"
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
        return {"status": "blocked", "blockers": ["native_task_arena_authority_consumed"]}
    except OSError:
        return {"status": "blocked", "blockers": ["native_task_arena_authority_consumption_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


#: A recovered zero may only be sealed against a global-zero guard this fresh.
#: Older evidence proves the account was empty at some past moment, not now.
MAX_RECOVERY_GUARD_AGE_SECONDS = 900


def _global_guard_proves_zero(report: Mapping[str, Any]) -> bool:
    """Does this guard report prove the whole account is at provider zero?"""

    rows = report.get("inventory_results")
    required = {
        row.get("provider"): row
        for row in rows or []
        if isinstance(row, Mapping) and row.get("required") is True
    }
    return (
        report.get("schema_version") == "gpu_spend_guard.v1"
        and report.get("status") == "passed"
        and report.get("provider_zero_verified") is True
        and report.get("live_instance_count") == 0
        and report.get("total_burn_per_hour_usd") == 0
        # These three providers form the historical minimum account-wide
        # inventory.  New required providers strengthen the report; they must
        # not make an otherwise valid zero unrecognizable merely because this
        # validator predates them.  Every required row, including additions
        # such as AWS, is still required to be succeeded and empty below.
        and {"runpod", "vast", "digitalocean"}.issubset(required)
        and all(
            row.get("status") == "succeeded" and row.get("row_count") == 0
            for row in required.values()
        )
    )


def materialize_native_task_arena_recovered_provider_zero(
    *,
    authority_path: str | Path,
    result_path: str | Path,
    global_zero_guard_path: str | Path,
    output_path: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Seal provider zero when the attempt's own account-wide sweep failed.

    The watchdog runs two inventories: one scoped to this attempt's instance
    label and one across the whole account. On 2026-08-18 an attempt tore its
    instance down, confirmed absence for its own prefix over a 200 response,
    and then could not complete the account-wide sweep -- so the lane could not
    seal its own zero, and an unsealed attempt blocks every authority chained
    after it. The money was spent; refusing to account for it does not make the
    ledger more honest, it makes it wrong.

    Everything the ordinary seal requires still holds here. Only the source of
    the account-wide evidence changes: a fresh ``gpu_spend_guard.v1`` report
    proving zero live resources across all three required providers, taken now
    rather than at teardown, and rejected if it is stale. The lane-scoped
    inventory must still be API-confirmed and empty -- a recovery is not a way
    to seal an attempt whose own instance was never observed gone.
    """

    authority_file = Path(authority_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    guard_file = Path(global_zero_guard_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_authority_unreadable")
    result = _read(result_file, "native_task_arena_result_unreadable")
    guard = _read(guard_file, "native_task_arena_recovery_guard_unreadable")
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = Path(str(result.get("object_store_cleanup_path") or "")).resolve()
    adapter_path = Path(str(result.get("adapter_result_path") or "")).resolve()
    teardown_path = Path(str(result.get("teardown_manifest_path") or "")).resolve()
    watchdog = _read(watchdog_path, "native_task_arena_watchdog_unreadable")
    cleanup = _read(cleanup_path, "native_task_arena_cleanup_unreadable")
    adapter = _read(adapter_path, "native_task_arena_adapter_unreadable")
    teardown = _read(teardown_path, "native_task_arena_teardown_unreadable")
    lane_inventory = watchdog.get("final_inventory")

    generated_at = str(guard.get("generated_at") or "")
    try:
        guard_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("native_task_arena_recovery_guard_invalid") from None
    if guard_at.tzinfo is None:
        guard_at = guard_at.replace(tzinfo=timezone.utc)
    moment = now or datetime.now(timezone.utc)
    age_seconds = (moment - guard_at).total_seconds()

    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or result.get("schema_version") != "native_task_arena_vast_run.v1"
        or result.get("status") not in {"completed", "blocked"}
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("status") != "completed"
        # the attempt's own instance must still have been observed gone
        or not isinstance(lane_inventory, Mapping)
        or lane_inventory.get("api_confirmed") is not True
        or lane_inventory.get("live_resource_count") != 0
        or adapter.get("continuing_spend_from_this_run") is not False
        # and the account must be empty right now
        or not _global_guard_proves_zero(guard)
        or age_seconds < 0
        or age_seconds > MAX_RECOVERY_GUARD_AGE_SECONDS
    ):
        raise ValueError("native_task_arena_recovered_provider_zero_invalid")

    receipt: dict[str, Any] = {
        "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed_recovered_provider_zero",
        "attempt_authority": _record(authority_file),
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(result_file),
        "provider_adapter": _record(adapter_path),
        "teardown": _record(teardown_path),
        "watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_zero_confirmed": True,
        # what the attempt itself observed, and what replaced the sweep it could not finish
        "inventory": dict(lane_inventory),
        "recovered_global_zero_guard": _record(guard_file),
        "recovered_global_zero_guard_generated_at": generated_at,
        "recovery_reason": "attempt_global_inventory_sweep_failed",
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("native_task_arena_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


def _zero_cost(value: Any) -> bool:
    """True only when the recorded spend is an explicit, exact zero.

    A missing or unparseable cost is not evidence of zero spend and must not
    unlock the no-allocation seal.
    """

    if isinstance(value, bool) or value is None:
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _definitive_preallocation_no_allocation(
    *, adapter: Mapping[str, Any], teardown: Mapping[str, Any]
) -> bool:
    """Prove a legacy preflight exit never reached a provider create call."""

    return (
        adapter.get("api_call_performed") is False
        and adapter.get("provider_create_attempted") is False
        and adapter.get("vast_instance_ids") == []
        and teardown.get("vast_instance_ids") == []
    )


def materialize_native_task_arena_pre_spend_closeout(
    *,
    authority_path: str | Path,
    allocator_result_path: str | Path,
    authority_consumption_path: str | Path,
    api_provider_zero_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Seal a consumed authority rejected by the shared gate before allocation.

    The immutable original result stays untouched.  This derivative binds its
    failed shared pre-spend receipt, the exclusive authority-consumption record,
    and a later authenticated global Vast-zero observation.  It cannot be used
    for a result that staged objects, armed a watchdog, or entered the adapter.
    """

    authority_file = Path(authority_path).expanduser().resolve()
    original_file = Path(allocator_result_path).expanduser().resolve()
    consumption_file = Path(authority_consumption_path).expanduser().resolve()
    api_zero_file = Path(api_provider_zero_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_pre_spend_authority_invalid")
    original = _read(original_file, "native_task_arena_pre_spend_result_invalid")
    consumption = _read(
        consumption_file, "native_task_arena_pre_spend_consumption_invalid"
    )
    api_zero = _validated_api_provider_zero(api_zero_file)
    authority_digest = str(authority.get("authorization_digest") or "")
    attempt_root_raw = str(original.get("attempt_root") or "")
    if not attempt_root_raw.startswith("/"):
        raise ValueError("native_task_arena_pre_spend_closeout_invalid")
    attempt_root = Path(attempt_root_raw).resolve()
    provider_run = attempt_root / "vast_provider_run"
    preflight_file = provider_run / "pre_spend_preflight.json"
    preflight = _read(
        preflight_file, "native_task_arena_pre_spend_preflight_invalid"
    )
    expected_blocker = next(
        (
            str(item)
            for item in original.get("blockers") or []
            if str(item).endswith("_pre_spend_preflight_not_passed")
        ),
        "",
    )
    observed_preflight_blockers = [
        str(item) for item in preflight.get("blockers") or [] if str(item)
    ]
    expected_blockers = sorted({expected_blocker, *observed_preflight_blockers})
    result_at = _aware_time(
        original.get("generated_at"), "native_task_arena_pre_spend_result_time_invalid"
    )
    zero_at = _aware_time(
        api_zero.get("observed_at_utc"),
        "native_task_arena_pre_spend_api_zero_time_invalid",
    )
    materialized_at = datetime.now(timezone.utc)
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority_digest
        != canonical_digest(authority, digest_field="authorization_digest")
        or original.get("schema_version") != "native_task_arena_vast_run.v1"
        or original.get("status") != "blocked"
        or original.get("blockers") != expected_blockers
        or original.get("provider_mutations_performed") != 0
        or original.get("estimated_cost_usd") is not None
        or original.get("continuing_spend_from_this_run") is not None
        or original.get("instance_id") not in (None, "")
        or original.get("vast_instance_ids") not in (None, [], ())
        or original.get("provider_instance_ids") not in (None, [], ())
        or original.get("provider_create_attempted") not in (None, False)
        or not expected_blocker
        or not observed_preflight_blockers
        or original.get("pre_spend_preflight") != preflight
        or preflight.get("schema_version") != "pre_spend_preflight.v1"
        or preflight.get("status") != "FAIL"
        or preflight.get("spend_allowed") is not False
        or consumption.get("schema_version") != CONSUMPTION_SCHEMA_VERSION
        or consumption.get("authorization_digest") != authority_digest
        or consumption.get("bundle_sha256") != authority.get("bundle_sha256")
        or consumption.get("blueprint_commit") != authority.get("blueprint_commit")
        or consumption.get("maximum_provider_allocations") != 1
        or original_file != attempt_root / "adp_arena_vast_result.json"
        or attempt_root.name != "attempt_001"
        or attempt_root.parent.name != "attempts"
        or attempt_root.parent.parent.name != _expected_job_dir(authority)
        or consumption_file.name != f"native-task-arena-{authority_digest[7:]}.json"
        or (provider_run / "vast_provider_adapter_result.json").exists()
        or (provider_run / "vast_teardown_manifest.json").exists()
        or (attempt_root / "object_store_staging").exists()
        or (attempt_root / "vast_independent_watchdog_handoff.json").exists()
        or zero_at < result_at
        or zero_at > materialized_at
        or (materialized_at - zero_at).total_seconds()
        > MAX_PREALLOCATION_API_ZERO_AGE_SECONDS
    ):
        raise ValueError("native_task_arena_pre_spend_closeout_invalid")

    destination = Path(output_dir).expanduser().resolve()
    teardown_path = destination / "native_task_arena_pre_spend_teardown.v1.json"
    result_path = destination / "native_task_arena_pre_spend_closed_result.v1.json"
    zero_path = destination / "native_task_arena_provider_zero.v1.json"
    if any(path.exists() or path.is_symlink() for path in (teardown_path, result_path, zero_path)):
        raise ValueError("native_task_arena_pre_spend_output_exists")
    generated_at = materialized_at.isoformat()
    reason = (
        observed_preflight_blockers[0]
        if len(observed_preflight_blockers) == 1
        else expected_blocker
    )
    authorization_consumption = {
        "status": "consumed",
        "authorization_digest": authority_digest,
        "consumption_record_sha256": _sha256(consumption_file),
    }
    teardown: dict[str, Any] = {
        "schema_version": PREALLOCATION_TEARDOWN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "not_required_pre_spend_preflight_blocked",
        "vast_instance_ids": [],
        "provider_mutations_performed": 0,
        "continuing_spend_from_this_run": False,
        "original_allocator_result": _record(original_file),
        "pre_spend_preflight": _record(preflight_file),
        "authority_consumption_record": _record(consumption_file),
        "receipt_digest": "",
    }
    teardown["receipt_digest"] = canonical_digest(
        teardown, digest_field="receipt_digest"
    )
    result: dict[str, Any] = {
        "schema_version": "native_task_arena_vast_run.v1",
        "generated_at": generated_at,
        "status": "sealed_blocked_attempt",
        "closeout_kind": PRE_SPEND_CLOSEOUT_KIND,
        "blockers": expected_blockers,
        "bundle_sha256": authority.get("bundle_sha256"),
        "estimated_cost_usd": 0.0,
        "hard_cap_usd": authority.get("hard_attempt_spend_cap_usd"),
        "hard_ttl_seconds": authority.get("maximum_single_resource_ttl_seconds"),
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "authorization_consumption": authorization_consumption,
        "original_allocator_result": _record(original_file),
        "pre_spend_preflight": _record(preflight_file),
        "authority_consumption_record": _record(consumption_file),
        "teardown_manifest_path": str(teardown_path),
        "scientific_attempt_started": False,
        "first_observation_reached": False,
        "candidate_policy_queried": False,
        "visual_evidence": {
            "status": "unavailable_before_first_observation",
            "media_gap": {"type": "before_first_observation", "reason": reason},
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    created: list[Path] = []
    try:
        _write_exclusive_json(teardown_path, teardown)
        created.append(teardown_path)
        _write_exclusive_json(result_path, result)
        created.append(result_path)
        zero: dict[str, Any] = {
            "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed_preallocation_provider_zero",
            "attempt_authority": _record(authority_file),
            "attempt_authority_digest": authority_digest,
            "terminal_result": _record(result_path),
            "teardown": _record(teardown_path),
            "pre_spend_preflight": _record(preflight_file),
            "authority_consumption_record": _record(consumption_file),
            "api_provider_zero": _record(api_zero_file),
            "sibling_preallocation_closeouts": [],
            "estimated_cost_usd": 0.0,
            "provider_zero_confirmed": True,
            "inventory": None,
            "inventory_scope": "no_provider_allocation",
            "global_inventory": [],
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "scientific_attempt_started": False,
            "evidence_times": {
                "allocator_result_generated_at": result_at.isoformat(),
                "api_provider_zero_observed_at": zero_at.isoformat(),
                "closeout_generated_at": generated_at,
                "maximum_api_zero_age_seconds": MAX_PREALLOCATION_API_ZERO_AGE_SECONDS,
            },
            "receipt_digest": "",
        }
        zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
        _write_exclusive_json(zero_path, zero)
        created.append(zero_path)
        _validate_pre_spend_closed_chain(
            authority=authority, result=result, zero=zero
        )
        validate_terminal_spend_chain(
            authority_path=authority_file,
            result_path=result_path,
            provider_zero_path=zero_path,
        )
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {
        "terminal_result_path": str(result_path),
        "teardown_manifest_path": str(teardown_path),
        "provider_zero_path": str(zero_path),
        "provider_zero_receipt_digest": zero["receipt_digest"],
        "attempt_authority_digest": authority_digest,
        "provider_mutation_performed": False,
        "scientific_attempt_started": False,
    }


def materialize_native_task_arena_preallocation_closeout(
    *,
    authority_path: str | Path,
    allocator_result_path: str | Path,
    watchdog_handoff_path: str | Path,
    object_store_cleanup_path: str | Path,
    api_provider_zero_path: str | Path,
    output_dir: str | Path,
    sibling_preallocation_closeout_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Seal a consumed authority that failed before its watchdog could arm.

    This does not upgrade the failed launch into an execution.  It derives a
    zero-cost terminal normalization only after the immutable allocator result,
    watchdog handoff, object cleanup, and a later authenticated global Vast-zero
    receipt all agree that no provider allocation or mutation occurred.
    """

    authority_file = Path(authority_path).expanduser().resolve()
    original_file = Path(allocator_result_path).expanduser().resolve()
    watchdog_file = Path(watchdog_handoff_path).expanduser().resolve()
    cleanup_file = Path(object_store_cleanup_path).expanduser().resolve()
    api_zero_file = Path(api_provider_zero_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_preallocation_authority_invalid")
    original = _read(original_file, "native_task_arena_preallocation_result_invalid")
    watchdog = _read(watchdog_file, "native_task_arena_preallocation_watchdog_invalid")
    cleanup = _read(cleanup_file, "native_task_arena_preallocation_cleanup_invalid")
    api_zero = _validated_api_provider_zero(api_zero_file)
    authority_digest = authority.get("authorization_digest")
    sibling_records: list[dict[str, Any]] = []
    sibling_digests: set[str] = set()
    for item in sibling_preallocation_closeout_paths:
        sibling_file = Path(item).expanduser().resolve()
        sibling = _read(
            sibling_file, "native_task_arena_preallocation_sibling_invalid"
        )
        _validate_preallocation_provider_zero_value(sibling)
        digest = str(sibling.get("attempt_authority_digest") or "")
        if not digest or digest == authority_digest or digest in sibling_digests:
            raise ValueError("native_task_arena_preallocation_sibling_invalid")
        sibling_authority_path, _ = _bound_record(
            sibling.get("attempt_authority"),
            "native_task_arena_preallocation_sibling_authority_unbound",
        )
        sibling_result_path, _ = _bound_record(
            sibling.get("terminal_result"),
            "native_task_arena_preallocation_sibling_result_unbound",
        )
        _validate_preallocation_closed_chain(
            authority=_read(
                sibling_authority_path,
                "native_task_arena_preallocation_sibling_authority_invalid",
            ),
            result=_read(
                sibling_result_path,
                "native_task_arena_preallocation_sibling_result_invalid",
            ),
            zero=sibling,
            allow_siblings=False,
        )
        sibling_digests.add(digest)
        sibling_records.append(_record(sibling_file))

    materialized_at = datetime.now(timezone.utc)
    result_at = _aware_time(
        original.get("generated_at"),
        "native_task_arena_preallocation_result_time_invalid",
    )
    zero_at = _aware_time(
        api_zero.get("observed_at_utc"),
        "native_task_arena_preallocation_api_zero_time_invalid",
    )
    attempt_root_raw = str(original.get("attempt_root") or "")
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority_digest
        != canonical_digest(authority, digest_field="authorization_digest")
        or original.get("schema_version") != "native_task_arena_vast_run.v1"
        or original.get("status") != "blocked"
        or original.get("blockers") != [_expected_watchdog_blocker(authority)]
        or original.get("provider_mutations_performed") != 0
        or original.get("all_staged_objects_absent") is not True
        or original.get("estimated_cost_usd") is not None
        or original.get("continuing_spend_from_this_run") is not None
        or original.get("retry_cap") != 0
        or original.get("authorization_consumption", {}).get("status") != "consumed"
        or original.get("authorization_consumption", {}).get("authorization_digest")
        != authority_digest
        or original.get("instance_id") not in (None, "")
        or original.get("vast_instance_ids") not in (None, [], ())
        or original.get("provider_instance_ids") not in (None, [], ())
        or original.get("provider_create_attempted") not in (None, False)
        or not _preallocation_attempt_layout_valid(
            authority=authority,
            original_path=original_file,
            watchdog_path=watchdog_file,
            cleanup_path=cleanup_file,
            attempt_root_raw=attempt_root_raw,
        )
        or original.get("independent_watchdog") != watchdog
        or watchdog.get("schema_version") != WATCHDOG_HANDOFF_SCHEMA_VERSION
        or watchdog.get("status") != "blocked"
        or watchdog.get("watchdog_armed_before_allocation") is not False
        or watchdog.get("independent_process") is not False
        or watchdog.get("provider_mutations_performed") != 0
        or not _preallocation_cleanup_valid(cleanup)
        or zero_at < result_at
        or zero_at > materialized_at
        or (materialized_at - zero_at).total_seconds()
        > MAX_PREALLOCATION_API_ZERO_AGE_SECONDS
    ):
        raise ValueError("native_task_arena_preallocation_evidence_invalid")

    destination = Path(output_dir).expanduser().resolve()
    teardown_path = destination / "native_task_arena_preallocation_teardown.v1.json"
    result_path = destination / "native_task_arena_preallocation_closed_result.v1.json"
    zero_path = destination / "native_task_arena_preallocation_provider_zero.v1.json"
    if any(path.exists() or path.is_symlink() for path in (teardown_path, result_path, zero_path)):
        raise ValueError("native_task_arena_preallocation_output_exists")
    generated_at = materialized_at.isoformat()
    teardown: dict[str, Any] = {
        "schema_version": PREALLOCATION_TEARDOWN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "not_required_independent_watchdog_not_armed_before_allocation",
        "vast_instance_ids": [],
        "provider_mutations_performed": 0,
        "continuing_spend_from_this_run": False,
        "original_allocator_result": _record(original_file),
        "watchdog_handoff": _record(watchdog_file),
        "receipt_digest": "",
    }
    teardown["receipt_digest"] = canonical_digest(
        teardown, digest_field="receipt_digest"
    )
    result: dict[str, Any] = {
        "schema_version": "native_task_arena_vast_run.v1",
        "generated_at": generated_at,
        "status": "sealed_blocked_attempt",
        "closeout_kind": PREALLOCATION_CLOSEOUT_KIND,
        "closeout_path": str(result_path),
        "blockers": list(original["blockers"]),
        "bundle_sha256": authority.get("bundle_sha256"),
        "estimated_cost_usd": 0.0,
        "hard_cap_usd": authority.get("hard_attempt_spend_cap_usd"),
        "hard_ttl_seconds": authority.get("maximum_single_resource_ttl_seconds"),
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "authorization_consumption": dict(original["authorization_consumption"]),
        "original_allocator_result": _record(original_file),
        "watchdog_handoff": _record(watchdog_file),
        "object_store_cleanup": _record(cleanup_file),
        "teardown_manifest_path": str(teardown_path),
        "scientific_attempt_started": False,
        "candidate_policy_queried": False,
        "visual_evidence": {
            "status": "unavailable_before_first_observation",
            "media_gap": {
                "type": "before_first_observation",
                "reason": _expected_watchdog_blocker(authority),
            },
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    created: list[Path] = []
    try:
        _write_exclusive_json(teardown_path, teardown)
        created.append(teardown_path)
        _write_exclusive_json(result_path, result)
        created.append(result_path)
        zero: dict[str, Any] = {
            "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed_preallocation_provider_zero",
            "attempt_authority": _record(authority_file),
            "attempt_authority_digest": authority_digest,
            "terminal_result": _record(result_path),
            "teardown": _record(teardown_path),
            "watchdog": _record(watchdog_file),
            "object_store_cleanup": _record(cleanup_file),
            "api_provider_zero": _record(api_zero_file),
            "sibling_preallocation_closeouts": sibling_records,
            "estimated_cost_usd": 0.0,
            "provider_zero_confirmed": True,
            "inventory": None,
            "inventory_scope": "no_provider_allocation",
            "global_inventory": [],
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "scientific_attempt_started": False,
            "evidence_times": {
                "allocator_result_generated_at": result_at.isoformat(),
                "api_provider_zero_observed_at": zero_at.isoformat(),
                "closeout_generated_at": materialized_at.isoformat(),
                "maximum_api_zero_age_seconds": (
                    MAX_PREALLOCATION_API_ZERO_AGE_SECONDS
                ),
            },
            "receipt_digest": "",
        }
        zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
        _write_exclusive_json(zero_path, zero)
        created.append(zero_path)
        _validate_preallocation_closed_chain(
            authority=authority, result=result, zero=zero
        )
        validate_terminal_spend_chain(
            authority_path=authority_file,
            result_path=result_path,
            provider_zero_path=zero_path,
        )
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {
        "terminal_result_path": str(result_path),
        "teardown_manifest_path": str(teardown_path),
        "provider_zero_path": str(zero_path),
        "provider_zero_receipt_digest": zero["receipt_digest"],
        "attempt_authority_digest": authority_digest,
        "provider_mutation_performed": False,
        "scientific_attempt_started": False,
    }


def materialize_native_task_arena_provider_zero(
    *, authority_path: str | Path, result_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Bind terminal teardown, object cleanup, and watchdog API inventory."""

    authority_file = Path(authority_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    authority = _read(authority_file, "native_task_arena_authority_unreadable")
    result = _read(result_file, "native_task_arena_result_unreadable")
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = Path(str(result.get("object_store_cleanup_path") or "")).resolve()
    adapter_path = Path(str(result.get("adapter_result_path") or "")).resolve()
    teardown_path = Path(str(result.get("teardown_manifest_path") or "")).resolve()
    watchdog = _read(watchdog_path, "native_task_arena_watchdog_unreadable")
    cleanup = _read(cleanup_path, "native_task_arena_cleanup_unreadable")
    adapter = _read(adapter_path, "native_task_arena_adapter_unreadable")
    teardown = _read(teardown_path, "native_task_arena_teardown_unreadable")
    global_inventory = watchdog.get("final_global_inventory")
    # The global sweep observes every instance on the provider account, so an
    # unrelated debug pod or a concurrent lane blocks this seal forever: the
    # watchdog receipt is frozen at write time and can never re-observe a
    # now-quiet account. When the watchdog itself marks the global sweep
    # informational, its lane-scoped inventory -- matched on this run's own
    # name prefix -- is the authority on whether THIS run is still spending.
    # Absent that flag the global sweep stays authoritative, so receipts
    # written before the watchdog scoped itself keep their original strictness.
    if watchdog.get("global_inventory_informational_only") is True:
        inventory = watchdog.get("final_inventory")
        inventory_scope = "recorded_instance_and_lane_prefix"
    else:
        inventory = global_inventory
        inventory_scope = "provider_global"

    # A run can end before it ever allocates -- no offer met the lane's
    # constraints, or admission refused. The watchdog then records a handoff
    # instead of a canary receipt, because it was armed but never had a
    # resource to observe. There is definitionally nothing to zero, yet the
    # inventory-based seal below can never be satisfied, which wedges the
    # whole chain: the successor's first step is sealing its predecessor.
    #
    # Accept that state only on proof that nothing was allocated -- zero
    # provider mutations, armed before allocation, and no continuing spend
    # anywhere. An orphaned resource requires an allocation to exist, so this
    # cannot mask one.
    definitive_preallocation_no_allocation = _definitive_preallocation_no_allocation(
        adapter=adapter,
        teardown=teardown,
    )
    no_allocation_seal = (
        watchdog.get("schema_version") == WATCHDOG_HANDOFF_SCHEMA_VERSION
        and watchdog.get("status") == "cancelled_no_allocation"
        and watchdog.get("provider_mutations_performed") == 0
        and watchdog.get("watchdog_armed_before_allocation") is True
        and (
            _zero_cost(result.get("estimated_cost_usd"))
            or (
                result.get("estimated_cost_usd") is None
                and definitive_preallocation_no_allocation
            )
        )
    )

    shared_invalid = (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or result.get("schema_version") != "native_task_arena_vast_run.v1"
        or result.get("status") not in {"completed", "blocked"}
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or teardown.get("continuing_spend_from_this_run") is not False
        or (
            adapter.get("continuing_spend_from_this_run") is not False
            and not (
                no_allocation_seal
                and definitive_preallocation_no_allocation
                and adapter.get("continuing_spend_from_this_run") is None
            )
        )
    )
    if no_allocation_seal:
        inventory = None
        inventory_scope = "no_provider_allocation"
        if shared_invalid:
            raise ValueError("native_task_arena_provider_zero_invalid")
    elif (
        shared_invalid
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or not isinstance(global_inventory, Mapping)
        or global_inventory.get("api_confirmed") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
    ):
        raise ValueError("native_task_arena_provider_zero_invalid")
    receipt: dict[str, Any] = {
        "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed",
        "attempt_authority": _record(authority_file),
        "attempt_authority_digest": authority["authorization_digest"],
        "terminal_result": _record(result_file),
        "provider_adapter": _record(adapter_path),
        "teardown": _record(teardown_path),
        "watchdog": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
        "provider_zero_confirmed": True,
        "inventory": inventory,
        "inventory_scope": inventory_scope,
        "global_inventory": global_inventory,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("native_task_arena_provider_zero_output_exists")
    ensure_dir(output.parent)
    write_json(output, receipt)
    return receipt


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "PRE_SPEND_CLOSEOUT_KIND",
    "PROVIDER_ZERO_SCHEMA_VERSION",
    "consume_native_task_arena_authority_once",
    "materialize_native_task_arena_paid_attempt_authority",
    "materialize_native_task_arena_pre_spend_closeout",
    "materialize_native_task_arena_preallocation_closeout",
    "materialize_native_task_arena_provider_zero",
    "validate_native_task_arena_paid_attempt_authority",
    "validate_terminal_spend_chain",
]
