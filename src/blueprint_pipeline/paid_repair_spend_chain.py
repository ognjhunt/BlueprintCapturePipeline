"""Neutral validation of retained paid-repair spend lineage.

This module deliberately parses legacy Aura receipt schemas only as immutable
historical spend evidence.  It does not expose an Aura execution path or select
Aura as a repair default.  Active ArtiFixer3D successors consume the same
fail-closed lineage validator without depending on an execution backend module.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


ARTIFIXER3D_PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "public_scene_artifixer3d_paid_attempt_authority.v1"
)
ARTIFIXER3D_RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_vast_run.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound(record: Any, *, code: str) -> Path:
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


PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = ARTIFIXER3D_PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
RESULT_SCHEMA_VERSION = ARTIFIXER3D_RESULT_SCHEMA_VERSION


def _validate_prior_authority_chain(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    """Recursively re-open the predecessor authority's complete spend chain."""

    seen = set() if seen is None else seen
    resolved = path.expanduser().resolve()
    if resolved in seen or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("artifixer3d_prior_authority_chain_invalid")
    seen.add(resolved)
    value = _read(resolved, code="artifixer3d_prior_authority_unreadable")
    if (
        value.get("schema_version") != "public_scene_aura_exact_residual_paid_attempt_authority.v1"
        or value.get("authorization_digest")
        != canonical_digest(value, digest_field="authorization_digest")
        or value.get("automatic_paid_retry_authorized") is not False
        or value.get("maximum_automatic_retries") != 0
        or value.get("maximum_paid_attempts") != 1
        or value.get("aggregate_goal_spend_cap_usd") != 12.0
        or isinstance(value.get("prior_goal_spend_usd"), bool)
        or not isinstance(value.get("prior_goal_spend_usd"), (int, float))
    ):
        raise ValueError("artifixer3d_prior_authority_invalid")
    for field in (
        "previous_terminal_execution_result",
        "previous_runtime_result",
        "previous_teardown",
        "previous_watchdog",
        "previous_object_store_cleanup",
        "prior_provider_runtime_campaign",
    ):
        _bound(value.get(field), code="artifixer3d_prior_authority_dependency_unbound")
    for record in value.get("additional_terminal_spend_receipts") or []:
        _bound(record, code="artifixer3d_prior_authority_spend_unbound")
    parent = value.get("prior_manual_corrected_attempt_authority")
    if parent is not None:
        _validate_prior_authority_chain(
            _bound(parent, code="artifixer3d_prior_authority_parent_unbound"),
            seen=seen,
        )
    return value


def _validate_prior_terminal_result(
    path: Path, *, prior_authority: Mapping[str, Any]
) -> tuple[dict[str, Any], float, float]:
    result = _read(path, code="artifixer3d_prior_terminal_result_unreadable")
    cost = result.get("estimated_cost_usd")
    if (
        result.get("schema_version") != "public_scene_aura_exact_residual_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != prior_authority.get("authorization_digest")
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError("artifixer3d_prior_terminal_result_invalid")
    teardown_path = Path(str(result.get("teardown_manifest_path") or "")).resolve()
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = path.parent / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    teardown = _read(teardown_path, code="artifixer3d_prior_teardown_unreadable")
    watchdog = _read(watchdog_path, code="artifixer3d_prior_watchdog_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_prior_cleanup_unreadable")
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("final_global_inventory", {}).get("live_resource_count") != 0
        or watchdog.get("final_global_inventory", {}).get("api_confirmed") is not True
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
    ):
        raise ValueError("artifixer3d_prior_terminal_closeout_invalid")
    return result, round(float(cost), 6)


def _validate_prior_artifixer_attempt(
    *,
    authority_path: Path,
    result_path: Path,
    cleanup_path: Path,
    provider_zero_path: Path,
) -> tuple[dict[str, Any], float]:
    """Re-open a predecessor ArtiFixer attempt, including its zero closeout."""

    authority = _read(authority_path, code="artifixer3d_predecessor_authority_unreadable")
    result = _read(result_path, code="artifixer3d_predecessor_result_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_predecessor_cleanup_unreadable")
    provider_zero = _read(
        provider_zero_path, code="artifixer3d_predecessor_provider_zero_unreadable"
    )
    inventory = provider_zero.get("inventory")
    cost = result.get("estimated_cost_usd")
    result_mutations = result.get("provider_mutations_performed")
    if (
        result_mutations is None
        and provider_zero.get("provider_mutations_performed_by_attempt") == 1
    ):
        adapter_path = _bound(
            provider_zero.get("provider_adapter"),
            code="artifixer3d_predecessor_adapter_unbound",
        )
        adapter = _read(adapter_path, code="artifixer3d_predecessor_adapter_unreadable")
        classification = adapter.get("provider_attempt_classification")
        if (
            adapter.get("schema_version") != "vast_provider_adapter_result.v1"
            or adapter.get("status") != "failed"
            or adapter.get("provider_create_attempted") is not True
            or adapter.get("api_call_performed") is not True
            or adapter.get("continuing_spend_from_this_run") is not False
            or not adapter.get("vast_instance_ids")
            or adapter.get("estimated_cost_usd") != cost
            or not isinstance(classification, Mapping)
            or classification.get("classification") != "pre_execution_provider_null"
            or classification.get("provider_bundle_started") is not False
            or classification.get("provider_entrypoint_started") is not False
            or classification.get("provider_output_returned") is not False
        ):
            raise ValueError("artifixer3d_predecessor_adapter_invalid")
        result_mutations = 1
    if cost is None and result.get("provider_mutations_performed") == 0:
        cost = 0.0
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("maximum_paid_attempts") != 1
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") not in {"blocked", "completed"}
        or result.get("retry_cap") != 0
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("all_staged_objects_absent") is not True
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or provider_zero.get("schema_version") != "artifixer3d_postblocked_provider_zero.v1"
        or provider_zero.get("attempt_authority_digest") != authority.get("authorization_digest")
        or provider_zero.get("provider_mutations_performed_by_attempt") != result_mutations
        or provider_zero.get("provider_zero_confirmed") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
    ):
        raise ValueError("artifixer3d_predecessor_attempt_invalid")
    attempt_cost = round(float(cost), 6)
    lineage_cost = attempt_cost
    predecessor = authority.get("prior_artifixer_attempt")
    if predecessor is not None:
        if not isinstance(predecessor, Mapping):
            raise ValueError("artifixer3d_predecessor_lineage_invalid")
        nested_authority_path = _bound(
            predecessor.get("authority"),
            code="artifixer3d_predecessor_lineage_authority_unbound",
        )
        nested_result_path = _bound(
            predecessor.get("terminal_result"),
            code="artifixer3d_predecessor_lineage_result_unbound",
        )
        nested_cleanup_path = _bound(
            predecessor.get("object_store_cleanup"),
            code="artifixer3d_predecessor_lineage_cleanup_unbound",
        )
        nested_zero_path = _bound(
            predecessor.get("provider_zero"),
            code="artifixer3d_predecessor_lineage_zero_unbound",
        )
        nested_authority, nested_attempt_cost, nested_lineage_cost = (
            _validate_prior_artifixer_attempt(
                authority_path=nested_authority_path,
                result_path=nested_result_path,
                cleanup_path=nested_cleanup_path,
                provider_zero_path=nested_zero_path,
            )
        )
        recorded_lineage_cost = predecessor.get(
            "lineage_cost_usd", predecessor.get("terminal_cost_usd")
        )
        if (
            predecessor.get("authority_digest") != nested_authority.get("authorization_digest")
            or predecessor.get("terminal_cost_usd") != nested_attempt_cost
            or recorded_lineage_cost != nested_lineage_cost
        ):
            raise ValueError("artifixer3d_predecessor_lineage_mismatch")
        lineage_cost = round(lineage_cost + nested_lineage_cost, 6)
    return authority, attempt_cost, lineage_cost


def validate_artifixer3d_terminal_spend_chain(
    *,
    authority_path: str | Path,
    result_path: str | Path,
    cleanup_path: str | Path,
    provider_zero_path: str | Path,
) -> dict[str, Any]:
    """Expose a verified terminal campaign spend anchor to successor modules."""

    resolved = tuple(
        Path(path).expanduser().resolve()
        for path in (authority_path, result_path, cleanup_path, provider_zero_path)
    )
    authority, attempt_cost, lineage_cost = _validate_prior_artifixer_attempt(
        authority_path=resolved[0],
        result_path=resolved[1],
        cleanup_path=resolved[2],
        provider_zero_path=resolved[3],
    )
    aggregate_before = authority.get("aggregate_goal_spend_before_attempt_usd")
    aggregate_cap = authority.get("aggregate_goal_spend_cap_usd")
    if (
        isinstance(aggregate_before, bool)
        or not isinstance(aggregate_before, (int, float))
        or isinstance(aggregate_cap, bool)
        or not isinstance(aggregate_cap, (int, float))
        or not math.isfinite(float(aggregate_before))
        or not math.isfinite(float(aggregate_cap))
        or float(aggregate_before) < 0
        or float(aggregate_cap) <= 0
    ):
        raise ValueError("artifixer3d_terminal_spend_anchor_invalid")
    return {
        "authority": authority,
        "authority_digest": authority["authorization_digest"],
        "attempt_cost_usd": attempt_cost,
        "lineage_cost_usd": lineage_cost,
        "aggregate_goal_spend_after_attempt_usd": round(float(aggregate_before) + attempt_cost, 6),
        "aggregate_goal_spend_cap_usd": round(float(aggregate_cap), 6),
        "records": {
            "authority": _record(resolved[0]),
            "terminal_result": _record(resolved[1]),
            "object_store_cleanup": _record(resolved[2]),
            "provider_zero": _record(resolved[3]),
        },
    }



#: The appearance campaign's spend anchor when it has no paid predecessor.
CAMPAIGN_START_SCHEMA_VERSION = "appearance_campaign_spend_start.v1"

#: The shared cap every link of the campaign is checked against.
AGGREGATE_GOAL_SPEND_CAP_USD = 12.0


def validate_campaign_start_receipt(path: Path) -> tuple[dict[str, Any], float]:
    """Anchor the campaign's spend on a measurement rather than a predecessor.

    `_validate_prior_authority_chain` anchors on a completed AuraFusion360 paid
    attempt: an authority carrying six bound dependency records, and a terminal
    result with `status: completed`. Those exist only after a real paid Aura
    run, and there was never one -- every Aura artifact is `dry_run_ready`, the
    launch queue holds no Aura attempt, no Aura authority was consumed, and the
    spend ledger totals zero. Retiring the lane made that anchor unreachable
    rather than merely unused, so ArtiFixer3D could not be authorized at all.

    This is the replacement, and it is deliberately not "skip the anchor". The
    anchor's whole job is to carry the campaign's running spend into the
    `prior_spend + hard_cap_usd > aggregate_cap` check, so removing it would
    uncap the campaign. This keeps the number and changes only where it comes
    from: a sealed receipt that records what was *measured*, and the evidence it
    was measured from.

    A zero-spend anchor must say so explicitly. A receipt claiming no prior
    spend while naming prior paid attempts is refused rather than believed.
    """

    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("appearance_campaign_start_invalid")
    value = _read(resolved, code="appearance_campaign_start_unreadable")
    spend = value.get("prior_goal_spend_usd")
    observed = value.get("measured_paid_attempts")
    if (
        value.get("schema_version") != CAMPAIGN_START_SCHEMA_VERSION
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("aggregate_goal_spend_cap_usd") != AGGREGATE_GOAL_SPEND_CAP_USD
        or value.get("provider_mutation_performed") is not False
        or isinstance(spend, bool)
        or not isinstance(spend, (int, float))
        or not math.isfinite(float(spend))
        or float(spend) < 0
        or not isinstance(observed, list)
        or not isinstance(value.get("measured_from"), list)
        or not value.get("measured_from")
    ):
        raise ValueError("appearance_campaign_start_invalid")
    # The two halves have to agree. A receipt that names paid attempts and
    # still claims zero prior spend is the shape a fabricated anchor takes.
    if bool(observed) != (float(spend) > 0):
        raise ValueError("appearance_campaign_start_spend_disagrees_with_evidence")
    for record in value.get("measured_from") or []:
        _bound(record, code="appearance_campaign_start_evidence_unbound")
    return value, round(float(spend), 6)

__all__ = [
    "ARTIFIXER3D_PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "ARTIFIXER3D_RESULT_SCHEMA_VERSION",
    "_validate_prior_artifixer_attempt",
    "_validate_prior_authority_chain",
    "_validate_prior_terminal_result",
    "validate_artifixer3d_terminal_spend_chain",
    "validate_campaign_start_receipt",
    "CAMPAIGN_START_SCHEMA_VERSION",
    "AGGREGATE_GOAL_SPEND_CAP_USD",
]
