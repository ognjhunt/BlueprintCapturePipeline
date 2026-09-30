"""Pure validation of retained launch sync and provider-zero observations.

These contracts do not reconcile resources, contact the WebApp, or mutate launch
state. The reconciler reexports the original helper names for compatibility.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .launch_immutable_input_writer import TaskEvaluationLaunchError

LAUNCH_RECEIPT_DIGEST_CANONICALIZATION = "rfc8785"
DIRECT_EXECUTION_ADOPTION_SCHEMA_VERSION = "task_evaluation_native_direct_execution_adoption.v1"
CONFIGURATION_COMPLETE_OFFERING_STATUSES = frozenset(
    {"launch_ready", "configured_controls_pending"}
)


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_sha256_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        text.startswith("sha256:")
        and len(text) == len("sha256:") + 64
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _guard_provider_zero(
    *,
    guard: Mapping[str, Any],
    required_providers: Sequence[str],
    max_age_seconds: int,
    now: datetime,
    not_before: datetime | None,
    not_before_subject: str = "launch",
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not required_providers:
        blockers.append("gpu_required_provider_scope_missing")
    if guard.get("schema_version") != "gpu_spend_guard.v1":
        blockers.append("gpu_spend_guard_schema_invalid")
    generated_at = _timestamp(guard.get("generated_at"))
    if generated_at is None:
        blockers.append("gpu_spend_guard_timestamp_invalid")
    else:
        age = (now - generated_at).total_seconds()
        if age < -60 or age > max_age_seconds:
            blockers.append("gpu_spend_guard_stale")
        if not_before is not None and generated_at < not_before:
            blockers.append(f"gpu_spend_guard_predates_{not_before_subject}")
    if guard.get("reap_mode") is not True:
        blockers.append("gpu_spend_guard_reap_mode_missing")
    provider_zero = guard.get("provider_zero")
    provider_zero = provider_zero if isinstance(provider_zero, Mapping) else {}
    guard_scope = {
        str(provider) for provider in provider_zero.get("required_provider_ids") or []
    }
    # The profile binds this launch's provider scope. A wider guard can be
    # globally unverified because a different provider's inventory failed;
    # its confirmed empty inventory still proves this narrower scope. Retain
    # the full-scope flags when claiming the guard's entire provider set.
    if not set(required_providers) < guard_scope:
        if guard.get("provider_zero_verified") is not True:
            blockers.append("gpu_provider_zero_not_verified")
        if provider_zero.get("status") != "verified":
            blockers.append("gpu_provider_zero_status_unverified")
    if guard.get("live_instance_count") != 0:
        blockers.append("gpu_provider_nonzero")
    if guard.get("total_burn_per_hour_usd") not in (0, 0.0):
        blockers.append("gpu_provider_nonzero_burn")
    if provider_zero.get("global_live_instance_count") != 0:
        blockers.append("gpu_provider_zero_global_inventory_nonzero")
    if provider_zero.get("global_total_burn_per_hour_usd") not in (0, 0.0):
        blockers.append("gpu_provider_zero_global_burn_nonzero")
    if guard.get("reap_candidate_ids") not in ([], ()):
        blockers.append("gpu_orphan_reap_candidates_remaining")
    for result in guard.get("reap_results") or []:
        if isinstance(result, Mapping) and result.get("status") != "terminated":
            blockers.append("gpu_orphan_reap_not_confirmed")

    inventories = {
        str(row.get("provider")): row
        for row in guard.get("inventory_results") or []
        if isinstance(row, Mapping)
    }
    for provider in required_providers:
        inventory = inventories.get(str(provider))
        if inventory is None:
            blockers.append(f"gpu_inventory_missing:{provider}")
        elif inventory.get("status") != "succeeded":
            blockers.append(f"gpu_inventory_not_confirmed:{provider}")
        elif inventory.get("row_count") != 0:
            blockers.append(f"gpu_inventory_nonzero:{provider}")
        elif inventory.get("required") is not True:
            blockers.append(f"gpu_inventory_scope_not_required:{provider}")
        elif provider not in guard_scope:
            blockers.append(f"gpu_provider_zero_scope_missing:{provider}")
    return not blockers, sorted(set(blockers))


def validated_succeeded_webapp_sync_row(
    *, receipt: Mapping[str, Any], attempt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the WebApp's exact terminal binding before claiming website origin."""

    response = attempt.get("response")
    response = response if isinstance(response, Mapping) else {}
    terminal = receipt.get("terminal_evidence")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    scene_configuration = terminal.get("scene_configuration")
    scene_configuration = (
        scene_configuration if isinstance(scene_configuration, Mapping) else {}
    )
    offering = scene_configuration.get("configured_scene_offering")
    offering = offering if isinstance(offering, Mapping) else {}
    offering_digest = offering.get("offering_digest")
    offering_status = offering.get("status")
    offering_ack_invalid = bool(offering) and (
        not _is_sha256_digest(offering_digest)
        or offering_status not in CONFIGURATION_COMPLETE_OFFERING_STATUSES
        or attempt.get("configured_scene_offering_digest") != offering_digest
        or attempt.get("configured_scene_offering_status") != offering_status
        or response.get("configured_scene_offering_digest") != offering_digest
        or response.get("configured_scene_offering_status") != offering_status
    )
    direct_projection = receipt.get("website_projection")
    direct_projection = (
        direct_projection if isinstance(direct_projection, Mapping) else {}
    )
    direct_projection_invalid = (
        receipt.get("schema_version") == DIRECT_EXECUTION_ADOPTION_SCHEMA_VERSION
        and (
            direct_projection.get("configured_scene_offering_status")
            != "configured_controls_pending"
            or direct_projection.get("native_construction_status") != "blocked"
            or direct_projection.get("native_construction_blockers")
            != receipt.get("blockers")
            or direct_projection.get("qualification_upgrade_performed") is not False
            or attempt.get("configured_scene_offering_status")
            != "configured_controls_pending"
            or attempt.get("native_construction_status") != "blocked"
            or attempt.get("native_construction_blockers")
            != receipt.get("blockers")
            or attempt.get("qualification_upgrade_performed") is not False
        )
    )
    if (
        attempt.get("schema_version")
        != "task_evaluation_launch_webapp_sync_result.v1"
        or attempt.get("status") != "succeeded"
        or attempt.get("provider_mutation_performed") is not False
        or attempt.get("sync_result_digest")
        != canonical_digest(attempt, digest_field="sync_result_digest")
        or not isinstance(attempt.get("attempt_number"), int)
        or isinstance(attempt.get("attempt_number"), bool)
        or attempt["attempt_number"] < 1
        or _timestamp(attempt.get("attempted_at")) is None
        or any(
            attempt.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
        )
        or any(
            response.get(field) != receipt.get(field)
            for field in ("launch_id", "run_id", "request_digest", "receipt_digest")
        )
        or response.get("schema_version")
        != "task_evaluation_launch_web_sync_receipt.v1"
        or response.get("status") != receipt.get("status")
        or not isinstance(response.get("already_exists"), bool)
        or offering_ack_invalid
        or direct_projection_invalid
    ):
        raise TaskEvaluationLaunchError("webapp_sync_succeeded_invalid")
    committed_receipt = {
        "sync_result_digest": attempt.get("sync_result_digest"),
        "launch_id": receipt.get("launch_id"),
        "run_id": receipt.get("run_id"),
        "request_digest": receipt.get("request_digest"),
        "receipt_digest": receipt.get("receipt_digest"),
        "response_schema_version": response.get("schema_version"),
        "terminal_status": response.get("status"),
        "already_exists": response.get("already_exists"),
    }
    if offering:
        committed_receipt.update(
            {
                "configured_scene_offering_digest": offering_digest,
                "configured_scene_offering_status": offering_status,
            }
        )
    if receipt.get("schema_version") == DIRECT_EXECUTION_ADOPTION_SCHEMA_VERSION:
        committed_receipt.update(
            {
                "configured_scene_offering_status": (
                    "configured_controls_pending"
                ),
                "native_construction_status": "blocked",
                "native_construction_blockers": list(receipt["blockers"]),
                "qualification_upgrade_performed": False,
            }
        )
    return {
        "launch_id": receipt.get("launch_id"),
        "status": "webapp_sync_succeeded",
        "attempts": attempt["attempt_number"],
        "blockers": [],
        "webapp_record_bound": True,
        "website_trigger_proven": True,
        "provider_mutation_performed": False,
        "allocator_invoked": False,
        "automatic_retry_performed": False,
        "receipt": committed_receipt,
    }
