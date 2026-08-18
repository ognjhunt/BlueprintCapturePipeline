"""Provider-neutral spend and staging guards shared by Vast probe lanes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


VAST_WAM_CONTAINER_MISSING_MAX_SECONDS_ENV = "BLUEPRINT_VAST_WAM_CONTAINER_MISSING_MAX_SECONDS"
DEFAULT_VAST_WAM_CONTAINER_MISSING_MAX_SECONDS = 720


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _attempt_estimated_cost(attempt: Mapping[str, Any]) -> float:
    for key in ("estimated_cost_usd_using_observed_rate", "estimated_cost_usd"):
        value = _number(attempt.get(key))
        if value is not None:
            return max(0.0, value)
    return 0.0


def _session_estimated_cost(path: Path) -> tuple[float, str | None]:
    if not path.is_file():
        return 0.0, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("session_budget_ledger_root_must_be_mapping")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return 0.0, f"session_budget_ledger_parse_failed:{type(exc).__name__}"
    for key in ("total_observed_estimated_cost_usd", "estimated_cost_usd"):
        value = _number(payload.get(key))
        if value is not None:
            return max(0.0, value), None
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        return (
            sum(
                _attempt_estimated_cost(item)
                for item in attempts
                if isinstance(item, Mapping)
            ),
            None,
        )
    return 0.0, None


def target_spend_guard(
    *,
    budget_path: Path,
    target_spend_usd: float,
    max_hourly_rate: float,
    max_live_minutes: int,
    allow_target_spend_overrun: bool,
) -> dict[str, Any]:
    prior_cost, parse_error = _session_estimated_cost(budget_path)
    projected_incremental = (
        max(0.0, max_hourly_rate) * max(0, max_live_minutes) / 60.0
    )
    blockers: list[str] = []
    if parse_error:
        blockers.append("session_budget_ledger_parse_failed")
    elif not allow_target_spend_overrun:
        if prior_cost >= target_spend_usd:
            blockers.append("session_estimated_spend_target_exhausted")
        elif prior_cost + projected_incremental > target_spend_usd:
            blockers.append("requested_max_spend_would_exceed_target")
    return {
        "schema_version": "vast_authorized_probe_target_spend_guard.v1",
        "status": "blocked" if blockers else "passed",
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_path.is_file(),
        "budget_parse_error": parse_error,
        "target_spend_usd": target_spend_usd,
        "prior_estimated_cost_usd": round(prior_cost, 6),
        "projected_max_incremental_cost_usd": round(projected_incremental, 6),
        "projected_total_estimated_cost_usd": round(
            prior_cost + projected_incremental, 6
        ),
        "remaining_to_target_before_request_usd": round(
            target_spend_usd - prior_cost, 6
        ),
        "allow_target_spend_overrun": allow_target_spend_overrun,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }


def staging_verification_guard(
    *,
    verify_staging_urls: bool,
    allow_unverified_public_staging_for_paid_launch: bool,
    staging_manifest: Mapping[str, Any],
    public_staging_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    bundle_url_ready = staging_manifest.get("provider_fetchable_bundle_uri_ready") is True
    output_url_ready = staging_manifest.get("provider_output_callback_ready") is True
    public_status = (
        public_staging_verification.get("status")
        if isinstance(public_staging_verification, Mapping)
        else "not_requested"
    )
    if not bundle_url_ready:
        blockers.append("provider_bundle_fetch_url_not_ready")
    if not output_url_ready:
        blockers.append("provider_output_put_url_not_ready")
    if (
        verify_staging_urls
        and not allow_unverified_public_staging_for_paid_launch
        and public_status != "passed"
    ):
        blockers.append("public_staging_url_verification_failed")
    if not verify_staging_urls and not allow_unverified_public_staging_for_paid_launch:
        blockers.append("public_staging_urls_not_verified_for_paid_launch")
    return {
        "schema_version": "vast_authorized_probe_staging_verification_guard.v1",
        "status": "blocked" if blockers else "passed",
        "verify_staging_urls": verify_staging_urls,
        "allow_unverified_public_staging_for_paid_launch": (
            allow_unverified_public_staging_for_paid_launch
        ),
        "provider_fetchable_bundle_uri_ready": bundle_url_ready,
        "provider_output_callback_ready": output_url_ready,
        "public_staging_verification_status": public_status,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }


def bounded_container_missing_retry_attempts(
    *,
    max_wait_seconds: int,
    retry_interval_seconds: int,
    max_missing_seconds: int,
) -> int:
    """Bound cold-container tolerance below the run deadline.

    Vast can expose a contract as running while Docker is still pulling the
    image and the named container does not yet exist. This window is long
    enough for a credible cold pull, but it cannot idle the entire paid run.
    """

    window = min(max(1, int(max_wait_seconds)), max(60, int(max_missing_seconds)))
    return max(1, int(window / max(1, int(retry_interval_seconds))))


def cold_pull_aware_heartbeat_no_progress_seconds(
    *,
    configured_seconds: int,
    provider_bundle_kind: str,
    allow_cold_image_pull: bool,
    min_cold_image_pull_live_minutes: int,
    startup_timeout_seconds: int,
    max_live_minutes: int,
) -> int:
    """Keep the heartbeat window consistent with an admitted WAM cold pull."""

    resolved = max(0, int(configured_seconds))
    # Every kind here ships a multi-gigabyte image. The list started as
    # {"wam"}, grew "paired_target_native_import" when that lane died
    # mid-pull, and on 2026-08-18 "adp_simready_isaac" repeated the same
    # failure twice on consecutive cold machines: the admission gate allowed
    # the cold pull, the live window was sized for it, and this window --
    # which never heard about the isaac lane -- expired first, every time,
    # deterministically, while looking like a provider flake.
    if (
        provider_bundle_kind
        not in {"wam", "paired_target_native_import", "adp_simready_isaac"}
        or not allow_cold_image_pull
    ):
        return resolved
    admitted_cold_pull_seconds = min(
        max(0, int(min_cold_image_pull_live_minutes)) * 60,
        max(0, int(startup_timeout_seconds)),
        max(0, int(max_live_minutes)) * 60,
    )
    return max(resolved, admitted_cold_pull_seconds)
