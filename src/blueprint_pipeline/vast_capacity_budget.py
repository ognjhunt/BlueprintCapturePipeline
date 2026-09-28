"""Transfer-aware pricing helpers for read-only Vast capacity admission."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def expected_transfer_bytes(
    request: Mapping[str, Any],
) -> tuple[int, int, list[str]]:
    """Parse non-negative exact transfer ceilings without accepting bools."""
    parsed: list[int] = []
    blockers: list[str] = []
    for field in (
        "expected_provider_download_bytes",
        "expected_provider_upload_bytes",
    ):
        raw = request.get(field, 0)
        if type(raw) is not int or raw < 0:
            blockers.append(f"vast_capacity_{field}_invalid")
            parsed.append(0)
        else:
            parsed.append(raw)
    return parsed[0], parsed[1], blockers


def bind_transfer_aware_budget(
    offers: list[dict[str, Any]],
    *,
    hard_ttl_seconds: int | None,
    hard_cap_usd: float | None,
    expected_provider_download_bytes: int,
    expected_provider_upload_bytes: int,
    projected_transfer_cost: Callable[..., float | None],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Price runtime plus transfer and retain only offers under the hard cap."""
    if hard_ttl_seconds is not None:
        for offer in offers:
            runtime_cost = float(offer["hourly_rate_usd"]) * hard_ttl_seconds / 3600.0
            transfer_cost = projected_transfer_cost(
                offer,
                expected_provider_download_bytes=expected_provider_download_bytes,
                expected_provider_upload_bytes=expected_provider_upload_bytes,
            )
            offer["projected_runtime_cost_usd"] = runtime_cost
            offer["projected_provider_transfer_cost_usd"] = transfer_cost
            offer["projected_full_ttl_cost_usd"] = (
                runtime_cost + transfer_cost if transfer_cost is not None else None
            )
    if hard_cap_usd is None or hard_ttl_seconds is None:
        return offers, []
    viable = [
        offer
        for offer in offers
        if offer.get("projected_full_ttl_cost_usd") is not None
        and float(offer["projected_full_ttl_cost_usd"]) <= hard_cap_usd
    ]
    return viable, ([] if viable else ["vast_capacity_full_ttl_exceeds_hard_cap"])
