"""Release qualification, quarantine, and promotion from campaign evidence."""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "production_gpu_reliability_qualification.v1"


def qualify_release(
    *,
    release_fingerprint: str,
    campaign_snapshots: Sequence[Mapping[str, Any]],
    bind_latencies_seconds: Sequence[float],
    cold_replenishment_seconds: Sequence[float],
    minimum_campaigns: int = 3,
    minimum_attempt_pass_rate: float = 0.95,
    warm_bind_p95_slo_seconds: float = 10.0,
    cold_replenishment_p95_slo_seconds: float = 1800.0,
    rollback_drill_passed: bool = False,
) -> dict[str, Any]:
    """Fail closed: local or incomplete samples can never promote a release."""

    def percentile95(values: Sequence[float]) -> float | None:
        rows = sorted(float(value) for value in values)
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]
        return statistics.quantiles(rows, n=100, method="inclusive")[94]

    campaigns = [dict(value) for value in campaign_snapshots]
    attempts = [dict(row) for campaign in campaigns for row in campaign.get("attempts", [])]
    passed = sum(row.get("state") == "passed" for row in attempts)
    pass_rate = passed / len(attempts) if attempts else 0.0
    bind_p95 = percentile95(bind_latencies_seconds)
    cold_p95 = percentile95(cold_replenishment_seconds)
    checks = {
        "minimum_campaign_sample": len(campaigns) >= int(minimum_campaigns),
        "all_campaigns_terminal": bool(campaigns)
        and all(value.get("terminal") is True for value in campaigns),
        "all_campaigns_exact_release": bool(campaigns)
        and all(
            value.get("release_fingerprint", release_fingerprint) == release_fingerprint
            for value in campaigns
        ),
        "attempt_pass_rate": bool(attempts) and pass_rate >= float(minimum_attempt_pass_rate),
        "warm_bind_p95": bind_p95 is not None and bind_p95 <= float(warm_bind_p95_slo_seconds),
        "cold_replenishment_p95": cold_p95 is not None
        and cold_p95 <= float(cold_replenishment_p95_slo_seconds),
        "rollback_drill": rollback_drill_passed is True,
    }
    blockers = [
        f"qualification_failed:{name}" for name, passed_check in checks.items() if not passed_check
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "release_fingerprint": release_fingerprint,
        "status": "promoted" if not blockers else "quarantined",
        "checks": checks,
        "metrics": {
            "campaign_count": len(campaigns),
            "attempt_count": len(attempts),
            "attempt_pass_rate": pass_rate,
            "warm_bind_p95_seconds": bind_p95,
            "cold_replenishment_p95_seconds": cold_p95,
        },
        "blockers": blockers,
        "promotion_requires_fresh_live_evidence": True,
    }
