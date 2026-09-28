"""Read-only provider funding evidence; never fund, allocate, or stop a resource.

Production can require a fresh balance before allocating. Existing billing and
teardown paths deliberately do not depend on this check. A balance is a snapshot,
not a reservation or an authorization to spend it.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from typing import Any
from pathlib import Path

from .decision_evidence_contracts import canonical_digest

ENABLED_ENV = "BLUEPRINT_VAST_CREDIT_GUARD_ENABLED"
RESERVE_ENV = "BLUEPRINT_VAST_CREDIT_RESERVE_USD"
WARNING_ENV = "BLUEPRINT_VAST_CREDIT_WARNING_USD"


def _amount(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def observe_vast_credit(
    *, api_key: str | None = None, request: Callable[..., Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Retain only credit and HTTP status, never the account response or error text."""
    if api_key is None:
        from .gpu_render_providers import VastRenderProvider
        api_key = VastRenderProvider()._key()
    if request is None:
        from .vast_provider_adapter import _api_json
        request = _api_json
    status, credit, blocker = None, None, "provider_credit_credentials_missing"
    if api_key:
        try:
            status, body = request(method="GET", path="/users/current/", api_key=api_key,
                                   timeout_seconds=10)
            if status == 200 and isinstance(body, Mapping):
                credit = _amount(body.get("credit"))
            blocker = "provider_credit_unverifiable" if credit is None else None
        except Exception:  # transport exceptions may contain tokens or account details
            blocker = "provider_credit_transport_failed"
    row = {
        "schema_version": "provider_credit_observation.v1", "provider": "vast",
        "observed_at_epoch": time.time() if now is None else now,
        "status": "observed" if blocker is None else "unknown",
        "http_status": status if isinstance(status, int) else None,
        "credit_usd": credit, "blockers": [blocker] if blocker else [],
        "provider_mutations_performed": 0, "raw_account_response_recorded": False,
    }
    row["observation_digest"] = canonical_digest(row, digest_field="observation_digest")
    return row


def credit_admission(
    observation: Mapping[str, Any], *, required_usd: float, reserve_usd: float = 1.0,
    now: float | None = None, maximum_age_seconds: float = 60,
) -> dict[str, Any]:
    observed = _amount(observation.get("observed_at_epoch"))
    current = time.time() if now is None else now
    credit = _amount(observation.get("credit_usd"))
    required, reserve = _amount(required_usd), _amount(reserve_usd)
    blockers = []
    if (observation.get("schema_version") != "provider_credit_observation.v1"
            or observation.get("provider") != "vast"
            or observation.get("observation_digest") != canonical_digest(
                observation, digest_field="observation_digest")
            or observation.get("status") != "observed" or observation.get("http_status") != 200
            or observation.get("blockers") != [] or credit is None):
        blockers.append("provider_credit_unverifiable")
    if observed is None or not 0 <= current - observed <= maximum_age_seconds:
        blockers.append("provider_credit_observation_stale")
    if required is None or required <= 0 or reserve is None or reserve < 0:
        blockers.append("provider_credit_requirement_invalid")
    elif credit is not None and credit < required + reserve:
        blockers.append("provider_credit_insufficient")
    return {"status": "blocked" if blockers else "admitted", "blockers": blockers,
            "required_usd": required, "reserve_usd": reserve,
            "observation": dict(observation), "credit_reserved": False}


def configured_vast_credit_admission(*, api_key: str, required_usd: float) -> dict[str, Any]:
    enabled = os.getenv(ENABLED_ENV, "false").strip().lower()
    if enabled in {"false", "0", ""}:
        return {"status": "not_configured", "blockers": []}
    if enabled not in {"true", "1"}:
        return {"status": "blocked", "blockers": ["provider_credit_guard_config_invalid"]}
    try:
        reserve = float(os.getenv(RESERVE_ENV, "1"))
    except ValueError:
        return {"status": "blocked", "blockers": ["provider_credit_guard_config_invalid"]}
    return credit_admission(observe_vast_credit(api_key=api_key), required_usd=required_usd,
                            reserve_usd=reserve)


def record_vast_credit_admission(job_dir: Path, api_key: str, required_usd: float) -> dict[str, Any]:
    """Called under the adapter's launch lock, before its no-allocation failure path."""
    from .common import write_json

    result = configured_vast_credit_admission(api_key=api_key, required_usd=required_usd)
    write_json(job_dir / "provider_credit_admission.json", result)
    return result


def render_credit_admission(request: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    guard = request.get("prelaunch_spend_guard")
    amount = guard.get("max_spend_usd") if isinstance(guard, Mapping) else None
    return configured_vast_credit_admission(api_key=api_key, required_usd=amount)
