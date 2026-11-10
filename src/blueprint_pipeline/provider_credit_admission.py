"""Read-only provider funding evidence; never fund, allocate, or stop a resource.

Production can require a fresh balance before allocating. Existing billing and
teardown paths deliberately do not depend on this check. A balance is a snapshot,
not an authorization to spend it. Launches admitted in other slots hold their
caps against it (``vast_credit_holds``): an admission with a job directory takes
a hold under one lock, and a read-only check subtracts the open holds.
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
#: The capacity tick pages while the Vast balance cannot cover two more launches'
#: $5 compute allowance on top of the reserve (owner decision D-09: page at twice
#: what the next launch needs), so funding is topped up before a launch is refused.
PAGE_CREDIT_USD = 10.0


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
        from .provider_credentials import _read_secret
        api_key = _read_secret("vast_api_key")
    if request is None:
        from .vast_api_transport import _api_json
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
    now: float | None = None, maximum_age_seconds: float = 60, held_usd: float = 0.0,
) -> dict[str, Any]:
    """``held_usd`` is what launches admitted in other slots still hold (``vast_credit_holds``)."""
    observed = _amount(observation.get("observed_at_epoch"))
    current = time.time() if now is None else now
    credit = _amount(observation.get("credit_usd"))
    required, reserve, held = _amount(required_usd), _amount(reserve_usd), _amount(held_usd)
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
    if required is None or required <= 0 or reserve is None or reserve < 0 or held is None or held < 0:
        blockers.append("provider_credit_requirement_invalid")
    elif credit is not None and credit < required + reserve + held:
        blockers.append("provider_credit_insufficient")
    return {"status": "blocked" if blockers else "admitted", "blockers": blockers,
            "required_usd": required, "reserve_usd": reserve, "held_by_other_launches_usd": held,
            "observation": dict(observation), "credit_reserved": False}


def credit_guard_setting() -> str:
    """``enabled``, ``disabled`` or ``invalid``; the guard is on unless a unit opts out.

    The default is on (P-D01, 2026-11-05) so a new unit that reaches a Vast
    launcher fails safe. Only an explicit ``false`` or ``0`` turns it off; an
    unset or empty value means on, and any other value is a config error that
    refuses the launch.
    """

    value = os.getenv(ENABLED_ENV, "true").strip().lower()
    if value in {"true", "1", ""}:
        return "enabled"
    if value in {"false", "0"}:
        return "disabled"
    return "invalid"


def configured_vast_credit_admission(*, api_key: str, required_usd: float,
                                     job_dir: Path | None = None) -> dict[str, Any]:
    """With ``job_dir`` an admitted launch holds its cap; without, the open holds are only subtracted."""
    setting = credit_guard_setting()
    if setting == "disabled":
        return {"status": "not_configured", "blockers": []}
    if setting == "invalid":
        return {"status": "blocked", "blockers": ["provider_credit_guard_config_invalid"]}
    try:
        reserve = float(os.getenv(RESERVE_ENV, "1"))
    except ValueError:
        return {"status": "blocked", "blockers": ["provider_credit_guard_config_invalid"]}
    from . import vast_credit_holds as holds

    observation = observe_vast_credit(api_key=api_key)
    if job_dir is not None:
        return holds.admit_and_hold(
            lambda held: credit_admission(observation, required_usd=required_usd, reserve_usd=reserve,
                                          held_usd=held),
            job_dir=job_dir, required_usd=required_usd)
    held, unreadable = holds.open_holds_usd()
    if unreadable:
        return {"status": "blocked", "blockers": unreadable, "credit_reserved": False}
    return credit_admission(observation, required_usd=required_usd, reserve_usd=reserve, held_usd=held or 0.0)


def record_vast_credit_admission(job_dir: Path, api_key: str, required_usd: float) -> dict[str, Any]:
    """Called under the adapter's launch lock, before its no-allocation failure path."""
    from .common import write_json

    result = configured_vast_credit_admission(api_key=api_key, required_usd=required_usd, job_dir=job_dir)
    write_json(job_dir / "provider_credit_admission.json", result)
    return result


def render_credit_admission(request: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    guard = request.get("prelaunch_spend_guard")
    amount = guard.get("max_spend_usd") if isinstance(guard, Mapping) else None
    return configured_vast_credit_admission(api_key=api_key, required_usd=amount)
