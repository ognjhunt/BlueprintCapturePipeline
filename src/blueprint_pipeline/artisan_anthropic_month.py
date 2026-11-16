"""ARTISAN's Anthropic spend in the account's month, at list price, as provider ``anthropic``.

No Anthropic bill is read, so without this the month (``account_monthly_spend_report``) would
never see an Anthropic dollar. Each attempt is held at its ARTISAN allowance before it starts and
settled to its meter's Anthropic subtotal afterwards; a lost run keeps its hold until its TTL ends.
The report counts the row as ``metered_only``, labelled ``metered_list_price_not_officially_read``.
A run that ended on Anthropic's no-funds refusal also turns its grant to OpenAI (rule (e)).
"""
from __future__ import annotations

import json
import math
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROVIDER = "anthropic"
BASIS = "metered_list_price_not_officially_read"
RESOURCE_CLASS = "artisan_anthropic_messages"
USAGE_MEMBER = "stages/stage-3/producer/artisan_runtime/blueprint_usage.json"
STAGE_INPUT_MEMBER = "stages/stage-3/producer/stage_production_input.v1.json"
NO_FUNDS_ORIGIN = "upstream_rejected_request_no_funds"
MAX_USAGE_BYTES = 2 * 1024**2


def _root(environ: Mapping[str, str] | None) -> Path | None:
    from .paid_spend_meter import meter_root
    return meter_root(environ)


def _job(fallback: Any, environ: Mapping[str, str] | None) -> str:
    from .paid_spend_meter import JOB_ID_ENV
    values = os.environ if environ is None else environ
    return str(values.get(JOB_ID_ENV) or fallback)


def _amount(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _meter_key(receipt: Mapping[str, Any]) -> str:
    return "artisan-anthropic-attempt:" + str(receipt["run_id"])


def hold_attempt(receipt: Mapping[str, Any], authority: Mapping[str, Any], *, ttl_seconds: float,
                 environ: Mapping[str, str] | None = None, now: float | None = None) -> bool:
    """Hold the allowance as Anthropic spend; on the host an unwritable hold raises, so the run never starts."""
    if (receipt or {}).get("replacement_authoring_text_provider") != PROVIDER:
        return False
    root = _root(environ)
    if root is None:
        return False
    import time
    from .paid_spend_meter import BILLABLE_UNTIL_FIELD, record_admission
    at = time.time() if now is None else float(now)
    allowance = authority["external_service_spend_caps"]["openai"]["stage_max_cost_usd"]["content_agents"]
    record_admission(job_id=_job(receipt["run_id"], environ), meter_key=_meter_key(receipt),
                     resource_class=RESOURCE_CLASS, held_usd=allowance, provider=PROVIDER, root=root, now=at,
                     basis=BASIS, **{BILLABLE_UNTIL_FIELD: at + float(ttl_seconds)})
    return True


def _member(archive: Path, name: str) -> Any:
    try:
        with zipfile.ZipFile(archive) as opened:
            info = opened.getinfo(name)
            return None if info.file_size > MAX_USAGE_BYTES else json.loads(opened.read(info))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _calls(archive: Path) -> list[Any] | None:
    usage = _member(archive, USAGE_MEMBER)
    calls = usage.get("calls") if isinstance(usage, Mapping) else None
    return calls if isinstance(calls, list) else None


def metered_anthropic_subtotal(archive: Path) -> float | None:
    """The run's Anthropic calls from its own meter; ``None`` if unreadable."""
    calls = _calls(archive)
    if calls is None:
        return None
    total = 0.0
    for call in calls:
        if not isinstance(call, Mapping) or call.get("kind") != "messages":
            continue
        # A call never priced counts at the most it could have cost.
        cost = _amount(call.get("cost_usd")) if call.get("status") == "priced" else _amount(call.get("reserved_usd"))
        if cost is None:
            return None
        total += cost
    return round(total, 6)


def settle_attempt(receipt: Mapping[str, Any], archive: Path | None, *, environ: Mapping[str, str] | None = None,
                   now: float | None = None) -> bool:
    """Settle the attempt's hold to its metered Anthropic subtotal; unreadable output keeps the hold."""
    if (receipt or {}).get("replacement_authoring_text_provider") != PROVIDER or archive is None:
        return False
    root = _root(environ)
    subtotal = metered_anthropic_subtotal(Path(archive)) if Path(archive).is_file() else None
    if subtotal is not None:
        ended_without_funds(receipt, Path(archive), environ=environ)
    if root is None or subtotal is None:
        return False
    from .paid_spend_meter import record_settlement
    try:
        record_settlement(job_id=_job(receipt["run_id"], environ), meter_key=_meter_key(receipt), settled_usd=subtotal,
                          basis=BASIS, provider=PROVIDER, root=root, now=now)
    except (OSError, ValueError):
        return False
    return True


def ended_without_funds(receipt: Mapping[str, Any], archive: Path, *,
                        environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The grant's OpenAI successor, when this run's last call was Anthropic's typed no-funds refusal."""
    calls = _calls(archive) or []
    last = calls[-1] if calls and isinstance(calls[-1], Mapping) else {}
    stage_input = _member(archive, STAGE_INPUT_MEMBER)
    configuration = stage_input.get("configuration") if isinstance(stage_input, Mapping) else None
    if (last.get("kind") != "messages" or last.get("cost_origin") != NO_FUNDS_ORIGIN
            or not isinstance(configuration, Mapping) or stage_input.get("run_id") != receipt.get("run_id")):
        return None
    from .artisan_provider_choice import record_no_funds_successor
    try:
        return record_no_funds_successor(configuration.get("artisan_provider_choice"), run_id=str(receipt["run_id"]),
                                         environ=environ)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
