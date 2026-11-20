"""ARTISAN's provider spend from the usage meter retained by the scene run.

No Anthropic bill is read, so without this the month (``account_monthly_spend_report``) would
never see an Anthropic dollar. Each attempt is held at its ARTISAN allowance before it starts and
settled to its meter's Anthropic subtotal afterwards; a lost run keeps its hold until its TTL ends.
The report counts the row as ``metered_only``, labelled ``metered_list_price_not_officially_read``.
A run that ended on Anthropic's no-funds refusal also turns its grant to OpenAI (rule (e)).

OpenAI's bill is read separately. The archive attributes the run's priced OpenAI calls
to its job, including the OpenAI design board on a run that uses Anthropic text.
"""
from __future__ import annotations

import json
import math
import os
import zipfile
import zlib
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

PROVIDER = "anthropic"
BASIS = "metered_list_price_not_officially_read"
RESOURCE_CLASS = "artisan_anthropic_messages"
USAGE_MEMBER = "stages/stage-3/producer/artisan_runtime/blueprint_usage.json"
STAGE_INPUT_MEMBER = "stages/stage-3/producer/stage_production_input.v1.json"
NO_FUNDS_ORIGIN = "upstream_rejected_request_no_funds"
MAX_USAGE_BYTES = 2 * 1024**2
METER_ROUNDING_TOLERANCE_USD = Decimal("0.000001")


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
                 job_id: str | None = None, environ: Mapping[str, str] | None = None,
                 now: float | None = None) -> bool:
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
    record_admission(job_id=job_id or _job(receipt["run_id"], environ), meter_key=_meter_key(receipt),
                     resource_class=RESOURCE_CLASS, held_usd=allowance, provider=PROVIDER, root=root, now=at,
                     basis=BASIS, **{BILLABLE_UNTIL_FIELD: at + float(ttl_seconds)})
    return True


def _member(archive: Path, name: str) -> Any:
    try:
        with zipfile.ZipFile(archive) as opened:
            return _read_member(opened, name)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _read_member(archive: zipfile.ZipFile, name: str) -> Any:
    info = archive.getinfo(name)
    return None if info.file_size > MAX_USAGE_BYTES else json.loads(archive.read(info))


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
                   job_id: str | None = None, now: float | None = None) -> bool:
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
        record_settlement(job_id=job_id or _job(receipt["run_id"], environ), meter_key=_meter_key(receipt),
                          settled_usd=subtotal,
                          basis=BASIS, provider=PROVIDER, root=root, now=now)
    except (OSError, ValueError):
        return False
    return True


def _metered_openai(archive: Path, receipt: Mapping[str, Any], *,
                    stage_cap: Decimal) -> list[tuple[str, Decimal, bool]] | None:
    """Price only recognizable OpenAI usage from the current run's own ARTISAN meter."""
    from .paid_spend_meter import money

    if archive.is_symlink():
        return None
    try:
        with zipfile.ZipFile(archive) as opened:
            names = opened.namelist()
            if names.count(STAGE_INPUT_MEMBER) != 1 or names.count(USAGE_MEMBER) != 1:
                return None
            stage = _read_member(opened, STAGE_INPUT_MEMBER)
            usage = _read_member(opened, USAGE_MEMBER)
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile, zlib.error):
        return None
    if not isinstance(stage, Mapping) or (stage.get("run_id"), stage.get("source_commit")) != (
            receipt["run_id"], receipt["source_commit"]):
        return None
    configuration = stage.get("configuration")
    if (not isinstance(configuration, Mapping) or configuration.get("authoring_backend") != "artisan_v1"
            or not isinstance(usage, Mapping) or usage.get("schema_version") != "artisan_upstream_usage.v1"
            or not isinstance(usage.get("calls"), list)):
        return None
    meter_cap = money(usage.get("maximum_cost_usd"))
    if meter_cap is None or not 0 < meter_cap <= stage_cap:
        return None
    seen, charges = set(), []
    for call in usage["calls"]:
        if not isinstance(call, Mapping) or not isinstance(call.get("id"), str) or not call["id"] or call["id"] in seen:
            return None
        seen.add(call["id"])
        kind, origin = call.get("kind"), call.get("cost_origin")
        image = kind == "generated_design_board" and origin == "upstream_image_budget_estimate"
        responses = kind in ("create", "compact") and origin == "upstream_priced_usage"
        # A failed call's upper bound is useful to the stage guard. The provider bill owns its actual cost.
        if call.get("status") != "priced" or not (image or responses):
            continue
        cost = call.get("cost_usd")
        amount = money(cost) if isinstance(cost, (int, float)) else None
        if amount is None:
            return None
        charges.append((call["id"], amount, image))
    total = sum((amount for _, amount, _ in charges), Decimal(0))
    return charges if total <= meter_cap + METER_ROUNDING_TOLERANCE_USD else None


def settle_openai_attempt(receipt: Mapping[str, Any], authority: Mapping[str, Any], *, archives: tuple[Path, ...],
                          job_id: str | None, environ: Mapping[str, str] | None = None,
                          now: float | None = None) -> bool:
    """Journal the job's OpenAI usage by call ID, so overlapping archives count each call once."""
    if not job_id or receipt.get("replacement_authoring_backend") != "artisan_v1":
        return False
    root = _root(environ)
    if root is None:
        return False
    from .paid_spend_meter import ARTISAN_OPENAI_METER_PREFIX, money, record_settlement

    openai = authority["external_service_spend_caps"].get("openai") or {}
    stage_cap = money((openai.get("stage_max_cost_usd") or {}).get("content_agents"))
    if stage_cap is None:
        return False

    charges = {}
    for archive in dict.fromkeys(archives):
        for call_id, cost, estimated in _metered_openai(archive, receipt, stage_cap=stage_cap) or ():
            if cost:
                charges.setdefault(call_id, (cost, estimated))
    if sum((cost for cost, _ in charges.values()), Decimal(0)) > stage_cap + METER_ROUNDING_TOLERANCE_USD:
        return False
    written = False
    for call_id, (cost, estimated) in charges.items():
        basis = "artisan_metered_image_estimate" if estimated else "artisan_measured_usage"
        try:
            record_settlement(job_id=job_id, meter_key=ARTISAN_OPENAI_METER_PREFIX + call_id,
                              settled_usd=cost, basis=basis, provider="openai", root=root, now=now)
        except (OSError, ValueError):
            continue
        written = True
    return written


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
