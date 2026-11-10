"""When the account's month reaches its limit at this month's measured daily rate.

Part of the account monthly spend report (``account_monthly_spend_report``),
so the owner hears long before the limit refuses anything. Nothing here admits
or refuses: the forecast and its alert gate nothing.

The daily rate is the month's measured provider spend (``spend_usd``, held
reservations excluded: they are upper bounds, not measurements) over the days
of the UTC month elapsed. From the month's counted figure (``counted_usd``,
holds included, as admission counts it), that rate reaches the limit on a
date, which reads as that date, ``not this month``, or ``unknown`` while less
than ``MIN_MEASURED_SECONDS`` of the month has passed. A report with gaps
(``complete`` false) measured a lower bound, and the forecast says so.

The alert (``alert: true``) stands while the date falls inside the month. It
is raised at most once a UTC day: once cleared, it cannot be raised again
until the next day (``alert_raised_on``, carried from the previous report of
the same month).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .paid_spend_meter import month_start, usd

SCHEMA = "blueprint.account_monthly_spend_forecast.v1"
DAY_SECONDS = 86_400
#: Less than a day of the month is too little to call a daily rate.
MIN_MEASURED_SECONDS = DAY_SECONDS


def _next_month_start(now: float) -> float:
    start = datetime.fromtimestamp(month_start(now), timezone.utc)
    return (start.replace(year=start.year + 1, month=1) if start.month == 12
            else start.replace(month=start.month + 1)).timestamp()


def forecast(*, now: float, spend_usd: Decimal, counted_usd: Decimal, limit_usd: Decimal | None,
             complete: bool, previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    elapsed = now - month_start(now)
    row: dict[str, Any] = {"schema_version": SCHEMA, "daily_rate_usd": None, "reaches_limit_on": None,
                           "rate_is_lower_bound": not complete, "measured_days": round(elapsed / DAY_SECONDS, 3)}
    if limit_usd is None:
        row.update(status="no_limit", reads="unknown", reason="account_monthly_spend_limit_not_set")
    elif counted_usd >= limit_usd:
        row.update(status="limit_reached", reads="reached")
    elif elapsed < MIN_MEASURED_SECONDS:
        row.update(status="unknown", reads="unknown", reason="less_than_one_day_of_the_month_measured")
    else:
        rate = spend_usd * DAY_SECONDS / Decimal(str(elapsed))
        row["daily_rate_usd"] = usd(rate)
        reach = None if rate <= 0 else now + float((limit_usd - counted_usd) / rate) * DAY_SECONDS
        if reach is None or reach >= _next_month_start(now):
            row.update(status="not_this_month", reads="not this month")
        else:
            on = datetime.fromtimestamp(reach, timezone.utc).date().isoformat()
            row.update(status="reaches_limit_this_month", reads=on, reaches_limit_on=on)
    inside = row["status"] == "reaches_limit_this_month"
    before = previous if isinstance(previous, Mapping) and previous.get("schema_version") == SCHEMA else {}
    raised_on = before.get("alert_raised_on")
    if inside and before.get("alert") is True:
        alert = True
    elif inside and raised_on != today:
        alert, raised_on = True, today
    else:
        alert = False
    return {**row, "alert": alert, "alert_raised_on": raised_on}


__all__ = ["MIN_MEASURED_SECONDS", "SCHEMA", "forecast"]
