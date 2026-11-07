"""Page when OpenAI, World Labs or Google Cloud spend crosses a line; never spend or block.

Beta tracker B-08: provider funding is monitored and paged. Vast pages on its
balance (C-07, ``provider_credit_admission``). This host can read no balance
for the other three, so each capacity tick (``control_plane_capacity_controller``)
pages on spend instead:

OpenAI
    The organization Costs API (``openai_organization_costs``), read with the
    Admin key that the launch dispatcher already uses (``OPENAI_ADMIN_API_KEY_FILE``).
    It counts every project in the organization. It reports whole UTC days and
    its data lags, so the 24 h line is checked against today so far and against
    yesterday, whose late costs keep arriving after midnight. A page therefore
    comes only after OpenAI reports the cost, and it can last until the UTC day
    that crossed the line is two days old.
World Labs (Atlas)
    The Atlas API has no balance, usage or billing endpoint. This reads
    Blueprint's own record of each Atlas attempt under the website job roots:
    its admission (the reviewed price) and its submission (send time and
    outcome). A completed operation counts at its reviewed price, one that may
    still be charged counts in full, and a failed or refused one counts zero,
    the rule in ``atlas_spend_settlement``. The balance itself is on the World
    Labs dashboard.
Google Cloud project ``blueprint-8c1ca`` (Gemini bills here)
    The monthly budget guard's own report (``gcp_monthly_budget_guard``). It
    pages at 80 % of the month's effective limit and never changes the limit.
The account (owner decision 2026-11-07, "Meter + monthly cap")
    The spend guard's account-wide monthly report (``account_monthly_spend``):
    a warn at 50 % of the limit, a page at 80 % and at 100 % (where new paid
    admissions are refused), and a warn that says no cap is set while none is
    configured. Each tick also leaves its OpenAI and World Labs readings where
    that report reads them (``configured_spend_check``).

A page code repeats hourly on the capacity alert route while it lasts and makes
the capacity level critical, as the Vast page does. Missing credentials, an
unreadable source, a malformed answer or a failed request are warn codes: they
never page and raise the level to warning at most. Nothing here reserves, admits
or refuses work, and no launch reads it: the account limit is enforced from the
spend guard's report, which only takes these readings as an input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .account_monthly_spend import ACCOUNT_PAGE_CODES, ACCOUNT_WARN_CODES
from .account_monthly_spend import account_alerts as _account_alerts
from .account_monthly_spend import describe as _describe_account
from .account_monthly_spend import observe_account_spend, persist_provider_observation
from .gcp_monthly_budget_admission import CONFIG_ENV as GCP_BUDGET_CONFIG_ENV

SCHEMA_VERSION = "provider_spend_observation.v1"
DISABLE_ENV = "BLUEPRINT_PROVIDER_SPEND_ALERTS"
#: The same names, and the same files, as the lanes that spend.
OPENAI_ADMIN_KEY_FILE_ENV = "OPENAI_ADMIN_API_KEY_FILE"
ATLAS_STORAGE_ROOT_ENV = "BLUEPRINT_WEBSITE_ATLAS_PREVIEW_STORAGE_ROOT"
DEFAULT_ATLAS_STORAGE_ROOT = Path("/var/lib/blueprint/pubsub-handoffs")
ATLAS_JOB_GLOB = "*/scenes/*/captures/*/pipeline/atlas"

OPENAI_24H_PAGE_ENV = "BLUEPRINT_OPENAI_SPEND_24H_PAGE_USD"
OPENAI_MONTH_PAGE_ENV = "BLUEPRINT_OPENAI_SPEND_MONTH_PAGE_USD"
WORLD_LABS_24H_PAGE_ENV = "BLUEPRINT_WORLD_LABS_SPEND_24H_PAGE_USD"
WORLD_LABS_MONTH_PAGE_ENV = "BLUEPRINT_WORLD_LABS_SPEND_MONTH_PAGE_USD"
GCP_PAGE_FRACTION_ENV = "BLUEPRINT_GCP_SPEND_PAGE_FRACTION"

#: B-03 (docs/launch-readiness/2026-11-cost-model.md, sections 2 and 6): one
#: website job's OpenAI cost at its caps is ARTISAN's cumulative $25 cap, plus
#: $0.91 of Atlas planner and review calls (measured), plus the $5 upstream
#: ledger that SAM, Gemini and other OpenAI calls share. capture-447 measured
#: $17.02 + $0.91.
OPENAI_JOB_USD = 25.0 + 0.91 + 5.0
#: One job's World Labs cost at its cap: the $25 default Atlas allowance (B-03
#: section 7, item 5). capture-447 cost about $8.60 at rate-card prices: the
#: completed splat and mesh, $5.50, plus about $3.10 (estimated) of poses and views.
WORLD_LABS_JOB_USD = 25.0
#: B-03 section 5 found 15 website jobs in the 30 days to 2026-11-02 (0.5 a
#: day); the WebApp admits at most 5 new jobs a day. A beta day is planned at 2
#: jobs and a beta month at 15.
BETA_JOBS_PER_DAY = 2
BETA_JOBS_PER_MONTH = 15
#: 20 % for spend outside website jobs (development, operator runs), rounded up to $5.
LINE_MARGIN = 1.2
OPENAI_24H_PAGE_USD = 75.0         # 2 x $30.91 x 1.2 = $74.18
OPENAI_MONTH_PAGE_USD = 560.0      # 15 x $30.91 x 1.2 = $556.38
WORLD_LABS_24H_PAGE_USD = 60.0     # 2 x $25 x 1.2
WORLD_LABS_MONTH_PAGE_USD = 450.0  # 15 x $25 x 1.2
#: The project's month pages at 80 % of the budget guard's effective limit
#: ($80 of the $100 ceiling), well before the guard refuses paid work at 100 %.
GCP_PAGE_FRACTION = 0.8

PAGE_CODES = frozenset({
    "openai_spend_24h_high", "openai_spend_month_high",
    "world_labs_spend_24h_high", "world_labs_spend_month_high",
    "gcp_spend_month_high",
    *ACCOUNT_PAGE_CODES,
})
#: Warn only: the check could not see. ``provider_spend_check_failed`` is the
#: capacity controller's own, for a check that raised.
WARN_CODES = frozenset({
    "openai_spend_credentials_missing", "openai_spend_transport_failed", "openai_spend_unverifiable",
    "world_labs_spend_unverifiable", "gcp_spend_unverifiable", "provider_spend_config_invalid",
    "provider_spend_check_failed",
    *ACCOUNT_WARN_CODES,
})

DAY_SECONDS = 86_400
OPENAI_TIMEOUT_SECONDS = 10.0
#: The Atlas walk stays narrow and bounded, so it cannot starve the host's other
#: timers; a walk cut short reports a lower bound and a warn.
SCAN_DEADLINE_SECONDS = 20.0
SCAN_MAX_DIRECTORIES = 50_000
MAX_JSON_BYTES = 4 * 1024 * 1024
#: The budget guard rewrites its report every few minutes.
GCP_REPORT_MAX_AGE_SECONDS = 30 * 60

_LINES = (
    ("openai_24h_usd", OPENAI_24H_PAGE_ENV, OPENAI_24H_PAGE_USD),
    ("openai_month_usd", OPENAI_MONTH_PAGE_ENV, OPENAI_MONTH_PAGE_USD),
    ("world_labs_24h_usd", WORLD_LABS_24H_PAGE_ENV, WORLD_LABS_24H_PAGE_USD),
    ("world_labs_month_usd", WORLD_LABS_MONTH_PAGE_ENV, WORLD_LABS_MONTH_PAGE_USD),
    ("gcp_fraction", GCP_PAGE_FRACTION_ENV, GCP_PAGE_FRACTION),
)
_LABELS = {"openai": "OpenAI", "world_labs": "World Labs", "gcp": "Google Cloud", "account": "Account"}
_SUMMARY_KEYS = (
    "status", "blockers", "reason", "http_status", "month", "day_usd", "month_to_date_usd", "spend_24h_usd",
    "held_usd", "attempts", "complete", "project_id", "period", "observed_cost_usd", "limit_usd", "generated_at",
    "cap_status", "spend_usd", "counted_usd", "fraction", "alert_threshold_percent", "alert_lines_usd",
    "limit_reached", "admission_allowed", "gaps",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


def _month_start(now: float) -> int:
    current = _utc(now)
    return int(datetime(current.year, current.month, 1, tzinfo=timezone.utc).timestamp())


def _small_json(path: Path) -> dict[str, Any] | None:
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def default_lines() -> dict[str, float]:
    return {key: default for key, _name, default in _LINES}


def lines_from_environment(environ: Mapping[str, str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Each page line, from its override when that is a usable number; a bad one keeps the default."""

    lines, warns = default_lines(), []
    for key, name, _default in _LINES:
        raw = str(environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value <= 0 or (key == "gcp_fraction" and value > 1):
            warns.append({"code": "provider_spend_config_invalid", "reason": name})
        else:
            lines[key] = value
    return lines, warns


def observe_openai_spend(*, now: float, admin_key_file: str | None, transport: Any = None,
                         timeout_seconds: float = OPENAI_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Organization USD for yesterday and today (UTC) and for the UTC month to date."""

    from .openai_organization_costs import OpenAICostsError, organization_daily_costs

    today = int(now // DAY_SECONDS) * DAY_SECONDS
    yesterday = today - DAY_SECONDS
    month_start = _month_start(now)
    row: dict[str, Any] = {"status": "unknown", "blockers": [], "month": _utc(now).strftime("%Y-%m"),
                           "day_usd": {}, "month_to_date_usd": None}
    key_file = str(admin_key_file or "").strip()
    if not key_file:
        return {**row, "blockers": ["openai_spend_credentials_missing"], "reason": "admin_key_file_unset"}
    try:
        totals = organization_daily_costs(admin_api_key_file=Path(key_file).expanduser(),
                                          start_time=min(month_start, yesterday), end_time=today + DAY_SECONDS,
                                          transport=transport, timeout_seconds=timeout_seconds)
    except OpenAICostsError as exc:
        code = str(exc)
        blocker = ("openai_spend_credentials_missing" if code.startswith("openai_admin_key")
                   else "openai_spend_transport_failed" if code == "openai_cost_request_failed"
                   else "openai_spend_unverifiable")
        return {**row, "blockers": [blocker], "reason": code,
                **({"http_status": exc.http_status} if exc.http_status is not None else {})}
    except Exception:  # noqa: BLE001 - transport errors may carry headers or account details
        return {**row, "blockers": ["openai_spend_transport_failed"]}
    return {**row, "status": "observed",
            "day_usd": {_utc(day).strftime("%Y-%m-%d"): round(totals.get(day, 0.0), 6) for day in (yesterday, today)},
            "month_to_date_usd": round(sum(usd for day, usd in totals.items() if day >= month_start), 6)}


def _attempt_price(state_dir: Path, state: Mapping[str, Any]) -> float | None:
    """The reviewed price the attempt was admitted at, else the one its binding records.

    A pose keeps its admission beside its state, a task one level up, an
    archived pose attempt inside its own directory (``atlas_spend_settlement``).
    """

    candidates = [state_dir / "controller_admission.json"]
    if state_dir.name == "atlas_pose":
        candidates.append(state_dir.parent / "controller_admission.json")
    for path in candidates:
        admission = _small_json(path)
        if (admission is not None and admission.get("provider") == "world_labs_atlas"
                and admission.get("allocation_binding_digest") == state.get("request_digest")):
            price = _number(admission.get("maximum_cost_usd"))
            if price is not None and price > 0:
                return price
    binding = state.get("binding")
    price = _number(binding.get("maximum_cost_usd")) if isinstance(binding, Mapping) else None
    return price if price is not None and price > 0 else None


def _atlas_charge(state_dir: Path, *, since: float) -> dict[str, Any] | None:
    """What one retained attempt bought since ``since``, at its last send; None for nothing.

    Raises ValueError when the attempt cannot be read or priced.
    """

    path = state_dir / "submission.json"
    try:
        info = os.lstat(path)
    except OSError:
        return None
    # A state last written before the window cannot hold a send inside it.
    if not stat.S_ISREG(info.st_mode) or info.st_mtime < since:
        return None
    state = _small_json(path)
    if state is None:
        raise ValueError("world_labs_spend_attempt_unreadable")
    sends = state.get("sends")
    last = sends[-1] if isinstance(sends, list) and sends else None
    sent_at = _number(last.get("at_epoch")) if isinstance(last, Mapping) else None
    if sent_at is None or sent_at < since:
        return None
    # Nothing was bought unless an operation exists or a send may have bought one.
    if not isinstance(state.get("operation_id"), str) and state.get("possibly_bought") is not True:
        return None
    from .atlas_spend_settlement import atlas_attempt_outcome

    outcome = atlas_attempt_outcome(state)
    if outcome is not None and outcome.get("outcome") != "completed":
        return None
    price = _attempt_price(state_dir, state)
    if price is None:
        raise ValueError("world_labs_spend_attempt_unreadable")
    return {"at_epoch": sent_at, "usd": price, "held": outcome is None, "job_id": _atlas_job_id(state_dir),
            "attempt": "atlas:" + hashlib.sha256(str(state_dir).encode()).hexdigest()[:32]}


def _atlas_job_id(state_dir: Path) -> str | None:
    """The scene the attempt belongs to (``site-<request id>`` for a website job), from its path."""

    parts = state_dir.parts
    for index in range(len(parts) - 1, 0, -1):
        if parts[index - 1] == "scenes" and index + 2 < len(parts) and parts[index + 1] == "captures":
            return parts[index]
    return None


def observe_world_labs_spend(*, now: float, storage_root: Path, deadline_seconds: float = SCAN_DEADLINE_SECONDS,
                             max_directories: int = SCAN_MAX_DIRECTORIES,
                             clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Blueprint's recorded Atlas purchases: the last 24 h and the UTC month to date."""

    month_start = _month_start(now)
    window_start = now - DAY_SECONDS
    since = min(month_start, window_start)
    row: dict[str, Any] = {"status": "unknown", "blockers": [], "month": _utc(now).strftime("%Y-%m"),
                           "spend_24h_usd": None, "month_to_date_usd": None, "held_usd": None,
                           "attempts": 0, "complete": False}
    root = Path(storage_root)
    try:
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            return {**row, "blockers": ["world_labs_spend_root_unreadable"]}
        base = root.resolve(strict=True)
        jobs = sorted(path for path in root.glob(ATLAS_JOB_GLOB)
                      if not path.is_symlink() and path.is_dir() and path.resolve().is_relative_to(base))
    except OSError:
        return {**row, "blockers": ["world_labs_spend_root_unreadable"]}
    blockers: set[str] = set()

    def unreadable(_error: OSError) -> None:
        blockers.add("world_labs_spend_scan_incomplete")

    deadline = clock() + deadline_seconds
    visited, cut_short, charges = 0, False, []
    for atlas in jobs:
        for directory, subdirectories, files in os.walk(atlas, onerror=unreadable, followlinks=False):
            subdirectories.sort()
            visited += 1
            if visited > max_directories or clock() > deadline:
                cut_short = True
                break
            here = Path(directory)
            if "submission.json" not in files or not (
                    here.name == "atlas_pose" or here.parent.name == "atlas_pose_attempts"):
                continue
            try:
                charge = _atlas_charge(here, since=since)
            except ValueError as exc:
                blockers.add(str(exc))
                continue
            if charge is not None:
                charges.append(charge)
        if cut_short:
            blockers.add("world_labs_spend_scan_incomplete")
            break
    month = [charge for charge in charges if charge["at_epoch"] >= month_start]
    return {**row, "status": "observed" if not blockers else "partial", "blockers": sorted(blockers),
            "spend_24h_usd": round(sum((c["usd"] for c in charges if c["at_epoch"] >= window_start), 0.0), 6),
            "month_to_date_usd": round(sum((c["usd"] for c in month), 0.0), 6),
            "held_usd": round(sum((c["usd"] for c in month if c["held"]), 0.0), 6),
            "attempts": len(month), "complete": not blockers,
            # Per attempt, for the per-job meter (``paid_spend_meter.atlas_rows``); not in the door summary.
            "charges": month}


def observe_gcp_spend(*, now: float, config_path: str | None) -> dict[str, Any]:
    """The budget guard's month cost and effective limit, read from its own report."""

    from .gcp_monthly_budget_admission import SCHEMA, _money, _period, _read, _time, load_policy

    row: dict[str, Any] = {"status": "unknown", "blockers": [], "project_id": None, "period": _period(_utc(now)),
                           "observed_cost_usd": None, "limit_usd": None}
    configured = str(config_path or "").strip()
    if not configured:
        return {**row, "blockers": ["gcp_budget_guard_not_configured"]}
    try:
        policy = load_policy(Path(configured))
        report = _read(Path(policy["report_path"]))
    except Exception:  # noqa: BLE001 - any unreadable policy or report is one warn
        return {**row, "blockers": ["gcp_budget_report_unreadable"]}
    if report.get("schema_version") != SCHEMA or report.get("project_id") != policy["project_id"]:
        return {**row, "blockers": ["gcp_budget_report_invalid"]}
    row["project_id"] = policy["project_id"]
    if report.get("period") != row["period"]:
        return {**row, "blockers": ["gcp_budget_report_other_month"]}
    try:
        limit = float(_money(report.get("limit_usd")))
        cost = None if report.get("observed_cost_usd") is None else float(_money(report["observed_cost_usd"]))
    except Exception:  # noqa: BLE001 - Decimal and value errors alike
        return {**row, "blockers": ["gcp_budget_report_invalid"]}
    blockers = []
    try:
        age = now - _time(str(report.get("generated_at"))).timestamp()
    except (TypeError, ValueError):
        age = math.inf
    if not 0 <= age <= GCP_REPORT_MAX_AGE_SECONDS:
        blockers.append("gcp_budget_report_stale")
    if cost is None:
        blockers.append("gcp_month_cost_not_reported")
    return {**row, "status": "observed" if cost is not None else "unknown", "blockers": blockers,
            "observed_cost_usd": cost, "limit_usd": limit, "generated_at": str(report.get("generated_at"))}


def _page(code: str, provider: str, spend_usd: float, line: float, window: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "provider": provider, "spend_usd": round(spend_usd, 2), "threshold_usd": line,
            **extra, "window": window}


def _warns(provider: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for blocker in row.get("blockers") or []:
        if provider == "openai":
            warn = {"code": str(blocker), "provider": provider}
            if row.get("reason"):
                warn["reason"] = row["reason"]
            if row.get("http_status") is not None:
                warn["http_status"] = row["http_status"]
        else:
            warn = {"code": f"{provider}_spend_unverifiable", "provider": provider, "reason": str(blocker)}
        rows.append(warn)
    return rows


def spend_alerts(providers: Mapping[str, Mapping[str, Any]], lines: Mapping[str, float]) -> list[dict[str, Any]]:
    """Page rows for spend at or over a line, then a warn row for each blind spot.

    A partial reading is a lower bound, so what it did count still pages.
    """

    alerts: list[dict[str, Any]] = []
    openai = providers.get("openai")
    if isinstance(openai, Mapping):
        days = {day: usd for day, usd in (openai.get("day_usd") or {}).items() if _number(usd) is not None}
        if days:
            day, usd = max(days.items(), key=lambda item: (item[1], item[0]))
            if usd >= lines["openai_24h_usd"]:
                alerts.append(_page("openai_spend_24h_high", "openai", usd, lines["openai_24h_usd"], f"utc_day:{day}"))
        month = _number(openai.get("month_to_date_usd"))
        if month is not None and month >= lines["openai_month_usd"]:
            alerts.append(_page("openai_spend_month_high", "openai", month, lines["openai_month_usd"],
                                f"utc_month:{openai.get('month')}"))
        alerts.extend(_warns("openai", openai))
    world = providers.get("world_labs")
    if isinstance(world, Mapping):
        day_usd, month = _number(world.get("spend_24h_usd")), _number(world.get("month_to_date_usd"))
        if day_usd is not None and day_usd >= lines["world_labs_24h_usd"]:
            alerts.append(_page("world_labs_spend_24h_high", "world_labs", day_usd, lines["world_labs_24h_usd"],
                                "rolling_24h"))
        if month is not None and month >= lines["world_labs_month_usd"]:
            alerts.append(_page("world_labs_spend_month_high", "world_labs", month, lines["world_labs_month_usd"],
                                f"utc_month:{world.get('month')}"))
        alerts.extend(_warns("world_labs", world))
    gcp = providers.get("gcp")
    if isinstance(gcp, Mapping):
        cost, limit = _number(gcp.get("observed_cost_usd")), _number(gcp.get("limit_usd"))
        if cost is not None and limit is not None and limit > 0:
            line = round(limit * lines["gcp_fraction"], 2)
            if cost >= line:
                alerts.append(_page("gcp_spend_month_high", "gcp", cost, line, f"billing_month:{gcp.get('period')}",
                                    limit_usd=limit))
        alerts.extend(_warns("gcp", gcp))
    account = providers.get("account")
    if isinstance(account, Mapping):
        alerts.extend(_account_alerts(account))
    return alerts


def observe_provider_spend(now: float, *, environ: Mapping[str, str] | None = None,
                           openai_transport: Any = None) -> dict[str, Any]:
    """One read of all three providers and the alerts it raises. Reads only; spends nothing."""

    env = os.environ if environ is None else environ
    lines, alerts = lines_from_environment(env)
    providers = {
        "openai": observe_openai_spend(now=now, admin_key_file=env.get(OPENAI_ADMIN_KEY_FILE_ENV),
                                       transport=openai_transport),
        "world_labs": observe_world_labs_spend(
            now=now, storage_root=Path(str(env.get(ATLAS_STORAGE_ROOT_ENV) or DEFAULT_ATLAS_STORAGE_ROOT))),
        "gcp": observe_gcp_spend(now=now, config_path=env.get(GCP_BUDGET_CONFIG_ENV)),
        "account": observe_account_spend(now=now, environ=env),
    }
    alerts.extend(spend_alerts(providers, lines))
    return {"schema_version": SCHEMA_VERSION, "observed_at_epoch": now, "lines": lines, "providers": providers,
            "alerts": alerts, "provider_mutations_performed": 0}


def configured_spend_check() -> Callable[[float], Mapping[str, Any]] | None:
    """The capacity service's check; ``BLUEPRINT_PROVIDER_SPEND_ALERTS=0`` turns it off."""

    if str(os.getenv(DISABLE_ENV) or "").strip().lower() in {"0", "false", "no", "off"}:
        return None

    def check(now: float) -> Mapping[str, Any]:
        observation = observe_provider_spend(now)
        # The spend guard's account report reads these OpenAI and World Labs readings.
        persist_provider_observation(observation)
        return observation

    return check


def summary_projection(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Amounts, lines and reason codes for the door-readable summary: no paths or keys."""

    providers = observation.get("providers")
    lines = observation.get("lines")
    return {"observed_at_epoch": observation.get("observed_at_epoch"),
            "lines": dict(lines) if isinstance(lines, Mapping) else None,
            "providers": {str(name): {key: row[key] for key in _SUMMARY_KEYS if key in row}
                          for name, row in (providers.items() if isinstance(providers, Mapping) else ())
                          if isinstance(row, Mapping)}}


def describe(alert: Mapping[str, Any]) -> str | None:
    """One human line for a spend page, or ``None`` for any other alert."""

    if alert.get("code") not in PAGE_CODES:
        return None
    if alert.get("code") in ACCOUNT_PAGE_CODES:
        return _describe_account(alert)
    provider = str(alert.get("provider") or "")
    spent, line, limit = (_number(alert.get(key)) for key in ("spend_usd", "threshold_usd", "limit_usd"))
    kind, _, value = str(alert.get("window") or "").partition(":")
    when = {"utc_day": f"on {value} UTC", "utc_month": f"in UTC month {value}", "rolling_24h": "in the last 24 h",
            "billing_month": f"in billing month {value}"}.get(kind, kind)
    of_limit = f" of ${limit:.2f}" if limit is not None else ""
    return f"{_LABELS.get(provider, provider)} ${spent or 0:.2f}{of_limit} {when} (line ${line or 0:.2f})"


__all__ = [
    "ATLAS_STORAGE_ROOT_ENV",
    "DISABLE_ENV",
    "GCP_BUDGET_CONFIG_ENV",
    "GCP_PAGE_FRACTION",
    "GCP_PAGE_FRACTION_ENV",
    "OPENAI_24H_PAGE_ENV",
    "OPENAI_24H_PAGE_USD",
    "OPENAI_ADMIN_KEY_FILE_ENV",
    "OPENAI_MONTH_PAGE_ENV",
    "OPENAI_MONTH_PAGE_USD",
    "PAGE_CODES",
    "WARN_CODES",
    "WORLD_LABS_24H_PAGE_ENV",
    "WORLD_LABS_24H_PAGE_USD",
    "WORLD_LABS_MONTH_PAGE_ENV",
    "WORLD_LABS_MONTH_PAGE_USD",
    "configured_spend_check",
    "default_lines",
    "describe",
    "lines_from_environment",
    "observe_gcp_spend",
    "observe_openai_spend",
    "observe_provider_spend",
    "observe_world_labs_spend",
    "spend_alerts",
    "summary_projection",
]
