"""One account-wide monthly spend limit, enforced where every paid admission passes.

Owner decision 2026-11-07, "Meter + monthly cap", and the amount set the same
day: USD 200 a UTC calendar month for the whole account. Alerts go at 50 %,
80 % and 100 % ($100, $160, $200). New paid admissions are refused by name,
``account_monthly_spend_limit_reached``, once the month's metered spend plus
held reservations reaches the limit. This is stage 1, which only adds: every
existing per-job budget, grant, reservation, provider ceiling and runaway
guard keeps applying (the GCP project's own USD 100 guard among them).

Configuration
    A reviewed, committed JSON file, ``config/account_monthly_spend_limit.v1.json``,
    which the deploy installs with the promoted source tree
    (``INSTALLED_CONFIG_PATH``). ``BLUEPRINT_ACCOUNT_MONTHLY_SPEND_LIMIT_CONFIG``
    may name another file. It is read the way the GCP budget guard reads its
    policy (``gcp_monthly_budget_admission.load_policy``): a regular file,
    never a symlink, small, validated, its sha256 bound into every report.
    A file that exists but does not validate refuses every new paid admission
    (``account_monthly_spend_limit_config_invalid``): money fails closed.
    No file at that path depends on where the process runs (``limit_required``).
    On the production control plane the limit must exist: every new paid
    admission is refused (``account_monthly_spend_limit_not_configured``), the
    report says ``cap_status: missing`` and the capacity tick pages. Anywhere
    else (tests, development, replays off the host) no file means no limit:
    the view and the alerts still run and say that no cap is set, and nothing
    is refused.

What counts
    For each provider, the larger of its reconciled month-to-date reading and
    what the meter (``paid_spend_meter``) settled with it this month, plus
    every hold the meter has open outside those readings. The readings are the
    ones that already exist:

    - Vast and DigitalOcean: the provider billing export
      (``provider_billing_reconciler``), cumulative since the cohort start,
      less the last retained reconciliation receipt before the month began;
    - Google Cloud: the larger of the monthly budget guard's report (when its
      billing month is this UTC month) and the export's month delta;
    - OpenAI (organization Costs API) and World Labs (Blueprint's Atlas
      purchase records): the capacity tick's provider spend observation
      (``provider_spend_alerts``), persisted for this module.

    A reading that is missing, stale or lower is flagged and never lowers the
    count within the month: the provider keeps what the previous report
    counted (``account_monthly_spend_report.build_report``). It never stops
    work by itself. Held reservations double count spend that an in-flight
    attempt has already billed, which can only make the limit trip earlier.
    A hold counts only in the months its attempt could still be billed in
    (``paid_spend_meter.hold_counts_in_month``); an older one is reported as
    an old unsettled hold and not counted.
    A clear stop (``paid_resource_admission.SPEND_EFFECTS``) is never refused.

Report and freshness
    The GPU spend guard rewrites the report every minute
    (``account_monthly_spend_report.refresh_report``).
    Admission requires a report for this month, bound to this configuration's
    sha256 and limit, written within ``REPORT_MAX_AGE_SECONDS``; anything else
    refuses while a limit is configured.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .launch_proof_policy import production_launch_mode
from .paid_spend_meter import money, month_of, month_start, usd

POLICY_SCHEMA = "blueprint.account_monthly_spend_limit.v1"
REPORT_SCHEMA = "blueprint.account_monthly_spend_report.v1"
OBSERVATION_SCHEMA = "blueprint.account_provider_month_observation.v1"
CONFIG_ENV = "BLUEPRINT_ACCOUNT_MONTHLY_SPEND_LIMIT_CONFIG"
ROOT_ENV = "BLUEPRINT_ACCOUNT_MONTHLY_SPEND_ROOT"
CONFIG_FILENAME = "account_monthly_spend_limit.v1.json"
#: Where the deploy's promoted source tree puts the committed configuration.
INSTALLED_CONFIG_PATH = Path("/opt/blueprint/task-evaluation-control-plane/config") / CONFIG_FILENAME
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / CONFIG_FILENAME
DEFAULT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/account-monthly-spend")
#: The units' ``EnvironmentFile``. Only the host install writes it; running code never creates it.
HOST_ENVIRONMENT_FILE = Path("/etc/blueprint/pipeline-control-plane.env")
REPORT_FILENAME = "report.json"
OBSERVATION_FILENAME = "provider_observation.json"
#: The spend guard rewrites the report every minute; slow runs under load took 5.5 min.
REPORT_MAX_AGE_SECONDS = 15 * 60
#: The capacity tick observes OpenAI and World Labs every ten minutes.
OBSERVATION_MAX_AGE_SECONDS = 30 * 60
#: The spend admission lock refuses billing evidence older than this.
BILLING_MAX_AGE_SECONDS = 24 * 60 * 60
ALERT_PERCENTS = (50, 80, 100)
ACCOUNT_PROVIDERS = ("digitalocean", "gcp", "openai", "vast", "world_labs")
LIMIT_REACHED = "account_monthly_spend_limit_reached"
#: No limit file on the production control plane: new spend is refused until one is installed.
LIMIT_NOT_CONFIGURED = "account_monthly_spend_limit_not_configured"
CONFIG_INVALID = "account_monthly_spend_limit_config_invalid"
REPORT_INVALID = "account_monthly_spend_report_missing_or_invalid"
REPORT_STALE = "account_monthly_spend_report_stale"
UNAVAILABLE = "account_monthly_spend_limit_unavailable"
MAX_POLICY_BYTES = 64 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_EXPORT_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_BASELINE_RECEIPTS = 24
REPORT_MAX_JOBS = 25
_POLICY_KEYS = {"schema_version", "monthly_limit_usd", "monthly_limit_overrides", "report_path", "decision"}
_AMOUNT = re.compile(r"[0-9]{1,6}(\.[0-9]{1,2})?")
_MONTH = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")
_AUDIT_DIRECTORY = re.compile(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z")
BILLING_EXPORT_SCHEMA = "blueprint.provider_billing_export.v1"
BILLING_EXPORT_SCOPE = "blueprint_beta_100_user_cohort"
BILLING_RECEIPT_SCHEMA = "blueprint.provider_billing_source_receipt.v1"
BILLING_EXPORT_ENV = "BLUEPRINT_GPU_BILLING_EXPORT"
BILLING_AUDIT_ROOT_ENV = "BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT"
#: Wall-clock epoch seconds; indirected so tests can pin the clock.
_clock = time.time


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


_month_start = month_start


def _epoch(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


def _regular_bytes(path: Path, limit: int, *, reject_writable: bool = False) -> bytes:
    """A regular file read without following a link; ValueError for anything else."""

    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("file_unsafe")
    if reject_writable and stat.S_IMODE(info.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("file_writable_by_group_or_world")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file_unsafe")
    return data


def _json_object(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("json_not_object")
    return value


def _limit_amount(value: Any) -> Decimal:
    if not isinstance(value, str) or not _AMOUNT.fullmatch(value) or Decimal(value) <= 0:
        raise ValueError("account_monthly_spend_limit_amount_invalid")
    return Decimal(value)


def load_policy(path: Path) -> dict[str, Any]:
    """Validate the owner's limit file; raises ValueError for anything not exactly right."""

    # As the GCP budget guard reads its policy: a regular small file, never a link.
    # Its mode follows the release checkout, so a group-write bit is not refused here.
    data = _regular_bytes(Path(path), MAX_POLICY_BYTES)
    policy = _json_object(data)
    if set(policy) - _POLICY_KEYS or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("account_monthly_spend_policy_schema_invalid")
    _limit_amount(policy.get("monthly_limit_usd"))
    overrides = policy.get("monthly_limit_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("account_monthly_spend_policy_overrides_invalid")
    for period, amount in overrides.items():
        if not _MONTH.fullmatch(str(period)):
            raise ValueError("account_monthly_spend_policy_overrides_invalid")
        _limit_amount(amount)
    report = policy.get("report_path")
    if not isinstance(report, str) or not Path(report).is_absolute() or ".." in Path(report).parts:
        raise ValueError("account_monthly_spend_policy_report_path_invalid")
    decision = policy.get("decision", {})
    if not isinstance(decision, dict) or not all(isinstance(v, str) for v in decision.values()):
        raise ValueError("account_monthly_spend_policy_decision_invalid")
    return {**policy, "policy_sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "config_path": str(path)}


def effective_limit(policy: Mapping[str, Any], period: str) -> Decimal:
    """The month's limit: a dated owner override, else the recurring amount."""

    overrides = policy.get("monthly_limit_overrides") or {}
    return _limit_amount(overrides.get(period, policy["monthly_limit_usd"]))


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = str(env.get(CONFIG_ENV) or "").strip()
    return Path(configured) if configured else INSTALLED_CONFIG_PATH


def configured_policy(environ: Mapping[str, str] | None = None) -> tuple[dict[str, Any] | None, str | None, Path]:
    """``(policy, None)`` when a limit is set, ``(None, None)`` when none is, ``(None, error)`` when unreadable."""

    path = config_path(environ)
    if not path.is_symlink() and not path.exists():
        return None, None, path
    try:
        return load_policy(path), None, path
    except (OSError, ValueError, UnicodeDecodeError, ArithmeticError):
        return None, CONFIG_INVALID, path


def limit_required(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process is the production control plane, where the limit must exist.

    Two signs, either one enough: the environment the production units run
    with (``BLUEPRINT_LAUNCH_PROOF_MODE=production``, read by
    ``launch_proof_policy``), and the host's installed environment file
    (``HOST_ENVIRONMENT_FILE``), which covers a host process started without
    that environment. The state directory is deliberately not a sign: any
    process that runs with the default roots creates it, so a test run as
    root on a development machine would turn that machine into "production".
    Tests, development machines and replays off the host have neither sign,
    so a missing limit file still means "no limit" there.
    """

    env = os.environ if environ is None else environ
    if production_launch_mode(env):
        return True
    try:
        return HOST_ENVIRONMENT_FILE.is_file()
    except OSError:
        return False


def cap_status(policy: Mapping[str, Any] | None, error: str | None, *, required: bool) -> str:
    """``set``, ``config_invalid``, ``missing`` (required here, no file) or ``not_set`` (no file, not required)."""

    if error:
        return "config_invalid"
    if policy is not None:
        return "set"
    return "missing" if required else "not_set"


def account_root(environ: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if environ is None else environ
    configured = str(env.get(ROOT_ENV) or "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_ROOT if DEFAULT_ROOT.parent.is_dir() else None


def report_path(policy: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None) -> Path | None:
    if policy is not None:
        return Path(policy["report_path"])
    root = account_root(environ)
    return root / REPORT_FILENAME if root is not None else None


def _write_public(path: Path, value: Mapping[str, Any]) -> None:
    """Atomic, readable by the paid lanes' service account, writable by its writer only."""

    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("account_monthly_spend_output_unsafe")
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o755)  # nosec B103 - the paid lanes must traverse this report directory; files 0644
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ----------------------------- readings -----------------------------


def persist_provider_observation(observation: Mapping[str, Any], *, environ: Mapping[str, str] | None = None) -> bool:
    """Leave the capacity tick's OpenAI and World Labs readings where the spend guard reads them."""

    try:
        root = account_root(environ)
        providers = observation.get("providers")
        if root is None or not isinstance(providers, Mapping):
            return False
        keep = ("status", "blockers", "month", "month_to_date_usd", "held_usd", "complete", "attempts", "charges")
        value = {"schema_version": OBSERVATION_SCHEMA, "observed_at_epoch": observation.get("observed_at_epoch"),
                 "providers": {name: {key: row[key] for key in keep if key in row}
                               for name, row in providers.items()
                               if name in {"openai", "world_labs"} and isinstance(row, Mapping)}}
        _write_public(root / OBSERVATION_FILENAME, value)
        return True
    except Exception:  # noqa: BLE001 - visibility only; the guard flags a missing observation
        return False


def read_provider_observation(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    root = account_root(environ)
    if root is None:
        return None
    try:
        return _json_object(_regular_bytes(root / OBSERVATION_FILENAME, MAX_REPORT_BYTES))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def current_world_labs(observation: Mapping[str, Any] | None, *, now: float) -> dict[str, Any] | None:
    """The retained World Labs reading, with its Atlas charges, when it is this month's."""

    if not isinstance(observation, Mapping) or observation.get("schema_version") != OBSERVATION_SCHEMA:
        return None
    providers = observation.get("providers") if isinstance(observation.get("providers"), Mapping) else {}
    row = providers.get("world_labs")
    return dict(row) if isinstance(row, Mapping) and row.get("month") == month_of(now) else None


# ----------------------------- the report -----------------------------


def summary(report: Mapping[str, Any], *, max_jobs: int = 10) -> dict[str, Any]:
    """The compact projection the spend guard's own report (``door status``) carries."""

    keys = ("schema_version", "generated_at", "period", "cap_status", "limit_usd", "alert_lines_usd", "spend_usd",
            "held_usd", "counted_usd", "fraction", "alert_threshold_percent", "limit_reached", "admission_allowed",
            "blockers", "complete", "gaps", "report_path", "policy_sha256", "old_unsettled_held_usd",
            "forecast", "reconciliation")
    meter = report.get("meter") if isinstance(report.get("meter"), Mapping) else {}
    return {**{key: report.get(key) for key in keys},
            "providers": {name: {**{key: row.get(key) for key in ("status", "counted_usd", "reconciled_month_usd",
                                                                   "metered_settled_usd", "reason", "observed_at")},
                                 **({"carried_forward": row["carried_forward"]} if row.get("carried_forward") else {})}
                          for name, row in (report.get("providers") or {}).items() if isinstance(row, Mapping)},
            "meter_totals": meter.get("totals"),
            "old_unsettled_holds": list(meter.get("old_unsettled_holds") or [])[:max_jobs],
            "jobs": list(meter.get("jobs") or [])[:max_jobs]}


# ----------------------------- admission -----------------------------


def read_report(path: Path) -> dict[str, Any]:
    return _json_object(_regular_bytes(Path(path), MAX_REPORT_BYTES, reject_writable=True))


def admission_blockers(*, now: float | None = None, environ: Mapping[str, str] | None = None) -> list[str]:
    """``[]`` when the month is under its limit, or no limit is configured where none is required.

    Otherwise the named refusal. With no limit file, the production control
    plane refuses (``limit_required``); any other process has no limit.
    """

    env = os.environ if environ is None else environ
    policy, error, _config = configured_policy(env)
    if policy is None:
        if error:
            return [error]
        return [LIMIT_NOT_CONFIGURED] if limit_required(env) else []
    moment = _clock() if now is None else float(now)
    period = month_of(moment)
    try:
        report = read_report(Path(policy["report_path"]))
        limit = effective_limit(policy, period)
        counted = money(report.get("counted_usd"))
        generated = _epoch(report.get("generated_at"))
        if (report.get("schema_version") != REPORT_SCHEMA or report.get("cap_status") != "set"
                or report.get("policy_sha256") != policy["policy_sha256"] or counted is None or generated is None
                or Decimal(str(report.get("limit_usd"))) != limit):
            return [REPORT_INVALID]
    except (OSError, ValueError, UnicodeDecodeError, ArithmeticError):
        return [REPORT_INVALID]
    if report.get("period") != period or not -60 <= moment - generated <= REPORT_MAX_AGE_SECONDS:
        return [REPORT_STALE]
    return [LIMIT_REACHED] if counted >= limit else []


# ----------------------------- alerts -----------------------------


def observe_account_spend(*, now: float, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The account row the provider spend alerts read: the guard's last report, judged for age."""

    env = os.environ if environ is None else environ
    policy, error, _config = configured_policy(env)
    row: dict[str, Any] = {"status": "unknown", "blockers": [], "period": month_of(now),
                           "cap_status": cap_status(policy, error, required=limit_required(env))}
    path = report_path(policy, env)
    if path is None:
        return {**row, "blockers": ["account_spend_report_unconfigured"]}
    try:
        report = read_report(path)
    except (OSError, ValueError, UnicodeDecodeError):
        return {**row, "blockers": ["account_spend_report_unreadable"]}
    generated = _epoch(report.get("generated_at"))
    blockers = []
    if report.get("schema_version") != REPORT_SCHEMA:
        blockers.append("account_spend_report_invalid")
    if report.get("period") != row["period"] or generated is None or not -60 <= now - generated <= REPORT_MAX_AGE_SECONDS:
        blockers.append("account_spend_report_stale")
    if policy is not None and report.get("policy_sha256") != policy["policy_sha256"]:
        blockers.append("account_spend_report_policy_mismatch")
    keys = ("limit_usd", "alert_lines_usd", "spend_usd", "held_usd", "counted_usd", "fraction",
            "alert_threshold_percent", "limit_reached", "admission_allowed", "complete", "gaps", "generated_at",
            "forecast", "reconciliation")
    return {**row, **{key: report.get(key) for key in keys}, "blockers": blockers,
            "status": "observed" if not blockers else "unknown"}


def account_alerts(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Page and warn rows for the account's month (provider ``account``)."""

    window = f"utc_month:{row.get('period')}"
    base = {"provider": "account", "window": window}
    if row.get("cap_status") == "config_invalid":
        return [{"code": "account_spend_limit_config_invalid", **base}]
    if row.get("cap_status") == "missing":
        # The production control plane has no limit file: whatever the report says, this pages.
        return [{"code": "account_spend_limit_missing", **base}]
    if row.get("blockers"):
        code = "account_spend_report_stale" if row.get("cap_status") == "set" else "account_spend_unverifiable"
        return [{"code": code, **base, "reason": ",".join(str(item) for item in row["blockers"])}]
    counted = money(row.get("counted_usd")) or Decimal(0)
    alerts: list[dict[str, Any]] = []
    if row.get("cap_status") != "set":
        alerts.append({"code": "account_spend_limit_not_set", **base, "spend_usd": usd(counted)})
    else:
        limit = money(row.get("limit_usd"))
        percent = row.get("alert_threshold_percent")
        if limit and percent in ALERT_PERCENTS:
            code = {50: "account_spend_month_50pct", 80: "account_spend_month_80pct",
                    100: "account_spend_month_limit_reached"}[percent]
            alerts.append({"code": code, **base, "spend_usd": usd(counted), "threshold_usd": usd(limit * percent / 100),
                           "limit_usd": usd(limit), "percent": percent})
    forecast = row.get("forecast") if isinstance(row.get("forecast"), Mapping) else {}
    if forecast.get("alert") is True and row.get("cap_status") == "set":
        alerts.append({"code": FORECAST_CODE, **base, "reaches_limit_on": forecast.get("reaches_limit_on"),
                       "daily_rate_usd": forecast.get("daily_rate_usd"), "spend_usd": usd(counted),
                       "limit_usd": row.get("limit_usd")})
    reconciliation = row.get("reconciliation") if isinstance(row.get("reconciliation"), Mapping) else {}
    if reconciliation.get("outside_tolerance"):
        alerts.append({"code": RECONCILIATION_CODE, **base,
                       "reason": ",".join(str(name) for name in reconciliation["outside_tolerance"])})
    if row.get("complete") is not True:
        gaps = [str(item) for item in row.get("gaps") or []]
        alerts.append({"code": "account_spend_unverifiable", **base,
                       "reason": ",".join(gaps[:6]) + (f",+{len(gaps) - 6}" if len(gaps) > 6 else "")})
    return alerts


#: Warn: at this month's measured rate the limit is reached inside the month (``account_spend_forecast``).
FORECAST_CODE = "account_spend_forecast_reaches_limit_this_month"
#: Warn: a provider's settled ledger and its billing delta differ past tolerance (``account_spend_reconciliation``).
RECONCILIATION_CODE = "account_spend_reconciliation_outside_tolerance"
ACCOUNT_PAGE_CODES = frozenset({"account_spend_month_80pct", "account_spend_month_limit_reached",
                                "account_spend_limit_config_invalid", "account_spend_limit_missing",
                                "account_spend_report_stale"})
ACCOUNT_WARN_CODES = frozenset({"account_spend_month_50pct", "account_spend_limit_not_set", "account_spend_unverifiable",
                                FORECAST_CODE, RECONCILIATION_CODE})


def describe(alert: Mapping[str, Any]) -> str | None:
    code = alert.get("code")
    _kind, _, month = str(alert.get("window") or "").partition(":")
    spent = alert.get("spend_usd")
    if code == "account_spend_limit_config_invalid":
        return "Account monthly limit config invalid: every new paid admission is refused"
    if code == "account_spend_limit_missing":
        return "Account monthly limit config missing on the control plane: every new paid admission is refused"
    if code == "account_spend_report_stale":
        return "Account monthly spend report stale or unreadable: new paid admissions refused until the spend guard rewrites it"
    if code in {"account_spend_month_80pct", "account_spend_month_limit_reached"}:
        line = f"Account ${float(spent or 0):.2f} of ${float(alert.get('limit_usd') or 0):.2f} in UTC month {month}"
        if code == "account_spend_month_limit_reached":
            return line + ": monthly limit reached, new paid admissions refused"
        return line + f" ({alert.get('percent')}% line ${float(alert.get('threshold_usd') or 0):.2f})"
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Show the account's month and the admission decision (reads only).")
    parser.add_argument("--full", action="store_true", help="print the whole report, every job included up to its cap")
    args = parser.parse_args(argv)
    policy, error, config = configured_policy()
    path = report_path(policy)
    try:
        report: Any = read_report(path) if path is not None else None
    except (OSError, ValueError, UnicodeDecodeError):
        report = None
    print(json.dumps({"config_path": str(config), "cap_status": cap_status(policy, error, required=limit_required()),
                      "report_path": str(path) if path else None,
        "admission_blockers": admission_blockers(),
        "report": report if args.full or not isinstance(report, Mapping) else summary(report)}, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ACCOUNT_PAGE_CODES", "ACCOUNT_WARN_CODES", "CONFIG_ENV", "CONFIG_INVALID", "INSTALLED_CONFIG_PATH",
    "LIMIT_NOT_CONFIGURED", "LIMIT_REACHED", "POLICY_SCHEMA", "REPORT_INVALID", "REPORT_SCHEMA", "REPORT_STALE",
    "ROOT_ENV", "SHIPPED_CONFIG_PATH", "UNAVAILABLE", "account_alerts", "admission_blockers", "cap_status",
    "configured_policy", "describe", "effective_limit", "limit_required", "load_policy", "observe_account_spend",
    "persist_provider_observation", "read_provider_observation", "read_report", "summary",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
