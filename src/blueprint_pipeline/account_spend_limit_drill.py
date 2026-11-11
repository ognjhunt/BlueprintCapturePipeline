"""A $0 drill of the account monthly limit on the real admission code (WP-13, G17, O-14).

Usage on the host (reads the installed limit, writes only to a fresh temporary directory)::

    PYTHONPATH=src python -m blueprint_pipeline.account_spend_limit_drill
    PYTHONPATH=src python -m blueprint_pipeline.account_spend_limit_drill --override-file drill_override.json

It takes the owner's limit file (``--config``, else the one the units read,
else the committed one), copies it into a temporary root with the report path
moved there, and walks a synthetic month's spend up through the limit:
0, 25, 49, 50, 79, 80, 99, 100 and 110 % of it. Each step records the spend as
a meter settlement in the temporary meter journal and then runs the real code
the host runs: the spend guard's report (``account_monthly_spend_report.refresh_report``),
the admission check (``account_monthly_spend.admission_blockers``), the paid
chokepoint itself (``paid_resource_admission.require_paid_resource_budget``),
and the capacity tick's alerts (``account_monthly_spend.account_alerts``) with
their page or warn severity (``account_monthly_spend.ACCOUNT_PAGE_CODES``, which the
capacity tick pages on).

It passes when, as the owner decided (E4 c): below 50 % no threshold alert; from
50 % a warning; from 80 % a page; at 100 % a page and the refusal
``account_monthly_spend_limit_reached`` from both the check and the chokepoint;
and nothing refused below 100 %. Every other alert is listed, not judged
(``account_spend_unverifiable`` is expected: the drill has no provider billing).

``--override-file`` names a JSON file ``{"monthly_limit_overrides": {"YYYY-MM": "amount"}}``:
the dated override O-14 describes, applied to the drill's copy only, and only
for the current UTC month (any other month is refused). The installed limit
file is never written. Nothing calls a provider; the drill costs $0. The work
root is refused under ``/var/lib/blueprint``, ``/etc/blueprint`` or
``/mnt/blueprint-work``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import account_monthly_spend as account
from .paid_spend_meter import METER_ROOT_ENV, SCREENING_QUEUE_ROOT_ENV, month_of, record_settlement, usd

STEPS_PERCENT = (0, 25, 49, 50, 79, 80, 99, 100, 110)
THRESHOLD_CODES = {50: "account_spend_month_50pct", 80: "account_spend_month_80pct",
                   100: "account_spend_month_limit_reached"}
FORBIDDEN_ROOTS = (Path("/var/lib/blueprint"), Path("/etc/blueprint"), Path("/mnt/blueprint-work"))
#: Read by the chokepoint from the process environment; cleared for the drill so only the account limit decides.
_CHOKEPOINT_ENV = (account.CONFIG_ENV, account.ROOT_ENV, "BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT",
                   "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG")
DRILL_RESOURCE_CLASS = "vast_provider_adapter"


class DrillRefused(ValueError):
    """The drill cannot run without touching real state or guessing a limit."""


def _expected_threshold(percent: int) -> int | None:
    reached = [line for line in THRESHOLD_CODES if percent >= line]
    return reached[-1] if reached else None


def _source_config(config: Path | None) -> Path:
    if config is not None:
        return config
    configured = account.config_path()
    return configured if configured.exists() else account.SHIPPED_CONFIG_PATH


def _override(path: Path | None, period: str) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = value.get("monthly_limit_overrides") if isinstance(value, Mapping) else None
    if not isinstance(overrides, Mapping) or set(overrides) != {period}:
        raise DrillRefused(f"override_file_must_name_only_this_month:{period}")
    return {period: str(overrides[period])}


@contextmanager
def _chokepoint_environment(env: Mapping[str, str]) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in _CHOKEPOINT_ENV}
    try:
        for name in _CHOKEPOINT_ENV:
            os.environ.pop(name, None)
        os.environ.update({name: env[name] for name in (account.CONFIG_ENV, account.ROOT_ENV)})
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _chokepoint_blockers() -> list[str]:
    from .paid_resource_admission import PaidResourceAdmissionBlocked, require_paid_resource_budget

    try:
        require_paid_resource_budget(resource_class=DRILL_RESOURCE_CLASS)
    except PaidResourceAdmissionBlocked as exc:
        return list(exc.blockers)
    return []


def run_drill(*, work_root: Path, config: Path | None = None, override_file: Path | None = None,
              now: float | None = None) -> dict[str, Any]:
    """Walk the month through the limit on the real code under ``work_root``; returns the drill record."""

    from .account_monthly_spend_report import refresh_report

    root = Path(work_root).resolve()
    if any(root == forbidden or forbidden in root.parents for forbidden in FORBIDDEN_ROOTS):
        raise DrillRefused("drill_work_root_is_host_state")
    moment = time.time() if now is None else float(now)
    period = month_of(moment)
    source = _source_config(config)
    policy = account.load_policy(source)
    overrides = {**(policy.get("monthly_limit_overrides") or {}), **_override(override_file, period)}
    config_copy = root / "config" / account.CONFIG_FILENAME
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    config_copy.write_text(json.dumps({
        "schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": policy["monthly_limit_usd"],
        "monthly_limit_overrides": overrides, "report_path": str(root / "account" / account.REPORT_FILENAME),
        "decision": policy.get("decision") or {}}, indent=2) + "\n", encoding="utf-8")
    env = {account.CONFIG_ENV: str(config_copy), account.ROOT_ENV: str(root / "account"),
           METER_ROOT_ENV: str(root / "meter"), SCREENING_QUEUE_ROOT_ENV: str(root / "screening")}
    (root / "screening").mkdir(parents=True, exist_ok=True)
    limit = account.effective_limit(account.load_policy(config_copy), period)
    account.persist_provider_observation({"observed_at_epoch": moment, "providers": {
        name: {"status": "observed", "blockers": [], "month": period, "month_to_date_usd": 0, "complete": True,
               "charges": []} for name in ("openai", "world_labs")}}, environ=env)
    steps, failures, recorded = [], [], Decimal(0)
    for index, percent in enumerate(STEPS_PERCENT):
        target = limit * percent / 100
        if target > recorded:
            record_settlement(job_id="limit-drill", meter_key=f"limit-drill-step-{index}", settled_usd=target - recorded,
                              basis="limit_drill_synthetic", provider="vast", root=root / "meter", now=moment)
            recorded = target
        report = refresh_report(now=moment, environ=env)
        check = account.admission_blockers(now=moment, environ=env)
        with _chokepoint_environment(env):
            chokepoint = _chokepoint_blockers()
        alerts = account.account_alerts(account.observe_account_spend(now=moment, environ=env))
        rows = [{"code": row["code"], "severity": "page" if row["code"] in account.ACCOUNT_PAGE_CODES else "warn"} for row in alerts]
        line = _expected_threshold(percent)
        thresholds = [row for row in rows if row["code"] in THRESHOLD_CODES.values()]
        expected = [] if line is None else [{"code": THRESHOLD_CODES[line], "severity": "warn" if line == 50 else "page"}]
        refusal = [account.LIMIT_REACHED] if percent >= 100 else []
        step = {"percent": percent, "counted_usd": report.get("counted_usd"), "limit_usd": usd(limit),
                "threshold_alerts": thresholds, "other_alerts": [row for row in rows if row not in thresholds],
                "admission_blockers": check, "chokepoint_blockers": chokepoint,
                "forecast": (report.get("forecast") or {}).get("reads")}
        if thresholds != expected:
            failures.append(f"step_{percent}pct_alerts:{json.dumps(thresholds)}!={json.dumps(expected)}")
        if check != refusal or chokepoint != refusal:
            failures.append(f"step_{percent}pct_refusal:{check}/{chokepoint}!={refusal}")
        steps.append(step)
    return {"schema_version": "blueprint.account_spend_limit_drill.v1", "status": "failed" if failures else "passed",
            "period": period, "config_source": str(source), "limit_usd": usd(limit),
            "dated_override": overrides.get(period), "work_root": str(root), "steps": steps, "failures": failures,
            "provider_calls_performed": 0, "spend_usd": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A $0 drill of the account monthly limit on the real admission code.")
    parser.add_argument("--config", type=Path, help="the owner's limit file (default: the units' own)")
    parser.add_argument("--override-file", type=Path, help='{"monthly_limit_overrides": {"YYYY-MM": "amount"}}')
    parser.add_argument("--work-root", type=Path, help="an empty scratch directory (default: a fresh temporary one)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary work root for inspection")
    args = parser.parse_args(argv)
    work = args.work_root or Path(tempfile.mkdtemp(prefix="account-limit-drill-"))
    try:
        result = run_drill(work_root=work, config=args.config, override_file=args.override_file)
    except (DrillRefused, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)[:300]}, indent=2))
        return 2
    finally:
        if args.work_root is None and not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


__all__ = ["DrillRefused", "STEPS_PERCENT", "run_drill"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
