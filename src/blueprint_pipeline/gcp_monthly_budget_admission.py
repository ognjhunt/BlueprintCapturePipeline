"""Read-only monthly GCP budget admission for every paid resource.

Paid-resource admission runs in the CPU control plane, so it reads only the
policy and the report that ``gcp_monthly_budget_guard`` writes. The guard itself
lists provider VMs and stays outside the control plane's import graph.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo

SCHEMA = "blueprint.gcp_monthly_budget_guard.v1"
CONFIG_ENV = "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG"
REPORT_ENV = "BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT"


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp_timezone_missing")
    return result


def _period(now: datetime) -> str:
    return now.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m")


def _money(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("cost_invalid")
    return result


def effective_limit(policy: Mapping[str, Any], now: datetime) -> Decimal:
    """A dated owner exception never changes the recurring monthly amount."""
    return _money(policy.get("monthly_limit_overrides", {}).get(_period(now), policy["limit_usd"]))


def _read(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > 1_000_000:
        raise ValueError("budget_file_unsafe")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("budget_file_invalid")
    return result


def load_policy(path: Path) -> dict:
    policy = _read(path)
    for key, pattern in {
        "project_id": r"[a-z][a-z0-9-]{4,62}",
        "project_number": r"[0-9]+",
        "billing_account_id": r"[A-F0-9]{6}-[A-F0-9]{6}-[A-F0-9]{6}",
        "budget_id": r"[a-f0-9-]{36}",
    }.items():
        if not re.fullmatch(pattern, str(policy.get(key, ""))):
            raise ValueError("budget_policy_identity_invalid")
    if _money(policy["limit_usd"]) <= 0:
        raise ValueError("budget_policy_limit_invalid")
    overrides = policy.get("monthly_limit_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("budget_policy_overrides_invalid")
    for period, limit in overrides.items():
        if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", period) or _money(limit) <= 0:
            raise ValueError("budget_policy_overrides_invalid")
    for key in ("topic", "subscription"):
        kind = "topics" if key == "topic" else "subscriptions"
        if not re.fullmatch(f"projects/{policy['project_id']}/{kind}/[a-zA-Z][a-zA-Z0-9._~-]+", str(policy.get(key, ""))):
            raise ValueError("budget_policy_pubsub_scope_invalid")
    for key in ("state_path", "report_path"):
        if not Path(policy[key]).is_absolute():
            raise ValueError("budget_policy_path_invalid")
    policy["policy_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return policy


def admission_blockers(*, report_path: str | None = None, project_id: str | None = None,
                       now: datetime | None = None) -> list[str]:
    path = report_path if report_path is not None else os.getenv(REPORT_ENV, "").strip()
    if not path:
        return []
    now = now or datetime.now(timezone.utc)
    try:
        configured = os.getenv(CONFIG_ENV, "").strip()
        if not configured:
            raise ValueError("gcp_monthly_budget_policy_missing")
        policy = load_policy(Path(configured))
        report = _read(Path(path))
        if (report.get("schema_version") != SCHEMA or report.get("period") != _period(now)
                or str(Path(path)) != policy["report_path"]
                or report.get("project_id") != policy["project_id"]
                or _money(report.get("limit_usd")) != effective_limit(policy, now)
                or report.get("policy_sha256") != policy["policy_sha256"]
                or project_id is not None and report.get("project_id") != project_id
                or report.get("admission_allowed") is not True or report.get("blockers")
                or report.get("billing_shutdown_authorized") is not True
                or report.get("cutoff_permission_verified") is not True
                or report.get("billing_enabled") is not True
                or _money(report["observed_cost_usd"]) >= _money(report["limit_usd"])
                or not timedelta(0) <= now - _time(report["generated_at"]) <= timedelta(minutes=5)):
            return ["gcp_monthly_budget_admission_blocked"]
    except Exception:
        return ["gcp_monthly_budget_report_missing_or_invalid"]
    return []


def project_admission_blockers(project_id: str) -> list[str]:
    configured = os.getenv(CONFIG_ENV, "").strip()
    if not configured:
        return []
    try:
        policy = load_policy(Path(configured))
        if policy["project_id"] != project_id:
            return []
        return admission_blockers(report_path=policy["report_path"], project_id=project_id)
    except Exception:
        return ["gcp_monthly_budget_policy_invalid"]
