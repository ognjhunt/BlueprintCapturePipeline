"""Project-scoped monthly admission and optional native billing shutoff.

Runs inside the existing spend-guard timer. Billing notifications are delayed;
this is an automatic stop, never a guarantee of a penny-exact provider invoice.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
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


def _read(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > 1_000_000:
        raise ValueError("budget_file_unsafe")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("budget_file_invalid")
    return result


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError("budget_file_unsafe")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o640)
    os.replace(temporary, path)


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
    for key in ("topic", "subscription"):
        kind = "topics" if key == "topic" else "subscriptions"
        if not re.fullmatch(f"projects/{policy['project_id']}/{kind}/[a-zA-Z][a-zA-Z0-9._~-]+", str(policy.get(key, ""))):
            raise ValueError("budget_policy_pubsub_scope_invalid")
    for key in ("state_path", "report_path"):
        if not Path(policy[key]).is_absolute():
            raise ValueError("budget_policy_path_invalid")
    policy["policy_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return policy


def validate_budget(policy: Mapping[str, Any], budget: Mapping[str, Any]) -> None:
    filters = budget.get("budgetFilter", {})
    amount = budget.get("amount", {}).get("specifiedAmount", {})
    expected_name = f"billingAccounts/{policy['billing_account_id']}/budgets/{policy['budget_id']}"
    allowed_projects = ([f"projects/{policy['project_id']}"], [f"projects/{policy['project_number']}"])
    value = _money(amount.get("units", 0)) + _money(amount.get("nanos", 0)) / Decimal(1_000_000_000)
    if (budget.get("name") != expected_name or filters.get("projects") not in allowed_projects
            or filters.get("calendarPeriod") != "MONTH" or filters.get("services")
            or filters.get("labels") or filters.get("subaccounts") or filters.get("resourceAncestors")
            or amount.get("currencyCode") != "USD" or value != _money(policy["limit_usd"])
            or budget.get("notificationsRule", {}).get("pubsubTopic") != policy["topic"]):
        raise ValueError("budget_live_scope_or_limit_mismatch")
    if set(filters) - {"projects", "calendarPeriod", "services", "labels", "subaccounts",
                       "resourceAncestors", "creditTypes", "creditTypesTreatment"}:
        raise ValueError("budget_live_filter_unknown")


def update_observation(policy: Mapping[str, Any], previous: Mapping[str, Any],
                       messages: list[dict], *, now: datetime) -> dict:
    period = _period(now)
    state = {"schema_version": SCHEMA, "project_id": policy["project_id"], "period": period,
             "limit_usd": str(policy["limit_usd"]), "observed_cost_usd": None,
             "source_observed_at": None}
    if previous:
        if (previous.get("schema_version") != SCHEMA or previous.get("project_id") != policy["project_id"]
                or _money(previous.get("limit_usd")) != _money(policy["limit_usd"])):
            raise ValueError("budget_previous_state_invalid")
        if previous.get("period") == period:
            state.update(previous)
    # A retained invoice/report can bootstrap a refusal, never permission to spend.
    seed = policy.get("over_budget_seed", {})
    if seed.get("period") == period:
        amount = _money(seed["observed_cost_usd"])
        if amount < _money(policy["limit_usd"]) or not re.fullmatch(r"[a-f0-9]{64}", str(seed.get("source_sha256", ""))):
            raise ValueError("budget_seed_must_only_block")
        if state["observed_cost_usd"] is None or amount > _money(state["observed_cost_usd"]):
            state.update(observed_cost_usd=str(amount), source_observed_at=seed["observed_at"])
    for received in messages:
        message = received.get("message", {})
        attributes = message.get("attributes", {})
        if (attributes.get("budgetId") != policy["budget_id"]
                or attributes.get("billingAccountId") != policy["billing_account_id"]
                or attributes.get("schemaVersion") != "1.0"):
            continue
        payload = json.loads(base64.b64decode(message["data"], validate=True))
        published = _time(message["publishTime"])
        interval = _time(payload["costIntervalStart"])
        local_interval = interval.astimezone(ZoneInfo("America/Los_Angeles"))
        if local_interval.strftime("%Y-%m") != period:
            continue
        if (payload.get("currencyCode") != "USD" or payload.get("budgetAmountType") != "SPECIFIED_AMOUNT"
                or _money(payload["budgetAmount"]) != _money(policy["limit_usd"])
                or (local_interval.day, local_interval.hour, local_interval.minute, local_interval.second) != (1, 0, 0, 0)
                or interval > published or published > now + timedelta(seconds=60)):
            raise ValueError("budget_notification_scope_invalid")
        amount = _money(payload["costAmount"])
        old = state["observed_cost_usd"]
        state["observed_cost_usd"] = str(max(amount, _money(old) if old is not None else Decimal(0)))
        if not state["source_observed_at"] or published > _time(state["source_observed_at"]):
            state["source_observed_at"] = published.isoformat()
    return state


class NativeClient:
    """Existing canonical ADC identity; fixed Google hosts and bounded requests."""
    def __init__(self):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.session = AuthorizedSession(credentials, refresh_timeout=10)

    def call(self, method: str, service: str, path: str, body: dict | None = None) -> dict:
        if service not in {"billingbudgets", "cloudbilling", "pubsub"} or not path.startswith("/v1/"):
            raise ValueError("budget_api_scope_invalid")
        response = self.session.request(method, f"https://{service}.googleapis.com{path}",
                                        json=body, timeout=10, allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"budget_api_{service}_http_{response.status_code}")
        return response.json() if response.content else {}

    def inventory(self, project: str) -> dict:
        from blueprint_pipeline.cloud_vm_render_providers import GCPRenderProvider

        return GCPRenderProvider().project_inventory(projects=[project])


def run_guard(policy: Mapping[str, Any], *, client=None, now: datetime | None = None) -> dict:
    lock = Path(policy["state_path"]).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o660)
    try:
        os.fchmod(descriptor, 0o660)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"admission_allowed": False, "native_cutoff_verified": False,
                    "blockers": ["gcp_budget_guard_already_running"]}
        return _run_guard(policy, client=client, now=now)
    finally:
        os.close(descriptor)


def _run_guard(policy: Mapping[str, Any], *, client=None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    report = {"schema_version": SCHEMA, "generated_at": now.isoformat(),
              "project_id": policy["project_id"], "period": _period(now),
              "limit_usd": str(policy["limit_usd"]), "policy_sha256": policy.get("policy_sha256"),
              "admission_allowed": False, "native_cutoff_verified": False, "blockers": [],
              "claim_boundary": "Automatic stop uses delayed Google cost reports; an exact invoice ceiling is not guaranteed."}
    try:
        client = client or NativeClient()
        budget = client.call("GET", "billingbudgets", f"/v1/billingAccounts/{policy['billing_account_id']}/budgets/{policy['budget_id']}")
        validate_budget(policy, budget)
        path = Path(policy["state_path"])
        previous = _read(path) if path.exists() else {}
        received = []
        try:
            received = client.call("POST", "pubsub", f"/v1/{policy['subscription']}:pull",
                                   {"maxMessages": 50, "returnImmediately": True}).get("receivedMessages", [])
        except Exception as exc:
            report["blockers"].append(f"budget_notifications_unavailable:{type(exc).__name__}")
        state = update_observation(policy, previous, received, now=now)
        _write(path, state)
        report.update(state)
        info_path = f"/v1/projects/{policy['project_id']}/billingInfo"
        info = client.call("GET", "cloudbilling", info_path)
        if info.get("projectId") != policy["project_id"]:
            raise ValueError("budget_project_billing_info_mismatch")
        if not isinstance(info.get("billingEnabled"), bool):
            raise ValueError("budget_project_billing_status_unknown")
        if info.get("billingEnabled") is True and info.get("billingAccountName") != f"billingAccounts/{policy['billing_account_id']}":
            raise ValueError("budget_project_billing_account_changed")
        cost = state.get("observed_cost_usd")
        exhausted = cost is not None and _money(cost) >= _money(policy["limit_usd"])
        report["billing_enabled"] = info.get("billingEnabled")
        if info.get("billingEnabled") is True:
            report["inventory"] = client.inventory(policy["project_id"])
            if report["inventory"].get("api_confirmed") is not True:
                report["blockers"].append("gcp_project_inventory_unverified")
        if exhausted:
            report["blockers"].append("gcp_monthly_budget_exhausted")
            if info.get("billingEnabled") is False:
                report["native_cutoff_verified"] = True
            elif policy.get("billing_shutdown_authorized") is True:
                client.call("PUT", "cloudbilling", info_path, {"billingAccountName": ""})
                confirmation = client.call("GET", "cloudbilling", info_path)
                report["billing_enabled"] = confirmation.get("billingEnabled")
                report["native_cutoff_verified"] = (confirmation.get("projectId") == policy["project_id"]
                                                     and confirmation.get("billingEnabled") is False)
                if not report["native_cutoff_verified"]:
                    report["blockers"].append("gcp_billing_cutoff_readback_failed")
            else:
                report["blockers"].append("gcp_billing_shutdown_activation_required")
        else:
            observed = state.get("source_observed_at")
            if cost is None or not observed or not timedelta(0) <= now - _time(observed) <= timedelta(hours=24):
                report["blockers"].append("gcp_monthly_cost_missing_or_stale")
            if info.get("billingEnabled") is not True:
                report["blockers"].append("gcp_billing_not_enabled")
        report["admission_allowed"] = not report["blockers"]
        _write(Path(policy["report_path"]), report)
        ids = [r["ackId"] for r in received if r.get("ackId")]
        if ids:
            client.call("POST", "pubsub", f"/v1/{policy['subscription']}:acknowledge", {"ackIds": ids})
    except Exception as exc:
        report["admission_allowed"] = False
        reason = str(exc) if re.fullmatch(r"[a-z0-9_]{1,128}", str(exc)) else type(exc).__name__
        report["blockers"].append(f"gcp_budget_guard_failed:{reason}")
    _write(Path(policy["report_path"]), report)
    return report


def run_configured_guard() -> dict | None:
    configured = os.getenv(CONFIG_ENV, "").strip()
    if not configured:
        return None
    return run_guard(load_policy(Path(configured)))


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
                or _money(report.get("limit_usd")) != _money(policy["limit_usd"])
                or report.get("policy_sha256") != policy["policy_sha256"]
                or project_id is not None and report.get("project_id") != project_id
                or report.get("admission_allowed") is not True or report.get("blockers")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_guard(load_policy(args.config))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["admission_allowed"] or result["native_cutoff_verified"] else 2)
