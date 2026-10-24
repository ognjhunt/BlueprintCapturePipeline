"""Project-scoped monthly admission and optional native billing shutoff.

Runs inside the existing spend-guard timer. Billing notifications are delayed;
this is an automatic stop, never a guarantee of a penny-exact provider invoice.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .gcp_monthly_budget_admission import (
    SCHEMA, CONFIG_ENV, REPORT_ENV as REPORT_ENV, _time, _period, _money, _read,
    effective_limit, load_policy, admission_blockers as admission_blockers,
    stale_report_wait as stale_report_wait, project_admission_blockers as project_admission_blockers,
)


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
             "source_observed_at": None, "provider_observed_at": None}
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
        if not state.get("provider_observed_at") or published > _time(state["provider_observed_at"]):
            state["provider_observed_at"] = published.isoformat()
    return state


class NativeClient:
    """Existing canonical ADC identity; fixed Google hosts and bounded requests."""
    def __init__(self):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.session = AuthorizedSession(credentials, refresh_timeout=10)

    def call(self, method: str, service: str, path: str, body: dict | None = None) -> dict:
        if service not in {"billingbudgets", "cloudbilling", "pubsub", "cloudresourcemanager"} or not path.startswith("/v1/"):
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
    # flock needs no file writes. Group read access lets the service share an
    # operator-owned inode without granting permission to modify that inode.
    descriptor = os.open(lock, os.O_CREAT | os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), 0o640)
    try:
        status = os.fstat(descriptor)
        if status.st_uid == os.geteuid():
            os.fchmod(descriptor, 0o640)
        elif status.st_mode & 0o007:
            raise ValueError("budget_lock_permissions_unsafe")
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
    limit = effective_limit(policy, now)
    report = {"schema_version": SCHEMA, "generated_at": now.isoformat(),
              "project_id": policy["project_id"], "period": _period(now),
              "limit_usd": str(limit), "base_limit_usd": str(policy["limit_usd"]),
              "policy_sha256": policy.get("policy_sha256"),
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
        report["limit_usd"] = str(limit)
        info_path = f"/v1/projects/{policy['project_id']}/billingInfo"
        info = client.call("GET", "cloudbilling", info_path)
        if info.get("projectId") != policy["project_id"]:
            raise ValueError("budget_project_billing_info_mismatch")
        if not isinstance(info.get("billingEnabled"), bool):
            raise ValueError("budget_project_billing_status_unknown")
        if info.get("billingEnabled") is True and info.get("billingAccountName") != f"billingAccounts/{policy['billing_account_id']}":
            raise ValueError("budget_project_billing_account_changed")
        cost = state.get("observed_cost_usd")
        exhausted = cost is not None and _money(cost) >= limit
        report["billing_enabled"] = info.get("billingEnabled")
        report["billing_shutdown_authorized"] = policy.get("billing_shutdown_authorized") is True
        permission = "resourcemanager.projects.deleteBillingAssignment"
        permissions = client.call("POST", "cloudresourcemanager",
            f"/v1/projects/{policy['project_id']}:testIamPermissions", {"permissions": [permission]})
        report["cutoff_permission_verified"] = permission in permissions.get("permissions", [])
        if info.get("billingEnabled") is True:
            report["inventory"] = client.inventory(policy["project_id"])
            if report["inventory"].get("api_confirmed") is not True:
                report["blockers"].append("gcp_project_inventory_unverified")
        if exhausted:
            report["blockers"].append("gcp_monthly_budget_exhausted")
            if info.get("billingEnabled") is False:
                report["native_cutoff_verified"] = True
            elif policy.get("billing_shutdown_authorized") is True and report["cutoff_permission_verified"]:
                client.call("PUT", "cloudbilling", info_path, {"billingAccountName": ""})
                confirmation = client.call("GET", "cloudbilling", info_path)
                report["billing_enabled"] = confirmation.get("billingEnabled")
                report["native_cutoff_verified"] = (confirmation.get("projectId") == policy["project_id"]
                                                     and confirmation.get("billingEnabled") is False)
                if not report["native_cutoff_verified"]:
                    report["blockers"].append("gcp_billing_cutoff_readback_failed")
            else:
                report["blockers"].append("gcp_billing_shutdown_activation_required"
                    if not report["billing_shutdown_authorized"] else "gcp_billing_cutoff_permission_missing")
        else:
            if not report["billing_shutdown_authorized"]:
                report["blockers"].append("gcp_billing_shutdown_activation_required")
            if not report["cutoff_permission_verified"]:
                report["blockers"].append("gcp_billing_cutoff_permission_missing")
            observed = state.get("provider_observed_at")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_guard(load_policy(args.config))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["admission_allowed"] or result["native_cutoff_verified"] else 2)
