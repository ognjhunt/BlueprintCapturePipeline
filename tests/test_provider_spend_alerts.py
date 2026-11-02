"""B-08: OpenAI, World Labs and Google Cloud spend pages; blind spots warn; nothing is blocked."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

import pytest

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline import openai_organization_costs as costs
from blueprint_pipeline import provider_spend_alerts as spend
from blueprint_pipeline.task_evaluation_supervisor import openai_cost_authority as authority

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
from operator_door.secrets_guard import scan_bytes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DAY = 86_400
NOW = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc).timestamp()
TODAY = int(NOW // DAY) * DAY
YESTERDAY = TODAY - DAY
MONTH_START = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp())
ADMIN_KEY = "sk-admin-fixture-not-a-provider-key"


def _admin_key(tmp_path: Path, mode: int = 0o600) -> Path:
    path = tmp_path / "openai_admin_api_key"
    path.write_text(ADMIN_KEY + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _page(days: dict[int, float], *, has_more: bool = False, next_page: str | None = None) -> dict:
    return {"object": "page", "has_more": has_more, "next_page": next_page, "data": [
        {"object": "bucket", "start_time": start, "end_time": start + DAY,
         "results": [{"object": "organization.costs.result", "amount": {"value": usd, "currency": "usd"},
                      "line_item": None, "project_id": None}]}
        for start, usd in sorted(days.items())]}


class Transport:
    """The Costs API, stubbed: answers pages in order and records each request."""

    def __init__(self, *pages):
        self.pages, self.calls = list(pages), []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers), timeout))
        page = self.pages.pop(0)
        if isinstance(page, BaseException):
            raise page
        return page


def _month(*, today: float = 0.0, yesterday: float = 0.0, other: float = 1.0) -> dict[int, float]:
    days = {start: other for start in range(MONTH_START, TODAY + DAY, DAY)}
    days.update({YESTERDAY: yesterday, TODAY: today})
    return days


def _observe(tmp_path, transport, *, now=NOW, env=None, key=True):
    environ = {spend.ATLAS_STORAGE_ROOT_ENV: str(tmp_path / "no-handoffs"), **(env or {})}
    if key:
        environ[spend.OPENAI_ADMIN_KEY_FILE_ENV] = str(_admin_key(tmp_path))
    return spend.observe_provider_spend(now, environ=environ, openai_transport=transport)


def _codes(result, provider=None):
    return {row["code"]: row for row in result["alerts"] if provider is None or row.get("provider") == provider}


# --- OpenAI -----------------------------------------------------------------------------------


def test_an_openai_day_over_the_line_pages_with_its_dollars(tmp_path):
    transport = Transport(_page(_month(today=80.0)))
    result = _observe(tmp_path, transport)

    page = _codes(result)["openai_spend_24h_high"]
    assert page == {"code": "openai_spend_24h_high", "provider": "openai", "spend_usd": 80.0,
                    "threshold_usd": spend.OPENAI_24H_PAGE_USD, "window": "utc_day:2026-11-20"}
    assert "openai_spend_month_high" not in _codes(result)
    # One organization-wide read: whole UTC days from the month start, no project or key filter.
    [(url, headers, timeout)] = transport.calls
    assert url.startswith(costs.OPENAI_COSTS_ENDPOINT + "?")
    query = dict(parse_qsl(urlsplit(url).query))
    assert query == {"start_time": str(MONTH_START), "end_time": str(TODAY + DAY), "bucket_width": "1d",
                     "limit": "20"}
    assert headers["Authorization"] == "Bearer " + ADMIN_KEY and timeout <= 10
    assert ADMIN_KEY not in json.dumps(result)
    assert result["providers"]["openai"]["day_usd"] == {"2026-11-19": 0.0, "2026-11-20": 80.0}


def test_late_openai_costs_for_yesterday_still_page_today(tmp_path):
    # Cost data lags: a burst late on the 19th lands in the 19th's bucket after midnight.
    result = _observe(tmp_path, Transport(_page(_month(today=0.0, yesterday=90.5))))
    page = _codes(result)["openai_spend_24h_high"]
    assert (page["spend_usd"], page["window"]) == (90.5, "utc_day:2026-11-19")


def test_openai_month_to_date_over_the_line_pages(tmp_path):
    result = _observe(tmp_path, Transport(_page(_month(today=30.0, yesterday=30.0, other=30.0))))
    page = _codes(result)["openai_spend_month_high"]
    assert page == {"code": "openai_spend_month_high", "provider": "openai", "spend_usd": 600.0,
                    "threshold_usd": spend.OPENAI_MONTH_PAGE_USD, "window": "utc_month:2026-11"}
    assert "openai_spend_24h_high" not in _codes(result)


def test_the_first_of_the_month_reads_yesterday_without_counting_it_in_the_month(tmp_path):
    now = datetime(2026, 12, 1, 5, 0, tzinfo=timezone.utc).timestamp()
    december = int(datetime(2026, 12, 1, tzinfo=timezone.utc).timestamp())
    transport = Transport(_page({december - DAY: 80.0, december: 2.0}))
    result = _observe(tmp_path, transport, now=now)
    query = dict(parse_qsl(urlsplit(transport.calls[0][0]).query))
    assert (query["start_time"], query["end_time"]) == (str(december - DAY), str(december + DAY))
    assert _codes(result)["openai_spend_24h_high"]["window"] == "utc_day:2026-11-30"
    assert result["providers"]["openai"]["month_to_date_usd"] == 2.0


def test_openai_spend_under_both_lines_is_quiet(tmp_path):
    result = _observe(tmp_path, Transport(_page(_month(today=74.99, yesterday=10.0, other=10.0))))
    assert not _codes(result, "openai")
    assert result["providers"]["openai"]["status"] == "observed"


def test_missing_openai_credentials_warn_without_a_request(tmp_path):
    transport = Transport()
    unset = _observe(tmp_path, transport, key=False)
    assert _codes(unset, "openai") == {"openai_spend_credentials_missing": {
        "code": "openai_spend_credentials_missing", "provider": "openai", "reason": "admin_key_file_unset"}}
    absent = _observe(tmp_path, transport, key=False,
                      env={spend.OPENAI_ADMIN_KEY_FILE_ENV: str(tmp_path / "absent")})
    assert _codes(absent)["openai_spend_credentials_missing"]["reason"] == "openai_admin_key_file_missing_or_symlink"
    loose = tmp_path / "loose"
    loose.mkdir()
    open_key = _admin_key(loose, mode=0o644)
    too_open = _observe(tmp_path, transport, key=False, env={spend.OPENAI_ADMIN_KEY_FILE_ENV: str(open_key)})
    assert _codes(too_open)["openai_spend_credentials_missing"]["reason"] == (
        "openai_admin_key_file_permissions_too_open")
    assert transport.calls == []
    assert all(row["code"] not in spend.PAGE_CODES for result in (unset, absent, too_open)
               for row in result["alerts"])


@pytest.mark.parametrize("broken", [
    {"object": "list", "data": [], "has_more": False},
    {"object": "page", "data": "rows", "has_more": False},
    {"object": "page", "data": [{"object": "bucket", "start_time": TODAY, "end_time": TODAY + DAY}],
     "has_more": False},
    _page({TODAY: 1.0}, has_more=True),
    {**_page({TODAY: 1.0}), "has_more": "no"},
])
def test_a_malformed_openai_page_is_a_warn(tmp_path, broken):
    result = _observe(tmp_path, Transport(broken))
    assert set(_codes(result, "openai")) == {"openai_spend_unverifiable"}
    assert result["providers"]["openai"]["status"] == "unknown"


@pytest.mark.parametrize("amount", [{"value": 5, "currency": "eur"}, {"value": float("nan"), "currency": "usd"},
                                    {"value": -1, "currency": "usd"}, {"value": True, "currency": "usd"}, None])
def test_a_malformed_openai_amount_is_a_warn_not_a_low_reading(tmp_path, amount):
    page = _page({TODAY: 500.0})
    page["data"][0]["results"].append({"object": "organization.costs.result", "amount": amount})
    result = _observe(tmp_path, Transport(page))
    assert set(_codes(result, "openai")) == {"openai_spend_unverifiable"}


def test_openai_pagination_is_followed_and_a_repeated_cursor_is_a_warn(tmp_path):
    first = _page({MONTH_START: 1.0}, has_more=True, next_page="cursor-2")
    result = _observe(tmp_path, Transport(first, _page({TODAY: 80.0})))
    assert _codes(result)["openai_spend_24h_high"]["spend_usd"] == 80.0
    looped = _observe(tmp_path, Transport(first, _page({TODAY: 1.0}, has_more=True, next_page="cursor-2")))
    assert _codes(looped, "openai")["openai_spend_unverifiable"]["reason"] == "openai_cost_pagination_invalid"


def test_an_openai_transport_failure_is_a_warn_and_never_records_the_key(tmp_path):
    for failure in (OSError("network down"), RuntimeError("Authorization: Bearer " + ADMIN_KEY)):
        result = _observe(tmp_path, Transport(failure))
        assert set(_codes(result, "openai")) == {"openai_spend_transport_failed"}
        assert ADMIN_KEY not in json.dumps(result) and "network down" not in json.dumps(result)


def test_a_rejected_admin_key_reports_its_http_status(tmp_path, monkeypatch):
    def refuse(request, timeout):
        assert request.full_url.startswith(costs.OPENAI_COSTS_ENDPOINT) and request.get_method() == "GET"
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(costs, "urlopen", refuse)
    result = _observe(tmp_path, None)
    row = _codes(result)["openai_spend_transport_failed"]
    assert (row["http_status"], row["reason"]) == (401, "openai_cost_request_failed")


def test_the_cost_authority_and_the_monitor_share_one_endpoint_key_reader_and_transport(tmp_path):
    assert authority.OPENAI_COSTS_ENDPOINT is costs.OPENAI_COSTS_ENDPOINT
    with pytest.raises(costs.OpenAICostsError, match="openai_cost_endpoint_invalid"):
        costs.get_costs_page("https://api.openai.com.evil.example/v1/organization/costs", {}, 1)
    with pytest.raises(authority.OpenAICostAuthorityError, match="openai_cost_endpoint_invalid"):
        authority.OpenAIOrganizationCostsClient._default_transport("https://example.com/v1/organization/costs", {}, 1)
    client = authority.OpenAIOrganizationCostsClient(project_id="proj", api_key_id="key",
                                                     admin_api_key_file=_admin_key(tmp_path, mode=0o640))
    with pytest.raises(authority.OpenAICostAuthorityError, match="openai_admin_key_file_permissions_too_open"):
        client._admin_key()


# --- Google Cloud (Gemini) --------------------------------------------------------------------


def _gcp(tmp_path, *, cost="80", limit="100", period="2026-11", generated=None, project="blueprint-8c1ca"):
    report = tmp_path / "gcp" / "latest.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    policy = tmp_path / "gcp" / "policy.json"
    policy.write_text(json.dumps({
        "project_id": "blueprint-8c1ca", "project_number": "123456789012",
        "billing_account_id": "ABCDEF-123456-ABCDEF", "budget_id": "0" * 8 + "-0000-0000-0000-" + "0" * 12,
        "limit_usd": 100, "topic": "projects/blueprint-8c1ca/topics/budget",
        "subscription": "projects/blueprint-8c1ca/subscriptions/budget-guard",
        "state_path": str(tmp_path / "gcp" / "state.json"), "report_path": str(report)}))
    generated = generated or datetime.fromtimestamp(NOW - 120, timezone.utc).isoformat()
    report.write_text(json.dumps({
        "schema_version": "blueprint.gcp_monthly_budget_guard.v1", "generated_at": generated,
        "project_id": project, "period": period, "limit_usd": limit, "base_limit_usd": "100",
        "observed_cost_usd": cost, "admission_allowed": True, "blockers": []}))
    return {spend.GCP_BUDGET_CONFIG_ENV: str(policy)}


def test_google_cloud_pages_at_eighty_percent_of_the_month_limit(tmp_path):
    quiet = _observe(tmp_path, Transport(_page({})), env=_gcp(tmp_path, cost="79.99"))
    assert not _codes(quiet, "gcp") and quiet["providers"]["gcp"]["observed_cost_usd"] == 79.99
    result = _observe(tmp_path, Transport(_page({})), env=_gcp(tmp_path, cost="80"))
    assert _codes(result, "gcp") == {"gcp_spend_month_high": {
        "code": "gcp_spend_month_high", "provider": "gcp", "spend_usd": 80.0, "threshold_usd": 80.0,
        "limit_usd": 100.0, "window": "billing_month:2026-11"}}
    # A dated owner exception moves the month's line with the limit; the limit is only read.
    october = _observe(tmp_path, Transport(_page({})), env=_gcp(tmp_path, cost="99", limit="125"))
    assert not _codes(october, "gcp")


@pytest.mark.parametrize("change, reason", [
    (lambda tmp_path: {}, "gcp_budget_guard_not_configured"),
    (lambda tmp_path: {spend.GCP_BUDGET_CONFIG_ENV: str(tmp_path / "absent.json")}, "gcp_budget_report_unreadable"),
    (lambda tmp_path: _gcp(tmp_path, period="2026-10"), "gcp_budget_report_other_month"),
    (lambda tmp_path: _gcp(tmp_path, project="other-project"), "gcp_budget_report_invalid"),
    (lambda tmp_path: _gcp(tmp_path, cost=None), "gcp_month_cost_not_reported"),
])
def test_google_cloud_blind_spots_are_warns(tmp_path, change, reason):
    result = _observe(tmp_path, Transport(_page({})), env=change(tmp_path))
    assert _codes(result, "gcp") == {"gcp_spend_unverifiable": {
        "code": "gcp_spend_unverifiable", "provider": "gcp", "reason": reason}}


def test_a_stale_google_cloud_report_warns_and_still_pages_its_month_cost(tmp_path):
    old = datetime.fromtimestamp(NOW - 3 * 3600, timezone.utc).isoformat()
    result = _observe(tmp_path, Transport(_page({})), env=_gcp(tmp_path, cost="90", generated=old))
    codes = _codes(result, "gcp")
    assert codes["gcp_spend_unverifiable"]["reason"] == "gcp_budget_report_stale"
    assert codes["gcp_spend_month_high"]["spend_usd"] == 90.0


# --- World Labs -------------------------------------------------------------------------------


def _attempt(directory: Path, *, price, sent_at, status="completed", operation_id="op-1", layout="task",
             terminal_error=False, admission=True):
    """One Atlas attempt as the adapters retain it: a task keeps its admission beside
    ``atlas_pose/``, a pose keeps it inside, an archived pose attempt is its own directory."""
    state_dir = directory if layout == "archived" else directory / "atlas_pose"
    state_dir.mkdir(parents=True, exist_ok=True)
    digest = "sha256:" + hashlib.sha256(str(directory).encode()).hexdigest()
    if admission:
        (directory if layout == "task" else state_dir).joinpath("controller_admission.json").write_text(json.dumps({
            "schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
            "provider": "world_labs_atlas", "resource_class": "provider_reconstruction_api",
            "maximum_cost_usd": price, "allocation_binding_digest": digest}))
    sends = ([{"at_epoch": sent_at, "status": 402}] if status == "rejected"
             else [{"at_epoch": sent_at - 600, "status": 402}, {"at_epoch": sent_at, "status": 201}])
    state = {"status": status, "request_digest": digest, "binding": {"maximum_cost_usd": price}, "sends": sends}
    if operation_id:
        state["operation_id"] = operation_id
    if status == "completed":
        state["result_digest"] = "sha256:" + "1" * 64
    if terminal_error:
        state["terminal_operation"] = {"id": operation_id, "done": True, "error": {"code": 13}}
    path = state_dir / "submission.json"
    path.write_text(json.dumps(state))
    os.utime(path, (sent_at + 60, sent_at + 60))


def _job(root: Path, name: str = "a") -> Path:
    return root / "bucket" / "scenes" / f"site-capture-{name}" / "captures" / f"walkthrough-capture-{name}" / (
        "pipeline/atlas")


def _world_labs(tmp_path) -> Path:
    root = tmp_path / "handoffs"
    atlas = _job(root)
    _attempt(atlas / "automation/splat/native", price=5.0, sent_at=NOW - 3600)             # bought today
    _attempt(atlas / "automation/splat/native/retry", price=5.0, sent_at=NOW - 1800,      # failed: free
             status="failed", terminal_error=True)
    _attempt(atlas, price=0.75, sent_at=NOW - 900, status="submitted", layout="pose")      # in flight: held
    _attempt(atlas / "automation/mesh", price=0.5, sent_at=NOW - 600, status="rejected",    # refused: nothing bought
             operation_id=None)
    _attempt(_job(root, "b") / "automation/views", price=2.0, sent_at=NOW - 5 * DAY)         # earlier this month
    _attempt(_job(root, "b") / "atlas_pose_attempts/0", price=1.25, sent_at=NOW - 2 * DAY,  # archived pose attempt
             layout="archived")
    _attempt(_job(root, "b") / "atlas_pose_attempts/1", price=9.0, sent_at=MONTH_START - 3 * DAY,
             layout="archived")                                                            # last month
    return root


def test_world_labs_counts_blueprints_own_attempt_records(tmp_path):
    result = _observe(tmp_path, Transport(_page({})), env={spend.ATLAS_STORAGE_ROOT_ENV: str(_world_labs(tmp_path))})
    row = result["providers"]["world_labs"]
    assert row["status"] == "observed" and row["complete"] is True
    assert (row["spend_24h_usd"], row["month_to_date_usd"], row["held_usd"]) == (5.75, 9.0, 0.75)
    assert row["attempts"] == 4
    assert not _codes(result, "world_labs")


def test_world_labs_spend_over_either_line_pages(tmp_path):
    env = {spend.ATLAS_STORAGE_ROOT_ENV: str(_world_labs(tmp_path)),
           spend.WORLD_LABS_24H_PAGE_ENV: "5.75", spend.WORLD_LABS_MONTH_PAGE_ENV: "7"}
    codes = _codes(_observe(tmp_path, Transport(_page({})), env=env), "world_labs")
    assert codes["world_labs_spend_24h_high"] == {
        "code": "world_labs_spend_24h_high", "provider": "world_labs", "spend_usd": 5.75,
        "threshold_usd": 5.75, "window": "rolling_24h"}
    assert codes["world_labs_spend_month_high"]["window"] == "utc_month:2026-11"


def test_world_labs_prices_an_attempt_from_its_binding_when_the_admission_is_gone(tmp_path):
    root = tmp_path / "handoffs"
    _attempt(_job(root) / "automation/splat/native", price=5.0, sent_at=NOW - 60, admission=False)
    result = _observe(tmp_path, Transport(_page({})), env={spend.ATLAS_STORAGE_ROOT_ENV: str(root)})
    assert result["providers"]["world_labs"]["spend_24h_usd"] == 5.0


def test_world_labs_blind_spots_are_warns(tmp_path):
    missing = _observe(tmp_path, Transport(_page({})), env={spend.ATLAS_STORAGE_ROOT_ENV: str(tmp_path / "absent")})
    assert _codes(missing, "world_labs") == {"world_labs_spend_unverifiable": {
        "code": "world_labs_spend_unverifiable", "provider": "world_labs", "reason": "world_labs_spend_root_unreadable"}}
    root = _world_labs(tmp_path)
    partial = spend.observe_world_labs_spend(now=NOW, storage_root=root, max_directories=3)
    assert partial["complete"] is False and partial["blockers"] == ["world_labs_spend_scan_incomplete"]
    # A partial scan is a lower bound (the walk is sorted: the pose attempt is the second
    # directory), so what it did count can still page.
    assert partial["spend_24h_usd"] == 0.75
    codes = {row["code"] for row in spend.spend_alerts({"world_labs": partial},
                                                       {**spend.default_lines(), "world_labs_24h_usd": 0.5})}
    assert codes == {"world_labs_spend_24h_high", "world_labs_spend_unverifiable"}


def test_world_labs_never_follows_a_symlinked_attempt(tmp_path):
    outside = tmp_path / "outside"
    _attempt(outside, price=50.0, sent_at=NOW - 60, layout="pose")
    root = tmp_path / "handoffs"
    atlas = _job(root)
    atlas.mkdir(parents=True)
    (atlas / "linked").symlink_to(outside, target_is_directory=True)
    row = spend.observe_world_labs_spend(now=NOW, storage_root=root)
    assert (row["spend_24h_usd"], row["attempts"]) == (0.0, 0)


# --- Lines and wiring -------------------------------------------------------------------------


def test_default_lines_follow_the_b03_job_costs():
    # B-03 (docs/launch-readiness/2026-11-cost-model.md): OpenAI per job at its caps is
    # ARTISAN's $25 + $0.91 measured Atlas planner/reviews + the $5 upstream ledger.
    assert spend.OPENAI_JOB_USD == pytest.approx(30.91)
    assert spend.WORLD_LABS_JOB_USD == 25.0
    assert (spend.BETA_JOBS_PER_DAY, spend.BETA_JOBS_PER_MONTH, spend.LINE_MARGIN) == (2, 15, 1.2)
    for line, jobs, job_usd in (
        (spend.OPENAI_24H_PAGE_USD, spend.BETA_JOBS_PER_DAY, spend.OPENAI_JOB_USD),
        (spend.OPENAI_MONTH_PAGE_USD, spend.BETA_JOBS_PER_MONTH, spend.OPENAI_JOB_USD),
        (spend.WORLD_LABS_24H_PAGE_USD, spend.BETA_JOBS_PER_DAY, spend.WORLD_LABS_JOB_USD),
        (spend.WORLD_LABS_MONTH_PAGE_USD, spend.BETA_JOBS_PER_MONTH, spend.WORLD_LABS_JOB_USD),
    ):
        assert 0 <= line - jobs * job_usd * spend.LINE_MARGIN < 5
    assert (spend.OPENAI_24H_PAGE_USD, spend.OPENAI_MONTH_PAGE_USD) == (75, 560)
    assert (spend.WORLD_LABS_24H_PAGE_USD, spend.WORLD_LABS_MONTH_PAGE_USD, spend.GCP_PAGE_FRACTION) == (60, 450, 0.8)


def test_lines_load_from_the_environment_and_a_bad_value_keeps_the_default(tmp_path):
    lowered = _observe(tmp_path, Transport(_page(_month(today=25.0))), env={spend.OPENAI_24H_PAGE_ENV: "20"})
    assert _codes(lowered)["openai_spend_24h_high"]["threshold_usd"] == 20.0
    for bad in ("abc", "0", "-5", "nan", "inf"):
        result = _observe(tmp_path, Transport(_page(_month(today=25.0))), env={spend.OPENAI_24H_PAGE_ENV: bad})
        assert "openai_spend_24h_high" not in _codes(result)
        assert _codes(result)["provider_spend_config_invalid"] == {
            "code": "provider_spend_config_invalid", "reason": spend.OPENAI_24H_PAGE_ENV}
    fraction = _observe(tmp_path, Transport(_page({})),
                        env={**_gcp(tmp_path, cost="51"), spend.GCP_PAGE_FRACTION_ENV: "0.5"})
    assert _codes(fraction)["gcp_spend_month_high"]["threshold_usd"] == 50.0
    widened = _observe(tmp_path, Transport(_page({})),
                       env={**_gcp(tmp_path, cost="81"), spend.GCP_PAGE_FRACTION_ENV: "1.5"})
    assert _codes(widened)["provider_spend_config_invalid"]["reason"] == spend.GCP_PAGE_FRACTION_ENV
    assert _codes(widened)["gcp_spend_month_high"]["threshold_usd"] == 80.0


def test_the_service_check_is_on_unless_disabled(monkeypatch):
    monkeypatch.delenv(spend.DISABLE_ENV, raising=False)
    assert callable(spend.configured_spend_check())
    monkeypatch.setenv(spend.DISABLE_ENV, "0")
    assert spend.configured_spend_check() is None


def _tick(tmp_path, spend_check, *, now=NOW, webhook="https://alerts.example/hook", posted=None):
    return capacity.run_controller(
        mounts=[str(tmp_path)], report_root=tmp_path / "capacity", reservation_root=tmp_path / "ledger",
        webhook_url=webhook, volume=None, ack="", token="", survey=None, now=now,
        disk_usage=lambda _path: SimpleNamespace(total=100 * 1024**3, used=10 * 1024**3, free=90 * 1024**3),
        poster=lambda _url, value: (posted if posted is not None else []).append(value),
        release_retirement_summary_path=tmp_path / "absent-retirement",
        break_glass_notes_root=tmp_path / "absent-notes", storage_gc_summary_path=tmp_path / "absent-gc",
        spend_check=spend_check)


def test_a_spend_page_rides_the_capacity_route_without_changing_the_disk_level(tmp_path, monkeypatch):
    observed = _observe(tmp_path, Transport(_page(_month(today=80.0))))
    posted: list = []
    report = _tick(tmp_path, lambda _now: observed, posted=posted)

    assert report["level"] == "ok" and report["alert_posted"] is True and posted == [report]
    row = next(a for a in report["alerts"] if a["code"] == "openai_spend_24h_high")
    assert row["severity"] == "page"
    payloads = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def urlopen(request, *, timeout):
        payloads.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(capacity.urllib.request, "urlopen", urlopen)
    capacity.post_alert("https://alerts.example/hook", report)
    [payload] = payloads
    assert payload["page"] is True and payload["severity"] == "page"
    assert payload["summary"] == "OpenAI $80.00 on 2026-11-20 UTC (line $75.00)"
    assert payload["text"].startswith("provider spend: OpenAI $80.00 on 2026-11-20 UTC (line $75.00) | ")
    assert "gcp gcp_spend_unverifiable" in payload["text"]
    assert payload["spend_runbook"] == "docs/runbooks/beta-operations.md#2a-provider-funding"
    summary = capacity.capacity_summary(report)
    summary_row = next(a for a in summary["alerts"] if a["code"] == "openai_spend_24h_high")
    assert (summary_row["spend_usd"], summary_row["threshold_usd"], summary_row["window"]) == (
        80.0, 75.0, "utc_day:2026-11-20")
    assert summary["provider_spend"]["providers"]["openai"]["day_usd"]["2026-11-20"] == 80.0
    raw = (tmp_path / "capacity" / "summary.json").read_bytes()
    assert scan_bytes(raw) is None and ADMIN_KEY.encode() not in raw
    # It repeats hourly while it lasts, like every page on this route.
    assert _tick(tmp_path, lambda _now: observed, now=NOW + 600)["alert_posted"] is False
    assert _tick(tmp_path, lambda _now: observed, now=NOW + 3600)["alert_posted"] is True


def test_spend_warns_alone_never_post_or_raise_the_level(tmp_path):
    blind = _observe(tmp_path, Transport(), key=False)
    posted: list = []
    report = _tick(tmp_path, lambda _now: blind, posted=posted)
    assert report["level"] == "ok" and posted == [] and report["alert_posted"] is False
    rows = [a for a in report["alerts"] if a.get("provider") in {"openai", "gcp", "world_labs"}]
    assert rows and {row["severity"] for row in rows} == {"warn"}


def test_a_failing_spend_check_is_a_warn_and_the_tick_finishes(tmp_path):
    def broken(_now):
        raise RuntimeError("Bearer " + ADMIN_KEY)

    report = _tick(tmp_path, broken)
    row = next(a for a in report["alerts"] if a["code"] == "provider_spend_check_failed")
    assert row == {"code": "provider_spend_check_failed", "severity": "warn"}
    assert ADMIN_KEY not in json.dumps(report) and (tmp_path / "capacity" / "latest.json").is_file()


def test_no_launch_or_admission_path_reads_the_spend_alerts():
    # Alerts only observe: the capacity tick is the one reader, and it never admits work.
    readers = set()
    for path in (REPO / "src" / "blueprint_pipeline").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "provider_spend_alerts" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            if any("provider_spend_alerts" in name.split(".") for name in names):
                readers.add(path.stem)
    assert readers == {"control_plane_capacity_controller"}


def test_the_capacity_unit_reads_the_same_admin_key_and_atlas_root_as_their_lanes():
    unit = (REPO / "deploy/systemd/blueprint-control-plane-capacity.service").read_text(encoding="utf-8")
    dispatcher = (REPO / "deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service").read_text(
        encoding="utf-8")
    atlas = (REPO / "deploy/systemd/blueprint-website-atlas-preview.service").read_text(encoding="utf-8")
    key_line = "Environment=OPENAI_ADMIN_API_KEY_FILE=/etc/blueprint/provider-secrets/openai_admin_api_key"
    root_line = "Environment=BLUEPRINT_WEBSITE_ATLAS_PREVIEW_STORAGE_ROOT=/var/lib/blueprint/pubsub-handoffs"
    assert key_line in unit and key_line in dispatcher
    assert root_line in unit and root_line in atlas
    assert "ReadOnlyPaths=/etc/blueprint/provider-secrets" in unit
