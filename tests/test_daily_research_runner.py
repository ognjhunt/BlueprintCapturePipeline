"""Hermetic business research lifecycle tests: no inference, credentials or sinks."""
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.daily_research.runner import (
    AGENT,
    MODEL,
    PROJECT,
    REMOTE_OUTPUT,
    SHEET,
    TEMPLATE,
    Ledger,
    Provider,
    Refusal,
    Runner,
    configuration,
    crm_snapshot,
    due_date,
    main,
    preflight,
    save_json,
    validate_output,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
DAY = "2026-09-30"


def output(day=DAY):
    c = {"organization": "Example Plant", "organization_url": "https://www.plant.example/",
         "site": "North plant, 123 Main St", "location": "Chicago, Illinois, US",
         "task": "Tray ingredient depositing", "potential_robot_match": "Robot depositing hypothesis",
         "qualification_status": "unqualified", "confidence": "low",
         "unknowns": ["Manual bottleneck and orderability unverified"],
         "proposed_next_action": "Review operator workflow evidence",
         "evidence": [{"claim": "Depositing is described", "url": "https://plant.example/tasks",
                       "publisher": "Plant", "source_date": None, "checked_date": day,
                       "classification": "operator", "claim_kind": "fact", "role": role,
                       "quote": "Ingredient depositing workstation"}
                      for role in ("task", "capability", "geography")]}
    return {"checked_date": day, "findings": ["One candidate requires review"],
            "blockers": ["Current orderability unknown"], "proposed_next_actions": ["Check sources"],
            "candidates": [c]}


class NotFound(Exception):
    status_code = 404


class FakeAPI:
    def __init__(self):
        self.agent = {"id": AGENT, "model": MODEL, "reasoning": {"effort": "medium"},
                      "multi_agent": {"enabled": False}, "tools": [{"type": "web_search"}]}
        self.template = {"id": TEMPLATE, "network": {"access": "disabled"},
                         "capability_directories": ["/workspace/capabilities/blueprint"],
                         "skills": [{"name": "deep-research"}, {"name": "blueprint-evidence-qualification"}]}
        self.sessions, self.payloads, self.cancellations = [], [], []
        self.turn_status, self.session_status = "completed", "idle"
        self.raw = (json.dumps(output(), indent=2) + "\n").encode()
        self.lost_create_reply = False
        self.absent, self.read_error, self.artifacts = False, False, True
        self.tool_count = 0
        self.completed_at = None
        self.environment_status = "connected"
        self.environment_missing = False
        self.calls = []

    def get(self, resource, resource_id):
        self.calls.append(("GET", resource, resource_id))
        if self.read_error and resource == "session":
            raise RuntimeError("sensitive upstream exception must not escape")
        if self.absent and resource in {"session", "environment"}:
            raise NotFound()
        if resource == "agent":
            return deepcopy(self.agent)
        if resource == "template":
            return deepcopy(self.template)
        if resource == "environment":
            if self.environment_missing:
                raise NotFound()
            return {"id": "env_1", "status": self.environment_status}
        s = deepcopy(self.sessions[0])
        s.update(status=self.session_status, agent=deepcopy(self.agent), usage=None)
        return s

    def create(self, payload):
        self.calls.append(("POST", "sessions"))
        self.payloads.append(deepcopy(payload))
        self.sessions.append({"id": "sess_1", "environment": {"id": "env_1", "type": "openai_hosted", "container_size": None},
                              "metadata": payload["metadata"]})
        if self.lost_create_reply:
            raise TimeoutError()
        return self.sessions[0]

    def listing(self, resource, session_id=None):
        if resource == "sessions":
            return deepcopy(self.sessions)
        if resource == "turns":
            return [{"id": "turn_1", "subagent_id": None, "status": self.turn_status, "completed_at": self.completed_at}]
        if resource == "items":
            return [{"id": "tool_" + str(i), "turn_id": "turn_1", "type": "web_search_call"} for i in range(self.tool_count)]
        if resource == "artifacts":
            return [{"id": "artifact_1", "turn_id": "turn_1", "path": REMOTE_OUTPUT}] if self.artifacts else []
        raise AssertionError(resource)

    def artifact(self, session_id, artifact_id):
        return self.raw

    def cancel(self, session_id, run_key):
        self.cancellations.append((session_id, run_key))


@pytest.fixture
def fixture(tmp_path):
    snapshot = tmp_path / "crm.json"
    headers = ["Prospect ID", "Organization", "Prospect type", "Site / team", "Contact name",
               "Contact details", "Verification", "Contact source URL", "Robot-team fit", "Task evidence URL",
               "Stage", "Owner", "Next action", "Next action date", "Task / job"]
    save_json(snapshot, {"sheet_id": SHEET, "captured_at": NOW.isoformat(), "complete": True,
                         "values": [["CRM"], [], [], [], headers]})
    cfg = {"enabled": True, "first_date": DAY, "approval_reference": "owner-recurring-soft-1",
           "scheduler_authority_reference": "approved-parent-cutover", "crm_snapshot": str(snapshot),
           "slack_channel_id": None, "max_runtime_seconds": 180, "soft_target_usd": 1}
    api = FakeAPI()
    ledger = Ledger(tmp_path / "state")
    runner = Runner(ledger, cfg, api, clock=lambda: NOW)
    yield runner, api, ledger
    ledger.db.close()


@pytest.mark.parametrize(("instant", "expected"), [
    ("2026-09-30T11:59:59+00:00", None), ("2026-09-30T12:00:00+00:00", DAY),
    ("2026-11-01T12:59:59+00:00", "2026-10-31"), ("2026-11-01T13:00:00+00:00", "2026-11-01"),
    ("2027-03-14T11:59:59+00:00", "2027-03-13"), ("2027-03-14T12:00:00+00:00", "2027-03-14"),
])
def test_dst_exact_latest_due_date(instant, expected):
    assert due_date(datetime.fromisoformat(instant), DAY) == expected


def test_due_requires_timezone():
    with pytest.raises(Refusal, match="timezone"):
        due_date(datetime(2026, 9, 30), DAY)  # noqa: DTZ001 - deliberately reject naive input


def test_saved_resources_one_create_and_exact_bytes(fixture):
    runner, api, ledger = fixture
    result = runner.start_or_resume()
    assert result["state"] == "awaiting_review"
    assert result["usage"] is None and result["budget_is_hard_cap"] is False
    assert result["reported_container_size"] is None
    assert runner.start_or_resume()["session_id"] == result["session_id"]
    assert len(api.payloads) == 1
    assert api.payloads[0]["agent_id"] == AGENT
    assert "agent" not in api.payloads[0]
    assert api.payloads[0]["environment"] == {"type": "openai_hosted", "container_size": "small", "environment_template_id": TEMPLATE}
    assert (ledger.root / (DAY + "-artifact.json")).read_bytes() == api.raw
    assert result["raw_output_digest"] == hashlib.sha256(api.raw).hexdigest()
    assert result["delivery"] == {}  # No broadcast or publication before parent review.


@pytest.mark.parametrize("environment_status", ["expired", "failed", "missing"])
def test_completed_turn_artifact_recovery_survives_environment_loss(fixture, environment_status):
    runner, api, ledger = fixture
    api.turn_status = "in_progress"
    assert runner.start_or_resume()["state"] == "running"
    api.turn_status = "completed"
    api.environment_status = environment_status
    api.environment_missing = environment_status == "missing"
    api.calls.clear()
    result = runner.start_or_resume(allow_create=False)
    assert result["state"] == "awaiting_review"
    assert result["turn_id"] == "turn_1" and result["artifact_downloaded"] is True
    assert (ledger.root / (DAY + "-artifact.json")).read_bytes() == api.raw
    assert not any(call[1] == "environment" for call in api.calls)
    assert len(api.payloads) == 1 and api.cancellations == []


def test_nonterminal_failed_environment_still_requests_cancellation(fixture):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    api.environment_status = "failed"
    result = runner.start_or_resume()
    assert result["state"] == "cancel_pending"
    assert result["error"] == "hosted_environment_failed"
    assert result["turn_id"] == "turn_1"
    assert len(api.cancellations) == 1 and len(api.payloads) == 1


def test_disabled_and_reconcile_never_start(fixture):
    runner, api, _ = fixture
    runner.config["enabled"] = False
    assert runner.start_or_resume(allow_create=False)["state"] == "nothing_to_reconcile"
    with pytest.raises(Refusal, match="disabled"):
        runner.start_or_resume()
    assert api.payloads == []


def test_preflight_get_only_and_configuration_drift(fixture):
    runner, api, _ = fixture
    assert preflight(api)["inference_started"] is False
    assert all(x[0] == "GET" for x in api.calls)
    api.agent["tools"].append({"type": "function"})
    with pytest.raises(Refusal, match="agent_configuration"):
        runner.start_or_resume()
    assert api.payloads == []


def test_lost_create_reply_reconciles_without_resubmission(fixture):
    runner, api, ledger = fixture
    api.lost_create_reply = True
    assert runner.start_or_resume()["state"] == "creation_unresolved"
    reopened = Ledger(ledger.root)
    try:
        resumed = Runner(reopened, runner.config, api, clock=lambda: NOW)
        assert resumed.start_or_resume()["state"] == "awaiting_review"
        assert len(api.payloads) == 1
    finally:
        reopened.db.close()


@pytest.mark.parametrize("matches", [0, 2])
def test_uncertain_create_never_guesses_absence(fixture, matches):
    runner, api, _ = fixture
    api.lost_create_reply = True
    runner.start_or_resume()
    api.sessions = api.sessions * matches
    assert runner.start_or_resume()["state"] == "creation_unresolved"
    assert len(api.payloads) == 1


def test_idle_is_not_success_and_deadline_cancel_once(fixture):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    assert runner.start_or_resume()["state"] == "running"
    runner.clock = lambda: NOW + timedelta(seconds=181)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    runner.start_or_resume()
    assert len(api.cancellations) == 1
    api.turn_status = "cancelled"
    assert runner.start_or_resume()["state"] == "cancelled"


def test_read_failure_still_requests_deadline_cancel(fixture):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    runner.start_or_resume()
    api.read_error = True
    runner.clock = lambda: NOW + timedelta(seconds=181)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.cancellations) == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_visible_and_blocks_next_date(fixture, status):
    runner, api, _ = fixture
    api.turn_status = status
    row = runner.start_or_resume()
    assert row["state"] == status and row["cleanup_required"] is True
    runner.clock = lambda: NOW + timedelta(days=1)
    with pytest.raises(Refusal, match="cleanup_unresolved"):
        runner.start_or_resume()
    assert len(api.payloads) == 1


def test_guard_rejects_excess_tool_calls(fixture):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    api.tool_count = 6
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.cancellations) == 1


def test_artifact_wait_is_bounded(fixture):
    runner, api, _ = fixture
    api.artifacts = False
    assert runner.start_or_resume()["state"] == "collecting"
    for _ in range(4):
        row = runner.start_or_resume()
    assert row["state"] == "failed" and "artifact" in row["error"]
    assert api.cancellations == []


@pytest.mark.parametrize("mutation", ["bad_json", "wrong_date", "qualified", "missing_source", "vendor_fact", "unknown_field"])
def test_invalid_output_never_reaches_review(fixture, mutation):
    runner, api, _ = fixture
    o = output()
    if mutation == "wrong_date":
        o["checked_date"] = "2026-09-29"
    if mutation == "qualified":
        o["candidates"][0]["qualification_status"] = "qualified"
    if mutation == "missing_source":
        o["candidates"][0]["evidence"][0]["url"] = ""
    if mutation == "vendor_fact":
        o["candidates"][0]["evidence"][0]["classification"] = "vendor"
    if mutation == "unknown_field":
        o["candidates"][0]["email_to_send"] = "no"
    api.raw = b"not json" if mutation == "bad_json" else json.dumps(o).encode()
    row = runner.start_or_resume()
    assert row["state"] == "failed"
    assert "packet" not in row and row["cleanup_required"] is True


def test_crm_and_batch_duplicate_detection_preserves_second_site(fixture):
    runner, _, _ = fixture
    snapshot = json.loads(Path(runner.config["crm_snapshot"]).read_text())
    c = output()["candidates"][0]
    snapshot["values"].append(["BP-000001", c["organization"], "Facility / site", c["site"], "", "", "", "", "", c["organization_url"], "", "", "", "", c["task"]])
    save_json(runner.config["crm_snapshot"], snapshot)
    _, known = crm_snapshot(runner.config["crm_snapshot"], NOW)
    o = output()
    second_site = deepcopy(c)
    second_site["site"] = "South plant"
    o["candidates"] = [c, second_site, deepcopy(second_site)]
    accepted, duplicates = validate_output(o, DAY, known)
    assert len(accepted) == 1 and accepted[0]["site"] == "South plant"
    assert len(duplicates) == 2


def test_stale_snapshot_prevents_paid_start(fixture):
    runner, api, _ = fixture
    runner.clock = lambda: NOW + timedelta(hours=27)
    with pytest.raises(Refusal, match="stale"):
        runner.start_or_resume()
    assert api.payloads == []


def decision(row):
    return {"packet_digest": row["packet_digest"], "reviewer_reference": "dot-review-1",
            "source_support_verified": True, "crm_rechecked": True,
            "accepted_keys": [c["candidate_key"] for c in row["packet"]["candidates"]],
            "summary": "One unqualified candidate for source review; no owner escalation needed."}


def test_review_binding_source_support_and_partial_delivery(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    review = decision(row)
    review["source_support_verified"] = False
    with pytest.raises(Refusal, match="review_evidence"):
        runner.review(DAY, review)
    review["source_support_verified"] = True
    row = runner.review(DAY, review)
    assert set(row["delivery"]) == {"sheets", "notion"}
    assert "owner" not in row["delivery"] and "slack" not in row["delivery"]
    for dest, item in list(row["delivery"].items()):
        receipt = {"destination": dest, "key": item["key"], "payload_digest": item["payload_digest"],
                   "readback_verified": True, "reference": "verified:" + dest}
        row = runner.receipt(DAY, receipt)
        assert row["delivery"][dest]["state"] == "acknowledged"
        assert runner.receipt(DAY, receipt) == row
        if dest != "notion":
            assert row["state"] == "reviewed"
    assert row["state"] == "completed"
    wrong = {**receipt, "payload_digest": "wrong"}
    with pytest.raises(Refusal, match="binding"):
        runner.receipt(DAY, wrong)


def test_cleanup_requires_approval_and_verified_absence(fixture):
    runner, api, _ = fixture
    row = runner.start_or_resume()
    receipt = {"session_id": row["session_id"], "environment_id": row["environment_id"]}
    with pytest.raises(Refusal, match="not_admitted"):
        runner.record_cleanup(DAY, receipt)
    receipt["action_time_approval_reference"] = "owner-approved-deletion-after-download"
    with pytest.raises(Refusal, match="still_present"):
        runner.record_cleanup(DAY, receipt)
    api.absent = True
    row = runner.record_cleanup(DAY, receipt)
    assert row["cleanup_required"] is False and row["billing_stop_verified"] is False
    assert not any(x[0] == "DELETE" for x in api.calls)


def _competing_process(root, queue):
    ledger = Ledger(root)
    try:
        with ledger.lock():
            queue.put("acquired")
    except Refusal as exc:
        queue.put(str(exc))
    finally:
        ledger.db.close()


def test_two_processes_cannot_overlap(fixture):
    _, _, ledger = fixture
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    with ledger.lock():
        process = context.Process(target=_competing_process, args=(ledger.root, queue))
        process.start()
        process.join(timeout=5)
        assert process.exitcode == 0
        assert queue.get(timeout=1) == "runner_overlap"


def _sdk_wire_probe():
    import httpx2 as httpx
    import openai

    assert openai.__version__ == "3.22.1", "wire proof requires the deployed SDK pin"
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b'{"artifact":"bytes"}')
        if request.url.path.endswith("/events"):
            return httpx.Response(200)
        if request.method == "POST":
            return httpx.Response(503, json={"error": {"message": "offline failure"}})
        return httpx.Response(200, json={"data": [], "has_more": False})

    p = Provider.__new__(Provider)
    p.client = openai.OpenAI(api_key="offline-placeholder", project=PROJECT, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=False))
    p.api = p.client.beta.agents
    assert p.listing("turns", "sess_1") == []
    assert p.artifact("sess_1", "artifact_1") == b'{"artifact":"bytes"}'
    p.cancel("sess_1", "blueprint-researcher:" + DAY)
    with pytest.raises(openai.APIStatusError):
        p.create({"agent_id": AGENT, "environment": {"type": "openai_hosted", "environment_template_id": TEMPLATE}, "input": "offline", "stream": False})
    assert len([r for r in calls if r.url.path == "/v1/agents/sessions"]) == 1
    assert all(r.headers["OpenAI-Project"] == PROJECT and r.headers["OpenAI-Beta"] == "agents=v1" for r in calls)
    assert all(r.method != "DELETE" for r in calls)
    p.client.close()


def test_sdk_wire_contract_no_retries_redirects_or_paid_calls():
    # The repository and this standalone runner intentionally use different
    # SDKs. Verify the actual runner pin in its isolated CPU interpreter.
    runtime = os.environ.get("BLUEPRINT_RESEARCH_SDK_PYTHON", sys.executable)
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [runtime, "-c", "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')",
         str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
        text=True, timeout=30, check=True,
    )
    assert result.stdout.strip() == "sdk_wire_contract_verified"


@pytest.mark.parametrize("caller_umask", [0o022, 0o027, 0o077])
def test_cli_status_without_provider_or_key(fixture, capsys, caller_umask):
    runner, _, ledger = fixture
    config = ledger.root / "config.json"
    save_json(config, runner.config)
    state = ledger.root / "cli-state"
    previous_umask = os.umask(caller_umask)
    try:
        assert main(["--config", str(config), "--state-dir", str(state), "status"]) == 0
        assert json.loads(capsys.readouterr().out) == []
        assert state.stat().st_mode & 0o777 == 0o700
        assert (state / "ledger.sqlite3").stat().st_mode & 0o777 == 0o600
        assert os.umask(caller_umask) == caller_umask
        later = ledger.root / "later-authority"
        later.mkdir(mode=0o750)
        assert later.stat().st_mode & 0o777 == 0o750 & ~caller_umask
    finally:
        os.umask(previous_umask)


@pytest.mark.parametrize("failure", ["configuration", "ledger_initialization"])
def test_cli_failure_restores_caller_umask(tmp_path, capsys, failure):
    config = tmp_path / "config.json"
    save_json(config, {})
    state = tmp_path / "state"
    if failure == "ledger_initialization":
        state.write_text("not a directory")
    previous_umask = os.umask(0o022)
    try:
        assert main(["--config", str(config), "--state-dir", str(state), "status"]) == 1
        error = "config_invalid" if failure == "configuration" else "local_or_provider_configuration_unavailable"
        assert json.loads(capsys.readouterr().out) == {"state": "blocked", "error": error}
        assert os.umask(0o022) == 0o022
        later = tmp_path / "later-authority"
        later.mkdir(mode=0o750)
        assert later.stat().st_mode & 0o777 == 0o750
    finally:
        os.umask(previous_umask)


def test_cli_database_close_failure_keeps_private_umask_then_restores_caller(
    fixture, monkeypatch, capsys,
):
    from tools.daily_research import runner as cli

    runner, _, ledger = fixture
    config = ledger.root / "config.json"
    save_json(config, runner.config)
    state = ledger.root / "cli-state"

    def failing_close_ledger(root):
        installed = Ledger(root)
        database = installed.db

        class Database:
            def __getattr__(self, name):
                return getattr(database, name)

            def close(self):
                (state / "close-private").touch(mode=0o666)
                database.close()
                raise RuntimeError("close failed")

        installed.db = Database()
        return installed

    monkeypatch.setattr(cli, "Ledger", failing_close_ledger)
    previous_umask = os.umask(0o022)
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            main(["--config", str(config), "--state-dir", str(state), "status"])
        assert json.loads(capsys.readouterr().out) == []
        assert (state / "close-private").stat().st_mode & 0o777 == 0o600
        assert os.umask(0o022) == 0o022
        later = ledger.root / "later-authority"
        later.mkdir(mode=0o750)
        assert later.stat().st_mode & 0o777 == 0o750
    finally:
        os.umask(previous_umask)


def test_review_cannot_approve_different_packet_or_repeat_decision(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    wrong = {**decision(row), "packet_digest": "wrong"}
    with pytest.raises(Refusal, match="binding"):
        runner.review(DAY, wrong)
    review = decision(row)
    runner.review(DAY, review)
    assert runner.review(DAY, review)["state"] == "reviewed"
    with pytest.raises(Refusal, match="already_bound"):
        runner.review(DAY, {**review, "summary": "changed"})


def test_domain_mismatch_does_not_count_as_operator_evidence():
    o = output()
    o["candidates"][0]["organization_url"] = "https://different.example/"
    with pytest.raises(Refusal, match="domain_mismatch"):
        validate_output(o, DAY, set())


def test_terminal_guard_and_crash_recovery_use_provider_timestamps(fixture):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    runner.start_or_resume()
    api.turn_status = "completed"
    api.completed_at = (NOW + timedelta(seconds=181)).timestamp()
    runner.clock = lambda: NOW + timedelta(days=1)
    assert runner.start_or_resume()["state"] == "failed"
    assert len(api.payloads) == 1


def test_snapshot_is_rechecked_after_research_and_zero_candidates_allowed(fixture):
    runner, api, _ = fixture
    o = output()
    o["candidates"] = []
    api.raw = json.dumps(o).encode()
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review" and row["packet"]["candidates"] == []


def test_boot_recovers_running_previous_date_before_any_new_create(fixture):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    runner.start_or_resume()
    runner.clock = lambda: NOW + timedelta(days=1)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.payloads) == 1


def test_enabled_placeholder_cutover_refused(fixture):
    runner, _, _ = fixture
    with pytest.raises(Refusal, match="cutover"):
        configuration({**runner.config, "scheduler_authority_reference": "PENDING-owner-cutover"})


def test_cancellation_retry_is_bounded_and_uses_same_identity(fixture, monkeypatch):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    runner.start_or_resume()

    def lose_reply(session_id, run_key):
        api.cancellations.append((session_id, run_key))
        raise TimeoutError()

    monkeypatch.setattr(api, "cancel", lose_reply)
    runner.clock = lambda: NOW + timedelta(seconds=181)
    row = runner.start_or_resume()
    assert row["state"] == "cancel_pending" and row["cancel_request_acknowledged"] is False
    runner.start_or_resume()
    runner.start_or_resume()
    runner.start_or_resume()
    assert len(api.cancellations) == 3
    assert len(set(api.cancellations)) == 1
    assert len(api.payloads) == 1


def test_cancellation_retries_after_connectivity_recovers(fixture, monkeypatch):
    runner, api, _ = fixture
    api.turn_status = "in_progress"
    runner.start_or_resume()
    original = api.cancel

    def fail_once(session_id, run_key):
        monkeypatch.setattr(api, "cancel", original)
        raise TimeoutError()

    monkeypatch.setattr(api, "cancel", fail_once)
    runner.clock = lambda: NOW + timedelta(seconds=181)
    assert runner.start_or_resume()["cancel_request_acknowledged"] is False
    assert runner.start_or_resume()["cancel_request_acknowledged"] is True
    assert len(api.payloads) == 1


def test_populated_crm_row_without_id_refuses_paid_start(fixture):
    runner, api, _ = fixture
    snapshot = json.loads(Path(runner.config["crm_snapshot"]).read_text())
    snapshot["values"].append(["", "Example Plant", "Facility / site", "North plant", "", "", "", "", "", "https://plant.example/tasks", "", "", "", "", "Tray ingredient depositing"])
    save_json(runner.config["crm_snapshot"], snapshot)
    with pytest.raises(Refusal, match="prospect_id_missing"):
        runner.start_or_resume()
    assert api.payloads == []


def test_completed_artifact_gap_refuses_cleanup(fixture):
    runner, api, _ = fixture
    api.artifacts = False
    for _ in range(5):
        row = runner.start_or_resume()
    assert row["state"] == "failed"
    api.absent = True
    receipt = {"session_id": "sess_1", "environment_id": "env_1", "action_time_approval_reference": "approved"}
    with pytest.raises(Refusal, match="artifact_not_downloaded"):
        runner.record_cleanup(DAY, receipt)


def test_invalid_json_raw_artifact_preserved_before_cleanup(fixture):
    runner, api, ledger = fixture
    api.raw = b"invalid JSON from provider"
    row = runner.start_or_resume()
    assert row["state"] == "failed" and row["artifact_downloaded"] is True
    assert (ledger.root / (DAY + "-artifact.json")).read_bytes() == api.raw
    api.absent = True
    receipt = {"session_id": "sess_1", "environment_id": "env_1", "action_time_approval_reference": "approved"}
    assert runner.record_cleanup(DAY, receipt)["cleanup_required"] is False


def test_completed_artifact_tampering_refuses_cleanup(fixture):
    runner, api, ledger = fixture
    runner.start_or_resume()
    (ledger.root / (DAY + "-artifact.json")).write_bytes(b"tampered")
    api.absent = True
    receipt = {"session_id": "sess_1", "environment_id": "env_1", "action_time_approval_reference": "approved"}
    with pytest.raises(Refusal, match="digest_mismatch"):
        runner.record_cleanup(DAY, receipt)


def test_stop_during_preflight_prevents_create(fixture, monkeypatch):
    runner, api, ledger = fixture
    original = api.get

    def stop_on_template(resource, resource_id):
        result = original(resource, resource_id)
        if resource == "template":
            runner.stop_requested = lambda: True
        return result

    monkeypatch.setattr(api, "get", stop_on_template)
    assert runner.start_or_resume()["state"] == "stopped_before_create"
    assert api.payloads == [] and ledger.rows() == []


def test_stop_after_durable_intent_prevents_create(fixture, monkeypatch):
    runner, api, ledger = fixture
    original = ledger.put

    def stop_after_intent(row):
        original(row)
        if row["state"] == "creating":
            runner.stop_requested = lambda: True

    monkeypatch.setattr(ledger, "put", stop_after_intent)
    row = runner.start_or_resume()
    assert row["state"] == "cancelled" and row["cleanup_required"] is False
    assert api.payloads == []


def test_stop_does_not_overwrite_newer_completed_review(fixture):
    runner, api, ledger = fixture
    api.turn_status = "in_progress"
    stale = runner.start_or_resume()
    api.turn_status = "completed"
    fresh = runner.start_or_resume()
    fresh = runner.review(DAY, decision(fresh))
    assert stale["state"] == "running" and fresh["state"] == "reviewed"
    assert runner.cancel_current(stale["date"], "observer_interrupted") == fresh
    assert ledger.get(DAY) == fresh and api.cancellations == []


def test_units_do_not_install_or_arm_existing_deployment():
    root = Path(__file__).resolve().parents[1]
    timer = (root / "tools/daily_research/systemd/blueprint-researcher-daily.timer").read_text()
    service = (root / "tools/daily_research/systemd/blueprint-researcher-daily.service").read_text()
    assert "07:00:00 America/Chicago" in timer and "RandomizedDelaySec=0" in timer
    assert "Persistent=true" in timer and "OnBootSec=2min" in timer
    assert "tools.daily_research.runner" in service
    assert "blueprint-researcher-daily" not in (root / "scripts/deploy_control_plane_commit.py").read_text()
    assert configuration(json.loads((root / "tools/daily_research/config.example.json").read_text()))["enabled"] is False


if __name__ == "__main__":
    _sdk_wire_probe()
    print("sdk_wire_contract_verified")
