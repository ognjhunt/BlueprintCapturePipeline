"""Hermetic business research lifecycle tests: no inference, credentials or sinks."""
import base64
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.daily_research import capabilities
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
    canonical,
    configuration,
    crm_snapshot,
    digest,
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


@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "\ufeff```json\n{}\n```"])
def test_research_json_fence_retains_raw_without_another_agent_turn(fixture, wrapper):
    runner, api, ledger = fixture
    original = api.raw
    api.raw = wrapper.format(original.decode()).encode()
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review"
    assert row["raw_output_digest"] == hashlib.sha256(api.raw).hexdigest()
    assert ledger.read_bytes(DAY + "-artifact.json") == api.raw
    receipt = row["artifact_format_normalization"]
    assert receipt["raw_sha256"] == row["raw_output_digest"]
    assert "single_json_fence" in receipt["transformations"]
    assert len(api.payloads) == 1


class NotFound(Exception):
    status_code = 404


class FakeAPI:
    def __init__(self):
        self.agent = {"id": AGENT, "model": MODEL, "reasoning": {"effort": "medium"},
                      "multi_agent": {"enabled": False}, "tools": [{"type": "web_search"}]}
        self.template = {"id": TEMPLATE, "network": {"access": "disabled"},
                         "capability_directories": ["/workspace/capabilities/blueprint"],
                         "skills": [], "plugins": [], "files": [
                             {"type": "inline", "path": capabilities.ROOT + "/" + name,
                              "size_bytes": size} for name, (size, _) in capabilities.TEMPLATE_FILES.items()]}
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
        self.vault_credentials = None

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
        if "vault_ids" in payload:
            self.sessions[-1]["vault_ids"] = deepcopy(payload["vault_ids"])
        if self.lost_create_reply:
            raise TimeoutError()
        return self.sessions[0]

    def resolve_mcp_vaults(self, connections):
        credentials = self.vault_credentials
        if credentials is None:
            credentials = {"vault_synthetic_" + tool["server_label"]: [{"id": tool["credential_id"],
                "vault_id": "vault_synthetic_" + tool["server_label"], "auth": {"type": "mcp_oauth",
                "mcp_server_url": tool["transport"]["server_url"]}}] for tool in connections}
        def page(values):
            return SimpleNamespace(data=[SimpleNamespace(model_dump=lambda mode, exclude_unset, value=value: deepcopy(value))
                for value in values], has_more=False)
        def vault_list(**query):
            self.calls.append(("GET", "vaults", query))
            return page([{"id": key} for key in credentials])
        def credential_list(vault_id, **query):
            self.calls.append(("GET", "vault_credentials", vault_id))
            return page(credentials[vault_id])
        provider = Provider.__new__(Provider)
        provider.api = SimpleNamespace(vaults=SimpleNamespace(list=vault_list,
            credentials=SimpleNamespace(list=credential_list)))
        return provider.resolve_mcp_vaults(connections)

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


@pytest.mark.parametrize("repeating", [True, False])
def test_vault_metadata_inventory_must_finish_before_attachment(repeating):
    calls = []
    def listing(**query):
        calls.append(query)
        return SimpleNamespace(data=[], has_more=True,
            last_id="vault_repeat" if repeating else "vault_page_" + str(len(calls)))
    with pytest.raises(Refusal, match="provider_pagination_invalid" if repeating else "provider_pagination_limit"):
        Provider._metadata_pages(SimpleNamespace(list=listing), status="active")
    assert len(calls) == (2 if repeating else 10)
    assert all(call["limit"] == 100 and call["status"] == "active" for call in calls)


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
    environment = api.payloads[0]["environment"]
    assert environment["type"] == "openai_hosted" and environment["container_size"] == "small"
    assert environment["environment_template_id"] == TEMPLATE
    assert environment["network"] == {"access": "disabled"}
    assert environment["capability_directories"] == [capabilities.ROOT]
    assert environment["setup_commands"] == capabilities.setup_commands()
    assert len(environment["files"]) == 5
    for item in environment["files"]:
        if item["path"] == "/workspace/inputs/blueprint-research-crm-identities.json":
            raw = base64.b64decode(item["data"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == result["metadata"]["research_crm_digest"]
            assert json.loads(raw) == result["research_crm_context"]
            continue
        name = item["path"].removeprefix(capabilities.ROOT + "/")
        raw = base64.b64decode(item["data"], validate=True)
        assert (len(raw), hashlib.sha256(raw).hexdigest()) == capabilities.FILES[name]
    assert result["create_payload"] == api.payloads[0]
    assert result["preflight"]["skill_binding"]["template_inline_content_verified"] is False
    assert (ledger.root / (DAY + "-artifact.json")).read_bytes() == api.raw
    assert result["raw_output_digest"] == hashlib.sha256(api.raw).hexdigest()
    assert result["delivery"] == {}  # The Blueprint QA agent must review before publication.


@pytest.mark.parametrize("mutation", [None, "changed", "missing", "symlink", "parent_symlink"])
def test_session_setup_checks_actual_reviewed_files(fixture, tmp_path, monkeypatch, mutation):
    runner, api, _ = fixture
    monkeypatch.setattr(capabilities, "ROOT", str(tmp_path / "mounted-capabilities"))
    api.template["capability_directories"] = [capabilities.ROOT]
    for item in api.template["files"]:
        item["path"] = item["path"].replace("/workspace/capabilities/blueprint", capabilities.ROOT)
    row = runner.start_or_resume()
    environment = api.payloads[0]["environment"]
    assert row["create_payload"]["environment"]["setup_commands"] == environment["setup_commands"]
    assert len(environment["setup_commands"]) == 1
    assert set(environment["setup_commands"][0]) == {"command"}
    for item in environment["files"]:
        if not item["path"].startswith(capabilities.ROOT + "/"):
            continue
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(item["data"], validate=True))
    target = Path(capabilities.ROOT) / "blueprint-evidence-qualification/SKILL.md"
    if mutation == "changed":
        raw = target.read_bytes()
        target.write_bytes(b"!" + raw[1:])  # Same size: require the actual SHA check.
    elif mutation == "missing":
        target.unlink()
    elif mutation == "symlink":
        original = tmp_path / "same-reviewed-bytes.md"
        original.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(original)
    elif mutation == "parent_symlink":
        root = Path(capabilities.ROOT)
        original = tmp_path / "same-reviewed-directory"
        root.rename(original)
        root.symlink_to(original, target_is_directory=True)
    result = subprocess.run(["sh", "-c", environment["setup_commands"][0]["command"]], check=False,
                            capture_output=True, text=True)
    assert result.returncode == (0 if mutation is None else 1)
    if mutation is not None:
        assert "reviewed_skill_file_" in result.stderr


def test_normal_history_requirement_fails_before_any_provider_action(fixture):
    runner, api, ledger = fixture
    runner.required_history = True
    with pytest.raises(Refusal, match="research_learning_input_required"):
        runner.start_or_resume()
    assert api.calls == api.payloads == [] and ledger.get(DAY) is None
    # Observation of a legacy ledger does not acquire new learning input.
    ledger.learning_context = lambda day: pytest.fail("legacy recovery must not create learning input")
    assert runner.start_or_resume(allow_create=False)["state"] == "nothing_to_reconcile"


def test_frozen_history_and_crm_identities_reach_agent_before_create_and_survive_recovery(fixture):
    runner, api, ledger = fixture
    original_crm = json.loads(Path(runner.config["crm_snapshot"]).read_bytes())
    original_crm["values"].append(["BP-000001", "Prior Plant", "Facility / site", "South plant", "Private Person",
        "private@example.invalid", "verified", "https://plant.example/contact", "Robot hypothesis",
        "https://plant.example/tasks", "contacted", "Internal Owner", "Follow up", "", "Pallet moving"])
    save_json(runner.config["crm_snapshot"], original_crm)
    raw = json.dumps({"date": DAY, "paidAnalysisCalls": 0, "sendsAuthorized": False,
        "overview": "Dated overview with counterevidence", "history": "Exact prior conversation " + "é" * 5000})
    learning = {"version": "blueprint.research-learning-input.v1", "date": DAY, "paidAnalysisCalls": 0,
        "sendsAuthorized": False, "content_json": raw, "inputHash": hashlib.sha256(raw.encode()).hexdigest(),
        "bindingHash": "a" * 64}
    reads = []
    ledger.learning_context = lambda day: reads.append(day) or deepcopy(learning)
    runner.required_history = True
    row = runner.start_or_resume()
    assert reads == [DAY] and len(api.payloads) == 1
    payload = api.payloads[0]
    inline = {entry["path"]: base64.b64decode(entry["data"], validate=True) for entry in payload["environment"]["files"]}
    assert inline["/workspace/inputs/blueprint-research-learning.json"] == raw.encode()
    crm = json.loads(inline["/workspace/inputs/blueprint-research-crm-identities.json"])
    assert crm["identities"][0]["organization"] == "Prior Plant"
    assert "private@example.invalid" not in payload["input"] and "private@example.invalid" not in str(inline)
    assert "No CRM is supplied" not in payload["input"]
    assert learning["inputHash"] in payload["input"]
    assert row["learning_context"] == learning
    assert row["metadata"]["learning_input_digest"] == learning["inputHash"]
    ledger.learning_context = lambda day: pytest.fail("recovery must use frozen prior bytes")
    resumed = runner.start_or_resume(allow_create=False)
    assert resumed["create_payload"] == payload and resumed["learning_context"] == learning
    assert len(api.payloads) == 1
    assert (ledger.root / (DAY + "-artifact.json")).read_bytes() == api.raw
    assert row["raw_output_digest"] == hashlib.sha256(api.raw).hexdigest()
    assert row["delivery"] == {}


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


@pytest.mark.parametrize("mutation", ["missing", "size", "duplicate", "extra", "file_id", "skill", "plugin", "path",
                                      "missing_skills", "null_skills", "missing_plugins", "null_plugins"])
def test_file_discovery_drift_refuses_before_intent_or_create(fixture, mutation):
    runner, api, ledger = fixture
    files = api.template["files"]
    if mutation == "missing":
        files.pop()
    elif mutation == "size":
        files[0]["size_bytes"] += 1
    elif mutation == "duplicate":
        files[0] = deepcopy(files[1])
    elif mutation == "extra":
        files.append({"type": "inline", "path": capabilities.ROOT + "/unreviewed.py", "size_bytes": 1})
    elif mutation == "file_id":
        files[0].update(type="file_id", file_id="file_other")
    elif mutation == "skill":
        api.template["skills"] = [{"name": "deep-research"}]
    elif mutation == "plugin":
        api.template["plugins"] = [{"name": "other"}]
    elif mutation.startswith("missing_"):
        del api.template[mutation.removeprefix("missing_")]
    elif mutation.startswith("null_"):
        api.template[mutation.removeprefix("null_")] = None
    else:
        files[0]["path"] = capabilities.ROOT + "/../other/SKILL.md"
    with pytest.raises(Refusal, match="template_skill_"):
        runner.start_or_resume()
    assert ledger.rows() == [] and api.payloads == []


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_local_skill_bytes_drift_refuses_before_create(fixture, tmp_path, monkeypatch, mutation):
    runner, api, ledger = fixture
    import shutil
    shutil.copytree(Path(capabilities.__file__).with_name("capabilities"), tmp_path / "capabilities")
    target = tmp_path / "capabilities/deep-research/SKILL.md"
    if mutation == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes().replace(b"Deep research", b"Evil research"))
    monkeypatch.setattr(capabilities, "__file__", str(tmp_path / "capabilities.py"))
    with pytest.raises(Refusal, match="reviewed_skill_file_"):
        runner.start_or_resume()
    assert ledger.rows() == [] and api.payloads == []


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
    snapshot["values"].append(["BP-000001", c["organization"], "Facility / site", c["site"], "", "", "", "", "", c["organization_url"], "", "", "", "", c["task"], "", "", c["location"]])
    save_json(runner.config["crm_snapshot"], snapshot)
    _, known = crm_snapshot(runner.config["crm_snapshot"], NOW)
    o = output()
    second_site = deepcopy(c)
    second_site["site"] = "South plant"
    second_site["location"] = "South plant physical location"
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
    from tests.daily_research_verification_fixture import assessment
    from tools.daily_research import verification
    return {"packet_digest": row["packet_digest"], "reviewer_reference": "dot-review-1",
            "source_support_verified": True, "crm_rechecked": True,
            "accepted_keys": [c["candidate_key"] for c in row["packet"]["candidates"]],
            "summary": "One unqualified candidate for source review; no owner escalation needed.",
            "lead_verification": verification.cohort(row["packet"]["candidates"],
                {c["candidate_key"]: assessment(c, NOW) for c in row["packet"]["candidates"]}, NOW)}


def test_review_boolean_cannot_promote_without_lead_verification(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    value = decision(row)
    value.pop("lead_verification")
    with pytest.raises(Refusal, match="lead_verification_required_before_promotion"):
        runner.review(DAY, value)


def test_review_rejects_forged_derived_verification_metrics(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    value = decision(row)
    value["lead_verification"]["verified_unique_site_task_candidates"] = 999
    with pytest.raises(Refusal, match="lead_verification_result_binding_invalid"):
        runner.review(DAY, value)


def test_review_derives_the_result_version_from_the_packet_pin(fixture):
    """Independent review S6: a decision cannot choose its own evaluator version."""
    from tests.daily_research_verification_fixture import assessment
    from tools.daily_research import verification
    runner, _, _ = fixture
    row = runner.start_or_resume()
    assert row["packet"]["lead_verification_result_version"] == verification.DIAGNOSTIC_RESULT_VERSION
    value = decision(row)
    value["lead_verification"] = verification.cohort(row["packet"]["candidates"],
        {c["candidate_key"]: assessment(c, NOW) for c in row["packet"]["candidates"]}, NOW,
        result_version=verification.RESULT_VERSION)
    with pytest.raises(Refusal, match="lead_verification_result_version_mismatch"):
        runner.review(DAY, value)


def test_review_accepts_portable_whole_number_metrics_after_bridge_roundtrip(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    value = decision(row)
    # JSON.parse/stringify in the real Node bridge changes 1.0 to 1.
    assert value["lead_verification"]["verification_coverage"] == 1.0
    value["lead_verification"]["verification_coverage"] = 1
    reviewed = runner.review(DAY, value)
    assert reviewed["review"] == value


def test_unresolved_review_replay_is_idempotent_and_retains_raw_discovery(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    value = decision(row)
    value.pop("lead_verification")
    value.update(accepted_keys=[], source_support_verified=False)
    reviewed = runner.review(DAY, value)
    assert reviewed["review"]["lead_verification"]["unresolved_count"] == len(row["packet"]["candidates"])
    assert runner.review(DAY, value) == reviewed
    assert reviewed["packet"]["candidates"]


def test_exact_dedupe_retains_every_candidate_and_distinct_physical_sites():
    from tools.daily_research import verification
    first = output()["candidates"][0]
    same_site_alias = {**deepcopy(first), "site": first["site"].upper() + "!"}
    second_site = {**deepcopy(first), "location": "Another physical location"}
    other_operator = {**deepcopy(first), "organization": "Another named operator at the same site"}
    o = output()
    o["candidates"] = [first, same_site_alias, second_site]
    accepted, duplicates = validate_output(o, DAY, set())
    assert len(accepted) == 2 and len(duplicates) == 1
    assert duplicates[0]["evidence"] == same_site_alias["evidence"]
    all_candidates = verification.packet_candidates({"candidates": accepted, "duplicates": duplicates,
                                                    "verification_cohort_version": verification.VERSION})
    assert len(all_candidates) == 3
    assert len({c["candidate_key"] for c in all_candidates}) == 3
    o["candidates"] = [first, other_operator]
    accepted, duplicates = validate_output(o, DAY, set())
    assert len(accepted) == 2 and not duplicates


def test_review_binding_source_support_and_partial_delivery(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    review = decision(row)
    review["source_support_verified"] = False
    with pytest.raises(Refusal, match="lead_verification_required"):
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
        p.create({"agent_id": AGENT, "environment": {"type": "openai_hosted", "environment_template_id": TEMPLATE,
                  "network": {"access": "disabled"}, "capability_directories": [capabilities.ROOT],
                  "files": capabilities.inline_files()}, "input": "offline", "stream": False})
    assert len([r for r in calls if r.url.path == "/v1/agents/sessions"]) == 1
    request = next(r for r in calls if r.url.path == "/v1/agents/sessions")
    sent = json.loads(request.content)
    assert sent["environment"]["files"] == capabilities.inline_files()
    assert sent["environment"]["network"] == {"access": "disabled"}
    assert all(r.headers["OpenAI-Project"] == PROJECT and r.headers["OpenAI-Beta"] == "agents=v1" for r in calls)
    assert all(r.method != "DELETE" for r in calls)
    p.client.close()


def _terminal_provider_probe():
    import importlib.util

    import httpx2 as httpx
    import openai
    assert openai.__version__ == "3.22.1", "wire proof requires the deployed SDK pin"
    path = Path(__file__).resolve().parents[1] / "tools/daily_research/operators/research-perplexity-canary.py"
    spec = importlib.util.spec_from_file_location("terminal_canary_probe", path)
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)
    monkeypatch = pytest.MonkeyPatch()
    requests = []
    def send(request):
        requests.append(request.method)
        return httpx.Response(200, json={})
    monkeypatch.setattr(openai, "DefaultHttpxClient", lambda **options:
        httpx.Client(transport=httpx.MockTransport(send), **options))
    api = canary.TerminalCollectionProvider(None, "offline-fake-key")
    try:
        api.client._client.get("https://api.openai.com/v1/agents")
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with pytest.raises(Refusal, match="terminal_qa_provider_mutation_forbidden"):
                api.client._client.request(method, "https://api.openai.com/v1/agents")
        for method in ("create", "cancel", "qa_input", "qa_retry_input", "repair_input",
                       "tool_result", "application_tool", "tool_admit"):
            with pytest.raises(Refusal, match="terminal_qa_provider_mutation_forbidden"):
                getattr(api, method)({})
        assert requests == ["GET"]
    finally:
        api.client.close()
        monkeypatch.undo()


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


def test_delegated_operator_domain_requires_semantic_agent_qa():
    o = output()
    o["candidates"][0]["organization_url"] = "https://different.example/"
    accepted, _ = validate_output(o, DAY, set())
    assert accepted[0]["operator_affiliation_qa_required"] is True
    assert accepted[0]["qualification_status"] == "unqualified"


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


def test_exact_dedupe_preserves_same_city_distinct_named_facilities():
    first = output()["candidates"][0]
    first = {**first, "site": "North plant, 1 Test Street", "location": "Chicago, Illinois, US"}
    second = {**deepcopy(first), "site": "South plant, 2 Test Street"}
    o = output()
    o["candidates"] = [first, second]
    accepted, duplicates = validate_output(o, DAY, set())
    assert len(accepted) == 2 and not duplicates
    assert len({c["candidate_key"] for c in accepted}) == 2


# --- Site universe slice (control.site_universe) --------------------------------------------------


class UniverseLedger(Ledger):
    """The disk ledger plus the two company-control reads that FirestoreLedger exposes."""

    def __init__(self, root, control=None, objects=None):
        super().__init__(root)
        self.universe, self.objects, self.universe_reads, self.intents = control, dict(objects or {}), [], []

    def site_universe_control(self):
        self.universe_reads.append("control")
        if isinstance(self.universe, Exception):
            raise self.universe
        return deepcopy(self.universe)

    def site_universe_object(self, pin):
        self.universe_reads.append("object")
        value = self.objects.get((pin["sha256"], pin["generation"]))
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise Refusal("site_universe_object_missing")
        return value

    def put(self, row):
        if row.get("state") == "creating":
            self.intents.append(len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()))
        super().put(row)


@pytest.fixture
def universe(tmp_path):
    """Search-profile creates, each in its own directory, with or without the control reads."""
    from tests.test_daily_research_search import fixture as search_fixture
    opened = []

    def run(name, *, control=..., objects=None, history=(), create=True):
        root = tmp_path / name
        root.mkdir()
        generator = search_fixture.__wrapped__(root)
        runner, api, ledger = next(generator)
        opened.append(generator)
        if control is not ...:
            ledger = runner.ledger = UniverseLedger(root / "universe", control, objects)
            opened.append(ledger)
        for row in history:
            ledger.put(row)
        if create:
            runner.start_or_resume()
        return runner, api, ledger

    yield run
    for value in reversed(opened):
        if isinstance(value, Ledger):
            value.db.close()
        else:
            next(value, None)


def universe_pin(rows=None, **changes):
    from tests.test_daily_research_site_universe import build_export, pin_for, site
    raw = build_export(rows or [site(number) for number in range(1, 10)])
    value = pin_for(raw, **{"slice_size": 6, **changes})
    return value, {(value["sha256"], value["generation"]): raw}


def without_payload_digest(payload):
    value = deepcopy(payload)
    value["metadata"].pop("payload_digest")
    return value


@pytest.mark.parametrize("control", [None, {"enabled": False},
                                     {"enabled": False, "sha256": "stale", "slice_size": 999, "uri": "anything"}])
def test_site_universe_flag_off_keeps_payload_metadata_input_and_row_byte_identical(universe, control):
    _, plain_api, plain = universe("plain")
    _, api, ledger = universe("flag-off", control=control)
    assert canonical(api.payloads) == canonical(plain_api.payloads) and len(api.payloads) == 1
    assert canonical(ledger.get(DAY)) == canonical(plain.get(DAY))
    assert ledger.universe_reads == ["control"]  # Nothing else is read.
    payload = api.payloads[0]
    assert "site_universe_slice_digest" not in payload["metadata"] and "site_universe" not in ledger.get(DAY)
    assert [f["path"] for f in payload["environment"]["files"]] == [f["path"] for f in plain_api.payloads[0]["environment"]["files"]]
    assert "site-universe" not in payload["input"] and "site_universe" not in payload["input"]


def test_site_universe_flag_on_freezes_one_slice_and_recovery_never_reads_it_again(universe):
    from tools.daily_research import site_universe
    _, plain_api, _ = universe("plain")
    value, objects = universe_pin()
    runner, api, ledger = universe("flag-on", control=value, objects=objects)
    assert len(api.payloads) == 1 and ledger.universe_reads == ["control", "object"]
    row, payload, plain = ledger.get(DAY), api.payloads[0], plain_api.payloads[0]
    record = row["site_universe"]
    assert record["state"] == "attached" and len(record["site_ids"]) == 6 and row["create_payload"] == payload
    [slice_file] = [f for f in payload["environment"]["files"] if f["path"] == site_universe.SLICE_PATH]
    raw = base64.b64decode(slice_file["data"])
    assert hashlib.sha256(raw).hexdigest() == payload["metadata"]["site_universe_slice_digest"] == record["slice_sha256"]
    assert row["metadata"] == payload["metadata"] and api.sessions[0]["metadata"] == payload["metadata"]
    # The payload is today's plus exactly the file, its digest and one paragraph after the CRM prefix.
    end = "compare semantics without assuming a match. "
    anchor = plain["input"][:plain["input"].index(end) + len(end)]
    expected = without_payload_digest(plain)
    expected["environment"]["files"].append(slice_file)
    expected["metadata"]["site_universe_slice_digest"] = record["slice_sha256"]
    expected["input"] = expected["input"].replace(anchor, anchor + site_universe.paragraph(record), 1)
    assert without_payload_digest(payload) == expected
    assert payload["metadata"]["payload_digest"] == digest(without_payload_digest(payload))
    assert payload["agent"] == plain["agent"]  # No new tool; the session's tool schemas are unchanged.
    # Collection reads only the frozen row: control and the bucket are never read again.
    ledger.universe, ledger.objects = AssertionError("control re-read"), {}
    output = json.loads(api.raw)
    first = output["candidates"][0]
    output["discovery_inventory"] = [
        {"operator": None, "site": None, "location": None, "task_hypothesis": None, "source_urls": [],
         "evidence_gap": "Screened only from the slice", "disposition": "screened", "site_universe_id": record["site_ids"][0]},
        {"operator": first["organization"], "site": first["site"], "location": first["location"],
         "task_hypothesis": first["task"], "source_urls": [first["evidence"][0]["url"]], "evidence_gap": "QA pending",
         "disposition": "candidate", "site_universe_id": record["site_ids"][1]}]
    api.raw, api.turn_status = canonical(output).encode(), "completed"
    collected = runner.start_or_resume(allow_create=False)
    assert collected["state"] == "awaiting_review" and len(api.payloads) == 1
    assert ledger.universe_reads == ["control", "object"]
    block = collected["packet"]["site_universe"]
    outcomes = {item["site_id"]: item for item in block["outcomes"]}
    assert outcomes[record["site_ids"][0]]["outcome"] == "screened"
    assert outcomes[record["site_ids"][1]]["candidate_key"] == collected["packet"]["candidates"][0]["candidate_key"]
    assert block["funnel"]["agent"]["untouched"] == 4 and collected["packet_digest"] == digest(collected["packet"])
    from tools.daily_research.runner import status_summary
    assert status_summary(collected)["site_universe"]["state"] == "attached"


def refused_cases():
    from tests.test_daily_research_site_universe import (
        build_export,
        outcome,
        pin_for,
        prior,
        sha,
        site,
    )
    rows = [site(number) for number in range(1, 10)]
    value, objects = universe_pin(rows)
    key = (value["sha256"], value["generation"])
    corrupt = b"\x1f\x8b corrupt export"
    corrupt_pin = pin_for(corrupt, slice_size=6)
    every = [prior("2026-09-29", [outcome(number) for number in range(1, 10)])]
    for history in every:
        history.update(state="completed", run_key="blueprint-researcher:" + history["date"])
    tampered = prior("2026-09-29", [outcome(1)], tamper=True)
    tampered.update(state="completed", run_key="blueprint-researcher:2026-09-29")
    tampered["packet"]["site_universe"] = "garbage"  # A damaged row that names no readable site.
    return [
        ("site_universe_pin_invalid", {**value, "slice_size": 99}, objects, ()),
        ("site_universe_slice_exceeds_research_window", {**value, "slice_size": 14}, objects, ()),
        ("site_universe_object_missing", value, {}, ()),
        ("site_universe_object_generation_mismatch", value, {key: Refusal("site_universe_object_generation_mismatch")}, ()),
        ("site_universe_object_digest_mismatch", value, {key: build_export(rows[:3])}, ()),
        ("site_universe_object_too_large", value, {key: Refusal("site_universe_object_too_large")}, ()),
        ("site_universe_object_unavailable", value, {key: Refusal("site_universe_object_unavailable")}, ()),
        ("site_universe_export_invalid", corrupt_pin, {(corrupt_pin["sha256"], "1001"): corrupt}, ()),
        ("site_universe_export_binding_mismatch", {**value, "snapshot_id": sha("other")}, objects, ()),
        ("site_universe_history_binding_invalid", value, objects, (tampered,)),
        ("site_universe_attach_unavailable", value, {key: RuntimeError("upstream text")}, ()),
        ("site_universe_slice_empty", value, objects, tuple(every)),
    ]


@pytest.mark.parametrize("case", range(12))
def test_every_site_universe_failure_still_creates_the_session_without_the_slice(universe, case):
    code, control, objects, history = refused_cases()[case]
    _, plain_api, _ = universe("plain", history=deepcopy(history))
    _, api, ledger = universe("refused", control=control, objects=objects, history=deepcopy(history))
    assert len(api.payloads) == 1 and canonical(api.payloads) == canonical(plain_api.payloads)
    record = ledger.get(DAY)["site_universe"]
    assert record["code"] == code and record["state"] == ("exhausted" if code == "site_universe_slice_empty" else "refused")
    from tools.daily_research.runner import status_summary
    assert status_summary(ledger.get(DAY))["site_universe"]["code"] == code


def test_unsupported_profile_records_its_code_and_reads_no_object(fixture, tmp_path):
    runner, api, _ = fixture
    value, objects = universe_pin()
    ledger = runner.ledger = UniverseLedger(tmp_path / "universe", value, objects)
    try:
        row = runner.start_or_resume()
        assert row["site_universe"]["code"] == "site_universe_profile_unsupported" and len(api.payloads) == 1
        assert ledger.universe_reads == ["control"] and "site_universe_slice_digest" not in api.payloads[0]["metadata"]
    finally:
        ledger.db.close()


def test_a_lost_lease_while_reading_the_pin_stops_the_run_as_today(universe):
    runner, api, ledger = universe("lost", control=Refusal("firestore_lease_lost"), create=False)
    with pytest.raises(Refusal, match="^firestore_lease_lost$"):
        runner.start_or_resume()
    assert not api.payloads and ledger.get(DAY) is None and ledger.universe_reads == ["control"]


def ceiling_case(scenario):
    from tests.test_daily_research_site_universe import outcome, prior
    value, objects = universe_pin()
    history = ()
    if scenario == "refused":
        objects = {}
    if scenario == "exhausted":
        every = prior("2026-09-29", [outcome(number) for number in range(1, 10)])
        every.update(state="completed", run_key="blueprint-researcher:2026-09-29")
        history = (every,)
    return value, objects, history


@pytest.mark.parametrize("delta", [100, 300, 10, 0])
@pytest.mark.parametrize(("scenario", "state", "code"), [
    ("attached", "refused", "site_universe_intent_resource_ceiling"),
    ("refused", "refused", "site_universe_object_missing"),
    ("exhausted", "exhausted", "site_universe_slice_empty")])
def test_a_flag_on_run_never_fails_the_intent_ceiling_where_flag_off_succeeds(universe, monkeypatch, scenario,
                                                                               state, code, delta):
    from tools.daily_research import search
    value, objects, history = ceiling_case(scenario)
    _, plain_api, plain = universe("plain", control=None, history=deepcopy(history))
    low = plain.intents[0]
    monkeypatch.setattr(search, "MAX_INTENT", low + delta)
    _, api, ledger = universe("flag-on", control=value, objects=objects, history=deepcopy(history))
    row = ledger.get(DAY)
    assert len(api.payloads) == 1 and canonical(api.payloads) == canonical(plain_api.payloads)
    assert ledger.intents[0] <= low + delta
    if delta >= 100:
        assert {key: row["site_universe"][key] for key in ("state", "code")} == {"state": state, "code": code}
        if delta == 100:  # No full record fits in 100 bytes, so each one shrinks to {state, code}.
            assert set(row["site_universe"]) == {"state", "code"}
    else:
        # No room for even the short record: the row is exactly the flag-off row.
        assert "site_universe" not in row and canonical(row) == canonical(plain.get(DAY))


@pytest.mark.parametrize("scenario", ["attached", "refused", "exhausted"])
def test_the_intent_ceiling_still_stops_a_run_that_flag_off_would_stop(universe, monkeypatch, scenario):
    from tools.daily_research import search
    value, objects, history = ceiling_case(scenario)
    _, _, plain = universe("plain", control=None, history=deepcopy(history))
    monkeypatch.setattr(search, "MAX_INTENT", plain.intents[0] - 1)
    with pytest.raises(Refusal, match="^research_profile_intent_resource_ceiling$"):
        universe("over", control=value, objects=objects, history=deepcopy(history))
