"""The site's quick answers on its job page, kept beside the capture for the job sheet, binding nothing.

Every read here is faked; no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import company_share_permission as permission
from blueprint_pipeline import website_site_job_facts as facts
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_context_contracts import site_job_facts_valid, validate_website_task_context

IDENTITY = {"request_id": "req-1", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1"}
FACTS = {"basis": "owner_stated_unverified", "updated_at": "2026-11-19T10:00:00.000Z", "answers": {
    "items_per_hour": {"code": "60_300", "answer": "60 to 300", "label": "How many an hour", "question": "How many an hour?"},
    "hours_per_day": {"code": "two_shifts", "answer": "Two shifts", "label": "Hours a day",
                      "question": "How many hours a day does this job run?"}}}
RIGHTS = {"statement_version": "2026-11-12.v4", "consent_status": "granted", "consent_revoked": False}


def _record(root: Path) -> dict:
    return json.loads(root.joinpath(*facts.RECORD_PATH).read_text())


def test_the_answers_are_kept_beside_the_capture_with_the_time_they_were_read(tmp_path) -> None:
    value = facts.record_site_job_facts(tmp_path, IDENTITY, 1_795_000_000.7, read=lambda _identity: FACTS)
    assert value == _record(tmp_path) == {"schema_version": "website_site_job_facts_record.v1",
                                          "read_at_epoch": 1_795_000_000, "site_job_facts": FACTS}
    # A site that answered none: kept as such, so the sheet can say so.
    facts.record_site_job_facts(tmp_path, IDENTITY, 1_795_000_100, read=lambda _identity: None)
    assert _record(tmp_path)["site_job_facts"] is None


@pytest.mark.parametrize("failure", [
    ValueError("website_control_task-context_http_400:task_context_request_invalid"),  # a website that does not know it yet
    OSError("unreachable"),
])
def test_a_failed_read_keeps_the_last_record_and_never_raises(tmp_path, failure) -> None:
    facts.record_site_job_facts(tmp_path, IDENTITY, 1_795_000_000, read=lambda _identity: FACTS)

    def fails(_identity):
        raise failure
    assert facts.record_site_job_facts(tmp_path, IDENTITY, 1_795_000_500, read=fails) is None
    assert _record(tmp_path)["read_at_epoch"] == 1_795_000_000


def test_answers_of_another_shape_are_not_kept(tmp_path) -> None:
    bad = {**FACTS, "answers": {"items_per_hour": {"code": "60_300", "answer": "Call 555 0142", "free": "text"}}}
    assert facts.record_site_job_facts(tmp_path, IDENTITY, 1, read=lambda _identity: bad) is None
    assert not tmp_path.joinpath(*facts.RECORD_PATH).exists()


@pytest.mark.parametrize("value", [
    None, [], {**FACTS, "basis": "measured"}, {**FACTS, "extra": 1}, {**FACTS, "updated_at": 5},
    {**FACTS, "answers": {"unknown_question": FACTS["answers"]["items_per_hour"]}},
    {**FACTS, "answers": {"items_per_hour": {**FACTS["answers"]["items_per_hour"], "code": 60}}},
    {**FACTS, "answers": {"items_per_hour": {**FACTS["answers"]["items_per_hour"], "answer": "a\nb"}}},
])
def test_only_the_stated_shape_is_valid(value) -> None:
    assert site_job_facts_valid(FACTS) and not site_job_facts_valid(value)


def _context(**extra) -> dict:
    value = {"schema_version": "website_site_task_context.v1", "purpose": "scene_preparation", "request_id": "req-1",
             "scene_id": "site-req-1", "capture_id": "walkthrough-req-1", "description": "Stack the cartons",
             "confirmed": False, "confirmed_at": None, "operator_answers": {}, "unresolved": [], "capture_rights": RIGHTS,
             **extra}
    return {**value, "context_digest": canonical_digest(value, digest_field="context_digest")}


def test_the_task_context_accepts_the_answers_in_their_shape_and_refuses_any_other() -> None:
    kwargs = {"request_id": "req-1", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1", "purpose": "scene_preparation"}
    assert validate_website_task_context(_context(site_job_facts=FACTS), **kwargs)["site_job_facts"] == FACTS
    with pytest.raises(ValueError, match="website_task_context_site_job_facts_invalid"):
        validate_website_task_context(_context(site_job_facts={"answers": "free text"}), **kwargs)


def test_a_null_answer_set_is_a_good_read_that_clears_the_answers_kept_before(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import website_task_context as context
    kwargs = {"request_id": "req-1", "scene_id": "site-req-1", "capture_id": "walkthrough-req-1", "purpose": "scene_preparation"}
    assert validate_website_task_context(_context(site_job_facts=None), **kwargs)["site_job_facts"] is None
    assert facts.record_site_job_facts(tmp_path, IDENTITY, 100.0, read=lambda identity: FACTS)["site_job_facts"] == FACTS
    monkeypatch.setattr(context, "website_webapp_request", lambda **kwargs: _context(site_job_facts=None))
    cleared = facts.record_site_job_facts(tmp_path, IDENTITY, 200.0)
    assert cleared is not None and cleared["site_job_facts"] is None
    assert json.loads(tmp_path.joinpath(*facts.RECORD_PATH).read_text())["site_job_facts"] is None


def test_only_the_job_sheets_read_asks_for_the_answers(monkeypatch) -> None:
    from blueprint_pipeline import website_task_context as context
    sent = []
    monkeypatch.setattr(context, "website_webapp_request", lambda **kwargs: sent.append(kwargs["payload"]) or _context(
        **({"site_job_facts": FACTS} if kwargs["payload"].get("include_site_job_facts") else {})))
    plain = context.load_current_website_task_context(request_id="req-1", scene_id="site-req-1",
                                                      capture_id="walkthrough-req-1", purpose="scene_preparation")
    assert "site_job_facts" not in plain and "include_site_job_facts" not in sent[-1]
    assert facts.read_live(IDENTITY) == FACTS
    assert sent[-1] == {"request_id": "req-1", "scene_id": "site-req-1", "purpose": "scene_preparation",
                        "include_site_job_facts": True}


def test_the_answers_are_read_beside_the_live_rights_of_a_shared_job_only(tmp_path, monkeypatch) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    read = []
    monkeypatch.setattr(facts, "read_live", lambda identity: read.append(identity) or FACTS)
    # The website's own reader, the job shared: the answers are read and kept.
    monkeypatch.setattr("blueprint_pipeline.website_task_context.load_current_website_task_context",
                        lambda **_kwargs: {"capture_rights": RIGHTS})
    check = permission.record_live(root, IDENTITY, permission.live_rights, 1_795_000_000)
    assert check["outcome"] == "covers_sharing" and read == [IDENTITY] and _record(root)["site_job_facts"] == FACTS
    # Not shared (an older statement): no read.
    monkeypatch.setattr("blueprint_pipeline.website_task_context.load_current_website_task_context",
                        lambda **_kwargs: {"capture_rights": {**RIGHTS, "statement_version": "2026-11-02.v2"}})
    assert permission.record_live(root, IDENTITY, permission.live_rights, 1_795_000_100)["outcome"] == "not_sharing"
    assert read == [IDENTITY]
    # Any other reader (a test's, an injected one): no read, whatever it says.
    assert permission.record_live(root, IDENTITY, lambda _identity: RIGHTS, 1_795_000_200)["outcome"] == "covers_sharing"
    assert read == [IDENTITY]


def test_a_failed_answers_read_changes_nothing_in_the_rights_record(tmp_path, monkeypatch) -> None:
    root = tmp_path / "capture"
    root.mkdir()

    def fails(_identity):
        raise OSError("website down")
    monkeypatch.setattr(facts, "read_live", fails)
    monkeypatch.setattr("blueprint_pipeline.website_task_context.load_current_website_task_context",
                        lambda **_kwargs: {"capture_rights": RIGHTS})
    check = permission.record_live(root, IDENTITY, permission.live_rights, 1_795_000_000)
    assert check["outcome"] == "covers_sharing" and check["granted_at_epoch"] == 1_795_000_000
    assert json.loads(permission.check_path(root).read_text()) == check
    assert not root.joinpath(*facts.RECORD_PATH).exists()
