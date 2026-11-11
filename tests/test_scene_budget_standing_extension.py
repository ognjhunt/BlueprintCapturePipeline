"""Owner decision 2026-11-10: a flagged job's ceiling extends by a reservation's shortfall.

The only hard money stop is the account's monthly limit. A job whose signed
grant carries ``automatic_budget_extension: true`` extends its scene ceiling
by exactly what a reservation lacks, when the account's fresh report has the
headroom and the job has paid attempts left; otherwise it waits as before and
its next timed re-ask asks again. Every figure here is synthetic.
"""
import copy
import hashlib
import json
import time
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import account_monthly_spend as account
from blueprint_pipeline import scene_budget_standing_extension as standing
from blueprint_pipeline import task_evaluation_scene_execution_budget as owner_budget
from blueprint_pipeline import website_preparation_authority
from blueprint_pipeline.paid_spend_meter import month_of
from blueprint_pipeline.task_evaluation_scene_execution_budget_evidence import effective_execution_budget
from blueprint_pipeline.task_evaluation_scene_intake import (
    SceneIntakeError, reserve_scene_attempt, stage_scene_intent,
)
from tests.test_task_evaluation_scene_intake import request as intake_request

GRANT_DIGEST = "sha256:" + "ab" * 32


def _request(submission="upload-1", *, spend=10, attempts=8):
    value = copy.deepcopy(intake_request())
    value["submission_id"] = submission
    value["execution"].update({"max_total_spend_usd": spend, "max_paid_attempts": attempts,
                               "purpose": "scene_preparation", "allowed_providers": ["vast", "openai"],
                               "policy_candidates": []})
    return value


def _stage(root, **kwargs):
    receipt = stage_scene_intent(value=_request(**kwargs), queue_root=root, authenticated_client="webapp",
                                 trusted_clients={"webapp"}, now=100)
    directory = root / receipt["intent_id"]
    return directory, json.loads((directory / "intent.json").read_text())


def _reserve(root, intent, attempt_id, cost, now=101):
    return reserve_scene_attempt(queue_root=root, intent_id=intent["intent_id"], attempt_id=attempt_id,
                                 source_commit="d" * 40, runtime_digest="sha256:" + "e" * 64,
                                 input_digest="sha256:" + "f" * 64, provider="vast", maximum_spend_usd=cost,
                                 now=now)


@pytest.fixture
def grant(monkeypatch):
    """The job's retained signed grant; ``flag`` is what the WebApp issued."""
    state = {"flag": True}

    def retained(*, request, queue_root, now):
        authority = {"schema_version": "website_scene_sponsorship.v1", "authority_digest": GRANT_DIGEST}
        if state["flag"] is not None:
            authority[standing.GRANT_FIELD] = state["flag"]
        return {}, authority, None

    monkeypatch.setattr(website_preparation_authority, "_retained_sources", retained)
    return state


@pytest.fixture
def account_report(tmp_path, monkeypatch):
    """Writes the account's limit file and report, as the spend guard does every minute."""
    policy = tmp_path / "account" / account.CONFIG_FILENAME
    report = tmp_path / "account" / "report.json"
    policy.parent.mkdir()
    policy.write_text(json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": "200",
                                  "report_path": str(report)}))
    monkeypatch.setenv(account.CONFIG_ENV, str(policy))

    def write(counted, *, age=0.0):
        moment = time.time()
        sha = "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest()
        if report.exists():
            report.unlink()
        report.write_text(json.dumps({
            "schema_version": account.REPORT_SCHEMA, "cap_status": "set", "policy_sha256": sha,
            "counted_usd": str(counted), "limit_usd": "200", "period": month_of(moment),
            "generated_at": datetime.fromtimestamp(moment - age, timezone.utc).isoformat()}))
        report.chmod(0o644)
    write(50)
    return write


def _records(directory):
    store = directory / standing.DIRECTORY
    return sorted((json.loads(p.read_text()) for p in store.glob("*.json")), key=lambda r: r["sequence"]) \
        if store.exists() else []


def test_a_flagged_job_extends_by_exactly_the_shortfall_and_records_it(tmp_path, grant, account_report):
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    row = _reserve(tmp_path, intent, "a2", 7.25)
    assert row["status"] == "reserved" and owner_budget.ATTEMPT_GRANT_FIELD not in row
    [record] = _records(directory)
    assert record["amount_usd"] == 3.25 and record["prior_ceiling_usd"] == 10
    assert record["resulting_ceiling_usd"] == 13.25 and record["issuer_channel"] == "standing_policy"
    assert record["reason"] == standing.REASON and record["attempt_id"] == "a2"
    assert record["grant_authority_digest"] == GRANT_DIGEST
    reading = record["account_reading"]
    assert (reading["counted_usd"], reading["limit_usd"], reading["headroom_usd"]) == ("50", "200", "150")
    assert standing.effective_scene_budget(directory, intent)["max_total_spend_usd"] == 13.25
    # The owner's own store and chain are untouched: an older release sees the owner's $10.
    assert not (directory / owner_budget.DIRECTORY).exists()
    assert effective_execution_budget(directory, intent)["max_total_spend_usd"] == 10
    # Nothing more is extended while the reservation fits.
    assert _reserve(tmp_path, intent, "a2", 7.25) == row
    assert len(_records(directory)) == 1


@pytest.mark.parametrize("flag", [None, False])
def test_an_unflagged_job_waits_as_today(tmp_path, grant, account_report, flag):
    grant["flag"] = flag
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 7.25)
    with pytest.raises(SceneIntakeError, match="attempt_spend_exceeds_authorized_limit"):
        _reserve(tmp_path, intent, "a3", 11)
    assert not (directory / standing.DIRECTORY).exists()


def test_no_extension_without_account_headroom(tmp_path, grant, account_report):
    account_report(196.75)  # $3.25 of headroom
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 7.5)
    assert _records(directory) == []
    assert _reserve(tmp_path, intent, "a2", 7.25)["status"] == "reserved"  # exactly the headroom fits
    assert _records(directory)[0]["account_reading"]["headroom_usd"] == "3.25"


def test_no_extension_without_a_configured_limit(tmp_path, grant, monkeypatch):
    monkeypatch.setenv(account.CONFIG_ENV, str(tmp_path / "absent.json"))
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 7.25)
    assert _records(directory) == []


def test_a_stale_report_waits_and_the_next_timed_re_ask_extends(tmp_path, grant, account_report):
    account_report(50, age=account.REPORT_MAX_AGE_SECONDS + 120)
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 7.25, now=110)
    assert _records(directory) == []
    # The spend guard rewrites the report; the next pass asks again with no operator step.
    account_report(50)
    assert _reserve(tmp_path, intent, "a2", 7.25, now=120)["status"] == "reserved"
    assert _records(directory)[0]["issued_at_epoch"] == 120


def test_no_extension_once_paid_attempts_are_spent(tmp_path, grant, account_report):
    directory, intent = _stage(tmp_path, attempts=1)
    _reserve(tmp_path, intent, "a1", 6)
    with pytest.raises(SceneIntakeError, match="attempt_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 7.25)
    assert _records(directory) == []


def test_two_jobs_cannot_extend_against_the_same_headroom_between_reports(tmp_path, grant, account_report):
    account_report(190)  # $10 of headroom until the next report
    first_dir, first = _stage(tmp_path, submission="upload-1")
    second_dir, second = _stage(tmp_path, submission="upload-2")
    for intent in (first, second):
        _reserve(tmp_path, intent, "a1", 6)
    assert _reserve(tmp_path, first, "a2", 10)["status"] == "reserved"  # extends by $6
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, second, "a2", 10)  # $4 left, $6 asked
    assert _records(second_dir) == []
    assert _reserve(tmp_path, second, "a2", 8)["status"] == "reserved"  # $4 fits
    assert _records(second_dir)[0]["account_reading"]["standing_extensions_since_report_usd"] in ("6", "6.0")
    assert _records(first_dir)[0]["amount_usd"] == 6


def test_owner_records_still_read_and_raise_the_ceiling_beside_standing_ones(tmp_path, grant, account_report):
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    _reserve(tmp_path, intent, "a2", 7.25)  # standing: $13.25
    owner = owner_budget.extend_scene_execution_budget(
        queue_root=tmp_path, intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
        owner=intent["request"]["owner"], authenticated_client="webapp", trusted_clients={"webapp"},
        max_total_spend_usd=20, max_paid_attempts=8, authorization_reference="owner_approval:test",
        ack=owner_budget.ACK, now=130)
    budget = standing.effective_scene_budget(directory, intent)
    assert budget["max_total_spend_usd"] == 20 and budget["extension_digest"] == owner["extension_digest"]
    row = _reserve(tmp_path, intent, "a3", 6.75, now=131)  # fits the owner's $20 exactly
    assert row[owner_budget.ATTEMPT_GRANT_FIELD] == owner["extension_digest"]
    _reserve(tmp_path, intent, "a4", 2, now=132)  # standing again, from the owner's $20
    assert _records(directory)[-1]["prior_ceiling_usd"] == 20
    assert _records(directory)[-1]["resulting_ceiling_usd"] == 22
    assert effective_execution_budget(directory, intent)["max_total_spend_usd"] == 20


def test_a_job_with_only_standing_records_reads_the_same_after_a_restart(tmp_path, grant, account_report,
                                                                         monkeypatch):
    directory, intent = _stage(tmp_path)
    _reserve(tmp_path, intent, "a1", 6)
    row = _reserve(tmp_path, intent, "a2", 7.25)
    # A new process: no grant reader, no account report; only the records on disk.
    grant["flag"] = None
    monkeypatch.setenv(account.CONFIG_ENV, str(tmp_path / "absent.json"))
    assert standing.effective_scene_budget(directory, intent)["max_total_spend_usd"] == 13.25
    assert _reserve(tmp_path, intent, "a2", 7.25) == row
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a3", 0.01)
    # A changed record fails closed instead of being read as a different ceiling.
    path = next((directory / standing.DIRECTORY).glob("*.json"))
    value = json.loads(path.read_text())
    value["resulting_ceiling_usd"] = 100
    path.chmod(0o640)
    path.write_text(json.dumps(value))
    with pytest.raises(SceneIntakeError, match="record_digest_invalid"):
        standing.effective_scene_budget(directory, intent)


def test_the_schema_bound_is_never_passed(tmp_path, grant, account_report):
    directory, intent = _stage(tmp_path, spend=990)
    _reserve(tmp_path, intent, "a1", 980)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(tmp_path, intent, "a2", 25)
    assert _records(directory) == []
