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


# --------------------------------------------------------------------------------------
# Owner, 2026-11-12: every website job extends its own budget inside the account's
# monthly limit, whether or not its site confirmed its task. A confirmed task's
# preparation retains no grant, so until this change its grant was never read and
# the job waited for a person (the reference job, scene-8ae57e1a...). These tests
# read the real retained files: no grant reader is replaced.
# --------------------------------------------------------------------------------------

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest  # noqa: E402
from tests.test_website_preparation_contract_v2 import VECTOR, NOW  # noqa: E402

PREPARATION_KEY = "website_preparation_authority"


def _confirmed_vector(*, flag=True, spend=20, attempts=16):
    task = copy.deepcopy(VECTOR["task_context_response_preparation"])
    task["confirmed"], task["confirmed_at"] = True, "2026-10-06T00:00:00Z"
    task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    grant.pop("assessment_preparation_proposal", None)
    grant["consent"]["task_confirmed"] = True
    grant.update(task_context_digest=task["context_digest"], preparation_max_total_spend_usd=spend + 5,
                 upstream_max_spend_usd=5, max_total_spend_usd=spend, max_paid_attempts=attempts,
                 scene_configuration_max_retries=1)
    if flag is not None:
        grant[standing.GRANT_FIELD] = flag
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    request = copy.deepcopy(VECTOR["prepared_scene_body"]["request"])
    request["consent"] = copy.deepcopy(grant["consent"])
    request["source"]["binding_id"] = "website-splat-" + request["source"]["content_digest"][7:39]
    request["task"]["task_id"] = "website-" + task["context_digest"][7:27]
    request["execution"].update(max_total_spend_usd=spend, max_paid_attempts=attempts, max_retries=1)
    return request, task, grant


def _ref(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _retain(tmp_path, request, task, grant, *, in_preparation=None, capture_grant=True, capture_root=None):
    """The files a website delivery and its preparation leave on the host, as they are written there."""
    capture = tmp_path / "captures" / request["submission_id"]
    prepared_root = capture / "pipeline" / "website_scene_preparation"
    prepared_root.mkdir(parents=True)
    if capture_grant:
        (capture / "pipeline" / "website_scene_sponsorship.json").write_text(json.dumps(grant))
    (prepared_root / "task_context.json").write_text(json.dumps(task))
    (prepared_root / "runtime_inputs.json").write_text(json.dumps({"synthetic": True}))
    preparation = {"intake_request": request, "binding": {"task_context_digest": task["context_digest"]},
                   **({PREPARATION_KEY: in_preparation} if in_preparation is not None else {})}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    (prepared_root / "preparation.json").write_text(json.dumps(preparation))
    registration = {
        "schema_version": "website_scene_source_registration.v1",
        "request_digest": cross_runtime_canonical_digest(request), "execution_authority_granted": False,
        "capture_source": {"canonical_path": str(capture_root or capture)},
        "references": {name: _ref(prepared_root / f"{name}.json")
                       for name in ("preparation", "runtime_inputs", "task_context")}}
    registration["registration_digest"] = canonical_digest(registration, digest_field="registration_digest")
    bindings = tmp_path / "website-source-bindings"
    bindings.mkdir(exist_ok=True)
    (bindings / (registration["request_digest"][7:] + ".json")).write_text(json.dumps(registration))
    return capture


def _stage_confirmed(tmp_path, request):
    root = tmp_path / "intents"
    receipt = stage_scene_intent(value=request, queue_root=root, authenticated_client="webapp",
                                 trusted_clients={"webapp"}, now=NOW)
    directory = root / receipt["intent_id"]
    return root, directory, json.loads((directory / "intent.json").read_text())


def _reserve_at(root, intent, attempt_id, cost):
    return _reserve(root, intent, attempt_id, cost, now=NOW + 60)


@pytest.fixture
def account_300(tmp_path, monkeypatch):
    """The account's limit file at the owner's USD 300 (2026-11-12) and a fresh report."""
    policy = tmp_path / "account" / account.CONFIG_FILENAME
    report = tmp_path / "account" / "report.json"
    policy.parent.mkdir()
    policy.write_text(json.dumps({"schema_version": account.POLICY_SCHEMA, "monthly_limit_usd": "300",
                                  "report_path": str(report)}))
    monkeypatch.setenv(account.CONFIG_ENV, str(policy))

    def write(counted):
        moment = time.time()
        if report.exists():
            report.unlink()
        report.write_text(json.dumps({
            "schema_version": account.REPORT_SCHEMA, "cap_status": "set",
            "policy_sha256": "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest(),
            "counted_usd": str(counted), "limit_usd": "300", "period": month_of(moment),
            "generated_at": datetime.fromtimestamp(moment, timezone.utc).isoformat()}))
    write("164.32")
    return write


def test_a_confirmed_task_reads_the_grant_its_capture_retained(tmp_path):
    request, task, grant = _confirmed_vector()
    _retain(tmp_path, request, task, grant)
    assert request["consent"]["task_confirmed"] is True
    assert standing.granted_authority(request, tmp_path / "intents", NOW) == grant


@pytest.mark.parametrize("fault", ["no_capture_grant", "unflagged", "flag_false", "other_budget", "other_owner",
                                   "outside_capture", "digest", "expired"])
def test_a_confirmed_task_without_its_own_flagged_grant_does_not_extend(tmp_path, fault):
    request, task, grant = _confirmed_vector(flag=None if fault == "unflagged" else
                                             False if fault == "flag_false" else True)
    if fault == "other_budget":
        grant["max_total_spend_usd"] = 21
        grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    elif fault == "other_owner":
        grant["owner"] = {**grant["owner"], "user_id": "someone-else"}
        grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    elif fault == "digest":
        grant["max_paid_attempts"] = 32  # changed after it was signed
    elsewhere = None
    if fault == "outside_capture":
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "pipeline").mkdir(parents=True)
        (elsewhere / "pipeline" / "website_scene_sponsorship.json").write_text(json.dumps(grant))
    _retain(tmp_path, request, task, grant, capture_grant=fault not in {"no_capture_grant", "outside_capture"},
            capture_root=elsewhere)
    moment = grant["expires_at_epoch"] + 1 if fault == "expired" else NOW
    assert standing.granted_authority(request, tmp_path / "intents", moment) is None


def test_a_grant_the_preparation_retained_is_read_first_and_never_replaced(tmp_path):
    request, task, grant = _confirmed_vector()
    unconfirmed = copy.deepcopy(request)
    unconfirmed["consent"]["task_confirmed"] = False
    unflagged = {key: value for key, value in grant.items() if key not in {standing.GRANT_FIELD, "authority_digest"}}
    unflagged["consent"] = dict(unconfirmed["consent"])
    unflagged_task = {**task, "confirmed": False, "confirmed_at": None}
    unflagged_task["context_digest"] = canonical_digest(unflagged_task, digest_field="context_digest")
    unflagged["task_context_digest"] = unflagged_task["context_digest"]
    unconfirmed["task"]["task_id"] = "website-" + unflagged_task["context_digest"][7:27]
    unflagged["authority_digest"] = canonical_digest(unflagged, digest_field="authority_digest")
    # The preparation retained an unflagged grant; a flagged one beside the capture changes nothing.
    flagged = {**unflagged, standing.GRANT_FIELD: True}
    flagged["authority_digest"] = canonical_digest(flagged, digest_field="authority_digest")
    _retain(tmp_path, unconfirmed, unflagged_task, flagged, in_preparation=unflagged)
    assert standing.granted_authority(unconfirmed, tmp_path / "intents", NOW) is None


def test_the_reference_job_extends_by_exactly_the_shortfall(tmp_path, account_300):
    # scene-8ae57e1a...: $20 ceiling, 16 paid attempts, a confirmed task, $18 still held.
    request, task, grant = _confirmed_vector(spend=20, attempts=16)
    _retain(tmp_path, request, task, grant)
    root, directory, intent = _stage_confirmed(tmp_path, request)
    _reserve_at(root, intent, "scene-configuration-held", 18)
    _reserve_at(root, intent, "scene-configuration-next", 18)
    [record] = _records(directory)
    assert (record["prior_ceiling_usd"], record["amount_usd"], record["resulting_ceiling_usd"]) == (20, 16, 36)
    assert record["grant_authority_digest"] == grant["authority_digest"]
    reading = record["account_reading"]
    assert (reading["limit_usd"], reading["counted_usd"], reading["headroom_usd"]) == ("300", "164.32", "135.68")
    # The owner's ceiling and attempt cap are untouched; only the standing chain records the shortfall.
    assert effective_execution_budget(directory, intent)["max_total_spend_usd"] == 20


def test_the_reference_job_after_its_gpu_bill_extends_by_four_dollars_and_a_cent(tmp_path, account_300):
    request, task, grant = _confirmed_vector(spend=20, attempts=16)
    _retain(tmp_path, request, task, grant)
    root, directory, intent = _stage_confirmed(tmp_path, request)
    _reserve_at(root, intent, "scene-configuration-held", 18)
    _reserve_at(root, intent, "scene-configuration-next", 6.01)
    [record] = _records(directory)
    assert (record["prior_ceiling_usd"], record["amount_usd"], record["resulting_ceiling_usd"]) == (20, 4.01, 24.01)


def test_a_confirmed_task_without_a_grant_waits_as_before(tmp_path, account_300):
    request, task, grant = _confirmed_vector()
    _retain(tmp_path, request, task, grant, capture_grant=False)
    root, directory, intent = _stage_confirmed(tmp_path, request)
    _reserve_at(root, intent, "scene-configuration-held", 18)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve_at(root, intent, "scene-configuration-next", 18)
    assert _records(directory) == []


def test_a_confirmed_task_is_refused_when_the_account_headroom_is_smaller_than_the_shortfall(tmp_path, account_300):
    account_300("284.01")  # $15.99 of headroom, $16 short
    request, task, grant = _confirmed_vector()
    _retain(tmp_path, request, task, grant)
    root, directory, intent = _stage_confirmed(tmp_path, request)
    _reserve_at(root, intent, "scene-configuration-held", 18)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve_at(root, intent, "scene-configuration-next", 18)
    assert _records(directory) == []
    account_300("284")  # exactly $16 of headroom
    assert _reserve_at(root, intent, "scene-configuration-next", 18)["status"] == "reserved"
    assert _records(directory)[0]["account_reading"]["headroom_usd"] == "16"


def test_nothing_extends_past_the_monthly_limit(tmp_path, account_300):
    account_300("300")  # the month is at its limit
    request, task, grant = _confirmed_vector()
    _retain(tmp_path, request, task, grant)
    root, directory, intent = _stage_confirmed(tmp_path, request)
    _reserve_at(root, intent, "scene-configuration-held", 18)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve_at(root, intent, "scene-configuration-next", 2.01)
    assert _records(directory) == []
    # Two shortfalls from one report never pass the limit together.
    account_300("290")  # $10 of headroom
    assert _reserve_at(root, intent, "scene-configuration-next", 8)["status"] == "reserved"  # $6 short
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve_at(root, intent, "scene-configuration-third", 5)  # $5 short, $4 left
    assert [r["amount_usd"] for r in _records(directory)] == [6]
