"""One paid attempt per source/task, and one resend of an unanswered one; no live model calls."""
import fcntl
import hashlib
import json

import pytest

from blueprint_pipeline import website_gemini_receipts as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
)


def context():
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req",
             "scene_id": "scene", "capture_id": "capture", "confirmed": True,
             "confirmed_at": "2026-09-19T00:00:00Z", "description": "Move the box"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def setup_call(tmp_path, monkeypatch):
    calls = []
    def reserve(**kwargs):
        calls.append(("reserve", kwargs))
        admission = {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
                     "resource_class": kwargs["resource_class"], "status": "admitted",
                     "allocation_binding_digest": kwargs["binding_digest"], "blockers": []}
        return admission, require_paid_resource_admission(admission,
            resource_class=kwargs["resource_class"], expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
    def invoke():
        calls.append(("invoke", None))
        return {"status": "completed", "targets": ["box"]}
    return dict(output_root=tmp_path, binding={"source": "video-one", "prompt": "task"},
                task_context=context(), maximum_cost_usd=1.04, preflight=lambda: None, invoke=invoke), calls


def test_completed_replay_needs_no_new_key_admission_or_call(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    first = module.retained_gemini_call(**args)
    args["preflight"] = lambda: pytest.fail("replay should not need a key")
    assert module.retained_gemini_call(**args) == first
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    assert calls[0][1]["provider"] == "google"


def timeout():
    raise TimeoutError("provider may have charged")


class Crash(BaseException):
    """The process dies here: nothing after this point runs."""


def first_receipt(root, args):
    request, digest = module._request(binding=args["binding"], task_context=args["task_context"],
                                      maximum_cost_usd=args["maximum_cost_usd"], source_video_digest=None)
    return request, digest, root / f"{digest[7:]}.json"


def test_unanswered_call_is_counted_once_and_resent_once(tmp_path, monkeypatch):
    # Owner decision 2026-11-10 ("same rule for Gemini").
    args, calls = setup_call(tmp_path, monkeypatch)
    request, digest, original = first_receipt(tmp_path, args)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    held = original.read_bytes()
    result = module.retained_gemini_call(**args)
    assert result == {"status": "completed", "targets": ["box"]}
    resend = canonical_digest({**request, "resend_of": digest, "resend_index": 1})
    assert [(kind, row and row["binding_digest"]) for kind, row in calls] == [
        ("reserve", digest), ("reserve", resend), ("invoke", None)]
    # The first receipt and its reservation stay as they were: held, never released or rewritten.
    assert original.read_bytes() == held and json.loads(held)["status"] == "submitting"
    record = json.loads((tmp_path / module.UNANSWERED / f"{digest[7:]}.json").read_text())
    assert {key: value for key, value in record.items() if key != "recorded_at"} == {
        "schema_version": "website_gemini_unanswered_dispatch.v1", "status": "counted_as_spent",
        "outcome": "unknown", "request_digest": digest,
        "receipt_sha256": "sha256:" + hashlib.sha256(held).hexdigest(),
        "admission": json.loads(held)["admission"], "maximum_cost_usd": 1.04,
        "reservation": "held_at_maximum_never_released_or_settled_as_zero", "resends_allowed": 1,
        "resend_request_digest": resend}
    receipt = json.loads((tmp_path / f"{resend[7:]}.json").read_text())
    assert receipt["status"] == "completed" and receipt["resend_of"] == digest and receipt["resend_index"] == 1
    assert receipt["request"] == {**request, "resend_of": digest, "resend_index": 1}
    assert receipt["admission"]["allocation_binding_digest"] == resend
    # Every later attempt reuses the resend: no key, reservation or call.
    for _ in range(2):
        assert module.retained_gemini_call(**{**args, "preflight": lambda: pytest.fail("no key needed"),
                                              "invoke": lambda: pytest.fail("never bought again")}) == result
    assert len(calls) == 3 and sorted(path.name for path in tmp_path.glob("*.json")) == sorted(
        [original.name, f"{resend[7:]}.json"])


def test_a_resend_left_unanswered_stops_for_review_and_is_never_sent_again(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            module.retained_gemini_call(**{**args, "invoke": timeout})
    for _ in range(3):
        with pytest.raises(ValueError, match="website_gemini_requires_reconciliation"):
            module.retained_gemini_call(**args)
    assert [kind for kind, _ in calls] == ["reserve", "reserve"]
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert sorted(row["status"] for row in receipts) == ["submitting", "submitting"]
    assert len(list((tmp_path / module.UNANSWERED).glob("*.json"))) == 1


def test_a_refused_resend_reservation_calls_nothing_and_asks_again_next_time(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    _, digest, _ = first_receipt(tmp_path, args)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    reserve = module.reserve_website_preparation_spend

    def refuse(**kwargs):
        calls.append(("refused", kwargs))
        raise PaidResourceAdmissionBlocked(["website_preparation_allowance_exhausted"])
    monkeypatch.setattr(module, "reserve_website_preparation_spend", refuse)
    record = tmp_path / module.UNANSWERED / f"{digest[7:]}.json"
    with pytest.raises(PaidResourceAdmissionBlocked):
        module.retained_gemini_call(**args)
    counted = record.read_bytes()
    with pytest.raises(PaidResourceAdmissionBlocked):
        module.retained_gemini_call(**args)
    assert record.read_bytes() == counted and len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
    assert module.retained_gemini_call(**args)["targets"] == ["box"]
    assert [kind for kind, _ in calls] == ["reserve", "refused", "refused", "reserve", "invoke"]
    assert calls[1][1]["binding_digest"] == calls[2][1]["binding_digest"] == calls[3][1]["binding_digest"] != digest
    assert record.read_bytes() == counted


def webapp(calls):
    """The website's reservation: a binding it already reserved is refused, never granted twice."""
    reserved = set()

    def reserve(**kwargs):
        if kwargs["binding_digest"] in reserved:
            calls.append(("already_reserved", kwargs))
            raise ValueError("already_reserved")
        reserved.add(kwargs["binding_digest"])
        calls.append(("reserve", kwargs))
        admission = {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "resource_class": kwargs["resource_class"],
                     "status": "admitted", "allocation_binding_digest": kwargs["binding_digest"], "blockers": []}
        return admission, require_paid_resource_admission(admission, resource_class=kwargs["resource_class"],
                                                          expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    return reserve


@pytest.mark.parametrize("point", ["first_call", "after_record", "after_reservation", "after_resend_receipt",
                                   "during_record_write"])
def test_a_crash_at_each_point_never_spends_beyond_one_resend(tmp_path, monkeypatch, point):
    args, calls = setup_call(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "reserve_website_preparation_spend", webapp(calls))
    _, digest, _ = first_receipt(tmp_path, args)

    def crash():
        raise Crash()
    with pytest.raises(Crash):
        module.retained_gemini_call(**{**args, "invoke": crash})  # dies after the first receipt
    record = tmp_path / module.UNANSWERED / f"{digest[7:]}.json"
    if point == "after_record":
        reserve = module.reserve_website_preparation_spend
        monkeypatch.setattr(module, "reserve_website_preparation_spend", lambda **_k: crash())
        with pytest.raises(Crash):
            module.retained_gemini_call(**args)
        assert record.is_file()
        monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
    elif point == "after_reservation":
        reserve = module.reserve_website_preparation_spend
        monkeypatch.setattr(module, "reserve_website_preparation_spend", lambda **k: (reserve(**k), crash()))
        with pytest.raises(Crash):
            module.retained_gemini_call(**args)
        monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
        # The website already holds the resend's reservation; without its grant nothing is called.
        for _ in range(2):
            with pytest.raises(ValueError, match="already_reserved"):
                module.retained_gemini_call(**args)
        assert [kind for kind, _ in calls] == ["reserve", "reserve", "already_reserved", "already_reserved"]
        return
    elif point == "after_resend_receipt":
        with pytest.raises(Crash):
            module.retained_gemini_call(**{**args, "invoke": crash})
        for _ in range(2):
            with pytest.raises(ValueError, match="requires_reconciliation"):
                module.retained_gemini_call(**args)
        assert [kind for kind, _ in calls] == ["reserve", "reserve"]
        return
    elif point == "during_record_write":
        # A record cut short before it was linked in place leaves only its temporary file.
        record.parent.mkdir()
        record.with_suffix(".tmp").write_text('{"schema_version": "website_gemi')
    assert module.retained_gemini_call(**args)["targets"] == ["box"]
    assert [kind for kind, _ in calls] == ["reserve", "reserve", "invoke"]
    assert module.retained_gemini_call(**args)["targets"] == ["box"] and len(calls) == 3
    assert json.loads(record.read_text())["status"] == "counted_as_spent"


def test_two_workers_cannot_both_resend(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    invoke = args["invoke"]

    def second_worker_meanwhile():
        # Another worker (or a restart) arriving while the resend is in flight is refused by the lock.
        with pytest.raises(ValueError, match="in_progress"):
            module.retained_gemini_call(**args)
        return invoke()
    module.retained_gemini_call(**{**args, "invoke": second_worker_meanwhile})
    assert [kind for kind, _ in calls] == ["reserve", "reserve", "invoke"]


@pytest.mark.parametrize("tamper", ["receipt_bytes", "record_resends", "record_symlink", "record_unreadable"])
def test_a_changed_receipt_or_record_refuses_and_never_spends(tmp_path, monkeypatch, tamper):
    args, calls = setup_call(tmp_path, monkeypatch)
    _, digest, original = first_receipt(tmp_path, args)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    def refuse(**_kwargs):
        raise PaidResourceAdmissionBlocked(["website_preparation_allowance_exhausted"])
    monkeypatch.setattr(module, "reserve_website_preparation_spend", refuse)
    with pytest.raises(PaidResourceAdmissionBlocked):
        module.retained_gemini_call(**args)  # counted; the resend's reservation is refused
    monkeypatch.setattr(module, "reserve_website_preparation_spend", lambda **_k: pytest.fail("never reserves"))
    record = tmp_path / module.UNANSWERED / f"{digest[7:]}.json"
    if tamper == "receipt_bytes":
        original.write_text(original.read_text() + " ")
    elif tamper == "record_resends":
        record.write_text(json.dumps({**json.loads(record.read_text()), "resends_allowed": 2}))
    elif tamper == "record_symlink":
        target = tmp_path / "elsewhere.txt"
        target.write_bytes(record.read_bytes())
        record.unlink()
        record.symlink_to(target)
    else:
        record.write_text("{")
    with pytest.raises(ValueError, match="unanswered_record_invalid"):
        module.retained_gemini_call(**args)
    assert [kind for kind, _ in calls] == ["reserve"]


@pytest.mark.parametrize("change", ["no_admission", "other_admission", "other_request", "unreadable"])
def test_an_unprovable_unanswered_receipt_is_never_counted_or_resent(tmp_path, monkeypatch, change):
    args, calls = setup_call(tmp_path, monkeypatch)
    _, digest, original = first_receipt(tmp_path, args)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    receipt = json.loads(original.read_text())
    if change == "no_admission":
        receipt.pop("admission")
    elif change == "other_admission":
        receipt["admission"]["allocation_binding_digest"] = "sha256:" + "0" * 64
    elif change == "other_request":
        receipt["request"]["maximum_cost_usd"] = 0.01
    original.write_text("{" if change == "unreadable" else json.dumps(receipt))
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.retained_gemini_call(**args)
    assert len(calls) == 1 and not (tmp_path / module.UNANSWERED).exists()


def test_a_request_that_supersedes_an_incomplete_one_is_already_its_resend(tmp_path, monkeypatch):
    # website_image_completion's later revisions name the incomplete request they replace.
    args, calls = setup_call(tmp_path, monkeypatch)
    args["binding"] = {**args["binding"], "supersedes_incomplete_request_digest": "sha256:" + "1" * 64}
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.retained_gemini_call(**args)
    assert len(calls) == 1 and not (tmp_path / module.UNANSWERED).exists()


def earlier_form(args):
    return {**args["binding"], "prompt": "earlier task"}, 0.5


def test_an_unanswered_earlier_form_is_resent_once_as_the_current_form(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    binding, cost = earlier_form(args)
    earlier_request, earlier, _ = first_receipt(tmp_path, {**args, "binding": binding, "maximum_cost_usd": cost})
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "binding": binding, "maximum_cost_usd": cost, "invoke": timeout})
    current_request, current, _ = first_receipt(tmp_path, args)
    result = module.retained_gemini_call(**args, earlier_requests=[(binding, cost)])
    resend = canonical_digest({**current_request, "resend_of": earlier, "resend_index": 1})
    assert [(kind, row and row["binding_digest"]) for kind, row in calls] == [
        ("reserve", earlier), ("reserve", resend), ("invoke", None)]
    assert calls[1][1]["maximum_cost_usd"] == 1.04
    assert json.loads((tmp_path / module.UNANSWERED / f"{earlier[7:]}.json").read_text())["request_digest"] == earlier
    assert not (tmp_path / f"{current[7:]}.json").exists()
    assert module.retained_gemini_call(**args, earlier_requests=[(binding, cost)]) == result
    # A later form that lists both still finds the decision's one resend.
    later = {**args["binding"], "prompt": "later task"}
    assert module.retained_gemini_call(**{**args, "binding": later},
                                       earlier_requests=[(args["binding"], 1.04), (binding, cost)]) == result
    assert len(calls) == 3


def test_an_earlier_forms_resend_is_never_sent_twice(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    binding, cost = earlier_form(args)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "binding": binding, "maximum_cost_usd": cost, "invoke": timeout})
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout}, earlier_requests=[(binding, cost)])
    later = {**args["binding"], "prompt": "later task"}
    for current, earlier in ((args["binding"], [(binding, cost)]),
                             (later, [(args["binding"], 1.04), (binding, cost)])):
        with pytest.raises(ValueError, match="requires_reconciliation"):
            module.retained_gemini_call(**{**args, "binding": current}, earlier_requests=earlier)
    assert [kind for kind, _ in calls] == ["reserve", "reserve"]


def test_a_form_changed_before_its_counted_resend_was_sent_stops_for_review(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    binding, cost = earlier_form(args)
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "binding": binding, "maximum_cost_usd": cost, "invoke": timeout})
    reserve = module.reserve_website_preparation_spend
    monkeypatch.setattr(module, "reserve_website_preparation_spend", lambda **_k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        module.retained_gemini_call(**args, earlier_requests=[(binding, cost)])  # counted for this form's resend
    monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
    later = {**args["binding"], "prompt": "later task"}
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.retained_gemini_call(**{**args, "binding": later}, earlier_requests=[(args["binding"], 1.04),
                                                                                    (binding, cost)])
    assert [kind for kind, _ in calls] == ["reserve"]


def test_two_bought_forms_of_one_decision_are_never_resent(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    binding, cost = earlier_form(args)
    for form in ({**args, "binding": binding, "maximum_cost_usd": cost}, args):
        with pytest.raises(TimeoutError):
            module.retained_gemini_call(**{**form, "invoke": timeout})
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.retained_gemini_call(**args, earlier_requests=[(binding, cost)])
    assert [kind for kind, _ in calls] == ["reserve", "reserve"]


def test_lease_lost_after_reservation_cannot_start_provider_call(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    reserve = module.reserve_website_preparation_spend
    lost = False

    def check(boundary):
        if lost:
            raise HandoffLeaseError("lost", boundary)

    def reserve_then_lose(**kwargs):
        nonlocal lost
        result = reserve(**kwargs)
        lost = True
        return result

    monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve_then_lose)
    with bind_handoff_lease(check), pytest.raises(HandoffLeaseError):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve"]
    # Its receipt cannot show the call was never made: it is counted as spent and sent once more.
    module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "reserve", "invoke"]
    assert calls[0][1]["binding_digest"] != calls[1][1]["binding_digest"]


@pytest.mark.parametrize("bad_grant", ["missing", "different_binding"])
def test_provider_call_requires_original_bound_opaque_grant(tmp_path, monkeypatch, bad_grant):
    args, calls = setup_call(tmp_path, monkeypatch)
    reserve = module.reserve_website_preparation_spend

    def replace_grant(**kwargs):
        admission, grant = reserve(**kwargs)
        if bad_grant == "missing":
            grant = object()
        else:
            grant = require_paid_resource_admission({**admission, "allocation_binding_digest": "sha256:" + "0" * 64},
                resource_class="evaluator_api", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        return admission, grant

    monkeypatch.setattr(module, "reserve_website_preparation_spend", replace_grant)
    with pytest.raises(PaidResourceAdmissionBlocked):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve"]


def test_already_started_call_can_close_its_receipt_after_lease_loss(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    invoke = args["invoke"]
    lost = False

    def check(boundary):
        if lost:
            raise HandoffLeaseError("lost", boundary)

    def finish_started_call():
        nonlocal lost
        lost = True
        return invoke()

    args["invoke"] = finish_started_call
    with bind_handoff_lease(check):
        result = module.retained_gemini_call(**args)
        assert module.retained_gemini_call(**args) == result
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["status"] == "completed"


def test_denied_cross_worker_reservation_cannot_invoke(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    def refuse(**kwargs):
        raise ValueError("already_reserved")
    monkeypatch.setattr(module, "reserve_website_preparation_spend", refuse)
    with pytest.raises(ValueError, match="already_reserved"):
        module.retained_gemini_call(**args)
    assert not calls
    assert not list(tmp_path.glob("*.json"))


def test_tampered_result_or_binding_cannot_be_reused(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    module.retained_gemini_call(**args)
    path = next(tmp_path.glob("*.json"))
    receipt = json.loads(path.read_text())
    receipt["result"]["targets"] = ["unrelated chair"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt_invalid"):
        module.retained_gemini_call(**args)
    assert len(calls) == 2


@pytest.mark.parametrize("changed", ["source", "prompt"])
def test_changed_analysis_input_requires_new_admission_and_preserves_prior_receipt(tmp_path, monkeypatch, changed):
    args, calls = setup_call(tmp_path, monkeypatch)
    module.retained_gemini_call(**args)
    prior = next(tmp_path.glob("*.json"))
    original = prior.read_bytes()
    args["binding"] = {**args["binding"], changed: "changed " + args["binding"][changed]}
    module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke", "reserve", "invoke"]
    assert calls[0][1]["binding_digest"] != calls[2][1]["binding_digest"]
    assert prior.read_bytes() == original and len(list(tmp_path.glob("*.json"))) == 2


def test_concurrent_worker_cannot_enter_locked_request(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    module.retained_gemini_call(**args)
    with next(tmp_path.glob("*.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="in_progress"):
            module.retained_gemini_call(**args)
    assert len(calls) == 2


def test_quote_includes_full_thinking_window_and_refuses_unpriced_model():
    assert module.gemini_quote(model="gemini-3.8-flash", input_tokens=1_048_576) == 1.04
    assert module.gemini_quote(model="gemini-3.8-flash", input_tokens=24000) == .27
    with pytest.raises(ValueError, match="pricing_refresh"):
        module.gemini_quote(model="unpriced", input_tokens=1)


def test_video_entrypoint_reuses_analysis_before_credentials_or_sdk(tmp_path, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    import sys
    from types import SimpleNamespace
    import google
    args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setenv(analysis.GATE_ENV, "true")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setattr(google, "genai", SimpleNamespace(), raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", google.genai)
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **kwargs: {**args["invoke"](), "input_video_sha256": analysis.sha256_file(video)})
    kwargs = dict(video_path=video, task_context=context(), output_root=tmp_path / "receipts")
    assert analysis.analyze_removal_targets(**kwargs)["targets"] == ["box"]
    monkeypatch.delenv(analysis.GATE_ENV)
    monkeypatch.setattr(analysis, "_api_key", lambda: (None, None))
    assert analysis.analyze_removal_targets(**kwargs)["targets"] == ["box"]
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    video.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="runtime_not_configured"):
        analysis.analyze_removal_targets(**kwargs)
    assert len(calls) == 2


#: The receipt origin/main wrote for this static fixture's analysis, before the
#: destination region was asked for (2026-10-31).
RETAINED_BEFORE_REGION = "d0d0be4de8fb4fc6181580b73485010315fd0cf56e022a72e658fc257d6810c3"


@pytest.mark.parametrize("status", ["completed", "submitting"])
def test_an_analysis_bought_before_the_region_prompt_is_never_bought_again(tmp_path, monkeypatch, status):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_k: pytest.fail("must not call the provider"))
    result = {"status": "completed", "targets": ["box"]}
    root = tmp_path / "receipts"
    root.mkdir()
    (root / f"{RETAINED_BEFORE_REGION}.json").write_text(json.dumps(
        {"status": status, "request_digest": "sha256:" + RETAINED_BEFORE_REGION,
         **({"result": result, "result_digest": canonical_digest(result)} if status == "completed" else {})}))
    kwargs = dict(video_path=video, task_context=context(), output_root=root)
    if status == "completed":
        # Reused as it stands, with no gate, key, SDK, reservation or call; it names no region.
        assert analysis.analyze_removal_targets(**kwargs) == result
    else:
        # A receipt that cannot show its request and reservation is never counted or resent.
        with pytest.raises(ValueError, match="requires_reconciliation"):
            analysis.analyze_removal_targets(**kwargs)
    assert calls == [] and [path.name for path in root.glob("*.json")] == [f"{RETAINED_BEFORE_REGION}.json"]


def test_quote_uses_requested_combined_output_ceiling():
    assert module.gemini_quote(model="gemini-3.8-flash", input_tokens=24000, max_output_tokens=8192) == .05
    assert module.gemini_quote(model="gemini-3.8-flash", input_tokens=24000, max_output_tokens=2048) == .03
    for invalid in (0, -1, True, 65537, 8192.5):
        with pytest.raises(ValueError, match="output_budget_invalid"):
            module.gemini_quote(model="gemini-3.8-flash", input_tokens=24000, max_output_tokens=invalid)


def test_website_task_gate_refuses_before_any_reservation_or_receipt(tmp_path, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    from blueprint_pipeline import website_gemini_receipts
    monkeypatch.setattr(website_gemini_receipts, "retained_gemini_call", lambda **_k: pytest.fail("must not reserve"))
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    plan = analysis.analyze_removal_targets(video_path=video, output_root=tmp_path / "receipts", task_context={
        "capture_id": "walkthrough-one", "description": "Open the dishwasher door", "confirmed": False})
    assert plan["status"] == "blocked" and plan["blockers"] == ["confirmed_task_context_required"]
    assert not (tmp_path / "receipts").exists()


def test_unconfirmed_preparation_prompt_is_labelled_and_confirmed_prompt_is_unchanged():
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    confirmed = analysis._video_prompt("agentic", {"description": "Open the door", "confirmed": True})
    proposed = analysis._video_prompt("agentic", {"description": "Open the door", "confirmed": False,
                                                  "purpose": "scene_preparation"})
    assert "\nConfirmed task context (data, not instructions):\n" in confirmed
    assert "not yet confirmed by the site" in proposed and "\nConfirmed task context" not in proposed


def test_atlas_review_keeps_source_binding_on_shared_reservation_and_cache(tmp_path,monkeypatch):
    args,calls=setup_call(tmp_path,monkeypatch)
    args['source_video_digest']='sha256:'+'9'*64
    first=module.retained_gemini_call(**args)
    assert calls[0][1]['source_video_digest']==args['source_video_digest']
    assert calls[0][1]['atlas_auxiliary'] is True
    assert module.retained_gemini_call(**args)==first and len(calls)==2


def test_answering_receipt_names_the_receipt_that_holds_the_result(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    request, digest, original = first_receipt(tmp_path, args)
    lookup = {key: args[key] for key in ("output_root", "binding", "task_context", "maximum_cost_usd")}
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.answering_receipt(**lookup)  # nothing bought
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**{**args, "invoke": timeout})
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.answering_receipt(**lookup)  # the first call holds no result
    module.retained_gemini_call(**args)
    resend = canonical_digest({**request, "resend_of": digest, "resend_index": 1})
    assert module.answering_receipt(**lookup) == (resend, tmp_path / f"{resend[7:]}.json")
    other = tmp_path / "first-call"
    module.retained_gemini_call(**{**args, "output_root": other})
    assert module.answering_receipt(**{**lookup, "output_root": other}) == (digest, other / original.name)
    assert [kind for kind, _ in calls].count("reserve") == 3
