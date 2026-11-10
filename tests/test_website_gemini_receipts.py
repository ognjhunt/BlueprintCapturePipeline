"""One paid attempt per source/task; no live model calls."""
import fcntl
import json
import time

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
                     "allocation_binding_digest": kwargs["binding_digest"], "blockers": [],
                     "provider": kwargs["provider"], "expires_at_epoch": time.time() + 600}
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


def test_a_killed_call_is_counted_spent_and_sent_again_once_per_new_reservation(tmp_path, monkeypatch):
    """No proof of $0: the first hold stays (counted at its maximum) and one new reservation pays the resend."""
    args, calls = setup_call(tmp_path, monkeypatch)
    answer = args["invoke"]
    def timeout():
        calls.append(("invoke", None))
        raise TimeoutError("provider may have charged")
    args["invoke"] = timeout
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**args)
    first = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert first["status"] == "submitting"
    # The resend is killed too: it spent only its own new reservation.
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke", "reserve", "invoke"]
    digests = [row[1]["binding_digest"] for row in calls if row[0] == "reserve"]
    assert digests[0] == first["request_digest"] and len(set(digests)) == 2
    counted = json.loads(next(tmp_path.glob("*.unanswered-1.json")).read_text())
    assert counted["status"] == "counted_as_spent" and counted["dispatch_digest"] == digests[0]
    assert counted["reservation"] == "held_at_maximum_never_released_or_settled_as_zero"
    # A third send needs a third admission; when the ceiling refuses it, nothing is dispatched.
    def refused(**kwargs):
        calls.append(("reserve", kwargs))
        raise ValueError("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")
    monkeypatch.setattr(module, "reserve_website_preparation_spend", refused)
    with pytest.raises(ValueError, match="budget_exhausted"):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls].count("invoke") == 2
    # Once admitted again, the answer is bought once and then reused.
    setup_call(tmp_path, monkeypatch)
    args["invoke"] = answer
    result = module.retained_gemini_call(**args)
    assert module.retained_gemini_call(**args) == result
    resend = json.loads(next(tmp_path.glob("*.resend-2.json")).read_text())
    assert resend["status"] == "completed" and resend["resend_index"] == 2


def test_an_unanswered_call_with_no_reservation_on_record_is_a_money_wait(tmp_path, monkeypatch):
    from blueprint_pipeline.website_preparation_retry import classify
    args, calls = setup_call(tmp_path, monkeypatch)
    digest = module._request(binding=args["binding"], task_context=args["task_context"],
                             maximum_cost_usd=args["maximum_cost_usd"], source_video_digest=None)[1]
    (tmp_path / f"{digest[7:]}.json").write_text(json.dumps({"status": "submitting", "request_digest": digest}))
    with pytest.raises(ValueError, match=module.UNRESERVED) as refused:
        module.retained_gemini_call(**args)
    assert calls == [] and classify(str(refused.value)) == "money"


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
    # Stopped before the dispatch: proof of $0, so the same reservation pays the next one.
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["status"] == "not_dispatched"
    module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    assert not list(tmp_path.glob("*.unanswered-*.json")) and not list(tmp_path.glob("*.resend-*.json"))


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
         **({"result": result, "result_digest": canonical_digest(result)} if status == "completed" else
            {"admission": {"status": "admitted", "allocation_binding_digest": "sha256:" + RETAINED_BEFORE_REGION}})}))
    kwargs = dict(video_path=video, task_context=context(), output_root=root)
    if status == "completed":
        # Reused as it stands, with no gate, key, SDK, reservation or call; it names no region.
        assert analysis.analyze_removal_targets(**kwargs) == result
        assert calls == [] and [path.name for path in root.glob("*.json")] == [f"{RETAINED_BEFORE_REGION}.json"]
    else:
        # A call that may have been charged is counted as spent; the current prompt is then bought
        # under its own new reservation, which first needs the runtime (not configured here).
        with pytest.raises(ValueError, match="runtime_not_configured"):
            analysis.analyze_removal_targets(**kwargs)
        assert calls == [] and (root / f"{RETAINED_BEFORE_REGION}.unanswered-1.json").is_file()


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


def _stub_genai(monkeypatch, analysis):
    import sys
    from types import SimpleNamespace

    import google
    monkeypatch.setenv(analysis.GATE_ENV, "true")
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setattr(google, "genai", SimpleNamespace(types=SimpleNamespace()), raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", google.genai)


def test_a_gemini_error_is_raised_never_retained_as_a_completed_blocked_plan(tmp_path, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    from blueprint_pipeline.website_preparation_retry import classify
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    _stub_genai(monkeypatch, analysis)

    class Unavailable(Exception):
        code = 503

    def failing(**_):
        raise Unavailable("upstream connect error")
    monkeypatch.setattr(analysis, "_invoke_agentic_video", failing)
    kwargs = dict(video_path=video, task_context=context(), output_root=tmp_path / "receipts")
    with pytest.raises(analysis.GeminiProviderError) as error:
        analysis.analyze_removal_targets(**kwargs)
    assert str(error.value) == "website_gemini_provider_error:gemini_clean_plate_provider_error:Unavailable:http_503"
    assert classify(str(error.value)) == "outage"
    # A 503 answer did no work: the intent is refused, never a completed plan.
    receipt, = (tmp_path / "receipts").glob("*.json")
    assert (json.loads(receipt.read_text())["status"], json.loads(receipt.read_text())["http_status"]) == ("refused", 503)
    # Diagnostics without a capture keep the blocked plan, as before.
    plan = analysis.analyze_removal_targets(video_path=video, task_context=None)
    assert plan["status"] == "blocked"


def test_a_blocked_plan_an_older_release_retained_is_still_read_and_not_bought_again(tmp_path, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    _stub_genai(monkeypatch, analysis)
    blocked = analysis.empty_removal_plan(status="blocked", model="gemini-3.8-flash", processing="static",
                                          blockers=["gemini_clean_plate_provider_error"], video_path=video,
                                          video_digest=analysis.sha256_file(video))
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_: blocked)
    kwargs = dict(video_path=video, task_context=context(), output_root=tmp_path / "receipts")
    assert analysis.analyze_removal_targets(**kwargs) == blocked  # what an older release stored
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_: pytest.fail("bought again"))
    assert analysis.analyze_removal_targets(**kwargs) == blocked
    assert [row[0] for row in calls] == ["reserve"]


def test_the_resend_chain_ends_after_three_unanswered_dispatches(tmp_path, monkeypatch):
    from blueprint_pipeline.website_preparation_retry import classify
    args, calls = setup_call(tmp_path, monkeypatch)
    def timeout():
        calls.append(("invoke", None))
        raise TimeoutError("no answer")
    args["invoke"] = timeout
    for _ in range(3):
        with pytest.raises(TimeoutError):
            module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke"] * 3
    # The 4th run sends nothing and reserves nothing; the refusal is not an outage, so the timed schedule ends it.
    with pytest.raises(ValueError, match=module.EXHAUSTED) as exhausted:
        module.retained_gemini_call(**args)
    assert len(calls) == 6 and classify(str(exhausted.value)) == "refusal"


@pytest.mark.parametrize("status", [429, 503])
def test_a_refusal_that_proves_no_work_reuses_the_same_reservation(tmp_path, monkeypatch, status):
    from blueprint_pipeline.website_preparation_retry import classify
    args, calls = setup_call(tmp_path, monkeypatch)
    answer = args["invoke"]

    class Refused(Exception):
        code = status
    def refuses():
        calls.append(("invoke", None))
        raise Refused("resource exhausted")
    args["invoke"] = refuses
    for _ in range(5):  # more than the unanswered bound: a quota outage never ends the request
        with pytest.raises(Refused) as refused:
            module.retained_gemini_call(**args)
    assert classify(f"{type(refused.value).__name__}: http_{status}") == "outage"
    args["invoke"] = answer
    module.retained_gemini_call(**args)
    # One hold, one dispatch digest, never counted as spent.
    assert [row[0] for row in calls] == ["reserve"] + ["invoke"] * 6
    assert not list(tmp_path.glob("*.unanswered-*.json")) and not list(tmp_path.glob("*.resend-*.json"))


def test_a_500_answer_is_still_counted_as_spent(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)

    class ServerError(Exception):
        code = 500
    def fails():
        raise ServerError("internal")
    args["invoke"] = fails
    with pytest.raises(ServerError):
        module.retained_gemini_call(**args)
    with pytest.raises(ServerError):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "reserve"] and list(tmp_path.glob("*.unanswered-1.json"))


def _refused_once(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    answer = args["invoke"]

    class Refused(Exception):
        code = 429
    def refuses():
        raise Refused("quota")
    args["invoke"] = refuses
    with pytest.raises(Refused):
        module.retained_gemini_call(**args)
    args["invoke"] = answer
    intent, = tmp_path.glob("*.json")
    return args, calls, intent


def test_an_expired_retained_reservation_is_replaced_by_one_new_one(tmp_path, monkeypatch):
    args, calls, intent = _refused_once(tmp_path, monkeypatch)
    stored = json.loads(intent.read_text())
    stored["admission"]["expires_at_epoch"] = time.time() - 1  # a quota outage outlasted the grant's window
    intent.write_text(json.dumps(stored))
    result = module.retained_gemini_call(**args)
    assert module.retained_gemini_call(**args) == result
    reserved = [row[1]["binding_digest"] for row in calls if row[0] == "reserve"]
    assert len(reserved) == len(set(reserved)) == 2 and [row[0] for row in calls].count("invoke") == 1
    # The old hold is kept as superseded, never released; nothing is counted as unanswered.
    assert json.loads(intent.read_text())["status"] == "superseded"
    assert not list(tmp_path.glob("*.unanswered-*.json"))
    replacement, = tmp_path.glob("*.replacement-*.json")
    assert json.loads(replacement.read_text())["status"] == "completed"


def test_a_budget_refusal_of_a_retained_reservation_is_a_money_wait(tmp_path, monkeypatch):
    from blueprint_pipeline import paid_resource_admission as admission
    args, calls, intent = _refused_once(tmp_path, monkeypatch)
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", lambda: ["account_monthly_spend_limit_reached"])
    with pytest.raises(PaidResourceAdmissionBlocked, match="account_monthly_spend_limit_reached"):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve"]
    assert json.loads(intent.read_text())["status"] == "refused"
    assert not list(tmp_path.glob("*.unanswered-*.json")) and not list(tmp_path.glob("*.replacement-*.json"))
