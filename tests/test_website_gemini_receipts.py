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
    # The website's task-context read, which a held reservation's reuse asks before any dispatch.
    from blueprint_pipeline import website_task_context
    monkeypatch.setattr(website_task_context, "load_current_website_task_context",
                        lambda **kw: calls.append(("source_read", kw)) or context())
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
    # The held reservation is reused only after the capture's current source is read.
    assert [row[0] for row in calls] == ["reserve", "source_read", "invoke"]
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
def test_an_analysis_bought_before_the_region_prompt_is_bought_once_more(tmp_path, monkeypatch, status):
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
        # It was never asked for the place the task names (owner direction 2026-11-11): the current
        # revision is bought once, under its own reservation, which first needs the runtime (not
        # configured here). The earlier receipt is kept as it is.
        with pytest.raises(ValueError, match="runtime_not_configured"):
            analysis.analyze_removal_targets(**kwargs)
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
    # One hold, one dispatch digest, never counted as spent; each reuse reads the current source first.
    assert [row[0] for row in calls] == ["reserve", "invoke"] + ["source_read", "invoke"] * 5
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
    assert [row[0] for row in calls] == ["reserve", "source_read"]
    assert json.loads(intent.read_text())["status"] == "refused"
    assert not list(tmp_path.glob("*.unanswered-*.json")) and not list(tmp_path.glob("*.replacement-*.json"))


def test_a_withdrawn_source_stops_a_reused_reservation_before_anything_is_sent(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context
    args, calls = setup_call(tmp_path, monkeypatch)

    class Refused(Exception):
        code = 429
    def refuses():
        calls.append(("invoke", None))
        raise Refused("resource exhausted")
    args["invoke"] = refuses
    with pytest.raises(Refused):
        module.retained_gemini_call(**args)
    def withdrawn(**_kw):
        raise ValueError("website_control_task-context_http_409:source_revoked")
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", withdrawn)
    args["invoke"] = lambda: pytest.fail("nothing may be sent once the site has withdrawn")
    with pytest.raises(ValueError, match="source_revoked"):
        module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke"]


def test_an_older_answer_is_bought_again_exactly_once_across_retries(tmp_path, monkeypatch):
    """Review 2026-11-12: a completed answer from before the destination region is re-bought once, under one
    reservation; every later run reuses that purchase, whatever it said."""
    import sys
    import types
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    from blueprint_pipeline.common import sha256_file
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    monkeypatch.setattr(analysis, "_truthy", lambda value: True)
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setitem(sys.modules, "google.genai", types.ModuleType("google.genai"))
    invokes = []

    def answer(**_kwargs):
        invokes.append(1)
        return {"status": "blocked", "targets": [], "blockers": ["task_target_requires_clarification:x"],
                "input_video_sha256": sha256_file(video), "prompt_revision": analysis.PROMPT_REVISION}
    monkeypatch.setattr(analysis, "_analyze_removal_targets", answer)
    root = tmp_path / "receipts"
    root.mkdir()
    old = {"status": "completed", "targets": ["box"]}
    (root / f"{RETAINED_BEFORE_REGION}.json").write_text(json.dumps({
        "status": "completed", "request_digest": "sha256:" + RETAINED_BEFORE_REGION, "result": old,
        "result_digest": canonical_digest(old)}))
    kwargs = dict(video_path=video, task_context=context(), output_root=root)
    results = [analysis.analyze_removal_targets(**kwargs) for _ in range(3)]
    assert len(invokes) == 1 and [row[0] for row in calls] == ["reserve"]
    assert results[0] == results[1] == results[2]


def test_a_blocked_region_answer_is_bought_once_more_and_never_on_each_retry(tmp_path, monkeypatch):
    """Review 2026-11-12: a region-revision answer that ended blocked is bought once more under the current
    revision, then that purchase decides every later run: six timed retries, and again on a second release."""
    import sys
    import types
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    from blueprint_pipeline.common import sha256_file
    _args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda *a: ("static", 13.525))
    monkeypatch.setattr(analysis, "_truthy", lambda value: True)
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setitem(sys.modules, "google.genai", types.ModuleType("google.genai"))
    invokes = []

    def answer(**_kwargs):
        invokes.append(1)
        return {"status": "blocked", "targets": [], "blockers": ["task_target_requires_clarification:x"],
                "input_video_sha256": sha256_file(video)}
    monkeypatch.setattr(analysis, "_analyze_removal_targets", answer)
    root = tmp_path / "receipts"
    kwargs = dict(video_path=video, task_context=context(), output_root=root)
    # The region revision's own blocked answer, bought by an earlier release.
    with monkeypatch.context() as earlier:
        earlier.setattr(analysis, "PROMPT_INSTRUCTION", analysis.REGION_PROMPT_INSTRUCTION)
        earlier.setattr(analysis, "RESPONSE_SCHEMA", analysis.REGION_RESPONSE_SCHEMA)
        analysis.analyze_removal_targets(**kwargs)
    assert len(invokes) == 1
    for release in ("a" * 40, "b" * 40):
        monkeypatch.setattr("blueprint_pipeline.team_scene_packet_worker.current_release_commit", lambda r=release: r)
        for _ in range(6):  # the six timed retries of a refusal, on each release
            analysis.analyze_removal_targets(**kwargs)
    # Bought once more under the current revision, then reused: two purchases in all, never one per retry.
    assert len(invokes) == 2 and [row[0] for row in calls].count("reserve") == 2


# --- 2026-11-17, test job 3: Gemini answered 402 "Your prepayment credits are depleted" -----------------------------

def _depleted(message="Your prepayment credits are depleted. Please go to AI Studio to manage your billing.",
              status="RESOURCE_EXHAUSTED", code=402):
    from tests.test_website_scene_handoff import GeminiClientError
    return GeminiClientError(code, status, message)


def test_a_credits_depleted_402_is_a_money_wait_on_the_same_hold_and_never_counted(tmp_path, monkeypatch):
    """The provider did no work: the receipt keeps its answer, the same reservation pays the next dispatch, and no
    number of such answers spends the unanswered bound. Each is a money wall the retry lane reads as money; once
    the account has funds the call completes under that first hold."""
    from blueprint_pipeline.hard_stops import is_wall
    from blueprint_pipeline.job_retry_policy import outage_retry_delay
    from blueprint_pipeline.website_preparation_retry import classify
    args, calls = setup_call(tmp_path, monkeypatch)
    answer = args["invoke"]

    def depleted():
        calls.append(("invoke", None))
        raise _depleted()
    args["invoke"] = depleted
    for _ in range(5):  # more than the unanswered bound
        with pytest.raises(module.CreditDepleted) as refused:
            module.retained_gemini_call(**args)
        assert is_wall(refused.value) and str(refused.value) == module.CREDIT_DEPLETED
    assert classify("website_scene_preparation: " + module.CREDIT_DEPLETED) == "money"
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert (receipt["status"], receipt["http_status"], receipt["reason"]) == ("refused", 402, module.CREDIT_DEPLETED)
    args["invoke"] = answer
    assert module.retained_gemini_call(**args) == {"status": "completed", "targets": ["box"]}
    assert [row[0] for row in calls] == ["reserve", "invoke"] + ["source_read", "invoke"] * 5
    assert not list(tmp_path.glob("*.unanswered-*.json")) and not list(tmp_path.glob("*.resend-*.json"))
    assert module.MONEY_RECHECK_SECONDS == outage_retry_delay(64)  # the money wait's longest interval


@pytest.mark.parametrize("error", [
    _depleted(message="Billing account disabled."), _depleted(status="PERMISSION_DENIED"), _depleted(code=403),
    type("Bare402", (Exception,), {"code": 402})("prepayment credits are depleted")])
def test_no_other_402_or_body_is_read_as_credits_depleted(tmp_path, monkeypatch, error):
    """Status 402 and the provider's own reason, both: any other answer is counted as spent, as before."""
    args, calls = setup_call(tmp_path, monkeypatch)
    args["invoke"] = lambda: (_ for _ in ()).throw(error)
    for _ in range(2):
        with pytest.raises(type(error)):
            module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "reserve"] and list(tmp_path.glob("*.unanswered-1.json"))
    assert not module.credit_depleted(error)


def _old_release_chain(tmp_path, monkeypatch, answers, *, ledger_error):
    """Dispatches of one request answered in turn by ``answers`` (an exception is raised, a dict returned) on the
    release before this change, which kept no 402 on the receipt; then a job ledger whose one failed attempt,
    spanning them all, ended on ``ledger_error``. Returns the call arguments and calls on this release."""
    from datetime import datetime, timezone
    root = tmp_path / "capture/pipeline/website_scene_preparation/assembly_coverage/drawer/receipts"
    started = datetime.now(timezone.utc)
    with pytest.MonkeyPatch.context() as release:
        args, _calls = setup_call(root, release)
        release.setattr(module, "credit_depleted", lambda _exc: False)  # that release
        for answer in answers:
            args["invoke"] = lambda answer=answer: answer if isinstance(answer, dict) else (_ for _ in ()).throw(answer)
            try:
                module.retained_gemini_call(**args)
            except Exception:  # noqa: BLE001 - each run ends on its answer, as on the host
                pass
    (tmp_path / "capture/pipeline_job_ledger.json").write_text(json.dumps({"attempt_history": [
        {"attempt_number": 7, "status": "failed_retryable", "error": ledger_error,
         "started_at": started.isoformat(), "failed_at": datetime.now(timezone.utc).isoformat()}]}))
    args, calls = setup_call(root, monkeypatch)
    return args, calls, root


def test_job3_s_old_402_chain_stays_counted_and_never_reaches_an_unreserved_resend(tmp_path, monkeypatch):
    """Test job 3's records: three 402s the release before kept as no answer (original, resend-1, resend-2). The
    ledger names no dispatch, so nothing is read as refused from it: the chain stays counted, the 4th run sends
    and reserves nothing and ends on the unanswered bound, never ``..._spend_unreserved`` (review probe a)."""
    args, calls, root = _old_release_chain(tmp_path, monkeypatch, [_depleted()] * 3 + [None],
                                           ledger_error=str(_depleted()))
    args["invoke"] = lambda: pytest.fail("sent again")
    for _ in range(2):
        with pytest.raises(ValueError, match=module.EXHAUSTED):
            module.retained_gemini_call(**args)
    assert calls == []
    assert sorted(json.loads(path.read_text())["status"] for path in root.glob("*.json")
                  if ".unanswered-" not in path.name) == ["submitting"] * 3


def test_a_request_whose_resend_was_answered_is_never_bought_again(tmp_path, monkeypatch):
    """A first look left unanswered, its resend answered, then another call's 402 ended the attempt: the request's
    latest record is the answer, returned as it stands; nothing is reserved or sent (review probe b)."""
    answered = {"status": "completed", "targets": ["box"]}
    args, calls, _root = _old_release_chain(tmp_path, monkeypatch, [TimeoutError("unusable look"), answered],
                                            ledger_error=str(_depleted()))
    args["invoke"] = lambda: pytest.fail("bought again")
    assert module.retained_gemini_call(**args) == answered and calls == []


def test_a_resend_after_a_top_up_that_times_out_is_an_ordinary_non_answer(tmp_path, monkeypatch):
    """A 402 kept on the receipt; after a top-up its hold pays the resend, which times out: the next run counts it
    as unanswered (not money) and sends once more under one new reservation; at the bound the chain ends on the
    unanswered code, never a money one. With a 402 between two timeouts, none of it is unreserved (probe 2)."""
    from blueprint_pipeline.website_preparation_retry import classify
    args, calls = setup_call(tmp_path, monkeypatch)
    answers = iter([_depleted(), TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3")])

    def invoke():
        calls.append(("invoke", None))
        raise next(answers)
    args["invoke"] = invoke
    with pytest.raises(module.CreditDepleted):
        module.retained_gemini_call(**args)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            module.retained_gemini_call(**args)
    with pytest.raises(ValueError, match=module.EXHAUSTED) as ended:
        module.retained_gemini_call(**args)
    assert classify("website_scene_preparation: " + str(ended.value)) == "refusal"
    assert [row[0] for row in calls] == ["reserve", "invoke", "source_read", "invoke"] + ["reserve", "invoke"] * 2
    args2, calls2 = setup_call(tmp_path / "probe2", monkeypatch)
    answers2 = iter([TimeoutError("t1"), _depleted(), TimeoutError("t2"), None])

    def invoke2():
        calls2.append(("invoke", None))
        answer = next(answers2)
        if answer is None:
            return {"status": "completed", "targets": ["box"]}
        raise answer
    args2["invoke"] = invoke2
    for expected in (TimeoutError, module.CreditDepleted, TimeoutError):
        with pytest.raises(expected):
            module.retained_gemini_call(**args2)
    assert module.retained_gemini_call(**args2) == {"status": "completed", "targets": ["box"]}
    assert [row[0] for row in calls2].count("reserve") == 3  # the original, its resend, the resend's resend


# --- Review (Codex P1): a later turn of one dispatch failing after an answered turn is never "refused" -------------

def _repair_turn_fails(calls, error):
    """The assembly views' invoke as it runs (``website_assembly_coverage._json_answer``): the first turn answered
    with text that does not parse, then the JSON repair turn fails with ``error``."""
    from blueprint_pipeline.website_assembly_coverage import _json_answer

    def invoke():
        calls.append(("invoke", None))
        return {"classification": _json_answer("{not json", lambda _error: (_ for _ in ()).throw(error))[0]}
    return invoke


def test_a_credits_depleted_402_on_the_repair_turn_is_spent_waits_on_money_and_then_takes_a_new_reservation(
        tmp_path, monkeypatch):
    """The first turn was answered (billed): the dispatch keeps no answer, so it is counted as spent, while the job
    still waits on money. After a top-up the resend is bought under one new reservation; the hold that paid the
    answered turn never pays again."""
    from blueprint_pipeline.hard_stops import is_wall
    args, calls = setup_call(tmp_path, monkeypatch)
    args["invoke"] = _repair_turn_fails(calls, _depleted())
    with pytest.raises(module.CreditDepleted) as waiting:
        module.retained_gemini_call(**args)
    assert is_wall(waiting.value) and module.answered_before(waiting.value)
    first = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert first["status"] == "submitting"  # no answer kept: never refused
    args["invoke"] = lambda: calls.append(("invoke", None)) or (_ for _ in ()).throw(_depleted())  # still out
    with pytest.raises(module.CreditDepleted):
        module.retained_gemini_call(**args)
    assert json.loads(next(tmp_path.glob("*.unanswered-1.json")).read_text())["status"] == "counted_as_spent"
    args["invoke"] = lambda: calls.append(("invoke", None)) or {"status": "completed", "targets": ["box"]}
    assert module.retained_gemini_call(**args) == {"status": "completed", "targets": ["box"]}
    assert [row[0] for row in calls] == ["reserve", "invoke", "reserve", "invoke", "source_read", "invoke"]
    reserved = [row[1]["binding_digest"] for row in calls if row[0] == "reserve"]
    assert reserved[0] == first["request_digest"] and len(set(reserved)) == 2


@pytest.mark.parametrize("status", [429, 503])
def test_a_429_or_503_on_the_repair_turn_is_spent_and_the_next_send_takes_a_new_reservation(
        tmp_path, monkeypatch, status):
    from blueprint_pipeline.website_preparation_retry import classify
    from tests.test_website_scene_handoff import GeminiClientError
    args, calls = setup_call(tmp_path, monkeypatch)
    error = GeminiClientError(status, "RESOURCE_EXHAUSTED" if status == 429 else "UNAVAILABLE", "try later")
    args["invoke"] = _repair_turn_fails(calls, error)
    with pytest.raises(module.LaterTurnFailed) as failed:
        module.retained_gemini_call(**args)
    assert failed.value.__cause__ is error and str(failed.value) == str(error)  # the retry lane reads the same text
    assert classify(f"website_scene_preparation: {failed.value}") == classify(f"website_scene_preparation: {error}")
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["status"] == "submitting"
    args["invoke"] = lambda: calls.append(("invoke", None)) or {"status": "completed", "targets": ["box"]}
    assert module.retained_gemini_call(**args) == {"status": "completed", "targets": ["box"]}
    assert [row[0] for row in calls] == ["reserve", "invoke", "reserve", "invoke"]
    assert list(tmp_path.glob("*.unanswered-1.json"))


def test_a_first_turn_402_still_reuses_its_hold(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    args["invoke"] = lambda: calls.append(("invoke", None)) or (_ for _ in ()).throw(_depleted())
    with pytest.raises(module.CreditDepleted) as waiting:
        module.retained_gemini_call(**args)
    assert not module.answered_before(waiting.value)
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["status"] == "refused"
    args["invoke"] = lambda: calls.append(("invoke", None)) or {"status": "completed", "targets": ["box"]}
    module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke", "source_read", "invoke"]


def test_the_clean_plate_s_repair_turn_and_its_clean_up_after_the_answer_are_marked(tmp_path, monkeypatch):
    """The two other requests after an answered turn: the clean plate's text repair turn, and deleting the uploaded
    video after the analysis answered. Each failure carries the marker; the provider error keeps its own name."""
    from tests.capture_audio_fakes import FakeMediaTools
    from tests.test_clean_plate_agentic_provider import NS, output
    from tests.test_website_scene_handoff import GeminiClientError
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    FakeMediaTools().install(monkeypatch)  # a hermetic video-only copy, as the provider tests make it
    path = tmp_path / "source.mov"
    path.write_bytes(b"source video")
    media = NS(name="files/test", uri="https://provider.test/file", state=NS(name="ACTIVE"))
    refused = GeminiClientError(429, "RESOURCE_EXHAUSTED", "try later")
    for answered in (True, False):
        def create(**_):
            if not answered:
                raise _depleted()
            return NS(status="completed", steps=[output('{"targets":[]}')])

        def delete(**_):
            raise refused
        genai = NS(Client=lambda **_: NS(interactions=NS(create=create),
                                         files=NS(upload=lambda **_: media, delete=delete)))
        with pytest.raises(Exception) as failed:
            analysis._invoke_agentic_video(api_key="fake", model=analysis.DEFAULT_MODEL, processing="static",
                                           video_path=path, genai=genai, types=NS(HttpOptions=NS, HttpRetryOptions=NS))
        assert module.answered_before(failed.value) is answered  # a clean-up before any answer stays as it was
    _gemini_sdk(monkeypatch)
    monkeypatch.setenv(analysis.GATE_ENV, "1")
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setattr(analysis, "_video_processing", lambda path, mode: ("static", 10.0))
    from tests.test_clean_plate_agentic_provider import _TASK, _door, _rack
    # An answer that breaks one rule (``test_a_mechanism_hint_on_a_static_target_is_repaired_in_one_turn``).
    monkeypatch.setattr(analysis, "_invoke_agentic_video", lambda **_: {
        "text": json.dumps({"targets": [_door(), _rack()]}), "video_processing": {"mode": "static"}})
    monkeypatch.setattr(analysis, "_invoke_text_repair", lambda **_: (_ for _ in ()).throw(_depleted()))
    with pytest.raises(analysis.GeminiProviderError) as failed:
        analysis._analyze_removal_targets(video_path=path, task_context=_TASK, raise_provider_errors=True)
    assert module.answered_before(failed.value) and module.credit_depleted(failed.value)
    assert str(failed.value).endswith(":GeminiClientError:http_402")


def _gemini_sdk(monkeypatch):
    """The SDK's modules, faked: the provider calls are, and the SDK need not be installed."""
    import sys
    import types as module_types
    google = sys.modules.get("google") or module_types.ModuleType("google")
    genai = module_types.ModuleType("google.genai")
    genai.types = module_types.ModuleType("google.genai.types")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
