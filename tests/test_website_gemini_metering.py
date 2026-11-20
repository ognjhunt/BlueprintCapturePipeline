"""Every Gemini call a website job makes counts against that job: reserved first, settled at its usage, metered.

No live model calls: answers are stand-ins with the provider's usage fields.
"""
import json
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import paid_spend_meter as meter
from blueprint_pipeline import website_gemini_receipts as module
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission,
)

NOW = time.time()


def context():
    value = {"schema_version": "website_site_task_context.v1", "request_id": "capture-job1",
             "scene_id": "site-capture-job1", "capture_id": "walkthrough-job1", "confirmed": True,
             "confirmed_at": "2026-11-19T00:00:00Z", "description": "Move the box"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def answer(prompt=1000, visible=200, thoughts=50, total=None):
    usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=visible,
                            thoughts_token_count=thoughts, tool_use_prompt_token_count=None,
                            total_token_count=total if total is not None else prompt + visible + thoughts)
    return SimpleNamespace(usage_metadata=usage)


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setenv(meter.METER_ROOT_ENV, str(tmp_path / "meter"))
    calls = []

    def reserve(**kwargs):
        calls.append(("reserve", kwargs))
        admission = {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "status": "admitted", "blockers": [],
                     "resource_class": kwargs["resource_class"], "provider": kwargs["provider"],
                     "allocation_binding_digest": kwargs["binding_digest"],
                     "maximum_cost_usd": kwargs["maximum_cost_usd"], "expires_at_epoch": time.time() + 600}
        grant = require_paid_resource_admission(
            website_task_context.metered_admission(admission, job_id=kwargs["task_context"]["request_id"],
                                                   provider=kwargs["provider"]),
            resource_class=kwargs["resource_class"], expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        return admission, grant

    settled = []

    def settle(*, task_context, operation, payload):
        settled.append(payload)
        command = payload["settlement"]
        return {**command, "actual_cost_usd": command["provider_charge_amount_usd"], "status": "settled"}

    monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
    monkeypatch.setattr(website_task_context, "preparation_webapp_request", settle)
    return SimpleNamespace(root=tmp_path / "receipts", calls=calls, settled=settled,
                           meter_root=tmp_path / "meter")


def call(job, invoke, maximum=0.5):
    return module.retained_gemini_call(output_root=job.root, binding={"model": "gemini-3.8-flash", "prompt": "look"},
                                       task_context=context(), maximum_cost_usd=maximum, preflight=lambda: None,
                                       invoke=invoke)


def view(job):
    events, gaps = meter.read_journal(job.meter_root)
    assert gaps == []
    return meter.meter_view(meter.journal_rows(events), now=time.time())


def test_an_answered_call_settles_on_the_job_ledger_at_its_usage_and_is_metered_against_the_job(job):
    def invoke():
        module.note_gemini_usage(answer(prompt=1000, visible=200, thoughts=50))
        module.note_gemini_usage(answer(prompt=300, visible=100, thoughts=0))  # a repair turn of the same call
        return {"seen": True}

    assert call(job, invoke) == {"seen": True}
    # (1300 input x $0.75 + 350 output x $3.75) per million tokens.
    cost = (1300 * 0.75 + 350 * 3.75) / 1_000_000
    [payload] = job.settled
    assert payload["request_id"] == "capture-job1" and payload["settlement"]["provider"] == "google"
    assert payload["settlement"]["completed_request_count"] == 1
    assert payload["settlement"]["provider_charge_amount_usd"] == pytest.approx(cost, abs=1e-6)
    receipt = json.loads(next(job.root.glob("*[0-9a-f].json")).read_text())
    assert receipt["status"] == "completed" and receipt["usage"]["cost_usd"] == pytest.approx(cost, abs=1e-6)
    assert payload["settlement"]["allocation_binding_digest"] == receipt["request_digest"]
    assert json.loads(next(job.root.glob("*.settlement.json")).read_text())["status"] == "settled"
    # The job's meter: one call, under the bill Gemini lands on, at what it cost; never "unattributed".
    [row] = view(job)["jobs"]
    assert row["job_id"] == "capture-job1" and row["admissions"] == 1 and row["unpriced_admissions"] == 0
    assert row["by_provider_settled_usd"] == {"gcp": pytest.approx(cost, abs=1e-6)}
    # A replay buys, settles and meters nothing again.
    assert call(job, lambda: pytest.fail("bought twice")) == {"seen": True}
    assert len(job.settled) == 1 and len(view(job)["jobs"]) == 1


def test_a_call_whose_usage_is_not_reported_keeps_its_quote_on_the_ledger_and_is_metered_at_it(job):
    def invoke():
        module.note_gemini_usage(SimpleNamespace())  # no usage fields at all
        return {"seen": True}

    call(job, invoke, maximum=0.37)
    assert job.settled == []
    assert not list(job.root.glob("*.settlement.json"))
    [row] = view(job)["jobs"]
    assert row["by_provider_settled_usd"] == {"gcp": 0.37}


def test_a_webapp_that_refuses_the_settlement_leaves_the_hold_and_the_answer_stands(job, monkeypatch):
    def refuse(**_kwargs):
        raise ValueError("website_control_preparation-settlement_http_409:website_scene_preparation_settlement_invalid")

    monkeypatch.setattr(website_task_context, "preparation_webapp_request", refuse)

    def invoke():
        module.note_gemini_usage(answer())
        return {"seen": True}

    assert call(job, invoke) == {"seen": True}
    assert not list(job.root.glob("*.settlement.json"))
    # The meter still records what the call cost.
    [row] = view(job)["jobs"]
    assert row["by_provider_settled_usd"]["gcp"] == pytest.approx((1000 * 0.75 + 250 * 3.75) / 1e6, abs=1e-6)


def test_a_dispatch_with_no_answer_is_metered_at_its_quote_when_counted_as_spent(job):
    def cut_off():
        raise TimeoutError("provider may have charged")

    with pytest.raises(TimeoutError):
        call(job, cut_off, maximum=0.4)
    # Counted when the next run walks the chain; the resend is bought under its own reservation and answered.
    def invoke():
        module.note_gemini_usage(answer(prompt=100, visible=10, thoughts=0))
        return {"seen": True}

    call(job, invoke, maximum=0.4)
    [row] = view(job)["jobs"]
    assert row["admissions"] == 2 and job.settled and len(job.settled) == 1
    assert row["by_provider_settled_usd"]["gcp"] == pytest.approx(0.4 + (100 * 0.75 + 10 * 3.75) / 1e6, abs=1e-6)


def test_usage_reads_both_answer_shapes_and_unknown_is_never_priced():
    assert module.gemini_response_usage(answer(prompt=10, visible=5, thoughts=2)) == {
        "status": "reported", "input_tokens": 10, "output_tokens": 7}
    # A total above the parts is billed output, never dropped.
    assert module.gemini_response_usage(answer(prompt=10, visible=5, thoughts=None, total=40))["output_tokens"] == 30
    interaction = SimpleNamespace(usage=SimpleNamespace(total_input_tokens=50, total_output_tokens=9, total_tokens=70))
    assert module.gemini_response_usage(interaction) == {"status": "reported", "input_tokens": 50, "output_tokens": 20}
    assert module.gemini_response_usage(SimpleNamespace(usage_metadata=SimpleNamespace()))["status"] == "unknown"
    no_total_or_thinking = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5))
    assert module.gemini_response_usage(no_total_or_thinking)["status"] == "unknown"
    assert module.gemini_usage_cost(model="gemini-3.8-flash", calls=[{"status": "unknown"}]) is None
    assert module.gemini_usage_cost(model="gemini-9", calls=[{"status": "reported", "input_tokens": 1,
                                                              "output_tokens": 1}]) is None
    # Outside a retained call nothing is collected.
    module.note_gemini_usage(answer())


def test_preparation_admissions_name_their_job_without_a_price():
    admission = {"allocation_binding_digest": "sha256:" + "a" * 64, "status": "admitted"}
    metered = website_task_context.metered_admission(admission, job_id="capture-job1", provider="google")
    assert metered["meter"] == {"job_id": "capture-job1", "provider": "gcp"} and "held_usd" not in metered["meter"]
    assert "meter" not in admission
    assert website_task_context.metered_admission(admission, job_id=None, provider="openai") == admission


def test_an_openai_preparation_settlement_is_metered_against_its_job(tmp_path, monkeypatch):
    monkeypatch.setenv(meter.METER_ROOT_ENV, str(tmp_path))
    binding = "sha256:" + "b" * 64
    website_task_context.meter_preparation_settlement(
        {"request_id": "capture-job2", "settlement": {"provider": "openai"}},
        {"status": "settled", "provider": "openai", "allocation_binding_digest": binding, "actual_cost_usd": 0.53})
    # Another provider, or a refusal, records nothing here.
    website_task_context.meter_preparation_settlement(
        {"request_id": "capture-job2"}, {"status": "settled", "provider": "world_labs_atlas",
                                         "allocation_binding_digest": binding, "actual_cost_usd": 5})
    events, _ = meter.read_journal(tmp_path)
    [row] = meter.journal_rows(events)
    assert (row["job_id"], row["meter_key"], row["provider"], row["status"]) == (
        "capture-job2", binding, "openai", "settled")
    assert float(row["settled_usd"]) == 0.53
