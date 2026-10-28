"""A semantic clarification reuses the bounded Atlas call/settlement lifecycle."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import atlas_scene_planner as planner
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_operation_coverage import clarification_request
from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
from tests.test_website_task_semantics import semantics_fixture


def review_fixture(tmp_path, monkeypatch, *, answers=None, cutoff=False):
    from agents.exceptions import ModelBehaviorError
    from blueprint_pipeline import paid_resource_admission, website_task_context
    from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit

    context, source, parent_receipt = semantics_fixture()
    semantics = parent_receipt["proposal"]["task_semantics"]
    semantics["covers_entire_job"] = False
    semantics["unresolved_aspects"] = ["Synthetic CAD parameters remain unqualified.",
                                       "Native physics and owner thresholds remain unconfirmed."]
    parent_receipt["proposal_digest"] = canonical_digest(parent_receipt["proposal"])
    parent = task_semantics_from_receipt(receipt=parent_receipt, task_context=context,
                                         source_video_digest=source)
    request = clarification_request(parent, task_context=context, source_video_digest=source)
    proposal = {"operation_coverage": {
        "operative_job": semantics["operative_job"], "task_family": semantics["task_family"],
        "steps": deepcopy(semantics["steps"]), "operation_coverage_complete": True,
        "unresolved_operations": [], "preparation_requirements": list(semantics["unresolved_aspects"]),
    }}
    survey = {"source_video_digest": source, "heldout_pixels_included": False,
              "display_rotation_degrees": 0, "frames": []}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    survey_path = tmp_path / "survey.json"
    survey_path.write_text(json.dumps(survey))
    calls = {"reserved": [], "settled": [], "specs": [], "inputs": [], "restored": []}
    monkeypatch.setattr(paid_resource_admission, "require_paid_resource_admission_grant", lambda *_, **__: None)
    monkeypatch.setattr(InferenceReservationAudit, "record_completion", lambda self, value: None)
    monkeypatch.setattr(website_task_context, "reserve_website_preparation_spend",
                        lambda **kw: (calls["reserved"].append(kw) or {}, object()))
    monkeypatch.setattr(website_task_context, "website_webapp_request",
                        lambda **kw: calls["settled"].append(kw["payload"]["settlement"]) or {"status": "settled"})

    class Invoker:
        config = SimpleNamespace(max_inference_cost_usd=1)

        def configure_reservation_audit(self, *, record_completion, restored_reserved_cost_usd, **_):
            self.record = record_completion
            calls["restored"].append(restored_reserved_cost_usd)

        def invoke(self, spec, inputs):
            calls["specs"].append(spec)
            calls["inputs"].append(deepcopy(inputs))
            if cutoff:
                self.record({"status": "invalid_structured_output", "reconciled_actual_cost_usd": .02,
                             "usage": {"output_tokens": spec.max_output_tokens, "cost_is_actual": True}})
                raise ModelBehaviorError("Invalid JSON cut off at the bounded allowance")
            output = deepcopy((answers or [proposal])[min(len(calls["specs"]) - 1, len(answers or [proposal]) - 1)])
            cost = .02 if output != proposal else .03
            self.record({"status": "completed", "reconciled_actual_cost_usd": cost,
                         "usage": {"input_tokens": 100, "output_tokens": 100, "cost_is_actual": True}})
            return SimpleNamespace(output=output, cost_usd=cost,
                usage={"input_tokens": 100, "output_tokens": 100, "cost_is_actual": True},
                model=spec.model, sdk_version="hermetic-test")

    options = dict(survey_path=survey_path, task_context=context,
        configuration={"policy": {"model": "gpt-6.1-sol", "maximum_review_cost_usd": 1}},
        output_root=tmp_path / "reviews", invoker=Invoker(), release_blockers=lambda: ("c" * 40, []),
        recovery={"scope": "operation_coverage", "operation_coverage_request": request, "previews": []})
    return options, calls, proposal


def test_text_only_clarification_is_capped_and_completed_receipt_resettles_without_another_call(tmp_path, monkeypatch):
    options, calls, proposal = review_fixture(tmp_path, monkeypatch)
    receipt = planner.review_survey(**options)
    assert receipt["proposal"] == proposal
    assert receipt["binding"]["operation_coverage_request_digest"] == canonical_digest(
        options["recovery"]["operation_coverage_request"])
    assert receipt["binding"]["maximum_cost_usd"] == .25
    assert receipt["binding"]["web_search"]["maximum_calls"] == 0
    assert len(calls["specs"]) == 1 and calls["specs"][0].max_output_tokens == 4000
    assert not calls["specs"][0].hosted_tools
    assert all(part["type"] == "input_text" for message in calls["inputs"][0] for part in message["content"])
    assert calls["reserved"][0]["maximum_cost_usd"] == .25
    assert calls["reserved"][0]["source_video_digest"] == receipt["binding"]["source_video_digest"]
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [.03]

    settlement = options["output_root"] / (receipt["binding_digest"][7:] + ".settlement.json")
    settlement.unlink()  # A worker stopped after retaining completion but before retaining settlement.
    assert planner.review_survey(**options) == receipt
    assert planner.review_survey(**options) == receipt
    assert len(calls["specs"]) == len(calls["reserved"]) == 1
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [.03, .03]
    assert len({row["allocation_binding_digest"] for row in calls["settled"]}) == 1


@pytest.mark.parametrize("mistake", ["quote", "lost_issue"])
def test_semantic_content_failure_settles_and_gets_one_bounded_repair(tmp_path, monkeypatch, mistake):
    options, calls, proposal = review_fixture(tmp_path, monkeypatch)
    invalid = deepcopy(proposal)
    if mistake == "quote":
        invalid["operation_coverage"]["steps"][0]["task_quote"] = "Invented wording absent from the task"
    else:
        invalid["operation_coverage"]["preparation_requirements"] = []
    options, calls, _ = review_fixture(tmp_path, monkeypatch, answers=[invalid, proposal])
    receipt = planner.review_survey(**options)
    assert receipt["proposal"] == proposal and receipt["binding"]["validation_repair"]["attempt"] == 1
    assert [spec.max_output_tokens for spec in calls["specs"]] == [4000, 4000]
    assert [row["maximum_cost_usd"] for row in calls["reserved"]] == [.25, .25]
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [.02, .03]
    assert calls["restored"] == [0, .02]
    assert planner.review_survey(**options) == receipt and len(calls["specs"]) == 2


def test_two_invalid_answers_cannot_buy_more_repairs_on_resume_or_a_later_release(tmp_path, monkeypatch):
    options, calls, proposal = review_fixture(tmp_path, monkeypatch)
    invalid = deepcopy(proposal)
    invalid["operation_coverage"]["preparation_requirements"] = []
    options, calls, _ = review_fixture(tmp_path, monkeypatch, answers=[invalid])
    for release in ("c" * 40, "c" * 40, "d" * 40):
        with pytest.raises(ValueError, match="atlas_scene_planner_output_invalid"):
            planner.review_survey(**{**options, "release_blockers": lambda: (release, [])})
    assert len(calls["specs"]) == len(calls["reserved"]) == len(calls["settled"]) == 2
    assert sum(row["maximum_cost_usd"] for row in calls["reserved"]) == .5
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [.02, .02]


def test_text_only_cutoff_is_settled_without_an_output_expansion_or_new_paid_attempt(tmp_path, monkeypatch):
    options, calls, _ = review_fixture(tmp_path, monkeypatch, cutoff=True)
    for _ in range(2):
        with pytest.raises(ValueError, match="atlas_scene_planner_output_truncated_retry_unavailable"):
            planner.review_survey(**options)
    assert [spec.max_output_tokens for spec in calls["specs"]] == [4000]
    assert [row["provider_charge_amount_usd"] for row in calls["settled"]] == [.02]


@pytest.mark.parametrize("kind", ["frames", "previews"])
def test_image_inputs_are_refused_before_any_text_clarification_reservation(tmp_path, monkeypatch, kind):
    options, calls, _ = review_fixture(tmp_path, monkeypatch)
    if kind == "frames":
        survey = json.loads(options["survey_path"].read_text())
        survey["frames"] = [{"source_image_path": "/must-not-be-read.png"}]
        survey["digest"] = canonical_digest(survey, digest_field="digest")
        options["survey_path"].write_text(json.dumps(survey))
    else:
        options["recovery"]["previews"] = [{"path": "/must-not-be-read.png"}]
    with pytest.raises(ValueError, match="website_task_operation_coverage_text_only_required"):
        planner.review_survey(**options)
    assert calls["reserved"] == calls["settled"] == calls["specs"] == []
