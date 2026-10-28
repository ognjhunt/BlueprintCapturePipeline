"""An unresolved operation reading can be clarified without asserting readiness."""

from copy import deepcopy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import website_operation_coverage as coverage
from blueprint_pipeline import website_task_semantics as semantics
from blueprint_pipeline.website_task_preparation import native_task_preparation_gap
from tests.test_website_task_semantics import semantics_fixture, _refresh_receipt


def unresolved_fixture():
    context, source, parent_receipt = semantics_fixture()
    reading = parent_receipt["proposal"]["task_semantics"]
    reading["covers_entire_job"] = False
    reading["unresolved_aspects"] = [
        "Synthetic ball dimensions, mass and collision geometry remain to be authored.",
        "Floor and destination region heights and metric scale remain estimates.",
        "Numeric settling and task success tolerances need owner confirmation.",
        "Robot reach, clearance, policy cameras and native physics remain unqualified.",
        "Provider authority, deployment and task execution are not established here.",
    ]
    _refresh_receipt(parent_receipt)
    parent = semantics.task_semantics_from_receipt(
        receipt=parent_receipt, task_context=context, source_video_digest=source
    )
    return context, source, parent


def clarification(context, source, parent, **changes):
    request = coverage.clarification_request(
        parent, task_context=context, source_video_digest=source
    )
    reading = parent["semantics"]
    result = {
        "operative_job": reading["operative_job"],
        "task_family": reading["task_family"],
        "steps": deepcopy(reading["steps"]),
        "operation_coverage_complete": True,
        "unresolved_operations": [],
        "preparation_requirements": list(reading["unresolved_aspects"]),
    }
    result.update(changes)
    binding = {
        "task_context_digest": context["context_digest"],
        "source_video_digest": source,
        "operation_coverage_request_digest": canonical_digest(request),
    }
    proposal = {"operation_coverage": result}
    return {
        "status": "completed",
        "binding": binding,
        "binding_digest": canonical_digest(binding),
        "proposal": proposal,
        "proposal_digest": canonical_digest(proposal),
        "source_commit": "c" * 40,
        "usage": {"input_tokens": 100, "output_tokens": 100},
        "cost_usd": 0.01,
        "claim_ceiling": "development_only",
    }


def joined(context, source, parent, **changes):
    return coverage.from_receipt(
        parent,
        clarification(context, source, parent, **changes),
        task_context=context,
        source_video_digest=source,
    )


def select(tmp_path, context, source, parent):
    (tmp_path / "selection_binding.json").write_text(
        json.dumps({"planner_receipt": parent["planner_receipt"]})
    )
    semantics.retain_task_semantics(
        job_dir=tmp_path,
        receipt=parent["planner_receipt"],
        task_context=context,
        source_video_digest=source,
    )


def test_preparation_requirements_do_not_prevent_complete_operation_routing():
    context, source, parent = unresolved_fixture()
    original_context, original_parent = deepcopy(context), deepcopy(parent)
    assert (
        semantics.resolved_task_profile(parent, task_context=context, source_video_digest=source)
        is None
    )
    held = native_task_preparation_gap(context, task_semantics=parent, source_video_digest=source)
    assert held["code"] == "website_task_operation_coverage_unresolved"
    assert held["missing"] == ["source_bound_operation_coverage_clarification"]
    assert "native_task_runtime_adapter" not in held["missing"]
    clarified = joined(context, source, parent)
    assert (
        semantics.resolved_task_profile(clarified, task_context=context, source_video_digest=source)
        == "rigid_pick_place"
    )
    assert (
        native_task_preparation_gap(context, task_semantics=clarified, source_video_digest=source)
        is None
    )
    assert clarified["semantics"] == parent["semantics"]
    assert clarified["semantics"]["covers_entire_job"] is False
    assert clarified["planner_receipt"] == parent["planner_receipt"]
    assert (
        clarified["operation_coverage_clarification"]["operation_coverage"][
            "preparation_requirements"
        ]
        == parent["semantics"]["unresolved_aspects"]
    )
    assert parent == original_parent and context == original_context
    assert context["confirmed"] is False and clarified["claim_ceiling"] == "development_only"


@pytest.mark.parametrize(
    "change", ["source", "context", "request", "status", "binding_seal", "proposal_seal"]
)
def test_clarification_receipt_is_bound_to_exact_parent_source_and_context(change):
    context, source, parent = unresolved_fixture()
    receipt = clarification(context, source, parent)
    if change == "source":
        receipt["binding"]["source_video_digest"] = "sha256:" + "d" * 64
    elif change == "context":
        receipt["binding"]["task_context_digest"] = "sha256:" + "d" * 64
    elif change == "request":
        receipt["binding"]["operation_coverage_request_digest"] = "sha256:" + "d" * 64
    elif change == "status":
        receipt["status"] = "submitting"
    elif change == "binding_seal":
        receipt["binding_digest"] = "sha256:" + "d" * 64
    else:
        receipt["proposal_digest"] = "sha256:" + "d" * 64
    if change in {"source", "context", "request"}:
        receipt["binding_digest"] = canonical_digest(receipt["binding"])
    with pytest.raises(ValueError, match="operation_coverage_binding_invalid"):
        coverage.from_receipt(parent, receipt, task_context=context, source_video_digest=source)


@pytest.mark.parametrize("change", ["quote", "step_lost", "issue_lost", "issue_duplicate"])
def test_every_original_operation_and_unresolved_issue_is_preserved(change):
    context, source, parent = unresolved_fixture()
    receipt = clarification(context, source, parent)
    reading = receipt["proposal"]["operation_coverage"]
    if change == "quote":
        reading["steps"][0]["task_quote"] = "Invented owner text"
    elif change == "step_lost":
        reading["steps"] = []
    elif change == "issue_lost":
        reading["preparation_requirements"].pop()
    else:
        reading["unresolved_operations"] = reading["preparation_requirements"][:1]
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    with pytest.raises(ValueError, match="operation_coverage_(quote_invalid|step_lost|issue_lost)"):
        coverage.from_receipt(parent, receipt, task_context=context, source_video_digest=source)


@pytest.mark.parametrize(
    "change", ["unresolved", "incomplete", "unsupported_extra", "multiple_atomic"]
)
def test_genuine_unknown_or_additional_work_stays_held(change):
    context, source, parent = unresolved_fixture()
    original = parent["semantics"]
    changes = {}
    if change == "unresolved":
        changes = {
            "unresolved_operations": [original["unresolved_aspects"][0]],
            "preparation_requirements": original["unresolved_aspects"][1:],
        }
    elif change == "incomplete":
        changes = {"operation_coverage_complete": False}
    else:
        step = deepcopy(original["steps"][0])
        if change == "unsupported_extra":
            step["operation"] = "unsupported_operation"
        changes = {"steps": [*original["steps"], step], "task_family": "multi_step"}
    value = joined(context, source, parent, **changes)
    assert (
        semantics.resolved_task_profile(value, task_context=context, source_video_digest=source)
        is None
    )
    gap = native_task_preparation_gap(context, task_semantics=value, source_video_digest=source)
    assert gap["code"] == "website_task_operation_coverage_unresolved"
    if change == "unsupported_extra":
        assert gap["unsupported_operations"] == ["unsupported_operation"]


def test_scale_abstention_does_not_change_an_independent_operation_reading():
    context, source, parent = unresolved_fixture()
    receipt = clarification(context, source, parent)
    receipt["proposal"].update(
        status="abstained", scale_references=[], summary="No metric reference established."
    )
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    value = coverage.from_receipt(parent, receipt, task_context=context, source_video_digest=source)
    assert (
        semantics.resolved_task_profile(value, task_context=context, source_video_digest=source)
        == "rigid_pick_place"
    )
    assert (
        value["operation_coverage_clarification"]["planner_receipt"]["proposal"]["status"]
        == "abstained"
    )


def test_multi_step_clarification_keeps_every_registered_operation_and_native_stage_order():
    context, source, receipt = semantics_fixture()
    context["description"] = "Wipe the bench and then press the button. Preserve all obstacles."
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    reading = receipt["proposal"]["task_semantics"]
    reading.update(
        task_family="multi_step",
        covers_entire_job=False,
        operative_job="Wipe the bench, then press the button.",
        steps=[
            {
                "operation": "wiping",
                "description": "Wipe the bench.",
                "task_quote": "Wipe the bench",
            },
            {
                "operation": "button_press",
                "description": "Press the button.",
                "task_quote": "press the button",
            },
        ],
        unresolved_aspects=["Surface physics and numeric native criteria need preparation."],
    )
    _refresh_receipt(receipt)
    parent = semantics.task_semantics_from_receipt(
        receipt=receipt, task_context=context, source_video_digest=source
    )
    value = joined(context, source, parent)
    assert (
        semantics.resolved_task_profile(value, task_context=context, source_video_digest=source)
        == "multi_step"
    )
    semantics.require_semantic_stage_order(
        task_context=context,
        task_semantics=value,
        source_video_digest=source,
        contract={
            "task_family": "multi_step",
            "stages": [{"task_family": "wiping"}, {"task_family": "button_press"}],
        },
    )
    with pytest.raises(ValueError, match="semantic_stages_mismatch"):
        semantics.require_semantic_stage_order(
            task_context=context,
            task_semantics=value,
            source_video_digest=source,
            contract={
                "task_family": "multi_step",
                "stages": [{"task_family": "button_press"}, {"task_family": "wiping"}],
            },
        )
    with pytest.raises(ValueError, match="operation_coverage_step_lost"):
        joined(context, source, parent, steps=list(reversed(reading["steps"])))
    repeated = joined(context, source, parent, steps=[*reading["steps"], reading["steps"][0]])
    assert (
        semantics.resolved_task_profile(repeated, task_context=context, source_video_digest=source)
        is None
    )


def test_selection_anchors_one_immutable_clarification_without_overwriting_parent(tmp_path):
    context, source, parent = unresolved_fixture()
    select(tmp_path, context, source, parent)
    path = tmp_path / "task_semantics" / (parent["digest"][7:] + ".json")
    original = path.read_bytes()
    receipt = clarification(context, source, parent, operation_coverage_complete=False)
    value = coverage.retain_operation_coverage(
        job_dir=tmp_path,
        parent_semantics=parent,
        receipt=receipt,
        task_context=context,
        source_video_digest=source,
    )
    assert (
        coverage.retain_operation_coverage(
            job_dir=tmp_path,
            parent_semantics=parent,
            receipt=receipt,
            task_context=context,
            source_video_digest=source,
        )
        == value
    )
    loaded = semantics.load_task_semantics(
        job_dir=tmp_path, task_context=context, source_video_digest=source
    )
    assert loaded == value and path.read_bytes() == original
    assert (
        coverage.clarification_request(loaded, task_context=context, source_video_digest=source)
        is None
    )
    with pytest.raises(ValueError, match="immutable_input_staging_conflict"):
        coverage.retain_operation_coverage(
            job_dir=tmp_path,
            parent_semantics=parent,
            receipt=clarification(context, source, parent),
            task_context=context,
            source_video_digest=source,
        )


def test_clarification_cannot_substitute_for_selected_parent_or_mutate_its_own_reading(tmp_path):
    context, source, parent = unresolved_fixture()
    receipt = clarification(context, source, parent)
    with pytest.raises(ValueError, match="operation_coverage_parent_not_selected"):
        coverage.retain_operation_coverage(
            job_dir=tmp_path,
            parent_semantics=parent,
            receipt=receipt,
            task_context=context,
            source_video_digest=source,
        )
    select(tmp_path, context, source, parent)
    value = coverage.retain_operation_coverage(
        job_dir=tmp_path,
        parent_semantics=parent,
        receipt=receipt,
        task_context=context,
        source_video_digest=source,
    )
    value["operation_coverage_clarification"]["operation_coverage"]["task_family"] = (
        "articulated_open_close"
    )
    value["digest"] = canonical_digest(value, digest_field="digest")
    with pytest.raises(ValueError, match="operation_coverage_changed"):
        semantics.validate_task_semantics(value, task_context=context, source_video_digest=source)
    changed_parent = deepcopy(parent["planner_receipt"])
    changed_parent["usage"]["output_tokens"] += 1
    semantics.retain_task_semantics(
        job_dir=tmp_path, receipt=changed_parent, task_context=context, source_video_digest=source
    )
    (tmp_path / "selection_binding.json").write_text(
        json.dumps({"planner_receipt": changed_parent})
    )
    newer = semantics.load_task_semantics(
        job_dir=tmp_path, task_context=context, source_video_digest=source
    )
    assert "operation_coverage_clarification" not in newer


@pytest.mark.parametrize("kind", ["complete", "explicit", "unknown", "legacy"])
def test_completed_legacy_and_explicit_task_readings_do_not_request_clarification(kind, tmp_path):
    context, source, parent = unresolved_fixture()
    receipt = deepcopy(parent["planner_receipt"])
    if kind == "complete":
        receipt["proposal"]["task_semantics"].update(covers_entire_job=True, unresolved_aspects=[])
    elif kind == "explicit":
        context["task_family"] = "rigid_pick_place"
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
        receipt["binding"]["task_context_digest"] = context["context_digest"]
    elif kind == "unknown":
        receipt["proposal"]["task_semantics"]["task_family"] = "washing"
    else:
        del receipt["proposal"]["task_semantics"]
    _refresh_receipt(receipt)
    value = semantics.retain_task_semantics(
        job_dir=tmp_path, receipt=receipt, task_context=context, source_video_digest=source
    )
    (tmp_path / "selection_binding.json").write_text(json.dumps({"planner_receipt": receipt}))
    if kind == "legacy":
        assert (
            value is None
            and semantics.load_task_semantics(
                job_dir=tmp_path, task_context=context, source_video_digest=source
            )
            is None
        )
    else:
        assert (
            coverage.clarification_request(value, task_context=context, source_video_digest=source)
            is None
        )
        assert (
            semantics.load_task_semantics(
                job_dir=tmp_path, task_context=context, source_video_digest=source
            )
            == value
        )
