"""Semantic refinement cannot turn a scene read into paid controller work."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import website_operation_coverage_review as review
from blueprint_pipeline import website_task_semantics as semantics
from blueprint_pipeline.website_task_context_contracts import holds_native_task_preparation
from tests.test_website_task_semantics import semantics_fixture


def selected_parent(job, *, context=None, source=None):
    original_context, original_source, receipt = semantics_fixture()
    context = deepcopy(context or original_context)
    source = source or original_source
    receipt["binding"].update(task_context_digest=context["context_digest"], source_video_digest=source)
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    receipt["proposal"]["task_semantics"].update(covers_entire_job=False,
        unresolved_aspects=["The synthetic asset and its parameters still need authoring.",
                            "Metric support geometry and numeric scoring tolerances need qualification."])
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    write_json(job / "selection_binding.json", {"planner_receipt": receipt})
    parent = semantics.retain_task_semantics(job_dir=job, receipt=receipt, task_context=context,
                                             source_video_digest=source)
    return context, source, receipt, parent


def clarification_receipt(request, context, source, *, job_label=None):
    original = request["original_semantics"]
    proposal = {"operation_coverage": {
        "operative_job": job_label or original["operative_job"], "task_family": original["task_family"],
        "steps": deepcopy(original["steps"]), "operation_coverage_complete": True,
        "unresolved_operations": [], "preparation_requirements": list(original["unresolved_aspects"]),
    }}
    binding = {"task_context_digest": context["context_digest"], "source_video_digest": source,
               "operation_coverage_request_digest": canonical_digest(request)}
    return {"status": "completed", "binding": binding, "binding_digest": canonical_digest(binding),
            "proposal": proposal, "proposal_digest": canonical_digest(proposal),
            "cost_usd": .02, "claim_ceiling": "development_only"}


def allocator_seams(monkeypatch):
    from blueprint_pipeline import paid_resource_allocator, website_atlas_pipeline
    calls = {"configuration": [], "planner": []}

    def configuration(context, source, *, create=True):
        calls["configuration"].append((deepcopy(context), source, create))
        assert create is False
        return {"policy": {"model": "gpt-6.1-sol", "maximum_review_cost_usd": 1},
                "grant_creation_allowed": False, "dispatch_authorized": False}

    def planner(**kwargs):
        calls["planner"].append(kwargs)
        survey = json.loads(kwargs["survey_path"].read_text())
        assert survey["frames"] == [] and kwargs["recovery"]["previews"] == []
        request = kwargs["recovery"]["operation_coverage_request"]
        return clarification_receipt(request, kwargs["task_context"], survey["source_video_digest"])

    monkeypatch.setattr(website_atlas_pipeline, "load_configuration", configuration)
    monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_plan", planner)
    return calls


def test_completed_scene_refines_through_canonical_allocator_then_adopts_immutable_child(tmp_path, monkeypatch):
    job = tmp_path / "atlas"
    context, source, _, parent = selected_parent(job)
    write_json(job / "result.json", {"retained": "complete scene export"})
    write_json(job / "pose/receipt.json", {"retained": "completed original pose"})
    preserved = {path: path.read_bytes() for path in job.rglob("*.json")}
    calls = allocator_seams(monkeypatch)
    result = review.clarify_operation_coverage(job_dir=job, task_context=context, source_video_digest=source)
    assert result["status"] == "clarification_retained"
    child = result["task_semantics"]
    assert child["semantics"] == parent["semantics"] and not child["semantics"]["covers_entire_job"]
    assert semantics.resolved_task_profile(child, task_context=context, source_video_digest=source) == "rigid_pick_place"
    assert all(path.read_bytes() == raw for path, raw in preserved.items())
    assert calls["configuration"][0][2] is False and len(calls["planner"]) == 1
    assert calls["planner"][0]["experimental"] is False
    assert review.clarify_operation_coverage(job_dir=job, task_context=context, source_video_digest=source) == {
        "status": "adopted_or_not_required", "task_semantics": child}
    assert len(calls["configuration"]) == len(calls["planner"]) == 1


@pytest.mark.parametrize("changed", ["context", "source"])
def test_changed_context_or_source_refuses_before_getter_or_allocator(tmp_path, monkeypatch, changed):
    job = tmp_path / "atlas"
    context, source, _, _ = selected_parent(job)
    calls = allocator_seams(monkeypatch)
    if changed == "context":
        context["description"] += " Then wash the object."
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    else:
        source = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="website_task_semantics_planner_binding_invalid"):
        review.clarify_operation_coverage(job_dir=job, task_context=context, source_video_digest=source)
    assert calls["configuration"] == calls["planner"] == []


@pytest.mark.parametrize("kind", ["legacy", "explicit_rigid", "explicit_articulated", "explicit_unknown"])
def test_legacy_completed_review_and_explicit_owner_families_never_buy_clarification(tmp_path, monkeypatch, kind):
    job = tmp_path / "atlas"
    context, source, receipt, _ = selected_parent(job)
    calls = allocator_seams(monkeypatch)
    if kind == "legacy":
        receipt["proposal"].pop("task_semantics")
        receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
        write_json(job / "selection_binding.json", {"planner_receipt": receipt})
    else:
        context["task_family"] = {"explicit_rigid": "rigid_pick_place", "explicit_articulated": "articulated_open_close",
                                  "explicit_unknown": "unsupported_owner_family"}[kind]
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    result = review.clarify_operation_coverage(job_dir=job, task_context=context, source_video_digest=source)
    assert result["status"] == "adopted_or_not_required"
    assert calls["configuration"] == calls["planner"] == []


def test_scale_consumer_retains_first_child_and_preserves_base_request_on_resume(tmp_path):
    job = tmp_path / "atlas"
    context, source, _, parent = selected_parent(job)
    first = review.scale_operation_coverage_options(job_dir=job, task_context=context, source_video_digest=source)
    assert first["operation_coverage_needed"] is True
    receipt = clarification_receipt(first["operation_coverage_request"], context, source)
    retained = first["operation_coverage_consumer"](receipt)
    second = review.scale_operation_coverage_options(job_dir=job, task_context=context, source_video_digest=source)
    assert second["operation_coverage_needed"] is False
    assert second["operation_coverage_request"] == first["operation_coverage_request"]
    later = clarification_receipt(second["operation_coverage_request"], context, source, job_label="Later scale reading")
    assert second["operation_coverage_consumer"](later) == retained
    assert retained["semantics"] == parent["semantics"]


@pytest.mark.parametrize("allow", [False, True])
def test_handoff_only_explicit_controller_flag_can_buy_clarification(tmp_path, monkeypatch, allow):
    from blueprint_pipeline import website_scene_handoff as handoff
    from tests.test_website_scene_handoff import inputs, atlas_ready, ATLAS
    from tests.test_website_task_preparation import SYNTHETIC_BALL
    kwargs = inputs(tmp_path)
    context, source, _ = semantics_fixture()
    context.update(capture_id="cap-1", scene_id="scene-1")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    job = tmp_path / "pipeline/atlas"
    selected_parent(job, context=context, source=source)
    kwargs["descriptor"]["metadata"]["site_task_context"] = context
    ready = atlas_ready(tmp_path)
    ready.update(subject_origin="synthetic_design")
    ready["source_geometry"]["binding"] = {"source_video_digest": source}
    monkeypatch.setattr(ATLAS, lambda **_: ready)
    write_json(tmp_path / "pipeline/removal.json", {"entries": [], "synthetic_subject": deepcopy(SYNTHETIC_BALL)})
    calls = allocator_seams(monkeypatch)

    class RoutingPassed(Exception):
        pass

    def downstream(**_):
        raise RoutingPassed("Stop before any masks, CAD, native physics or scorer")

    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", downstream)
    if allow:
        with pytest.raises(RoutingPassed):
            handoff.prepare_website_scene_handoff(**kwargs, allow_operation_coverage_review=True)
        assert len(calls["configuration"]) == len(calls["planner"]) == 1
    else:
        result = handoff.prepare_website_scene_handoff(**kwargs)  # The default is observational.
        assert result["blockers"] == ["website_task_operation_coverage_unresolved"]
        assert result["provider_mutation_performed"] is False
        assert holds_native_task_preparation(result, context)
        result["native_task_preparation_gap"]["missing"] = ["native_task_runtime_adapter"]
        result["digest"] = canonical_digest(result, digest_field="digest")
        assert not holds_native_task_preparation(result, context)
        assert calls["configuration"] == calls["planner"] == []


def test_family_handoff_does_not_claim_no_provider_mutation_after_a_clarification(tmp_path, monkeypatch):
    from blueprint_pipeline import website_scene_handoff as handoff
    from tests.test_website_scene_handoff import inputs, atlas_ready, ATLAS
    from tests.test_task_semantics_catalog_composition import catalog_semantics_fixture
    kwargs = inputs(tmp_path)
    context, source, artifact = catalog_semantics_fixture("button_press")
    context.update(capture_id="cap-1", scene_id="scene-1")
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    receipt = artifact["planner_receipt"]
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    receipt["proposal"]["task_semantics"].update(covers_entire_job=False,
        unresolved_aspects=["The numeric thresholds still require owner confirmation."])
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    job = tmp_path / "pipeline/atlas"
    write_json(job / "selection_binding.json", {"planner_receipt": receipt})
    semantics.retain_task_semantics(job_dir=job, receipt=receipt, task_context=context, source_video_digest=source)
    kwargs["descriptor"]["metadata"]["site_task_context"] = context
    ready = atlas_ready(tmp_path)
    ready["source_geometry"]["binding"] = {"source_video_digest": source}
    monkeypatch.setattr(ATLAS, lambda **_: ready)
    calls = allocator_seams(monkeypatch)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    monkeypatch.setattr(handoff, "compile_website_scene_preparation", lambda **_: {
        "status": "needs_input", "blockers": ["website_task_family_success_contract_required"],
        "task_evaluation_inputs": {}, "digest": "sha256:" + "b" * 64})
    result = handoff.prepare_website_scene_handoff(**kwargs, allow_operation_coverage_review=True)
    assert result["status"] == "needs_input", result
    assert "provider_mutation_performed" not in result
    assert len(calls["planner"]) == 1


@pytest.mark.parametrize("mode", ["atlas-status", "discover-selector"])
def test_selected_read_surfaces_never_invoke_site_orchestrator_or_clarifier(tmp_path, monkeypatch, mode):
    from blueprint_pipeline import selected_handoff_recovery, selected_atlas_preview_recovery
    from tests.test_selected_handoff_recovery import fixture, discovery_fixture

    def forbidden(**_):
        pytest.fail("A selected read must not invoke the site orchestrator or semantic paid recovery")

    monkeypatch.setattr(review, "clarify_operation_coverage", forbidden)
    monkeypatch.setattr("blueprint_pipeline.website_scene_handoff.prepare_website_scene_handoff", forbidden)
    if mode == "discover-selector":
        selection, _, _, client, listener, owner, retirement, _, native_calls = discovery_fixture(tmp_path)
    else:
        selection, _, client, listener, owner, retirement, _, _, native_calls = fixture(tmp_path)
        selection["mode"] = mode
        monkeypatch.setattr(selected_atlas_preview_recovery, "recover_selected_atlas_preview",
                            lambda *_, **__: {"status": "atlas_status_returned", "atlas_retry_performed": False})
    result = selected_handoff_recovery.recover_selected_handoff(selection, storage_root=tmp_path,
        client=client, listener=listener, owner_reader=owner, retirement_reader=retirement,
        membership_reader=lambda **_: (b"original membership", {"raw": []}, []),
        admission_reader=lambda *args: pytest.fail("A selected read must not create sponsorship"))
    assert result["status"] in {"atlas_status_returned", "selector_discovered"}
    assert not native_calls and result["provider_dispatch_performed"] is False
