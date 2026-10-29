from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.owned_capture_atlas_context import (
    build_owned_capture_atlas_context,
    require_owned_capture_atlas_source_binding,
    validate_owned_capture_atlas_context,
)


def _source() -> dict:
    return {
        "receipt": {
            "capture_session_id": "capture-session-reference",
            "intake_id": "licensed-reference-intake",
            "capture_digest": "sha256:" + "1" * 64,
            "envelope_digest": "sha256:" + "2" * 64,
        },
        "qa": {"qa_report_digest": "sha256:" + "3" * 64},
        "object_manifest": {"manifest_digest": "sha256:" + "4" * 64},
        "envelope": {
            "scene_id": "reference-scene",
            "capture_authority_profile": "monocular_video",
            "known_task_specification": {
                "task_id": "reference-task",
                "description": "Move the object from the shelf into the box, close its lid, and return to the start.",
                "task_family": "customer_defined_multistage_operation",
                "success_parameters_status": "proposed_unconfirmed",
                "operations": ["pickup", "containment", "articulation", "return"],
            },
            "governance": {
                "rights": "accepted", "consent": "not_required", "privacy": "cleared",
                "provider_constraints": {"external_processing_allowed": False,
                                         "automatic_model_inference_allowed": False},
                "retention": {"max_days": 30},
            },
            "permitted_reconstruction_providers": ["local", "world_labs_atlas"],
            "permitted_evidence_uses": ["scene_assessment", "reconstruction"],
            "coordinate_frame_declaration": {"metric_scale": "unverified"},
        },
    }


def test_context_preserves_complete_task_permissions_and_uncertainty_without_authority() -> None:
    source = _source()
    original = deepcopy(source)
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    assert source == original
    assert context["task_specification"] == source["envelope"]["known_task_specification"]
    assert context["description"] == source["envelope"]["known_task_specification"]["description"]
    assert context["task_family"] == "customer_defined_multistage_operation"
    assert context["task_confirmation_status"] == "proposed_unconfirmed"
    assert context["governance"] == source["envelope"]["governance"]
    assert context["coordinate_frame_declaration"] == {"metric_scale": "unverified"}
    assert context["authority_issued"] is False
    assert context["source_capture"]["capture_digest"] == source["receipt"]["capture_digest"]
    context["task_specification"]["operations"].clear()
    assert source == original


def test_missing_site_identity_does_not_require_an_invented_site_attestation() -> None:
    source = _source()
    source["envelope"].pop("scene_id")
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    assert context["source_scene_id"] is None
    assert context["scene_id"] == context["capture_id"]
    assert context["scene_identity_basis"] == "capture_session_identity"
    assert context["governance"]["consent"] == "not_required"


@pytest.mark.parametrize("task", [None, {}, {"description": ""}, {"description": "   "}])
def test_missing_task_is_an_explicit_input_gap(task) -> None:
    source = _source()
    source["envelope"]["known_task_specification"] = task
    with pytest.raises(ValueError, match="owned_capture_atlas_task_description_required"):
        build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")


@pytest.mark.parametrize("field,value", [
    ("authority_issued", True),
    ("description", "A simpler substituted task"),
    ("task_confirmation_status", "confirmed"),
    ("scene_id", "invented-site"),
    ("paid_compute_authorized", True),
])
def test_rehashed_context_cannot_change_task_confirmation_or_issue_authority(field, value) -> None:
    context = build_owned_capture_atlas_context(source=_source(), plan_id="reconstruction-test")
    context[field] = value
    context["context_digest"] = cross_runtime_canonical_digest(context, digest_field="context_digest")
    with pytest.raises(ValueError):
        validate_owned_capture_atlas_context(context)


def test_source_change_invalidates_retained_context() -> None:
    context = build_owned_capture_atlas_context(source=_source(), plan_id="reconstruction-test")
    context["source_capture"]["capture_digest"] = "sha256:" + "5" * 64
    with pytest.raises(ValueError, match="owned_capture_atlas_context_digest_mismatch"):
        validate_owned_capture_atlas_context(context)


@pytest.mark.parametrize("change", ["source_digest", "permissions", "task", "retention"])
def test_rehashed_context_is_not_current_source_or_disclosure_authority(change) -> None:
    source = _source()
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    if change == "source_digest":
        context["source_capture"]["capture_digest"] = "sha256:" + "5" * 64
    elif change == "permissions":
        context["governance"]["provider_constraints"]["external_processing_allowed"] = True
    elif change == "retention":
        context["governance"]["retention"]["max_days"] = 365
    else:
        context["task_specification"]["description"] = "An easier task"
        context["description"] = "An easier task"
    context["context_digest"] = cross_runtime_canonical_digest(context, digest_field="context_digest")
    with pytest.raises(ValueError, match="owned_capture_atlas_source_or_task_changed"):
        require_owned_capture_atlas_source_binding(context, source=source, plan_id="reconstruction-test")


def test_a_source_bound_context_still_carries_the_actual_disclosure_refusal() -> None:
    source = _source()
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    current = require_owned_capture_atlas_source_binding(context, source=source, plan_id="reconstruction-test")
    assert current["authority_issued"] is False
    assert current["governance"]["provider_constraints"]["external_processing_allowed"] is False


def test_atlas_planner_receives_complete_owned_task_and_actual_confirmation_status() -> None:
    import json
    from blueprint_pipeline.atlas_scene_planner import _input
    from blueprint_pipeline.atlas_preparation_context import requires_task_semantic_analysis

    context = build_owned_capture_atlas_context(source=_source(), plan_id="reconstruction-test")
    before = deepcopy(context)
    content = _input({"frames": [], "display_rotation_degrees": 0}, context)
    payload = json.loads(content[0]["content"][0]["text"])
    assert payload["task_specification"] == context["task_specification"]
    assert payload["task_confirmation_status"] == "proposed_unconfirmed"
    assert payload["declared_task_family"] == "customer_defined_multistage_operation"
    assert requires_task_semantic_analysis(context)
    assert context == before


def _owned_semantics_fixture():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    source = _source()
    task = source["envelope"]["known_task_specification"]
    task.update(description="Carry out all recorded requirements.",
                operations=[{"instruction": "Move the block into the box."},
                            {"instruction": "Close the lid after placement."}],
                target_tolerance_m=1.0)
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    binding = {"task_context_digest": context["context_digest"],
               "source_video_digest": context["source_capture"]["capture_digest"]}
    proposal = {"status": "proposed", "task_semantics": {
        "operative_job": "Place the block in the box, then close the lid.",
        "task_family": "multi_step", "steps": [
            {"operation": "rigid_pick_place", "description": "Put block in box",
             "task_quote": "Move the block into the box."},
            {"operation": "close_container", "description": "Close the lid",
             "task_quote": "Close the lid after placement."}],
        "covers_entire_job": True, "unresolved_aspects": [], "preparation_notes": []}}
    receipt = {"status": "completed", "binding": binding,
               "binding_digest": canonical_digest(binding), "proposal": proposal,
               "proposal_digest": canonical_digest(proposal)}
    return context, receipt


def test_nested_owned_operations_are_quoted_and_unknown_step_never_becomes_pick_place() -> None:
    from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt, resolved_task_profile
    from blueprint_pipeline.atlas_preparation_context import preparation_context_digest_matches
    from blueprint_pipeline.website_operation_coverage import operation_coverage_gap

    context, receipt = _owned_semantics_fixture()
    original = deepcopy(context)
    assert preparation_context_digest_matches(context)  # RFC 8785 includes 1.0.
    video = context["source_capture"]["capture_digest"]
    value = task_semantics_from_receipt(receipt=receipt, task_context=context, source_video_digest=video)
    assert [row["operation"] for row in value["semantics"]["steps"]] == ["rigid_pick_place", "close_container"]
    assert resolved_task_profile(value, task_context=context, source_video_digest=video) is None
    gap = operation_coverage_gap(value, task_context=context, source_video_digest=video)
    assert gap is not None
    assert gap["task_family"] == context["task_family"]
    assert gap["derived_task_family"] == "multi_step"
    assert "close_container" in str(gap)
    assert context == original


def test_owned_semantic_quotes_cannot_add_an_unrequested_easier_task() -> None:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt

    context, receipt = _owned_semantics_fixture()
    receipt["proposal"]["task_semantics"]["steps"][1]["task_quote"] = "Leave the box open."
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    with pytest.raises(ValueError, match="website_task_semantics_task_quote_invalid"):
        task_semantics_from_receipt(receipt=receipt, task_context=context,
                                    source_video_digest=context["source_capture"]["capture_digest"])


@pytest.mark.parametrize("retain_semantics", [True, False])
def test_owned_registered_family_never_discards_a_later_unexecutable_operation(retain_semantics) -> None:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt, effective_task_profile

    context, receipt = _owned_semantics_fixture()
    context["task_specification"]["task_family"] = context["task_family"] = "rigid_pick_place"
    context["context_digest"] = cross_runtime_canonical_digest(context, digest_field="context_digest")
    receipt["binding"]["task_context_digest"] = context["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    video = context["source_capture"]["capture_digest"]
    value = task_semantics_from_receipt(receipt=receipt, task_context=context, source_video_digest=video)
    assert effective_task_profile(task_context=context, task_semantics=value if retain_semantics else None,
                                  source_video_digest=video) is None
    assert context["task_family"] == "rigid_pick_place"


@pytest.mark.parametrize("family", ["rigid_pick_place", "articulated_open_close", "customer_custom"])
def test_existing_explicit_website_families_keep_their_routing_behavior(family) -> None:
    from blueprint_pipeline.atlas_preparation_context import requires_task_semantic_analysis

    assert not requires_task_semantic_analysis({"task_family": family})


def test_context_digest_matches_actual_webapp_numeric_canonicalization() -> None:
    source = _source()
    source["envelope"]["coordinate_frame_declaration"] = {"camera_translation": [1.0, -0.0, 1e-7]}
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    # Independently produced by committed WebApp crossRuntimeArtifactDigest
    # under Node, not Python json.dumps (which retains 1.0, -0.0 and 1e-07).
    assert context["context_digest"] == "sha256:552eb8609d935ab021647011c598ea267cf67ab3a97df47b1f429810f66106c9"
