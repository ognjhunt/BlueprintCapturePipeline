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


def test_context_digest_matches_actual_webapp_numeric_canonicalization() -> None:
    source = _source()
    source["envelope"]["coordinate_frame_declaration"] = {"camera_translation": [1.0, -0.0, 1e-7]}
    context = build_owned_capture_atlas_context(source=source, plan_id="reconstruction-test")
    # Independently produced by committed WebApp crossRuntimeArtifactDigest
    # under Node, not Python json.dumps (which retains 1.0, -0.0 and 1e-07).
    assert context["context_digest"] == "sha256:552eb8609d935ab021647011c598ea267cf67ab3a97df47b1f429810f66106c9"
