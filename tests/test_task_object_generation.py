import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline import task_object_generation as generation
from blueprint_pipeline.task_evaluation_scene_configuration_astra_driver import build_authoring_request
from tests import test_task_evaluation_scene_configuration_astra_driver as fixtures


@pytest.fixture
def retained(tmp_path):
    return fixtures.retained.__wrapped__(tmp_path)


def spec(identity="small_box", *, variant_of="source_book"):
    return {"object_id": identity, "description": "Small blue rigid package", "task_purpose": "Test package pick-and-place size variation",
            "dimensions_m": [.08, .06, .04], "geometry_features": ["closed rectangular package", "flat grasp faces"],
            "appearance_requirements": ["blue opaque wrapping"], "material_description": "cardboard with blue wrapping",
            "appearance": "opaque", "variant_of": variant_of}


def context(retained):
    return build_authoring_request(retained.input, retained.source, [retained.image], retained.rights).model_dump(mode="json")


def test_new_objects_and_variants_share_task_context_without_inventing_capture(retained, tmp_path):
    original = context(retained)
    frozen = json.dumps(original, sort_keys=True)
    requests = generation.build_generated_object_requests(context_request=original,
        specifications=[spec(), spec("new_package", variant_of=None)], output_root=tmp_path / "specs")
    assert json.dumps(original, sort_keys=True) == frozen
    assert len(requests) == 2
    for request in requests:
        assert request.dimensions_m == (.08, .06, .04)
        assert request.dimension_authority == "estimated"
        assert request.physical_review_input.measured.mass_kg is None
        assert request.physical_review_input.object_id == request.object_id
        assert all(frame.role == "task_context" for frame in request.source_frames)
        assert author.validate_request(request.model_dump(mode="json")) == request
        assert request.generated_specification.task_purpose in request.construction_constraints
        assert "not an observed physical object" in request.physical_review_input.evidence[0].excerpt
    assert requests[0].generated_specification.variant_of == "source_book"
    assert requests[1].generated_specification.variant_of is None


@pytest.mark.parametrize("specifications", [[spec("source_book")], [spec(), spec()], [spec(variant_of="unbound")]])
def test_generated_objects_cannot_replace_anchor_or_claim_unbound_parent(retained, tmp_path, specifications):
    with pytest.raises(ValueError):
        generation.build_generated_object_requests(context_request=context(retained),
                                                   specifications=specifications, output_root=tmp_path / "specs")
    assert not (tmp_path / "specs").exists()


def test_generated_review_requires_matching_requested_changes():
    review = author.AppearanceReview(source_object_recognizable=False, source_color_and_material_preserved=False,
        opaque_surfaces_opaque=True, required_parts_present=True, no_obvious_geometry_artifacts=True,
        blockers=[], repair_instructions="", unobserved_surface_limitations=[], requested_specification_satisfied=True)
    assert author.appearance_passed(review, generated=True)
    assert not author.appearance_passed(review)
    assert not author.appearance_passed(review.model_copy(update={"requested_specification_satisfied": None}), generated=True)
    assert not author.appearance_passed(review.model_copy(update={"required_parts_present": False}), generated=True)


def test_captured_requests_and_retained_reviews_keep_existing_digest_shape(retained):
    original = context(retained)
    assert "generated_specification" not in original
    author.validate_request(original)
    old_review = {"source_object_recognizable": True, "source_color_and_material_preserved": True,
        "opaque_surfaces_opaque": True, "required_parts_present": True, "no_obvious_geometry_artifacts": True,
        "blockers": [], "repair_instructions": "", "unobserved_surface_limitations": []}
    assert author.AppearanceReview.model_validate(old_review).model_dump(mode="json") == old_review


def test_variant_intent_reaches_cad_blender_and_shared_batch_execution(retained, tmp_path):
    requests = generation.build_generated_object_requests(context_request=context(retained),
        specifications=[spec("blocked_package"), spec("working_package")], output_root=tmp_path / "specs")
    brief = author.VisualBrief(object_identity="package", observed_parts=[], appearance_requirements=["blue wrapping"],
        unknown_regions=["generated new object"], cad_brief_markdown="Closed rectangular package", proposed_material="cardboard",
        proposed_appearance="opaque")
    assert "blue opaque wrapping" in author.compact_cad_handoff(requests[1], brief)
    assert "blue opaque wrapping" in author.blender_author_prompt(requests[1], brief, "")
    invoker, calls = SimpleNamespace(), []
    def execute(**kwargs):
        calls.append(kwargs)
        request = kwargs["request_value"]
        if request["object_id"] == "blocked_package":
            raise RuntimeError("provider_rate_limit")
        return {"status": "candidate_authored_pending_native_qualification", "object_id": request["object_id"],
                "request_digest": request["request_digest"]}
    result = generation.execute_generated_object_batch(requests=requests, output_root=tmp_path / "batch",
        invoker=invoker, mac_executor=None, blender_runner=None, blender_executable="test-only", executor=execute)
    assert [row["status"] for row in result["objects"]] == ["held", "candidate_authored"]
    assert all(call["invoker"] is invoker for call in calls)
    assert result["evaluation_ready"] is False and result["native_qualification_required"] is True


def test_generated_request_cannot_inherit_measured_physics(retained, tmp_path):
    request = generation.build_generated_object_requests(context_request=context(retained),
        specifications=[spec()], output_root=tmp_path / "specs")[0].model_dump(mode="json")
    request["physical_review_input"]["measured"]["mass_kg"] = {
        "value": .2, "basis": "measured", "interval": {"lower": .19, "upper": .21},
        "rationale": "Anchor mass", "uncertainty": "Wrong object", "evidence_ids": ["generated_design"]}
    with pytest.raises(ValueError, match="authority_mismatch"):
        author.AuthoringRequest.model_validate(request)
