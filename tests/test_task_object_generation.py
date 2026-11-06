import pytest

from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline.task_evaluation_scene_configuration_astra_driver import build_authoring_request
from tests import test_task_evaluation_scene_configuration_astra_driver as fixtures


@pytest.fixture
def retained(tmp_path):
    return fixtures.retained.__wrapped__(tmp_path)


def context(retained):
    return build_authoring_request(retained.input, retained.source, [retained.image], retained.rights).model_dump(mode="json")


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
