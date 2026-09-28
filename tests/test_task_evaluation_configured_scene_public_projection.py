from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_configured_scene_public_projection import (
    ConfiguredScenePublicProjectionError,
    build_public_display_projection,
    validate_public_display_authorization,
)
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError,
    validate_launch_preparation_request,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    test_configuration_request as configuration_request_fixture,
)


def _authorized_request() -> dict[str, object]:
    request = configuration_request_fixture()
    scene = request["scene"]
    task = request["task"]
    rights = scene["rights"]
    human_authority = next(
        row["artifact"] for row in rights["evidence"] if row["role"] == "human_authority_record"
    )
    authority = {
        "schema_version": ("task_evaluation_configured_scene_public_display_authorization.v1"),
        "status": "authorized",
        "scope": "configured_scene_derived_listing",
        "scene_identity": copy.deepcopy(scene["identity"]),
        "task_identity": copy.deepcopy(task["identity"]),
        "subject_identity": copy.deepcopy(task["subject"]["identity"]),
        "rights_admission_digest": rights["admission"]["digest"],
        "human_authority_record_digest": human_authority["digest"],
        "public_slug": "interiorgs-841007-planar-mug-push",
        "title": "Planar Mug Push",
        "summary": "A robot-neutral configured scene for a planar mug push.",
        "category": "Rigid relocation",
        "allowed_fields": [
            "status",
            "scene_identity",
            "task_identity",
            "task_kind",
            "task_strategy",
            "public_title",
            "public_summary",
            "public_category",
            "thumbnail",
            "proof_boundary",
        ],
        "thumbnail_publication_authorized": True,
        "derived_metadata_publication_authorized": True,
        "private_artifact_uri_publication_authorized": False,
        "raw_media_publication_authorized": False,
        "authority_reference": "owner-public-display-authorization-20260828",
        "authorized_by": "blueprint-owner",
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    rights["public_display_authorization"] = authority
    return request


def _revision_and_offering(request: dict[str, object]):
    revision = {
        "status": "configured",
        "scene_identity": copy.deepcopy(request["scene"]["identity"]),
        "task_template": {
            "identity": copy.deepcopy(request["task"]["identity"]),
        },
        "replacement": {
            "identity": copy.deepcopy(request["task"]["subject"]["identity"]),
        },
        "revision_digest": "sha256:" + "a" * 64,
    }
    offering = {
        "status": "configured_controls_pending",
        "scene_identity": copy.deepcopy(revision["scene_identity"]),
        "task": {
            "identity": copy.deepcopy(revision["task_template"]["identity"]),
            "subject_identity": copy.deepcopy(revision["replacement"]["identity"]),
        },
        "presentation": {
            "task_thumbnail": {"digest": "sha256:" + "b" * 64},
        },
        "offering_digest": "",
    }
    return revision, offering


def test_missing_authorization_remains_private_without_projection() -> None:
    request = configuration_request_fixture()
    revision, offering = _revision_and_offering(request)

    assert validate_public_display_authorization(request) is None
    assert (
        build_public_display_projection(
            request=request,
            revision=revision,
            offering=offering,
            source_offering_digest=canonical_digest(offering, digest_field="offering_digest"),
            diagnostic_only=False,
        )
        is None
    )


def test_explicit_authorization_is_accepted_by_canonical_request() -> None:
    request = _authorized_request()

    assert validate_launch_preparation_request(request) == request


def test_projection_refuses_diagnostic_result_even_when_authorized() -> None:
    request = _authorized_request()
    revision, offering = _revision_and_offering(request)

    with pytest.raises(
        ConfiguredScenePublicProjectionError,
        match="configured_scene_public_projection_nonqualifying",
    ):
        build_public_display_projection(
            request=request,
            revision=revision,
            offering=offering,
            source_offering_digest=canonical_digest(offering, digest_field="offering_digest"),
            diagnostic_only=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", "Internal path /var/lib/blueprint/scene.usdc"),
        ("summary", "Internal path /tmp/blueprint/scene.usdc"),
        ("title", "s3://private-bucket/task-thumbnail.png"),
        ("category", "API_KEY material"),
    ],
)
def test_public_metadata_refuses_paths_private_uris_and_secret_markers(
    field: str, value: str
) -> None:
    request = _authorized_request()
    authority = request["scene"]["rights"]["public_display_authorization"]
    authority[field] = value
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_public_display_authorization_invalid",
    ):
        validate_launch_preparation_request(request)


def test_authorization_digest_and_rights_binding_are_fail_closed() -> None:
    request = _authorized_request()
    authority = request["scene"]["rights"]["public_display_authorization"]
    authority["human_authority_record_digest"] = "sha256:" + "f" * 64
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_public_display_authorization_invalid",
    ):
        validate_launch_preparation_request(request)


def test_projection_digest_cross_binds_exact_scene_revision_and_thumbnail() -> None:
    request = _authorized_request()
    revision, offering = _revision_and_offering(request)
    source_offering_digest = canonical_digest(offering, digest_field="offering_digest")

    projection = build_public_display_projection(
        request=request,
        revision=revision,
        offering=offering,
        source_offering_digest=source_offering_digest,
        diagnostic_only=False,
    )

    assert projection is not None
    assert projection["source_offering_digest"] == source_offering_digest
    assert projection["scene_identity_digest"] == canonical_digest(offering["scene_identity"])
    assert projection["configured_scene_revision_digest"] == revision["revision_digest"]
    assert (
        projection["task_thumbnail_digest"] == offering["presentation"]["task_thumbnail"]["digest"]
    )
    assert projection["projection_digest"] == canonical_digest(
        projection, digest_field="projection_digest"
    )
