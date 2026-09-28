"""Join website source/consent records for derived native assets (ADP-009D/day21).

Keep the original records and hashes. This admits only the configured revision's
exact derived assets; it grants no raw-video upload, training or redistribution.
"""

from collections.abc import Mapping

from .decision_evidence_contracts import canonical_digest

SOURCE = "website_scene_preparation.v1"
RIGHTS = "website_native_rights_admission.v1"


def reference_valid(value, *, expected_schema, scene_id):
    if not isinstance(value, Mapping) or value.get("claim_ceiling") != "development_only":
        return False
    if value.get("digest") != canonical_digest(value, digest_field="digest"):
        return False
    if (
        expected_schema == "task_evaluation_scene_source_manifest.v1"
        and value.get("schema_version") == SOURCE
    ):
        config = value.get("authoring_inputs", {}).get("configuration", {})
        return (
            value.get("status") == "intake_ready"
            and value.get("blockers") == []
            and config.get("scene_id") == scene_id
        )
    if (
        expected_schema == "task_evaluation_scene_rights_admission.v1"
        and value.get("schema_version") == RIGHTS
    ):
        original = value.get("scene_id")
        return (
            bool(original)
            and scene_id in (original, original + "-development")
            and value.get("status") == "admitted_for_internal_development"
            and value.get("private_provider_processing_allowed") is True
            and value.get("provider_training_allowed") is False
            and value.get("public_redistribution_allowed") is False
            and value.get("physical_measurement_proven") is False
        )
    return False


def validate_join(source, rights, *, scene_id):
    """A website pair must share source, owner, task, consent and authority."""
    if source.get("schema_version") != SOURCE and rights.get("schema_version") != RIGHTS:
        return False
    from .task_evaluation_scene_intake import validate_request
    from .website_development_test import environment

    try:
        request = source["intake_request"]
        validate_request(request, now=request["consent"]["accepted_at_epoch"])
        environment(source)
        disclosure = rights["provider_disclosure"]
        valid = (
            reference_valid(
                source,
                expected_schema="task_evaluation_scene_source_manifest.v1",
                scene_id=scene_id,
            )
            and reference_valid(
                rights,
                expected_schema="task_evaluation_scene_rights_admission.v1",
                scene_id=scene_id,
            )
            and rights["preparation_digest"] == source["digest"]
            and rights["owner"] == request["owner"]
            and rights["consent"] == request["consent"]
            and rights["execution_authority"] == request["execution"]
            and rights["task_context_digest"] == source["binding"]["task_context_digest"]
            and disclosure.get("prepared_background_allowed") is True
            and disclosure.get("captured_frame_derivatives_allowed") is True
            and disclosure.get("raw_capture_video_allowed") is False
            and disclosure.get("provider_training_allowed") is False
            and disclosure.get("public_redistribution_allowed") is False
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("native_task_arena_website_claim_binding_invalid") from exc
    if not valid:
        raise ValueError("native_task_arena_website_claim_binding_invalid")
    return True
