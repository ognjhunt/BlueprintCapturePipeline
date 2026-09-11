"""Bind one authenticated consent to the exact private-processing terms it accepted."""
from __future__ import annotations

from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, sha

DEFAULT_TERMS_PATH = "/etc/blueprint/task-evaluation-private-scene-provider-terms.json"


def accepted_provider_terms(intent, path):
    terms = read(path, digest_field="terms_digest")
    consent = intent["request"]["consent"]
    require(terms.get("schema_version") == "blueprint_private_scene_provider_terms.v1"
            and terms["terms_digest"] == consent["provider_terms_reference"]
            and terms.get("use_scope") == "noncommercial_internal_research"
            and terms.get("provider_training_authorized") is False
            and terms.get("public_redistribution_authorized") is False
            and consent["private_processing_authorized"] is True
            and consent["provider_training_authorized"] is False,
            "scene_provider_terms_scope_mismatch")
    providers = terms.get("providers", {})
    require(providers.get("vast", {}).get("purpose") ==
            "private full-scene source processing and simulator execution on owned rented resources"
            and terms.get("full_scene_reencoding_does_not_reduce_disclosure_scope") is True,
            "scene_full_source_processing_not_authorized")
    openai = providers.get("openai", {})
    require(openai.get("purpose") == "derived image editing and bounded authoring/review/supervision"
            and openai.get("api_training_opt_in") is False
            and openai.get("standard_abuse_monitoring_retention_days") == 30,
            "scene_derived_review_terms_not_authorized")
    return terms


def derive_review_terms(*, intent, provider_terms_path):
    from .public_scene_sam31_track_selection_review import AI_RIGHTS_SCHEMA_VERSION
    from .task_evaluation_sam31_preparation_review_authority import TERMS
    from .task_evaluation_public_scene_attempt_factory import record
    from datetime import datetime, timezone

    accepted_provider_terms(intent, provider_terms_path)
    consent = intent["request"]["consent"]
    terms = {"schema_version": AI_RIGHTS_SCHEMA_VERSION, **TERMS,
             "status": "accepted_for_private_derived_visual_review",
             "accepted_by": consent["accepted_by"],
             "accepted_on": datetime.fromtimestamp(consent["accepted_at_epoch"], timezone.utc).isoformat(),
             "human_authority_reference": "scene-intent:" + intent["intent_digest"],
             "source_provider_terms": record(provider_terms_path),
             "intent_digest": intent["intent_digest"],
             "artifact_prepared_by": "durable_scene_controller"}
    terms["attestation_digest"] = canonical_digest(terms, digest_field="attestation_digest")
    return terms


def validate_review_terms_binding(*, intent, provider_terms_path):
    path = Path(provider_terms_path)
    if intent["request"]["consent"]["provider_terms_reference"] == sha(path):
        return  # Legacy consent directly pinned the standalone review terms.
    projected = read(path, digest_field="attestation_digest")
    ref = projected.get("source_provider_terms")
    require(isinstance(ref, dict), "scene_owner_provider_terms_not_bound")
    source = checked_file(ref["path"], ref)
    require(projected == derive_review_terms(intent=intent, provider_terms_path=source),
            "scene_owner_provider_terms_projection_changed")
