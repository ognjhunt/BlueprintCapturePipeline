"""Fresh owner/source handoff uses real factory producers without model/provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline import task_evaluation_public_scene_configuration_binding as binding
from blueprint_pipeline import task_evaluation_public_scene_attempt_factory as factory
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent, reserve_scene_attempt
from blueprint_pipeline.task_evaluation_scene_provider_terms import (
    accepted_provider_terms, derive_review_terms, validate_review_terms_binding,
)
from tests import test_task_evaluation_public_scene_attempt_factory as factory_fixtures
from tests.test_task_evaluation_public_scene_attempt_factory import write, ref


@pytest.fixture
def context(tmp_path, monkeypatch, request):
    return factory_fixtures.context.__wrapped__(tmp_path, monkeypatch, request)


def provider_terms():
    value = {"schema_version": "blueprint_private_scene_provider_terms.v1",
        "use_scope": "noncommercial_internal_research", "provider_training_authorized": False,
        "public_redistribution_authorized": False, "full_scene_reencoding_does_not_reduce_disclosure_scope": True,
        "providers": {"vast": {"purpose": "private full-scene source processing and simulator execution on owned rented resources"},
            "openai": {"purpose": "derived image editing and bounded authoring/review/supervision",
                       "api_training_opt_in": False, "standard_abuse_monitoring_retention_days": 30}}}
    value["terms_digest"] = canonical_digest(value, digest_field="terms_digest")
    return value


@pytest.mark.parametrize("context", [{"green_region": True}], indirect=True)
def test_fresh_source_binding_reaches_actual_factory_and_reuses_conversion(context, tmp_path, monkeypatch):
    args, source = context
    old_intent = json.loads(args["intent_path"].read_text())
    request = deepcopy(old_intent["request"])
    terms = provider_terms()
    terms_path = write(tmp_path / "provider-terms.json", terms)
    request["submission_id"] += "-fresh"
    request["consent"]["provider_terms_reference"] = terms["terms_digest"]
    intent_root = args["intent_path"].parent.parent
    intent = stage_scene_intent(value=request, queue_root=intent_root,
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"}, now=time.time())
    intent = json.loads((intent_root / intent["intent_id"] / "intent.json").read_text())
    prepared = {"intent_digest": intent["intent_digest"], "references": {
        "installation_receipt": ref(source["installation_receipt"]), "publisher_intake": ref(source["publisher_intake"]),
        "source_preparation_receipt": ref(source["source_preparation"]),
        **{key: ref(path) for key, path in source["rights_evidence"].items()}}}
    prepared_path = write(tmp_path / "public_source_preparation.json", prepared, "receipt_digest")
    release = json.loads(args["release_binding_path"].read_text())
    (Path(release["runtime_publication_root"]) / "splat-render" / release["source_commit"]).mkdir()
    machinery = json.loads(args["machinery_path"].read_text())
    key_file = tmp_path / "sam31-review.key"
    key_file.write_text("fixture-only-not-a-provider-key")
    key_file.chmod(0o600)
    preparation = machinery["preparation"]
    write(Path(preparation["sam31_review_cost_scope_attestation_path"]).parent / "openai_key_binding_sam31_visual_review.v1.json",
        {"schema_version": "openai_project_service_key_binding.v1", "paid_resource_class": "sam31_ai_visual_review",
         "project_id": preparation["openai_project_id"], "api_key_id": preparation["openai_api_key_id"], "key_file": str(key_file)})
    machinery["preparation"]["completed_review_execution_path"] = "/old-scene/review.json"
    for key in ("privacy_use_authorization", "trade_controls_review"):
        path = Path(machinery["provider_references"][key]["path"])
        value = json.loads(path.read_text())
        value.update(provider_id="vast", provider_country_allowlist=["US"], publisher_scene_id="other-scene")
        write(path, value, "receipt_digest")
        machinery["provider_references"][key] = ref(path)
    write(args["machinery_path"], machinery, "machinery_digest")
    prior_binding = json.loads(args["source_binding_path"].read_text())
    old_conversion = json.loads(Path(prior_binding["references"]["standard_splat_conversion_receipt"]["path"]).read_text())
    decoder_calls = []
    def decode(**kwargs):
        from blueprint_pipeline.standard_splat_conversion import build_standard_splat_conversion_request
        conversion_request = build_standard_splat_conversion_request(json.loads(Path(kwargs["request_path"]).read_text()))
        decoder_calls.append(conversion_request["request_digest"])
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True)
        raw = Path(conversion_request["source"]["relative_path"])
        standard = output / conversion_request["output_filename"]
        standard.write_bytes(raw.read_bytes())  # Hermetic decoder edge; three synthetic Gaussian rows.
        receipt = deepcopy(old_conversion)
        receipt["source"].update(conversion_request["source"])
        receipt["output"].update(relative_path=standard.name, **{k: ref(standard)[k] for k in ("sha256", "size_bytes")})
        write(Path(kwargs["receipt_output"]), receipt, "receipt_digest")
        return receipt
    from blueprint_pipeline import standard_splat_conversion
    monkeypatch.setattr(standard_splat_conversion, "materialize_standard_splat_conversion", decode)
    choice = {"publisher_scene_id": "841757", "binding_id": request["source"]["binding_id"],
              "source_content_digest": request["source"]["content_digest"]}
    config = {"machinery_path": str(args["machinery_path"]), "public_source_provider_terms_path": str(terms_path)}
    resolved = binding.bind_registered_public_configuration(intent=intent, choice=choice, config=config,
        release=release, prepared_path=prepared_path)
    assert resolved.status == "resolved" and len(decoder_calls) == 1
    active_machinery = json.loads(resolved.machinery_path.read_text())
    assert "completed_review_execution_path" not in active_machinery["preparation"]
    assert active_machinery["preparation"]["runtime_root"].endswith(release["source_commit"])
    assert active_machinery["preparation"]["openai_api_key_file"] == str(key_file)
    scope = json.loads(Path(active_machinery["preparation"]["sam31_review_cost_scope_attestation_path"]).read_text())
    assert scope["issued_by_agent"] is scope["derived_from_operator_scope_binding"] is True
    for key in ("privacy_use_authorization", "trade_controls_review"):
        scoped = json.loads(Path(active_machinery["provider_references"][key]["path"]).read_text())
        assert scoped["publisher_scene_id"] == "841757"
        assert scoped["intent_digest"] == intent["intent_digest"]
    repeated = binding.bind_registered_public_configuration(intent=intent, choice=choice, config=config,
        release=release, prepared_path=prepared_path)
    assert repeated == resolved and len(decoder_calls) == 1
    source_binding = json.loads(resolved.binding_path.read_text())
    reserve_scene_attempt(queue_root=intent_root, intent_id=intent["intent_id"], attempt_id="fresh-attempt",
        source_commit=release["source_commit"], runtime_digest=release["runtime_digest"],
        input_digest=source_binding["binding_digest"], provider="vast", maximum_spend_usd=4.5, now=time.time())
    result = factory.materialize_public_scene_attempt(intent_path=intent_root / intent["intent_id"] / "intent.json",
        source_binding_path=resolved.binding_path, machinery_path=resolved.machinery_path,
        release_binding_path=args["release_binding_path"], output_root=tmp_path / "fresh-factory", attempt_id="fresh-attempt")
    assert result["status"] == "publication_ready"
    request = json.loads(Path(result["submission_request"]["path"]).read_text())
    assert "surface_target" in request["task"] and "destination" not in request["task"]


def test_consent_projection_cannot_expand_terms_or_change_owner(tmp_path):
    terms = provider_terms()
    source = write(tmp_path / "terms.json", terms)
    intent = {"intent_digest": "sha256:"+"1"*64, "request": {"consent": {
        "provider_terms_reference": terms["terms_digest"], "accepted_by": "one-owner",
        "accepted_at_epoch": time.time(), "private_processing_authorized": True, "provider_training_authorized": False}}}
    accepted_provider_terms(intent, source)
    projected = derive_review_terms(intent=intent, provider_terms_path=source)
    path = write(tmp_path / "review.json", projected)
    validate_review_terms_binding(intent=intent, provider_terms_path=path)
    projected["accepted_by"] = "another-owner"
    write(path, projected, "attestation_digest")
    with pytest.raises(ValueError, match="projection_changed"):
        validate_review_terms_binding(intent=intent, provider_terms_path=path)
    terms["providers"]["vast"]["purpose"] = "derived frames only"
    write(source, terms, "terms_digest")
    intent["request"]["consent"]["provider_terms_reference"] = terms["terms_digest"]
    with pytest.raises(ValueError, match="full_source_processing_not_authorized"):
        accepted_provider_terms(intent, source)
