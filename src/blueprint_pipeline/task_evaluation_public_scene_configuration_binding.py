"""Join fresh publisher preparation to the existing immutable attempt factory.

Called only by the durable scene controller. All scene/task/owner values come
from its retained intent. Source conversion is local, content-addressed and
reused; this module does not allocate, invoke a model, or claim qualification.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_public_scene_attempt_factory import (
    BINDING_SCHEMA, MACHINERY_SCHEMA, PURPOSES, _write, record, source_reference_names,
)
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, source_inputs
from .task_evaluation_scene_provider_terms import (
    DEFAULT_TERMS_PATH, accepted_provider_terms, derive_review_terms,
)


def _owner(intent):
    consent = intent["request"]["consent"]
    return {"accepted_by": consent["accepted_by"],
        "accepted_on": datetime.fromtimestamp(consent["accepted_at_epoch"], timezone.utc).isoformat(),
        "authority_reference": "scene-intent:" + intent["intent_digest"],
        "private_derived_frame_disclosure_authorized": True, "provider_retention_terms_accepted": True,
        "provider_training_terms_accepted": True, "provider_training_authorized": False,
        "task_success_contract_confirmed": True, "source_calibration_gpu_render_authorized": True,
        "sam31_visual_review_authorized": True, "sam31_visual_review_maximum_cost_usd": 1.0}


def _seed(intent, choice, commit):
    task = intent["request"]["task"]
    require(task["strategy"] == "pick_and_place", "public_source_task_strategy_unsupported")
    # Retain the numerical owner projection exactly. Configuration adds derived
    # runtime constraints separately, never overwriting the owner's task.
    return {"schema_version": "task_evaluation_minimal_task_request.v1",
        "expected_production_commit": commit, "publisher_scene_id": choice["publisher_scene_id"],
        "team_namespace": "scene-" + intent["intent_id"].removeprefix("scene-")[:40],
        "run_prefix": "public-scene-" + choice["publisher_scene_id"],
        "scene_identity": {"id": "interiorgs-" + choice["publisher_scene_id"], "version": "v1"},
        "task_identity": {"id": task["task_id"], "version": "v1"},
        "output_identity": {"id": task["task_id"] + "-configured", "version": "v1"},
        "strategy": task["strategy"],
        **{key: deepcopy(task[key]) for key in ("subject", "support", "destination", "success")},
        "appearance_removal_method": "sam31", "resolved_seed": 1,
        "human_authority": _owner(intent)}


def _conversion(*, inputs, refs, root, release):
    from .standard_splat_conversion import build_standard_splat_conversion_request, materialize_standard_splat_conversion

    source = inputs["raw"]["appearance_3dgs"]
    conversions = root / "standard-conversions"
    # A compatible completed conversion survives a deployment. Verify both
    # bytes now; the attempt factory performs its current-release rebinding.
    for path in sorted(conversions.glob("*/standard_splat_conversion_receipt.v1.json")):
        receipt = read(path, digest_field="receipt_digest")
        checked_file(source["path"], receipt["source"])
        checked_file(path.parent / receipt["output"]["relative_path"], receipt["output"])
        return path
    output = conversions / release["source_commit"]
    failure = output / "standard_splat_conversion_failure.v1.json"
    require(not failure.exists(), "public_source_conversion_failure_requires_repair")
    conversion_request = build_standard_splat_conversion_request({
        "schema_version": "standard_splat_conversion_request.v1", "program_id": "arm-decision-proof-v1",
        "frozen_before_conversion": True, "learned_policy_outcomes_observed": False,
        "source": {"relative_path": str(source["path"]), "dataset": "spatialverse/InteriorGS",
            "revision": source["publisher_revision"], "license": "InteriorGS custom noncommercial research terms",
            "sha256": source["sha256"], "size_bytes": source["size_bytes"]},
        "rights": {"conversion_execution_location": "local_only", "raw_private_upload_authorized": False,
                   "training_authorized": False, "terms_digest": refs["interiorgs_terms"]["sha256"]},
        "output_filename": "standard_source.ply"})
    request_path = _write(root / "standard_conversion_request.json", conversion_request)
    receipt_path = output / "standard_splat_conversion_receipt.v1.json"
    materialize_standard_splat_conversion(request_path=request_path, repo_root=release["repo_root"],
        data_root=root, output_root=output, receipt_output=receipt_path,
        production_runtime_root=Path(release["runtime_publication_root"]) / "splat-render" / release["source_commit"])
    return receipt_path


def _source_authorities(*, seed, intent, conversion_path, refs, provider_terms_path, root):
    from .sam31_contribution_disclosure import AUTHORITY_SCHEMA, validate_full_source_disclosure

    conversion = read(conversion_path, digest_field="receipt_digest")
    source, output = conversion["source"], conversion["output"]
    owner = seed["human_authority"]
    original = Path(source["relative_path"])
    binding = {"publisher_scene_id": seed["publisher_scene_id"], "dataset": source["dataset"],
        "publisher_revision": source["revision"], "original_source_sha256": source["sha256"],
        "original_source_size_bytes": source["size_bytes"], "standard_splat_sha256": output["sha256"],
        "standard_splat_size_bytes": output["size_bytes"], "retained_gaussian_count": output["gaussian_count"],
        "source_gaussian_count": source["source_gaussian_count"],
        "publisher_terms_digest": conversion["rights"]["terms_digest"]}
    authorities = {}
    for purpose in PURPOSES:
        value = {"schema_version": AUTHORITY_SCHEMA, "status": "authorized",
            "authority_kind": "explicit_human_full_source_provider_processing",
            "authorized_by": owner["accepted_by"], "authorized_on": owner["accepted_on"],
            "authority_reference": owner["authority_reference"], "agent_accepted_terms": False,
            "artifact_prepared_by": "durable_scene_controller", "intent_digest": intent["intent_digest"],
            "source_commit": conversion["repository"]["commit"], "provider_id": "vast", "purpose": purpose,
            "source_binding": binding, "full_source_scene_content_upload_authorized": True,
            "private_provider_processing_authorized": True, "publisher_rights_permit_private_full_source_processing": True,
            "provider_retention_terms_accepted": True, "provider_training_terms_accepted": True,
            "format_conversion_does_not_reduce_disclosure_scope": True,
            "public_redistribution_authorized": False, "provider_training_authorized": False,
            "publisher_rights_basis": {"kind": "publisher_license_private_processing",
                "publisher_terms_evidence": refs["interiorgs_terms"],
                "private_processing_permission_evidence": record(provider_terms_path),
                "scope_explanation": "The authenticated owner accepted private full-scene processing on owned rented "
                    "resources for noncommercial research under the retained publisher terms. This records that "
                    "private-compute interpretation; it grants no public redistribution or new publisher permission."}}
        value["authorization_digest"] = canonical_digest(value, digest_field="authorization_digest")
        authorities[purpose] = record(_write(root / "source-authorities" / (purpose + ".json"), value))
    owner["full_source_provider_disclosure_authorities"] = authorities
    for purpose in PURPOSES:
        validate_full_source_disclosure(task_authority=owner, conversion_path=conversion_path,
            standard_splat_path=conversion_path.parent / output["relative_path"], original_source_path=original,
            expected_source_commit=conversion["repository"]["commit"], publisher_scene_id=seed["publisher_scene_id"],
            approved_roots=(root, provider_terms_path.parent), purpose=purpose)


def _machinery(*, intent, scene_id, base_path, provider_terms_path, root, release):
    base = read(base_path, digest_field="machinery_digest")
    require(base.get("schema_version") == MACHINERY_SCHEMA, "public_source_machinery_invalid")
    value = deepcopy(base)
    value.pop("retained_prefix_only_binding_ids", None)
    value["preparation"].pop("completed_review_execution_path", None)
    value["preparation"]["runtime_root"] = str(Path(release["runtime_publication_root"]) / "splat-render" / release["source_commit"])
    owner = _owner(intent)
    from .task_evaluation_supervisor.openai_cost_authority import derive_operator_scope_attestation
    preparation = value["preparation"]
    prior_scope = Path(preparation["sam31_review_cost_scope_attestation_path"])
    key_binding_path = prior_scope.parent / "openai_key_binding_sam31_visual_review.v1.json"
    key_binding = read(key_binding_path)
    require(key_binding.get("schema_version") == "openai_project_service_key_binding.v1"
            and key_binding.get("paid_resource_class") == "sam31_ai_visual_review"
            and key_binding.get("project_id") == preparation["openai_project_id"]
            and key_binding.get("api_key_id") == preparation["openai_api_key_id"]
            and Path(key_binding.get("key_file", "")).is_file(), "public_source_review_key_binding_invalid")
    preparation["openai_api_key_file"] = key_binding["key_file"]
    scope = derive_operator_scope_attestation(provider_id="openai", paid_resource_class="sam31_ai_visual_review",
        project_id=key_binding["project_id"], api_key_id=key_binding["api_key_id"], operator_id=owner["accepted_by"],
        exclusive_from=datetime.fromtimestamp(intent["request"]["consent"]["accepted_at_epoch"], timezone.utc)-timedelta(hours=1),
        exclusive_until=datetime.fromtimestamp(intent["request"]["execution"]["expires_at_epoch"], timezone.utc)+timedelta(days=1))
    preparation["sam31_review_cost_scope_attestation_path"] = str(_write(root / "sam31_review_cost_scope.json", scope))
    value["sam31_review_key_binding"] = record(key_binding_path)
    for role in ("privacy_use_authorization", "trade_controls_review"):
        ref = base["provider_references"][role]
        prior = read(checked_file(ref["path"], ref), digest_field="receipt_digest")
        require(prior.get("provider_country_allowlist") == ["US"] and prior.get("provider_id") == "vast",
                "public_source_provider_country_review_missing")
        scoped = deepcopy(prior)
        scoped.update(publisher_scene_id=scene_id,
            authorized_by=owner["accepted_by"], authorized_on=owner["accepted_on"],
            authority_reference=owner["authority_reference"], prior_authority=ref,
            processing_scope="Exact source-bound calibrated derivatives for private noncommercial research; no training.",
            review_basis="Same pinned checkpoint and US provider scope as the retained review; fresh authenticated owner consent.",
            intent_digest=intent["intent_digest"], artifact_prepared_by="durable_scene_controller")
        scoped["receipt_digest"] = canonical_digest(scoped, digest_field="receipt_digest")
        value["provider_references"][role] = record(_write(root / (role + ".json"), scoped))
    value["review_terms"] = record(_write(root / "sam31_review_terms.json",
        derive_review_terms(intent=intent, provider_terms_path=provider_terms_path)))
    value["registered_source_intent_digest"] = intent["intent_digest"]
    value["source_machinery"] = record(base_path)
    value["machinery_digest"] = canonical_digest(value, digest_field="machinery_digest")
    return _write(root / "machinery.json", value)


def bind_registered_public_configuration(*, intent, choice, config, release, prepared_path):
    from .task_evaluation_scene_progression import SourceResolution
    from .task_evaluation_public_scene_attempt_factory import materialize_public_scene_attempt

    root = prepared_path.parent
    terms_path = Path(config.get("public_source_provider_terms_path") or DEFAULT_TERMS_PATH)
    require(terms_path.is_file(), "public_source_accepted_provider_terms_missing")
    accepted_provider_terms(intent, terms_path)
    prepared = read(prepared_path, digest_field="receipt_digest")
    require(prepared["intent_digest"] == intent["intent_digest"], "public_source_owner_binding_mismatch")
    refs = deepcopy(prepared["references"])
    seed_path = root / "accepted_task_seed.json"
    if seed_path.exists():
        seed = read(seed_path)
    else:
        seed = _seed(intent, choice, release["source_commit"])
        inputs = source_inputs(installation_path=Path(refs["installation_receipt"]["path"]),
            publisher_path=Path(refs["publisher_intake"]["path"]),
            preparation_path=Path(refs["source_preparation_receipt"]["path"]), task=seed, commit=release["source_commit"])
        conversion_path = _conversion(inputs=inputs, refs=refs, root=root, release=release)
        _source_authorities(seed=seed, intent=intent, conversion_path=conversion_path,
            refs=refs, provider_terms_path=terms_path, root=root)
        seed["source_input_references"] = {"standard_splat_conversion_receipt": record(conversion_path)}
        _write(seed_path, seed)
    refs["standard_splat_conversion_receipt"] = seed["source_input_references"]["standard_splat_conversion_receipt"]
    if seed["destination"].get("kind") != "green_region":
        require(isinstance(seed["destination"].get("simready_result"), dict), "public_source_destination_asset_binding_required")
        refs["destination_simready_result"] = seed["destination"]["simready_result"]
    require(set(refs) == source_reference_names(seed), "public_source_reference_set_invalid")
    for ref in refs.values():
        checked_file(ref["path"], ref)
    binding = {"schema_version": BINDING_SCHEMA, "status": "admitted_for_private_processing",
        "binding_id": choice["binding_id"], "source_content_digest": choice["source_content_digest"],
        "publisher_scene_id": choice["publisher_scene_id"], "owner": intent["request"]["owner"],
        "rights_reference": intent["request"]["consent"]["rights_reference"],
        "intent_task_digest": cross_runtime_canonical_digest(intent["request"]["task"]),
        "accepted_task_seed": record(seed_path), "references": refs}
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    binding_path = _write(root / "source_binding.json", binding)
    machinery_path = _machinery(intent=intent, scene_id=choice["publisher_scene_id"], base_path=Path(config["machinery_path"]),
        provider_terms_path=terms_path, root=root / "configuration" / release["source_commit"], release=release)
    return SourceResolution("resolved", binding_path, machinery_path, materialize_public_scene_attempt,
                            analysis_reference=record(prepared_path))
