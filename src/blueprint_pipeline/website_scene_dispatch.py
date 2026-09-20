"""Resolve website-prepared derivatives for the existing scene progression worker.

ADP-030/040, day 28. Registration is local and grants no execution authority.
The existing authenticated owner intent, release and allocator own dispatch.
"""
from __future__ import annotations

import os
from pathlib import Path

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_public_scene_attempt_factory import record, RELEASE_SCHEMA
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
from .task_evaluation_scene_progression_state import require, safe_path
from . import task_evaluation_scene_intake as intake


def binding_root(config=None):
    config = config or {}
    configured = config.get("website_source_binding_root") or os.getenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT")
    if configured:
        return safe_path(configured)
    intent_root = config.get("intent_root") or os.getenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT")
    require(bool(intent_root), "website_scene_intake_root_missing")
    return safe_path(Path(intent_root).parent / "website-source-bindings")


def _index_path(root, request):
    return root / (cross_runtime_canonical_digest(request)[7:] + ".json")


def register_website_preparation(*, preparation_path, runtime_inputs_path, task_context_path, root, now):
    from .website_native_background import prepare_construction_stages, construction_rights_admission
    preparation = read(preparation_path, digest_field="digest")
    context = read(task_context_path, digest_field="context_digest")
    construction_rights_admission(preparation=preparation, task_context=context, now=now)
    prepare_construction_stages(runtime_inputs_path=Path(runtime_inputs_path), preparation_path=Path(preparation_path))
    value = {"schema_version": "website_scene_source_registration.v1",
        "request_digest": cross_runtime_canonical_digest(preparation["intake_request"]),
        "references": {"preparation": record(preparation_path), "runtime_inputs": record(runtime_inputs_path),
                       "task_context": record(task_context_path)},
        "provider_mutation_performed": False, "execution_authority_granted": False,
        "claim_ceiling": "development_only"}
    value["registration_digest"] = canonical_digest(value, digest_field="registration_digest")
    root = safe_path(root)
    require(root.is_absolute(), "website_source_root_invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    path = _index_path(root, preparation["intake_request"])
    if not path.exists():
        intake.write_exclusive(path, value)
    require(read(path, digest_field="registration_digest") == value, "website_source_registration_conflict")
    return record(path)


def resolve_website_source(*, intent, config):
    from .task_evaluation_scene_progression import SourceResolution
    path = _index_path(binding_root(config), intent["request"])
    if not path.is_file():
        return SourceResolution("awaiting_source", blockers=("website_prepared_source_pending",))
    registration = read(path, digest_field="registration_digest")
    require(registration.get("schema_version") == "website_scene_source_registration.v1"
            and registration.get("request_digest") == cross_runtime_canonical_digest(intent["request"])
            and registration.get("execution_authority_granted") is False, "website_source_registration_invalid")
    refs = registration["references"]
    for ref in refs.values():
        checked_file(ref["path"], ref)
    preparation = read(refs["preparation"]["path"], digest_field="digest")
    require(preparation["intake_request"] == intent["request"], "website_source_request_changed")
    from .website_native_background import prepare_construction_stages
    construction = prepare_construction_stages(runtime_inputs_path=Path(refs["runtime_inputs"]["path"]),
                                               preparation_path=Path(refs["preparation"]["path"]))
    files = {row["path"]: row["size_bytes"] for row in construction["references"]}
    files.update({ref["path"]: ref["size_bytes"] for ref in refs.values()})
    value = {"schema_version": "website_scene_source_binding.v1", "binding_id": intent["request"]["source"]["binding_id"],
        "source_content_digest": intent["request"]["source"]["content_digest"], "intent_digest": intent["intent_digest"],
        "task_digest": intent["task_content_digest"], "owner": intent["request"]["owner"],
        "references": refs, "registration": record(path), "physical_scale_measured": False,
        "required_staging_bytes": 2 * sum(files.values()) + 256 * 1024**2,
        "provider_mutation_performed": False, "claim_ceiling": "development_only"}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    output = safe_path(config["factory_output_root"]) / intent["intent_id"] / "website-source"
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    bound = output / (value["binding_digest"][7:] + ".json")
    if not bound.exists():
        intake.write_exclusive(bound, value)
    require(read(bound, digest_field="binding_digest") == value, "website_source_binding_conflict")
    machinery = config.get("website_source_machinery_path") or config.get("completed_source_machinery_path")
    require(bool(machinery), "website_source_machinery_missing")
    return SourceResolution("resolved", binding_path=bound, machinery_path=safe_path(machinery),
                            materializer=materialize_website_attempt)


def materialize_website_attempt(*, intent_path, source_binding_path, machinery_path,
                               release_binding_path, output_root, attempt_id, now=None):
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    from .task_evaluation_scene_preparation_attempts import preparation_attempt_path
    from .website_native_submission import materialize_website_submission
    intent = reopen_scene_intent(record(intent_path), now=now)
    binding = read(source_binding_path, digest_field="binding_digest")
    machinery = read(machinery_path, digest_field="machinery_digest")
    release = read(release_binding_path, digest_field="release_digest")
    require(binding.get("schema_version") == "website_scene_source_binding.v1"
            and binding.get("intent_digest") == intent["intent_digest"]
            and binding.get("owner") == intent["request"]["owner"]
            and binding.get("task_digest") == intent["task_content_digest"]
            and release.get("schema_version") == RELEASE_SCHEMA
            and machinery.get("schema_version") in {"task_evaluation_website_scene_machinery.v1",
                                                   "task_evaluation_completed_scene_machinery.v1"},
            "website_factory_binding_invalid")
    attempt = intake._read(preparation_attempt_path(Path(intent_path).parent, attempt_id), "attempt_digest")
    require(attempt.get("intent_digest") == intent["intent_digest"]
            and attempt.get("input_digest") == binding["binding_digest"]
            and attempt.get("source_commit") == release["source_commit"], "website_factory_attempt_mismatch")
    task = {"scene_intent_authority": record(intent_path), **binding["references"]}
    output = safe_path(output_root)
    require(output.is_absolute() and not output.is_relative_to(Path(release["repo_root"])), "website_factory_output_invalid")
    submission = output / "submission"
    materialize_website_submission(task=task, expected_production_commit=release["source_commit"],
        **{key + "_path": checked_file(release[key]["path"], release[key])
           for key in ("deploy_receipt", "release_provenance", "release_environment")},
        runtime_publication_root=release["runtime_publication_root"], namespace_timestamp=release["namespace_timestamp"],
        release_admission_mode=release["release_admission_mode"], staging_root=submission)
    result = {"schema_version": "website_scene_attempt_factory.v1", "status": "publication_ready",
        "intent_digest": intent["intent_digest"], "attempt_digest": attempt["attempt_digest"],
        "source_commit": release["source_commit"], "submission_manifest": record(submission / "bundle_manifest.v1.json"),
        "submission_request": record(submission / "scene_configuration_preparation_request.v1.json"),
        "frozen_policy_candidates": intent["request"]["execution"]["policy_candidates"],
        "physical_scale_measured": False, "physical_registration_proven": False,
        "provider_mutation_performed": False, "claim_scope": "development_only"}
    result["factory_digest"] = canonical_digest(result, digest_field="factory_digest")
    return result
