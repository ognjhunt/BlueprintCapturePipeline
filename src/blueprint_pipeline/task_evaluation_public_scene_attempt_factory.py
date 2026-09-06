"""Materialize a public-scene attempt from retained owner/source/runtime contracts.

This factory calls the canonical CPU producers. It never installs publisher
assets, queries a model, uploads a source, or allocates a provider. All outputs
are development-only inputs and authorities, not scientific success evidence.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive
from .task_evaluation_scene_configuration_submission_inputs import (
    checked_file, read, require, sha, source_inputs, release_inputs,
)
from .task_evaluation_scene_owner_authority import (
    reopen_scene_intent, task_contract_projection, owner_numeric_task, descriptive_task_match,
)

BINDING_SCHEMA = "task_evaluation_public_source_binding.v1"
MACHINERY_SCHEMA = "task_evaluation_public_scene_machinery.v1"
RELEASE_SCHEMA = "task_evaluation_public_scene_release_binding.v1"
FACTORY_SCHEMA = "task_evaluation_public_scene_attempt_factory.v1"
SOURCE_ROLES = {"appearance_3dgs", "semantic_metadata", "scene_structure", "collision_usd", "publisher_scene_usdz"}
PURPOSES = ("exact_source_calibration_gpu_render", "released_code_segment_contribution_sweep",
            "configured_scene_partitioned_source_processing")
SOURCE_REFS = {"installation_receipt", "publisher_intake", "source_preparation_receipt",
               "destination_simready_result", "standard_splat_conversion_receipt",
               "interiorgs_terms", "interiorgs_readme", "sage_readme"}
PROVIDER_REFS = {"worker_stack_manifest", "runtime_image_build_receipt", "license_use_authorization",
                 "privacy_use_authorization", "trade_controls_review"}
PROVIDER_OPTIONS = {"runtime_image_identity", "method_version", "output_probability_threshold",
                    "max_num_objects", "multiplex_count", "use_fa3", "compile_model", "warm_up",
                    "async_loading_frames"}


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _reference(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"}
            and Path(ref["path"]).is_absolute(), "public_factory_reference_invalid")
    return checked_file(ref["path"], ref)


def _write(path, value):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "public_factory_output_unsafe")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        write_launch_preparation_record_exclusive(path, value)
    except FileExistsError:
        require(read(path) == value, "public_factory_immutable_conflict")
    return path


def _produce(function, output, **kwargs):
    """Canonical producers rerun to scratch when validating an interrupted step."""
    output = Path(output)
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        return function(output_path=output, **kwargs)
    with tempfile.TemporaryDirectory(prefix="factory-revalidate-", dir=output.parent) as scratch:
        value = function(output_path=Path(scratch) / output.name, **kwargs)
        require(read(output) == value, "public_factory_immutable_conflict")
        return value


def public_source_content_digest(installation):
    rows = [{key: row[key] for key in ("role", "sha256", "size_bytes")}
            for row in installation.get("files", []) if row.get("role") in SOURCE_ROLES]
    require(len(rows) == len(SOURCE_ROLES) and {r["role"] for r in rows} == SOURCE_ROLES,
            "public_factory_source_roles_invalid")
    return canonical_digest({"publisher_scene_id": str(installation["scene_id"]),
                             "assets": sorted(rows, key=lambda r: r["role"])})


def _prefix_candidates(binding, machinery, release, task):
    """Discover prior exact task jobs; a retained hint is optional, never opt-in."""
    from .task_evaluation_sam31_phase_queue import PHASES
    candidates = []
    if binding.get("prefix_candidate"):
        candidates.append(binding["prefix_candidate"])
    queue = Path(machinery.get("child_queue_root", "/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions"))
    registry = Path(machinery["profile_registry_root"])
    paths = sorted((queue / "completed").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = {(c["parent_request_digest"], c["source_plan"]["sha256"]) for c in candidates}
    discoveries = []
    for path in paths[:4096]:
        try:
            job = read(path, digest_field="job_digest")
            if job.get("phase") not in PHASES:
                continue
            plan_path = _reference(job["plan_ref"])
            plan = read(plan_path, digest_field="plan_digest")
            key = (job["parent_request_digest"], job["plan_ref"]["sha256"])
            if (key in seen or plan.get("task_identity") != task["task_identity"]
                    or str(plan.get("publisher_scene_id")) != str(binding["publisher_scene_id"])):
                continue
            profile = registry / (plan["server_profile_sha256"].removeprefix("sha256:") + ".json")
            if not profile.is_file() or sha(profile) != plan["server_profile_sha256"]:
                continue
            seen.add(key)
            discoveries.append((PHASES.index(job["phase"]), path.stat().st_mtime, {
                "source_plan": job["plan_ref"], "source_profile": record(profile),
                "parent_request_digest": job["parent_request_digest"],
                "sam31_billing_source": release.get("sam31_billing_source")}))
        except (OSError, ValueError, KeyError, TypeError):
            # Unrelated/invalid historical records are not reuse authority.
            continue
    discoveries.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return (candidates + [row[2] for row in discoveries])[:16]


def _current_conversion(*, refs, inputs, release, machinery, output):
    """Retain original raw files; only canonical local format conversion may rerun."""
    from .standard_splat_conversion import build_standard_splat_conversion_request, materialize_standard_splat_conversion
    old_path = _reference(refs["standard_splat_conversion_receipt"])
    old = read(old_path, digest_field="receipt_digest")
    original = inputs["raw"]["appearance_3dgs"]["path"]
    checked_file(original, old["source"])
    checked_file(old_path.parent / old["output"]["relative_path"], old["output"])
    if old.get("repository", {}).get("commit") == release["source_commit"]:
        return old_path
    source = {key: value for key, value in old["source"].items()
              if key not in {"source_bytes_unchanged", "source_gaussian_count"}}
    data = Path(machinery["preparation"]["server_data_root"])
    require(original.is_relative_to(data), "public_factory_raw_source_outside_data_root")
    source["relative_path"] = original.relative_to(data).as_posix()
    request = build_standard_splat_conversion_request({
        "schema_version": "standard_splat_conversion_request.v1", "program_id": "arm-decision-proof-v1",
        "frozen_before_conversion": True, "learned_policy_outcomes_observed": False,
        "source": source, "rights": old["rights"], "output_filename": Path(old["output"]["relative_path"]).name})
    request_path = _write(output / "conversion_request.json", request)
    converted = output / "conversion" / "standard_splat_conversion_receipt.v1.json"
    if not converted.exists():
        materialize_standard_splat_conversion(request_path=request_path, repo_root=release["repo_root"],
            data_root=data, output_root=converted.parent,
            production_runtime_root=machinery["preparation"]["runtime_root"])
    value = read(converted, digest_field="receipt_digest")
    require(value.get("request_digest") == request["request_digest"]
            and value.get("repository", {}).get("commit") == release["source_commit"],
            "public_factory_conversion_binding_changed")
    checked_file(converted.parent / value["output"]["relative_path"], value["output"])
    return converted


def _source_authorities(*, task, seed, refs, conversion_path, original_source, output, roots, commit):
    """Derive current execution scope while preserving the admitted publisher basis."""
    from .sam31_contribution_disclosure import validate_full_source_disclosure
    old_path = _reference(refs["standard_splat_conversion_receipt"])
    old = read(old_path, digest_field="receipt_digest")
    old_standard = old_path.parent / old["output"]["relative_path"]
    current = read(conversion_path, digest_field="receipt_digest")
    standard = conversion_path.parent / current["output"]["relative_path"]
    inherited = seed.get("human_authority", {})
    authorities = {}
    for purpose in PURPOSES:
        original_proof = validate_full_source_disclosure(task_authority=inherited,
            conversion_path=old_path, standard_splat_path=old_standard, original_source_path=original_source,
            expected_source_commit=old["repository"]["commit"], publisher_scene_id=task["publisher_scene_id"],
            approved_roots=roots, purpose=purpose)
        authority_ref = original_proof["disclosure_authority"]
        value = read(_reference(authority_ref), digest_field="authorization_digest")
        # The publisher permission/basis is retained verbatim and revalidated.
        # Owner consent supplies this attempt's execution permission only.
        value.update(source_commit=commit, authorized_by=task["human_authority"]["accepted_by"],
            authorized_on=task["human_authority"]["accepted_on"],
            authority_reference=task["human_authority"]["authority_reference"],
            original_disclosure_authority=authority_ref, scene_intent_authority=task["scene_intent_authority"])
        value["source_binding"].update(standard_splat_sha256=current["output"]["sha256"],
            standard_splat_size_bytes=current["output"]["size_bytes"],
            retained_gaussian_count=current["output"]["gaussian_count"])
        value["authorization_digest"] = canonical_digest(value, digest_field="authorization_digest")
        authorities[purpose] = record(_write(output / "source-authorities" / (purpose + ".json"), value))
    task["human_authority"].pop("full_source_provider_disclosure_authority", None)
    task["human_authority"]["full_source_provider_disclosure_authorities"] = authorities
    for purpose in PURPOSES:
        validate_full_source_disclosure(task_authority=task["human_authority"],
            conversion_path=conversion_path, standard_splat_path=standard, original_source_path=original_source,
            expected_source_commit=commit, publisher_scene_id=task["publisher_scene_id"],
            approved_roots=roots, purpose=purpose)


def materialize_public_scene_attempt(*, intent_path, source_binding_path, machinery_path,
                                     release_binding_path, output_root, attempt_id, now=None):
    """Build publication-ready inputs from an already reserved immutable attempt."""
    from .task_evaluation_scene_intake import _read as read_intake
    from .task_evaluation_release_retention import release_reference_lock
    from .task_evaluation_scene_configuration_submission import _validate_task, materialize_scene_configuration_submission
    from .sam31_provider_launch_packet import materialize_sam31_execution_authorization, materialize_sam31_provider_profile
    from .task_evaluation_sam31_preparation_profile import materialize_sam31_preparation_profile
    from .task_evaluation_sam31_preparation_review_authority import materialize_sam31_review_authority
    from .task_evaluation_sam31_profile_registry import register_sam31_profile
    from .task_evaluation_sam31_prefix_adoption import select_completed_prefix_adoption

    moment = time.time() if now is None else now
    intent_ref = record(intent_path)
    intent = reopen_scene_intent(intent_ref, now=moment)
    binding = read(source_binding_path, digest_field="binding_digest")
    machinery = read(machinery_path, digest_field="machinery_digest")
    release = read(release_binding_path, digest_field="release_digest")
    require(binding.get("schema_version") == BINDING_SCHEMA
            and machinery.get("schema_version") == MACHINERY_SCHEMA
            and release.get("schema_version") == RELEASE_SCHEMA, "public_factory_schema_invalid")
    request = intent["request"]
    require(request["source"]["kind"] == "public_scene"
            and request["source"]["binding_id"] == binding.get("binding_id")
            and request["source"]["content_digest"] == binding.get("source_content_digest")
            and binding.get("owner") == request["owner"]
            and binding.get("rights_reference") == request["consent"]["rights_reference"],
            "public_factory_source_or_owner_binding_mismatch")
    require(binding.get("status") == "admitted_for_private_processing", "public_factory_source_not_admitted")
    seed_ref = binding.get("accepted_task_seed")
    if seed_ref is None:
        return {"schema_version": FACTORY_SCHEMA, "status": "needs_input",
                "blockers": ["accepted_public_task_seed_required"], "provider_mutation_performed": False}
    seed = read(_reference(seed_ref))
    projection = task_contract_projection(seed)
    descriptive = None
    numeric_match = cross_runtime_canonical_digest(projection) == cross_runtime_canonical_digest(owner_numeric_task(request["task"]))
    if not numeric_match:
        descriptive = descriptive_task_match(owner_task=request["task"], seed=seed, source_binding=binding)
    if (binding.get("intent_task_digest") != cross_runtime_canonical_digest(request["task"])
            or (not numeric_match and descriptive is None)):
        return {"schema_version": FACTORY_SCHEMA, "status": "needs_input",
                "blockers": ["accepted_public_task_seed_does_not_match_owner_task"],
                "provider_mutation_performed": False}
    if descriptive is not None:
        authority = seed.get("success_contract_authority", {})
        proposal_ref = seed.get("task_parameter_provenance", {}).get("source_proposal")
        if authority.get("author_source") != "agent_proposal" or not proposal_ref:
            return {"schema_version": FACTORY_SCHEMA, "status": "needs_input",
                    "blockers": ["retained_task_parameter_proposal_required_for_descriptive_task"],
                    "provider_mutation_performed": False}
        proposal = read(_reference(proposal_ref), digest_field="proposal_digest")
        require(authority.get("agent_proposal") == proposal and authority.get("proposal_digest") == proposal["proposal_digest"]
                and proposal.get("schema_version") == "task_evaluation_task_parameter_proposal.v1"
                and proposal.get("status") == "proposal_only" and isinstance(proposal.get("success"), dict)
                and seed["task_parameter_provenance"].get("measured_thresholds_claimed") is False
                and all(seed["success"].get(key) == value for key, value in proposal["success"].items()),
                "public_factory_retained_numeric_proposal_changed")
    require(str(seed.get("publisher_scene_id")) == str(binding.get("publisher_scene_id"))
            and seed.get("appearance_removal_method") == "sam31", "public_factory_task_source_mismatch")
    require(set(binding.get("references", {})) == SOURCE_REFS, "public_factory_source_references_invalid")
    refs = binding["references"]
    paths = {name: _reference(ref) for name, ref in refs.items()}
    require(public_source_content_digest(read(paths["installation_receipt"], digest_field="receipt_digest"))
            == binding["source_content_digest"], "public_factory_source_content_changed")
    require(set(machinery.get("provider_references", {})) == PROVIDER_REFS
            and set(machinery.get("provider_options", {})) == PROVIDER_OPTIONS,
            "public_factory_provider_configuration_invalid")
    provider_paths = {key: _reference(ref) for key, ref in machinery["provider_references"].items()}
    require({"vast", "openai"}.issubset(request["execution"]["allowed_providers"]),
            "public_factory_provider_not_authorized")
    maximum = machinery.get("maximum_preparation_spend_usd")
    require(type(maximum) in (int, float) and 4.5 <= maximum <= request["execution"]["max_total_spend_usd"],
            "public_factory_preparation_cap_insufficient")
    require(isinstance(attempt_id, str) and attempt_id and "/" not in attempt_id and "\\" not in attempt_id,
            "public_factory_attempt_id_invalid")
    attempt_path = Path(intent_path).parent / "attempts" / (attempt_id + ".json")
    attempt = read_intake(attempt_path, "attempt_digest")
    commit = release["source_commit"]
    require(attempt.get("intent_digest") == intent["intent_digest"]
            and attempt.get("source_commit") == commit and attempt.get("runtime_digest") == release["runtime_digest"]
            and attempt.get("input_digest") == binding["binding_digest"]
            and attempt.get("provider") == "vast" and attempt.get("maximum_spend_usd") == maximum,
            "public_factory_attempt_binding_mismatch")
    if "robot_binding_id" in request["task"]:
        from .task_evaluation_controls_autoprovision import CATALOG_SCHEMA, _asset, payload_digest
        catalog = read(_reference(machinery["robot_catalog"]), digest_field="catalog_digest")
        require(catalog.get("schema_version") == CATALOG_SCHEMA, "public_factory_robot_catalog_invalid")
        robot = catalog["bindings"].get(request["task"]["robot_binding_id"])
        require(isinstance(robot, dict) and robot.get("expected_production_commit") == commit,
                "public_factory_robot_binding_mismatch")
        _asset(robot["robot_asset_usd"])
        _asset(robot["embodiment_camera_template"])
        require(payload_digest(Path(robot["runtime_source_payload_dir"])) == robot["runtime_digest"],
                "public_factory_robot_runtime_changed")
    if "episode_interpretation" in request["task"]:
        interpretation = request["task"]["episode_interpretation"]
        require(type(interpretation) is bool or (isinstance(interpretation, dict)
                and set(interpretation) == {"enabled"} and type(interpretation["enabled"]) is bool),
                "public_factory_episode_interpretation_invalid")
    release_paths = {key: _reference(release[key]) for key in (
        "deploy_receipt", "release_provenance", "release_environment")}
    release_inputs(deploy_path=release_paths["deploy_receipt"], provenance_path=release_paths["release_provenance"],
        publication_root=Path(release["runtime_publication_root"]), commit=commit,
        release_admission_mode=release["release_admission_mode"])
    output = Path(output_root)
    roots = tuple(Path(p) for p in machinery["preparation"]["approved_roots"])
    require(output.is_absolute() and not any(p.is_symlink() for p in (output, *output.parents))
            and any(output.is_relative_to(p) for p in roots)
            and not output.is_relative_to(Path(release["repo_root"])), "public_factory_output_root_invalid")
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    with release_reference_lock(output, exclusive=True):
        identity = {"intent": intent_ref, "attempt": record(attempt_path), "source_binding": record(source_binding_path),
                    "machinery": record(machinery_path), "release": record(release_binding_path)}
        identity_path = output / "factory_inputs.json"
        if identity_path.exists():
            retained_identity = read(identity_path)
            require(all(retained_identity.get(key) == value for key, value in identity.items())
                    and type(retained_identity.get("factory_started_at_epoch")) in (int, float),
                    "public_factory_immutable_conflict")
            identity = retained_identity
        else:
            identity["factory_started_at_epoch"] = moment
            _write(identity_path, identity)
        factory_moment = identity["factory_started_at_epoch"]
        task = deepcopy(seed)
        for key in ("robot_binding_id", "episode_interpretation"):
            if key in request["task"]:
                task[key] = request["task"][key]
        task["expected_production_commit"] = commit
        task["run_prefix"] = "scene-" + intent["intent_id"].removeprefix("scene-")[:20] + "-" + attempt_id
        task["scene_intent_authority"] = {"intent": intent_ref, "intent_digest": intent["intent_digest"],
                                          "attempt": record(attempt_path)}
        accepted_on = datetime.fromtimestamp(request["consent"]["accepted_at_epoch"], timezone.utc).isoformat()
        owner = task["human_authority"] = {}
        owner.update(accepted_by=request["owner"]["user_id"], accepted_on=accepted_on,
            authority_reference="scene-intent:" + intent["intent_digest"],
            private_derived_frame_disclosure_authorized=True, provider_retention_terms_accepted=True,
            provider_training_terms_accepted=True, provider_training_authorized=False,
            task_success_contract_confirmed=True, source_calibration_gpu_render_authorized=True,
            sam31_visual_review_authorized=True, sam31_visual_review_maximum_cost_usd=1.0)
        if descriptive is not None:
            task["task_identity"]["id"] = request["task"]["task_id"]
            task["owner_description_seed_binding"] = {"source_binding": record(source_binding_path), "match": descriptive}
        if task.get("success_contract_authority") is not None:
            task["success_contract_authority"].update(accepted_by=owner["accepted_by"],
                authority_reference=owner["authority_reference"], delegation_authority_reference=owner["authority_reference"],
                confirmed_by_team_id=task["team_namespace"])
        inputs = source_inputs(installation_path=paths["installation_receipt"], publisher_path=paths["publisher_intake"],
            preparation_path=paths["source_preparation_receipt"], task=task, commit=commit)
        conversion = _current_conversion(refs=refs, inputs=inputs, release=release, machinery=machinery, output=output)
        _source_authorities(task=task, seed=seed, refs=refs, conversion_path=conversion,
            original_source=inputs["raw"]["appearance_3dgs"]["path"], output=output, roots=roots, commit=commit)
        task.setdefault("source_input_references", {}).update({key: refs[key] for key in (
            "installation_receipt", "source_preparation_receipt", "destination_simready_result")})
        task["source_input_references"]["standard_splat_conversion_receipt"] = record(conversion)
        task.setdefault("configuration_provenance", {})["execution_release_rebinding"] = {
            "prior_task_request": seed_ref, "prior_execution_commit": seed.get("expected_production_commit"),
            "new_execution_commit": commit, "source_file_identities_unchanged": True,
            "prior_task_identity": seed["task_identity"], "current_task_identity": task["task_identity"],
            "numeric_task_proposal_reexecuted": False, "original_proposal_receipts_preserved": True}
        if "request_digest" in task:
            task["request_digest"] = canonical_digest(task, digest_field="request_digest")
        _validate_task(task)
        task_path = _write(output / "task_request.json", task)
        authorization_path = output / "sam31_execution_authorization.json"
        _produce(materialize_sam31_execution_authorization, authorization_path,
            source_commit_sha=commit, runtime_image_identity=machinery["provider_options"]["runtime_image_identity"],
            authorized_by=owner["accepted_by"], authorized_on=accepted_on, authority_reference=owner["authority_reference"])
        provider_path = output / "sam31_provider_profile.json"
        _produce(materialize_sam31_provider_profile, provider_path,
            **{key + "_path": path for key, path in provider_paths.items()},
            execution_authorization_path=authorization_path, source_commit_sha=commit, **machinery["provider_options"])
        terms = _reference(machinery["review_terms"])
        review_path = output / "sam31_review_authority.json"
        _produce(materialize_sam31_review_authority, review_path, task_request_path=task_path,
                 provider_terms_evidence_path=terms)
        adopted_path = None
        selection = {"status": "no_reusable_prefix", "rejected_candidates": [{"blocker": "no_retained_prefix_candidate"}]}
        candidates = _prefix_candidates(binding, machinery, release, task)
        best, best_kwargs = None, None
        selection_reports = []
        if candidates:
            require(release.get("provider_zero") is not None, "public_factory_prefix_reconciliation_required")
            zero_path = _reference(release["provider_zero"])
            from .task_evaluation_sam31_prefix_adoption import _zero, PREFIX_LENGTHS
            _zero(zero_path, at=factory_moment)
        for candidate in candidates:
            kwargs = dict(source_plan_path=_reference(candidate["source_plan"]),
                source_profile_path=_reference(candidate["source_profile"]),
                parent_request_digest=candidate["parent_request_digest"],
                current_host_inputs={"task_request": record(task_path), **{name: refs[name] for name in (
                    "installation_receipt", "publisher_intake", "source_preparation_receipt", "interiorgs_terms")}},
                current_provider_profile_path=provider_path, current_repo_root=release["repo_root"],
                expected_source_commit=commit, provider_zero_path=zero_path, approved_roots=roots,
                queue_root=machinery["child_queue_root"], parent_queue_root=machinery["parent_queue_root"],
                execution_root=machinery["execution_root"], now_epoch=factory_moment,
                sam31_billing_source_path=(_reference(candidate["sam31_billing_source"])
                    if candidate.get("sam31_billing_source") else None),
                release_binding_root=machinery["release_retention_binding_root"])
            result = select_completed_prefix_adoption(**kwargs, output_path=None)
            selection_reports.append(result)
            if result["status"] == "reusable_prefix_selected" and (best is None or
                    PREFIX_LENGTHS[result["through_phase"]] > PREFIX_LENGTHS[best["through_phase"]]):
                best, best_kwargs = result, kwargs
                if best["through_phase"] == "segment_cutout":
                    break
        if best is not None:
            adopted_path = output / "completed_prefix_adoption.json"
            if not adopted_path.exists():
                best = select_completed_prefix_adoption(**best_kwargs, output_path=adopted_path)
            else:
                require(read(adopted_path) == best["adoption"], "public_factory_adoption_changed")
            selection = {**best, "candidate_selections": selection_reports}
        elif selection_reports:
            selection = {"status": "no_reusable_prefix", "candidate_selections": selection_reports}
        _write(output / "prefix_selection.json", selection)
        preparation = dict(machinery["preparation"])
        preparation.update(source_commit=commit, repo_root=release["repo_root"], sam31_provider_profile_path=provider_path,
            sam31_review_rights_attestation_path=review_path, completed_prefix_adoption_path=adopted_path)
        profile = materialize_sam31_preparation_profile(**preparation)
        profile_path = _write(output / "sam31_preparation_profile.json", profile)
        registry = register_sam31_profile(profile_path=profile_path, registry_root=machinery["profile_registry_root"])
        submission_root = output / "submission"
        manifest_path = submission_root / "bundle_manifest.v1.json"
        if not manifest_path.exists():
            materialize_scene_configuration_submission(task_request_path=task_path,
                installation_receipt_path=paths["installation_receipt"], publisher_intake_path=paths["publisher_intake"],
                source_preparation_receipt_path=paths["source_preparation_receipt"],
                destination_simready_result_path=paths["destination_simready_result"],
                deploy_receipt_path=release_paths["deploy_receipt"], release_provenance_path=release_paths["release_provenance"],
                release_environment_path=release_paths["release_environment"],
                runtime_publication_root=release["runtime_publication_root"],
                rights_evidence={key: paths[key] for key in ("interiorgs_terms", "interiorgs_readme", "sage_readme")},
                staging_root=submission_root, expected_production_commit=commit,
                namespace_timestamp=release["namespace_timestamp"], sam31_server_profile_path=profile_path,
                sam31_completed_prefix_adoption_path=adopted_path, release_admission_mode=release["release_admission_mode"],
                scene_intent_digest=intent["intent_digest"])
        manifest = read(manifest_path, digest_field="manifest_digest")
        for row in manifest["files"]:
            checked_file(submission_root / row["relative_path"], {"sha256": row["digest"], "size_bytes": row["size_bytes"]})
        receipt = {"schema_version": FACTORY_SCHEMA, "status": "publication_ready", "identity": identity,
            "intent_digest": intent["intent_digest"], "attempt_digest": attempt["attempt_digest"], "source_commit": commit,
            "task_request": record(task_path), "sam31_provider_profile": record(provider_path),
            "sam31_preparation_profile": record(profile_path), "profile_registry": registry,
            "submission_manifest": record(manifest_path), "prefix_selection": record(output / "prefix_selection.json"),
            "submission_request": record(submission_root / "scene_configuration_preparation_request.v1.json"),
            "frozen_policy_candidates": request["execution"]["policy_candidates"],
            "original_source_reinstalled": False, "task_model_queried": False,
            "provider_mutation_performed": False, "source_uploaded": False, "claim_scope": "development_only"}
        receipt["factory_digest"] = canonical_digest(receipt, digest_field="factory_digest")
        _write(output / "factory_receipt.json", receipt)
        return receipt
