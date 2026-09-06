"""Real CPU producers build a reusable publication-ready development fixture."""
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline import task_evaluation_public_scene_attempt_factory as factory
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent, reserve_scene_attempt
from blueprint_pipeline.task_evaluation_scene_owner_authority import task_contract_projection
from blueprint_pipeline.task_evaluation_sam31_preparation_review_authority import TERMS
from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from tests import test_task_evaluation_sam31_preparation_profile as profile_fixtures
from tests import test_public_scene_removal_selection as source_fixtures
from tests.test_task_evaluation_scene_configuration_submission import production_fixture, SHA
from tests.test_task_evaluation_scene_intake import request as owner_request


def write(path, value, field=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return path


def replace_commit(value, commit):
    if isinstance(value, dict):
        return {key: replace_commit(v, commit) for key, v in value.items()}
    if isinstance(value, list):
        return [replace_commit(v, commit) for v in value]
    return commit if value == SHA else value


def ref(path):
    return factory.record(path)


@pytest.fixture
def context(tmp_path, monkeypatch, request):
    options = getattr(request, "param", {})
    machine_root = tmp_path / "machine"
    machine_root.mkdir()
    profile = profile_fixtures.inputs.__wrapped__(machine_root, monkeypatch)
    commit = profile["source_commit"]
    monkeypatch.setattr(source_fixtures, "_fixture", lambda root: production_fixture(root, room_topology=True))
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    source = source_fixtures._source_fixture(raw_root)
    seed = json.loads(source["task_request"].read_text())
    seed.update(expected_production_commit=commit, appearance_removal_method="sam31")
    installation = json.loads(source["installation_receipt"].read_text())
    raw_row = next(r for r in installation["files"] if r.get("role") == "appearance_3dgs")
    raw_path = source["installation_receipt"].parent / raw_row["relative_path"]
    conversion_root = tmp_path / "retained-conversion"
    conversion_root.mkdir()
    standard = conversion_root / "standard.ply"
    standard.write_bytes(raw_path.read_bytes())
    terms_ref = ref(source["rights_evidence"]["interiorgs_terms"])
    conversion = {"schema_version": "standard_splat_conversion_receipt.v1",
        "status": "standard_splat_conversion_materialized", "repository": {"commit": commit},
        "claim_ceiling": "local_format_conversion_only", "source": {
            **{key: raw_row[key] for key in ("sha256", "size_bytes")},
            "source_bytes_unchanged": True, "source_gaussian_count": 3,
            "dataset": "spatialverse/InteriorGS", "revision": "d" * 40, "license": "fixture-private"},
        "output": {**{key: ref(standard)[key] for key in ("sha256", "size_bytes")},
            "relative_path": standard.name, "gaussian_count": 3, "gaussian_count_preserved": True,
            "standard_3dgs_schema_validated": True},
        "rights": {"conversion_execution_location": "local_only", "raw_private_upload_authorized": False,
            "training_authorized": False, "terms_digest": terms_ref["sha256"]}}
    conversion_path = write(conversion_root / "standard_splat_conversion_receipt.v1.json", conversion, "receipt_digest")
    authorities = {}
    for purpose in factory.PURPOSES:
        authority = {"schema_version": "public_scene_full_source_provider_disclosure_authority.v1",
            "status": "authorized", "authority_kind": "explicit_human_full_source_provider_processing",
            "authorized_by": seed["human_authority"]["accepted_by"], "authorized_on": "2026-09-05",
            "authority_reference": "synthetic-publisher-permission-only", "agent_accepted_terms": False,
            "source_commit": commit, "provider_id": "vast", "purpose": purpose,
            "source_binding": {"publisher_scene_id": "841757", "dataset": "spatialverse/InteriorGS",
                "publisher_revision": "d" * 40, "original_source_sha256": raw_row["sha256"],
                "original_source_size_bytes": raw_row["size_bytes"], "standard_splat_sha256": ref(standard)["sha256"],
                "standard_splat_size_bytes": standard.stat().st_size, "retained_gaussian_count": 3,
                "source_gaussian_count": 3, "publisher_terms_digest": terms_ref["sha256"]},
            **{key: True for key in ("full_source_scene_content_upload_authorized", "private_provider_processing_authorized",
                "publisher_rights_permit_private_full_source_processing", "provider_retention_terms_accepted",
                "provider_training_terms_accepted", "format_conversion_does_not_reduce_disclosure_scope")},
            "public_redistribution_authorized": False, "provider_training_authorized": False,
            "publisher_rights_basis": {"kind": "publisher_license_private_processing",
                "scope_explanation": "Synthetic fixture; not real publisher admission.",
                "publisher_terms_evidence": terms_ref, "private_processing_permission_evidence": terms_ref}}
        authorities[purpose] = ref(write(tmp_path / (purpose + ".json"), authority, "authorization_digest"))
    seed["human_authority"]["full_source_provider_disclosure_authorities"] = authorities
    if options.get("descriptive") and not options.get("missing_proposal"):
        proposal = {"schema_version": "task_evaluation_task_parameter_proposal.v1",
                    "status": "proposal_only", "success": seed["success"], "fixture_only": True}
        proposal_path = write(tmp_path / "retained-proposal.json", proposal, "proposal_digest")
        seed["success_contract_authority"] = {"author_source": "agent_proposal", "confirmation_status": "confirmed",
            "agent_proposal": proposal, "proposal_digest": proposal["proposal_digest"],
            "author_id": "test-fixture-machine-seed"}
        seed["task_parameter_provenance"] = {"source_proposal": ref(proposal_path), "measured_thresholds_claimed": False}
    write(source["task_request"], seed)
    terms = write(tmp_path / "review-terms.json", {"schema_version": review.AI_RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_private_derived_visual_review", **TERMS,
        "accepted_by": "nijelhunt_1", "accepted_on": "2026-09-05", "human_authority_reference": "fixture-terms"}, "attestation_digest")
    now = time.time()
    owner = owner_request()
    owner["source"] = {"kind": "public_scene", "binding_id": "public-scene-1",
        "content_digest": factory.public_source_content_digest(installation)}
    owner["task"] = task_contract_projection(seed)
    if options.get("descriptive"):
        owner["task"]["task_id"] = "my-book-task"
        owner["task"].update({key: {"description": description, "authority": "owner_confirmed"}
            for key, description in {"subject": options.get("subject", "book"), "support": "TV cabinet",
                "destination": "tray", "success": options.get("success",
                    "Place the object fully inside the destination, release it, and move the gripper clear.")}.items()})
    owner["execution"].update(max_total_spend_usd=20, max_paid_attempts=4,
        allowed_providers=["vast", "openai"], expires_at_epoch=now + 3600)
    owner["consent"].update(accepted_at_epoch=now - 1, provider_terms_reference=ref(terms)["sha256"])
    intake_root = tmp_path / "intents"
    intent = stage_scene_intent(value=owner, queue_root=intake_root, authenticated_client="blueprint-webapp",
        trusted_clients={"blueprint-webapp"}, now=now)
    intent_path = intake_root / intent["intent_id"] / "intent.json"
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intake_root))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "blueprint-webapp")
    references = {"installation_receipt": ref(source["installation_receipt"]),
        "publisher_intake": ref(source["publisher_intake"]), "source_preparation_receipt": ref(source["source_preparation"]),
        "destination_simready_result": ref(source["destination_simready"]),
        "standard_splat_conversion_receipt": ref(conversion_path),
        **{key: ref(path) for key, path in source["rights_evidence"].items()}}
    binding = {"schema_version": factory.BINDING_SCHEMA, "status": "admitted_for_private_processing",
        "binding_id": "public-scene-1", "source_content_digest": owner["source"]["content_digest"],
        "publisher_scene_id": "841757", "owner": owner["owner"], "rights_reference": owner["consent"]["rights_reference"],
        "intent_task_digest": cross_runtime_canonical_digest(owner["task"]), "accepted_task_seed": ref(source["task_request"]),
        "references": references}
    binding_path = write(tmp_path / "binding.json", binding, "binding_digest")
    provider = json.loads(Path(profile["sam31_provider_profile_path"]).read_text())
    provider_refs = {"worker_stack_manifest": provider["worker_stack_manifest"],
        "runtime_image_build_receipt": provider["runtime_image_build_receipt"],
        **{new: provider["authorization_sources"][old] for new, old in (
            ("license_use_authorization", "license_use"), ("privacy_use_authorization", "privacy_use"),
            ("trade_controls_review", "trade_controls"))}}
    preparation = {key: value for key, value in profile.items() if key not in {
        "source_commit", "repo_root", "sam31_provider_profile_path", "sam31_review_rights_attestation_path"}}
    preparation.update(server_data_root=tmp_path, approved_roots=[tmp_path])
    machinery = {"schema_version": factory.MACHINERY_SCHEMA, "maximum_preparation_spend_usd": 4.5,
        "provider_references": {key: {k: row[k] for k in ("path", "sha256", "size_bytes")} for key, row in provider_refs.items()},
        "provider_options": {key: provider["compile" if key == "compile_model" else key] for key in factory.PROVIDER_OPTIONS},
        "preparation": {key: str(value) if isinstance(value, Path) else [str(p) for p in value] if key == "approved_roots" else value
                        for key, value in preparation.items()}, "review_terms": ref(terms),
        "profile_registry_root": str(tmp_path / "profile-registry")}
    machinery_path = write(tmp_path / "machinery.json", machinery, "machinery_digest")
    provenance = replace_commit(json.loads(source["release_provenance"].read_text()), commit)
    write(source["release_provenance"], provenance)
    deploy = replace_commit(json.loads(source["deploy_receipt"].read_text()), commit)
    deploy["release_provenance"]["sha256"] = ref(source["release_provenance"])["sha256"]
    write(source["deploy_receipt"], deploy)
    for kind in ("scene-configuration", "splat-render"):
        old = source["runtime_publication_root"] / kind / (SHA + ".publication.v1.json")
        write(old.with_name(commit + ".publication.v1.json"), replace_commit(json.loads(old.read_text()), commit), "receipt_digest")
    release = {"schema_version": factory.RELEASE_SCHEMA, "source_commit": commit,
        "runtime_digest": "sha256:" + "f" * 64, "repo_root": str(profile["repo_root"]),
        "runtime_publication_root": str(source["runtime_publication_root"]), "namespace_timestamp": "20260906T010000Z",
        "release_admission_mode": "promoted", **{key: ref(source[key]) for key in (
            "deploy_receipt", "release_provenance", "release_environment")}}
    release_path = write(tmp_path / "release.json", release, "release_digest")
    reserve_scene_attempt(queue_root=intake_root, intent_id=intent["intent_id"], attempt_id="attempt-1",
        source_commit=commit, runtime_digest=release["runtime_digest"], input_digest=binding["binding_digest"],
        provider="vast", maximum_spend_usd=4.5, now=now)
    return dict(intent_path=intent_path, source_binding_path=binding_path, machinery_path=machinery_path,
        release_binding_path=release_path, output_root=tmp_path / "factory", attempt_id="attempt-1"), source


def test_real_producers_materialize_then_revalidate_same_attempt_without_raw_reinstall(context, monkeypatch):
    args, source = context
    original_open = Path.open
    def no_secret_read(path, *a, **kw):
        assert path.name not in {"hf.secret", "admin.secret"}, "secret contents must never be read"
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, "open", no_secret_read)
    before = {key: source[key].read_bytes() for key in ("installation_receipt", "source_preparation", "task_request")}
    receipt = factory.materialize_public_scene_attempt(**args)
    assert receipt["status"] == "publication_ready"
    assert receipt["provider_mutation_performed"] is False
    assert receipt["original_source_reinstalled"] is False
    assert receipt["task_model_queried"] is False
    assert all(source[key].read_bytes() == value for key, value in before.items())
    task = json.loads(Path(receipt["task_request"]["path"]).read_text())
    assert task["human_authority"]["accepted_by"] == "u1"
    request = json.loads(Path(receipt["submission_request"]["path"]).read_text())
    assert request["scene_intent_digest"] == receipt["intent_digest"]
    assert len(receipt["frozen_policy_candidates"]) == 2
    assert factory.materialize_public_scene_attempt(**args) == receipt


def test_prior_queue_jobs_are_discovered_without_operator_prefix_opt_in(context, tmp_path):
    from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase
    args, _ = context
    receipt = factory.materialize_public_scene_attempt(**args)
    plan_path = args["output_root"] / "submission/configuration/sam31_preparation_plan.v1.json"
    plan = json.loads(plan_path.read_text())
    queue = tmp_path / "child-queue"
    queued = enqueue_sam31_phase(queue_root=queue, parent_preparation_id="prior-preparation",
        parent_request_digest="sha256:" + "a" * 64, expected_source_commit=receipt["source_commit"],
        plan_ref=ref(plan_path), phase="calibrated_views", inputs=plan["host_inputs"])
    job_path = Path(queued["job_path"])
    job_path.rename(queue / "completed" / job_path.name)
    binding = json.loads(args["source_binding_path"].read_text())
    machinery = json.loads(args["machinery_path"].read_text())
    machinery["child_queue_root"] = str(queue)
    candidates = factory._prefix_candidates(binding, machinery, {}, json.loads(Path(receipt["task_request"]["path"]).read_text()))
    assert len(candidates) == 1
    assert candidates[0]["source_plan"] == ref(plan_path)
    assert candidates[0]["parent_request_digest"] == "sha256:" + "a" * 64


def test_current_conversion_uses_canonical_cpu_producer_without_reinstalling_raw(tmp_path, monkeypatch):
    from tests.test_sam31_contribution_disclosure import converted_job
    job, _, _, raw_path, old_path = converted_job(tmp_path, monkeypatch)
    before = raw_path.read_bytes()
    old = json.loads(old_path.read_text())
    old["repository"]["commit"] = "b" * 40
    write(old_path, old, "receipt_digest")
    retained = old_path.read_bytes()
    data = tmp_path / "conversion-fixture/data"
    new_path = factory._current_conversion(refs={"standard_splat_conversion_receipt": ref(old_path)},
        inputs={"raw": {"appearance_3dgs": {"path": raw_path}}},
        release={"source_commit": job["expected_source_commit"], "repo_root": tmp_path / "conversion-fixture/repo"},
        machinery={"preparation": {"server_data_root": data, "runtime_root": None}}, output=data / "factory")
    assert new_path != old_path
    assert json.loads(new_path.read_text())["repository"]["commit"] == job["expected_source_commit"]
    assert raw_path.read_bytes() == before and old_path.read_bytes() == retained


@pytest.mark.parametrize("context", [{"descriptive": True}], indirect=True)
def test_gui_descriptions_bind_the_existing_numeric_seed_without_owner_measurement_claim(context):
    args, _ = context
    receipt = factory.materialize_public_scene_attempt(**args)
    assert receipt["status"] == "publication_ready"
    task = json.loads(Path(receipt["task_request"]["path"]).read_text())
    match = task["owner_description_seed_binding"]["match"]
    assert match["owner_task"]["subject"]["description"] == "book"
    assert match["matches"]["subject"]["source_instance_id"] == "115"
    assert match["numeric_parameters_owner_measured"] is False
    assert task["success_contract_authority"]["author_source"] == "agent_proposal"
    assert task["success_contract_authority"]["accepted_by"] == "u1"
    assert task["task_identity"]["id"] == "my-book-task"
    assert match["administrative_task_id_rebinding"] is True
    assert match["original_seed_task_id"] == "scene-841757-book-to-tray"
    assert task["task_parameter_provenance"]["measured_thresholds_claimed"] is False


@pytest.mark.parametrize("context", [
    {"descriptive": True, "subject": "sofa"},
    {"descriptive": True, "success": "Place the object fully inside the destination and never touch the table."},
    {"descriptive": True, "missing_proposal": True},
], indirect=True)
def test_unmatched_descriptions_or_missing_numeric_provenance_need_input(context):
    args, _ = context
    result = factory.materialize_public_scene_attempt(**args)
    assert result["status"] == "needs_input"
    assert not args["output_root"].exists()


@pytest.mark.parametrize("fault", ["task", "source", "owner", "attempt", "revoked"])
def test_changed_source_task_owner_or_attempt_cannot_materialize(context, fault):
    args, source = context
    if fault == "revoked":
        (args["intent_path"].parent / "revoked.json").write_text('{}')
    elif fault == "attempt":
        args = {**args, "attempt_id": "missing-attempt"}
    elif fault == "source":
        installation = json.loads(source["installation_receipt"].read_text())
        path = source["installation_receipt"].parent / next(r for r in installation["files"] if r.get("role") == "appearance_3dgs")["relative_path"]
        path.chmod(0o640)
        path.write_bytes(b"changed original source")
    else:
        binding = json.loads(args["source_binding_path"].read_text())
        if fault == "owner":
            binding["owner"]["user_id"] = "another-owner"
        else:
            binding["intent_task_digest"] = "sha256:" + "0" * 64
        write(args["source_binding_path"], binding, "binding_digest")
    if fault == "task":
        assert factory.materialize_public_scene_attempt(**args)["status"] == "needs_input"
    else:
        with pytest.raises(ValueError):
            factory.materialize_public_scene_attempt(**args)
    assert not (args["output_root"] / "submission").exists()
