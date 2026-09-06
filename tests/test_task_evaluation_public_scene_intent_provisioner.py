"""The public-scene intent provisioner transforms a scene's already-retained
artifacts into a persistent public_scene intent + public-source binding +
public-scene machinery, and the REAL scene-progression worker then re-prepares
the scene to publication_ready with no provider allocation.

Every producer here is a real CPU producer (the same ones the public-scene
attempt factory fixture uses); the provisioner is exercised against real
retained artifacts and its output is consumed by the real progression engine.
No network, no 841757 host access, no provider allocation.
"""
import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_public_scene_intent_provisioner as provisioner
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from blueprint_pipeline import task_evaluation_public_scene_attempt_factory as factory
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_sam31_preparation_review_authority import TERMS
from blueprint_pipeline.task_evaluation_scene_owner_authority import task_contract_projection
from tests import test_task_evaluation_sam31_preparation_profile as profile_fixtures
from tests import test_public_scene_removal_selection as source_fixtures
from tests.test_task_evaluation_scene_configuration_submission import production_fixture, SHA
from tests.test_task_evaluation_scene_intake import request as owner_request


def _write(path, value, field=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return path


def _replace_commit(value, commit):
    if isinstance(value, dict):
        return {key: _replace_commit(v, commit) for key, v in value.items()}
    if isinstance(value, list):
        return [_replace_commit(v, commit) for v in value]
    return commit if value == SHA else value


def _ref(path):
    return factory.record(path)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    """Produce the scene's already-retained artifacts with real CPU producers.

    Returns (retained_paths, owner_authority, extras). The provisioner is given
    ONLY paths to these retained artifacts plus the owner's standing authority;
    it never sees a pre-assembled binding/machinery/intent.
    """
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
    terms_ref = _ref(source["rights_evidence"]["interiorgs_terms"])
    conversion = {"schema_version": "standard_splat_conversion_receipt.v1",
        "status": "standard_splat_conversion_materialized", "repository": {"commit": commit},
        "claim_ceiling": "local_format_conversion_only", "source": {
            **{key: raw_row[key] for key in ("sha256", "size_bytes")},
            "source_bytes_unchanged": True, "source_gaussian_count": 3,
            "dataset": "spatialverse/InteriorGS", "revision": "d" * 40, "license": "fixture-private"},
        "output": {**{key: _ref(standard)[key] for key in ("sha256", "size_bytes")},
            "relative_path": standard.name, "gaussian_count": 3, "gaussian_count_preserved": True,
            "standard_3dgs_schema_validated": True},
        "rights": {"conversion_execution_location": "local_only", "raw_private_upload_authorized": False,
            "training_authorized": False, "terms_digest": terms_ref["sha256"]}}
    conversion_path = _write(conversion_root / "standard_splat_conversion_receipt.v1.json", conversion, "receipt_digest")

    # Retained private-source disclosure authorities (one per execution purpose).
    for purpose in factory.PURPOSES:
        authority = {"schema_version": "public_scene_full_source_provider_disclosure_authority.v1",
            "status": "authorized", "authority_kind": "explicit_human_full_source_provider_processing",
            "authorized_by": seed["human_authority"]["accepted_by"], "authorized_on": "2026-09-05",
            "authority_reference": "synthetic-publisher-permission-only", "agent_accepted_terms": False,
            "source_commit": commit, "provider_id": "vast", "purpose": purpose,
            "source_binding": {"publisher_scene_id": "841757", "dataset": "spatialverse/InteriorGS",
                "publisher_revision": "d" * 40, "original_source_sha256": raw_row["sha256"],
                "original_source_size_bytes": raw_row["size_bytes"], "standard_splat_sha256": _ref(standard)["sha256"],
                "standard_splat_size_bytes": standard.stat().st_size, "retained_gaussian_count": 3,
                "source_gaussian_count": 3, "publisher_terms_digest": terms_ref["sha256"]},
            **{key: True for key in ("full_source_scene_content_upload_authorized", "private_provider_processing_authorized",
                "publisher_rights_permit_private_full_source_processing", "provider_retention_terms_accepted",
                "provider_training_terms_accepted", "format_conversion_does_not_reduce_disclosure_scope")},
            "public_redistribution_authorized": False, "provider_training_authorized": False,
            "publisher_rights_basis": {"kind": "publisher_license_private_processing",
                "scope_explanation": "Synthetic fixture; not real publisher admission.",
                "publisher_terms_evidence": terms_ref, "private_processing_permission_evidence": terms_ref}}
        seed["human_authority"].setdefault("full_source_provider_disclosure_authorities", {})[purpose] = _ref(
            _write(tmp_path / (purpose + ".json"), authority, "authorization_digest"))
    _write(source["task_request"], seed)

    review_terms = _write(tmp_path / "review-terms.json", {"schema_version": review.AI_RIGHTS_SCHEMA_VERSION,
        "status": "accepted_for_private_derived_visual_review", **TERMS,
        "accepted_by": "nijelhunt_1", "accepted_on": "2026-09-05", "human_authority_reference": "fixture-terms"},
        "attestation_digest")

    # The retained SAM provider profile + the retained SAM preparation config.
    provider_profile_path = Path(profile["sam31_provider_profile_path"])
    preparation = {key: str(value) if isinstance(value, Path) else value
                   for key, value in profile.items() if key not in {
                       "source_commit", "repo_root", "sam31_provider_profile_path",
                       "sam31_review_rights_attestation_path"}}
    preparation.update(server_data_root=str(tmp_path), approved_roots=[str(tmp_path)])
    sam_preparation_config_path = _write(tmp_path / "sam-preparation-config.json", preparation)

    # The exact release binding (a real release producer output).
    provenance = _replace_commit(json.loads(source["release_provenance"].read_text()), commit)
    _write(source["release_provenance"], provenance)
    deploy = _replace_commit(json.loads(source["deploy_receipt"].read_text()), commit)
    deploy["release_provenance"]["sha256"] = _ref(source["release_provenance"])["sha256"]
    _write(source["deploy_receipt"], deploy)
    for kind in ("scene-configuration", "splat-render"):
        old = source["runtime_publication_root"] / kind / (SHA + ".publication.v1.json")
        _write(old.with_name(commit + ".publication.v1.json"),
               _replace_commit(json.loads(old.read_text()), commit), "receipt_digest")
    release = {"schema_version": factory.RELEASE_SCHEMA, "source_commit": commit,
        "runtime_digest": "sha256:" + "f" * 64, "repo_root": str(profile["repo_root"]),
        "runtime_publication_root": str(source["runtime_publication_root"]), "namespace_timestamp": "20260906T010000Z",
        "release_admission_mode": "promoted", **{key: _ref(source[key]) for key in (
            "deploy_receipt", "release_provenance", "release_environment")}}
    release_path = _write(tmp_path / "release.json", release, "release_digest")

    installation_fresh = json.loads(source["installation_receipt"].read_text())
    now = time.time()
    base = owner_request()
    execution = {**base["execution"], "max_total_spend_usd": 20, "max_paid_attempts": 4,
                 "allowed_providers": ["vast", "openai"], "expires_at_epoch": now + 3600}
    # An honest owner records acceptance of exactly the retained review terms:
    # consent.provider_terms_reference is the sha256 of the retained review-terms
    # file. The provisioner VALIDATES this equality; it never manufactures it.
    consent = {**base["consent"], "accepted_at_epoch": now - 1,
               "provider_terms_reference": factory.record(review_terms)["sha256"]}
    owner_authority = {"owner": base["owner"], "submission_id": base["submission_id"],
                       "execution": execution, "consent": consent}

    retained_paths = {
        "accepted_task_request": source["task_request"],
        "installation_receipt": source["installation_receipt"],
        "publisher_intake": source["publisher_intake"],
        "source_preparation_receipt": source["source_preparation"],
        "destination_simready_result": source["destination_simready"],
        "standard_splat_conversion_receipt": conversion_path,
        "interiorgs_terms": source["rights_evidence"]["interiorgs_terms"],
        "interiorgs_readme": source["rights_evidence"]["interiorgs_readme"],
        "sage_readme": source["rights_evidence"]["sage_readme"],
        "sam_provider_profile": provider_profile_path,
        "sam_preparation_config": sam_preparation_config_path,
        "review_terms": review_terms,
        "release_binding": release_path,
    }
    extras = {"commit": commit, "content_digest": factory.public_source_content_digest(installation_fresh),
              "seed_task": task_contract_projection(seed), "now": now}
    return retained_paths, owner_authority, extras


def _provision(tmp_path, retained_paths, owner_authority, *, now, **overrides):
    intents = tmp_path / "intents"
    bindings = tmp_path / "public-source-bindings"
    machinery_out = tmp_path / "config" / "task-evaluation-public-scene-machinery.json"
    kwargs = dict(retained=retained_paths, owner_authority=owner_authority, binding_id="public-scene-841757",
        profile_registry_root=str(tmp_path / "profile-registry"), maximum_preparation_spend_usd=4.5,
        intent_root=intents, public_source_binding_root=bindings, machinery_output_path=machinery_out,
        now=now)
    kwargs.update(overrides)
    return provisioner.provision_public_scene_intent(**kwargs), intents, bindings, machinery_out


def _progression_config(tmp_path, intents, bindings, machinery_out, release_path, *, source_kinds=None):
    config = dict(schema_version=engine.CONFIG_SCHEMA, intent_root=str(intents),
        public_source_binding_root=str(bindings), machinery_path=str(machinery_out),
        release_binding_path=str(release_path),
        factory_output_root=str(tmp_path / "progression-output"), trusted_clients=["blueprint-webapp"],
        submission_enabled=False)
    if source_kinds is not None:
        config["supported_source_kinds"] = source_kinds
    return _write(tmp_path / "progression-config.json", config, "config_digest")


def _bind_runtime_env(monkeypatch, intents, commit):
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intents))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "blueprint-webapp")
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: commit)


def test_provisioned_intent_reaches_publication_ready_without_allocation(retained, tmp_path, monkeypatch):
    retained_paths, owner_authority, extras = retained
    result, intents, bindings, machinery_out = _provision(
        tmp_path, retained_paths, owner_authority, now=extras["now"])

    # The provisioner returns the provisioned identity and the paths it wrote.
    assert result["status"] == "public_scene_intent_provisioned"
    assert result["binding_id"] == "public-scene-841757"
    assert result["intent_id"].startswith("scene-")
    assert result["provider_mutation_performed"] is False
    assert Path(result["binding_path"]).is_file()
    assert Path(result["machinery_path"]).is_file()
    assert (intents / result["intent_id"] / "intent.json").is_file()

    # The binding + intent bind the retained content, not fabricated identity.
    binding = json.loads(Path(result["binding_path"]).read_text())
    assert binding["schema_version"] == factory.BINDING_SCHEMA
    assert binding["source_content_digest"] == extras["content_digest"]
    assert set(binding["references"]) == factory.SOURCE_REFS
    intent = json.loads((intents / result["intent_id"] / "intent.json").read_text())
    assert intent["request"]["source"] == {"kind": "public_scene", "binding_id": "public-scene-841757",
                                            "content_digest": extras["content_digest"]}
    assert intent["request"]["task"] == extras["seed_task"]

    machinery = json.loads(Path(result["machinery_path"]).read_text())
    assert machinery["schema_version"] == factory.MACHINERY_SCHEMA
    assert set(machinery["provider_references"]) == factory.PROVIDER_REFS
    assert set(machinery["provider_options"]) == factory.PROVIDER_OPTIONS

    # The REAL scene-progression worker re-prepares the scene to publication_ready.
    _bind_runtime_env(monkeypatch, intents, extras["commit"])
    config = _progression_config(tmp_path, intents, bindings, machinery_out, retained_paths["release_binding"])
    run = engine.process_scene_intents(config_path=config)
    assert run["provider_allocation_performed"] is False
    assert run["results"][0]["status"] == "awaiting_execution", run
    assert run["results"][0]["phase"] == "publication_ready", run

    # Idempotent: a second progression pass is byte-identical, and re-provisioning
    # the same retained scene does not mutate the binding.
    assert engine.process_scene_intents(config_path=config) == run
    again, *_ = _provision(tmp_path, retained_paths, owner_authority, now=extras["now"])
    assert again["binding_digest"] == result["binding_digest"]
    assert again["intent_id"] == result["intent_id"]


@pytest.mark.parametrize("absent, expected", [
    ("installation_receipt", "public_scene_source_evidence_absent"),
    ("source_preparation_receipt", "public_scene_source_evidence_absent"),
    ("standard_splat_conversion_receipt", "public_scene_source_evidence_absent"),
    ("accepted_task_request", "public_scene_task_authority_absent"),
    ("interiorgs_terms", "public_scene_rights_absent"),
    ("sam_provider_profile", "public_scene_provider_machinery_absent"),
    ("release_binding", "public_scene_release_absent"),
])
def test_absent_retained_artifact_fails_closed(retained, tmp_path, absent, expected):
    retained_paths, owner_authority, extras = retained
    retained_paths[absent] = retained_paths[absent].parent / "does-not-exist.json"
    with pytest.raises(provisioner.PublicSceneIntentProvisionError, match=expected):
        _provision(tmp_path, retained_paths, owner_authority, now=extras["now"])
    assert not (tmp_path / "public-source-bindings").exists() or not list(
        (tmp_path / "public-source-bindings").glob("*.json"))


def test_conversion_receipt_from_another_release_is_accepted_content_bound(retained, tmp_path):
    # Content identity, not commit identity: a structurally valid conversion recorded
    # at a different (real) release is accepted. The standard-splat converter needs the
    # 3DGS decoder present only in the SAM preparation, so the exact-release rebind is
    # left to that decoder-equipped preparation; a deploy must not invalidate the
    # retained per-scene document (piece 1).
    retained_paths, owner_authority, extras = retained
    conversion = json.loads(retained_paths["standard_splat_conversion_receipt"].read_text())
    conversion["repository"]["commit"] = "a" * 40  # a different, real release commit
    _write(retained_paths["standard_splat_conversion_receipt"], conversion, "receipt_digest")
    # Not refused: the provisioner accepts the content-valid conversion and provisions.
    result = _provision(tmp_path, retained_paths, owner_authority, now=extras["now"])
    assert result


def test_structurally_invalid_conversion_still_fails_closed(retained, tmp_path):
    retained_paths, owner_authority, extras = retained
    conversion = json.loads(retained_paths["standard_splat_conversion_receipt"].read_text())
    conversion["repository"]["commit"] = "not-a-real-commit"  # not 40 hex → structurally invalid
    _write(retained_paths["standard_splat_conversion_receipt"], conversion, "receipt_digest")
    with pytest.raises(provisioner.PublicSceneIntentProvisionError, match="public_scene_source_evidence_absent"):
        _provision(tmp_path, retained_paths, owner_authority, now=extras["now"])


def test_owner_consent_with_unaccepted_provider_terms_is_refused(retained, tmp_path):
    # A5: the provisioner must NOT rewrite the owner's accepted provider-terms
    # reference to match the retained review file. If the owner's recorded consent
    # references different terms than the retained review-terms bytes, we have no
    # evidence they accepted these provider terms; the provisioner fails closed
    # and stages no intent, rather than fabricating acceptance.
    retained_paths, owner_authority, extras = retained
    owner_authority["consent"]["provider_terms_reference"] = "sha256:" + "9" * 64  # not the review terms
    with pytest.raises(provisioner.PublicSceneIntentProvisionError,
                       match="public_scene_provider_terms_not_accepted"):
        _provision(tmp_path, retained_paths, owner_authority, now=extras["now"])
    assert not (tmp_path / "intents").exists() or not list((tmp_path / "intents").glob("scene-*"))
    assert not (tmp_path / "public-source-bindings").exists() or not list(
        (tmp_path / "public-source-bindings").glob("*.json"))


def test_provisioning_writes_dependencies_before_publishing_the_intent(retained, tmp_path):
    # A11: the persistent intent is the signal the scene-progression worker acts
    # on; it must never become visible unless its immutable binding + machinery
    # are already durably written. Force an immutable conflict on the binding
    # write and assert no intent was staged (dependencies precede publication).
    retained_paths, owner_authority, extras = retained
    intents = tmp_path / "intents"
    bindings = tmp_path / "public-source-bindings"
    machinery_out = tmp_path / "config" / "machinery.json"
    bindings.mkdir(parents=True)
    (bindings / "public-scene-841757.json").write_text('{"different":true}')  # conflicting bytes
    with pytest.raises(provisioner.PublicSceneIntentProvisionError,
                       match="public_scene_provision_immutable_conflict"):
        provisioner.provision_public_scene_intent(
            retained=retained_paths, owner_authority=owner_authority, binding_id="public-scene-841757",
            profile_registry_root=str(tmp_path / "reg"), maximum_preparation_spend_usd=4.5,
            intent_root=intents, public_source_binding_root=bindings, machinery_output_path=machinery_out,
            now=extras["now"])
    assert not intents.exists() or not list(intents.glob("scene-*")), "no intent may be staged on dep failure"


def test_missing_disclosure_authority_fails_closed_as_rights_absent(retained, tmp_path):
    retained_paths, owner_authority, extras = retained
    seed = json.loads(retained_paths["accepted_task_request"].read_text())
    ref = next(iter(seed["human_authority"]["full_source_provider_disclosure_authorities"].values()))
    Path(ref["path"]).unlink()
    with pytest.raises(provisioner.PublicSceneIntentProvisionError, match="public_scene_rights_absent"):
        _provision(tmp_path, retained_paths, owner_authority, now=extras["now"])


def test_public_scene_intent_under_mesh_only_config_is_refused(retained, tmp_path, monkeypatch):
    retained_paths, owner_authority, extras = retained
    result, intents, bindings, machinery_out = _provision(
        tmp_path, retained_paths, owner_authority, now=extras["now"])
    _bind_runtime_env(monkeypatch, intents, extras["commit"])
    config = _progression_config(tmp_path, intents, bindings, machinery_out, retained_paths["release_binding"],
                                 source_kinds=["mesh", "gaussian_splat"])
    run = engine.process_scene_intents(config_path=config)
    assert run["results"][0]["status"] == "needs_input"
    assert run["results"][0]["blockers"] == ["source_kind_not_supported_by_progression"]
    assert run["provider_allocation_performed"] is False


def test_cli_provisions_from_a_retained_manifest(retained, tmp_path, capsys):
    retained_paths, owner_authority, extras = retained
    manifest = _write(tmp_path / "retained-manifest.json",
                      {key: str(path) for key, path in retained_paths.items()})
    authority = _write(tmp_path / "owner-authority.json", owner_authority)
    intents = tmp_path / "intents"
    bindings = tmp_path / "public-source-bindings"
    machinery_out = tmp_path / "config" / "machinery.json"
    assert provisioner.main([
        "--retained-manifest", str(manifest), "--owner-authority", str(authority),
        "--binding-id", "public-scene-841757", "--profile-registry-root", str(tmp_path / "registry"),
        "--intent-root", str(intents), "--public-source-binding-root", str(bindings),
        "--machinery-output-path", str(machinery_out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "public_scene_intent_provisioned"
    assert Path(printed["binding_path"]).is_file()
    assert (intents / printed["intent_id"] / "intent.json").is_file()


def test_source_preparation_is_materialized_by_the_canonical_producer(tmp_path):
    """The provisioner reuses materialize_public_scene_source_preparation to obtain
    the source-preparation receipt from a real installed scene when no retained
    receipt is supplied. Proven against a real USD scene (fake-USD fixtures cannot
    exercise the USD collision inspection)."""
    from blueprint_pipeline import public_scene_source_preparation as source_prep
    from tests.test_public_scene_source_preparation import _installed_scene, _objects

    installed = _installed_scene(tmp_path)
    receipt_path = provisioner.resolve_source_preparation_receipt(
        installation_receipt_path=installed, task_objects=_objects(explicit_support=True),
        expected_source_commit=public_scene_host_input_intake._verified_checkout_head(),
        approved_roots=(tmp_path,), output_root=tmp_path / "prepared")
    receipt = json.loads(Path(receipt_path).read_text())
    assert receipt["schema_version"] == source_prep.SCHEMA_VERSION
    assert receipt["status"] == "source_context_prepared_pending_calibrated_views"
    # A retained receipt is adopted verbatim, not re-derived.
    assert provisioner.resolve_source_preparation_receipt(
        source_preparation_receipt_path=receipt_path) == Path(receipt_path)


def test_provisioned_machinery_carries_completed_prefix_reuse_roots(retained, tmp_path):
    # A2: the factory's completed-prefix adoption path reads child_queue_root,
    # parent_queue_root, execution_root and release_retention_binding_root directly
    # from the machinery. The provisioner must emit those canonical host roots so a
    # re-attempt of the same owner intent can adopt already-completed GPU stages
    # (never repeat paid work) instead of KeyError-ing on a missing machinery key.
    from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import (
        DEFAULT_QUEUE, DEFAULT_PARENT_QUEUE, DEFAULT_EXECUTION)
    from blueprint_pipeline.task_evaluation_release_retention import DEFAULT_EVIDENCE_BINDING_ROOT
    retained_paths, owner_authority, extras = retained
    result, _intents, _bindings, _machinery_out = _provision(
        tmp_path, retained_paths, owner_authority, now=extras["now"])
    machinery = json.loads(Path(result["machinery_path"]).read_text())
    assert machinery["child_queue_root"] == str(DEFAULT_QUEUE)
    assert machinery["parent_queue_root"] == str(DEFAULT_PARENT_QUEUE)
    assert machinery["execution_root"] == str(DEFAULT_EXECUTION)
    assert machinery["release_retention_binding_root"] == str(DEFAULT_EVIDENCE_BINDING_ROOT)
