"""A completed splat+collision-mesh intent reaches publication-ready and a queue row.

This mirrors ``tests/test_task_evaluation_scene_progression.py`` (the public-scene
path) but for an owner-uploaded finished 3DGS asset plus its collision mesh. No
manual step runs between ``process_scene_intents`` and a queued construction
submission, and every un-measured physical property stays a typed input, never a
fabricated measurement.
"""
import json
import time
from pathlib import Path

from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_progression_state as state
from blueprint_pipeline import public_scene_host_input_intake
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA, record
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
)
from tests.test_task_evaluation_completed_scene_source import MESH, upload
from tests.test_task_evaluation_scene_configuration_submission import SHA, production_fixture
from tests.test_task_evaluation_scene_intake import request as owner_request
from tests.test_provided_scene_splat import splat_bytes


def _write(path, value, field=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return path


def _owner(store, now, *, source_kind="gaussian_splat"):
    import struct
    data = bytearray(splat_bytes())
    offset = data.index(b"end_header\n") + len(b"end_header\n")
    for i in range(64):
        xyz = ((0.02 + (i % 4) * 0.03, 0.02 + (i // 4) * 0.04, 0.76)
               if i < 16 else (-0.8 + (i % 8) * 0.3, -0.8 + (i // 8) * 0.3, 0.749))
        struct.pack_into("<3f", data, offset + i * 14 * 4, *xyz)
    _, appearance = upload(store, bytes(data), "appearance1", "provided_scene_splat", "room.ply")
    submitted, collision = upload(store, MESH, "collision1", "provided_scene_mesh", "room.usda")
    owner = owner_request()
    from blueprint_pipeline.task_evaluation_scene_policy_capability import supported_policy_candidates
    owner["execution"]["policy_candidates"] = supported_policy_candidates()
    owner["owner"] = {"user_id": submitted["customer_id"], "organization_id": submitted["organization_id"]}
    owner["consent"].update(accepted_by=owner["owner"]["user_id"], accepted_at_epoch=now - 1,
                            rights_reference=appearance["envelope_digest"], provider_terms_reference="terms-v1")
    owner["source"] = {"kind": "gaussian_splat", "binding_id": "appearance1",
        "content_digest": appearance["capture_digest"],
        "collision_mesh": {"binding_id": "collision1", "content_digest": collision["capture_digest"],
            "rights_reference": collision["envelope_digest"], "frame_relation": "owner_declared_common_frame"}}
    if source_kind == "mesh":
        owner["consent"]["rights_reference"] = collision["envelope_digest"]
        owner["source"] = {"kind": "mesh", "binding_id": "collision1", "content_digest": collision["capture_digest"]}
    owner["task"] = {"task_id": "book-into-tray", "strategy": "pick_and_place",
        "subject": {"description": "Book"}, "support": {"description": "Table"},
        "destination": {"relation": "inside", "visible_label": "document tray",
            "position_world_m": [0.5, 0.5, 0.75], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "success": {"control_frequency_hz": 15, "maximum_episode_seconds": 24.0, "minimum_lift_m": 0.05,
            "pregrasp_clearance_m": 0.1, "minimum_planar_displacement_m": 0.1,
            "maximum_final_planar_target_error_m": 0.05, "maximum_retries": 0, "maximum_regrasps": 0}}
    owner["execution"].update(max_total_spend_usd=40, max_paid_attempts=4,
                              allowed_providers=["vast", "openai"], expires_at_epoch=now + 3600)
    return owner


def _config(tmp_path, monkeypatch, *, submission_enabled=False, extra=None, source_kind="gaussian_splat", real_destination=False):
    now = time.time()
    fixture = production_fixture(tmp_path)
    if real_destination:
        from tests.test_task_evaluation_passive_destination_simready import test_exact_step_visual_gets_five_colliders_and_static_qualification
        destination_root = tmp_path / "destination-fixture"
        destination_root.mkdir()
        test_exact_step_visual_gets_five_colliders_and_static_qualification(destination_root)
        fixture["destination_simready"] = destination_root / "output/passive_destination_simready_result.v1.json"
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: SHA)
    store = tmp_path / "store"
    intake_root = tmp_path / "intents"
    owner = _owner(store, now, source_kind=source_kind)
    receipt = stage_scene_intent(value=owner, queue_root=intake_root, authenticated_client="blueprint-webapp",
                                 trusted_clients={"blueprint-webapp"}, now=now)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intake_root.resolve()))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "blueprint-webapp")
    monkeypatch.setenv("PIPELINE_CAPTURE_INTAKE_STORE_ROOT", str(store.resolve()))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_OWNER_SOURCE_STORE_ROOT", str(tmp_path / "owner-sources"))
    machinery = {"schema_version": "task_evaluation_completed_scene_machinery.v1",
                 "maximum_preparation_spend_usd": 4.5, "provider": "vast",
                 "destination_catalog": [{"binding_id": "document-tray-v2",
                    "owner_description_aliases": ["document tray"],
                    "simready_result": record(fixture["destination_simready"])}],
                 "simulation_physics_bounds": {"mass_kg_bounds": [0.1, 0.8],
                    "static_friction_bounds": [0.4, 0.8], "dynamic_friction_bounds": [0.3, 0.6],
                    "restitution_bounds": [0.0, 0.1]}}
    machinery_path = _write(tmp_path / "completed-machinery.json", machinery, "machinery_digest")
    (tmp_path / "repo").mkdir(exist_ok=True)
    release = {"schema_version": RELEASE_SCHEMA, "source_commit": SHA, "runtime_digest": "sha256:" + "f" * 64,
        "repo_root": str(tmp_path / "repo"), "runtime_publication_root": str(fixture["runtime_publication_root"]),
        "namespace_timestamp": "20260906T010000Z", "release_admission_mode": "promoted",
        **{key: record(fixture[key]) for key in ("deploy_receipt", "release_provenance", "release_environment")}}
    release_path = _write(tmp_path / "release.json", release, "release_digest")
    config = {"schema_version": engine.CONFIG_SCHEMA, "intent_root": str(intake_root),
        "public_source_binding_root": str(tmp_path / "unused-bindings"),
        "machinery_path": str(machinery_path), "release_binding_path": str(release_path),
        "completed_source_machinery_path": str(machinery_path), "capture_store_root": str(store),
        "factory_output_root": str(tmp_path / "progression-output"), "trusted_clients": ["blueprint-webapp"],
        "submission_enabled": submission_enabled}
    if extra:
        config.update(extra)
    config_path = _write(tmp_path / "progression-config.json", config, "config_digest")
    return config_path, receipt["intent_id"], intake_root, now


def test_completed_asset_reaches_publication_ready_without_manual_step(tmp_path, monkeypatch):
    config_path, intent_id, intake_root, now = _config(tmp_path, monkeypatch)
    first = engine.process_scene_intents(config_path=config_path, now=now)
    assert first["results"][0]["phase"] == "publication_ready", first
    assert first["provider_allocation_performed"] is False
    # Idempotent across a worker restart: no second attempt, byte-identical run.
    assert engine.process_scene_intents(config_path=config_path, now=now) == first

    directory = intake_root / intent_id
    intent = json.loads((directory / "intent.json").read_text())
    progress = state.load_progression(directory, intent)
    factory = json.loads(Path(progress["state"]["factory"]["path"]).read_text())
    assert factory["status"] == "publication_ready"
    assert factory["provider_mutation_performed"] is False
    assert factory["source_uploaded"] is False
    assert factory["physical_registration_proven"] is False
    assert factory["physical_scale_measured"] is False
    assert factory["claim_scope"] == "development_only"

    req = json.loads(Path(factory["submission_request"]["path"]).read_text())
    assert req["scene_intent_digest"] == intent["intent_digest"]
    assert req["run_mode"] == "scene_configuration"
    assert req["scene"]["appearance"]["kind"] == "gaussian_splat"
    assert req["scene"]["geometry"]["kind"] == "observed_mesh"
    assert req["scene"]["rights"]["source_bytes_redistributable"] is False
    assert req["scene"]["rights"]["provider_disclosure_scope"] == "derived_only"
    assert req["task"]["kind"] == "rigid_relocation"
    assert req["task"]["strategy"] == "pick_and_place"
    assert req["task"]["subject"]["mode"] == "construct_from_scene_object"
    # The launch contract must accept the assembled request unmodified.
    assert validate_launch_preparation_request(req) == req

    manifest = json.loads(Path(factory["submission_manifest"]["path"]).read_text())
    assert manifest["raw_source_upload_allowed"] is False
    assert manifest["provider_allocated"] is False


def test_missing_destination_pose_stays_a_typed_blocker(tmp_path, monkeypatch):
    config_path, intent_id, intake_root, now = _config(tmp_path, monkeypatch)
    # A second, pose-free intent under a fresh store exercises the blocker branch:
    # the factory must fail closed rather than invent a placement.
    owner = _owner(tmp_path / "store2", now)
    owner["submission_id"] = "upload-no-pose"
    owner["task"]["destination"] = {"relation": "on", "visible_label": "table"}
    receipt = stage_scene_intent(value=owner, queue_root=intake_root, authenticated_client="blueprint-webapp",
                                 trusted_clients={"blueprint-webapp"}, now=now)
    value = json.loads(config_path.read_text())
    value["capture_store_root"] = str(tmp_path / "store2")
    value["paused_intent_ids"] = [intent_id]  # only advance the pose-free intent
    _write(config_path, value, "config_digest")
    result = engine.process_scene_intents(config_path=config_path, now=now)
    row = next(r for r in result["results"] if r["intent_id"] == receipt["intent_id"])
    assert row["status"] == "needs_input", result
    assert "task_destination_pose_required" in row["blockers"], row


def test_completed_asset_produces_a_preparation_queue_row(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
        ensure_launch_preparation_queue_root, stage_launch_preparation_request,
    )
    queue = tmp_path / "queue"
    ensure_launch_preparation_queue_root(queue)
    config_path, intent_id, intake_root, now = _config(tmp_path, monkeypatch, submission_enabled=True,
        extra={"preparation_queue_root": str(queue), "publication_lock_root": str(tmp_path / "publication-locks")})

    def publish(**kwargs):
        return dict(status="published_and_read_back", source_commit=kwargs["expected_source_commit"],
                    raw_source_uploaded=False, provider_allocated=False,
                    manifest_sha256=record(kwargs["manifest_path"])["sha256"])

    def post(*, request_path, config):
        req = json.loads(Path(request_path).read_text())
        stage_launch_preparation_request(value=req, queue_root=queue, submitted_by="test-webapp")
        return {"schema_version": "task_evaluation_launch_preparation_web_submission_receipt.v1",
                "status": "submitted", "webapp_request_digest": cross_runtime_canonical_digest(req),
                "preparation_id": req["preparation_id"], "paid_execution_requested_by_this_tool": False}

    result = engine.process_scene_intents(config_path=config_path, publisher=publish, submitter=post, now=now)
    assert result["results"][0]["status"] == "running", result
    rows = [p for statename in ("pending", "processing") for p in (queue / statename).glob("*.json")]
    assert len(rows) == 1, rows
    link = json.loads((intake_root / intent_id / "preparation-link.json").read_text())
    # R3: the preparation link never carries the paid construction reservation; that
    # moves to the versioned activation link at the activation transition.
    assert "scene_configuration_attempt" not in link


def test_real_publisher_retains_owner_sources_and_worker_reads_them_without_network(tmp_path, monkeypatch):
    import os
    import pwd
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.task_evaluation_owner_source_store import PREFIX
    from tests.test_task_evaluation_scene_configuration_submission_publication import Store

    config_path, intent_id, intake_root, now = _config(tmp_path, monkeypatch)
    engine.process_scene_intents(config_path=config_path, now=now)
    progress = state.load_progression(intake_root / intent_id,
        json.loads((intake_root / intent_id / "intent.json").read_text()))
    factory = json.loads(Path(progress["state"]["factory"]["path"]).read_text())
    manifest_path = Path(factory["submission_manifest"]["path"])
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: SHA)
    store = Store()
    result = publication.publish_scene_configuration_submission(manifest_path=manifest_path,
        receipt_path=tmp_path / "published.json", expected_source_commit=SHA,
        service_account=pwd.getpwuid(os.geteuid()).pw_name, client=store, lock_root=tmp_path / "locks")
    assert result["status"] == "published_and_read_back"
    assert len(result["host_only_source_objects"]) == 2
    assert not any("/source/" in key or "host-only-owner-sources" in key for key in store.puts)
    monkeypatch.setattr(worker, "_s3_client", lambda _: (_ for _ in ()).throw(AssertionError("source network access")))
    for index, row in enumerate(result["host_only_source_objects"]):
        assert row["uri"].startswith(PREFIX)
        output = tmp_path / f"read-{index}"
        worker.default_reference_fetcher(row["uri"], output, row["size_bytes"])
        assert record(output)["sha256"] == row["digest"]


def test_owner_revocation_prevents_real_publication(tmp_path, monkeypatch):
    import pytest
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_publication import _validated_inventory

    config_path, intent_id, intake_root, now = _config(tmp_path, monkeypatch)
    engine.process_scene_intents(config_path=config_path, now=now)
    progress = json.loads((intake_root / intent_id / "progression.json").read_text())
    factory = json.loads(Path(progress["state"]["factory"]["path"]).read_text())
    (intake_root / intent_id / "revoked.json").write_text("{}")
    with pytest.raises(ValueError, match="scene_owner_authority_revoked"):
        _validated_inventory(Path(factory["submission_manifest"]["path"]).parent, SHA)


def test_owner_task_id_bridges_controls_autoprovision(tmp_path, monkeypatch):
    """The owner's declared task_id must survive unchanged into the preparation
    request's task identity and the preparation link, because controls
    autoprovision requires ``link["task_id"] == owner_intent.task.task_id`` and
    ``preparation.task.identity.id == link["task_id"]`` simultaneously. A
    synthesized ``completed-task-<digest>`` identity breaks that bridge.
    """
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
        ensure_launch_preparation_queue_root, stage_launch_preparation_request,
    )
    queue = tmp_path / "queue"
    ensure_launch_preparation_queue_root(queue)
    config_path, intent_id, intake_root, now = _config(tmp_path, monkeypatch, submission_enabled=True,
        extra={"preparation_queue_root": str(queue), "publication_lock_root": str(tmp_path / "publication-locks")})

    def publish(**kwargs):
        return dict(status="published_and_read_back", source_commit=kwargs["expected_source_commit"],
                    raw_source_uploaded=False, provider_allocated=False,
                    manifest_sha256=record(kwargs["manifest_path"])["sha256"])

    def post(*, request_path, config):
        req = json.loads(Path(request_path).read_text())
        stage_launch_preparation_request(value=req, queue_root=queue, submitted_by="test-webapp")
        return {"schema_version": "task_evaluation_launch_preparation_web_submission_receipt.v1",
                "status": "submitted", "webapp_request_digest": cross_runtime_canonical_digest(req),
                "preparation_id": req["preparation_id"], "paid_execution_requested_by_this_tool": False}

    engine.process_scene_intents(config_path=config_path, publisher=publish, submitter=post, now=now)

    directory = intake_root / intent_id
    intent = json.loads((directory / "intent.json").read_text())
    owner_task_id = intent["request"]["task"]["task_id"]
    assert owner_task_id == "book-into-tray"

    link = json.loads((directory / "preparation-link.json").read_text())
    assert link["task_id"] == owner_task_id, link

    factory = json.loads(Path(state.load_progression(directory, intent)["state"]["factory"]["path"]).read_text())
    req = json.loads(Path(factory["submission_request"]["path"]).read_text())
    assert req["task"]["identity"]["id"] == owner_task_id, req["task"]["identity"]
