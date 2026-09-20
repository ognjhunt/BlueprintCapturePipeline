"""Real website compiler to publisher inventory; no provider or model calls."""
import copy
import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_scene_configuration_submission_publication import _validated_inventory
from blueprint_pipeline.website_native_submission import materialize_website_submission
from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_website_native_appearance import inputs
from tests.test_task_evaluation_scene_configuration_submission import production_fixture, SHA


def setup(tmp_path, monkeypatch, *, development=False):
    capture = tmp_path / "capture"
    capture.mkdir()
    args, _, _ = inputs(capture)
    now = time.time()
    args["now"] = now
    args["spend"] = copy.deepcopy(args["spend"])
    args["spend"]["expires_at_epoch"] = now + 3600
    args["spend"]["consent"]["accepted_at_epoch"] = now - 1
    preparation = compile_website_scene_preparation(**args)
    if development:
        from blueprint_pipeline.website_development_test import prepare_development_test, ENV
        monkeypatch.setenv(ENV, json.dumps([args["task_context"]["context_digest"]]))
        preparation["status"] = "needs_input"
        preparation["blockers"] = ["support_surface_not_found_under_subject"]
        preparation["support"] = None
        preparation["digest"] = canonical_digest(preparation, digest_field="digest")
        preparation, _ = prepare_development_test(preparation=preparation,
            source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=capture / "native")
        args["output_root"] = capture / "native"
    else:
        prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
            source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=capture / "native")
    write_json(capture / "context.json", args["task_context"])
    intake_root = tmp_path / "intents"
    accepted = stage_scene_intent(value=preparation["intake_request"], queue_root=intake_root,
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"}, now=now)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intake_root))
    task = {"scene_intent_authority": record(intake_root / accepted["intent_id"] / "intent.json"),
            "preparation": record(args["output_root"] / "preparation.json"),
            "runtime_inputs": record(capture / "native/runtime_inputs.json"),
            "task_context": record(capture / "context.json")}
    fixture = production_fixture(tmp_path / "release")
    kwargs = {"task": task, "expected_production_commit": SHA, "namespace_timestamp": "20260919T120000Z",
              "release_admission_mode": "promoted", "staging_root": tmp_path / "submission",
              "runtime_publication_root": fixture["runtime_publication_root"],
              **{key + "_path": fixture[key] for key in ("deploy_receipt", "release_provenance", "release_environment")}}
    return kwargs, accepted


def test_prepared_capture_materializes_a_publishable_native_request_without_raw_video(tmp_path, monkeypatch):
    kwargs, accepted = setup(tmp_path, monkeypatch)
    preparation = json.loads(Path(kwargs["task"]["preparation"]["path"]).read_text())
    assert "collision_mesh" not in preparation["intake_request"]["source"]
    assert preparation["binding"]["collision_mesh_digest"]
    first = materialize_website_submission(**kwargs)
    assert materialize_website_submission(**kwargs) == first
    root = kwargs["staging_root"]
    manifest, rows = _validated_inventory(root, SHA)
    assert manifest["source"] == "website_capture_derivatives"
    assert all(row["publication_allowed"] for row in rows)
    assert not any(Path(row["relative_path"]).suffix in {".mov", ".mp4"} for row in rows)
    request = json.loads((root / "scene_configuration_preparation_request.v1.json").read_text())
    assert request["scene_intent_digest"] == accepted["intent_digest"]
    caps = request["spend"]["external_service_caps"]["openai"]
    assert caps["stage_max_cost_usd"] == {"artifixer_semantic_teacher": 0,
        "artifixer_visual_review": 0, "content_agents": 5}
    assert caps["maximum_cost_usd"] == 5
    assert request["spend"]["hard_cap_usd"] == 11
    assert request["scene"]["geometry"]["kind"] == "other_derived"
    assert request["scene"]["rights"]["provider_disclosure_scope"] == "derived_only"
    assert request["scene"]["website_native_inputs"]["frames"]
    assert request["task"]["surface_target"]["non_colliding"] is True
    recipe = json.loads((root / "configuration/recipe.json").read_text())
    assert [s["adapter"]["id"] for s in recipe["stage_sequence"][:2]] == [
        "website_prepared_appearance", "website_prepared_collision"]
    # Estimated physics and placement uncertainty ride with the task, so the
    # result can abstain rather than claim feasibility the estimate cannot carry.
    task = json.loads((root / "configuration/task.json").read_text())
    screen = task["physical_property_screen"]
    assert screen["basis"] == "estimated" and screen["bounds"]["mass_kg"][0] < screen["bounds"]["mass_kg"][1]
    assert screen["sensitivity"] in {"robust_within_range", "outcome_depends_on_estimate", "blocked_by_estimate"}
    assert screen["feasibility_claim_allowed"] == (screen["sensitivity"] == "robust_within_range")
    assert screen["reference_gripper"]["model"] == "robotiq_2f85"
    assert task["scale_authority"] in {"registration_estimate", "provider_declared_estimate"}
    assert "placement_uncertainty_m" in task and task["physical_world_truth_claimed"] is False


def test_publication_rechecks_owner_revocation(tmp_path, monkeypatch):
    kwargs, _ = setup(tmp_path, monkeypatch)
    materialize_website_submission(**kwargs)
    intent_path = Path(kwargs["task"]["scene_intent_authority"]["path"])
    write_json(intent_path.parent / "revoked.json", {"revoked": True})
    with pytest.raises(ValueError, match="revoked"):
        _validated_inventory(kwargs["staging_root"], SHA)


def prepare_website_construction(tmp_path, monkeypatch, development):
    import os
    import pwd
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
    from blueprint_pipeline import public_scene_host_input_intake
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
    from blueprint_pipeline.website_scene_dispatch import register_website_preparation
    from tests.test_task_evaluation_scene_configuration_submission_publication import Store
    kwargs, accepted = setup(tmp_path, monkeypatch, development=development)
    task = kwargs["task"]
    root = tmp_path / "bindings"
    register_website_preparation(preparation_path=task["preparation"]["path"],
        runtime_inputs_path=task["runtime_inputs"]["path"], task_context_path=task["task_context"]["path"],
        root=root, now=time.time())
    # The WebApp forwards ECMAScript JSON: whole-number floats become integers.
    # Its accepted intent must still find the original Python source registration.
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_json
    intent_path = Path(task["scene_intent_authority"]["path"])
    intent = json.loads(intent_path.read_text())
    intent["request"] = json.loads(cross_runtime_canonical_json(intent["request"]))
    write_json(intent_path, intent)
    def sealed(name, value, field):
        value[field] = canonical_digest(value, digest_field=field)
        path = tmp_path / name
        write_json(path, value)
        return path
    machinery_path = sealed("machinery.json", {"schema_version": "task_evaluation_website_scene_machinery.v1",
        "maximum_preparation_spend_usd": 0, "provider": "vast"}, "machinery_digest")
    release_path = sealed("release.json", {"schema_version": RELEASE_SCHEMA, "source_commit": SHA,
        "runtime_digest": "sha256:" + "f" * 64, "repo_root": str(tmp_path / "repo"),
        "runtime_publication_root": str(kwargs["runtime_publication_root"]),
        "namespace_timestamp": kwargs["namespace_timestamp"], "release_admission_mode": "promoted",
        **{key: record(kwargs[key + "_path"]) for key in ("deploy_receipt", "release_provenance", "release_environment")}},
        "release_digest")
    queue = tmp_path / "preparations"
    ensure_launch_preparation_queue_root(queue)
    config_path = sealed("config.json", {"schema_version": engine.CONFIG_SCHEMA,
        "intent_root": str(tmp_path / "intents"), "public_source_binding_root": str(tmp_path / "unused"),
        "website_source_binding_root": str(root), "website_source_machinery_path": str(machinery_path),
        "release_binding_path": str(release_path), "factory_output_root": str(tmp_path / "factories"),
        "trusted_clients": ["blueprint-webapp"], "submission_enabled": True,
        "submission_transport": "local_owned_queue", "preparation_queue_root": str(queue),
        "service_account": pwd.getpwuid(os.geteuid()).pw_name,
        "publication_lock_root": str(tmp_path / "locks")}, "config_digest")
    monkeypatch.setattr(public_scene_host_input_intake, "_verified_checkout_head", lambda: SHA)
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: SHA)
    store = Store()
    def publish(**kw):
        return publication.publish_scene_configuration_submission(**kw, client=store)
    result = engine.process_scene_intents(config_path=config_path, publisher=publish)
    assert result["results"][0]["status"] == "running", result
    rows = list((queue / "pending").glob("*.json"))
    assert len(rows) == 1
    queued = json.loads(rows[0].read_text())
    assert queued["request"]["scene_intent_digest"] == accepted["intent_digest"]
    assert queued["request"]["scene"]["website_native_inputs"]["frames"]
    assert store.puts
    assert not any(key.endswith((".mov", ".mp4")) for key in store.puts)
    intent_dir = tmp_path / "intents" / accepted["intent_id"]
    assert not list((intent_dir / "attempts").glob("*.json"))
    attempts = list((intent_dir / "preparation-attempts").glob("*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["maximum_spend_usd"] == 0
    engine.process_scene_intents(config_path=config_path, publisher=publish)
    assert list((queue / "pending").glob("*.json")) == rows
    from urllib.parse import urlsplit
    from blueprint_pipeline.task_evaluation_launch_preparation_worker import process_launch_preparation_queue
    def fetch(uri, destination, maximum_bytes):
        data = store.objects[urlsplit(uri).path.lstrip("/")]
        assert len(data) <= maximum_bytes
        destination.write_bytes(data)
    consumed = process_launch_preparation_queue(queue_root=queue, input_root=tmp_path / "worker-inputs",
        allowed_uri_prefixes=["s3://blueprint/task-evaluation/production-inputs/"],
        service_account=pwd.getpwuid(os.geteuid()).pw_name, source_commit=SHA, fetcher=fetch,
        construction_queue_root=tmp_path / "construction")
    assert consumed["results"][0]["status"] == "queued_for_production_scene_configuration", consumed["results"][0].get("blockers")
    assert len(list((tmp_path / "construction/pending").glob("*.json"))) == 1

    return json.loads(next((tmp_path / "construction/pending").glob("*.json")).read_text())


@pytest.mark.parametrize("development", [False, True])
def test_existing_progression_publishes_and_queues_website_source_without_manual_step(tmp_path, monkeypatch, development):
    prepare_website_construction(tmp_path, monkeypatch, development)


def test_collected_website_world_registers_the_source_for_progression(tmp_path, monkeypatch):
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    args, _, _ = inputs(pipeline)
    # This seam consumes World Labs' Y-down exports, whereas the reusable
    # fixture is Y-up. Rotate the actual collider bytes into the provider frame.
    import numpy as np
    import trimesh
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    collider = Path(args["base_scene"]["collision_mesh_path"])
    mesh = trimesh.load(collider, force="mesh")
    mesh.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))
    mesh.export(collider)
    args["base_scene"]["collision_mesh_digest"] = _sha256_file(collider)
    context = args["task_context"]
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(tmp_path / "intents"))
    assets = pipeline / "assets.json"
    write_json(assets, {"world_id": "world-1", "downloads": [
        {"kind": kind, "local_path": args["base_scene"][key + "_path"],
         "sha256": args["base_scene"][key + "_digest"][7:]}
        for kind, key in (("splat_ply", "splat"), ("collider_mesh_glb", "collision_mesh"))]})
    removal = pipeline / "removal.json"
    write_json(removal, args["removal_manifest"])
    result = prepare_website_scene_handoff(descriptor={"capture_id": context["capture_id"],
        "scene_id": context["scene_id"], "metadata": {"site_task_context": context,
            "website_scene_execution_authority": args["spend"]}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "task_masks": args["task_masks"],
            "source_geometry": args["source_geometry"], "removal_manifest_path": str(removal)},
        provider_run={"status": "ready", "world_id": "world-1", "provider_run_id": "op-1",
            "worldlabs_asset_materialization": {"manifest_path": str(assets)}}, capture_root=tmp_path, now=args["now"])
    assert result.get("source_registration"), result
    registration = json.loads(Path(result["source_registration"]["path"]).read_text())
    assert registration["execution_authority_granted"] is False
    assert Path(registration["references"]["runtime_inputs"]["path"]).is_file()
    assert result["simulator_ready"] is False
