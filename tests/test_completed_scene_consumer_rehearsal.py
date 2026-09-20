"""Run completed uploads through real consumers; fake only external transports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import grp
import zipfile
import pytest
from urllib.parse import urlsplit

from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root, stage_launch_preparation_request
from blueprint_pipeline.task_evaluation_owner_source_store import PREFIX
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import SceneConfigurationAdapterRegistry
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import builtin_scene_configuration_adapter_handlers
from blueprint_pipeline.task_evaluation_scene_configuration_orchestrator import _load_envelope, _verified_stage_configurations
from blueprint_pipeline.task_evaluation_scene_configuration_source_preflight import validate_scene_configuration_source_preflight
from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
from tests.test_task_evaluation_completed_scene_progression import _config
from tests.test_task_evaluation_scene_configuration_submission import SHA
from tests.test_task_evaluation_scene_configuration_submission_publication import Store

ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
pytestmark = pytest.mark.slow


def _prepare(tmp_path, monkeypatch, *, source_kind="mesh", render_materializer=None):
    queue = ensure_launch_preparation_queue_root(tmp_path / "preparations")
    config, intent_id, intents, now = _config(tmp_path, monkeypatch, submission_enabled=True,
        source_kind=source_kind, real_destination=True, extra={"preparation_queue_root": str(queue),
            "publication_lock_root": str(tmp_path / "locks"), "service_account": ACCOUNT,
            "submission_transport": "local_owned_queue", "activation_enabled": False})
    store = Store()
    monkeypatch.setattr(publication, "_verified_checkout_head", lambda: SHA)
    def publish(**kwargs):
        return publication.publish_scene_configuration_submission(**kwargs, client=store)
    def submit(*, request_path, config):
        request = json.loads(request_path.read_text())
        stage_launch_preparation_request(value=request, queue_root=queue, submitted_by="test-webapp")
        return {"schema_version": "task_evaluation_launch_preparation_web_submission_receipt.v1",
            "status": "submitted", "webapp_request_digest": cross_runtime_canonical_digest(request),
            "preparation_id": request["preparation_id"], "paid_execution_requested_by_this_tool": False}
    first = engine.process_scene_intents(config_path=config, publisher=publish, submitter=submit, now=now)
    assert first["results"][0]["status"] == "running", first
    def fetch(uri, destination, maximum_bytes):
        if uri.startswith(PREFIX):
            return worker.default_reference_fetcher(uri, destination, maximum_bytes)
        data = store.objects[urlsplit(uri).path.lstrip("/")]
        assert len(data) == maximum_bytes
        destination.write_bytes(data)
    inputs = tmp_path / "worker-inputs"
    construction = tmp_path / "construction"
    kwargs = {"scene_render_input_materializer": render_materializer} if render_materializer else {}
    result = worker.process_launch_preparation_queue(queue_root=queue, input_root=inputs,
        allowed_uri_prefixes=["s3://blueprint/task-evaluation/"], service_account=ACCOUNT,
        source_commit=SHA, fetcher=fetch, construction_queue_root=construction, **kwargs)
    receipts = [json.loads(path.read_text()) for path in (queue / "results").glob("*.json")]
    assert len(receipts) == 1 and receipts[0]["status"] == "queued_for_production_scene_configuration", (result, receipts)
    envelopes = list((construction / "pending").glob("*.json"))
    assert len(envelopes) == 1
    envelope = _load_envelope(envelopes[0])
    configurations = _verified_stage_configurations(envelope=envelope, input_root=inputs)
    values = {key: value[0] for key, value in configurations.items()}
    validate_immutable_stage_configurations(envelope=envelope, configurations=values)
    validate_scene_configuration_source_preflight(envelope=envelope, configurations=values)
    assert envelope["request"]["task"]["identity"]["id"] == "book-into-tray"
    assert envelope["provider_mutation_performed"] is False
    return envelope, configurations, config, store, intent_id, intents


def test_mesh_upload_reaches_real_construction_consumer_and_static_qualification(tmp_path, monkeypatch):
    envelope, configurations, config, store, intent_id, intents = _prepare(tmp_path, monkeypatch)
    assert envelope["render_inputs_result"]["derived_frames"] == []
    from blueprint_pipeline.task_evaluation_scene_configuration_bundle import build_scene_configuration_provider_bundle
    from blueprint_pipeline.task_evaluation_scene_configuration_provider_preflight import scene_configuration_bundle_contract
    from blueprint_pipeline.provider_archive import extract_provider_archive
    from scripts.task_evaluation_scene_configuration_provider_runner import _hydrate_envelope
    from tests.astra_toolchain_fixture import astra_toolchain_fixture
    from blueprint_pipeline.task_evaluation_scene_configuration_astra_runtime import declared_python_profile
    envelope_path = tmp_path / "consumer-envelope.json"
    envelope_path.write_text(json.dumps(envelope))
    # Fresh submissions select Astra, so even this CPU-only consumer rehearsal
    # must carry the sealed Astra profile instead of a legacy base wheelhouse.
    toolchain = astra_toolchain_fixture(tmp_path / "toolchain", SHA, monkeypatch)
    assert declared_python_profile(toolchain) == "astra_asset_authoring"
    bundle = build_scene_configuration_provider_bundle(construction_envelope_path=envelope_path,
        toolchain_root=toolchain, repository_root=Path(__file__).resolve().parents[1],
        output_root=tmp_path / "bundle", expected_source_commit=SHA)
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        _, _, blockers = scene_configuration_bundle_contract(archive)
    assert blockers == []
    extracted = tmp_path / "extracted"
    extract_provider_archive(Path(bundle["bundle_path"]), extracted)
    runtime = extracted / "provider_runtime"
    portable = json.loads((runtime / "input/portable_construction_envelope.v1.json").read_text())
    envelope = _hydrate_envelope(runtime, portable)
    configurations = {row["stage_id"]: (json.loads(Path(row["materialized_path"]).read_text()), Path(row["materialized_path"]))
                      for row in envelope["stage_configuration_references"]}
    registry = SceneConfigurationAdapterRegistry(builtin_scene_configuration_adapter_handlers())
    results = []
    # Source appearance excision, collision excision, exact-geometry physics
    # authoring and independent USD qualification all execute on the CPU.
    for stage in envelope["recipe"]["stage_sequence"][:4]:
        configuration, path = configurations[stage["stage_id"]]
        output = tmp_path / "executed" / stage["stage_id"]
        output.mkdir(parents=True)
        result = registry.execute(stage=stage, envelope=envelope, configuration=configuration,
            configuration_path=path, dependency_results=tuple(results), output_root=output)
        assert result["status"] == "completed"
        assert result["provider_mutations_performed"] == 0
        results.append(result)
    assert any(row["role"] == "static_qualification_receipt" for row in results[-1]["output_artifacts"])


def test_splat_and_mesh_pair_reaches_real_renderer_orchestration_and_construction_admission(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import materialize_scene_configuration_render_inputs
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha
    calls = []
    def raster(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["output_dir"])
        root.mkdir(parents=True)
        rows = []
        for camera in kwargs["cameras"]:
            path = root / (camera["camera_id"] + ".png")
            Image.new("RGB", (1024, 1024), (90, 80, 70)).save(path)
            rows.append({"camera_id": camera["camera_id"], "relative_path": path.name,
                         "digest": sha(path), "width": 1024, "height": 1024})
        result = {"schema_version": "sealed_camera_render_manifest.v1", "status": "rendered_exact_cameras",
            "authorization_class": "method_input", "splat_digest": kwargs["source_splat_digest"],
            "renders": rows, "render_count": len(rows)}
        result["sealed_camera_render_manifest_digest"] = canonical_digest(result, digest_field="sealed_camera_render_manifest_digest")
        return result
    def materialize(**kwargs):
        return materialize_scene_configuration_render_inputs(**kwargs, renderer=raster,
            runtime_resolver=lambda **_: {"node": "/fixture/node", "browser_executable": "/fixture/chrome",
                "renderer_root": "/fixture/renderer", "identity": {"runtime_digest": "sha256:" + "d" * 64,
                    "mode": "immutable_host_runtime", "source_commit": SHA, "full_byte_service_account_readback_passed": True}})
    envelope, _, _, _, _, _ = _prepare(tmp_path, monkeypatch, source_kind="gaussian_splat", render_materializer=materialize)
    render = envelope["render_inputs_result"]
    assert len(calls) == 1 and render["derived_frame_count"] == 8
    assert render["derived_gaussian_cutout"]["removed_count"] == 16
    assert render["derived_gaussian_cutout"]["retained_count"] == 48
    assert render["source_object_masks"]["observed_segmentation_truth"] is False


@pytest.mark.parametrize("source", ["mesh", "website"])
def test_completed_upload_reaches_execute_only_dispatch_boundary(tmp_path, monkeypatch, source):
    """Rehearse real consumer joins beyond the old preparation-context checks.

    Storage, accounting, and Website readiness use test transports. Real
    preparation and dispatcher checks run, stopping before allocator execution.
    """
    from datetime import datetime, timedelta, timezone
    import socket
    from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as activation
    from blueprint_pipeline.task_evaluation_launch_activation_worker import process_launch_activation_queue
    from scripts.prepare_paid_lane_launch import _load_scene_configuration_context, validate_paid_lane_launch, prepare_paid_lane_launch
    from tests.astra_toolchain_fixture import astra_toolchain_fixture
    from tests.test_task_evaluation_scene_configuration_activation_automation import _provider_zero, _publisher
    from tests.test_project_spend_reconciliation import _human_baseline
    from blueprint_pipeline.project_spend_reconciliation import materialize_project_spend_reconciliation

    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("rehearsal attempted network access"))

    if source == "website":
        from tests.test_website_native_submission import prepare_website_construction
        envelope = prepare_website_construction(tmp_path, monkeypatch, development=True)
    else:
        envelope, _, _, _, _, _ = _prepare(tmp_path, monkeypatch)
    request = envelope["request"]
    queue = tmp_path / "preparations"
    result_path = next((queue / "results").glob("*.json"))
    registry = tmp_path / "activation-intents"
    baseline, _ = _human_baseline(tmp_path / "baseline.json")
    project_spend = tmp_path / "project-spend.json"
    materialize_project_spend_reconciliation(
        baseline_authority_path=baseline, posted_reconciliation_paths=[], expected_coverage_ids=[],
        completeness_reference=str(baseline), authorized_by="fixture-owner",
        authorized_on=datetime.now(timezone.utc).isoformat(), output_path=project_spend)
    activation.provision_scene_configuration_activation_intent(
        expected_production_commit=SHA, team_namespace=request["team_namespace"],
        scene_id=request["scene"]["identity"]["id"], task_id=request["task"]["identity"]["id"],
        authorization_reference="scene-intent:" + request["scene_intent_digest"],
        authorized_by=ACCOUNT, profile_revision="rehearsal", valid_for_seconds=3600,
        project_spend_reconciliation_path=project_spend,
        rights_scope="internal_noncommercial_research_only",
        maximum_hard_cap_usd=request["spend"]["hard_cap_usd"], release_reference="development-rehearsal",
        intent_root=registry, materialization_root=tmp_path / "activation-inputs", release_scoped=True)
    lineage = _publisher("scene-configuration-activation-lineage")
    window = _publisher("coordinator-release-windows")
    now = datetime.now(timezone.utc)
    staged = activation.advance_scene_configuration_activation(
        preparation_result_path=result_path, preparation_queue_root=queue,
        activation_queue_root=tmp_path / "activation-queue", progression_root=tmp_path / "activation-progression",
        intent_root=registry, provider_zero_collector=lambda: _provider_zero(now - timedelta(seconds=1)),
        lineage_publisher_factory=lambda: lineage, release_window_publisher_factory=lambda: window,
        now=now, running_commit=SHA)
    assert staged["status"] == "scene_configuration_activation_queued", staged
    payloads = {**lineage.published, **window.published}

    def fetch(uri, destination, maximum_bytes):
        data = payloads[uri]
        assert len(data) <= maximum_bytes
        destination.write_bytes(data)

    plans = []

    def inspect_plan(*, lane, context_path, **_):
        context = _load_scene_configuration_context(context_path, expected_lane=lane)
        plans.append(validate_paid_lane_launch(lane, context))
        try:
            return prepare_paid_lane_launch(lane, context, runner=_preparation_runner(tmp_path, monkeypatch, website=source == "website"))
        except Exception as exc:
            pytest.fail(str(exc))

    controls = tmp_path / "controls-intents"
    controls.mkdir()
    run = process_launch_activation_queue(
        queue_root=tmp_path / "activation-queue", preparation_queue_root=queue,
        preparation_input_root=tmp_path / "worker-inputs", activation_root=tmp_path / "activations",
        allowed_uri_prefixes=["s3://blueprint/task-evaluation/production-inputs/"],
        service_account=ACCOUNT, service_group=grp.getgrgid(os.getegid()).gr_name, repository_root=Path(__file__).resolve().parents[1],
        destination_prefix="s3://blueprint/task-evaluation/production-inputs/rehearsal",
        release_window_prefix="s3://blueprint/task-evaluation/production-inputs/coordinator-release-windows/",
        profile_dir=tmp_path / "profiles", webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "authorizations", scene_construction_queue_root=tmp_path / "construction",
        scene_configuration_toolchain_root=astra_toolchain_fixture(tmp_path / "toolchain", SHA, monkeypatch),
        configured_controls_autostart_intent_root=controls, source_commit=SHA, fetcher=fetch, preparer=inspect_plan)
    assert len(plans) == 1, run
    assert plans[0]["status"] == "validated_no_commands_run"
    assert plans[0]["provider_allocation_performed"] is False
    assert plans[0]["paid_inference_performed"] is False
    assert {row["step_id"] for row in plans[0]["planned_steps"]} >= {
        "provider_bundle", "paid_authority", "allocator_dry_run", "live_profile", "standing_authorization"}
    assert run["results"][0]["status"] == "profile_authority_materialized_no_execution", run
    profile = json.loads(next((tmp_path / "profiles").glob("*.json")).read_text())
    assert profile["task_evaluation_run"]["evaluation_episode_executed"] is False
    _rehearse_dispatch(tmp_path, monkeypatch, profile)


def _preparation_runner(tmp_path, monkeypatch, *, website=False):
    """Execute the actual no-allocation graph, replacing only external I/O."""
    import hashlib
    import importlib
    from blueprint_pipeline import robot_eval_provider_input_setup as publication
    from blueprint_pipeline import task_evaluation_scene_configuration_vast as vast
    from tests.test_task_evaluation_scene_configuration_bundle import _configure_scene_openai_runtime_files

    objects = {}

    def upload(path, uri, *, exclusive):
        assert exclusive and uri not in objects
        objects[uri] = Path(path).read_bytes()
        digest = hashlib.sha256(objects[uri]).hexdigest()
        return {"status": "uploaded", "full_byte_readback_verified": True,
                "source_sha256": digest, "remote_sha256": digest,
                "source_size_bytes": len(objects[uri]), "remote_size_bytes": len(objects[uri]),
                "receipt_digest": "sha256:" + digest}

    monkeypatch.setattr(publication, "upload_file", upload)
    monkeypatch.setattr(vast, "_collect_openai_cost_snapshot", lambda **_: {"total_cost_usd": 0.0})
    _configure_scene_openai_runtime_files(tmp_path, monkeypatch)
    if website:
        for name in (*vast._OPENAI_RUNTIME_FILE_ENVS, *vast._OPENAI_RUNTIME_VALUE_ENVS):
            if "ARTIFIXER" in name:
                monkeypatch.delenv(name, raising=False)

    def run(argv):
        assert "--execute" not in argv
        if "blueprint_pipeline.paid_resource_allocator" in argv:
            def arg(name):
                return argv[argv.index(name) + 1]
            result = vast.run_scene_configuration_vast(
                job_dir=arg("--scene-configuration-job-dir"),
                bundle_receipt_path=arg("--scene-configuration-bundle-receipt"),
                paid_attempt_authority_path=arg("--scene-configuration-attempt-authority"),
                paid_resource_admission_grant=None, execute=False,
                scene_construction_queue_root=tmp_path / "construction")
            Path(arg("--adapter-output")).write_text(json.dumps(result))
            assert result["status"] == "dry_run_ready", result
            return 0
        if argv[1] == "-m":
            name, args = argv[2], argv[3:]
        else:
            name, args = "scripts." + Path(argv[1]).stem, argv[2:]
        assert name in {
            "blueprint_pipeline.task_evaluation_scene_configuration_bundle",
            "blueprint_pipeline.task_evaluation_scene_configuration_paid_authority",
            "scripts.publish_task_evaluation_immutable_manifest",
            "scripts.build_task_evaluation_scene_configuration_live_profile",
            "scripts.rehearse_lane_terminal_contract",
            "scripts.publish_task_evaluation_launch_profiles",
            "scripts.materialize_task_evaluation_standing_launch_authorization"}
        return importlib.import_module(name).main(args)

    return run


def _rehearse_dispatch(tmp_path, monkeypatch, profile):
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher
    from tests.test_task_evaluation_launch_dispatcher import _request
    request = _request(profile)
    request["authorization"]["spend"]["max_spend_usd"] = profile["allocator"]["max_spend_usd"]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    path = tmp_path / "dispatch-request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setenv(dispatcher.EXECUTE_ENV, "true")
    monkeypatch.setenv(dispatcher.SECRET_PROFILE_ID_ENV, profile["required_controls"]["secret_profile_id"])
    monkeypatch.setenv(dispatcher.STANDING_AUTHORIZATION_DIR_ENV, str(tmp_path / "authorizations"))
    calls = []

    def stop_at_allocator(argv):
        assert "--execute" in argv
        calls.append(argv)
        # All controller execute-only gates ran; no provider call or fake result.
        return 75

    receipt = dispatcher.dispatch_launch_request(
        request_path=path, profile_dir=tmp_path / "profiles", state_root=tmp_path / "dispatch",
        execute=True, allocator_runner=stop_at_allocator,
        publication_readiness_probe=lambda **_: {"status": "ready", "provider_mutation_performed": False, "spend_authority_granted": False})
    assert len(calls) == 1, receipt
    assert receipt["status"] == "blocked"
    assert receipt["provider_mutation_attempted"] is False
