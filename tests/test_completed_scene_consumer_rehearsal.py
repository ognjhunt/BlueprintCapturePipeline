"""Run completed uploads through real consumers; fake only external transports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import zipfile
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
    from tests.test_task_evaluation_scene_configuration_bundle import _toolchain
    envelope_path = tmp_path / "consumer-envelope.json"
    envelope_path.write_text(json.dumps(envelope))
    bundle = build_scene_configuration_provider_bundle(construction_envelope_path=envelope_path,
        toolchain_root=_toolchain(tmp_path / "toolchain", SHA), repository_root=Path(__file__).resolve().parents[1],
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
