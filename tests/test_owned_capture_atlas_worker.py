"""Owned-source adapter against retained bytes and the shared real worker lifecycle.

Provider/controller/archive responses are hermetic fixtures, not reconstruction
or robot-success evidence. Tests protect source identity, resumed stages,
revocation/cleanup, explicit queue admission and private owner downloads.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest

from blueprint_pipeline import owned_capture_atlas_worker as worker
from blueprint_pipeline import website_atlas_pipeline as controller
from blueprint_pipeline import website_atlas_pose as pose
from blueprint_pipeline import website_atlas_preview as lifecycle
from blueprint_pipeline import task_evaluation_configured_scene_object_store as archives
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.owned_capture_atlas_http import register_owned_capture_atlas_routes
from blueprint_pipeline.reconstruction_control_plane import prepare_owned_capture_atlas_plan
from blueprint_pipeline.website_atlas_preview_store import PreviewUnavailable
from tests.test_reconstruction_control_plane import _seed_owned_atlas_source
from tests.test_owned_capture_atlas_project import configuration as configuration


@pytest.fixture(autouse=True)
def configuration_transport(monkeypatch):
    from blueprint_pipeline import owned_capture_atlas_control
    # Both worker execution and CPU queue admission use the same signed loader;
    # existing controller test doubles must also cover the extracted CPU seam.
    monkeypatch.setattr(owned_capture_atlas_control, "load_configuration",
                        lambda *args, **kwargs: controller.load_configuration(*args, **kwargs))


@pytest.fixture
def owned(tmp_path):
    state, store = tmp_path / "work" / "reconstruction_control_plane", tmp_path / "store"
    session, intake, digest = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=["navigation_clearance"],
        idempotency_key="owned-plan-1")
    return SimpleNamespace(state=state, store=store, plan=plan, session=session, intake=intake,
        digest=digest, path=state / "plans" / plan["plan_id"] / "atlas" / worker.JOB_FILE)


@pytest.fixture
def context(owned):
    return owned.plan["source_context"]


def _arguments(owned):
    return dict(state_root=owned.state, capture_store_root=owned.store, plan_id=owned.plan["plan_id"],
        task_context_digest=owned.plan["source_context"]["context_digest"], source_video_digest=owned.digest,
        owner_user_id="owner", idempotency_key="owned-execution-1")


def _enqueue(owned, configuration, monkeypatch):
    calls = []

    def configuration_read(context, source_digest, *, create=True):
        calls.append((context["context_digest"], source_digest, create))
        assert create is False
        return configuration

    monkeypatch.setattr(controller, "load_configuration", configuration_read)
    value = worker.enqueue_owned_atlas(**_arguments(owned))
    assert len(calls) == 1
    return value


def _advance(owned, **kwargs):
    return worker.advance_job(state_root=owned.state, capture_store_root=owned.store,
                               plan_id=owned.plan["plan_id"], **kwargs)


def _revoke(owned):
    marker = hashlib.sha256(f"{owned.session}\0{owned.intake}".encode()).hexdigest()
    path = owned.store / "lifecycle_tombstones" / f"{marker}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"revoked":true}')


def test_queue_is_noncreating_idempotent_and_does_not_invoke_provider(owned, configuration, monkeypatch):
    monkeypatch.setattr(controller, "run_atlas_pipeline", lambda **_: pytest.fail("queue ran controller"))
    value = _enqueue(owned, configuration, monkeypatch)
    assert value["state"] == "processing" and value["stage"] == "queued"
    assert value["proof_boundary"]["robot_execution_established"] is False
    assert worker.enqueue_owned_atlas(**_arguments(owned)) == value
    assert worker.enqueue_owned_atlas(**{**_arguments(owned), "idempotency_key": "another-execution-key"}) == value
    assert len(list(owned.state.glob("plans/*/atlas/owned_job.json"))) == 1
    assert not list(owned.state.rglob("site-*"))
    assert not list(owned.state.rglob("walkthrough-*"))


@pytest.mark.parametrize("field,value", [("owner_user_id", "different-owner"),
    ("task_context_digest", "sha256:" + "f" * 64), ("source_video_digest", "sha256:" + "f" * 64)])
def test_queue_rejects_wrong_owner_source_and_task_before_job(owned, configuration, monkeypatch, field, value):
    monkeypatch.setattr(controller, "load_configuration", lambda *_, **__: configuration)
    with pytest.raises(ValueError):
        worker.enqueue_owned_atlas(**{**_arguments(owned), field: value})
    assert not owned.path.exists()


def test_queue_rechecks_revocation_after_remote_authority_read(owned, configuration, monkeypatch):
    def read(*_, **__):
        _revoke(owned)
        return configuration
    monkeypatch.setattr(controller, "load_configuration", read)
    with pytest.raises(ValueError, match="capture_revoked_or_expired"):
        worker.enqueue_owned_atlas(**_arguments(owned))
    assert not owned.path.exists()


def test_worker_resumes_same_original_source_and_archives_without_execution_claim(owned, configuration, monkeypatch):
    _enqueue(owned, configuration, monkeypatch)
    calls, published = [], []
    from tests.test_owned_capture_atlas_reconstruction import _write_retained_result
    retained_result, _ = _write_retained_result(owned.path.parent, owned.plan)
    bundle = owned.path.parent / "download.zip"
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()

    def run(**options):
        calls.append(options)
        assert options["source_video"].read_bytes() == b"retained-video"
        assert options["task_context"] == owned.plan["source_context"]
        assert options["max_polls"] == 1
        assert options["output_root"] == owned.path.parent
        options["checkpoint"]("test-before-provider")
        if len(calls) == 1:
            return {"status": "processing", "stage": "source_pose"}
        return retained_result

    def publish(**options):
        published.append(options)
        return {"uri": "s3://private-test/retained-bundle", "digest": digest, "size_bytes": bundle.stat().st_size,
                "remote_verified_at": "hermetic-readback"}

    monkeypatch.setattr(controller, "run_atlas_pipeline", run)
    monkeypatch.setattr(archives, "verify_dedicated_artifact_store", lambda: None)
    monkeypatch.setattr(archives, "publish_configured_scene_artifact", publish)
    assert _advance(owned) == "processing"
    assert _advance(owned) == "ready"
    assert _advance(owned) == "ready"  # A terminal read never buys the scene again.
    assert len(calls) == 2 and len(published) == 1
    value = worker.inspect_owned_atlas(state_root=owned.state, capture_store_root=owned.store,
        plan_id=owned.plan["plan_id"], owner_user_id="owner")
    assert value["bundle"]["digest"] == digest
    assert value["provider_cleanup_state"] == "complete"
    assert value["scale_status"] == "unscaled_provider_world_units"
    assert not any(value["proof_boundary"].values())
    assert value["execution_digest"] == cross_runtime_canonical_digest(value, digest_field="execution_digest")
    assert value["reconstruction"]["execution_result"]["cost_usd"] is None
    assert json.loads((owned.path.parent / "owned_status.json").read_text()) == worker.projection(json.loads(owned.path.read_text()))


def test_revoked_owned_source_stops_controller_and_cleans_retained_assets(owned, configuration, monkeypatch):
    _enqueue(owned, configuration, monkeypatch)
    monkeypatch.setattr(controller, "run_atlas_pipeline", lambda **_: pytest.fail("revoked source reached controller"))
    asset_path = owned.path.parent / "atlas_pose" / "submission.json"
    asset_path.parent.mkdir()
    asset_path.write_text('{"created_assets":["source-asset"]}')
    monkeypatch.setattr(pose, "atlas_api_config", lambda: ("https://api.example.test/api/v2", "hermetic-key"))
    sent = []

    def transport(method, url, headers, data):
        sent.append((method, url))
        return 200, b'{}'

    _revoke(owned)
    assert _advance(owned, transport=transport) == "withdrawn"
    assert sent == [("DELETE", "https://api.example.test/api/v2/assets/source-asset")]
    value = json.loads(owned.path.read_text())
    assert value["reason"] == "capture_revoked_or_expired"
    assert value["provider_cleanup"]["state"] == "complete"
    assert value["provider_cleanup"]["deleted"] == ["source-asset"]


def test_incomplete_completion_is_not_ready_or_downloadable(owned, configuration, monkeypatch):
    _enqueue(owned, configuration, monkeypatch)
    monkeypatch.setattr(controller, "run_atlas_pipeline", lambda **_: {
        "status": "ready", "stage": "completion", "digest": "sha256:" + "b" * 64,
        "scene_completion": {"status": "needs_input", "reason": "atlas_geometry_gap_unfilled"},
        "download": {"zip_path": "must-not-be-read"}})
    monkeypatch.setattr(lifecycle, "_archive", lambda *_: pytest.fail("incomplete scene was archived"))
    assert _advance(owned) == "needs_input"
    with pytest.raises(PreviewUnavailable):
        worker.open_owned_bundle(state_root=owned.state, capture_store_root=owned.store,
            plan_id=owned.plan["plan_id"], owner_user_id="owner", digest="sha256:" + "b" * 64)


def test_lock_blocks_duplicate_step_and_deadline_does_not_reset(owned, configuration, monkeypatch):
    _enqueue(owned, configuration, monkeypatch)
    monkeypatch.setattr(controller, "run_atlas_pipeline", lambda **_: pytest.fail("locked/expired job ran"))
    with (owned.path.parent / lifecycle.LOCK_FILE).open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _advance(owned) == "busy"
    original = json.loads(owned.path.read_text())["enqueued_at_epoch"]
    assert _advance(owned, now=lambda: original + lifecycle.MAXIMUM_JOB_SECONDS + 1) == "failed"
    assert json.loads(owned.path.read_text())["enqueued_at_epoch"] == original


def test_signed_webapp_issuer_is_required_and_authority_fields_are_not_client_input(owned, configuration, monkeypatch):
    monkeypatch.setattr(controller, "load_configuration", lambda *_, **__: configuration)
    app = FastAPI()

    def already_authenticated(request: Request):
        # Only this fixture substitutes the existing signature verifier; the
        # route must still reject an authenticated non-WebApp service issuer.
        request.state.intake_client_id = request.headers.get("x-test-verified-issuer", "")

    register_owned_capture_atlas_routes(app, require_admission=already_authenticated,
                                         roots=lambda: (owned.state, owned.store))
    client = TestClient(app)
    args = _arguments(owned)
    body = {"schema_version": "owned_capture_atlas_execution_submission.v1",
            **{k: v for k, v in args.items() if k not in {"state_root", "capture_store_root"}}}
    url = "/api/live-pipeline/reconstructions/atlas-execute"
    assert client.post(url, json=body).status_code == 403
    headers = {"x-blueprint-pipeline-signature": "fixture-verified", "x-test-verified-issuer": "blueprint-webapp"}
    assert client.post(url, headers=headers, json={**body, "provider_identity": {"project_id": "forged"}}).status_code == 422
    result = client.post(url, headers=headers, json=body)
    assert result.status_code == 200 and result.json()["state"] == "processing"
    read = {"schema_version": "owned_capture_atlas_inspection_submission.v1",
            "plan_id": owned.plan["plan_id"], "owner_user_id": "owner"}
    inspected = client.post('/api/live-pipeline/reconstructions/atlas-inspect', headers=headers, json=read)
    assert inspected.status_code == 200 and inspected.json() == result.json()
    assert client.post('/api/live-pipeline/reconstructions/atlas-inspect', headers=headers,
                      json={**read, "owner_user_id": "another-owner"}).status_code == 409


def test_existing_timer_is_safe_with_no_owned_configuration(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline.capture_upload_intake import CAPTURE_UPLOAD_STORE_ROOT_ENV
    monkeypatch.delenv(CAPTURE_UPLOAD_STORE_ROOT_ENV, raising=False)
    assert worker.tick_from_environment() == {}
    assert lifecycle.main(["tick", "--storage-root", str(tmp_path / "empty-storage")]) == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_existing_timer_discovers_explicit_owned_job_from_same_server_roots(owned, configuration, monkeypatch, tmp_path, capsys):
    from blueprint_pipeline.capture_upload_intake import CAPTURE_UPLOAD_STORE_ROOT_ENV
    from blueprint_pipeline.live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
    from blueprint_pipeline.live_pipeline_intake_runtime_controls import INTAKE_WORK_DIR_ENV
    _enqueue(owned, configuration, monkeypatch)
    manifest = tmp_path / "live-manifest.json"
    manifest.write_text('{}')
    monkeypatch.setenv(CAPTURE_UPLOAD_STORE_ROOT_ENV, str(owned.store))
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest))
    monkeypatch.setenv(INTAKE_WORK_DIR_ENV, str(owned.state.parent))
    calls = []
    monkeypatch.setattr(controller, "run_atlas_pipeline", lambda **options: calls.append(options) or {
        "status": "processing", "stage": "source_pose"})
    assert lifecycle.main(["tick", "--storage-root", str(tmp_path / "empty-website-storage")]) == 0
    assert json.loads(capsys.readouterr().out) == {"owned_processing": 1}
    assert len(calls) == 1
    assert calls[0]["task_context"]["capture_id"] == owned.session


def test_image_network_boundary_keeps_original_source_fence_across_threads(owned):
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import Request
    from blueprint_pipeline.atlas_preparation_runtime import preparation_disclosure_opener
    from blueprint_pipeline.owned_capture_atlas_runtime import bind_owned_atlas_source
    calls = []
    def opener(request, **_):
        calls.append(request.get_method())
        return "hermetic-response"
    with bind_owned_atlas_source(state_root=owned.state, capture_store_root=owned.store,
                                 plan_id=owned.plan["plan_id"]) as context:
        captured = preparation_disclosure_opener(context, opener)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(captured, Request("https://provider.example.test/edit", method="POST")).result() == "hermetic-response"
        _revoke(owned)
        with pytest.raises(ValueError, match="capture_revoked_or_expired"):
            pool.submit(captured, Request("https://provider.example.test/edit", method="POST")).result()
        assert pool.submit(captured, Request("https://provider.example.test/output", method="GET")).result() == "hermetic-response"
    assert calls == ["POST", "GET"]


def test_explicit_owned_retry_reuses_prefix_is_idempotent_and_does_not_renew_allowance(owned, configuration, monkeypatch):
    _enqueue(owned, configuration, monkeypatch)
    retained = owned.path.parent / "completed-stage.bin"
    retained.write_bytes(b"already purchased stage")
    initial = json.loads(owned.path.read_text())
    authority = json.dumps(configuration["authority"], sort_keys=True)

    def refuse(**_):
        raise ValueError("owned_fixture_replay_boundary")
    monkeypatch.setattr(controller, "run_atlas_pipeline", refuse)
    assert _advance(owned) == "failed"
    args = {**_arguments(owned), "idempotency_key": "owned-retry-1"}
    retried = worker.retry_owned_atlas(**args)
    assert retried["state"] == "processing"
    assert worker.retry_owned_atlas(**args) == retried
    current = json.loads(owned.path.read_text())
    assert len(current["retries"]) == 1
    assert current["retries"][0]["reason"] == "owned_fixture_replay_boundary"
    assert current["enqueued_at_epoch"] == initial["enqueued_at_epoch"]
    assert current["authorization_digest"] == initial["authorization_digest"]
    assert json.dumps(configuration["authority"], sort_keys=True) == authority
    assert retained.read_bytes() == b"already purchased stage"


@pytest.mark.parametrize("change", ["owner", "source", "revoked", "authority", "expired"])
def test_owned_retry_refuses_changed_binding_and_expired_clock(owned, configuration, monkeypatch, change):
    _enqueue(owned, configuration, monkeypatch)
    job = json.loads(owned.path.read_text())
    job.update(state="failed", reason="owned_fixture_replay_boundary")
    owned.path.write_text(json.dumps(job))
    args = {**_arguments(owned), "idempotency_key": "owned-retry-1"}
    if change == "owner":
        args["owner_user_id"] = "different-owner"
    if change == "source":
        args["source_video_digest"] = "sha256:" + "f" * 64
    if change == "revoked":
        _revoke(owned)
    if change == "authority":
        configuration["authority"]["original_allowance_digest"] = "sha256:" + "7" * 64
        configuration["authority"]["authorization_digest"] = cross_runtime_canonical_digest(
            configuration["authority"], digest_field="authorization_digest")
    if change == "expired":
        args["now"] = lambda: job["enqueued_at_epoch"] + lifecycle.MAXIMUM_JOB_SECONDS + 1
    with pytest.raises(ValueError, match="job_authority_changed" if change == "authority" else ".*"):
        worker.retry_owned_atlas(**args)
    assert json.loads(owned.path.read_text()) == job
