"""Consented Atlas previews run beside preparation and end in plain, delivered states (D1-D6, 2026-10-22)."""
import fcntl
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline import website_atlas_preview_store as store
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.delenv(preview.ALLOWLIST_ENV, raising=False)
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    return SimpleNamespace(storage=tmp_path / "storage", capture=capture, pipeline=capture / "pipeline",
                           job=capture / "pipeline/atlas" / preview.JOB_FILE)


def _descriptor(capture_id="walkthrough-req1", request="req1"):
    return SimpleNamespace(site_submission_id=request, metadata={}, capture_id=capture_id, scene_id=f"site-{request}")


def _enqueue(site, descriptor=None, source=None):
    return preview.enqueue_preview(descriptor=descriptor or _descriptor(), capture_root=site.capture,
                                   pipeline_dir=site.pipeline,
                                   source_video=source or (lambda: site.capture / "raw/walkthrough.mov"))


def _withdraw(site):
    marker = site.capture.parent.parent / "website_withdrawal/tombstone.json"
    marker.parent.mkdir(parents=True)
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": "site-req1", "request_id": "req1"}
    value["digest"] = canonical_digest(value, digest_field="digest")
    marker.write_text(json.dumps(value))


def _job(site):
    return json.loads(site.job.read_text())


class World:
    """The task context, controller, archive store, WebApp and World Labs, as each answers."""

    def __init__(self, monkeypatch, site, steps):
        from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
        from blueprint_pipeline import website_atlas_pipeline as pipeline, website_atlas_pose as pose
        from blueprint_pipeline import website_task_context as context
        self.site, self.steps, self.runs, self.pushed, self.deleted, self.published = site, list(steps), [], [], [], []
        # The worker's clock starts where intake recorded the job.
        self.clock, self.refuse_push, self.missing = _job(site)["enqueued_at_epoch"], None, {"gone1"}
        self.operations, self.cancelled = {}, []
        monkeypatch.setattr(context, "load_current_website_task_context",
                            lambda **kw: {"context_digest": "sha256:" + "c" * 64, **kw})
        monkeypatch.setattr(pipeline, "run_atlas_pipeline", self.run)
        monkeypatch.setattr(store, "verify_dedicated_artifact_store", lambda: None)
        monkeypatch.setattr(store, "publish_configured_scene_artifact", self.publish)
        monkeypatch.setattr(context, "website_webapp_request", self.push)
        monkeypatch.setattr(pose, "atlas_api_config", lambda: ("https://api.example.worldlabs.ai/api/v2", "key"))

    def now(self):
        return self.clock

    def run(self, **kwargs):
        self.runs.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(kwargs) if callable(step) else step

    def publish(self, *, path, artifact_kind, filename):
        data = Path(path).read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        self.published.append((artifact_kind, filename))
        return {"uri": f"s3://bucket/blueprint/artifacts/{artifact_kind}/sha256/{digest[7:]}/{filename}",
                "digest": digest, "size_bytes": len(data), "remote_verified_at": "2026-10-22T00:00:00Z"}

    def push(self, *, capture_id, operation, payload):
        assert operation == "atlas-preview" and capture_id == "walkthrough-req1"
        if self.refuse_push:
            raise ValueError(self.refuse_push)
        self.pushed.append(payload["preview"])
        return {"schema_version": "website_atlas_preview_acceptance.v1", "accepted": True,
                "revision": payload["preview"]["revision"], "preview_digest": payload["preview"]["preview_digest"]}

    def transport(self, method, url, headers, data):
        assert headers["WLT-Api-Key"] == "key"
        if "/operations/" in url:
            operation = url.rsplit("/", 1)[1]
            if method == "POST":
                self.cancelled.append(operation.removesuffix(":cancel"))
                return 200, b"{}"
            value = self.operations.get(operation)
            return (404, b"{}") if value is None else (200, json.dumps({"id": operation, **value}).encode())
        assert method == "DELETE"
        asset = url.rsplit("/", 1)[1]
        self.deleted.append(asset)
        return (404, b'{"error":"gone"}') if asset in self.missing else (200, b"{}")

    def advance(self):
        return preview.advance_job(self.site.job, now=self.now, transport=self.transport)


def _completed(kwargs):
    """What the controller leaves after a full run: records naming every asset, and the bundle."""
    root = kwargs["output_root"]
    for relative, value in {"pose/atlas_pose/submission.json": {"created_assets": ["upload1", "gone1"]},
                            "pose/atlas_posed_rgbd_result.json": {"operation_receipt": {"input_usage": {"frames": [
                                {"outputs": [{"asset_id": "view1", "path": "x"}]}]}}},
                            "mesh/atlas_pose/submission.json": {"created_assets": ["splat1"]},
                            "mesh/result.json": {"outputs": [{"asset_id": "mesh1", "role": "mesh"}]}}.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(json.dumps(value))
    bundle = root / "downloads/atlas-scene-0123.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", "{}")
    digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    return {"schema_version": "website_atlas_pipeline_result.v1", "status": "needs_input", "digest": "sha256:" + "d" * 64,
            "splat_available": True, "mesh_available": True,
            "download": {"zip_path": str(bundle), "zip_sha256": digest, "manifest_digest": "sha256:" + "e" * 64}}


def test_routing_is_off_unless_set_and_an_unreadable_setting_routes_nothing(monkeypatch, site):
    monkeypatch.delenv(preview.ROUTING_ENV)
    resolved = []
    assert _enqueue(site, source=lambda: resolved.append(1)) is None
    assert resolved == [] and not site.job.exists()  # an unrouted upload is never read for this
    for value, expected in (("off", False), ("Consented", False), ("allow list", False), ("consented", True)):
        monkeypatch.setenv(preview.ROUTING_ENV, value)
        assert preview.routed("walkthrough-req1") is expected
    monkeypatch.setenv(preview.ROUTING_ENV, "allowlist")
    monkeypatch.setenv(preview.ALLOWLIST_ENV, '["walkthrough-req1"]')
    assert preview.routed("walkthrough-req1") and not preview.routed("walkthrough-req2")
    for malformed in ('{"walkthrough-req1": true}', "[walkthrough-req1]", '["walkthrough-req1", 3]'):
        monkeypatch.setenv(preview.ALLOWLIST_ENV, malformed)
        assert not preview.routed("walkthrough-req1")


def test_routed_walkthroughs_get_one_job_and_intake_never_fails_for_it(site, tmp_path):
    job = _enqueue(site)
    assert (job["state"], job["source_video"], job["revision"]) == ("processing", "raw/walkthrough.mov", 1)
    assert (site.job.parent / preview.LOCK_FILE).is_file()  # withdrawal deletion holds the same lock
    assert _enqueue(site) == job == _job(site)  # a redelivered upload keeps its job
    assert _enqueue(site, descriptor=_descriptor(capture_id="supplement-" + "a" * 40)) is None
    assert _enqueue(site, descriptor=_descriptor(request="other")) is None
    assert preview.enqueue_preview(descriptor=None, capture_root=site.capture, pipeline_dir=site.pipeline,
                                   source_video=lambda: None) is None
    site.job.unlink()
    outside = tmp_path / "elsewhere.mov"
    outside.write_bytes(b"x")
    failed = _enqueue(site, source=lambda: outside)
    assert (failed["state"], failed["reason"]) == ("failed", "atlas_preview_source_video_unavailable")
    site.job.unlink()
    _withdraw(site)
    assert _enqueue(site) is None and not site.job.exists()


def test_a_routed_upload_becomes_an_archived_preview_and_leaves_nothing_at_world_labs(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [{"status": "processing", "stage": "pose"}, _completed])
    assert world.advance() == "processing"
    assert world.runs[0]["max_polls"] == 1 and world.runs[0]["output_root"] == site.job.parent
    assert world.runs[0]["source_video"] == site.capture / "raw/walkthrough.mov"
    assert _job(site)["stage"] == "pose" and world.published == [] and world.deleted == []
    world.clock += 60
    assert world.advance() == "ready"
    job = _job(site)
    assert world.published == [("atlas-scene-bundle", "walkthrough-req1.zip")]
    assert job["archive"]["filename"] == "walkthrough-req1.zip" and job["revision"] == 2
    assert sorted(world.deleted) == ["gone1", "mesh1", "splat1", "upload1", "view1"]
    assert job["provider_cleanup"]["state"] == "complete" and job["provider_cleanup"]["remaining"] == []
    assert [row["state"] for row in world.pushed] == ["processing", "ready"]
    ready = world.pushed[-1]
    assert ready["bundle"] == {key: job["archive"][key] for key in ("digest", "size_bytes", "filename")}
    assert ready["claim_ceiling"] == "development_only" and ready["scale_status"] == "unscaled_provider_world_units"
    assert ready["splat_available"] and ready["mesh_available"]
    assert ready["preview_digest"] == cross_runtime_canonical_digest(
        {key: value for key, value in ready.items() if key != "preview_digest"})
    # A finished job is left alone: no more runs, pushes, deletions or writes.
    before = site.job.read_text()
    world.clock += 7200
    assert world.advance() == "ready" and site.job.read_text() == before
    assert len(world.runs) == 2 and len(world.pushed) == 2


def test_rejected_background_without_download_waits_without_publishing_a_bundle(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [{"status": "needs_input", "download": None,
                                       "digest": "sha256:rejected"}])
    assert world.advance() == "needs_input"
    job = _job(site)
    assert job["reason"] == "atlas_background_review_required" and job["archive"] is None
    assert world.published == [] and world.pushed[-1]["bundle"] is None


def test_a_withdrawal_stops_the_job_before_the_next_paid_stage_and_cleans_world_labs(monkeypatch, site):
    _enqueue(site)

    def withdrawn_before_pose(kwargs):
        (kwargs["output_root"] / "pose/atlas_pose").mkdir(parents=True)
        (kwargs["output_root"] / "pose/atlas_pose/submission.json").write_text('{"created_assets": ["upload1"]}')
        _withdraw(site)
        kwargs["checkpoint"]("pose")
        raise AssertionError("the checkpoint must stop the run")
    world = World(monkeypatch, site, [withdrawn_before_pose])
    assert world.advance() == "withdrawn"
    job = _job(site)
    assert (job["reason"], job["archive"]) == ("website_capture_withdrawn", None) and world.published == []
    assert world.deleted == ["upload1"] and [row["state"] for row in world.pushed] == ["withdrawn"]
    assert world.pushed[0]["bundle"] is None
    world.clock += 60
    assert world.advance() == "withdrawn" and len(world.runs) == 1


def test_a_tombstone_seen_at_the_next_tick_ends_the_job_without_running(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [])
    _withdraw(site)
    assert world.advance() == "withdrawn" and world.runs == []


@pytest.mark.parametrize("refusal,reason", [
    (ValueError("website_control_atlas-configuration_http_409:website_atlas_source_consent_required"),
     "website_atlas_source_consent_required"),
    # A lapsed price review is the owner's to renew, like a missing grant.
    (ValueError("atlas_runtime_price_review_expired"), "atlas_runtime_price_review_expired"),
])
def test_a_missing_grant_waits_then_resumes_when_the_owner_grant_exists(monkeypatch, site, refusal, reason):
    _enqueue(site)
    world = World(monkeypatch, site, [refusal, {"status": "processing", "stage": "plan"}])
    assert world.advance() == "needs_input"
    job = _job(site)
    assert job["reason"] == reason
    assert job["retry_at_epoch"] == world.clock + preview.AUTHORITY_RECHECK_SECONDS
    assert job["provider_cleanup"] is None  # still waiting, so nothing is torn down yet
    world.clock += 60
    assert world.advance() == "needs_input" and len(world.runs) == 1
    world.clock += preview.AUTHORITY_RECHECK_SECONDS
    assert world.advance() == "processing"
    # Each tick tells the WebApp only its newest state.
    assert [(row["state"], row["reason"], row["revision"]) for row in world.pushed] == [
        ("needs_input", reason, 2), ("processing", None, 3)]


@pytest.mark.parametrize("error,state,reason", [
    (ValueError("atlas_runtime_budget_insufficient"), "failed", "atlas_runtime_budget_insufficient"),
    (ValueError("atlas_scene_planner_abstained:people move through every frame"), "needs_input",
     "atlas_scene_planner_abstained"),
    (ValueError("website_control_task-context_http_409:source_revoked"), "withdrawn", "website_capture_withdrawn"),
    (type("AtlasHttpError", (RuntimeError,), {"status": 403})("website_atlas_api_403:forbidden"), "failed",
     "website_atlas_api_403"),
])
def test_typed_refusals_end_in_one_plain_final_state(monkeypatch, site, error, state, reason):
    _enqueue(site)
    world = World(monkeypatch, site, [error])
    assert world.advance() == state
    job = _job(site)
    assert job["reason"] == reason and job["retry_at_epoch"] is None
    world.clock += preview.MAXIMUM_JOB_SECONDS
    assert world.advance() == state and len(world.runs) == 1


def test_transport_failures_back_off_and_end_failed_once_retries_are_exhausted(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [OSError("connection reset")] * preview.MAXIMUM_TRANSIENT_FAILURES)
    world.advance()
    job = _job(site)
    assert job["state"] == "processing" and job["transient_failures"] == 1 and job["next_attempt_at_epoch"] > world.clock
    assert job["last_transient_reason"] == "atlas_preview_oserror"
    world.clock += 30
    world.advance()
    assert len(world.runs) == 1  # not before the backoff ends
    for _ in range(preview.MAXIMUM_TRANSIENT_FAILURES - 1):
        world.clock = _job(site)["next_attempt_at_epoch"]
        world.advance()
    job = _job(site)
    assert (job["state"], job["reason"]) == ("failed", "atlas_preview_retries_exhausted")
    assert len(world.runs) == preview.MAXIMUM_TRANSIENT_FAILURES


def test_a_webapp_withdrawal_answer_ends_the_job(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [{"status": "processing", "stage": "plan"}])
    world.refuse_push = "website_control_atlas-preview_http_409:source_revoked"
    assert world.advance() == "withdrawn"
    world.refuse_push = None
    world.clock += 60
    assert world.advance() == "withdrawn" and world.pushed[-1]["state"] == "withdrawn"


def test_a_job_another_worker_holds_is_skipped(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [])
    with (site.job.parent / "preview.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert world.advance() == "busy"
    assert world.runs == []


def test_tick_advances_every_job_and_one_bad_job_never_stops_the_others(monkeypatch, site):
    _enqueue(site)
    world = World(monkeypatch, site, [{"status": "processing", "stage": "plan"}])
    other = site.storage / "bucket/scenes/site-req2/captures/walkthrough-req2/pipeline/atlas" / preview.JOB_FILE
    other.parent.mkdir(parents=True)
    other.write_text("not json")
    assert preview.tick(site.storage, now=world.now) == {"processing": 1, "unreadable": 1}


class _Body(io.BytesIO):
    def iter_chunks(self, size):
        while chunk := self.read(size):
            yield chunk


class _Bucket:
    def __init__(self, key, data, digest):
        self.objects = {key: (data, {"sha256": digest[7:]})}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key][0]), "Metadata": self.objects[Key][1]}

    def get_object(self, *, Bucket, Key):
        return {"Body": _Body(self.objects[Key][0])}


def _ready(site, data=b"zip bytes"):
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    site.job.parent.mkdir(parents=True, exist_ok=True)
    _enqueue(site)
    site.job.write_text(json.dumps({**_job(site), "state": "ready", "archive": {"digest": digest}}))
    from blueprint_pipeline.task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/atlas-scene-bundle/sha256/{digest[7:]}/walkthrough-req1.zip"
    return digest, _Bucket(key, data, digest)


def test_owner_downloads_read_only_the_ready_unwithdrawn_bundle_named_for_the_capture(site):
    digest, bucket = _ready(site)
    body, size = store.open_bundle(request_id="req1", capture_id="walkthrough-req1", digest=digest,
                                     storage_root=site.storage, client=bucket, bucket="b")
    assert body.read() == b"zip bytes" and size == 9
    refused = [dict(request_id="req1", capture_id="walkthrough-req2", digest=digest),
               dict(request_id="req1", capture_id="walkthrough-req1", digest="sha256:" + "0" * 64),
               dict(request_id="../req1", capture_id="walkthrough-../req1", digest=digest),
               dict(request_id="*", capture_id="walkthrough-*", digest=digest)]
    for selectors in refused:
        with pytest.raises(store.PreviewUnavailable):
            store.open_bundle(**selectors, storage_root=site.storage, client=bucket, bucket="b")
    # After the workspace is retired only the object named for this capture and digest remains.
    assert store.open_bundle(request_id="req1", capture_id="walkthrough-req1", digest=digest,
                               storage_root=site.storage / "elsewhere", client=bucket, bucket="b")[1] == 9
    _withdraw(site)
    with pytest.raises(store.PreviewUnavailable):
        store.open_bundle(request_id="req1", capture_id="walkthrough-req1", digest=digest,
                            storage_root=site.storage, client=bucket, bucket="b")


def test_download_route_streams_with_digest_headers_only_for_the_webapp_client(monkeypatch, site):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from blueprint_pipeline import website_atlas_preview_http as http

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    digest, bucket = _ready(site)
    monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(site.storage))
    monkeypatch.setattr(http, "open_bundle", lambda **kw: store.open_bundle(**kw, client=bucket, bucket="b"))
    client = TestClient(app)
    url = f"/api/live-pipeline/website-atlas-previews/walkthrough-req1/bundle?request_id=req1&digest={digest}"
    signed = {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}
    response = client.get(url, headers=signed)
    assert response.status_code == 200 and response.content == b"zip bytes"
    assert response.headers["x-blueprint-artifact-sha256"] == digest and response.headers["content-length"] == "9"
    assert response.headers["content-type"] == "application/zip"
    assert "atlas-preview-walkthrough-req1.zip" in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get(url, headers={**signed, "x-test-client": "robot-team"}).status_code == 403
    assert client.get(url.replace("req1&", "req2&"), headers=signed).status_code == 404

    def broken(**_kw):
        raise RuntimeError("https://signed.example/secret?X-Amz-Signature=abc")
    monkeypatch.setattr(http, "open_bundle", broken)
    failed = client.get(url, headers=signed)
    assert failed.status_code == 503 and "secret" not in failed.text


def test_a_withdrawal_after_the_last_paid_stage_still_prevents_the_archive(monkeypatch, site):
    _enqueue(site)

    def finished_then_withdrawn(kwargs):
        result = _completed(kwargs)
        _withdraw(site)  # during fusion or export, after the mesh checkpoint
        return result
    world = World(monkeypatch, site, [finished_then_withdrawn])
    assert world.advance() == "withdrawn"
    assert world.published == [] and _job(site)["archive"] is None
    assert world.pushed[-1]["state"] == "withdrawn" and world.pushed[-1]["bundle"] is None


class _Versions:
    """``list_object_versions``/``delete_object`` as boto3 answers them on B2."""

    def __init__(self):
        self.objects, self.deleted = {}, []

    def list_object_versions(self, *, Bucket, Prefix, **_markers):
        rows = [{"Key": key, "VersionId": version} for key, versions in self.objects.items()
                if key.startswith(Prefix) for version in versions]
        return {"Versions": rows, "DeleteMarkers": [], "IsTruncated": False}

    def delete_object(self, *, Bucket, Key, VersionId):
        self.deleted.append((Key, VersionId))
        self.objects[Key] = [version for version in self.objects[Key] if version != VersionId]


def test_a_copy_published_while_the_withdrawal_landed_is_taken_back(monkeypatch, site):
    from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
    _enqueue(site)
    world = World(monkeypatch, site, [_completed])
    versions = _Versions()
    monkeypatch.setattr(deletion.B2ObjectVersions, "configured", classmethod(lambda cls: cls(versions, "bucket")))
    publish = world.publish

    def publish_while_withdrawn(**kwargs):
        reference = publish(**kwargs)
        versions.objects[reference["uri"].split("s3://bucket/", 1)[1]] = ["v1", "v2"]
        _withdraw(site)
        return reference
    monkeypatch.setattr(world, "publish", publish_while_withdrawn)
    from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
    monkeypatch.setattr(store, "publish_configured_scene_artifact", publish_while_withdrawn)
    assert world.advance() == "withdrawn"
    job = _job(site)
    assert job["archive_retraction"] == "complete" and len(versions.deleted) == 2
    assert all(not rows for rows in versions.objects.values())
    assert world.pushed[-1]["state"] == "withdrawn" and world.pushed[-1]["bundle"] is None


def test_a_cleanup_reset_without_its_lists_still_deletes_and_delivers(monkeypatch, site):
    """2026-10-23: a reset to pending without "deleted" raised KeyError on every tick."""
    _enqueue(site)
    world = World(monkeypatch, site, [_completed])
    assert world.advance() == "ready"
    job = _job(site)
    job.update(provider_cleanup={"state": "pending", "attempts": 0, "next_attempt_at_epoch": 0},
               revision=job["revision"] + 1)
    site.job.write_text(json.dumps(job))
    world.deleted.clear()
    assert world.advance() == "ready"
    job = _job(site)
    assert job["provider_cleanup"]["state"] == "complete" and job["provider_cleanup"]["remaining"] == []
    assert sorted(world.deleted) == ["gone1", "mesh1", "splat1", "upload1", "view1"]
    # The revision the reset raised reaches the WebApp.
    assert job["delivered_revision"] == job["revision"] and world.pushed[-1]["revision"] == job["revision"]


def test_a_task_confirmed_mid_run_is_picked_up_by_the_next_tick_not_failed(monkeypatch, site):
    """2026-10-24: the owner confirmed the task while the fresh capture's preview ran."""
    _enqueue(site)
    changed = ValueError("website_control_atlas-reservation_http_409:website_scene_sponsorship_binding_invalid")
    world = World(monkeypatch, site, [changed, _completed])
    assert world.advance() == "processing"
    job = _job(site)
    assert job["transient_failures"] == 1 and job["last_transient_reason"] == "website_scene_sponsorship_binding_invalid"
    world.clock = job["next_attempt_at_epoch"]
    assert world.advance() == "ready" and len(world.runs) == 2


def test_a_failed_job_resumes_once_an_operator_retries_it_after_a_fix(monkeypatch, site):
    """2026-10-23: a fixed planner check could not reach a job its old check had failed."""
    _enqueue(site)
    world = World(monkeypatch, site, [ValueError("atlas_scene_planner_transients_invalid"), _completed])
    assert world.advance() == "failed"
    failed = _job(site)
    assert failed["provider_cleanup"]["state"] == "complete" and world.deleted == []  # nothing made yet
    assert world.advance() == "failed" and len(world.runs) == 1  # a failure is final until retried
    world.clock += 60
    job = preview.retry_failed_job(site.job, now=world.now)
    assert (job["state"], job["reason"], job["revision"]) == ("processing", None, failed["revision"] + 1)
    assert job["retries"] == [{"reason": "atlas_scene_planner_transients_invalid", "at_epoch": world.clock}]
    assert job["provider_cleanup"]["state"] == "pending"
    assert world.advance() == "ready" and len(world.runs) == 2
    # The resumed run's assets are deleted at World Labs, not left behind.
    assert sorted(world.deleted) == ["gone1", "mesh1", "splat1", "upload1", "view1"]
    assert _job(site)["provider_cleanup"]["state"] == "complete"
    assert world.pushed[-1]["state"] == "ready" and world.pushed[-1]["revision"] > job["revision"]
    # Only a recoverable failure can be retried.
    with pytest.raises(ValueError, match="requires_recoverable_failure"):
        preview.retry_failed_job(site.job, now=world.now)


def test_a_job_stopped_for_background_review_resumes_but_one_needing_a_new_capture_does_not(monkeypatch, site):
    """2026-10-24: a cleanup stopped at needs_input could not resume once its cause was fixed."""
    _enqueue(site)
    stopped = {"status": "needs_input", "reason": "atlas_object_mask_view_coverage_incomplete"}
    world = World(monkeypatch, site, [stopped, _completed])
    assert world.advance() == "needs_input"
    assert world.advance() == "needs_input" and len(world.runs) == 1  # stopped until resumed
    job = preview.retry_failed_job(site.job, now=world.now)
    assert job["state"] == "processing" and job["retries"][0]["reason"] == "atlas_background_review_required"
    assert world.advance() == "ready" and len(world.runs) == 2
    site.job.write_text(json.dumps({**_job(site), "state": "needs_input", "reason": "atlas_scene_planner_abstained"}))
    with pytest.raises(ValueError, match="requires_recoverable_failure"):
        preview.retry_failed_job(site.job, now=world.now)


def test_a_retry_refuses_an_expired_or_withdrawn_job(monkeypatch, site):
    _enqueue(site)
    job = _job(site)
    site.job.write_text(json.dumps({**job, "state": "failed", "reason": "atlas_preview_expired"}))
    with pytest.raises(ValueError, match="requires_recoverable_failure"):
        preview.retry_failed_job(site.job)
    site.job.write_text(json.dumps({**job, "state": "failed", "reason": "atlas_scene_planner_transients_invalid"}))
    _withdraw(site)
    with pytest.raises(ValueError, match="website_capture_withdrawn"):
        preview.retry_failed_job(site.job)
    assert _job(site)["state"] == "failed"


def test_cleanup_waits_for_operations_it_never_collected_and_deletes_their_outputs(monkeypatch, site):
    _enqueue(site)

    def submitted_then_withdrawn(kwargs):
        stage = kwargs["output_root"] / "generated/atlas_pose"
        stage.mkdir(parents=True)
        (stage / "submission.json").write_text(json.dumps({"status": "submitted", "operation_id": "op1"}))
        _withdraw(site)
        kwargs["checkpoint"]("mesh")
    world = World(monkeypatch, site, [submitted_then_withdrawn])
    world.operations["op1"] = {"done": False}
    assert world.advance() == "withdrawn"
    cleanup = _job(site)["provider_cleanup"]
    assert cleanup["state"] == "pending" and cleanup["operations_running"] == ["op1"] and world.cancelled == ["op1"]
    world.operations["op1"] = {"done": True, "response": {"frames": [
        {"imageAsset": {"assetId": "out1"}, "depth": {"depthAsset": {"assetId": "out2"}}}]}}
    world.clock = cleanup["next_attempt_at_epoch"]
    world.advance()
    cleanup = _job(site)["provider_cleanup"]
    assert cleanup["state"] == "complete" and sorted(world.deleted) == ["out1", "out2"]
    assert cleanup["operations_running"] == []


def test_cleanup_collects_nested_automation_outputs(tmp_path):
    from blueprint_pipeline.website_atlas_preview import atlas_asset_ids
    stage = tmp_path / "automation/background/masked/0"
    stage.mkdir(parents=True)
    (stage / "result.json").write_text(json.dumps({"outputs": [{"asset_id": "asset_masked_output"}]}))
    assert atlas_asset_ids(tmp_path) == ["asset_masked_output"]


def test_cleanup_collects_a_resubmitted_pose_attempt_set_aside(tmp_path):
    from blueprint_pipeline.website_atlas_preview import atlas_asset_ids
    archived = tmp_path / "pose" / "atlas_pose_attempts" / "000"
    archived.mkdir(parents=True)
    (archived / "submission.json").write_text(json.dumps({"status": "failed", "created_assets": ["asset_frame_0"],
        "uploads": [{"frame_id": "f0", "asset_id": "asset_frame_0"}]}))
    current = tmp_path / "pose" / "atlas_pose"
    current.mkdir()
    (current / "submission.json").write_text(json.dumps({"created_assets": ["asset_frame_1"]}))
    assert atlas_asset_ids(tmp_path) == ["asset_frame_0", "asset_frame_1"]


def test_cleanup_covers_duplicate_purchases_that_reconciliation_found(tmp_path):
    stage = tmp_path / "generated" / "atlas_pose"
    stage.mkdir(parents=True)
    (stage / "submission.json").write_text(json.dumps({"status": "submitted", "operation_id": "op_kept",
        "reconciliations": [{"outcome": "bought", "operation_id": "op_kept",
                             "duplicate_operation_ids": ["op_duplicate"]}]}))
    assert preview._pending_operations(tmp_path) == ["op_kept", "op_duplicate"]
