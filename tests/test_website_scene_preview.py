"""The site's web-sized scene preview: built from retained scenes, published once per revision.

Hermetic: temporary capture roots, fakes for the rights read, the private artifact
store, the WebApp acceptance and the B2 version listing. No provider, bucket or
host is reached, and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_scene_preview.py

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import website_scene_preview as preview
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _splat(path: Path, count: int, *, seed: int = 0, sh_rest: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    data = SplatData(
        count=count, xyz=rng.normal(size=(count, 3)).astype(np.float32) * np.array([4, 1, 1], np.float32),
        opacity=rng.normal(size=count).astype(np.float32), f_dc=rng.normal(size=(count, 3)).astype(np.float32),
        scales=rng.normal(-3, 1, size=(count, 3)).astype(np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1)), properties=(),
        sh_rest=rng.normal(size=(count, 9)).astype(np.float32) if sh_rest else None)
    return write_standard_3dgs_ply(data, path)


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "storage/source-bucket/scenes/site-req1/captures/walkthrough-req1"
    _write(root / "raw/manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                        "capture_id": "walkthrough-req1", "consent_status": "granted"})
    return root


def _base_scene(capture: Path, *, seed: int = 0, count: int = 400) -> Path:
    level = capture / "pipeline/website_scene_preparation/atlas/level/abc"
    splat = _splat(level / "levelled_splat.ply", count, seed=seed, sh_rest=False)
    provider = _splat(capture / "pipeline/atlas/downloads/atlas-scene-1/splat/atlas_refined_splat.ply", count, seed=seed)
    return _write(capture / "pipeline/website_scene_preparation/base_scene.json", {
        "splat_path": str(splat), "splat_digest": _sha(splat), "up_axis": "Y", "provider": "world_labs_atlas",
        "levelling": {"provider_splat_path": str(provider), "provider_splat_digest": _sha(provider)}})


def _sealed(value: dict) -> dict:
    return {**value, "digest": canonical_digest(value, digest_field="digest")}


def _reconstruction(capture: Path, count: int = 300) -> Path:
    job_dir = capture / "pipeline/atlas"
    bundle_dir = job_dir / "downloads/atlas-scene-2"
    splat = _splat(bundle_dir / "splat/atlas_refined_splat.ply", count, seed=7)
    bundle = _sealed({"schema_version": "atlas_download_bundle.v1", "splat_available": True,
                      "assets": [{"path": "splat/atlas_refined_splat.ply", "sha256": _sha(splat),
                                  "bytes": splat.stat().st_size},
                                 {"path": "mesh/registered/mesh.glb", "sha256": "sha256:" + "1" * 64, "bytes": 3}]})
    _write(bundle_dir / "manifest.json", bundle)
    result = _sealed({"schema_version": "website_atlas_pipeline_result.v1",
                      "download": {"directory": str(bundle_dir), "manifest_digest": bundle["digest"]}})
    _write(job_dir / "result.json", result)
    _write(job_dir / "preview_job.json", {"state": "ready", "result_digest": result["digest"]})
    return splat


def _withdraw(capture: Path) -> None:
    site = capture.parents[1]
    _write(site / "website_withdrawal/tombstone.json", _sealed(
        {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": "site-req1", "request_id": "req1",
         "consent_revoked_at": "2026-11-07T00:00:00Z"}))


class World:
    """The rights read, private store and WebApp acceptance a pass talks to."""

    def __init__(self, *, rights: bool = True, on_archive=None, delivery_error: Exception | None = None):
        self.rights, self.on_archive, self.delivery_error = rights, on_archive, delivery_error
        self.archived, self.delivered, self.retracted, self.rights_reads = [], [], [], 0

    def check_rights(self, identity):
        self.rights_reads += 1
        if not self.rights:
            raise ValueError("website_scene_preview_rights_refused")

    def archive(self, path, *, identity):
        digest = _sha(path)
        self.archived.append(digest)
        if self.on_archive:
            self.on_archive()
        return {"uri": f"s3://bucket/x/website-scene-preview/sha256/{digest[7:]}/walkthrough-req1.ply",
                "digest": digest, "size_bytes": path.stat().st_size, "filename": "walkthrough-req1.ply"}

    def retract(self, uri):
        self.retracted.append(uri)

    def deliver(self, record, *, identity):
        if self.delivery_error:
            raise self.delivery_error
        self.delivered.append(record)

    def publish(self, capture, **options):
        return preview.publish_scene_preview(capture, check_rights=self.check_rights, archive=self.archive,
                                             retract=self.retract, deliver=self.deliver, now=lambda: 1_794_000_000,
                                             **options)


def test_the_preview_keeps_the_most_visible_gaussians_under_the_cap_and_is_deterministic(tmp_path):
    source = _splat(tmp_path / "source.ply", 1000)
    raw = read_standard_3dgs_ply(source)
    raw_bad = SplatData(count=raw.count, xyz=raw.xyz.copy(), opacity=raw.opacity.copy(), f_dc=raw.f_dc,
                        scales=raw.scales, quats=raw.quats, properties=(), sh_rest=raw.sh_rest)
    raw_bad.opacity[0] = np.nan  # a broken Gaussian never survives, however visible it claims to be
    raw_bad.scales[0] = 10
    write_standard_3dgs_ply(raw_bad, source)
    cap = 1024 + 100 * preview.RECORD_BYTES
    built = preview.build_preview(source, tmp_path / "a.ply", max_bytes=cap)
    again = preview.build_preview(source, tmp_path / "b.ply", max_bytes=cap)
    assert built == again and (tmp_path / "a.ply").read_bytes() == (tmp_path / "b.ply").read_bytes()
    assert built["gaussian_count"] == 100 and built["source_gaussian_count"] == 1000
    assert built["size_bytes"] <= cap and built["format"] == "ply_3dgs_sh0"
    assert built["coordinate_frame"] == "right_up_back_y_up"
    kept = read_standard_3dgs_ply(tmp_path / "a.ply")
    # Band-0 colour only, in the source's own coordinates: rows are copied, never moved.
    assert kept.sh_rest is None and kept.properties == (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3")
    written = read_standard_3dgs_ply(source)
    importance = written.opacity_sigmoid.astype(np.float64) * np.exp(
        written.scales.astype(np.float64).sum(axis=1) * 2 / 3)
    importance[0] = -1
    expected = np.sort(np.argsort(-importance, kind="stable")[:100])
    assert np.array_equal(kept.xyz, raw.xyz[expected])
    assert not np.isnan(kept.opacity).any()
    view = built["start_view"]
    assert len(view["position"]) == 3 and view["target"][0] > view["position"][0]  # along the long x side
    # A scene that already fits keeps every Gaussian.
    assert preview.build_preview(source, tmp_path / "c.ply")["gaussian_count"] == 999


def test_a_file_that_is_not_a_splat_is_refused(tmp_path):
    (tmp_path / "mesh.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError, match="website_scene_preview_source_not_3dgs_ply"):
        preview.build_preview(tmp_path / "mesh.ply", tmp_path / "out.ply")


def test_the_base_scene_is_published_once_per_scene_revision(capture):
    _base_scene(capture)
    world = World()
    first = world.publish(capture)
    assert first["status"] == "published"
    assert len(world.archived) == 1 and len(world.delivered) == 1
    record = world.delivered[0]
    assert set(record) == {"schema_version", "capture_id", "scene_revision", "state", "source_kind", "format",
                           "coordinate_frame", "digest", "size_bytes", "gaussian_count", "source_gaussian_count",
                           "start_view", "claim_ceiling", "updated_at_epoch", "preview_digest"}
    assert record["schema_version"] == "website_scene_preview.v1" and record["state"] == "ready"
    assert record["capture_id"] == "walkthrough-req1" and record["source_kind"] == "atlas_base_scene"
    assert record["preview_digest"] == cross_runtime_canonical_digest(record, digest_field="preview_digest")
    assert record["size_bytes"] <= preview.MAX_PREVIEW_BYTES
    directory = capture / "pipeline/website_scene_preparation"
    manifest = json.loads((directory / "scene_preview.json").read_text())
    assert manifest["scene_revision"] == record["scene_revision"] == first["scene_revision"]
    assert manifest["source"]["kind"] == "atlas_base_scene"
    # The levelled base scene's own splat, recorded relative to the capture.
    assert manifest["source"]["path"] == "pipeline/website_scene_preparation/atlas/level/abc/levelled_splat.ply"
    assert _sha(directory / "scene_preview.ply") == record["digest"] == manifest["archive"]["digest"]
    assert manifest["delivered"]["preview_digest"] == record["preview_digest"]

    # The same revision again builds, uploads, reads rights for and sends nothing.
    reads = world.rights_reads
    assert world.publish(capture) == {"status": "current", "scene_revision": first["scene_revision"]}
    assert len(world.archived) == 1 and len(world.delivered) == 1 and world.rights_reads == reads

    # A new scene is a new revision, published once.
    _base_scene(capture, seed=3)
    second = world.publish(capture)
    assert second["status"] == "published" and second["scene_revision"] != first["scene_revision"]
    assert len(world.archived) == 2 and len(world.delivered) == 2


def test_an_interrupted_delivery_resumes_without_rebuilding_or_reuploading(capture):
    _base_scene(capture)
    failing = World(delivery_error=ValueError("website_control_scene-preview_http_503:website_scene_preview_unavailable"))
    assert failing.publish(capture) == {"status": "failed",
                                        "reason": "website_control_scene-preview_http_503:website_scene_preview_unavailable"}
    assert failing.retracted == []
    manifest_path = capture / "pipeline/website_scene_preparation/scene_preview.json"
    built = json.loads(manifest_path.read_text())
    world = World()
    world.archived = []
    # The archive is already the built preview; only the delivery is repeated, with the same record.
    world.archive = lambda *_a, **_k: pytest.fail("re-uploaded an archived preview")
    assert world.publish(capture)["status"] == "published"
    assert world.delivered[0]["preview_digest"] == built["preview_digest"]
    assert json.loads(manifest_path.read_text())["built_at_epoch"] == built["built_at_epoch"]


def test_without_derived_scene_rights_nothing_is_built_archived_or_sent(capture, monkeypatch):
    _base_scene(capture)
    from blueprint_pipeline import website_task_context
    contexts = []

    def context(**kwargs):
        contexts.append(kwargs)
        return {"capture_rights": {"derived_scene_generation_allowed": False, "consent_revoked": False,
                                   "consent_status": "unknown"}}
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", context)
    world = World()
    result = preview.publish_scene_preview(capture, archive=world.archive, deliver=world.deliver,
                                           retract=world.retract)
    assert result == {"status": "refused", "reason": "website_scene_preview_rights_refused"}
    assert contexts == [{"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
                         "purpose": "scene_preparation"}]
    assert world.archived == [] and world.delivered == []
    assert not (capture / "pipeline/website_scene_preparation/scene_preview.ply").exists()
    assert not (capture / "pipeline/website_scene_preparation/scene_preview.json").exists()

    # A read that fails is a refusal too, never a guess.
    def unavailable(**_kwargs):
        raise ValueError("website_control_task-context_http_409:source_revoked")
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", unavailable)
    assert preview.publish_scene_preview(capture, archive=world.archive, deliver=world.deliver)["status"] == "refused"
    assert world.archived == []


def test_a_withdrawn_capture_publishes_nothing(capture):
    _base_scene(capture)
    _withdraw(capture)
    world = World()
    assert world.publish(capture) == {"status": "withdrawn"}
    assert world.rights_reads == 0 and world.archived == [] and world.delivered == []
    assert not (capture / "pipeline/website_scene_preparation/scene_preview.ply").exists()


def test_a_withdrawal_that_lands_during_the_upload_takes_the_copy_back(capture):
    _base_scene(capture)
    world = World(on_archive=lambda: _withdraw(capture))
    assert world.publish(capture) == {"status": "withdrawn"}
    assert len(world.retracted) == 1 and world.delivered == []
    manifest = json.loads((capture / "pipeline/website_scene_preparation/scene_preview.json").read_text())
    assert manifest["archive"]["retracted"] is True


def test_a_webapp_rights_refusal_takes_the_copy_back(capture):
    _base_scene(capture)
    world = World(delivery_error=ValueError("website_control_scene-preview_http_409:source_revoked"))
    result = world.publish(capture)
    assert result == {"status": "refused", "reason": "website_control_scene-preview_http_409:source_revoked"}
    assert len(world.retracted) == 1
    # The next pass reads rights again before anything is uploaded again.
    again = World(rights=False)
    assert again.publish(capture)["status"] == "refused" and again.archived == []


def test_a_finished_reconstruction_is_shown_until_the_base_scene_exists(capture):
    splat = _reconstruction(capture)
    assert preview.scene_source(capture)["kind"] == "atlas_reconstruction"
    assert preview.scene_source(capture)["digest"] == _sha(splat)
    world = World()
    assert world.publish(capture)["status"] == "published"
    assert world.delivered[0]["source_kind"] == "atlas_reconstruction"
    _base_scene(capture)
    assert world.publish(capture)["status"] == "published"
    assert world.delivered[-1]["source_kind"] == "atlas_base_scene"
    # An unfinished reconstruction shows nothing.
    job = json.loads((capture / "pipeline/atlas/preview_job.json").read_text())
    _write(capture / "pipeline/atlas/preview_job.json", {**job, "state": "processing"})
    assert preview._reconstruction_source(capture) is None


def test_a_splat_outside_the_capture_is_never_read(capture, tmp_path):
    outside = _splat(tmp_path / "elsewhere.ply", 10)
    _write(capture / "pipeline/website_scene_preparation/base_scene.json",
           {"splat_path": str(outside), "splat_digest": _sha(outside), "levelling": {}})
    world = World()
    assert world.publish(capture) == {"status": "no_viewable_scene"}
    assert not (capture / "pipeline/website_scene_preparation/scene_preview.lock").exists()


def test_the_tick_covers_every_capture_and_one_failure_stops_none(capture, monkeypatch):
    _base_scene(capture)
    other = capture.parent / ("supplement-" + "a" * 40)
    _reconstruction(other)
    _write(capture.parents[2] / "site-req2/captures/not-a-website-capture/pipeline/atlas/result.json", {})
    storage = capture.parents[4]
    assert preview.capture_roots(storage) == sorted([capture, other])
    seen = []

    def publish(root, **_options):
        seen.append(root.name)
        if root == other:
            raise RuntimeError("synthetic")
        return {"status": "current"}
    monkeypatch.setattr(preview, "publish_scene_preview", publish)
    assert preview.tick(storage) == {"current": 1, "unreadable": 1}
    assert sorted(seen) == sorted([capture.name, other.name])
    from blueprint_pipeline import website_atlas_preview
    monkeypatch.setattr(preview, "tick", lambda _root: (_ for _ in ()).throw(RuntimeError("boom")))
    assert website_atlas_preview.scene_preview_tick(storage) == {"tick_failed": 1}


def test_supplement_previews_are_named_for_their_job():
    assert preview.preview_filename("walkthrough-req1", "req1") == "walkthrough-req1.ply"
    child = "supplement-" + "b" * 40
    assert preview.preview_filename(child, "req1") == f"req1-{child}.ply"
    with pytest.raises(preview.ScenePreviewUnavailable):
        preview.preview_filename("walkthrough-req2", "req1")


def test_delivery_uses_the_signed_scene_preview_operation_and_checks_the_acceptance(monkeypatch):
    from blueprint_pipeline import website_task_context
    calls = []
    record = {"scene_revision": "sha256:" + "1" * 64, "preview_digest": "sha256:" + "2" * 64}
    acceptance = {"schema_version": "website_scene_preview_acceptance.v1", "accepted": True, **record}

    def request(**kwargs):
        calls.append(kwargs)
        return dict(acceptance)
    monkeypatch.setattr(website_task_context, "website_webapp_request", request)
    identity = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1"}
    preview._deliver(record, identity=identity)
    assert calls == [{"capture_id": "walkthrough-req1", "operation": "scene-preview",
                      "payload": {"request_id": "req1", "scene_id": "site-req1", "preview": record}}]
    acceptance["preview_digest"] = "sha256:" + "3" * 64
    with pytest.raises(ValueError, match="acceptance_invalid"):
        preview._deliver(record, identity=identity)


def test_the_signed_transport_admits_the_scene_preview_operation(monkeypatch):
    from blueprint_pipeline import website_task_context
    sent = []

    class Response:
        body = b'{"accepted": true}'
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.invalid")
    monkeypatch.setattr(website_task_context, "load_pipeline_sync_token", lambda: "fixture-sync-token")
    monkeypatch.setattr(website_task_context, "safe_request", lambda url, **_kw: sent.append(url) or Response())
    assert website_task_context.website_webapp_request(capture_id="walkthrough-req1", operation="scene-preview",
                                                       payload={}) == {"accepted": True}
    assert sent == ["https://webapp.invalid/api/internal/pipeline/creator-captures/walkthrough-req1/scene-preview"]


class _Bucket:
    def __init__(self, objects):
        self.objects = objects

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise type("NoSuchKey", (Exception,), {"response": {"Error": {"Code": "404"}}})()
        body = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": {"sha256": hashlib.sha256(body).hexdigest()}}

    def get_object(self, *, Bucket, Key):
        return {"Body": _Body(self.objects[Key])}


class _Body(io.BytesIO):
    def iter_chunks(self, size):
        while chunk := self.read(size):
            yield chunk


def test_the_webapp_reads_only_an_archived_unwithdrawn_preview(capture, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from blueprint_pipeline import website_atlas_preview_http as http
    from blueprint_pipeline.task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    _base_scene(capture)
    World().publish(capture)
    manifest = json.loads((capture / "pipeline/website_scene_preparation/scene_preview.json").read_text())
    digest = manifest["archive"]["digest"]
    body = (capture / "pipeline/website_scene_preparation/scene_preview.ply").read_bytes()
    bucket = _Bucket({f"{LARGE_ARTIFACT_KEY_PREFIX}/website-scene-preview/sha256/{digest[7:]}/walkthrough-req1.ply": body})

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    storage = capture.parents[4]
    monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(storage))
    monkeypatch.setattr(http, "open_scene_preview",
                        lambda **kw: preview.open_scene_preview(**kw, client=bucket, bucket="b"))
    client = TestClient(app)
    url = f"/api/live-pipeline/website-scene-previews/walkthrough-req1/scene.ply?request_id=req1&digest={digest}"
    signed = {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}
    response = client.get(url, headers=signed)
    assert response.status_code == 200 and response.content == body
    assert response.headers["x-blueprint-artifact-sha256"] == digest
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get(url, headers={**signed, "x-test-client": "robot-team"}).status_code == 403
    assert client.get(url.replace("req1&", "req2&"), headers=signed).status_code == 404
    assert client.get(url.replace(digest[7:], "0" * 64), headers=signed).status_code == 404
    _withdraw(capture)
    assert client.get(url, headers=signed).status_code == 404
