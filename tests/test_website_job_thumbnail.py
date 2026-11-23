"""A still of each job's simulation for its thumbnail: rendered once per delivered scene preview and sent to the WebApp.

Hermetic: temporary capture roots and fakes for the rights read, the WebApp's acceptance and the preview's own
archive. One test renders a small synthetic splat through the real CPU renderer. Nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_job_thumbnail.py

import base64
import hashlib
import json
import math

import numpy as np
import pytest

from blueprint_pipeline import website_job_thumbnail as thumbnail
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from tests.test_website_scene_preview import World, _base_scene, _withdraw, capture  # noqa: F401 - fixture

PNG = b"\x89PNG\r\n\x1a\n" + b"fixture-png"
IDENTITY = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1"}


def _manifest(root):
    return json.loads((root / "pipeline/website_scene_preparation/scene_preview.json").read_text())


def _receipt(root):
    return json.loads((root / "pipeline/website_scene_preparation/job_thumbnail.json").read_text())


class Side:
    """The rights read, the render and the WebApp a thumbnail pass talks to."""

    def __init__(self, *, rights=True, empty=False, errors=()):
        self.rights, self.empty, self.errors = rights, empty, list(errors)
        self.renders, self.delivered = [], []

    def check_rights(self, identity):
        if not self.rights:
            raise ValueError("website_scene_preview_rights_refused")

    def render(self, source, start_view):
        self.renders.append((source, dict(start_view)))
        return {"empty": True, "visible_gaussians": 0} if self.empty else {
            "empty": False, "png": PNG, "width": 480, "height": 300, "visible_gaussians": 5, "renderer": "r"}

    def deliver(self, payload, *, identity):
        if self.errors:
            raise self.errors.pop(0)
        self.delivered.append(payload)

    def publish(self, root):
        return thumbnail.publish_job_thumbnail(root, check_rights=self.check_rights, render=self.render,
                                               deliver=self.deliver, now=lambda: 1_794_000_000)


@pytest.fixture
def previewed(capture):  # noqa: F811 - the scene preview tests' capture
    _base_scene(capture)
    assert World().publish(capture)["status"] == "published"
    return capture


def test_the_camera_stands_at_the_start_view_and_looks_at_its_target():
    camera = thumbnail.camera_for({"position": [1.0, 1.5, 2.0], "target": [4.0, 1.5, 2.0]})
    rotation = camera["rotation"]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    # The camera looks down its -Z axis, along the start view, with +Y up.
    assert np.allclose(-rotation[:, 2], [1.0, 0.0, 0.0]) and rotation[1, 1] > 0.99
    assert camera["intrinsics"]["width"] == 480 and camera["intrinsics"]["height"] == 300
    with pytest.raises(ValueError, match="start_view_invalid"):
        thumbnail.camera_for({"position": [0, 0, 0], "target": [0, 0, 0]})


def _wall(path, *, behind=False):
    """A bright wall of 300 Gaussians three metres in front of (or behind) a camera at the origin looking down -Z."""
    xs, ys = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-2, 2, 15))
    count = xs.size
    xyz = np.stack([xs.ravel(), ys.ravel(), np.full(count, 3.0 if behind else -3.0)], axis=1).astype(np.float32)
    data = SplatData(count=count, xyz=xyz, opacity=np.full(count, 4.0, np.float32),
                     f_dc=np.full((count, 3), 1.0, np.float32), scales=np.full((count, 3), math.log(0.25), np.float32),
                     quats=np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1)), properties=())
    return write_standard_3dgs_ply(data, path)


def test_a_real_render_from_the_start_view_is_a_480_by_300_png_and_an_empty_one_is_said(tmp_path):
    view = {"position": [0.0, 0.0, 0.0], "target": [0.0, 0.0, -1.0]}
    made = thumbnail.render_png(_wall(tmp_path / "wall.ply"), view)
    assert made["empty"] is False and made["visible_gaussians"] > 0
    png = made["png"]
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) <= thumbnail.MAX_PNG_BYTES
    assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == (480, 300)
    # Nothing in front of the camera: said, never encoded.
    assert thumbnail.render_png(_wall(tmp_path / "behind.ply", behind=True), view) == {"empty": True, "visible_gaussians": 0}


def test_near_black_or_barely_covered_images_count_as_empty():
    assert thumbnail.looks_empty(np.zeros((30, 48, 3)), np.ones((30, 48)))
    assert thumbnail.looks_empty(np.full((30, 48, 3), 0.8), np.zeros((30, 48)))
    assert not thumbnail.looks_empty(np.full((30, 48, 3), 0.8), np.ones((30, 48)))


def test_the_payload_is_exactly_the_contract():
    payload = thumbnail.thumbnail_payload(IDENTITY, preview_digest="sha256:" + "a" * 64, png=PNG, width=480, height=300)
    assert set(payload) == {"request_id", "scene_id", "thumbnail"}
    assert payload["request_id"] == "req1" and payload["scene_id"] == "site-req1"
    image = payload["thumbnail"]
    assert set(image) == {"schema_version", "capture_id", "source", "camera", "scene_preview_digest", "width", "height",
                          "png_base64", "digest"}
    assert image == {"schema_version": "website_job_thumbnail.v1", "capture_id": "walkthrough-req1",
                     "source": "simulation_render", "camera": "scene_preview_start_view",
                     "scene_preview_digest": "sha256:" + "a" * 64, "width": 480, "height": 300,
                     "png_base64": base64.b64encode(PNG).decode(), "digest": "sha256:" + hashlib.sha256(PNG).hexdigest()}


def test_one_thumbnail_per_delivered_preview_and_a_new_preview_gets_a_new_one(previewed):
    side = Side()
    first = side.publish(previewed)
    assert first["status"] == "delivered" and len(side.renders) == 1 and len(side.delivered) == 1
    manifest = _manifest(previewed)
    sent = side.delivered[0]["thumbnail"]
    assert sent["scene_preview_digest"] == manifest["preview_digest"] == manifest["delivered"]["preview_digest"]
    assert side.renders[0][1] == manifest["preview"]["start_view"]
    receipt = _receipt(previewed)
    assert receipt["outcome"] == "delivered" and receipt["digest"] == sent["digest"] == first["digest"]
    assert receipt["render_inputs_digest"].startswith("sha256:")
    # The same preview again: nothing rendered or sent.
    assert side.publish(previewed) == {"status": "current", "outcome": "delivered"}
    assert len(side.renders) == 1 and len(side.delivered) == 1
    # A new scene, so a new preview, gets a new thumbnail.
    _base_scene(previewed, seed=3)
    assert World().publish(previewed)["status"] == "published"
    assert side.publish(previewed)["status"] == "delivered" and len(side.delivered) == 2
    assert side.delivered[1]["thumbnail"]["scene_preview_digest"] == _manifest(previewed)["preview_digest"]


def test_nothing_before_the_preview_itself_was_accepted(capture):  # noqa: F811
    side = Side()
    assert side.publish(capture) == {"status": "no_delivered_preview"}
    _base_scene(capture)
    assert World(delivery_error=RuntimeError("offline")).publish(capture)["status"] == "failed"
    assert side.publish(capture) == {"status": "no_delivered_preview"}
    assert not side.renders and not side.delivered


def test_a_transient_failure_retries_without_rendering_again_and_a_refusal_is_final(previewed):
    side = Side(errors=[ValueError("website_control_job-thumbnail_http_503:unclassified")])
    assert side.publish(previewed)["status"] == "failed"
    assert _receipt(previewed)["attempts"] == 1
    assert side.publish(previewed)["status"] == "delivered"
    assert len(side.renders) == 1 and len(side.delivered) == 1

    _base_scene(previewed, seed=4)
    World().publish(previewed)
    refused = Side(errors=[ValueError("website_control_job-thumbnail_http_409:website_job_thumbnail_stale")])
    assert refused.publish(previewed)["status"] == "refused"
    assert _receipt(previewed)["outcome"] == "refused"
    assert refused.publish(previewed) == {"status": "current", "outcome": "refused"}
    assert not refused.delivered


def test_repeated_failures_give_up_for_that_preview(previewed):
    side = Side(errors=[ConnectionError("reset")] * thumbnail.MAX_ATTEMPTS)
    for _ in range(thumbnail.MAX_ATTEMPTS):
        assert side.publish(previewed)["status"] == "failed"
    assert side.publish(previewed)["status"] == "gave_up"
    assert not side.delivered


def test_an_empty_render_is_recorded_and_never_sent(previewed):
    side = Side(empty=True)
    assert side.publish(previewed) == {"status": "skipped_empty"}
    assert side.publish(previewed) == {"status": "current", "outcome": "skipped_empty"}
    assert len(side.renders) == 1 and not side.delivered


def test_a_withdrawn_capture_or_doubtful_rights_render_and_send_nothing(previewed):
    doubtful = Side(rights=False)
    assert doubtful.publish(previewed)["status"] == "refused"
    assert not doubtful.renders and not (previewed / "pipeline/website_scene_preparation/job_thumbnail.json").exists()
    _withdraw(previewed)
    side = Side()
    assert side.publish(previewed) == {"status": "withdrawn"}
    assert not side.renders and not side.delivered


def test_delivery_uses_the_signed_job_thumbnail_operation_and_checks_the_acceptance(monkeypatch):
    from blueprint_pipeline import website_task_context
    payload = thumbnail.thumbnail_payload(IDENTITY, preview_digest="sha256:" + "a" * 64, png=PNG, width=480, height=300)
    calls, acceptance = [], {"schema_version": "website_job_thumbnail_acceptance.v1", "accepted": True,
                             "digest": payload["thumbnail"]["digest"]}
    monkeypatch.setattr(website_task_context, "website_webapp_request", lambda **kw: calls.append(kw) or dict(acceptance))
    thumbnail._deliver(payload, identity=IDENTITY)
    assert calls == [{"capture_id": "walkthrough-req1", "operation": "job-thumbnail", "payload": payload}]
    acceptance["digest"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="acceptance_invalid"):
        thumbnail._deliver(payload, identity=IDENTITY)


def test_the_signed_transport_admits_the_job_thumbnail_operation(monkeypatch):
    from blueprint_pipeline import website_task_context
    sent = []

    class Response:
        body = b'{"accepted": true}'
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.invalid")
    monkeypatch.setattr(website_task_context, "load_pipeline_sync_token", lambda: "fixture-sync-token")
    monkeypatch.setattr(website_task_context, "safe_request", lambda url, **_kw: sent.append(url) or Response())
    assert website_task_context.website_webapp_request(capture_id="walkthrough-req1", operation="job-thumbnail",
                                                       payload={}) == {"accepted": True}
    assert sent == ["https://webapp.invalid/api/internal/pipeline/creator-captures/walkthrough-req1/job-thumbnail"]


def test_the_tick_keeps_to_its_render_budget(previewed, tmp_path):
    side = Side()
    options = {"check_rights": side.check_rights, "render": side.render, "deliver": side.deliver}
    storage = previewed.parents[4]
    assert thumbnail.tick(storage, renders=0, **options) == {"deferred": 1, "stopped_render_budget": 1}
    assert thumbnail.tick(storage, **options) == {"delivered": 1}
    assert thumbnail.tick(storage, **options) == {"current": 1}


def test_withdrawal_deletion_cannot_take_the_lock_while_a_render_runs(previewed):
    """Codex review of #3705: deletion takes the preview's lock, so the render holds it from start to finish."""
    from blueprint_pipeline.website_scene_preview import ScenePreviewBusy, preview_lock
    side = Side()
    rendering = side.render

    def render_while_deletion_tries(source, start_view):
        with pytest.raises(ScenePreviewBusy):
            with preview_lock(previewed):
                pass
        return rendering(source, start_view)

    side.render = render_while_deletion_tries
    assert side.publish(previewed)["status"] == "delivered"


def test_a_withdrawal_just_before_the_lock_leaves_no_image_and_no_receipt(previewed, monkeypatch):
    """Withdrawn after the read-only checks and before the lock: rechecked under the lock, nothing is written."""
    calls = iter([False, True])
    monkeypatch.setattr(thumbnail, "capture_withdrawn", lambda root: next(calls, True))
    side = Side()
    assert side.publish(previewed) == {"status": "withdrawn"}
    directory = previewed / "pipeline/website_scene_preparation"
    assert not (directory / "job_thumbnail.png").exists() and not (directory / "job_thumbnail.json").exists()
    assert side.renders == [] and side.delivered == []
