"""No provider calls: durable task execution and the public beta schema boundary."""
import json
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import website_atlas_pose as pose
from blueprint_pipeline import website_atlas_tasks as tasks
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def camera(x=0):
    return {"intrinsics": {"width": 1280, "height": 720, "fx": 900, "fy": 900, "cx": 640, "cy": 360},
            "extrinsics": {"position": [x, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}


def request():
    return {"contextFrames": [{"imageAsset": {"assetId": "asset_source"}, "camera": camera(),
                              "depth": {"depthAsset": {"assetId": "asset_source_depth"}}}],
            "targetCameras": [camera(.1)], "returnDepth": True}


class Fake:
    def __init__(self):
        self.posts = self.polls = self.reservations = 0
        self.admission = None
        self.uncertain = False
        self.pending = False

    def reserve(self, *, capture_id, operation, payload):
        assert operation == "preparation-spend"
        if self.admission:
            return {**self.admission, "status": "already_reserved"}
        self.reservations += 1
        self.admission = {**payload["spend"], "schema_version": "paid_lane_admission.v1", "status": "admitted",
                          "blockers": [], "external_disclosure_allowed": True, "expires_at_epoch": 1e12}
        return self.admission

    @staticmethod
    def content(asset_id):
        return (pose._EXR_MAGIC if "depth" in asset_id else b"\x89PNG\r\n\x1a\n") + asset_id.encode()

    def transport(self, method, url, headers, data):
        if url.startswith("https://read.example.test/"):
            assert "WLT-Api-Key" not in headers
            return 200, self.content(url.rsplit("/", 1)[-1])
        path = url.split("/api/v2")[1]
        if path == "/tasks:atlasGenerate":
            self.posts += 1
            assert headers["Idempotency-Key"].startswith("bp-atlas-task-")
            if self.uncertain:
                raise pose.AtlasTransportError("test uncertain POST")
            value = {"id": "op_task_1", "done": False}
        elif path == "/operations/op_task_1":
            self.polls += 1
            value = {"id": "op_task_1", "done": not self.pending,
                     "response": {"frames": [{"imageAsset": {"assetId": "asset_output"}, "camera": camera(.1),
                       "depth": {"depthAsset": {"assetId": "asset_depth"},
                                 "confidenceAsset": {"assetId": "asset_confidence"}}}], "promptUsed": "same kitchen"}}
        elif path.endswith(":createReadUrl"):
            value = {"readUrl": "https://read.example.test/" + path.split("/")[-1].split(":")[0]}
        elif path.startswith("/assets/"):
            asset_id = path.split("/")[-1]
            value = {"id": asset_id, "sizeBytes": len(self.content(asset_id)), "mimeType": "image/png"}
        else:
            raise AssertionError((method, path))
        return 200, json.dumps(value).encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    fake = Fake()
    monkeypatch.setenv(pose.ATLAS_API_BASE_URL_ENV, "https://api.atlas-beta.worldlabs.ai/api/v2")
    monkeypatch.setenv(pose.ATLAS_API_KEY_ENV, "test-key-not-secret")
    monkeypatch.setenv(pose.ATLAS_PROVIDER_RETENTION_ENV, "project_library_retention_approved")
    monkeypatch.setattr(control, "website_webapp_request", fake.reserve)
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("a" * 40, True, True))
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_a, **_kw: ([], "a" * 40))
    context = {"schema_version": "website_site_task_context.v1", "request_id": "r", "scene_id": "s", "capture_id": "c",
               "description": "Inspect the dishwasher", "purpose": "scene_preparation", "confirmed": False, "confirmed_at": None}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    body = request()
    review = {"schema_version": "website_atlas_task_price_review.v1", "task": "atlasGenerate",
              "request_digest": canonical_digest(body), "maximum_cost_usd": .05,
              "rate_card_reference": "fixture only", "reviewed_by": "test owner",
              "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    def run(**kw):
        return allocator.run_sponsored_website_atlas_task(task="atlasGenerate", request=body,
            output_root=tmp_path, source_video_digest="sha256:" + "b" * 64, task_context=context,
            price_review=review, transport=fake.transport, sleep=lambda _: None, max_polls=1, **kw)
    return fake, body, review, run


def test_retains_all_outputs_and_returns_replayed_result_without_another_purchase(case):
    fake, _, _, run = case
    result = run()
    assert result["status"] == "completed" and len(result["outputs"]) == 3
    assert result["metric_measurement_proven"] is False and result["generated_content"] is True
    assert result["billing"]["status"] == "unreported"
    assert run() == result
    assert fake.posts == fake.reservations == 1


def test_polling_resumes_without_new_reservation(case):
    fake, _, _, run = case
    fake.pending = True
    assert run()["status"] == "processing"
    fake.pending = False
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 1


def test_uncertain_send_never_causes_a_fresh_purchase(case):
    fake, _, _, run = case
    fake.uncertain = True
    with pytest.raises(pose.AtlasTransportError):
        run()
    with pytest.raises(ValueError, match="submission_requires_reconciliation"):
        run()
    assert fake.posts == fake.reservations == 1


def test_price_and_payload_refuse_before_reservation(case):
    fake, body, review, run = case
    review["request_digest"] = "wrong"
    with pytest.raises(ValueError, match="reviewed_price_required"):
        run()
    assert fake.reservations == fake.posts == 0
    body["contextFrames"][0]["imageAsset"] = {"url": "https://example.test/private.png"}
    with pytest.raises(ValueError, match="private_project_assets_required"):
        run()


def test_depth_requires_a_translation_and_target_grid_is_exact():
    body = request()
    body["targetCameras"] = [camera()]
    with pytest.raises(ValueError, match="depth_requires_translation"):
        tasks.validate_task_request("atlasGenerate", body)
    body["targetCameras"][0]["intrinsics"]["width"] = 720
    with pytest.raises(ValueError, match="target_grid_invalid"):
        tasks.validate_task_request("atlasGenerate", body)


def test_masked_views_must_be_paired_and_splat_mesh_is_a_separate_task():
    c = camera()
    body = {"contextFrames": [{"imageAsset": {"assetId": "asset_image"},
                              "maskAsset": {"assetId": "asset_mask"}, "camera": c}],
            "targetCameras": [camera(.1)]}
    with pytest.raises(ValueError, match="view_pairing_invalid"):
        tasks.validate_task_request("atlasMasked", body)
    body["targetCameras"] = [c]
    tasks.validate_task_request("atlasMasked", body)
    tasks.validate_task_request("splats2Mesh", {"splatAsset": {"assetId": "asset_splat"}, "textureMode": "image_texture"})
    with pytest.raises(ValueError, match="unavailable"):
        tasks.validate_task_request("createSplat", {})


def test_response_shape_cameras_depth_and_mesh_bytes_are_not_assumed(case):
    fake, _, _, run = case
    original = fake.transport
    def changed(method, url, headers, data):
        status, payload = original(method, url, headers, data)
        if "/operations/" in url:
            value = json.loads(payload)
            value["response"]["frames"][0]["camera"] = camera(.2)
            payload = json.dumps(value).encode()
        return status, payload
    fake.transport = changed
    with pytest.raises(ValueError, match="response_camera_mismatch"):
        run()
    assert fake.posts == 1


def test_camera_wire_rounding_is_allowed_but_grid_and_view_changes_are_not():
    requested = camera(.1)
    returned = camera(.10000000149)
    returned["intrinsics"]["fx"] += .00005
    assert tasks._camera_matches(returned, requested)
    returned["intrinsics"]["width"] = 1279
    assert not tasks._camera_matches(returned, requested)
    assert not tasks._camera_matches(camera(.11), requested)


def test_portrait_context_cannot_be_assumed_to_anchor_generated_depth():
    body = request()
    k = body["contextFrames"][0]["camera"]["intrinsics"]
    k.update(width=720, height=1280, cx=360, cy=640)
    with pytest.raises(ValueError, match="anchor_baseline_missing"):
        tasks.validate_task_request("atlasGenerate", body)
    body["targetCameras"].append(camera(.2))
    tasks.validate_task_request("atlasGenerate", body)


def test_chisel_rollout_and_depth_constraints_are_checked_before_spend():
    body = {"targetCameras": [camera()], "prompt": "Same kitchen", "returnDepth": True}
    with pytest.raises(ValueError, match="depth_requires_translated_targets"):
        tasks.validate_task_request("atlasChisel", body)
    body["targetCameras"] = [camera(), camera(.1)]
    tasks.validate_task_request("atlasChisel", body)
    body["sourceFrames"] = [{"imageAsset": {"assetId": "source"}, "camera": camera()}] * 63
    with pytest.raises(ValueError, match="rollout_limit_exceeded"):
        tasks.validate_task_request("atlasChisel", body)


def test_undeclared_uploads_and_wrong_asset_bytes_do_not_become_meshes(case, tmp_path):
    fake, body, _, run = case
    body["contextFrames"][0]["imageAsset"] = {"assetId": "upload:unbound"}
    with pytest.raises(ValueError, match="upload_references_mismatch"):
        run()
    assert fake.reservations == 0
    client = pose.AtlasClient(base_url="https://api.atlas-beta.worldlabs.ai/api/v2",
        api_key="test-key", transport=fake.transport)
    with pytest.raises(ValueError, match="mesh_not_glb"):
        tasks._collect(client, {"response": {"mesh": {"assetId": "asset_output"}}}, tmp_path)



def test_textured_mesh_may_exceed_the_image_output_ceiling_but_images_may_not(tmp_path, monkeypatch):
    import struct
    monkeypatch.setattr(pose, "MAX_OUTPUT_BYTES", 64)
    monkeypatch.setattr(pose, "MAX_MESH_OUTPUT_BYTES", 4096)
    blobs = {"asset_mesh": b"glTF" + struct.pack("<II", 2, 200) + bytes(188),
             "asset_image": b"\x89PNG\r\n\x1a\n" + bytes(100)}
    def transport(method, url, headers, data):
        if url.startswith("https://read.example.test/"):
            return 200, blobs[url.rsplit("/", 1)[-1]]
        path = url.split("/api/v2")[1]
        if path.endswith(":createReadUrl"):
            return 200, json.dumps({"readUrl": "https://read.example.test/" + path.split("/")[-1].split(":")[0]}).encode()
        asset_id = path.split("/")[-1]
        return 200, json.dumps({"id": asset_id, "sizeBytes": len(blobs[asset_id]), "mimeType": "model/gltf-binary"}).encode()
    client = pose.AtlasClient(base_url="https://api.atlas-beta.worldlabs.ai/api/v2", api_key="test-key", transport=transport)
    retained = tasks._collect(client, {"response": {"mesh": {"assetId": "asset_mesh"}}}, tmp_path)
    assert retained[0]["bytes"] == 200 and retained[0]["path"].endswith(".glb")
    with pytest.raises(ValueError, match="output_too_large"):
        tasks._collect(client, {"response": {"frames": [{"imageAsset": {"assetId": "asset_image"}}]}}, tmp_path)
