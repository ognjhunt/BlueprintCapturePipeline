"""Admitted Atlas images2PosedRGBD operation: hermetic lifecycle tests.

Every World Labs call goes to a fake transport and every spend reservation to a
fake WebApp endpoint. Nothing here is a provider receipt, usage proof or price;
the reviewed price is a test value.
"""

from __future__ import annotations

import json
import io
import re
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import paid_resource_admission as admission_gate
from blueprint_pipeline import safe_outbound_http
from blueprint_pipeline import website_atlas_pose as atlas
from blueprint_pipeline import website_scene_geometry as geometry
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
)

BASE = "https://api.atlas-beta.worldlabs.ai/api/v2"
KEY = "test-atlas-key-not-a-secret"
EXR = b"\x76\x2f\x31\x01"
PNG = b"\x89PNG\r\n\x1a\n"
OPTIONS = {"refinePoses": True, "base64Outputs": False}
LANDSCAPE, PORTRAIT = (32, 18), (18, 32)


def _today():
    return datetime.now(timezone.utc).date()


def _review(**overrides):
    review = {"schema_version": "website_atlas_pose_price_review.v1",
              "endpoint": "/api/v2/tasks:images2PosedRGBD", "request_options": dict(OPTIONS),
              "target_resolutions": [[1280, 720], [720, 1280]], "maximum_cost_usd": 0.75,
              "max_input_images": 16, "rate_card_reference": "test fixture, not a real rate card",
              "reviewed_by": "test-owner", "reviewed_on": _today().isoformat()}
    review.update(overrides)
    return review


def _task_context():
    context = {"schema_version": "website_site_task_context.v1", "request_id": "request-atlas",
               "scene_id": "site-atlas", "capture_id": "website-capture", "purpose": "scene_preparation",
               "confirmed": False, "confirmed_at": None, "description": "Move the blue box to the tray"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return context


def _camera(width, height):
    return {"intrinsics": {"width": width, "height": height, "fx": 900.0, "fy": 900.0,
                           "cx": width / 2, "cy": height / 2},
            "extrinsics": {"coordinateSystem": "rub", "position": [0.0, 0.0, 0.0],
                           "quaternion": [0.0, 0.0, 0.0, 1.0]}}


class FakeWebApp:
    """The signed preparation-spend endpoint: one dispatch per binding."""

    def __init__(self, events, *, lifetime_seconds=3600):
        self.events, self.spend, self.reserved = events, [], {}
        self.lifetime_seconds = lifetime_seconds

    def __call__(self, *, capture_id, operation, payload):
        # Atlas reports no usage on its operations, so nothing may settle.
        assert operation == "preparation-spend"
        command = payload["spend"]
        self.events.append(("reserve", command["allocation_binding_digest"]))
        self.spend.append(command)
        key = command["allocation_binding_digest"]
        if key in self.reserved:
            return {**self.reserved[key], "status": "already_reserved"}
        value = {**command, "schema_version": "paid_lane_admission.v1", "status": "admitted",
                 "blockers": [], "external_disclosure_allowed": True,
                 "expires_at_epoch": time.time() + self.lifetime_seconds}
        self.reserved[key] = value
        return dict(value)


class FakeAtlas:
    """The documented Atlas beta task API (atlas-beta.worldlabs.ai/docs/api-reference).

    ``submit_outcomes`` answers successive task POSTs: an HTTP status, or
    "timeout" for a POST whose answer never arrived. ``mutations`` bends the
    finished operation or its files for negative tests.
    """

    def __init__(self, events, *, polls_until_done=1, submit_outcomes=(), put_failure_index=None,
                 operation_error=None, mutations=(), ready_after_checks=0, grant_headers=None,
                 upload_scheme="https", operation_id="op_atlas_1"):
        self.events, self.polls_until_done = events, polls_until_done
        self.submit_outcomes, self.put_failure_index = list(submit_outcomes), put_failure_index
        self.operation_error, self.mutations = operation_error, set(mutations)
        self.ready_after_checks, self.grant_headers = ready_after_checks, grant_headers
        self.upload_scheme, self.operation_id = upload_scheme, operation_id
        self.calls, self.created, self.uploaded, self.submissions = [], [], {}, []
        self.raw_submissions = []
        self.ready_checks, self.puts, self.polls = {}, 0, 0
        # Operations the project holds, oldest first: (id, request). ``foreign``
        # adds other callers' operations as {id: trace}; a None trace is unreadable.
        self.bought, self.foreign, self.traces = [], {}, []

    @staticmethod
    def _json(status, value):
        return status, json.dumps(value).encode()

    def _output(self, name):
        data = (EXR if name.startswith("asset_depth") and "depth_not_exr" not in self.mutations else PNG)
        return data + name.encode() * 4

    def __call__(self, method, url, headers, data):
        self.calls.append((method, url))
        self.events.append(("atlas", method, url))
        if not url.startswith(BASE + "/"):
            # Provider-issued URLs never see the key, in any casing.
            assert not any(key.lower() == "wlt-api-key" for key in headers), url
        if url.startswith("https://upload.example.test/"):
            asset_id = url.rsplit("/", 1)[1]
            assert headers["x-upload-token"] == asset_id and headers["Content-Type"] == "image/png"
            index, self.puts = self.puts, self.puts + 1
            if index == self.put_failure_index:
                return 503, b"upload backend unavailable"
            self.uploaded[asset_id] = data
            return 200, b""
        if url.startswith("https://read.example.test/"):
            assert method == "GET"
            body = self._output(url.rsplit("/", 1)[1].split(".")[0])
            return 200, (body[:-3] if "truncated" in self.mutations else body)
        assert headers["WLT-Api-Key"] == KEY
        path = url[len(BASE):]
        body = json.loads(data) if data else None
        if method == "POST" and path == "/assets":
            assert set(body) == {"asset"} and set(body["asset"]) == {"displayName"}
            asset_id = f"asset_{len(self.created):04x}"
            self.created.append((asset_id, body["asset"]["displayName"]))
            return self._json(201, {"id": asset_id, "displayName": body["asset"]["displayName"],
                                    "mimeType": "application/octet-stream", "state": "CREATED"})
        match = re.fullmatch(r"/assets/([A-Za-z0-9_]+):(createUploadUrl|completeUpload|createReadUrl)", path)
        if method == "POST" and match:
            asset_id, action = match.groups()
            if action == "createUploadUrl":
                assert body["contentType"] == "image/png" and body["contentLength"] > 0
                return self._json(200, {"uploadUrl": f"{self.upload_scheme}://upload.example.test/{asset_id}",
                                        "method": "PUT", "expiresAt": "2026-10-19T20:00:00Z",
                                        "headers": self.grant_headers or {"x-upload-token": asset_id}})
            if action == "completeUpload":
                self.ready_checks[asset_id] = 0
                return self._json(200, {"id": asset_id, "mimeType": "image/png",
                                        "sizeBytes": str(len(self.uploaded[asset_id])),
                                        "state": "UPLOADING" if self.ready_after_checks else "READY"})
            suffix = ".exr" if asset_id.startswith("asset_depth") else ".png"
            return self._json(200, {"readUrl": f"https://read.example.test/{asset_id}{suffix}",
                                    "expiresAt": "2026-10-19T20:00:00Z"})
        match = re.fullmatch(r"/assets/([A-Za-z0-9_]+)", path)
        if method == "GET" and match:
            asset_id = match.group(1)
            if asset_id in self.uploaded and "inputs_deleted" in self.mutations:
                return self._json(404, {"detail": "Asset not found"})
            if asset_id in self.uploaded:
                self.ready_checks[asset_id] += 1
                ready = self.ready_checks[asset_id] >= self.ready_after_checks
                return self._json(200, {"id": asset_id, "sizeBytes": str(len(self.uploaded[asset_id])),
                                        "state": "READY" if ready else "UPLOADING"})
            size = None if "size_unreported" in self.mutations else str(len(self._output(asset_id)))
            return self._json(200, {"id": asset_id, "state": "READY", "sizeBytes": size})
        if method == "POST" and path == "/tasks:images2PosedRGBD":
            self.submissions.append((body, headers.get("Idempotency-Key")))
            self.raw_submissions.append(data)
            outcome = self.submit_outcomes.pop(0) if self.submit_outcomes else 201
            if outcome in {201, "lost"}:
                suffix = f"_{len(self.bought)}" if self.bought else ""
                self.bought.append((self.operation_id + suffix, body))
            if outcome in {"timeout", "lost"}:
                # "lost": Atlas accepted the task, but its answer never arrived.
                raise atlas.AtlasTransportError("website_atlas_transport_failed:TimeoutError")
            if outcome == "bad_id":
                return self._json(201, {"id": "operations/op_atlas_1", "done": False})
            if outcome != 201:
                return self._json(outcome, {"code": "REFUSED_FOR_TEST"})
            operation_id = self.bought[-1][0]
            return self._json(201, {"id": operation_id, "done": False, "error": None, "response": None,
                                    "name": f"accounts/a/projects/p/operations/{operation_id}",
                                    "metadata": {"task": "images2PosedRGBD"}})
        if method == "GET" and path.startswith("/operations?"):
            query = dict(item.split("=", 1) for item in path.split("?", 1)[1].split("&"))
            assert {"createdAfter", "createdBefore", "pageSize"} <= set(query)
            # Newest first, as documented.
            listed = [*self.foreign, *(operation_id for operation_id, _ in reversed(self.bought))]
            return self._json(200, {"operations": [{"id": operation_id, "ownerId": "acct_test", "metadata": None,
                                                    "done": False, "error": None} for operation_id in listed],
                                    "nextPageToken": None})
        match = re.fullmatch(r"/operations/([A-Za-z0-9_]+):trace", path)
        if method == "GET" and match:
            operation_id = match.group(1)
            self.traces.append(operation_id)
            if operation_id in self.foreign:
                trace = self.foreign[operation_id]
                return self._json(404, {"detail": "not found"}) if trace is None else self._json(200, trace)
            request = dict(self.bought)[operation_id]
            # Projected through the current public schema: defaults the caller never sent.
            return self._json(200, {"operationId": operation_id, "taskName": "images2PosedRGBD", "taskVersion": "v1",
                                    "request": {**request, "webhookUrl": None, "canonicalizeToFirstFrame": True,
                                                "purgeContentOnOperationCompletion": None},
                                    "response": None, "error": None})
        match = re.fullmatch(r"/operations/([A-Za-z0-9_]+)", path)
        if method == "GET" and match and (match.group(1) == self.operation_id
                                          or match.group(1) in dict(self.bought)):
            if "gone" in self.mutations:
                return self._json(404, {"code": "NOT_FOUND"})
            self.polls += 1
            operation = {"id": "op_other" if "operation_id" in self.mutations else match.group(1),
                         "done": self.polls >= self.polls_until_done, "error": None, "response": None,
                         "metadata": {"task": "images2PosedRGBD"}}
            if operation["done"] and self.operation_error:
                operation["error"] = self.operation_error
            elif operation["done"]:
                request = self.submissions[-1][0]
                count = len(request["frames"]) - ("frame_count" in self.mutations)
                width, height = (640, 480) if "grid" in self.mutations else request["targetResolution"]
                frames = [{"imageAsset": {"assetId": f"asset_rgb_{i}", "url": f"https://cdn.example.test/rgb_{i}.png"},
                           "maskAsset": {"assetId": f"asset_mask_{i}", "url": f"https://cdn.example.test/m_{i}.png"},
                           "camera": _camera(width, height), "centroidWorld": [0.0, 0.0, 1.0],
                           "depth": {"depthAsset": {"assetId": f"asset_depth_{i}",
                                                    "url": f"https://cdn.example.test/d_{i}.exr"},
                                     "confidenceAsset": {"assetId": f"asset_conf_{i}",
                                                         "url": f"https://cdn.example.test/c_{i}.png"}}}
                          for i in range(count)]
                if "bad_mask" in self.mutations:
                    frames[-1]["maskAsset"] = {"assetId": "../elsewhere"}
                operation["response"] = {"poseRefinementAccepted": True, "frames": frames}
            return self._json(200, operation)
        raise AssertionError(f"undocumented Atlas call: {method} {path}")


@pytest.fixture
def atlas_case(tmp_path, monkeypatch, request):
    sizes = getattr(request, "param", [LANDSCAPE] * 3)
    source = tmp_path / "raw.mov"
    source.write_bytes(b"original video")

    def decode(self, **kwargs):
        root = kwargs["output_root"]
        root.mkdir(parents=True)
        rows = []
        for i, size in enumerate(sizes):
            image = root / f"{i}.png"
            Image.new("RGB", size, (i * 40, 10, 0)).save(image)
            rows.append({"candidate_relative_path": image.name, "frame_id": f"decoded-{i:09d}",
                         "frame_digest": geometry._sha256_file(image), "t_video_sec": float(i)})
        (root / "candidate.json").write_text(json.dumps({"heldout_pixels_included": False, "frames": rows}))
        (root / "index.json").write_text(json.dumps({"stream_metadata": {"display_rotation_degrees": 0}}))
        return {"asset_references": {"candidate_dataset_manifest": {"relative_path": "candidate.json"},
                                     "decoded_observation_index": {"relative_path": "index.json"}}}

    monkeypatch.setattr(geometry.LocalDecodedObservationAdapter, "execute", decode)
    kwargs = dict(source_video=source, output_root=tmp_path / "out", capture_id="website-capture")
    pending = geometry.run_website_scene_geometry(**kwargs)
    assert pending["status"] == "pending" and len(pending["frames"]) == len(sizes)
    monkeypatch.setenv(atlas.ATLAS_API_BASE_URL_ENV, BASE)
    monkeypatch.setenv(atlas.ATLAS_API_KEY_ENV, KEY)
    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, json.dumps(_review()))
    monkeypatch.setenv(atlas.ATLAS_PROVIDER_RETENTION_ENV, "account_does_not_retain")
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("1" * 40, True, True))
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_a, **_k: ([], "1" * 40))
    events: list = []
    webapp = FakeWebApp(events)
    monkeypatch.setattr(control, "website_webapp_request", webapp)
    context = _task_context()

    def run(transport, capture_id="website-capture", **options):
        options.setdefault("sleep", lambda _seconds: None)
        return allocator.run_sponsored_website_atlas_pose(
            output_root=kwargs["output_root"], capture_id=capture_id, task_context=context,
            transport=transport, **options)

    return SimpleNamespace(kwargs=kwargs, root=kwargs["output_root"], pending=pending, events=events,
                           webapp=webapp, context=context, run=run)


def _state(case):
    return json.loads((case.root / "atlas_pose" / "submission.json").read_text())


def test_portable_export_indexes_preserved_views_separately_from_native_outputs(atlas_case, tmp_path):
    """Synthetic metadata fixture: portable links and provenance, no model/geometry claim."""
    import zipfile
    from blueprint_pipeline.atlas_result_bundle import export_pose_bundle

    atlas_case.run(FakeAtlas(atlas_case.events))
    native = tmp_path / 'fake-generated'
    native.mkdir()
    image = native / 'native.png'
    Image.new('RGB', (4, 4), 'magenta').save(image)
    source_video = _result(atlas_case)['binding']['source_video_digest']
    value = {'status': 'completed', 'binding': {'source_video_digest': source_video},
             'operation': {'id': 'fake-generation'}, 'outputs': [{'role': 'frames-0-imageAsset',
                'path': image.name, 'sha256': geometry._sha256_file(image), 'bytes': image.stat().st_size}]}
    value['digest'] = canonical_digest(value, digest_field='digest')
    (native / 'result.json').write_text(json.dumps(value))
    (native / 'atlas_pose').mkdir()
    (native / 'atlas_pose/submission.json').write_text(json.dumps({'status': 'completed', 'result_digest': value['digest']}))
    context, fixed, known, allowed = [tmp_path / name for name in ('context.png', 'fixed.png', 'known.png', 'allowed.png')]
    Image.new('RGB', (4, 4), 'green').save(context)
    Image.new('RGB', (4, 4), 'green').save(fixed)
    Image.new('L', (4, 4), 255).save(known)
    Image.new('L', (4, 4), 0).save(allowed)
    record = {'schema_version': 'atlas_scene_preservation.v1', 'frame_id': 'completion-a',
              'native_result_digest': value['digest'], 'native_frame_index': 0, 'known_pixels_changed': 0,
              'source_geometry_modified': False, 'native_outputs_modified': False,
              'context_image_digest': geometry._sha256_file(context),
              'assets': [{'role': role, 'sha256': geometry._sha256_file(path)} for role, path in
                         (('composite_rgb', fixed), ('known_pixels', known), ('generated_geometry_mask', allowed))]}
    receipt = tmp_path / 'preservation.json'
    record['digest'] = canonical_digest(record, digest_field='digest')
    receipt.write_text(json.dumps(record))
    args = dict(pose_root=atlas_case.root, output_root=tmp_path / 'export', generated_roots=[native],
                analysis_paths=[context, fixed, known, allowed, receipt])
    exported = export_pose_bundle(**args)
    with zipfile.ZipFile(exported['zip_path']) as bundle:
        manifest = json.loads(bundle.read('manifest.json'))
        view = manifest['scene_views'][0]
        assert bundle.read(view['image_path']) == fixed.read_bytes()
        assert bundle.read(view['known_mask_path']) == known.read_bytes()
        assert bundle.read(view['generated_geometry_mask_path']) == allowed.read_bytes()
        assert manifest['approved_context_pixels_preserved'] is True
        assert bundle.read('generated/0/native.png') == image.read_bytes()
    duplicates = [Path(exported['directory']) / row['path'] for row in manifest['assets']
                  if row['sha256'] == geometry._sha256_file(fixed)]
    assert len(duplicates) >= 2
    assert len({path.stat().st_ino for path in duplicates}) == 1
    record['native_result_digest'] = 'not-this-operation'
    record['digest'] = canonical_digest(record, digest_field='digest')
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='preservation_receipt_invalid'):
        export_pose_bundle(**args)


def test_portable_export_is_verified_deterministic_and_excludes_unlisted_files(atlas_case, tmp_path):
    import zipfile
    from blueprint_pipeline.atlas_result_bundle import export_pose_bundle
    atlas_case.run(FakeAtlas(atlas_case.events))
    first = export_pose_bundle(pose_root=atlas_case.root, output_root=tmp_path / "exports")
    Path(first["directory"], "unlisted.txt").write_text("not part of the admitted run")
    second = export_pose_bundle(pose_root=atlas_case.root, output_root=tmp_path / "exports")
    assert first["zip_sha256"] == second["zip_sha256"]
    with zipfile.ZipFile(second["zip_path"]) as archive:
        assert "unlisted.txt" not in archive.namelist()
        # Source frames stay with the job (D5): the views are the provider's own posed images.
        assert not [name for name in archive.namelist() if name.startswith("inputs/")]
        assert any(name.startswith("views/") for name in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["provider_input_count"] == 3
        assert manifest["simulation_qualified"] is False
        assert manifest["mesh_available"] is False and manifest["splat_available"] is False
    result = _result(atlas_case)
    output = result["operation_receipt"]["input_usage"]["frames"][0]["outputs"][0]
    (atlas_case.root / output["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="retained_output_changed"):
        export_pose_bundle(pose_root=atlas_case.root, output_root=tmp_path / "exports")


def test_portable_export_carries_bound_splat_and_mesh_and_refuses_changed_ones(atlas_case, tmp_path):
    import zipfile
    from blueprint_pipeline.atlas_result_bundle import export_pose_bundle
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    atlas_case.run(FakeAtlas(atlas_case.events))
    result = _result(atlas_case)
    splat_root, mesh_root = tmp_path / "splat", tmp_path / "mesh"
    splat_root.mkdir()
    spz = splat_root / "atlas_depth_splat.spz"
    spz.write_bytes(b"spz bytes")
    splat = {"pose_result_digest": result["digest"], "splat_kind": "depth_fused_draft",
             "outputs": [{"path": spz.name, "sha256": _sha256_file(spz)}]}
    splat["digest"] = canonical_digest(splat, digest_field="digest")
    (splat_root / "atlas_depth_splat_manifest.json").write_text(json.dumps(splat))
    (mesh_root / "outputs").mkdir(parents=True)
    glb = mesh_root / "outputs" / "mesh.glb"
    glb.write_bytes(b"glTF mesh bytes")
    mesh = {"status": "completed", "binding": {"source_video_digest": result["binding"]["source_video_digest"]},
            "outputs": [{"path": "outputs/mesh.glb", "sha256": _sha256_file(glb), "bytes": glb.stat().st_size}]}
    mesh["digest"] = canonical_digest(mesh, digest_field="digest")
    (mesh_root / "result.json").write_text(json.dumps(mesh))
    (mesh_root / "atlas_pose").mkdir()
    (mesh_root / "atlas_pose" / "submission.json").write_text(
        json.dumps({"status": "completed", "result_digest": mesh["digest"]}))
    with pytest.raises(ValueError, match="mesh_unregistered"):
        export_pose_bundle(pose_root=atlas_case.root, output_root=tmp_path / "exports", mesh_root=mesh_root)
    (mesh_root / "registered").mkdir()
    supported = mesh_root / "registered" / "atlas_mesh_supported.glb"
    supported.write_bytes(b"glTF supported mesh")
    record = {"source_glb_sha256": _sha256_file(glb), "glb_frame": "atlas_world_rotated_180_x",
              "outputs": [{"path": supported.name, "sha256": _sha256_file(supported)}]}
    record["digest"] = canonical_digest(record, digest_field="digest")
    (mesh_root / "registered" / "atlas_mesh_registration.json").write_text(json.dumps(record))
    bundle = export_pose_bundle(pose_root=atlas_case.root, output_root=tmp_path / "exports",
                                splat_root=splat_root, mesh_root=mesh_root)
    with zipfile.ZipFile(bundle["zip_path"]) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert {"splat/atlas_depth_splat.spz", "splat/atlas_depth_splat_manifest.json", "mesh/mesh.glb",
            "mesh/registered/atlas_mesh_supported.glb"} <= names
    assert manifest["mesh_frame"] == "atlas_world_rotated_180_x"
    assert manifest["splat_available"] is True and manifest["mesh_available"] is True
    assert manifest["splat_kind"] == "depth_fused_draft" and manifest["simulation_qualified"] is False
    spz.write_bytes(b"changed")
    with pytest.raises(ValueError, match="splat_output_changed"):
        export_pose_bundle(pose_root=atlas_case.root, output_root=tmp_path / "exports", splat_root=splat_root)


def _result(case):
    return json.loads((case.root / "atlas_posed_rgbd_result.json").read_text())


def _later(seconds):
    return lambda: time.time() + seconds


def test_admitted_operation_uploads_submits_polls_and_retains_bound_outputs(atlas_case):
    fake = FakeAtlas(atlas_case.events, polls_until_done=2)
    first = atlas_case.run(fake, max_polls=1)
    assert first["status"] == "processing"
    assert first["blockers"] == ["website_atlas_pose_operation_pending"]
    # The signed reservation precedes every provider call, under Atlas's own
    # identity rather than Marble's.
    assert atlas_case.events[0][0] == "reserve"
    assert [command["provider"] for command in atlas_case.webapp.spend] == ["world_labs_atlas"]
    state = _state(atlas_case)
    assert state["status"] == "submitted" and state["operation_id"] == "op_atlas_1"

    second = atlas_case.run(fake, max_polls=3)
    assert second["status"] == "completed" and second["submitted_frame_count"] == 3
    # Polling and retention are reads under the existing operation: no new
    # reservation and no second billable submission.
    assert len(atlas_case.webapp.spend) == 1 and len(fake.submissions) == 1
    body, idempotency_key = fake.submissions[0]
    asset_ids = [row["asset_id"] for row in state["uploads"]]
    assert body == {"frames": [{"imageAsset": {"assetId": asset_id}} for asset_id in asset_ids],
                    "targetResolution": [1280, 720], **OPTIONS}
    assert idempotency_key == state["idempotency_key"] and len(idempotency_key) <= 128
    assert state["created_assets"] == asset_ids
    pending = json.loads((atlas_case.root / "atlas_pose_inputs.json").read_text())
    assert [fake.uploaded[asset_id] for asset_id in asset_ids] == [
        Path(row["image_path"]).read_bytes() for row in pending["frames"]]
    # Uploaded names identify the attempt, never the site or the capture.
    assert all(re.fullmatch(r"bp-[0-9a-f]{12}-\d{4}", name) for _, name in fake.created)

    result = _result(atlas_case)
    assert result["digest"] == canonical_digest(result, digest_field="digest")
    assert result["binding"] == pending["binding"] and result["endpoint"] == "/api/v2/tasks:images2PosedRGBD"
    assert result["claim_ceiling"] == "development_only"
    assert result["metric_measurement_proven"] is False and result["physical_evidence"] is False
    # Signed output URLs are short-lived credentials; asset IDs are the record.
    assert "cdn.example.test" not in (atlas_case.root / "atlas_posed_rgbd_result.json").read_text()
    receipt = result["operation_receipt"]
    assert receipt["operation_id"] == "op_atlas_1" and receipt["request"] == body
    assert receipt["billing"]["status"] == "unreported"
    assert receipt["binding"]["provider_retention"] == "account_does_not_retain"
    usage = receipt["input_usage"]
    assert usage["submitted_frame_count"] == 3
    assert [(row["frame_id"], row["image_digest"], row["asset_id"]) for row in usage["frames"]] == [
        (row["frame_id"], row["image_digest"], asset_id) for row, asset_id in zip(pending["frames"], asset_ids)]
    for frame in usage["frames"]:
        assert {row["role"] for row in frame["outputs"]} == {"image", "depth", "confidence", "mask"}
        for row in frame["outputs"]:
            assert "sha256:" + sha256((atlas_case.root / row["path"]).read_bytes()).hexdigest() == row["sha256"]
        depth = next(row for row in frame["outputs"] if row["role"] == "depth")
        assert (atlas_case.root / depth["path"]).read_bytes().startswith(EXR)

    # The CPU reader binds the producer's receipt: actual input usage is now
    # known, and every evidence gate stays where it was.
    context = geometry.run_website_scene_geometry(**atlas_case.kwargs)
    assert context["actual_provider_input_count"] == 3 and context["provider_mutation_performed"] is True
    assert context["posed_rgbd_available"] is True and context["native_geometry_available"] is False
    assert context["blockers"] == ["website_atlas_final_scene_assets_required",
                                   "website_atlas_native_geometry_binding_required"]
    assert context["claim_ceiling"] == "development_only" and context["metric_measurement_proven"] is False

    calls = len(fake.calls)
    again = atlas_case.run(fake)
    assert again["status"] == "completed" and len(fake.calls) == calls


def test_production_key_file_is_explicit_protected_and_never_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv(atlas.ATLAS_API_BASE_URL_ENV, BASE)
    monkeypatch.delenv(atlas.ATLAS_API_KEY_ENV, raising=False)
    monkeypatch.delenv("WLT_API_KEY", raising=False)
    key = tmp_path / "atlas-key"
    key.write_text("test-file-key")
    key.chmod(0o600)
    monkeypatch.setenv("WORLDLABS_ATLAS_API_KEY_FILE", str(key))
    assert atlas.atlas_api_config() == (BASE, "test-file-key")
    key.chmod(0o644)
    with pytest.raises(ValueError, match="key_file_unsafe"):
        atlas.atlas_api_config()


@pytest.mark.parametrize("missing,code", [
    ("price", "website_atlas_reviewed_price_required"),
    ("retention", "website_atlas_provider_retention_decision_required"),
    ("key", "website_atlas_api_key_missing"),
    ("base", "website_atlas_api_base_url_missing"),
])
def test_unconfigured_atlas_refuses_before_any_reservation_or_upload(atlas_case, monkeypatch, missing, code):
    monkeypatch.delenv({"price": atlas.ATLAS_PRICE_REVIEW_ENV, "key": atlas.ATLAS_API_KEY_ENV,
                        "base": atlas.ATLAS_API_BASE_URL_ENV,
                        "retention": atlas.ATLAS_PROVIDER_RETENTION_ENV}[missing])
    fake = FakeAtlas(atlas_case.events)
    with pytest.raises(ValueError, match=code):
        atlas_case.run(fake)
    assert atlas_case.events == []
    assert not (atlas_case.root / "atlas_pose" / "submission.json").exists()


def test_an_unknown_retention_decision_is_refused(atlas_case, monkeypatch):
    monkeypatch.setenv(atlas.ATLAS_PROVIDER_RETENTION_ENV, "keep_forever")
    with pytest.raises(ValueError, match="website_atlas_provider_retention_decision_required"):
        atlas_case.run(FakeAtlas(atlas_case.events))
    assert atlas_case.events == []


def test_unadmitted_release_refuses_before_any_reservation_or_upload(atlas_case, monkeypatch):
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_a, **_k: (["dirty_checkout"], "1" * 40))
    with pytest.raises(ValueError, match="website_atlas_pose_release_not_admitted:dirty_checkout"):
        atlas_case.run(FakeAtlas(atlas_case.events))
    assert atlas_case.events == []


def test_another_captures_task_context_is_refused(atlas_case):
    # The sponsorship charged and the frames disclosed must be one capture's.
    with pytest.raises(ValueError, match="website_atlas_pose_task_capture_mismatch"):
        atlas_case.run(FakeAtlas(atlas_case.events), capture_id="another-capture")
    assert atlas_case.events == []


@pytest.mark.parametrize("override,code", [
    ({"schema_version": "website_atlas_pose_price_review.v0"}, "website_atlas_reviewed_price_invalid:schema"),
    ({"endpoint": "/api/v2/tasks:atlasGenerate"}, "website_atlas_reviewed_price_invalid:endpoint"),
    ({"request_options": {**OPTIONS, "refinePoses": False}}, "website_atlas_reviewed_price_options_mismatch"),
    ({"request_options": {**OPTIONS, "refinePoses": 1}}, "website_atlas_reviewed_price_invalid:request_options"),
    ({"target_resolutions": [[720, 1280]]}, "website_atlas_reviewed_price_options_mismatch"),
    ({"target_resolutions": [[1280.0, 720]]}, "website_atlas_reviewed_price_invalid:target_resolutions"),
    ({"target_resolutions": [[1280, 720], [1920, 1080]]}, "website_atlas_reviewed_price_invalid:target_res"),
    ({"maximum_cost_usd": 0}, "website_atlas_reviewed_price_invalid:maximum_cost_usd"),
    ({"maximum_cost_usd": True}, "website_atlas_reviewed_price_invalid:maximum_cost_usd"),
    ({"maximum_cost_usd": float("inf")}, "website_atlas_reviewed_price_invalid:maximum_cost_usd"),
    ({"maximum_cost_usd": 10 ** 400}, "website_atlas_reviewed_price_invalid:maximum_cost_usd"),
    ({"maximum_cost_usd": 1000.01}, "website_atlas_reviewed_price_invalid:maximum_cost_usd"),
    ({"max_input_images": 2}, "website_atlas_reviewed_price_input_count_exceeded"),
    ({"max_input_images": 1}, "website_atlas_reviewed_price_invalid:max_input_images"),
    ({"rate_card_reference": ""}, "website_atlas_reviewed_price_invalid:rate_card_reference"),
    ({"reviewed_by": "  "}, "website_atlas_reviewed_price_invalid:reviewed_by"),
    ({"reviewed_on": "yesterday"}, "website_atlas_reviewed_price_invalid:reviewed_on"),
    ({"reviewed_on": 20261019}, "website_atlas_reviewed_price_invalid:reviewed_on"),
    ({"reviewed_on": "2026-W42"}, "website_atlas_reviewed_price_invalid:reviewed_on"),
    ({"reviewed_on": timedelta(days=1)}, "website_atlas_reviewed_price_invalid:reviewed_on"),
    ({"reviewed_on": timedelta(days=-31)}, "website_atlas_reviewed_price_invalid:reviewed_on"),
])
def test_price_review_must_be_complete_current_and_cover_the_request(monkeypatch, override, code):
    # Relative dates resolve against one date that the check is also given:
    # a run that crosses UTC midnight must not turn "tomorrow" into today.
    today = _today()
    override = {key: (today + value).isoformat() if isinstance(value, timedelta) else value
                for key, value in override.items()}
    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV,
                       json.dumps(_review(**{"reviewed_on": today.isoformat(), **override})))
    with pytest.raises(ValueError, match=code):
        atlas.atlas_pose_spend_line(planned_input_count=3, target_resolution=[1280, 720], today=today)


def test_reviewed_price_is_the_reservation_ceiling_however_it_was_typed(monkeypatch):
    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, "not json")
    with pytest.raises(ValueError, match="website_atlas_reviewed_price_invalid:json"):
        atlas.atlas_pose_spend_line(planned_input_count=3, target_resolution=[1280, 720])
    lines = []
    for price in (1, 1.0):
        monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, json.dumps(_review(maximum_cost_usd=price)))
        lines.append(atlas.atlas_pose_spend_line(planned_input_count=3, target_resolution=[1280, 720]))
    # Re-exporting the same review must not change the binding mid-attempt.
    assert lines[0] == lines[1] and lines[0]["maximum_cost_usd"] == 1.0


@pytest.mark.parametrize("url", [
    "http://api.atlas-beta.worldlabs.ai/api/v2", "https://api.atlas-beta.worldlabs.ai",
    "https://api.atlas-beta.worldlabs.ai/api/v1", "https://worldlabs.ai.example.test/api/v2",
    "https://api.atlas-beta.worldlabs.ai/api/v2?debug=1", "https://user:pw@api.worldlabs.ai/api/v2",
    "https://api.worldlabs.ai:8443/api/v2",
])
def test_api_root_must_be_an_explicit_world_labs_api_v2_https_root(monkeypatch, url):
    monkeypatch.setenv(atlas.ATLAS_API_BASE_URL_ENV, url)
    monkeypatch.setenv(atlas.ATLAS_API_KEY_ENV, KEY)
    with pytest.raises(ValueError, match="website_atlas_api_base_url_invalid"):
        atlas.atlas_api_config()
    monkeypatch.setenv(atlas.ATLAS_API_BASE_URL_ENV, BASE + "/")
    assert atlas.atlas_api_config() == (BASE, KEY)
    # http.client would refuse this key and quote it in its error.
    monkeypatch.setenv(atlas.ATLAS_API_KEY_ENV, "first-line\nsecond-line")
    with pytest.raises(ValueError, match="^website_atlas_api_key_invalid$"):
        atlas.atlas_api_config()


def test_an_uncertain_submission_is_resent_with_the_same_key_and_body(atlas_case):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout"])
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    state = _state(atlas_case)
    # Intent and the exact body were durable before the POST.
    assert state["status"] == "submitting" and "operation_id" not in state
    assert state["request"] == fake.submissions[0][0]
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    # The documented recovery: identical bytes, same Idempotency-Key.
    assert len(fake.submissions) == 2 and fake.submissions[0] == fake.submissions[1]
    assert fake.raw_submissions[0] == fake.raw_submissions[1]
    assert _state(atlas_case)["resends"] == 1


PAST_WINDOW = atlas.UNCERTAIN_RESEND_WINDOW_SECONDS + 1


def _bindings(case):
    return {command["allocation_binding_digest"] for command in case.webapp.spend}


def test_an_uncertain_submission_that_bought_is_adopted_from_the_operation_list(atlas_case):
    atlas_case.webapp.lifetime_seconds = 7 * 86400  # the automatic grant's validity
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["lost"])
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    # It may have bought an operation, so it can never be set aside.
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_not_closed"):
        atlas.archive_closed_atlas_pose_attempt(atlas_case.root)
    # Past the resend window the key may have lapsed: the list, not a resend, decides.
    assert atlas_case.run(fake, now=_later(PAST_WINDOW), max_polls=3)["status"] == "completed"
    assert len(fake.submissions) == 1 and len(_bindings(atlas_case)) == 1
    state = _state(atlas_case)
    assert state["operation_id"] == "op_atlas_1" and state["api_base"] == BASE
    assert state["reconciliations"][0]["outcome"] == "bought"
    assert state["reconciliations"][0]["duplicate_operation_ids"] == []
    # Matching is read from the trace, so the defaults it adds do not hide the purchase.
    assert fake.traces == ["op_atlas_1"]


def test_an_uncertain_submission_proven_unbought_is_resent_once_under_the_same_key(atlas_case):
    atlas_case.webapp.lifetime_seconds = 7 * 86400  # the automatic grant's validity
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout"])
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    assert atlas_case.run(fake, now=_later(PAST_WINDOW), max_polls=3)["status"] == "completed"
    # No new reservation, and the documented recovery: identical bytes, same key.
    assert len(_bindings(atlas_case)) == 1 and len(fake.submissions) == 2
    assert fake.submissions[0] == fake.submissions[1] and fake.raw_submissions[0] == fake.raw_submissions[1]
    state = _state(atlas_case)
    assert state["reconciliations"][0]["outcome"] == "not_bought"
    assert state["reconciliations"][0]["project_asset"] in fake.uploaded
    assert state["reconciliation_reopens"] == 1


def test_reconciliation_waits_until_the_list_can_decide(atlas_case):
    atlas_case.webapp.lifetime_seconds = 7 * 86400  # the automatic grant's validity
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout"] * 4)
    for _ in range(4):
        with pytest.raises(atlas.AtlasTransportError):
            atlas_case.run(fake)
    # Every resend is spent, but the last send is too recent to read "nothing bought".
    with pytest.raises(atlas.AtlasReconciliationPending, match="settling"):
        atlas_case.run(fake)
    # An unreadable trace in the window could be ours.
    fake.foreign["op_unreadable"] = None
    with pytest.raises(atlas.AtlasReconciliationPending, match="listing_unresolved"):
        atlas_case.run(fake, now=_later(atlas.RECONCILE_SETTLE_SECONDS + 1))
    # So could another task that carries a frame only this attempt uploaded.
    asset = _state(atlas_case)["created_assets"][0]
    fake.foreign["op_unreadable"] = {"operationId": "op_unreadable", "taskName": "atlasGenerate", "taskVersion": "v1",
                                     "request": {"contextFrames": [{"imageAsset": {"assetId": asset}}]}}
    with pytest.raises(atlas.AtlasReconciliationPending, match="listing_unresolved"):
        atlas_case.run(fake, now=_later(atlas.RECONCILE_SETTLE_SECONDS + 1))
    # Someone else's operation is ruled out by its trace.
    fake.foreign["op_unreadable"] = {"operationId": "op_unreadable", "taskName": "images2PosedRGBD",
                                     "taskVersion": "v1", "request": {"frames": [{"imageAsset": {"assetId": "asset_x"}}]}}
    # A key that cannot read the request's own frames may list another project.
    fake.mutations.add("inputs_deleted")
    with pytest.raises(atlas.AtlasReconciliationPending, match="project_unproven"):
        atlas_case.run(fake, now=_later(atlas.RECONCILE_SETTLE_SECONDS + 1))
    fake.mutations.discard("inputs_deleted")
    assert len(fake.submissions) == 4
    assert atlas_case.run(fake, now=_later(atlas.RECONCILE_SETTLE_SECONDS + 1), max_polls=3)["status"] == "completed"
    assert len(fake.submissions) == 5 and len(set(fake.raw_submissions)) == 1 and len(_bindings(atlas_case)) == 1


def test_a_repeatedly_unbought_attempt_closes_after_the_reopen_limit(atlas_case):
    atlas_case.webapp.lifetime_seconds = 7 * 86400  # the automatic grant's validity
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout"] * (atlas.RECONCILE_REOPEN_LIMIT + 1))
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    for cycle in range(1, atlas.RECONCILE_REOPEN_LIMIT + 1):
        with pytest.raises(atlas.AtlasTransportError):
            atlas_case.run(fake, now=_later(cycle * 2 * PAST_WINDOW))
    later = _later((atlas.RECONCILE_REOPEN_LIMIT + 1) * 2 * PAST_WINDOW)
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_rejected"):
        atlas_case.run(fake, now=later)
    assert len(fake.submissions) == atlas.RECONCILE_REOPEN_LIMIT + 1
    assert _state(atlas_case)["provider_rejection"]["reconciled_unbought"] is True
    # Proven to have bought nothing, the closed attempt can be set aside.
    assert atlas.archive_closed_atlas_pose_attempt(atlas_case.root) == 1


def test_a_duplicate_purchase_adopts_the_original_and_records_the_rest(atlas_case):
    atlas_case.webapp.lifetime_seconds = 7 * 86400  # the automatic grant's validity
    # A lapsed key let a resend buy a second operation.
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["lost", "lost"])
    for _ in range(2):
        with pytest.raises(atlas.AtlasTransportError):
            atlas_case.run(fake)
    # Another operation in the window whose trace is unreadable could be a third
    # purchase whose outputs would escape cleanup: nothing is adopted until it reads.
    fake.foreign["op_unreadable"] = None
    with pytest.raises(atlas.AtlasReconciliationPending, match="listing_unresolved"):
        atlas_case.run(fake, now=_later(PAST_WINDOW))
    assert "operation_id" not in _state(atlas_case)
    fake.foreign["op_unreadable"] = {"operationId": "op_unreadable", "taskName": "atlasGenerate", "taskVersion": "v1",
                                     "request": {"prompt": "someone else's"}}
    assert atlas_case.run(fake, now=_later(PAST_WINDOW), max_polls=3)["status"] == "completed"
    found = _state(atlas_case)["reconciliations"][0]
    assert found["operation_id"] == "op_atlas_1" and found["duplicate_operation_ids"] == ["op_atlas_1_1"]


def test_an_uncertain_submission_past_retention_fails_and_can_be_set_aside(atlas_case):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout"])
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_failed"):
        atlas_case.run(fake, now=_later(atlas.OPERATION_RETENTION_SECONDS))
    assert _state(atlas_case)["terminal_error"] == {"reason": "reconciliation_window_lapsed"}
    assert not [call for call in fake.calls if "/operations" in call[1]]
    assert atlas.archive_closed_atlas_pose_attempt(atlas_case.root) == 1


def test_trace_matching_needs_every_sent_field_unchanged():
    sent = {"frames": [{"imageAsset": {"assetId": "asset_1"}}], "targetResolution": [1280, 720], "refinePoses": True}
    assert atlas.request_matches_trace(sent, {**sent, "webhookUrl": None})
    assert atlas.request_matches_trace({"fx": 900.0000001}, {"fx": 900.0})
    assert not atlas.request_matches_trace(sent, {**sent, "refinePoses": 1})
    assert not atlas.request_matches_trace(sent, {**sent, "targetResolution": [720, 1280]})
    assert not atlas.request_matches_trace(sent, {key: sent[key] for key in ("frames", "refinePoses")})
    assert not atlas.request_matches_trace(sent, {**sent, "frames": [*sent["frames"], {"imageAsset": {"assetId": "x"}}]})


@pytest.mark.parametrize("status", [408, 429, 503])
def test_throttling_and_server_errors_are_not_refusals(atlas_case, status):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=[status])
    with pytest.raises(atlas.AtlasHttpError, match=f"website_atlas_api_{status}"):
        atlas_case.run(fake)
    assert _state(atlas_case)["status"] == "submitting"
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    assert len(fake.submissions) == 2 and fake.raw_submissions[0] == fake.raw_submissions[1]


def test_a_credit_refusal_is_resent_after_the_documented_interval(atlas_case):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=[402])
    with pytest.raises(atlas.AtlasHttpError, match="website_atlas_api_402"):
        atlas_case.run(fake)
    state = _state(atlas_case)
    assert state["status"] == "rejected" and state["provider_rejection"] == {"status": 402, "retryable": True}
    # Not in a tight loop: the billing docs ask for a top-up first.
    with pytest.raises(ValueError, match="website_atlas_pose_resend_not_due"):
        atlas_case.run(fake)
    assert len(fake.submissions) == 1
    later = _later(atlas.CREDIT_RESEND_INTERVAL_SECONDS + 1)
    assert atlas_case.run(fake, now=later, max_polls=3)["status"] == "completed"
    assert len(fake.submissions) == 2 and fake.raw_submissions[0] == fake.raw_submissions[1]


def test_a_definitive_refusal_closes_the_attempt_until_it_is_archived(atlas_case):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=[400])
    with pytest.raises(atlas.AtlasHttpError, match="website_atlas_api_400"):
        atlas_case.run(fake)
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_rejected"):
        atlas_case.run(fake)
    assert len(fake.submissions) == 1
    # A coded transition, not a hand edit: the closed attempt is set aside and
    # the next one is a new binding with its own reservation.
    assert atlas.archive_closed_atlas_pose_attempt(atlas_case.root) == 1
    assert (atlas_case.root / "atlas_pose_attempts" / "000" / "submission.json").is_file()
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    bindings = [command["allocation_binding_digest"] for command in atlas_case.webapp.spend]
    assert len(set(bindings)) == 2 and _state(atlas_case)["binding"]["attempt"] == 1


def test_a_paid_answer_is_kept_before_it_is_judged(atlas_case):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["bad_id"])
    with pytest.raises(ValueError, match="website_atlas_pose_operation_id_missing"):
        atlas_case.run(fake)
    state = _state(atlas_case)
    assert state["status"] == "submitting" and state["possibly_bought"] is True
    assert state["sends"][-1]["status"] == 201 and "operations/op_atlas_1" in state["sends"][-1]["body"]
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_not_closed"):
        atlas.archive_closed_atlas_pose_attempt(atlas_case.root)


@pytest.mark.parametrize("first", ["timeout", "bad_id", 503])
def test_a_refusal_after_a_send_that_may_have_bought_never_closes_the_attempt(atlas_case, first):
    # A later 4xx cannot prove the earlier send bought nothing, so archiving
    # it could pay for a second operation under a new key.
    fake = FakeAtlas(atlas_case.events, submit_outcomes=[first, 422])
    with pytest.raises((atlas.AtlasTransportError, atlas.AtlasHttpError, ValueError)):
        atlas_case.run(fake)
    with pytest.raises(atlas.AtlasHttpError, match="website_atlas_api_422"):
        atlas_case.run(fake)
    state = _state(atlas_case)
    assert state["status"] == "submitting" and state["possibly_bought"] is True
    assert [send["status"] for send in state["sends"]] == [None if first == "timeout" else
                                                           (201 if first == "bad_id" else first), 422]
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_not_closed"):
        atlas.archive_closed_atlas_pose_attempt(atlas_case.root)


def test_a_credit_refusal_after_an_uncertain_send_stays_uncertain(atlas_case):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout", 402])
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    with pytest.raises(atlas.AtlasHttpError, match="website_atlas_api_402"):
        atlas_case.run(fake)
    state = _state(atlas_case)
    assert state["status"] == "submitting" and state["possibly_bought"] is True
    # The uncertain window, not the credit interval, governs the next resend.
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"


def test_an_uncertain_credit_resend_gets_its_own_window(atlas_case):
    # A 402 bought nothing, so the window starts at the first uncertain send.
    atlas_case.webapp.lifetime_seconds = 86400
    fake = FakeAtlas(atlas_case.events, submit_outcomes=[402, "timeout"])
    with pytest.raises(atlas.AtlasHttpError, match="website_atlas_api_402"):
        atlas_case.run(fake)
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake, now=_later(7200))
    assert atlas_case.run(fake, now=_later(7260), max_polls=3)["status"] == "completed"
    assert len(fake.submissions) == 3 and len(set(fake.raw_submissions)) == 1


def test_failed_operation_is_recorded_and_yields_no_result(atlas_case):
    fake = FakeAtlas(atlas_case.events, operation_error={"code": "INTERNAL", "message": "synthetic failure"})
    with pytest.raises(ValueError, match="website_atlas_pose_operation_failed"):
        atlas_case.run(fake)
    assert _state(atlas_case)["status"] == "failed"
    assert not (atlas_case.root / "atlas_posed_rgbd_result.json").exists()
    calls = len(fake.calls)
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_failed"):
        atlas_case.run(fake)
    assert len(fake.calls) == calls
    assert atlas.archive_closed_atlas_pose_attempt(atlas_case.root) == 1


class FailingFirstAtlas(FakeAtlas):
    """The first ``failures`` bought operations finish with one documented error, later ones succeed."""

    def __init__(self, events, *, failures, code, reason, **kwargs):
        super().__init__(events, **kwargs)
        self.failures = failures
        self.error = {"code": code, "message": "synthetic", "details": [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "domain": "api.worldlabs.ai", "reason": reason}]}

    def __call__(self, method, url, headers, data):
        match = re.fullmatch(re.escape(BASE) + r"/operations/([A-Za-z0-9_]+)", url)
        if method == "GET" and match:
            order = [operation_id for operation_id, _ in self.bought]
            index = order.index(match.group(1)) if match.group(1) in order else 0
            self.operation_error = self.error if index < self.failures else None
        return super().__call__(method, url, headers, data)


@pytest.mark.parametrize("code,reason", [(14, "TRANSIENT_UPSTREAM"), (8, "THROTTLED"), (4, "EXECUTION_TIMEOUT")])
def test_documented_resubmittable_pose_failure_starts_a_separately_reserved_attempt(atlas_case, code, reason):
    fake = FailingFirstAtlas(atlas_case.events, failures=1, code=code, reason=reason)
    result = atlas_case.run(fake, max_polls=2)
    assert result["status"] == "completed" and result["operation_id"] == "op_atlas_1_1"
    # The failed attempt is set aside intact; the new one has its own reservation and key.
    assert (atlas_case.root / "atlas_pose_attempts" / "000" / "submission.json").is_file()
    assert len(atlas_case.webapp.reserved) == 2 and len({key for _, key in fake.submissions}) == 2
    assert atlas_case.run(fake, max_polls=2) == result and len(fake.submissions) == 2


def test_an_execution_failed_pose_is_resubmitted_once(atlas_case):
    fake = FailingFirstAtlas(atlas_case.events, failures=9, code=13, reason="EXECUTION_FAILED")
    with pytest.raises(ValueError, match="website_atlas_pose_operation_failed"):
        atlas_case.run(fake, max_polls=2)
    assert len(fake.submissions) == 2 and len(atlas_case.webapp.reserved) == 2
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_failed"):
        atlas_case.run(fake, max_polls=2)
    assert len(fake.submissions) == 2


@pytest.mark.parametrize("code,reason", [(4, "THROTTLED"), (3, "CONTENT_POLICY")])
def test_other_pose_failures_stay_final(atlas_case, code, reason):
    fake = FailingFirstAtlas(atlas_case.events, failures=9, code=code, reason=reason)
    with pytest.raises(ValueError, match="website_atlas_pose_operation_failed"):
        atlas_case.run(fake, max_polls=2)
    assert len(fake.submissions) == 1 and not (atlas_case.root / "atlas_pose_attempts").exists()


def test_resubmittable_pose_failures_stop_after_three_attempts_across_restarts(atlas_case):
    fake = FailingFirstAtlas(atlas_case.events, failures=9, code=14, reason="TRANSIENT_UPSTREAM")
    with pytest.raises(ValueError, match="website_atlas_pose_operation_failed"):
        atlas_case.run(fake, max_polls=2)
    assert len(fake.submissions) == atlas.MAX_POSE_ATTEMPTS == 3 and len(atlas_case.webapp.reserved) == 3
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_failed"):
        atlas_case.run(fake, max_polls=2)
    assert len(fake.submissions) == 3


def test_a_missing_operation_is_gone_only_after_the_retention_period(atlas_case):
    atlas_case.webapp.lifetime_seconds = 40 * 86400
    fake = FakeAtlas(atlas_case.events, mutations={"gone"})
    # Eventual consistency, another project's key or another host can all
    # answer 404 for an operation that is still running and billed.
    with pytest.raises(ValueError, match="website_atlas_pose_operation_not_found"):
        atlas_case.run(fake)
    assert _state(atlas_case)["status"] == "submitted"
    with pytest.raises(ValueError, match="website_atlas_pose_attempt_not_closed"):
        atlas.archive_closed_atlas_pose_attempt(atlas_case.root)
    retained = _later(atlas.OPERATION_RETENTION_SECONDS + 60)
    with pytest.raises(ValueError, match="website_atlas_pose_operation_gone"):
        atlas_case.run(fake, now=retained)
    assert _state(atlas_case)["status"] == "failed"
    assert atlas.archive_closed_atlas_pose_attempt(atlas_case.root) == 1


def test_a_changed_api_root_never_polls_a_submitted_operation(atlas_case, monkeypatch):
    fake = FakeAtlas(atlas_case.events, polls_until_done=2)
    assert atlas_case.run(fake, max_polls=1)["status"] == "processing"
    monkeypatch.setenv(atlas.ATLAS_API_BASE_URL_ENV, "https://api.worldlabs.ai/api/v2")
    calls = len(fake.calls)
    with pytest.raises(ValueError, match="website_atlas_pose_api_base_changed"):
        atlas_case.run(fake, max_polls=3)
    assert len(fake.calls) == calls


def test_an_attempt_that_never_submitted_can_be_set_aside(atlas_case, monkeypatch):
    # Uploads buy no operation. A stale price review must not strand the capture.
    fake = FakeAtlas(atlas_case.events, put_failure_index=1)
    with pytest.raises(atlas.AtlasHttpError):
        atlas_case.run(fake)
    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, json.dumps(_review(maximum_cost_usd=0.8)))
    with pytest.raises(ValueError, match="website_atlas_pose_already_bound_to_other_inputs"):
        atlas_case.run(fake)
    assert atlas.archive_closed_atlas_pose_attempt(atlas_case.root) == 1
    fake.put_failure_index = None
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    archived = json.loads((atlas_case.root / "atlas_pose_attempts" / "000" / "submission.json").read_text())
    assert archived["status"] == "uploading" and archived["created_assets"] == ["asset_0000", "asset_0001"]


def test_the_provider_poll_retains_a_finished_pose_through_its_own_reader(atlas_case, tmp_path):
    from blueprint_pipeline import website_atlas_preview as preview
    fake = FakeAtlas(atlas_case.events, polls_until_done=2)
    assert atlas_case.run(fake, max_polls=1)["status"] == "processing"
    job = tmp_path / "preview_job.json"
    job.write_text(json.dumps({"state": "processing", "enqueued_at_epoch": int(time.time()),
                               "next_attempt_at_epoch": 0}))
    before, reserved = job.read_bytes(), len(atlas_case.webapp.spend)
    calls = len(fake.calls)
    counts = preview.poll_jobs([(job, lambda path: json.loads(path.read_text()), lambda _job: True)], transport=fake)
    assert counts == {"completed": 1} and _state(atlas_case)["status"] == "completed"
    assert all(method == "GET" or url.endswith(":createReadUrl") for method, url in fake.calls[calls:])
    assert job.read_bytes() == before and len(atlas_case.webapp.spend) == reserved and len(fake.submissions) == 1
    # The tick's own run finds the retained result: no poll, no purchase.
    calls = len(fake.calls)
    assert atlas_case.run(fake)["status"] == "completed" and len(fake.calls) == calls


WEBHOOK = "https://paperclip.tryblueprint.io/api/live-pipeline/atlas-webhook"


def test_a_new_pose_asks_for_the_completion_webhook_outside_what_was_bought(atlas_case, monkeypatch):
    monkeypatch.setenv(atlas.ATLAS_WEBHOOK_URL_ENV, WEBHOOK)
    fake = FakeAtlas(atlas_case.events)
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    body, key = fake.submissions[0]
    state = _state(atlas_case)
    assert body["webhookUrl"] == WEBHOOK and key == "bp-atlas-" + state["request_digest"][7:39]
    # The request record, its binding and the receipt are the task request alone.
    assert "webhookUrl" not in state["request"] and "webhook" not in json.dumps(state["binding"])
    assert "webhookUrl" not in _result(atlas_case)["operation_receipt"]["request"]
    assert geometry.run_website_scene_geometry(**atlas_case.kwargs)["actual_provider_input_count"] == 3


def test_a_pose_sent_before_the_webhook_was_configured_is_resent_byte_for_byte(atlas_case, monkeypatch):
    fake = FakeAtlas(atlas_case.events, submit_outcomes=["timeout"])
    with pytest.raises(atlas.AtlasTransportError):
        atlas_case.run(fake)
    monkeypatch.setenv(atlas.ATLAS_WEBHOOK_URL_ENV, WEBHOOK)
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    assert fake.raw_submissions[0] == fake.raw_submissions[1] and b"webhookUrl" not in fake.raw_submissions[1]
    assert fake.submissions[0][1] == fake.submissions[1][1]


def test_the_reader_reports_an_operation_in_progress(atlas_case):
    fake = FakeAtlas(atlas_case.events, polls_until_done=2)
    assert atlas_case.run(fake, max_polls=1)["status"] == "processing"
    view = geometry.run_website_scene_geometry(**atlas_case.kwargs)
    # Site pixels went to World Labs and an operation was bought: say so.
    assert view["status"] == "pending" and view["atlas_pose_status"] == "submitted"
    assert view["provider_mutation_performed"] is True and view["actual_provider_input_count"] is None
    assert view["blockers"] == ["website_atlas_admitted_pose_operation_required"]


def test_the_reader_treats_an_unfinished_result_as_pending(atlas_case):
    assert atlas_case.run(FakeAtlas(atlas_case.events), max_polls=3)["status"] == "completed"
    # A crash between writing the result and recording completion.
    path = atlas_case.root / "atlas_pose" / "submission.json"
    state = json.loads(path.read_text())
    path.write_text(json.dumps({**state, "status": "submitted"}))
    view = geometry.run_website_scene_geometry(**atlas_case.kwargs)
    assert view["atlas_pose_status"] == "submitted" and view["actual_provider_input_count"] is None


def test_changed_source_pixels_are_refused_before_any_reservation(atlas_case):
    pending = json.loads((atlas_case.root / "atlas_pose_inputs.json").read_text())
    Path(pending["frames"][1]["image_path"]).write_bytes(b"edited pixels")
    with pytest.raises(ValueError, match="website_atlas_pose_input_changed"):
        atlas_case.run(FakeAtlas(atlas_case.events))
    assert atlas_case.events == []


def test_budget_race_retains_pose_admission_before_any_upload_and_reuses_it(atlas_case, monkeypatch):
    fake = FakeAtlas(atlas_case.events)
    stale = False

    def reserve(**kwargs):
        nonlocal stale
        first = not atlas_case.webapp.reserved
        result = atlas_case.webapp(**kwargs)
        if first:
            stale = True
        return result

    monkeypatch.setattr(control, "website_webapp_request", reserve)
    monkeypatch.setattr(admission_gate, "_monthly_budget_blockers",
                        lambda _: ["gcp_monthly_budget_admission_blocked"] if stale else [])
    with pytest.raises(PaidResourceAdmissionBlocked, match="gcp_monthly_budget"):
        atlas_case.run(fake)
    retained = json.loads((atlas_case.root / "atlas_pose/controller_admission.json").read_text())
    assert retained == {**next(iter(atlas_case.webapp.reserved.values())), "source_commit": "1" * 40}
    assert fake.calls == [] and len(atlas_case.webapp.reserved) == 1
    stale = False
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    assert len(atlas_case.webapp.reserved) == len(fake.submissions) == 1


def test_interrupted_uploads_resume_and_every_created_asset_stays_on_record(atlas_case):
    fake = FakeAtlas(atlas_case.events, put_failure_index=1)
    with pytest.raises(atlas.AtlasHttpError, match="website_atlas_api_503"):
        atlas_case.run(fake)
    state = _state(atlas_case)
    assert state["status"] == "uploading" and len(state["uploads"]) == 1
    # The asset whose bytes may have reached the provider is on record.
    assert state["created_assets"] == ["asset_0000", "asset_0001"]
    fake.put_failure_index = None
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    # The WebApp answers the second reservation with already_reserved; only
    # the exact retained admission turns that back into a grant.
    assert len(atlas_case.webapp.spend) == 2 and len(fake.submissions) == 1
    state = _state(atlas_case)
    assert state["created_assets"] == ["asset_0000", "asset_0001", "asset_0002", "asset_0003"]
    assert [row["imageAsset"]["assetId"] for row in fake.submissions[0][0]["frames"]] == [
        row["asset_id"] for row in state["uploads"]]


def test_a_changed_binding_mid_upload_refuses_before_a_second_reservation(atlas_case, monkeypatch):
    fake = FakeAtlas(atlas_case.events, put_failure_index=1)
    with pytest.raises(atlas.AtlasHttpError):
        atlas_case.run(fake)
    # A re-reviewed price is a different binding. Reserving it would charge the
    # sponsorship for an attempt the adapter must then refuse.
    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, json.dumps(_review(maximum_cost_usd=0.8)))
    fake.put_failure_index = None
    with pytest.raises(ValueError, match="website_atlas_pose_already_bound_to_other_inputs"):
        atlas_case.run(fake)
    assert len(atlas_case.webapp.spend) == 1 and fake.submissions == []


def test_a_slow_ready_asset_is_rechecked_never_uploaded_twice(atlas_case):
    # Each re-upload would leave another copy of the site's pixels with the
    # provider. Recorded assets are re-checked on the next pass instead.
    fake = FakeAtlas(atlas_case.events, ready_after_checks=len(atlas.READY_DELAYS_SECONDS) + 2)
    with pytest.raises(ValueError, match="website_atlas_asset_not_ready"):
        atlas_case.run(fake)
    state = _state(atlas_case)
    assert state["status"] == "uploading" and [row["ready"] for row in state["uploads"]] == [False] * 3
    assert len(fake.created) == 3 and fake.submissions == []
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    assert len(fake.created) == 3 and len(fake.submissions) == 1


def test_an_admission_about_to_expire_refuses_before_any_upload(atlas_case):
    atlas_case.webapp.lifetime_seconds = atlas.ADMISSION_MARGIN_SECONDS // 2
    fake = FakeAtlas(atlas_case.events)
    with pytest.raises(ValueError, match="website_atlas_pose_admission_expired"):
        atlas_case.run(fake)
    assert fake.calls == []


@pytest.mark.parametrize("mutation,code", [
    ("bad_mask", "website_atlas_identifier_invalid"),
    ("frame_count", "website_atlas_pose_frames_invalid"),
    ("grid", "website_atlas_pose_grid_not_requested"),
    ("operation_id", "website_atlas_pose_operation_identity_mismatch"),
    ("depth_not_exr", "website_atlas_depth_not_exr"),
    ("truncated", "website_atlas_transfer_size_mismatch"),
    ("size_unreported", "website_atlas_output_size_unreported"),
])
def test_returned_bundle_that_does_not_match_the_request_is_refused(atlas_case, mutation, code):
    fake = FakeAtlas(atlas_case.events, mutations={mutation})
    with pytest.raises(ValueError, match=code):
        atlas_case.run(fake, max_polls=3)
    assert not (atlas_case.root / "atlas_posed_rgbd_result.json").exists()
    if mutation in {"bad_mask", "frame_count", "grid", "operation_id"}:
        # Every reference is checked before the first download.
        assert not any(url.endswith(":createReadUrl") for _, url in fake.calls)


@pytest.mark.parametrize("tamper,code", [
    ("output_bytes", "website_atlas_retained_output_changed"),
    ("asset_binding", "website_atlas_pose_receipt_invalid"),
    ("non_mapping_asset", "website_atlas_pose_receipt_invalid"),
    ("no_admission", "website_atlas_pose_receipt_invalid"),
])
def test_reader_refuses_outputs_or_receipts_that_no_longer_bind(atlas_case, tamper, code):
    assert atlas_case.run(FakeAtlas(atlas_case.events), max_polls=3)["status"] == "completed"
    path = atlas_case.root / "atlas_posed_rgbd_result.json"
    result = json.loads(path.read_text())
    receipt = result["operation_receipt"]
    if tamper == "output_bytes":
        depth = next(row for row in receipt["input_usage"]["frames"][0]["outputs"] if row["role"] == "depth")
        (atlas_case.root / depth["path"]).write_bytes(EXR + b"edited")
    elif tamper == "no_admission":
        # A self-consistent result with no admitted run behind it reports nothing.
        (atlas_case.root / "atlas_pose" / "controller_admission.json").unlink()
    else:
        if tamper == "asset_binding":
            receipt["input_usage"]["frames"][0]["asset_id"] = "asset_elsewhere"
        else:
            receipt["request"]["frames"][0]["imageAsset"] = "asset_0000"
        result["digest"] = canonical_digest(result, digest_field="digest")
        path.write_text(json.dumps(result))
        # Keep the producer's record in step, so only the asset checks can refuse.
        state_path = atlas_case.root / "atlas_pose" / "submission.json"
        state_path.write_text(json.dumps({**json.loads(state_path.read_text()), "result_digest": result["digest"]}))
    with pytest.raises(ValueError, match=code):
        geometry.run_website_scene_geometry(**atlas_case.kwargs)


def test_default_geometry_route_never_calls_atlas_even_with_a_task_context(atlas_case, monkeypatch):
    monkeypatch.setattr(atlas, "_safe_transport", lambda *_a, **_k: pytest.fail("the default route called Atlas"))
    result = geometry.run_website_scene_geometry(**atlas_case.kwargs, task_context=atlas_case.context)
    assert result["status"] == "pending" and result["provider_mutation_performed"] is False
    assert result["actual_provider_input_count"] == 0
    assert result["blockers"] == ["website_atlas_admitted_pose_operation_required"]
    assert atlas_case.events == []


def test_adapter_refuses_any_upload_without_the_exact_grant(atlas_case):
    pending = atlas.load_pending_atlas_pose_inputs(atlas_case.root)
    fake = FakeAtlas(atlas_case.events)
    client = atlas.AtlasClient(base_url=BASE, api_key=KEY, transport=fake)
    grid = atlas.atlas_pose_target_resolution(pending)
    binding = atlas.atlas_pose_request_binding(
        pose_inputs=pending, spend=atlas.atlas_pose_spend_line(planned_input_count=3, target_resolution=grid),
        task_context=atlas_case.context, target_resolution=grid,
        provider_retention="account_does_not_retain", attempt=0)
    digest = canonical_digest(binding)
    admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                 "resource_class": "provider_reconstruction_api", "provider": "world_labs_atlas",
                 "maximum_cost_usd": 0.75, "external_disclosure_allowed": True,
                 "expires_at_epoch": time.time() + 3600, "allocation_binding_digest": digest}
    other = require_paid_resource_admission({**admission, "allocation_binding_digest": "sha256:" + "0" * 64},
        resource_class="provider_reconstruction_api", expected_schema_version="paid_lane_admission.v1")
    grant = require_paid_resource_admission(admission, resource_class="provider_reconstruction_api",
                                            expected_schema_version="paid_lane_admission.v1")

    def submit(admission=admission, grant=grant):
        return atlas.submit_atlas_pose_operation(
            pose_inputs=pending, output_root=atlas_case.root, client=client, binding=binding,
            admission=admission, grant=grant, sleep=lambda _seconds: None, now=time.time)

    for wrong in (None, other):
        with pytest.raises(PaidResourceAdmissionBlocked):
            submit(grant=wrong)
    for field, value, code in (("external_disclosure_allowed", False, "disclosure_not_authorized"),
                               ("maximum_cost_usd", 0.5, "budget_insufficient"),
                               ("provider", "world_labs", "spend_binding_missing")):
        with pytest.raises(ValueError, match=f"website_atlas_pose_{code}"):
            submit(admission={**admission, field: value})
    assert fake.calls == []


@pytest.mark.parametrize("atlas_case", [[PORTRAIT] * 3], indirect=True)
def test_portrait_capture_requests_a_portrait_grid(atlas_case):
    fake = FakeAtlas(atlas_case.events)
    assert atlas_case.run(fake, max_polls=3)["status"] == "completed"
    assert fake.submissions[0][0]["targetResolution"] == [720, 1280]


@pytest.mark.parametrize("atlas_case", [[LANDSCAPE, PORTRAIT, LANDSCAPE], [(32, 24)] * 3, [(24, 24)] * 3],
                         indirect=True, ids=["mixed_orientation", "four_by_three", "square"])
def test_views_that_atlas_would_crop_are_refused_before_any_reservation(atlas_case):
    # Atlas scales each view to cover the grid and center-crops it; a 4:3
    # frame on a 16:9 grid would lose a quarter of the site.
    with pytest.raises(ValueError, match="website_atlas_pose_view_crop_too_large"):
        atlas_case.run(FakeAtlas(atlas_case.events))
    assert atlas_case.events == []


def test_the_key_never_reaches_upload_hosts(atlas_case):
    fake = FakeAtlas(atlas_case.events, grant_headers={"x-upload-token": "t", "wlt-api-key": "echo"})
    with pytest.raises(ValueError, match="website_atlas_upload_grant_invalid"):
        atlas_case.run(fake)
    assert not any(url.startswith("https://upload.") for _, url in fake.calls)
    fake = FakeAtlas(atlas_case.events, upload_scheme="http")
    with pytest.raises(ValueError, match="website_atlas_provider_url_invalid"):
        atlas_case.run(fake)
    assert not any(url.startswith("http://upload.") for _, url in fake.calls)


def test_the_default_transport_uses_the_audited_boundary(monkeypatch):
    seen = []

    def request(url, *, method, data, headers, timeout_seconds, policy):
        seen.append((method, url, dict(headers), policy))
        if url.endswith("/broken-header"):
            raise ValueError(f"Invalid header value {headers['WLT-Api-Key']!r}")
        if url.endswith("/refused"):
            error = urllib.error.HTTPError(url, 402, "Payment Required", {}, None)
            monkeypatch.setattr(error, "read", lambda *_a: (_ for _ in ()).throw(TimeoutError()))
            raise error
        return safe_outbound_http.SafeHttpResponse(status=200, body=b'{"id": "op_1"}', url=url, final_url=url)

    monkeypatch.setattr(safe_outbound_http, "request", request)
    client = atlas.AtlasClient(base_url=BASE, api_key=KEY)
    assert client.call("GET", "/operations/op_1") == {"id": "op_1"}
    assert client.fetch_bytes("https://storage.example.test/object.exr") == b'{"id": "op_1"}'
    (_, _, api_headers, api_policy), (_, _, read_headers, read_policy) = seen
    assert api_policy.allowed_hosts == frozenset({"api.atlas-beta.worldlabs.ai"}) and api_headers["WLT-Api-Key"] == KEY
    assert read_policy.allowed_hosts == frozenset({"storage.example.test"}) and "WLT-Api-Key" not in read_headers
    # A definitive status survives a lost error body.
    assert client.send("POST", "/refused", {}) == (402, b"")
    with pytest.raises(atlas.AtlasTransportError) as error:
        client.call("GET", "/broken-header")
    assert KEY not in str(error.value) and error.value.__cause__ is None


def test_retry_after_survives_audited_error_transport_without_retaining_other_headers(monkeypatch):
    def request(url, **kwargs):
        if url.endswith("/tasks:atlasGenerate"):
            raise urllib.error.HTTPError(url, 429, "Busy", {"Retry-After": "120", "Set-Cookie": "private"},
                                         io.BytesIO(b'{"code":"CONCURRENCY_LIMIT_EXCEEDED"}'))
        return safe_outbound_http.SafeHttpResponse(status=200, body=b"{}", url=url, final_url=url)

    monkeypatch.setattr(safe_outbound_http, "request", request)
    client = atlas.AtlasClient(base_url=BASE, api_key=KEY)
    assert client.send("POST", "/tasks:atlasGenerate", {})[0] == 429
    assert client.retry_after == "120"
    assert "private" not in repr(vars(client))
    client.call("GET", "/operations/op_1")
    assert client.retry_after is None


@pytest.mark.parametrize("model", ["atlas-beta", "worldlabs-atlas"])
def test_the_marble_request_builder_never_names_another_model(monkeypatch, model):
    from blueprint_pipeline.provider_preview import WorldLabsPreviewProvider
    monkeypatch.setenv("WORLDLABS_DEFAULT_MODEL", model)
    descriptor = {"scene_id": "scene-1", "capture_id": "capture-1", "metadata": {
        "worldlabs_input_video_uri": "gs://bucket/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/in.mp4"}}
    # Refused while building the request, before any video upload.
    with pytest.raises(ValueError, match="worldlabs_model_not_a_world_model"):
        WorldLabsPreviewProvider()._build_request_manifest(descriptor=descriptor, capture_root=Path("/tmp/capture"))
