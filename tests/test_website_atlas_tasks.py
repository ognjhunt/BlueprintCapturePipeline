"""No provider calls: durable task execution and the public beta schema boundary."""
import json
import time
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import paid_resource_admission as admission_gate
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
        # "lost": the task is accepted but its answer never arrives. A status code refuses it.
        self.outcome = None
        self.bought = []

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
            if isinstance(self.outcome, int):
                return self.outcome, b'{"detail": "refused for test"}'
            if not self.uncertain:
                self.bought.append(json.loads(data))
            if self.uncertain or self.outcome == "lost":
                raise pose.AtlasTransportError("test uncertain POST")
            value = {"id": "op_task_1", "done": False}
        elif path.startswith("/operations?"):
            value = {"operations": [{"id": "op_task_1", "metadata": None, "done": False}] if self.bought else [],
                     "nextPageToken": None}
        elif path == "/operations/op_task_1:trace":
            value = {"operationId": "op_task_1", "taskName": "atlasGenerate", "taskVersion": "v1",
                     "request": {**self.bought[0], "modelParameters": {"seed": 42}}, "response": None, "error": None}
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


def test_known_budget_refusal_does_not_reserve_or_touch_atlas(case, monkeypatch, tmp_path):
    fake, _, _, run = case
    monkeypatch.setattr(admission_gate, "_monthly_budget_blockers",
                        lambda _: ["gcp_monthly_budget_admission_blocked"])
    with pytest.raises(admission_gate.PaidResourceAdmissionBlocked, match="gcp_monthly_budget"):
        run()
    assert fake.reservations == fake.posts == 0
    assert not (tmp_path / "controller_admission.json").exists()
    assert not (tmp_path / "atlas_pose/submission.json").exists()


def test_budget_expiring_during_reservation_retains_receipt_and_resumes_once(case, monkeypatch, tmp_path):
    fake, _, _, run = case
    stale = False

    def reserve(**kwargs):
        nonlocal stale
        first = fake.admission is None
        result = fake.reserve(**kwargs)
        if first:
            stale = True
        return result

    monkeypatch.setattr(control, "website_webapp_request", reserve)
    monkeypatch.setattr(admission_gate, "_monthly_budget_blockers",
                        lambda _: ["gcp_monthly_budget_admission_blocked"] if stale else [])
    with pytest.raises(admission_gate.PaidResourceAdmissionBlocked, match="gcp_monthly_budget"):
        run()
    retained = json.loads((tmp_path / "controller_admission.json").read_text())
    assert retained == {**fake.admission, "source_commit": "a" * 40}
    assert (tmp_path / "controller_admission.json").stat().st_mode & 0o777 == 0o600
    assert fake.reservations == 1 and fake.posts == 0
    assert not (tmp_path / "atlas_pose/submission.json").exists()
    stale = False  # The real monitor supplies a fresh report on the next tick.
    assert run()["status"] == "completed"
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 1


class TransientFake(Fake):
    def __init__(self, failures=1, code=14, reason="TRANSIENT_UPSTREAM"):
        super().__init__()
        self.failures, self.code, self.reason = failures, code, reason
        self.admissions, self.keys = {}, []

    def reserve(self, *, capture_id, operation, payload):
        spend = payload["spend"]
        key = spend["allocation_binding_digest"]
        if key not in self.admissions:
            self.reservations += 1
            self.admissions[key] = {**spend, "schema_version": "paid_lane_admission.v1",
                "status": "admitted", "blockers": [], "external_disclosure_allowed": True,
                "expires_at_epoch": 1e12}
        return self.admissions[key]

    def transport(self, method, url, headers, data):
        if url.endswith("/tasks:atlasGenerate"):
            self.keys.append(headers["Idempotency-Key"])
            status, body = super().transport(method, url, headers, data)
            return status, json.dumps({"id": f"op_task_{self.posts}", "done": False}).encode()
        if "/operations/op_task_" in url and not url.endswith(":trace"):
            index = int(url.rsplit("_", 1)[-1])
            self.polls += 1
            if index <= self.failures:
                return 200, json.dumps({"id": f"op_task_{index}", "done": True,
                    "error": {"code": self.code, "message": "upstream unavailable",
                              "details": [{"domain": "api.worldlabs.ai", "reason": self.reason}]}}).encode()
            return 200, json.dumps({"id": f"op_task_{index}", "done": not self.pending,
                "response": {"frames": [{"imageAsset": {"assetId": "asset_output"}, "camera": camera(.1),
                    "depth": {"depthAsset": {"assetId": "asset_depth"},
                              "confidenceAsset": {"assetId": "asset_confidence"}}}], "promptUsed": "same kitchen"}}).encode()
        return super().transport(method, url, headers, data)


def transient_case(case, monkeypatch, failures=1, code=14, reason="TRANSIENT_UPSTREAM"):
    _, _, _, run = case
    fake = TransientFake(failures, code, reason)
    monkeypatch.setattr(control, "website_webapp_request", fake.reserve)
    # The fixture closure intentionally uses Fake.transport; replace its target.
    original = case[0]
    original.transport = fake.transport
    return fake, run


def test_terminal_upstream_retry_keeps_separate_spend_and_replays_without_purchase(case, monkeypatch, tmp_path):
    fake, run = transient_case(case, monkeypatch)
    result = run()
    assert result["status"] == "completed" and result["operation"]["id"] == "op_task_2"
    assert fake.posts == fake.reservations == len(set(fake.keys)) == 2
    assert result["billing"]["total_reserved_usd"] == .1
    assert result["retry_history"][0]["operation_id"] == "op_task_1"
    state = json.loads((tmp_path / "atlas_pose/submission.json").read_text())
    assert state["terminal_operation"]["error"]["code"] == 14
    assert state["completed_operation_id"] == "op_task_2"
    assert all((tmp_path / row["path"]).is_file() for row in result["outputs"])
    assert run() == result and fake.posts == fake.reservations == 2


def test_upstream_retry_limit_persists_across_restarts(case, monkeypatch):
    fake, run = transient_case(case, monkeypatch, failures=9)
    for _ in range(2):
        with pytest.raises(ValueError, match="atlas_task_operation_failed"):
            run()
        assert fake.posts == fake.reservations == len(set(fake.keys)) == 3


def test_pending_retry_resumes_the_same_purchase(case, monkeypatch):
    fake, run = transient_case(case, monkeypatch)
    fake.pending = True
    assert run()["operation_id"] == "op_task_2"
    assert run()["status"] == "processing"
    assert fake.posts == fake.reservations == 2
    fake.pending = False
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 2


def test_two_failures_keep_all_receipts_and_portable_outputs(case, monkeypatch, tmp_path):
    fake, run = transient_case(case, monkeypatch, failures=2)
    result = run()
    assert result["operation"]["id"] == "op_task_3"
    assert [r["operation_id"] for r in result["retry_history"]] == ["op_task_1", "op_task_2"]
    assert result["billing"]["total_reserved_usd"] == .15
    assert all((tmp_path / row["path"]).is_file() for row in result["outputs"])
    assert run() == result and fake.posts == fake.reservations == 3


def test_retry_refuses_an_external_symlink(case, monkeypatch, tmp_path):
    fake, run = transient_case(case, monkeypatch)
    (tmp_path / "retry").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="retry_root_symlink"):
        run()
    assert fake.posts == fake.reservations == 1


def test_nontransient_terminal_failure_never_retries(case, monkeypatch):
    fake, run = transient_case(case, monkeypatch, code=3)
    with pytest.raises(ValueError, match="atlas_task_operation_failed"):
        run()
    assert fake.posts == fake.reservations == 1


@pytest.mark.parametrize("code,reason,resubmitted", [
    (8, "THROTTLED", True), (4, "EXECUTION_TIMEOUT", True),
    # The provider says to investigate an execution failure before resubmitting it.
    (13, "EXECUTION_FAILED", False),
    # A reason is resubmittable only with its documented code; code 4 alone can mean expiry.
    (4, "THROTTLED", False), (8, "EXECUTION_TIMEOUT", False)])
def test_documented_resubmittable_failures_retry_with_new_spend(case, monkeypatch, code, reason, resubmitted):
    fake, run = transient_case(case, monkeypatch, code=code, reason=reason)
    if resubmitted:
        assert run()["operation"]["id"] == "op_task_2"
        assert fake.posts == fake.reservations == len(set(fake.keys)) == 2
    else:
        with pytest.raises(ValueError, match="atlas_task_operation_failed"):
            run()
        assert fake.posts == fake.reservations == 1


def test_transient_retry_requires_new_budget_and_release_admission(case, monkeypatch):
    fake, run = transient_case(case, monkeypatch)
    def refuse_retry(*_a, **_kw):
        return (["release_changed"] if fake.posts else [], "a" * 40)
    monkeypatch.setattr(allocator, "_source_checkout_blockers", refuse_retry)
    with pytest.raises(ValueError, match="release_not_admitted"):
        run()
    assert fake.posts == fake.reservations == 1


def test_transient_retry_stops_before_post_when_budget_refuses(case, monkeypatch):
    fake, run = transient_case(case, monkeypatch)
    def budget(**kw):
        if fake.reservations:
            raise ValueError("test_budget_exhausted")
        return fake.reserve(**kw)
    monkeypatch.setattr(control, "website_webapp_request", budget)
    with pytest.raises(ValueError, match="test_budget_exhausted"):
        run()
    assert fake.posts == fake.reservations == 1


def test_polling_resumes_without_new_reservation(case):
    fake, _, _, run = case
    fake.pending = True
    assert run()["status"] == "processing"
    fake.pending = False
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 1


@pytest.mark.parametrize("code", sorted(tasks._CAPACITY_CODES))
@pytest.mark.parametrize("retry_after", ["1800", "Fri, 23 Oct 2026 16:30:00 GMT"])
def test_capacity_refusal_resumes_same_purchase_after_durable_deadline(case, tmp_path, code, retry_after):
    fake, _, _, run = case
    original = fake.transport
    sent = []
    received = datetime(2026, 10, 23, 16, tzinfo=timezone.utc).timestamp()

    def limited(method, url, headers, data):
        if url.endswith("/tasks:atlasGenerate"):
            sent.append((headers["Idempotency-Key"], data))
            if len(sent) == 1:
                fake.posts += 1
                return 429, json.dumps({"code": code}).encode(), {"Retry-After": retry_after}
        assert "/operations?" not in url, "A definitive refusal does not need purchase reconciliation"
        return original(method, url, headers, data)

    fake.transport = limited
    first = run(now=lambda: received)
    assert first["status"] == "processing" and first["waiting_for"] == "provider_capacity"
    deadline = first["retry_not_before_epoch"]
    assert received + 1800 <= deadline <= received + 1805
    state = json.loads((tmp_path / "atlas_pose/submission.json").read_text())
    assert state["status"] == "capacity_wait" and not state.get("operation_id")
    assert state["sends"][0]["capacity_refusal"]["accepted"] is False
    # Each run constructs a fresh client; a restart neither shortens the wait nor resends early.
    assert run(now=lambda: deadline - .01) == first
    assert fake.posts == fake.reservations == 1
    assert run(now=lambda: deadline)["status"] == "completed"
    assert sent[0] == sent[1] and fake.posts == 2 and fake.reservations == 1
    assert run()["status"] == "completed" and fake.posts == 2


@pytest.mark.parametrize("header", [None, "not a date", "nan", "inf", "-10"])
def test_capacity_retry_without_valid_hint_uses_bounded_backoff(header):
    for attempt, expected in [(1, 60), (2, 120), (20, 300)]:
        result = tasks._capacity_refusal(b'{"code":"TASK_UNAVAILABLE"}', header,
                                          now=1000, attempt=attempt)
        assert 1000 + expected <= result["retry_not_before_epoch"] <= 1005 + expected


def test_unknown_or_legacy_429_still_requires_reconciliation(case, tmp_path):
    fake, _, _, run = case
    fake.outcome = 429
    with pytest.raises(pose.AtlasHttpError):
        run()
    assert json.loads((tmp_path / "atlas_pose/submission.json").read_text())["status"] == "submitting"
    with pytest.raises(pose.AtlasReconciliationPending):
        run()
    assert fake.posts == fake.reservations == 1


def test_capacity_retry_rechecks_release_before_resubmission(case, monkeypatch):
    fake, _, _, run = case
    original = fake.transport

    def limited(method, url, headers, data):
        if url.endswith("/tasks:atlasGenerate"):
            fake.posts += 1
            return 429, b'{"code":"CONCURRENCY_LIMIT_EXCEEDED"}'
        return original(method, url, headers, data)

    fake.transport = limited
    pending = run()
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_a, **_kw: (["release_changed"], "a" * 40))
    with pytest.raises(ValueError, match="release_not_admitted"):
        run(now=lambda: pending["retry_not_before_epoch"])
    assert fake.posts == fake.reservations == 1


def _later(seconds):
    return lambda: time.time() + seconds


def test_uncertain_send_never_causes_a_fresh_purchase(case):
    fake, _, _, run = case
    fake.uncertain = True
    with pytest.raises(pose.AtlasTransportError):
        run()
    # Too soon for the operation list to show that nothing was bought.
    with pytest.raises(pose.AtlasReconciliationPending, match="settling"):
        run()
    assert fake.posts == fake.reservations == 1
    # Proven unbought, the same bytes go again under the same key, on the same reservation.
    fake.uncertain = False
    assert run(now=_later(pose.RECONCILE_SETTLE_SECONDS + 1))["status"] == "completed"
    assert fake.posts == 2 and fake.reservations == 1


def test_a_lost_answer_is_adopted_from_the_operation_list(case, tmp_path):
    fake, _, _, run = case
    fake.outcome = "lost"
    with pytest.raises(pose.AtlasTransportError):
        run()
    # The trace carries schema defaults we never sent; every field we sent still matches.
    assert run()["status"] == "completed"
    assert fake.posts == fake.reservations == 1
    state = json.loads((tmp_path / "atlas_pose" / "submission.json").read_text())
    assert state["reconciliations"][0]["outcome"] == "bought"


def test_a_definitive_refusal_closes_the_task_without_a_purchase(case):
    fake, _, _, run = case
    fake.outcome = 422
    with pytest.raises(pose.AtlasHttpError, match="website_atlas_api_422"):
        run()
    with pytest.raises(ValueError, match="atlas_task_attempt_rejected"):
        run()
    assert fake.posts == fake.reservations == 1


def test_a_repeatedly_unbought_task_closes_after_the_reopen_limit(case):
    fake, _, _, run = case
    fake.uncertain = True
    for cycle in range(pose.RECONCILE_REOPEN_LIMIT + 1):
        with pytest.raises(pose.AtlasTransportError):
            run(now=_later(cycle * 2 * pose.RECONCILE_SETTLE_SECONDS))
    with pytest.raises(ValueError, match="atlas_task_attempt_rejected"):
        run(now=_later((pose.RECONCILE_REOPEN_LIMIT + 1) * 2 * pose.RECONCILE_SETTLE_SECONDS))
    assert fake.posts == pose.RECONCILE_REOPEN_LIMIT + 1 and fake.reservations == 1


def test_a_legacy_uncertain_record_reconciles_from_its_write_time(case, tmp_path):
    fake, _, _, run = case
    fake.outcome = "lost"
    with pytest.raises(pose.AtlasTransportError):
        run()
    path = tmp_path / "atlas_pose" / "submission.json"
    state = json.loads(path.read_text())
    state.pop("sends")
    path.write_text(json.dumps(state))
    assert run()["status"] == "completed" and fake.posts == 1


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
