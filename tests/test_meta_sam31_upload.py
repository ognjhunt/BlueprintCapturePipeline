"""A multi-frame SAM request sends its clip as one uploaded file, deleted however the run ends.

capture-39ecdc97 (2026-11-11): the Atlas background step sent 16 lossless frames inline as a
base64 data URL (about 16.5 MiB per request, once per prompt). Nine prompts were answered and the
tenth was refused with 413. These tests hold the upload transport, its cleanup on every ending,
the resume from the inline run's retained answers, and the money rules around a failed upload.
"""
from __future__ import annotations

from io import BytesIO
import json
from urllib.error import HTTPError

import pytest
from PIL import Image

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission

MASK = '<0f>0<|box;x1=1;y1=0;x2=2;y2=1;w=4;h=4|><|mask;x=0;y=0;data=2,2,!!!!!&T:u9`|>'


def _answer():
    return {"status": "completed", "usage": {"video_frames_processed": 50},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": MASK}]}]}


def _job(tmp_path, monkeypatch, prompts=3):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    rows, artifacts = [], []
    for index, colour in enumerate(("blue", "red")):
        image = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (4, 4), colour).save(image)
        rows.append({"source_frame_id": f"decoded-{index}", "width": 4, "height": 4})
        artifacts.append({"source_frame_id": f"decoded-{index}", "path": str(image), "sha256": _sha256_file(image)})
    texts = [{"prompt_id": f"target-{i}", "output_label": f"target-{i}", "text": f"object {i}"} for i in range(prompts)]
    args = dict(frame_registry=rows, frame_artifacts=artifacts, prompts=texts, output_root=tmp_path / "out",
                admission={}, admission_grant=None, task_context={"capture_id": "fixture"})
    digest = canonical_digest(sam.request_binding(frame_registry=rows, frame_artifacts=artifacts, prompts=texts))
    return args, digest, args["output_root"] / digest[7:]


def _reservations(monkeypatch):
    rows = []

    def reserve(**kwargs):
        # The website answers a second reservation of one binding "already_reserved".
        assert kwargs["binding_digest"] not in [row["binding_digest"] for row in rows], "already_reserved"
        rows.append(kwargs)
        value = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "allocation_binding_digest": kwargs["binding_digest"], "request_count": kwargs["request_count"],
                 "maximum_cost_usd": kwargs["maximum_cost_usd"], "external_disclosure_allowed": True}
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
                                                      expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    return rows


class Provider:
    """Answers uploads, deletions and prompts; records each call."""

    def __init__(self, *, refuse=(), refuse_upload=None, refuse_delete=False):
        self.calls, self.uploads, self.deleted = [], 0, []
        self.refuse, self.refuse_upload, self.refuse_delete = dict(refuse), refuse_upload, refuse_delete

    def __call__(self, request, **_kwargs):
        if request.full_url == sam.FILES_ENDPOINT:
            self.calls.append("upload")
            if self.refuse_upload is not None:
                code, self.refuse_upload = self.refuse_upload, None
                raise HTTPError(sam.FILES_ENDPOINT, code, "Refused", {}, None)
            self.uploads += 1
            return BytesIO(json.dumps({"id": f"file-clip{self.uploads}"}).encode())
        if request.get_method() == "DELETE":
            self.calls.append("delete")
            if self.refuse_delete:
                raise ConnectionResetError("no answer")
            self.deleted.append(request.full_url.rsplit("/", 1)[1])
            return BytesIO(b'{"deleted":true}')
        payload = json.loads(request.data)
        media = payload["input"][0]["content"][1]
        assert "video_url" not in media and media["type"] == "input_video"
        text = payload["input"][0]["content"][0]["text"]
        self.calls.append((text, media["file_id"]))
        if self.refuse.get(text):
            self.refuse[text] -= 1
            raise HTTPError(sam.ENDPOINT, 413, "Request Entity Too Large", {}, None)
        return BytesIO(json.dumps(_answer()).encode())


def test_every_prompt_names_one_uploaded_clip_and_the_upload_is_deleted(tmp_path, monkeypatch):
    args, digest, root = _job(tmp_path, monkeypatch)
    reservations = _reservations(monkeypatch)
    provider = Provider()
    result = sam.run_meta_sam31(**args, opener=provider)
    assert result["status"] == "completed"
    assert provider.calls == ["upload", ("object 0", "file-clip1"), ("object 1", "file-clip1"),
                              ("object 2", "file-clip1"), "delete"]
    assert provider.deleted == ["file-clip1"]
    assert json.loads((root / sam.UPLOAD_CLEANUP).read_text()) == {"file_id": "file-clip1", "deleted": True}
    # The transport is not part of the bought request: the binding is the one inline runs used.
    assert canonical_digest(json.loads((root / "binding.json").read_text())) == digest
    assert "video_transport" not in json.loads((root / "binding.json").read_text())
    # The reservation per request is unchanged: one hold for the three prompts at the per-request price.
    assert len(reservations) == 1 and reservations[0]["request_count"] == 3
    assert reservations[0]["maximum_cost_usd"] == pytest.approx(3 * 50 * sam.PRICE_PER_FRAME_USD)
    # A replay buys nothing and calls nothing: its upload is already recorded deleted.
    replay = Provider()
    assert sam.run_meta_sam31(**{**args, "task_context": None}, opener=replay) == result
    assert replay.calls == []


def test_an_inline_run_refused_413_resumes_and_buys_only_its_missing_prompts(tmp_path, monkeypatch):
    # The reference job's state: prompts answered under the inline transport, then one refused 413.
    args, digest, root = _job(tmp_path, monkeypatch, prompts=4)
    root.mkdir(parents=True)
    sam.write_json(root / "binding.json", sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=args["prompts"]))
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    for index in (0, 1):
        sam.write_json(root / f"intent-{index}.json", {"binding_digest": digest, "clip_digest": _sha256_file(clip),
                                                       "allocation_binding_digest": digest})
        sam.write_json(root / f"response-{index}.json", {"binding_digest": digest, "clip_digest": _sha256_file(clip),
            "allocation_binding_digest": digest, "response": _answer(), "reserved_cost_usd": 0.01})
    kept = [(root / f"response-{index}.json").read_bytes() for index in (0, 1)]
    sam.write_json(root / "intent-2.json", {"binding_digest": digest, "clip_digest": _sha256_file(clip),
                                            "allocation_binding_digest": digest})
    sam.write_json(root / "failure-2.json", {"http_status": 413, "binding_digest": digest})
    reservations = _reservations(monkeypatch)
    provider = Provider()
    result = sam.run_meta_sam31(**args, opener=provider)
    assert result["status"] == "completed"
    assert provider.calls == ["upload", ("object 2", "file-clip1"), ("object 3", "file-clip1"), "delete"]
    assert [(root / f"response-{index}.json").read_bytes() for index in (0, 1)] == kept
    assert len(reservations) == 1 and reservations[0]["request_count"] == 2
    assert reservations[0]["maximum_cost_usd"] == pytest.approx(2 * 50 * sam.PRICE_PER_FRAME_USD)
    assert reservations[0]["binding_digest"] != digest
    assert json.loads((root / "refused/attempt-1/failure-2.json").read_text())["http_status"] == 413


def test_a_refused_prompt_deletes_the_upload_and_the_next_run_uploads_again(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    reservations = _reservations(monkeypatch)
    provider = Provider(refuse={"object 1": 1})
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls[-1] == "delete" and provider.deleted == ["file-clip1"]
    result = sam.run_meta_sam31(**args, opener=provider)
    assert result["status"] == "completed"
    # Never the deleted file's id: a fresh upload, itself deleted when the run ends.
    assert provider.calls[-4:] == ["upload", ("object 1", "file-clip2"), ("object 2", "file-clip2"), "delete"]
    assert provider.deleted == ["file-clip1", "file-clip2"]
    archived = root / sam.UPLOAD_ARCHIVE / "attempt-1"
    assert json.loads((archived / sam.UPLOAD_RECEIPT).read_text())["file_id"] == "file-clip1"
    assert json.loads((archived / sam.UPLOAD_CLEANUP).read_text())["deleted"] is True
    assert len(reservations) == 2


def test_a_delete_left_undone_by_a_crash_is_finished_by_the_next_run_even_a_replay(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    _reservations(monkeypatch)
    provider = Provider(refuse_delete=True)
    result = sam.run_meta_sam31(**args, opener=provider)
    assert json.loads((root / sam.UPLOAD_CLEANUP).read_text()) == {"file_id": "file-clip1", "deleted": False}
    # A process killed before its deletion leaves no cleanup record at all.
    (root / sam.UPLOAD_CLEANUP).unlink()
    later = Provider()
    assert sam.run_meta_sam31(**{**args, "task_context": None}, opener=later) == result
    assert later.calls == ["delete"] and later.deleted == ["file-clip1"]
    assert json.loads((root / sam.UPLOAD_CLEANUP).read_text()) == {"file_id": "file-clip1", "deleted": True}


def test_a_crash_after_the_upload_resumes_with_that_upload_and_then_deletes_it(tmp_path, monkeypatch):
    args, digest, root = _job(tmp_path, monkeypatch, prompts=1)
    root.mkdir(parents=True)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    sam.write_json(root / sam.UPLOAD_INTENT, {"clip_digest": _sha256_file(clip)})
    sam.write_json(root / sam.UPLOAD_RECEIPT, {"file_id": "file-earlier", "clip_digest": _sha256_file(clip)})
    _reservations(monkeypatch)
    provider = Provider()
    sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == [("object 0", "file-earlier"), "delete"]
    assert provider.deleted == ["file-earlier"]


def test_a_refused_upload_reuses_its_own_reservation_and_leaves_no_unknown_copy(tmp_path, monkeypatch):
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as provider_copies

    args, digest, root = _job(tmp_path, monkeypatch)
    reservations = _reservations(monkeypatch)
    reused = []

    def reuse(*, task_context, admission, binding_digest):
        reused.append(binding_digest)
        assert admission["allocation_binding_digest"] == binding_digest
        return require_paid_resource_admission(admission, resource_class="evaluator_api",
                                               expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reuse_website_sam_spend", reuse)
    provider = Provider(refuse_upload=413)
    with pytest.raises(ValueError, match="meta_sam_upload_http_413"):
        sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == ["upload"] and not list(root.glob("intent-*.json"))
    _, unknown = provider_copies._discover(tmp_path)
    assert unknown == []
    result = sam.run_meta_sam31(**args, opener=provider)
    assert result["status"] == "completed"
    # One hold in all: the run that stopped before dispatching anything pays the next one.
    assert len(reservations) == 1 and reused == [digest]
    assert provider.calls[-1] == "delete" and provider.deleted == ["file-clip1"]


def test_an_unanswered_upload_still_stops_for_reconciliation(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    _reservations(monkeypatch)

    def no_answer(request, **_kwargs):
        raise ConnectionResetError("no answer")

    with pytest.raises(ConnectionResetError):
        sam.run_meta_sam31(**args, opener=no_answer)
    with pytest.raises(ValueError, match="video_upload_requires_reconciliation"):
        sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("an uncertain upload is never repeated"))


def test_a_retained_meta_reservation_is_reusable_only_for_its_own_binding(monkeypatch):
    admission = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "allocation_binding_digest": "sha256:" + "a" * 64, "expires_at_epoch": 4e9}
    assert website_task_context.reuse_website_preparation_spend(
        admission=admission, binding_digest="sha256:" + "a" * 64, provider="meta")
    with pytest.raises(website_task_context.RetainedAdmissionUnusable):
        website_task_context.reuse_website_preparation_spend(
            admission=admission, binding_digest="sha256:" + "b" * 64, provider="meta")
    with pytest.raises(website_task_context.RetainedAdmissionUnusable):
        website_task_context.reuse_website_preparation_spend(admission=admission, binding_digest="sha256:" + "a" * 64)
