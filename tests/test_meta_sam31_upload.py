"""A multi-frame SAM request sends its clip as one uploaded file, deleted however the run ends.

capture-39ecdc97 (2026-11-11): the Atlas background step sent 16 lossless frames inline as a
base64 data URL (about 16.5 MiB per request, once per prompt). Nine prompts were answered and the
tenth was refused with 413. These tests hold the upload transport, its cleanup on every ending,
the resume from the inline run's retained answers, and the money rules: one hold per binding,
reused while the website still honours it and the capture's source is current, replaced (never
stacked) once it is not.
"""
from __future__ import annotations

import fcntl
from io import BytesIO
import json
import time
from urllib.error import HTTPError

import pytest
from PIL import Image

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission

MASK = '<0f>0<|box;x1=1;y1=0;x2=2;y2=1;w=4;h=4|><|mask;x=0;y=0;data=2,2,!!!!!&T:u9`|>'
CONTEXT = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
           "context_digest": "sha256:" + "c" * 64}


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
                admission={}, admission_grant=None, task_context=dict(CONTEXT))
    digest = canonical_digest(sam.request_binding(frame_registry=rows, frame_artifacts=artifacts, prompts=texts))
    return args, digest, args["output_root"] / digest[7:]


class Website:
    """The website's reservation transaction and its task-context read, as each answers."""

    def __init__(self, monkeypatch):
        self.holds, self.reads, self.withdrawn, self.context = [], 0, False, dict(CONTEXT)
        monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", self.reserve)
        monkeypatch.setattr(website_task_context, "load_current_website_task_context", self.read)

    def reserve(self, *, task_context, binding_digest, maximum_cost_usd, request_count, source_video_digest=None):
        # One reservation per binding digest: a second ask is "already_reserved", never a new hold.
        assert binding_digest not in [row["allocation_binding_digest"] for row in self.holds], "already_reserved"
        value = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "task_context_digest": task_context["context_digest"], "allocation_binding_digest": binding_digest,
                 "maximum_cost_usd": maximum_cost_usd, "request_count": request_count,
                 "external_disclosure_allowed": True, "expires_at_epoch": time.time() + 3600}
        if source_video_digest is not None:
            value["source_video_digest"] = source_video_digest
        self.holds.append(value)
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
                                                      expected_schema_version="paid_lane_admission.v1")

    def read(self, *, request_id, scene_id, capture_id, purpose=None):
        self.reads += 1
        if self.withdrawn:
            raise ValueError("website_control_task-context_http_409:source_revoked")
        return dict(self.context)


def _unused_live_holds(website, root):
    """Holds the website still counts that have requests left unused and have not expired."""
    live = []
    for hold in website.holds:
        used = sum(json.loads(path.read_text()).get("allocation_binding_digest") == hold["allocation_binding_digest"]
                   for path in root.rglob("intent-*.json") if root / "refused" not in path.parents)
        if hold["request_count"] - used > 0 and hold["expires_at_epoch"] > time.time():
            live.append(hold["allocation_binding_digest"])
    return live


class Provider:
    """Answers uploads, deletions and prompts; records each call."""

    def __init__(self, *, refuse=None, refuse_upload=None, refuse_delete=False):
        self.calls, self.uploads, self.deleted = [], 0, []
        self.refuse, self.refuse_upload, self.refuse_delete = dict(refuse or {}), refuse_upload, refuse_delete

    def __call__(self, request, **_kwargs):
        if request.full_url == sam.FILES_ENDPOINT:
            self.calls.append("upload")
            answer, self.refuse_upload = self.refuse_upload, None
            if isinstance(answer, BaseException):
                raise answer
            if isinstance(answer, int):
                raise HTTPError(sam.FILES_ENDPOINT, answer, "Refused", {}, None)
            if isinstance(answer, bytes):
                return BytesIO(answer)
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
        if text in self.refuse:
            raise HTTPError(sam.ENDPOINT, self.refuse.pop(text), "Refused", {}, None)
        return BytesIO(json.dumps(_answer()).encode())


def test_every_prompt_names_one_uploaded_clip_and_the_upload_is_deleted(tmp_path, monkeypatch):
    args, digest, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    provider = Provider()
    result = sam.run_meta_sam31(**args, opener=provider)
    assert result["status"] == "completed"
    assert provider.calls == ["upload", ("object 0", "file-clip1"), ("object 1", "file-clip1"),
                              ("object 2", "file-clip1"), "delete"]
    assert json.loads((root / sam.UPLOAD_CLEANUP).read_text()) == {"file_id": "file-clip1", "deleted": True}
    # The transport is not part of the bought request: the binding is the one inline runs used.
    assert canonical_digest(json.loads((root / "binding.json").read_text())) == digest
    # The reservation per request is unchanged: one hold for the three prompts at the per-request price.
    assert len(website.holds) == 1 and website.holds[0]["request_count"] == 3
    assert website.holds[0]["maximum_cost_usd"] == pytest.approx(3 * 50 * sam.PRICE_PER_FRAME_USD)
    replay = Provider()
    assert sam.run_meta_sam31(**{**args, "task_context": None}, opener=replay) == result
    assert replay.calls == []


def test_an_inline_run_refused_413_resumes_and_buys_only_its_missing_prompts(tmp_path, monkeypatch):
    # The reference job's state: prompts answered under the inline transport, then one refused 413.
    args, digest, root = _job(tmp_path, monkeypatch, prompts=4)
    root.mkdir(parents=True)
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
    website = Website(monkeypatch)
    provider = Provider()
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.calls == ["upload", ("object 2", "file-clip1"), ("object 3", "file-clip1"), "delete"]
    assert [(root / f"response-{index}.json").read_bytes() for index in (0, 1)] == kept
    assert len(website.holds) == 1 and website.holds[0]["request_count"] == 2
    assert website.holds[0]["maximum_cost_usd"] == pytest.approx(2 * 50 * sam.PRICE_PER_FRAME_USD)


def test_a_refused_prompt_retried_later_is_paid_by_the_same_hold_and_uploads_afresh(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    provider = Provider(refuse={"object 1": 413})
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls[-1] == "delete" and provider.deleted == ["file-clip1"]
    assert _unused_live_holds(website, root) == [website.holds[0]["allocation_binding_digest"]]
    reads = website.reads
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    # The provider did no work for the refused prompt: its unused requests pay the retry, no second hold.
    assert len(website.holds) == 1 and website.reads == reads + 1
    assert provider.calls[-4:] == ["upload", ("object 1", "file-clip2"), ("object 2", "file-clip2"), "delete"]
    assert provider.deleted == ["file-clip1", "file-clip2"]
    assert _unused_live_holds(website, root) == []


def test_an_expired_hold_is_replaced_once_under_a_new_digest_and_never_jams_the_job(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    with pytest.raises(ValueError, match="meta_sam_upload_http_413"):
        sam.run_meta_sam31(**args, opener=Provider(refuse_upload=413))
    record_path = next((root / sam.ADMISSION_ARCHIVE).glob("*.json"))
    record = json.loads(record_path.read_text())
    record["admission"]["expires_at_epoch"] = time.time() - 1  # the website's window for that hold has passed
    record_path.write_text(json.dumps(record))
    website.holds[0]["expires_at_epoch"] = time.time() - 1
    provider = Provider()
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert json.loads(record_path.read_text())["status"] == "superseded"
    assert len(website.holds) == 2
    assert website.holds[1]["allocation_binding_digest"] != website.holds[0]["allocation_binding_digest"]
    assert _unused_live_holds(website, root) == []
    # A later pass finds everything answered: nothing more is reserved.
    assert sam.run_meta_sam31(**args, opener=Provider()) and len(website.holds) == 2


def test_a_hold_for_another_task_context_is_not_reused(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    with pytest.raises(ValueError):
        sam.run_meta_sam31(**args, opener=Provider(refuse_upload=413))
    changed = {**CONTEXT, "context_digest": "sha256:" + "d" * 64}
    website.context = changed
    assert sam.run_meta_sam31(**{**args, "task_context": changed}, opener=Provider())["status"] == "completed"
    assert len(website.holds) == 2 and website.holds[1]["task_context_digest"] == changed["context_digest"]


def test_a_withdrawn_source_stops_a_reused_hold_before_any_frame_leaves(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    with pytest.raises(ValueError):
        sam.run_meta_sam31(**args, opener=Provider(refuse_upload=413))
    website.withdrawn = True
    provider = Provider()
    with pytest.raises(ValueError, match="source_revoked"):
        sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == [] and len(website.holds) == 1


@pytest.mark.parametrize("failure,raised", [
    (TimeoutError("timed out"), "timed out"),
    (500, "meta_sam_upload_http_500"), (502, "meta_sam_upload_http_502"), (504, "meta_sam_upload_http_504"),
    (b'{"id":"weird id"}', "meta_sam_upload_unusable_answer_http_200"),
])
def test_an_upload_with_no_clear_answer_is_recorded_as_a_possible_copy_and_the_job_goes_on(
        tmp_path, monkeypatch, failure, raised):
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as provider_copies

    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    with pytest.raises((OSError, ValueError), match=raised):
        sam.run_meta_sam31(**args, opener=Provider(refuse_upload=failure))
    provider = Provider()
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.calls[0] == "upload" and provider.deleted == ["file-clip1"]
    # The first upload's possible copy has no id to delete by: it is kept for deletion by hand.
    assert [path.parent.name for path in sam.possible_provider_copies(root)] == ["attempt-1"]
    _, unknown = provider_copies._discover(tmp_path)
    assert len(unknown) == 1 and unknown[0]["provider"] == "meta"
    # The upload bought nothing: its hold paid the retry, never a second one.
    assert len(website.holds) == 1 and _unused_live_holds(website, root) == []


def test_a_refused_upload_leaves_no_copy_and_no_second_hold(tmp_path, monkeypatch):
    from blueprint_pipeline import website_capture_withdrawal_provider_copies as provider_copies

    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    with pytest.raises(ValueError, match="meta_sam_upload_http_413"):
        sam.run_meta_sam31(**args, opener=Provider(refuse_upload=413))
    assert not list(root.glob("intent-*.json"))
    assert sam.run_meta_sam31(**args, opener=Provider())["status"] == "completed"
    assert provider_copies._discover(tmp_path)[1] == [] and sam.possible_provider_copies(root) == []
    assert len(website.holds) == 1


def test_a_failed_delete_is_recorded_and_finished_by_the_next_run_even_a_replay(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    Website(monkeypatch)
    result = sam.run_meta_sam31(**args, opener=Provider(refuse_delete=True))
    assert json.loads((root / sam.UPLOAD_CLEANUP).read_text()) == {"file_id": "file-clip1", "deleted": False}
    (root / sam.UPLOAD_CLEANUP).unlink()  # a process killed before its deletion leaves no record at all
    later = Provider()
    assert sam.run_meta_sam31(**{**args, "task_context": None}, opener=later) == result
    assert later.calls == ["delete"] and later.deleted == ["file-clip1"]


def test_a_crash_after_the_upload_resumes_with_that_upload_and_then_deletes_it(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch, prompts=1)
    root.mkdir(parents=True)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    sam.write_json(root / sam.UPLOAD_INTENT, {"clip_digest": _sha256_file(clip)})
    sam.write_json(root / sam.UPLOAD_RECEIPT, {"file_id": "file-earlier", "clip_digest": _sha256_file(clip)})
    Website(monkeypatch)
    provider = Provider()
    sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == [("object 0", "file-earlier"), "delete"] and provider.deleted == ["file-earlier"]


def test_a_second_run_on_the_same_binding_waits_and_never_deletes_the_first_runs_upload(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    Website(monkeypatch)
    root.mkdir(parents=True)
    sam.write_json(root / sam.UPLOAD_RECEIPT, {"file_id": "file-in-use", "clip_digest": "sha256:" + "a" * 64})
    with (root / sam.REQUEST_LOCK).open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provider = Provider()
        with pytest.raises(BlockingIOError):
            sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == []


def test_a_continuous_video_upload_is_deleted_when_its_prompt_is_refused(tmp_path, monkeypatch):
    # The clean-plate and robot-access path (``video_artifact``) uses the same wrapper.
    args, _, _ = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    rows = [{**args["frame_registry"][0], "source_frame_id": f"f{i}"} for i in range(3)]
    artifacts = [{**args["frame_artifacts"][0], "source_frame_id": row["source_frame_id"]} for row in rows]
    source = sam.encode_clip(registry=rows, artifacts=artifacts, root=tmp_path)
    registry, video = sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source),
                                                   root=tmp_path / "continuous")
    provider = Provider(refuse={"object 0": 413})
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        sam.run_meta_sam31(**{**args, "frame_registry": registry, "frame_artifacts": [], "video_artifact": video},
                           opener=provider)
    assert provider.calls == ["upload", ("object 0", "file-clip1"), "delete"]
    assert provider.deleted == ["file-clip1"] and len(website.holds) == 1


def test_a_retained_meta_reservation_is_reusable_only_for_its_own_binding():
    admission = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "allocation_binding_digest": "sha256:" + "a" * 64, "expires_at_epoch": time.time() + 600}
    assert website_task_context.reuse_website_preparation_spend(
        admission=admission, binding_digest="sha256:" + "a" * 64, provider="meta")
    with pytest.raises(website_task_context.RetainedAdmissionUnusable):
        website_task_context.reuse_website_preparation_spend(
            admission=admission, binding_digest="sha256:" + "b" * 64, provider="meta")
    with pytest.raises(website_task_context.RetainedAdmissionUnusable):
        website_task_context.reuse_website_preparation_spend(
            admission={**admission, "expires_at_epoch": time.time() - 1}, binding_digest="sha256:" + "a" * 64,
            provider="meta")


class GoneUpload(Provider):
    """Answers a prompt 404 while its file id is in ``gone``: the provider no longer holds that upload."""

    def __init__(self, gone):
        super().__init__()
        self.gone = set(gone)

    def __call__(self, request, **kwargs):
        if request.full_url == sam.ENDPOINT:
            file_id = json.loads(request.data)["input"][0]["content"][1]["file_id"]
            if file_id in self.gone:
                text = json.loads(request.data)["input"][0]["content"][0]["text"]
                self.calls.append((text, file_id))
                raise HTTPError(sam.ENDPOINT, 404, "Not Found", {}, None)
        return super().__call__(request, **kwargs)


def test_a_404_on_a_dispatch_uploads_again_and_resends_once_under_the_same_hold(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    provider = GoneUpload({"file-clip1"})
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.calls == ["upload", ("object 0", "file-clip1"), "upload", ("object 0", "file-clip2"),
                              ("object 1", "file-clip2"), ("object 2", "file-clip2"), "delete"]
    # The vanished file is recorded gone, never deleted by id; the second upload is deleted.
    assert provider.deleted == ["file-clip2"]
    assert len(website.holds) == 1 and _unused_live_holds(website, root) == []


def test_a_second_404_in_one_run_ends_the_step_with_its_code(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    website = Website(monkeypatch)
    provider = GoneUpload({"file-clip1", "file-clip2"})
    with pytest.raises(ValueError, match="meta_sam_http_404"):
        sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == ["upload", ("object 0", "file-clip1"), "upload", ("object 0", "file-clip2"), "delete"]
    assert len(website.holds) == 1


@pytest.mark.parametrize("refusal", [400, 403, 413, 422])
def test_any_other_refusal_of_a_dispatch_is_not_resent(tmp_path, monkeypatch, refusal):
    args, _, _ = _job(tmp_path, monkeypatch)
    Website(monkeypatch)
    provider = Provider(refuse={"object 0": refusal})
    with pytest.raises(ValueError, match=f"meta_sam_http_{refusal}"):
        sam.run_meta_sam31(**args, opener=provider)
    assert provider.calls == ["upload", ("object 0", "file-clip1"), "delete"]


def test_an_undeleted_upload_reopens_a_finished_preview_cleanup(tmp_path, monkeypatch):
    # The clean-plate masks of a capture whose preview job ended and cleaned up long ago.
    capture = tmp_path / "scenes/site-req1/captures/walkthrough-req1"
    job = capture / "pipeline/atlas/preview_job.json"
    job.parent.mkdir(parents=True)
    job.write_text("{}")
    args, _, root = _job(tmp_path, monkeypatch)
    args["output_root"] = capture / "pipeline/clean_plate/task_masks/x"
    Website(monkeypatch)
    sam.run_meta_sam31(**args, opener=Provider(refuse_delete=True))
    request = job.parent / sam.CLEANUP_REQUEST
    assert json.loads(request.read_text())["upload_root"].startswith(str(args["output_root"]))
    # A capture with no preview job has no such loop: the request is not left anywhere.
    assert sam.request_capture_cleanup(tmp_path / "elsewhere/pipeline/clean_plate/y") is None
