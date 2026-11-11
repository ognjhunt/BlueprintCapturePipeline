"""A multi-frame request sent inline uses an encoding whose whole request fits a fixed ceiling.

capture-39ecdc97 (2026-11-11): 16 lossless 1920x1080 frames (crf 0, yuv444p, about 13 MB) made a
17,324,561-byte request that the provider refused with 413 after nine like it were answered. An
inline request stores nothing at the provider; only its size had to change. A retained clip is
reused rather than encoded again, so another encoder build never re-buys a binding's answers.
"""
from __future__ import annotations

from io import BytesIO
import json
import time
from urllib.error import HTTPError

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.job_retry_policy import provider_says_retry_later, request_cannot_succeed
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission

CONTEXT = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
           "context_digest": "sha256:" + "c" * 64}


def _mask(size):
    return f'<0f>0<|box;x1=1;y1=0;x2=2;y2=1;w={size};h={size}|><|mask;x=0;y=0;data=2,2,!!!!!&T:u9`|>'


def _answer(size=64):
    return {"status": "completed", "usage": {"video_frames_processed": 50},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": _mask(size)}]}]}


def _frames(tmp_path, frames, size, noise):
    rng = np.random.default_rng(7)
    rows, artifacts = [], []
    for index in range(frames):
        y, x = np.mgrid[0:size, 0:size]
        base = (np.stack([x * 3 + index * 9, y * 2, (x + y) * 1.5], axis=-1) % 256).astype(np.int16)
        pixels = np.clip(base + rng.integers(-noise, noise + 1, base.shape), 0, 255).astype(np.uint8)
        image = tmp_path / f"frame-{size}-{index}.png"
        Image.fromarray(pixels).save(image)
        rows.append({"source_frame_id": f"decoded-{index}", "width": size, "height": size})
        artifacts.append({"source_frame_id": f"decoded-{index}", "path": str(image), "sha256": _sha256_file(image)})
    return rows, artifacts


def _job(tmp_path, monkeypatch, prompts=3, frames=4, size=64):
    """Textured frames (a phone video's detail compresses; lossless keeps all of it)."""
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    rows, artifacts = _frames(tmp_path, frames, size, noise=12)
    texts = [{"prompt_id": f"target-{i}", "output_label": f"target-{i}", "text": f"object {i}"} for i in range(prompts)]
    args = dict(frame_registry=rows, frame_artifacts=artifacts, prompts=texts, output_root=tmp_path / "out",
                admission={}, admission_grant=None, task_context=dict(CONTEXT))
    digest = canonical_digest(sam.request_binding(frame_registry=rows, frame_artifacts=artifacts, prompts=texts))
    return args, digest, args["output_root"] / digest[7:]


def _reservations(monkeypatch):
    holds = []

    def reserve(*, task_context, binding_digest, maximum_cost_usd, request_count, source_video_digest=None):
        assert binding_digest not in [row["allocation_binding_digest"] for row in holds], "already_reserved"
        value = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "task_context_digest": task_context["context_digest"], "allocation_binding_digest": binding_digest,
                 "maximum_cost_usd": maximum_cost_usd, "request_count": request_count,
                 "external_disclosure_allowed": True, "expires_at_epoch": time.time() + 3600}
        holds.append(value)
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
                                                      expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    return holds


class Provider:
    """Answers inline prompts; ``fail`` maps a prompt to the statuses its next dispatches are answered with."""

    def __init__(self, fail=None, size=64):
        self.bodies, self.sent, self.fail, self.size = [], [], {k: list(v) for k, v in (fail or {}).items()}, size

    def __call__(self, request, **_kwargs):
        assert request.full_url == sam.ENDPOINT, "an inline request uploads nothing"
        payload = json.loads(request.data)
        media = payload["input"][0]["content"][1]
        assert media["type"] == "input_video" and media["video_url"].startswith("data:video/mp4;base64,")
        text = payload["input"][0]["content"][0]["text"]
        self.bodies.append(request.data)
        self.sent.append(text)
        if self.fail.get(text):
            raise HTTPError(sam.ENDPOINT, self.fail[text].pop(0), "Answered", {}, None)
        return BytesIO(json.dumps(_answer(self.size)).encode())


def _lossless_body(args, tmp_path):
    root = tmp_path / "lossless"
    root.mkdir(exist_ok=True)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    return sam._inline_body_bytes(clip, args["prompts"])


def _no_encoding(monkeypatch):
    monkeypatch.setattr(sam, "encode_clip", lambda **_: pytest.fail("the retained clip must be reused"))


# --- the size ladder -------------------------------------------------------------------------

def test_a_clip_too_large_lossless_goes_out_under_the_ceiling_at_the_first_rung(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    lossless = _lossless_body(args, tmp_path)
    # The job's proportions: lossless is past the ceiling, as 17.3 MB was past where refusals began.
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", int(lossless * 0.6))
    holds = _reservations(monkeypatch)
    provider = Provider()
    result = sam.run_meta_sam31(**args, opener=provider)
    assert len(provider.bodies) == 3 and max(map(len, provider.bodies)) <= sam.INLINE_REQUEST_CEILING_BYTES < lossless
    record = json.loads((root / sam.CLIP_ENCODING_FILE).read_text())
    assert record["encoding"] == sam.INLINE_CLIP_LADDER[0]["encoding"] == result["clip_encoding"]
    assert json.loads((root / "response-0.json").read_text())["clip_encoding"] == record["encoding"]
    assert len(holds) == 1 and holds[0]["request_count"] == 3  # the per-request reservation is unchanged


def test_large_frames_step_down_the_whole_ladder_past_a_rung_over_the_clip_bound(tmp_path, monkeypatch):
    # Noisy 1080p frames: the first rungs' clips pass the clip bound, which must not stop the ladder.
    rows, artifacts = _frames(tmp_path, 2, 1080, noise=40)
    prompts = [{"prompt_id": "a", "output_label": "a", "text": "chair"}]
    sizes = {}
    for rung in sam.INLINE_CLIP_LADDER:
        root = tmp_path / rung["encoding"]
        root.mkdir()
        sizes[rung["crf"]] = sam.encode_clip(registry=rows, artifacts=artifacts, root=root, rung=rung).stat().st_size
    assert sizes[22] > sizes[34]
    monkeypatch.setattr(sam, "CLIP_MAX_BYTES", (sizes[30] + sizes[26]) // 2)
    root = tmp_path / "ladder"
    root.mkdir()
    clip, record = sam.encode_inline_clip(registry=rows, artifacts=artifacts, root=root, prompts=prompts)
    assert record["encoding"] == "h264_crf30_yuv444p_threads2_v1" and clip.stat().st_size == sizes[30]
    # No rung fits: the typed code, which ends the step at once.
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", 1000)
    with pytest.raises(ValueError, match=sam.INLINE_CLIP_TOO_LARGE):
        sam.encode_inline_clip(registry=rows, artifacts=artifacts, root=root, prompts=prompts)
    assert request_cannot_succeed(sam.INLINE_CLIP_TOO_LARGE)


def test_a_clip_that_fits_at_no_rung_ends_typed_before_anything_is_reserved_or_sent(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", 1000)
    holds = _reservations(monkeypatch)
    with pytest.raises(ValueError, match=sam.INLINE_CLIP_TOO_LARGE):
        sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("nothing may be sent"))
    assert holds == []


def test_the_quality_floor_is_the_last_rung():
    assert sam.INLINE_CLIP_QUALITY_FLOOR == sam.INLINE_CLIP_LADDER[-1]


# --- a retained clip is reused, never encoded again -------------------------------------------

def test_encoder_threads_change_the_bytes_so_a_retained_clip_is_never_re_encoded(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    digests = {}
    for threads in (1, 2, 4):
        root = tmp_path / f"t{threads}"
        root.mkdir()
        rung = {**sam.INLINE_CLIP_LADDER[0], "threads": threads}
        digests[threads] = _sha256_file(sam.encode_clip(registry=args["frame_registry"],
                                                        artifacts=args["frame_artifacts"], root=root, rung=rung))
    again = tmp_path / "t2-again"
    again.mkdir()
    assert _sha256_file(sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"],
                                        root=again, rung=sam.INLINE_CLIP_LADDER[0])) == digests[2]


def test_a_complete_binding_under_another_encoder_setting_sends_and_reserves_nothing(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    holds = _reservations(monkeypatch)
    first = sam.run_meta_sam31(**args, opener=Provider())
    monkeypatch.setattr(sam, "INLINE_CLIP_LADDER", tuple({**rung, "threads": 1} for rung in sam.INLINE_CLIP_LADDER))
    _no_encoding(monkeypatch)
    assert sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("nothing may be sent")) == first
    assert len(holds) == 1


def test_a_partial_binding_with_a_fitting_retained_clip_buys_only_the_missing_prompts(tmp_path, monkeypatch):
    args, digest, root = _job(tmp_path, monkeypatch)
    holds = _reservations(monkeypatch)
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        sam.run_meta_sam31(**args, opener=Provider({"object 1": [413]}))
    kept = (root / "response-0.json").read_bytes()
    monkeypatch.setattr(sam, "INLINE_CLIP_LADDER", tuple({**rung, "threads": 4} for rung in sam.INLINE_CLIP_LADDER))
    _no_encoding(monkeypatch)
    provider = Provider()
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.sent == ["object 1", "object 2"] and (root / "response-0.json").read_bytes() == kept
    assert not (root / sam.SUPERSEDED_CLIP_ARCHIVE).exists()


def test_a_retained_clip_too_large_is_re_encoded_and_re_bought_once(tmp_path, monkeypatch):
    # The stopped job's state: answers bought with the lossless clip, then a 413 on one prompt.
    args, digest, root = _job(tmp_path, monkeypatch, prompts=4)
    root.mkdir(parents=True)
    old_clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    old_digest = _sha256_file(old_clip)
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", int(_lossless_body(args, tmp_path) * 0.6))
    for index in (0, 1):
        sam.write_json(root / f"intent-{index}.json", {"binding_digest": digest, "clip_digest": old_digest,
                                                       "allocation_binding_digest": digest})
        sam.write_json(root / f"response-{index}.json", {"binding_digest": digest, "clip_digest": old_digest,
            "allocation_binding_digest": digest, "response": _answer(), "reserved_cost_usd": 0.01})
    sam.write_json(root / "intent-2.json", {"binding_digest": digest, "clip_digest": old_digest,
                                            "allocation_binding_digest": digest})
    sam.write_json(root / "failure-2.json", {"http_status": 413, "binding_digest": digest})
    holds = _reservations(monkeypatch)
    provider = Provider()
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.sent == ["object 0", "object 1", "object 2", "object 3"]
    assert len(holds) == 1 and holds[0]["request_count"] == 4 and holds[0]["allocation_binding_digest"] != digest
    record = json.loads((root / sam.SUPERSEDED_CLIP_ARCHIVE / "attempt-1/superseded.json").read_text())
    assert record["reason"] == "clip_changed" and record["earlier_clip_digests"] == [old_digest]
    # The third run reuses the new clip and its answers: nothing is encoded, sent or reserved.
    _no_encoding(monkeypatch)
    assert sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("rebought"))["status"] == "completed"
    assert len(holds) == 1


def test_a_caller_with_its_own_grant_gets_a_typed_code_when_a_re_buy_needs_a_new_one(tmp_path, monkeypatch):
    args, digest, root = _job(tmp_path, monkeypatch)
    _reservations(monkeypatch)
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        sam.run_meta_sam31(**args, opener=Provider({"object 1": [413]}))
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", 10)  # the retained clip no longer fits
    monkeypatch.setattr(sam, "encode_inline_clip", lambda **kw: (
        sam.encode_clip(registry=kw["registry"], artifacts=kw["artifacts"], root=kw["root"],
                        rung=sam.INLINE_CLIP_LADDER[-1]), {"encoding": sam.INLINE_CLIP_LADDER[-1]["encoding"]}))
    admission = {**build_paid_lane_admission(resource_class="evaluator_api"), "allocation_binding_digest": digest,
                 "maximum_cost_usd": 1.0, "external_disclosure_allowed": True}
    grant = require_paid_resource_admission(admission, resource_class="evaluator_api",
                                            expected_schema_version="paid_lane_admission.v1")
    with pytest.raises(ValueError, match="meta_sam_caller_grant_covers_another_clip"):
        sam.run_meta_sam31(**{**args, "admission": admission, "admission_grant": grant},
                           opener=lambda *a, **k: pytest.fail("nothing may be sent"))
    assert request_cannot_succeed("meta_sam_caller_grant_covers_another_clip")


def test_the_scale_reference_masks_go_through_one_re_buy(tmp_path, monkeypatch):
    from blueprint_pipeline.atlas_reference_masks import segment_scale_references

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    rows, artifacts = _frames(tmp_path, 2, 64, noise=12)
    frames = [{"frame_id": row["source_frame_id"], "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
               "image_path": art["path"], "image_digest": art["sha256"]} for row, art in zip(rows, artifacts)]
    references = [{"frame_id": "decoded-0", "segmentation_prompt": "door"},
                  {"frame_id": "decoded-1", "segmentation_prompt": "table"}]
    holds = _reservations(monkeypatch)
    arguments = dict(frames=frames, references=references, root=tmp_path / "refs", task_context=dict(CONTEXT),
                     source_digest="sha256:" + "e" * 64, checkpoint=lambda _stage: None)
    provider = Provider({"table": [413]})
    real = sam.run_meta_sam31
    monkeypatch.setattr(sam, "run_meta_sam31", lambda **kw: real(**kw, opener=provider))
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        segment_scale_references(**arguments)
    root = next((tmp_path / "refs").iterdir())
    # The retained clip stops fitting (a lower ceiling, as the lossless clip of the stopped job did):
    # one re-buy of every prompt at the next rung that fits, then none.
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES",
                        sam._inline_body_bytes(root / "retained-frames.mp4", [{"text": "table"}]) - 1)
    assert len(segment_scale_references(**arguments)) == 2
    assert provider.sent == ["door", "table", "door", "table"] and len(holds) == 2
    assert json.loads((root / "tracks.json").read_text())["clip_encoding"] != sam.INLINE_CLIP_LADDER[0]["encoding"]
    segment_scale_references(**arguments)
    assert len(provider.sent) == 4 and len(holds) == 2


# --- answers that mean "try again" -----------------------------------------------------------

@pytest.mark.parametrize("status", [408, 409, 425, 429, 503])
def test_a_try_again_refusal_is_resent_by_the_next_run(tmp_path, monkeypatch, status):
    args, _, root = _job(tmp_path, monkeypatch)
    holds = _reservations(monkeypatch)
    provider = Provider({"object 1": [status]})
    with pytest.raises(ValueError, match=f"meta_sam_http_{status}") as raised:
        sam.run_meta_sam31(**args, opener=provider)
    assert provider_says_retry_later(str(raised.value)) and not request_cannot_succeed(str(raised.value))
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.sent == ["object 0", "object 1", "object 1", "object 2"]
    # It did no work: nothing is counted as spent; the next dispatch has its own hold (no release exists).
    assert not (root / sam.UNANSWERED_ARCHIVE).exists() and len(holds) == 2


@pytest.mark.parametrize("status", [500, 502, 504])
def test_an_answered_error_that_may_have_done_the_work_is_resent_once_under_its_own_hold(tmp_path, monkeypatch, status):
    args, _, root = _job(tmp_path, monkeypatch)
    holds = _reservations(monkeypatch)
    provider = Provider({"object 1": [status]})
    with pytest.raises(ValueError, match=f"meta_sam_http_{status}") as raised:
        sam.run_meta_sam31(**args, opener=provider)
    assert provider_says_retry_later(str(raised.value))
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert provider.sent == ["object 0", "object 1", "object 1", "object 2"]
    counted = json.loads(next((root / sam.UNANSWERED_ARCHIVE).glob("attempt-*/" + sam.COUNTED_AS_SPENT)).read_text())
    assert counted["answered_http_status"] == status and counted["prompt_index"] == 1
    assert len(holds) == 2 and holds[1]["request_count"] == 2
    assert json.loads((root / "response-1.json").read_text())["resend_index"] == 1


def test_a_resend_that_also_fails_ends_the_step_without_waiting_for_a_person(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    holds = _reservations(monkeypatch)
    provider = Provider({"object 1": [502, 502]})
    for _ in range(2):
        with pytest.raises(ValueError, match="meta_sam_http_502"):
            sam.run_meta_sam31(**args, opener=provider)
    with pytest.raises(ValueError, match="meta_sam_submission_requires_reconciliation") as raised:
        sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("never a second resend"))
    assert request_cannot_succeed(str(raised.value)) and len(holds) == 2
