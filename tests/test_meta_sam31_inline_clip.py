"""A multi-frame request sent inline uses an encoding whose whole request fits a fixed ceiling.

capture-39ecdc97 (2026-11-11): 16 lossless 1920x1080 frames (crf 0, yuv444p, about 13 MB) made a
17,324,561-byte request that the provider refused with 413 after nine like it were answered. An
inline request stores nothing at the provider; only its size had to change.
"""
from __future__ import annotations

from io import BytesIO
import json
import time

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission

MASK = '<0f>0<|box;x1=1;y1=0;x2=2;y2=1;w=64;h=64|><|mask;x=0;y=0;data=2,2,!!!!!&T:u9`|>'
CONTEXT = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
           "context_digest": "sha256:" + "c" * 64}


def _answer():
    return {"status": "completed", "usage": {"video_frames_processed": 50},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": MASK}]}]}


def _job(tmp_path, monkeypatch, prompts=3, frames=4, size=64):
    """Textured frames (a phone video's detail compresses; lossless keeps all of it)."""
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    rng = np.random.default_rng(7)
    rows, artifacts = [], []
    for index in range(frames):
        y, x = np.mgrid[0:size, 0:size]
        base = (np.stack([x * 3 + index * 9, y * 2, (x + y) * 1.5], axis=-1) % 256).astype(np.int16)
        pixels = np.clip(base + rng.integers(-12, 13, base.shape), 0, 255).astype(np.uint8)
        image = tmp_path / f"frame-{index}.png"
        Image.fromarray(pixels).save(image)
        rows.append({"source_frame_id": f"decoded-{index}", "width": size, "height": size})
        artifacts.append({"source_frame_id": f"decoded-{index}", "path": str(image), "sha256": _sha256_file(image)})
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
    def __init__(self):
        self.bodies = []

    def __call__(self, request, **_kwargs):
        assert request.full_url == sam.ENDPOINT, "an inline request uploads nothing"
        self.bodies.append(request.data)
        media = json.loads(request.data)["input"][0]["content"][1]
        assert media["type"] == "input_video" and media["video_url"].startswith("data:video/mp4;base64,")
        return BytesIO(json.dumps(_answer()).encode())


def _lossless_body(args, tmp_path):
    root = tmp_path / "lossless"
    root.mkdir()
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    return sam._inline_body_bytes(clip, args["prompts"])


def test_a_clip_too_large_lossless_goes_out_under_the_ceiling_at_the_first_rung(tmp_path, monkeypatch):
    args, _, root = _job(tmp_path, monkeypatch)
    lossless = _lossless_body(args, tmp_path)
    # The job's proportions: lossless is past the ceiling, as 17.3 MB was past where refusals began.
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", int(lossless * 0.6))
    holds = _reservations(monkeypatch)
    provider = Provider()
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert len(provider.bodies) == 3 and max(map(len, provider.bodies)) <= sam.INLINE_REQUEST_CEILING_BYTES < lossless
    record = json.loads((root / sam.CLIP_ENCODING_FILE).read_text())
    assert record["encoding"] == sam.INLINE_CLIP_LADDER[0]["encoding"]
    assert record["largest_request_bytes"] == max(map(len, provider.bodies))
    assert json.loads((root / "response-0.json").read_text())["clip_encoding"] == record["encoding"]
    # The per-request reservation is unchanged.
    assert len(holds) == 1 and holds[0]["request_count"] == 3


def test_the_inline_encoding_is_the_same_from_run_to_run(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    digests = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        clip, _ = sam.encode_inline_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"],
                                         root=root, prompts=args["prompts"])
        digests.append(_sha256_file(clip))
    assert digests[0] == digests[1]


def test_a_clip_that_fits_at_no_rung_ends_typed_before_anything_is_reserved_or_sent(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", 1000)
    holds = _reservations(monkeypatch)
    with pytest.raises(ValueError, match=sam.INLINE_CLIP_TOO_LARGE):
        sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("nothing may be sent"))
    assert holds == []


def test_answers_bought_with_another_clip_are_set_aside_and_every_prompt_is_bought_again(tmp_path, monkeypatch):
    # The stopped job's state: answers bought with the lossless clip, then a 413 on one prompt.
    args, digest, root = _job(tmp_path, monkeypatch, prompts=4)
    root.mkdir(parents=True)
    old_clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    old_digest = _sha256_file(old_clip)
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
    assert len(provider.bodies) == 4
    # One new hold for all four, under a digest the first reservation never used.
    assert len(holds) == 1 and holds[0]["request_count"] == 4 and holds[0]["allocation_binding_digest"] != digest
    record = json.loads((root / sam.SUPERSEDED_CLIP_ARCHIVE / "attempt-1/superseded.json").read_text())
    assert record["reason"] == "clip_changed" and record["earlier_clip_digests"] == [old_digest]
    assert record["prompt_indices"] == [0, 1]
    new_digest = json.loads((root / sam.CLIP_ENCODING_FILE).read_text())["clip_digest"]
    assert all(json.loads((root / f"response-{i}.json").read_text())["clip_digest"] == new_digest for i in range(4))
    # A later pass reuses the new answers: nothing more is bought.
    assert sam.run_meta_sam31(**{**args, "task_context": None}, opener=lambda *a, **k: pytest.fail("rebought"))
    assert len(holds) == 1
