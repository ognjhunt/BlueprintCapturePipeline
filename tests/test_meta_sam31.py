from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.paid_resource_admission import build_paid_lane_admission, require_paid_resource_admission
from blueprint_pipeline.website_task_masks import decode_track_mask


def response(text=None, status="completed"):
    if text is None:
        text = '<0f>0<|box;x1=1;y1=0;x2=2;y2=1;w=4;h=4|><|mask;x=0;y=0;data=2,2,!!!!!&T:u9`|>'
    return {"status": status, "usage": {"video_frames_processed": 1}, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def registry():
    return [{"source_frame_id": "original-at-3.2s", "width": 4, "height": 4}]


PROMPT = {"prompt_id": "blue-container", "output_label": "task-object", "text": "blue container"}


def test_official_decoder_places_local_mask_in_source_image_without_inventing_confidence():
    track = sam.parse_tracks(response(), prompt=PROMPT, registry=registry())[0]
    obs = track["observations"][0]
    assert obs["source_frame_id"] == "original-at-3.2s"
    assert obs["runs"] == [{"start": 1, "length": 1}, {"start": 6, "length": 1}]
    assert decode_track_mask(obs).sum() == 2
    assert "confidence" not in track


@pytest.mark.parametrize("text,error", [
    (response()["output"][0]["content"][0]["text"].replace("<0f>", "<2f>"), "frame_index"),
    (response()["output"][0]["content"][0]["text"].replace("w=4", "w=5"), "dimensions_mismatch"),
    ("malformed result", "malformed"),
    (response()["output"][0]["content"][0]["text"].replace("x2=2", "x2=5"), "malformed"),
])
def test_invalid_provider_output_cannot_become_a_task_mask(text, error):
    with pytest.raises(ValueError, match=error):
        sam.parse_tracks(response(text), prompt=PROMPT, registry=registry())


def test_empty_is_no_match_and_incomplete_is_not_success():
    assert sam.parse_tracks(response(""), prompt=PROMPT, registry=registry()) == []
    with pytest.raises(ValueError, match="incomplete"):
        sam.parse_tracks(response(status="incomplete"), prompt=PROMPT, registry=registry())


def inputs(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "blue").save(image)
    rows = registry()
    artifacts = [{"source_frame_id": rows[0]["source_frame_id"], "path": str(image), "sha256": _sha256_file(image)}]
    digest = canonical_digest(sam.request_binding(frame_registry=rows, frame_artifacts=artifacts, prompts=[PROMPT]))
    authority = {**build_paid_lane_admission(resource_class="evaluator_api"), "allocation_binding_digest": digest,
                 "maximum_cost_usd": 0.01, "external_disclosure_allowed": True}
    grant = require_paid_resource_admission(authority, resource_class="evaluator_api", expected_schema_version="paid_lane_admission.v1")
    return dict(frame_registry=rows, frame_artifacts=artifacts, prompts=[PROMPT], output_root=tmp_path / "out",
                admission=authority, admission_grant=grant)


def test_real_cpu_encoding_preserves_one_frame_per_source(tmp_path):
    args = inputs(tmp_path)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=tmp_path)
    assert clip.stat().st_size > 0
    Path(args["frame_artifacts"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source_frame_changed"):
        sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=tmp_path)


def test_exact_request_is_retained_and_replay_does_not_charge_again(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    calls = []
    def opener(request, **kwargs):
        calls.append(request)
        payload = json.loads(request.data)
        assert request.full_url == sam.ENDPOINT
        assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        return BytesIO(json.dumps(response()).encode())
    args = inputs(tmp_path)
    first = sam.run_meta_sam31(**args, opener=opener)
    second = sam.run_meta_sam31(**{**args, "admission_grant": None, "admission": {}}, opener=opener)
    assert first == second and len(calls) == 1
    assert first["tracks"][0]["label"] == "task-object"
    for file in args["output_root"].rglob("*.json"):
        assert "fixture-meta-secret" not in file.read_text()


def test_timeout_is_not_retried_and_missing_grant_cannot_call_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        raise TimeoutError()
    args = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="grant_missing"):
        sam.run_meta_sam31(**{**args, "admission_grant": None}, opener=opener)
    assert calls == []
    with pytest.raises(TimeoutError):
        sam.run_meta_sam31(**args, opener=opener)
    with pytest.raises(ValueError, match="requires_reconciliation"):
        sam.run_meta_sam31(**args, opener=opener)
    assert len(calls) == 1


def test_per_frame_budget_is_checked_before_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    args["admission"]["maximum_cost_usd"] = 0.00001
    with pytest.raises(ValueError, match="authorization_missing"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("unauthorized call"))


def test_video_uses_video_payload_and_blocks_usage_beyond_reserved_frames(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    args["frame_registry"].append({**args["frame_registry"][0], "source_frame_id": "second"})
    args["frame_artifacts"].append({**args["frame_artifacts"][0], "source_frame_id": "second"})
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=args["prompts"]))
    args["admission"]["allocation_binding_digest"] = digest
    args["admission_grant"] = require_paid_resource_admission(args["admission"], resource_class="evaluator_api",
        expected_schema_version="paid_lane_admission.v1")
    def opener(request, **kwargs):
        media = json.loads(request.data)["input"][0]["content"][1]
        assert media["type"] == "input_video"
        result = response()
        result["usage"]["video_frames_processed"] = 51
        return BytesIO(json.dumps(result).encode())
    with pytest.raises(ValueError, match="usage_exceeds_reservation"):
        sam.run_meta_sam31(**args, opener=opener)
