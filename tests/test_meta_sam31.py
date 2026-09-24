from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil

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


def test_each_mask_is_decoded_once_while_preserving_official_validation(monkeypatch):
    import meta_sam_parser
    from meta_sam_parser import _segmentation

    monkeypatch.setattr(sam, "_sam_parser_node", lambda: None)
    decode = meta_sam_parser.decode_mask_to_raster
    calls = []

    def counted(mask):
        calls.append(mask)
        return decode(mask)

    monkeypatch.setattr(meta_sam_parser, "decode_mask_to_raster", counted)
    monkeypatch.setattr(_segmentation, "decode_mask_to_raster", counted)
    assert sam.parse_tracks(response(), prompt=PROMPT, registry=registry())
    assert len(calls) == 1


def test_official_js_parser_matches_python_track_runs(monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is supplied by the production runtime prerequisite")
    monkeypatch.setattr(sam, "_sam_parser_node", lambda: None)
    expected = sam.parse_tracks(response(), prompt=PROMPT, registry=registry())
    monkeypatch.setattr(sam, "_sam_parser_node", lambda: Path(node))
    assert sam.parse_tracks(response(), prompt=PROMPT, registry=registry()) == expected


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
    monkeypatch.setattr(sam, "parse_tracks", lambda *a, **kw: pytest.fail("validated decoded masks must be reused"))
    second = sam.run_meta_sam31(**{**args, "admission_grant": None, "admission": {}}, opener=opener)
    assert first == second and len(calls) == 1
    assert first["tracks"][0]["label"] == "task-object"
    for file in args["output_root"].rglob("*.json"):
        assert "fixture-meta-secret" not in file.read_text()


def test_partial_prompt_resume_reserves_only_missing_response(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    prompts = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(3)]
    args.update(prompts=prompts, admission={}, admission_grant=None,
                task_context={"capture_id": "fixture"})
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=prompts))
    root = args["output_root"] / digest[7:]
    root.mkdir(parents=True)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    for index in (0, 1):
        sam.write_json(root / f"intent-{index}.json", {
            "binding_digest": digest, "clip_digest": _sha256_file(clip)})
        sam.write_json(root / f"response-{index}.json", {
            "binding_digest": digest, "clip_digest": _sha256_file(clip),
            "response": response(), "reserved_cost_usd": 0.0025})
    original_bytes = [(root / f"response-{index}.json").read_bytes() for index in (0, 1)]
    reservations = []

    def reserve(**kwargs):
        reservations.append(kwargs)
        value = {**build_paid_lane_admission(resource_class="evaluator_api"),
                 "allocation_binding_digest": kwargs["binding_digest"],
                 "maximum_cost_usd": kwargs["maximum_cost_usd"],
                 "external_disclosure_allowed": True}
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
            expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    calls = []

    def opener(request, **kwargs):
        calls.append(json.loads(request.data)["input"][0]["content"][0]["text"])
        return BytesIO(json.dumps(response()).encode())

    result = sam.run_meta_sam31(**args, opener=opener)
    assert result["status"] == "completed"
    assert calls == ["object 2"]
    assert len(reservations) == 1
    assert reservations[0]["request_count"] == 1
    assert reservations[0]["maximum_cost_usd"] == 0.0025
    assert reservations[0]["binding_digest"] != digest
    assert json.loads((root / "response-2.json").read_text())["allocation_binding_digest"] == reservations[0]["binding_digest"]
    assert [(root / f"response-{index}.json").read_bytes() for index in (0, 1)] == original_bytes
    assert sam.run_meta_sam31(**{**args, "task_context": None}, opener=lambda *a, **kw: pytest.fail("retained")) == result
    assert len(reservations) == 1


def test_partial_prompt_resume_refuses_uncertain_missing_prompt_before_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    prompts = [{**PROMPT, "prompt_id": f"target-{i}"} for i in range(2)]
    args.update(prompts=prompts, admission={}, admission_grant=None, task_context={"capture_id": "fixture"})
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=prompts))
    root = args["output_root"] / digest[7:]
    root.mkdir(parents=True)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    sam.write_json(root / "response-0.json", {"binding_digest": digest,
        "clip_digest": _sha256_file(clip), "response": response()})
    sam.write_json(root / "intent-1.json", {"binding_digest": digest, "clip_digest": _sha256_file(clip)})
    monkeypatch.setattr("blueprint_pipeline.website_task_context.reserve_website_sam_spend",
        lambda **kwargs: pytest.fail("uncertain provider call must not get another grant"))
    with pytest.raises(ValueError, match="requires_reconciliation"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("uncertain provider call"))
    (root / "intent-1.json").unlink()
    saved = json.loads((root / "response-0.json").read_text())
    saved["clip_digest"] = "sha256:changed"
    sam.write_json(root / "response-0.json", saved)
    with pytest.raises(ValueError, match="retained_response_changed"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("tampered provider response"))


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


def test_continuous_video_keeps_original_frames_and_timestamps(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    rows = [{**args["frame_registry"][0], "source_frame_id": f"f{i}"} for i in range(3)]
    artifacts = [{**args["frame_artifacts"][0], "source_frame_id": row["source_frame_id"]} for row in rows]
    source = sam.encode_clip(registry=rows, artifacts=artifacts, root=tmp_path)
    source_digest = _sha256_file(source)
    registry, video = sam.prepare_continuous_video(source=source, source_digest=source_digest, root=tmp_path / "continuous")
    assert [row["decoded_pts_seconds"] for row in registry] == [0, 1, 2]
    assert [row["source_frame_id"] for row in registry] == ["decoded-000000000", "decoded-000000001", "decoded-000000002"]
    args.update(frame_registry=registry, frame_artifacts=[], video_artifact=video)
    digest = canonical_digest(sam.request_binding(frame_registry=registry, frame_artifacts=[], prompts=[PROMPT], video_artifact=video))
    args["admission"]["allocation_binding_digest"] = digest
    args["admission_grant"] = require_paid_resource_admission(args["admission"], resource_class="evaluator_api",
        expected_schema_version="paid_lane_admission.v1")
    calls = []
    def opener(request, **kwargs):
        calls.append(request)
        if request.full_url == "https://api.meta.ai/v1/files":
            assert b'name="purpose"\r\n\r\nuser_data' in request.data
            return BytesIO(b'{"id":"file-fixture"}')
        if request.get_method() == "DELETE":
            assert request.full_url == "https://api.meta.ai/v1/files/file-fixture"
            return BytesIO(b'{"deleted":true}')
        assert json.loads(request.data)["input"][0]["content"][1]["type"] == "input_video"
        assert json.loads(request.data)["input"][0]["content"][1]["file_id"] == "file-fixture"
        text = response()["output"][0]["content"][0]["text"].replace("<0f>", "<2f>")
        return BytesIO(json.dumps(response(text)).encode())
    result = sam.run_meta_sam31(**args, opener=opener)
    assert result["tracks"][0]["observations"][0]["source_frame_id"] == "decoded-000000002"
    assert sam.run_meta_sam31(**{**args, "admission_grant": None, "admission": {}}, opener=opener) == result
    assert len(calls) == 3
    Path(video["path"]).write_bytes(b"changed encoded video")
    with pytest.raises(ValueError, match="continuous_video_invalid"):
        sam.run_meta_sam31(**args, opener=opener)
    with pytest.raises(ValueError, match="source_video_changed"):
        sam.prepare_continuous_video(source=source, source_digest="sha256:wrong", root=tmp_path / "bad")


def test_completed_continuous_video_reuses_cpu_work_and_rejects_tampering(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    rows = [{**args["frame_registry"][0], "source_frame_id": f"f{i}"} for i in range(3)]
    artifacts = [{**args["frame_artifacts"][0], "source_frame_id": row["source_frame_id"]} for row in rows]
    source = sam.encode_clip(registry=rows, artifacts=artifacts, root=tmp_path)
    kwargs = dict(source=source, source_digest=_sha256_file(source), root=tmp_path / "continuous")
    result = sam.prepare_continuous_video(**kwargs)
    monkeypatch.setattr(sam.subprocess, "run", lambda *a, **k: pytest.fail("completed encode must be reused"))
    assert sam.prepare_continuous_video(**kwargs) == result
    Path(result[1]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared_video_receipt_invalid"):
        sam.prepare_continuous_video(**kwargs)


def test_continuous_files_api_clip_can_exceed_old_inline_limit(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    probe = {"streams": [{"width": 1920, "height": 1080}],
             "frames": [{"best_effort_timestamp_time": str(i)} for i in range(2)]}
    monkeypatch.setattr(sam, "_probe_video", lambda _: probe)

    def encode(argv, **_kwargs):
        with Path(argv[-1]).open("wb") as output:
            output.truncate(33 * 1024**2)

    monkeypatch.setattr(sam.subprocess, "run", encode)
    rows, video = sam.prepare_continuous_video(
        source=source, source_digest=_sha256_file(source), root=tmp_path / "continuous")
    assert len(rows) == 2
    assert Path(video["path"]).stat().st_size == 33 * 1024**2
    assert video["encoding"] == "upright_h264_crf18_veryfast_threads2_all_source_frames_v2"


def test_failed_encoder_cannot_publish_partial_video(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    monkeypatch.setattr(sam, "_probe_video", lambda _: {"frames": [{}, {}]})
    def timeout(argv, **kwargs):
        assert argv[argv.index("-preset") + 1] == "veryfast"
        assert argv[argv.index("-threads") + 1] == "2"
        Path(argv[-1]).write_bytes(b"partial")
        raise sam.subprocess.TimeoutExpired(argv, 120)
    monkeypatch.setattr(sam.subprocess, "run", timeout)
    root = tmp_path / "output"
    with pytest.raises(sam.subprocess.TimeoutExpired):
        sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=root)
    assert not (root / "continuous-upright.mp4").exists()
    assert not (root / "continuous-video.json").exists()
    assert not list(root.glob("*.mp4"))


def test_decoded_mask_cache_is_bound_to_retained_provider_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture")
    args = inputs(tmp_path)
    sam.run_meta_sam31(**args, opener=lambda *a, **kw: BytesIO(json.dumps(response()).encode()))
    cache_path = next(args["output_root"].rglob("parsed-response-0.json"))
    cache = json.loads(cache_path.read_text())
    cache["tracks"][0]["label"] = "substituted"
    cache_path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="retained_tracks_changed"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("must not purchase a retry"))
