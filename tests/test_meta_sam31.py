from __future__ import annotations

from io import BytesIO
import json
import time
import math
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


def _two_answered_one_pending(tmp_path, monkeypatch):
    """Prompts 0 and 1 answered and kept, prompt 2 not yet sent: the shape the owner's Austin job stopped in."""
    from blueprint_pipeline import website_task_context

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    prompts = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(3)]
    args.update(prompts=prompts, admission={}, admission_grant=None,
                task_context={"capture_id": "fixture", "request_id": "itest-austin"})
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=prompts))
    root = args["output_root"] / digest[7:]
    root.mkdir(parents=True)
    clip = sam.encode_clip(registry=args["frame_registry"], artifacts=args["frame_artifacts"], root=root)
    for index in (0, 1):
        sam.write_json(root / f"intent-{index}.json", {"binding_digest": digest, "clip_digest": _sha256_file(clip)})
        sam.write_json(root / f"response-{index}.json", {"binding_digest": digest, "clip_digest": _sha256_file(clip),
                                                         "response": response(), "reserved_cost_usd": 0.0025})
    reservations, sources = [], []

    def reserve(**kwargs):
        if any(row["binding_digest"] == kwargs["binding_digest"] for row in reservations):
            pytest.fail("already_reserved: the WebApp lets only the first transaction dispatch")
        reservations.append(kwargs)
        value = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "allocation_binding_digest": kwargs["binding_digest"], "maximum_cost_usd": kwargs["maximum_cost_usd"],
                 "external_disclosure_allowed": True, "expires_at_epoch": time.time() + 3600}
        kwargs["retain_admission"](value)   # as the real reservation does, before its local recheck
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
            expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    monkeypatch.setattr(website_task_context, "require_current_website_source", sources.append)
    return args, root, reservations, sources


def test_a_hold_reserved_before_a_run_stopped_pays_the_next_run_and_is_never_reserved_twice(tmp_path, monkeypatch):
    """Stress round 2 (2026-11-24): the owner's Austin job reserved its pending prompt, then timed out reading a kept
    answer before sending it; every later run was answered ``already_reserved`` and refused. The hold is kept the
    moment it is granted, and a run that sent nothing under it reuses it after reading the site's current source."""
    import subprocess

    args, root, reservations, sources = _two_answered_one_pending(tmp_path, monkeypatch)
    real_parse = sam.parse_tracks
    monkeypatch.setattr(sam, "parse_tracks", lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("node", 379)))
    with pytest.raises(subprocess.TimeoutExpired):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("nothing is sent before the kept answers read"))
    assert len(reservations) == 1 and not (root / "intent-2.json").exists()
    held = json.loads((root / f"held-reservation-{reservations[0]['binding_digest'][7:]}.json").read_text())
    assert held["status"] == "held" and held["admission"]["allocation_binding_digest"] == reservations[0]["binding_digest"]
    assert "fixture-meta-secret" not in json.dumps(held)

    monkeypatch.setattr(sam, "parse_tracks", real_parse)
    calls = []

    def opener(request, **kwargs):
        calls.append(json.loads(request.data)["input"][0]["content"][0]["text"])
        return BytesIO(json.dumps(response()).encode())

    result = sam.run_meta_sam31(**args, opener=opener)
    assert result["status"] == "completed" and calls == ["object 2"]
    assert len(reservations) == 1                        # the same hold paid it: no second reservation
    assert sources == [args["task_context"]]             # the site's current source was read before the reuse
    assert json.loads((root / "response-2.json").read_text())["allocation_binding_digest"] == reservations[0]["binding_digest"]


def test_an_expired_kept_hold_stays_counted_and_the_next_reservation_is_a_new_binding(tmp_path, monkeypatch):
    args, root, reservations, _sources = _two_answered_one_pending(tmp_path, monkeypatch)
    prompts = args["prompts"]
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=prompts))
    pending = canonical_digest({"kind": "meta_sam31_pending_prompts_v1", "request_binding_digest": digest,
                                "pending_prompt_indices": [2]})
    expired = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
               "allocation_binding_digest": pending, "maximum_cost_usd": 0.0025, "external_disclosure_allowed": True,
               "expires_at_epoch": time.time() - 1}
    sam._write_reservation(root, pending, expired)
    calls = []
    result = sam.run_meta_sam31(**args, opener=lambda request, **kw: calls.append(1) or BytesIO(json.dumps(response()).encode()))
    assert result["status"] == "completed" and calls == [1]
    assert len(reservations) == 1 and reservations[0]["binding_digest"] not in (pending, digest)
    assert json.loads((root / f"held-reservation-{pending[7:]}.json").read_text())["status"] == "superseded"
    assert sam._superseded_reservations(root) == [pending]


def _website_holds_without_record(monkeypatch, *, holding):
    """A website that answers ``already_reserved`` for every binding in ``holding`` (it granted those to a run that
    kept no record), and grants any other once, as the real reservation does."""
    from blueprint_pipeline import website_task_context
    from blueprint_pipeline.paid_resource_admission import PaidResourceAlreadyReserved

    reservations = []

    def reserve(**kwargs):
        value = {**build_paid_lane_admission(resource_class="evaluator_api"), "provider": "meta",
                 "allocation_binding_digest": kwargs["binding_digest"], "maximum_cost_usd": kwargs["maximum_cost_usd"],
                 "external_disclosure_allowed": True, "expires_at_epoch": time.time() + 3600}
        if kwargs["binding_digest"] in holding or kwargs["binding_digest"] in [r["binding_digest"] for r in reservations]:
            raise PaidResourceAlreadyReserved(["paid_resource_admission_not_admitted"], {**value, "status": "already_reserved"})
        reservations.append(kwargs)
        kwargs["retain_admission"](value)
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
            expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    return reservations


def _pending_digest(args):
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=args["prompts"]))
    return digest, canonical_digest({"kind": "meta_sam31_pending_prompts_v1", "request_binding_digest": digest,
                                     "pending_prompt_indices": [2]})


def test_a_hold_the_website_keeps_with_no_record_here_is_set_aside_and_the_prompt_bought_once_more(tmp_path, monkeypatch):
    """The owner's Austin job (2026-11-24): its pending prompt was reserved by a run that kept no record, then every
    later run was answered ``already_reserved`` and refused for good. With no intent under it, nothing was sent: the
    hold is set aside (still counted) and the prompt is reserved under a new binding, once."""
    args, root, _reservations, _sources = _two_answered_one_pending(tmp_path, monkeypatch)
    digest, pending = _pending_digest(args)
    reservations = _website_holds_without_record(monkeypatch, holding={pending})
    calls = []

    def opener(request, **kwargs):
        calls.append(json.loads(request.data)["input"][0]["content"][0]["text"])
        return BytesIO(json.dumps(response()).encode())

    result = sam.run_meta_sam31(**args, opener=opener)
    assert result["status"] == "completed" and calls == ["object 2"]
    assert len(reservations) == 1 and reservations[0]["binding_digest"] not in (pending, digest)
    assert json.loads((root / f"held-reservation-{pending[7:]}.json").read_text())["status"] == "orphaned"
    assert json.loads((root / "response-2.json").read_text())["allocation_binding_digest"] == reservations[0]["binding_digest"]


def test_a_binding_set_aside_twice_is_refused_rather_than_bought_a_third_time(tmp_path, monkeypatch):
    from blueprint_pipeline.paid_resource_admission import PaidResourceAlreadyReserved

    args, root, _reservations, _sources = _two_answered_one_pending(tmp_path, monkeypatch)
    _digest, pending = _pending_digest(args)

    class Everything(set):
        def __contains__(self, item):
            return True

    reservations = _website_holds_without_record(monkeypatch, holding=Everything())
    for _ in range(sam.MAX_ORPHANED_RESERVATIONS):
        with pytest.raises(PaidResourceAlreadyReserved):
            sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("nothing is sent without a grant"))
    assert reservations == [] and len(sam._superseded_reservations(root, frozenset({"orphaned"}))) == sam.MAX_ORPHANED_RESERVATIONS
    with pytest.raises(PaidResourceAlreadyReserved):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("nothing is sent without a grant"))
    assert len(sam._superseded_reservations(root, frozenset({"orphaned"}))) == sam.MAX_ORPHANED_RESERVATIONS


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
    # An intent that names no reservation cannot be counted as spent, so it is never resent.
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


def test_timeout_is_resent_only_under_its_own_grant_and_missing_grant_cannot_call_provider(tmp_path, monkeypatch):
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked

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
    # The unanswered dispatch is counted as spent; its one resend needs its own reservation,
    # and the grant for the first never pays for it.
    with pytest.raises(PaidResourceAdmissionBlocked, match="grant_binding_mismatch"):
        sam.run_meta_sam31(**args, opener=opener)
    assert len(calls) == 1
    counted = json.loads(next(args["output_root"].rglob(sam.COUNTED_AS_SPENT)).read_text())
    assert counted["status"] == "counted_as_spent" and counted["allocation_binding_digest"] == args["admission"]["allocation_binding_digest"]


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


def test_failed_encoder_cannot_publish_partial_video(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    monkeypatch.setattr(sam, "_probe_video", lambda _: {"streams": [{"time_base": "1/600"}], "frames": [
        {"best_effort_timestamp_time": "0.0"}, {"best_effort_timestamp_time": "0.033"}]})
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


def test_variable_frame_rate_phone_clip_keeps_source_timestamps(tmp_path):
    # An iPhone-like clip: rotated, 1/600 track timescale, one off-grid frame interval.
    plain, source = tmp_path / "plain.mov", tmp_path / "vfr.mov"
    sam.subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30",
        "-frames:v", "30", "-vf", "settb=1/600,setpts='N*20+gte(N,10)*9'", "-fps_mode", "passthrough",
        "-enc_time_base", "1/600", "-c:v", "libx264", "-video_track_timescale", "600", str(plain)],
        check=True, capture_output=True)
    sam.subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation:v:0", "90", "-i", str(plain),
        "-c", "copy", "-video_track_timescale", "600", str(source)], check=True, capture_output=True)
    original = sam._probe_video(source)
    assert original["streams"][0]["time_base"] == "1/600"
    expected = [float(frame["best_effort_timestamp_time"]) for frame in original["frames"]]
    root = tmp_path / "continuous"
    registry, video = sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=root)
    assert [row["decoded_pts_seconds"] for row in registry] == expected
    assert (registry[0]["width"], registry[0]["height"]) == (48, 64)
    encoded = [float(frame["best_effort_timestamp_time"]) for frame in sam._probe_video(Path(video["path"]))["frames"]]
    assert all(abs((a - encoded[0]) - (b - expected[0])) <= 0.002 for a, b in zip(encoded, expected, strict=True))
    assert video["encoding"] == "upright_h264_crf18_maxrate_fit_veryfast_threads2_source_timebase_all_source_frames_v4"
    # A receipt from the earlier encode is still reused rather than re-bought.
    receipt = json.loads((root / "continuous-video.json").read_text())
    receipt["video"]["encoding"] = "upright_h264_crf18_veryfast_threads2_all_source_frames_v2"
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    (root / "continuous-video.json").write_text(json.dumps(receipt))
    assert sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=root)[1] == receipt["video"]


def test_source_without_fine_timebase_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    monkeypatch.setattr(sam, "_probe_video", lambda _: {"streams": [{"time_base": "1001/30000"}], "frames": [{}, {}]})
    monkeypatch.setattr(sam.subprocess, "run", lambda *a, **k: pytest.fail("must not encode"))
    with pytest.raises(ValueError, match="meta_sam_source_time_base_invalid"):
        sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=tmp_path / "out")


def _oversize_encoder(tmp_path, monkeypatch, oversize_crfs):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    frames = {"streams": [{"width": 1080, "height": 1920, "time_base": "1/600"}],
              "frames": [{"best_effort_timestamp_time": "0.0"}, {"best_effort_timestamp_time": "15.0"}]}
    monkeypatch.setattr(sam, "_probe_video", lambda _: frames)
    crfs = []
    def encode(argv, **kwargs):
        # Rotated phone clips otherwise encode on 1/(29.97 fps) and drift from
        # the source timestamps; the demuxer timebase and timescale must carry over.
        assert argv[argv.index("-enc_time_base") + 1] == "-1"
        assert argv[argv.index("-video_track_timescale") + 1] == "600"
        # A 30-second clip is capped at 90% of the 32 MiB bound on the first pass.
        assert argv[argv.index("-maxrate") + 1] == argv[argv.index("-bufsize") + 1] == str(int(0.9 * 32 * 1024**2 * 8 / 30))
        crf = int(argv[argv.index("-crf") + 1])
        crfs.append(crf)
        size = sam._CONTINUOUS_VIDEO_MAX_BYTES + 1 if crf in oversize_crfs else 1024
        with open(argv[-1], "wb") as file:
            file.truncate(size)
    monkeypatch.setattr(sam.subprocess, "run", encode)
    return source, crfs


def test_oversized_continuous_video_steps_down_quality_ladder(tmp_path, monkeypatch):
    source, crfs = _oversize_encoder(tmp_path, monkeypatch, {18})
    root = tmp_path / "output"
    registry, video = sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=root)
    assert crfs == [18, 23]
    assert video["encoding"] == "upright_h264_crf23_maxrate_fit_veryfast_threads2_source_timebase_all_source_frames_v4"
    assert [(row["width"], row["height"]) for row in registry] == [(1080, 1920)] * 2
    assert Path(video["path"]).stat().st_size == 1024
    assert sorted(path.name for path in root.glob("*.mp4")) == ["continuous-upright.mp4"]
    # The recorded rung is accepted on reuse.
    assert sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=root)[1] == video


def test_continuous_video_over_bound_at_every_rung_fails_closed(tmp_path, monkeypatch):
    source, crfs = _oversize_encoder(tmp_path, monkeypatch, set(sam._CONTINUOUS_CRF_LADDER))
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="meta_sam_inline_video_limits_exceeded"):
        sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source), root=root)
    assert crfs == list(sam._CONTINUOUS_CRF_LADDER)
    assert not list(root.glob("*.mp4"))
    assert not (root / "continuous-video.json").exists()


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


@pytest.mark.parametrize("refusal", [404, 429, 503])
def test_a_refusal_the_provider_answered_is_dispatched_again_under_a_fresh_reservation(tmp_path, monkeypatch, refusal):
    # capture-5b067498 (2026-11-06): the second target's request named a video upload from an
    # earlier pass that the provider no longer held, and it answered 404. That is an outcome,
    # not an uncertainty: the next pass uploads again and reserves the prompt afresh.
    from urllib.error import HTTPError

    from blueprint_pipeline import website_task_context

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    prompts = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(2)]
    args.update(prompts=prompts, admission={}, admission_grant=None, task_context={"capture_id": "fixture"})
    reservations = []

    def reserve(**kwargs):
        reservations.append(kwargs)
        value = {**build_paid_lane_admission(resource_class="evaluator_api"),
                 "allocation_binding_digest": kwargs["binding_digest"],
                 "maximum_cost_usd": kwargs["maximum_cost_usd"], "external_disclosure_allowed": True}
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
            expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    calls = []

    def refuses_second(request, **kwargs):
        text = json.loads(request.data)["input"][0]["content"][0]["text"]
        calls.append(text)
        if text == "object 1" and calls.count("object 1") == 1:
            raise HTTPError(sam.ENDPOINT, refusal, "Refused", {}, None)
        return BytesIO(json.dumps(response()).encode())

    # 429 (rate limited) and 503 (unavailable) are refusals too: retried, never a reconciliation hold.
    with pytest.raises(ValueError, match=f"meta_sam_http_{refusal}"):
        sam.run_meta_sam31(**args, opener=refuses_second)
    result = sam.run_meta_sam31(**args, opener=refuses_second)
    assert result["status"] == "completed"
    assert calls == ["object 0", "object 1", "object 1"]
    # The refused dispatch and the new one are separate reservations; neither replays the other.
    assert len(reservations) == 2 and reservations[0]["binding_digest"] != reservations[1]["binding_digest"]
    assert reservations[1]["request_count"] == 1
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=prompts))
    archived = args["output_root"] / digest[7:] / "refused" / "attempt-1"
    assert json.loads((archived / "failure-1.json").read_text())["http_status"] == refusal
    assert (archived / "intent-1.json").is_file()
    # An answered refusal is no unknown outcome: nothing is counted as spent or marked a resend.
    assert not (archived.parents[1] / sam.UNANSWERED_ARCHIVE).exists()
    assert "resend_of" not in json.loads((archived.parents[1] / "response-1.json").read_text())


def test_a_server_error_answer_whose_intent_names_no_reservation_is_never_resent(tmp_path, monkeypatch):
    # A 500 that names its reservation is resent once under a new one (test_meta_sam31_inline_clip);
    # an intent naming none cannot be counted as spent, so it is never resent, and the step ends.
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
        frame_artifacts=args["frame_artifacts"], prompts=args["prompts"]))
    root = args["output_root"] / digest[7:]
    root.mkdir(parents=True)
    sam.write_json(root / "intent-0.json", {"binding_digest": digest})
    sam.write_json(root / "failure-0.json", {"http_status": 500, "binding_digest": digest})
    with pytest.raises(ValueError, match="requires_reconciliation"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("a 5xx may have been processed"))
    assert sam.retire_refused_dispatches(root, [0]) == 0
    assert not (root / sam.UNANSWERED_ARCHIVE).exists() and (root / "intent-0.json").is_file()


def test_the_parser_time_limit_grows_with_the_response():
    assert sam.parser_timeout_seconds("x" * 1000) == 300
    assert sam.parser_timeout_seconds("x" * (2 * 1024 * 1024)) == 600
    assert sam.parser_timeout_seconds("x" * (64 * 1024 * 1024)) == 3600


def test_the_parser_time_limit_also_grows_with_the_boxes_it_decodes():
    """Owner's Austin walkthrough (2026-11-24): 3,010 masks, mostly 1079x1617 boxes, in 1.26 MiB of text, timed out
    twice at the text's 379 s; the boxes' 1,912 megapixels give 956 s. The text's measure still holds as a floor."""
    box = "<0f>0<|box;x1=0;y1=302;x2=1079;y2=1919;w=1080;h=1920|><|mask;x=0;y=0;data=1024,684,!!|>"
    assert sam.parser_timeout_seconds(box * 1000) == math.ceil(0.5 * 1000 * 1079 * 1617 / 1e6)   # 873 s
    assert sam.parser_timeout_seconds(box) == 300                                                # the floor
    assert sam.parser_timeout_seconds(box * 10_000) == 3600                                      # the ceiling
    small = "<0f>0<|box;x1=10;y1=10;x2=20;y2=20;w=1080;h=1920|>" + "x" * (2 * 1024 * 1024)
    assert sam.parser_timeout_seconds(small) == math.ceil(300 * len(small) / 1024**2)            # text measure wins


def _job_spend(monkeypatch):
    """The job's reservation path as the website answers it: one admission per distinct binding."""
    from blueprint_pipeline import website_task_context

    reservations = []

    def reserve(**kwargs):
        assert kwargs["binding_digest"] not in [row["binding_digest"] for row in reservations], "already_reserved"
        reservations.append(kwargs)
        value = {**build_paid_lane_admission(resource_class="evaluator_api"),
                 "allocation_binding_digest": kwargs["binding_digest"],
                 "maximum_cost_usd": kwargs["maximum_cost_usd"], "external_disclosure_allowed": True}
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
            expected_schema_version="paid_lane_admission.v1")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    # Nothing here may release a reservation or settle one, at zero or otherwise.
    monkeypatch.setattr(website_task_context, "preparation_webapp_request",
                        lambda **_: pytest.fail("a SAM reservation was released or settled"))
    return reservations


def _two_prompts(tmp_path):
    args = inputs(tmp_path)
    prompts = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(2)]
    args.update(prompts=prompts, admission={}, admission_grant=None, task_context={"capture_id": "fixture"})
    digest = canonical_digest(sam.request_binding(frame_registry=args["frame_registry"],
                                                  frame_artifacts=args["frame_artifacts"], prompts=prompts))
    return args, digest, args["output_root"] / digest[7:]


def _provider(calls, *, drop):
    """Answers every prompt except those in ``drop``, whose answer never arrives (a crash looks the same)."""
    def opener(request, **_kwargs):
        text = json.loads(request.data)["input"][0]["content"][0]["text"]
        calls.append(text)
        if text in drop:
            raise ConnectionResetError("no answer")
        return BytesIO(json.dumps(response()).encode())
    return opener


def test_an_unanswered_dispatch_is_counted_as_spent_and_resent_once_under_its_own_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    reservations = _job_spend(monkeypatch)
    args, digest, root = _two_prompts(tmp_path)
    calls = []
    with pytest.raises(ConnectionResetError):
        sam.run_meta_sam31(**args, opener=_provider(calls, drop={"object 1"}))
    first_intent = (root / "intent-1.json").read_bytes()
    assert json.loads(first_intent)["allocation_binding_digest"] == digest == reservations[0]["binding_digest"]
    answered = json.loads((root / "response-0.json").read_text())

    result = sam.run_meta_sam31(**args, opener=_provider(calls, drop=set()))
    assert result["status"] == "completed" and calls == ["object 0", "object 1", "object 1"]
    # The first reservation is kept as it was; the resend has its own, distinct binding, for one prompt.
    resend_of = "sha256:" + sam.hashlib.sha256(first_intent).hexdigest()
    mark = {"resend_of": resend_of, "resend_index": 1}
    assert len(reservations) == 2 and reservations[0]["binding_digest"] == digest
    assert reservations[1]["binding_digest"] == canonical_digest({
        "kind": "meta_sam31_pending_prompts_v1", "request_binding_digest": digest, "pending_prompt_indices": [1],
        "unanswered_resends": [{"prompt_index": 1, **mark}]}) != digest
    assert (reservations[1]["request_count"], reservations[1]["maximum_cost_usd"]) == (1, 0.0025)
    # The unanswered first dispatch is recorded as spent, naming the reservation that stays held.
    archived = root / sam.UNANSWERED_ARCHIVE / "attempt-1"
    counted = json.loads((archived / sam.COUNTED_AS_SPENT).read_text())
    assert (archived / "intent-1.json").read_bytes() == first_intent
    assert {key: counted[key] for key in ("status", "outcome", "prompt_index", "allocation_binding_digest",
                                          "intent_sha256", "reservation", "resends_allowed")} == {
        "status": "counted_as_spent", "outcome": "unknown", "prompt_index": 1, "allocation_binding_digest": digest,
        "intent_sha256": resend_of, "reservation": "held_at_maximum_never_released_or_settled_as_zero",
        "resends_allowed": 1}
    # Both are recorded: the resend's receipt names its own reservation and what it resends.
    resent = json.loads((root / "response-1.json").read_text())
    assert resent["allocation_binding_digest"] == reservations[1]["binding_digest"]
    assert {key: resent[key] for key in mark} == mark
    # An answered prompt is untouched.
    assert json.loads((root / "response-0.json").read_text()) == answered
    assert set(answered) == {"binding_digest", "clip_digest", "allocation_binding_digest", "response",
                             "reserved_cost_usd"} and answered["allocation_binding_digest"] == digest
    assert sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("retained")) == result
    assert len(reservations) == 2


def test_a_resend_that_also_gets_no_answer_stops_for_review(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    reservations = _job_spend(monkeypatch)
    args, digest, root = _two_prompts(tmp_path)
    calls = []
    for _ in range(2):
        with pytest.raises(ConnectionResetError):
            sam.run_meta_sam31(**args, opener=_provider(calls, drop={"object 1"}))
    resend_intent = json.loads((root / "intent-1.json").read_text())
    assert resend_intent["resend_index"] == 1 and resend_intent["allocation_binding_digest"] == reservations[1]["binding_digest"]
    with pytest.raises(ValueError, match="meta_sam_submission_requires_reconciliation"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("a second resend"))
    assert calls == ["object 0", "object 1", "object 1"] and len(reservations) == 2
    assert len(list((root / sam.UNANSWERED_ARCHIVE).glob("attempt-*"))) == 1
    assert json.loads((root / "intent-1.json").read_text()) == resend_intent


def test_a_refused_resend_is_dispatched_again_but_never_resent_a_second_time(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    reservations = _job_spend(monkeypatch)
    args, digest, root = _two_prompts(tmp_path)
    calls = []
    with pytest.raises(ConnectionResetError):
        sam.run_meta_sam31(**args, opener=_provider(calls, drop={"object 1"}))

    def refuses(request, **_kwargs):
        calls.append(json.loads(request.data)["input"][0]["content"][0]["text"])
        raise HTTPError(sam.ENDPOINT, 409, "Conflict", {}, None)
    with pytest.raises(ValueError, match="meta_sam_http_409"):
        sam.run_meta_sam31(**args, opener=refuses)
    # The refusal is an answer (#3423): the prompt goes again under a fresh reservation, still marked
    # as this prompt's one resend, so a lost answer to it stops for review.
    with pytest.raises(ConnectionResetError):
        sam.run_meta_sam31(**args, opener=_provider(calls, drop={"object 1"}))
    assert json.loads((root / "intent-1.json").read_text())["resend_index"] == 1
    with pytest.raises(ValueError, match="meta_sam_submission_requires_reconciliation"):
        sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("a second resend"))
    assert len(reservations) == 3 and len({row["binding_digest"] for row in reservations}) == 3
    assert calls == ["object 0", "object 1", "object 1", "object 1"]


def test_an_archive_a_crash_interrupted_is_finished_not_counted_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    reservations = _job_spend(monkeypatch)
    args, digest, root = _two_prompts(tmp_path)
    calls = []
    with pytest.raises(ConnectionResetError):
        sam.run_meta_sam31(**args, opener=_provider(calls, drop={"object 1"}))
    original = root / "intent-1.json"
    first = original.read_bytes()
    sam.count_unanswered_dispatches(root, [1], binding_digest=digest)
    archived = root / sam.UNANSWERED_ARCHIVE / "attempt-1"
    # As if the process died after the record was written and before the intent moved.
    shutil.move(str(archived / "intent-1.json"), str(original))
    assert sam.run_meta_sam31(**args, opener=_provider(calls, drop=set()))["status"] == "completed"
    assert (archived / "intent-1.json").read_bytes() == first
    assert json.loads(original.read_text())["resend_index"] == 1  # the resend's own intent
    assert len(list((root / sam.UNANSWERED_ARCHIVE).glob("attempt-*"))) == 1 and len(reservations) == 2


@pytest.mark.parametrize("cap,resent", [(0.005, False), (0.0075, True)])
def test_a_resend_never_exceeds_the_callers_own_cap(tmp_path, monkeypatch, cap, resent):
    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    reservations = _job_spend(monkeypatch)
    args, digest, root = _two_prompts(tmp_path)
    args["maximum_cost_usd"] = cap  # two prompts at 0.0025 each; only the larger cap leaves room for one more
    calls = []
    with pytest.raises(ConnectionResetError):
        sam.run_meta_sam31(**args, opener=_provider(calls, drop={"object 1"}))
    if resent:
        assert sam.run_meta_sam31(**args, opener=_provider(calls, drop=set()))["status"] == "completed"
        assert len(reservations) == 2 and calls[-1] == "object 1"
    else:
        with pytest.raises(ValueError, match="meta_sam_submission_requires_reconciliation"):
            sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("resend beyond the caller's cap"))
        assert len(reservations) == 1 and not (root / sam.UNANSWERED_ARCHIVE).exists()
        assert (root / "intent-1.json").is_file()


# --- settling a website reservation (stress round 2, 2026-11-24) -----------------------------------------------

WEBSITE_CONTEXT = {"capture_id": "walkthrough-req1", "request_id": "req1", "scene_id": "site-req1",
                   "context_digest": "sha256:" + "c" * 64}


def _website_run(tmp_path, monkeypatch, *, settle):
    """Three prompts reserved on the job's ledger as one reservation, answered; ``settle`` is the website."""
    from blueprint_pipeline import website_task_context

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    args = inputs(tmp_path)
    prompts = [{**PROMPT, "prompt_id": f"target-{i}", "text": f"object {i}"} for i in range(3)]
    args.update(prompts=prompts, admission={}, admission_grant=None, task_context=WEBSITE_CONTEXT)
    reservations, sent = [], []

    def reserve(**kwargs):
        reservations.append(kwargs)
        value = {**build_paid_lane_admission(resource_class="evaluator_api"),
                 "allocation_binding_digest": kwargs["binding_digest"],
                 "maximum_cost_usd": kwargs["maximum_cost_usd"], "external_disclosure_allowed": True}
        return value, require_paid_resource_admission(value, resource_class="evaluator_api",
            expected_schema_version="paid_lane_admission.v1")

    def webapp(*, task_context, operation, payload):
        assert operation == "preparation-settlement" and task_context is WEBSITE_CONTEXT
        sent.append(payload["settlement"])
        return settle(payload["settlement"])

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    monkeypatch.setattr(website_task_context, "preparation_webapp_request", webapp)
    return args, reservations, sent


def _accepts(command):
    return {**command, "actual_cost_usd": command["provider_charge_amount_usd"], "status": "settled"}


def _answers(request, **kwargs):
    return BytesIO(json.dumps(response()).encode())


def test_a_reservation_whose_prompts_are_all_answered_is_settled_once_at_its_usage(tmp_path, monkeypatch):
    """Before this no Meta reservation was settled: each held its whole quote on the job's ledger for the job's life."""
    args, reservations, sent = _website_run(tmp_path, monkeypatch, settle=_accepts)
    assert sam.run_meta_sam31(**args, opener=_answers)["status"] == "completed"
    assert len(reservations) == 1 and len(sent) == 1
    command = sent[0]
    assert command["provider"] == "meta" and command["completed_request_count"] == 3
    assert command["allocation_binding_digest"] == reservations[0]["binding_digest"]
    assert command["task_context_digest"] == WEBSITE_CONTEXT["context_digest"]
    # One image per prompt, at the reviewed image price: never above what was reserved.
    assert command["provider_charge_amount_usd"] == 0.0075 <= reservations[0]["maximum_cost_usd"]
    assert command["usage_receipt_digest"].startswith("sha256:")
    # Replayed from the retained answers: nothing is bought or settled again.
    assert sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("retained"))["status"] == "completed"
    assert len(sent) == 1 and len(reservations) == 1


def test_a_website_that_refuses_the_meta_settlement_keeps_the_hold_and_is_asked_again_later(tmp_path, monkeypatch):
    """The website's settlement reader lists no ``meta`` variant until its release adds it: the refusal changes
    nothing (the row keeps its quote, so the cap over-counts, never under-counts), the step still completes, and a
    later run of the binding asks again."""
    answers = {"accept": False}

    def website(command):
        if not answers["accept"]:
            raise ValueError("website_control_preparation-settlement_http_400:unclassified")
        return _accepts(command)
    args, _reservations, sent = _website_run(tmp_path, monkeypatch, settle=website)
    assert sam.run_meta_sam31(**args, opener=_answers)["status"] == "completed"
    assert len(sent) == 1
    root = next(path for path in args["output_root"].iterdir() if path.is_dir())
    assert not list(root.glob("reservation-*.settlement.json"))
    answers["accept"] = True
    sam.run_meta_sam31(**args, opener=lambda *a, **kw: pytest.fail("retained"))
    assert len(sent) == 2 and sent[0] == sent[1] and len(list(root.glob("reservation-*.settlement.json"))) == 1


def test_a_reservation_with_a_prompt_answered_under_another_is_never_settled(tmp_path, monkeypatch):
    """An unanswered dispatch is counted as spent and resent under a new reservation: the first keeps its hold."""
    args, _reservations, sent = _website_run(tmp_path, monkeypatch, settle=_accepts)
    sam.run_meta_sam31(**args, opener=_answers)
    root = next(path for path in args["output_root"].iterdir() if path.is_dir())
    for path in root.glob("reservation-*.settlement.json"):
        path.unlink()
    answer = json.loads((root / "response-1.json").read_text())
    sam.write_json(root / "response-1.json", {**answer, "allocation_binding_digest": "sha256:" + "e" * 64})
    assert sam.settle_answered_reservations(root, task_context=WEBSITE_CONTEXT) == []
    assert len(sent) == 1


def test_an_answer_is_priced_from_the_usage_meta_reported():
    video = {"response": {"usage": {"video_frames_processed": 37}}}
    assert sam._answered_cost(video, image_request=False) == pytest.approx(37 * sam.PRICE_PER_FRAME_USD)
    assert sam._answered_cost({"response": {"usage": {}}}, image_request=False) is None
    assert sam._answered_cost({"response": {"usage": {"video_frames_processed": True}}}, image_request=False) is None
    assert sam._answered_cost({}, image_request=True) == sam.PROFILE["price_per_image_usd"]
