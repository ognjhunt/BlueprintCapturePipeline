"""A deferred contact surface can recover without buying the source clip again."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from blueprint_pipeline import meta_sam31 as sam, website_task_grounding as grounding
from blueprint_pipeline import website_task_masks as masks
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def _snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def retained(tmp_path, monkeypatch, request):
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"synthetic original source video")
    source_digest = _sha256_file(source_video)
    context = {"description": "Place the blue object on the wooden desk."}
    context_digest = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    frames = [{"frame_id": f"decoded-{i:09d}", "timestamp_seconds": float(i),
               "width": 4, "height": 4, "source_width": 4, "source_height": 4} for i in range(3)]
    source = {"geometry_available": False, "geometry_input_digest": "sha256:source-input",
              "binding": {"source_video_digest": source_digest}, "frames": frames[:2]}
    source["digest"] = canonical_digest(source, digest_field="digest")
    options = getattr(request, "param", {})
    targets = [{"target_id": identifier, "semantic_label": identifier, "segmentation_prompt": identifier,
                "task_effect": effect, "disposition": disposition,
                "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": box}]}
               for identifier, effect, disposition, box in [
                   ("object", "manipulated", "remove", [0, 0, .5, .5]),
                   ("floor", "static_contact", "keep", [0, .5, .5, .5]),
                   ("desk", options.get("desk_effect", "static_contact"), "keep", [0, 0, .5, .5])]]
    plan = {"task_context_sha256": context_digest, "targets": targets}
    binding = {"geometry_digest": None, "source_frames_digest": source["digest"],
               "geometry_input_digest": source["geometry_input_digest"], "task_targets": targets,
               "task_context_sha256": context_digest, "profile_digest": canonical_digest(sam.PROFILE),
               "mask_input_pixels": "continuous_source_video_v1"}
    root = tmp_path / "task_masks" / canonical_digest(binding)[7:23]
    root.mkdir(parents=True)
    encoded = root / "continuous-upright.mp4"
    encoded.write_bytes(b"synthetic retained continuous video")
    registry = [{"source_frame_id": frame["frame_id"], "model_frame_index": index,
                 "decoded_pts_seconds": frame["timestamp_seconds"], "width": 4, "height": 4,
                 "retained_video_digest": source_digest} for index, frame in enumerate(frames)]
    video = {"path": str(encoded), "sha256": _sha256_file(encoded), "source_video_digest": source_digest,
             "encoding": "upright_h264_crf18_veryfast_threads2_all_source_frames_v2"}
    continuous = {"registry": registry, "video": video}
    continuous["digest"] = canonical_digest(continuous, digest_field="digest")
    write_json(root / "continuous-video.json", continuous)
    def probe(path):
        assert path == encoded
        return {"streams": [{"width": 4, "height": 4}], "frames": [{}, {}, {}]}
    monkeypatch.setattr(sam, "_probe_video", probe)
    monkeypatch.setattr(sam.subprocess, "run", lambda *_, **__: pytest.fail("retained recovery must not encode a video"))
    prompts = [{"prompt_id": target["target_id"], "text": target["segmentation_prompt"],
                "output_label": target["target_id"], "anchor_frame_index": 2} for target in targets]
    provider_binding = sam.request_binding(frame_registry=registry, frame_artifacts=[], prompts=prompts,
                                          video_artifact=video)
    provider_digest = canonical_digest(provider_binding)
    paid_root = root / provider_digest[7:]
    paid_root.mkdir()
    write_json(paid_root / "binding.json", provider_binding)

    def track(identifier, label, start):
        return {"track_id": identifier, "label": label, "observations": [
            {"source_frame_id": row["source_frame_id"], "width": 4, "height": 4,
             "runs": [{"start": start, "length": 2}, {"start": start + 4, "length": 2}]}
            for row in registry]}

    tracks = [track("object-1", "object", 0), track("floor-1", "floor", 8),
              track("desk-a", "desk", 8), track("desk-b", "desk", 10)]
    receipts = []
    for index, prompt in enumerate(prompts):
        response_path = paid_root / f"response-{index}.json"
        write_json(response_path, {"binding_digest": provider_digest, "clip_digest": video["sha256"],
                                   "response": {"status": "completed", "usage": {"video_frames_processed": 3}}})
        response_digest = _sha256_file(response_path)
        receipts.append({"path": str(response_path), "sha256": response_digest})
        parsed = {"binding": {"response_digest": response_digest, "request_digest": provider_digest,
                              "parser_revision": 1, "profile": sam.PROFILE},
                  "tracks": [row for row in tracks if row["label"] == prompt["output_label"]],
                  "parser_runtime": "synthetic_test_fixture"}
        parsed["digest"] = canonical_digest(parsed, digest_field="digest")
        write_json(paid_root / f"parsed-response-{index}.json", parsed)
    result = {"schema_version": "website_meta_sam31_tracks.v1", "status": "completed",
              "binding_digest": provider_digest, "profile": sam.PROFILE, "tracks": tracks, "responses": receipts,
              "claim_ceiling": "development_only"}
    write_json(paid_root / "tracks.json", result)
    initial = {"schema_version": "website_task_masks.v1", "status": "object_removal_ready",
               "binding": binding, "source_geometry_digest": None, "claim_ceiling": "development_only",
               "source_video_digest": source_digest, "source_frame_registry": registry,
               "deferred_target_ids": ["floor", "desk"],
               "targets": [{**targets[0], "source_track": tracks[0], "already_retained": True}]}
    initial["digest"] = canonical_digest(initial, digest_field="digest")
    write_json(root / "task_masks.object_removal.json", initial)
    posed_frames = []
    for frame in (frames[0], frames[2]):
        depth = tmp_path / f"{frame['frame_id']}.npz"
        np.savez_compressed(depth, depth_m=np.ones((4, 4)), valid_mask=np.ones((4, 4), dtype=bool))
        posed_frames.append({**frame, "geometry_path": str(depth), "geometry_digest": _sha256_file(depth),
                             "intrinsics": np.eye(3).tolist(), "world_from_camera": np.eye(4).tolist()})
    geometry = {"binding": {"source_video_digest": source_digest, "input_digest": "sha256:posed"},
                "frames": posed_frames}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    monkeypatch.setattr(sam, "meta_api_key", lambda: pytest.fail("retained clip must not require a provider key"))
    monkeypatch.setattr(sam, "_upload_video", lambda **_: pytest.fail("must not upload the completed video"))
    monkeypatch.setattr(sam, "parse_tracks", lambda *_, **__: pytest.fail("must reuse validated decoded masks"))
    monkeypatch.setattr(masks, "resolve_video_segmentation_concept", lambda **_: pytest.fail("no new video concept search"))
    return {"plan": plan, "source_geometry": source, "task_masks": initial, "output_root": root.parent,
            "view_plan_root": tmp_path / "mask_view_plan", "task_context": context,
            "source_video": source_video, "recovery_geometry": geometry}, paid_root


@pytest.mark.parametrize("single_image", [False, True])
def test_recovery_uses_calibrated_source_views_and_preserves_completed_evidence(retained, monkeypatch, single_image):
    kwargs, paid_root = retained
    original_paid = _snapshot(paid_root)
    original_masks = copy.deepcopy(kwargs["task_masks"])
    calls = []

    def ground(**kw):
        calls.append("ground:" + kw["target"]["target_id"])
        assert [row["model_frame_index"] for row in kw["registry"]] == [0, 2]
        assert kw["target"]["target_id"] == "desk"
        frame = kw["registry"][-1]
        box = [.5, 0, .5, .5] if single_image else [0, .5, .5, .5]
        return {**kw["target"], "segmentation_prompt": "wooden desktop",
                "spatial_evidence": [{"timestamp_seconds": 2.0, "box_xywh_normalized": box}],
                "grounding": {"source_frame_id": frame["source_frame_id"], "confidence": .95}}

    def segment(**kw):
        calls.append("image:" + kw["target"]["target_id"])
        assert single_image
        frame_id = kw["target"]["grounding"]["source_frame_id"]
        return {"track_id": "desk-image", "label": "desk", "coverage": "single_observed_frame",
                "observations": [{"source_frame_id": frame_id, "width": 4, "height": 4,
                                  "runs": [{"start": 2, "length": 2}, {"start": 6, "length": 2}]}]}

    monkeypatch.setattr(grounding, "ground_task_target", ground)
    monkeypatch.setattr(masks, "segment_grounded_static_target", segment)
    result = masks.complete_retained_static_task_masks(**kwargs)
    assert calls == ["ground:desk"] + (["image:desk"] if single_image else [])
    assert result["status"] == "completed"
    assert result["targets"][0] == original_masks["targets"][0]
    assert result["targets"][-1]["source_track"]["track_id"] == ("desk-image" if single_image else "desk-a")
    assert result["retained_static_recovery"] == {
        "source_task_masks_digest": original_masks["digest"],
        "reference_geometry_digest": kwargs["recovery_geometry"]["digest"],
        "full_video_reused": True, "static_contact_only": True}
    assert kwargs["task_masks"] == original_masks
    assert _snapshot(paid_root) == original_paid
    # The new observation may be outside the old clean-plate sample, but it must
    # lift onto an actually calibrated view of this same source video.
    bound = masks.bind_task_masks_to_geometry(task_masks=result, source_geometry=kwargs["recovery_geometry"])
    assert bound["targets"][-1]["estimated_visible_bounds"]["metric_measurement_proven"] is False
    before_replay = _snapshot(kwargs["output_root"])
    assert masks.complete_retained_static_task_masks(**kwargs) == result
    assert _snapshot(kwargs["output_root"]) == before_replay
    completed_path = paid_root.parent / "task_masks.json"
    completed_path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="retained_static_masks_manifest_changed"):
        masks.complete_retained_static_task_masks(**kwargs)
    assert completed_path.read_text() == '{"changed":true}'
    assert _snapshot(paid_root) == original_paid


@pytest.mark.parametrize("change,error", [
    ("context", "recovery_context_changed"), ("calibration", "recovery_geometry_invalid"),
    ("video", "source_video_changed"),
    ("no_opt_in", "task_target_track_ambiguous"), ("parsed", "provider_receipts_invalid"),
])
def test_recovery_refuses_changed_evidence_and_noncontact_targets_before_model_calls(retained, monkeypatch, change, error):
    kwargs, paid_root = retained
    if change == "context":
        kwargs["task_context"]["description"] = "Another task"
    elif change == "calibration":
        kwargs["recovery_geometry"]["frames"][0]["source_width"] = 8
        kwargs["recovery_geometry"]["digest"] = canonical_digest(kwargs["recovery_geometry"], digest_field="digest")
    elif change == "video":
        kwargs["source_video"].write_bytes(b"changed source")
    elif change == "no_opt_in":
        kwargs.pop("task_context")
        kwargs.pop("source_video")
        kwargs.pop("recovery_geometry")
    elif change == "parsed":
        path = paid_root / "parsed-response-2.json"
        path.write_text('{"changed": true}')
    monkeypatch.setattr(grounding, "ground_task_target", lambda **_: pytest.fail("unproved recovery must not call a model"))
    with pytest.raises(ValueError, match=error):
        masks.complete_retained_static_task_masks(**kwargs)


@pytest.mark.parametrize("retained", [{"desk_effect": "static_obstacle"}], indirect=True)
def test_noncontact_obstacle_never_uses_a_single_frame_recovery(retained, monkeypatch):
    kwargs, _ = retained
    monkeypatch.setattr(grounding, "ground_task_target", lambda **_: pytest.fail("only kept contact surfaces can recover"))
    with pytest.raises(ValueError, match="task_target_track_ambiguous:desk"):
        masks.complete_retained_static_task_masks(**kwargs)


def test_direct_recovery_entry_also_binds_the_current_task_context(retained):
    kwargs, _ = retained
    kwargs["task_context"]["description"] = "Different task"
    with pytest.raises(ValueError, match="recovery_context_changed"):
        masks.run_website_task_masks(plan=kwargs["plan"], source_geometry=kwargs["source_geometry"],
            output_root=kwargs["output_root"], source_video=kwargs["source_video"],
            task_context=kwargs["task_context"], retained_static_masks=kwargs["task_masks"],
            recovery_geometry=kwargs["recovery_geometry"])


@pytest.mark.parametrize("change,error", [
    (None, None), ("missing", "retained_video_required"), ("binding", "retained_video_required"),
    ("parsed", "retained_tracks_changed"), ("result", "retained_tracks_changed"),
    ("missing_parsed", "retained_video_required"), ("profile", "retained_video_required"),
    ("symlink", "retained_video_required"),
])
def test_cache_only_video_mode_never_creates_an_implicit_paid_attempt(retained, change, error):
    kwargs, paid_root = retained
    binding = json.loads((paid_root / "binding.json").read_text())
    video = json.loads((paid_root.parent / "continuous-video.json").read_text())["video"]
    if change == "missing":
        (paid_root / "response-2.json").unlink()
    elif change == "binding":
        (paid_root / "binding.json").write_text('{}')
    elif change in {"parsed", "result"}:
        path = paid_root / ("parsed-response-2.json" if change == "parsed" else "tracks.json")
        path.write_text('{}')
    elif change == "missing_parsed":
        (paid_root / "parsed-response-2.json").unlink()
    elif change == "symlink":
        path = paid_root / "response-2.json"
        content = path.read_bytes()
        path.unlink()
        target = paid_root.parent / "other-response.json"
        target.write_bytes(content)
        path.symlink_to(target)
    elif change == "profile":
        binding["prompts"][0]["text"] = "changed concept"
    before = _snapshot(paid_root)
    args = dict(frame_registry=binding["frame_registry"], frame_artifacts=[], prompts=binding["prompts"],
                output_root=paid_root.parent, admission={}, video_artifact=video,
                require_retained_video=True, opener=lambda *_, **__: pytest.fail("no paid or upload calls"))
    if error:
        with pytest.raises(ValueError, match=error):
            sam.run_meta_sam31(**args)
    else:
        assert sam.run_meta_sam31(**args) == json.loads((paid_root / "tracks.json").read_text())
    assert _snapshot(paid_root) == before


def test_missing_cached_video_cannot_create_a_new_directory(tmp_path):
    with pytest.raises(ValueError, match="retained_video_required"):
        sam.run_meta_sam31(frame_registry=[], frame_artifacts=[], prompts=[], admission={},
                          output_root=tmp_path / "missing", require_retained_video=True)
    assert not (tmp_path / "missing").exists()
