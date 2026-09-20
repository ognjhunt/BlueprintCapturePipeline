from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.website_task_masks import decode_track_mask, estimate_target_bounds, select_task_track
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def _track(identifier="cup-1", *, start=0, label="task-cup"):
    return {"track_id": identifier, "label": label, "observations": [
        {"source_frame_id": "frame-0", "height": 4, "width": 4,
         "runs": [{"start": start, "length": 2, "probability": 0.9},
                  {"start": start + 4, "length": 2, "probability": 0.9}]}]}


def _target():
    return {"target_id": "task-cup", "spatial_evidence": [
        {"timestamp_seconds": 0, "box_xywh_normalized": [0, 0, 0.5, 0.5]}]}


def test_same_class_neighbor_is_not_removed():
    selected = select_task_track(target=_target(), tracks=[_track("wrong", start=2), _track("right")],
                                 frames=[{"frame_id": "frame-0", "timestamp_seconds": 0}])
    assert selected["track_id"] == "right"


def test_coarse_timestamp_uses_nearest_observed_mask_with_bounded_tolerance():
    frames = [{"frame_id": "unobserved", "timestamp_seconds": 0},
              {"frame_id": "frame-0", "timestamp_seconds": 1 / 30}]
    assert select_task_track(target=_target(), tracks=[_track()], frames=frames)["track_id"] == "cup-1"
    frames[1]["timestamp_seconds"] = 0.6
    with pytest.raises(ValueError, match="track_ambiguous"):
        select_task_track(target=_target(), tracks=[_track()], frames=frames)


def test_temporal_tolerance_does_not_cherry_pick_a_later_spatial_match():
    track = _track(start=2)
    later = {**_track()["observations"][0], "source_frame_id": "later"}
    track["observations"].append(later)
    with pytest.raises(ValueError, match="track_ambiguous"):
        select_task_track(target=_target(), tracks=[track], frames=[
            {"frame_id": "frame-0", "timestamp_seconds": 0.03},
            {"frame_id": "later", "timestamp_seconds": 0.1}])


def test_full_video_identity_is_selected_before_geometry_sampling(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_masks as masks

    target = {**_target(), "semantic_label": "blue box", "task_effect": "manipulated", "disposition": "remove"}
    target["spatial_evidence"][0]["timestamp_seconds"] = 8.0
    full_track = _track()
    full_track["observations"][0]["source_frame_id"] = "anchor"
    full_track["observations"].append({**_track(start=2)["observations"][0], "source_frame_id": "sampled"})
    registry = [{"source_frame_id": frame_id, "decoded_pts_seconds": timestamp, "width": 4, "height": 4}
                for frame_id, timestamp in [("anchor", 8 + 1 / 30), ("sampled", 8.1)]]
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    monkeypatch.setattr(masks, "run_meta_sam31", lambda **kw: {"tracks": [full_track]})
    result = masks.run_website_task_masks(
        plan={"targets": [target], "task_context_sha256": "task"},
        source_geometry={"digest": "source", "geometry_available": False,
                         "binding": {"source_video_digest": "video"},
                         "frames": [{"frame_id": "sampled", "timestamp_seconds": 8.1, "width": 4, "height": 4}]},
        source_video=tmp_path / "source.mov", output_root=tmp_path / "masks")
    assert result["targets"][0]["track"]["track_id"] == "cup-1"
    assert [row["source_frame_id"] for row in result["targets"][0]["track"]["observations"]] == ["sampled"]
    assert len(result["targets"][0]["source_track"]["observations"]) == 2


@pytest.mark.parametrize("tracks", [[], [_track(start=2)], [_track("a"), _track("b")], [_track(label="other-target")]])
def test_missing_or_ambiguous_instance_holds_editing(tracks):
    with pytest.raises(ValueError, match="track_ambiguous"):
        select_task_track(target=_target(), tracks=tracks, frames=[{"frame_id": "frame-0", "timestamp_seconds": 0}])


def test_masks_cannot_escape_the_image():
    observation = _track()["observations"][0]
    observation["runs"][0]["length"] = 17
    with pytest.raises(ValueError, match="mask_run_invalid"):
        decode_track_mask(observation)


def test_masked_geometry_preserves_world_position_and_estimated_scale(tmp_path):
    geometry_path = tmp_path / "geometry.npz"
    np.savez_compressed(geometry_path, depth_m=np.full((4, 4), 2.0), valid_mask=np.ones((4, 4), dtype=bool))
    pose = np.eye(4)
    pose[:3, 3] = [10, 20, 30]
    frame = {"frame_id": "frame-0", "geometry_path": str(geometry_path),
             "geometry_digest": _sha256_file(geometry_path), "intrinsics": np.eye(3).tolist(),
             "world_from_camera": pose.tolist()}
    bounds = estimate_target_bounds(_track(), [frame])
    assert bounds["center"] == [11, 21, 32]
    assert bounds["unit"] == "estimated_meters"
    assert bounds["metric_measurement_proven"] is False
    assert bounds["complete_object_dimensions"] is False
    geometry_path.write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="geometry_changed"):
        estimate_target_bounds(_track(), [frame])


@pytest.mark.parametrize("task_frame_refinement", [False, True])
def test_retained_masks_gain_estimated_bounds_without_retracking(tmp_path, monkeypatch, task_frame_refinement):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_task_masks import bind_task_masks_to_geometry
    geometry_path = tmp_path / "depth.npz"
    np.savez_compressed(geometry_path, depth_m=np.ones((4, 4)), valid_mask=np.ones((4, 4), dtype=bool))
    frame = {"frame_id": "frame-0", "width": 4, "height": 4, "geometry_path": str(geometry_path),
             "geometry_digest": _sha256_file(geometry_path), "intrinsics": np.eye(3).tolist(),
             "world_from_camera": np.eye(4).tolist()}
    geometry = {"frames": [frame], "binding": {"source_video_digest": "video", "input_digest": "inputs"}}
    geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    masks = {"source_geometry_digest": None, "source_video_digest": "video",
             "binding": {"geometry_input_digest": "inputs"},
             "targets": [{"target_id": "cup", "source_track": _track(), "estimated_visible_bounds": None}]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    if task_frame_refinement:
        geometry["binding"].update(input_digest="refined-inputs", tracking_input_digest="inputs", task_masks_digest=masks["digest"])
        geometry["digest"] = canonical_digest(geometry, digest_field="digest")
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.run_meta_sam31", lambda **kw: pytest.fail("must reuse SAM tracks"))
    bound = bind_task_masks_to_geometry(task_masks=masks, source_geometry=geometry)
    assert bound["targets"][0]["estimated_visible_bounds"]["unit"] == "estimated_meters"
    assert bound["source_geometry_digest"] == geometry["digest"]
    assert masks["targets"][0]["estimated_visible_bounds"] is None
    masks["binding"]["geometry_input_digest"] = "another-input"
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    with pytest.raises(ValueError, match="geometry_binding_mismatch"):
        bind_task_masks_to_geometry(task_masks=masks, source_geometry=geometry)


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("defer_geometry", [False, True])
def test_hosted_masks_use_original_upright_pixels_and_bind_to_depth(tmp_path, monkeypatch, multiple, defer_geometry):
    from PIL import Image
    from blueprint_pipeline.website_task_masks import run_website_task_masks

    original = tmp_path / "original.png"
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:2, 4:] = [0, 0, 255]
    Image.fromarray(pixels).save(original)
    geometry_path = tmp_path / "geometry.npz"
    np.savez_compressed(geometry_path, depth_m=np.ones((4, 2)), valid_mask=np.ones((4, 2), dtype=bool))
    frame = {"frame_id": "frame-0", "timestamp_seconds": 0,
             "source_image_path": str(original), "source_image_digest": _sha256_file(original),
             "display_rotation_degrees": 90, "width": 2, "height": 4,
             "geometry_path": str(geometry_path), "geometry_digest": _sha256_file(geometry_path),
             "intrinsics": np.eye(3).tolist(), "world_from_camera": np.eye(4).tolist()}
    target = {**_target(), "semantic_label": "small container beside picture", "segmentation_prompt": "blue object",
              "task_effect": "manipulated", "disposition": "remove"}
    calls = []

    def hosted(**kwargs):
        calls.append(kwargs)
        assert len(kwargs["prompts"]) == 1
        assert kwargs["prompts"][0]["text"] == "blue object"
        assert (kwargs["frame_registry"][0]["width"], kwargs["frame_registry"][0]["height"]) == (4, 8)
        with Image.open(kwargs["frame_artifacts"][0]["path"]) as submitted:
            assert submitted.size == (4, 8)
            assert submitted.getpixel((0, 0))[2] > 240
        tracks = [{"track_id": "selected", "label": "task-cup", "observations": [
            {"source_frame_id": "frame-0", "width": 4, "height": 8,
             "runs": [{"start": y * 4, "length": 2} for y in range(4)]}]}]
        if multiple:
            tracks.append({"track_id": "second", "label": "task-cup", "observations": [
                {"source_frame_id": "frame-0", "width": 4, "height": 8,
                 "runs": [{"start": y * 4 + 2, "length": 2} for y in range(4, 8)]}]})
        return {"tracks": tracks}

    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.run_meta_sam31", hosted)
    targets = [target]
    if multiple:
        targets.append({**target, "target_id": "second-task-cup", "spatial_evidence": [
            {"timestamp_seconds": 0, "box_xywh_normalized": [0.5, 0.5, 0.5, 0.5]}]})
    kwargs = dict(plan={"targets": targets, "task_context_sha256": "task"},
                  source_geometry={"digest": "geometry", "frames": [frame],
                                   "binding": {"source_video_digest": "video"}}, output_root=tmp_path / "masks")
    if defer_geometry:
        kwargs["source_geometry"]["geometry_available"] = False
        kwargs["source_geometry"]["frames"] = [{k: v for k, v in frame.items()
            if k not in {"geometry_path", "geometry_digest", "intrinsics", "world_from_camera"}}]
    result = run_website_task_masks(**kwargs)
    observation = result["targets"][0]["track"]["observations"][0]
    assert result["targets"][0]["source_track"]["observations"][0]["width"] == 4
    np.testing.assert_array_equal(decode_track_mask(observation), [[True, False], [True, False], [False, False], [False, False]])
    assert observation["source_mask_width"] == 4
    if defer_geometry:
        assert result["source_geometry_digest"] is None
        assert result["targets"][0]["estimated_visible_bounds"] is None
    else:
        assert result["targets"][0]["estimated_visible_bounds"]["metric_measurement_proven"] is False
    if multiple:
        assert [t["track"]["track_id"] for t in result["targets"]] == ["selected", "second"]
    original.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source_frame_changed"):
        run_website_task_masks(**kwargs)
    assert len(calls) == 1


def test_exact_frame_grounding_cannot_match_a_neighboring_timestamp():
    target = {**_target(), "grounding": {"source_frame_id": "exact"}}
    with pytest.raises(ValueError, match="track_ambiguous"):
        select_task_track(target=target, tracks=[_track()], frames=[
            {"frame_id": "exact", "timestamp_seconds": 0}, {"frame_id": "frame-0", "timestamp_seconds": 0.03}])


@pytest.mark.parametrize("recovery", ["anchor", "concept", "crop", "image", "unchanged", "wrong_instance"])
def test_controller_recovers_ambiguous_video_anchor_with_bounded_exact_frame_evidence(tmp_path, monkeypatch, recovery):
    from blueprint_pipeline import website_task_masks as masks, website_task_grounding as grounding
    target = {**_target(), "semantic_label": "white support", "segmentation_prompt": "white container",
              "task_effect": "static_contact", "disposition": "keep"}
    target["spatial_evidence"][0]["box_xywh_normalized"] = [0, 0.5, 0.5, 0.5]
    frame = {"frame_id": "frame-0", "timestamp_seconds": 0, "width": 4, "height": 4}
    registry = [{"source_frame_id": "frame-0", "decoded_pts_seconds": 0, "width": 4, "height": 4}]
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    calls = []
    def hosted(**kw):
        calls.append(kw)
        wrong = (recovery != "anchor" and len(calls) == 1) or recovery in {"image", "wrong_instance"}
        return {"tracks": [_track(start=2 if wrong else 0)]}
    monkeypatch.setattr(masks, "run_meta_sam31", hosted)
    grounds = []
    def ground(**kw):
        grounds.append(kw)
        concept = "white container" if recovery == "unchanged" or (recovery == "crop" and len(grounds) == 1) else "white book"
        return {**target, **_target(), "segmentation_prompt": concept, "grounding": {"source_frame_id": "frame-0"}}
    monkeypatch.setattr(grounding, "ground_task_target", ground)
    image_calls = []
    def image_fallback(**kw):
        image_calls.append(kw)
        if recovery == "wrong_instance":
            raise ValueError("task_target_track_ambiguous:task-cup")
        assert recovery == "image" and kw["target"]["disposition"] == "keep"
        return _track()
    monkeypatch.setattr(masks, "segment_grounded_static_target", image_fallback)
    kwargs = dict(plan={"targets": [target], "task_context_sha256": "task"},
        source_geometry={"digest": "source", "geometry_available": False,
                         "binding": {"source_video_digest": "video"}, "frames": [frame]},
        source_video=tmp_path / "source.mov", task_context={"confirmed": True}, output_root=tmp_path / "masks")
    if recovery in {"unchanged", "wrong_instance"}:
        with pytest.raises(ValueError, match="track_ambiguous"):
            masks.run_website_task_masks(**kwargs)
        assert len(calls) == (1 if recovery == "unchanged" else 2)
        assert len(grounds) == (2 if recovery == "unchanged" else 1)
        return
    result = masks.run_website_task_masks(**kwargs)
    assert len(image_calls) == (1 if recovery == "image" else 0)
    assert len(grounds) == (2 if recovery == "crop" else 1)
    assert len(calls) == (1 if recovery == "anchor" else 2)
    if recovery == "crop":
        assert grounds[1]["failed_segmentation_prompt"] == "white container"
    if recovery != "anchor":
        assert calls[1]["prompts"][0]["text"] == "white book"
        assert calls[1]["video_artifact"] == calls[0]["video_artifact"]
    assert result["targets"][0]["disposition"] == "keep"
    assert result["targets"][0]["source_track"]["observations"][0]["runs"][0]["start"] == 0
    assert result["targets"][0]["grounding"]["source_frame_id"] == "frame-0"


@pytest.mark.parametrize("case", ["match", "empty", "wrong_instance", "bad_grid", "changed_image", "manipulated"])
def test_static_source_crop_maps_masks_back_without_inventing_temporal_coverage(tmp_path, monkeypatch, case):
    from PIL import Image
    from blueprint_pipeline import website_task_masks as masks

    image = tmp_path / "original.png"
    Image.new("RGB", (100, 100), "white").save(image)
    target = {"target_id": "support", "task_effect": "static_contact", "disposition": "keep",
              "segmentation_prompt": "white book", "grounding": {"source_frame_id": "decoded-41",
                  "source_image_path": str(image), "image_digest": _sha256_file(image)},
              "spatial_evidence": [{"timestamp_seconds": 1.367, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1]}]}
    registry = [{"source_frame_id": "decoded-41", "model_frame_index": 41, "decoded_pts_seconds": 1.367,
                 "width": 100, "height": 100}]
    calls = []
    def hosted(**kw):
        calls.append(kw)
        assert "video_artifact" not in kw
        assert kw["prompts"][0]["text"] == "book"
        assert kw["task_context"] == {"confirmed": True}
        frame = kw["frame_registry"][0]
        assert frame["model_frame_index"] == 0 and frame["source_frame_id"] == "decoded-41"
        left, top, right, bottom = frame["source_crop_box_pixels"]
        w, h = right - left, bottom - top
        x, y, width, height = (20 - left, 30 - top, 20, 10) if case != "wrong_instance" else (0, 0, 2, 2)
        observation = {"source_frame_id": "decoded-41", "width": w, "height": h + (case == "bad_grid"),
                       "runs": [{"start": row * w + x, "length": width} for row in range(y, y + height)]}
        return {"binding_digest": "sha256:provider", "tracks": [] if case == "empty" else [
            {"track_id": "support-0", "label": "support", "observations": [observation]}]}
    monkeypatch.setattr(masks, "run_meta_sam31", hosted)
    if case == "changed_image":
        image.write_bytes(b"changed")
    if case == "manipulated":
        target.update(task_effect="manipulated", disposition="remove")
    kwargs = dict(target=target, registry=registry, task_context={"confirmed": True}, output_root=tmp_path / "masks")
    if case != "match":
        errors = {"empty": "track_ambiguous", "wrong_instance": "track_ambiguous", "bad_grid": "mapping_invalid",
                  "changed_image": "image_changed", "manipulated": "requires_static_contact"}
        with pytest.raises(ValueError, match=errors[case]):
            masks.segment_grounded_static_target(**kwargs)
        assert len(calls) == (0 if case in {"changed_image", "manipulated"} else 1)
        return
    result = masks.segment_grounded_static_target(**kwargs)
    assert result["coverage"] == "single_observed_frame" and len(result["observations"]) == 1
    mask = masks.decode_track_mask(result["observations"][0])
    assert mask.shape == (100, 100) and mask.sum() == 200 and mask[30:40, 20:40].all()
    assert result["observations"][0]["source_frame_id"] == "decoded-41"


@pytest.mark.parametrize("support_disposition,support_effect", [("keep", "static_contact"), ("remove", "static_contact"), ("keep", "manipulated")])
def test_visual_masks_defer_kept_support_but_cannot_enter_simulation(tmp_path, monkeypatch, support_disposition, support_effect):
    from blueprint_pipeline import website_task_masks as masks
    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
    from blueprint_pipeline import website_task_grounding as grounding

    blue = {**_target(), "semantic_label": "blue box", "segmentation_prompt": "blue box",
            "task_effect": "manipulated", "disposition": "remove"}
    support = {**_target(), "target_id": "support", "semantic_label": "white support", "segmentation_prompt": "white book",
               "task_effect": support_effect, "disposition": support_disposition}
    registry = [{"source_frame_id": "frame-0", "decoded_pts_seconds": 0, "width": 4, "height": 4}]
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    provider_calls = []
    def hosted(**kw):
        provider_calls.append(kw)
        return {"tracks": [_track()]}
    monkeypatch.setattr(masks, "run_meta_sam31", hosted)
    def unresolved(**kw):
        raise ValueError("support_not_resolved")
    monkeypatch.setattr(grounding, "ground_task_target", unresolved)
    kwargs = dict(plan={"targets": [blue, support], "task_context_sha256": "task"},
        source_geometry={"digest": "source", "geometry_available": False, "binding": {"source_video_digest": "video"},
                         "frames": [{"frame_id": "frame-0", "timestamp_seconds": 0, "width": 4, "height": 4}]},
        source_video=tmp_path / "source.mov", task_context={"confirmed": True}, output_root=tmp_path / "masks")
    if support_disposition != "keep" or support_effect == "manipulated":
        with pytest.raises(ValueError, match="support_not_resolved"):
            masks.run_website_task_masks(**kwargs, defer_kept_static=True)
        assert not list((tmp_path / "masks").rglob("task_masks.object_removal.json"))
        return
    result = masks.run_website_task_masks(**kwargs, defer_kept_static=True)
    assert result["status"] == "object_removal_ready" and result["deferred_target_ids"] == ["support"]
    assert [t["target_id"] for t in result["targets"]] == ["task-cup"]
    assert list((tmp_path / "masks").rglob("task_masks.object_removal.json"))
    assert not list((tmp_path / "masks").rglob("task_masks.json"))
    with pytest.raises(ValueError, match="static_task_masks_pending"):
        masks.bind_task_masks_to_geometry(task_masks=result, source_geometry={})
    with pytest.raises(ValueError, match="static_task_masks_pending"):
        compile_website_scene_preparation(task_context={}, task_masks=result, removal_manifest={}, source_geometry={},
            base_scene={}, output_root=tmp_path / "native", spend={}, now=0)
    with pytest.raises(ValueError, match="support_not_resolved"):
        masks.run_website_task_masks(**kwargs)
    assert provider_calls[0] == provider_calls[1]  # strict continuation reuses identical paid request bindings
