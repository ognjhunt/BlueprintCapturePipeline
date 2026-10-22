"""Automatic estimates/editing preserve raw geometry, source caps and quality rejection."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_measurements as sizes
from blueprint_pipeline import atlas_background_automation as edits
from blueprint_pipeline.atlas_automation_plan import AutomationPlan
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def view(tmp_path, key="a", position=0, width=80, height=60):
    camera = {
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 100.0,
            "fy": 100.0,
            "cx": width / 2,
            "cy": height / 2,
        },
        "extrinsics": {
            "position": [position, 0.0, 0.0],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
            "coordinateSystem": "rub",
        },
    }
    f = posed_frame(
        frame_id=key,
        rgb=np.full((height, width, 3), 90, np.uint8),
        depth=np.full((height, width), 2, np.float32),
        confidence=np.full((height, width), 255, np.uint8),
        camera=camera,
    )
    f["pixel_alignment"] = {"matrix": np.eye(3).tolist(), "source_grid": [width, height]}
    path = tmp_path / f"{key}.png"
    Image.fromarray(f["rgb"]).save(path)
    f.update(image_path=str(path), image_digest=_sha256_file(path))
    return f


def plan():
    return {
        "scale_references": [
            {
                "frame_id": "a",
                "label": "visible known-sized item",
                "endpoints_normalized": [[0.25, 0.5], [0.75, 0.5]],
                "assumed_distance_m": 0.8,
                "plausible_distance_range_m": [0.7, 0.9],
                "assumption": "class size assumption",
            }
        ],
        "objects": [
            {
                "target_id": "obj",
                "label": "task object",
                "segmentation_prompt": "task object",
                "disposition": "remove",
                "regions": [{"frame_id": "a", "box_xywh_normalized": [0.2, 0.2, 0.6, 0.6]}],
            }
        ],
        "removal_instruction": "Remove the complete task object and its parts; reveal the realistic empty space.",
        "mask_margin_fraction": 0.01,
    }


def track(frames):
    observations = []
    for f in frames:
        h, w = f["depth"].shape
        observations.append(
            {
                "source_frame_id": f["frame_id"],
                "width": w,
                "height": h,
                "runs": [
                    {"start": int(y * w + w // 4), "length": w // 2}
                    for y in range(h // 4, 3 * h // 4)
                ],
            }
        )
    return {
        "target_id": "obj",
        "label": "task object",
        "disposition": "remove",
        "track": {"observations": observations},
    }


def test_estimates_include_units_ranges_objects_and_preserve_raw_geometry(tmp_path):
    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    before = deepcopy(frames[0]["depth"])
    result = sizes.estimate_sizes(
        frames=frames,
        automation=plan(),
        pose_digest="pose",
        output=tmp_path / "estimate.json",
        tracks=[track(frames)],
    )
    assert result["meters_per_unit_estimate"] == pytest.approx(1.0)
    assert result["meters_per_unit_range"] == pytest.approx([0.875, 1.125])
    assert result["scene"]["extent_xyz_feet_estimate"] is not None
    assert result["objects"][0]["extent_xyz_m_estimate"] is not None
    assert result["metric_measurement_proven"] is False and result["raw_geometry_rescaled"] is False
    assert np.array_equal(frames[0]["depth"], before)
    assert (
        sizes.estimate_sizes(
            frames=frames,
            automation=plan(),
            pose_digest="pose",
            output=tmp_path / "estimate.json",
            tracks=[track(frames)],
        )
        == result
    )
    with pytest.raises(ValueError, match="binding_changed"):
        sizes.estimate_sizes(
            frames=frames,
            automation=plan(),
            pose_digest="changed",
            output=tmp_path / "estimate.json",
            tracks=[track(frames)],
        )


def test_missing_reference_depth_yields_relative_sizes_without_inventing_meters(tmp_path):
    f = view(tmp_path)
    f["depth"][:] = np.nan
    result = sizes.estimate_sizes(
        frames=[f], automation=plan(), pose_digest="pose", output=tmp_path / "none.json"
    )
    assert result["status"] == "relative_sizes_only" and result["meters_per_unit_estimate"] is None
    assert result["reference_gaps"][0]["reason"] == "atlas_reference_depth_missing"
    assert result["scene"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "change",
    [
        {"assumed_distance_m": float("nan")},
        {"endpoints_normalized": [[0, 0], [2, 1]]},
        {"plausible_distance_range_m": [1.0, 2.0]},
    ],
)
def test_invalid_scale_priors_cannot_reach_depth_sampling(change):
    value = plan()
    value["scale_references"][0].update(change)
    with pytest.raises(ValueError):
        AutomationPlan.model_validate(value)


def test_source_crop_is_checked_instead_of_silently_pairing_wrong_pixels(tmp_path):
    f = view(tmp_path)
    checked = sizes.source_to_posed(Path(f["image_path"]), f["rgb"])
    assert checked["method"] == "pixel_checked_resize"
    with pytest.raises(ValueError, match="alignment_unavailable"):
        sizes.source_to_posed(Path(f["image_path"]), np.full((60, 80, 3), 240, np.uint8))


def test_portrait_tiles_cover_full_height_and_transform_camera_rays(tmp_path):
    f = view(tmp_path, width=720, height=1280)
    prepared = edits.prepare_masked_frames(
        frames=[f], targets=[track([f])], margin=0.01, root=tmp_path / "masks"
    )
    tiles = edits.tile_views(
        frames=[f], prepared=prepared, anchor=prepared[0], root=tmp_path / "tiles"
    )
    assert len(tiles) == 4
    first, last = tiles[0], tiles[-1]
    assert first["pixel_transform"]["crop_xy"][1] == 0
    assert last["pixel_transform"]["crop_xy"][1] + 720 == round(1280 * 1280 / 720)
    for t in tiles:
        assert Image.open(t["keep_mask_path"]).mode == "L"
        assert Image.open(t["image_path"]).size == (1280, 720)
        assert np.asarray(Image.open(t["keep_mask_path"])).min() == 255
        k = t["camera"]["intrinsics"]
        sx, sy = t["pixel_transform"]["resize_xy"]
        x, y = t["pixel_transform"]["crop_xy"]
        assert k["fx"] == pytest.approx(f["intrinsics"]["fx"] * sx)
        assert k["cy"] == pytest.approx(f["intrinsics"]["cy"] * sy - y)


def test_batching_has_no_scene_frame_ceiling_and_every_batch_keeps_clean_anchor(tmp_path):
    a = view(tmp_path)
    b = view(tmp_path, "b", 0.2)
    prepared = edits.prepare_masked_frames(
        frames=[a, b], targets=[track([a, b])], margin=0.01, root=tmp_path / "masks"
    )
    tiles = edits.tile_views(
        frames=[a, b], prepared=prepared, anchor=prepared[0], root=tmp_path / "tiles"
    )
    rows = [
        {**tiles[i % len(tiles)], "frame_id": str(i), "clean_anchor": i == 0} for i in range(100)
    ]
    groups = edits.batches(rows)
    assert {r["frame_id"] for group in groups for r in group} == {str(i) for i in range(100)}
    assert all(g[0]["clean_anchor"] and len(g) <= 16 for g in groups)
    for group in groups:
        request, uploads = edits.masked_request(
            group, "Remove complete appliance; keep the surrounding room."
        )
        assert request["contextFrames"][0]["camera"] == request["targetCameras"][0]
        assert len(uploads) == 2 * len(group)


def test_empty_or_misaligned_mask_track_refuses_before_editing(tmp_path):
    f = view(tmp_path)
    t = track([f])
    t["track"]["observations"][0]["width"] = 79
    with pytest.raises(ValueError, match="grid_mismatch"):
        edits.prepare_masked_frames(frames=[f], targets=[t], margin=0.01, root=tmp_path / "bad")


def test_rejected_anchor_never_calls_atlas_or_splat_builder(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit

    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(
        edits, "segment_objects", lambda **_: [{**plan()["objects"][0], **track(frames)}]
    )
    monkeypatch.setattr(
        image_edit,
        "_validated_backend",
        lambda *a, **k: ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"),
    )
    monkeypatch.setattr(image_edit, "complete_background_images", lambda **k: k["frames"])
    monkeypatch.setattr(
        image_edit,
        "verify_completed_background",
        lambda **_: {"status": "blocked", "review": {"task_objects_removed": False}},
    )

    def forbidden(**_):
        raise AssertionError("Rejected pixels reached Atlas")

    stages = []
    result = edits.run_background(
        frames=frames,
        objects=plan()["objects"],
        automation=plan(),
        inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]},
        pose={"digest": "pose"},
        root=tmp_path / "run",
        context={},
        policy={"rgbd_view_cost_usd": 0.05},
        source_digest="source",
        task_runner=forbidden,
        release_blockers=lambda: ("commit", []),
        configuration={"remaining_cost_usd": 10},
        checkpoint=stages.append,
    )
    assert result["status"] == "quality_rejected" and result["stage"] == "anchor"
    assert stages == ["automation_segmentation", "automation_anchor", "automation_anchor_review"]


def test_insufficient_entire_path_budget_refuses_before_sam(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(
        image_edit,
        "_validated_backend",
        lambda *a, **k: ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"),
    )
    monkeypatch.setattr(
        edits, "segment_objects", lambda **_: (_ for _ in ()).throw(AssertionError("SAM called"))
    )
    with pytest.raises(ValueError, match="budget_insufficient"):
        edits.run_background(
            frames=frames,
            objects=plan()["objects"],
            automation=plan(),
            inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]},
            pose={},
            root=tmp_path / "run",
            context={},
            policy={"rgbd_view_cost_usd": 0.05},
            source_digest="source",
            task_runner=None,
            release_blockers=lambda: ("commit", []),
            configuration={"remaining_cost_usd": 0},
        )


def test_failed_automation_preserves_measurement_result_without_reconstructing(
    tmp_path, monkeypatch
):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline import atlas_depth_splat

    f = view(tmp_path)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [f]))
    monkeypatch.setattr(
        edits, "run_background", lambda **_: {"status": "quality_rejected", "stage": "anchor"}
    )
    monkeypatch.setattr(
        atlas_depth_splat,
        "write_depth_splat",
        lambda **_: pytest.fail("rejected edit was reconstructed"),
    )
    result = worker.run_automation(
        pose_root=tmp_path,
        inputs={},
        review={"automation": plan()},
        output_root=tmp_path / "out",
        task_context={},
        configuration={"policy": {}},
        task_runner=lambda **_: pytest.fail("provider called"),
        pose_runner=None,
        release_blockers=lambda: ("commit", []),
    )
    assert (
        result["status"] == "quality_rejected" and result["measurements"]["status"] == "estimated"
    )
    assert result["raw_capture_preserved"] and not result["simulation_qualified"]


def test_boxes_are_clipped_in_returned_grid_and_outside_boxes_refuse_before_sam(tmp_path):
    f = view(tmp_path)
    f["pixel_alignment"]["matrix"][0][2] = 50
    args = dict(frames=[f], objects=plan()["objects"], inputs={"frames": [{"frame_id": "a"}]})
    box = edits.object_evidence(**args)["obj"][0]["box_xywh_normalized"]
    assert box[0] + box[2] == pytest.approx(1)
    f["pixel_alignment"]["matrix"][0][2] = 500
    with pytest.raises(ValueError, match="outside_returned_view"):
        edits.object_evidence(**args)


def test_offline_measurements_retain_public_citation_lineage(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker

    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [view(tmp_path)]))
    source = {
        "frame_id": "a",
        "label": plan()["scale_references"][0]["label"],
        "url": "https://manufacturer.example/spec",
        "status": "unconfirmed_reference",
    }
    result = worker.run_automation(
        pose_root=tmp_path,
        inputs={},
        review={"automation": plan(), "scale_reference_sources": [source], "anchor_frame_id": "a"},
        output_root=tmp_path / "offline",
        task_context={},
        configuration={},
        execute=False,
        task_runner=lambda **_: pytest.fail("offline provider call"),
        pose_runner=None,
        release_blockers=None,
    )
    assert result["measurements"]["scale_references"][0]["public_reference_readback"] == source


def test_passed_edit_constructs_and_exports_generated_splat_after_review(tmp_path, monkeypatch):
    from blueprint_pipeline import (
        atlas_scene_automation as worker,
        atlas_depth_splat,
        atlas_result_bundle,
    )

    f = view(tmp_path)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [f]))
    monkeypatch.setattr(
        edits,
        "run_background",
        lambda **_: {
            "status": "passed",
            "provider": "atlas",
            "stage_roots": [],
            "result_digests": ["edited-task"],
            "visual_review_digest": "review",
            "batch_frame_ids": [],
        },
    )
    monkeypatch.setattr(worker, "_edited_frames", lambda _: [f])
    calls = []

    def build(**kw):
        calls.append(kw)
        assert (
            kw["generated_result_digests"] == ["edited-task"]
            and kw["visual_review_digest"] == "review"
        )
        return {"digest": "generated-splat"}

    monkeypatch.setattr(atlas_depth_splat, "write_depth_splat", build)
    monkeypatch.setattr(
        atlas_result_bundle,
        "export_pose_bundle",
        lambda **kw: {"zip_path": "generated-background.zip"},
    )
    result = worker.run_automation(
        pose_root=tmp_path,
        inputs={},
        review={"automation": plan()},
        output_root=tmp_path / "out",
        task_context={},
        configuration={"policy": {}},
        task_runner=lambda **_: pytest.fail("unexpected provider call"),
        pose_runner=None,
        release_blockers=lambda: ("commit", []),
    )
    assert result["status"] == "completed" and len(calls) == 1
    assert result["background"]["download"]["zip_path"] == "generated-background.zip"
    assert result["background"]["mesh_available"] is False


def test_public_reference_quote_requires_retained_page_text_and_matching_units(tmp_path):
    from blueprint_pipeline.atlas_reference_research import retain_references

    class Fetcher:
        log = []

        def fetch(self, url):
            path = tmp_path / "public-page.txt"
            path.write_text("Range width: 30 inches. Category dimensions vary by model.")
            self.log.append({"url": url, "status": "ok", "text_path": str(path)})
            return {"status": "ok", "url": url}

    proposal = {"automation": plan()}
    reference = proposal["automation"]["scale_references"][0]
    reference.update(
        assumed_distance_m=0.762,
        citation_url="https://manufacturer.example/spec",
        citation_quote="Range width: 30 inches.",
    )
    valid = retain_references(proposal, tmp_path, fetcher=Fetcher())
    assert (
        valid[0]["status"] == "retained_public_reference" and not valid[0]["physical_measurement"]
    )
    reference["citation_quote"] = "Range width: 99 inches."
    invalid = retain_references(proposal, tmp_path, fetcher=Fetcher())
    assert invalid[0]["status"] == "unconfirmed_reference" and invalid[0]["quote_found"] is False
