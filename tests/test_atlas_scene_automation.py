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


def test_reviewed_background_resume_reuses_images_before_any_provider_preflight(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    image = view(tmp_path)
    row = {"frame_id": "a", "image_path": image["image_path"],
           "image_digest": image["image_digest"], "original_image_path": image["image_path"],
           "original_image_digest": image["image_digest"]}
    review_binding = {"frames": [{"frame_id": "a", "image_digest": row["image_digest"]}]}
    background = {"status": "fallback_prepared", "ready_for_reconstruction": True,
                  "frames": [row], "review": {"status": "passed", "binding": review_binding,
                  "request_digest": canonical_digest(review_binding), "review": {
                      "consistent_background": True, "task_objects_removed": True,
                      "unrelated_objects_preserved": True}}}
    args = dict(frames=[image], objects=plan()["objects"], automation=plan(),
                inputs={"digest": "inputs"}, pose={"digest": "pose"}, root=tmp_path / "run",
                context={}, policy={}, source_digest="source", task_runner=None,
                release_blockers=lambda: ("commit", []))
    args["root"].mkdir()
    binding = {"source_video_digest": "source", "source_pose_digest": "pose", "inputs_digest": "inputs",
               "automation_digest": canonical_digest(plan()), "task_context_digest": canonical_digest({}),
               "selected_frame_ids": None, "editor_first": True}
    path = args["root"] / "prepared_background.json"
    edits._prepared_background(path, binding, background)
    monkeypatch.setattr(edits, "_provider_preflight", lambda: pytest.fail("bought completed edits again"))
    monkeypatch.setattr(edits, "object_evidence", lambda **_: pytest.fail("repeated completed masks"))
    assert edits.run_background(**args) == background
    with pytest.raises(ValueError, match="binding_changed"):
        edits.run_background(**{**args, "source_digest": "different-video"})
    Path(row["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="pixels_changed"):
        edits.run_background(**args)


def test_rejected_or_unbound_background_cannot_be_adopted(tmp_path):
    for review in ({"status": "blocked"}, {"status": "passed", "binding": {"frames": []}}):
        with pytest.raises(ValueError, match="review_invalid"):
            edits._prepared_background(tmp_path / "prepared.json", {}, {
                "status": "fallback_prepared", "ready_for_reconstruction": True,
                "frames": [{"frame_id": "a", "image_digest": "sha256:a"}], "review": review})
    assert not (tmp_path / "prepared.json").exists()


def test_source_cleanup_needs_no_pose_and_keeps_only_current_reviewed_frames(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    a, b = view(tmp_path), view(tmp_path, "b", 0.1)
    inputs = {"frames": [{k: f[k] for k in ("frame_id", "image_path", "image_digest")} for f in (a, b)],
              "binding": {"source_video_digest": "source"}, "digest": "inputs"}
    observed = []

    def reject(**kw):
        observed.append(kw)
        assert [f["frame_id"] for f in kw["frames"]] == ["a"]
        assert "depth" not in kw["frames"][0] and "extrinsics" not in kw["frames"][0]
        assert not kw["frames"][0]["camera_calibrated"]
        return {"status": "quality_rejected"}

    monkeypatch.setattr(edits, "run_background", reject)
    result = worker.run_automation(pose_root=None, inputs=inputs,
        review={"automation": plan(), "coverage_requirements": [{"required_frame_ids": ["a"]}]},
        output_root=tmp_path / "out", task_context={}, configuration={"policy": {}},
        task_runner=lambda **_: pytest.fail("rejected image reached Atlas"),
        pose_runner=lambda **_: pytest.fail("rejected image reached Atlas"), release_blockers=None)
    assert result["status"] == "quality_rejected" and result["source_pose_digest"] is None
    assert observed[0]["selected_frame_ids"] == ["a"]
    assert result["digest"] == canonical_digest(result, digest_field="digest")


def test_accepted_rgb_passes_real_pose_capture_boundary_before_any_paid_runner(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker, website_atlas_pose as atlas

    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, "")
    frame, second = view(tmp_path, width=80, height=45), view(tmp_path, "b", .1, width=80, height=45)
    root = tmp_path / "job"
    legacy = root / "fallback_pose/atlas_pose_inputs.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"refused_legacy_handoff": true}')
    called = []

    def validate(**kw):
        inputs = atlas.load_pending_atlas_pose_inputs(kw["output_root"])
        assert inputs["generated_input_content"] is True
        assert Path(inputs["frames"][0]["image_path"]).is_relative_to(kw["output_root"])
        assert inputs["full_view_sources"][0]["image_digest"] == frame["image_digest"]
        from PIL import Image
        from blueprint_pipeline.atlas_context_fov import full_view_sources
        full_view_sources(inputs, kw["output_root"])
        assert Image.open(inputs["frames"][0]["image_path"]).width == Image.open(inputs["frames"][0]["image_path"]).height
        assert atlas.atlas_pose_target_resolution(inputs) == [1280, 720]
        called.append(kw)
        return {"status": "processing", "operation_id": "not-purchased"}

    result, state = worker._fallback_pose({"frames": [frame, second]}, root, "source", {"capture_id": "c"},
        {"pose_input_cost_usd": .01, "rate_card_reference": "reviewed", "reviewed_by": "test",
         "reviewed_on": "2026-10-22"}, validate, 2)
    assert result is None and state["status"] == "processing" and len(called) == 1
    assert legacy.read_text() == '{"refused_legacy_handoff": true}'
    (legacy.parent / "atlas_pose").mkdir()
    (legacy.parent / "atlas_pose/submission.json").write_text('{"status":"submitting"}')
    with pytest.raises(ValueError, match="legacy_pose_requires_reconciliation"):
        worker._fallback_pose({"frames": [frame]}, root, "source", {}, {}, validate, 2)
    assert len(called) == 1


def test_rejected_atlas_fallback_enters_existing_repair_and_retains_passed_set(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit, website_image_repair_agent as repair
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    monkeypatch.setattr(edits, "segment_objects", lambda **_: [{**plan()["objects"][0], **track(frames)}])
    monkeypatch.setattr(image_edit, "_validated_backend", lambda *a, **k:
                        ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"))
    monkeypatch.setattr(image_edit, "complete_background_images", lambda **k: [{
        **row, "original_image_path": row["image_path"], "original_image_digest": row["image_digest"],
    } for row in k["frames"]])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **k:
                        {"status": "passed" if k["output_root"].name == "anchor_review" else "blocked"})
    monkeypatch.setattr(repair, "image_repair_enabled", lambda: True)
    calls = []

    def repaired(**kw):
        calls.append(kw)
        binding = {"frames": [{"frame_id": r["frame_id"], "image_digest": r["image_digest"]}
                              for r in kw["selected"]]}
        return kw["selected"], {"status": "passed", "binding": binding,
                                "request_digest": canonical_digest(binding), "review": {
                                    "consistent_background": True, "task_objects_removed": True,
                                    "unrelated_objects_preserved": True}}

    monkeypatch.setattr(repair, "repair_rejected_views", repaired)

    def task(**kw):
        return {"status": "completed", "outputs": [
            {"role": f"frames-{i}-imageAsset", "path": "unused.png", "sha256": "unused"}
            for i in range(len(kw["request"]["contextFrames"]))]}

    args = dict(frames=frames, objects=plan()["objects"], automation=plan(), editor_first=False,
                inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]},
                pose={"digest": "pose"}, root=tmp_path / "run", context={},
                policy={"rgbd_view_cost_usd": 0.05, "rate_card_reference": "reviewed",
                        "reviewed_by": "operator", "reviewed_on": "2026-10-22"},
                source_digest="video", task_runner=task,
                release_blockers=lambda: ("commit", []), configuration={"remaining_cost_usd": 10})
    result = edits.run_background(**args)
    assert result["status"] == "fallback_prepared" and len(calls) == 1
    assert calls[0]["max_repairs"] == 2 and calls[0]["source_video_digest"] == "video"
    assert calls[0]["repair_reference_path"].is_file()
    assert edits.run_background(**args) == result and len(calls) == 1


@pytest.mark.parametrize("makes_progress", [True, False])
def test_editor_repairs_follow_latest_review_and_stop_when_pixels_do_not_change(tmp_path, monkeypatch, makes_progress):
    from blueprint_pipeline import website_image_completion as image_edit, website_image_repair_agent as repair
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    prepared = edits.prepare_masked_frames(frames=frames, targets=[track(frames)], margin=.01,
                                           root=tmp_path / "originals")
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(image_edit, "complete_background_images", lambda **kw: kw["frames"])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **_: {"status": "blocked"})
    monkeypatch.setattr(repair, "image_repair_enabled", lambda: True)
    calls = []

    def fix(**kw):
        calls.append(kw)
        selected = list(map(dict, kw["selected"]))
        if makes_progress:
            index = len(calls) - 1
            path = tmp_path / f"repair-{index}.png"
            Image.new("RGB", (80, 60), "red" if index == 0 else "blue").save(path)
            selected[index].update(image_path=str(path), image_digest=_sha256_file(path))
        binding = {"frames": [{"frame_id": row["frame_id"], "image_digest": row["image_digest"]}
                              for row in selected]}
        passed = makes_progress and len(calls) == 2
        review = {"status": "passed" if passed else "blocked", "binding": binding,
                  "request_digest": canonical_digest(binding), "review": {
                      "consistent_background": passed, "task_objects_removed": True,
                      "unrelated_objects_preserved": True}}
        return selected, review

    monkeypatch.setattr(repair, "repair_rejected_views", fix)
    result = edits._reviewed_editor_background(prepared=prepared, anchor=prepared[0],
        plan={"task_context_sha256": "task", "targets": []}, automation=plan(), root=root, context={},
        source_digest="video", measurements={}, selected_frame_ids=None, check=lambda _: None, prepared_binding={})
    assert len(calls) == (2 if makes_progress else 1)
    assert result["status"] == ("fallback_prepared" if makes_progress else "quality_rejected")
    if makes_progress:
        assert calls[1]["failed_review"]["status"] == "blocked"
        assert calls[1]["selected"][0]["image_path"].endswith("repair-0.png")
        assert calls[1]["output_root"].name == "fallback_repairs-0002"
    assert (root / "prepared_background.json").exists() is makes_progress


def _sam_person_tracks(frames):
    """Two people in the first view (one instance each), nobody in the second."""
    first = frames[0]
    h, w = first["rgb"].shape[:2]
    people = []
    for index, columns in enumerate([(2, 10), (60, 70)]):
        runs = [{"start": y * w + columns[0], "length": columns[1] - columns[0]} for y in range(10, 50)]
        people.append({"track_id": f"p{index}", "label": "transient-0", "observations": [
            {"source_frame_id": first["frame_id"], "width": w, "height": h, "runs": runs}]})
    return people


def _transient_run(tmp_path, monkeypatch, tracks):
    from blueprint_pipeline import meta_sam31, website_image_completion as image_edit

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    for frame in frames:
        del frame["depth"]  # Editor-first runs before any depth is bought.
    prompts, edits_seen = [], []
    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    monkeypatch.setattr(meta_sam31, "run_meta_sam31",
                        lambda **kw: prompts.extend(kw["prompts"]) or {"tracks": tracks(frames)})
    monkeypatch.setattr(image_edit, "_validated_backend", lambda *a, **k:
                        ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"))
    monkeypatch.setattr(image_edit, "complete_background_images",
                        lambda **k: edits_seen.append(k) or k["frames"])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **_: {"status": "blocked"})
    result = edits.run_background(
        frames=frames, objects=[], automation={**plan(), "objects": []},
        inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]}, pose={"digest": "pose"},
        root=tmp_path / "run", context={}, policy={"rgbd_view_cost_usd": 0.05}, source_digest="video",
        task_runner=None, release_blockers=lambda: ("commit", []), configuration={"remaining_cost_usd": 10},
        transients=[{"target_id": "transient-0", "label": "person", "segmentation_prompt": "person"}])
    return result, prompts, edits_seen


def test_every_person_in_the_one_video_is_erased_even_with_no_task_object_to_remove(tmp_path, monkeypatch):
    result, prompts, edits_seen = _transient_run(tmp_path, monkeypatch, _sam_person_tracks)
    assert prompts == [{"prompt_id": "transient-0", "output_label": "transient-0", "text": "person"}]
    anchor = edits_seen[0]["frames"][0]
    # Both people are erased, not one chosen instance.
    assert anchor["frame_id"] == "a" and anchor["removed_transient_ids"] == ["transient-0"]
    assert anchor["removed_task_object_ids"] == [] and anchor["remaining_pixel_count"] > 2 * 8 * 40
    mask = np.asarray(Image.open(anchor["remaining_mask_path"])) == 255
    assert mask[30, 5] and mask[30, 65] and not mask[30, 35]
    empty = np.asarray(Image.open(tmp_path / "run/originals/0001-edit.png"))
    assert not empty.any()  # Nobody in the second view: nothing to edit there.
    transient = next(t for t in edits_seen[0]["targets"] if t["target_id"] == "transient-0")
    assert transient["task_effect"] == "transient" and transient["disposition"] == "remove"
    assert result["status"] == "quality_rejected" and result["stage"] == "anchor"


def test_a_scene_with_no_people_found_and_nothing_to_remove_buys_no_edit(tmp_path, monkeypatch):
    result, prompts, edits_seen = _transient_run(tmp_path, monkeypatch, lambda frames: [])
    assert result["status"] == "not_requested" and result["reason"] == "all_objects_kept"
    assert prompts and edits_seen == []


def test_an_object_track_missing_a_view_refuses_but_an_absent_transient_is_empty(tmp_path):
    a, b = view(tmp_path), view(tmp_path, "b", 0.1)
    partial = track([a])
    with pytest.raises(ValueError, match="mask_view_coverage_incomplete"):
        edits.prepare_masked_frames(frames=[a, b], targets=[partial], margin=0.01, root=tmp_path / "bad")
    prepared = edits.prepare_masked_frames(
        frames=[a, b], targets=[{**partial, "task_effect": "transient"}], margin=0.01, root=tmp_path / "ok")
    assert prepared[0]["removed_transient_ids"] == ["obj"] and prepared[1]["remaining_pixel_count"] == 0
    assert "removed_transient_ids" not in prepared[1]
