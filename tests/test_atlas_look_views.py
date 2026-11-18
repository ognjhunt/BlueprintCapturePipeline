"""Generated views of weakly observed regions, checked against the capture, for the room's look.

Owner decision, 2026-11-18: generated Atlas views may enter the refined splat's input when the real
capture and its depth are the base and the views agree with them. Synthetic frames: a gradient plane
and an observed box (``tests.test_atlas_gap_composite``), filmed from one direction.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat as depth
from blueprint_pipeline import atlas_look_views as look
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_depth_splat import camera
from tests.test_atlas_gap_composite import real_frames, scene

CENTROID = [0.05, 0.0, -1.5]
POLICY = {"rgbd_view_cost_usd": .05, "additional_rgbd_view_cost_usd": .01, "rate_card_reference": "card",
          "reviewed_by": "owner", "reviewed_on": "2026-11-18", "refined_splat_cost_usd": 5, "mesh_export_cost_usd": .5}
POSE = {"digest": "sha256:" + "a" * 64}


def region(region_id="table", **changes):
    return {"id": region_id, "centroid": CENTROID, "extent": [.3, .3, .1], "angle_spread_deg": 2.9,
            "mm_per_px": 1.2, "flatness": .1, "view_count": 3, "seen_by": ["real-0", "real-1", "real-2"],
            "orbit_deg": [15, 30], **changes}


def write_map(capture, regions, **fields):
    path = capture / look.WEAK_REGIONS_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": look.WEAK_REGIONS_SCHEMA, "regions": regions, **fields}))
    return path


class Atlas:
    """``atlasGenerate`` behind the paid admission: one purchase per stage, then polled and adopted."""

    def __init__(self, *, polls_before_done=0):
        self.purchases, self.calls, self.polls_before_done = [], [], polls_before_done

    def __call__(self, *, task, request, uploads, output_root, source_video_digest, task_context, price_review,
                 max_polls):
        assert task == "atlasGenerate" and price_review["request_digest"] == canonical_digest(request)
        self.calls.append((output_root, price_review["request_digest"]))
        submission = output_root / "atlas_pose/submission.json"
        if not submission.exists():
            self.purchases.append((output_root, price_review["maximum_cost_usd"]))
            submission.parent.mkdir(parents=True, exist_ok=True)
            submission.write_text(json.dumps({"status": "submitted", "request_digest": price_review["request_digest"]}))
        polled = sum(1 for root, _ in self.calls if root == output_root)
        if polled <= self.polls_before_done:
            return {"status": "processing", "operation_id": "op-" + output_root.name}
        if not (output_root / "result.json").exists():
            outputs, frames = [], []
            for index, target in enumerate(request["targetCameras"]):
                image, exr = output_root / f"{index}.png", output_root / f"{index}.exr"
                cv2.imwrite(str(image), np.full((720, 1280, 3), 90 + index, np.uint8))
                depth.write_exr_depth(exr, np.full((720, 1280), 1.5, np.float32))
                for role, path in ((f"frames-{index}-imageAsset", image), (f"frames-{index}-depth-depthAsset", exr)):
                    outputs.append({"role": role, "path": path.name, "sha256": look._sha256_file(path)})
                frames.append({"camera": target})
            result = {"status": "completed", "outputs": outputs, "operation": {"id": "op", "response": {"frames": frames}},
                      "billing": {"reserved_usd": price_review["maximum_cost_usd"]}}
            result["digest"] = canonical_digest(result, digest_field="digest")
            (output_root / "result.json").write_text(json.dumps(result))
        return {"status": "completed"}


def run(output_root, atlas, *, frames=None, bought=False, remaining=20.0, policy=POLICY):
    return look.look_views_for_splat(frames=frames or real_frames(), pose=POSE, output_root=output_root,
        policy=policy, task_context={"context_digest": "t"}, source_digest="sha256:" + "s" * 64,
        configuration=lambda: {"policy": policy, "remaining_cost_usd": remaining}, task_runner=atlas,
        bought=bought, check=lambda *_: None, max_polls=1)


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A website capture's Atlas root with its weak-region map; the check passes the views turned right."""
    capture = tmp_path / "capture"
    write_map(capture, [region()])
    monkeypatch.setattr(look, "check_view", lambda view, references, footprint: (
        {"passed": True, "checked_fraction": .6, "contradicted_fraction": .01}
        if view["extrinsics"]["position"][0] > CENTROID[0] else
        {"passed": False, "reason": "contradicts_filmed_frames", "checked_fraction": .6,
         "contradicted_fraction": .4}))
    return capture / "pipeline/atlas"


# --- cameras ---------------------------------------------------------------------------------------------------

def test_orbit_cameras_keep_the_filmed_distance_height_and_intrinsics_and_look_at_the_centroid():
    frames = real_frames()
    targets, skipped = look.orbit_cameras([region()], frames)
    assert skipped == [] and [row["orbit_degrees"] for row in targets] == [-15, 15, -30, 30]
    base = next(f for f in frames if f["frame_id"] == targets[0]["base_frame_id"])
    centroid = np.asarray(CENTROID)
    distance = np.linalg.norm(base["position"] - centroid)
    for row in targets:
        e, k = row["camera"]["extrinsics"], row["camera"]["intrinsics"]
        position = np.asarray(e["position"])
        assert np.linalg.norm(position - centroid) == pytest.approx(distance)
        assert position[1] == pytest.approx(base["position"][1])
        # The view direction points at the centroid, and the turn from the filmed camera is the orbit angle.
        forward = -depth._rotation_matrix(e["quaternion"])[:, 2]
        assert forward @ (centroid - position) / np.linalg.norm(centroid - position) == pytest.approx(1)
        before, after = base["position"] - centroid, position - centroid
        turned = math.degrees(math.acos(before @ after / np.linalg.norm(before) / np.linalg.norm(after)))
        assert turned == pytest.approx(abs(row["orbit_degrees"]), abs=1e-6)
        # Atlas's target grid, the filmed camera's rays: the portrait frame turned a quarter.
        assert (k["width"], k["height"], k["cx"], k["cy"]) == (1280, 720, 640, 360)
        assert k["fx"] == pytest.approx(base["intrinsics"]["fy"] * 720 / 48)
        assert row["camera_id"].startswith("look-") and row["context_frame_ids"][0] == base["frame_id"]
        # The region projects in the middle of the view, and is only part of it.
        footprint = look.region_footprint(row)
        assert footprint[360, 640] and 0 < footprint.mean() < 1


def test_regions_and_views_are_capped_weakest_first_and_unposed_regions_are_named():
    regions = [region(f"r{i}", angle_spread_deg=float(i)) for i in range(8)]
    regions.append(region("ghost", angle_spread_deg=0.1, seen_by=["not-posed"]))
    targets, skipped = look.orbit_cameras(regions, real_frames())
    kept = {row["region_id"] for row in targets}
    assert kept == {"r0", "r1", "r2", "r3", "r4", "r5"} and len(targets) == look.MAX_VIEWS == 24
    assert {"region_id": "ghost", "reason": "seen_by_cameras_not_posed"} in skipped
    assert {row["region_id"] for row in skipped if row["reason"] == "over_region_limit"} == {"r6", "r7"}
    assert len(look.orbit_cameras(regions, real_frames(), max_views=6)[0]) == 6
    assert look.orbit_angles([-30, 30]) == [-30, 30] and look.orbit_angles([0, 20]) == [-20, 20]


def test_the_map_is_read_only_when_it_is_the_contract(tmp_path):
    assert look.load_weak_regions(tmp_path) is None
    path = write_map(tmp_path, [region(), {"id": "broken", "centroid": [0, 0]}])
    assert [row["id"] for row in look.load_weak_regions(tmp_path)] == ["table"]
    path.write_text(json.dumps({"schema_version": "other.v1", "regions": [region()]}))
    assert look.load_weak_regions(tmp_path) is None
    path.write_text("{")
    assert look.load_weak_regions(tmp_path) is None


# --- the consistency check -------------------------------------------------------------------------------------

def test_a_view_that_agrees_with_the_filmed_frames_passes_and_contradicting_or_unchecked_views_fail():
    from blueprint_pipeline.atlas_gap_composite import observation_references
    references = observation_references(real_frames())
    cam = camera((.1, .05, .1), yaw_degrees=4)
    agreeing = look.check_view(scene(cam, "look-agrees"), references)
    assert agreeing["passed"] and agreeing["checked_fraction"] >= look.MIN_CHECKED_FRACTION
    assert agreeing["contradicted_fraction"] <= look.MAX_CONTRADICTED_FRACTION
    deeper = scene(cam, "look-deeper")
    deeper["depth"] = deeper["depth"] * 1.2
    deeper = look.check_view(deeper, references)
    assert not deeper["passed"] and deeper["reason"] == "contradicts_filmed_frames"
    # A small object it erased passes the whole view, not the region the view was made for.
    target = {"target_world_estimate": CENTROID, "region_extent": [.3, .3, .1]}
    erased = scene(cam, "look-erased", box=False)
    assert look.check_view(erased, references)["passed"]
    erased = look.check_view(erased, references, look.region_footprint(target, cam))
    assert erased["reason"] == "contradicts_filmed_frames_in_region" and erased["region_contradicted_fraction"] > .5
    assert look.check_view(scene(cam, "look-kept"), references, look.region_footprint(target, cam))["passed"]
    recoloured = scene(cam, "look-recoloured")
    recoloured["rgb"] = 255 - recoloured["rgb"]
    assert look.check_view(recoloured, references)["reason"] == "contradicts_filmed_frames"
    away = scene(camera((0, 0, -1.2), yaw_degrees=170), "look-away")
    assert look.check_view(away, references)["reason"] == "too_little_overlap_with_filmed_frames"


# --- the stage -------------------------------------------------------------------------------------------------

def test_passing_views_become_splat_frames_failing_ones_are_dropped_and_all_are_recorded(job):
    atlas = Atlas()
    value = run(job, atlas)
    record = value["record"]
    assert record["status"] == "checked" and len(record["views"]) == 4 and record["dropped_view_count"] == 2
    assert len(atlas.purchases) == 1 and atlas.purchases[0][1] == pytest.approx(.05 + 3 * .01)
    assert record["bought_cost_usd"] == pytest.approx(.08)
    dropped = [row for row in record["views"] if not row["check"]["passed"]]
    assert {row["check"]["reason"] for row in dropped} == {"contradicts_filmed_frames"}
    assert [frame["frame_id"] for frame in value["views"]] == record["passed_view_ids"]
    for frame in value["views"]:
        keep = frame["keep_mask"]
        # Only the region is the generated part: the splat posts it only through the whole-view review.
        assert keep.any() and not keep.all() and frame["rgb"].shape == (720, 1280, 3)
        assert frame["depth"].shape == (720, 1280) and frame["intrinsics"]["width"] == 1280
    # The request is anchored on the real photographs that saw the region and their depth.
    plan = json.loads((job / look.ROOT / look.PLAN).read_text())
    assert plan["batches"][0]["request_digest"] == atlas.calls[0][1]
    assert record["digest"] == canonical_digest(record, digest_field="digest")


def test_a_rerun_buys_nothing_new(job):
    atlas = Atlas(polls_before_done=1)
    assert run(job, atlas)["status"] == "processing" and len(atlas.purchases) == 1
    first = run(job, atlas)
    assert len(atlas.purchases) == 1 and atlas.calls[0][1] == atlas.calls[1][1]
    calls = len(atlas.calls)
    again = run(job, atlas)
    assert len(atlas.calls) == calls and len(atlas.purchases) == 1
    assert again["record"] == first["record"]
    assert [f["frame_id"] for f in again["views"]] == [f["frame_id"] for f in first["views"]]
    assert np.array_equal(again["views"][0]["rgb"], first["views"][0]["rgb"])
    # After the splat is bought the record is read back; it is never generated again.
    assert run(job, atlas, bought=True)["record"] == first["record"] and len(atlas.calls) == calls


def test_a_sealed_plan_keeps_its_cameras_when_the_map_changes_mid_flight(job):
    atlas = Atlas(polls_before_done=1)
    assert run(job, atlas)["status"] == "processing"
    write_map(job.parent.parent, [region(orbit_deg=[5, 10]), region("shelf", angle_spread_deg=1.4)])
    value = run(job, atlas)
    assert len(atlas.purchases) == 1 and atlas.calls[0][1] == atlas.calls[1][1]
    assert {row["orbit_degrees"] for row in value["record"]["views"]} == {-15, 15, -30, 30}


def test_no_map_a_bought_splat_or_another_root_generates_nothing_and_writes_nothing(tmp_path, job):
    atlas = Atlas()
    bare = tmp_path / "bare/pipeline/atlas"
    bare.mkdir(parents=True)
    assert run(bare, atlas) == {"views": [], "record": None} and not any(bare.iterdir())
    assert run(job, atlas, bought=True) == {"views": [], "record": None}
    other = tmp_path / "capture/pipeline/rebuild"
    assert run(other, atlas) == {"views": [], "record": None} and not other.exists()
    assert atlas.calls == [] and not (job / look.ROOT).exists()


def test_money_the_allowance_and_walls_decide_before_anything_is_bought(job, monkeypatch):
    atlas = Atlas()
    short = run(job, atlas, remaining=5.2)
    assert short["views"] == [] and short["record"]["status"] == "allowance_insufficient" and atlas.calls == []
    assert run(job, atlas, remaining=100)["record"]["status"] == "allowance_insufficient"  # sealed, never re-bought
    from blueprint_pipeline.hard_stops import MoneyStop

    def refused(**_):
        raise MoneyStop("account_monthly_spend_limit_reached")
    other = job.parent.parent.parent / "second/pipeline/atlas"
    write_map(other.parent.parent, [region()])
    with pytest.raises(MoneyStop):
        run(other, refused)
    assert not (other / look.ROOT / look.RECORD).exists()

    def failed(**_):
        raise ValueError("atlas_task_operation_failed")
    third = job.parent.parent.parent / "third/pipeline/atlas"
    write_map(third.parent.parent, [region()])
    value = run(third, failed)
    assert value["views"] == [] and value["record"]["batches"][0]["status"] == "failed"
    elsewhere = job.parent.parent.parent / "fifth/pipeline/atlas"
    write_map(elsewhere.parent.parent, [region()], pose_result_digest="sha256:" + "b" * 64)
    assert run(elsewhere, atlas)["record"]["status"] == "weak_regions_pose_mismatch" and atlas.calls == []
    same = job.parent.parent.parent / "sixth/pipeline/atlas"
    write_map(same.parent.parent, [region()], pose_result_digest=POSE["digest"])
    assert run(same, Atlas())["record"]["status"] == "checked"
    unpriced = job.parent.parent.parent / "fourth/pipeline/atlas"
    write_map(unpriced.parent.parent, [region()])
    policy = {key: v for key, v in POLICY.items() if key != "rgbd_view_cost_usd"}
    assert run(unpriced, atlas, policy=policy)["record"]["status"] == "rgbd_view_price_unreviewed"


# --- the splat input, the result and the scene -------------------------------------------------------------------

def _appearance(job, monkeypatch, *, form):
    from blueprint_pipeline import atlas_native_splat, paid_resource_allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    frames = real_frames()
    selection = {"digest": "sha256:" + "4" * 64, "pose_refinement_accepted": True, "flags": [],
                 "sent_frame_ids": ["real-0", "real-1", "real-2", "real-3"], "left_out": [], "heldout_frame_ids": []}
    monkeypatch.setattr(pipeline, "_splat_input_selection", lambda **_: selection)
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: {"policy": POLICY, "remaining_cost_usd": 17})
    atlas = Atlas()
    monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_task", lambda experimental, **kw: atlas(**kw))
    sent = []

    def splat(**kw):
        sent.append(kw["frames"])
        rows = [{"frame_id": f["frame_id"], **({"whole_view_review": {"admit_whole": True}}
                                               if f.get("keep_mask") is not None and form == "documented_reviewed_whole"
                                               else {})}
                for f in kw["frames"] if f.get("keep_mask") is None or form == "documented_reviewed_whole"]
        return {"status": "completed", "splat_root": str(job / "refined_splat-whole"),
                "splat_manifest": {"frames": rows}, "refined_splat": {"request_form": form}}
    monkeypatch.setattr(atlas_native_splat, "run_native_splat", splat)
    appearance = pipeline._preserved_appearance(frames=frames, pose=POSE, draft={"digest": "d"}, output_root=job,
        policy=POLICY, task_context={"context_digest": "t"}, source_digest="s", configuration={},
        check=lambda *_: None, experimental=False, max_polls=1, review={}, source_root=job)
    return appearance, sent, atlas


def test_passing_views_reach_the_splat_input_and_the_result_says_the_look_holds_them(job, monkeypatch):
    appearance, sent, atlas = _appearance(job, monkeypatch, form="documented_reviewed_whole")
    ids = [frame["frame_id"] for frame in sent[0]]
    record = json.loads((job / look.ROOT / look.RECORD).read_text())
    assert ids == ["real-0", "real-1", "real-2", "real-3", *record["passed_view_ids"]]
    disclosed = appearance["look_views"]
    assert disclosed["generated_views_in"] == "look" and disclosed["in_look_view_ids"] == record["passed_view_ids"]
    assert disclosed["record_digest"] == record["digest"] and disclosed["generated_view_count"] == 4


def test_the_real_frames_only_fallback_says_the_look_holds_none(job, monkeypatch):
    appearance, sent, _ = _appearance(job, monkeypatch, form="documented_minimal")
    assert "generated_views_in" not in appearance["look_views"] and appearance["look_views"]["in_look_view_ids"] == []


def test_without_a_map_the_splat_input_and_the_appearance_are_exactly_mains(tmp_path, monkeypatch):
    job = tmp_path / "capture/pipeline/atlas"
    job.mkdir(parents=True)
    appearance, sent, atlas = _appearance(job, monkeypatch, form="documented_minimal")
    assert [frame["frame_id"] for frame in sent[0]] == ["real-0", "real-1", "real-2", "real-3"]
    assert all(frame is real or frame["frame_id"] == real["frame_id"] for frame, real in zip(sent[0], real_frames()))
    assert "look_views" not in appearance and atlas.calls == [] and not any(job.iterdir())


def _bound(job, *, form="documented_reviewed_whole", admit=True, extra=None):
    record = json.loads((job / look.ROOT / look.RECORD).read_text())
    source = {"digest": POSE["digest"], "operation_receipt": {"input_usage": {"frames": [
        {"frame_id": f"real-{i}"} for i in range(4)]}}}
    rows = [{"frame_id": f"real-{i}"} for i in range(4)] + [
        {"frame_id": key, "whole_view_review": {"admit_whole": admit}} for key in record["passed_view_ids"]]
    splat = {"splat_kind": "atlas_refined", "task": "posedImages2Splats", "request_form": form,
             "pose_result_digest": POSE["digest"], "frames": rows + (extra or [])}
    appearance = {"look_views": {"generated_views_in": "look", "record_digest": record["digest"],
                                 "record_relative_path": "look_views/look_views.json"}}
    return look.bound_look_views(splat=splat, source=source, appearance=appearance, job_dir=job)


def test_the_scene_takes_a_look_only_bound_to_its_checked_views(job):
    run(job, Atlas())
    assert _bound(job) == {"generated_views_in": "look", "look_views_digest": json.loads(
        (job / look.ROOT / look.RECORD).read_text())["digest"], "view_count": 2,
        "basis": "filmed_frames_with_generated_views_checked_against_them"}
    for changes in ({"form": "documented_minimal"}, {"admit": False},
                    {"extra": [{"frame_id": "unrecorded", "whole_view_review": {"admit_whole": True}}]}):
        with pytest.raises(ValueError, match="atlas_look_views_not_bound"):
            _bound(job, **changes)
    assert look.bound_look_views(splat={}, source={}, appearance={}, job_dir=job) is None
