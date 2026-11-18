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
from blueprint_pipeline.atlas_gap_composite import _digest as composite_digest
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
    """The map as ``atlas_coverage_map.write_weak_regions`` seals it in the Atlas job root."""
    path = capture / look.WEAK_REGIONS_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": look.WEAK_REGIONS_SCHEMA, "status": "mapped", "regions": regions,
              "pose_result_digest": POSE["digest"], **fields}
    record["digest"] = canonical_digest(record, digest_field="digest")
    path.write_text(json.dumps(record))
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
            submission.write_text(json.dumps({"status": "submitted", "request_digest": price_review["request_digest"],
                                              "binding": {"request_template": request, "uploads": uploads,
                                                          "maximum_cost_usd": price_review["maximum_cost_usd"]}}))
        else:
            # The adapter's own rule on resume: the request and its reserved price are what was sent.
            sent = json.loads(submission.read_text())["binding"]
            assert request == sent["request_template"] and price_review["maximum_cost_usd"] == sent["maximum_cost_usd"]
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
    monkeypatch.setattr(look, "check_view", lambda view, references, footprint, **_: (
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


def test_regions_and_views_are_capped_in_the_maps_order_and_unposed_regions_are_named():
    # The map ranks the regions the job's route views saw first; a wider spread there still comes first.
    regions = [region("ghost", angle_spread_deg=0.1, seen_by=["not-posed"])]
    regions += [region(f"r{i}", angle_spread_deg=float(8 - i)) for i in range(8)]
    targets, skipped = look.orbit_cameras(regions, real_frames())
    kept = {row["region_id"] for row in targets}
    assert kept == {"r0", "r1", "r2", "r3", "r4", "r5"} and len(targets) == look.MAX_VIEWS == 24
    assert {"region_id": "ghost", "reason": "seen_by_cameras_not_available"} in skipped
    assert {row["region_id"] for row in skipped if row["reason"] == "over_region_limit"} == {"r6", "r7"}
    assert len(look.orbit_cameras(regions, real_frames(), max_views=6)[0]) == 6
    assert look.orbit_angles([-30, 30]) == [-30, 30] and look.orbit_angles([0, 20]) == [-20, 20]


def test_the_map_is_read_by_the_coverage_maps_own_sealed_reader(tmp_path):
    assert look._mapped_regions(tmp_path) == (None, None)
    path = write_map(tmp_path, [region(), {"id": "broken", "centroid": [0, 0]}])
    regions, mapped_pose = look._mapped_regions(tmp_path)
    assert [row["id"] for row in regions] == ["table"] and mapped_pose == POSE["digest"]
    sealed = json.loads(path.read_text())
    for changed in ({**sealed, "regions": [region("edited")]},  # no longer the sealed record
                    {**sealed, "status": "not_mapped"}):
        if changed.get("regions") == sealed["regions"]:
            changed["digest"] = canonical_digest(changed, digest_field="digest")
        path.write_text(json.dumps(changed))
        assert look._mapped_regions(tmp_path) == (None, None)
    path.write_text("{")
    assert look._mapped_regions(tmp_path) == (None, None)


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
    unnamed = job.parent.parent.parent / "seventh/pipeline/atlas"
    write_map(unnamed.parent.parent, [region()], pose_result_digest=None)  # The map must name its pose.
    assert run(unnamed, atlas)["record"]["status"] == "weak_regions_pose_mismatch" and atlas.calls == []
    same = job.parent.parent.parent / "sixth/pipeline/atlas"
    write_map(same.parent.parent, [region()], pose_result_digest=POSE["digest"])
    assert run(same, Atlas())["record"]["status"] == "checked"
    unpriced = job.parent.parent.parent / "fourth/pipeline/atlas"
    write_map(unpriced.parent.parent, [region()])
    policy = {key: v for key, v in POLICY.items() if key != "rgbd_view_cost_usd"}
    assert run(unpriced, atlas, policy=policy)["record"]["status"] == "rgbd_view_price_unreviewed"


def test_a_request_in_flight_is_polled_as_it_was_sent_when_the_price_changes(job):
    """Review (genviews): a live paid operation is never abandoned; it is polled at the price it was reserved under."""
    atlas = Atlas(polls_before_done=1)
    assert run(job, atlas)["status"] == "processing" and len(atlas.purchases) == 1
    dearer = {**POLICY, "rgbd_view_cost_usd": .07, "additional_rgbd_view_cost_usd": .02}
    value = run(job, atlas, policy=dearer)  # The fake adapter asserts the request and price are as sent.
    assert value["record"]["status"] == "checked" and len(atlas.purchases) == 1
    assert value["record"]["batches"][0]["maximum_cost_usd"] == pytest.approx(.08)
    # Its price withdrawn while it runs: still polled and settled, never left live.
    other = job.parent.parent.parent / "withdrawn/pipeline/atlas"
    write_map(other.parent.parent, [region()])
    atlas = Atlas(polls_before_done=1)
    assert run(other, atlas)["status"] == "processing"
    unpriced = {key: v for key, v in POLICY.items() if key != "rgbd_view_cost_usd"}
    assert run(other, atlas, policy=unpriced)["record"]["status"] == "checked" and len(atlas.purchases) == 1


def test_an_error_that_is_no_wall_falls_back_to_the_real_frames_unless_a_request_is_live(job, monkeypatch):
    from blueprint_pipeline import atlas_scene_completion

    def mismatch(*_):
        raise ValueError("atlas_depth_splat_grid_mismatch")
    monkeypatch.setattr(atlas_scene_completion, "load_generated_views", mismatch)
    value = run(job, Atlas())
    assert value["views"] == [] and value["record"]["status"] == "failed"
    assert value["record"]["reason"] == "atlas_depth_splat_grid_mismatch"
    assert run(job, Atlas())["record"] == value["record"]  # Sealed: the splat goes on with its real frames.
    # A request still live is not left behind: the error is raised, and the next tick polls it.
    live = job.parent.parent.parent / "live/pipeline/atlas"
    write_map(live.parent.parent, [region()])
    atlas = Atlas(polls_before_done=5)
    assert run(live, atlas)["status"] == "processing"
    monkeypatch.setattr(look, "orbit_cameras", lambda *_, **__: (_ for _ in ()).throw(IndexError("late")))
    plan = live / look.ROOT / look.PLAN
    plan.unlink()  # Forces a re-plan that fails, with the first request still live.
    with pytest.raises(IndexError):
        run(live, atlas)
    assert not (live / look.ROOT / look.RECORD).exists()


def test_a_checkpoint_stop_is_raised_as_it_is(job):
    class Withdrawn(Exception):
        pass

    def check(_stage):
        raise Withdrawn()
    with pytest.raises(Withdrawn):
        look.look_views_for_splat(frames=real_frames(), pose=POSE, output_root=job, policy=POLICY,
            task_context={"context_digest": "t"}, source_digest="s", configuration=lambda: {"policy": POLICY},
            task_runner=Atlas(), bought=False, check=check)
    assert not (job / look.ROOT / look.RECORD).exists()


# --- the splat input, the result and the scene -------------------------------------------------------------------

def _appearance(job, monkeypatch, *, form, mapped=False):
    """``_preserved_appearance`` with the map the fixture wrote, or the one ``atlas_coverage_map`` writes."""
    from blueprint_pipeline import atlas_coverage_map, atlas_native_splat, paid_resource_allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    if not mapped:
        monkeypatch.setattr(atlas_coverage_map, "coverage_before_splat", lambda **_: None)
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


@pytest.mark.parametrize("map_state", ["absent", "not_mapped", "no_regions"])
def test_without_a_mapped_region_the_splat_input_and_the_appearance_are_exactly_mains(tmp_path, monkeypatch,
                                                                                        map_state):
    job = tmp_path / "capture/pipeline/atlas"
    job.mkdir(parents=True)
    if map_state != "absent":
        write_map(job.parent.parent, [], **({"status": "not_mapped"} if map_state == "not_mapped" else {}))
    before = sorted(job.rglob("*"))
    appearance, sent, atlas = _appearance(job, monkeypatch, form="documented_minimal")
    assert [frame["frame_id"] for frame in sent[0]] == ["real-0", "real-1", "real-2", "real-3"]
    assert "look_views" not in appearance and atlas.calls == [] and sorted(job.rglob("*")) == before


def test_the_coverage_map_written_before_the_splat_feeds_the_look_views(job, monkeypatch):
    """Merged with ``atlas_coverage_map``: its map, written before the purchase, names the regions orbited."""
    monkeypatch.setattr(look, "check_view", lambda view, references, footprint, **_: {"passed": True})
    appearance, sent, atlas = _appearance(job, monkeypatch, form="documented_reviewed_whole", mapped=True)
    from blueprint_pipeline.atlas_coverage_map import read_weak_regions
    mapped = read_weak_regions(job)
    record = json.loads((job / look.ROOT / look.RECORD).read_text())
    assert mapped["status"] == "mapped" and record["weak_regions_digest"] == look._sha256_file(job / "weak_regions.json")
    assert {row["region_id"] for row in record["views"]} <= {row["id"] for row in mapped["regions"]}
    assert [frame["frame_id"] for frame in sent[0]][4:] == record["passed_view_ids"] and atlas.purchases
    assert appearance["look_views"]["in_look_view_ids"] == record["passed_view_ids"]


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


def test_each_view_is_composited_once_and_paired_only_against_the_real_frames_that_overlap_it(tmp_path, monkeypatch):
    """The serial room step renders each view once, against at most CHECK_FRAMES_PER_VIEW real frames."""
    from blueprint_pipeline import atlas_gap_composite, atlas_native_splat, atlas_whole_view_review
    frames = [scene(camera((x, y, 0)), f"real-{i}") for i, (x, y) in enumerate(
        (x, y) for x in (0, .05, .1, .15, .2, .25) for y in (0, .1))]
    capture = tmp_path / "capture"
    write_map(capture, [region(seen_by=[f["frame_id"] for f in frames])])
    monkeypatch.setattr(look, "check_view", lambda view, references, footprint, **_: {"passed": True})
    rendered = []
    real_render = atlas_gap_composite.render_observed
    monkeypatch.setattr(atlas_gap_composite, "render_observed", lambda references, camera: rendered.append(
        [ref["frame_id"] for ref in references]) or real_render(references, camera))
    job = capture / "pipeline/atlas"
    sent = frames[:10]  # The splat's selection left two of the real frames out.
    value = look.look_views_for_splat(frames=frames, pose=POSE, output_root=job, policy=POLICY,
        task_context={"context_digest": "t"}, source_digest="s", configuration=lambda: {"policy": POLICY},
        task_runner=Atlas(), bought=False, sent=sent)
    record = value["record"]
    assert len(value["views"]) == 4 and len(frames) > look.CHECK_FRAMES_PER_VIEW
    # One render a view, against its region's frames only.
    assert len(rendered) == 4 and all(len(ids) == look.CHECK_FRAMES_PER_VIEW for ids in rendered)
    assert all(ids == record["views"][0]["check_frame_ids"] for ids in rendered)
    sent_ids = {f["frame_id"] for f in sent}
    for view in value["views"]:
        row = next(r for r in record["views"] if r["frame_id"] == view["frame_id"])
        assert view["gap_composite"] == row["gap_composite"]
        assert composite_digest(view["rgb"]) == row["gap_composite"]["rgb_sha256"]
        assert view["overlapping_frame_ids"] == [key for key in row["check_frame_ids"] if key in sent_ids]
    # The splat path posts the composited views as they are: nothing is rendered again.
    merged = atlas_native_splat.composited_views([*sent, *value["views"]])
    assert len(rendered) == 4 and sum(1 for _, view, _ in merged if view.get("gap_composite")) == 4
    # The whole-view review pairs each view among the sent frames that overlap it only.
    paired = []
    real_overlap = atlas_whole_view_review._overlap
    monkeypatch.setattr(atlas_whole_view_review, "_overlap", lambda real, view: paired.append(
        (view["frame_id"], real["frame_id"])) or real_overlap(real, view))
    review = atlas_whole_view_review.review_whole_views(views=merged, root=tmp_path / "review", task_context={},
        policy={"maximum_review_cost_usd": 0}, source_video_digest="s")
    assert review["status"] == "over_review_cap" and len(review["gap_composites"]) == 4
    for view in value["views"]:
        assert {real for key, real in paired if key == view["frame_id"]} == set(view["overlapping_frame_ids"])
    # Fewer than three of its region's frames sent: the sent seen_by frames nearest its camera.
    row = record["views"][0]
    few = [f for f in frames if f["frame_id"] not in row["check_frame_ids"][1:]]
    named = look.reference_frame_ids(row, few)
    assert named[0] == row["check_frame_ids"][0]
    assert set(named) == {f["frame_id"] for f in few} & set(row["seen_by_frame_ids"])


def test_a_refined_splat_retry_resends_the_same_request_and_its_records_match_the_look(job, monkeypatch):
    """``atlas_refined_splat_retry``: when the first attempt's forms failed, the fresh attempt posts the same
    frames, the look views included, under the first attempt's retained review; the result, the record
    and the scene binding then agree on which generated views the look holds."""
    from blueprint_pipeline import atlas_coverage_map, atlas_native_splat, paid_resource_allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline.atlas_refined_splat_forms import MINIMAL, WHOLE
    from blueprint_pipeline import atlas_refined_splat_retry as retry
    monkeypatch.setattr(atlas_coverage_map, "coverage_before_splat", lambda **_: None)
    selection = {"digest": "sha256:" + "4" * 64, "pose_refinement_accepted": True, "flags": [],
                 "sent_frame_ids": ["real-0", "real-1", "real-2", "real-3"], "left_out": [], "heldout_frame_ids": []}
    monkeypatch.setattr(pipeline, "_splat_input_selection", lambda **_: selection)
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: {"policy": POLICY, "remaining_cost_usd": 17})
    monkeypatch.setattr(pipeline, "_job_window_end", lambda *_: 2_000_000_000 + 5 * 86400)
    clock = [2_000_000_000]
    monkeypatch.setattr(pipeline.time, "time", lambda: clock[0])
    atlas = Atlas()
    monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_task", lambda experimental, **kw: atlas(**kw))
    calls = []

    def splat(**kw):
        calls.append(kw)
        look_ids = [f["frame_id"] for f in kw["frames"] if f.get("keep_mask") is not None]
        if kw["output_root"] == job / "refined_splat":
            # The first attempt: its review admitted every look view, then both forms failed at the provider.
            whole = job / "refined_splat-whole"
            review = {"schema_version": "atlas_whole_view_review.v1", "status": "reviewed", "verdicts": [
                {"position": 4 + i, "frame_id": key, "admit_whole": True, "reason": "compared"}
                for i, key in enumerate(look_ids)]}
            (whole / "native/atlas_pose").mkdir(parents=True, exist_ok=True)
            (whole / "whole_view_review.json").write_text(json.dumps(review))
            (whole / "input_binding.json").write_text(json.dumps({"request_form": WHOLE, "whole_view_review": review}))
            for root in (whole, job / "refined_splat-minimal"):
                (root / "native/atlas_pose").mkdir(parents=True, exist_ok=True)
                (root / "native/atlas_pose/submission.json").write_text(json.dumps(
                    {"status": "failed", "terminal_operation": {"error": {"code": 13}}}))
            raise ValueError("atlas_task_operation_failed")
        rows = [{"frame_id": f["frame_id"], **({"whole_view_review": {"admit_whole": True}}
                                               if f.get("keep_mask") is not None else {})} for f in kw["frames"]]
        return {"status": "completed", "splat_root": str(kw["output_root"]) + "-whole",
                "splat_manifest": {"splat_kind": "atlas_refined", "task": "posedImages2Splats", "request_form": WHOLE,
                                   "pose_result_digest": POSE["digest"], "frames": rows},
                "refined_splat": {"request_form": WHOLE}}
    monkeypatch.setattr(atlas_native_splat, "run_native_splat", splat)
    options = dict(frames=real_frames(), pose=POSE, draft={"digest": "d"}, output_root=job, policy=POLICY,
                   task_context={"context_digest": "t"}, source_digest="s", configuration={},
                   check=lambda *_: None, experimental=False, max_polls=1, review={}, source_root=job)
    waiting = pipeline._preserved_appearance(**options)
    assert waiting["stage"] == retry.WAIT_STAGE
    clock[0] = waiting["next_attempt_at_epoch"]
    appearance = pipeline._preserved_appearance(**options)
    first, again = calls[0], calls[-1]
    assert again["output_root"] == retry.attempt_root(job, 1) and again["forms"] == (WHOLE, MINIMAL)
    assert [f["frame_id"] for f in again["frames"]] == [f["frame_id"] for f in first["frames"]]
    assert again["whole_view_review"] == json.loads((job / "refined_splat-whole/whole_view_review.json").read_text())
    assert len(atlas.purchases) == 1  # The look views were generated once, for both attempts.
    record = json.loads((job / look.ROOT / look.RECORD).read_text())
    assert appearance["look_views"]["generated_views_in"] == "look"
    assert appearance["look_views"]["in_look_view_ids"] == record["passed_view_ids"]
    source = {"digest": POSE["digest"], "operation_receipt": {"input_usage": {"frames": [
        {"frame_id": f"real-{i}"} for i in range(4)]}}}
    bound = look.bound_look_views(splat=appearance["splat_manifest"], source=source,
                                  appearance={"look_views": appearance["look_views"]}, job_dir=job)
    assert bound["view_count"] == len(record["passed_view_ids"]) and bound["look_views_digest"] == record["digest"]


def _room(count=30):
    """Filmed views along a line and turning, so the room's views spread in position and direction."""
    return [scene(camera((.05 * (i % 6), .02 * (i // 6), .03 * i), yaw_degrees=(i % 7) - 3), f"real-{i}")
            for i in range(count)]


def test_generation_context_fills_the_providers_rollout_region_views_then_task_views_then_the_room(tmp_path):
    """AGENTS (src/blueprint_pipeline): generation context uses the provider's configured capacity, never a
    global count, and keeps task views and wider room context. The check alone stays at eight frames."""
    from blueprint_pipeline.atlas_scene_chisel import ROLLOUT_VIEWS, photograph_count
    frames = _room()
    seen = ["real-0", "real-1", "real-2", "real-3"]
    capture = tmp_path / "capture"
    write_map(capture, [region(seen_by=seen)])
    atlas = Atlas()
    requests = []
    original = atlas.__call__

    def runner(**kw):
        requests.append(kw["request"])
        return original(**kw)
    value = look.look_views_for_splat(frames=frames, pose=POSE, output_root=capture / "pipeline/atlas", policy=POLICY,
        task_context={"context_digest": "t"}, source_digest="s", configuration=lambda: {"policy": POLICY},
        task_runner=runner, bought=False, task_frame_ids=["real-29", "real-15"])
    plan = json.loads((capture / "pipeline/atlas" / look.ROOT / look.PLAN).read_text())
    target = plan["targets"][0]
    context = target["context_frame_ids"]
    by_id = {frame["frame_id"]: frame for frame in frames}
    # The region's own views first (nearest its filmed camera), then the job's task views, then the room.
    assert set(context[:4]) == set(seen) and context[0] == target["base_frame_id"]
    assert context[4:6] == ["real-29", "real-15"] and len(context) == len(set(context)) > 8
    # Filled to the rollout: context photographs plus two views per target, and no other view would fit.
    photographs = sum(photograph_count(by_id[key]) for key in context)
    targets = len(requests[0]["targetCameras"])
    assert photographs == len(requests[0]["contextFrames"]) and photographs + 2 * targets <= ROLLOUT_VIEWS
    assert all(photographs + photograph_count(frame) + 2 * targets > ROLLOUT_VIEWS
               for frame in frames if frame["frame_id"] not in context)
    # The check is a review, not reconstruction context: at most eight of the region's own views.
    assert target["check_frame_ids"] == context[:4] and len(target["check_frame_ids"]) <= look.CHECK_FRAMES_PER_VIEW
    # The price is per generated view: more context photographs do not change it.
    assert value["record"]["batches"][0]["maximum_cost_usd"] == pytest.approx(.05 + (targets - 1) * .01)


def test_held_out_frames_never_reach_generation_context_or_the_check(tmp_path, monkeypatch):
    """Review (P2): the held-out score measures the look against views nothing generated saw."""
    from blueprint_pipeline import atlas_gap_composite, atlas_scene_generate
    frames = _room(12)
    held = ["real-1", "real-7"]
    capture = tmp_path / "capture"
    write_map(capture, [region(seen_by=["real-0", "real-1", "real-2", "real-3"])])
    monkeypatch.setattr(look, "check_view", lambda view, references, footprint, **_: {"passed": True})
    prepared, rendered = [], []
    real_prepare, real_render = atlas_scene_generate.prepare_rgbd_context, atlas_gap_composite.render_observed
    monkeypatch.setattr(atlas_scene_generate, "prepare_rgbd_context", lambda *, frames, root: prepared.extend(
        frame["frame_id"] for frame in frames) or real_prepare(frames=frames, root=root))
    monkeypatch.setattr(atlas_gap_composite, "render_observed", lambda references, camera: rendered.extend(
        ref["frame_id"] for ref in references) or real_render(references, camera))
    sent = [frame for frame in frames if frame["frame_id"] not in held]
    value = look.look_views_for_splat(frames=frames, pose=POSE, output_root=capture / "pipeline/atlas", policy=POLICY,
        task_context={"context_digest": "t"}, source_digest="s", configuration=lambda: {"policy": POLICY},
        task_runner=Atlas(), bought=False, sent=sent, heldout_frame_ids=held, task_frame_ids=["real-7", "real-9"])
    record = value["record"]
    assert value["views"] and prepared and rendered
    named = {key for row in record["views"] for field in ("context_frame_ids", "check_frame_ids", "seen_by_frame_ids")
             for key in row[field]} | set(prepared) | set(rendered)
    assert not named & set(held) and "real-9" in record["views"][0]["context_frame_ids"]
    assert not {key for view in value["views"] for key in view["overlapping_frame_ids"]} & set(held)


def test_the_pipeline_keeps_held_out_frames_out_of_the_look_views(job, monkeypatch):
    """``_preserved_appearance`` hands the selection's held-out and task views to the look views."""
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    monkeypatch.setattr(look, "check_view", lambda view, references, footprint, **_: {"passed": True})
    monkeypatch.setattr(pipeline, "_heldout_quality", lambda **_: {"status": "not_scored"})
    held = ["real-1"]

    def selection(**_):
        return {"digest": "sha256:" + "4" * 64, "pose_refinement_accepted": True, "flags": [],
                "sent_frame_ids": ["real-0", "real-2", "real-3"], "left_out": [], "heldout_frame_ids": held,
                "route_frame_ids": ["real-3"]}
    from blueprint_pipeline import atlas_coverage_map, atlas_native_splat, paid_resource_allocator
    monkeypatch.setattr(atlas_coverage_map, "coverage_before_splat", lambda **_: None)
    monkeypatch.setattr(pipeline, "_splat_input_selection", selection)
    monkeypatch.setattr(pipeline, "load_configuration", lambda *_, **__: {"policy": POLICY, "remaining_cost_usd": 17})
    atlas = Atlas()
    monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_task", lambda experimental, **kw: atlas(**kw))
    sent = []
    monkeypatch.setattr(atlas_native_splat, "run_native_splat", lambda **kw: sent.append(kw["frames"]) or {
        "status": "completed", "splat_root": str(job / "refined_splat-minimal"), "splat_manifest": {"frames": []},
        "refined_splat": {"request_form": "documented_minimal"}})
    pipeline._preserved_appearance(frames=real_frames(), pose=POSE, draft={"digest": "d"}, output_root=job,
        policy=POLICY, task_context={"context_digest": "t"}, source_digest="s", configuration={},
        check=lambda *_: None, experimental=False, max_polls=1, review={}, source_root=job)
    record = json.loads((job / look.ROOT / look.RECORD).read_text())
    assert record["views"] and all("real-1" not in row["context_frame_ids"] + row["check_frame_ids"]
                                   for row in record["views"])
    assert "real-1" not in [frame["frame_id"] for frame in sent[0]]
