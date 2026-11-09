"""A rebuild's one windowed retry: its pose limited to the earlier pose's consistent moving footage.

Every motion shape is synthetic: cameras along a line in pose units, a time per view, a scene depth.
The window comes from ``select_splat_frames``'s own segment decisions, never from fixed times.
"""
from __future__ import annotations

import json
import math

import pytest

from blueprint_pipeline import atlas_splat_input_quality as quality
from blueprint_pipeline import atlas_splat_rebuild as rebuilds
from blueprint_pipeline import atlas_splat_rebuild_window as windows
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_splat_input_quality import _ready_job

DEPTH = 3.0


def _views(parts, *, seconds=.25):
    """Posed views along -Z: each part is (kind, count); ``jump`` moves 2 units in one step."""
    frames, z, t = [], 0.0, 0.0
    for kind, count in parts:
        if kind == "jump":
            z -= 2.0
            continue
        for _ in range(count):
            yaw = math.radians(len(frames) * .5) if kind == "still" else 0.0
            frames.append({"frame_id": f"v{len(frames):03d}", "timestamp_seconds": round(t, 3),
                           "extrinsics": {"position": [0.0, 1.5, z],
                                          "quaternion": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]}})
            t += seconds
            if kind == "move":
                z -= .1
    return frames


def _window(frames, *, route=(), depth=DEPTH, timestamps=None):
    times = {row["frame_id"]: row["timestamp_seconds"] for row in frames} if timestamps is None else timestamps
    record = quality.select_splat_frames(frames=frames, agreement=None, timestamps=times, scene_depth=depth,
                                         pose_refinement_accepted=False, route_frame_ids=set(route))
    return record, windows.window_from_selection(record, times)


def _time(frames, key):
    return next(row["timestamp_seconds"] for row in frames if row["frame_id"] == key)


def test_one_continuous_moving_shot_makes_no_window_and_no_retry():
    _, window = _window(_views([("move", 30)]))
    assert window == {"status": "unchanged", "reason": "earlier_pose_kept_every_view"}


def test_a_pan_or_tripod_shot_has_no_moving_segment_and_makes_no_window():
    record, window = _window(_views([("still", 30)]))
    assert "splat_no_parallax_capture" in record["flags"]
    assert window == {"status": "unavailable", "reason": "earlier_pose_no_moving_segment"}


def test_a_stationary_opening_then_a_jump_keeps_only_the_moving_footage_after_it():
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    route = ["v000", "v030", "v049"]
    record, window = _window(frames, route=route)
    assert [(row["kind"], row["decision"]) for row in record["motion"]["segments"]] == [
        ("no_parallax", "left_out_across_pose_jump"), ("moving", "primary")]
    assert window["status"] == "windowed"
    assert window["intervals_seconds"] == [[_time(frames, "v020"), _time(frames, "v049")]]
    assert window["excluded_frame_ids"] == [] and window["capped_no_parallax_spans"] == []


def test_several_jumps_keep_the_segment_select_splat_frames_keeps_by_route_then_baseline():
    frames = _views([("move", 12), ("jump", 0), ("move", 10), ("jump", 0), ("move", 25)])
    # The reviewed route lies in the middle shot: it is kept although the last shot is longer.
    record, window = _window(frames, route=["v013", "v018"])
    assert [row["decision"] for row in record["motion"]["segments"]] == [
        "left_out_across_pose_jump", "primary", "left_out_across_pose_jump"]
    assert window["intervals_seconds"] == [[_time(frames, "v012"), _time(frames, "v021")]]
    # With no reviewed route in any shot, the most baseline wins.
    record, window = _window(frames)
    assert window["intervals_seconds"] == [[_time(frames, "v022"), _time(frames, "v046")]]


def test_a_stationary_run_inside_the_moving_shot_is_capped_as_the_splat_selection_caps_it():
    frames = _views([("move", 15), ("still", 12), ("move", 15)])
    record, window = _window(frames, route=["v000", "v041"])
    segments = record["motion"]["segments"]
    assert [(row["kind"], row["decision"]) for row in segments] == [
        ("moving", "primary"), ("no_parallax", "capped"), ("moving", "kept")]
    capped = segments[1]
    assert len(capped["kept_frame_ids"]) == quality.STATIONARY_FRAME_CAP
    assert window["status"] == "windowed" and window["intervals_seconds"] == [
        [_time(frames, "v000"), _time(frames, "v041")]]
    span = window["capped_no_parallax_spans"][0]
    assert span["kept_frame_ids"] == capped["kept_frame_ids"]
    # The survey's frames in that run, other than the views the cap kept, are excluded from the next selection.
    review = _review(frames)
    derived = windows.windowed_review(review, frames, window)
    still = {key for key in capped["frame_ids"]}
    assert set(derived["excluded_frame_ids"]) == still - set(capped["kept_frame_ids"])


@pytest.mark.parametrize("case", ["no_times", "no_depth"])
def test_an_unusable_earlier_pose_makes_no_window_and_raises_nothing(case):
    frames = _views([("still", 10), ("jump", 0), ("move", 20)])
    if case == "no_times":
        _, window = _window(frames, timestamps={"v000": 0.0, "v001": .25})
        assert window == {"status": "unavailable", "reason": "earlier_pose_too_few_timed_views"}
    else:
        _, window = _window(frames, depth=None)
        assert window == {"status": "unavailable", "reason": "earlier_pose_scene_depth_unavailable"}


def _review(frames, *, required=("v000", "v018", "v030", "v049"), anchor="v018"):
    first, *middle, last = required
    return {"schema_version": "atlas_scene_state_review.v1", "subject_origin": "no_manipulated_subject",
            "anchor_frame_id": anchor,
            "static_intervals_seconds": [[0.0, frames[-1]["timestamp_seconds"]]],
            "coverage_requirements": [
                {"role": "task_region", "label": "Start area", "required_frame_ids": [first]},
                {"role": "task_region", "label": "Route", "required_frame_ids": list(middle)},
                {"role": "surrounding_scene", "label": "Room", "required_frame_ids": list(required)}],
            "uncaptured_regions": ["Behind the camera."], "excluded_frame_ids": [],
            "additional_input_frame_ids": [first, middle[-1]]}


def test_reviewed_frames_outside_the_window_are_left_out_and_named_and_the_anchor_moves():
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    _, window = _window(frames, route=["v000", "v030"])
    review = _review(frames)
    derived = windows.windowed_review(review, frames, window)
    assert derived["status"] == "windowed"
    value = derived["review"]
    # The start area had only a frame before the jump: its requirement is left out and named.
    assert [(row["label"], row["required_frame_ids"]) for row in value["coverage_requirements"]] == [
        ("Route", ["v030"]), ("Room", ["v030", "v049"])]
    assert derived["left_out_required_frame_ids"] == ["v000", "v018"]
    # The anchor moves to the in-window reviewed frame nearest it in time, and the original is kept.
    assert value["anchor_frame_id"] == "v030" and value["original_task_anchor_frame_id"] == "v018"
    assert derived["flags"] == ["pose_window_applied", "pose_window_required_frames_left_out",
                                "task_route_frames_left_out", "pose_window_anchor_moved"]
    assert value["uncaptured_regions"][0] == "Behind the camera."
    assert any(line.startswith("Start area: frames v000 left out") for line in value["uncaptured_regions"])
    assert value["static_intervals_seconds"] == window["intervals_seconds"]
    assert value["additional_input_frame_ids"] == ["v030"]
    assert value["pose_window"]["original_review_digest"] == canonical_digest(review)
    # The reviewed review itself is untouched.
    assert review["anchor_frame_id"] == "v018" and "pose_window" not in review


def test_a_window_that_would_lose_a_reviewed_coverage_role_makes_no_review():
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    _, window = _window(frames)
    review = _review(frames, required=("v000", "v005", "v010", "v019"), anchor="v005")
    assert windows.windowed_review(review, frames, window) == {
        "status": "unavailable", "reason": "pose_window_omits_reviewed_coverage",
        "left_out_required_frame_ids": ["v000", "v005", "v010", "v019"]}


def _earlier_rebuild(tmp_path, *, frames):
    """A ready job whose first rebuild stopped at a pose World Labs did not refine."""
    job_dir, job, parent = _ready_job(tmp_path)
    (job_dir / "survey").mkdir()
    survey = {"frames": [{"frame_id": row["frame_id"], "timestamp_seconds": row["timestamp_seconds"]}
                         for row in frames]}
    (job_dir / "survey/atlas_video_survey.json").write_text(json.dumps(survey))
    review = _review(frames)
    (job_dir / "selection_binding.json").write_text(json.dumps(
        {"survey_relative_path": "survey/atlas_video_survey.json", "review": review}))
    first = rebuilds.request_rebuild(job_dir, job, at_epoch=1)
    first_root = job_dir / first["root"]
    source = first_root / "pose/task-coverage-1/full-view-source-v1"
    source.mkdir(parents=True)
    pose = {"operation": {"response": {"poseRefinementAccepted": False}},
            "operation_receipt": {"input_usage": {"frames": [{"frame_id": row["frame_id"]} for row in frames]}}}
    pose["digest"] = canonical_digest(pose, digest_field="digest")
    (source / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    rebuilds.record_stop(first_root, first, parent, {"stage": "source_pose",
                                                     "reason": "atlas_splat_rebuild_pose_refinement_not_accepted"})
    return job_dir, job, parent, first, review


def test_a_rebuild_stopped_at_an_unrefined_pose_qualifies_for_one_windowed_retry_at_the_same_cost(tmp_path):
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    job_dir, job, parent, first, _ = _earlier_rebuild(tmp_path, frames=frames)
    first_root = job_dir / first["root"]
    outcome_before = (first_root / rebuilds.OUTCOME_FILE).read_bytes()
    pose_before = sorted(path.read_bytes() for path in (first_root / "pose").rglob("*.json"))
    eligible = rebuilds.rebuild_eligibility(job_dir, job)
    assert eligible["attempt"] == rebuilds.WINDOW_ATTEMPT
    assert eligible["previous_request_digest"] == first["digest"] and eligible["previous_root"] == first["root"]
    assert eligible["reasons"] == ["previous_rebuild_pose_refinement_not_accepted"]
    assert eligible["previous_posed_frame_count"] == len(frames)
    # Money unchanged: the same expected and maximum cost the first rebuild showed.
    assert {key: eligible[key] for key in ("expected_cost_usd", "maximum_cost_usd")} == {
        key: first[key] for key in ("expected_cost_usd", "maximum_cost_usd")}
    retry = rebuilds.request_rebuild(job_dir, job, at_epoch=2)
    root = job_dir / retry["root"]
    assert root != first_root and root.parent == first_root.parent and retry["attempt"] == rebuilds.WINDOW_ATTEMPT
    # The first request is kept in its own root; its outcome and pose are untouched.
    assert json.loads((first_root / rebuilds.RETAINED_REQUEST_FILE).read_text()) == first
    assert (first_root / rebuilds.OUTCOME_FILE).read_bytes() == outcome_before
    assert sorted(path.read_bytes() for path in (first_root / "pose").rglob("*.json")) == pose_before
    assert json.loads((root / "parent_result.json").read_text()) == parent
    # Never a third attempt.
    assert rebuilds.rebuild_eligibility(job_dir, job) is None


@pytest.mark.parametrize("stop", ["refinement_not_accepted", "allowance", "accepted_elsewhere"])
def test_only_an_unrefined_pose_stop_qualifies_for_the_retry(tmp_path, stop):
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    job_dir, job, _, first, _ = _earlier_rebuild(tmp_path, frames=frames)
    root = job_dir / first["root"]
    if stop == "allowance":
        outcome = json.loads((root / rebuilds.OUTCOME_FILE).read_text())
        (root / rebuilds.OUTCOME_FILE).write_text(json.dumps(
            {**outcome, "stage": "refined_splat", "reason": "atlas_splat_rebuild_allowance_insufficient"}))
    elif stop == "accepted_elsewhere":
        path = next((root / "pose").rglob("atlas_posed_rgbd_result.json"))
        pose = json.loads(path.read_text())
        pose["operation"]["response"]["poseRefinementAccepted"] = True
        pose["digest"] = canonical_digest(pose, digest_field="digest")
        path.write_text(json.dumps(pose))
    eligible = rebuilds.rebuild_eligibility(job_dir, job)
    assert (eligible is not None) == (stop == "refinement_not_accepted")


def _runner(job_dir):
    return dict(job_dir=job_dir, task_context={}, policy={"refined_splat_cost_usd": 5},
                held_policy={"refined_splat_cost_usd": 5}, configuration={}, source_digest="s",
                check=lambda *_: None, experimental=False, max_polls=1)


@pytest.mark.parametrize("outcome", ["stopped", "published"])
def test_the_windowed_retry_poses_only_the_window_and_its_stop_is_final(tmp_path, monkeypatch, outcome):
    from blueprint_pipeline import atlas_video_inputs, website_atlas_pipeline as pipeline
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    job_dir, job, parent, _, review = _earlier_rebuild(tmp_path, frames=frames)
    retry = rebuilds.request_rebuild(job_dir, job, at_epoch=2)
    root = job_dir / retry["root"]
    earlier = []
    monkeypatch.setattr(windows, "_earlier_selection", lambda previous_root, reviewed: earlier.append(
        previous_root) or (dict(_window(frames, route=["v000", "v030"])[0], pose_result_digest="sha256:" + "e" * 64),
                           {row["frame_id"]: row["timestamp_seconds"] for row in frames}))
    selected = []
    monkeypatch.setattr(atlas_video_inputs, "prepare_video_inputs", lambda **kw: selected.append(kw) or {"frames": []})

    def scene(**kw):
        assert kw["review"]["anchor_frame_id"] == "v030" and kw["output_root"] == root
        if outcome == "stopped":
            return {"status": "needs_input", "stage": "source_pose",
                    "reason": "atlas_splat_rebuild_pose_refinement_not_accepted"}
        return {"status": "needs_input", "appearance": {"basis": "atlas_refined_splat"}, "download": {"zip": "new"},
                "source_pose": {"status": "completed", "pose_relative_path": "pose/task-coverage-2/full-view-source-v1"},
                "digest": "sha256:" + "5" * 64}
    monkeypatch.setattr(pipeline, "_preserved_scene", scene)
    value = pipeline._run_splat_rebuild(**_runner(job_dir))
    assert earlier == [job_dir / retry["previous_root"]]
    # The selection draws only from the window, under the derived review, recorded once before any purchase.
    window = json.loads((root / "pose" / rebuilds.WINDOW_FILE).read_text())
    assert window["status"] == "windowed" and window["digest"] == canonical_digest(window, digest_field="digest")
    assert window["intervals_seconds"] == [[_time(frames, "v020"), _time(frames, "v049")]]
    assert window["original_review_digest"] == canonical_digest(review)
    assert selected[0]["review"] == window["review"] and selected[0]["splat_frame_target"] == 96
    assert json.loads((root / "atlas_scene_review.json").read_text()) == window["review"]
    if outcome == "stopped":
        assert value == parent
        saved = json.loads((root / rebuilds.OUTCOME_FILE).read_text())
        assert saved["attempt"] == rebuilds.WINDOW_ATTEMPT and saved["status"] == "stopped"
        assert rebuilds.rebuild_eligibility(job_dir, job) is None
    else:
        marker = value["splat_rebuild"]
        assert marker["attempt"] == rebuilds.WINDOW_ATTEMPT and marker["previous_request_digest"] == retry[
            "previous_request_digest"]
        assert marker["pose_window"]["intervals_seconds"] == window["intervals_seconds"]
        assert marker["pose_window"]["left_out_required_frame_ids"] == ["v000", "v018"]
        assert marker["pose_window"]["review_digest"] == canonical_digest(window["review"])
        assert rebuilds.rebuild_eligibility(job_dir, {**job, "result_digest": value["digest"]}) is None
    # Read back, never re-derived or run again.
    monkeypatch.setattr(pipeline, "_preserved_scene", lambda **_: pytest.fail("retry ran twice"))
    assert pipeline._run_splat_rebuild(**_runner(job_dir)) == value


@pytest.mark.parametrize("shape", ["clean", "tripod", "unreadable"])
def test_a_retry_with_no_usable_window_stops_by_name_and_buys_nothing(tmp_path, monkeypatch, shape):
    from blueprint_pipeline import atlas_video_inputs, website_atlas_pipeline as pipeline
    frames = {"clean": _views([("move", 40)]), "tripod": _views([("still", 40)]),
              "unreadable": _views([("move", 40)])}[shape]
    job_dir, job, parent, _, _ = _earlier_rebuild(tmp_path, frames=frames)
    retry = rebuilds.request_rebuild(job_dir, job, at_epoch=2)
    if shape == "unreadable":
        def broken(*_):
            raise ValueError("atlas_depth_splat_frame_count_mismatch")
        monkeypatch.setattr(windows, "_earlier_selection", broken)
    else:
        monkeypatch.setattr(windows, "_earlier_selection", lambda *_: (
            dict(_window(frames)[0], pose_result_digest="sha256:" + "e" * 64),
            {row["frame_id"]: row["timestamp_seconds"] for row in frames}))
    monkeypatch.setattr(atlas_video_inputs, "prepare_video_inputs", lambda **_: pytest.fail("selection made"))
    monkeypatch.setattr(pipeline, "_preserved_scene", lambda **_: pytest.fail("pose bought"))
    assert pipeline._run_splat_rebuild(**_runner(job_dir)) == parent
    saved = json.loads((job_dir / retry["root"] / rebuilds.OUTCOME_FILE).read_text())
    status = {"clean": "unchanged", "tripod": "unavailable", "unreadable": "unavailable"}[shape]
    assert saved["stage"] == "pose_window" and saved["reason"] == "atlas_splat_rebuild_pose_window_" + status
    assert "pose_window_" + status in saved["flags"]
    assert rebuilds.rebuild_eligibility(job_dir, job) is None


def test_the_retry_pose_root_admits_only_task_views_inside_its_window(tmp_path, monkeypatch):
    from PIL import Image

    from blueprint_pipeline import atlas_privacy_screen, atlas_source_pose_inputs as source_inputs
    from blueprint_pipeline import website_scene_geometry
    pipeline_root = tmp_path / "pipeline"
    pose_root = pipeline_root / "atlas/splat-rebuild/x-pose-window/pose"
    pose_root.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()

    def row(key, time):
        path = images / f"{key}.png"
        Image.new("RGB", (4, 4)).save(path)
        return {"frame_id": key, "timestamp_seconds": time, "source_image_path": str(path), "image_path": str(path),
                "source_image_digest": "d", "image_digest": "d", "display_rotation_degrees": 0}
    original = {"binding": {"source_video_digest": "sha256:v"}, "frames": [row("a", 9.0)], "digest": "sha256:o"}
    candidates = [row("early", 1.0), row("late", 12.0), row("dropped", 11.0)]
    clean = pipeline_root / "clean_plate"
    (clean / "source_geometry/worker_inputs").mkdir(parents=True)
    sources = {"schema_version": "website_source_frames.v1", "binding": {"source_video_digest": "sha256:v"},
               "geometry_input_digest": "sha256:g"}
    sources["digest"] = canonical_digest(sources, digest_field="digest")
    (clean / "clean_plate_stage_manifest.json").write_text(json.dumps({"source_frames": sources}))
    (clean / "source_geometry/worker_inputs/geometry_inputs.json").write_text(json.dumps(
        {"digest": "sha256:g", "binding": sources["binding"], "heldout_pixels_included": False}))
    monkeypatch.setattr(website_scene_geometry, "_bound_frames",
                        lambda value, root, geometry: candidates if value.get("digest") == "sha256:g"
                        else value["frames"])
    monkeypatch.setattr(atlas_privacy_screen, "blur_faces", lambda image: (image, []))
    # No window: every task view joins, as before, and the binding is as before.
    plain = tmp_path / "plain/pipeline/atlas/pose"
    plain.mkdir(parents=True)
    (tmp_path / "plain/pipeline/clean_plate").symlink_to(clean, target_is_directory=True)
    _, unwindowed = source_inputs.task_source_inputs(plain, original)
    assert [row["frame_id"] for row in unwindowed["frames"]] == ["a", "early", "late", "dropped"]
    assert "pose_window_digest" not in unwindowed["binding"]
    window = {"schema_version": rebuilds.WINDOW_SCHEMA, "status": "windowed", "request_digest": "sha256:r",
              "intervals_seconds": [[8.0, 13.0]], "review": {"excluded_frame_ids": ["dropped"]}}
    window["digest"] = canonical_digest(window, digest_field="digest")
    (pose_root / rebuilds.WINDOW_FILE).write_text(json.dumps(window))
    _, windowed = source_inputs.task_source_inputs(pose_root, original)
    assert [row["frame_id"] for row in windowed["frames"]] == ["a", "late"]
    assert windowed["binding"]["pose_window_digest"] == window["digest"]


def test_atlas_retry_reopens_the_job_once_more_for_the_windowed_retry_and_never_again(tmp_path):
    from blueprint_pipeline.website_atlas_preview import retry_failed_job
    frames = _views([("still", 20), ("jump", 0), ("move", 30)])
    job_dir, job, _, first, _ = _earlier_rebuild(tmp_path, frames=frames)
    reopened = retry_failed_job(job_dir / "preview_job.json")
    assert reopened["state"] == "processing" and reopened["retries"][-1]["reason"] == "refined_splat_rebuild_requested"
    request = json.loads((job_dir / rebuilds.REQUEST_FILE).read_text())
    assert request["attempt"] == rebuilds.WINDOW_ATTEMPT and request["previous_request_digest"] == first["digest"]
    (job_dir / "preview_job.json").write_text(json.dumps(job))
    with pytest.raises(ValueError, match="atlas_preview_retry_requires_recoverable_failure"):
        retry_failed_job(job_dir / "preview_job.json")
