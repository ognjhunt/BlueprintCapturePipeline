"""Source selection cannot promote state review, heldout pixels or generated views."""
import json
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import atlas_video_inputs as video
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_adaptive_selection_keeps_overlap_and_skips_duplicates_blur_and_disconnection(monkeypatch):
    frames = [{"frame_id": str(i), "sharpness": 100, "clipped_fraction": .1} for i in range(6)]
    frames[1]["sharpness"] = 0
    answers = iter([(100, 2), (100, 35), (4, 90), (100, 40)])
    monkeypatch.setattr(video, "_overlap", lambda *_: next(answers))
    selected, diagnostics = video.select_views(frames=frames, features=[None] * 6, maximum_frames=2)
    assert selected == [0, 3]
    assert [r["decision"] for r in diagnostics] == ["near_duplicate", "selected", "disconnected_view", "frame_budget_reached"]


def test_no_static_review_no_preparation(tmp_path):
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = tmp_path / "survey.json"
    path.write_text(json.dumps(survey))
    with pytest.raises(ValueError, match="atlas_scene_state_review_required"):
        video.prepare_video_inputs(survey_path=path, review={}, output_root=tmp_path)


def test_no_paid_set_with_only_duplicate_views(monkeypatch):
    monkeypatch.setattr(video, "_overlap", lambda *_: (100, 1))
    with pytest.raises(ValueError, match="insufficient_overlapping_distinct_views"):
        video.select_views(frames=[{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0} for i in range(3)],
                           features=[None] * 3, maximum_frames=24)


def test_task_closeups_without_surrounding_scene_are_refused():
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0} for i in range(3)]
    with pytest.raises(ValueError, match="task_and_scene_coverage_review_required"):
        video.select_coverage_views(frames=frames, features=[None] * 3, maximum_frames=3, anchor_id="0",
            requirements=[{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["0"]}])


def test_required_context_uses_overlap_bridges_and_never_silently_drops_context(monkeypatch):
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0} for i in range(5)]
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 25) if abs(a-b) == 1 else (0, 0))
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["4"]},
                    {"role": "surrounding_scene", "label": "counter and floor", "required_frame_ids": ["0"]}]
    selected, _ = video.select_coverage_views(frames=frames, features=list(range(5)), maximum_frames=5,
        requirements=requirements, anchor_id="4")
    assert selected == list(range(5))
    with pytest.raises(ValueError, match="coverage_exceeds_frame_budget"):
        video.select_coverage_views(frames=frames, features=list(range(5)), maximum_frames=4,
            requirements=requirements, anchor_id="4")
    frames[2]["sharpness"] = 0
    with pytest.raises(ValueError, match="coverage_gap"):
        video.select_coverage_views(frames=frames, features=list(range(5)), maximum_frames=5,
            requirements=requirements, anchor_id="4")


def test_financial_frame_budget_has_no_invented_64_frame_ceiling():
    review = {"selection_budget": {"maximum_pose_cost_usd": 4.76, "input_cost_usd": .01,
        "rate_card_reference": "fixture only", "reviewed_by": "test operator",
        "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    assert video.selection_frame_budget(review) == 476
    review["selection_budget"]["input_cost_usd"] = 0
    with pytest.raises(ValueError, match="reviewed_selection_budget_required"):
        video.selection_frame_budget(review)


def test_candidate_densification_preserves_existing_hidden_ids():
    from blueprint_pipeline.atlas_video_extension import gap_indexes
    times = [i / 10 for i in range(31)]
    result = gap_indexes(times=times, base_indexes=[0, 5, 10, 15, 20, 25, 30],
        visible_indexes={0, 5, 20, 25, 30}, intervals=[[0, 2.5]], step_seconds=.2)
    assert result and set(result).isdisjoint({10, 15})
    assert all(5 < i < 20 for i in result)


def test_survey_scales_with_video_and_per_run_cadence_instead_of_64_frames():
    short = [i / 30 for i in range(31)]
    long = [i / 30 for i in range(3001)]
    assert video.initial_survey_count(long, {}) > 64
    assert video.initial_survey_count(short, {}) < 64
    assert video.initial_survey_count(long, {"initial_candidate_interval_seconds": .1}) > video.initial_survey_count(long, {})


def test_scene_specific_quality_and_overlap_policy_can_admit_sparse_texture(monkeypatch):
    frames = [{"frame_id": str(i), "sharpness": 10, "clipped_fraction": .2} for i in range(2)]
    requirements = [{"role": "task_object", "label": "scene-specific object", "required_frame_ids": ["0"]},
                    {"role": "surrounding_scene", "label": "scene-specific support", "required_frame_ids": ["1"]}]
    monkeypatch.setattr(video, "_overlap", lambda *_: (12, 20))
    selected, _ = video.select_coverage_views(frames=frames, features=[None, None], maximum_frames=476,
        requirements=requirements, anchor_id="0", policy={"minimum_sharpness": 5, "minimum_overlap_inliers": 10})
    assert selected == [0, 1]


def test_provider_inputs_are_face_screened_before_their_digest_and_raw_frames_stay_local(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline import atlas_privacy_screen as screen
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    survey_root = tmp_path / "survey"
    (survey_root / "frames").mkdir(parents=True)
    rows = []
    for index in range(2):
        path = survey_root / "frames" / f"{index}.png"
        Image.new("RGB", (64, 48), (40 * index, 90, 160)).save(path)
        rows.append({"frame_id": str(index), "timestamp_seconds": index, "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path)})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False, "frames": rows,
              "display_rotation_degrees": 0, "source_video_digest": "sha256:" + "a" * 64, "sample_count": 2,
              "heldout_frame_count": 0}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (survey_root / "atlas_video_survey.json").write_text(json.dumps(survey))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "test", "scene_state": "unchanged",
              "static_intervals_seconds": [[0, 1]], "anchor_frame_id": "0", "coverage_requirements": [
                  {"role": "task_object", "label": "object", "required_frame_ids": ["0"]},
                  {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["1"]}],
              "selection_budget": {"maximum_pose_cost_usd": 1, "input_cost_usd": .01, "rate_card_reference": "test",
                                   "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    monkeypatch.setattr(video, "_features", lambda *_: None)
    monkeypatch.setattr(video, "select_coverage_views", lambda **_: ([0, 1], []))
    screened = []

    def blur(image):
        screened.append(image.size)
        pixels = image.copy()
        pixels.paste((0, 0, 0), (0, 0, 8, 8))
        return pixels, [[0, 0, 8, 8]] if len(screened) == 1 else []
    monkeypatch.setattr(screen, "blur_faces", blur)
    output = tmp_path / "pose"
    video.prepare_video_inputs(survey_path=survey_root / "atlas_video_survey.json", review=review, output_root=output)
    assert screened == [(64, 48), (64, 48)]
    for row in rows:
        sent = output / "atlas_source_views" / f"{row['frame_id']}.png"
        with Image.open(sent) as image:
            assert image.getpixel((1, 1)) == (0, 0, 0)  # the provider receives only screened pixels
        assert _sha256_file(output / "atlas_observations" / f"{row['frame_id']}.png") == row["source_image_digest"]
    selection = json.loads((output / "atlas_view_selection.json").read_text())
    assert [frame["image_digest"] for frame in selection["frames"]] == [
        _sha256_file(output / "atlas_source_views" / f"{row['frame_id']}.png") for row in rows]
    assert selection["privacy_screen"]["blurred_faces_by_frame"] == {"0": 1, "1": 0}
    assert selection["privacy_screen"]["model_sha256"] == screen.MODEL_SHA256
    pending = json.loads((output / "atlas_pose_inputs.json").read_text())
    assert [frame["image_digest"] for frame in pending["frames"]] == [frame["image_digest"] for frame in selection["frames"]]


def _timeline(count):
    return [{"frame_id": str(i), "timestamp_seconds": i * .5, "sharpness": 50, "clipped_fraction": 0} for i in range(count)]


def test_accuracy_first_adds_every_distinct_sharp_connected_view_and_keeps_the_cover(monkeypatch):
    frames = _timeline(8)
    frames[5]["sharpness"] = 1  # blurred
    frames[6]["clipped_fraction"] = .9  # blown out

    def overlap(a, b):
        a, b = sorted((a, b))
        if (a, b) == (1, 2):
            return 100, 3  # frame 2 barely moved: a near duplicate of 1
        if 7 in (a, b):
            return 2, 40  # nothing matches frame 7: disconnected
        return 100, 30
    monkeypatch.setattr(video, "_overlap", overlap)
    kept, diagnostics = video.add_distinct_views(frames=frames, features=list(range(8)), selected=[0, 4],
                                                 maximum_frames=64)
    # Widest gap first: 2 is admitted before 1, and 1 then barely differs from it.
    assert kept == [0, 2, 3, 4]
    decisions = {row["frame_id"]: row["decision"] for row in diagnostics}
    assert decisions == {"1": "near_duplicate", "2": "distinct_view", "3": "distinct_view", "7": "disconnected_view"}


def _chain(count):
    """Overlap only between consecutive frames: the hardest case for keeping inputs connected."""
    return lambda a, b: (100, 30) if abs(a - b) == 1 else (0, 0.0)


def test_earlier_context_is_reached_through_its_own_bridges(monkeypatch):
    monkeypatch.setattr(video, "_overlap", _chain(10))
    kept, _ = video.add_distinct_views(frames=_timeline(10), features=list(range(10)), selected=[8, 9],
                                       maximum_frames=64)
    assert kept == list(range(10))


def test_a_binding_capacity_keeps_every_input_connected_and_never_drops_the_cover(monkeypatch):
    monkeypatch.setattr(video, "_overlap", _chain(10))
    kept, diagnostics = video.add_distinct_views(frames=_timeline(10), features=list(range(10)), selected=[0, 1],
                                                 maximum_frames=5)
    assert kept == [0, 1, 2, 3, 4]  # each added view overlaps a kept one; no stranded far views
    assert {row["decision"] for row in diagnostics} == {"distinct_view", "beyond_input_capacity"}
    # With wider overlap, the same capacity spreads the inputs over the whole interval.
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 30) if abs(a - b) <= 5 else (0, 0.0))
    kept, _ = video.add_distinct_views(frames=_timeline(20), features=list(range(20)), selected=[10],
                                       maximum_frames=5)
    assert kept == [0, 5, 10, 15, 19] or (min(kept) <= 1 and max(kept) >= 18 and len(kept) == 5)
    with pytest.raises(ValueError, match="coverage_exceeds_frame_budget"):
        video.add_distinct_views(frames=_timeline(20), features=list(range(20)), selected=[1, 2, 3], maximum_frames=2)


def test_accuracy_target_is_independent_of_dollars_which_only_refuse(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline import atlas_privacy_screen as screen
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    root = tmp_path / "survey"
    (root / "frames").mkdir(parents=True)
    rows = []
    for index in range(70):
        path = root / "frames" / f"{index}.png"
        Image.new("RGB", (32, 24), (index, 90, 160)).save(path)
        rows.append({"frame_id": str(index), "timestamp_seconds": index * .5, "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path), "sharpness": 50, "clipped_fraction": 0})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False, "frames": rows,
              "display_rotation_degrees": 0, "source_video_digest": "sha256:" + "a" * 64, "sample_count": 70,
              "heldout_frame_count": 0}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (root / "atlas_video_survey.json").write_text(json.dumps(survey))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "test", "scene_state": "unchanged",
              "static_intervals_seconds": [[0, 40]], "anchor_frame_id": "0", "coverage_requirements": [
                  {"role": "task_object", "label": "object", "required_frame_ids": ["0"]},
                  {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["1"]}],
              "selection_budget": {"maximum_pose_cost_usd": 10, "input_cost_usd": .01, "rate_card_reference": "test",
                                   "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    monkeypatch.setattr(video, "_features", lambda row, rotation: int(row["frame_id"]))
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 30))
    monkeypatch.setattr(screen, "blur_faces", lambda image: (image, []))
    video.prepare_video_inputs(survey_path=root / "atlas_video_survey.json", review=review, output_root=tmp_path / "pose")
    selection = json.loads((tmp_path / "pose/atlas_view_selection.json").read_text())
    from blueprint_pipeline.website_atlas_pose import MAX_POSE_INPUT_IMAGES
    assert selection["selected_frame_count"] == MAX_POSE_INPUT_IMAGES == 64
    assert selection["frame_policy"] == {"basis": "accuracy_first", "provider_input_capacity": 64, "frame_target": 64,
                                         "connected_cover_frames": 2, "distinct_views_added": 62}
    # A budget that cannot pay for the accurate set refuses; it never quietly poses fewer frames.
    review["selection_budget"]["maximum_pose_cost_usd"] = .3
    with pytest.raises(ValueError, match="budget_below_accurate_inputs"):
        video.prepare_video_inputs(survey_path=root / "atlas_video_survey.json", review=review,
                                   output_root=tmp_path / "pose-low")
