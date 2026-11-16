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
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0, "timestamp_seconds": float(i)}
              for i in range(5)]
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
    # The room beyond the blurred frame is left out and named, not silently dropped (before 2026-11-16: coverage_gap).
    selected, diagnostics = video.select_coverage_views(frames=frames, features=list(range(5)), maximum_frames=5,
        requirements=requirements, anchor_id="4")
    assert selected == [3, 4] and {"decision": video.SURROUNDINGS_NOT_CONNECTED, "frame_ids": ["0"]} in diagnostics


def _neighbour_chain(monkeypatch, count=6, blurred=(1,)):
    """Frames overlap only their neighbours; a blurred frame breaks the chain there."""
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0} for i in range(count)]
    for index in blurred:
        frames[index]["sharpness"] = 0
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 25) if abs(a - b) == 1 else (0, 0))
    return frames


def test_a_surrounding_scene_gap_is_recorded_while_the_task_area_still_selects(monkeypatch):
    frames = _neighbour_chain(monkeypatch)
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["4"]},
                    {"role": "surrounding_scene", "label": "counter and floor", "required_frame_ids": ["0", "3"]},
                    {"role": "surrounding_scene", "label": "window wall", "required_frame_ids": ["1"]}]
    selected, diagnostics = video.select_coverage_views(frames=frames, features=list(range(6)), maximum_frames=6,
                                                        requirements=requirements, anchor_id="4")
    # Frame 0 lies beyond the blurred frame 1; frame 3 still shows the room around the task.
    assert selected == [3, 4]
    assert [(row["frame_id"], row["label"], row["decision"]) for row in diagnostics
            if row["decision"] in video.SURROUNDING_GAP_DECISIONS] == [
        ("0", "counter and floor", "surrounding_scene_unconnected"),
        ("1", "window wall", "surrounding_scene_below_quality")]


def test_a_blurry_task_anchor_still_refuses(monkeypatch):
    frames = _neighbour_chain(monkeypatch, blurred=(4,))
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["4"]},
                    {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["3"]}]
    with pytest.raises(ValueError, match="atlas_coverage_anchor_quality_insufficient"):
        video.select_coverage_views(frames=frames, features=list(range(6)), maximum_frames=6,
                                    requirements=requirements, anchor_id="4")


@pytest.mark.parametrize("role, frame_id, decision, code", [
    ("support_surface", "0", video.TASK_AREA_UNCONNECTED, "task_views_left_out"),
    ("approach_space", "0", video.TASK_AREA_UNCONNECTED, "task_views_left_out"),
    ("task_region", "1", video.TASK_AREA_BELOW_QUALITY, "task_views_left_out"),
    ("surrounding_scene", "0", "surrounding_scene_unconnected", "surroundings_not_connected"),
    ("surrounding_scene", "1", "surrounding_scene_below_quality", "surroundings_not_connected"),
])
def test_a_view_that_does_not_connect_is_left_out_named_and_the_job_goes_on(monkeypatch, role, frame_id,
                                                                             decision, code):
    # Before 2026-11-16: atlas_task_and_scene_coverage_gap (or anchor quality), and the site was asked to film again.
    from blueprint_pipeline import atlas_scene_estimates as estimates
    frames = [{**row, "timestamp_seconds": float(i)} for i, row in enumerate(_neighbour_chain(monkeypatch))]
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["4"]},
                    {"role": role, "label": "the gap", "required_frame_ids": [frame_id]}]
    if role != "surrounding_scene":
        requirements.append({"role": "surrounding_scene", "label": "room", "required_frame_ids": ["3"]})
    selected, diagnostics = video.select_coverage_views(frames=frames, features=list(range(6)), maximum_frames=6,
                                                        requirements=requirements, anchor_id="4")
    assert 4 in selected and int(frame_id) not in selected and len(selected) >= 2
    assert {"frame_id": frame_id, "label": "the gap", "decision": decision} in diagnostics
    rows = estimates.scene_estimates(levelling={}, registration={}, selection={"diagnostics": diagnostics})
    assert rows == [{"code": "scene_estimate:" + code, "source": "view_selection", "evidence": {"frame_ids": [frame_id]}}]
    assert video._covered({}, diagnostics) is False and video._covered({}, []) is True


def test_a_task_view_that_overlaps_no_other_view_still_refuses(monkeypatch):
    frames = [{"frame_id": str(i), "sharpness": 50, "clipped_fraction": 0, "timestamp_seconds": float(i)}
              for i in range(3)]
    monkeypatch.setattr(video, "_overlap", lambda a, b: (0, 0))
    requirements = [{"role": "task_object", "label": "dishwasher", "required_frame_ids": ["0"]},
                    {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["2"]}]
    # One view and nothing that overlaps it: no scene can be posed from it.
    with pytest.raises(ValueError, match="atlas_insufficient_overlapping_distinct_views"):
        video.select_coverage_views(frames=frames, features=list(range(3)), maximum_frames=3,
                                    requirements=requirements, anchor_id="0")


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


@pytest.mark.parametrize("dropped", [False, True])
def test_provider_inputs_are_face_screened_before_their_digest_and_raw_frames_stay_local(tmp_path, monkeypatch,
                                                                                         dropped):
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
    gap = {"frame_id": "1", "label": "doorway", "decision": "surrounding_scene_unconnected"}
    monkeypatch.setattr(video, "select_coverage_views", lambda **_: ([0, 1], [gap] if dropped else []))
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
    # A surrounding frame left out is reported as uncaptured; a full cover adds no field.
    if dropped:
        assert selection["uncaptured_regions"] == ["doorway: frame 1 not connected by overlapping views"]
    else:
        assert "uncaptured_regions" not in selection
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


@pytest.mark.parametrize("left_out, options", [
    ({"excluded_frame_ids": ["1"]}, {}),
    # On the direct path a frame showing a person is never posed as captured.
    ({"transient_frame_ids": ["1"]}, {}),
    # Nor on a geometry-only job, which edits nothing even when its plan would remove the subject.
    ({"transient_frame_ids": ["1"], "automation": {"objects": [{"disposition": "remove"}]}},
     {"exclude_transients": True}),
])
def test_frames_the_review_excluded_inside_a_still_interval_never_become_candidates(tmp_path, monkeypatch, left_out,
                                                                                    options):
    from PIL import Image
    from blueprint_pipeline import atlas_privacy_screen as screen
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    survey_root = tmp_path / "survey"
    (survey_root / "frames").mkdir(parents=True)
    rows = []
    for index in range(3):
        path = survey_root / "frames" / f"{index}.png"
        Image.new("RGB", (64, 48), (40 * index, 90, 160)).save(path)
        rows.append({"frame_id": str(index), "timestamp_seconds": index, "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path), "sharpness": 50, "clipped_fraction": 0})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False, "frames": rows,
              "display_rotation_degrees": 0, "source_video_digest": "sha256:" + "a" * 64, "sample_count": 3,
              "heldout_frame_count": 0}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (survey_root / "atlas_video_survey.json").write_text(json.dumps(survey))
    # Someone fills frame 1 while working; the rest of the interval stays usable.
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "test", "scene_state": "unchanged",
              "static_intervals_seconds": [[0, 2]], "anchor_frame_id": "0", **left_out,
              "coverage_requirements": [
                  {"role": "task_object", "label": "object", "required_frame_ids": ["0"]},
                  {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["2"]}],
              "selection_budget": {"maximum_pose_cost_usd": 1, "input_cost_usd": .01, "rate_card_reference": "test",
                                   "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    monkeypatch.setattr(video, "_features", lambda *_: None)
    candidates = []
    monkeypatch.setattr(video, "select_coverage_views",
                        lambda **kw: candidates.append([f["frame_id"] for f in kw["frames"]]) or ([0, 1], []))
    monkeypatch.setattr(screen, "blur_faces", lambda image: (image, []))
    video.prepare_video_inputs(survey_path=survey_root / "atlas_video_survey.json", review=review,
                               output_root=tmp_path / "pose", **options)
    assert candidates == [["0", "2"]]
    selection = json.loads((tmp_path / "pose" / "atlas_view_selection.json").read_text())
    assert [frame["frame_id"] for frame in selection["frames"]] == ["0", "2"]


@pytest.mark.parametrize("review, edited", [
    ({"automation": {"objects": [{"disposition": "remove"}]}}, True),
    ({"automation": {"objects": [{"disposition": "keep"}]}, "transient_frame_ids": ["9"],
      "coverage_requirements": [{"required_frame_ids": ["0"]}], "additional_input_frame_ids": ["9"]}, True),
    # A person only in frames the editor never processes needs no edit.
    ({"automation": {"objects": [{"disposition": "keep"}]}, "transient_frame_ids": ["5"],
      "coverage_requirements": [{"required_frame_ids": ["0"]}]}, False),
    ({}, False),
])
def test_only_a_removal_or_a_transient_in_an_edited_view_takes_the_reviewed_background(review, edited):
    assert video.takes_reviewed_background(review) is edited


def test_splat_support_views_preserve_quality_connectivity_unique_pixels_and_cover(monkeypatch):
    rows = [{**row, "source_image_digest": str(i)} for i, row in enumerate(_timeline(40))]
    rows[10]["sharpness"] = 0
    rows[11]["clipped_fraction"] = 1
    rows[12]["source_image_digest"] = rows[0]["source_image_digest"]
    monkeypatch.setattr(video, "_overlap", lambda a, b: (0, 0) if 13 in (a, b) else (100, 1))
    kept, added = video.add_overlapping_support_views(frames=rows, features=list(range(40)),
        selected=[0, 20, 39], minimum_frames=16)
    assert len(kept) == 16 and {0, 20, 39} <= set(kept)
    assert not {10, 11, 12, 13} & set(kept)
    assert len({rows[i]["source_image_digest"] for i in kept}) == 16
    assert {row["decision"] for row in added} == {"overlapping_support_view"}


def test_splat_support_views_report_the_available_set_without_fabricating_a_count(monkeypatch):
    rows = [{**row, "source_image_digest": str(i)} for i, row in enumerate(_timeline(4))]
    monkeypatch.setattr(video, "_overlap", lambda *_: (100, 1))
    kept, added = video.add_overlapping_support_views(frames=rows, features=list(range(4)),
        selected=[0, 3], minimum_frames=16)
    assert kept == [0, 1, 2, 3] and len(added) == 2


def test_dense_splat_inputs_are_private_digest_bound_and_reused_without_reselection(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline import atlas_privacy_screen as screen
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    root = tmp_path / "survey"
    root.mkdir()
    rows = []
    for i in range(40):
        path = root / f"{i}.png"
        Image.new("RGB", (32, 24), (i, 80, 160)).save(path)
        rows.append({"frame_id": str(i), "timestamp_seconds": i * .25, "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path), "sharpness": 100, "clipped_fraction": 0})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False, "frames": rows,
              "display_rotation_degrees": 0, "source_video_digest": "sha256:" + "a" * 64,
              "sample_count": 40, "heldout_frame_count": 4}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = root / "atlas_video_survey.json"
    path.write_text(json.dumps(survey))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "test", "scene_state": "unchanged",
              "static_intervals_seconds": [[0, 10]], "anchor_frame_id": "0", "excluded_frame_ids": ["20"],
              "coverage_requirements": [
                  {"role": "task_object", "label": "object", "required_frame_ids": ["0"]},
                  {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["39"]}],
              "selection_budget": {"maximum_pose_cost_usd": 1, "input_cost_usd": .01, "rate_card_reference": "test",
                  "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    monkeypatch.setattr(video, "_features", lambda row, _: int(row["frame_id"]))
    monkeypatch.setattr(video, "_overlap", lambda *_: (100, 1))
    monkeypatch.setattr(screen, "blur_faces", lambda image: (image, []))
    options = dict(survey_path=path, review=review, output_root=tmp_path / "pose", minimum_frames=16, maximum_frames=30)
    first = video.prepare_video_inputs(**options)
    plan = json.loads((options["output_root"] / "atlas_view_selection.json").read_text())
    assert len(first["frames"]) == 16 and "20" not in {row["frame_id"] for row in first["frames"]}
    assert first["heldout_pixels_included"] is False and first["heldout_frame_count"] == 4
    assert plan["frame_policy"]["overlapping_support_views_added"] == 14
    assert plan["frame_policy"]["distinct_views_added"] == 0
    assert plan["frame_policy"]["minimum_source_frames_met"] is True
    monkeypatch.setattr(video, "_features", lambda *_: pytest.fail("retained inputs recomputed"))
    assert video.prepare_video_inputs(**options) == first
    with pytest.raises(ValueError, match="retained_video_inputs_invalid"):
        video.prepare_video_inputs(**{**options, "minimum_frames": 0})


def test_splat_fill_takes_the_sharpest_view_of_each_empty_time_window_without_the_displacement_rule(monkeypatch):
    rows = [{**row, "source_image_digest": str(i), "sharpness": 100 if i % 10 == 7 else 50}
            for i, row in enumerate(_timeline(100))]
    monkeypatch.setattr(video, "_overlap", lambda *_: (100, 1))  # every candidate barely moved
    kept, added = video.add_time_spread_views(frames=rows, features=list(range(100)), selected=[0, 99],
                                              frame_target=10)
    # Ten equal windows; the kept ends hold the first and last, each other window gives its sharpest view.
    assert kept == [0, 17, 27, 37, 47, 57, 67, 77, 87, 99]
    assert [row["decision"] for row in added] == ["time_spread_view"] * 8


def test_splat_fill_keeps_the_sharpness_clipping_overlap_and_unique_image_gates(monkeypatch):
    rows = [{**row, "source_image_digest": str(i)} for i, row in enumerate(_timeline(20))]
    rows[3]["sharpness"] = 1  # blurred
    rows[7]["clipped_fraction"] = .9  # blown out
    rows[15]["source_image_digest"] = rows[0]["source_image_digest"]  # a repeated decoded image
    monkeypatch.setattr(video, "_overlap", lambda a, b: (2, 40) if 11 in (a, b) else (100, 1))  # 11 matches nothing
    kept, added = video.add_time_spread_views(frames=rows, features=list(range(20)), selected=[0, 19],
                                              frame_target=20)
    assert kept == [i for i in range(20) if i not in {3, 7, 11, 15}]  # short of the target, never padded
    assert len(added) == 14
    for policy in ({"minimum_sharpness": 60}, {"maximum_clipped_fraction": -1}, {"minimum_overlap_inliers": 101}):
        assert video.add_time_spread_views(frames=rows, features=list(range(20)), selected=[0, 19],
                                           frame_target=20, policy=policy) == ([0, 19], [])


def _forward_motion_job(tmp_path, *, count=152, budget=10.0, origin="no_manipulated_subject"):
    """A 0.1 s survey of a forward dolly: features move 1.5 px a frame and overlap 30 frames ahead."""
    from PIL import Image
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    root = tmp_path / "survey"
    root.mkdir(exist_ok=True)
    rows = []
    for i in range(count):
        path = root / f"{i}.png"
        Image.new("RGB", (32, 24), (i, 90, 160)).save(path)
        rows.append({"frame_id": str(i), "timestamp_seconds": round(i * .1, 3), "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path), "sharpness": 80, "clipped_fraction": .01})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False, "frames": rows,
              "display_rotation_degrees": 0, "source_video_digest": "sha256:" + "a" * 64,
              "sample_count": count, "heldout_frame_count": 0}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = root / "atlas_video_survey.json"
    path.write_text(json.dumps(survey))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "test", "scene_state": "unchanged",
              "subject_origin": origin, "static_intervals_seconds": [[0, count * .1]], "anchor_frame_id": "0",
              "coverage_requirements": [
                  {"role": "task_region", "label": "aisle", "required_frame_ids": ["0"]},
                  {"role": "surrounding_scene", "label": "end shelf", "required_frame_ids": [str(count - 1)]}],
              "selection_budget": {"maximum_pose_cost_usd": budget, "input_cost_usd": .01,
                                   "rate_card_reference": "test", "reviewed_by": "test",
                                   "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    return path, review


def _forward_motion(monkeypatch):
    from blueprint_pipeline import atlas_privacy_screen as screen
    monkeypatch.setattr(video, "_features", lambda row, _: int(row["frame_id"]))
    monkeypatch.setattr(video, "_overlap",
                        lambda a, b: (200, 1.5 * abs(a - b)) if abs(a - b) <= 30 else (0, 0.0))
    monkeypatch.setattr(screen, "blur_faces", lambda image: (image, []))


def test_a_forward_motion_splat_selection_reaches_its_target_instead_of_stalling(tmp_path, monkeypatch):
    from blueprint_pipeline.website_atlas_pose import SPLAT_FRAME_TARGET, SPLAT_POSE_INPUT_CAPACITY
    path, review = _forward_motion_job(tmp_path)
    _forward_motion(monkeypatch)
    plain = video.prepare_video_inputs(survey_path=path, review=review, output_root=tmp_path / "plain")
    # The near-duplicate rule alone keeps one view per 12 px of motion: the capture-5b067498 stall.
    assert len(plain["frames"]) < 32
    options = dict(survey_path=path, review=review, output_root=tmp_path / "splat",
                   splat_frame_target=SPLAT_FRAME_TARGET)
    first = video.prepare_video_inputs(**options)
    plan = json.loads((tmp_path / "splat/atlas_view_selection.json").read_text())
    policy = plan["frame_policy"]
    assert len(first["frames"]) == plan["selected_frame_count"] == SPLAT_FRAME_TARGET == 96
    assert policy == {"basis": "accuracy_first", "provider_input_capacity": SPLAT_POSE_INPUT_CAPACITY,
                      "frame_target": 96, "connected_cover_frames": policy["connected_cover_frames"],
                      "distinct_views_added": len(plain["frames"]) - policy["connected_cover_frames"],
                      "splat_frame_target": 96, "time_spread_views_added": 96 - len(plain["frames"]),
                      "splat_frame_target_met": True}
    # The cover and distinct views stay; the fill spreads the rest evenly over the interval.
    assert {row["frame_id"] for row in plain["frames"]} <= {row["frame_id"] for row in first["frames"]}
    times = sorted(row["timestamp_seconds"] for row in plan["frames"])
    assert times[0] == 0 and times[-1] == 15.1
    assert max(b - a for a, b in zip(times, times[1:])) <= 2 * 15.1 / 96  # never more than two windows apart
    decisions = [row["decision"] for row in plan["diagnostics"]]
    assert decisions.count("time_spread_view") == policy["time_spread_views_added"]
    spread = {row["frame_id"] for row in plan["diagnostics"] if row["decision"] == "time_spread_view"}
    assert len([row for row in plan["diagnostics"] if row["frame_id"] in spread]) == len(spread)
    inputs = json.loads((tmp_path / "splat" / video.GEOMETRY_INPUTS_FILE).read_text())
    assert inputs["binding"]["splat_frame_target"] == 96 and inputs["binding"]["maximum_frames"] == 96
    # A paid-stage wait reuses the retained handoff; a different profile never rewrites it.
    monkeypatch.setattr(video, "_features", lambda *_: pytest.fail("retained inputs recomputed"))
    assert video.prepare_video_inputs(**options) == first
    for changed in ({"splat_frame_target": 0}, {"splat_frame_target": 64}):
        with pytest.raises(ValueError, match="retained_video_inputs_invalid"):
            video.prepare_video_inputs(**{**options, **changed})
    for invalid in (1, SPLAT_POSE_INPUT_CAPACITY + 1, True):
        with pytest.raises(ValueError, match="atlas_selection_limit_invalid"):
            video.prepare_video_inputs(**{**options, "output_root": tmp_path / "invalid",
                                          "splat_frame_target": invalid})


def test_dollars_refuse_a_splat_selection_they_cannot_pose_and_never_shrink_it(tmp_path, monkeypatch):
    path, review = _forward_motion_job(tmp_path, budget=.5)  # pays for 50 frames, not 96
    _forward_motion(monkeypatch)
    with pytest.raises(ValueError, match="budget_below_accurate_inputs"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=tmp_path / "pose",
                                   splat_frame_target=96)
    assert not (tmp_path / "pose" / video.GEOMETRY_INPUTS_FILE).exists()


def test_a_selection_retained_at_64_resumes_with_its_own_capacity(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline, website_atlas_pose as pose_api
    path, review = _forward_motion_job(tmp_path)
    _forward_motion(monkeypatch)
    output = tmp_path / "pose"
    # A preserved job with a refined-splat price selected before this release.
    first = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
    plan = json.loads((output / "atlas_view_selection.json").read_text())
    assert plan["frame_policy"] == {"basis": "accuracy_first", "provider_input_capacity": 64, "frame_target": 64,
                                    "connected_cover_frames": plan["frame_policy"]["connected_cover_frames"],
                                    "distinct_views_added": plan["frame_policy"]["distinct_views_added"]}
    options = pipeline._splat_input_options(output, review=review, policy={"refined_splat_cost_usd": 5},
                                            edit_scope="all_selected_views")
    assert options == {}  # the recorded profile, not the new default
    monkeypatch.setattr(video, "_features", lambda *_: pytest.fail("retained inputs recomputed"))
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output, **options) == first
    # A later change to the configured capacity reads the plan's own recorded capacity instead.
    monkeypatch.setattr(pose_api, "MAX_POSE_INPUT_IMAGES", 128)
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output, **options) == first


def test_views_added_after_the_cover_clear_the_clips_relative_sharpness_floor(tmp_path, monkeypatch):
    """capture-5b067498: its soft opening passed the absolute floor at about half the clip's median sharpness."""
    path, review = _forward_motion_job(tmp_path)
    survey = json.loads(path.read_text())
    soft = {str(i) for i in range(30)}
    for row in survey["frames"]:
        row["sharpness"] = 40 if row["frame_id"] in soft else 80
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path.write_text(json.dumps(survey))
    review["survey_digest"] = survey["digest"]
    _forward_motion(monkeypatch)
    options = dict(survey_path=path, review=review, output_root=tmp_path / "pose", splat_frame_target=96,
                   relative_sharpness_fraction=video.RELATIVE_SHARPNESS_FRACTION)
    first = video.prepare_video_inputs(**options)
    plan = json.loads((tmp_path / "pose/atlas_view_selection.json").read_text())
    policy = plan["frame_policy"]
    assert policy["relative_sharpness_fraction"] == .6 and policy["relative_sharpness_floor"] == 48.0
    # The reviewed cover keeps its required soft view, named; no soft view is added after it.
    below = policy["cover_frames_below_relative_sharpness_floor"]
    assert "0" in below and set(below) <= soft
    assert {row["frame_id"] for row in first["frames"]} & soft == set(below)
    assert len(first["frames"]) == 96 and policy["splat_frame_target_met"] is True
    named = {row["frame_id"] for row in plan["diagnostics"] if row["decision"] == "below_relative_sharpness_floor"}
    assert named == soft - set(below)
    inputs = json.loads((tmp_path / "pose" / video.GEOMETRY_INPUTS_FILE).read_text())
    assert inputs["binding"]["relative_sharpness_fraction"] == .6
    # A wait reuses the retained handoff; a different profile never rewrites it.
    monkeypatch.setattr(video, "_features", lambda *_: pytest.fail("retained inputs recomputed"))
    assert video.prepare_video_inputs(**options) == first
    with pytest.raises(ValueError, match="retained_video_inputs_invalid"):
        video.prepare_video_inputs(**{**options, "relative_sharpness_fraction": 0})
    for invalid in (1, -.1, True, float("nan")):
        with pytest.raises(ValueError, match="atlas_selection_sharpness_profile_invalid"):
            video.prepare_video_inputs(**{**options, "relative_sharpness_fraction": invalid})


class _EditorReached(Exception):
    pass


def _left_out_job(tmp_path, monkeypatch, gap_role, *, blurred=False):
    """Six frames, 0-4 overlapping their neighbours; frame 5 overlaps nothing (or, blurred, is overlapped but
    below the quality policy). Frame 5 is reviewed as ``gap_role``; frame 2 is the anchor."""
    from PIL import Image
    from blueprint_pipeline import atlas_privacy_screen as screen
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    root = tmp_path / "survey"
    (root / "frames").mkdir(parents=True)
    rows = []
    for index in range(6):
        path = root / "frames" / f"{index}.png"
        Image.new("RGB", (32, 24), (index * 30, 90, 160)).save(path)
        rows.append({"frame_id": str(index), "timestamp_seconds": index * .5, "source_image_path": str(path),
                     "source_image_digest": _sha256_file(path), "sharpness": 5 if blurred and index == 5 else 50,
                     "clipped_fraction": 0})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False, "frames": rows,
              "display_rotation_degrees": 0, "source_video_digest": "sha256:" + "a" * 64, "sample_count": 6,
              "heldout_frame_count": 0}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (root / "atlas_video_survey.json").write_text(json.dumps(survey))
    requirements = [{"role": "task_object", "label": "object", "required_frame_ids": ["2"]},
                    {"role": gap_role, "label": "the gap", "required_frame_ids": ["5"]}]
    if gap_role != "surrounding_scene":
        requirements.append({"role": "surrounding_scene", "label": "room", "required_frame_ids": ["0"]})
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "reviewed_by": "test", "scene_state": "unchanged",
              "static_intervals_seconds": [[0, 3]], "anchor_frame_id": "2", "coverage_requirements": requirements,
              "automation": {"objects": [{"disposition": "remove"}]},
              "selection_budget": {"maximum_pose_cost_usd": 10, "input_cost_usd": .01, "rate_card_reference": "test",
                                   "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    monkeypatch.setattr(video, "_features", lambda row, rotation: int(row["frame_id"]))
    connected = (lambda a, b: abs(a - b) == 1) if blurred else (lambda a, b: abs(a - b) == 1 and 5 not in (a, b))
    monkeypatch.setattr(video, "_overlap", lambda a, b: (100, 30) if connected(a, b) else (0, 0))
    monkeypatch.setattr(screen, "blur_faces", lambda image: (image, []))
    return root / "atlas_video_survey.json", review


@pytest.mark.parametrize("gap_role, code", [("support_surface", "scene_estimate:task_views_left_out"),
                                            ("surrounding_scene", "scene_estimate:surroundings_not_connected")])
def test_a_left_out_view_reaches_the_background_editor_and_its_flags_reach_the_result(tmp_path, monkeypatch,
                                                                                      gap_role, code):
    """The editor-first route (an object to remove): the selection leaves frame 5 out and names it, the
    reviewed-background step asks only for the frames it kept, and the result says what was left out."""
    from blueprint_pipeline import atlas_background_automation as background
    from blueprint_pipeline import atlas_scene_automation as automation
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    survey_path, review = _left_out_job(tmp_path, monkeypatch, gap_role)
    assert video.takes_reviewed_background(review)
    inputs = video.prepare_video_inputs(survey_path=survey_path, review=review, output_root=tmp_path / "pose")
    assert inputs["left_out_frame_ids"] == ["5"] and "5" not in {row["frame_id"] for row in inputs["frames"]}
    monkeypatch.setattr(automation, "measurement_plan", lambda review: review["automation"])

    def editor(**kwargs):
        raise _EditorReached(kwargs["selected_frame_ids"])
    monkeypatch.setattr(background, "run_background", editor)
    with pytest.raises(_EditorReached) as reached:
        automation.run_automation(pose_root=None, inputs=inputs, review=review, output_root=tmp_path / "automation",
                                  task_context={}, configuration={"policy": {}}, task_runner=None, pose_runner=None,
                                  release_blockers=lambda: ([], ""))
    assert "5" not in reached.value.args[0] and "2" in reached.value.args[0]
    selection = json.loads((tmp_path / "pose/atlas_view_selection.json").read_text())
    result = pipeline.finalize_background_result(output_root=tmp_path, source_digest="sha256:" + "a" * 64,
        task_context={"context_digest": "sha256:" + "c" * 64}, automation={"status": "completed", "background": {}},
        source_pose=None, model="m", selection=selection)
    published = json.loads((tmp_path / "result.json").read_text())
    assert published == result and published["task_and_surrounding_scene_covered"] is False
    assert any(row.startswith("the gap: frame 5 ") for row in published["uncaptured_regions"])
    assert code in {row["code"] for row in published["scene_estimates"]}


@pytest.mark.parametrize("gap_role, blurred, decision", [
    ("support_surface", False, video.TASK_AREA_UNCONNECTED),
    ("surrounding_scene", False, "surrounding_scene_unconnected"),
    ("task_region", True, video.TASK_AREA_BELOW_QUALITY)])
def test_a_left_out_view_survives_the_next_tick_and_an_unrecorded_one_does_not(tmp_path, monkeypatch, gap_role,
                                                                               blurred, decision):
    """Tick 2 reads the retained handoff (``_retained_video_inputs``): the same inputs and flags as tick 1."""
    from blueprint_pipeline import atlas_scene_estimates as estimates
    survey_path, review = _left_out_job(tmp_path, monkeypatch, gap_role, blurred=blurred)
    first = video.prepare_video_inputs(survey_path=survey_path, review=review, output_root=tmp_path / "pose")
    monkeypatch.setattr(video, "_features", lambda *_: pytest.fail("tick 2 re-ran the selection"))
    second = video.prepare_video_inputs(survey_path=survey_path, review=review, output_root=tmp_path / "pose")
    assert second == first and second["left_out_frame_ids"] == ["5"]
    plan_path = tmp_path / "pose/atlas_view_selection.json"
    plan = json.loads(plan_path.read_text())
    assert {"frame_id": "5", "label": "the gap", "decision": decision} in plan["diagnostics"]
    assert estimates.scene_estimates(levelling={}, registration={}, selection=plan)
    # Tampered: the plan no longer records why frame 5 is missing.
    tampered = {**plan, "diagnostics": [row for row in plan["diagnostics"] if row.get("frame_id") != "5"
                                        and row.get("decision") != video.SURROUNDINGS_NOT_CONNECTED],
                "task_and_surrounding_scene_covered": True}  # consistent with the tampered decisions
    plan_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=survey_path, review=review, output_root=tmp_path / "pose")
    # Tampered in both records alike, digests rebound: frame 5 is missing for no recorded reason.
    inputs_path = tmp_path / "pose" / video.GEOMETRY_INPUTS_FILE
    inputs = json.loads(inputs_path.read_text())
    inputs.pop("left_out_frame_ids")
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    inputs_path.write_text(json.dumps(inputs))
    plan_path.write_text(json.dumps({**tampered, "input_digest": inputs["digest"]}))
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=survey_path, review=review, output_root=tmp_path / "pose")
