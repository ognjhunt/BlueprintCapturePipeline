"""A paid-stage wait reuses validated CPU inputs without rebuilding or weakening them."""
import json
from datetime import datetime, timezone

import pytest
from PIL import Image

from blueprint_pipeline import atlas_privacy_screen as privacy
from blueprint_pipeline import atlas_video_inputs as video
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "survey"
    source.mkdir()
    frames = []
    for index in range(3):
        path = source / f"{index}.png"
        Image.new("RGB", (64, 48), (index * 40, 80, 160)).save(path)
        frames.append({"frame_id": str(index), "timestamp_seconds": index,
                       "source_image_path": str(path), "source_image_digest": _sha256_file(path),
                       "sharpness": 100, "clipped_fraction": 0})
    survey = {"schema_version": video.SURVEY_SCHEMA, "heldout_pixels_included": False,
              "frames": frames, "display_rotation_degrees": 0,
              "source_video_digest": "sha256:" + "a" * 64, "sample_count": 4, "heldout_frame_count": 1}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = source / "survey.json"
    path.write_text(json.dumps(survey))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"],
              "moving_people_excluded": True, "same_scene_state": True, "reviewed_by": "fixture",
              "scene_state": "unchanged", "static_intervals_seconds": [[0, 2]], "anchor_frame_id": "0",
              "coverage_requirements": [
                  {"role": "task_object", "label": "source", "required_frame_ids": ["0"]},
                  {"role": "support_surface", "label": "destination", "required_frame_ids": ["1"]},
                  {"role": "surrounding_scene", "label": "room", "required_frame_ids": ["2"]}],
              "selection_budget": {"maximum_pose_cost_usd": 1, "input_cost_usd": .01,
                                   "rate_card_reference": "fixture", "reviewed_by": "fixture",
                                   "reviewed_on": datetime.now(timezone.utc).date().isoformat()}}
    monkeypatch.setattr(video, "_features", lambda *_: None)
    monkeypatch.setattr(video, "select_coverage_views", lambda **_: ([0, 1, 2], []))
    monkeypatch.setattr(privacy, "blur_faces", lambda image: (image.copy(), []))
    output = tmp_path / "pose"
    result = video.prepare_video_inputs(survey_path=path, review=review, output_root=output)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("completed input preparation was recomputed")

    monkeypatch.setattr(video, "_features", no_rebuild)
    monkeypatch.setattr(video, "select_coverage_views", no_rebuild)
    monkeypatch.setattr(video, "add_distinct_views", no_rebuild)
    monkeypatch.setattr(privacy, "blur_faces", no_rebuild)
    return path, review, output, result


def test_paid_stage_retry_reuses_the_exact_cpu_handoff_without_writes(prepared):
    path, review, output, result = prepared
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    assert video.prepare_video_inputs(survey_path=path, review=review, output_root=output) == result
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before


@pytest.mark.parametrize("artifact", ["source", "observation", "screened", "missing_screened"])
def test_reuse_refuses_changed_or_missing_source_and_screened_bytes(prepared, artifact):
    path, review, output, _ = prepared
    target = {"source": path.parent / "0.png", "observation": output / "atlas_observations/0.png",
              "screened": output / "atlas_source_views/0.png",
              "missing_screened": output / "atlas_source_views/0.png"}[artifact]
    if artifact == "missing_screened":
        target.unlink()
    else:
        target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="atlas_source_frame_changed|website_source_geometry_artifact_changed"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)


@pytest.mark.parametrize("change", ["metadata", "privacy", "heldout", "selection", "required_support"])
def test_self_hashed_edits_cannot_change_the_retained_handoff(prepared, change):
    path, review, output, _ = prepared
    input_path, plan_path = output / video.GEOMETRY_INPUTS_FILE, output / "atlas_view_selection.json"
    inputs, plan = json.loads(input_path.read_text()), json.loads(plan_path.read_text())
    if change == "metadata":
        inputs["frames"][0]["timestamp_seconds"] = .25
    elif change == "privacy":
        inputs["binding"]["privacy_screen"]["blurred_faces_by_frame"]["0"] = False
        plan["privacy_screen"] = inputs["binding"]["privacy_screen"]
    elif change == "heldout":
        inputs["heldout_frame_count"] = 0
    elif change == "selection":
        plan["frames"][0]["image_digest"] = "sha256:" + "b" * 64
    else:
        inputs["frames"] = [row for row in inputs["frames"] if row["frame_id"] != "1"]
        plan["frames"] = [row for row in plan["frames"] if row["frame_id"] != "1"]
        plan["selected_frame_count"] = 2
        plan["frame_policy"]["connected_cover_frames"] = 2
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    plan["input_digest"] = inputs["digest"]
    input_path.write_text(json.dumps(inputs))
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)


def test_reuse_refuses_a_changed_review(prepared):
    path, review, output, _ = prepared
    with pytest.raises(ValueError, match="atlas_selection_review_conflict"):
        video.prepare_video_inputs(survey_path=path, review={**review, "scene_state": "changed"}, output_root=output)


def test_reuse_refuses_a_changed_privacy_screen_version(prepared, monkeypatch):
    path, review, output, _ = prepared
    monkeypatch.setattr(privacy, "screen_identity", lambda: {"model_sha256": "changed"})
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)


def test_changed_capacity_requires_a_new_handoff_instead_of_rewriting_it(prepared):
    path, review, output, _ = prepared
    with pytest.raises(ValueError, match="atlas_retained_video_inputs_invalid"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output, maximum_frames=3)


def test_incomplete_preparation_uses_the_existing_build_path(prepared):
    path, review, output, _ = prepared
    (output / "atlas_view_selection.json").unlink()
    with pytest.raises(AssertionError, match="completed input preparation was recomputed"):
        video.prepare_video_inputs(survey_path=path, review=review, output_root=output)
