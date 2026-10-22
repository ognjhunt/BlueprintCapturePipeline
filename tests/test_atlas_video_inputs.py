"""Source selection cannot promote state review, heldout pixels or generated views."""
import json

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
