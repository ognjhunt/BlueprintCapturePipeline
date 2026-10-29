from types import SimpleNamespace
import pytest
from blueprint_pipeline import website_capture_preparation as preparation
from blueprint_pipeline.website_target_background_correction import retains_rejected_selection
from tests.test_website_target_background_correction import rejected_job


def test_negative_owner_answer_still_supersedes_a_privately_archived_observed_room(tmp_path, monkeypatch):
    root, context, source = rejected_job(tmp_path)
    def forbidden(**_):
        pytest.fail("An explicit negative answer must be retained before the room-only completion path")
    monkeypatch.setattr("blueprint_pipeline.website_observed_room_limit.retain_observed_room_limit", forbidden)
    result = preparation.run_website_capture_preparation(
        descriptor=SimpleNamespace(metadata={"site_task_context":context}), capture_root=root,
        pipeline_dir=root / "pipeline", source_video=source, derived_scene_generation_allowed=True,
        gates=[], prepare_media=forbidden, check_privacy=forbidden, make_gate=forbidden)
    assert result["status"] == "completed" and result["native_execution_complete"] is False
    assert retains_rejected_selection(result["website_scene_preparation"], context)
    assert result["website_scene_preparation"]["selection_rejection"]["additional_footage_required"] is False
