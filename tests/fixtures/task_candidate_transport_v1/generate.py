"""Reproduce task-discovery/approval transport bytes; development-only, no provider calls."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from blueprint_pipeline.task_candidate_control_plane import (
    build_task_discovery_webapp_publication,
    process_task_candidate_decision_submission,
    publish_task_candidate_discovery,
)
from blueprint_pipeline.task_candidate_discovery import build_task_candidate_discovery
from tests.test_task_candidate_control_plane import _discovery, _submission


def fixtures():
    base = _discovery()
    proposal = {**base["task_candidates"][0], "confidence": 1.0, "estimated_evaluation_cost_usd": 0.0,
        "proposed_measurable_success_condition": {
            **base["task_candidates"][0]["proposed_measurable_success_condition"], "threshold": 1.0}}
    discovery = build_task_candidate_discovery(discovery_id="software-transport-discovery",
        source_capture=base["source_capture"], capture_qa_report_digest=base["capture_qa_report_digest"],
        scene_analysis=base["scene_analysis"], candidate_proposals=[proposal], proposal_method=base["proposal_method"])
    publication = build_task_discovery_webapp_publication(capture_session_id="capture-review-1",
        intake_id="intake-1", discovery=discovery)
    submission = _submission(discovery)
    submission["command"]["action"] = "edit_and_approve"
    submission["command"]["edited_task"] = {
        "description": "Move the observed blue tote into the marked box.", "task_family": "rigid_object_pick_place",
        "measurable_success_conditions": [{"metric": "object_center_distance", "operator": "<=", "threshold": 1.0, "units": "m"}],
        "reset_contract": {"description": "Restore the observed tote and marked box.", "position_tolerance_m": 1.0},
    }
    with tempfile.TemporaryDirectory() as directory, patch(
        "blueprint_pipeline.task_candidate_control_plane.utc_now_iso", return_value="2026-10-21T12:00:00Z"):
        publish_task_candidate_discovery(state_root=directory, capture_session_id="capture-review-1",
            intake_id="intake-1", discovery=discovery)
        decision = process_task_candidate_decision_submission(state_root=directory, submission=submission)
        replay = process_task_candidate_decision_submission(state_root=directory, submission=submission)
    return {"provenance": {"scope": "development_only", "provider_execution": False, "physical_evidence": False,
            "generator": "tests/fixtures/task_candidate_transport_v1/generate.py", "clock_fixed_for_reproducibility": True},
        "publication": publication, "submission": submission, "decision": decision, "replay": replay}


if __name__ == "__main__":
    Path(__file__).with_name("producer-examples.json").write_text(
        json.dumps(fixtures(), sort_keys=True, indent=2, ensure_ascii=False) + "\n")
