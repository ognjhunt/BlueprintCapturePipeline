"""Reviewer probe (finding 5): a launch-time reach finding on exactly its own row, end to end."""
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline.robot_arm_path_clearance import REACH_CANNOT_SPAN_FLAG
from tests.test_beta_policy_screening_coordinator import plan as queue_plan
from tests.test_beta_policy_screening_known_good_control import (PerRow, done, run, summaries,  # noqa: F401
                                                                  PUBLICATION_SCHEMA)


class Flagging(PerRow):
    def prepare_candidate(self, **kwargs):
        self.preparations += 1
        extra = {"launch_scene_check_flags": [REACH_CANNOT_SPAN_FLAG]} if kwargs["candidate"]["candidate_id"] == "candidate-001" else {}
        return {"sealed": "local-inputs", **extra}


def test_flag_on_exactly_its_row(tmp_path, summaries):  # noqa: F811
    coordinator.enqueue_screening_campaign(queue_plan(count=3), queue_root=tmp_path)
    backend = Flagging(done("candidate-000", "candidate-001", "candidate-002"))
    [mid] = run(tmp_path, backend, passes=2)
    [pub] = run(tmp_path, backend, passes=8)
    [again] = run(tmp_path, backend, passes=2)  # a re-publication after more passes
    rows = {r["candidate_id"]: r for r in pub["candidates"]}
    print({k: (r["status"], r.get("scene_check_flags")) for k, r in rows.items()}, pub.get("scene_feasibility"))
    assert rows["candidate-001"]["status"] == "completed"
    assert rows["candidate-001"].get("scene_check_flags") == [REACH_CANNOT_SPAN_FLAG]
    assert "scene_check_flags" not in rows["candidate-000"] and "scene_check_flags" not in rows["candidate-002"]
    assert {r["candidate_id"]: r.get("scene_check_flags") for r in again["candidates"]} == {
        k: r.get("scene_check_flags") for k, r in rows.items()}
    Draft202012Validator(json.loads(Path(PUBLICATION_SCHEMA).read_text())).validate(pub)
