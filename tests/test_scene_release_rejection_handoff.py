"""A new worker's no-execution refusal cannot strand an old owner intent."""
import json

import pytest

from blueprint_pipeline.task_evaluation_scene_progression import _queue
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import stage_launch_preparation_request
from tests.test_task_evaluation_launch_preparation_worker import production_request_with_fetchable_bytes


@pytest.mark.parametrize("defect", [None, "success", "provider_mutation_performed", "paid_execution_requested",
                                  "catalog_mutation_performed", "different_reason", "different_parent"])
def test_foreign_release_is_accepted_only_as_exact_no_execution_refusal(tmp_path, defect):
    request, _ = production_request_with_fetchable_bytes()
    stage_launch_preparation_request(value=request, queue_root=tmp_path, submitted_by="fixture")
    pending = next((tmp_path / "pending").glob("*.json"))
    blocked = tmp_path / "blocked" / pending.name
    pending.rename(blocked)
    result = {"schema_version": "task_evaluation_launch_preparation_result.v1", "status": "blocked",
        "preparation_id": request["preparation_id"], "source_commit": "f" * 40,
        "blockers": ["launch_preparation_worker_source_commit_mismatch"],
        "paid_execution_requested": False, "provider_mutation_performed": False, "catalog_mutation_performed": False}
    if defect == "success":
        result["status"] = "queued_for_production_scene_configuration"
    elif defect == "different_reason":
        result["blockers"] = ["unrelated_failure"]
    elif defect == "different_parent":
        result["preparation_id"] += "-different"
    elif defect:
        result[defect] = True
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path = tmp_path / "results" / pending.name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result))
    before = path.read_bytes()
    if defect:
        with pytest.raises(ValueError, match="preparation_result_mismatch"):
            _queue(request, tmp_path)
    else:
        observed = _queue(request, tmp_path)
        assert observed["status"] == "blocked" and observed["result"] == result
        assert observed["result"]["source_commit"] != request["expected_production_commit"]
    assert path.read_bytes() == before
