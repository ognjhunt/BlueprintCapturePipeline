"""Durable successor attempts preserve failed evidence and aggregate authority."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_intake import reserve_scene_attempt
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
from tests.test_task_evaluation_scene_intake import request, stage, attempt


def write(path, value, field=None):
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return record(path)


def setup(tmp_path, retries=1):
    value = request()
    value["execution"].update(max_retries=retries, max_paid_attempts=8, max_total_spend_usd=20)
    intent = stage(tmp_path, value)
    first = attempt(tmp_path, intent)
    producer = write(tmp_path / "producer.json", {"status": "failed", "allocation_created": False})
    failure = write(tmp_path / "failure.json", {"schema_version": "task_evaluation_scene_attempt_failure.v1",
        "attempt_digest": first["attempt_digest"], "status": "failed", "failure_kind": "create_ambiguous",
        "observed_at_epoch": 102, "producer_result": producer}, "failure_digest")
    guard = write(tmp_path / "guard.json", {"schema_version": "gpu_spend_guard.v1",
        "generated_at": datetime.fromtimestamp(103, timezone.utc).isoformat(), "reap_mode": True,
        "provider_zero_verified": True, "live_instance_count": 0, "total_burn_per_hour_usd": 0,
        "reap_candidate_ids": [], "reap_results": [], "provider_zero": {"status": "verified",
            "global_live_instance_count": 0, "global_total_burn_per_hour_usd": 0,
            "required_provider_ids": ["vast"]},
        "inventory_results": [{"provider": "vast", "status": "succeeded", "row_count": 0, "required": True}]})
    owner = write(tmp_path / "owner.json", {"schema_version": "task_evaluation_scene_attempt_ownership.v1",
        "attempt_digest": first["attempt_digest"], "status": "closed_without_resource",
        "active_writer_count": 0, "unresolved_create_count": 0, "provider_guard": guard,
        "observed_at_epoch": 103}, "ownership_digest")
    return intent, first, {"failure": failure, "provider_guard": guard, "ownership_reconciliation": owner}


def recover(root, intent, evidence, name="a2", now=104):
    return reserve_scene_attempt(queue_root=root, intent_id=intent["intent_id"], attempt_id=name,
        source_commit="c" * 40, runtime_digest="sha256:" + "e" * 64,
        input_digest="sha256:" + "f" * 64, provider="vast", maximum_spend_usd=2,
        now=now, recovery_from_attempt_id="a1", recovery_evidence=evidence)


def test_recovery_is_new_immutable_attempt_and_keeps_failure_and_exposure(tmp_path):
    intent, first, evidence = setup(tmp_path)
    retained = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    second = recover(tmp_path, intent, evidence)
    assert second["recovery"]["prior_attempt_digest"] == first["attempt_digest"]
    assert recover(tmp_path, intent, evidence, now=800) == second
    assert all(p.read_bytes() == raw for p, raw in retained.items())
    rows = list((tmp_path / intent["intent_id"] / "attempts").glob("*.json"))
    assert len(rows) == 2
    assert sum(json.loads(p.read_text())["maximum_spend_usd"] for p in rows) == 4


@pytest.mark.parametrize("fault", ["zero", "stale", "writer", "ambiguous", "wrong_attempt", "bytes"])
def test_no_successor_without_current_global_and_ownership_reconciliation(tmp_path, fault):
    intent, _first, evidence = setup(tmp_path)
    if fault == "bytes":
        (tmp_path / "producer.json").write_text('{}')
    elif fault in {"zero", "stale"}:
        path = tmp_path / "guard.json"
        value = json.loads(path.read_text())
        value["provider_zero"]["global_live_instance_count"] = 1 if fault == "zero" else 0
        if fault == "stale":
            value["generated_at"] = datetime.fromtimestamp(100, timezone.utc).isoformat()
        evidence["provider_guard"] = write(path, value)
    else:
        path = tmp_path / "owner.json"
        value = json.loads(path.read_text())
        key = {"writer": "active_writer_count", "ambiguous": "unresolved_create_count", "wrong_attempt": "attempt_digest"}[fault]
        value[key] = "sha256:" + "0" * 64 if fault == "wrong_attempt" else 1
        evidence["ownership_reconciliation"] = write(path, value, "ownership_digest")
    with pytest.raises(ValueError):
        recover(tmp_path, intent, evidence)
    assert len(list((tmp_path / intent["intent_id"] / "attempts").glob("*.json"))) == 1


def test_zero_retry_consent_cannot_be_bypassed_by_paid_attempt_capacity(tmp_path):
    intent, _, evidence = setup(tmp_path, retries=0)
    with pytest.raises(ValueError, match="retry_cap_exhausted"):
        recover(tmp_path, intent, evidence)


def test_concurrent_successors_cannot_reset_retry_count(tmp_path):
    intent, _, evidence = setup(tmp_path)
    def run(name):
        try:
            return recover(tmp_path, intent, copy.deepcopy(evidence), name)
        except ValueError:
            return None
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(run, ["a2", "a3"]))
    assert sum(row is not None for row in rows) == 1
