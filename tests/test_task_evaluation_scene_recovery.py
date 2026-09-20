"""Durable successor attempts preserve failed evidence and aggregate authority."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

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


def test_unrelated_scientific_failure_cannot_be_relabeled_as_create_refusal(tmp_path):
    intent, _, evidence = setup(tmp_path)
    producer = write(tmp_path / "producer.json", {"status": "failed", "blockers": ["invalid_tracking"]})
    failure = json.loads((tmp_path / "failure.json").read_text())
    failure["producer_result"] = producer
    evidence["failure"] = write(tmp_path / "failure.json", failure, "failure_digest")
    with pytest.raises(ValueError, match="create_failure_evidence_missing"):
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


def _dead_machine_producer(**overrides):
    """A gaussian-excision run whose machine started then exited before any output."""
    value = {
        "schema_version": "adp009b_gaussian_excision_vast_run.v1",
        "status": "blocked",
        "blockers": [
            "gaussian_excision_execution_not_completed",
            "gaussian_excision_provider_output_zip_missing",
            "vast_heartbeat_instance_exited",
        ],
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "raw_secret_values_recorded": False,
    }
    value.update(overrides)
    return value


def test_dead_provider_machine_is_recognised_as_a_provider_null():
    """Scene 840938, 2026-09-16: a $0.06 dead box parked the whole hands-off run.

    Vast created the instance, the container never reached the on-start heartbeat,
    and the run sealed bare blockers with no provider_attempt_classification, so
    scene recovery recognised nothing and the intent stopped at
    source_preparation/preparation_failed with max_retries untouched.
    """
    from blueprint_pipeline.task_evaluation_scene_recovery import (
        gaussian_excision_dead_machine_evidence, provider_null_evidence, recovery_budget,
    )

    producer = _dead_machine_producer()
    assert gaussian_excision_dead_machine_evidence(producer) is True
    assert provider_null_evidence(producer) is True
    # A machine that ran and died consumed a real resource: ordinary retry budget,
    # not the marketplace-miss budget reserved for $0 no-instance misses.
    assert recovery_budget("provider_null", producer) == "retry"


def test_dead_machine_recovery_requires_proof_the_machine_died():
    """Execution failures on a live machine must never be recycled as provider nulls."""
    from blueprint_pipeline.task_evaluation_scene_recovery import (
        gaussian_excision_dead_machine_evidence,
    )

    # Same job, but the heartbeat proof is absent: the work itself failed.
    work_failed = _dead_machine_producer(blockers=[
        "gaussian_excision_execution_not_completed",
        "gaussian_excision_provider_output_zip_missing",
    ])
    assert gaussian_excision_dead_machine_evidence(work_failed) is False

    # An unrelated blocker alongside the proof is not a clean dead machine either.
    mixed = _dead_machine_producer(blockers=[
        "vast_heartbeat_instance_exited", "gaussian_excision_scene_inputs_invalid",
    ])
    assert gaussian_excision_dead_machine_evidence(mixed) is False


@pytest.mark.parametrize("override", [
    {"status": "completed"},
    {"continuing_spend_from_this_run": True},
    {"all_staged_objects_absent": False},
    {"raw_secret_values_recorded": True},
    {"retained_owned": True},
    {"schema_version": "adp009b_gaussian_excision_vast_run.v2"},
    {"blockers": []},
])
def test_dead_machine_recovery_fails_closed(override):
    """Anything still running, retained, or differently sealed is not recoverable."""
    from blueprint_pipeline.task_evaluation_scene_recovery import (
        gaussian_excision_dead_machine_evidence,
    )

    assert gaussian_excision_dead_machine_evidence(_dead_machine_producer(**override)) is False


@pytest.mark.parametrize("mismatched_execution", [False, True])
def test_preparation_release_reconciles_its_bound_execution_provider(tmp_path, mismatched_execution):
    from blueprint_pipeline.task_evaluation_scene_progression_recovery import reconcile_ownership
    _, execution, _ = setup(tmp_path)
    preparation = {**execution, "provider": "control_plane", "maximum_spend_usd": 0}
    if mismatched_execution:
        execution = {**execution, "source_commit": "f" * 40}
    kwargs = dict(attempt=preparation, execution_attempt=execution,
        failure_path=tmp_path / "failure.json", output_root=tmp_path / "reconciliation", now=104,
        config={"provider_guard_path": str(tmp_path / "guard.json"), "ownership_roots": [str(tmp_path)],
            "child_execution_root": str(tmp_path / "children"), "launch_execution_root": str(tmp_path / "launches"),
            "child_queue_root": str(tmp_path), "launch_queue_root": str(tmp_path)})
    if mismatched_execution:
        with pytest.raises(ValueError, match="recovery_execution_attempt_mismatch"):
            reconcile_ownership(**kwargs)
    else:
        receipt = reconcile_ownership(**kwargs)
        owner = json.loads(Path(receipt["ownership_reconciliation"]["path"]).read_text())
        assert owner["attempt_digest"] == preparation["attempt_digest"]
        assert owner["status"] == "closed_without_resource"
