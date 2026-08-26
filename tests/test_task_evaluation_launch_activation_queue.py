from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_activation_queue import (
    TaskEvaluationLaunchActivationQueueError,
    launch_activation_status,
    stage_launch_activation_request,
)
from tests.test_task_evaluation_launch_activation_contract import request


def test_stages_safe_authority_gated_activation_and_exact_replay(tmp_path) -> None:
    value = request()
    first = stage_launch_activation_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="blueprint-webapp"
    )
    replay = stage_launch_activation_request(
        value=value, queue_root=tmp_path / "queue", submitted_by="blueprint-webapp"
    )
    assert first["status"] == "queued_for_authority_gated_activation"
    assert first["already_exists"] is False
    assert replay["already_exists"] is True
    assert first["request_digest"] == replay["request_digest"]
    assert "queue_path" not in first
    assert first["receipt_digest"] == canonical_digest(
        first, digest_field="receipt_digest"
    )
    assert first["provider_mutation_performed_inside_http_request"] is False
    assert first["catalog_mutation_performed_inside_http_request"] is False
    assert first["standing_authorization_published_inside_http_request"] is False
    assert first["paid_execution_requested"] is False
    assert launch_activation_status(
        activation_id=value["activation_id"], queue_root=tmp_path / "queue"
    )["status"] == "pending"


def test_changed_activation_id_payload_conflicts_before_second_queue_entry(
    tmp_path,
) -> None:
    queue = tmp_path / "queue"
    first = request()
    changed = copy.deepcopy(first)
    changed["authorization"]["profile_revision"] = "r2"
    stage_launch_activation_request(
        value=first, queue_root=queue, submitted_by="blueprint-webapp"
    )
    with pytest.raises(
        TaskEvaluationLaunchActivationQueueError,
        match="launch_activation_id_immutable_conflict",
    ):
        stage_launch_activation_request(
            value=changed, queue_root=queue, submitted_by="blueprint-webapp"
        )
    assert len(list((queue / "pending").glob("*.json"))) == 1


def test_concurrent_different_writers_leave_one_activation_identity(tmp_path) -> None:
    queue = tmp_path / "queue"
    first = request()
    second = copy.deepcopy(first)
    second["authorization"]["profile_revision"] = "r2"

    def stage(value):
        try:
            return stage_launch_activation_request(
                value=value, queue_root=queue, submitted_by="blueprint-webapp"
            )["request_digest"]
        except TaskEvaluationLaunchActivationQueueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(stage, (first, second)))
    assert sum(item.startswith("sha256:") for item in outcomes) == 1
    assert outcomes.count("launch_activation_id_immutable_conflict") == 1
    assert len(list((queue / "pending").glob("*.json"))) == 1
    assert list(queue.rglob("*.tmp")) == []
