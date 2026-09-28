from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    TaskEvaluationLaunchPreparationQueueError,
    launch_preparation_status,
    stage_launch_preparation_request,
    write_launch_preparation_record_exclusive,
)
import blueprint_pipeline.task_evaluation_launch_preparation_queue as queue_module
from tests.test_task_evaluation_launch_preparation_contract import request


def test_stages_exact_no_spend_request_and_reports_status(tmp_path) -> None:
    receipt = stage_launch_preparation_request(
        value=request(), queue_root=tmp_path / "queue", submitted_by="webapp-service"
    )
    assert receipt["status"] == "queued_for_no_spend_preparation"
    assert receipt["provider_mutation_performed_inside_http_request"] is False
    assert receipt["catalog_mutation_performed_inside_http_request"] is False
    assert receipt["paid_execution_requested"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    path = tmp_path / "queue" / "pending" / receipt["queue_path"].split("/")[-1]
    envelope = json.loads(path.read_text())
    assert envelope["schema_version"] == ENVELOPE_SCHEMA_VERSION
    assert envelope["request_digest"] == receipt["request_digest"]
    assert envelope["envelope_digest"] == canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    assert path.stat().st_mode & 0o777 == 0o440
    assert launch_preparation_status(
        preparation_id=request()["preparation_id"], queue_root=tmp_path / "queue"
    )["status"] == "pending"


def test_exact_replay_is_idempotent_and_changed_request_conflicts(tmp_path) -> None:
    queue = tmp_path / "queue"
    first = stage_launch_preparation_request(
        value=request(), queue_root=queue, submitted_by="webapp-service"
    )
    replay = stage_launch_preparation_request(
        value=request(), queue_root=queue, submitted_by="webapp-service"
    )
    assert replay["already_exists"] is True
    assert replay["request_digest"] == first["request_digest"]

    changed = copy.deepcopy(request())
    changed["task"]["identity"]["version"] = "v2"
    with pytest.raises(
        TaskEvaluationLaunchPreparationQueueError,
        match="launch_preparation_id_immutable_conflict",
    ):
        stage_launch_preparation_request(
            value=changed, queue_root=queue, submitted_by="webapp-service"
        )


def test_status_fails_closed_on_tampered_envelope(tmp_path) -> None:
    queue = tmp_path / "queue"
    receipt = stage_launch_preparation_request(
        value=request(), queue_root=queue, submitted_by="webapp-service"
    )
    path = queue / "pending" / receipt["queue_path"].split("/")[-1]
    path.chmod(0o640)
    envelope = json.loads(path.read_text())
    envelope["request"]["run_id"] = "different-run"
    path.write_text(json.dumps(envelope))
    with pytest.raises(
        TaskEvaluationLaunchPreparationQueueError,
        match="launch_preparation_queue_envelope_invalid",
    ):
        launch_preparation_status(
            preparation_id=request()["preparation_id"], queue_root=queue
        )


def test_concurrent_writers_cannot_create_two_identities(tmp_path) -> None:
    queue = tmp_path / "queue"
    first = request()
    second = copy.deepcopy(first)
    second["task"]["identity"]["version"] = "v2"

    def stage(value):
        try:
            return stage_launch_preparation_request(
                value=value, queue_root=queue, submitted_by="webapp-service"
            )["request_digest"]
        except TaskEvaluationLaunchPreparationQueueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(stage, (first, second)))

    assert sum(item.startswith("sha256:") for item in outcomes) == 1
    assert outcomes.count("launch_preparation_id_immutable_conflict") == 1
    assert len(list((queue / "identities").glob("*.json"))) == 1
    assert len(list((queue / "pending").glob("*.json"))) == 1
    assert list(queue.rglob("*.tmp")) == []


def test_exclusive_writer_publishes_only_complete_bytes(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "sealed.json"
    first_write_started = Event()
    allow_first_writer_to_finish = Event()
    writer_lock = Lock()
    first_writer_id: int | None = None
    original_write = queue_module.os.write

    def delayed_first_write(descriptor: int, payload: memoryview) -> int:
        nonlocal first_writer_id
        with writer_lock:
            if first_writer_id is None:
                first_writer_id = get_ident()
        if get_ident() == first_writer_id and not first_write_started.is_set():
            first_write_started.set()
            assert allow_first_writer_to_finish.wait(timeout=5)
        return original_write(descriptor, payload)

    monkeypatch.setattr(queue_module.os, "write", delayed_first_write)
    second = {"writer": "second", "payload": "complete"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            write_launch_preparation_record_exclusive,
            destination,
            {"writer": "first", "payload": "delayed"},
        )
        try:
            assert first_write_started.wait(timeout=5)
            assert not destination.exists()
            second_future = pool.submit(
                write_launch_preparation_record_exclusive,
                destination,
                second,
            )
            assert second_future.result(timeout=5) is None
        finally:
            allow_first_writer_to_finish.set()
        with pytest.raises(FileExistsError):
            first_future.result(timeout=5)

    assert json.loads(destination.read_text()) == second
    assert list(tmp_path.glob("*.tmp")) == []
