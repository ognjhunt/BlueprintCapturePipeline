"""Admission delivery preserves task identity across loss and independent refusals."""

import json
import os
import time

from blueprint_pipeline.agent_execution.contracts import AgentTask
from blueprint_pipeline.agent_execution.production import TaskRecord
from blueprint_pipeline.agent_execution.webapp_delivery import WebappAdmissionOutbox, admission_payload
from tests.test_agent_production_service import fixture


def owned_record(service, task):
    value = service.record(task.task_id).model_dump(mode="json")
    value["owner_client_ids"] = ["blueprint-webapp"]
    return TaskRecord.model_validate(value)


def acknowledgement(payload):
    return {"schema_version": "blueprint_webapp_agent_admission_receipt.v1", "admission": payload, "proof_effect": "none"}


def test_admission_projection_contains_no_prompts_paths_credentials_or_authority(tmp_path):
    service, task, _, _ = fixture(tmp_path)
    assert admission_payload(service.record(task.task_id)) is None
    payload = admission_payload(owned_record(service, task))
    assert payload["task_digest"] == task.task_digest
    assert payload["runtime"] == "openai_agents_sdk"
    text = json.dumps(payload)
    for private in (str(tmp_path), "sk-fixture", "authority_envelope", "customer_question", task.instructions):
        assert private not in text


def test_lost_reply_retains_same_payload_and_success_deduplicates_after_restart(tmp_path):
    service, task, _, _ = fixture(tmp_path)
    records, calls = {}, []
    def post(payload):
        calls.append(payload.copy())
        records.setdefault(payload["task_id"], payload.copy())
        if len(calls) == 1:
            raise OSError("connection lost after receiver stored record")
        return acknowledgement(records[payload["task_id"]])
    outbox = WebappAdmissionOutbox(service.journal, post=post)
    record = owned_record(service, task)
    outbox.queue(record)
    assert outbox.flush()[0]["status"] == "delivery_pending"
    resumed = WebappAdmissionOutbox(service.journal, post=post)
    assert resumed.flush()[0]["status"] == "stored_in_webapp"
    resumed.queue(record)
    assert resumed.flush() == []
    assert calls[0] == calls[1]
    assert len(records) == 1


def test_foreign_readback_is_not_delivery_and_does_not_starve_other_tasks(tmp_path):
    service, task, _, _ = fixture(tmp_path)
    calls = []
    record = owned_record(service, task)
    value = task.model_dump(mode="json")
    value["task_id"] = "other-task"
    other = TaskRecord.model_validate({**record.model_dump(mode="json"), "task": AgentTask.model_validate(value).model_dump(mode="json")})
    def post(payload):
        calls.append(payload["task_id"])
        response = acknowledgement(payload)
        if payload["task_id"] == task.task_id:
            response["admission"] = {**payload, "task_digest": "sha256:" + "f" * 64}
        return response
    outbox = WebappAdmissionOutbox(service.journal, post=post)
    outbox.queue(record)
    outbox.queue(other)
    # Queue timestamps can be tied/coarse or ahead of the current wall clock.
    # A refused entry must still move behind every currently waiting entry.
    future = time.time_ns() + 60_000_000_000
    for path in outbox.pending.glob('*.json'):
        order = 0 if json.loads(path.read_text())['task_id'] == task.task_id else 1
        os.utime(path, ns=(future, future + order))
    assert outbox.flush()[0]["status"] == "delivery_pending"
    assert outbox.flush()[0]["status"] == "stored_in_webapp"
    assert calls == [task.task_id, other.task.task_id]
