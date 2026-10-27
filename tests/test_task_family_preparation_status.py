"""Family handoffs retain source-bound status without legacy intake retries."""
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "site-req1/captures/walkthrough-req1"
    root.mkdir(parents=True)
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
        "capture_id": "walkthrough-req1", "task_family": "cloth_folding", "description": "Fold the observed cloth",
        "confirmed": True, "confirmed_at": "2026-10-27T00:00:00Z"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    write_json(root / "pipeline/website_task_context.json", context)
    selected = {key: context[key] for key in ("request_id", "scene_id", "capture_id")}
    selected.update(completion_marker_generation="19", producer_delivery_key="sha256:" + "a" * 64,
        source_payload_sha256="b" * 64, task_context_digest=context["context_digest"])
    producer = {"delivery_key": selected["producer_delivery_key"],
        "raw_video": {"generation": "7", "object_name": "owned/video.mp4"}}
    delivery_path = root / "delivery.json"
    write_json(delivery_path, {"source_finalize": {"generation": "19", "bucket": "fixture"}, "producer_delivery": producer})
    birth = {key: context[key] for key in ("request_id", "scene_id", "capture_id")}
    birth.update(delivery_key=selected["producer_delivery_key"], birth_delivery_raw_ref={"path": str(delivery_path)},
        capture_owner_user_id="owner1", raw_video=producer["raw_video"])
    ledger = {"schema_version": "pipeline_job_ledger.v1", "revision": 1, "attempt_count": 1, "status": "completed",
        **{key: selected[key] for key in ("scene_id", "capture_id", "producer_delivery_key", "source_payload_sha256")}}
    monkeypatch.setattr("blueprint_pipeline.handoff_job_state._read_job_ledger", lambda _: copy.deepcopy(ledger))
    monkeypatch.setattr("blueprint_pipeline.handoff_job_state._output_commit", lambda *_, **__: {"committed": True})
    monkeypatch.setattr("blueprint_pipeline.task_evaluation_scene_retirement_generations.capture_birth_source_projection",
        lambda *_, **__: birth)
    monkeypatch.setattr("blueprint_pipeline.capture_original_owner_observer.load_original_owner_observation",
        lambda **_: {"capture_owner": {"user_id": "owner1"}, "producer_delivery": producer})
    monkeypatch.setattr("blueprint_pipeline.consent_takedown.read_consent_state", lambda _: {"state": "granted"})
    monkeypatch.setattr("blueprint_pipeline.website_task_context.load_current_website_task_context", lambda **_: context)
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "needs_input", "scene_id": context["scene_id"],
        "capture_id": context["capture_id"], "job_disposition": "accepted", "simulator_ready": False,
        "visual_reconstruction_ready": True,
        "blockers": ["website_task_family_success_contract_required", "website_task_family_native_runtime_inputs_required"],
        "task_evaluation_inputs": {"task_family": "cloth_folding", "scope": {"scene_id": context["scene_id"],
            "task_id": "family-task", "task_context_digest": context["context_digest"]},
            "task_success_contract": None, "native_runtime_contract": None,
            "runtime_adapter": "task_evaluation_task_family_native_adapter", "automatic_paid_execution_authorized": False}}
    return root, selected, handoff


def retain(source):
    root, _, handoff = source
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(root / "pipeline/website_scene_preparation/handoff.json", handoff)


@pytest.mark.parametrize("schema", [status.STATUS_V1, status.STATUS_V2, status.STATUS_V3])
def test_completed_family_specific_missing_inputs_survive_existing_status_read(source, schema):
    retain(source)
    result = status.read_preparation_status(capture_root=source[0], selectors=source[1], schema_version=schema)
    assert result["state"] == "awaiting_inputs" and result["code"] == "preparation_inputs_pending"


@pytest.mark.parametrize("fault", ["context", "runtime", "paid", "output"])
def test_family_handoff_cannot_hide_changed_context_or_invent_authority(source, monkeypatch, fault):
    if fault == "context":
        source[2]["task_evaluation_inputs"]["scope"]["task_context_digest"] = "sha256:" + "f" * 64
    elif fault == "runtime":
        source[2]["runtime_inputs"] = {"status": "ready"}
    elif fault == "paid":
        source[2]["task_evaluation_inputs"]["automatic_paid_execution_authorized"] = True
    else:
        monkeypatch.setattr("blueprint_pipeline.handoff_job_state._output_commit", lambda *_, **__: None)
    retain(source)
    with pytest.raises(status.PreparationStatusUnavailable):
        status.read_preparation_status(capture_root=source[0], selectors=source[1])
