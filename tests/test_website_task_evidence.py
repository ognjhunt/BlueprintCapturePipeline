"""Synthetic owner evidence reaches preparation without becoming measured truth."""
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_evidence import (
    ingest_website_task_evidence,
    owner_success_criteria,
)


def test_consumes_exact_webapp_projection_vector(tmp_path):
    vector = json.loads((Path(__file__).parent / "fixtures/website-context-vector.json").read_text())
    context = vector["context"]
    def download(image, target):
        Path(target).write_bytes(vector["photo_utf8"].encode())
    receipt = ingest_website_task_evidence(task_context=context, output_root=tmp_path, download=download)
    assert receipt["task_context_digest"] == context["context_digest"]
    assert receipt["operator_task_details"] == vector["brief"]["operatorTaskDetails"]
    assert receipt["success_criteria"]["targets"] == vector["brief"]["successCriteria"]
    assert Path(receipt["items"][0]["images"][0]["path"]).read_bytes() == b"synthetic-photo"


def task():
    data = b"synthetic-photo"
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
             "capture_id": "walkthrough-req1", "confirmed": True, "confirmed_at": "2026-10-06T00:00:00Z",
             "description": "Move the carton", "capture_rights": {"derived_scene_generation_allowed": True},
             "operator_task_details": {"item_weight": "approximately 5 lb, not weighed", "item_make_model": "ACME X2"},
             "success_criteria": {"successDefinition": "Arrives intact", "successRate": 95, "cycleTimeSeconds": None, "unknown": False},
             "task_items": [{"item_id": "item_box", "label": "Carton", "basis": "operator_added", "images": [{
                 "image_id": "img1", "storage_path": "scenes/site-req1/items/item_box/img1.jpg",
                 "source": {"generation": "123", "size_bytes": len(data), "crc32c": "AAAAAA==",
                            "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}}]}]}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value, data


def test_separately_supplied_photo_is_retained_once_and_cannot_invent_geometry(tmp_path):
    context, data = task()
    calls = []
    def download(image, target):
        calls.append(image)
        Path(target).write_bytes(data)
    receipt = ingest_website_task_evidence(task_context=context, output_root=tmp_path, download=download)
    assert receipt["items"][0]["status"] == "unsupported"
    image = receipt["items"][0]["images"][0]
    assert Path(image["path"]).read_bytes() == data
    assert image["sha256"] == context["task_items"][0]["images"][0]["source"]["sha256"]
    assert receipt["physical_measurements_verified"] is False
    assert receipt["operator_task_details"] == context["operator_task_details"]
    assert ingest_website_task_evidence(task_context=context, output_root=tmp_path, download=download) == receipt
    assert len(calls) == 1


def test_owner_targets_preserve_units_and_explicitly_refuse_unsupported_translation():
    context, _ = task()
    result = owner_success_criteria(context)
    assert result["targets"] == context["success_criteria"]
    assert result["success_rate_unit"] == "percent" and result["cycle_time_unit"] == "seconds"
    assert result["status"] == "unsupported"
    assert owner_success_criteria({})["status"] == "not_supplied"


@pytest.mark.parametrize("fault", ["revoked", "foreign", "changed", "unknown_generation"])
def test_invalid_or_unknown_sources_never_become_authoring_inputs(tmp_path, fault):
    context, data = task()
    if fault == "revoked":
        context["capture_rights"]["derived_scene_generation_allowed"] = False
    elif fault == "foreign":
        context["task_items"][0]["images"][0]["storage_path"] = "scenes/site-foreign/items/box/x.jpg"
    elif fault == "unknown_generation":
        context["task_items"][0]["images"][0]["source"] = None
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    calls = []
    def download(image, target):
        calls.append(image)
        Path(target).write_bytes(b"changed" if fault == "changed" else data)
    if fault in {"revoked", "foreign", "changed"}:
        with pytest.raises(ValueError):
            ingest_website_task_evidence(task_context=context, output_root=tmp_path, download=download)
    else:
        receipt = ingest_website_task_evidence(task_context=context, output_root=tmp_path, download=download)
        assert receipt["items"][0]["blocker"] == "website_item_original_generation_unknown"
        assert calls == []
