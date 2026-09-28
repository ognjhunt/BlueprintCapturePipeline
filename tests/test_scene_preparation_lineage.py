# Covers: src/blueprint_pipeline/task_evaluation_scene_preparation_lineage.py
"""ADP-009D/day-28: retained lineage proves identity, never retirement authority."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from blueprint_pipeline import task_evaluation_scene_preparation_lineage as lineage
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)

ROOTS = {
    "intent_root": "/retained/intents",
    "preparation_queue_root": "/retained/preparations",
    "preparation_input_root": "/retained/inputs",
}
INTENT = "scene-owner-1"
BASE = ROOTS["intent_root"] + "/" + INTENT
COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64


def seal(value, field, *, cross=False):
    value = copy.deepcopy(value)
    digest = cross_runtime_canonical_digest if cross else canonical_digest
    value[field] = digest(value, digest_field=field)
    return value


def record(path, value):
    return path, json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(preparation="prep-1"):
    task = {"task_id": "task-1", "historical_number": 1.0}
    intent = seal({
        "schema_version": "task_evaluation_scene_intent.v1", "intent_id": INTENT,
        "request": {"schema_version": "task_evaluation_scene_intake_request.v1",
                    "submission_id": "submission-1",
                    "owner": {"user_id": "owner@site", "organization_id": "org:1"},
                    "task": task, "source": {"content_digest": DIGEST}},
        "source_content_digest": DIGEST,
        "task_content_digest": cross_runtime_canonical_digest(task),
    }, "intent_digest", cross=True)
    request = {
        "schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "scene_configuration", "preparation_id": preparation,
        "expected_production_commit": COMMIT, "team_namespace": "team-1",
        "scene": {"identity": {"id": "scene-1"}},
        "task": {"identity": {"id": "task-1"}},
        "scene_intent_digest": intent["intent_digest"],
        "execution_adapter": {"runtime_source_bundle": {"digest": DIGEST}},
    }
    digest = canonical_digest(request)
    filename = preparation + "-" + digest[7:] + ".json"
    link = seal({
        "schema_version": "task_evaluation_scene_preparation_link.v1",
        "intent_id": INTENT, "intent_digest": intent["intent_digest"],
        "preparation_id": preparation, "request_digest": digest,
        "expected_production_commit": COMMIT, "team_namespace": "team-1",
        "scene_id": "scene-1", "task_id": "task-1", "result_filename": filename,
    }, "link_digest")
    envelope = seal({
        "schema_version": "task_evaluation_launch_preparation_envelope.v1",
        "request": request, "request_digest": digest,
    }, "envelope_digest")
    return {
        "intent_id": INTENT, "intent_record": record(BASE + "/intent.json", intent),
        "preparation_links": [record(BASE + "/preparations/" + digest[7:] + ".json", link)],
        "preparation_envelopes": [record(ROOTS["preparation_queue_root"] + "/completed/" + filename, envelope)],
        "configuration_attempt_records": [], "roots": dict(ROOTS),
    }


def change(args, field, edit, seal_field=None, *, cross=False):
    position = None if field == "intent_record" else 0
    path, raw = args[field] if position is None else args[field][position]
    value = json.loads(raw)
    edit(value)
    if seal_field:
        value = seal(value, seal_field, cross=cross)
    updated = record(path, value)
    if position is None:
        args[field] = updated
    else:
        args[field][position] = updated


def refuses(args, code=None):
    with pytest.raises(lineage.SceneLineageError) as error:
        lineage.join_scene_preparation_lineage(**args)
    assert str(error.value).startswith("scene_lineage_")
    assert len(str(error.value)) < 100
    if code:
        assert str(error.value) == code


def test_base_join_retains_raw_provenance_and_no_authority():
    args = fixture()
    report = lineage.join_scene_preparation_lineage(**args)
    assert report["schema_version"] == "task_evaluation_scene_preparation_lineage.v1"
    assert report["status"] == "joined"
    assert report["scope"] == "supplied_retained_preparation_records"
    assert report["preparation_count"] == 1
    row = report["preparations"][0]
    assert row["workspace_path"] == "/retained/inputs/prep-1"
    assert row["configuration_attempt"] is None
    for source, retained in [(report["intent_provenance"], args["intent_record"]),
                             (row["source_provenance"][0], args["preparation_envelopes"][0])]:
        assert source["path"] == retained[0]
        assert source["sha256"] == "sha256:" + hashlib.sha256(retained[1]).hexdigest()
        assert source["size_bytes"] == len(retained[1])
    assert report["mutations"] == 0
    for field in ("execution_authorized", "complete_scene_inventory", "references_checked",
                  "finished_state_checked", "payload_presence_checked"):
        assert report[field] is False
    assert report["requires_fresh_reference_check"] is True


@pytest.mark.parametrize("field,seal_field", [
    ("intent_record", "intent_digest"), ("preparation_links", "link_digest"),
    ("preparation_envelopes", "envelope_digest"),
])
def test_broken_seal_refuses(field, seal_field):
    args = fixture()
    change(args, field, lambda value: value.update({seal_field: DIGEST}))
    refuses(args)


@pytest.mark.parametrize("key,value", [
    ("intent_id", "other"), ("intent_digest", DIGEST),
    ("expected_production_commit", "c" * 40), ("team_namespace", "other"),
    ("scene_id", "other"), ("task_id", "other"), ("result_filename", "other.json"),
    ("request_digest", DIGEST),
])
def test_resealed_link_does_not_match_owner_request_tuple(key, value):
    args = fixture()
    change(args, "preparation_links", lambda row: row.update({key: value}), "link_digest")
    refuses(args)


@pytest.mark.parametrize("path", ["relative.json", BASE + "/preparations/../intent.json",
                                     BASE + "//intent.json", "/foreign/intent.json"])
def test_unsafe_or_wrong_intent_path_refuses(path):
    args = fixture()
    args["intent_record"] = (path, args["intent_record"][1])
    refuses(args)


@pytest.mark.parametrize("key,value", [
    ("schema_version", "unknown"), ("submission_id", "../bad"),
    ("owner", {"user_id": "bad/owner", "organization_id": "org"}),
    ("task", {"task_id": "bad/task"}), ("source", {"content_digest": "bad"}),
])
def test_resealed_intent_request_minimal_structure_refuses(key, value):
    args = fixture()
    change(args, "intent_record", lambda row: row["request"].update({key: value}),
           "intent_digest", cross=True)
    refuses(args)


def test_duplicate_json_keys_refuse():
    args = fixture()
    path, raw = args["intent_record"]
    args["intent_record"] = (path, b'{"intent_id":"duplicate",' + raw[1:])
    refuses(args)


def test_missing_envelope_refuses_and_empty_supplied_scope_is_not_complete():
    args = fixture()
    args["preparation_envelopes"] = []
    refuses(args)
    args["preparation_links"] = []
    report = lineage.join_scene_preparation_lineage(**args)
    assert report["preparation_count"] == 0
    assert report["complete_scene_inventory"] is False


@pytest.mark.parametrize("limit,value", [("MAX_RECORD_BYTES", 8), ("MAX_TOTAL_BYTES", 8),
                                        ("MAX_RECORDS", 2), ("MAX_OUTPUT_BYTES", 8),
                                        ("MAX_PATH_BYTES", 8), ("MAX_PATH_COMPONENTS", 2)])
def test_injected_resource_caps_refuse_without_truncating(monkeypatch, limit, value):
    monkeypatch.setattr(lineage, limit, value)
    refuses(fixture())
