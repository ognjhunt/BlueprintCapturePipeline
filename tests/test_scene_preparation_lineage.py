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


def rebind(args, *, task_id=None):
    """Keep all downstream seals/paths consistent to isolate an intent rule."""
    intent = json.loads(args["intent_record"][1])
    envelope = json.loads(args["preparation_envelopes"][0][1])
    request = envelope["request"]
    request["scene_intent_digest"] = intent["intent_digest"]
    if task_id:
        request["task"]["identity"]["id"] = task_id
    digest = canonical_digest(request)
    envelope["request_digest"] = digest
    link = json.loads(args["preparation_links"][0][1])
    link.update(intent_digest=intent["intent_digest"], request_digest=digest,
                result_filename=link["preparation_id"] + "-" + digest[7:] + ".json")
    if task_id:
        link["task_id"] = task_id
    args["preparation_links"][0] = record(BASE + "/preparations/" + digest[7:] + ".json",
                                           seal(link, "link_digest"))
    args["preparation_envelopes"][0] = record(ROOTS["preparation_queue_root"] + "/completed/" + link["result_filename"],
                                               seal(envelope, "envelope_digest"))


def activation(args, *, base=True):
    link = json.loads(args["preparation_links"][0][1])
    attempt_id = "scene-configuration-" + link["request_digest"][7:31]
    path = BASE + "/attempts/" + attempt_id + ".json"
    attempt = seal({
        "schema_version": "task_evaluation_scene_attempt.v1", "attempt_id": attempt_id,
        "intent_id": INTENT, "intent_digest": link["intent_digest"],
        "source_commit": COMMIT, "input_digest": link["request_digest"],
        "runtime_digest": DIGEST, "provider": "vast", "maximum_spend_usd": 1.0,
    }, "attempt_digest", cross=True)
    args["configuration_attempt_records"] = [record(path, attempt)]
    bind_attempt(args, link=link)
    if not base:
        args["preparation_links"] = args["preparation_links"][1:]


def bind_attempt(args, *, link=None):
    if link is None:
        link = json.loads(args["preparation_links"][-1][1])
    path, raw = args["configuration_attempt_records"][0]
    link["scene_configuration_attempt"] = {
        "path": path, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
    }
    bound = record(BASE + "/preparations/" + link["request_digest"][7:] + ".activation.json",
                   seal(link, "link_digest"))
    if args["preparation_links"][-1][0].endswith(".activation.json"):
        args["preparation_links"][-1] = bound
    else:
        args["preparation_links"].append(bound)


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
    rebind(args)
    refuses(args, "scene_lineage_intent_request_invalid")


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


def test_consistent_request_and_link_cannot_rebind_the_selected_task():
    args = fixture()
    rebind(args, task_id="different-task")
    refuses(args, "scene_lineage_request_digest_invalid")


@pytest.mark.parametrize("base", [False, True])
def test_activation_joins_exact_attempt_once_retaining_distinct_variant_seals(base):
    args = fixture()
    activation(args, base=base)
    report = lineage.join_scene_preparation_lineage(**args)
    assert report["preparation_count"] == 1
    row = report["preparations"][0]
    assert row["configuration_attempt"]["provider"] == "vast"
    assert row["configuration_attempt"]["input_digest"] == row["request_digest"]
    sources = row["source_provenance"]
    link_sources = [p for p in sources if p["role"] == "link"]
    assert len(link_sources) == (2 if base else 1)
    if base:
        assert len({p["seal_digest"] for p in link_sources}) == 2
    assert any(p["role"] == "attempt" for p in sources)
    assert "maximum_spend_usd" not in row["configuration_attempt"]


@pytest.mark.parametrize("field,value", [
    ("input_digest", DIGEST), ("runtime_digest", "sha256:" + "c" * 64),
    ("provider", "other"), ("attempt_id", "scene-configuration-wrong"),
    ("intent_id", "other"), ("intent_digest", DIGEST),
    ("source_commit", "c" * 40), ("schema_version", "unknown"),
])
def test_fully_resealed_byte_bound_attempt_must_match_complete_identity(field, value):
    args = fixture()
    activation(args)
    change(args, "configuration_attempt_records", lambda row: row.update({field: value}),
           "attempt_digest", cross=True)
    bind_attempt(args)
    refuses(args, "scene_lineage_attempt_identity_invalid")


@pytest.mark.parametrize("failure", ["missing", "wrong_sha", "wrong_size", "wrong_path", "broken_seal"])
def test_activation_requires_raw_byte_proof_and_attempt_seal(failure):
    args = fixture()
    activation(args)
    if failure == "missing":
        args["configuration_attempt_records"] = []
    elif failure == "broken_seal":
        change(args, "configuration_attempt_records", lambda row: row.update(attempt_digest=DIGEST))
        bind_attempt(args)
    else:
        key, value = {"wrong_sha": ("sha256", DIGEST), "wrong_size": ("size_bytes", 1),
                      "wrong_path": ("path", "/foreign/attempt.json")}[failure]
        path, raw = args["preparation_links"][-1]
        link = json.loads(raw)
        link["scene_configuration_attempt"][key] = value
        args["preparation_links"][-1] = record(path, seal(link, "link_digest"))
    refuses(args)


@pytest.mark.parametrize("value", [None, "malformed", {"path": "/foreign", "sha256": DIGEST, "size_bytes": 1}])
def test_base_refuses_any_attempt_field_before_proof_validation(value):
    args = fixture()
    change(args, "preparation_links", lambda row: row.update(scene_configuration_attempt=value), "link_digest")
    refuses(args, "scene_lineage_link_role_invalid")


def test_activation_missing_attempt_field_is_a_role_error():
    args = fixture()
    path, raw = args["preparation_links"][0]
    args["preparation_links"][0] = (path[:-5] + ".activation.json", raw)
    refuses(args, "scene_lineage_link_role_invalid")


def test_all_historical_preparations_are_kept_and_reordering_is_deterministic():
    args, older = fixture("prep-z"), fixture("prep-a")
    args["preparation_links"] += older["preparation_links"]
    args["preparation_envelopes"] += older["preparation_envelopes"]
    report = lineage.join_scene_preparation_lineage(**args)
    assert [r["preparation_id"] for r in report["preparations"]] == ["prep-a", "prep-z"]
    args["preparation_links"].reverse()
    args["preparation_envelopes"].reverse()
    assert lineage.join_scene_preparation_lineage(**args) == report


def test_duplicate_state_copies_and_orphan_envelopes_refuse():
    args = fixture()
    path, raw = args["preparation_envelopes"][0]
    args["preparation_envelopes"].append((path.replace("/completed/", "/processing/"), raw))
    refuses(args, "scene_lineage_envelope_ambiguous")
    args = fixture()
    args["preparation_envelopes"] += fixture("foreign-prep")["preparation_envelopes"]
    refuses(args, "scene_lineage_envelope_unmatched")


def test_same_preparation_id_cannot_name_two_historical_requests():
    args, other = fixture(), fixture()
    rebind(other, task_id="different-task")
    args["preparation_links"] += other["preparation_links"]
    args["preparation_envelopes"] += other["preparation_envelopes"]
    refuses(args, "scene_lineage_link_ambiguous")
