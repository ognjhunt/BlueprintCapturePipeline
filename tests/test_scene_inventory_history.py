# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_inventory_seed.py
"""ADP-009D/day-28: retained history is descriptive, never cleanup authority."""
from __future__ import annotations

import copy
import importlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_scene_source_attempt_lineage import fixture as source_fixture, pair, ref

ROLES = ("events", "attempts", "source_snapshots", "factories", "source_submissions",
         "preparation_links", "preparation_envelopes", "preparation_results",
         "configuration_progressions", "activation_envelopes")


def api():
    return importlib.import_module("blueprint_pipeline.task_evaluation_scene_inventory_seed")


def seal(value, field, cross=False):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    value = copy.deepcopy(value)
    digest = cross_runtime_canonical_digest if cross else canonical_digest
    value[field] = digest(value, digest_field=field)
    return value


def fixture(*, source=False):
    child = source_fixture()
    roots = dict(child["roots"], preparation_queue_root="/retained/queue",
                 preparation_input_root="/retained/inputs", configuration_progression_root="/retained/progression",
                 activation_queue_root="/retained/activations", content_store_root="/retained/content")
    records = {role: [] for role in ROLES}
    records.update(intent=child["intent_record"], projection=None)
    if source:
        for old, new in (("attempt_records", "attempts"), ("snapshot_records", "source_snapshots"),
                         ("factory_records", "factories"), ("submission_records", "source_submissions")):
            records[new] = child[old]
    return {"intent_id": child["intent_id"], "roots": roots, "records": records}


def event(args, state=None, *, remote=None, updates=None):
    intent = json.loads(args["records"]["intent"][1])
    events = args["records"]["events"]
    previous = json.loads(events[-1][1]) if events else None
    value = {"schema_version": "task_evaluation_scene_progression_event.v1",
             "intent_id": args["intent_id"], "intent_digest": intent["intent_digest"],
             "sequence": len(events) + 1, "previous_event_digest": previous["event_digest"] if previous else None,
             "observed_at_epoch": 12.0, "status": "preparing", "phase": "source_preparation",
             "state": state or {}, "blockers": [], "result_reference": remote}
    value.update(updates or {})
    value = seal(value, "event_digest", cross=True)
    base = args["roots"]["intent_root"] + "/" + args["intent_id"]
    events.append(pair(base + f"/progression-events/{value['sequence']:06d}.json", value))
    return value


def project(args, value):
    projection = {"schema_version": "task_evaluation_scene_progression.v1",
                  "intent_id": value["intent_id"], "intent_digest": value["intent_digest"],
                  "event_sequence": value["sequence"], "last_event_digest": value["event_digest"],
                  **{key: value[key] for key in ("status", "phase", "blockers", "result_reference", "state")},
                  "updated_at_epoch": value["observed_at_epoch"], "provider_allocation_performed": False}
    args["records"]["projection"] = pair(args["roots"]["intent_root"] + "/" + args["intent_id"] + "/progression.json",
                                            seal(projection, "progression_digest", cross=True))


def refuses(args, suffix=None):
    with pytest.raises(api().SceneInventoryError) as exc:
        api().join_retained_scene_inventory_seed(**args)
    assert str(exc.value).startswith("scene_inventory_") and len(str(exc.value)) < 100
    if suffix:
        assert str(exc.value) == "scene_inventory_" + suffix


def test_empty_history_is_unknown_and_never_an_authorized_empty_inventory():
    result = api().join_retained_scene_inventory_seed(**fixture())
    assert result["history"]["projection_state"] == "unavailable"
    assert not result["history"]["chain_validated"] and not result["history"]["host_history_complete"]
    assert result["status"] == "kept_unresolved" and "history_unavailable" in result["unresolved_reasons"]
    assert result["members"] == [] and result["mutations"] == 0
    for flag in ("execution_authorized", "complete_scene_inventory", "historical_attempt_inventory_complete",
                 "remaining_branches_complete", "payload_presence_checked", "payload_members_verified",
                 "byte_accounting_complete", "finished_state_checked", "references_checked", "consumer_fence_checked",
                 "publication_readback_checked", "remote_availability_checked", "cleanup_authorized"):
        assert result[flag] is False


@pytest.mark.parametrize("projection", ["current", "stale", "missing"])
def test_contiguous_history_accepts_stale_or_missing_projection_without_repair(projection):
    args = fixture()
    first = event(args)
    last = event(args, updates={"observed_at_epoch": 11})  # no producer monotonic clock promise
    if projection != "missing":
        project(args, first if projection == "stale" else last)
    result = api().join_retained_scene_inventory_seed(**args)
    assert result["history"] == {"supplied_event_count": 2, "supplied_tail_sequence": 2,
                                 "supplied_tail_digest": last["event_digest"], "projection_state": projection,
                                 "chain_validated": True, "host_history_complete": False}


@pytest.mark.parametrize("updates", [{"sequence": True}, {"sequence": 2}, {"previous_event_digest": "sha256:" + "a" * 64},
                                     {"intent_id": "foreign"}, {"intent_digest": "sha256:" + "a" * 64},
                                     {"observed_at_epoch": True}, {"observed_at_epoch": -1}, {"phase": "x/y"},
                                     {"status": "unknown"}, {"blockers": [2]}, {"state": []}, {"unexpected": 1}])
def test_resealed_bad_event_contract_refuses(updates):
    args = fixture()
    event(args, updates=updates)
    refuses(args)


def test_gapped_duplicate_fork_and_wrong_projection_refuse():
    for mode in ("gap", "duplicate", "fork", "projection"):
        args = fixture()
        first = event(args)
        second = event(args)
        if mode == "gap":
            args["records"]["events"].pop(0)
        elif mode == "duplicate":
            args["records"]["events"].append(args["records"]["events"][0])
        elif mode == "fork":
            second["previous_event_digest"] = "sha256:" + "a" * 64
            args["records"]["events"][1] = pair(args["records"]["events"][1][0], seal(second, "event_digest", cross=True))
        else:
            project(args, first)
            path, raw = args["records"]["projection"]
            changed = json.loads(raw)
            changed["provider_allocation_performed"] = True
            args["records"]["projection"] = pair(path, seal(changed, "progression_digest", cross=True))
        refuses(args)


def test_projection_without_events_refuses():
    args = fixture()
    project(args, event(args))
    args["records"]["events"] = []
    refuses(args)


def test_every_historical_version_remains_an_exact_obligation_not_latest_wins():
    args = fixture(source=True)
    current = args["records"]["attempts"][0]
    old = ref(current)
    old["sha256"] = "sha256:" + "a" * 64
    old["size_bytes"] += 1
    event(args, {"attempt": old, "release_predecessors": [{"attempt": ref(current), "factory": None}],
                 "recovery_predecessors": [{"attempt": old}]})
    event(args, {"attempt": ref(current)})
    result = api().join_retained_scene_inventory_seed(**args)
    rows = result["obligations"]
    assert len(rows) == 2
    matched = next(row for row in rows if row["sha256"] == ref(current)["sha256"])
    missing = next(row for row in rows if row["sha256"] == old["sha256"])
    assert matched["event_sequences"] == [1, 2] and matched["status"] == "matched_retained_bytes"
    assert missing["reason"] == "historical_reference_bytes_unavailable" and missing["status"] == "kept_unresolved"
    assert len(result["members"]) == 1  # current source only, absent old bytes never fabricated


@pytest.mark.parametrize("value", [{}, {"path": "/wrong"}, 2, False])
def test_nonnull_malformed_predecessor_factory_refuses(value):
    args = fixture(source=True)
    event(args, {"release_predecessors": [{"attempt": ref(args["records"]["attempts"][0]), "factory": value}]})
    refuses(args)


def test_remote_result_stays_remote_even_if_its_digest_matches_local_record():
    args = fixture(source=True)
    remote = {"uri": "https://example.test/result.json", "digest": ref(args["records"]["attempts"][0])["sha256"], "size_bytes": 1}
    event(args, remote=remote)
    result = api().join_retained_scene_inventory_seed(**args)
    row = result["obligations"][0]
    assert row["role"] == "remote_result_reference" and "path" not in row
    assert row["status"] == "kept_deferred" and row["uri"] == remote["uri"]


@pytest.mark.parametrize("updates", [{"uri": "/local"}, {"uri": "https://example.test/r?q=x"},
                                     {"size_bytes": 0}, {"size_bytes": True}, {"digest": "sha256:no"}, {"path": "/local"}])
def test_remote_reference_shape_refuses(updates):
    args = fixture()
    event(args, remote={"uri": "s3://bucket/key", "digest": "sha256:" + "b" * 64, "size_bytes": 1, **updates})
    refuses(args)


def test_source_missing_proof_and_paid_rows_survive_without_history_references():
    for paid in (False, True):
        args = fixture(source=True)
        if paid:
            path, raw = args["records"]["attempts"][0]
            value = json.loads(raw)
            value["schema_version"] = "task_evaluation_scene_attempt.v1"
            path = path.replace("/preparation-attempts/", "/attempts/")
            args["records"]["attempts"] = [pair(path, seal(value, "attempt_digest", cross=True))]
            for role in ("source_snapshots", "factories", "source_submissions"):
                args["records"][role] = []
        else:
            args["records"]["source_snapshots"].pop()
        result = api().join_retained_scene_inventory_seed(**args)
        row = result["source_attempt_obligations"][0]
        assert row["child_status"] == ("kept_out_of_scope" if paid else "kept_unresolved")
        assert row["seed_disposition"] == ("kept_deferred" if paid else "kept_unresolved")
        assert result["members"] == [] and row["reasons"]


def test_bounds_precede_parser_or_hash(monkeypatch):
    args = fixture()
    module = api()
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 1)
    monkeypatch.setattr(module.retained, "_record", lambda *args: pytest.fail("parsed before total preflight"))
    refuses(args, "bytes_limit")


def test_history_reference_occurrences_budgeted_before_dedup(monkeypatch):
    args = fixture(source=True)
    reference = ref(args["records"]["attempts"][0])
    event(args, {"attempt": reference, "release_predecessors": [{"attempt": reference}]})
    monkeypatch.setattr(api(), "MAX_REFERENCES", 1)
    refuses(args, "references_limit")


def test_reordering_input_preserves_deterministic_output():
    args = fixture(source=True)
    event(args, {"factory": ref(args["records"]["factories"][0])})
    event(args)
    expected = api().join_retained_scene_inventory_seed(**args)
    for role in ROLES:
        args["records"][role].reverse()
    assert api().join_retained_scene_inventory_seed(**args) == expected
