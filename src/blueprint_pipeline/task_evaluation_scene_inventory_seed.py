"""Pure retained scene inventory seed; no host completeness or cleanup authority.

ADP-009D/day-28: history preserves missing exact versions; existing pure joins
bind supplied owner/workspace identity. Recorded bytes are never measured disk.
"""
from __future__ import annotations

import re
from typing import Any

from . import task_evaluation_scene_preparation_lineage as retained
from . import task_evaluation_scene_source_attempt_lineage as source
from .decision_evidence_contracts import cross_runtime_canonical_digest

MAX_RECORD_BYTES = MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RECORDS = MAX_REFERENCES = MAX_ROWS = 10_000
MAX_NODES, MAX_DEPTH = 100_000, 64
_ROLES = {"events", "attempts", "source_snapshots", "factories", "source_submissions",
          "preparation_links", "preparation_envelopes", "preparation_results",
          "configuration_progressions", "activation_envelopes"}
_ROOTS = {"intent_root", "factory_output_root", "preparation_queue_root", "preparation_input_root",
          "configuration_progression_root", "activation_queue_root", "content_store_root"}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EVENT_FIELDS = {"schema_version", "intent_id", "intent_digest", "sequence", "previous_event_digest",
                 "observed_at_epoch", "status", "phase", "state", "blockers", "result_reference", "event_digest"}
_EVENT_STATUSES = {"accepted", "preparing", "awaiting_source", "awaiting_execution", "running", "completed", "needs_input", "blocked"}
_REF_ROLES = {"attempt": "attempts", "factory": "factories", "preparation_link": "preparation_links",
              "preparation_result": "preparation_results", "activation_link": "preparation_links"}
_DEFERRED = {"preparation_failure", "configuration_failure", "failure", "activation", "submission",
             "publication", "settlement", "lookahead"}


class SceneInventoryError(ValueError):
    """Fixed bounded refusal without supplied private text."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneInventoryError("scene_inventory_" + code)


def _shape(value: Any, pattern: re.Pattern, code: str) -> None:
    _require(retained._matches(value, pattern), code)


def _identity(record: tuple) -> tuple:
    p = record[1]
    return p["path"], p["sha256"], p["size_bytes"]


def _raw_reference(value: Any) -> tuple:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"}, "reference_invalid")
    path = retained._path(value["path"])
    _shape(value["sha256"], retained._DIGEST, "reference_invalid")
    _require(type(value["size_bytes"]) is int and value["size_bytes"] > 0, "reference_invalid")
    return path, value["sha256"], value["size_bytes"]


def _decode(records: dict) -> dict:
    count = 1 + int(records["projection"] is not None) + sum(len(records[role]) for role in _ROLES)
    _require(count <= MAX_RECORDS, "records_limit")
    groups = {"intent": [records["intent"]], "projection": [] if records["projection"] is None else [records["projection"]],
              **{role: records[role] for role in sorted(_ROLES)}}
    total = 0
    for group in groups.values():
        for pair in group:
            _require(isinstance(pair, (list, tuple)) and len(pair) == 2 and type(pair[1]) is bytes
                     and 0 < len(pair[1]) <= MAX_RECORD_BYTES, "record_invalid")
            _require(isinstance(pair[0], str) and len(pair[0]) <= retained.MAX_PATH_BYTES, "path_invalid")
            retained._path(pair[0])
            total += len(pair[1])
            _require(total <= MAX_TOTAL_BYTES, "bytes_limit")
    decoded, identities, paths = {}, set(), set()
    for role, group in groups.items():
        decoded[role] = []
        for pair in group:
            row = retained._record(pair, role, set())
            key = _identity(row)
            _require(key not in identities, "record_duplicate")
            _require(key[0] not in paths or role == "preparation_results", "immutable_ambiguous")
            identities.add(key)
            paths.add(key[0])
            row[1].update(seal_field=None, seal_digest=None)
            decoded[role].append(row)
    return decoded


def _remote(value: Any) -> None:
    _require(isinstance(value, dict) and set(value) == {"uri", "digest", "size_bytes"}
             and isinstance(value["uri"], str) and value["uri"].startswith(("https://", "s3://", "b2://", "gs://", "r2://"))
             and "?" not in value["uri"] and type(value["size_bytes"]) is int and value["size_bytes"] > 0, "remote_invalid")
    _shape(value["digest"], retained._DIGEST, "remote_invalid")


def _projection(event: dict) -> dict:
    value = {"schema_version": "task_evaluation_scene_progression.v1", "intent_id": event["intent_id"],
             "intent_digest": event["intent_digest"], "event_sequence": event["sequence"],
             "last_event_digest": event["event_digest"], "updated_at_epoch": event["observed_at_epoch"],
             "provider_allocation_performed": False,
             **{key: event[key] for key in ("status", "phase", "blockers", "result_reference", "state")}}
    value["progression_digest"] = cross_runtime_canonical_digest(value, digest_field="progression_digest")
    return value


def _history(decoded: dict, intent: dict, roots: dict, reasons: set) -> tuple:
    directory = retained._child(roots["intent_root"], intent["intent_id"])
    events = {}
    for value, provenance in decoded["events"]:
        sequence = value.get("sequence")
        _require(set(value) == _EVENT_FIELDS and value.get("schema_version") == "task_evaluation_scene_progression_event.v1"
                 and type(sequence) is int and 1 <= sequence <= MAX_RECORDS and sequence not in events, "event_invalid")
        _require(provenance["path"] == retained._child(directory, "progression-events", f"{sequence:06d}.json")
                 and value["intent_id"] == intent["intent_id"] and value["intent_digest"] == intent["intent_digest"], "event_identity_invalid")
        retained._seal(value, provenance, "event_digest", cross=True)
        _require(value["status"] in _EVENT_STATUSES and retained._matches(value["phase"], retained._ID)
                 and type(value["observed_at_epoch"]) in (int, float) and value["observed_at_epoch"] >= 0
                 and isinstance(value["state"], dict) and isinstance(value["blockers"], list)
                 and all(isinstance(code, str) for code in value["blockers"]), "event_invalid")
        if value["result_reference"] is not None:
            _remote(value["result_reference"])
        events[sequence] = value
    previous = None
    for number in range(1, len(events) + 1):
        _require(number in events and events[number]["previous_event_digest"] == (previous["event_digest"] if previous else None), "event_chain_invalid")
        previous = events[number]
    state = "missing" if events else "unavailable"
    if decoded["projection"]:
        value, provenance = decoded["projection"][0]
        _require(provenance["path"] == retained._child(directory, "progression.json"), "projection_invalid")
        sequence = value.get("event_sequence")
        _require(type(sequence) is int and sequence in events and value == _projection(events[sequence]), "projection_invalid")
        retained._seal(value, provenance, "progression_digest", cross=True)
        state = "current" if sequence == len(events) else "stale"
    else:
        reasons.add("projection_missing" if events else "history_unavailable")
    return {"supplied_event_count": len(events), "supplied_tail_sequence": len(events) if events else None,
            "supplied_tail_digest": previous["event_digest"] if previous else None, "projection_state": state,
            "chain_validated": bool(events), "host_history_complete": False}, events


def _obligations(events: dict, decoded: dict, reasons: set) -> tuple:
    local, remote, occurrences, deferred = {}, {}, 0, 0
    index = {_identity(row): (role, row) for role, rows in decoded.items() for row in rows}
    def add(role, ref, sequence):
        nonlocal occurrences
        occurrences += 1
        _require(occurrences <= MAX_REFERENCES, "references_limit")
        key = (role, *_raw_reference(ref))
        local.setdefault(key, set()).add(sequence)
    for sequence, value in events.items():
        state = value["state"]
        for role in _REF_ROLES:
            if role in state and state[role] is not None:
                add(role, state[role], sequence)
        for name in ("release_predecessors", "recovery_predecessors"):
            if name in state:
                _require(isinstance(state[name], list), "predecessor_invalid")
                for row in state[name]:
                    _require(isinstance(row, dict) and "attempt" in row, "predecessor_invalid")
                    add("attempt", row["attempt"], sequence)
                    if row.get("factory") is not None:
                        add("factory", row["factory"], sequence)
                    deferred += len(set(row) - {"attempt", "factory", "new_source_commit"})
        deferred += len(set(state) & _DEFERRED)
        ref = value["result_reference"]
        if ref is not None:
            occurrences += 1
            _require(occurrences <= MAX_REFERENCES, "references_limit")
            remote.setdefault((ref["uri"], ref["digest"], ref["size_bytes"]), set()).add(sequence)
    rows = []
    for key, sequences in sorted(local.items()):
        role, path, digest, size = key
        match = index.get((path, digest, size))
        reason = None if match and match[0] == _REF_ROLES[role] else (
            "historical_reference_role_unproven" if match else "historical_reference_bytes_unavailable")
        if reason:
            reasons.add(reason)
        rows.append({"role": role, "path": path, "sha256": digest, "size_bytes": size, "event_sequences": sorted(sequences),
                     "status": "matched_retained_bytes" if reason is None else "kept_deferred" if match else "kept_unresolved",
                     "reason": reason, "source_provenance": match[1][1] if reason is None else None})
    for (uri, digest, size), sequences in sorted(remote.items()):
        rows.append({"role": "remote_result_reference", "uri": uri, "digest": digest, "size_bytes": size,
                     "event_sequences": sorted(sequences), "status": "kept_deferred", "reason": "remote_result_scope_unproven"})
    return rows, deferred + len(remote)


def _member(path: str, kind: str, binding: dict, sources: list) -> dict:
    return {"path": path, "kind": kind, "binding": binding, "source_provenance": sources,
            "presence_checked": False, "exclusive_ownership_proven": False, "measured_bytes": None}


def _sources(intent_id: str, records: dict, roots: dict, reasons: set, members: list) -> list:
    child = source.join_scene_source_attempt_lineage(intent_id=intent_id, intent_record=records["intent"],
        attempt_records=records["attempts"], snapshot_records=records["source_snapshots"],
        factory_records=records["factories"], submission_records=records["source_submissions"],
        roots={key: roots[key] for key in ("intent_root", "factory_output_root")})
    rows = []
    for row in child["attempts"]:
        status = row["status"]
        disposition = "matched_retained_bytes" if row["workspace_membership_bound"] else (
            "kept_deferred" if status == "kept_out_of_scope" else "kept_unresolved")
        rows.append({"attempt_id": row["attempt_id"], "attempt_digest": row["attempt_digest"], "child_status": status,
                     "reasons": row["reasons"], "workspace_path": row["workspace_path"], "source_provenance": row["source_provenance"],
                     "seed_disposition": disposition})
        if disposition == "kept_unresolved":
            reasons.update(row["reasons"])
        if row["workspace_membership_bound"]:
            members.append(_member(row["workspace_path"], "administrative_source_workspace",
                                   {"intent_id": intent_id, "attempt_id": row["attempt_id"], "attempt_digest": row["attempt_digest"]},
                                   row["source_provenance"]))
    return rows


def join_retained_scene_inventory_seed(*, intent_id: str, records: Any, roots: Any) -> dict:
    """Join bounded supplied records without import-time or call-time host I/O."""
    try:
        return _join(intent_id, records, roots)
    except SceneInventoryError:
        raise
    except (retained.SceneLineageError, source.SceneSourceLineageError, ValueError, TypeError, KeyError,
            OverflowError, RecursionError, UnicodeError):
        raise SceneInventoryError("scene_inventory_input_invalid") from None


def _join(intent_id: str, records: Any, roots: Any) -> dict:
    _shape(intent_id, retained._ID, "parameters_invalid")
    _require(isinstance(records, dict) and set(records) == _ROLES | {"intent", "projection"}
             and all(isinstance(records[role], (list, tuple)) for role in _ROLES)
             and isinstance(roots, dict) and set(roots) == _ROOTS, "parameters_invalid")
    for value in roots.values():
        _require(isinstance(value, str) and len(value) <= retained.MAX_PATH_BYTES, "path_invalid")
    roots = {key: retained._path(value) for key, value in roots.items()}
    decoded = _decode(records)
    intent, provenance = decoded["intent"][0]
    retained._intent(intent, provenance, intent_id, roots["intent_root"])
    reasons, members = set(), []
    history, events = _history(decoded, intent, roots, reasons)
    source_rows = _sources(intent_id, records, roots, reasons, members)
    obligations, deferred = _obligations(events, decoded, reasons)
    _require(not any(records[role] for role in ("preparation_links", "preparation_envelopes", "preparation_results",
                                               "configuration_progressions", "activation_envelopes")), "downstream_not_implemented")
    result = {"schema_version": "task_evaluation_scene_inventory_seed.v1", "status": "kept_unresolved" if reasons else "joined_supplied_seed",
              "scope": "supplied_retained_history_and_preparation_records", "intent_id": intent_id,
              "intent_digest": intent["intent_digest"], "intent_provenance": provenance, "history": history,
              "obligations": obligations, "members": sorted(members, key=lambda row: (row["kind"], row["path"])),
              "source_attempt_obligations": source_rows, "shared_cache_references": [], "deferred_result_references": [],
              "unresolved_reasons": sorted(reasons), "deferred_branch_count": deferred + sum(row["seed_disposition"] == "kept_deferred" for row in source_rows),
              "mutations": 0, "execution_authorized": False, "complete_scene_inventory": False,
              "historical_attempt_inventory_complete": False, "remaining_branches_complete": False,
              "payload_presence_checked": False, "payload_members_verified": False, "byte_accounting_complete": False,
              "finished_state_checked": False, "references_checked": False, "consumer_fence_checked": False,
              "publication_readback_checked": False, "remote_availability_checked": False,
              "cleanup_authorized": False, "requires_fresh_reference_check": True}
    _require(sum(len(result[key]) for key in ("obligations", "members", "source_attempt_obligations",
                                             "shared_cache_references", "deferred_result_references")) <= MAX_ROWS, "rows_limit")
    _require(len(retained._encoded(result)) <= MAX_OUTPUT_BYTES, "output_limit")
    return result
