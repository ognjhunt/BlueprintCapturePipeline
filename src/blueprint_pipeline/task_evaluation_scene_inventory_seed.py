"""Pure retained scene inventory seed; no host completeness or cleanup authority.

ADP-009D/day-28: history preserves missing exact versions; existing pure joins
bind supplied owner/workspace identity. Recorded bytes are never measured disk.
"""
from __future__ import annotations

import re
import hashlib
from pathlib import PurePosixPath
from typing import Any

from . import task_evaluation_scene_preparation_lineage as retained
from . import task_evaluation_scene_source_attempt_lineage as source
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

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
_DEFERRED = {"preparation_failure", "configuration_failure", "failure", "activation", "launch", "submission",
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


def _supported_path(role: str, path: str, roots: dict, intent_id: str) -> bool:
    root = roots["factory_output_root"] if role == "factory" else roots["preparation_queue_root"] if role == "preparation_result" else roots["intent_root"]
    try:
        parts = PurePosixPath(path).relative_to(PurePosixPath(root)).parts
    except ValueError:
        return False
    if role == "preparation_result":
        return len(parts) == 2 and parts[0] == "results" and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}-[0-9a-f]{64}\.json", parts[1]) is not None
    if not parts or parts[0] != intent_id:
        return False
    if role == "factory":
        return len(parts) in (3, 4) and retained._matches(parts[1], retained._ID) and parts[2:] in (("factory.json",), ("materialized", "factory_receipt.json"))
    if role == "attempt":
        return len(parts) == 3 and parts[1] in {"attempts", "preparation-attempts"} and parts[2].endswith(".json") and retained._matches(parts[2][:-5], retained._ID)
    return len(parts) == 3 and parts[1] == "preparations" and re.fullmatch(r"[0-9a-f]{64}(?:\.activation)?\.json", parts[2]) is not None


def _obligations(events: dict, decoded: dict, reasons: set, roots: dict, intent_id: str) -> tuple:
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
        supported = _supported_path(role, path, roots, intent_id)
        reason = "historical_reference_role_unproven" if not supported or (match and match[0] != _REF_ROLES[role]) else (
            None if match else "historical_reference_bytes_unavailable")
        if reason:
            reasons.add(reason)
        rows.append({"role": role, "path": path, "sha256": digest, "size_bytes": size, "event_sequences": sorted(sequences),
                     "status": "matched_retained_bytes" if reason is None else "kept_deferred" if reason == "historical_reference_role_unproven" else "kept_unresolved",
                     "reason": reason, "source_provenance": match[1][1] if reason is None else None})
    for (uri, digest, size), sequences in sorted(remote.items()):
        rows.append({"role": "remote_result_reference", "uri": uri, "digest": digest, "size_bytes": size,
                     "event_sequences": sorted(sequences), "status": "kept_deferred", "reason": "remote_result_scope_unproven"})
    return rows, deferred + len(remote), occurrences


def _member(path: str, kind: str, binding: dict, sources: list) -> dict:
    return {"path": path, "kind": kind, "binding": binding, "source_provenance": sorted(sources, key=lambda p: (p["role"], p["path"], p["sha256"])),
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


def _missing(group: dict, role: str, path: str | None, root: str, states: set, reason: str, missing: dict) -> None:
    missing["count"] += 1
    _require(missing["count"] <= MAX_REFERENCES, "references_limit")
    missing["joins"].append({"preparation_id": group["link"]["preparation_id"], "request_digest": group["link"]["request_digest"],
        "role": role, "expected_path": path, "expected_root": root, "expected_states": sorted(states), "reason": reason,
        "status": "kept_unresolved", "source_provenance": sorted(group["sources"], key=lambda p: (p["role"], p["path"], p["sha256"]))})


def _preparations(decoded: dict, records: dict, intent: dict, roots: dict, reasons: set, members: list, missing: dict) -> dict:
    directory = retained._child(roots["intent_root"], intent["intent_id"])
    grouped, envelopes, attempts = {}, {}, {_identity(row): row for row in decoded["attempts"]}
    for value, provenance in decoded["preparation_links"]:
        link = retained._link(value, provenance, intent, directory)
        key = link["preparation_id"]
        if key in grouped:
            old, sources = grouped[key]
            _require(all(old[field] == link[field] for field in retained._IDENTITY)
                     and all(p["variant"] != provenance["variant"] for p in sources), "link_ambiguous")
            sources.append(provenance)
            if provenance["variant"] == "activation":
                grouped[key] = link, sources
        else:
            grouped[key] = link, [provenance]
    for row in decoded["preparation_envelopes"]:
        envelopes.setdefault(PurePosixPath(row[1]["path"]).name, []).append(row)
    complete_links, complete_envelopes, complete_attempts, context = [], [], [], {}
    raw_index = {pair[0]: pair for role in ("preparation_links", "preparation_envelopes", "attempts") for pair in records[role]}
    for key, (link, sources) in sorted(grouped.items()):
        matches = envelopes.pop(link["result_filename"], [])
        _require(len(matches) <= 1, "envelope_ambiguous")
        request = retained._envelope(*matches[0], roots["preparation_queue_root"], link, intent) if matches else None
        sources = list(sources) + ([matches[0][1]] if matches else [])
        absent = set() if matches else {"preparation_envelope_missing"}
        attempt = None
        if "scene_configuration_attempt" in link:
            reference = link["scene_configuration_attempt"]
            attempt = attempts.get(_raw_reference(reference))
            if attempt is None:
                absent.add("configuration_attempt_missing")
                missing["count"] += 1
                _require(missing["count"] <= MAX_REFERENCES, "references_limit")
                missing["raw"].append({"role": "attempt", **reference, "event_sequences": [], "status": "kept_unresolved",
                                       "reason": "configuration_attempt_missing", "source_provenance": None})
            else:
                value, provenance = attempt
                expected_id = "scene-configuration-" + link["request_digest"][7:31]
                _require(value.get("schema_version") == "task_evaluation_scene_attempt.v1"
                         and value.get("attempt_id") == expected_id and value.get("provider") == "vast"
                         and value.get("input_digest") == link["request_digest"]
                         and value.get("source_commit") == link["expected_production_commit"]
                         and provenance["path"] == retained._child(directory, "attempts", expected_id + ".json"), "attempt_identity_invalid")
                if request is not None:
                    _require(value["runtime_digest"] == request["execution_adapter"]["runtime_source_bundle"]["digest"], "attempt_runtime_invalid")
                sources.append(provenance)
        reasons.update(absent)
        context[key] = {"link": link, "request": request, "sources": sources, "complete": not absent,
                        "queue_state": matches[0][1].get("queue_state") if matches else None}
        if not matches:
            _missing(context[key], "preparation_envelope", None, roots["preparation_queue_root"], retained._STATES,
                     "preparation_envelope_missing", missing)
        if not absent:
            complete_links.extend(raw_index[p["path"]] for p in sources if p["role"] == "preparation_links")
            complete_envelopes.append(raw_index[matches[0][1]["path"]])
            if attempt:
                complete_attempts.append(raw_index[attempt[1]["path"]])
    _require(not envelopes, "envelope_unmatched")
    if complete_links:
        child = retained.join_scene_preparation_lineage(intent_id=intent["intent_id"], intent_record=records["intent"],
            preparation_links=complete_links, preparation_envelopes=complete_envelopes, configuration_attempt_records=complete_attempts,
            roots={key: roots[key] for key in ("intent_root", "preparation_queue_root", "preparation_input_root")})
        for row in child["preparations"]:
            members.append(_member(row["workspace_path"], "preparation_workspace",
                {key: row[key] for key in ("preparation_id", "request_digest", "expected_production_commit", "team_namespace", "scene_id", "task_id")},
                row["source_provenance"]))
    return context


def _typed_references(request: dict, budget: dict) -> dict:
    references, uris = {}, {}
    def visit(node, components):
        budget["nodes"] += 1
        _require(budget["nodes"] <= MAX_NODES and len(components) <= MAX_DEPTH, "traversal_limit")
        if isinstance(node, dict):
            fields = {"uri", "digest", "size_bytes"}
            if fields <= set(node) or len(fields & set(node)) >= 2:
                _require(set(node) == fields and isinstance(node["uri"], str)
                         and re.fullmatch(r"(gs|s3|https)://[^\s]+", node["uri"]) is not None
                         and retained._matches(node["digest"], retained._DIGEST)
                         and type(node["size_bytes"]) is int and node["size_bytes"] >= 1, "typed_reference_invalid")
                budget["typed"] += 1
                _require(budget["typed"] <= MAX_REFERENCES, "references_limit")
                path = ".".join(components)
                _require(len(path.encode("utf-8")) <= retained.MAX_PATH_BYTES and path not in references, "contract_path_invalid")
                identity = node["digest"], node["size_bytes"]
                _require(node["uri"] not in uris or uris[node["uri"]] == identity, "uri_identity_conflict")
                uris[node["uri"]] = identity
                references[path] = node
                return
            for key, value in node.items():
                visit(value, (*components, key))
        elif isinstance(node, list):
            for number, value in enumerate(node):
                visit(value, (*components, str(number)))
    visit(request, ())
    return references


def _result_references(value: dict, provenance: dict, context: dict, roots: dict, reasons: set,
                       members: list, shared: dict, deferred: list, budget: dict, promote: bool, missing: dict) -> None:
    references = value.get("references", [])
    _require(isinstance(references, list), "result_references_invalid")
    budget["result"] += len(references)
    _require(budget["result"] <= MAX_REFERENCES, "references_limit")
    request_refs = context["typed"]
    workspace = retained._child(roots["preparation_input_root"], context["link"]["preparation_id"])
    observed, projected = set(), {}
    for row in references:
        _require(isinstance(row, dict) and isinstance(row.get("contract_path"), str)
                 and 0 < len(row["contract_path"].encode("utf-8")) <= retained.MAX_PATH_BYTES
                 and isinstance(row.get("uri"), str) and retained._matches(row.get("digest"), retained._DIGEST)
                 and type(row.get("size_bytes")) is int and row["size_bytes"] >= 0
                 and type(row.get("content_addressed_reuse")) is bool
                 and type(row.get("full_byte_service_account_readback_passed")) is bool, "result_reference_invalid")
        path = retained._path(row.get("materialized_path"))
        _require(workspace in (str(parent) for parent in PurePosixPath(path).parents)
                 and PurePosixPath(path).name == row["digest"][7:], "materialized_path_invalid")
        identity = row["digest"], row["size_bytes"]
        _require(path not in projected or projected[path][0] == identity, "materialized_identity_conflict")
        projected.setdefault(path, (identity, []))[1].append(row["contract_path"])
        contract = request_refs.get(row["contract_path"])
        if contract is not None:
            _require(all(row[field] == contract[field] for field in ("uri", "digest", "size_bytes")), "request_reference_rebound")
            observed.add(row["contract_path"])
        if contract is None or not promote:
            deferred.append({**row, "preparation_id": context["link"]["preparation_id"], "source_provenance": [provenance],
                             "binding_strength": "result_receipt_only", "reason": "deferred_parent_reference_proof"})
        if promote and contract is not None:
            existing = budget["members"].get(path)
            if existing is None:
                existing = _member(path, "preparation_projected_file", {"preparation_id": context["link"]["preparation_id"],
                    "request_digest": context["link"]["request_digest"]}, list(context["sources"]) + [provenance])
                existing.update(binding_strength="request_typed_reference", receipt_digest=row["digest"], receipt_size_bytes=row["size_bytes"], contract_paths=[])
                members.append(existing)
                budget["members"][path] = existing
                budget["edges"][path] = set()
                budget["provenances"][path] = {(p["path"], p["sha256"], p["size_bytes"]) for p in existing["source_provenance"]}
            _require((existing["receipt_digest"], existing["receipt_size_bytes"]) == identity, "materialized_identity_conflict")
            budget["edges"][path].add(row["contract_path"])
            if _identity((value, provenance)) not in budget["provenances"][path]:
                budget["provenances"][path].add(_identity((value, provenance)))
                existing["source_provenance"].append(provenance)
            _require(row["digest"] not in shared or shared[row["digest"]]["size_bytes"] == row["size_bytes"], "cache_identity_conflict")
            shared[row["digest"]] = {"digest": row["digest"], "size_bytes": row["size_bytes"],
                "path": retained._child(roots["content_store_root"], row["digest"][7:]), "exclusive_scene_membership": False}
    if promote and set(request_refs) - observed:
        reasons.add("request_projection_missing")
        for contract_path in sorted(set(request_refs) - observed):
            missing["count"] += 1
            _require(missing["count"] <= MAX_REFERENCES, "references_limit")
            missing["projections"].append({"preparation_id": context["link"]["preparation_id"], "request_digest": context["link"]["request_digest"],
                "contract_path": contract_path, **request_refs[contract_path], "reason": "request_projection_missing", "status": "kept_unresolved",
                "source_provenance": sorted(list(context["sources"]) + [provenance], key=lambda p: (p["role"], p["path"], p["sha256"]))})


def _results(decoded: dict, context: dict, roots: dict, reasons: set, members: list, missing: dict) -> tuple:
    shared, deferred, materialized, seen = {}, [], {}, set()
    budget = {"nodes": 0, "typed": 0, "result": 0, "members": {}, "edges": {}, "provenances": {}}
    for group in context.values():
        group["typed"] = _typed_references(group["request"], budget) if group["request"] is not None else {}
    by_filename = {group["link"]["result_filename"]: group for group in context.values()}
    for value, provenance in sorted(decoded["preparation_results"], key=_identity):
        path = PurePosixPath(provenance["path"])
        _require(str(path.parent) == retained._child(roots["preparation_queue_root"], "results")
                 and path.name in by_filename, "result_path_invalid")
        group = by_filename[path.name]
        link, request = group["link"], group["request"]
        _require(value.get("schema_version") == "task_evaluation_launch_preparation_result.v1"
                 and value.get("preparation_id") == link["preparation_id"]
                 and retained._matches(value.get("source_commit"), _COMMIT), "result_invalid")
        retained._seal(value, provenance, "result_digest")
        missing["count"] += 1
        _require(missing["count"] <= MAX_REFERENCES, "references_limit")
        missing["raw"].append({"role": "preparation_result", **{key: provenance[key] for key in ("path", "sha256", "size_bytes")},
                               "event_sequences": [], "status": "matched_retained_bytes", "reason": None, "source_provenance": provenance})
        status = value.get("status")
        _require(isinstance(status, str), "result_invalid")
        flags = ("provider_mutation_performed", "catalog_mutation_performed", "paid_execution_requested")
        mismatch = (group["queue_state"] == "blocked" and status == "blocked"
                    and value.get("blockers") == ["launch_preparation_worker_source_commit_mismatch"]
                    and all(value.get(field) is False for field in flags))
        _require(value["source_commit"] == link["expected_production_commit"] or mismatch, "result_commit_invalid")
        known = status in {"inputs_materialized_awaiting_construction_adapter", "queued_for_production_scene_configuration"}
        if known or status in {"blocked", "awaiting_capacity"}:
            _require(all(value.get(field) is False for field in flags), "result_scope_invalid")
        if status in {"blocked", "awaiting_capacity"}:
            _require(isinstance(value.get("blockers"), list) and all(isinstance(code, str) for code in value["blockers"]), "result_invalid")
        if known:
            _require(type(value.get("reference_count")) is int and isinstance(value.get("references"), list)
                     and value["reference_count"] == len(value["references"])
                     and value.get("team_namespace") == link["team_namespace"]
                     and isinstance(value.get("run_id"), str), "result_identity_invalid")
            if request is not None:
                _require(value["run_id"] == request.get("run_id"), "result_identity_invalid")
        promote = known and group["complete"]
        _result_references(value, provenance, group, roots, reasons, members, shared, deferred, budget, promote, missing)
        seen.add(link["preparation_id"])
        if promote:
            materialized[(link["preparation_id"], value["result_digest"])] = value, provenance
        else:
            reasons.add("preparation_result_scope_unproven")
    if set(context) - seen:
        reasons.add("preparation_result_missing")
        for key in sorted(set(context) - seen):
            group = context[key]
            _missing(group, "preparation_result", retained._child(roots["preparation_queue_root"], "results", group["link"]["result_filename"]),
                     roots["preparation_queue_root"], {"results"}, "preparation_result_missing", missing)
    for path, member in budget["members"].items():
        member["contract_paths"] = sorted(budget["edges"][path])
        member["source_provenance"].sort(key=lambda p: (p["role"], p["path"], p["sha256"]))
    return shared, deferred, materialized


def _configurations(decoded: dict, context: dict, materialized: dict, roots: dict, reasons: set, members: list, missing: dict) -> None:
    envelopes, seen = {}, set()
    identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
    for value, provenance in decoded["activation_envelopes"]:
        _require(value.get("schema_version") == "task_evaluation_launch_activation_envelope.v1", "activation_invalid")
        retained._seal(value, provenance, "envelope_digest")
        request = value.get("request")
        _require(isinstance(request, dict) and request.get("schema_version") == "task_evaluation_launch_activation_request.v1"
                 and request.get("lane") == "task_evaluation_scene_configuration", "activation_invalid")
        _shape(request.get("activation_id"), identifier, "activation_invalid")
        digest = canonical_digest(request)
        _require(value.get("request_digest") == digest and all(value.get(field) is False for field in
                 ("provider_mutation_performed_inside_intake", "catalog_mutation_performed_inside_intake",
                  "standing_authorization_published_inside_intake", "paid_execution_requested")), "activation_invalid")
        filename = request["activation_id"] + "-" + digest[7:] + ".json"
        if len(filename.encode("utf-8")) > 255:
            filename = "activation-" + hashlib.sha256(request["activation_id"].encode("utf-8")).hexdigest() + "-" + digest[7:] + ".json"
        path = PurePosixPath(provenance["path"])
        _require(path.name == filename and path.parent.name in {"pending", "processing", "prepared", "blocked"}
                 and str(path.parent.parent) == roots["activation_queue_root"] and digest not in envelopes, "activation_ambiguous")
        envelopes[digest] = value, provenance
    for value, provenance in decoded["configuration_progressions"]:
        preparation_id = value.get("preparation_id")
        _shape(preparation_id, retained._ID, "configuration_invalid")
        _require(provenance["path"] == retained._child(roots["configuration_progression_root"], "scene-configuration-activations", preparation_id, "activation_progression.json")
                 and value.get("schema_version") == "task_evaluation_scene_configuration_activation_progression.v1"
                 and value.get("status") == "scene_configuration_activation_queued" and preparation_id in context
                 and value.get("provider_mutation_performed") is False and value.get("paid_execution_requested") is False, "configuration_invalid")
        retained._seal(value, provenance, "progression_digest")
        for field in ("intent_digest", "preparation_request_digest", "preparation_result_digest", "activation_request_digest"):
            _shape(value.get(field), retained._DIGEST, "configuration_invalid")
        _shape(value.get("activation_id"), identifier, "configuration_invalid")
        group = context[preparation_id]
        seen.add(preparation_id)
        for field in ("team_namespace", "scene_id", "task_id", "expected_production_commit"):
            _require(value.get(field) == group["link"][field], "configuration_identity_invalid")
        _require(value["preparation_request_digest"] == group["link"]["request_digest"], "configuration_identity_invalid")
        if group["request"] is not None:
            _require(value.get("run_id") == group["request"].get("run_id"), "configuration_identity_invalid")
        envelope = envelopes.pop(value["activation_request_digest"], None)
        result = materialized.get((preparation_id, value["preparation_result_digest"]))
        if envelope is not None:
            request = envelope[0]["request"]
            _require(request.get("activation_id") == value["activation_id"] and request.get("team_namespace") == value["team_namespace"]
                     and request.get("expected_production_commit") == value["expected_production_commit"]
                     and request.get("preparation") == {"preparation_id": preparation_id,
                         "request_digest": value["preparation_request_digest"], "result_digest": value["preparation_result_digest"]}, "configuration_identity_invalid")
        if envelope is None or result is None:
            reasons.add("configuration_join_unresolved")
            _missing({**group, "sources": list(group["sources"]) + [provenance]}, "configuration_join", None,
                     roots["activation_queue_root"], {"pending", "processing", "prepared", "blocked"}, "configuration_join_unresolved", missing)
        else:
            members.append(_member(str(PurePosixPath(provenance["path"]).parent), "configuration_progression_workspace",
                {"preparation_id": preparation_id, "request_digest": value["preparation_request_digest"], "result_digest": value["preparation_result_digest"]},
                list(group["sources"]) + [provenance, result[1], envelope[1]]))
    for key in sorted(set(context) - seen):
        reasons.add("configuration_join_unresolved")
        _missing(context[key], "configuration_progression", retained._child(roots["configuration_progression_root"],
            "scene-configuration-activations", key, "activation_progression.json"), roots["configuration_progression_root"], set(),
            "configuration_join_unresolved", missing)
    _require(not envelopes, "activation_unmatched")


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
    obligations, deferred, count = _obligations(events, decoded, reasons, roots, intent_id)
    missing = {"count": count, "joins": [], "projections": [], "raw": []}
    source_rows = _sources(intent_id, records, roots, reasons, members)
    context = _preparations(decoded, records, intent, roots, reasons, members, missing)
    shared, deferred_results, materialized = _results(decoded, context, roots, reasons, members, missing)
    _configurations(decoded, context, materialized, roots, reasons, members, missing)
    def obligation_key(row):
        return row["role"], row.get("path", row.get("uri", "")), row.get("sha256", row.get("digest", "")), row["size_bytes"]
    obligation_index = {obligation_key(row): row for row in obligations}
    for row in missing["raw"]:
        obligation_index.setdefault(obligation_key(row), row)
    obligations = [obligation_index[key] for key in sorted(obligation_index)]
    result = {"schema_version": "task_evaluation_scene_inventory_seed.v1", "status": "kept_unresolved" if reasons else "joined_supplied_seed",
              "scope": "supplied_retained_history_and_preparation_records", "intent_id": intent_id,
              "intent_digest": intent["intent_digest"], "intent_provenance": provenance, "history": history,
              "obligations": obligations, "members": sorted(members, key=lambda row: (row["kind"], row["path"])),
              "source_attempt_obligations": source_rows, "shared_cache_references": [shared[key] for key in sorted(shared)],
              "deferred_result_references": sorted(deferred_results, key=lambda row: (row["preparation_id"], row["contract_path"], row["digest"])),
              "preparation_join_obligations": sorted(missing["joins"], key=lambda row: (row["preparation_id"], row["request_digest"], row["role"])),
              "request_projection_obligations": sorted(missing["projections"], key=lambda row: (row["preparation_id"], row["contract_path"], row["digest"])),
              "unresolved_reasons": sorted(reasons), "deferred_branch_count": deferred + len(deferred_results) + sum(row["seed_disposition"] == "kept_deferred" for row in source_rows),
              "mutations": 0, "execution_authorized": False, "complete_scene_inventory": False,
              "historical_attempt_inventory_complete": False, "remaining_branches_complete": False,
              "payload_presence_checked": False, "payload_members_verified": False, "byte_accounting_complete": False,
              "finished_state_checked": False, "references_checked": False, "consumer_fence_checked": False,
              "publication_readback_checked": False, "remote_availability_checked": False,
              "cleanup_authorized": False, "requires_fresh_reference_check": True}
    _require(sum(len(result[key]) for key in ("obligations", "members", "source_attempt_obligations",
                                             "shared_cache_references", "deferred_result_references", "preparation_join_obligations", "request_projection_obligations")) <= MAX_ROWS, "rows_limit")
    _require(len(retained._encoded(result)) <= MAX_OUTPUT_BYTES, "output_limit")
    return result
