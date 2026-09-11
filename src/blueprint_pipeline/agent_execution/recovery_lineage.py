"""Derive exact recovery bindings only across recorded compatible releases.

The operator's exact binding is the immutable opt-in anchor. This module writes
only its own derivation receipts; the existing controller still owns retries,
consent, spend, provider admission and execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from ..decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .contracts import AgentExecutionError, digest
from .controller_recovery import ControllerRecoveryBinding, validate_controller_binding

SCHEMA = "blueprint_agent_controller_recovery_derivation.v1"
STATES = ("pending", "processing", "awaiting_source_preparation", "materialized", "completed", "blocked")


def _require(condition, code):
    if not condition:
        raise AgentExecutionError("agent_recovery_lineage_" + code)


def _read(path, *, field=None, cross=False):
    from .production import _read_private
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    raw = _read_private(path)
    value = json.loads(raw)
    if field:
        measure = cross_runtime_canonical_digest if cross else canonical_digest
        _require(value.get(field) == measure(value, digest_field=field), "digest_invalid")
    return value, {"path": str(path), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _reference(ref, *, root=None, field=None, cross=False):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"}, "reference_invalid")
    path = Path(ref["path"])
    _require(root is None or (Path(root).is_absolute() and ".." not in Path(root).parts
                             and path.is_relative_to(root)), "reference_outside_owner_root")
    value, observed = _read(path, field=field, cross=cross)
    _require(observed == ref, "reference_changed")
    return value


def _scope(service, anchor):
    config, intent = validate_controller_binding(anchor)
    directory = Path(config["intent_root"]) / anchor.intent_id
    execution = intent["request"]["execution"]
    _require(not (directory / "revoked.json").exists()
             and time.time() < execution["expires_at_epoch"]
             and anchor.intent_id not in config.get("paused_intent_ids", []), "owner_authority_revoked")
    _require(config.get("only_intent_id") in (None, anchor.intent_id), "controller_owner_changed")
    allowances = [row for row in service.config.automatic_supervision_allowances if row.intent_id == anchor.intent_id]
    _require(len(allowances) <= 1, "allowance_ambiguous")
    if allowances:
        allowance = allowances[0]
        _require(allowance.intent_digest == anchor.intent_digest and time.time() < allowance.expires_at
                 and allowance.expires_at <= execution["expires_at_epoch"]
                 and allowance.maximum_reserved_inference_usd <= execution["max_total_spend_usd"], "allowance_revoked")
    return config, intent, directory, {
        "execution_bounds": execution,
        "supervision_allowance_digest": allowances[0].allowance_digest if allowances else None,
        "max_task_budget_usd": service.config.max_task_budget_usd,
    }


def _events(directory, intent):
    paths = sorted((directory / "progression-events").glob("*.json"))
    _require(0 < len(paths) <= 10000, "event_inventory_invalid")
    rows, previous = [], None
    for number, path in enumerate(paths, 1):
        event, ref = _read(path, field="event_digest", cross=True)
        _require(event.get("schema_version") == "task_evaluation_scene_progression_event.v1"
                 and event.get("intent_id") == intent["intent_id"]
                 and event.get("intent_digest") == intent["intent_digest"]
                 and event.get("sequence") == number and path.name == f"{number:06d}.json"
                 and event.get("previous_event_digest") == previous, "event_chain_invalid")
        rows.append((event, ref))
        previous = event["event_digest"]
    current, _ = _read(directory / "progression.json", field="progression_digest", cross=True)
    last = rows[-1][0]
    _require(current.get("schema_version") == "task_evaluation_scene_progression.v1"
             and current.get("event_sequence") == last["sequence"]
             and current.get("last_event_digest") == last["event_digest"]
             and current.get("updated_at_epoch") == last["observed_at_epoch"]
             and all(current.get(key) == last.get(key) for key in
                     ("intent_id", "intent_digest", "status", "phase", "state", "result_reference"))
             and current.get("blockers", []) == last.get("blockers", []), "projection_not_current")
    return rows, current


def _link(anchor, ref, directory):
    value = _reference(ref, root=directory / "preparations", field="link_digest")
    _require(value.get("schema_version") == "task_evaluation_scene_preparation_link.v1"
             and value.get("intent_id") == anchor.intent_id and value.get("intent_digest") == anchor.intent_digest
             and Path(ref["path"]).name == value["request_digest"][7:] + ".json", "preparation_link_invalid")
    return value


def _attempt(ref, directory, anchor):
    value = _reference(ref, root=directory / "attempts", field="attempt_digest", cross=True)
    _require(value.get("schema_version") == "task_evaluation_scene_attempt.v1"
             and value.get("intent_id") == anchor.intent_id and value.get("intent_digest") == anchor.intent_digest
             and Path(ref["path"]).name == value["attempt_id"] + ".json", "attempt_invalid")
    return value


def _attempt_namespace(anchor, attempt):
    # Match the existing public-scene factory's tenant/attempt namespace.
    return "scene-" + canonical_digest({"intent_digest": anchor.intent_digest,
                                       "attempt_digest": attempt["attempt_digest"]})[7:55]


def _release_edge(previous_ref, current_ref, previous_link_ref, state, directory, anchor, config):
    before, after = (_attempt(ref, directory, anchor) for ref in (previous_ref, current_ref))
    edges = [row for row in state.get("release_predecessors", [])
             if row.get("attempt") == previous_ref and row.get("new_source_commit") == after["source_commit"]]
    _require(len(edges) == 1 and edges[0].get("basis") == "terminal_preparation_and_reconciled_global_ownership"
             and before["source_commit"] != after["source_commit"]
             and all(before.get(key) == after.get(key) for key in
                     ("input_digest", "provider", "maximum_spend_usd")), "compatible_release_successor_required")
    refs = edges[0].get("reconciliation", {})
    _require(set(refs) == {"failure", "provider_guard", "ownership_reconciliation"}, "release_reconciliation_missing")
    output = Path(config["factory_output_root"]) / anchor.intent_id / before["attempt_id"]
    transition = _reference(refs["failure"], root=output, field="failure_digest")
    prior_link = _link(anchor, previous_link_ref, directory)
    prior_envelope = _reference(transition.get("parent_envelope"), root=Path(config["preparation_queue_root"]),
                                field="envelope_digest")
    ownership = _reference(refs["ownership_reconciliation"], root=output, field="ownership_digest")
    guard = _reference(refs["provider_guard"], root=output)
    _require(transition.get("schema_version") == "task_evaluation_scene_release_transition.v1"
             and transition.get("attempt_digest") == before["attempt_digest"]
             and prior_envelope.get("request_digest") == prior_link["request_digest"]
             and prior_envelope.get("request", {}).get("expected_production_commit") == before["source_commit"]
             and prior_envelope.get("request", {}).get("team_namespace") == prior_link["team_namespace"]
             == _attempt_namespace(anchor, before)
             and transition.get("parent_state") in {"blocked", "completed", "materialized"}
             and ownership.get("schema_version") == "task_evaluation_scene_attempt_ownership.v1"
             and ownership.get("attempt_digest") == before["attempt_digest"]
             and ownership.get("status") == "closed_without_resource"
             and ownership.get("active_writer_count") == ownership.get("unresolved_create_count") == 0
             and ownership.get("provider_guard") == refs["provider_guard"]
             and guard.get("schema_version") == "gpu_spend_guard.v1" and guard.get("status") == "passed",
             "release_reconciliation_invalid")
    return {"previous_attempt": previous_ref, "successor_attempt": current_ref, "reconciliation": refs}


def _derive(service, anchor, *, parent_request_digest, parent_queue_root):
    config, intent, directory, limits = _scope(service, anchor)
    queue = Path(config["preparation_queue_root"])
    _require(str(queue) == str(parent_queue_root), "unrelated_parent_queue")
    rows, current = _events(directory, intent)
    ref = current.get("state", {}).get("preparation_link")
    link = _link(anchor, ref, directory)
    _require(link["request_digest"] == parent_request_digest
             and link["expected_production_commit"] == service.config.source_commit, "current_parent_changed")
    original, anchor_ref = _read(anchor.preparation_link_path, field="link_digest")
    _require(anchor_ref["sha256"] == anchor.preparation_link_sha256
             and all(link.get(key) == original.get(key) for key in ("scene_id", "task_id")),
             "frozen_task_changed")
    if link["request_digest"] == anchor.parent_request_digest:
        return anchor, None
    anchor_rows = [index for index, (event, _) in enumerate(rows)
                   if event.get("state", {}).get("preparation_link") == anchor_ref]
    target_rows = [index for index, (event, _) in enumerate(rows)
                   if event.get("state", {}).get("preparation_link") == ref]
    _require(anchor_rows and target_rows and anchor_rows[0] < target_rows[0], "recorded_anchor_required")
    first, last = anchor_rows[0], target_rows[0]
    prior_ref = rows[first][0]["state"].get("attempt")
    prior_link_ref = anchor_ref
    _attempt(prior_ref, directory, anchor)
    edges = []
    for event, _ in rows[first + 1:last + 1]:
        next_ref = event.get("state", {}).get("attempt")
        if next_ref is not None and next_ref != prior_ref:
            edges.append(_release_edge(prior_ref, next_ref, prior_link_ref, event["state"], directory, anchor, config))
            prior_ref = next_ref
        if event.get("state", {}).get("preparation_link") is not None:
            prior_link_ref = event["state"]["preparation_link"]
    _require(bool(edges) and current["state"].get("attempt") == prior_ref, "successor_attempt_not_recorded")
    current_attempt = _attempt(prior_ref, directory, anchor)
    _require(current_attempt["source_commit"] == link["expected_production_commit"]
             and link["team_namespace"] == _attempt_namespace(anchor, current_attempt),
             "successor_release_changed")
    matches = [queue / state / link["result_filename"] for state in STATES
               if (queue / state / link["result_filename"]).exists()]
    _require(len(matches) == 1, "parent_queue_ambiguous")
    envelope, envelope_ref = _read(matches[0], field="envelope_digest")
    request = envelope.get("request", {})
    _require(envelope.get("schema_version") == "task_evaluation_launch_preparation_envelope.v1"
             and envelope.get("request_digest") == canonical_digest(request) == parent_request_digest
             and request.get("preparation_id") == link["preparation_id"]
             and request.get("expected_production_commit") == service.config.source_commit
             and request.get("team_namespace") == link["team_namespace"]
             and request.get("scene", {}).get("identity", {}).get("id") == link["scene_id"]
             and request.get("task", {}).get("identity", {}).get("id") == link["task_id"], "parent_envelope_invalid")
    identity = {"anchor_digest": digest(anchor.model_dump(mode="json")),
                "parent_request_digest": parent_request_digest, "unchanged_authority": limits,
                "source_commit": service.config.source_commit}
    derived = ControllerRecoveryBinding(**{**anchor.model_dump(mode="json"),
        "recovery_id": "recover-" + digest(identity)[7:39], "parent_request_digest": parent_request_digest,
        "preparation_link_path": ref["path"], "preparation_link_sha256": ref["sha256"],
        "allow_controller_successors": False})
    validate_controller_binding(derived)
    payload = {"schema_version": SCHEMA, "anchor": anchor.model_dump(mode="json"),
        "anchor_digest": digest(anchor.model_dump(mode="json")), "binding": derived.model_dump(mode="json"),
        "source_commit": service.config.source_commit, "parent_queue_root": str(queue),
        "anchor_event": rows[first][1], "successor_event": rows[last][1], "lineage_edges": edges,
        "parent_envelope": {"sha256": envelope_ref["sha256"], "size_bytes": envelope_ref["size_bytes"]},
        "unchanged_authority": limits, "provider_mutation_performed": False, "historical_records_modified": False}
    payload["derivation_digest"] = digest(payload)
    return derived, payload


def _receipt_path(service, binding):
    return service.journal.root / "controller-recovery-bindings" / (digest(binding.model_dump(mode="json"))[7:] + ".json")


def resolve_recovery_binding(service, *, intent_id, parent_request_digest, parent_queue_root):
    anchors = [row for row in service.config.automatic_recovery_bindings if row.intent_id == intent_id]
    _require(len(anchors) <= 1, "anchor_ambiguous")
    if not anchors:
        return None
    anchor = anchors[0]
    if not anchor.allow_controller_successors:
        return anchor if anchor.parent_request_digest == parent_request_digest else None
    binding, payload = _derive(service, anchor, parent_request_digest=parent_request_digest, parent_queue_root=parent_queue_root)
    if payload is not None:
        from ..common import write_json
        path = _receipt_path(service, binding)
        with service.journal.own_task("controller-recovery-lineage:" + intent_id):
            if path.exists():
                _require(_read(path)[0] == payload, "derivation_changed")
            else:
                write_json(path, payload)
                path.chmod(0o440)
    return binding


def recovery_binding_authorized(service, binding):
    if binding in service.config.automatic_recovery_bindings and not binding.allow_controller_successors:
        return True
    try:
        anchors = [row for row in service.config.automatic_recovery_bindings
                   if row.intent_id == binding.intent_id and row.allow_controller_successors]
        _require(len(anchors) == 1, "anchor_revoked")
        config, _, _, _ = _scope(service, anchors[0])
        derived, expected = _derive(service, anchors[0], parent_request_digest=binding.parent_request_digest,
                                    parent_queue_root=config["preparation_queue_root"])
        _require(derived == binding, "binding_changed")
        if expected is not None:
            _require(_read(_receipt_path(service, binding))[0] == expected, "derivation_changed")
        return True
    except (AgentExecutionError, ValueError, OSError, KeyError, TypeError):
        return False
