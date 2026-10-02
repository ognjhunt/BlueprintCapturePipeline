"""Pure retained source-attempt lineage for ADP-009D/day-28 disk safety.

Membership is a supplied-record join, never a filesystem, publication,
complete inventory, finished-state, reference clearance or eviction proof.
Shared parser/path/resource limits remain owned by preparation lineage.
"""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work_order, _work, _work_call, _work_items, _work_kwargs

import re
from typing import Any

from . import task_evaluation_scene_preparation_lineage as retained
from .decision_evidence_contracts import canonical_digest

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
_ADMIN_SCHEMA = "task_evaluation_scene_preparation_attempt.v1"
_PAID_SCHEMA = "task_evaluation_scene_attempt.v1"
_ATTEMPT_FIELDS = {"schema_version", "intent_id", "intent_digest", "attempt_id", "source_commit",
                   "runtime_digest", "input_digest", "provider", "maximum_spend_usd", "status",
                   "paid_authority_granted", "provider_allocation_permitted", "attempt_digest"}
_BINDINGS = {"task_evaluation_completed_scene_source.v1": "completed",
             "website_scene_source_binding.v1": "website",
             "task_evaluation_public_source_binding.v1": "public"}
_FACTORIES = {"task_evaluation_completed_scene_attempt_factory.v1": "completed",
              "website_scene_attempt_factory.v1": "website",
              "task_evaluation_public_scene_attempt_factory.v1": "public"}
_MACHINERY = {"completed": {"task_evaluation_completed_scene_machinery.v1"},
              "website": {"task_evaluation_website_scene_machinery.v1",
                          "task_evaluation_completed_scene_machinery.v1"},
              "public": {"task_evaluation_public_scene_machinery.v1"}}
_SNAPSHOTS = (("source_binding", "source_binding.json", "binding_digest"),
              ("machinery", "machinery.json", "machinery_digest"),
              ("release", "release_binding.json", "release_digest"))
_SUBMISSIONS = (("submission_request", "scene_configuration_preparation_request.v1.json"),
                ("submission_manifest", "bundle_manifest.v1.json"))
_SHARED_BLOCKERS = {"path_invalid", "record_invalid", "record_duplicate", "json_invalid", "seal_invalid",
                    "intent_invalid", "intent_request_invalid", "intent_content_invalid", "bytes_limit"}


class SceneSourceLineageError(ValueError):
    """Fixed bounded refusal, without supplied record/path/exception text."""


def _require(condition: bool, code: str, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    if not condition:
        raise SceneSourceLineageError("scene_source_lineage_" + code)


def _shape(value: Any, pattern: re.Pattern[str], code: str, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    _require(retained._matches(value, pattern, **_work_kwargs(work_budget)), code, **_work_kwargs(work_budget))


def _reference(value: Any, path: str, record: tuple | None, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    _require(isinstance(value, dict) and (_work_collect(work_budget, set, value) if work_budget is not None else set(value)) == {"path", "sha256", "size_bytes"}
             and value.get("path") == path and retained._path(value["path"], **_work_kwargs(work_budget)) == path
             and retained._matches(value.get("sha256"), retained._DIGEST, **_work_kwargs(work_budget))
             and type(value.get("size_bytes")) is int and value["size_bytes"] > 0, "reference_invalid", **_work_kwargs(work_budget))
    if record is not None:
        provenance = record[1]
        _require(all(value[k] == provenance[k] for k in (_work_items(("path", "sha256", "size_bytes"), work_budget) if work_budget is not None else ("path", "sha256", "size_bytes"))), "reference_rebound", **_work_kwargs(work_budget))


def _attempt(value: dict, provenance: dict, intent: dict, directory: str, *, work_budget=None) -> str:
    if work_budget is not None:
        _work(work_budget)
    attempt_id = value.get("attempt_id")
    _shape(attempt_id, retained._ID, "attempt_invalid", **_work_kwargs(work_budget))
    admin = value.get("schema_version") == _ADMIN_SCHEMA
    _require(admin or value.get("schema_version") == _PAID_SCHEMA, "attempt_invalid", **_work_kwargs(work_budget))
    if admin:
        _require((_work_collect(work_budget, set, value) if work_budget is not None else set(value)) == _ATTEMPT_FIELDS, "attempt_invalid", **_work_kwargs(work_budget))
    retained._seal(value, provenance, "attempt_digest", cross=True, **_work_kwargs(work_budget))
    _require(value.get("intent_id") == intent["intent_id"]
             and value.get("intent_digest") == intent["intent_digest"], "attempt_identity_invalid", **_work_kwargs(work_budget))
    _shape(value.get("source_commit"), _COMMIT, "attempt_invalid", **_work_kwargs(work_budget))
    for key in (_work_items(("input_digest", "runtime_digest"), work_budget) if work_budget is not None else ("input_digest", "runtime_digest")):
        _shape(value.get(key), retained._DIGEST, "attempt_invalid", **_work_kwargs(work_budget))
    if admin:
        spend = value.get("maximum_spend_usd")
        _require(value.get("provider") == "control_plane" and type(spend) in (int, float) and spend == 0
                 and value.get("status") == "preparation_only" and value.get("paid_authority_granted") is False
                 and value.get("provider_allocation_permitted") is False, "attempt_invalid", **_work_kwargs(work_budget))
    aliases = {retained._child(directory, alias, attempt_id + ".json", **_work_kwargs(work_budget)): alias
               for alias in (_work_items(("preparation-attempts", "attempts"), work_budget) if work_budget is not None else ("preparation-attempts", "attempts"))}
    _require(provenance["path"] in aliases, "attempt_path_invalid", **_work_kwargs(work_budget))
    alias = aliases[provenance["path"]]
    _require(admin or alias == "attempts", "attempt_path_invalid", **_work_kwargs(work_budget))
    return alias


def _binding(value: dict, provenance: dict, intent: dict, attempt: dict, *, work_budget=None) -> str:
    if work_budget is not None:
        _work(work_budget)
    family = _BINDINGS.get(value.get("schema_version"))
    _require(family is not None, "binding_invalid", **_work_kwargs(work_budget))
    retained._seal(value, provenance, "binding_digest", **_work_kwargs(work_budget))
    source = intent["request"]["source"]
    _require(value.get("binding_id") == source["binding_id"]
             and value.get("source_content_digest") == source["content_digest"]
             and value.get("owner") == intent["request"]["owner"]
             and attempt["input_digest"] == value["binding_digest"], "binding_identity_invalid", **_work_kwargs(work_budget))
    if family == "public":
        _require(source["kind"] == "public_scene" and value.get("status") == "admitted_for_private_processing"
                 and value.get("intent_task_digest") == intent["task_content_digest"], "binding_invalid", **_work_kwargs(work_budget))
    else:
        _require(value.get("intent_digest") == intent["intent_digest"]
                 and value.get("task_digest") == intent["task_content_digest"], "binding_identity_invalid", **_work_kwargs(work_budget))
    if family == "completed":
        _require(source["kind"] in {"mesh", "gaussian_splat"}
                 and value.get("source_kind") == source["kind"]
                 and value.get("status") == "source_task_objects_bound", "binding_invalid", **_work_kwargs(work_budget))
    return family


def _machinery(value: dict, intent: dict, family: str | None, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    schema = value.get("schema_version")
    _require(schema in set().union(*_MACHINERY.values())
             and (family is None or schema in _MACHINERY[family]), "machinery_invalid", **_work_kwargs(work_budget))
    if schema == "task_evaluation_public_scene_machinery.v1":
        bindings = value.get("retained_prefix_only_binding_ids")
        _require(isinstance(bindings, list) and all(retained._matches(item, retained._ID, **_work_kwargs(work_budget)) for item in (_work_items(bindings, work_budget) if work_budget is not None else bindings))
                 and intent["request"]["source"]["binding_id"] in bindings, "machinery_mode_invalid", **_work_kwargs(work_budget))


def _release(value: dict, provenance: dict, attempt: dict, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    _require(value.get("schema_version") == "task_evaluation_public_scene_release_binding.v1", "release_invalid", **_work_kwargs(work_budget))
    retained._seal(value, provenance, "release_digest", **_work_kwargs(work_budget))
    _shape(value.get("source_commit"), _COMMIT, "release_invalid", **_work_kwargs(work_budget))
    _shape(value.get("runtime_digest"), retained._DIGEST, "release_invalid", **_work_kwargs(work_budget))
    _require(value["source_commit"] == attempt["source_commit"]
             and value["runtime_digest"] == attempt["runtime_digest"], "release_identity_invalid", **_work_kwargs(work_budget))


def _request(value: dict, intent: dict, attempt: dict, *, work_budget=None) -> dict:
    if work_budget is not None:
        _work(work_budget)
    _require(value.get("schema_version") == "task_evaluation_launch_preparation_request.v1"
             and value.get("run_mode") == "scene_configuration"
             and value.get("scene_intent_digest") == intent["intent_digest"]
             and value.get("expected_production_commit") == attempt["source_commit"], "request_invalid", **_work_kwargs(work_budget))
    identity = {k: value.get(k) for k in (_work_items(("preparation_id", "team_namespace", "expected_production_commit"), work_budget) if work_budget is not None else ("preparation_id", "team_namespace", "expected_production_commit"))}
    for key in (_work_items(("preparation_id", "team_namespace"), work_budget) if work_budget is not None else ("preparation_id", "team_namespace")):
        _shape(identity[key], retained._ID, "request_invalid", **_work_kwargs(work_budget))
    for key in (_work_items(("scene", "task"), work_budget) if work_budget is not None else ("scene", "task")):
        item = value.get(key)
        _require(isinstance(item, dict) and isinstance(item.get("identity"), dict), "request_invalid", **_work_kwargs(work_budget))
        identifier = item["identity"].get("id")
        _shape(identifier, retained._ID, "request_invalid", **_work_kwargs(work_budget))
        identity[key + "_id"] = identifier
    _require(identity["task_id"] == intent["request"]["task"]["task_id"], "request_identity_invalid", **_work_kwargs(work_budget))
    publication = value.get("publication")
    _require(isinstance(publication, dict), "request_invalid", **_work_kwargs(work_budget))
    _namespace(publication.get("input_namespace"), **_work_kwargs(work_budget))
    identity["request_digest"] = (_work_call(work_budget, canonical_digest, value) if work_budget is not None else canonical_digest(value))
    return identity


def _namespace(value: Any, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    _shape(value, _NAMESPACE, "namespace_invalid", **_work_kwargs(work_budget))


def _manifest(value: dict, provenance: dict, attempt: dict, request: dict | None, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    _require(value.get("schema_version") == "task_evaluation_scene_configuration_submission_manifest.v1"
             and value.get("status") == "validated_pending_production_publication_and_submission"
             and value.get("source_commit") == attempt["source_commit"], "manifest_invalid", **_work_kwargs(work_budget))
    retained._seal(value, provenance, "manifest_digest", **_work_kwargs(work_budget))
    _shape(value.get("request_digest"), retained._DIGEST, "manifest_invalid", **_work_kwargs(work_budget))
    _namespace(value.get("input_namespace"), **_work_kwargs(work_budget))
    if request is not None:
        _require(value["request_digest"] == (_work_call(work_budget, canonical_digest, request) if work_budget is not None else canonical_digest(request))
                 and value["input_namespace"] == request["publication"]["input_namespace"], "manifest_identity_invalid", **_work_kwargs(work_budget))


def _factory(value: dict, provenance: dict, intent: dict, attempt: dict, family: str | None, *, work_budget=None) -> str:
    if work_budget is not None:
        _work(work_budget)
    chosen = _FACTORIES.get(value.get("schema_version"))
    _require(chosen is not None and (family is None or family == chosen)
             and value.get("status") == "publication_ready" and value.get("provider_mutation_performed") is False,
             "factory_invalid", **_work_kwargs(work_budget))
    retained._seal(value, provenance, "factory_digest", **_work_kwargs(work_budget))
    _require(value.get("intent_digest") == intent["intent_digest"]
             and value.get("attempt_digest") == attempt["attempt_digest"]
             and value.get("source_commit") == attempt["source_commit"], "factory_identity_invalid", **_work_kwargs(work_budget))
    if chosen == "completed":
        _require(value.get("source_kind") == intent["request"]["source"]["kind"]
                 and value["source_kind"] in {"mesh", "gaussian_splat"}, "factory_invalid", **_work_kwargs(work_budget))
    if chosen == "public":
        _require(intent["request"]["source"]["kind"] == "public_scene", "factory_invalid", **_work_kwargs(work_budget))
    return chosen


def _workspace(intent: dict, intent_provenance: dict, attempt: dict, provenance: dict,
               alias: str, root: str, pools: list[dict], *, emission_budget=None, work_budget=None) -> dict:
    if work_budget is not None:
        _work(work_budget)
    workspace = retained._child(root, intent["intent_id"], attempt["attempt_id"], **_work_kwargs(work_budget))
    sources, reasons = (emission_budget.reserve_provenance((provenance,)) if emission_budget is not None else [provenance]), set()
    snapshots = {}
    for role, filename, seal in (_work_items(_SNAPSHOTS, work_budget) if work_budget is not None else _SNAPSHOTS):
        record = pools[0].pop(retained._child(workspace, filename, **_work_kwargs(work_budget)), None)
        if record is None:
            reasons.add("source_snapshot_missing")
            continue
        record[1]["role"] = role
        retained._seal(*record, seal, **_work_kwargs(work_budget))
        snapshots[role] = record
        sources.append(record[1])
    family = _binding(*snapshots["source_binding"], intent, attempt, **_work_kwargs(work_budget)) if "source_binding" in snapshots else None
    if "release" in snapshots:
        _release(*snapshots["release"], attempt, **_work_kwargs(work_budget))
    factories = []
    for path in (_work_items((retained._child(workspace, "factory.json", **_work_kwargs(work_budget)),
                 retained._child(workspace, "materialized", "factory_receipt.json", **_work_kwargs(work_budget))), work_budget) if work_budget is not None else (retained._child(workspace, "factory.json"),
                 retained._child(workspace, "materialized", "factory_receipt.json"))):
        record = pools[1].pop(path, None)
        if record is not None:
            chosen = _factory(*record, intent, attempt, family, **_work_kwargs(work_budget))
            _require(not (chosen == "website" and path.endswith("/factory_receipt.json")), "factory_path_invalid", **_work_kwargs(work_budget))
            family = chosen
            factories.append(record)
            sources.append(record[1])
    _require(not factories or all(record[0] == factories[0][0] for record in (_work_items(factories, work_budget) if work_budget is not None else factories)), "factory_ambiguous", **_work_kwargs(work_budget))
    if "machinery" in snapshots:
        _machinery(snapshots["machinery"][0], intent, family, **_work_kwargs(work_budget))
    factory = factories[0][0] if factories else None
    if factory is None:
        reasons.add("source_factory_missing")
    if factory is not None and family == "public":
        identity = factory.get("identity")
        _require(isinstance(identity, dict) and (_work_collect(work_budget, set, identity) if work_budget is not None else set(identity)) == {
            "intent", "attempt", "source_binding", "machinery", "release", "factory_started_at_epoch"},
            "factory_identity_invalid", **_work_kwargs(work_budget))
        started = identity["factory_started_at_epoch"]
        _require(type(started) in (int, float) and started >= 0, "factory_identity_invalid", **_work_kwargs(work_budget))
        _reference(identity["intent"], intent_provenance["path"], (intent, intent_provenance), **_work_kwargs(work_budget))
        _reference(identity["attempt"], provenance["path"], (attempt, provenance), **_work_kwargs(work_budget))
        for role, filename, _ in (_work_items(_SNAPSHOTS, work_budget) if work_budget is not None else _SNAPSHOTS):
            _reference(identity[role], retained._child(workspace, filename, **_work_kwargs(work_budget)), snapshots.get(role), **_work_kwargs(work_budget))
    submissions = {}
    for role, filename in (_work_items(_SUBMISSIONS, work_budget) if work_budget is not None else _SUBMISSIONS):
        path = retained._child(workspace, "materialized", "submission", filename, **_work_kwargs(work_budget))
        record = pools[2].pop(path, None)
        if factory is not None:
            _reference(factory.get(role), path, record, **_work_kwargs(work_budget))
        if record is None:
            reasons.add("source_submission_missing")
            continue
        record[1]["role"] = role
        submissions[role] = record
        sources.append(record[1])
    request = submissions["submission_request"][0] if "submission_request" in submissions else None
    identity = _request(request, intent, attempt, **_work_kwargs(work_budget)) if request is not None else None
    if "submission_manifest" in submissions:
        _manifest(*submissions["submission_manifest"], attempt, request, **_work_kwargs(work_budget))
    bound = not reasons
    return {"attempt_id": attempt["attempt_id"], "attempt_digest": attempt["attempt_digest"],
            "attempt_schema": attempt["schema_version"], "attempt_alias": alias,
            "source_commit": attempt["source_commit"], "runtime_digest": attempt["runtime_digest"],
            "input_digest": attempt["input_digest"], "source_family": family,
            "status": "bound_retained_workspace" if bound else "kept_unresolved",
            "reasons": (_work_order(work_budget, sorted, reasons) if work_budget is not None else sorted(reasons)), "workspace_path": workspace, "workspace_membership_bound": bound,
            "snapshot_binding_strength": ("factory_raw_references" if family == "public"
                                          else "sealed_snapshots_at_expected_paths") if bound else None,
            "preparation_identity": identity if bound else None,
            "source_provenance": (_work_order(work_budget, sorted, sources, key=lambda row: (row["role"], row["path"])) if work_budget is not None else sorted(sources, key=lambda row: (row["role"], row["path"])))}


def _paid_row(attempt: dict, provenance: dict, alias: str, *, emission_budget=None, work_budget=None) -> dict:
    if work_budget is not None:
        _work(work_budget)
    sources = emission_budget.reserve_provenance((provenance,)) if emission_budget is not None else [provenance]
    return {"attempt_id": attempt["attempt_id"], "attempt_digest": attempt["attempt_digest"],
            "attempt_schema": attempt["schema_version"], "attempt_alias": alias,
            "source_commit": attempt["source_commit"], "runtime_digest": attempt["runtime_digest"],
            "input_digest": attempt["input_digest"], "source_family": None, "status": "kept_out_of_scope",
            "reasons": ["paid_attempt_out_of_scope"], "workspace_path": None, "workspace_membership_bound": False,
            "snapshot_binding_strength": None, "preparation_identity": None, "source_provenance": sources}


def join_scene_source_attempt_lineage(*, intent_id: str, intent_record: Any, attempt_records: Any,
                                    snapshot_records: Any, factory_records: Any, submission_records: Any,
                                    roots: Any) -> dict:
    """Join supplied historical records without I/O, runtime imports or authority."""
    try:
        return _join(intent_id, intent_record, attempt_records, snapshot_records, factory_records,
                     submission_records, roots)
    except retained.SceneLineageError as exc:
        suffix = str(exc).removeprefix("scene_lineage_")
        _require(str(exc).startswith("scene_lineage_") and suffix in _SHARED_BLOCKERS, "input_invalid")
        raise SceneSourceLineageError("scene_source_lineage_" + suffix) from None
    except SceneSourceLineageError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        raise SceneSourceLineageError("scene_source_lineage_input_invalid") from None


def _join(intent_id, intent_record, attempts, snapshots, factories, submissions, roots, *, emission_budget=None, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
        _require(emission_budget is not None and getattr(emission_budget, 'work_budget', None) is work_budget,
                 'parameters_invalid', **_work_kwargs(work_budget))
    if emission_budget is not None:
        emission_budget = emission_budget.scope(max_bytes=retained.MAX_OUTPUT_BYTES, max_rows=retained.MAX_RECORDS,
                                                max_references=retained.MAX_RECORDS)
    _require(retained._matches(intent_id, retained._ID, **_work_kwargs(work_budget)) and isinstance(roots, dict)
             and (_work_collect(work_budget, set, roots) if work_budget is not None else set(roots)) == {"intent_root", "factory_output_root"}, "parameters_invalid", **_work_kwargs(work_budget))
    groups = (attempts, snapshots, factories, submissions)
    _require(all(isinstance(group, (list, tuple)) for group in (_work_items(groups, work_budget) if work_budget is not None else groups))
             and 1 + sum(len(group) for group in (_work_items(groups, work_budget) if work_budget is not None else groups)) <= retained.MAX_RECORDS, "records_limit", **_work_kwargs(work_budget))
    for value in (_work_items(roots.values(), work_budget) if work_budget is not None else roots.values()):
        _require(isinstance(value, str) and len(value) <= retained.MAX_PATH_BYTES, "path_invalid", **_work_kwargs(work_budget))
    roots = {key: retained._path(value, **_work_kwargs(work_budget)) for key, value in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items())}
    retained._preflight(([intent_record], *groups), **_work_kwargs(work_budget))
    for group in (_work_items(([intent_record], *groups), work_budget) if work_budget is not None else ([intent_record], *groups)):
        for pair in (_work_items(group, work_budget) if work_budget is not None else group):
            _require(isinstance(pair[0], str) and len(pair[0]) <= retained.MAX_PATH_BYTES, "path_invalid", **_work_kwargs(work_budget))
            retained._path(pair[0], **_work_kwargs(work_budget))
    seen = set()
    intent, intent_provenance = retained._record(intent_record, "intent", seen, **_work_kwargs(work_budget))
    retained._intent(intent, intent_provenance, intent_id, roots["intent_root"], **_work_kwargs(work_budget))
    source = intent["request"]["source"]
    _shape(source.get("binding_id"), retained._ID, "intent_source_invalid", **_work_kwargs(work_budget))
    _require(source.get("kind") in {"capture_bundle", "mesh", "gaussian_splat", "public_scene"}, "intent_source_invalid", **_work_kwargs(work_budget))
    decoded = [[retained._record(pair, role, seen, **_work_kwargs(work_budget)) for pair in (_work_items(group, work_budget) if work_budget is not None else group)]
               for role, group in (_work_items(zip(("attempt", "snapshot", "factory", "submission"), groups), work_budget) if work_budget is not None else zip(("attempt", "snapshot", "factory", "submission"), groups))]
    for group in (_work_items(decoded, work_budget) if work_budget is not None else decoded):
        for _, provenance in (_work_items(group, work_budget) if work_budget is not None else group):
            provenance.update(seal_field=None, seal_digest=None)
    pools = [{record[1]["path"]: record for record in (_work_items(group, work_budget) if work_budget is not None else group)} for group in (_work_items(decoded[1:], work_budget) if work_budget is not None else decoded[1:])]
    directory = retained._child(roots["intent_root"], intent_id, **_work_kwargs(work_budget))
    rows, ids = (emission_budget.rows() if emission_budget is not None else []), set()
    for value, provenance in (_work_items(decoded[0], work_budget) if work_budget is not None else decoded[0]):
        alias = _attempt(value, provenance, intent, directory, **_work_kwargs(work_budget))
        _require(value["attempt_id"] not in ids, "attempt_ambiguous", **_work_kwargs(work_budget))
        ids.add(value["attempt_id"])
        rows.append(_workspace(intent, intent_provenance, value, provenance, alias, roots["factory_output_root"], pools, emission_budget=emission_budget, **_work_kwargs(work_budget))
                    if value["schema_version"] == _ADMIN_SCHEMA else _paid_row(value, provenance, alias, emission_budget=emission_budget, **_work_kwargs(work_budget)))
    _require(not any(pools), "record_unmatched", **_work_kwargs(work_budget))
    result = {"schema_version": "task_evaluation_scene_source_attempt_lineage.v1", "status": "joined_supplied_records",
              "scope": "supplied_retained_source_attempt_records", "intent_id": intent_id,
              "intent_digest": intent["intent_digest"], "intent_provenance": intent_provenance,
              "attempt_count": len(rows), "bound_workspace_count": sum(row["workspace_membership_bound"] for row in (_work_items(rows, work_budget) if work_budget is not None else rows)),
              "attempts": (_work_order(work_budget, sorted, rows, key=lambda row: row["attempt_id"]) if work_budget is not None else sorted(rows, key=lambda row: row["attempt_id"])), "mutations": 0, "execution_authorized": False,
              "complete_scene_inventory": False, "historical_attempt_inventory_complete": False,
              "payload_presence_checked": False, "payload_members_verified": False, "publication_readback_checked": False,
              "remote_availability_checked": False, "finished_state_checked": False, "references_checked": False,
              "consumer_fence_checked": False, "requires_fresh_reference_check": True}
    if emission_budget is not None:
        emission_budget.check_document(result)
    else:
        _require(len(retained._encoded(result, **_work_kwargs(work_budget))) <= retained.MAX_OUTPUT_BYTES, "output_limit", **_work_kwargs(work_budget))
    return result
