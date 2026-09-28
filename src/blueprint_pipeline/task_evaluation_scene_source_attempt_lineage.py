"""Pure retained source-attempt lineage for ADP-009D/day-28 disk safety.

Membership is a supplied-record join, never a filesystem, publication,
complete inventory, finished-state, reference clearance or eviction proof.
Shared parser/path/resource limits remain owned by preparation lineage.
"""
from __future__ import annotations

import re
from typing import Any

from . import task_evaluation_scene_preparation_lineage as retained
from .decision_evidence_contracts import canonical_digest

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
_ADMIN_SCHEMA = "task_evaluation_scene_preparation_attempt.v1"
_ATTEMPT_FIELDS = {"schema_version", "intent_id", "intent_digest", "attempt_id", "source_commit",
                   "runtime_digest", "input_digest", "provider", "maximum_spend_usd", "status",
                   "paid_authority_granted", "provider_allocation_permitted", "attempt_digest"}
_BINDINGS = {"task_evaluation_completed_scene_source.v1": "completed",
             "website_scene_source_binding.v1": "website"}
_FACTORIES = {"task_evaluation_completed_scene_attempt_factory.v1": "completed",
              "website_scene_attempt_factory.v1": "website"}
_MACHINERY = {"completed": {"task_evaluation_completed_scene_machinery.v1"},
              "website": {"task_evaluation_website_scene_machinery.v1",
                          "task_evaluation_completed_scene_machinery.v1"}}
_SNAPSHOTS = (("source_binding", "source_binding.json", "binding_digest"),
              ("machinery", "machinery.json", "machinery_digest"),
              ("release", "release_binding.json", "release_digest"))
_SUBMISSIONS = (("submission_request", "scene_configuration_preparation_request.v1.json"),
                ("submission_manifest", "bundle_manifest.v1.json"))


class SceneSourceLineageError(ValueError):
    """Fixed bounded refusal, without supplied record/path/exception text."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneSourceLineageError("scene_source_lineage_" + code)


def _shape(value: Any, pattern: re.Pattern[str], code: str) -> None:
    _require(retained._matches(value, pattern), code)


def _reference(value: Any, path: str, record: tuple | None) -> None:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"}
             and value.get("path") == path and retained._path(value["path"]) == path
             and retained._matches(value.get("sha256"), retained._DIGEST)
             and type(value.get("size_bytes")) is int and value["size_bytes"] > 0, "reference_invalid")
    if record is not None:
        provenance = record[1]
        _require(all(value[k] == provenance[k] for k in ("path", "sha256", "size_bytes")), "reference_rebound")


def _attempt(value: dict, provenance: dict, intent: dict, directory: str) -> str:
    attempt_id = value.get("attempt_id")
    _shape(attempt_id, retained._ID, "attempt_invalid")
    _require(value.get("schema_version") == _ADMIN_SCHEMA and set(value) == _ATTEMPT_FIELDS,
             "attempt_invalid")
    retained._seal(value, provenance, "attempt_digest", cross=True)
    _require(value.get("intent_id") == intent["intent_id"]
             and value.get("intent_digest") == intent["intent_digest"], "attempt_identity_invalid")
    _shape(value.get("source_commit"), _COMMIT, "attempt_invalid")
    for key in ("input_digest", "runtime_digest"):
        _shape(value.get(key), retained._DIGEST, "attempt_invalid")
    spend = value.get("maximum_spend_usd")
    _require(value.get("provider") == "control_plane" and type(spend) in (int, float) and spend == 0
             and value.get("status") == "preparation_only" and value.get("paid_authority_granted") is False
             and value.get("provider_allocation_permitted") is False, "attempt_invalid")
    aliases = {retained._child(directory, alias, attempt_id + ".json"): alias
               for alias in ("preparation-attempts", "attempts")}
    _require(provenance["path"] in aliases, "attempt_path_invalid")
    return aliases[provenance["path"]]


def _binding(value: dict, provenance: dict, intent: dict, attempt: dict) -> str:
    family = _BINDINGS.get(value.get("schema_version"))
    _require(family is not None, "binding_invalid")
    retained._seal(value, provenance, "binding_digest")
    source = intent["request"]["source"]
    _require(value.get("binding_id") == source["binding_id"]
             and value.get("source_content_digest") == source["content_digest"]
             and value.get("owner") == intent["request"]["owner"]
             and value.get("intent_digest") == intent["intent_digest"]
             and value.get("task_digest") == intent["task_content_digest"]
             and attempt["input_digest"] == value["binding_digest"], "binding_identity_invalid")
    if family == "completed":
        _require(source["kind"] in {"mesh", "gaussian_splat"}
                 and value.get("source_kind") == source["kind"]
                 and value.get("status") == "source_task_objects_bound", "binding_invalid")
    return family


def _release(value: dict, provenance: dict, attempt: dict) -> None:
    _require(value.get("schema_version") == "task_evaluation_public_scene_release_binding.v1", "release_invalid")
    retained._seal(value, provenance, "release_digest")
    _shape(value.get("source_commit"), _COMMIT, "release_invalid")
    _shape(value.get("runtime_digest"), retained._DIGEST, "release_invalid")
    _require(value["source_commit"] == attempt["source_commit"]
             and value["runtime_digest"] == attempt["runtime_digest"], "release_identity_invalid")


def _request(value: dict, intent: dict, attempt: dict) -> dict:
    _require(value.get("schema_version") == "task_evaluation_launch_preparation_request.v1"
             and value.get("run_mode") == "scene_configuration"
             and value.get("scene_intent_digest") == intent["intent_digest"]
             and value.get("expected_production_commit") == attempt["source_commit"], "request_invalid")
    identity = {k: value.get(k) for k in ("preparation_id", "team_namespace", "expected_production_commit")}
    for key in ("preparation_id", "team_namespace"):
        _shape(identity[key], retained._ID, "request_invalid")
    for key in ("scene", "task"):
        item = value.get(key)
        _require(isinstance(item, dict) and isinstance(item.get("identity"), dict), "request_invalid")
        identifier = item["identity"].get("id")
        _shape(identifier, retained._ID, "request_invalid")
        identity[key + "_id"] = identifier
    _require(identity["task_id"] == intent["request"]["task"]["task_id"], "request_identity_invalid")
    publication = value.get("publication")
    _require(isinstance(publication, dict), "request_invalid")
    _namespace(publication.get("input_namespace"))
    identity["request_digest"] = canonical_digest(value)
    return identity


def _namespace(value: Any) -> None:
    _shape(value, _NAMESPACE, "namespace_invalid")


def _manifest(value: dict, provenance: dict, attempt: dict, request: dict | None) -> None:
    _require(value.get("schema_version") == "task_evaluation_scene_configuration_submission_manifest.v1"
             and value.get("status") == "validated_pending_production_publication_and_submission"
             and value.get("source_commit") == attempt["source_commit"], "manifest_invalid")
    retained._seal(value, provenance, "manifest_digest")
    _shape(value.get("request_digest"), retained._DIGEST, "manifest_invalid")
    _namespace(value.get("input_namespace"))
    if request is not None:
        _require(value["request_digest"] == canonical_digest(request)
                 and value["input_namespace"] == request["publication"]["input_namespace"], "manifest_identity_invalid")


def _factory(value: dict, provenance: dict, intent: dict, attempt: dict, family: str | None) -> str:
    chosen = _FACTORIES.get(value.get("schema_version"))
    _require(chosen is not None and (family is None or family == chosen)
             and value.get("status") == "publication_ready" and value.get("provider_mutation_performed") is False,
             "factory_invalid")
    retained._seal(value, provenance, "factory_digest")
    _require(value.get("intent_digest") == intent["intent_digest"]
             and value.get("attempt_digest") == attempt["attempt_digest"]
             and value.get("source_commit") == attempt["source_commit"], "factory_identity_invalid")
    if chosen == "completed":
        _require(value.get("source_kind") == intent["request"]["source"]["kind"]
                 and value["source_kind"] in {"mesh", "gaussian_splat"}, "factory_invalid")
    return chosen


def _workspace(intent: dict, attempt: dict, provenance: dict, alias: str, root: str, pools: list[dict]) -> dict:
    workspace = retained._child(root, intent["intent_id"], attempt["attempt_id"])
    sources = [provenance]
    snapshots = {}
    for role, filename, seal in _SNAPSHOTS:
        record = pools[0].pop(retained._child(workspace, filename), None)
        _require(record is not None, "snapshot_missing")
        record[1]["role"] = role
        retained._seal(*record, seal)
        snapshots[role] = record
        sources.append(record[1])
    family = _binding(*snapshots["source_binding"], intent, attempt)
    _require(snapshots["machinery"][0].get("schema_version") in _MACHINERY[family], "machinery_invalid")
    _release(*snapshots["release"], attempt)
    factories = []
    for path in (retained._child(workspace, "factory.json"),
                 retained._child(workspace, "materialized", "factory_receipt.json")):
        record = pools[1].pop(path, None)
        if record is not None:
            _require(not (family == "website" and path.endswith("/factory_receipt.json")), "factory_path_invalid")
            _factory(*record, intent, attempt, family)
            factories.append(record)
            sources.append(record[1])
    _require(factories and all(record[0] == factories[0][0] for record in factories), "factory_ambiguous")
    factory = factories[0][0]
    submissions = {}
    for role, filename in _SUBMISSIONS:
        path = retained._child(workspace, "materialized", "submission", filename)
        record = pools[2].pop(path, None)
        _reference(factory.get(role), path, record)
        _require(record is not None, "submission_missing")
        record[1]["role"] = role
        submissions[role] = record
        sources.append(record[1])
    request = submissions["submission_request"][0]
    identity = _request(request, intent, attempt)
    _manifest(*submissions["submission_manifest"], attempt, request)
    return {"attempt_id": attempt["attempt_id"], "attempt_digest": attempt["attempt_digest"],
            "attempt_schema": attempt["schema_version"], "attempt_alias": alias,
            "source_commit": attempt["source_commit"], "runtime_digest": attempt["runtime_digest"],
            "input_digest": attempt["input_digest"], "source_family": family, "status": "bound_retained_workspace",
            "reasons": [], "workspace_path": workspace, "workspace_membership_bound": True,
            "snapshot_binding_strength": "sealed_snapshots_at_expected_paths", "preparation_identity": identity,
            "source_provenance": sorted(sources, key=lambda row: (row["role"], row["path"]))}


def join_scene_source_attempt_lineage(*, intent_id: str, intent_record: Any, attempt_records: Any,
                                    snapshot_records: Any, factory_records: Any, submission_records: Any,
                                    roots: Any) -> dict:
    """Join supplied historical records without I/O, runtime imports or authority."""
    try:
        return _join(intent_id, intent_record, attempt_records, snapshot_records, factory_records,
                     submission_records, roots)
    except retained.SceneLineageError as exc:
        suffix = str(exc).removeprefix("scene_lineage_")
        _require(re.fullmatch(r"[a-z_]{1,64}", suffix) is not None, "input_invalid")
        raise SceneSourceLineageError("scene_source_lineage_" + suffix) from None
    except SceneSourceLineageError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        raise SceneSourceLineageError("scene_source_lineage_input_invalid") from None


def _join(intent_id, intent_record, attempts, snapshots, factories, submissions, roots):
    _require(retained._matches(intent_id, retained._ID) and isinstance(roots, dict)
             and set(roots) == {"intent_root", "factory_output_root"}, "parameters_invalid")
    groups = (attempts, snapshots, factories, submissions)
    _require(all(isinstance(group, (list, tuple)) for group in groups)
             and 1 + sum(len(group) for group in groups) <= retained.MAX_RECORDS, "records_limit")
    roots = {key: retained._path(value) for key, value in roots.items()}
    retained._preflight(([intent_record], *groups))
    seen = set()
    intent, intent_provenance = retained._record(intent_record, "intent", seen)
    retained._intent(intent, intent_provenance, intent_id, roots["intent_root"])
    source = intent["request"]["source"]
    _shape(source.get("binding_id"), retained._ID, "intent_source_invalid")
    _require(source.get("kind") in {"capture_bundle", "mesh", "gaussian_splat", "public_scene"}, "intent_source_invalid")
    decoded = [[retained._record(pair, role, seen) for pair in group]
               for role, group in zip(("attempt", "snapshot", "factory", "submission"), groups)]
    for group in decoded:
        for _, provenance in group:
            provenance.update(seal_field=None, seal_digest=None)
    pools = [{record[1]["path"]: record for record in group} for group in decoded[1:]]
    directory = retained._child(roots["intent_root"], intent_id)
    rows, ids = [], set()
    for value, provenance in decoded[0]:
        alias = _attempt(value, provenance, intent, directory)
        _require(value["attempt_id"] not in ids, "attempt_ambiguous")
        ids.add(value["attempt_id"])
        rows.append(_workspace(intent, value, provenance, alias, roots["factory_output_root"], pools))
    _require(not any(pools), "record_unmatched")
    result = {"schema_version": "task_evaluation_scene_source_attempt_lineage.v1", "status": "joined_supplied_records",
              "scope": "supplied_retained_source_attempt_records", "intent_id": intent_id,
              "intent_digest": intent["intent_digest"], "intent_provenance": intent_provenance,
              "attempt_count": len(rows), "bound_workspace_count": sum(row["workspace_membership_bound"] for row in rows),
              "attempts": sorted(rows, key=lambda row: row["attempt_id"]), "mutations": 0, "execution_authorized": False,
              "complete_scene_inventory": False, "historical_attempt_inventory_complete": False,
              "payload_presence_checked": False, "payload_members_verified": False, "publication_readback_checked": False,
              "remote_availability_checked": False, "finished_state_checked": False, "references_checked": False,
              "consumer_fence_checked": False, "requires_fresh_reference_check": True}
    _require(len(retained._encoded(result)) <= retained.MAX_OUTPUT_BYTES, "output_limit")
    return result
