"""Pure retained preparation lineage for ADP-009D/day-28 disk safety.

Only supplied bytes are joined. Paths are lexical provenance, not observations
of payload existence, exclusive ownership, complete inventory or authorization.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_scene_preparation_link_contract import validate_preparation_link

MAX_RECORD_BYTES = MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 10_000
MAX_PATH_BYTES = 4096
MAX_PATH_COMPONENTS = 64
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._@-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STATES = {"pending", "processing", "awaiting_source_preparation", "awaiting_capacity",
           "materialized", "completed", "blocked"}
_ROOTS = {"intent_root", "preparation_queue_root", "preparation_input_root"}
_IDENTITY = ("schema_version", "intent_id", "intent_digest", "preparation_id",
             "request_digest", "expected_production_commit", "team_namespace",
             "scene_id", "task_id", "result_filename")


class SceneLineageError(ValueError):
    """Fixed bounded blocker; never includes supplied text or paths."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneLineageError("scene_lineage_" + code)


def _matches(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _path(value: Any) -> str:
    _require(isinstance(value, str), "path_invalid")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise SceneLineageError("scene_lineage_path_invalid") from None
    _require(size <= MAX_PATH_BYTES and value.startswith("/") and not value.startswith("//")
             and not any(ord(c) < 32 or ord(c) == 127 or c in "\\<>*" for c in value),
             "path_invalid")
    parts = value[1:].split("/") if value != "/" else []
    _require(len(parts) <= MAX_PATH_COMPONENTS and all(p not in {"", ".", ".."} for p in parts),
             "path_invalid")
    return value


def _child(root: str, *parts: str) -> str:
    return _path(root.rstrip("/") + "/" + "/".join(parts))


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, "json_invalid")
        result[key] = value
    return result


def _numeric(text: str) -> int | float:
    number = float(text) if any(c in text for c in ".eE") else int(text)
    try:
        valid = math.isfinite(number)
    except OverflowError:
        valid = False
    _require(valid, "json_invalid")
    return number


def _encoded(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        raise SceneLineageError("scene_lineage_json_invalid") from None


def _record(pair: Any, role: str, seen: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(isinstance(pair, (tuple, list)) and len(pair) == 2, "record_invalid")
    path, raw = pair
    path = _path(path)
    _require(path not in seen, "record_duplicate")
    seen.add(path)
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_RECORD_BYTES, "record_invalid")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_int=_numeric, parse_float=_numeric,
                           parse_constant=lambda _: _require(False, "json_invalid"))
    except (ValueError, UnicodeError, OverflowError, RecursionError):
        raise SceneLineageError("scene_lineage_json_invalid") from None
    _require(isinstance(value, dict), "record_invalid")
    _encoded(value)  # Reject decoded lone surrogates before seals or provenance.
    return value, {"role": role, "path": path, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                   "size_bytes": len(raw)}


def _preflight(groups: tuple[Any, ...]) -> None:
    total = 0
    for group in groups:
        for pair in group:
            _require(isinstance(pair, (tuple, list)) and len(pair) == 2, "record_invalid")
            raw = pair[1]
            _require(type(raw) is bytes and 0 < len(raw) <= MAX_RECORD_BYTES, "record_invalid")
            total += len(raw)
            _require(total <= MAX_TOTAL_BYTES, "bytes_limit")


def _seal(value: dict[str, Any], provenance: dict[str, Any], field: str,
          *, cross: bool = False) -> None:
    digest = cross_runtime_canonical_digest if cross else canonical_digest
    _require(_matches(value.get(field), _DIGEST)
             and value[field] == digest(value, digest_field=field), "seal_invalid")
    provenance.update(seal_field=field, seal_digest=value[field])


def _intent(value: dict[str, Any], provenance: dict[str, Any], intent_id: str,
            root: str) -> None:
    _require(provenance["path"] == _child(root, intent_id, "intent.json")
             and value.get("schema_version") == "task_evaluation_scene_intent.v1"
             and value.get("intent_id") == intent_id, "intent_invalid")
    _seal(value, provenance, "intent_digest", cross=True)
    request = value.get("request")
    _require(isinstance(request, dict)
             and request.get("schema_version") == "task_evaluation_scene_intake_request.v1"
             and _matches(request.get("submission_id"), _ID), "intent_request_invalid")
    owner, task, source = (request.get(k) for k in ("owner", "task", "source"))
    _require(isinstance(owner, dict) and set(owner) == {"user_id", "organization_id"}
             and all(_matches(v, _OWNER) for v in owner.values())
             and isinstance(task, dict) and _matches(task.get("task_id"), _ID)
             and isinstance(source, dict) and _matches(source.get("content_digest"), _DIGEST),
             "intent_request_invalid")
    _require(value.get("source_content_digest") == source["content_digest"]
             and value.get("task_content_digest") == cross_runtime_canonical_digest(task),
             "intent_content_invalid")


def _link(value: dict[str, Any], provenance: dict[str, Any], intent: dict[str, Any],
          intent_directory: str) -> dict[str, Any]:
    path = provenance["path"]
    variant = "activation" if path.endswith(".activation.json") else "base"
    _require(("scene_configuration_attempt" in value) == (variant == "activation"),
             "link_role_invalid")
    try:
        link = validate_preparation_link(value)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise SceneLineageError("scene_lineage_link_invalid") from None
    suffix = ".activation.json" if variant == "activation" else ".json"
    _require(path == _child(intent_directory, "preparations", link["request_digest"][7:] + suffix)
             and link["intent_id"] == intent["intent_id"]
             and link["intent_digest"] == intent["intent_digest"], "link_identity_invalid")
    _seal(link, provenance, "link_digest")
    provenance["variant"] = variant
    return link


def _envelope(value: dict[str, Any], provenance: dict[str, Any], root: str,
              link: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    path = PurePosixPath(provenance["path"])
    _require(path.parent.name in _STATES
             and str(path.parent.parent) == root and path.name == link["result_filename"]
             and value.get("schema_version") == "task_evaluation_launch_preparation_envelope.v1",
             "envelope_invalid")
    _seal(value, provenance, "envelope_digest")
    request = value.get("request")
    _require(isinstance(request, dict)
             and request.get("schema_version") == "task_evaluation_launch_preparation_request.v1"
             and request.get("run_mode") == "scene_configuration", "request_invalid")
    for name in ("preparation_id", "expected_production_commit", "team_namespace"):
        _require(request.get(name) == link[name], "request_identity_invalid")
    for name in ("scene", "task"):
        item = request.get(name)
        _require(isinstance(item, dict) and isinstance(item.get("identity"), dict)
                 and item["identity"].get("id") == link[name + "_id"], "request_identity_invalid")
    _require(request.get("scene_intent_digest") == intent["intent_digest"]
             and link["task_id"] == intent["request"]["task"]["task_id"]
             and value.get("request_digest") == link["request_digest"] == canonical_digest(request),
             "request_digest_invalid")
    adapter = request.get("execution_adapter")
    _require(isinstance(adapter, dict) and isinstance(adapter.get("runtime_source_bundle"), dict)
             and _matches(adapter["runtime_source_bundle"].get("digest"), _DIGEST), "runtime_invalid")
    provenance["queue_state"] = path.parent.name
    return request


def _attempt(link: dict[str, Any], request: dict[str, Any], intent: dict[str, Any],
             directory: str, remaining: dict[str, Any], sources: list[dict[str, Any]]) -> Any:
    if "scene_configuration_attempt" not in link:
        return None
    reference = link["scene_configuration_attempt"]
    path = _path(reference["path"])
    _require(path in remaining, "attempt_missing")
    value, provenance = remaining[path]
    _require(reference["sha256"] == provenance["sha256"]
             and reference["size_bytes"] == provenance["size_bytes"], "attempt_bytes_invalid")
    _seal(value, provenance, "attempt_digest", cross=True)
    attempt_id = "scene-configuration-" + link["request_digest"][7:31]
    _require(value.get("schema_version") == "task_evaluation_scene_attempt.v1"
             and value.get("intent_id") == intent["intent_id"]
             and value.get("intent_digest") == intent["intent_digest"]
             and value.get("source_commit") == link["expected_production_commit"]
             and value.get("input_digest") == link["request_digest"]
             and value.get("runtime_digest") == request["execution_adapter"]["runtime_source_bundle"]["digest"]
             and value.get("provider") == "vast" and value.get("attempt_id") == attempt_id
             and path == _child(directory, "attempts", attempt_id + ".json"), "attempt_identity_invalid")
    del remaining[path]
    sources.append(provenance)
    return {k: value[k] for k in ("attempt_id", "input_digest", "runtime_digest", "source_commit",
                                 "provider", "attempt_digest")}


def join_scene_preparation_lineage(*, intent_id: str, intent_record: Any,
                                   preparation_links: Any, preparation_envelopes: Any,
                                   configuration_attempt_records: Any, roots: Any) -> dict[str, Any]:
    """Join only supplied historical records; perform no I/O or authority checks."""
    try:
        return _join(intent_id, intent_record, preparation_links, preparation_envelopes,
                     configuration_attempt_records, roots)
    except SceneLineageError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        raise SceneLineageError("scene_lineage_input_invalid") from None


def _join(intent_id: str, intent_record: Any, links: Any, envelopes: Any,
          attempts: Any, roots: Any) -> dict[str, Any]:
    _require(_matches(intent_id, _ID) and isinstance(roots, dict) and set(roots) == _ROOTS,
             "parameters_invalid")
    roots = {key: _path(value) for key, value in roots.items()}
    _require(all(isinstance(group, (list, tuple)) for group in (links, envelopes, attempts))
             and 1 + len(links) + len(envelopes) + len(attempts) <= MAX_RECORDS, "records_limit")
    _preflight(([intent_record], links, envelopes, attempts))
    seen: set[str] = set()
    intent, intent_provenance = _record(intent_record, "intent", seen)
    _intent(intent, intent_provenance, intent_id, roots["intent_root"])
    decoded = [[_record(pair, role, seen) for pair in group]
               for role, group in (("link", links), ("envelope", envelopes), ("attempt", attempts))]
    remaining_attempts = {p["path"]: (value, p) for value, p in decoded[2]}
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    directory = _child(roots["intent_root"], intent_id)
    for value, provenance in decoded[0]:
        link = _link(value, provenance, intent, directory)
        previous = grouped.get(link["preparation_id"])
        _require(previous is None or all(previous[0][k] == link[k] for k in _IDENTITY), "link_ambiguous")
        if previous is None:
            grouped[link["preparation_id"]] = (link, [provenance])
        else:
            _require(all(p["variant"] != provenance["variant"] for p in previous[1]), "link_ambiguous")
            previous[1].append(provenance)
            chosen = link if provenance["variant"] == "activation" else previous[0]
            grouped[link["preparation_id"]] = (chosen, previous[1])
    remaining: dict[str, list[Any]] = {}
    for value, provenance in decoded[1]:
        remaining.setdefault(PurePosixPath(provenance["path"]).name, []).append((value, provenance))
    rows = []
    for preparation_id, (link, sources) in grouped.items():
        matches = remaining.pop(link["result_filename"], [])
        _require(len(matches) == 1, "envelope_ambiguous")
        envelope, provenance = matches[0]
        request = _envelope(envelope, provenance, roots["preparation_queue_root"], link, intent)
        sources.append(provenance)
        attempt = _attempt(link, request, intent, directory, remaining_attempts, sources)
        rows.append({**{k: link[k] for k in ("preparation_id", "request_digest", "expected_production_commit",
                                           "team_namespace", "scene_id", "task_id", "result_filename")},
                     "workspace_path": _child(roots["preparation_input_root"], preparation_id),
                     "source_provenance": sorted(sources, key=lambda p: (p["role"], p.get("variant", ""),
                                                                        p.get("queue_state", ""), p["path"])),
                     "configuration_attempt": attempt})
    _require(not remaining, "envelope_unmatched")
    _require(not remaining_attempts, "attempt_unmatched")
    result = {"schema_version": "task_evaluation_scene_preparation_lineage.v1", "status": "joined",
              "scope": "supplied_retained_preparation_records", "intent_id": intent_id,
              "intent_digest": intent["intent_digest"], "intent_provenance": intent_provenance,
              "preparation_count": len(rows),
              "preparations": sorted(rows, key=lambda row: (row["preparation_id"], row["request_digest"])),
              "mutations": 0, "execution_authorized": False, "complete_scene_inventory": False,
              "references_checked": False, "finished_state_checked": False,
              "payload_presence_checked": False, "requires_fresh_reference_check": True}
    _require(len(_encoded(result)) <= MAX_OUTPUT_BYTES, "output_limit")
    return result
