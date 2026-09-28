"""Pure retained preparation lineage for ADP-009D/day-28 disk safety.

Only supplied bytes are joined. Paths are lexical provenance, not observations
of payload existence, exclusive ownership, complete inventory or authorization.
"""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_order, _work, _work_call, _work_hash, _work_items, _work_kwargs, _work_parse

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


def _require(condition: bool, code: str, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    if not condition:
        raise SceneLineageError("scene_lineage_" + code)


def _matches(value: Any, pattern: re.Pattern[str], *, work_budget=None) -> bool:
    if work_budget is not None:
        _work(work_budget)
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _path(value: Any, *, work_budget=None) -> str:
    if work_budget is not None:
        _work(work_budget)
    _require(isinstance(value, str), "path_invalid", **_work_kwargs(work_budget))
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise SceneLineageError("scene_lineage_path_invalid") from None
    _require(size <= MAX_PATH_BYTES and value.startswith("/") and not value.startswith("//")
             and not any(ord(c) < 32 or ord(c) == 127 or c in "\\<>*" for c in (_work_items(value, work_budget) if work_budget is not None else value)),
             "path_invalid", **_work_kwargs(work_budget))
    parts = value[1:].split("/") if value != "/" else []
    _require(len(parts) <= MAX_PATH_COMPONENTS and all(p not in {"", ".", ".."} for p in (_work_items(parts, work_budget) if work_budget is not None else parts)),
             "path_invalid", **_work_kwargs(work_budget))
    return value


def _child(root: str, *parts: str, work_budget=None) -> str:
    if work_budget is not None:
        _work(work_budget)
    return _path(root.rstrip("/") + "/" + "/".join(parts), **_work_kwargs(work_budget))


def _pairs(pairs: list[tuple[str, Any]], *, work_budget=None) -> dict[str, Any]:
    if work_budget is not None:
        _work(work_budget)
    result = {}
    for key, value in (_work_items(pairs, work_budget) if work_budget is not None else pairs):
        _require(key not in result, "json_invalid", **_work_kwargs(work_budget))
        result[key] = value
    return result


def _numeric(text: str, *, work_budget=None) -> int | float:
    if work_budget is not None:
        _work(work_budget)
    number = float(text) if any(c in text for c in (_work_items(".eE", work_budget) if work_budget is not None else ".eE")) else int(text)
    try:
        valid = math.isfinite(number)
    except OverflowError:
        valid = False
    _require(valid, "json_invalid", **_work_kwargs(work_budget))
    return number


def _encoded(value: Any, *, work_budget=None) -> bytes:
    if work_budget is not None:
        _work(work_budget)
    try:
        return (_work_call(work_budget, json.dumps, value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")) if work_budget is not None else json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":"))).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        raise SceneLineageError("scene_lineage_json_invalid") from None


def _record(pair: Any, role: str, seen: set[str], *, work_budget=None) -> tuple[dict[str, Any], dict[str, Any]]:
    if work_budget is not None:
        _work(work_budget)
    _require(isinstance(pair, (tuple, list)) and len(pair) == 2, "record_invalid", **_work_kwargs(work_budget))
    path, raw = pair
    path = _path(path, **_work_kwargs(work_budget))
    _require(path not in seen, "record_duplicate", **_work_kwargs(work_budget))
    seen.add(path)
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_RECORD_BYTES, "record_invalid", **_work_kwargs(work_budget))
    try:
        value = (_work_parse(work_budget, json.loads, raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_int=_numeric, parse_float=_numeric,
                           parse_constant=lambda _: _require(False, "json_invalid")) if work_budget is not None else json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_int=_numeric, parse_float=_numeric,
                           parse_constant=lambda _: _require(False, "json_invalid")))
    except (ValueError, UnicodeError, OverflowError, RecursionError):
        raise SceneLineageError("scene_lineage_json_invalid") from None
    _require(isinstance(value, dict), "record_invalid", **_work_kwargs(work_budget))
    _encoded(value, **_work_kwargs(work_budget))  # Reject decoded lone surrogates before seals or provenance.
    return value, {"role": role, "path": path, "sha256": "sha256:" + (_work_hash(work_budget, hashlib.sha256, raw) if work_budget is not None else hashlib.sha256(raw)).hexdigest(),
                   "size_bytes": len(raw)}


def _preflight(groups: tuple[Any, ...], *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    total = 0
    for group in (_work_items(groups, work_budget) if work_budget is not None else groups):
        for pair in (_work_items(group, work_budget) if work_budget is not None else group):
            _require(isinstance(pair, (tuple, list)) and len(pair) == 2, "record_invalid", **_work_kwargs(work_budget))
            raw = pair[1]
            _require(type(raw) is bytes and 0 < len(raw) <= MAX_RECORD_BYTES, "record_invalid", **_work_kwargs(work_budget))
            if work_budget is not None:
                work_budget.preflight(raw.decode('utf-8'))
            total += len(raw)
            _require(total <= MAX_TOTAL_BYTES, "bytes_limit", **_work_kwargs(work_budget))


def _seal(value: dict[str, Any], provenance: dict[str, Any], field: str,
          *, cross: bool = False, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    digest = cross_runtime_canonical_digest if cross else canonical_digest
    _require(_matches(value.get(field), _DIGEST, **_work_kwargs(work_budget))
             and value[field] == (_work_call(work_budget, digest, value, digest_field=field) if work_budget is not None else digest(value, digest_field=field)), "seal_invalid", **_work_kwargs(work_budget))
    provenance.update(seal_field=field, seal_digest=value[field])


def _intent(value: dict[str, Any], provenance: dict[str, Any], intent_id: str,
            root: str, *, work_budget=None) -> None:
    if work_budget is not None:
        _work(work_budget)
    _require(provenance["path"] == _child(root, intent_id, "intent.json", **_work_kwargs(work_budget))
             and value.get("schema_version") == "task_evaluation_scene_intent.v1"
             and value.get("intent_id") == intent_id, "intent_invalid", **_work_kwargs(work_budget))
    _seal(value, provenance, "intent_digest", cross=True, **_work_kwargs(work_budget))
    request = value.get("request")
    _require(isinstance(request, dict)
             and request.get("schema_version") == "task_evaluation_scene_intake_request.v1"
             and _matches(request.get("submission_id"), _ID, **_work_kwargs(work_budget)), "intent_request_invalid", **_work_kwargs(work_budget))
    owner, task, source = (request.get(k) for k in (_work_items(("owner", "task", "source"), work_budget) if work_budget is not None else ("owner", "task", "source")))
    _require(isinstance(owner, dict) and set(owner) == {"user_id", "organization_id"}
             and all(_matches(v, _OWNER, **_work_kwargs(work_budget)) for v in (_work_items(owner.values(), work_budget) if work_budget is not None else owner.values()))
             and isinstance(task, dict) and _matches(task.get("task_id"), _ID, **_work_kwargs(work_budget))
             and isinstance(source, dict) and _matches(source.get("content_digest"), _DIGEST, **_work_kwargs(work_budget)),
             "intent_request_invalid", **_work_kwargs(work_budget))
    _require(value.get("source_content_digest") == source["content_digest"]
             and value.get("task_content_digest") == (_work_call(work_budget, cross_runtime_canonical_digest, task) if work_budget is not None else cross_runtime_canonical_digest(task)),
             "intent_content_invalid", **_work_kwargs(work_budget))


def _link(value: dict[str, Any], provenance: dict[str, Any], intent: dict[str, Any],
          intent_directory: str, *, work_budget=None) -> dict[str, Any]:
    if work_budget is not None:
        _work(work_budget)
    path = provenance["path"]
    variant = "activation" if path.endswith(".activation.json") else "base"
    _require(("scene_configuration_attempt" in value) == (variant == "activation"),
             "link_role_invalid", **_work_kwargs(work_budget))
    try:
        link = (_work_call(work_budget, validate_preparation_link, value) if work_budget is not None else validate_preparation_link(value))
    except (ValueError, TypeError, KeyError, OverflowError):
        raise SceneLineageError("scene_lineage_link_invalid") from None
    suffix = ".activation.json" if variant == "activation" else ".json"
    _require(path == _child(intent_directory, "preparations", link["request_digest"][7:] + suffix, **_work_kwargs(work_budget))
             and link["intent_id"] == intent["intent_id"]
             and link["intent_digest"] == intent["intent_digest"], "link_identity_invalid", **_work_kwargs(work_budget))
    _seal(link, provenance, "link_digest", **_work_kwargs(work_budget))
    provenance["variant"] = variant
    return link


def _envelope(value: dict[str, Any], provenance: dict[str, Any], root: str,
              link: dict[str, Any], intent: dict[str, Any], *, work_budget=None) -> dict[str, Any]:
    if work_budget is not None:
        _work(work_budget)
    path = PurePosixPath(provenance["path"])
    _require(path.parent.name in _STATES
             and str(path.parent.parent) == root and path.name == link["result_filename"]
             and value.get("schema_version") == "task_evaluation_launch_preparation_envelope.v1",
             "envelope_invalid", **_work_kwargs(work_budget))
    _seal(value, provenance, "envelope_digest", **_work_kwargs(work_budget))
    request = value.get("request")
    _require(isinstance(request, dict)
             and request.get("schema_version") == "task_evaluation_launch_preparation_request.v1"
             and request.get("run_mode") == "scene_configuration", "request_invalid", **_work_kwargs(work_budget))
    for name in (_work_items(("preparation_id", "expected_production_commit", "team_namespace"), work_budget) if work_budget is not None else ("preparation_id", "expected_production_commit", "team_namespace")):
        _require(request.get(name) == link[name], "request_identity_invalid", **_work_kwargs(work_budget))
    for name in (_work_items(("scene", "task"), work_budget) if work_budget is not None else ("scene", "task")):
        item = request.get(name)
        _require(isinstance(item, dict) and isinstance(item.get("identity"), dict)
                 and item["identity"].get("id") == link[name + "_id"], "request_identity_invalid", **_work_kwargs(work_budget))
    _require(request.get("scene_intent_digest") == intent["intent_digest"]
             and link["task_id"] == intent["request"]["task"]["task_id"]
             and value.get("request_digest") == link["request_digest"] == (_work_call(work_budget, canonical_digest, request) if work_budget is not None else canonical_digest(request)),
             "request_digest_invalid", **_work_kwargs(work_budget))
    adapter = request.get("execution_adapter")
    _require(isinstance(adapter, dict) and isinstance(adapter.get("runtime_source_bundle"), dict)
             and _matches(adapter["runtime_source_bundle"].get("digest"), _DIGEST, **_work_kwargs(work_budget)), "runtime_invalid", **_work_kwargs(work_budget))
    provenance["queue_state"] = path.parent.name
    return request


def _attempt(link: dict[str, Any], request: dict[str, Any], intent: dict[str, Any],
             directory: str, remaining: dict[str, Any], sources: list[dict[str, Any]], *, emission_budget=None, work_budget=None) -> Any:
    if work_budget is not None:
        _work(work_budget)
    if "scene_configuration_attempt" not in link:
        return None
    reference = link["scene_configuration_attempt"]
    path = _path(reference["path"], **_work_kwargs(work_budget))
    _require(path in remaining, "attempt_missing", **_work_kwargs(work_budget))
    value, provenance = remaining[path]
    _require(reference["sha256"] == provenance["sha256"]
             and reference["size_bytes"] == provenance["size_bytes"], "attempt_bytes_invalid", **_work_kwargs(work_budget))
    _seal(value, provenance, "attempt_digest", cross=True, **_work_kwargs(work_budget))
    attempt_id = "scene-configuration-" + link["request_digest"][7:31]
    _require(value.get("schema_version") == "task_evaluation_scene_attempt.v1"
             and value.get("intent_id") == intent["intent_id"]
             and value.get("intent_digest") == intent["intent_digest"]
             and value.get("source_commit") == link["expected_production_commit"]
             and value.get("input_digest") == link["request_digest"]
             and value.get("runtime_digest") == request["execution_adapter"]["runtime_source_bundle"]["digest"]
             and value.get("provider") == "vast" and value.get("attempt_id") == attempt_id
             and path == _child(directory, "attempts", attempt_id + ".json", **_work_kwargs(work_budget)), "attempt_identity_invalid", **_work_kwargs(work_budget))
    del remaining[path]
    sources.append(provenance)
    return {k: value[k] for k in (_work_items(("attempt_id", "input_digest", "runtime_digest", "source_commit",
                                 "provider", "attempt_digest"), work_budget) if work_budget is not None else ("attempt_id", "input_digest", "runtime_digest", "source_commit",
                                 "provider", "attempt_digest"))}


def join_scene_preparation_lineage(*, intent_id: str, intent_record: Any,
                                   preparation_links: Any, preparation_envelopes: Any,
                                   configuration_attempt_records: Any, roots: Any, work_budget=None) -> dict[str, Any]:
    """Join only supplied historical records; perform no I/O or authority checks."""
    if work_budget is not None:
        _work(work_budget)
    try:
        return _join(intent_id, intent_record, preparation_links, preparation_envelopes,
                     configuration_attempt_records, roots, **_work_kwargs(work_budget))
    except SceneLineageError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        raise SceneLineageError("scene_lineage_input_invalid") from None


def _join(intent_id: str, intent_record: Any, links: Any, envelopes: Any,
          attempts: Any, roots: Any, *, emission_budget=None, work_budget=None) -> dict[str, Any]:
    if work_budget is not None:
        _work(work_budget)
    if emission_budget is not None:
        emission_budget = emission_budget.scope(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_RECORDS, max_references=MAX_RECORDS)
    _require(_matches(intent_id, _ID, **_work_kwargs(work_budget)) and isinstance(roots, dict) and set(roots) == _ROOTS,
             "parameters_invalid", **_work_kwargs(work_budget))
    roots = {key: _path(value, **_work_kwargs(work_budget)) for key, value in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items())}
    _require(all(isinstance(group, (list, tuple)) for group in (_work_items((links, envelopes, attempts), work_budget) if work_budget is not None else (links, envelopes, attempts)))
             and 1 + len(links) + len(envelopes) + len(attempts) <= MAX_RECORDS, "records_limit", **_work_kwargs(work_budget))
    _preflight(([intent_record], links, envelopes, attempts), **_work_kwargs(work_budget))
    seen: set[str] = set()
    intent, intent_provenance = _record(intent_record, "intent", seen, **_work_kwargs(work_budget))
    _intent(intent, intent_provenance, intent_id, roots["intent_root"], **_work_kwargs(work_budget))
    decoded = [[_record(pair, role, seen, **_work_kwargs(work_budget)) for pair in (_work_items(group, work_budget) if work_budget is not None else group)]
               for role, group in (_work_items((("link", links), ("envelope", envelopes), ("attempt", attempts)), work_budget) if work_budget is not None else (("link", links), ("envelope", envelopes), ("attempt", attempts)))]
    remaining_attempts = {p["path"]: (value, p) for value, p in (_work_items(decoded[2], work_budget) if work_budget is not None else decoded[2])}
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    directory = _child(roots["intent_root"], intent_id, **_work_kwargs(work_budget))
    for value, provenance in (_work_items(decoded[0], work_budget) if work_budget is not None else decoded[0]):
        link = _link(value, provenance, intent, directory, **_work_kwargs(work_budget))
        previous = grouped.get(link["preparation_id"])
        _require(previous is None or all(previous[0][k] == link[k] for k in (_work_items(_IDENTITY, work_budget) if work_budget is not None else _IDENTITY)), "link_ambiguous", **_work_kwargs(work_budget))
        if previous is None:
            grouped[link["preparation_id"]] = (link, emission_budget.reserve_provenance((provenance,)) if emission_budget is not None else [provenance])
        else:
            _require(all(p["variant"] != provenance["variant"] for p in (_work_items(previous[1], work_budget) if work_budget is not None else previous[1])), "link_ambiguous", **_work_kwargs(work_budget))
            previous[1].append(provenance)
            chosen = link if provenance["variant"] == "activation" else previous[0]
            grouped[link["preparation_id"]] = (chosen, previous[1])
    remaining: dict[str, list[Any]] = {}
    for value, provenance in (_work_items(decoded[1], work_budget) if work_budget is not None else decoded[1]):
        remaining.setdefault(PurePosixPath(provenance["path"]).name, []).append((value, provenance))
    rows = emission_budget.rows() if emission_budget is not None else []
    for preparation_id, (link, sources) in (_work_items(grouped.items(), work_budget) if work_budget is not None else grouped.items()):
        matches = remaining.pop(link["result_filename"], [])
        _require(len(matches) == 1, "envelope_ambiguous", **_work_kwargs(work_budget))
        envelope, provenance = matches[0]
        request = _envelope(envelope, provenance, roots["preparation_queue_root"], link, intent, **_work_kwargs(work_budget))
        sources.append(provenance)
        attempt = _attempt(link, request, intent, directory, remaining_attempts, sources, emission_budget=emission_budget, **_work_kwargs(work_budget))
        rows.append({**{k: link[k] for k in (_work_items(("preparation_id", "request_digest", "expected_production_commit",
                                           "team_namespace", "scene_id", "task_id", "result_filename"), work_budget) if work_budget is not None else ("preparation_id", "request_digest", "expected_production_commit",
                                           "team_namespace", "scene_id", "task_id", "result_filename"))},
                     "workspace_path": _child(roots["preparation_input_root"], preparation_id, **_work_kwargs(work_budget)),
                     "source_provenance": (_work_order(work_budget, sorted, sources, key=lambda p: (p["role"], p.get("variant", ""),
                                                                        p.get("queue_state", ""), p["path"])) if work_budget is not None else sorted(sources, key=lambda p: (p["role"], p.get("variant", ""),
                                                                        p.get("queue_state", ""), p["path"]))),
                     "configuration_attempt": attempt})
    _require(not remaining, "envelope_unmatched", **_work_kwargs(work_budget))
    _require(not remaining_attempts, "attempt_unmatched", **_work_kwargs(work_budget))
    result = {"schema_version": "task_evaluation_scene_preparation_lineage.v1", "status": "joined",
              "scope": "supplied_retained_preparation_records", "intent_id": intent_id,
              "intent_digest": intent["intent_digest"], "intent_provenance": intent_provenance,
              "preparation_count": len(rows),
              "preparations": (_work_order(work_budget, sorted, rows, key=lambda row: (row["preparation_id"], row["request_digest"])) if work_budget is not None else sorted(rows, key=lambda row: (row["preparation_id"], row["request_digest"]))),
              "mutations": 0, "execution_authorized": False, "complete_scene_inventory": False,
              "references_checked": False, "finished_state_checked": False,
              "payload_presence_checked": False, "requires_fresh_reference_check": True}
    if emission_budget is not None:
        emission_budget.check_document(result)
    else:
        _require(len(_encoded(result, **_work_kwargs(work_budget))) <= MAX_OUTPUT_BYTES, "output_limit", **_work_kwargs(work_budget))
    return result
