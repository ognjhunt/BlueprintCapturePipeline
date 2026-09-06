"""Durable, idempotent terminal-result index written by the paid run's producers.

The Spec E terminal reconciler
(``task_evaluation_scene_terminal_reconciler.reconcile_terminal_owner_result``)
joins six sealed receipts under ``terminal_result_root/<intent-id>/`` into the
persistent owner status. Those receipts are produced by DIFFERENT real producers,
at different locations and file names, during the paid policy-canary run and its
teardown (the policy-canary projection + authenticated Website sync, the launch
request/profile bridge, the launch reconciler's post-teardown provider-zero
receipt). Nothing gathered them into the one owner-scoped directory the reconciler
reads (audit finding A6), so the reconciler was inert in production.

This module is that missing index. It is invoked at paid-run completion with the
paths of the ALREADY-SEALED producer outputs, validates each against exactly the
schema the reconciler consumes, and writes the six reconciler-named files
idempotently (byte-identical on a re-run). It additionally PRODUCES the durable
result publication (``terminal_result_publication.json``) -- the one receipt with
no upstream producer -- from the projection's own identity and the durable result
reference, sealed so the reconciler's A7 integrity checks bind it to the run. It
never launches, retries, tears down, allocates a provider, or rewrites a producer
receipt; it only re-files sealed evidence and seals the publication over it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_policy_canary_result import (
    TaskEvaluationPolicyCanaryResultError,
    validate_policy_canary_result,
)
from .task_evaluation_scene_terminal_reconciler import (
    PROVIDER_ZERO_SCHEMA,
    PUBLICATION_SCHEMA,
)

INDEX_SCHEMA = "task_evaluation_scene_terminal_result_index.v1"
_WEBAPP_SYNC_SCHEMA = "task_evaluation_policy_canary_webapp_sync_result.v1"
_LAUNCH_REQUEST_SCHEMA = "task_evaluation_launch_request.v1"
_LAUNCH_PROFILE_SCHEMA = "task_evaluation_launch_profile.v1"
_RESULT_URI_PREFIXES = ("https://", "b2://", "gs://", "r2://")

#: Reconciler filename -> (source key, expected schema_version | None). ``None``
#: for the projection, which is validated by ``validate_policy_canary_result``.
_COPIED = {
    "policy_canary_result_projection.json": ("projection_path", None),
    "policy_canary_webapp_sync.json": ("webapp_sync_path", _WEBAPP_SYNC_SCHEMA),
    "provider_zero_closure.json": ("post_teardown_provider_zero_path", PROVIDER_ZERO_SCHEMA),
    "launch_request.json": ("launch_request_path", _LAUNCH_REQUEST_SCHEMA),
    "launch_profile.json": ("launch_profile_path", _LAUNCH_PROFILE_SCHEMA),
}


class TerminalResultIndexError(ValueError):
    """A producer receipt is absent or does not match the reconciler's contract."""


def _fail(code: str) -> None:
    raise TerminalResultIndexError("terminal_result_index_" + code)


def _safe(path: str | Path) -> Path:
    item = Path(path)
    if not item.is_absolute() or any(p.is_symlink() for p in (item, *item.parents)):
        _fail("path_unsafe")
    return item


def _read(path: Path, *, reason: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(reason)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _fail(reason)
    if not isinstance(value, dict):
        _fail(reason)
    return value


def _put(path: Path, value: dict[str, Any]) -> Path:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_text(encoding="utf-8") != serialized:
            _fail("immutable_conflict")
        return path
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path


def index_terminal_owner_result(*, intent_id: str, terminal_result_root: str | Path,
                                projection_path: str | Path, webapp_sync_path: str | Path,
                                post_teardown_provider_zero_path: str | Path,
                                launch_request_path: str | Path, launch_profile_path: str | Path,
                                result_uri: str, result_size_bytes: int) -> dict[str, Any]:
    """Gather the paid run's sealed producer receipts into the reconciler's
    owner-scoped directory and produce the durable result publication.

    ``result_uri``/``result_size_bytes`` describe the durable, already-published
    run result (the reconciler requires a completed-unqualified result to carry a
    sealed publication). All inputs are validated against the reconciler's exact
    contract; a missing or off-contract receipt fails closed. Re-running with the
    same sealed inputs is byte-identical.
    """
    sources = {
        "projection_path": _safe(projection_path), "webapp_sync_path": _safe(webapp_sync_path),
        "post_teardown_provider_zero_path": _safe(post_teardown_provider_zero_path),
        "launch_request_path": _safe(launch_request_path), "launch_profile_path": _safe(launch_profile_path),
    }
    # The projection is the join key; validate it with the canonical validator.
    projection_raw = _read(sources["projection_path"], reason="projection_absent")
    try:
        projection = validate_policy_canary_result(projection_raw)
    except TaskEvaluationPolicyCanaryResultError:
        _fail("projection_invalid")
    run_id = projection["run_id"]
    projection_digest = projection["projection_digest"]

    directory = _safe(Path(terminal_result_root) / intent_id)
    directory.mkdir(parents=True, exist_ok=True, mode=0o750)

    written: dict[str, str] = {}
    for filename, (source_key, schema) in _COPIED.items():
        value = _read(sources[source_key], reason="receipt_absent")
        if schema is not None and value.get("schema_version") != schema:
            _fail("receipt_schema_invalid")
        written[filename] = str(_put(directory / filename, value))

    # A6: produce the durable result publication -- the one receipt with no
    # upstream producer -- bound to THIS run and sealed for the reconciler's A7
    # integrity checks. Never assert allocation.
    if not (isinstance(result_uri, str) and result_uri.startswith(_RESULT_URI_PREFIXES) and "?" not in result_uri):
        _fail("result_uri_invalid")
    if not (type(result_size_bytes) is int and result_size_bytes > 0):
        _fail("result_size_invalid")
    publication = {"schema_version": PUBLICATION_SCHEMA, "run_id": run_id, "uri": result_uri,
                   "digest": projection_digest, "size_bytes": result_size_bytes,
                   "provider_allocated": False, "publication_digest": ""}
    publication["publication_digest"] = canonical_digest(publication, digest_field="publication_digest")
    written["terminal_result_publication.json"] = str(_put(directory / "terminal_result_publication.json", publication))

    return {"schema_version": INDEX_SCHEMA, "status": "terminal_result_indexed",
            "intent_id": intent_id, "run_id": run_id, "directory": str(directory),
            "files": written, "provider_mutation_performed": False}


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index a paid run's terminal receipts for the owner reconciler.")
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--terminal-result-root", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--webapp-sync", required=True)
    parser.add_argument("--post-teardown-provider-zero", required=True)
    parser.add_argument("--launch-request", required=True)
    parser.add_argument("--launch-profile", required=True)
    parser.add_argument("--result-uri", required=True)
    parser.add_argument("--result-size-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    result = index_terminal_owner_result(
        intent_id=args.intent_id, terminal_result_root=args.terminal_result_root,
        projection_path=args.projection, webapp_sync_path=args.webapp_sync,
        post_teardown_provider_zero_path=args.post_teardown_provider_zero,
        launch_request_path=args.launch_request, launch_profile_path=args.launch_profile,
        result_uri=args.result_uri, result_size_bytes=args.result_size_bytes)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
