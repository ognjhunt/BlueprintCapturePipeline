"""Validate retained census proposals without inspecting or mutating targets."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import secrets
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from .control_plane_lane_scratch import (
    MAX_TTL_SECONDS, LaneScratchError, _creation_lease, _id, _REASON,
)

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ROWS = 10_000
MAX_PATH_BYTES = 4096
MAX_PATH_COMPONENTS = 64
INVENTORY_SCHEMA = "control_plane_lane_scratch_census.v1"
ANNOTATIONS_SCHEMA = "control_plane_lane_scratch_annotations.v1"
VALIDATION_SCHEMA = "control_plane_lane_scratch_decision_validation.v1"
ACTIONS = ("keep", "register", "offload", "delete")


class CensusDecisionError(ValueError):
    """A retained input or proposal failed validation; only a typed code is public."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _refuse(code: str) -> None:
    raise CensusDecisionError(code)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def encode_validation_report(report: dict[str, Any]) -> bytes:
    """One deterministic encoding for stdout and retained validation artifacts."""
    return (json.dumps(report, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _bound(value: int) -> int:
    if type(value) is not int or not 0 < value <= MAX_JSON_BYTES:
        _refuse("census_input_too_large")
    return value


@contextmanager
def _opened_parent(path: Path):
    if ".." in Path(path).parts:
        _refuse("census_input_unsafe")
    absolute = Path(os.path.abspath(path))
    with ExitStack() as descriptors:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.callback(os.close, directory)
        for component in absolute.parts[1:-1]:
            directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
            descriptors.callback(os.close, directory)
        yield directory, absolute.name


def read_census_input_record(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> tuple[bytes, tuple[int, int]]:
    """Read bounded input bytes and retain their inode identity for artifact alias checks."""
    limit = _bound(max_bytes)
    try:
        with _opened_parent(path) as (directory, name):
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    _refuse("census_input_unsafe")
                if before.st_size > limit:
                    _refuse("census_input_too_large")
                pieces = []
                remaining = limit + 1
                while remaining:
                    part = os.read(descriptor, min(65536, remaining))
                    if not part:
                        break
                    pieces.append(part)
                    remaining -= len(part)
                payload = b"".join(pieces)
                after = os.fstat(descriptor)
                if len(payload) > limit:
                    _refuse("census_input_too_large")

                def identity(info):
                    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

                if len(payload) != before.st_size or identity(before) != identity(after):
                    _refuse("census_input_unsafe")
                return payload, (before.st_dev, before.st_ino)
            finally:
                os.close(descriptor)
    except CensusDecisionError:
        raise
    except (TypeError, ValueError) as exc:
        raise CensusDecisionError("census_input_unsafe") from exc
    except OSError as exc:
        code = "census_input_unsafe" if exc.errno in (errno.ELOOP, errno.ENOTDIR) else "census_input_unreadable"
        raise CensusDecisionError(code) from exc


def read_census_input(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    """Read a bounded regular input through no-follow ancestor descriptors."""
    return read_census_input_record(path, max_bytes=max_bytes)[0]


def write_census_validation_report(path: Path, payload: bytes, *, input_paths, input_identities) -> None:
    """Publish only a report, using one retained parent for alias checking and writing."""
    if len(payload) > MAX_JSON_BYTES:
        _refuse("census_validation_output_too_large")
    if os.path.abspath(path) in {os.path.abspath(source) for source in input_paths}:
        _refuse("census_input_unsafe")
    try:
        with _opened_parent(path) as (directory, name):
            try:
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None and (not stat.S_ISREG(current.st_mode)
                                       or (current.st_dev, current.st_ino) in input_identities):
                _refuse("census_input_unsafe")
            temporary = f".{name}.{secrets.token_hex(8)}.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
    except (OSError, TypeError, ValueError) as exc:
        raise CensusDecisionError("census_input_unsafe") from exc


def _document(payload: bytes, limit: int) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        _refuse("census_json_invalid")
    if len(payload) > limit:
        _refuse("census_input_too_large")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=nonfinite)
        if not isinstance(value, dict):
            _refuse("census_json_invalid")
        encode_validation_report(value)  # Reject overflowing floats and invalid Unicode too.
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CensusDecisionError("census_json_invalid") from exc


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _path(value: Any) -> PurePosixPath:
    try:
        byte_count = len(value.encode("utf-8")) if isinstance(value, str) else 0
    except UnicodeError as exc:
        raise CensusDecisionError("census_row_ambiguous") from exc
    if (not isinstance(value, str) or not value.startswith("/") or value.startswith("//")
            or "<redacted>" in value or "\\" in value
            or byte_count > MAX_PATH_BYTES
            or value.count("/") > MAX_PATH_COMPONENTS
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        _refuse("census_row_ambiguous")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value:
        _refuse("census_row_ambiguous")
    return path



def _counter(value: Any) -> bool:
    return type(value) is int and value >= 0


def _inventory_rows(census: dict, roots: tuple[PurePosixPath, ...]) -> dict[str, dict]:
    fields = {"schema_version", "status", "observed_at_epoch", "rows", "candidate_count",
              "entries_visited", "unique_allocated_bytes", "scan_errors", "mutations"}
    if set(census) != fields or census["schema_version"] != INVENTORY_SCHEMA:
        _refuse("census_inventory_invalid")
    if census["status"] != "complete" or census["scan_errors"] != []:
        _refuse("census_inventory_incomplete")
    if (not _number(census["observed_at_epoch"])
            or any(not _counter(census[key]) for key in
                   ("candidate_count", "entries_visited", "unique_allocated_bytes", "mutations"))
            or census["mutations"] != 0):
        _refuse("census_inventory_invalid")
    rows = census["rows"]
    if (not isinstance(rows, list) or len(rows) > MAX_ROWS
            or census["candidate_count"] != len(rows)):
        _refuse("census_inventory_invalid")
    row_fields = {"path", "family", "owner_guess", "owner_guess_basis", "allocated_bytes",
                  "newest_mtime_epoch", "age_seconds", "unreadable", "shared_names",
                  "references", "owner_decision", "approved_expiry"}
    indexed: dict[str, dict] = {}
    known_refs = {"process", "queue", "pin", "live_release", "active_run"}
    for row in rows:
        if isinstance(row, dict) and "references" not in row:
            _refuse("census_reference_invalid")
        if not isinstance(row, dict) or not row_fields <= set(row):
            _refuse("census_inventory_invalid")
        if (any(not _counter(row[key]) for key in ("allocated_bytes", "shared_names", "unreadable"))
                or any(row[key] is not None and not _number(row[key]) for key in
                       ("newest_mtime_epoch", "age_seconds"))
                or row["owner_decision"] is not None or row["approved_expiry"] is not None
                or any(not isinstance(row[key], str) for key in
                       ("family", "owner_guess", "owner_guess_basis"))):
            _refuse("census_inventory_invalid")
        if row["unreadable"] != 0:
            _refuse("census_inventory_incomplete")
        path = _path(row["path"])
        if path in roots or not any(root in path.parents for root in roots) or str(path) in indexed:
            _refuse("census_row_ambiguous")
        refs = row["references"]
        if (not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in known_refs for ref in refs)
                or len(set(refs)) != len(refs)):
            _refuse("census_reference_invalid")
        indexed[str(path)] = row
    for path in indexed:
        if any(str(parent) in indexed for parent in PurePosixPath(path).parents):
            _refuse("census_row_ambiguous")
    if sum(row["allocated_bytes"] for row in rows) != census["unique_allocated_bytes"]:
        _refuse("census_inventory_invalid")
    return indexed


def _decision_metadata(decision: dict, roots: tuple[PurePosixPath, ...], now: float) -> None:
    action = decision["action"]
    base = {"path", "action", "owner"}
    try:
        _id(decision.get("owner"), "owner")
        if action == "keep":
            expiry = decision.get("expires_at_epoch")
            if (set(decision) != base | {"expires_at_epoch"} or not _number(expiry)
                    or not 0 < expiry - now <= MAX_TTL_SECONDS):
                _refuse("census_decision_metadata_invalid")
        elif action in ("delete", "offload"):
            reason = decision.get("reason")
            if (set(decision) != base | {"reason"} or not isinstance(reason, str)
                    or _REASON.fullmatch(reason) is None):
                _refuse("census_decision_metadata_invalid")
        else:
            required = {"lane", "name", "reason", "class_intent", "cleanup", "ttl_seconds"}
            optional = {"run_ref", "scene_ref", "size_budget_bytes"}
            if (not base | required <= set(decision) or set(decision) - base - required - optional
                    or ("run_ref" in decision) == ("scene_ref" in decision)
                    or decision.get("run_ref", decision.get("scene_ref")) is None
                    or ("size_budget_bytes" in decision and decision["size_budget_bytes"] is None)):
                _refuse("census_decision_metadata_invalid")
            metadata = {key: value for key, value in decision.items() if key not in base}
            _creation_lease(owner=decision["owner"], now=lambda: now, **metadata)
            path = PurePosixPath(decision["path"])
            for root in roots:
                if root not in path.parents:
                    continue
                relative = path.relative_to(root).parts
                if (len(relative) == 3 and relative[0] == "lanes"
                        and relative[1:] != (decision["lane"], decision["name"])):
                    _refuse("census_decision_metadata_invalid")
    except (LaneScratchError, TypeError, ValueError, OverflowError) as exc:
        raise CensusDecisionError("census_decision_metadata_invalid") from exc


def validate_census_annotations(
    census_bytes: bytes, annotation_bytes: bytes, *, now: float, allowed_roots,
    max_input_bytes: int = MAX_JSON_BYTES, max_output_bytes: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    """Validate byte-bound retained proposals; never stat, resolve or operate on targets."""
    census = _document(census_bytes, _bound(max_input_bytes))
    annotations = _document(annotation_bytes, _bound(max_input_bytes))
    output_limit = _bound(max_output_bytes)
    if not _number(now):
        _refuse("census_annotations_invalid")
    try:
        if isinstance(allowed_roots, (str, bytes)):
            _refuse("census_inventory_invalid")
        roots = tuple(_path(str(root)) if isinstance(root, (str, PurePosixPath)) else
                      _path(None) for root in allowed_roots)
    except TypeError as exc:
        raise CensusDecisionError("census_inventory_invalid") from exc
    if not roots:
        _refuse("census_inventory_invalid")
    rows = _inventory_rows(census, roots)
    if (annotations.get("schema_version") != ANNOTATIONS_SCHEMA
            or set(annotations) != {"schema_version", "census_digest", "decisions"}):
        _refuse("census_annotations_invalid")
    digest = _digest(census_bytes)
    if annotations["census_digest"] != digest:
        _refuse("census_identity_mismatch")
    decisions = annotations["decisions"]
    if not isinstance(decisions, list) or len(decisions) > MAX_ROWS:
        _refuse("census_annotations_invalid")
    validated = {}
    counts = dict.fromkeys(ACTIONS, 0)
    for decision in decisions:
        if not isinstance(decision, dict):
            _refuse("census_annotations_invalid")
        path = decision.get("path")
        if not isinstance(path, str) or path not in rows:
            _refuse("census_decision_unknown_target")
        if path in validated:
            _refuse("census_decision_duplicate")
        action = decision.get("action")
        if not isinstance(action, str) or action not in ACTIONS:
            _refuse("census_decision_action_invalid")
        _decision_metadata(decision, roots, now)
        refs = rows[path]["references"]
        if refs and action in ("delete", "offload"):
            _refuse("census_decision_referenced")
        validated[path] = decision | {"references": list(refs)}
        counts[action] += 1
    if len(validated) != len(rows):
        _refuse("census_decision_missing")
    report = {"schema_version": VALIDATION_SCHEMA, "status": "validated",
              "census_digest": digest, "annotations_digest": _digest(annotation_bytes),
              "decision_count": len(validated), "decision_counts": counts,
              "decisions": [validated[path] for path in sorted(validated)],
              "mutations": 0, "execution_authorized": False,
              "requires_fresh_reference_check": True}
    if len(encode_validation_report(report)) > output_limit:
        _refuse("census_validation_output_too_large")
    return report
