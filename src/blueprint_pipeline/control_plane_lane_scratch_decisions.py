"""Validate retained census proposals without inspecting or mutating targets."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ROWS = 10_000
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


def read_census_input(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    """Read a bounded regular input through no-follow ancestor descriptors."""
    limit = _bound(max_bytes)
    absolute = Path(os.path.abspath(path))
    descriptors: list[int] = []
    try:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        for component in absolute.parts[1:-1]:
            directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
            descriptors.append(directory)
        descriptor = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        descriptors.append(descriptor)
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
        return payload
    except OSError as exc:
        code = "census_input_unsafe" if exc.errno in (errno.ELOOP, errno.ENOTDIR) else "census_input_unreadable"
        raise CensusDecisionError(code) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


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
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _path(value: Any) -> PurePosixPath:
    if (not isinstance(value, str) or not value.startswith("/") or value.startswith("//")
            or "<redacted>" in value or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        _refuse("census_row_ambiguous")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value:
        _refuse("census_row_ambiguous")
    return path


def validate_census_annotations(
    census_bytes: bytes, annotation_bytes: bytes, *, now: float, allowed_roots,
    max_input_bytes: int = MAX_JSON_BYTES, max_output_bytes: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    """Validate byte-bound retained proposals; never stat, resolve or operate on targets."""
    limit = _bound(max_input_bytes)
    output_limit = _bound(max_output_bytes)
    census = _document(census_bytes, limit)
    annotations = _document(annotation_bytes, limit)
    if not _number(now):
        _refuse("census_annotations_invalid")
    try:
        roots = tuple(_path(str(root)) for root in allowed_roots)
    except TypeError as exc:
        raise CensusDecisionError("census_inventory_invalid") from exc
    if not roots:
        _refuse("census_inventory_invalid")
    if census.get("schema_version") != INVENTORY_SCHEMA:
        _refuse("census_inventory_invalid")
    if census.get("status") != "complete" or census.get("scan_errors") != []:
        _refuse("census_inventory_incomplete")
    if (census.get("rows") != [] or census.get("candidate_count") != 0
            or census.get("mutations") != 0 or census.get("unique_allocated_bytes") != 0):
        _refuse("census_inventory_invalid")
    if (annotations.get("schema_version") != ANNOTATIONS_SCHEMA
            or set(annotations) != {"schema_version", "census_digest", "decisions"}):
        _refuse("census_annotations_invalid")
    digest = _digest(census_bytes)
    if annotations["census_digest"] != digest:
        _refuse("census_identity_mismatch")
    if annotations["decisions"] != []:
        _refuse("census_decision_unknown_target")
    report = {"schema_version": VALIDATION_SCHEMA, "status": "validated",
              "census_digest": digest, "annotations_digest": _digest(annotation_bytes),
              "decision_count": 0, "decision_counts": dict.fromkeys(ACTIONS, 0),
              "decisions": [], "mutations": 0, "execution_authorized": False,
              "requires_fresh_reference_check": True}
    if len(encode_validation_report(report)) > output_limit:
        _refuse("census_validation_output_too_large")
    return report
