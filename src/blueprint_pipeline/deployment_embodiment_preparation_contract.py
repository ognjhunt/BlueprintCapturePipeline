"""One explicitly funded public robot asset batch; no task or policy authority."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .deployment_embodiment_assets import asset_recipes
from .deployment_embodiment_runtime_configuration import runtime_profiles

PROBE_KIND = "embodiment-asset-preparation"
BUNDLE_KIND = "embodiment_asset_preparation"
REQUEST_SCHEMA = "embodiment_asset_preparation_request.v1"
INPUT_SCHEMA = "embodiment_asset_preparation_inputs.v1"
RESULT_SCHEMA = "embodiment_asset_preparation_result.v1"
RESULT_FILENAME = "embodiment_asset_preparation_result.v1.json"
INPUT_FILENAME = "embodiment_asset_preparation_inputs.v1.json"
MAX_OUTPUT_BYTES = 4 * 1024**3
MAX_SPEND_USD = 4.0
MAX_RATE_USD = .80
MAX_TTL_SECONDS = 14_400
INSTANCE_PREFIX = "blueprint-embodiment-preparation-"
STATE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/embodiment-preparation")
WORK_ROOT = Path("/mnt/blueprint-work/embodiment-preparation")
MODEL_ROOT = Path("/var/lib/blueprint/robot-models")
REQUEST_FIELDS = {"schema_version", "authorization_id", "release_commit", "embodiment_ids", "max_spend_usd",
                  "max_hourly_rate_usd", "ttl_seconds", "expires_at_epoch", "authorization_reference", "request_digest"}


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise ValueError("embodiment_preparation_" + reason)


def sealed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = canonical_digest(result, digest_field=key)
    return result


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path, *, limit: int = 8 * 1024**2) -> dict[str, Any]:
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= limit, "record_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "record_invalid")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a durable complete record; a matching retry changes no bytes."""
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        require(not path.is_symlink() and path.read_bytes() == payload, "existing_record_conflict")
        return
    temporary = path.with_name("." + path.name + "." + os.urandom(8).hex())
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_request(value: Mapping[str, Any], *, now: float | None = None,
                     allow_expired: bool = False) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request_schema_invalid")
    request = dict(value)
    require(set(request) == REQUEST_FIELDS and request.get("schema_version") == REQUEST_SCHEMA, "request_schema_invalid")
    require(isinstance(request["authorization_id"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", request["authorization_id"]), "authorization_id_invalid")
    require(isinstance(request["release_commit"], str)
            and re.fullmatch(r"[0-9a-f]{40}", request["release_commit"]), "release_commit_invalid")
    ids = request["embodiment_ids"]
    profiles = {row["embodiment_id"] for row in runtime_profiles()["profiles"]}
    available = {row["embodiment_id"] for row in asset_recipes()["recipes"] if row["status"] == "source_available"}
    require(isinstance(ids, list) and 1 <= len(ids) <= 24 and all(isinstance(x, str) for x in ids)
            and len(set(ids)) == len(ids) and set(ids) <= profiles & available, "catalog_selection_invalid")
    rate, cap, ttl, expires = (request[k] for k in ("max_hourly_rate_usd", "max_spend_usd", "ttl_seconds", "expires_at_epoch"))
    require(all(type(x) is int or (type(x) is float and math.isfinite(x)) for x in (rate, cap, expires))
            and 0 < rate <= MAX_RATE_USD and 0 < cap <= MAX_SPEND_USD
            and type(ttl) is int and 1800 <= ttl <= MAX_TTL_SECONDS and rate * ttl / 3600 <= cap,
            "budget_invalid")
    current = time.time() if now is None else now
    require(allow_expired or current + ttl + 300 <= expires <= current + 7 * 86400, "authorization_expiry_invalid")
    reference = request["authorization_reference"]
    require(isinstance(reference, str) and 1 <= len(reference) <= 300 and reference.isprintable(), "authorization_reference_invalid")
    require(request["request_digest"] == canonical_digest(request, digest_field="request_digest"), "request_digest_invalid")
    return request


def resource_name(request: Mapping[str, Any]) -> str:
    return INSTANCE_PREFIX + hashlib.sha256(request["authorization_id"].encode()).hexdigest()[:32]


def validate_preparation_result(value: Mapping[str, Any], *, expected_inputs_digest: str | None = None,
                                expected_ids: list[str] | None = None, output_root: Path | None = None) -> dict:
    result = dict(value)
    require(result.get("schema_version") == RESULT_SCHEMA and result.get("status") == "completed"
            and result.get("claim_ceiling") == "robot_asset_import_only"
            and result.get("candidate_policy_queried") is False and result.get("task_executed") is False
            and result.get("provider_teardown_verified") is False
            and result.get("result_digest") == canonical_digest(result, digest_field="result_digest"), "result_invalid")
    require(expected_inputs_digest is None or result.get("inputs_digest") == expected_inputs_digest, "result_inputs_changed")
    rows = result.get("results")
    require(isinstance(rows, list) and 1 <= len(rows) <= 24 and all(isinstance(r, dict) for r in rows), "result_rows_invalid")
    ids = [row.get("embodiment_id") for row in rows]
    require(all(isinstance(x, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", x) for x in ids)
            and len(set(ids)) == len(ids) and (expected_ids is None or ids == expected_ids), "result_selection_changed")
    require(all(row.get("status") in {"imported", "import_failed", "not_imported", "not_retained"} for row in rows),
            "result_status_invalid")
    files = result.get("files")
    require(isinstance(files, list) and len(files) <= 100_000, "output_inventory_invalid")
    seen, total = set(), 0
    for row in files:
        require(isinstance(row, dict) and set(row) == {"path", "size_bytes", "sha256"}, "output_row_invalid")
        require(isinstance(row["path"], str) and isinstance(row["sha256"], str), "output_path_invalid")
        path = Path(row["path"])
        require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and str(path) == row["path"]
                and row["path"] not in seen and row["path"] != RESULT_FILENAME
                and type(row["size_bytes"]) is int and row["size_bytes"] >= 0
                and re.fullmatch(r"sha256:[0-9a-f]{64}", row["sha256"]), "output_path_invalid")
        if path.parts[0] == "bodies":
            require(len(path.parts) >= 3 and path.parts[1] in ids, "unexpected_body_output")
        seen.add(row["path"])
        total += row["size_bytes"]
        if output_root is not None:
            actual = output_root / path
            require(not any(p.is_symlink() for p in (actual, *actual.parents)) and actual.is_file()
                    and actual.stat().st_size == row["size_bytes"] and sha(actual) == row["sha256"], "output_bytes_changed")
    require(total <= MAX_OUTPUT_BYTES, "output_bound_exceeded")
    if output_root is not None:
        actual = {p.relative_to(output_root).as_posix() for p in output_root.rglob("*")
                  if p.is_file() and p != output_root / RESULT_FILENAME}
        require(actual == seen, "output_inventory_changed")
    return result
