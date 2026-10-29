"""Operator-owned, data-defined candidates for a site's diagnostic screening.

Beta journey: a prepared job can compare an arbitrary set of configurations.
This catalog describes identities and requirements, never execution authority.
The planner separately requires a matching validated runtime binding and a
bounded cost quote. The ADP pair registry and its selection rules are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .decision_evidence_contracts import canonical_digest

CATALOG_SCHEMA_VERSION = "beta_policy_candidate_catalog.v1"
CATALOG_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_CATALOG_PATH"
MAX_CATALOG_BYTES = 8 * 1024 * 1024
ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,191}$"
CANDIDATE_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$"
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_DIGEST = {"type": "string", "pattern": DIGEST_PATTERN}
_OPTIONAL_DIGEST = {"anyOf": [_DIGEST, {"type": "null"}]}
_ID = {"type": "string", "pattern": ID_PATTERN}
_TEXT = {"type": "string", "minLength": 1, "maxLength": 2048}


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required if required is not None else list(properties)}


POLICY_IDENTITY_SCHEMA = _object({
    "kind": {"enum": ["checkpoint", "hosted"]},
    "source": _TEXT,
    "revision": {"type": ["string", "null"], "maxLength": 2048},
    "digest": _OPTIONAL_DIGEST,
    "digest_kind": {"enum": ["downloaded_snapshot_inventory", "hub_tree_listing",
                               "provider_metadata", "unverified"]},
})
INTERFACE_SCHEMA = _object({
    "policy_interface_id": _ID,
    "robot_profile_id": _ID,
    "observation_profile_id": _ID,
    "action_schema_id": _ID,
    "control_hz": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
    "action_horizon": {"type": "integer", "minimum": 1, "maximum": 10000},
    "execution_horizon": {"type": "integer", "minimum": 1, "maximum": 10000},
    "camera_roles": {"type": "array", "minItems": 1, "uniqueItems": True, "items": _ID},
    "image_height": {"type": "integer", "minimum": 1, "maximum": 16384},
    "image_width": {"type": "integer", "minimum": 1, "maximum": 16384},
    "image_encoding": {"enum": ["rgb_uint8", "rgb_float32", "jpeg", "png"]},
    "image_preprocessing": _TEXT,
    "history_frames": {"type": "integer", "minimum": 1, "maximum": 10000},
    "normalization_digest": _OPTIONAL_DIGEST,
})
# Native families retain the served controller and bounds alongside the shared
# interface fields. Existing DROID catalog entries keep their frozen shape.
INTERFACE_SCHEMA["properties"].update({"controller": _ID, "action_bounds": {
    "type": "array", "minItems": 1, "items": {"type": "array", "minItems": 2, "maxItems": 2,
                                                "items": {"type": "number"}}}})
RUNTIME_SCHEMA = _object({
    "backend": _ID,
    "runtime_id": _ID,
    "minimum_gpu_memory_gb": {"type": ["number", "null"], "exclusiveMinimum": 0},
})
ELIGIBILITY_SCHEMA = _object({
    "enabled": {"type": "boolean"},
    "commercial_use": {"enum": ["allowed", "allowed_with_conditions", "under_review", "not_allowed"]},
    "license_id": _TEXT,
    "license_note": _TEXT,
    "requires_credentials": {"type": "array", "uniqueItems": True, "maxItems": 16,
                             "items": {"type": "string", "pattern": r"^[A-Z][A-Z0-9_]{0,127}$"}},
    "blockers": {"type": "array", "uniqueItems": True, "maxItems": 64,
                 "items": {"type": "string", "minLength": 1, "maxLength": 300}},
})
CANDIDATE_SCHEMA = _object({
    "candidate_id": {"type": "string", "pattern": CANDIDATE_ID_PATTERN},
    "display_name": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": r"^\S(?:[\s\S]*\S)?$"},
    "policy": POLICY_IDENTITY_SCHEMA,
    "interface": {"anyOf": [INTERFACE_SCHEMA, {"type": "null"}]},
    "runtime": {"anyOf": [RUNTIME_SCHEMA, {"type": "null"}]},
    "supported_task_kinds": {"type": "array", "uniqueItems": True, "items": _ID},
    "eligibility": ELIGIBILITY_SCHEMA,
    "configuration_digest": _DIGEST,
})
# Company/task relevance is descriptive. Only runtime bindings grant execution.
CANDIDATE_SCHEMA["properties"]["deployment"] = _object({
    "configuration_id": _ID,
    "policy_route_id": _ID,
    "companies": {"type": "array", "items": _TEXT, "uniqueItems": True},
    "relationship": {"enum": ["publisher_demonstrated", "company_deployment", "public_controller_integration",
                                "planning_candidate"]},
    "evidence_url": _TEXT,
}, required=["configuration_id", "companies", "relationship", "evidence_url"])
CATALOG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": CATALOG_SCHEMA_VERSION,
    **_object({
        "schema_version": {"const": CATALOG_SCHEMA_VERSION},
        "catalog_id": _ID,
        "revision": _TEXT,
        "candidates": {"type": "array", "minItems": 1, "maxItems": 256, "items": CANDIDATE_SCHEMA},
        "catalog_digest": _DIGEST,
    }),
}


class BetaPolicyCatalogError(ValueError):
    """A catalog cannot safely describe executable candidate identities."""


def configuration_digest(candidate: Mapping[str, Any]) -> str:
    return canonical_digest(dict(candidate), digest_field="configuration_digest")


def seal_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal operator-authored data; sealing does not admit any execution."""
    if not isinstance(value, Mapping) or not isinstance(value.get("candidates"), list):
        raise BetaPolicyCatalogError("beta_catalog_candidates_missing")
    result = deepcopy(dict(value))
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, dict):
            raise BetaPolicyCatalogError("beta_catalog_candidate_invalid")
        candidate["configuration_digest"] = configuration_digest(candidate)
    result["catalog_digest"] = canonical_digest(result, digest_field="catalog_digest")
    return validate_catalog(result)


def validate_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise BetaPolicyCatalogError("beta_catalog_not_json") from exc
    errors = sorted(Draft202012Validator(CATALOG_SCHEMA).iter_errors(result), key=lambda row: str(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path)
        raise BetaPolicyCatalogError("beta_catalog_schema_invalid:" + path)
    ids = [candidate["candidate_id"] for candidate in result["candidates"]]
    if len(set(ids)) != len(ids):
        raise BetaPolicyCatalogError("beta_catalog_duplicate_candidate_id")
    for candidate in result["candidates"]:
        if candidate["configuration_digest"] != configuration_digest(candidate):
            raise BetaPolicyCatalogError("beta_catalog_candidate_digest_mismatch:" + candidate["candidate_id"])
        interface = candidate["interface"]
        if interface and interface["execution_horizon"] > interface["action_horizon"]:
            raise BetaPolicyCatalogError("beta_catalog_execution_horizon_exceeds_prediction:" + candidate["candidate_id"])
        if candidate["eligibility"]["enabled"] and (interface is None or candidate["runtime"] is None):
            raise BetaPolicyCatalogError("beta_catalog_enabled_interface_missing:" + candidate["candidate_id"])
        if not candidate["eligibility"]["enabled"] and not candidate["eligibility"]["blockers"]:
            raise BetaPolicyCatalogError("beta_catalog_disabled_reason_missing:" + candidate["candidate_id"])
        # Secrets are references only. URL userinfo is never an acceptable
        # checkpoint/service identity, even for a disabled catalog row.
        from urllib.parse import urlsplit
        source = urlsplit(candidate["policy"]["source"])
        if source.username is not None or source.password is not None:
            raise BetaPolicyCatalogError("beta_catalog_credential_in_source")
    if result["catalog_digest"] != canonical_digest(result, digest_field="catalog_digest"):
        raise BetaPolicyCatalogError("beta_catalog_digest_mismatch")
    return result


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise BetaPolicyCatalogError("beta_catalog_duplicate_json_key")
        result[key] = value
    return result


def _load_catalog_data(path: str | Path | None = None) -> dict[str, Any]:
    """Read only an explicit, locally operator-owned catalog; never customer JSON.

    Matching the existing team registry, root or the process owner can own the
    file. O_NOFOLLOW and fstat bind the ownership check to the bytes read.
    The deployment installer owns the parent-directory access policy.
    """
    raw = str(path) if path is not None else os.environ.get(CATALOG_ENV, "")
    catalog_path = Path(raw)
    if not raw or not catalog_path.is_absolute():
        raise BetaPolicyCatalogError("beta_catalog_path_required")
    try:
        descriptor = os.open(catalog_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            status = os.fstat(handle.fileno())
            if (not stat.S_ISREG(status.st_mode) or status.st_uid not in {0, os.getuid()}
                    or status.st_mode & 0o022 or status.st_size > MAX_CATALOG_BYTES):
                raise BetaPolicyCatalogError("beta_catalog_not_operator_owned")
            data = handle.read(MAX_CATALOG_BYTES + 1)
    except OSError as exc:
        raise BetaPolicyCatalogError("beta_catalog_unavailable") from exc
    if len(data) > MAX_CATALOG_BYTES:
        raise BetaPolicyCatalogError("beta_catalog_too_large")
    try:
        value = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetaPolicyCatalogError("beta_catalog_invalid_json") from exc
    if not isinstance(value, dict):
        raise BetaPolicyCatalogError("beta_catalog_not_mapping")
    return value


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    """Load an owned catalog and verify its schema and every identity seal."""
    return validate_catalog(_load_catalog_data(path))


def catalog_candidate(catalog: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    checked = validate_catalog(catalog)
    for candidate in checked["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise BetaPolicyCatalogError("beta_catalog_candidate_unknown:" + candidate_id)


def load_default_catalog() -> dict[str, Any]:
    """Load the installed override or the reviewed checkout's default catalog."""
    override = os.environ.get(CATALOG_ENV)
    path = Path(override) if override else Path(__file__).resolve().parents[2] / "config" / "beta_policy_candidates.v1.json"
    return load_catalog(path)


def main(argv: list[str] | None = None) -> int:
    """Operator catalog inspection and explicit non-overwriting reseal."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "inspect"):
        commands.add_parser(command).add_argument("--catalog", type=Path)
    sealing = commands.add_parser("seal")
    sealing.add_argument("--input", type=Path, required=True)
    sealing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            catalog = seal_catalog(_load_catalog_data(args.input))
            if not args.output.is_absolute():
                raise BetaPolicyCatalogError("beta_catalog_output_path_required")
            try:
                descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError as exc:
                raise BetaPolicyCatalogError("beta_catalog_output_exists") from exc
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(catalog, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        else:
            catalog = load_catalog(args.catalog) if args.catalog else load_default_catalog()
    except (BetaPolicyCatalogError, OSError) as exc:
        code = str(exc) if isinstance(exc, BetaPolicyCatalogError) else "beta_catalog_io_failed"
        print(json.dumps({"status": "blocked", "blocker": code}))
        return 2
    result: dict[str, Any] = {"status": "sealed" if args.command == "seal" else "valid",
              "catalog_id": catalog["catalog_id"], "revision": catalog["revision"],
              "catalog_digest": catalog["catalog_digest"], "candidate_count": len(catalog["candidates"])}
    if args.command == "inspect":
        result["candidates"] = [{"candidate_id": row["candidate_id"], "display_name": row["display_name"],
            "configuration_digest": row["configuration_digest"], "enabled": row["eligibility"]["enabled"],
            "commercial_use": row["eligibility"]["commercial_use"], "blockers": row["eligibility"]["blockers"],
            "interface": row["interface"], "runtime": row["runtime"]} for row in catalog["candidates"]]
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = ["CATALOG_ENV", "CATALOG_SCHEMA", "CATALOG_SCHEMA_VERSION", "BetaPolicyCatalogError",
           "catalog_candidate", "configuration_digest", "load_catalog", "load_default_catalog", "main", "seal_catalog", "validate_catalog"]


if __name__ == "__main__":
    raise SystemExit(main())
