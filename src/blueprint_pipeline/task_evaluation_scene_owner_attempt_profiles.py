"""Carry persisted owner reservations into immutable non-policy profiles.

These adapters allocate no providers. They reserve the main configuration
attempt, retain controls' already reserved phase identities, and reopen live
consent at profile creation or immediately before a fresh paid child call.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_intake as intake
from .task_evaluation_scene_execution_authority import (
    OWNER_FIELDS, bind_scene_attempt, require_scene_execution_authority,
)

RECORD_SCHEMA = "task_evaluation_scene_owner_attempt.v1"


class SceneOwnerAttemptProfileError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneOwnerAttemptProfileError("scene_owner_profile_" + code)


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _require(source.is_absolute() and not any(p.is_symlink() for p in (source, *source.parents)), "path_unsafe")
    value = json.loads(source.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    if path.exists():
        _require(_read(path) == value, "immutable_record_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    intake.write_exclusive(path, value)


def _root(queue_root: str | Path | None = None) -> Path:
    value = str(queue_root or os.getenv(intake.ROOT_ENV, ""))
    _require(bool(value) and Path(value).is_absolute(), "owner_store_missing")
    root = Path(value)
    _require(not any(p.is_symlink() for p in (root, *root.parents)), "owner_store_unsafe")
    return root


def owner_reference_present(value: Mapping[str, Any]) -> bool:
    return any(str(value.get(key) or "").startswith("scene-intent:")
               for key in ("reference", "authority_reference", "authorization_reference"))


def make_owner_attempt_record(*, owner_fields: Mapping[str, Any], phase: str,
                             team_namespace: str, scene_id: str, task_id: str,
                             runtime_source_bundle_digest: str) -> dict[str, Any]:
    _require(set(owner_fields) == OWNER_FIELDS, "owner_fields_missing")
    value = {"schema_version": RECORD_SCHEMA, **dict(owner_fields), "phase": phase,
             "team_namespace": team_namespace, "scene_id": scene_id, "task_id": task_id,
             "runtime_source_bundle_digest": runtime_source_bundle_digest}
    value["owner_attempt_digest"] = canonical_digest(value, digest_field="owner_attempt_digest")
    return value


def validate_owner_attempt_record(value: Mapping[str, Any], *, phase: str, source_commit: str,
        scene_id: str, task_id: str, maximum_spend_usd: float, provider: str = "vast",
        team_namespace: str | None = None, runtime_source_bundle_digest: str | None = None,
        queue_root: str | Path | None = None, reopen_records: bool = True, now: float | None = None) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == OWNER_FIELDS | {
        "schema_version", "phase", "team_namespace", "scene_id", "task_id",
        "runtime_source_bundle_digest", "owner_attempt_digest"} and
        value.get("schema_version") == RECORD_SCHEMA and
        value.get("owner_attempt_digest") == canonical_digest(value, digest_field="owner_attempt_digest"),
        "owner_record_invalid")
    _require(value["phase"] == phase and value["scene_id"] == scene_id and value["task_id"] == task_id
             and (team_namespace is None or value["team_namespace"] == team_namespace), "scope_mismatch")
    bundle_digest = value.get("runtime_source_bundle_digest")
    _require(isinstance(bundle_digest, str) and intake._DIGEST.fullmatch(bundle_digest) is not None
             and (runtime_source_bundle_digest is None or bundle_digest == runtime_source_bundle_digest),
             "runtime_bundle_mismatch")
    fields = {key: value[key] for key in OWNER_FIELDS}
    require_scene_execution_authority(fields, source_commit=source_commit,
        maximum_spend_usd=maximum_spend_usd, provider=provider,
        queue_root=queue_root, reopen_records=reopen_records, now=now)
    if reopen_records:
        root = _root(queue_root)
        owner = intake._read(root / fields["scene_attempt_binding"]["intent_id"] / "intent.json", "intent_digest")
        _require(owner["request"]["task"]["task_id"] == task_id, "owner_task_mismatch")
    return fields


def profile_owner_fields(*, path: str | Path | None, authority: Mapping[str, Any],
        phase: str, source_commit: str, scene_id: str, task_id: str,
        maximum_spend_usd: float, provider: str = "vast", team_namespace: str | None = None) -> dict[str, Any]:
    if path is None:
        _require(not owner_reference_present(authority) and not OWNER_FIELDS.intersection(authority),
                 "owner_attempt_path_missing")
        return {}
    return validate_owner_attempt_record(_read(path), phase=phase, source_commit=source_commit,
        scene_id=scene_id, task_id=task_id, maximum_spend_usd=maximum_spend_usd,
        provider=provider, team_namespace=team_namespace)


def retain_native_owner_attempt(*, activation_request: Mapping[str, Any],
        preparation_request: Mapping[str, Any], output_path: Path) -> str | None:
    authorization = activation_request["authorization"]
    record = authorization.get("scene_owner_attempt")
    if record is None:
        _require(not owner_reference_present(authorization) and
                 "scene_intent_digest" not in preparation_request, "native_owner_attempt_missing")
        return None
    lane = str(activation_request["lane"])
    phase = "controls" if lane.startswith("native_task_arena_controls") else "construction"
    _require(lane in {"native_task_arena_construction", "native_task_arena_construction_after_destination",
                     "native_task_arena_controls"}, "native_phase_unsupported")
    spend = preparation_request["spend"]
    validate_owner_attempt_record(record, phase=phase,
        source_commit=activation_request["expected_production_commit"],
        scene_id=preparation_request["scene"]["identity"]["id"],
        task_id=preparation_request["task"]["identity"]["id"],
        team_namespace=activation_request["team_namespace"],
        maximum_spend_usd=spend["hard_cap_usd"], provider=spend["selected_provider"],
        runtime_source_bundle_digest=preparation_request["execution_adapter"]["runtime_source_bundle"]["digest"])
    if "scene_intent_digest" in preparation_request:
        _require(preparation_request["scene_intent_digest"] == record["scene_intent_digest"], "native_owner_mismatch")
    _write_once(output_path, record)
    return str(output_path)


def reserve_configuration_owner_attempt(*, activation_request: Mapping[str, Any],
        preparation_request: Mapping[str, Any], output_path: Path) -> str | None:
    digest = preparation_request.get("scene_intent_digest")
    if digest is None:
        _require("scene_intent_digest" not in preparation_request and not
                 owner_reference_present(activation_request["authorization"]), "configuration_owner_missing")
        return None
    from .task_evaluation_controls_autoprovision import validate_preparation_link
    root = _root()
    request_digest = canonical_digest(preparation_request)
    owner = None
    for path in root.glob("scene-*/intent.json"):
        _read(path)
        candidate = intake._read(path, "intent_digest")
        if candidate["intent_digest"] == digest:
            owner, directory = candidate, path.parent
            break
    _require(owner is not None, "configuration_owner_unknown")
    candidates = (directory / "preparations" / (request_digest.removeprefix("sha256:") + ".json"),
                  directory / "preparation-link.json")
    link_path = next((path for path in candidates if path.is_file()), None)
    _require(link_path is not None, "preparation_link_missing")
    link = validate_preparation_link(_read(link_path))
    identity = {"intent_id": owner["intent_id"], "intent_digest": digest,
        "request_digest": request_digest, "preparation_id": preparation_request["preparation_id"],
        "expected_production_commit": activation_request["expected_production_commit"],
        "team_namespace": activation_request["team_namespace"],
        "scene_id": preparation_request["scene"]["identity"]["id"],
        "task_id": preparation_request["task"]["identity"]["id"]}
    _require(all(link.get(k) == v for k, v in identity.items()) and
             preparation_request["expected_production_commit"] == identity["expected_production_commit"],
             "preparation_link_mismatch")
    runtime = preparation_request["execution_adapter"]["runtime_source_bundle"]["digest"]
    spend = preparation_request["spend"]
    _require(spend["selected_provider"] == "vast", "configuration_provider_mismatch")
    attempt_id = "scene-configuration-" + request_digest[7:31]
    attempt = intake.reserve_scene_attempt(queue_root=root, intent_id=owner["intent_id"],
        attempt_id=attempt_id, source_commit=identity["expected_production_commit"],
        runtime_digest=runtime, input_digest=request_digest, provider="vast", maximum_spend_usd=spend["hard_cap_usd"])
    reference = link.get("scene_configuration_attempt")
    if reference is not None:
        path = directory / "attempts" / (attempt_id + ".json")
        _require(reference["path"] == str(path) and reference["size_bytes"] == path.stat().st_size and
                 reference["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                 "configuration_attempt_reference_mismatch")
    record = make_owner_attempt_record(owner_fields=bind_scene_attempt(attempt), phase="scene_configuration",
        team_namespace=identity["team_namespace"], scene_id=identity["scene_id"], task_id=identity["task_id"],
        runtime_source_bundle_digest=runtime)
    validate_owner_attempt_record(record, phase="scene_configuration", source_commit=identity["expected_production_commit"],
        scene_id=identity["scene_id"], task_id=identity["task_id"], maximum_spend_usd=spend["hard_cap_usd"],
        team_namespace=identity["team_namespace"], runtime_source_bundle_digest=runtime, queue_root=root)
    _write_once(output_path, record)
    return str(output_path)


def require_fresh_task_owner(task: Mapping[str, Any], *, source_commit: str,
        maximum_spend_usd: float, output_path: Path | None = None) -> dict[str, Any]:
    """Only fresh allocation uses this guard; retained billing/teardown do not."""
    from .task_evaluation_scene_owner_authority import validate_task_scene_owner
    from .task_evaluation_scene_configuration_submission_inputs import checked_file
    authority = task.get("scene_intent_authority")
    if authority is None:
        _require(not OWNER_FIELDS.intersection(task) and
                 not owner_reference_present(task.get("human_authority") or {}), "task_owner_binding_missing")
        return {}
    owner = validate_task_scene_owner(task)
    _require(task.get("expected_production_commit") == source_commit, "task_release_mismatch")
    reference = authority.get("attempt")
    _require(isinstance(reference, Mapping), "task_attempt_missing")
    path = checked_file(reference["path"], reference)
    _require(Path(path).parent == _root() / owner["intent_id"] / "attempts", "task_attempt_path_invalid")
    attempt = intake._read(Path(path), "attempt_digest")
    fields = bind_scene_attempt(attempt)
    _require(fields["scene_intent_digest"] == owner["intent_digest"], "task_owner_mismatch")
    require_scene_execution_authority(fields, source_commit=source_commit,
        maximum_spend_usd=maximum_spend_usd, provider="vast")
    if output_path is not None:
        _write_once(output_path, fields)
    return fields


def record_owner_attempt(
    operations: dict[str, Any],
    mode: str,
    activation_request: Mapping[str, Any],
    preparation_request: Mapping[str, Any],
    activation_root: Path,
) -> None:
    """Retain the native ("native") or configuration ("config") owner attempt into ``operations``."""
    retain = retain_native_owner_attempt if mode == "native" else reserve_configuration_owner_attempt
    attempt = retain(
        activation_request=activation_request,
        preparation_request=preparation_request,
        output_path=Path(activation_root) / "scene_owner_attempt.json",
    )
    if attempt is not None:
        operations["scene_owner_attempt"] = attempt
