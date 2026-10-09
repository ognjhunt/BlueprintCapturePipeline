"""Pure retained scene-intent contracts, shared by issuance and evidence readers.

ADP-050/day-28 and ADP-009D: reopen owner consent without importing execution
or mutation services. Issuance, reservations and recovery remain in intake.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as canonical_digest

TASK_STRATEGIES = ("pick_and_place", "articulated_open_close")
ARTICULATION_JOINT_TYPES = ("prismatic", "revolute")
REQUEST_SCHEMA = "task_evaluation_scene_intake_request.v1"
INTENT_SCHEMA = "task_evaluation_scene_intent.v1"
ATTEMPT_SCHEMA = "task_evaluation_scene_attempt.v1"
ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"
CLIENTS_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
#: The two frozen policy candidates this ADP-009D run actually supports end to
#: end (scene setup CANDIDATE_IDS, policy-canary handoff, dispatch). Intake
#: rejects any other pair up front instead of accepting it and failing late,
#: after construction spend, at the handoff (A10). Do not broaden this here.
SUPPORTED_POLICY_CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")


class SceneIntakeError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneIntakeError("scene_intake_" + code)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_digest(result, digest_field=field)
    return result


def validate_request(value: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    _require(set(value) == {"schema_version", "submission_id", "owner", "source", "task",
                            "execution", "consent"}, "request_fields_invalid")
    _require(value.get("schema_version") == REQUEST_SCHEMA, "schema_invalid")
    _require(_identifier(value.get("submission_id")), "submission_id_invalid")
    owner = value.get("owner")
    _require(isinstance(owner, Mapping) and set(owner) == {"user_id", "organization_id"}
             and all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._@-]{0,127}", v)
                     for v in owner.values()), "owner_invalid")
    source = value.get("source")
    _require(isinstance(source, Mapping) and set(source) - {"collision_mesh"} == {"kind", "binding_id", "content_digest"},
             "source_invalid")
    _require(source["kind"] in {"capture_bundle", "mesh", "gaussian_splat", "public_scene"}
             and _identifier(source["binding_id"])
             and isinstance(source["content_digest"], str)
             and _DIGEST.fullmatch(source["content_digest"]) is not None, "source_invalid")
    if "collision_mesh" in source:
        companion = source["collision_mesh"]
        _require(source["kind"] == "gaussian_splat" and isinstance(companion, Mapping)
                 and set(companion) == {"binding_id", "content_digest", "rights_reference", "frame_relation"}
                 and _identifier(companion.get("binding_id")) and companion["binding_id"] != source["binding_id"]
                 and isinstance(companion.get("content_digest"), str)
                 and _DIGEST.fullmatch(companion["content_digest"]) is not None
                 and isinstance(companion.get("rights_reference"), str)
                 and _DIGEST.fullmatch(companion["rights_reference"]) is not None
                 and companion.get("frame_relation") == "owner_declared_common_frame", "collision_mesh_binding_invalid")
    task = value.get("task")
    _require(isinstance(task, Mapping) and task.get("strategy") in TASK_STRATEGIES
             and _identifier(task.get("task_id")), "task_invalid")
    _require(type(task.get("reuse_completed_stages", True)) is bool, "task_reuse_mode_invalid")
    # A relocation binds a destination; an articulated open/close binds the
    # mechanism instead (the moving part never leaves its assembly).
    required = (("subject", "support", "articulation", "success")
                if task["strategy"] == "articulated_open_close"
                else ("subject", "support", "destination", "success"))
    for key in required:
        _require(isinstance(task.get(key), Mapping) and bool(task[key]), "task_" + key + "_missing")
    if task["strategy"] == "articulated_open_close":
        _require(task["articulation"].get("joint_type") in ARTICULATION_JOINT_TYPES
                 and isinstance(task["articulation"].get("part_label"), str)
                 and task["articulation"]["part_label"].strip() != "", "task_articulation_invalid")
    from .task_evaluation_scene_execution_scope import validate_execution
    validate_execution(value, now=now)
    consent = value.get("consent")
    _require(isinstance(consent, Mapping) and set(consent) == {
        "accepted_by", "accepted_at_epoch", "rights_reference", "provider_terms_reference",
        "private_processing_authorized", "provider_training_authorized", "task_confirmed",
        "spend_authorized"}, "consent_invalid")
    _require(consent["accepted_by"] == owner["user_id"]
             and _number(consent["accepted_at_epoch"])
             and now - 86400 <= consent["accepted_at_epoch"] <= now, "consent_actor_or_time_invalid")
    _require(all(isinstance(consent[k], str) and 1 <= len(consent[k]) <= 1000
                 for k in ("rights_reference", "provider_terms_reference")), "consent_references_missing")
    from .website_preparation_authority import preparation_consent_valid
    _require(consent["private_processing_authorized"] is True
             and consent["provider_training_authorized"] is False
             and (consent["task_confirmed"] is True or preparation_consent_valid(value))
             and consent["spend_authorized"] is True,
             "consent_missing")
    # Detach mutable caller state and reject non-JSON/NaN task values.
    try:
        detached = json.loads(json.dumps(value, allow_nan=False))
        canonical_digest(detached)
        return detached
    except (TypeError, ValueError) as exc:
        raise SceneIntakeError("scene_intake_json_invalid") from exc


def _read(path: Path, field: str) -> dict[str, Any]:
    _require(not path.is_symlink(), "record_unsafe")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SceneIntakeError("scene_intake_record_unreadable") from exc
    _require(isinstance(value, dict) and value.get(field) == canonical_digest(value, digest_field=field),
             "record_digest_invalid")
    return value
