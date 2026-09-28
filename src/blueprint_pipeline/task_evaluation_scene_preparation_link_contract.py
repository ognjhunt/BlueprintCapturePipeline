"""Pure historical preparation-link contract without runtime admission imports.

The controls worker re-exports this existing contract. Validation is lexical and
in-memory only; a valid link does not grant current execution authority.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

LINK_SCHEMA = "task_evaluation_scene_preparation_link.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("controls_autoprovision_" + code)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def validate_preparation_link(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(value) - {"scene_configuration_attempt"} == {"schema_version", "intent_id", "intent_digest", "preparation_id",
        "request_digest", "expected_production_commit", "team_namespace", "scene_id", "task_id",
        "result_filename", "link_digest"}, "link_fields_invalid")
    _require(value["schema_version"] == LINK_SCHEMA and value["link_digest"] ==
             canonical_digest(value, digest_field="link_digest"), "link_digest_invalid")
    for key in ("intent_id", "preparation_id", "team_namespace", "scene_id", "task_id"):
        _require(_identifier(value[key]), "link_identity_invalid")
    for key in ("intent_digest", "request_digest"):
        _require(isinstance(value[key], str) and _DIGEST.fullmatch(value[key]) is not None,
                 "link_identity_invalid")
    _require(isinstance(value["expected_production_commit"], str) and
             _COMMIT.fullmatch(value["expected_production_commit"]) is not None,
             "link_release_invalid")
    expected = value["preparation_id"] + "-" + value["request_digest"].removeprefix("sha256:") + ".json"
    _require(value["result_filename"] == expected, "link_filename_invalid")
    if "scene_configuration_attempt" in value:
        reference = value["scene_configuration_attempt"]
        _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256", "size_bytes"}
                 and isinstance(reference.get("path"), str) and Path(reference["path"]).is_absolute()
                 and isinstance(reference.get("sha256"), str) and _DIGEST.fullmatch(reference["sha256"]) is not None
                 and type(reference.get("size_bytes")) is int and reference["size_bytes"] > 0,
                 "link_configuration_attempt_invalid")
    return dict(value)

