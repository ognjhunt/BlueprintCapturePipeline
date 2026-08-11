"""Fail-closed contracts for agent-authored CAD replacement candidates.

This module deliberately has no geometry generator.  It binds requests and
outputs from admitted CAD-agent backends, while retaining deterministic tools
only for format conversion, measurement, and verification.  A scene may carry
one to five independent replacement-object slots, and each slot may compare
multiple agent backends without sharing a candidate or task-specific receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import (
    MAX_REPLACEMENT_OBJECTS,
    DualTaskRehearsalContractError,
    validate_task_freeze,
)


REQUEST_SCHEMA_VERSION = "simready_cad_agent_request.v1"
OUTPUT_SCHEMA_VERSION = "simready_cad_agent_output.v1"
MATRIX_SCHEMA_VERSION = "scene_replacement_cad_agent_matrix.v1"
REFERENCE_MANIFEST_SCHEMA_VERSION = "simready_cad_agent_reference_manifest.v1"
REFERENCE_BINDING_AUDIT_SCHEMA_VERSION = "cad_agent_reference_binding_audit.v1"
INSPECTION_SCHEMA_VERSION = "cad_agent_step_inspection.v1"
EXECUTION_SCHEMA_VERSION = "cad_agent_execution_receipt.v1"

ADMITTED_BACKENDS = {
    "earthtojake_text_to_cad": ("codex_skill_step_first",),
    "pan_chera_multi_agent_cad": (
        "codex_agent_direct_repo_route",
        "openai_compatible_api_aider_first",
    ),
}
BACKEND_REPOSITORIES = {
    "earthtojake_text_to_cad": "https://github.com/earthtojake/text-to-cad",
    "pan_chera_multi_agent_cad": "https://github.com/Pan-Chera/Multi-Agent-CAD",
}
FORBIDDEN_BACKEND_IDS = frozenset(
    {
        "deterministic_graph_to_cad",
        "graph_to_opencascade",
        "primitive_to_cad",
        "blueprint_deterministic_cad",
    }
)
CLAIM_BOUNDARY = {
    "agent_authored_cad_candidate": True,
    "observed_exterior_reference_only": True,
    "unseen_geometry_is_generated_candidate": True,
    "simready_qualified": False,
    "native_simulator_import_qualified": False,
    "physical_equivalence": False,
}


class SimReadyCadAgentContractError(ValueError):
    """Stable sorted failures for CAD-agent requests, outputs, and matrices."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise SimReadyCadAgentContractError(["cad_agent_payload_invalid"]) from exc
    if not isinstance(result, dict):
        raise SimReadyCadAgentContractError(["cad_agent_payload_invalid"])
    return result


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _git_id(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise SimReadyCadAgentContractError(["cad_agent_bound_file_invalid"])
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _file_record_valid(value: Any, *, verify_files: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    path_text = str(value.get("path") or "")
    if (
        not path_text
        or not isinstance(value.get("size_bytes"), int)
        or int(value.get("size_bytes", 0)) <= 0
        or not _digest(value.get("sha256"))
    ):
        return False
    if not verify_files:
        return True
    path = Path(path_text).expanduser().resolve()
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == value["size_bytes"]
        and _sha256(path) == value["sha256"]
    )


def _file_record_identity(value: Any) -> tuple[int, str] | None:
    if not isinstance(value, Mapping):
        return None
    if not isinstance(value.get("size_bytes"), int) or not _digest(
        value.get("sha256")
    ):
        return None
    return (int(value["size_bytes"]), str(value["sha256"]))


def _file_records_equivalent(left: Any, right: Any) -> bool:
    left_identity = _file_record_identity(left)
    return left_identity is not None and left_identity == _file_record_identity(right)


def _file_record_lists_equivalent(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        _file_records_equivalent(left_record, right_record)
        for left_record, right_record in zip(left, right, strict=True)
    )


def _read_json_record(record: Mapping[str, Any], code: str) -> dict[str, Any]:
    if not _file_record_valid(record, verify_files=True):
        raise SimReadyCadAgentContractError([code])
    try:
        value = json.loads(Path(str(record["path"])).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyCadAgentContractError([code]) from exc
    if not isinstance(value, dict):
        raise SimReadyCadAgentContractError([code])
    return value


def _reference_object_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    slot = row.get("replacement_slot")
    return (
        slot if isinstance(slot, int) and not isinstance(slot, bool) else -1,
        str(row.get("task_id") or ""),
        str(row.get("asset_id") or ""),
    )


def validate_cad_agent_reference_manifest(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    manifest = _clone(value)
    errors: list[str] = []
    if manifest.get("schema_version") != REFERENCE_MANIFEST_SCHEMA_VERSION:
        errors.append("cad_agent_reference_manifest_schema_invalid")
    if not str(manifest.get("scene_id") or "").strip():
        errors.append("cad_agent_reference_manifest_scene_invalid")
    objects = manifest.get("objects")
    if (
        not isinstance(objects, list)
        or not objects
        or len(objects) > MAX_REPLACEMENT_OBJECTS
    ):
        errors.append("cad_agent_reference_manifest_object_count_invalid")
        objects = []
    keys: list[tuple[int, str, str]] = []
    slots: list[int] = []
    for row in objects:
        if not isinstance(row, Mapping):
            errors.append("cad_agent_reference_manifest_object_invalid")
            continue
        key = _reference_object_key(row)
        slot, task_id, asset_id = key
        keys.append(key)
        slots.append(slot)
        if (
            slot < 1
            or slot > MAX_REPLACEMENT_OBJECTS
            or not task_id
            or not asset_id
        ):
            errors.append("cad_agent_reference_manifest_object_invalid")
        if not _file_record_valid(row.get("task_freeze"), verify_files=verify_files):
            errors.append("cad_agent_reference_manifest_task_freeze_invalid")
        references = row.get("reference_images")
        if (
            not isinstance(references, list)
            or not references
            or any(
                not _file_record_valid(record, verify_files=verify_files)
                for record in references
            )
        ):
            errors.append("cad_agent_reference_manifest_reference_images_invalid")
        if str(row.get("reference_authority") or "") not in {
            "observed_scene_reconnaissance_frames",
            "evaluation_authorized_observed_scene_frames",
        }:
            errors.append("cad_agent_reference_manifest_authority_invalid")
    if len(keys) != len(set(keys)) or len(slots) != len(set(slots)):
        errors.append("cad_agent_reference_manifest_object_identity_invalid")
    if manifest.get("maximum_replacement_objects") != MAX_REPLACEMENT_OBJECTS:
        errors.append("cad_agent_reference_manifest_capacity_invalid")
    if manifest.get("manifest_digest") != canonical_digest(
        manifest, digest_field="manifest_digest"
    ):
        errors.append("cad_agent_reference_manifest_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return manifest


def seal_cad_agent_reference_manifest(
    *,
    scene_id: str,
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload_objects: list[dict[str, Any]] = []
    for row in objects:
        references = row.get("reference_image_paths")
        if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
            raise SimReadyCadAgentContractError(
                ["cad_agent_reference_manifest_reference_images_invalid"]
            )
        payload_objects.append(
            {
                "replacement_slot": int(row.get("replacement_slot", 0)),
                "task_id": str(row.get("task_id") or ""),
                "asset_id": str(row.get("asset_id") or ""),
                "task_freeze": file_record(str(row.get("task_freeze_path") or "")),
                "reference_authority": str(
                    row.get("reference_authority")
                    or "observed_scene_reconnaissance_frames"
                ),
                "reference_images": [file_record(path) for path in references],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_MANIFEST_SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "objects": payload_objects,
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
    }
    payload["manifest_digest"] = canonical_digest(
        payload, digest_field="manifest_digest"
    )
    return validate_cad_agent_reference_manifest(payload)


def _select_reference_manifest_object(
    *,
    manifest: Mapping[str, Any],
    scene_id: str,
    replacement_slot: int,
    task_id: str,
    asset_id: str,
) -> dict[str, Any]:
    try:
        admitted = validate_cad_agent_reference_manifest(manifest)
    except SimReadyCadAgentContractError:
        raise
    if admitted.get("scene_id") != scene_id:
        raise SimReadyCadAgentContractError(
            ["cad_agent_reference_manifest_scene_mismatch"]
        )
    key = (replacement_slot, task_id, asset_id)
    matches = [
        row
        for row in admitted.get("objects", [])
        if _reference_object_key(row) == key
    ]
    if len(matches) != 1:
        raise SimReadyCadAgentContractError(
            ["cad_agent_reference_manifest_object_missing"]
        )
    return _clone(matches[0])


def validate_cad_agent_reference_binding_audit(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    audit = _clone(value)
    errors: list[str] = []
    if audit.get("schema_version") != REFERENCE_BINDING_AUDIT_SCHEMA_VERSION:
        errors.append("cad_agent_reference_binding_audit_schema_invalid")
    manifest_record = audit.get("reference_manifest")
    manifest: dict[str, Any] = {}
    if not _file_record_valid(manifest_record, verify_files=verify_files):
        errors.append("cad_agent_reference_binding_audit_manifest_invalid")
    elif verify_files:
        try:
            manifest = validate_cad_agent_reference_manifest(
                _read_json_record(
                    manifest_record,
                    "cad_agent_reference_binding_audit_manifest_invalid",
                ),
                verify_files=verify_files,
            )
        except SimReadyCadAgentContractError as exc:
            errors.extend(exc.codes)
            manifest = {}
    scene_id = str(audit.get("scene_id") or "")
    if manifest and manifest.get("scene_id") != scene_id:
        errors.append("cad_agent_reference_binding_audit_scene_mismatch")
    rows = audit.get("candidate_rows")
    if not isinstance(rows, list) or not rows:
        errors.append("cad_agent_reference_binding_audit_rows_invalid")
        rows = []
    identities: list[tuple[int, str, str, str]] = []
    object_keys: set[tuple[int, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"cad_agent_reference_binding_audit_row_invalid:{index}")
            continue
        output_record = row.get("cad_agent_output")
        if not _file_record_valid(output_record, verify_files=verify_files):
            errors.append(f"cad_agent_reference_binding_audit_output_invalid:{index}")
            continue
        output: dict[str, Any] = {}
        verify_output_artifacts = (
            verify_files
            and row.get("cad_agent_output_artifacts_verified") is not False
        )
        if verify_files:
            try:
                output = validate_cad_agent_output(
                    _read_json_record(
                        output_record,
                        f"cad_agent_reference_binding_audit_output_invalid:{index}",
                    ),
                    verify_files=verify_output_artifacts,
                )
            except SimReadyCadAgentContractError as exc:
                errors.extend(exc.codes)
                output = {}
        request = output.get("request") or {}
        slot = request.get("replacement_slot")
        task_id = str(request.get("task_id") or "")
        asset_id = str(request.get("asset_id") or "")
        backend_id = str(((request.get("backend") or {}).get("backend_id")) or "")
        identity = (
            slot if isinstance(slot, int) and not isinstance(slot, bool) else -1,
            task_id,
            asset_id,
            backend_id,
        )
        identities.append(identity)
        object_keys.add(identity[:3])
        if (
            row.get("replacement_slot") != identity[0]
            or row.get("task_id") != task_id
            or row.get("asset_id") != asset_id
            or row.get("cad_agent_backend_id") != backend_id
            or row.get("cad_agent_output_receipt_digest")
            != output.get("receipt_digest")
            or row.get("cad_agent_output_artifacts_verified")
            != verify_output_artifacts
        ):
            errors.append(f"cad_agent_reference_binding_audit_row_join_invalid:{index}")
        if manifest and output:
            try:
                selected = _select_reference_manifest_object(
                    manifest=manifest,
                    scene_id=scene_id,
                    replacement_slot=identity[0],
                    task_id=task_id,
                    asset_id=asset_id,
                )
            except SimReadyCadAgentContractError as exc:
                errors.extend(exc.codes)
            else:
                inputs = request.get("inputs") or {}
                if (
                    not _file_records_equivalent(
                        selected.get("task_freeze"), inputs.get("task_freeze")
                    )
                    or not _file_record_lists_equivalent(
                        selected.get("reference_images"),
                        inputs.get("reference_images"),
                    )
                    or row.get("reference_manifest_object_digest")
                    != canonical_digest(selected)
                ):
                    errors.append(
                        f"cad_agent_reference_binding_audit_manifest_join_invalid:{index}"
                    )
    if len(identities) != len(set(identities)):
        errors.append("cad_agent_reference_binding_audit_duplicate_candidate")
    if audit.get("replacement_object_count") != len(object_keys):
        errors.append("cad_agent_reference_binding_audit_object_count_invalid")
    if (
        audit.get("historical_requests_rewritten") is not False
        or audit.get("agent_outputs_regenerated") is not False
        or audit.get("all_candidate_references_manifest_bound") is not True
    ):
        errors.append("cad_agent_reference_binding_audit_claim_boundary_invalid")
    if audit.get("audit_digest") != canonical_digest(
        audit, digest_field="audit_digest"
    ):
        errors.append("cad_agent_reference_binding_audit_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return audit


def materialize_cad_agent_reference_binding_audit(
    *,
    scene_id: str,
    reference_manifest_path: str | Path,
    cad_agent_output_paths: Sequence[str | Path],
    output_path: str | Path | None = None,
    verify_cad_output_artifact_files: bool = True,
) -> dict[str, Any]:
    manifest_record = file_record(reference_manifest_path)
    manifest = validate_cad_agent_reference_manifest(
        _read_json_record(
            manifest_record,
            "cad_agent_reference_binding_audit_manifest_invalid",
        )
    )
    rows: list[dict[str, Any]] = []
    object_keys: set[tuple[int, str, str]] = set()
    for path in cad_agent_output_paths:
        output_record = file_record(path)
        output = validate_cad_agent_output(
            _read_json_record(
                output_record,
                "cad_agent_reference_binding_audit_output_invalid",
            ),
            verify_files=verify_cad_output_artifact_files,
        )
        request = output["request"]
        slot = int(request["replacement_slot"])
        task_id = str(request["task_id"])
        asset_id = str(request["asset_id"])
        backend_id = str(request["backend"]["backend_id"])
        selected = _select_reference_manifest_object(
            manifest=manifest,
            scene_id=str(scene_id),
            replacement_slot=slot,
            task_id=task_id,
            asset_id=asset_id,
        )
        object_keys.add((slot, task_id, asset_id))
        rows.append(
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "cad_agent_backend_id": backend_id,
                "cad_agent_output": output_record,
                "cad_agent_output_receipt_digest": output["receipt_digest"],
                "cad_agent_output_artifacts_verified": bool(
                    verify_cad_output_artifact_files
                ),
                "reference_manifest_object_digest": canonical_digest(selected),
                "reference_image_count": len(selected["reference_images"]),
                "request_contains_reference_manifest_field": bool(
                    (request.get("inputs") or {}).get("reference_manifest")
                ),
                "binding_status": "manifest_bound",
            }
        )
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_BINDING_AUDIT_SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "reference_manifest": manifest_record,
        "candidate_rows": sorted(
            rows,
            key=lambda row: (
                row["replacement_slot"],
                row["task_id"],
                row["asset_id"],
                row["cad_agent_backend_id"],
            ),
        ),
        "replacement_object_count": len(object_keys),
        "historical_requests_rewritten": False,
        "agent_outputs_regenerated": False,
        "all_candidate_references_manifest_bound": True,
        "claim_boundary": {
            "historical_request_reference_binding_audited": True,
            "cad_geometry_regenerated": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
    }
    payload["audit_digest"] = canonical_digest(payload, digest_field="audit_digest")
    validated = validate_cad_agent_reference_binding_audit(payload)
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(validated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return validated


def _archive_binds_commit(record: Any, commit: Any, *, verify_files: bool) -> bool:
    if not _file_record_valid(record, verify_files=verify_files):
        return False
    if not verify_files:
        return True
    try:
        with ZipFile(str(record["path"])) as archive:
            return archive.comment.decode("ascii") == str(commit)
    except (BadZipFile, OSError, UnicodeDecodeError):
        return False


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    result = [_finite(item) for item in value]
    if any(item is None or item <= 0.0 for item in result):
        return None
    return [float(item) for item in result]


def validate_cad_agent_request(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    request = _clone(value)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("cad_agent_request_schema_invalid")
    for field in ("request_id", "scene_id", "task_id", "asset_id"):
        if not str(request.get(field) or "").strip():
            errors.append(f"cad_agent_request_{field}_invalid")
    slot = request.get("replacement_slot")
    if (
        not isinstance(slot, int)
        or isinstance(slot, bool)
        or slot < 1
        or slot > MAX_REPLACEMENT_OBJECTS
    ):
        errors.append("cad_agent_request_replacement_slot_invalid")

    backend = request.get("backend")
    if not isinstance(backend, Mapping):
        errors.append("cad_agent_request_backend_invalid")
    else:
        backend_id = str(backend.get("backend_id") or "")
        if (
            backend_id not in ADMITTED_BACKENDS
            or backend_id in FORBIDDEN_BACKEND_IDS
            or backend.get("execution_mode") not in ADMITTED_BACKENDS.get(backend_id, ())
            or backend.get("agent_authored_geometry") is not True
            or backend.get("deterministic_geometry_generator_used") is not False
            or backend.get("graph_geometry_used_for_cad_authoring") is not False
            or backend.get("deterministic_format_conversion_only") is not True
            or backend.get("repository_url") != BACKEND_REPOSITORIES.get(backend_id)
            or not _git_id(backend.get("commit"))
            or not _git_id(backend.get("tree"))
            or not _archive_binds_commit(
                backend.get("source_archive"),
                backend.get("commit"),
                verify_files=verify_files,
            )
            or backend.get("license") != "MIT"
            or not str(backend.get("model_id") or "").strip()
        ):
            errors.append("cad_agent_request_backend_invalid")

    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("cad_agent_request_inputs_invalid")
    else:
        if inputs.get("geometry_authority") != "agent_authored_from_observed_evidence":
            errors.append("cad_agent_request_geometry_authority_invalid")
        if (
            inputs.get("collision_graph_is_geometry_authority") is not False
            or inputs.get("primitive_graph_used_as_cad_candidate") is not False
        ):
            errors.append("cad_agent_request_graph_geometry_forbidden")
        if not _file_record_valid(inputs.get("task_freeze"), verify_files=verify_files):
            errors.append("cad_agent_request_task_freeze_invalid")
        if not _file_record_valid(inputs.get("cad_brief"), verify_files=verify_files):
            errors.append("cad_agent_request_brief_invalid")
        references = inputs.get("reference_images")
        if (
            not isinstance(references, list)
            or not references
            or any(
                not _file_record_valid(record, verify_files=verify_files)
                for record in references
            )
        ):
            errors.append("cad_agent_request_reference_images_invalid")
        manifest_record = inputs.get("reference_manifest")
        if not _file_record_valid(manifest_record, verify_files=verify_files):
            errors.append("cad_agent_request_reference_manifest_invalid")
        if inputs.get("reference_binding_source") != "manifest_derived":
            errors.append("cad_agent_request_reference_binding_invalid")
        if not _digest(inputs.get("reference_manifest_object_digest")):
            errors.append("cad_agent_request_reference_manifest_object_digest_invalid")
        if verify_files and _file_record_valid(manifest_record, verify_files=True):
            manifest = _read_json_record(
                manifest_record, "cad_agent_request_reference_manifest_invalid"
            )
            try:
                selected = _select_reference_manifest_object(
                    manifest=manifest,
                    scene_id=str(request.get("scene_id") or ""),
                    replacement_slot=(
                        slot if isinstance(slot, int) and not isinstance(slot, bool) else -1
                    ),
                    task_id=str(request.get("task_id") or ""),
                    asset_id=str(request.get("asset_id") or ""),
                )
            except SimReadyCadAgentContractError as exc:
                errors.extend(exc.codes)
            else:
                if not _file_records_equivalent(
                    selected.get("task_freeze"), inputs.get("task_freeze")
                ):
                    errors.append("cad_agent_request_reference_manifest_join_invalid")
                if not _file_record_lists_equivalent(
                    selected.get("reference_images"), references
                ):
                    errors.append("cad_agent_request_reference_manifest_join_invalid")
                if inputs.get("reference_manifest_object_digest") != canonical_digest(
                    selected
                ):
                    errors.append("cad_agent_request_reference_manifest_join_invalid")
        if _vector3(inputs.get("metric_envelope_mm")) is None:
            errors.append("cad_agent_request_metric_envelope_invalid")
        tolerance = _finite(inputs.get("envelope_tolerance_mm"))
        if tolerance is None or tolerance <= 0.0 or tolerance > 1.0:
            errors.append("cad_agent_request_envelope_tolerance_invalid")
        if not errors and verify_files:
            freeze = _read_json_record(
                inputs["task_freeze"], "cad_agent_request_task_freeze_invalid"
            )
            try:
                validated_freeze = validate_task_freeze(freeze)
            except DualTaskRehearsalContractError as exc:
                raise SimReadyCadAgentContractError(
                    ["cad_agent_request_task_freeze_invalid", *exc.errors]
                ) from exc
            if (
                validated_freeze.get("task_id") != request.get("task_id")
                or (
                    (validated_freeze.get("removal_plan") or {}).get(
                        "replacement_asset_id"
                    )
                    != request.get("asset_id")
                )
            ):
                errors.append("cad_agent_request_task_binding_mismatch")

    budget = request.get("execution_budget")
    if not isinstance(budget, Mapping):
        errors.append("cad_agent_request_budget_invalid")
    else:
        cap = _finite(budget.get("hard_cap_usd"))
        backend_id = str((backend or {}).get("backend_id") or "")
        expected_cap = 0.0 if backend_id == "earthtojake_text_to_cad" else 1.0
        if (
            cap is None
            or abs(cap - expected_cap) > 1e-9
            or budget.get("maximum_attempts") != 1
            or budget.get("automatic_retries") != 0
        ):
            errors.append("cad_agent_request_budget_invalid")

    if request.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("cad_agent_request_claim_boundary_invalid")
    if request.get("request_digest") != canonical_digest(
        request, digest_field="request_digest"
    ):
        errors.append("cad_agent_request_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return request


def seal_cad_agent_request(
    *,
    request_id: str,
    scene_id: str,
    task_id: str,
    asset_id: str,
    replacement_slot: int,
    backend: Mapping[str, Any],
    task_freeze_path: str | Path,
    cad_brief_path: str | Path,
    metric_envelope_mm: Sequence[float],
    reference_manifest_path: str | Path | None = None,
    reference_image_paths: Sequence[str | Path] | None = None,
    envelope_tolerance_mm: float = 0.25,
) -> dict[str, Any]:
    task_freeze_record = file_record(task_freeze_path)
    if (
        not isinstance(replacement_slot, int)
        or isinstance(replacement_slot, bool)
        or replacement_slot < 1
        or replacement_slot > MAX_REPLACEMENT_OBJECTS
    ):
        raise SimReadyCadAgentContractError(
            ["cad_agent_request_replacement_slot_invalid"]
        )
    if reference_manifest_path is None:
        raise SimReadyCadAgentContractError(
            ["cad_agent_request_reference_manifest_required"]
        )
    else:
        if reference_image_paths is not None:
            raise SimReadyCadAgentContractError(
                ["cad_agent_request_manual_reference_images_forbidden"]
            )
        reference_manifest_record = file_record(reference_manifest_path)
        manifest = _read_json_record(
            reference_manifest_record, "cad_agent_request_reference_manifest_invalid"
        )
        selected = _select_reference_manifest_object(
            manifest=manifest,
            scene_id=str(scene_id),
            replacement_slot=int(replacement_slot),
            task_id=str(task_id),
            asset_id=str(asset_id),
        )
        if not _file_records_equivalent(selected.get("task_freeze"), task_freeze_record):
            raise SimReadyCadAgentContractError(
                ["cad_agent_request_reference_manifest_join_invalid"]
            )
        reference_records = [_clone(record) for record in selected["reference_images"]]
    payload: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": str(request_id),
        "scene_id": str(scene_id),
        "task_id": str(task_id),
        "asset_id": str(asset_id),
        "replacement_slot": int(replacement_slot),
        "backend": _clone(backend),
        "inputs": {
            "geometry_authority": "agent_authored_from_observed_evidence",
            "collision_graph_is_geometry_authority": False,
            "primitive_graph_used_as_cad_candidate": False,
            "task_freeze": task_freeze_record,
            "cad_brief": file_record(cad_brief_path),
            "reference_images": reference_records,
            "reference_manifest": reference_manifest_record,
            "reference_binding_source": "manifest_derived",
            "reference_manifest_object_digest": canonical_digest(selected),
            "metric_envelope_mm": [float(item) for item in metric_envelope_mm],
            "envelope_tolerance_mm": float(envelope_tolerance_mm),
        },
        "execution_budget": {
            "hard_cap_usd": (
                0.0
                if backend.get("backend_id") == "earthtojake_text_to_cad"
                else 1.0
            ),
            "maximum_attempts": 1,
            "automatic_retries": 0,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    payload["request_digest"] = canonical_digest(
        payload, digest_field="request_digest"
    )
    return validate_cad_agent_request(payload)


def validate_cad_agent_output(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    if receipt.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        errors.append("cad_agent_output_schema_invalid")
    if receipt.get("status") not in {"candidate_authored", "blocked"}:
        errors.append("cad_agent_output_status_invalid")
    request = receipt.get("request")
    if not isinstance(request, Mapping):
        errors.append("cad_agent_output_request_invalid")
    else:
        try:
            admitted_request = validate_cad_agent_request(
                request, verify_files=verify_files
            )
        except SimReadyCadAgentContractError as exc:
            errors.extend(exc.codes)
            admitted_request = {}
        if receipt.get("request_digest") != admitted_request.get("request_digest"):
            errors.append("cad_agent_output_request_digest_mismatch")

    if receipt.get("status") == "candidate_authored":
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            errors.append("cad_agent_output_artifacts_invalid")
        else:
            for role in ("generator_source", "step", "inspection_receipt"):
                if not _file_record_valid(
                    artifacts.get(role), verify_files=verify_files
                ):
                    errors.append(f"cad_agent_output_{role}_invalid")
            if _file_record_valid(
                artifacts.get("inspection_receipt"), verify_files=verify_files
            ):
                inspection = _read_json_record(
                    artifacts["inspection_receipt"],
                    "cad_agent_output_inspection_receipt_invalid",
                )
                try:
                    inspection = validate_step_inspection_receipt(
                        inspection, verify_files=verify_files
                    )
                except SimReadyCadAgentContractError as exc:
                    errors.extend(exc.codes)
                    inspection = {}
                if (
                    inspection.get("step", {}).get("sha256")
                    != (artifacts.get("step") or {}).get("sha256")
                    or inspection.get("measured_envelope_mm")
                    != receipt.get("measured_envelope_mm")
                ):
                    errors.append("cad_agent_output_inspection_join_invalid")
            snapshots = artifacts.get("snapshots")
            if (
                not isinstance(snapshots, list)
                or not snapshots
                or any(
                    not _file_record_valid(row, verify_files=verify_files)
                    for row in snapshots
                )
            ):
                errors.append("cad_agent_output_snapshots_invalid")
        actual = _vector3(receipt.get("measured_envelope_mm"))
        requested = _vector3(
            ((request or {}).get("inputs") or {}).get("metric_envelope_mm")
        )
        tolerance = _finite(
            ((request or {}).get("inputs") or {}).get("envelope_tolerance_mm")
        )
        if actual is None or requested is None or tolerance is None:
            errors.append("cad_agent_output_metric_envelope_invalid")
        elif any(abs(a - b) > tolerance for a, b in zip(actual, requested, strict=True)):
            errors.append("cad_agent_output_metric_envelope_mismatch")
        execution = receipt.get("execution")
        if (
            not isinstance(execution, Mapping)
            or execution.get("agent_authored_geometry") is not True
            or execution.get("deterministic_geometry_generator_used") is not False
            or execution.get("graph_geometry_used_for_cad_authoring") is not False
            or not str(execution.get("model_id") or "")
            or not _file_record_valid(
                execution.get("execution_receipt"), verify_files=verify_files
            )
        ):
            errors.append("cad_agent_output_execution_invalid")
        elif _file_record_valid(
            execution.get("execution_receipt"), verify_files=verify_files
        ):
            execution_receipt = _read_json_record(
                execution["execution_receipt"],
                "cad_agent_output_execution_receipt_invalid",
            )
            try:
                execution_receipt = validate_cad_agent_execution_receipt(
                    execution_receipt, verify_files=verify_files
                )
            except SimReadyCadAgentContractError as exc:
                errors.extend(exc.codes)
                execution_receipt = {}
            if (
                execution_receipt.get("request_digest")
                != admitted_request.get("request_digest")
                or execution_receipt.get("output_step", {}).get("sha256")
                != (receipt.get("artifacts") or {}).get("step", {}).get("sha256")
            ):
                errors.append("cad_agent_output_execution_join_invalid")
        cost = _finite(receipt.get("actual_cost_usd"))
        cap = _finite(((request or {}).get("execution_budget") or {}).get("hard_cap_usd"))
        if cost is None or cost < 0.0 or cap is None or cost > cap + 1e-9:
            errors.append("cad_agent_output_cost_invalid")
        if receipt.get("blocker") is not None:
            errors.append("cad_agent_output_blocker_invalid")
    else:
        blocker = receipt.get("blocker")
        if not isinstance(blocker, Mapping) or not str(blocker.get("code") or ""):
            errors.append("cad_agent_output_blocker_invalid")
        if receipt.get("artifacts") not in (None, {}):
            errors.append("cad_agent_output_blocked_artifacts_invalid")

    if receipt.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("cad_agent_output_claim_boundary_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("cad_agent_output_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return receipt


def seal_cad_agent_output(
    *,
    request: Mapping[str, Any],
    generator_source_path: str | Path,
    step_path: str | Path,
    inspection_receipt_path: str | Path,
    snapshot_paths: Sequence[str | Path],
    execution_receipt_path: str | Path,
    measured_envelope_mm: Sequence[float],
    actual_cost_usd: float,
) -> dict[str, Any]:
    admitted_request = validate_cad_agent_request(request)
    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "candidate_authored",
        "request": admitted_request,
        "request_digest": admitted_request["request_digest"],
        "artifacts": {
            "generator_source": file_record(generator_source_path),
            "step": file_record(step_path),
            "inspection_receipt": file_record(inspection_receipt_path),
            "snapshots": [file_record(path) for path in snapshot_paths],
        },
        "measured_envelope_mm": [float(item) for item in measured_envelope_mm],
        "execution": {
            "backend_id": admitted_request["backend"]["backend_id"],
            "execution_mode": admitted_request["backend"]["execution_mode"],
            "model_id": admitted_request["backend"]["model_id"],
            "agent_authored_geometry": True,
            "deterministic_geometry_generator_used": False,
            "graph_geometry_used_for_cad_authoring": False,
            "execution_receipt": file_record(execution_receipt_path),
        },
        "actual_cost_usd": float(actual_cost_usd),
        "artifacts_are_agent_authored_candidates_not_simready_qualification": True,
        "blocker": None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    payload["receipt_digest"] = canonical_digest(
        payload, digest_field="receipt_digest"
    )
    return validate_cad_agent_output(payload)


def validate_step_inspection_receipt(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    if receipt.get("schema_version") != INSPECTION_SCHEMA_VERSION:
        errors.append("cad_agent_step_inspection_schema_invalid")
    if receipt.get("status") != "passed":
        errors.append("cad_agent_step_inspection_status_invalid")
    if not _file_record_valid(receipt.get("step"), verify_files=verify_files):
        errors.append("cad_agent_step_inspection_step_invalid")
    if _vector3(receipt.get("measured_envelope_mm")) is None:
        errors.append("cad_agent_step_inspection_envelope_invalid")
    center = receipt.get("measured_center_mm")
    if (
        not isinstance(center, list)
        or len(center) != 3
        or any(_finite(item) is None for item in center)
    ):
        errors.append("cad_agent_step_inspection_center_invalid")
    topology = receipt.get("topology")
    if (
        not isinstance(topology, Mapping)
        or not isinstance(topology.get("face_count"), int)
        or topology.get("face_count", 0) <= 0
        or not isinstance(topology.get("edge_count"), int)
        or topology.get("edge_count", 0) <= 0
    ):
        errors.append("cad_agent_step_inspection_topology_invalid")
    inspector = receipt.get("inspector")
    if (
        not isinstance(inspector, Mapping)
        or inspector.get("backend") != "build123d_import_step"
        or not str(inspector.get("build123d_version") or "")
        or not str(inspector.get("ocp_version") or "")
        or not _file_record_valid(
            inspector.get("module_source"), verify_files=verify_files
        )
    ):
        errors.append("cad_agent_step_inspection_inspector_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("cad_agent_step_inspection_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return receipt


def materialize_step_inspection_receipt(
    *, step_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Independently reopen and measure exact STEP bytes with build123d/OCP."""

    try:
        import build123d
        import OCP
        from build123d import import_step
    except ImportError as exc:  # pragma: no cover - CAD-tool environment only
        raise SimReadyCadAgentContractError(
            ["cad_agent_step_inspection_dependency_missing"]
        ) from exc
    step_record = file_record(step_path)
    try:
        shape = import_step(step_record["path"])
        bounds = shape.bounding_box()
        size = [float(bounds.size.X), float(bounds.size.Y), float(bounds.size.Z)]
        center = [
            float(bounds.center().X),
            float(bounds.center().Y),
            float(bounds.center().Z),
        ]
        face_count = len(shape.faces())
        edge_count = len(shape.edges())
    except Exception as exc:  # pragma: no cover - malformed external CAD bytes
        raise SimReadyCadAgentContractError(
            ["cad_agent_step_inspection_import_failed"]
        ) from exc
    payload: dict[str, Any] = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "passed",
        "step": step_record,
        "measured_envelope_mm": size,
        "measured_center_mm": center,
        "topology": {"face_count": face_count, "edge_count": edge_count},
        "inspector": {
            "backend": "build123d_import_step",
            "build123d_version": str(getattr(build123d, "__version__", "unknown")),
            "ocp_version": str(getattr(OCP, "__version__", "unknown")),
            "python_version": platform.python_version(),
            "module_source": file_record(__file__),
        },
        "claim_boundary": {
            "step_bytes_reopened": True,
            "metric_envelope_measured": True,
            "visual_quality_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
    }
    payload["receipt_digest"] = canonical_digest(
        payload, digest_field="receipt_digest"
    )
    validated = validate_step_inspection_receipt(payload)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validated


def validate_cad_agent_execution_receipt(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    if receipt.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        errors.append("cad_agent_execution_schema_invalid")
    if receipt.get("status") != "candidate_authored":
        errors.append("cad_agent_execution_status_invalid")
    if str(receipt.get("backend_id") or "") not in ADMITTED_BACKENDS:
        errors.append("cad_agent_execution_backend_invalid")
    if not _digest(receipt.get("request_digest")):
        errors.append("cad_agent_execution_request_digest_invalid")
    for field in ("generator_source", "cad_brief", "output_step"):
        if not _file_record_valid(receipt.get(field), verify_files=verify_files):
            errors.append(f"cad_agent_execution_{field}_invalid")
    event_rows = receipt.get("event_rows")
    if (
        receipt.get("agent_authored_geometry") is not True
        or receipt.get("deterministic_geometry_generator_used") is not False
        or receipt.get("graph_geometry_used_for_cad_authoring") is not False
        or not isinstance(event_rows, list)
        or not event_rows
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("event") or "")
            or row.get("status") not in {"passed", "failed_then_repaired"}
            for row in event_rows or []
        )
    ):
        errors.append("cad_agent_execution_trace_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("cad_agent_execution_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return receipt


def seal_cad_agent_execution_receipt(
    *,
    request: Mapping[str, Any],
    generator_source_path: str | Path,
    cad_brief_path: str | Path,
    output_step_path: str | Path,
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    admitted = validate_cad_agent_request(request)
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "candidate_authored",
        "backend_id": admitted["backend"]["backend_id"],
        "execution_mode": admitted["backend"]["execution_mode"],
        "model_id": admitted["backend"]["model_id"],
        "request_digest": admitted["request_digest"],
        "generator_source": file_record(generator_source_path),
        "cad_brief": file_record(cad_brief_path),
        "output_step": file_record(output_step_path),
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "event_rows": [_clone(row) for row in event_rows],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    payload["receipt_digest"] = canonical_digest(
        payload, digest_field="receipt_digest"
    )
    return validate_cad_agent_execution_receipt(payload)


def validate_cad_agent_matrix(value: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _clone(value)
    errors: list[str] = []
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        errors.append("cad_agent_matrix_schema_invalid")
    objects = matrix.get("objects")
    if (
        not isinstance(objects, list)
        or not objects
        or len(objects) > MAX_REPLACEMENT_OBJECTS
    ):
        errors.append("cad_agent_matrix_object_count_invalid")
        objects = []
    slots: list[int] = []
    asset_ids: list[str] = []
    task_ids: list[str] = []
    for row in objects:
        if not isinstance(row, Mapping):
            errors.append("cad_agent_matrix_object_invalid")
            continue
        slot = row.get("replacement_slot")
        slots.append(slot if isinstance(slot, int) and not isinstance(slot, bool) else -1)
        asset_ids.append(str(row.get("asset_id") or ""))
        task_ids.append(str(row.get("task_id") or ""))
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(ADMITTED_BACKENDS):
            errors.append("cad_agent_matrix_candidates_invalid")
            continue
        backend_ids: list[str] = []
        for candidate in candidates:
            try:
                receipt = validate_cad_agent_output(candidate)
            except SimReadyCadAgentContractError as exc:
                errors.extend(exc.codes)
                continue
            request = receipt["request"]
            backend_ids.append(request["backend"]["backend_id"])
            if (
                request["replacement_slot"] != slot
                or request["asset_id"] != row.get("asset_id")
                or request["task_id"] != row.get("task_id")
            ):
                errors.append("cad_agent_matrix_candidate_binding_mismatch")
        if len(backend_ids) != len(set(backend_ids)):
            errors.append("cad_agent_matrix_duplicate_backend")
        if set(backend_ids) != set(ADMITTED_BACKENDS):
            errors.append("cad_agent_matrix_required_backends_missing")
    if (
        any(slot < 1 or slot > MAX_REPLACEMENT_OBJECTS for slot in slots)
        or len(slots) != len(set(slots))
        or len(asset_ids) != len(set(asset_ids))
        or len(task_ids) != len(set(task_ids))
    ):
        errors.append("cad_agent_matrix_object_identity_invalid")
    if matrix.get("maximum_replacement_objects") != MAX_REPLACEMENT_OBJECTS:
        errors.append("cad_agent_matrix_capacity_invalid")
    if matrix.get("required_backends_per_object") != sorted(ADMITTED_BACKENDS):
        errors.append("cad_agent_matrix_required_backends_invalid")
    if matrix.get("deterministic_cad_backends_admitted") is not False:
        errors.append("cad_agent_matrix_deterministic_backend_invalid")
    if matrix.get("matrix_digest") != canonical_digest(
        matrix, digest_field="matrix_digest"
    ):
        errors.append("cad_agent_matrix_digest_invalid")
    if errors:
        raise SimReadyCadAgentContractError(errors)
    return matrix


def seal_cad_agent_matrix(*, objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "objects": [_clone(row) for row in objects],
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "required_backends_per_object": sorted(ADMITTED_BACKENDS),
        "deterministic_cad_backends_admitted": False,
    }
    payload["matrix_digest"] = canonical_digest(
        payload, digest_field="matrix_digest"
    )
    return validate_cad_agent_matrix(payload)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-step")
    inspect_parser.add_argument("--step", required=True)
    inspect_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect-step":
        receipt = materialize_step_inspection_receipt(
            step_path=args.step,
            output_path=args.output,
        )
        json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    raise AssertionError("unreachable")


__all__ = [
    "ADMITTED_BACKENDS",
    "BACKEND_REPOSITORIES",
    "CLAIM_BOUNDARY",
    "EXECUTION_SCHEMA_VERSION",
    "INSPECTION_SCHEMA_VERSION",
    "MATRIX_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "REFERENCE_BINDING_AUDIT_SCHEMA_VERSION",
    "REFERENCE_MANIFEST_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "SimReadyCadAgentContractError",
    "file_record",
    "materialize_cad_agent_reference_binding_audit",
    "materialize_step_inspection_receipt",
    "seal_cad_agent_reference_manifest",
    "seal_cad_agent_execution_receipt",
    "seal_cad_agent_matrix",
    "seal_cad_agent_output",
    "seal_cad_agent_request",
    "validate_cad_agent_matrix",
    "validate_cad_agent_reference_binding_audit",
    "validate_cad_agent_reference_manifest",
    "validate_cad_agent_execution_receipt",
    "validate_cad_agent_output",
    "validate_cad_agent_request",
    "validate_step_inspection_receipt",
]


if __name__ == "__main__":
    raise SystemExit(_main())
