"""Derive replacement native-qualification receipts from executable evidence.

``simready_replacement_native_qualification.v1`` is the construction boundary
used by the native task runtime.  Static CAD/graph readback can prove authored
structure, but it cannot prove native simulator import.  This module therefore
materializes the qualified receipt only from two file-backed inputs:

* a static graph-asset qualification receipt for exact authored USD structure;
* a native import receipt produced by a simulator/import lane.

The resulting receipt remains claim-bounded: simulator import evidence is not
appearance qualification, physical equivalence, or task success.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    validate_scene_freeze,
    validate_task_freeze,
)
from .simready_graph_asset_static_qualification import (
    SCHEMA_VERSION as STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
)


SCHEMA_VERSION = "simready_replacement_native_qualification.v1"
NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION = "simready_replacement_native_import_receipt.v1"
NATIVE_IMPORT_EXECUTION_SCHEMA_VERSION = (
    "simready_replacement_native_import_execution.v1"
)


class SimReadyReplacementNativeQualificationError(ValueError):
    """Stable validation failures for replacement native-qualification joins."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path_value: str | Path, *, role: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SimReadyReplacementNativeQualificationError(
            [f"simready_replacement_native_{role}_path_invalid"]
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyReplacementNativeQualificationError(
            [f"simready_replacement_native_{role}_receipt_invalid"]
        ) from exc
    if not isinstance(value, dict):
        raise SimReadyReplacementNativeQualificationError(
            [f"simready_replacement_native_{role}_receipt_invalid"]
        )
    return path, value


def _file_record(path: Path, value: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema_version": value["schema_version"],
        "canonical_digest": value[digest_field],
    }


def _canonical_receipt_valid(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    status: str | None,
    digest_field: str = "receipt_digest",
) -> bool:
    return (
        value.get("schema_version") == schema_version
        and (status is None or value.get("status") == status)
        and value.get(digest_field)
        == canonical_digest(value, digest_field=digest_field)
    )


def _identity_errors(
    receipt: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    role: str,
) -> list[str]:
    return [
        f"simready_replacement_native_{role}_identity_mismatch:{field}"
        for field, expected_value in expected.items()
        if str(receipt.get(field) or "") != expected_value
    ]


def validate_simready_replacement_native_import_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the task-neutral receipt shape expected from a native import lane."""

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SimReadyReplacementNativeQualificationError(
            ["simready_replacement_native_import_receipt_invalid"]
        ) from exc
    errors: list[str] = []
    if not _canonical_receipt_valid(
        payload,
        schema_version=NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION,
        status="native_import_qualified",
    ):
        errors.append("simready_replacement_native_import_receipt_invalid")
    for field in (
        "scene_id",
        "scene_freeze_digest",
        "task_id",
        "task_freeze_digest",
        "source_object_instance_id",
        "asset_id",
        "replacement_qualification_id",
    ):
        if not str(payload.get(field) or ""):
            errors.append(f"simready_replacement_native_import_identity_missing:{field}")
    if not _digest(payload.get("replacement_asset_sha256")):
        errors.append("simready_replacement_native_import_asset_digest_invalid")
    if payload.get("native_isaac_executed") is not True:
        errors.append("simready_replacement_native_import_execution_missing")
    if payload.get("native_simulator_import_qualified") is not True:
        errors.append("simready_replacement_native_import_not_qualified")
    if payload.get("physical_equivalence_claimed") is not False:
        errors.append("simready_replacement_native_import_physical_claim_invalid")
    import_identity = payload.get("simulator_import_identity")
    if not isinstance(import_identity, Mapping) or not str(
        import_identity.get("runtime") or ""
    ):
        errors.append("simready_replacement_native_import_identity_unbound")
    if errors:
        raise SimReadyReplacementNativeQualificationError(errors)
    return payload


def validate_simready_replacement_native_import_execution(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a retained native import probe execution receipt.

    This is the output expected from the future native import worker.  It is not
    itself the construction-facing qualification: it must first be joined to the
    frozen scene/task and exact static graph-asset qualification by
    ``materialize_simready_replacement_native_import_receipt``.
    """

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SimReadyReplacementNativeQualificationError(
            ["simready_replacement_native_import_execution_invalid"]
        ) from exc
    errors: list[str] = []
    if not _canonical_receipt_valid(
        payload,
        schema_version=NATIVE_IMPORT_EXECUTION_SCHEMA_VERSION,
        status="completed",
        digest_field="execution_digest",
    ):
        errors.append("simready_replacement_native_import_execution_invalid")
    if payload.get("native_isaac_executed") is not True:
        errors.append("simready_replacement_native_import_execution_missing")
    if payload.get("native_simulator_import_qualified") is not True:
        errors.append("simready_replacement_native_import_execution_not_qualified")
    if payload.get("physical_equivalence_claimed") is not False:
        errors.append("simready_replacement_native_import_execution_physical_claim_invalid")
    if not _digest(payload.get("replacement_asset_sha256")):
        errors.append("simready_replacement_native_import_execution_asset_digest_invalid")
    import_identity = payload.get("simulator_import_identity")
    if not isinstance(import_identity, Mapping) or not str(
        import_identity.get("runtime") or ""
    ):
        errors.append("simready_replacement_native_import_execution_identity_unbound")
    readback = payload.get("native_readback")
    if not isinstance(readback, Mapping) or readback.get("asset_imported") is not True:
        errors.append("simready_replacement_native_import_execution_readback_missing")
    if errors:
        raise SimReadyReplacementNativeQualificationError(errors)
    return payload


def materialize_simready_replacement_native_import_receipt(
    *,
    scene_freeze_receipt_path: str | Path,
    task_freeze_receipt_path: str | Path,
    static_qualification_receipt_path: str | Path,
    native_import_execution_receipt_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Join retained native execution to scene/task/static asset identity."""

    scene_path, raw_scene = _read_json(scene_freeze_receipt_path, role="scene_freeze")
    task_path, raw_task = _read_json(task_freeze_receipt_path, role="task_freeze")
    static_path, static = _read_json(
        static_qualification_receipt_path,
        role="static_qualification",
    )
    execution_path, raw_execution = _read_json(
        native_import_execution_receipt_path,
        role="native_import_execution",
    )
    try:
        scene = validate_scene_freeze(raw_scene)
        task = validate_task_freeze(raw_task)
        execution = validate_simready_replacement_native_import_execution(raw_execution)
    except (
        DualTaskRehearsalContractError,
        SimReadyReplacementNativeQualificationError,
    ) as exc:
        codes = getattr(exc, "errors", getattr(exc, "codes", (str(exc),)))
        raise SimReadyReplacementNativeQualificationError(
            [
                f"simready_replacement_native_import_input_invalid:{code}"
                for code in codes
            ]
        ) from exc

    removal = task["removal_plan"]
    source = task["source_object"]
    common_identity = {
        "scene_id": str(scene["selected_scene_id"]),
        "scene_freeze_digest": str(scene["scene_freeze_digest"]),
        "task_id": str(task["task_id"]),
        "task_freeze_digest": str(task["task_freeze_digest"]),
        "source_object_instance_id": str(source["instance_id"]),
        "asset_id": str(removal["replacement_asset_id"]),
        "replacement_qualification_id": str(removal["replacement_qualification_id"]),
    }
    errors: list[str] = []
    if task["scene_freeze_digest"] != scene["scene_freeze_digest"]:
        errors.append("simready_replacement_native_import_task_scene_mismatch")
    if not _canonical_receipt_valid(
        static,
        schema_version=STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
        status="authored_structure_statically_qualified",
    ):
        errors.append("simready_replacement_native_import_static_qualification_invalid")
    static_identity = {
        key: common_identity[key] for key in ("task_id", "task_freeze_digest", "asset_id")
    }
    errors.extend(
        _identity_errors(static, static_identity, role="import_static_qualification")
    )
    static_asset = (static.get("replacement_usd") or {}).get("sha256")
    if not _digest(static_asset):
        errors.append("simready_replacement_native_import_static_asset_digest_invalid")
    if execution.get("asset_id") != common_identity["asset_id"]:
        errors.append("simready_replacement_native_import_execution_asset_mismatch")
    if execution.get("replacement_asset_sha256") != static_asset:
        errors.append("simready_replacement_native_import_asset_digest_mismatch")
    if errors:
        raise SimReadyReplacementNativeQualificationError(errors)

    result: dict[str, Any] = {
        "schema_version": NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION,
        "status": "native_import_qualified",
        **common_identity,
        "replacement_asset_sha256": str(execution["replacement_asset_sha256"]),
        "native_isaac_executed": True,
        "native_simulator_import_qualified": True,
        "physical_equivalence_claimed": False,
        "execution_digest": execution["execution_digest"],
        "static_qualification_receipt_digest": static["receipt_digest"],
        "simulator_import_identity": json.loads(
            json.dumps(execution["simulator_import_identity"], sort_keys=True)
        ),
        "native_readback": json.loads(
            json.dumps(execution["native_readback"], sort_keys=True)
        ),
        "evidence_receipts": {
            "scene_freeze": _file_record(
                scene_path, scene, digest_field="scene_freeze_digest"
            ),
            "task_freeze": _file_record(
                task_path, task, digest_field="task_freeze_digest"
            ),
            "static_qualification": _file_record(
                static_path, static, digest_field="receipt_digest"
            ),
            "native_import_execution": _file_record(
                execution_path, execution, digest_field="execution_digest"
            ),
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination.exists() or destination.is_symlink():
            raise SimReadyReplacementNativeQualificationError(
                ["simready_replacement_native_import_output_exists"]
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def materialize_simready_replacement_native_qualification(
    *,
    scene_freeze_receipt_path: str | Path,
    task_freeze_receipt_path: str | Path,
    static_qualification_receipt_path: str | Path,
    native_import_receipt_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Emit ``simready_replacement_native_qualification.v1`` from evidence files."""

    scene_path, raw_scene = _read_json(scene_freeze_receipt_path, role="scene_freeze")
    task_path, raw_task = _read_json(task_freeze_receipt_path, role="task_freeze")
    static_path, static = _read_json(
        static_qualification_receipt_path,
        role="static_qualification",
    )
    native_path, raw_native = _read_json(
        native_import_receipt_path,
        role="native_import",
    )
    try:
        scene = validate_scene_freeze(raw_scene)
        task = validate_task_freeze(raw_task)
        native = validate_simready_replacement_native_import_receipt(raw_native)
    except (
        DualTaskRehearsalContractError,
        SimReadyReplacementNativeQualificationError,
    ) as exc:
        codes = getattr(exc, "errors", getattr(exc, "codes", (str(exc),)))
        raise SimReadyReplacementNativeQualificationError(
            [
                f"simready_replacement_native_input_invalid:{code}"
                for code in codes
            ]
        ) from exc

    errors: list[str] = []
    if task["scene_freeze_digest"] != scene["scene_freeze_digest"]:
        errors.append("simready_replacement_native_task_scene_mismatch")
    removal = task["removal_plan"]
    source = task["source_object"]
    common_identity = {
        "scene_id": str(scene["selected_scene_id"]),
        "scene_freeze_digest": str(scene["scene_freeze_digest"]),
        "task_id": str(task["task_id"]),
        "task_freeze_digest": str(task["task_freeze_digest"]),
        "source_object_instance_id": str(source["instance_id"]),
        "asset_id": str(removal["replacement_asset_id"]),
        "replacement_qualification_id": str(removal["replacement_qualification_id"]),
    }
    if not _canonical_receipt_valid(
        static,
        schema_version=STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
        status="authored_structure_statically_qualified",
    ):
        errors.append("simready_replacement_native_static_qualification_invalid")
    static_identity = {
        key: common_identity[key] for key in ("task_id", "task_freeze_digest", "asset_id")
    }
    errors.extend(_identity_errors(static, static_identity, role="static_qualification"))
    errors.extend(_identity_errors(native, common_identity, role="native_import"))
    static_asset = (static.get("replacement_usd") or {}).get("sha256")
    if not _digest(static_asset):
        errors.append("simready_replacement_native_static_asset_digest_invalid")
    if native.get("replacement_asset_sha256") != static_asset:
        errors.append("simready_replacement_native_asset_digest_mismatch")
    if native.get("native_simulator_import_qualified") is not True:
        errors.append("simready_replacement_native_import_not_qualified")
    if errors:
        raise SimReadyReplacementNativeQualificationError(errors)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "native_simulator_import_qualified",
        **common_identity,
        "replacement_asset_sha256": str(native["replacement_asset_sha256"]),
        "native_simulator_import_qualified": True,
        "native_import_receipt_digest": native["receipt_digest"],
        "static_qualification_receipt_digest": static["receipt_digest"],
        "evidence_receipts": {
            "scene_freeze": _file_record(
                scene_path, scene, digest_field="scene_freeze_digest"
            ),
            "task_freeze": _file_record(
                task_path, task, digest_field="task_freeze_digest"
            ),
            "static_qualification": _file_record(
                static_path, static, digest_field="receipt_digest"
            ),
            "native_import": _file_record(
                native_path, native, digest_field="receipt_digest"
            ),
        },
        "simulator_import_identity": json.loads(
            json.dumps(native["simulator_import_identity"], sort_keys=True)
        ),
        "claim_boundary": {
            "native_simulator_import_qualified": True,
            "simulator_execution_is_not_physical_truth": True,
            "appearance_materially_qualified": False,
            "joint_physics_behavior_qualified": bool(
                native.get("joint_physics_behavior_qualified", False)
            ),
            "contact_or_support_qualified": bool(
                native.get("contact_or_support_qualified", False)
            ),
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination.exists() or destination.is_symlink():
            raise SimReadyReplacementNativeQualificationError(
                ["simready_replacement_native_output_exists"]
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


__all__ = [
    "NATIVE_IMPORT_EXECUTION_SCHEMA_VERSION",
    "NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SimReadyReplacementNativeQualificationError",
    "materialize_simready_replacement_native_import_receipt",
    "materialize_simready_replacement_native_qualification",
    "validate_simready_replacement_native_import_execution",
    "validate_simready_replacement_native_import_receipt",
]
