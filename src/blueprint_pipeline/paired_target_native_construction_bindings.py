"""Seal paired-target replacement evidence for native Arena construction.

The original repeatable-replacement runtime consumes an Aura-era construction
binding.  Paired-target scenes have a different, stronger evidence chain:
registered USD bytes, one shared source-collider-deleted scene, and a native
Isaac import result covering every co-present replacement.  This adapter joins
that chain without manufacturing legacy Gaussian-removal receipts.

A valid binding admits exact bytes to a native construction canary only.  It
does not claim camera readback, reachability, contact, controls, policy success,
or physical equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


SCHEMA_VERSION = "paired_target_native_construction_bindings.v1"
MANIPULATION_PREFLIGHT_SCHEMA = "paired_target_native_manipulation_preflight.v1"
PAIRED_PREFLIGHT_SCHEMA = "paired_target_native_preflight.v1"
NATIVE_IMPORT_SCHEMA = "paired_target_native_import_runtime_result.v1"
REGISTERED_ASSET_SCHEMA = "registered_replacement_asset.v1"


class PairedTargetNativeConstructionBindingsError(ValueError):
    """Stable fail-closed paired-target construction binding error."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_bound_record(
    record: Any,
    *,
    role: str,
    expected_schema: str,
    digest_field: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise PairedTargetNativeConstructionBindingsError(
            [f"paired_target_construction_record_invalid:{role}"]
        )
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeConstructionBindingsError(
            [f"paired_target_construction_record_invalid:{role}"]
        ) from exc
    if (
        path.is_symlink()
        or not isinstance(value, dict)
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
        or value.get("schema_version") != expected_schema
        or not _digest(value.get(digest_field))
        or value.get(digest_field)
        != canonical_digest(value, digest_field=digest_field)
        or record.get(digest_field) != value.get(digest_field)
    ):
        raise PairedTargetNativeConstructionBindingsError(
            [f"paired_target_construction_record_invalid:{role}"]
        )
    return path, value


def _file_record(
    path: Path,
    *,
    schema_version: str | None = None,
    canonical_digest_value: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_file_invalid"]
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **({"schema_version": schema_version} if schema_version else {}),
        **(
            {"canonical_digest": canonical_digest_value}
            if canonical_digest_value
            else {}
        ),
    }


def validate_paired_target_native_construction_bindings(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a sealed, path-backed 1--5 object construction binding."""

    try:
        payload = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_invalid"]
        ) from exc
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("paired_target_construction_schema_invalid")
    if payload.get("status") != "paired_targets_admitted_for_native_construction":
        errors.append("paired_target_construction_status_invalid")
    if not str(payload.get("scene_id") or ""):
        errors.append("paired_target_construction_scene_invalid")
    if not _digest(payload.get("task_freeze_set_digest")):
        errors.append("paired_target_construction_task_freeze_set_invalid")
    rows = payload.get("bindings")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_REPLACEMENT_OBJECTS:
        errors.append("paired_target_construction_binding_count_invalid")
        rows = []
    if payload.get("replacement_object_count") != len(rows):
        errors.append("paired_target_construction_replacement_count_invalid")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            errors.append(f"paired_target_construction_binding_invalid:{index}")
            continue
        row = dict(raw)
        for field in ("task_id", "asset_id"):
            text = str(row.get(field) or "")
            if (
                not text
                or PurePosixPath(text).name != text
                or text in {".", ".."}
                or not text.replace("_", "a").replace("-", "a").isalnum()
            ):
                errors.append(
                    f"paired_target_construction_identity_invalid:{index}:{field}"
                )
        for field in (
            "task_freeze_digest",
            "registered_asset_receipt_digest",
            "replacement_asset_sha256",
            "native_import_probe_result_digest",
        ):
            if not _digest(row.get(field)):
                errors.append(
                    f"paired_target_construction_digest_invalid:{index}:{field}"
                )
        if row.get("native_simulator_import_qualified") is not True:
            errors.append(f"paired_target_construction_import_missing:{index}")
        evidence = row.get("evidence_receipts")
        if not isinstance(evidence, Mapping):
            errors.append(f"paired_target_construction_evidence_missing:{index}")
        else:
            expected = {
                "task_freeze": row.get("task_freeze_digest"),
                "registered_asset": row.get("registered_asset_receipt_digest"),
                "registered_usd": row.get("replacement_asset_sha256"),
                "native_import_probe": row.get(
                    "native_import_probe_result_digest"
                ),
            }
            for role, digest in expected.items():
                record = evidence.get(role)
                canonical = (
                    record.get("canonical_digest")
                    if isinstance(record, Mapping)
                    else None
                )
                if (
                    not isinstance(record, Mapping)
                    or not str(record.get("path") or "")
                    or not isinstance(record.get("size_bytes"), int)
                    or isinstance(record.get("size_bytes"), bool)
                    or int(record.get("size_bytes") or 0) <= 0
                    or not _digest(record.get("sha256"))
                    or (
                        role == "registered_usd"
                        and record.get("sha256") != digest
                    )
                    or (
                        role != "registered_usd"
                        and canonical != digest
                    )
                ):
                    errors.append(
                        f"paired_target_construction_evidence_invalid:{index}:{role}"
                    )
        normalized.append(row)
    asset_ids = [str(row.get("asset_id") or "") for row in normalized]
    task_ids = [str(row.get("task_id") or "") for row in normalized]
    if len(asset_ids) != len(set(asset_ids)) or len(task_ids) != len(set(task_ids)):
        errors.append("paired_target_construction_identity_not_unique")
    for field in (
        "native_camera_readback_qualified",
        "native_reachability_qualified",
        "controls_executed",
        "learned_policies_executed",
    ):
        if payload.get(field) is not False:
            errors.append(f"paired_target_construction_boundary_invalid:{field}")
    payload["bindings"] = sorted(normalized, key=lambda row: str(row.get("asset_id")))
    if payload.get("construction_digest") != canonical_digest(
        payload, digest_field="construction_digest"
    ):
        errors.append("paired_target_construction_digest_invalid")
    if errors:
        raise PairedTargetNativeConstructionBindingsError(errors)
    return payload


def materialize_paired_target_native_construction_bindings(
    *, manipulation_preflight_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Join paired inputs after import and before or after Arena compilation."""

    source = Path(manipulation_preflight_path).expanduser().resolve()
    try:
        manipulation = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_manipulation_preflight_invalid"]
        ) from exc
    if (
        source.is_symlink()
        or not isinstance(manipulation, dict)
        or manipulation.get("schema_version") != MANIPULATION_PREFLIGHT_SCHEMA
        or manipulation.get("receipt_digest")
        != canonical_digest(manipulation, digest_field="receipt_digest")
        or manipulation.get("native_import_qualified") is not True
        or manipulation.get("native_construction_bindings_ready") is not True
        or manipulation.get("blockers") != []
        or manipulation.get("native_reachability_executed") is not False
        or manipulation.get("controls_executed") is not False
        or manipulation.get("learned_policies_executed") is not False
    ):
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_manipulation_preflight_invalid"]
        )
    manipulation_tasks = manipulation.get("tasks")
    if (
        not isinstance(manipulation_tasks, list)
        or not 1 <= len(manipulation_tasks) <= MAX_REPLACEMENT_OBJECTS
        or any(not isinstance(row, Mapping) for row in manipulation_tasks)
    ):
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_manipulation_preflight_invalid"]
        )
    phase = manipulation.get("preflight_phase")
    if phase == "pre_arena":
        expected_pending = sorted(
            f"{row.get('task_id')}:native_task_arena_packet_request_missing"
            for row in manipulation_tasks
        )
        if (
            manipulation.get("status") != "ready_for_native_construction_bindings"
            or manipulation.get("pending_requirements") != expected_pending
            or any(
                row.get("native_construction_binding_ready") is not True
                or row.get("native_task_arena_request") is not None
                or row.get("pending_requirements")
                != ["native_task_arena_packet_request_missing"]
                for row in manipulation_tasks
            )
        ):
            raise PairedTargetNativeConstructionBindingsError(
                ["paired_target_construction_manipulation_preflight_invalid"]
            )
    elif phase == "arena_packet":
        if (
            manipulation.get("status")
            != "ready_for_native_arena_packet_materialization"
            or manipulation.get("pending_requirements") != []
            or any(
                row.get("native_construction_binding_ready") is not True
                or not isinstance(row.get("native_task_arena_request"), Mapping)
                for row in manipulation_tasks
            )
        ):
            raise PairedTargetNativeConstructionBindingsError(
                ["paired_target_construction_manipulation_preflight_invalid"]
            )
    else:
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_manipulation_preflight_invalid"]
        )
    _, preflight = _read_bound_record(
        manipulation.get("paired_target_preflight"),
        role="paired_preflight",
        expected_schema=PAIRED_PREFLIGHT_SCHEMA,
        digest_field="receipt_digest",
    )
    _, native_import = _read_bound_record(
        manipulation.get("native_import_result"),
        role="native_import",
        expected_schema=NATIVE_IMPORT_SCHEMA,
        digest_field="result_digest",
    )
    if (
        preflight.get("scene_id") != manipulation.get("scene_id")
        or native_import.get("scene_id") != manipulation.get("scene_id")
        or native_import.get("status") != "completed"
        or native_import.get("all_replacements_import_qualified") is not True
        or native_import.get("candidate_policy_queried") is not False
    ):
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_source_mismatch"]
        )
    preflight_by_task = {row["task_id"]: row for row in preflight["tasks"]}
    import_by_task = {row["task_id"]: row for row in native_import["replacements"]}
    if (
        manipulation.get("replacement_object_count")
        != len(manipulation.get("tasks") or [])
        or preflight.get("replacement_object_count") != len(preflight_by_task)
        or native_import.get("replacement_count") != len(import_by_task)
        or manipulation.get("task_freeze_set_digest") is None
    ):
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_task_set_mismatch"]
        )
    bindings: list[dict[str, Any]] = []
    for index, task in enumerate(manipulation.get("tasks") or []):
        task_id = str(task.get("task_id") or "")
        asset_id = str(task.get("asset_id") or "")
        parent = preflight_by_task.get(task_id)
        imported = import_by_task.get(task_id)
        if (
            not isinstance(parent, Mapping)
            or not isinstance(imported, Mapping)
            or parent.get("asset_id") != asset_id
            or imported.get("asset_id") != asset_id
            or imported.get("native_simulator_import_qualified") is not True
            or imported.get("blockers") != []
        ):
            raise PairedTargetNativeConstructionBindingsError(
                [f"paired_target_construction_task_mismatch:{index}"]
            )
        task_freeze_path, task_freeze = _read_bound_record(
            task.get("task_freeze"),
            role=f"task_freeze:{index}",
            expected_schema="dual_task_task_freeze.v1",
            digest_field="task_freeze_digest",
        )
        registered_path, registered = _read_bound_record(
            parent.get("registered_replacement_asset_receipt"),
            role=f"registered_asset:{index}",
            expected_schema=REGISTERED_ASSET_SCHEMA,
            digest_field="receipt_digest",
        )
        usd_record = parent.get("registered_replacement_usd")
        usd_path = Path(str((usd_record or {}).get("path") or "")).expanduser().resolve()
        if (
            not isinstance(usd_record, Mapping)
            or usd_path.is_symlink()
            or not usd_path.is_file()
            or usd_path.stat().st_size != usd_record.get("size_bytes")
            or _sha256(usd_path) != usd_record.get("sha256")
            or registered.get("output_usd") != dict(usd_record)
            or registered.get("task_id") != task_id
            or registered.get("asset_id") != asset_id
            or registered.get("task_freeze_digest")
            != task_freeze.get("task_freeze_digest")
        ):
            raise PairedTargetNativeConstructionBindingsError(
                [f"paired_target_construction_registered_asset_mismatch:{index}"]
            )
        probe_relative = str(imported.get("probe_result_path") or "")
        probe_path = (Path(manipulation["native_import_result"]["path"]).parent / probe_relative).resolve()
        if (
            not probe_relative
            or Path(probe_relative).is_absolute()
            or ".." in Path(probe_relative).parts
            or probe_path.is_symlink()
            or not probe_path.is_file()
            or _sha256(probe_path) != imported.get("probe_result_sha256")
        ):
            raise PairedTargetNativeConstructionBindingsError(
                [f"paired_target_construction_native_import_probe_invalid:{index}"]
            )
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PairedTargetNativeConstructionBindingsError(
                [f"paired_target_construction_native_import_probe_invalid:{index}"]
            ) from exc
        if (
            not isinstance(probe, Mapping)
            or probe.get("result_digest") != imported.get("probe_result_digest")
            or probe.get("result_digest")
            != canonical_digest(probe, digest_field="result_digest")
            or probe.get("asset_id") != asset_id
            or probe.get("native_simulator_import_qualified") is not True
            or probe.get("candidate_policy_queried") is not False
        ):
            raise PairedTargetNativeConstructionBindingsError(
                [f"paired_target_construction_native_import_probe_invalid:{index}"]
            )
        bindings.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_digest": task_freeze["task_freeze_digest"],
                "registered_asset_receipt_digest": registered["receipt_digest"],
                "replacement_asset_sha256": usd_record["sha256"],
                "native_import_probe_result_digest": imported["probe_result_digest"],
                "native_simulator_import_qualified": True,
                "evidence_receipts": {
                    "task_freeze": _file_record(
                        task_freeze_path,
                        schema_version="dual_task_task_freeze.v1",
                        canonical_digest_value=task_freeze["task_freeze_digest"],
                    ),
                    "registered_asset": _file_record(
                        registered_path,
                        schema_version=REGISTERED_ASSET_SCHEMA,
                        canonical_digest_value=registered["receipt_digest"],
                    ),
                    "registered_usd": _file_record(usd_path),
                    "native_import_probe": _file_record(
                        probe_path,
                        schema_version=str(probe.get("schema_version") or ""),
                        canonical_digest_value=probe["result_digest"],
                    ),
                },
            }
        )
    if (
        not 1 <= len(bindings) <= MAX_REPLACEMENT_OBJECTS
        or set(preflight_by_task) != {row["task_id"] for row in bindings}
        or set(import_by_task) != {row["task_id"] for row in bindings}
    ):
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_task_set_mismatch"]
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "paired_targets_admitted_for_native_construction",
        "scene_id": manipulation["scene_id"],
        "task_freeze_set_digest": manipulation["task_freeze_set_digest"],
        "replacement_object_count": len(bindings),
        "collision_scene": preflight["collision_scene"],
        "paired_target_preflight": manipulation["paired_target_preflight"],
        "native_import_result": manipulation["native_import_result"],
        "manipulation_preflight": _file_record(
            source,
            schema_version=MANIPULATION_PREFLIGHT_SCHEMA,
            canonical_digest_value=manipulation["receipt_digest"],
        ),
        "bindings": sorted(bindings, key=lambda row: row["asset_id"]),
        "native_camera_readback_qualified": False,
        "native_reachability_qualified": False,
        "controls_executed": False,
        "learned_policies_executed": False,
        "claim_boundary": (
            "exact_paired_scene_and_replacement_bytes_admitted_for_native_"
            "construction_only;camera_reachability_contact_controls_policy_and_"
            "physical_claims_unproven"
        ),
        "construction_digest": "",
    }
    payload["construction_digest"] = canonical_digest(
        payload, digest_field="construction_digest"
    )
    payload = validate_paired_target_native_construction_bindings(payload)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetNativeConstructionBindingsError(
            ["paired_target_construction_output_exists"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = [
    "PairedTargetNativeConstructionBindingsError",
    "SCHEMA_VERSION",
    "materialize_paired_target_native_construction_bindings",
    "validate_paired_target_native_construction_bindings",
]
