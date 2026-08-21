"""Admit exact task-object bytes to native GPU construction.

This module is the single seam between authored/registered task geometry and a
paid native construction launch.  It deliberately has two phases:

``prepare_native_task_execution_candidate`` is provider-zero.  It verifies the
exact registered USD bytes and records every dynamic collider's intended GPU
cooking representation and dimensions.

``seal_native_task_execution_admission`` joins that immutable candidate to an
exact-runtime physics-cook/step result and the final construction packet.  A
construction launcher consumes only the sealed admission receipt, never a
weaker "USD opened" result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_gpu_collision_qualification import (
    NativeTaskGpuCollisionQualificationError,
    audit_native_task_gpu_collisions,
)


CANDIDATE_SCHEMA_VERSION = "native_task_execution_candidate.v1"
ADMISSION_SCHEMA_VERSION = "native_task_execution_admission.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "paired_target_native_import_runtime_result.v1"
PACKET_RECEIPT_SCHEMA_VERSION = "native_task_arena_packet_receipt.v1"
SCENE_PLAN_SCHEMA_VERSION = "native_task_arena_scene_plan.v1"


class NativeTaskExecutionAdmissionError(ValueError):
    """Stable fail-closed preparation or admission failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _identifier(value: Any) -> str:
    text = str(value or "")
    if (
        not text
        or PurePosixPath(text).name != text
        or text in {".", ".."}
        or not text.replace("_", "a").replace("-", "a").isalnum()
    ):
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_candidate_identity_invalid"]
        )
    return text


def _runtime_image(value: Any) -> str:
    text = str(value or "")
    marker = "@sha256:"
    if marker not in text or len(text.rsplit(marker, 1)[1]) != 64:
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_candidate_runtime_image_unpinned"]
        )
    return text


def prepare_native_task_execution_candidate(
    *,
    scene_id: str,
    runtime_image: str,
    assets: Sequence[Mapping[str, Any]],
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Qualify exact registered assets before any provider authority is issued."""

    if not str(scene_id):
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_candidate_scene_invalid"]
        )
    pinned_image = _runtime_image(runtime_image)
    if not assets:
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_candidate_assets_missing"]
        )
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(assets):
        try:
            task_id = _identifier(raw.get("task_id"))
            asset_id = _identifier(raw.get("asset_id"))
        except NativeTaskExecutionAdmissionError as exc:
            errors.extend(f"{error}:{index}" for error in exc.errors)
            continue
        identity = (task_id, asset_id)
        if identity in identities:
            errors.append(f"native_task_execution_candidate_identity_duplicate:{index}")
            continue
        identities.add(identity)
        path = Path(str(raw.get("path") or "")).expanduser().resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in {".usd", ".usda", ".usdc"}
            or path.stat().st_size != raw.get("size_bytes")
            or _sha256(path) != raw.get("sha256")
        ):
            errors.append(f"native_task_execution_candidate_asset_invalid:{index}")
            continue
        try:
            audit = audit_native_task_gpu_collisions(path)
        except NativeTaskGpuCollisionQualificationError as exc:
            errors.extend(exc.errors)
            continue
        if audit.get("status") != "qualified" or audit.get("blockers") != []:
            errors.extend(
                str(error) for error in audit.get("blockers") or [
                    f"native_task_execution_candidate_gpu_collision_invalid:{index}"
                ]
            )
            continue
        collision_intent = {
            "schema_version": "native_task_collision_intent.v1",
            "asset_sha256": audit["usd_sha256"],
            "dynamic_collision_prim_count": audit["dynamic_collision_prim_count"],
            "dynamic_mesh_colliders": audit["dynamic_mesh_colliders"],
            "dynamic_primitive_colliders": audit["dynamic_primitive_colliders"],
            "maximum_convex_hull_aspect_ratio": audit[
                "maximum_convex_hull_aspect_ratio"
            ],
            "intent_digest": "",
        }
        collision_intent["intent_digest"] = canonical_digest(
            collision_intent, digest_field="intent_digest"
        )
        rows.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "registered_asset": {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": audit["usd_sha256"],
                },
                "registered_static_qualification_digest": raw.get(
                    "registered_static_qualification_digest"
                ),
                "collision_intent": collision_intent,
                "provider_zero_qualification_completed": True,
            }
        )
    if errors:
        raise NativeTaskExecutionAdmissionError(errors)
    rows.sort(key=lambda row: (row["task_id"], row["asset_id"]))
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": "prepared_for_exact_runtime_gpu_cook",
        "scene_id": str(scene_id),
        "runtime_image": pinned_image,
        "asset_count": len(rows),
        "assets": rows,
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "native_gpu_cooking_readback_still_required": True,
        "construction_authorized": False,
        "physical_equivalence_claimed": False,
        "candidate_digest": "",
    }
    candidate["candidate_digest"] = canonical_digest(
        candidate, digest_field="candidate_digest"
    )
    validated = validate_native_task_execution_candidate(candidate, reopen_files=True)
    if destination is not None:
        output = Path(destination).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise NativeTaskExecutionAdmissionError(
                ["native_task_execution_candidate_output_exists"]
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, validated)
    return validated


def validate_native_task_execution_candidate(
    value: Mapping[str, Any], *, reopen_files: bool = False
) -> dict[str, Any]:
    """Validate the provider-zero candidate and optionally re-audit source bytes."""

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_candidate_invalid"]
        ) from exc
    errors: list[str] = []
    rows = payload.get("assets")
    if (
        payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or payload.get("status") != "prepared_for_exact_runtime_gpu_cook"
        or not str(payload.get("scene_id") or "")
        or not isinstance(rows, list)
        or not rows
        or payload.get("asset_count") != len(rows)
        or payload.get("provider_mutation_performed") is not False
        or payload.get("paid_resource_allocated") is not False
        or payload.get("native_gpu_cooking_readback_still_required") is not True
        or payload.get("construction_authorized") is not False
        or payload.get("physical_equivalence_claimed") is not False
    ):
        errors.append("native_task_execution_candidate_invalid")
        rows = rows if isinstance(rows, list) else []
    try:
        _runtime_image(payload.get("runtime_image"))
    except NativeTaskExecutionAdmissionError as exc:
        errors.extend(exc.errors)
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"native_task_execution_candidate_asset_invalid:{index}")
            continue
        identity = (str(row.get("task_id") or ""), str(row.get("asset_id") or ""))
        record = row.get("registered_asset")
        intent = row.get("collision_intent")
        if identity in seen:
            errors.append(f"native_task_execution_candidate_identity_duplicate:{index}")
        seen.add(identity)
        if (
            not all(identity)
            or not isinstance(record, Mapping)
            or not _digest(record.get("sha256"))
            or not isinstance(record.get("size_bytes"), int)
            or isinstance(record.get("size_bytes"), bool)
            or int(record.get("size_bytes") or 0) <= 0
            or not isinstance(intent, Mapping)
            or intent.get("schema_version") != "native_task_collision_intent.v1"
            or intent.get("asset_sha256") != record.get("sha256")
            or not isinstance(intent.get("dynamic_collision_prim_count"), int)
            or int(intent.get("dynamic_collision_prim_count") or 0) <= 0
            or intent.get("intent_digest")
            != canonical_digest(dict(intent), digest_field="intent_digest")
            or row.get("provider_zero_qualification_completed") is not True
            or not _digest(row.get("registered_static_qualification_digest"))
        ):
            errors.append(f"native_task_execution_candidate_asset_invalid:{index}")
            continue
        if reopen_files:
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record.get("size_bytes")
                or _sha256(path) != record.get("sha256")
            ):
                errors.append(f"native_task_execution_candidate_asset_mutated:{index}")
                continue
            audit = audit_native_task_gpu_collisions(path)
            if (
                audit.get("status") != "qualified"
                or audit.get("dynamic_collision_prim_count")
                != intent.get("dynamic_collision_prim_count")
                or audit.get("dynamic_mesh_colliders")
                != intent.get("dynamic_mesh_colliders")
                or audit.get("dynamic_primitive_colliders")
                != intent.get("dynamic_primitive_colliders")
            ):
                errors.append(
                    f"native_task_execution_candidate_collision_intent_mismatch:{index}"
                )
    if payload.get("candidate_digest") != canonical_digest(
        payload, digest_field="candidate_digest"
    ):
        errors.append("native_task_execution_candidate_digest_invalid")
    if errors:
        raise NativeTaskExecutionAdmissionError(errors)
    return payload


def seal_native_task_execution_admission(
    *,
    candidate: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    packet_receipt: Mapping[str, Any],
    scene_plan: Mapping[str, Any],
    task_id: str,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Join local intent, exact-runtime cooking, and the final packet."""

    prepared = validate_native_task_execution_candidate(candidate)
    errors: list[str] = []
    if (
        runtime_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or runtime_result.get("status") != "completed"
        or runtime_result.get("scene_id") != prepared.get("scene_id")
        or runtime_result.get("execution_candidate_digest")
        != prepared.get("candidate_digest")
        or runtime_result.get("all_replacements_import_qualified") is not True
        or runtime_result.get("native_gpu_physics_qualified") is not True
        or runtime_result.get("result_digest")
        != canonical_digest(dict(runtime_result), digest_field="result_digest")
        or runtime_result.get("blockers") != []
    ):
        errors.append("native_task_execution_runtime_result_invalid")
    if (
        packet_receipt.get("schema_version") != PACKET_RECEIPT_SCHEMA_VERSION
        or packet_receipt.get("status") != "construction_packet_completed"
        or packet_receipt.get("scene_id") != prepared.get("scene_id")
        or packet_receipt.get("task_id") != task_id
        or packet_receipt.get("receipt_digest")
        != canonical_digest(dict(packet_receipt), digest_field="receipt_digest")
    ):
        errors.append("native_task_execution_packet_receipt_invalid")
    if (
        scene_plan.get("schema_version") != SCENE_PLAN_SCHEMA_VERSION
        or scene_plan.get("scene_id") != prepared.get("scene_id")
        or scene_plan.get("task_id") != task_id
        or scene_plan.get("plan_digest")
        != canonical_digest(dict(scene_plan), digest_field="plan_digest")
        or packet_receipt.get("arena_scene_plan_digest")
        != scene_plan.get("plan_digest")
    ):
        errors.append("native_task_execution_scene_plan_invalid")
    candidate_by_task = {
        str(row.get("task_id") or ""): row for row in prepared.get("assets") or []
    }
    runtime_by_task = {
        str(row.get("task_id") or ""): row
        for row in runtime_result.get("replacements") or []
        if isinstance(row, Mapping)
    }
    asset = candidate_by_task.get(task_id)
    runtime = runtime_by_task.get(task_id)
    if (
        not isinstance(asset, Mapping)
        or not isinstance(runtime, Mapping)
        or runtime.get("asset_id") != asset.get("asset_id")
        or runtime.get("native_simulator_import_qualified") is not True
        or runtime.get("native_gpu_physics_qualified") is not True
        or runtime.get("collision_intent_digest")
        != (asset.get("collision_intent") or {}).get("intent_digest")
        or runtime.get("blockers") != []
    ):
        errors.append("native_task_execution_task_runtime_invalid")
    source_binding = next(
        (
            row
            for row in packet_receipt.get("source_bindings") or []
            if isinstance(row, Mapping) and row.get("asset_id") == (asset or {}).get("asset_id")
        ),
        None,
    )
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("staged_sha256")
        != ((asset or {}).get("registered_asset") or {}).get("sha256")
    ):
        errors.append("native_task_execution_packet_asset_mismatch")
    if errors:
        raise NativeTaskExecutionAdmissionError(errors)
    admission: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": "admitted_for_native_gpu_construction",
        "scene_id": prepared["scene_id"],
        "task_id": task_id,
        "asset_id": asset["asset_id"],
        "registered_asset_sha256": asset["registered_asset"]["sha256"],
        "runtime_image": prepared["runtime_image"],
        "execution_candidate_digest": prepared["candidate_digest"],
        "collision_intent_digest": asset["collision_intent"]["intent_digest"],
        "native_runtime_result_digest": runtime_result["result_digest"],
        "packet_receipt_digest": packet_receipt["receipt_digest"],
        "scene_plan_digest": scene_plan["plan_digest"],
        "native_gpu_cooking_readback_qualified": True,
        "native_simulation_step_qualified": True,
        "construction_authorized": True,
        "controls_executed": False,
        "learned_policy_executed": False,
        "physical_equivalence_claimed": False,
        "receipt_digest": "",
    }
    admission["receipt_digest"] = canonical_digest(
        admission, digest_field="receipt_digest"
    )
    validated = validate_native_task_execution_admission(admission)
    if destination is not None:
        output = Path(destination).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise NativeTaskExecutionAdmissionError(
                ["native_task_execution_admission_output_exists"]
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, validated)
    return validated


def validate_native_task_execution_admission(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a sealed final-packet construction admission receipt."""

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_admission_invalid"]
        ) from exc
    if (
        payload.get("schema_version") != ADMISSION_SCHEMA_VERSION
        or payload.get("status") != "admitted_for_native_gpu_construction"
        or not all(
            str(payload.get(field) or "")
            for field in ("scene_id", "task_id", "asset_id")
        )
        or any(
            not _digest(payload.get(field))
            for field in (
                "registered_asset_sha256",
                "execution_candidate_digest",
                "collision_intent_digest",
                "native_runtime_result_digest",
                "packet_receipt_digest",
                "scene_plan_digest",
                "receipt_digest",
            )
        )
        or payload.get("native_gpu_cooking_readback_qualified") is not True
        or payload.get("native_simulation_step_qualified") is not True
        or payload.get("construction_authorized") is not True
        or payload.get("controls_executed") is not False
        or payload.get("learned_policy_executed") is not False
        or payload.get("physical_equivalence_claimed") is not False
        or payload.get("receipt_digest")
        != canonical_digest(payload, digest_field="receipt_digest")
    ):
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_admission_invalid"]
        )
    _runtime_image(payload.get("runtime_image"))
    return payload


def require_native_task_execution_admission(
    packet_dir: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    """Reopen the final packet and require an exact matching admission receipt."""

    root = Path(packet_dir).expanduser().resolve()
    try:
        admission = json.loads(
            (root / "native_task_execution_admission.v1.json").read_text(
                encoding="utf-8"
            )
        )
        packet = json.loads(
            (root / "native_task_arena_packet_receipt.v1.json").read_text(
                encoding="utf-8"
            )
        )
        plan = json.loads(
            (root / "native_task_arena_scene_plan.v1.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_admission_missing"]
        ) from exc
    validated = validate_native_task_execution_admission(admission)
    errors: list[str] = []
    if (
        packet.get("receipt_digest")
        != canonical_digest(packet, digest_field="receipt_digest")
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
        or validated.get("packet_receipt_digest") != packet.get("receipt_digest")
        or validated.get("scene_plan_digest") != plan.get("plan_digest")
        or packet.get("arena_scene_plan_digest") != plan.get("plan_digest")
        or validated.get("scene_id") != packet.get("scene_id")
        or validated.get("task_id") != packet.get("task_id")
    ):
        errors.append("native_task_execution_admission_packet_mismatch")
    if expected_scene_id is not None and validated.get("scene_id") != expected_scene_id:
        errors.append("native_task_execution_admission_scene_mismatch")
    if expected_task_id is not None and validated.get("task_id") != expected_task_id:
        errors.append("native_task_execution_admission_task_mismatch")
    if errors:
        raise NativeTaskExecutionAdmissionError(errors)
    return validated


def native_task_execution_admission_required(packet_dir: str | Path) -> bool:
    """Return whether this packet uses the execution-admission construction schema."""

    root = Path(packet_dir).expanduser().resolve()
    try:
        request = json.loads(
            (root / "native_task_arena_packet_request.v1.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskExecutionAdmissionError(
            ["native_task_execution_packet_request_invalid"]
        ) from exc
    return (
        (request.get("construction_bindings") or {}).get("schema_version")
        == "paired_target_native_construction_bindings.v2"
    )


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "NativeTaskExecutionAdmissionError",
    "prepare_native_task_execution_candidate",
    "native_task_execution_admission_required",
    "require_native_task_execution_admission",
    "seal_native_task_execution_admission",
    "validate_native_task_execution_admission",
    "validate_native_task_execution_candidate",
]
