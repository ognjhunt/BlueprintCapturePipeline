"""Apply a sealed semantic-frame registration to a composed replacement USD.

This is deterministic pose composition, not CAD generation. The source visual
composition remains the geometry/material authority and the registration
receipt remains the orientation authority. The resulting USD is the single
visual+collision+joint asset imported by native runtimes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .agent_cad_graph_visual_composition import (
    COMPOSITION_SCHEMA_VERSION,
    validate_agent_cad_visual_composition,
)
from .decision_evidence_contracts import canonical_digest
from .replacement_asset_frame_registration import (
    validate_replacement_asset_frame_registration,
)


SCHEMA_VERSION = "registered_replacement_asset.v1"


class RegisteredReplacementAssetError(ValueError):
    """Stable fail-closed registered replacement failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RegisteredReplacementAssetError("registered_replacement_file_invalid")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegisteredReplacementAssetError(code) from exc
    if candidate.is_symlink() or not isinstance(value, dict):
        raise RegisteredReplacementAssetError(code)
    return candidate, value


def _matrix_close(left: Any, right: Any, tolerance: float = 1e-7) -> bool:
    try:
        return all(
            abs(float(left[row][column]) - float(right[row][column])) <= tolerance
            for row in range(4)
            for column in range(4)
        )
    except (IndexError, TypeError, ValueError):
        return False


def validate_registered_replacement_asset(
    value: dict[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    payload = json.loads(json.dumps(value, allow_nan=False))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "registered_replacement_materialized_pending_native_import"
        or payload.get("receipt_digest") != canonical_digest(payload, digest_field="receipt_digest")
        or payload.get("agent_authored_display_colors_preserved") is not True
        or payload.get("generated_texture_maps_present") is not False
        or payload.get("neutral_fallback_present") is not False
        or payload.get("native_import_qualified") is not False
        or payload.get("deterministic_pose_composition_only") is not True
        or payload.get("geometry_generated_or_modified") is not False
        or payload.get("physical_equivalence_proven") is not False
    ):
        raise RegisteredReplacementAssetError("registered_replacement_receipt_invalid")
    output = payload.get("output_usd")
    registration = payload.get("frame_registration")
    composition = payload.get("visual_composition_receipt")
    if not all(isinstance(record, dict) for record in (output, registration, composition)):
        raise RegisteredReplacementAssetError("registered_replacement_receipt_invalid")
    if verify_files:
        for record in (output, registration, composition):
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            if _record(path) != {key: record[key] for key in ("path", "size_bytes", "sha256")}:
                raise RegisteredReplacementAssetError("registered_replacement_bound_file_invalid")
        try:
            registration_value = json.loads(Path(registration["path"]).read_text(encoding="utf-8"))
            composition_value = json.loads(Path(composition["path"]).read_text(encoding="utf-8"))
            registration_value = validate_replacement_asset_frame_registration(registration_value)
            composition_value = validate_agent_cad_visual_composition(composition_value)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RegisteredReplacementAssetError(
                "registered_replacement_bound_receipt_invalid"
            ) from exc
        if (
            registration_value.get("registration_digest") != registration.get("registration_digest")
            or composition_value.get("receipt_digest") != composition.get("receipt_digest")
            or any(
                payload.get(field) != payload_value.get(field)
                for payload_value in (registration_value, composition_value)
                for field in ("scene_id", "task_id", "asset_id")
            )
        ):
            raise RegisteredReplacementAssetError("registered_replacement_bound_receipt_invalid")
    return payload


def materialize_registered_replacement_asset(
    *,
    visual_composition_receipt_path: str | Path,
    frame_registration_path: str | Path,
    output_usd_path: str | Path,
    output_receipt_path: str | Path,
) -> dict[str, Any]:
    """Materialize one exact registered visual+physics replacement asset."""

    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RegisteredReplacementAssetError("registered_replacement_usd_runtime_missing") from exc
    composition_path, composition = _read(
        visual_composition_receipt_path, "registered_replacement_composition_invalid"
    )
    registration_path, registration = _read(
        frame_registration_path, "registered_replacement_registration_invalid"
    )
    try:
        composition = validate_agent_cad_visual_composition(composition)
        registration = validate_replacement_asset_frame_registration(registration)
    except Exception as exc:
        raise RegisteredReplacementAssetError("registered_replacement_input_invalid") from exc
    if composition.get("schema_version") != COMPOSITION_SCHEMA_VERSION or any(
        composition.get(field) != registration.get(field)
        for field in ("scene_id", "task_id", "asset_id")
    ):
        raise RegisteredReplacementAssetError("registered_replacement_identity_mismatch")
    source = Path(composition["output_usd"]["path"]).expanduser().resolve()
    if _record(source) != composition["output_usd"]:
        raise RegisteredReplacementAssetError("registered_replacement_source_invalid")
    destination = Path(output_usd_path).expanduser().resolve()
    receipt_path = Path(output_receipt_path).expanduser().resolve()
    if any(path.exists() or path.is_symlink() for path in (destination, receipt_path)):
        raise RegisteredReplacementAssetError("registered_replacement_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None or not stage.Export(str(destination)):
        raise RegisteredReplacementAssetError("registered_replacement_export_failed")
    registered = Usd.Stage.Open(str(destination), load=Usd.Stage.LoadAll)
    root = registered.GetDefaultPrim() if registered is not None else None
    if registered is None or not root or not root.IsA(UsdGeom.Xform):
        raise RegisteredReplacementAssetError("registered_replacement_root_invalid")
    xform = UsdGeom.Xformable(root)
    before = xform.GetLocalTransformation()
    translation = before.ExtractTranslation()
    matrix = registration["T_observed_world_axes_from_asset_local_axes"]
    rotation_matrix = Gf.Matrix3d(*[float(matrix[r][c]) for r in range(3) for c in range(3)])
    registered_transform = Gf.Matrix4d(1.0)
    registered_transform.SetRotate(rotation_matrix)
    registered_transform.SetTranslateOnly(translation)
    xform.ClearXformOpOrder()
    # Clearing the op order deliberately leaves the authored properties in the
    # layer. Reusing a common translate/orient name can therefore collide with
    # an existing float-precision op even when it is no longer ordered. A
    # uniquely suffixed Matrix4d op is precision-stable for every source graph.
    xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "assetFrameRegistration").Set(
        registered_transform
    )
    root.SetCustomDataByKey(
        "blueprint:assetFrameRegistrationDigest", registration["registration_digest"]
    )
    root.SetCustomDataByKey("blueprint:identityOrientationAssumed", False)
    registered.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination), load=Usd.Stage.LoadAll)
    actual = UsdGeom.Xformable(reopened.GetDefaultPrim()).GetLocalTransformation()
    expected = registered_transform
    if not _matrix_close(actual, expected):
        raise RegisteredReplacementAssetError("registered_replacement_transform_mismatch")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "registered_replacement_materialized_pending_native_import",
        "scene_id": composition["scene_id"],
        "task_id": composition["task_id"],
        "asset_id": composition["asset_id"],
        "visual_composition_receipt": {
            **_record(composition_path),
            "receipt_digest": composition["receipt_digest"],
        },
        "frame_registration": {
            **_record(registration_path),
            "registration_digest": registration["registration_digest"],
        },
        "output_usd": _record(destination),
        "T_observed_world_axes_from_asset_local_axes": matrix,
        "source_root_translation_preserved": [float(value) for value in translation],
        "agent_authored_display_colors_preserved": True,
        "generated_texture_maps_present": False,
        "neutral_fallback_present": False,
        "native_import_qualified": False,
        "deterministic_pose_composition_only": True,
        "geometry_generated_or_modified": False,
        "physical_equivalence_proven": False,
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validate_registered_replacement_asset(result)
