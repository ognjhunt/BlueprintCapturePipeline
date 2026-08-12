"""Compose actual USD depth sweeps for one through five co-present assets.

The result is an image-space z-minimum of independently rasterized, exact USD
geometry.  It is deliberately narrower than simulator composition: it proves
only that the replacement silhouettes used by the Gaussian-removal seam are
co-present in the same calibrated camera/state cells.  It does not assert
native import, contact, material appearance, or physical behavior.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .articulated_usd_depth_sweep import GENERAL_DEPTH_SWEEP_SCHEMA
from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS


REQUEST_SCHEMA = "public_scene_replacement_depth_composition_request.v1"
COMPOSITION_SCHEMA = "public_scene_replacement_depth_composition.v1"


class ReplacementDepthCompositionError(ValueError):
    """Stable fail-closed errors for co-present replacement depth composition."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


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


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReplacementDepthCompositionError([code]) from exc
    if not isinstance(result, dict):
        raise ReplacementDepthCompositionError([code])
    return result


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplacementDepthCompositionError([code]) from exc
    if not isinstance(value, dict):
        raise ReplacementDepthCompositionError([code])
    return value


def _path(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ReplacementDepthCompositionError([code])
    return path


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_absolute(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise ReplacementDepthCompositionError([code])
    path = _path(str(value.get("path") or ""), code=code)
    if path.stat().st_size != value.get("size_bytes") or _sha256(path) != value.get("sha256"):
        raise ReplacementDepthCompositionError([code])
    return path


def _verify_relative(root: Path, value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise ReplacementDepthCompositionError([code])
    relative = str(value.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ReplacementDepthCompositionError([code])
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ReplacementDepthCompositionError([code])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ReplacementDepthCompositionError([code])
    return path


def build_replacement_depth_composition_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the non-outcome composition request for one through five assets."""

    request = _clone(value, code="replacement_depth_composition_request_not_json")
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("replacement_depth_composition_request_schema_invalid")
    if not str(request.get("task_id") or "").strip() or not _digest(
        request.get("task_freeze_digest")
    ):
        errors.append("replacement_depth_composition_task_identity_invalid")
    if not str(request.get("scored_task_asset_id") or "").strip():
        errors.append("replacement_depth_composition_scored_asset_invalid")
    if request.get("frozen_before_removal_execution") is not True:
        errors.append("replacement_depth_composition_not_frozen")
    if request.get("learned_policy_outcomes_accessed") is not False:
        errors.append("replacement_depth_composition_policy_outcome_leakage")
    if any(key in request for key in ("coverage_qualified", "inpainting_result_qualified")):
        errors.append("replacement_depth_composition_caller_outcome_forbidden")
    sweeps = request.get("input_sweep_manifest_paths")
    if not isinstance(sweeps, list) or not 1 <= len(sweeps) <= MAX_REPLACEMENT_OBJECTS:
        errors.append("replacement_depth_composition_input_count_invalid")
        sweeps = []
    normalized = [str(path or "").strip() for path in sweeps]
    if any(not path for path in normalized) or len(normalized) != len(set(normalized)):
        errors.append("replacement_depth_composition_input_paths_invalid")
    if errors:
        raise ReplacementDepthCompositionError(errors)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != request["request_digest"]:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_request_digest_mismatch"]
        )
    return request


def _cells_key(cells: Any) -> list[tuple[str, str]]:
    if not isinstance(cells, list) or not cells:
        raise ReplacementDepthCompositionError(["replacement_depth_composition_cells_invalid"])
    key: list[tuple[str, str]] = []
    for row in cells:
        if not isinstance(row, Mapping):
            raise ReplacementDepthCompositionError(["replacement_depth_composition_cells_invalid"])
        camera_id = str(row.get("camera_id") or "")
        cell_id = str(row.get("cell_id") or "")
        if not camera_id or not cell_id:
            raise ReplacementDepthCompositionError(["replacement_depth_composition_cells_invalid"])
        key.append((camera_id, cell_id))
    if len(key) != len(set(key)):
        raise ReplacementDepthCompositionError(["replacement_depth_composition_cells_invalid"])
    return key


def _load_sweep(path: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    manifest = _read_object(path, code="replacement_depth_composition_input_unreadable")
    if (
        manifest.get("schema_version") != GENERAL_DEPTH_SWEEP_SCHEMA
        or manifest.get("status") != "actual_usd_geometry_depth_rasterized"
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("actual_usd_geometry_depth_rasterized") is not True
        or manifest.get("caller_supplied_coverage_mask") is not False
        or isinstance(manifest.get("resolution_scale"), bool)
        or not isinstance(manifest.get("resolution_scale"), (int, float))
        or not 0.0 < float(manifest["resolution_scale"]) <= 1.0
        or not str(manifest.get("asset_id") or "")
        or not _digest(manifest.get("task_freeze_digest"))
        or not _digest(manifest.get("camera_contract_digest"))
        or not _digest(manifest.get("camera_rows_digest"))
        or manifest.get("scene_state_role")
        not in {"task_subject", "co_present_passive"}
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_input_manifest_invalid"]
        )
    arrays_path = _verify_relative(
        path.parent,
        manifest.get("arrays"),
        code="replacement_depth_composition_input_array_invalid",
    )
    try:
        depth = np.asarray(np.load(arrays_path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_input_array_unreadable"]
        ) from exc
    cells = _cells_key(manifest.get("cells"))
    if (
        depth.ndim != 3
        or depth.shape[0] != len(cells)
        or depth.shape[1] <= 0
        or depth.shape[2] <= 0
        or np.any(np.isnan(depth))
        or np.any(np.isneginf(depth))
        or np.any((np.isfinite(depth)) & (depth <= 0.0))
        or manifest.get("depth_dimensions") != [int(depth.shape[2]), int(depth.shape[1])]
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_input_array_invalid"]
        )
    return manifest, depth, {
        **_input_record(path),
        "manifest_digest": manifest["manifest_digest"],
    }


def materialize_replacement_depth_composition(
    *, request_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Combine exact input sweeps with per-pixel nearest-depth composition."""

    request_file = _path(
        request_path, code="replacement_depth_composition_request_missing"
    )
    request = build_replacement_depth_composition_request(
        _read_object(request_file, code="replacement_depth_composition_request_unreadable")
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_output_not_empty"]
        )
    loaded: list[tuple[dict[str, Any], np.ndarray, dict[str, Any]]] = []
    for value in request["input_sweep_manifest_paths"]:
        loaded.append(
            _load_sweep(
                _path(value, code="replacement_depth_composition_input_manifest_missing")
            )
        )
    asset_ids = [str(manifest["asset_id"]) for manifest, _depth, _record in loaded]
    if len(asset_ids) != len(set(asset_ids)):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_asset_ids_invalid"]
        )
    if request["scored_task_asset_id"] not in asset_ids:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_scored_asset_not_in_inputs"]
        )
    scored = next(
        (item for item in loaded if item[0]["asset_id"] == request["scored_task_asset_id"]),
        None,
    )
    assert scored is not None
    if scored[0]["task_freeze_digest"] != request["task_freeze_digest"]:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_scored_task_freeze_mismatch"]
        )
    if (
        scored[0].get("scene_state_role") != "task_subject"
        or any(
            manifest.get("scene_state_role") != "co_present_passive"
            for manifest, _depth, _record in loaded
            if manifest["asset_id"] != request["scored_task_asset_id"]
        )
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_scene_state_roles_invalid"]
        )
    reference_manifest, reference_depth, _reference_record = loaded[0]
    reference_cells = _cells_key(reference_manifest["cells"])
    reference_camera_contract = reference_manifest["camera_contract_digest"]
    reference_camera_rows = reference_manifest["camera_rows_digest"]
    reference_resolution_scale = float(reference_manifest["resolution_scale"])
    for manifest, depth, _record_value in loaded[1:]:
        if (
            _cells_key(manifest["cells"]) != reference_cells
            or depth.shape != reference_depth.shape
            or manifest["camera_contract_digest"] != reference_camera_contract
            or manifest["camera_rows_digest"] != reference_camera_rows
            or float(manifest["resolution_scale"]) != reference_resolution_scale
        ):
            raise ReplacementDepthCompositionError(
                ["replacement_depth_composition_input_cell_or_camera_mismatch"]
            )
    output.mkdir(parents=True, exist_ok=True)
    composed = np.minimum.reduce(np.stack([depth for _manifest, depth, _record in loaded]))
    arrays_path = output / "replacement_depth_composition.npy"
    np.save(arrays_path, composed.astype(np.float32), allow_pickle=False)
    scored_manifest = scored[0]
    receipt: dict[str, Any] = {
        "schema_version": COMPOSITION_SCHEMA,
        "status": "co_present_replacement_depth_rasterized",
        "request": {
            **_input_record(request_file),
            "request_digest": request["request_digest"],
        },
        "task_id": request["task_id"],
        "task_freeze_digest": request["task_freeze_digest"],
        "scored_task_asset_id": request["scored_task_asset_id"],
        "replacement_asset_ids": sorted(asset_ids),
        "input_sweeps": [
            {
                **record,
                "asset_id": manifest["asset_id"],
                "task_freeze_digest": manifest["task_freeze_digest"],
                "finite_depth_pixel_count_by_cell": [
                    int(np.isfinite(cell_depth).sum()) for cell_depth in _depth
                ],
                "visible_in_any_composed_camera": bool(np.isfinite(_depth).any()),
            }
            for manifest, _depth, record in loaded
        ],
        "camera_contract_digest": reference_camera_contract,
        "camera_rows_digest": reference_camera_rows,
        "cells": scored_manifest["cells"],
        "state_cell_count": len(reference_cells),
        "camera_count": len({camera_id for camera_id, _cell_id in reference_cells}),
        "resolution_scale": reference_resolution_scale,
        "depth_dimensions": [int(composed.shape[2]), int(composed.shape[1])],
        "finite_depth_pixel_count_by_cell": [
            int(np.isfinite(depth).sum()) for depth in composed
        ],
        "arrays": _record(arrays_path, output),
        "actual_usd_geometry_depth_rasterized": True,
        "actual_composed_depth_rasterized": True,
        "caller_supplied_coverage_mask": False,
        "native_simulator_readback": False,
        "physical_equivalence_proven": False,
        "claim_ceiling": "co_present_actual_usd_geometry_depth_only",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (output / f"{COMPOSITION_SCHEMA}.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


def validate_replacement_depth_composition(
    value: Mapping[str, Any], *, receipt_path: str | Path
) -> dict[str, Any]:
    """Recompute and verify a file-backed co-present depth receipt.

    This deliberately does not trust a digest-shaped composition claim.  It
    reopens every constituent USD-depth sweep and checks that the stored array
    is their exact nearest-depth composition.
    """

    receipt_file = _path(
        receipt_path, code="replacement_depth_composition_receipt_missing"
    )
    receipt = _clone(value, code="replacement_depth_composition_receipt_not_json")
    if (
        receipt.get("schema_version") != COMPOSITION_SCHEMA
        or receipt.get("status") != "co_present_replacement_depth_rasterized"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("actual_usd_geometry_depth_rasterized") is not True
        or receipt.get("actual_composed_depth_rasterized") is not True
        or receipt.get("caller_supplied_coverage_mask") is not False
        or isinstance(receipt.get("resolution_scale"), bool)
        or not isinstance(receipt.get("resolution_scale"), (int, float))
        or not 0.0 < float(receipt["resolution_scale"]) <= 1.0
        or not str(receipt.get("task_id") or "")
        or not _digest(receipt.get("task_freeze_digest"))
        or not str(receipt.get("scored_task_asset_id") or "")
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_receipt_invalid"]
        )
    request_path = _verify_absolute(
        receipt.get("request"), code="replacement_depth_composition_request_record_invalid"
    )
    request = build_replacement_depth_composition_request(
        _read_object(request_path, code="replacement_depth_composition_request_unreadable")
    )
    if (
        receipt.get("request", {}).get("request_digest") != request["request_digest"]
        or request.get("task_id") != receipt.get("task_id")
        or request.get("task_freeze_digest") != receipt.get("task_freeze_digest")
        or request.get("scored_task_asset_id") != receipt.get("scored_task_asset_id")
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_request_join_invalid"]
        )
    input_rows = receipt.get("input_sweeps")
    if not isinstance(input_rows, list) or not 1 <= len(input_rows) <= MAX_REPLACEMENT_OBJECTS:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_input_rows_invalid"]
        )
    loaded: list[tuple[dict[str, Any], np.ndarray, dict[str, Any]]] = []
    for row in input_rows:
        sweep_path = _verify_absolute(
            row, code="replacement_depth_composition_input_record_invalid"
        )
        manifest, depth, record = _load_sweep(sweep_path)
        if (
            row.get("manifest_digest") != manifest["manifest_digest"]
            or row.get("asset_id") != manifest["asset_id"]
            or row.get("task_freeze_digest") != manifest["task_freeze_digest"]
            or row.get("finite_depth_pixel_count_by_cell")
            != [int(np.isfinite(cell_depth).sum()) for cell_depth in depth]
            or row.get("visible_in_any_composed_camera")
            is not bool(np.isfinite(depth).any())
        ):
            raise ReplacementDepthCompositionError(
                ["replacement_depth_composition_input_record_join_invalid"]
            )
        loaded.append((manifest, depth, record))
    input_paths = [record["path"] for _manifest, _depth, record in loaded]
    requested_input_paths = [
        str(_path(value, code="replacement_depth_composition_input_manifest_missing"))
        for value in request["input_sweep_manifest_paths"]
    ]
    if input_paths != requested_input_paths:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_request_input_join_invalid"]
        )
    asset_ids = [str(manifest["asset_id"]) for manifest, _depth, _record in loaded]
    if (
        len(asset_ids) != len(set(asset_ids))
        or sorted(asset_ids) != receipt.get("replacement_asset_ids")
        or receipt.get("scored_task_asset_id") not in asset_ids
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_asset_join_invalid"]
        )
    scored = next(
        item for item in loaded if item[0]["asset_id"] == receipt["scored_task_asset_id"]
    )
    if (
        scored[0].get("scene_state_role") != "task_subject"
        or any(
            manifest.get("scene_state_role") != "co_present_passive"
            for manifest, _depth, _record in loaded
            if manifest["asset_id"] != receipt["scored_task_asset_id"]
        )
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_scene_state_roles_invalid"]
        )
    reference_manifest, reference_depth, _reference_record = loaded[0]
    reference_cells = _cells_key(reference_manifest["cells"])
    for manifest, depth, _record_value in loaded[1:]:
        if (
            _cells_key(manifest["cells"]) != reference_cells
            or depth.shape != reference_depth.shape
            or manifest["camera_contract_digest"] != reference_manifest["camera_contract_digest"]
            or manifest["camera_rows_digest"] != reference_manifest["camera_rows_digest"]
            or float(manifest["resolution_scale"])
            != float(reference_manifest["resolution_scale"])
        ):
            raise ReplacementDepthCompositionError(
                ["replacement_depth_composition_input_cell_or_camera_mismatch"]
            )
    if (
        scored[0]["task_freeze_digest"] != receipt["task_freeze_digest"]
        or receipt.get("cells") != scored[0].get("cells")
        or receipt.get("camera_contract_digest")
        != reference_manifest.get("camera_contract_digest")
        or receipt.get("camera_rows_digest") != reference_manifest.get("camera_rows_digest")
        or float(receipt["resolution_scale"])
        != float(reference_manifest["resolution_scale"])
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_receipt_join_invalid"]
        )
    array_path = _verify_relative(
        receipt_file.parent,
        receipt.get("arrays"),
        code="replacement_depth_composition_output_array_invalid",
    )
    try:
        composed = np.asarray(np.load(array_path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_output_array_unreadable"]
        ) from exc
    expected = np.minimum.reduce(np.stack([depth for _manifest, depth, _record in loaded]))
    if (
        composed.shape != expected.shape
        or not np.array_equal(composed, expected)
        or receipt.get("depth_dimensions")
        != [int(composed.shape[2]), int(composed.shape[1])]
        or receipt.get("finite_depth_pixel_count_by_cell")
        != [int(np.isfinite(depth).sum()) for depth in composed]
    ):
        raise ReplacementDepthCompositionError(
            ["replacement_depth_composition_output_array_invalid"]
        )
    return receipt


__all__ = [
    "COMPOSITION_SCHEMA",
    "REQUEST_SCHEMA",
    "ReplacementDepthCompositionError",
    "build_replacement_depth_composition_request",
    "materialize_replacement_depth_composition",
    "validate_replacement_depth_composition",
]
