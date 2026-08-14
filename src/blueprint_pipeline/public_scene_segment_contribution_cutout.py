"""Materialize repair-supported cutouts from full-view object segments.

This lane deliberately answers a different question from conservative Gaussian
ownership: which source Gaussians make a renderer-detectable contribution to
the frozen object segment in any calibrated view?  Every selected Gaussian is
removed from a derived PLY, even when it also contributes protected background.
The complete projected deleted layer must therefore become later repair support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    validate_task_freeze_set,
)
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply_subset_exact,
)
from .public_scene_gaussian_excision_audit import (
    CONTRIBUTION_CLASS_ORDER,
    CONTRIBUTION_EVIDENCE_SCHEMA,
    FREEZE_SCHEMA,
)


SWEEP_KIND = "repair_supported_full_view_segment_contribution_sweep.v1"
CUTOUT_SET_SCHEMA = "adp009d_segment_contribution_cutout_set.v1"
TOOL_REQUEST_SCHEMA = "fresh_scene_segment_cutout_tool_request.v1"
SELECTION_RULE = (
    "union_across_repetitions_of_any_view_target_core_plus_uncertain_"
    "contribution_at_frozen_threshold.v1"
)


class SegmentContributionCutoutError(ValueError):
    """Stable fail-closed errors for segment-contribution materialization."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentContributionCutoutError([code]) from exc
    if not isinstance(value, dict):
        raise SegmentContributionCutoutError([code])
    return value


def _input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verified_relative(root: Path, record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise SegmentContributionCutoutError([code])
    relative = str(record.get("relative_path") or "")
    path = (root / relative).resolve()
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or root.resolve() not in path.parents
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise SegmentContributionCutoutError([code])
    return path


def materialize_segment_contribution_sweep_freeze(
    *, excision_freeze_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Derive an all-camera contribution freeze without changing source truth."""

    source_path = Path(excision_freeze_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise SegmentContributionCutoutError(["segment_sweep_source_freeze_missing"])
    if output.exists() and any(output.iterdir()):
        raise SegmentContributionCutoutError(["segment_sweep_output_not_empty"])
    source = _read(source_path, code="segment_sweep_source_freeze_unreadable")
    if (
        source.get("schema_version") != FREEZE_SCHEMA
        or source.get("freeze_digest")
        != canonical_digest(source, digest_field="freeze_digest")
        or source.get("status") != "frozen_before_excision_execution"
    ):
        raise SegmentContributionCutoutError(["segment_sweep_source_freeze_invalid"])
    rows = source.get("masks")
    camera_split = source.get("camera_split")
    if not isinstance(rows, list) or not isinstance(camera_split, Mapping):
        raise SegmentContributionCutoutError(["segment_sweep_camera_set_invalid"])
    camera_ids = sorted(str(row.get("camera_id") or "") for row in rows if isinstance(row, Mapping))
    if len(camera_ids) != len(rows) or "" in camera_ids or len(set(camera_ids)) != len(rows):
        raise SegmentContributionCutoutError(["segment_sweep_camera_set_invalid"])

    output.mkdir(parents=True)
    mask_output = output / "masks"
    mask_output.mkdir()
    rewritten_rows: list[dict[str, Any]] = []
    for row in rows:
        assert isinstance(row, Mapping)
        camera_id = str(row["camera_id"])
        zones = row.get("zones")
        if not isinstance(zones, Mapping):
            raise SegmentContributionCutoutError([f"segment_sweep_masks_invalid:{camera_id}"])
        rewritten = dict(row)
        rewritten_zones: dict[str, Any] = {}
        for zone in CONTRIBUTION_CLASS_ORDER:
            source_mask = _verified_relative(
                source_path.parent,
                zones.get(zone),
                code=f"segment_sweep_masks_invalid:{camera_id}:{zone}",
            )
            destination = mask_output / f"{camera_id}.{zone}.png"
            shutil.copy2(source_mask, destination)
            rewritten_zones[zone] = _record(destination, output)
        rewritten["zones"] = rewritten_zones
        rewritten_rows.append(rewritten)

    baseline = source.get("historical_baseline")
    if isinstance(baseline, Mapping) and isinstance(baseline.get("indices"), Mapping):
        source_indices = _verified_relative(
            source_path.parent,
            baseline["indices"],
            code="segment_sweep_historical_indices_invalid",
        )
        destination = output / "historical_obb_source_indices.npy"
        shutil.copy2(source_indices, destination)
        baseline = {**baseline, "indices": _record(destination, output)}

    sweep = dict(source)
    sweep["masks"] = rewritten_rows
    sweep["historical_baseline"] = baseline
    sweep["camera_split"] = {
        "camera_count": len(camera_ids),
        "calibration_camera_count": len(camera_ids),
        "calibration_camera_ids": camera_ids,
        "heldout_camera_count": 0,
        "heldout_camera_ids": [],
        "method": "all_frozen_segment_views.v1",
        "outcome_fields_accessed": False,
        "camera_contract_sha256": (source.get("camera_contract") or {}).get("sha256"),
        "camera_split_digest": "",
    }
    sweep["camera_split"]["camera_split_digest"] = canonical_digest(
        sweep["camera_split"], digest_field="camera_split_digest"
    )
    sweep["segment_contribution_sweep"] = {
        "kind": SWEEP_KIND,
        "source_excision_freeze": {
            **_input_record(source_path),
            "freeze_digest": source["freeze_digest"],
        },
        "selection_classes": ["target_core", "uncertain"],
        "selection_mask_semantics": "historical_outer_mask_exact_union",
        "all_frozen_cameras_included": True,
        "factual_gaussian_ownership_claimed": False,
        "protected_background_coupling_allowed_only_with_complete_repair_support": True,
    }
    sweep["freeze_digest"] = canonical_digest(sweep, digest_field="freeze_digest")
    path = output / f"{FREEZE_SCHEMA}.json"
    path.write_text(canonical_json(sweep) + "\n", encoding="utf-8")
    return sweep


def _load_arrays(
    *, manifest_path: Path, manifest: Mapping[str, Any], shape: tuple[int, int, int], decimals: int
) -> list[np.ndarray]:
    rows = manifest.get("repetitions")
    if not isinstance(rows, list) or len(rows) < 2:
        raise SegmentContributionCutoutError(["segment_cutout_repetitions_invalid"])
    arrays: list[np.ndarray] = []
    for row in rows:
        path = _verified_relative(
            manifest_path.parent, row, code="segment_cutout_repetition_changed"
        )
        try:
            with np.load(path, allow_pickle=False) as archive:
                value = np.asarray(archive["per_view_class_contribution"], dtype=np.float64)
        except (OSError, ValueError, KeyError) as exc:
            raise SegmentContributionCutoutError(["segment_cutout_repetition_invalid"]) from exc
        if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0.0):
            raise SegmentContributionCutoutError(["segment_cutout_repetition_invalid"])
        arrays.append(np.round(value, decimals=decimals))
    return arrays


def materialize_segment_contribution_cutout_set(
    *,
    source_standard_splat_path: str | Path,
    task_freeze_paths: Sequence[str | Path],
    sweep_freeze_paths_by_task: Mapping[str, str | Path],
    contribution_manifest_paths_by_task: Mapping[str, str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize an overlapping 1--5 object cutout and shared derived PLY."""

    source_path = Path(source_standard_splat_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise SegmentContributionCutoutError(["segment_cutout_source_splat_missing"])
    if output.exists() and any(output.iterdir()):
        raise SegmentContributionCutoutError(["segment_cutout_output_not_empty"])
    task_paths = [Path(path).expanduser().resolve() for path in task_freeze_paths]
    try:
        tasks = [_read(path, code="segment_cutout_task_freeze_unreadable") for path in task_paths]
        task_set = validate_task_freeze_set(tasks)
    except DualTaskRehearsalContractError as exc:
        raise SegmentContributionCutoutError(
            [f"segment_cutout_{code}" for code in exc.errors]
        ) from exc
    task_ids = sorted(str(task["task_id"]) for task in tasks)
    if set(sweep_freeze_paths_by_task) != set(task_ids):
        raise SegmentContributionCutoutError(["segment_cutout_sweep_keys_invalid"])
    if set(contribution_manifest_paths_by_task) != set(task_ids):
        raise SegmentContributionCutoutError(["segment_cutout_manifest_keys_invalid"])
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    task_path_by_id = {str(task["task_id"]): path for task, path in zip(tasks, task_paths, strict=True)}

    source = read_standard_3dgs_ply(source_path)
    source_sha = _sha256(source_path)
    source_size = source_path.stat().st_size
    selected_by_task: dict[str, np.ndarray] = {}
    candidate_rows: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    for slot, task_id in enumerate(task_ids, start=1):
        task = tasks_by_id[task_id]
        sweep_path = Path(sweep_freeze_paths_by_task[task_id]).expanduser().resolve()
        manifest_path = Path(contribution_manifest_paths_by_task[task_id]).expanduser().resolve()
        sweep = _read(sweep_path, code=f"segment_cutout_sweep_unreadable:{task_id}")
        manifest = _read(manifest_path, code=f"segment_cutout_manifest_unreadable:{task_id}")
        sweep_meta = sweep.get("segment_contribution_sweep")
        camera_split = sweep.get("camera_split")
        scene = sweep.get("scene")
        if (
            sweep.get("schema_version") != FREEZE_SCHEMA
            or sweep.get("freeze_digest") != canonical_digest(sweep, digest_field="freeze_digest")
            or not isinstance(sweep_meta, Mapping)
            or sweep_meta.get("kind") != SWEEP_KIND
            or sweep_meta.get("selection_classes") != ["target_core", "uncertain"]
            or sweep_meta.get("all_frozen_cameras_included") is not True
            or not isinstance(camera_split, Mapping)
            or camera_split.get("heldout_camera_ids") != []
            or camera_split.get("calibration_camera_count") != camera_split.get("camera_count")
            or not isinstance(scene, Mapping)
            or scene.get("task_id") != task_id
            or scene.get("target_instance_id") != task["source_object"]["instance_id"]
            or scene.get("removal_id") != task["removal_plan"]["removal_id"]
            or scene.get("mask_set_id") != task["removal_plan"]["mask_set_id"]
        ):
            raise SegmentContributionCutoutError([f"segment_cutout_sweep_invalid:{task_id}"])
        bound_source = sweep.get("source_standard_splat")
        camera_ids = camera_split.get("calibration_camera_ids")
        if (
            not isinstance(bound_source, Mapping)
            or bound_source.get("sha256") != source_sha
            or bound_source.get("size_bytes") != source_size
            or not isinstance(camera_ids, list)
            or len(camera_ids) != camera_split.get("camera_count")
        ):
            raise SegmentContributionCutoutError([f"segment_cutout_source_or_cameras_invalid:{task_id}"])
        if (
            manifest.get("schema_version") != CONTRIBUTION_EVIDENCE_SCHEMA
            or manifest.get("manifest_digest")
            != canonical_digest(manifest, digest_field="manifest_digest")
            or manifest.get("freeze_digest") != sweep["freeze_digest"]
            or manifest.get("class_order") != list(CONTRIBUTION_CLASS_ORDER)
            or manifest.get("camera_ids") != camera_ids
            or manifest.get("heldout_cameras_accessed_for_classification") is not False
            or not isinstance(manifest.get("method"), Mapping)
            or manifest["method"].get("released_code_executed") is not True
        ):
            raise SegmentContributionCutoutError([f"segment_cutout_manifest_invalid:{task_id}"])
        decimals = sweep.get("policy", {}).get("contribution_quantization_decimals")
        threshold = sweep.get("policy", {}).get("minimum_per_view_contribution")
        if (
            isinstance(decimals, bool)
            or not isinstance(decimals, int)
            or not 3 <= decimals <= 12
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 < float(threshold) <= 1.0
        ):
            raise SegmentContributionCutoutError([f"segment_cutout_policy_invalid:{task_id}"])
        arrays = _load_arrays(
            manifest_path=manifest_path,
            manifest=manifest,
            shape=(len(camera_ids), len(CONTRIBUTION_CLASS_ORDER), source.count),
            decimals=decimals,
        )
        flags = [
            np.any((array[:, 1, :] + array[:, 2, :]) >= float(threshold), axis=0)
            for array in arrays
        ]
        selected_mask = np.logical_or.reduce(flags)
        unanimous_mask = np.logical_and.reduce(flags)
        selected = np.flatnonzero(selected_mask).astype(np.int64)
        if not selected.size:
            raise SegmentContributionCutoutError([f"segment_cutout_selection_empty:{task_id}"])
        protected = np.logical_or.reduce(
            [np.any(array[:, 0, :] >= float(threshold), axis=0) for array in arrays]
        )
        selected_by_task[task_id] = selected
        candidate_root = output / "task_candidates" / f"slot_{slot:02d}"
        candidate_root.mkdir(parents=True)
        indices_path = candidate_root / "deleted_source_indices.npy"
        np.save(indices_path, selected, allow_pickle=False)
        deleted_ply = write_standard_3dgs_ply_subset_exact(
            source_path, candidate_root / "deleted_source_gaussians.ply", selected
        )
        retained = np.flatnonzero(~selected_mask).astype(np.int64)
        retained_ply = write_standard_3dgs_ply_subset_exact(
            source_path, candidate_root / "retained_scene_gaussians.ply", retained
        )
        preservation = verify_standard_3dgs_ply_subset_exact(source_path, retained_ply, retained)
        if preservation.get("retained_rows_byte_exact") is not True:
            raise SegmentContributionCutoutError([f"segment_cutout_retained_rows_changed:{task_id}"])
        candidate_rows.append(
            {
                "slot": slot,
                "task_id": task_id,
                "task_freeze_digest": task["task_freeze_digest"],
                "task_freeze": _input_record(task_path_by_id[task_id]),
                "sweep_freeze": {**_input_record(sweep_path), "freeze_digest": sweep["freeze_digest"]},
                "contribution_manifest": {
                    **_input_record(manifest_path),
                    "manifest_digest": manifest["manifest_digest"],
                },
                "selection": {
                    "rule": SELECTION_RULE,
                    "threshold": float(threshold),
                    "camera_count": len(camera_ids),
                    "repetition_count": len(arrays),
                    "repetition_disagreement_count": int((selected_mask & ~unanimous_mask).sum()),
                    "protected_coupled_selected_count": int((selected_mask & protected).sum()),
                },
                "counts": {
                    "source": source.count,
                    "deleted_total": int(selected.size),
                    "retained_total": int(retained.size),
                },
                "preservation": preservation,
                "outputs": {
                    "deleted_source_indices": _record(indices_path, output),
                    "deleted_source_gaussians": _record(deleted_ply, output),
                    "retained_scene_gaussians": _record(retained_ply, output),
                },
            }
        )

    union = np.unique(np.concatenate([selected_by_task[task_id] for task_id in task_ids]))
    retained_union = np.setdiff1d(np.arange(source.count, dtype=np.int64), union, assume_unique=True)
    shared = output / "shared_scene_union"
    shared.mkdir()
    union_path = shared / "deleted_source_indices.npy"
    retained_path = shared / "retained_source_indices.npy"
    np.save(union_path, union, allow_pickle=False)
    np.save(retained_path, retained_union, allow_pickle=False)
    deleted_ply = write_standard_3dgs_ply_subset_exact(
        source_path, shared / "deleted_source_gaussians.ply", union
    )
    retained_ply = write_standard_3dgs_ply_subset_exact(
        source_path, shared / "retained_scene_gaussians.ply", retained_union
    )
    preservation = verify_standard_3dgs_ply_subset_exact(source_path, retained_ply, retained_union)
    if preservation.get("retained_rows_byte_exact") is not True:
        raise SegmentContributionCutoutError(["segment_cutout_shared_retained_rows_changed"])
    overlaps = []
    for left_index, left in enumerate(task_ids):
        for right in task_ids[left_index + 1 :]:
            overlaps.append(
                {
                    "left_task_id": left,
                    "right_task_id": right,
                    "shared_deleted_gaussian_count": int(
                        np.intersect1d(selected_by_task[left], selected_by_task[right], assume_unique=True).size
                    ),
                }
            )
    receipt: dict[str, Any] = {
        "schema_version": CUTOUT_SET_SCHEMA,
        "status": "repair_supported_segment_contribution_cutout_materialized_pending_full_deleted_layer_projection",
        "source_standard_splat": _input_record(source_path),
        "task_set": task_set,
        "selection": {
            "rule": SELECTION_RULE,
            "frozen_segment_classes": ["target_core", "uncertain"],
            "all_frozen_cameras_required": True,
            "task_overlap_allowed_and_recorded": True,
            "replacement_usd_used": False,
            "learned_policy_or_simulator_output_used": False,
        },
        "task_candidates": candidate_rows,
        "cross_task_overlaps": overlaps,
        "shared_scene_union": {
            "counts": {
                "source": source.count,
                "deleted_total": int(union.size),
                "retained_total": int(retained_union.size),
            },
            "preservation": preservation,
            "outputs": {
                "deleted_source_indices": _record(union_path, output),
                "retained_source_indices": _record(retained_path, output),
                "deleted_source_gaussians": _record(deleted_ply, output),
                "retained_scene_gaussians": _record(retained_ply, output),
            },
        },
        "claim_boundary": {
            "canonical_source_altered": False,
            "candidate_derived_layers_only": True,
            "factual_gaussian_ownership_established": False,
            "protected_background_deletion_permitted_only_as_explicit_repair_support": True,
            "full_deleted_layer_projection_complete": False,
            "inpainting_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    path = output / f"{CUTOUT_SET_SCHEMA}.json"
    path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_segment_contribution_cutout_set_from_tool_request(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Execute the registered non-spend cutout request after digest validation."""

    value = dict(request)
    task_freezes = value.get("task_freeze_paths")
    sweep_paths = value.get("sweep_freeze_paths_by_task")
    manifest_paths = value.get("contribution_manifest_paths_by_task")
    if (
        value.get("schema_version") != TOOL_REQUEST_SCHEMA
        or value.get("request_digest")
        != canonical_digest(value, digest_field="request_digest")
        or not isinstance(task_freezes, list)
        or not 1 <= len(task_freezes) <= 5
        or not isinstance(sweep_paths, Mapping)
        or not isinstance(manifest_paths, Mapping)
        or set(sweep_paths) != set(manifest_paths)
        or len(sweep_paths) != len(task_freezes)
    ):
        raise SegmentContributionCutoutError(["segment_cutout_tool_request_invalid"])
    return materialize_segment_contribution_cutout_set(
        source_standard_splat_path=str(value.get("source_standard_splat_path") or ""),
        task_freeze_paths=[str(path) for path in task_freezes],
        sweep_freeze_paths_by_task={
            str(task_id): str(path) for task_id, path in sweep_paths.items()
        },
        contribution_manifest_paths_by_task={
            str(task_id): str(path) for task_id, path in manifest_paths.items()
        },
        output_root=output_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Expose sweep-freeze and digest-bound cutout requests to production CLIs."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep-freeze")
    sweep.add_argument("--excision-freeze", required=True)
    sweep.add_argument("--output-root", required=True)
    cutout = commands.add_parser("cutout-from-request")
    cutout.add_argument("--request", required=True)
    cutout.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "sweep-freeze":
        result = materialize_segment_contribution_sweep_freeze(
            excision_freeze_path=args.excision_freeze,
            output_root=args.output_root,
        )
    else:
        request_path = Path(args.request).expanduser().resolve()
        request = _read(request_path, code="segment_cutout_tool_request_unreadable")
        if request_path.is_symlink():
            raise SegmentContributionCutoutError(
                ["segment_cutout_tool_request_unreadable"]
            )
        result = materialize_segment_contribution_cutout_set_from_tool_request(
            request=request,
            output_root=args.output_root,
        )
    print(canonical_json(result))
    return 0


__all__ = [
    "CUTOUT_SET_SCHEMA",
    "SELECTION_RULE",
    "SWEEP_KIND",
    "SegmentContributionCutoutError",
    "main",
    "materialize_segment_contribution_cutout_set",
    "materialize_segment_contribution_cutout_set_from_tool_request",
    "materialize_segment_contribution_sweep_freeze",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
