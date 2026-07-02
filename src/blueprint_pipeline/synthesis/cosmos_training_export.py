"""Cosmos Predict fine-tuning/export substrate for site-grounded captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ..common import ensure_dir, read_json_any, utc_now_iso, write_json
from ..geometry_sources import SyntheticGeometryExportError, geometry_export_gate
from ..local_capture import resolve_local_capture_context
from .cosmos_capture_bootstrap import extract_video_bootstrap_records, resolve_video_bootstrap_sources
from .future_anchor_regrounding import build_future_anchor_regrounding_manifest
from .plucker_rays import compute_plucker_map
from .reference_selection import (
    build_legacy_reference_selection_manifest,
    build_reference_selection_comparison,
    build_reference_selection_manifest,
)
from .sparse_view_interpolation import build_sparse_view_interpolation_manifest
from .trajectory_augmentation import build_synthetic_trajectory_manifest


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        "".join(json.dumps(dict(row), separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normalized_intrinsics(record: Mapping[str, Any]) -> Dict[str, float] | None:
    """Return validated intrinsics or None when calibration is missing.

    Guessed focal lengths / image sizes must never feed conditioning maps;
    a record without real fx/fy/width/height is rejected, not defaulted.
    Only the principal point may fall back to image center, and that fact is
    recorded.
    """
    intrinsics = dict(record.get("intrinsics") or {}) if isinstance(record.get("intrinsics"), Mapping) else {}
    width = intrinsics.get("width") or intrinsics.get("w")
    height = intrinsics.get("height") or intrinsics.get("h")
    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    if not width or not height or not fx or not fy:
        return None
    width_f = float(width)
    height_f = float(height)
    fx_f = float(fx)
    fy_f = float(fy)
    if width_f <= 0 or height_f <= 0 or fx_f <= 0 or fy_f <= 0:
        return None
    principal_point_defaulted = intrinsics.get("cx") is None or intrinsics.get("cy") is None
    return {
        "fx": fx_f,
        "fy": fy_f,
        "cx": float(intrinsics.get("cx") if intrinsics.get("cx") is not None else width_f / 2.0),
        "cy": float(intrinsics.get("cy") if intrinsics.get("cy") is not None else height_f / 2.0),
        "width": width_f,
        "height": height_f,
        "principal_point_defaulted": float(bool(principal_point_defaulted)),
    }


def _record_pose_matrix(record: Mapping[str, Any]) -> "np.ndarray | None":
    """Return the record's 4x4 world-from-camera pose or None when absent.

    Missing/misshaped poses are rejected upstream — never identity-filled.
    """
    raw = record.get("T_world_camera")
    if raw is None:
        return None
    pose = np.array(raw, dtype=np.float32)
    if pose.ndim == 1 and pose.size == 16:
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return pose


def _split_name(frame_id: str) -> str:
    digest = hashlib.sha256(frame_id.encode("utf-8")).hexdigest()
    return "val" if int(digest[:2], 16) < 51 else "train"


def export_cosmos_training_substrate(
    *,
    capture_root: str | Path,
    k_references: int = 4,
    max_video_bootstrap_frames: int = 12,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_root = context.pipeline_root
    export_root = pipeline_root / "cosmos_training_export"
    plucker_root = export_root / "plucker"
    ensure_dir(export_root)
    ensure_dir(plucker_root)

    geometry_summary_path = pipeline_root / "geometry" / "geometry_summary.json"
    geometry_summary = read_json_any(geometry_summary_path) if geometry_summary_path.is_file() else {}
    try:
        geometry_provenance = geometry_export_gate(
            geometry_summary if isinstance(geometry_summary, Mapping) else {},
            export_name="cosmos_training_export",
        )
    except SyntheticGeometryExportError as exc:
        blocked_manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "reason": str(exc),
            "capture_id": context.capture_id,
            "scene_id": context.scene_id,
            "geometry_provenance": {
                "synthetic_geometry": True,
                "export_allowed_by": None,
            },
        }
        write_json(export_root / "manifest.json", blocked_manifest)
        return blocked_manifest

    dense_index_path = context.capture_root / "world_model_export" / "dense_index.jsonl"
    task_anchor_path = pipeline_root / "evaluation_prep" / "task_anchor_manifest.json"
    protected_regions_path = pipeline_root / "evaluation_prep" / "protected_regions_manifest.json"
    conditioning_bundle_path = pipeline_root / "scene_memory" / "conditioning_bundle.json"

    dense_records = [
        row for row in _read_jsonl(dense_index_path)
        if bool(row.get("included_in_index")) and str(row.get("frame_uri") or "").strip()
    ]
    task_anchor_manifest = read_json_any(task_anchor_path) if task_anchor_path.is_file() else {}
    protected_regions_manifest = read_json_any(protected_regions_path) if protected_regions_path.is_file() else {}
    conditioning_bundle = read_json_any(conditioning_bundle_path) if conditioning_bundle_path.is_file() else {}
    source_mode = "dense_index"
    bootstrap_origin = None
    bootstrap_source_manifest_path: Path | None = None
    reference_selection_manifest_path = export_root / "reference_selection_manifest.json"
    reference_selection_comparison_path = export_root / "reference_selection_comparison.json"
    synthetic_trajectory_manifest_path = export_root / "synthetic_trajectory_manifest.json"
    sparse_view_interpolation_manifest_path = export_root / "sparse_view_interpolation_manifest.json"
    future_anchor_regrounding_manifest_path = export_root / "future_anchor_regrounding_manifest.json"

    if len(dense_records) < 2:
        bootstrap_sources = resolve_video_bootstrap_sources(
            context=context,
            conditioning_bundle=conditioning_bundle if isinstance(conditioning_bundle, Mapping) else {},
        )
        dense_records = extract_video_bootstrap_records(
            bootstrap_sources=bootstrap_sources,
            export_root=export_root,
            max_frames=max_video_bootstrap_frames,
        ) if bootstrap_sources else []
        if dense_records:
            source_mode = "video_bootstrap"
            bootstrap_origin = str(bootstrap_sources.get("origin") or "unknown")
            bootstrap_source_manifest_path = export_root / "bootstrap_source_manifest.json"
            write_json(
                bootstrap_source_manifest_path,
                {
                    "schema_version": "v1",
                    "generated_at": utc_now_iso(),
                    "origin": bootstrap_origin,
                    "video_path": bootstrap_sources.get("video_path"),
                    "poses_path": bootstrap_sources.get("poses_path"),
                    "intrinsics_path": bootstrap_sources.get("intrinsics_path"),
                    "source_video_uri": bootstrap_sources.get("source_video_uri"),
                },
            )

    if len(dense_records) < 2:
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "missing",
            "reason": "insufficient_dense_index_records",
            "geometry_provenance": geometry_provenance,
            "source_mode": source_mode,
            "bootstrap_origin": bootstrap_origin,
            "bootstrap_source_manifest_path": str(bootstrap_source_manifest_path.resolve()) if bootstrap_source_manifest_path else None,
            "reference_selection_manifest_path": None,
            "reference_selection_comparison_path": None,
            "synthetic_trajectory_manifest_path": None,
            "sparse_view_interpolation_manifest_path": None,
            "future_anchor_regrounding_manifest_path": None,
            "paired_reference_target_path": None,
            "k_reference_conditioning_path": None,
        }
        write_json(export_root / "manifest.json", manifest)
        return manifest

    reference_selection_manifest = build_reference_selection_manifest(
        records=dense_records,
        k=k_references,
        selection_name="cosmos_training_export",
    )
    legacy_reference_selection_manifest = build_legacy_reference_selection_manifest(
        records=dense_records,
        k=k_references,
        selection_name="cosmos_training_export_legacy_baseline",
    )
    reference_selection_comparison = build_reference_selection_comparison(
        current_manifest=reference_selection_manifest,
        legacy_manifest=legacy_reference_selection_manifest,
        selection_name="cosmos_training_export",
    )
    synthetic_trajectory_manifest = build_synthetic_trajectory_manifest(
        records=dense_records,
        selection_entries=list(reference_selection_manifest.get("entries") or []),
        augmentation_name="cosmos_training_export",
    )
    sparse_view_interpolation_manifest = build_sparse_view_interpolation_manifest(
        records=dense_records,
        selection_entries=list(reference_selection_manifest.get("entries") or []),
        trajectory_entries=list(synthetic_trajectory_manifest.get("entries") or []),
        interpolation_name="cosmos_training_export",
    )
    future_anchor_regrounding_manifest = build_future_anchor_regrounding_manifest(
        records=dense_records,
        selection_entries=list(reference_selection_manifest.get("entries") or []),
        task_anchor_manifest=task_anchor_manifest if isinstance(task_anchor_manifest, Mapping) else {},
        protected_regions_manifest=protected_regions_manifest if isinstance(protected_regions_manifest, Mapping) else {},
        regrounding_name="cosmos_training_export",
    )
    write_json(reference_selection_manifest_path, reference_selection_manifest)
    write_json(reference_selection_comparison_path, reference_selection_comparison)
    write_json(synthetic_trajectory_manifest_path, synthetic_trajectory_manifest)
    write_json(sparse_view_interpolation_manifest_path, sparse_view_interpolation_manifest)
    write_json(future_anchor_regrounding_manifest_path, future_anchor_regrounding_manifest)

    trajectory_entries = {
        str(entry.get("target_frame_id") or ""): dict(entry)
        for entry in list(synthetic_trajectory_manifest.get("entries") or [])
        if str(entry.get("target_frame_id") or "").strip()
    }
    sparse_interpolation_entries = {
        str(entry.get("target_frame_id") or ""): dict(entry)
        for entry in list(sparse_view_interpolation_manifest.get("entries") or [])
        if str(entry.get("target_frame_id") or "").strip()
    }
    future_anchor_entries = {
        str(entry.get("target_frame_id") or ""): dict(entry)
        for entry in list(future_anchor_regrounding_manifest.get("entries") or [])
        if str(entry.get("target_frame_id") or "").strip()
    }

    paired_rows: List[Dict[str, Any]] = []
    k_reference_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    split_summary = {"train": 0, "val": 0}

    for selection in list(reference_selection_manifest.get("entries") or []):
        target_index = int(selection.get("target_index") or 0)
        if target_index < 0 or target_index >= len(dense_records):
            continue
        record = dense_records[target_index]
        selected_references = list(selection.get("selected_references") or [])
        references = [
            dense_records[int(item.get("candidate_index"))]
            for item in selected_references
            if int(item.get("candidate_index")) >= 0 and int(item.get("candidate_index")) < len(dense_records)
        ]
        if not references:
            continue
        frame_id = str(record.get("frame_id") or "").strip()
        intrinsics = _normalized_intrinsics(record)
        target_T = _record_pose_matrix(record)
        if intrinsics is None or target_T is None:
            # Never guess calibration or identity-fill poses into training
            # targets — skip the record and make the rejection auditable.
            rejected_rows.append(
                {
                    "frame_id": frame_id or f"frame_{target_index}",
                    "target_index": target_index,
                    "reasons": [
                        *(["intrinsics_missing_or_implausible"] if intrinsics is None else []),
                        *(["pose_missing_or_misshaped"] if target_T is None else []),
                    ],
                }
            )
            continue
        split = _split_name(frame_id or f"frame_{target_index}")
        split_summary[split] += 1
        plucker = compute_plucker_map(
            T_world_camera=target_T,
            intrinsics=intrinsics,
            height=max(16, int(intrinsics["height"])),
            width=max(16, int(intrinsics["width"])),
        )
        plucker_path = plucker_root / f"{frame_id or f'target_{target_index}'}.npz"
        np.savez_compressed(plucker_path, plucker=plucker)
        trajectory_entry = trajectory_entries.get(frame_id, {})
        sparse_interpolation_entry = sparse_interpolation_entries.get(frame_id, {})
        future_anchor_entry = future_anchor_entries.get(frame_id, {})

        paired_rows.append(
            {
                "capture_id": context.capture_id,
                "scene_id": context.scene_id,
                "frame_id": frame_id,
                "split": split,
                "target_frame_uri": record.get("frame_uri"),
                "primary_reference_frame_uri": references[0].get("frame_uri"),
                "primary_reference_id": selected_references[0].get("reference_id"),
                "selected_reference_ids": list(selection.get("selected_reference_ids") or []),
                "selected_reference_frame_ids": list(selection.get("selected_reference_frame_ids") or []),
                "reference_selection_score": selected_references[0].get("score"),
                "reference_temporal_gap_sec": selected_references[0].get("temporal_gap_sec"),
                "reference_pose_distance_m": selected_references[0].get("pose_distance_m"),
                "rejected_near_duplicate_count": selection.get("rejected_near_duplicate_count"),
                "target_reference_decoupling_mode": (
                    (selection.get("decoupling") or {})
                    if isinstance(selection.get("decoupling"), Mapping)
                    else {}
                ).get("mode"),
                "plucker_conditioning_path": str(plucker_path.resolve()),
                "task_anchor_manifest_path": str(task_anchor_path.resolve()) if task_anchor_path.is_file() else None,
                "protected_regions_manifest_path": str(protected_regions_path.resolve()) if protected_regions_path.is_file() else None,
                "reference_selection_manifest_path": str(reference_selection_manifest_path.resolve()),
                "reference_selection_comparison_path": str(reference_selection_comparison_path.resolve()),
                "synthetic_trajectory_manifest_path": str(synthetic_trajectory_manifest_path.resolve()),
                "sparse_view_interpolation_manifest_path": str(sparse_view_interpolation_manifest_path.resolve()),
                "future_anchor_regrounding_manifest_path": str(future_anchor_regrounding_manifest_path.resolve()),
                "trajectory_context_id": trajectory_entry.get("trajectory_context_id"),
                "synthetic_trajectory_status": trajectory_entry.get("status"),
                "synthetic_trajectory_reason": trajectory_entry.get("reason"),
                "synthetic_waypoint_count": trajectory_entry.get("synthetic_waypoint_count"),
                "synthetic_waypoint_ids": list(trajectory_entry.get("synthetic_waypoint_ids") or []),
                "sparse_interpolation_context_id": sparse_interpolation_entry.get("interpolation_context_id"),
                "sparse_view_interpolation_status": sparse_interpolation_entry.get("status"),
                "sparse_view_interpolation_reason": sparse_interpolation_entry.get("reason"),
                "interpolated_view_count": sparse_interpolation_entry.get("interpolated_view_count"),
                "interpolated_view_ids": list(sparse_interpolation_entry.get("interpolated_view_ids") or []),
                "future_anchor_context_id": future_anchor_entry.get("future_anchor_context_id"),
                "future_anchor_status": future_anchor_entry.get("status"),
                "future_anchor_reason": future_anchor_entry.get("reason"),
                "future_anchor_count": future_anchor_entry.get("future_anchor_count"),
                "future_anchor_reference_ids": list(future_anchor_entry.get("future_anchor_reference_ids") or []),
                "future_anchor_frame_ids": list(future_anchor_entry.get("future_anchor_frame_ids") or []),
                "anchor_observations": list(record.get("anchor_observations") or []),
                "source_mode": str(record.get("source_mode") or source_mode),
            }
        )
        k_reference_rows.append(
            {
                "capture_id": context.capture_id,
                "scene_id": context.scene_id,
                "frame_id": frame_id,
                "split": split,
                "target_frame_uri": record.get("frame_uri"),
                "selected_reference_ids": list(selection.get("selected_reference_ids") or []),
                "selected_reference_frame_ids": list(selection.get("selected_reference_frame_ids") or []),
                "reference_frame_uris": [ref.get("frame_uri") for ref in references if str(ref.get("frame_uri") or "").strip()],
                "reference_embedding_uris": [ref.get("embedding_uri") for ref in references if str(ref.get("embedding_uri") or "").strip()],
                "reference_scores": [item.get("score") for item in selected_references],
                "reference_temporal_gaps_sec": [item.get("temporal_gap_sec") for item in selected_references],
                "reference_pose_distances_m": [item.get("pose_distance_m") for item in selected_references],
                "rejected_near_duplicate_count": selection.get("rejected_near_duplicate_count"),
                "target_reference_decoupling_mode": (
                    (selection.get("decoupling") or {})
                    if isinstance(selection.get("decoupling"), Mapping)
                    else {}
                ).get("mode"),
                "plucker_conditioning_path": str(plucker_path.resolve()),
                "conditioning_bundle_path": str(conditioning_bundle_path.resolve()) if conditioning_bundle_path.is_file() else None,
                "reference_selection_manifest_path": str(reference_selection_manifest_path.resolve()),
                "reference_selection_comparison_path": str(reference_selection_comparison_path.resolve()),
                "synthetic_trajectory_manifest_path": str(synthetic_trajectory_manifest_path.resolve()),
                "sparse_view_interpolation_manifest_path": str(sparse_view_interpolation_manifest_path.resolve()),
                "future_anchor_regrounding_manifest_path": str(future_anchor_regrounding_manifest_path.resolve()),
                "trajectory_context_id": trajectory_entry.get("trajectory_context_id"),
                "synthetic_trajectory_status": trajectory_entry.get("status"),
                "synthetic_trajectory_reason": trajectory_entry.get("reason"),
                "synthetic_waypoint_count": trajectory_entry.get("synthetic_waypoint_count"),
                "synthetic_waypoint_ids": list(trajectory_entry.get("synthetic_waypoint_ids") or []),
                "sparse_interpolation_context_id": sparse_interpolation_entry.get("interpolation_context_id"),
                "sparse_view_interpolation_status": sparse_interpolation_entry.get("status"),
                "sparse_view_interpolation_reason": sparse_interpolation_entry.get("reason"),
                "interpolated_view_count": sparse_interpolation_entry.get("interpolated_view_count"),
                "interpolated_view_ids": list(sparse_interpolation_entry.get("interpolated_view_ids") or []),
                "future_anchor_context_id": future_anchor_entry.get("future_anchor_context_id"),
                "future_anchor_status": future_anchor_entry.get("status"),
                "future_anchor_reason": future_anchor_entry.get("reason"),
                "future_anchor_count": future_anchor_entry.get("future_anchor_count"),
                "future_anchor_reference_ids": list(future_anchor_entry.get("future_anchor_reference_ids") or []),
                "future_anchor_frame_ids": list(future_anchor_entry.get("future_anchor_frame_ids") or []),
                "source_mode": str(record.get("source_mode") or source_mode),
            }
        )

    paired_path = export_root / "paired_reference_target.jsonl"
    k_reference_path = export_root / "k_reference_conditioning.jsonl"
    rejection_manifest_path = export_root / "export_rejection_manifest.json"
    split_path = export_root / "train_val_split.json"
    trainer_config_path = export_root / "trainer_config.json"
    checkpoint_layout_path = export_root / "checkpoint_layout.json"
    inference_backend_path = export_root / "inference_backend_shape.json"

    _write_jsonl(paired_path, paired_rows)
    _write_jsonl(k_reference_path, k_reference_rows)
    write_json(
        rejection_manifest_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "rejected_count": len(rejected_rows),
            "rejections": rejected_rows,
        },
    )
    write_json(
        split_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "train_count": split_summary["train"],
            "val_count": split_summary["val"],
            "train_ratio": round(split_summary["train"] / float(max(1, len(paired_rows))), 4),
        },
    )
    write_json(
        trainer_config_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "model_family": "nvidia/Cosmos-Predict2.5-2B",
            "adapter_type": "lora",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "conditioning_modes": [
                "paired_reference_target",
                "k_reference_conditioning",
                "plucker_conditioning",
            ],
            "source_mode": source_mode,
            "reference_selection_policy": reference_selection_manifest.get("policy"),
            "target_reference_decoupling_mode": str(
                (reference_selection_manifest.get("policy") or {}).get("target_reference_decoupling_mode") or "unknown"
            ),
            "dataset_paths": {
                "paired_reference_target": str(paired_path.resolve()),
                "k_reference_conditioning": str(k_reference_path.resolve()),
                "train_val_split": str(split_path.resolve()),
                "reference_selection_manifest": str(reference_selection_manifest_path.resolve()),
                "reference_selection_comparison": str(reference_selection_comparison_path.resolve()),
                "synthetic_trajectory_manifest": str(synthetic_trajectory_manifest_path.resolve()),
                "sparse_view_interpolation_manifest": str(sparse_view_interpolation_manifest_path.resolve()),
                "future_anchor_regrounding_manifest": str(future_anchor_regrounding_manifest_path.resolve()),
            },
        },
    )
    write_json(
        checkpoint_layout_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "root_dir": str((export_root / "checkpoints").resolve()),
            "expected_files": [
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "trainer_state.json",
            ],
        },
    )
    write_json(
        inference_backend_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "backend_name": "cosmos_predict_lora_adapter",
            "default_conditioning_mode": "k_reference_conditioning",
            "plucker_conditioning_required": True,
            "conditioning_bundle_path": str(conditioning_bundle_path.resolve()) if conditioning_bundle_path.is_file() else None,
            "protected_regions_manifest_path": str(protected_regions_path.resolve()) if protected_regions_path.is_file() else None,
            "task_anchor_manifest_path": str(task_anchor_path.resolve()) if task_anchor_path.is_file() else None,
            "source_mode": source_mode,
            "reference_selection_policy": reference_selection_manifest.get("policy"),
            "target_reference_decoupling_mode": str(
                (reference_selection_manifest.get("policy") or {}).get("target_reference_decoupling_mode") or "unknown"
            ),
            "reference_selection_quality_comparison": reference_selection_comparison,
            "synthetic_trajectory_augmentation": {
                "manifest_path": str(synthetic_trajectory_manifest_path.resolve()),
                "policy": synthetic_trajectory_manifest.get("policy"),
                "augmented_target_count": synthetic_trajectory_manifest.get("augmented_target_count"),
                "synthetic_waypoint_count": synthetic_trajectory_manifest.get("synthetic_waypoint_count"),
            },
            "sparse_view_interpolation": {
                "manifest_path": str(sparse_view_interpolation_manifest_path.resolve()),
                "policy": sparse_view_interpolation_manifest.get("policy"),
                "interpolated_target_count": sparse_view_interpolation_manifest.get("interpolated_target_count"),
                "interpolated_view_count": sparse_view_interpolation_manifest.get("interpolated_view_count"),
            },
            "future_anchor_regrounding": {
                "manifest_path": str(future_anchor_regrounding_manifest_path.resolve()),
                "policy": future_anchor_regrounding_manifest.get("policy"),
                "re_grounded_target_count": future_anchor_regrounding_manifest.get("re_grounded_target_count"),
            },
        },
    )

    manifest = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if paired_rows else "missing",
        "reason": None if paired_rows else "insufficient_decoupled_reference_targets",
        "capture_id": context.capture_id,
        "scene_id": context.scene_id,
        "source_mode": source_mode,
        "bootstrap_origin": bootstrap_origin,
        "geometry_provenance": geometry_provenance,
        "rejected_record_count": len(rejected_rows),
        "export_rejection_manifest_path": str(rejection_manifest_path.resolve()),
        "paired_reference_target_path": str(paired_path.resolve()),
        "k_reference_conditioning_path": str(k_reference_path.resolve()),
        "train_val_split_path": str(split_path.resolve()),
        "trainer_config_path": str(trainer_config_path.resolve()),
        "checkpoint_layout_path": str(checkpoint_layout_path.resolve()),
        "inference_backend_shape_path": str(inference_backend_path.resolve()),
        "bootstrap_source_manifest_path": str(bootstrap_source_manifest_path.resolve()) if bootstrap_source_manifest_path else None,
        "reference_selection_manifest_path": str(reference_selection_manifest_path.resolve()),
        "reference_selection_comparison_path": str(reference_selection_comparison_path.resolve()),
        "synthetic_trajectory_manifest_path": str(synthetic_trajectory_manifest_path.resolve()),
        "sparse_view_interpolation_manifest_path": str(sparse_view_interpolation_manifest_path.resolve()),
        "future_anchor_regrounding_manifest_path": str(future_anchor_regrounding_manifest_path.resolve()),
        "reference_selection_policy": reference_selection_manifest.get("policy"),
        "target_reference_decoupling_mode": str(
            (reference_selection_manifest.get("policy") or {}).get("target_reference_decoupling_mode") or "unknown"
        ),
        "reference_selection_quality_comparison": reference_selection_comparison,
        "synthetic_trajectory_augmentation": {
            "policy": synthetic_trajectory_manifest.get("policy"),
            "augmented_target_count": synthetic_trajectory_manifest.get("augmented_target_count"),
            "skipped_sparse_context_count": synthetic_trajectory_manifest.get("skipped_sparse_context_count"),
            "synthetic_waypoint_count": synthetic_trajectory_manifest.get("synthetic_waypoint_count"),
        },
        "sparse_view_interpolation": {
            "policy": sparse_view_interpolation_manifest.get("policy"),
            "interpolated_target_count": sparse_view_interpolation_manifest.get("interpolated_target_count"),
            "skipped_sparse_target_count": sparse_view_interpolation_manifest.get("skipped_sparse_target_count"),
            "interpolated_view_count": sparse_view_interpolation_manifest.get("interpolated_view_count"),
        },
        "future_anchor_regrounding": {
            "policy": future_anchor_regrounding_manifest.get("policy"),
            "re_grounded_target_count": future_anchor_regrounding_manifest.get("re_grounded_target_count"),
            "skipped_target_count": future_anchor_regrounding_manifest.get("skipped_target_count"),
        },
        "conditioning_bundle_path": str(conditioning_bundle_path.resolve()) if conditioning_bundle_path.is_file() else None,
        "task_anchor_manifest_path": str(task_anchor_path.resolve()) if task_anchor_path.is_file() else None,
        "protected_regions_manifest_path": str(protected_regions_path.resolve()) if protected_regions_path.is_file() else None,
        "paired_example_count": len(paired_rows),
        "k_reference_example_count": len(k_reference_rows),
        "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
        "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
        "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
        "train_count": split_summary["train"],
        "val_count": split_summary["val"],
        "protected_region_count": len(list(protected_regions_manifest.get("regions") or []))
        if isinstance(protected_regions_manifest, Mapping)
        else 0,
        "task_count": len(list(task_anchor_manifest.get("tasks") or []))
        if isinstance(task_anchor_manifest, Mapping)
        else 0,
        "conditioning_modes": [
            "paired_reference_target",
            "k_reference_conditioning",
            "plucker_conditioning",
        ],
    }
    write_json(export_root / "manifest.json", manifest)
    return manifest
