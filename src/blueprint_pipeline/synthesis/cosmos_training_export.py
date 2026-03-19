"""Cosmos Predict fine-tuning/export substrate for site-grounded captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ..common import ensure_dir, read_json_any, resolve_gs_uri_to_path, utc_now_iso, write_json
from ..local_capture import resolve_local_capture_context
from .plucker_rays import compute_plucker_map


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


def _normalized_intrinsics(record: Mapping[str, Any]) -> Dict[str, float]:
    intrinsics = dict(record.get("intrinsics") or {}) if isinstance(record.get("intrinsics"), Mapping) else {}
    width = float(intrinsics.get("width") or intrinsics.get("w") or 640.0)
    height = float(intrinsics.get("height") or intrinsics.get("h") or 480.0)
    return {
        "fx": float(intrinsics.get("fx") or max(width, 1.0)),
        "fy": float(intrinsics.get("fy") or max(height, 1.0)),
        "cx": float(intrinsics.get("cx") or width / 2.0),
        "cy": float(intrinsics.get("cy") or height / 2.0),
        "width": width,
        "height": height,
    }


def _select_references(records: Sequence[Mapping[str, Any]], *, target_index: int, k: int) -> List[Dict[str, Any]]:
    target_time = float(records[target_index].get("t_capture_sec") or 0.0)
    candidates: List[tuple[float, Dict[str, Any]]] = []
    for index, record in enumerate(records):
        if index == target_index:
            continue
        frame_uri = str(record.get("frame_uri") or "").strip()
        if not frame_uri:
            continue
        delta = abs(float(record.get("t_capture_sec") or 0.0) - target_time)
        candidates.append((delta, dict(record)))
    candidates.sort(key=lambda item: item[0])
    return [item for _delta, item in candidates[: max(1, k)]]


def _split_name(frame_id: str) -> str:
    digest = hashlib.sha256(frame_id.encode("utf-8")).hexdigest()
    return "val" if int(digest[:2], 16) < 51 else "train"


def _read_pose_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    return [row for row in rows if row.get("T_world_camera") or row.get("transform")]


def _extract_video_bootstrap_records(
    *,
    context,
    conditioning_bundle: Mapping[str, Any],
    export_root: Path,
    max_frames: int,
) -> List[Dict[str, Any]]:
    raw_video_uri = str(conditioning_bundle.get("raw_video_uri") or "").strip()
    arkit = dict(conditioning_bundle.get("arkit") or {}) if isinstance(conditioning_bundle.get("arkit"), Mapping) else {}
    poses_uri = str(arkit.get("poses_uri") or "").strip()
    intrinsics_uri = str(arkit.get("intrinsics_uri") or "").strip()
    if not raw_video_uri or not poses_uri or not intrinsics_uri:
        return []

    video_path = resolve_gs_uri_to_path(raw_video_uri, context.storage_root)
    poses_path = resolve_gs_uri_to_path(poses_uri, context.storage_root)
    intrinsics_path = resolve_gs_uri_to_path(intrinsics_uri, context.storage_root)
    if not video_path.is_file() or not poses_path.is_file() or not intrinsics_path.is_file():
        return []

    pose_rows = _read_pose_rows(poses_path)
    intrinsics_payload = read_json_any(intrinsics_path)
    intrinsics = dict(intrinsics_payload) if isinstance(intrinsics_payload, Mapping) else {}
    if not pose_rows:
        return []

    try:
        import cv2  # type: ignore[import]
    except ImportError:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    output_dir = export_root / "video_bootstrap_frames"
    ensure_dir(output_dir)

    target_count = min(max_frames, len(pose_rows))
    if target_count < 2:
        capture.release()
        return []

    pose_indices = [
        round(index * (len(pose_rows) - 1) / float(max(1, target_count - 1)))
        for index in range(target_count)
    ]
    records: List[Dict[str, Any]] = []
    for export_index, pose_index in enumerate(sorted(dict.fromkeys(pose_indices))):
        pose_row = pose_rows[pose_index]
        frame_index = round(pose_index * (total_frames - 1) / float(max(1, len(pose_rows) - 1)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frame_path = output_dir / f"frame_{export_index:04d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            continue
        records.append(
            {
                "frame_id": f"video_bootstrap_{export_index:04d}",
                "frame_uri": str(frame_path.resolve()),
                "embedding_uri": None,
                "included_in_index": True,
                "t_capture_sec": pose_row.get("t_capture_sec", pose_row.get("t_device_sec", float(pose_index))),
                "T_world_camera": pose_row.get("T_world_camera") or pose_row.get("transform"),
                "intrinsics": intrinsics,
                "anchor_observations": [],
                "source_mode": "video_bootstrap",
                "source_video_uri": raw_video_uri,
            }
        )
    capture.release()
    return records


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

    if len(dense_records) < 2 and isinstance(conditioning_bundle, Mapping):
        dense_records = _extract_video_bootstrap_records(
            context=context,
            conditioning_bundle=conditioning_bundle,
            export_root=export_root,
            max_frames=max_video_bootstrap_frames,
        )
        if dense_records:
            source_mode = "video_bootstrap"

    if len(dense_records) < 2:
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "missing",
            "reason": "insufficient_dense_index_records",
            "source_mode": source_mode,
            "paired_reference_target_path": None,
            "k_reference_conditioning_path": None,
        }
        write_json(export_root / "manifest.json", manifest)
        return manifest

    paired_rows: List[Dict[str, Any]] = []
    k_reference_rows: List[Dict[str, Any]] = []
    split_summary = {"train": 0, "val": 0}

    for target_index, record in enumerate(dense_records):
        references = _select_references(dense_records, target_index=target_index, k=k_references)
        if not references:
            continue
        frame_id = str(record.get("frame_id") or "").strip()
        split = _split_name(frame_id or f"frame_{target_index}")
        split_summary[split] += 1
        intrinsics = _normalized_intrinsics(record)
        target_T = np.array(record.get("T_world_camera") or np.eye(4), dtype=np.float32)
        if target_T.ndim == 1 and target_T.size == 16:
            target_T = target_T.reshape(4, 4)
        if target_T.shape != (4, 4):
            target_T = np.eye(4, dtype=np.float32)
        plucker = compute_plucker_map(
            T_world_camera=target_T,
            intrinsics=intrinsics,
            height=max(16, int(intrinsics["height"])),
            width=max(16, int(intrinsics["width"])),
        )
        plucker_path = plucker_root / f"{frame_id or f'target_{target_index}'}.npz"
        np.savez_compressed(plucker_path, plucker=plucker)

        paired_rows.append(
            {
                "capture_id": context.capture_id,
                "scene_id": context.scene_id,
                "frame_id": frame_id,
                "split": split,
                "target_frame_uri": record.get("frame_uri"),
                "primary_reference_frame_uri": references[0].get("frame_uri"),
                "plucker_conditioning_path": str(plucker_path.resolve()),
                "task_anchor_manifest_path": str(task_anchor_path.resolve()) if task_anchor_path.is_file() else None,
                "protected_regions_manifest_path": str(protected_regions_path.resolve()) if protected_regions_path.is_file() else None,
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
                "reference_frame_uris": [ref.get("frame_uri") for ref in references if str(ref.get("frame_uri") or "").strip()],
                "reference_embedding_uris": [ref.get("embedding_uri") for ref in references if str(ref.get("embedding_uri") or "").strip()],
                "plucker_conditioning_path": str(plucker_path.resolve()),
                "conditioning_bundle_path": str(conditioning_bundle_path.resolve()) if conditioning_bundle_path.is_file() else None,
                "source_mode": str(record.get("source_mode") or source_mode),
            }
        )

    paired_path = export_root / "paired_reference_target.jsonl"
    k_reference_path = export_root / "k_reference_conditioning.jsonl"
    split_path = export_root / "train_val_split.json"
    trainer_config_path = export_root / "trainer_config.json"
    checkpoint_layout_path = export_root / "checkpoint_layout.json"
    inference_backend_path = export_root / "inference_backend_shape.json"

    _write_jsonl(paired_path, paired_rows)
    _write_jsonl(k_reference_path, k_reference_rows)
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
            "dataset_paths": {
                "paired_reference_target": str(paired_path.resolve()),
                "k_reference_conditioning": str(k_reference_path.resolve()),
                "train_val_split": str(split_path.resolve()),
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
        },
    )

    manifest = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "ready" if paired_rows else "missing",
        "capture_id": context.capture_id,
        "scene_id": context.scene_id,
        "source_mode": source_mode,
        "paired_reference_target_path": str(paired_path.resolve()),
        "k_reference_conditioning_path": str(k_reference_path.resolve()),
        "train_val_split_path": str(split_path.resolve()),
        "trainer_config_path": str(trainer_config_path.resolve()),
        "checkpoint_layout_path": str(checkpoint_layout_path.resolve()),
        "inference_backend_shape_path": str(inference_backend_path.resolve()),
        "conditioning_bundle_path": str(conditioning_bundle_path.resolve()) if conditioning_bundle_path.is_file() else None,
        "task_anchor_manifest_path": str(task_anchor_path.resolve()) if task_anchor_path.is_file() else None,
        "protected_regions_manifest_path": str(protected_regions_path.resolve()) if protected_regions_path.is_file() else None,
        "paired_example_count": len(paired_rows),
        "k_reference_example_count": len(k_reference_rows),
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
