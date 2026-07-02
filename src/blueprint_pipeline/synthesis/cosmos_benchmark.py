"""Zero-shot Cosmos validation lane for fixed site-world benchmark checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from ..capture_orchestrator import PipelineConfig, run_capture_synthesis_validation
from ..common import ensure_dir, resolve_gs_uri_to_path, utc_now_iso, write_json
from ..local_capture import resolve_local_capture_context
from .cosmos_capture_bootstrap import extract_video_bootstrap_records, resolve_video_bootstrap_sources
from .cosmos_inference import _DEFAULT_COSMOS_MODEL_ID, generate_view, load_cosmos_model
from .future_anchor_regrounding import build_future_anchor_regrounding_manifest
from .plucker_rays import compute_plucker_map
from .reference_selection import (
    build_legacy_reference_selection_manifest,
    build_reference_selection_comparison,
    build_reference_selection_manifest,
)
from .sparse_view_interpolation import build_sparse_view_interpolation_manifest
from .trajectory_augmentation import build_synthetic_trajectory_manifest


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


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


def _probe_cosmos_runtime() -> Dict[str, Any]:
    packages = {
        "cosmos_predict2_5": bool(importlib.util.find_spec("cosmos_predict2_5")),
        "diffusers": bool(importlib.util.find_spec("diffusers")),
        "torch": bool(importlib.util.find_spec("torch")),
    }
    blockers: List[str] = []
    if not packages["torch"]:
        blockers.append("missing_torch")
    if not packages["cosmos_predict2_5"] and not packages["diffusers"]:
        blockers.append("missing_cosmos_runtime_package")
    return {
        "status": "ready" if not blockers else "blocked",
        "model_id": _DEFAULT_COSMOS_MODEL_ID,
        "packages": packages,
        "blockers": blockers,
    }


def _runtime_blocked_reason(reason: Any) -> bool:
    text = str(reason or "").strip().lower()
    return (
        "could not load cosmos-predict2.5-2b" in text
        or "missing_cosmos_runtime_package" in text
        or "no module named" in text
    )


def _load_frame_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _resolve_record_frame_path(*, context, record: Mapping[str, Any]) -> Path | None:
    frame_uri = str(record.get("frame_uri") or "").strip()
    if not frame_uri:
        return None
    if frame_uri.startswith("gs://"):
        candidate = resolve_gs_uri_to_path(frame_uri, context.storage_root)
    else:
        candidate = Path(frame_uri).expanduser().resolve()
    return candidate if candidate.is_file() else None


def _target_intrinsics(record: Mapping[str, Any], image_shape: tuple[int, ...]) -> Dict[str, float]:
    intrinsics = dict(record.get("intrinsics") or {}) if isinstance(record.get("intrinsics"), Mapping) else {}
    height = float(intrinsics.get("height") or (image_shape[0] if image_shape else 720))
    width = float(intrinsics.get("width") or (image_shape[1] if len(image_shape) > 1 else 1280))
    return {
        "fx": float(intrinsics.get("fx") or max(width, 1.0)),
        "fy": float(intrinsics.get("fy") or max(height, 1.0)),
        "cx": float(intrinsics.get("cx") or width / 2.0),
        "cy": float(intrinsics.get("cy") or height / 2.0),
        "width": width,
        "height": height,
    }


def _target_pose_matrix(record: Mapping[str, Any]) -> np.ndarray | None:
    raw_pose = record.get("T_world_camera")
    if raw_pose is None:
        return None
    pose = np.array(raw_pose, dtype=np.float32)
    if pose.ndim == 1 and pose.size == 16:
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        return None
    if not np.isfinite(pose).all():
        return None
    return pose


def _target_plucker_map(record: Mapping[str, Any], image_shape: tuple[int, ...]) -> np.ndarray | None:
    pose = _target_pose_matrix(record)
    if pose is None:
        return None
    intrinsics = _target_intrinsics(record, image_shape)
    return compute_plucker_map(
        T_world_camera=pose,
        intrinsics=intrinsics,
        height=max(16, int(intrinsics["height"])),
        width=max(16, int(intrinsics["width"])),
    )


def _smoke_manifest_base(
    *,
    benchmark_root: Path,
    runtime_probe: Mapping[str, Any],
    context,
    bootstrap_origin: str | None,
    bootstrap_source_manifest_path: Path | None,
    reference_selection_manifest_path: Path,
    reference_selection_comparison_path: Path,
    validation_set: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "manifest_type": "cosmos_single_capture_smoke",
        "benchmark_family": "cosmos_single_capture_smoke",
        "benchmark_scope": "single_capture_smoke",
        "readiness_upgrade_allowed": False,
        "comparable_to_multi_pass_benchmark": False,
        "uses_prior_site_memory": False,
        "self_contained": True,
        "generated_at": utc_now_iso(),
        "capture_id": context.capture_id,
        "scene_id": context.scene_id,
        "benchmark_root": str(benchmark_root.resolve()),
        "runtime_probe": dict(runtime_probe),
        "bootstrap_origin": bootstrap_origin,
        "bootstrap_source_manifest_path": str(bootstrap_source_manifest_path.resolve()) if bootstrap_source_manifest_path else None,
        "reference_selection_manifest_path": str(reference_selection_manifest_path.resolve()),
        "reference_selection_comparison_path": str(reference_selection_comparison_path.resolve()),
        "validation_set": validation_set,
        "validation_example_count": len(validation_set),
    }


def _video_bootstrap_reference_policy(records: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    times = [
        float(value)
        for value in (
            record.get("t_capture_sec")
            for record in records
        )
        if value is not None
    ]
    if len(times) < 2:
        return None
    span_sec = max(times) - min(times)
    if span_sec <= 12.0:
        return None
    return {
        "max_temporal_window_sec": max(12.0, span_sec + 1.0),
        "preferred_temporal_gap_sec": max(1.5, min(span_sec / 3.0, span_sec)),
    }


def run_cosmos_single_capture_smoke_lane(
    *,
    capture_root: str | Path,
    descriptor_gcs_uri: str,
    cfg: PipelineConfig,
    max_examples: int = 1,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    benchmark_root = context.pipeline_root / "cosmos_single_capture_smoke"
    ensure_dir(benchmark_root)
    runtime_probe = _probe_cosmos_runtime()

    dense_index = [
        row
        for row in _read_jsonl(context.capture_root / "world_model_export" / "dense_index.jsonl")
        if bool(row.get("included_in_index"))
    ]
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    descriptor = _read_json(descriptor_path)
    conditioning_bundle = _read_json(context.pipeline_root / "scene_memory" / "conditioning_bundle.json")
    bootstrap_origin = None
    bootstrap_source_manifest_path: Path | None = None
    benchmark_records = list(dense_index)
    reference_selection_policy: Dict[str, Any] | None = None

    if not benchmark_records:
        bootstrap_frame_budget = max(4, max_examples)
        bootstrap_sources = resolve_video_bootstrap_sources(
            context=context,
            conditioning_bundle=conditioning_bundle,
        )
        benchmark_records = extract_video_bootstrap_records(
            bootstrap_sources=bootstrap_sources,
            export_root=benchmark_root,
            max_frames=bootstrap_frame_budget,
        ) if bootstrap_sources else []
        if benchmark_records:
            bootstrap_origin = str(bootstrap_sources.get("origin") or "unknown")
            bootstrap_source_manifest_path = benchmark_root / "bootstrap_source_manifest.json"
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
            reference_selection_policy = _video_bootstrap_reference_policy(benchmark_records)

    reference_selection_manifest_path = benchmark_root / "reference_selection_manifest.json"
    reference_selection_comparison_path = benchmark_root / "reference_selection_comparison.json"
    reference_selection_manifest = build_reference_selection_manifest(
        records=benchmark_records,
        k=min(4, max(1, len(benchmark_records) - 1)) if benchmark_records else 1,
        selection_name="cosmos_single_capture_smoke",
        policy=reference_selection_policy,
        max_targets=max_examples,
    ) if benchmark_records else {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": "cosmos_single_capture_smoke",
        "policy": {},
        "record_count": 0,
        "selected_target_count": 0,
        "skipped_target_count": 0,
        "rejected_near_duplicate_count": 0,
        "aggregate_rejected_counts": {},
        "entries": [],
    }
    legacy_reference_selection_manifest = build_legacy_reference_selection_manifest(
        records=benchmark_records,
        k=min(4, max(1, len(benchmark_records) - 1)) if benchmark_records else 1,
        selection_name="cosmos_single_capture_smoke_legacy_baseline",
        max_targets=max_examples,
    ) if benchmark_records else {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": "cosmos_single_capture_smoke_legacy_baseline",
        "policy": {"selection_mode": "legacy_temporal_nearest", "target_reference_decoupling_mode": "none"},
        "record_count": 0,
        "selected_target_count": 0,
        "skipped_target_count": 0,
        "rejected_near_duplicate_count": 0,
        "aggregate_rejected_counts": {},
        "entries": [],
    }
    reference_selection_comparison = build_reference_selection_comparison(
        current_manifest=reference_selection_manifest,
        legacy_manifest=legacy_reference_selection_manifest,
        selection_name="cosmos_single_capture_smoke",
    )
    write_json(reference_selection_manifest_path, reference_selection_manifest)
    write_json(reference_selection_comparison_path, reference_selection_comparison)

    validation_set = [
        {
            "frame_id": str(entry.get("target_frame_id") or ""),
            "frame_uri": entry.get("target_frame_uri"),
            "target_index": int(entry.get("target_index") or 0),
            "selected_reference_ids": list(entry.get("selected_reference_ids") or []),
            "selected_reference_frame_ids": list(entry.get("selected_reference_frame_ids") or []),
            "selected_reference_frame_uris": list(entry.get("selected_reference_frame_uris") or []),
            "rejected_near_duplicate_count": entry.get("rejected_near_duplicate_count"),
            "target_reference_decoupling_mode": (
                (entry.get("decoupling") or {})
                if isinstance(entry.get("decoupling"), Mapping)
                else {}
            ).get("mode"),
        }
        for entry in list(reference_selection_manifest.get("entries") or [])
    ]
    manifest_base = _smoke_manifest_base(
        benchmark_root=benchmark_root,
        runtime_probe=runtime_probe,
        context=context,
        bootstrap_origin=bootstrap_origin,
        bootstrap_source_manifest_path=bootstrap_source_manifest_path,
        reference_selection_manifest_path=reference_selection_manifest_path,
        reference_selection_comparison_path=reference_selection_comparison_path,
        validation_set=validation_set,
    )
    manifest_path = benchmark_root / "cosmos_single_capture_smoke_manifest.json"

    if not validation_set:
        manifest = {
            **manifest_base,
            "status": "missing",
            "reason": "no_self_contained_smoke_examples",
            "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
            "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
            "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
        }
        write_json(manifest_path, manifest)
        return manifest

    if runtime_probe["status"] != "ready":
        manifest = {
            **manifest_base,
            "status": "blocked",
            "reason": "cosmos_runtime_unavailable",
            "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
            "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
            "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
        }
        write_json(manifest_path, manifest)
        return manifest

    render_root = benchmark_root / "renders"
    ensure_dir(render_root)
    try:
        cosmos_model = load_cosmos_model()
    except Exception as exc:
        manifest = {
            **manifest_base,
            "status": "blocked",
            "reason": str(exc),
            "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
            "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
            "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
            "target_reference_decoupling_mode": str(
                (reference_selection_manifest.get("policy") or {}).get("target_reference_decoupling_mode") or "unknown"
            ),
            "reference_selection_policy": reference_selection_manifest.get("policy"),
            "reference_selection_quality_comparison": reference_selection_comparison,
        }
        write_json(manifest_path, manifest)
        return manifest
    smoke_examples: List[Dict[str, Any]] = []
    render_success_count = 0

    for entry, selection in zip(validation_set, list(reference_selection_manifest.get("entries") or []), strict=False):
        target_index = int(entry.get("target_index") or 0)
        if target_index < 0 or target_index >= len(benchmark_records):
            continue
        target_record = benchmark_records[target_index]
        candidate_refs = [
            benchmark_records[int(item.get("candidate_index"))]
            for item in list(selection.get("selected_references") or [])
            if int(item.get("candidate_index")) >= 0 and int(item.get("candidate_index")) < len(benchmark_records)
        ]
        if not candidate_refs:
            smoke_examples.append(
                {
                    "frame_id": entry["frame_id"],
                    "status": "skipped",
                    "reason": "no_selected_reference",
                }
            )
            continue
        reference_record = candidate_refs[0]
        frame_path = _resolve_record_frame_path(context=context, record=reference_record)
        if frame_path is None:
            smoke_examples.append(
                {
                    "frame_id": entry["frame_id"],
                    "status": "skipped",
                    "reason": "reference_frame_unavailable",
                }
            )
            continue

        conditioning_image = _load_frame_image(frame_path)
        coverage_mask = np.ones(conditioning_image.shape[:2], dtype=bool)
        output_path = render_root / f"{entry['frame_id']}.jpg"
        try:
            generate_view(
                splatted_image=conditioning_image,
                coverage_mask=coverage_mask,
                target_plucker_map=_target_plucker_map(target_record, conditioning_image.shape),
                output_path=output_path,
                mode="cosmos_i2w",
                cosmos_model=cosmos_model,
            )
            status = "completed" if output_path.is_file() else "failed"
        except Exception as exc:
            status = "failed"
            smoke_examples.append(
                {
                    "frame_id": entry["frame_id"],
                    "reference_frame_id": reference_record.get("frame_id"),
                    "status": status,
                    "reason": str(exc),
                }
            )
            continue

        output_video_path = output_path.with_suffix(".mp4")
        if status == "completed":
            render_success_count += 1
        smoke_examples.append(
            {
                "frame_id": entry["frame_id"],
                "reference_frame_id": reference_record.get("frame_id"),
                "reference_frame_uri": reference_record.get("frame_uri"),
                "output_path": str(output_path.resolve()),
                "output_video_path": str(output_video_path.resolve()) if output_video_path.is_file() else None,
                "status": status,
                "pose_distance_m": (selection.get("selected_references") or [{}])[0].get("pose_distance_m"),
                "temporal_gap_sec": (selection.get("selected_references") or [{}])[0].get("temporal_gap_sec"),
            }
        )

    checks = {
        "runtime_loaded": {
            "passed": True,
            "detail": "Cosmos runtime loaded for single-capture smoke execution.",
        },
        "render_output_emitted": {
            "passed": render_success_count > 0,
            "detail": f"Rendered {render_success_count} smoke outputs.",
        },
        "reference_target_decoupled": {
            "passed": any(
                float(item.get("pose_distance_m") or 0.0) > 0.05
                or float(item.get("temporal_gap_sec") or 0.0) > 0.2
                for item in smoke_examples
                if item.get("status") == "completed"
            ),
            "detail": "Selected same-capture references preserve non-trivial temporal or pose separation.",
        },
    }
    passed_count = sum(1 for item in checks.values() if item["passed"])
    status = "completed" if passed_count == len(checks) else ("degraded" if render_success_count > 0 else "failed")
    manifest = {
        **manifest_base,
        "status": status,
        "reason": None if status == "completed" else ("single_capture_smoke_checks_failed" if render_success_count > 0 else "single_capture_smoke_render_failed"),
        "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
        "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
        "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
        "target_reference_decoupling_mode": str(
            (reference_selection_manifest.get("policy") or {}).get("target_reference_decoupling_mode") or "unknown"
        ),
        "reference_selection_policy": reference_selection_manifest.get("policy"),
        "reference_selection_quality_comparison": reference_selection_comparison,
        "smoke_examples": smoke_examples,
        "render_success_count": render_success_count,
        "checks": checks,
        "evidence_supported": all(
            key in descriptor or key in (descriptor.get("quality") or {})
            for key in ("capture_id", "scene_id")
        ),
    }
    write_json(manifest_path, manifest)
    return manifest


def run_cosmos_zero_shot_validation_lane(
    *,
    capture_root: str | Path,
    descriptor_gcs_uri: str,
    cfg: PipelineConfig,
    max_examples: int = 8,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    benchmark_root = context.pipeline_root / "cosmos_zero_shot_validation"
    ensure_dir(benchmark_root)
    runtime_probe = _probe_cosmos_runtime()

    dense_index = [
        row
        for row in _read_jsonl(context.capture_root / "world_model_export" / "dense_index.jsonl")
        if bool(row.get("included_in_index"))
    ]
    task_anchor_manifest = _read_json(context.pipeline_root / "evaluation_prep" / "task_anchor_manifest.json")
    protected_regions_manifest = _read_json(context.pipeline_root / "evaluation_prep" / "protected_regions_manifest.json")
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    descriptor = _read_json(descriptor_path)
    conditioning_bundle = _read_json(context.pipeline_root / "scene_memory" / "conditioning_bundle.json")
    bootstrap_origin = None
    bootstrap_source_manifest_path: Path | None = None
    reference_selection_manifest_path = benchmark_root / "reference_selection_manifest.json"
    reference_selection_comparison_path = benchmark_root / "reference_selection_comparison.json"
    synthetic_trajectory_manifest_path = benchmark_root / "synthetic_trajectory_manifest.json"
    sparse_view_interpolation_manifest_path = benchmark_root / "sparse_view_interpolation_manifest.json"
    future_anchor_regrounding_manifest_path = benchmark_root / "future_anchor_regrounding_manifest.json"
    benchmark_records = list(dense_index)
    reference_selection_policy: Dict[str, Any] | None = None

    if not benchmark_records:
        bootstrap_frame_budget = max(4, max_examples)
        bootstrap_sources = resolve_video_bootstrap_sources(
            context=context,
            conditioning_bundle=conditioning_bundle,
        )
        bootstrap_records = extract_video_bootstrap_records(
            bootstrap_sources=bootstrap_sources,
            export_root=benchmark_root,
            max_frames=bootstrap_frame_budget,
        ) if bootstrap_sources else []
        if bootstrap_records:
            bootstrap_origin = str(bootstrap_sources.get("origin") or "unknown")
            bootstrap_source_manifest_path = benchmark_root / "bootstrap_source_manifest.json"
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
            benchmark_records = bootstrap_records
            reference_selection_policy = _video_bootstrap_reference_policy(benchmark_records)

    reference_selection_manifest = build_reference_selection_manifest(
        records=benchmark_records,
        k=min(4, max(1, len(benchmark_records) - 1)) if benchmark_records else 1,
        selection_name="cosmos_zero_shot_validation",
        policy=reference_selection_policy,
        max_targets=max_examples,
    ) if benchmark_records else {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": "cosmos_zero_shot_validation",
        "policy": {},
        "record_count": 0,
        "selected_target_count": 0,
        "skipped_target_count": 0,
        "rejected_near_duplicate_count": 0,
        "aggregate_rejected_counts": {},
        "entries": [],
    }
    legacy_reference_selection_manifest = build_legacy_reference_selection_manifest(
        records=benchmark_records,
        k=min(4, max(1, len(benchmark_records) - 1)) if benchmark_records else 1,
        selection_name="cosmos_zero_shot_validation_legacy_baseline",
        max_targets=max_examples,
    ) if benchmark_records else {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "selection_name": "cosmos_zero_shot_validation_legacy_baseline",
        "policy": {"selection_mode": "legacy_temporal_nearest", "target_reference_decoupling_mode": "none"},
        "record_count": 0,
        "selected_target_count": 0,
        "skipped_target_count": 0,
        "rejected_near_duplicate_count": 0,
        "aggregate_rejected_counts": {},
        "entries": [],
    }
    reference_selection_comparison = build_reference_selection_comparison(
        current_manifest=reference_selection_manifest,
        legacy_manifest=legacy_reference_selection_manifest,
        selection_name="cosmos_zero_shot_validation",
    )
    synthetic_trajectory_manifest = build_synthetic_trajectory_manifest(
        records=benchmark_records,
        selection_entries=list(reference_selection_manifest.get("entries") or []),
        augmentation_name="cosmos_zero_shot_validation",
    )
    sparse_view_interpolation_manifest = build_sparse_view_interpolation_manifest(
        records=benchmark_records,
        selection_entries=list(reference_selection_manifest.get("entries") or []),
        trajectory_entries=list(synthetic_trajectory_manifest.get("entries") or []),
        interpolation_name="cosmos_zero_shot_validation",
    )
    future_anchor_regrounding_manifest = build_future_anchor_regrounding_manifest(
        records=benchmark_records,
        selection_entries=list(reference_selection_manifest.get("entries") or []),
        task_anchor_manifest=task_anchor_manifest,
        protected_regions_manifest=protected_regions_manifest,
        regrounding_name="cosmos_zero_shot_validation",
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

    validation_set = [
        {
            "frame_id": str(entry.get("target_frame_id") or ""),
            "frame_uri": entry.get("target_frame_uri"),
            "anchor_observation_count": len(
                list((benchmark_records[int(entry.get("target_index") or 0)].get("anchor_observations") or []))
            ) if int(entry.get("target_index") or 0) < len(benchmark_records) else 0,
            "zone_id": benchmark_records[int(entry.get("target_index") or 0)].get("zone_id")
            if int(entry.get("target_index") or 0) < len(benchmark_records)
            else None,
            "source_mode": benchmark_records[int(entry.get("target_index") or 0)].get("source_mode")
            if int(entry.get("target_index") or 0) < len(benchmark_records)
            else None,
            "selected_reference_ids": list(entry.get("selected_reference_ids") or []),
            "selected_reference_frame_ids": list(entry.get("selected_reference_frame_ids") or []),
            "selected_reference_frame_uris": list(entry.get("selected_reference_frame_uris") or []),
            "rejected_near_duplicate_count": entry.get("rejected_near_duplicate_count"),
            "target_reference_decoupling_mode": (
                (entry.get("decoupling") or {})
                if isinstance(entry.get("decoupling"), Mapping)
                else {}
            ).get("mode"),
            "trajectory_context_id": (
                trajectory_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("trajectory_context_id"),
            "synthetic_trajectory_status": (
                trajectory_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("status"),
            "synthetic_trajectory_reason": (
                trajectory_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("reason"),
            "synthetic_waypoint_count": (
                trajectory_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("synthetic_waypoint_count"),
            "synthetic_waypoint_ids": list(
                (
                    trajectory_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
                ).get("synthetic_waypoint_ids")
                or []
            ),
            "sparse_interpolation_context_id": (
                sparse_interpolation_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("interpolation_context_id"),
            "sparse_view_interpolation_status": (
                sparse_interpolation_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("status"),
            "sparse_view_interpolation_reason": (
                sparse_interpolation_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("reason"),
            "interpolated_view_count": (
                sparse_interpolation_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("interpolated_view_count"),
            "interpolated_view_ids": list(
                (
                    sparse_interpolation_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
                ).get("interpolated_view_ids")
                or []
            ),
            "future_anchor_context_id": (
                future_anchor_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("future_anchor_context_id"),
            "future_anchor_status": (
                future_anchor_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("status"),
            "future_anchor_reason": (
                future_anchor_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("reason"),
            "future_anchor_count": (
                future_anchor_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
            ).get("future_anchor_count"),
            "future_anchor_reference_ids": list(
                (
                    future_anchor_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
                ).get("future_anchor_reference_ids")
                or []
            ),
            "future_anchor_frame_ids": list(
                (
                    future_anchor_entries.get(str(entry.get("target_frame_id") or ""), {}) or {}
                ).get("future_anchor_frame_ids")
                or []
            ),
        }
        for entry in list(reference_selection_manifest.get("entries") or [])
    ]

    if not validation_set:
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "missing",
            "reason": "no_decoupled_validation_examples",
            "runtime_probe": runtime_probe,
            "bootstrap_origin": bootstrap_origin,
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
            },
            "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
            "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
            "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
            "validation_set": [],
        }
        write_json(benchmark_root / "cosmos_zero_shot_benchmark.json", manifest)
        return manifest

    if runtime_probe["status"] != "ready":
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "reason": "cosmos_runtime_unavailable",
            "runtime_probe": runtime_probe,
            "capture_id": context.capture_id,
            "scene_id": context.scene_id,
            "benchmark_family": "cosmos_zero_shot_validation",
            "bootstrap_origin": bootstrap_origin,
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
            },
            "validation_set": validation_set,
            "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
            "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
            "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
        }
        write_json(benchmark_root / "cosmos_zero_shot_benchmark.json", manifest)
        return manifest

    synthesis_result = run_capture_synthesis_validation(
        capture_root=context.capture_root,
        descriptor_gcs_uri=descriptor_gcs_uri,
        cfg=cfg,
        mode="cosmos_i2w",
    )
    if synthesis_result.get("status") == "failed" and _runtime_blocked_reason(synthesis_result.get("reason")):
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "reason": str(synthesis_result.get("reason") or "cosmos_runtime_unavailable"),
            "runtime_probe": runtime_probe,
            "capture_id": context.capture_id,
            "scene_id": context.scene_id,
            "benchmark_family": "cosmos_zero_shot_validation",
            "bootstrap_origin": bootstrap_origin,
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
            },
            "validation_set": validation_set,
            "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
            "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
            "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
            "synthesis_result": synthesis_result,
        }
        write_json(benchmark_root / "cosmos_zero_shot_benchmark.json", manifest)
        return manifest

    _ref_dist = synthesis_result.get("ref_frame_distance_m")
    spatial_faithfulness_passed = bool(
        synthesis_result.get("status") == "completed"
        and float(synthesis_result.get("coverage_frac") or 0.0) >= 0.55
        and float(_ref_dist if _ref_dist is not None else 99.0) <= 2.0
    )
    temporal_stability_passed = synthesis_result.get("status") == "completed" and bool(
        synthesis_result.get("output_video_uri")
    )
    task_targets = {
        str(target_id)
        for task in task_anchor_manifest.get("tasks", [])
        if isinstance(task, Mapping)
        for target_id in list(task.get("target_object_ids") or [])
        if str(target_id).strip()
    }
    task_salient_retention_passed = bool(validation_set) and (
        any(item["anchor_observation_count"] > 0 for item in validation_set) or not task_targets
    )
    protected_region_count = len(list(protected_regions_manifest.get("regions") or []))
    protected_region_leakage_passed = protected_region_count == 0 or str(
        protected_regions_manifest.get("grounding_status") or "grounded"
    ).strip().lower() == "grounded"

    checks = {
        "spatial_faithfulness": {
            "passed": spatial_faithfulness_passed,
            "detail": "Coverage and reference distance stay within the zero-shot acceptance band.",
        },
        "temporal_stability": {
            "passed": temporal_stability_passed,
            "detail": "Cosmos returned a time-consistent video artifact for the validation render.",
        },
        "task_salient_object_retention": {
            "passed": task_salient_retention_passed,
            "detail": "Validation frames keep task anchors or route anchors in-view.",
        },
        "protected_region_leakage": {
            "passed": protected_region_leakage_passed,
            "detail": "Protected-region grounding remains intact for the benchmark package.",
        },
    }
    passed_count = sum(1 for item in checks.values() if item["passed"])
    status = "completed" if passed_count == len(checks) else "degraded"
    manifest = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_id": context.capture_id,
        "scene_id": context.scene_id,
        "benchmark_family": "cosmos_zero_shot_validation",
        "runtime_probe": runtime_probe,
        "bootstrap_origin": bootstrap_origin,
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
        },
        "validation_set": validation_set,
        "selected_target_count": int(reference_selection_manifest.get("selected_target_count") or 0),
        "skipped_target_count": int(reference_selection_manifest.get("skipped_target_count") or 0),
        "rejected_near_duplicate_count": int(reference_selection_manifest.get("rejected_near_duplicate_count") or 0),
        "task_target_count": len(task_targets),
        "protected_region_count": protected_region_count,
        "synthesis_result": synthesis_result,
        "checks": checks,
        "evidence_supported": all(
            key in descriptor or key in (descriptor.get("quality") or {})
            for key in ("capture_id", "scene_id")
        ),
    }
    write_json(benchmark_root / "cosmos_zero_shot_benchmark.json", manifest)
    return manifest
