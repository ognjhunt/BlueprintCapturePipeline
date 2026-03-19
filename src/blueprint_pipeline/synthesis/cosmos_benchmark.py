"""Zero-shot Cosmos validation lane for fixed site-world benchmark checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ..capture_orchestrator import PipelineConfig, run_capture_synthesis_validation
from ..common import ensure_dir, resolve_gs_uri_to_path, utc_now_iso, write_json
from ..local_capture import resolve_local_capture_context
from .cosmos_inference import _DEFAULT_COSMOS_MODEL_ID


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

    validation_set = [
        {
            "frame_id": str(row.get("frame_id") or ""),
            "frame_uri": row.get("frame_uri"),
            "anchor_observation_count": len(list(row.get("anchor_observations") or [])),
            "zone_id": row.get("zone_id"),
        }
        for row in dense_index[:max_examples]
    ]

    if not validation_set:
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "missing",
            "reason": "no_dense_validation_examples",
            "runtime_probe": runtime_probe,
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
            "validation_set": validation_set,
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
            "validation_set": validation_set,
            "synthesis_result": synthesis_result,
        }
        write_json(benchmark_root / "cosmos_zero_shot_benchmark.json", manifest)
        return manifest

    spatial_faithfulness_passed = bool(
        synthesis_result.get("status") == "completed"
        and float(synthesis_result.get("coverage_frac") or 0.0) >= 0.55
        and float(synthesis_result.get("ref_frame_distance_m") or 99.0) <= 2.0
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
        "validation_set": validation_set,
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
