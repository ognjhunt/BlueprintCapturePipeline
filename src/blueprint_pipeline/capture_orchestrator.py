"""Lane-aware capture pipeline entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import PipelineError, parse_gs_uri, resolve_gs_uri_to_path
from .evaluation_prep_stage import run_evaluation_prep_stage
from .qualification import run_qualification_pipeline

_SUPPORTED_LANES = {"qualification", "scene_memory", "evaluation_prep", "all"}


@dataclass(frozen=True)
class PipelineConfig:
    gcs_root: Path = Path(os.getenv("GCS_ROOT", "/mnt/gcs"))


def _normalize_lane_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value not in _SUPPORTED_LANES:
        raise ValueError(f"Unsupported pipeline lane: {raw}")
    return value


def _normalize_requested_lanes(values: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        lane = _normalize_lane_value(value)
        if lane is None:
            continue
        if lane == "all":
            for expanded in ("qualification", "scene_memory", "evaluation_prep"):
                if expanded not in normalized:
                    normalized.append(expanded)
            continue
        if lane == "evaluation_prep" and "qualification" not in normalized:
            normalized.append("qualification")
        if lane not in normalized:
            normalized.append(lane)
    ordered: List[str] = []
    for lane in ("qualification", "scene_memory", "evaluation_prep"):
        if lane in normalized and lane not in ordered:
            ordered.append(lane)
    return ordered


def _load_descriptor_requested_lanes(descriptor_gcs_uri: str, gcs_root: Any) -> List[str]:
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, gcs_root)
    try:
        raw_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw_payload = {}
    raw_requested_outputs = raw_payload.get("requested_outputs")
    if isinstance(raw_requested_outputs, str):
        requested_outputs = [raw_requested_outputs]
    elif isinstance(raw_requested_outputs, (list, tuple, set)):
        requested_outputs = [str(value) for value in raw_requested_outputs]
    else:
        requested_outputs = []
    normalized_outputs = {str(value).strip().lower() for value in requested_outputs if str(value).strip()}
    if "deeper_evaluation" in normalized_outputs:
        return ["qualification", "scene_memory", "evaluation_prep"]
    if normalized_outputs & {"managed_tuning", "data_licensing"}:
        return ["qualification", "scene_memory"]
    return ["qualification"]


def resolve_requested_lanes(
    *,
    descriptor_gcs_uri: str,
    gcs_root: Any,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
) -> List[str]:
    explicit_lane = _normalize_lane_value(lane)
    if explicit_lane:
        return _normalize_requested_lanes([explicit_lane])

    env_lane = _normalize_lane_value(os.getenv("PIPELINE_LANE"))
    if env_lane:
        return _normalize_requested_lanes([env_lane])

    normalized_requested = _normalize_requested_lanes(requested_lanes)
    if normalized_requested:
        return normalized_requested

    descriptor_requested = _normalize_requested_lanes(_load_descriptor_requested_lanes(descriptor_gcs_uri, gcs_root))
    return descriptor_requested or ["qualification"]


def run_capture_pipeline(
    *,
    descriptor_gcs_uri: str,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    lanes = resolve_requested_lanes(
        descriptor_gcs_uri=descriptor_gcs_uri,
        gcs_root=cfg.gcs_root,
        lane=lane,
        requested_lanes=requested_lanes,
    )

    results: List[Dict[str, Any]] = []
    qualification_result: Optional[Dict[str, Any]] = None
    for selected_lane in lanes:
        if selected_lane in {"qualification", "scene_memory"}:
            if qualification_result is None:
                qualification_result = run_qualification_pipeline(
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    config=cfg,
                    requested_lanes=lanes,
                )
            if selected_lane == "qualification":
                results.append(qualification_result)
            else:
                results.append(
                    {
                        "status": "completed",
                        "lane": "scene_memory",
                        "scene_id": qualification_result.get("scene_id"),
                        "capture_id": qualification_result.get("capture_id"),
                        "pipeline_prefix": qualification_result.get("pipeline_prefix"),
                        "source": "qualification_artifacts",
                    }
                )
            continue
        if selected_lane == "evaluation_prep":
            if qualification_result is None:
                qualification_result = run_qualification_pipeline(
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    config=cfg,
                    requested_lanes=lanes,
                )
            evaluation_prep_result = run_evaluation_prep_stage(
                capture_root=resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root).parent,
                provider_name="manual",
            )
            lane_result = {
                "status": "completed",
                "lane": "evaluation_prep",
                "scene_id": qualification_result.get("scene_id"),
                "capture_id": qualification_result.get("capture_id"),
                "pipeline_prefix": qualification_result.get("pipeline_prefix"),
                "source": "evaluation_prep_artifacts",
                "manifest_path": evaluation_prep_result.get("manifest_path"),
            }
            results.append(lane_result)
            continue
        raise ValueError(f"Unsupported pipeline lane: {selected_lane}")

    parsed = parse_gs_uri(descriptor_gcs_uri)
    return {
        "status": "completed",
        "descriptor_gcs_uri": descriptor_gcs_uri,
        "bucket": parsed.bucket,
        "lanes": lanes,
        "results": results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run lane-aware capture pipeline")
    parser.add_argument(
        "--descriptor-gcs-uri",
        required=True,
        help="gs:// URI for capture_descriptor.json",
    )
    parser.add_argument(
        "--lane",
        default=None,
        help="qualification, scene_memory, evaluation_prep, or all",
    )
    args = parser.parse_args(argv)

    try:
        run_capture_pipeline(
            descriptor_gcs_uri=args.descriptor_gcs_uri,
            lane=args.lane,
        )
    except (PipelineError, ValueError) as exc:
        print(f"[capture-orchestrator] FAILED: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - safety net
        print(f"[capture-orchestrator] FAILED (unexpected): {exc}")
        return 1

    print("[capture-orchestrator] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
