"""Lane-aware capture pipeline entrypoint."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

from .capture_bridge import CaptureDescriptor
from .common import PipelineError, parse_gs_uri, resolve_gs_uri_to_path
from .qualification import run_qualification_pipeline
from .swap_orchestrator import OrchestratorConfig, run_swap_pipeline

_SUPPORTED_LANES = {"qualification", "scene_memory", "advanced_geometry", "all"}


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
            for expanded in ("qualification", "scene_memory", "advanced_geometry"):
                if expanded not in normalized:
                    normalized.append(expanded)
            continue
        if lane == "advanced_geometry" and "scene_memory" not in normalized:
            normalized.append("scene_memory")
        if lane not in normalized:
            normalized.append(lane)
    ordered: List[str] = []
    for lane in ("qualification", "scene_memory", "advanced_geometry"):
        if lane in normalized and lane not in ordered:
            ordered.append(lane)
    return ordered


def _load_descriptor_requested_lanes(descriptor_gcs_uri: str, gcs_root: Any) -> List[str]:
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, gcs_root)
    descriptor = CaptureDescriptor.from_file(descriptor_path)
    return list(descriptor.requested_lanes)


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

    descriptor_requested = _normalize_requested_lanes(
        _load_descriptor_requested_lanes(descriptor_gcs_uri, gcs_root)
    )
    return descriptor_requested or ["qualification"]


def run_capture_pipeline(
    *,
    descriptor_gcs_uri: str,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
    config: Optional[OrchestratorConfig] = None,
    nurec_client: Any = None,
    blueprint_runner: Any = None,
) -> Dict[str, Any]:
    cfg = config or OrchestratorConfig()
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
        if selected_lane == "advanced_geometry":
            results.append(
                run_swap_pipeline(
                    descriptor_gcs_uri=descriptor_gcs_uri,
                    config=cfg,
                    nurec_client=nurec_client,
                    blueprint_runner=blueprint_runner,
                )
            )
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
        help="qualification, scene_memory, advanced_geometry, or all",
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
