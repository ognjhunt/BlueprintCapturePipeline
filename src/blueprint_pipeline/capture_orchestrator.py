"""Capture pipeline entrypoint: a website capture's qualification (site preparation) lane."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional

from .core.common import PipelineError, parse_gs_uri, resolve_gs_uri_to_path
from .device_capture_pipeline_switch import closed_device_capture_record
from .core.logging_utils import log_event
from .local_capture import resolve_local_capture_context
from .core.pipeline_settings import PipelineConfig
from .site_package_orchestrator import run_qualification_pipeline
from .core.lane_resume import (
    lane_ledger_input_fingerprint,
    lane_resume_disabled,
    read_completed_lane_result,
    record_lane_completion,
)

logger = logging.getLogger(__name__)

QUALIFICATION_LANE = "qualification"


def _require_qualification_lane(lane: Optional[str], requested_lanes: Optional[List[str]]) -> None:
    """Only the qualification lane exists; any other requested lane is refused, not ignored."""
    values = [lane] if lane is not None else []
    if isinstance(requested_lanes, str):
        values.append(requested_lanes)
    elif requested_lanes:
        values.extend(str(value) for value in requested_lanes)
    for value in values:
        normalized = str(value or "").strip().lower()
        if normalized and normalized != QUALIFICATION_LANE:
            raise ValueError(f"Unsupported pipeline lane: {value}")


def run_capture_pipeline(
    *,
    descriptor_gcs_uri: str,
    lane: Optional[str] = None,
    requested_lanes: Optional[List[str]] = None,
    config: Optional[PipelineConfig] = None,
    input_binding_digest: Optional[str] = None,
) -> Dict[str, Any]:
    _require_qualification_lane(lane, requested_lanes)
    cfg = config or PipelineConfig()
    lanes = [QUALIFICATION_LANE]
    descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, cfg.gcs_root)
    # The lane ledger skips a qualification already completed for the same capture
    # input fingerprint. BLUEPRINT_LANE_RESUME_DISABLED=1 disables it. run_e2e's
    # stage binding (input_binding_digest) joins that fingerprint, so a stage it
    # reruns from retained outputs is never skipped here.
    resume_root = resolve_local_capture_context(descriptor_path).capture_root
    lane_ledger_fingerprint: Optional[Dict[str, Any]] = None
    if not lane_resume_disabled():
        try:
            lane_ledger_fingerprint = lane_ledger_input_fingerprint(
                capture_root=resume_root,
                descriptor_path=descriptor_path,
                input_binding_digest=input_binding_digest,
            )
        except OSError:
            logger.warning(
                "capture_pipeline.lane_resume_fingerprint_failed descriptor=%s",
                descriptor_gcs_uri,
                exc_info=True,
            )
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.started",
        descriptor_gcs_uri=descriptor_gcs_uri,
        descriptor_path=str(descriptor_path),
        requested_lane=lane,
        lanes=lanes,
    )

    lane_result: Optional[Mapping[str, Any]] = None
    if lane_ledger_fingerprint is not None:
        resumed_result = read_completed_lane_result(
            capture_root=resume_root,
            lane=QUALIFICATION_LANE,
            fingerprint=lane_ledger_fingerprint,
        )
        if resumed_result is not None:
            log_event(
                logger,
                logging.INFO,
                "capture_pipeline.lane_skipped_already_completed",
                descriptor_gcs_uri=descriptor_gcs_uri,
                selected_lane=QUALIFICATION_LANE,
                reason="lane_ledger_marker_matches_capture_input_fingerprint",
                marker_dir="pipeline/lane_ledger",
            )
            lane_result = {**resumed_result, "resumed_from_lane_ledger": True}
    if lane_result is None:
        log_event(
            logger,
            logging.INFO,
            "capture_pipeline.lane_started",
            descriptor_gcs_uri=descriptor_gcs_uri,
            selected_lane=QUALIFICATION_LANE,
        )
        lane_result = run_qualification_pipeline(
            descriptor_gcs_uri=descriptor_gcs_uri,
            config=cfg,
            requested_lanes=lanes,
        )
        if lane_ledger_fingerprint is not None:
            record_lane_completion(
                capture_root=resume_root,
                lane=QUALIFICATION_LANE,
                fingerprint=lane_ledger_fingerprint,
                lane_result=lane_result,
            )
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.lane_completed",
        descriptor_gcs_uri=descriptor_gcs_uri,
        selected_lane=QUALIFICATION_LANE,
        lane=lane_result.get("lane") or QUALIFICATION_LANE,
        lane_status=lane_result.get("status"),
        result_count=1,
        manifest_path=lane_result.get("manifest_path"),
        source=lane_result.get("source"),
    )

    parsed = parse_gs_uri(descriptor_gcs_uri)
    result = {
        "status": "completed",
        "artifact_purpose": "evaluation_preparation",
        "robot_evaluation_performed": False,
        "descriptor_gcs_uri": descriptor_gcs_uri,
        "bucket": parsed.bucket,
        "lanes": lanes,
        "results": [dict(lane_result)],
    }
    log_event(
        logger,
        logging.INFO,
        "capture_pipeline.completed",
        descriptor_gcs_uri=descriptor_gcs_uri,
        bucket=parsed.bucket,
        lanes=lanes,
        result_count=1,
        lane_failure_count=0,
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """The ``blueprint-pipeline`` Cloud Run job's entry: a permanent no-op.

    The job served the capture app, whose processing path was removed on 2026-11-11
    (``device_capture_pipeline_switch``). The image and Terraform still name this
    entry, so it accepts the job's arguments, writes the one closed record and exits
    0: the execution is not retried and nothing waits on an operator.
    """

    parser = argparse.ArgumentParser(description="Retired capture-app job entry; runs nothing")
    parser.add_argument("--descriptor-gcs-uri",
                        default=(os.getenv("PIPELINE_DESCRIPTOR_GCS_URI") or "").strip() or None)
    parser.add_argument("--bucket", default=(os.getenv("PIPELINE_BUCKET") or "").strip() or None)
    parser.add_argument("--scene-id", default=(os.getenv("PIPELINE_SCENE_ID") or "").strip() or None)
    parser.add_argument("--capture-id", default=(os.getenv("PIPELINE_CAPTURE_ID") or "").strip() or None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--allow-legacy-lanes", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    scene_id, capture_id = args.scene_id, args.capture_id
    if args.descriptor_gcs_uri:
        try:
            parts = parse_gs_uri(args.descriptor_gcs_uri).key.split("/")
        except (PipelineError, ValueError):
            parts = []
        if len(parts) > 3 and parts[0] == "scenes" and parts[2] == "captures":
            scene_id, capture_id = parts[1], parts[3]
    record = closed_device_capture_record(entry="cloud_run_job", scene_id=scene_id, capture_id=capture_id)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
