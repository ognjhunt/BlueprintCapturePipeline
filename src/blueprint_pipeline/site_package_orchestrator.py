"""Capture->site-package orchestration spine (formerly ``qualification.py``).

A website upload branches from fresh task context directly into task-aware media
and native scene preparation (``website_capture_preparation``). The capture app's
branch (object indexing, Gemini review, privacy post-processing, World Labs input
preparation, canonical site-package assembly, provider preview, WebApp sync) was
removed on 2026-11-11 with the app's processing path; a descriptor that is not a
website upload ends with ``device_capture_pipeline_removed``.

``blueprint_pipeline.qualification`` remains a deprecated import alias for this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import (
    MAXIMUM_HIDDEN_ZONE_BOUND,
    PipelineError,
    StageError,
    ensure_dir,
    ensure_local_uri_path,
    infer_storage_root_from_scene_path,
    parse_bool,
    parse_gs_uri,
    resolve_gs_uri_to_path,
    to_pipeline_prefix,
    utc_now_iso,
)
from .core.handoff_lease import guarded_handoff_write_json as write_json
from .consent_normalization import resolve_consent_signals
from .clean_plate_stage import run_clean_plate_stage
from .website_task_context import load_current_website_task_context, load_website_scene_sponsorship
from .website_capture_entry import is_website_entry_source


@dataclass
class QualificationGate:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _capture_rights(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("capture_rights") if isinstance(metadata.get("capture_rights"), Mapping) else {}
    consent_signals = resolve_consent_signals(raw)
    consent_revoked = consent_signals["consent_revoked"]
    return {
        "derived_scene_generation_allowed": parse_bool(
            raw.get("derived_scene_generation_allowed"),
            default=False,
        )
        and not consent_revoked,
        "data_licensing_allowed": parse_bool(
            raw.get("data_licensing_allowed"),
            default=False,
        )
        and not consent_revoked,
        "capture_contributor_payout_eligible": parse_bool(
            raw.get("capture_contributor_payout_eligible"),
            default=False,
        ),
        "consent_status": consent_signals["consent_status"]
        or str(raw.get("consent_status") or "unknown"),
        "consent_revoked": consent_revoked,
        "consent_revoked_at": consent_signals["consent_revoked_at"],
        "permission_document_uri": str(raw.get("permission_document_uri") or "").strip() or None,
        "consent_scope": _string_list(raw.get("consent_scope")),
        "commercialization_terms": _mapping(
            raw.get("commercialization_terms")
            or raw.get("commercializationTerms")
            or raw.get("commercial_terms")
            or raw.get("commercialTerms")
        ),
        "operator_revenue_terms": _mapping(
            raw.get("operator_revenue_terms")
            or raw.get("operatorRevenueTerms")
            or raw.get("revenue_share_terms")
            or raw.get("revenueShareTerms")
        ),
        "exclusivity_terms": _mapping(
            raw.get("exclusivity_terms") or raw.get("exclusivityTerms")
        ),
    }


def _worldlabs_derived_rights_allowed(*, metadata: Mapping[str, Any]) -> bool:
    """Whether the capture is rights-cleared for derived scene generation.

    PIPE-04: the WorldLabs preview input video is a derived, reviewer-facing
    transformation of the capture, so it must not be generated unless
    ``derived_scene_generation_allowed`` is set — mirroring scene-memory readiness
    (see the ``derived_scene_rights`` gate).
    """
    rights = _capture_rights(metadata if isinstance(metadata, Mapping) else {})
    return bool(rights["derived_scene_generation_allowed"])


# Privacy pipeline statuses that completed some privacy processing. Delivery
# gates remove fallback/local-proof statuses from this set because those are
# review inputs, not verified removal.
_PRIVACY_POSTPROCESS_CLEARED_STATUSES = frozenset(
    {
        "no_people_detected",
        "person_removed", "website_capture_admitted",
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    }
)


def _privacy_postprocess_gate(*, privacy_status: str, delivery_run: bool) -> QualificationGate:
    """Build the privacy_postprocess_gate.

    PIPE-03: for a *delivery* run (one that will build buyer/reviewer-facing
    downstream artifacts), ``not_run`` is NON-passing — privacy must actually have
    executed and cleared. For non-delivery / local flows, ``not_run`` remains
    acceptable so existing test/local pipelines keep working. ``failed_closed`` never
    passes.
    """
    status = str(privacy_status or "").strip().lower()
    passing_statuses = set(_PRIVACY_POSTPROCESS_CLEARED_STATUSES)
    fallback_statuses = {
        "face_anonymized_fallback",
        "full_frame_redacted_local_proof",
    }
    if delivery_run:
        passing_statuses.difference_update(fallback_statuses)
    if not delivery_run:
        passing_statuses.add("not_run")
    passed = status in passing_statuses
    detail = f"privacy_status={privacy_status}"
    if not passed and status == "not_run" and delivery_run:
        detail = (
            f"privacy_status={privacy_status}; delivery runs require privacy post-processing "
            "to run and clear (enable PRIVACY_PIPELINE_ENABLED / production launch mode)"
        )
    elif not passed and status in fallback_statuses and delivery_run:
        detail = (
            f"privacy_status={privacy_status}; delivery runs require verified privacy "
            "removal, not fallback/local redaction proof"
        )
    return QualificationGate("privacy_postprocess_gate", passed, detail)


def _resolve_optional_uri_to_path(uri: Optional[str], storage_root: Path) -> Optional[Path]:
    if not uri:
        return None
    try:
        return ensure_local_uri_path(uri, gcs_root=storage_root, scratch_dir=storage_root / ".downloads")
    except Exception:
        return None


_GENERIC_CAPABILITY_ENVELOPE = {
    "minimum_path_width_m": 0.95,
    "preferred_path_width_m": 1.15,
    "maximum_threshold_height_m": 0.04,
    "maximum_target_reach_distance_m": 1.1,
    "maximum_workcell_span_m": 2.5,
    "maximum_hidden_zone_bound": MAXIMUM_HIDDEN_ZONE_BOUND,
    "maximum_uncertainty_score": 0.3,
}


def _write_failure(
    *,
    pipeline_dir: Path,
    descriptor_uri: str,
    stage: str,
    error: Exception,
    gates: List[QualificationGate],
) -> None:
    payload = {
        "schema_version": "v1",
        "lane": "qualification",
        "status": "failed",
        "stage": stage,
        "descriptor_uri": descriptor_uri,
        "error": str(error),
        "failed_at": utc_now_iso(),
        "gates": [gate.to_dict() for gate in gates],
    }
    write_json(pipeline_dir / ".qualification_pipeline_failed.json", payload)
    write_json(pipeline_dir / ".swap_pipeline_failed.json", payload)


def run_qualification_pipeline(
    *,
    descriptor_gcs_uri: str,
    config: Any,
    requested_lanes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    stage = "intake"
    gates: List[QualificationGate] = []
    scene_id = "unknown_scene"
    capture_id = "unknown_capture"
    pipeline_prefix = "_pipeline_failures"
    pipeline_dir = config.gcs_root / pipeline_prefix
    ensure_dir(pipeline_dir)

    try:
        parsed_uri = parse_gs_uri(descriptor_gcs_uri)
        bucket = parsed_uri.bucket
        descriptor_path = resolve_gs_uri_to_path(descriptor_gcs_uri, config.gcs_root)
        storage_root = infer_storage_root_from_scene_path(descriptor_path)
        from .local_capture import resolve_local_capture_context
        capture_root = resolve_local_capture_context(descriptor_path).capture_root
        descriptor = CaptureDescriptor.from_file(descriptor_path)
        scene_id = descriptor.scene_id
        capture_id = descriptor.capture_id
        pipeline_prefix = to_pipeline_prefix(scene_id, capture_id)
        pipeline_dir = storage_root / pipeline_prefix
        ensure_dir(pipeline_dir)
        if is_website_entry_source(descriptor.metadata.get("capture_entry_source")):
            stage = "website_task_context"
            from .website_atlas_preview import enqueue_preview
            from .website_source_video import preparation_source_video
            from .website_task_evidence import prepare_website_task_descriptor

            # The one file every website preparation stage reads: a new intake's clip normalized
            # once (long edge <= 1920 px, <= 30 fps, original kept), or the input a job started with.
            def website_source_video() -> Path | None:
                return preparation_source_video(
                    capture_root=capture_root,
                    raw_video=_resolve_optional_uri_to_path(descriptor.raw_video_uri, storage_root))
            # A routed upload's Atlas preview is a side lane: recording it never raises or holds this run.
            enqueue_preview(descriptor=descriptor, capture_root=capture_root, pipeline_dir=pipeline_dir,
                            source_video=website_source_video)
            descriptor = prepare_website_task_descriptor(
                descriptor=descriptor, capture_root=capture_root, pipeline_dir=pipeline_dir, bucket=bucket,
                load_task_context=load_current_website_task_context, load_sponsorship=load_website_scene_sponsorship)
            stage = "website_scene_preparation"
            from .website_capture_preparation import run_website_capture_preparation
            return run_website_capture_preparation(
                descriptor=descriptor, capture_root=capture_root, pipeline_dir=pipeline_dir,
                source_video=website_source_video(),
                derived_scene_generation_allowed=_worldlabs_derived_rights_allowed(metadata=descriptor.metadata),
                gates=gates, prepare_media=run_clean_plate_stage,
                check_privacy=_privacy_postprocess_gate, make_gate=QualificationGate,
            )
        # The capture app's branch was removed on 2026-11-11; the host listener ends such a handoff first.
        raise StageError("intake", "device_capture_pipeline_removed")

    except Exception as exc:
        if isinstance(exc, StageError):
            stage = exc.stage
        ensure_dir(pipeline_dir)
        try:
            _write_failure(
                pipeline_dir=pipeline_dir,
                descriptor_uri=descriptor_gcs_uri,
                stage=stage,
                error=exc,
                gates=gates,
            )
            failure_quality_report = {
                "schema_version": "v1",
                "lane": "qualification",
                "scene_id": scene_id,
                "capture_id": capture_id,
                "status": "failed",
                "generated_at": utc_now_iso(),
                "failed_stage": stage,
                "error": str(exc),
                "gates": [gate.to_dict() for gate in gates],
            }
            write_json(pipeline_dir / "qualification_quality_report.json", failure_quality_report)
            write_json(pipeline_dir / "swap_quality_report.json", failure_quality_report)
        except Exception:
            pass
        raise PipelineError(str(exc)) from exc
