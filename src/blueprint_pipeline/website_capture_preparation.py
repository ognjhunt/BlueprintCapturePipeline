"""Direct website preparation; native source/privacy/execution admission stays mandatory."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .capture_bridge import CaptureDescriptor
from .clean_plate_stage import CleanPlatePolicy
from .common import StageError
from .core.handoff_lease import guarded_handoff_write_json as write_json
from .core.handoff_lease import handoff_lease_checkpoint


def run_website_capture_preparation(
    *,
    descriptor: CaptureDescriptor,
    capture_root: Path,
    pipeline_dir: Path,
    source_video: Path | None,
    derived_scene_generation_allowed: bool,
    gates: list[Any],
    prepare_media: Callable[..., Mapping[str, Any]],
    check_privacy: Callable[..., Any],
    make_gate: Callable[[str, bool, str], Any],
) -> dict[str, Any]:
    """Prepare the website's scene without legacy neutral-readiness reports.

    Fresh task context and sponsorship have already been loaded. The existing
    clean-plate and native handoff own source, privacy, geometry and execution
    admission; generic object-index/rubric and optional narratives add no
    authority to that path.
    """
    from .website_scene_handoff import prepare_website_scene_handoff
    from .website_task_context_contracts import (
        holds_native_task_preparation,
        retains_task_family_preparation,
        sponsors_scene_preparation,
    )

    from .website_target_background_correction import retain_rejected_selection
    handoff_lease_checkpoint("website_preparation_before_target_rejection")
    rejection = (retain_rejected_selection(capture_root=capture_root,
        task_context=descriptor.metadata.get("site_task_context") or {}, source_video=source_video)
        if derived_scene_generation_allowed else None)
    if rejection is not None:
        write_json(pipeline_dir / "provider_run_manifest.json", {
            "schema_version": "v1", "provider_name": None, "status": "not_requested",
            "website_scene_preparation": rejection})
        return {"status": "completed", "lane": "qualification", "website_scene_preparation": rejection,
                "native_execution_complete": False}

    privacy_processing = {
        "status": "pending_website_review",
        "mode": "task_aware_prepared_images",
        "raw_retained": True,
        "fail_closed": True,
    }
    preview_requested = sponsors_scene_preparation(descriptor.metadata) or any(
        str(value or "").strip().lower() in {"preview_simulation", "preview"}
        for value in descriptor.requested_outputs
    )
    handoff_lease_checkpoint("website_preparation_before_media")
    clean_plate = prepare_media(
        capture_root=capture_root,
        privacy_processing=privacy_processing,
        worldlabs_input={"status": "not_requested", "manifest_uri": None, "output_video_uri": None},
        task_context=descriptor.metadata.get("site_task_context"),
        website_source_video=source_video,
        policy=CleanPlatePolicy(
            enabled=derived_scene_generation_allowed and preview_requested
        ),
    )
    handoff_lease_checkpoint("website_preparation_after_media")
    gates.append(make_gate(
        "clean_plate_stage", clean_plate.get("status") in {"noop", "objects_removed"},
        f"status={clean_plate.get('status')} mode={clean_plate.get('mode')}",
    ))
    if clean_plate.get("status") not in {"noop", "objects_removed"}:
        raise StageError("clean_plate", ",".join(clean_plate.get("blockers") or ["website_preparation_pending"]))
    privacy_processing.update(
        status=clean_plate.get("privacy_status"),
        fail_closed=not bool(clean_plate.get("privacy_verified")),
    )
    privacy_gate = check_privacy(
        privacy_status=str(privacy_processing.get("status") or ""), delivery_run=True,
    )
    gates.append(privacy_gate)
    if not privacy_gate.passed or clean_plate.get("privacy_verified") is not True:
        raise StageError("privacy_postprocess", "website_scene_preparation_pending")

    descriptor_payload = descriptor.to_dict()
    descriptor_payload.update(
        privacy_processed_video_uri=None,
        world_model_video_uri=None,
        privacy_status=privacy_processing["status"],
        privacy_mode=privacy_processing["mode"],
        privacy_manifest_uri=None,
    )
    metadata = dict(descriptor.metadata)
    metadata["privacy_processing"] = dict(privacy_processing)
    metadata["clean_plate"] = {key: clean_plate.get(key) for key in (
        "status", "mode", "adp_item", "claim_ceiling", "movable_removal_count",
        "person_target_count", "privacy_verified", "clean_plate_video_uri",
        "stage_manifest_uri", "source_geometry",
    )}
    descriptor_payload["metadata"] = metadata
    # Keep producer members immutable; this is the pipeline-owned descriptor.
    write_json(capture_root / "capture_descriptor.json", descriptor_payload)
    handoff_lease_checkpoint("website_preparation_before_native_handoff")
    preparation = prepare_website_scene_handoff(
        descriptor=descriptor_payload, clean_plate=clean_plate,
        capture_root=capture_root, now=time.time(),
        allow_operation_coverage_review=True, allow_static_target_recovery=True,
    )
    handoff_lease_checkpoint("website_preparation_after_native_handoff")
    write_json(pipeline_dir / "provider_run_manifest.json", {
        "schema_version": "v1", "provider_name": None, "status": "not_requested",
        "website_scene_preparation": preparation,
    })
    outbox = preparation.get("website_intake_outbox") or {}
    context = descriptor.metadata.get("site_task_context") or {}
    if (outbox.get("state") != "forward_pending"
            and not holds_native_task_preparation(preparation, context)
            and not retains_task_family_preparation(preparation, context)):
        raise StageError("website_scene_preparation", ",".join(
            preparation.get("blockers")
            or (preparation.get("runtime_inputs") or {}).get("blockers")
            or ["website_native_intake_pending"]
        ))
    return {
        "status": "completed", "lane": "qualification",
        "website_scene_preparation": preparation, "native_execution_complete": False,
    }
