"""Operator-facing raw capture preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import optional_read_json, utc_now_iso
from .local_capture import LocalCaptureContext, resolve_local_capture_context
from .materialization import preview_capture_bundle


def _string_value(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _uses_open_capture_intake_fallback(
    *,
    context: LocalCaptureContext,
    manifest: Mapping[str, Any],
    capture_context: Mapping[str, Any],
) -> bool:
    special_task_type = _string_value(
        capture_context.get("special_task_type"),
        capture_context.get("specialTaskType"),
        manifest.get("special_task_type"),
        manifest.get("specialTaskType"),
    ).lower()
    if special_task_type != "open_capture":
        return False

    task_hypothesis = optional_read_json(context.raw_root / "task_hypothesis.json") or {}
    accepted_statuses = {"accepted", "accepted_with_warnings"}
    hypothesis_status = _string_value(
        task_hypothesis.get("status"),
        capture_context.get("task_hypothesis_status"),
        capture_context.get("taskHypothesisStatus"),
    ).lower()
    return hypothesis_status in accepted_statuses


def _required_raw_entries(
    context: LocalCaptureContext,
    *,
    manifest: Mapping[str, Any],
    capture_context: Mapping[str, Any],
) -> Dict[str, Path]:
    required = {
        "manifest": context.raw_root / "manifest.json",
        "capture_context": context.raw_root / "capture_context.json",
        "capture_upload_complete": context.raw_complete_path,
    }
    if not _uses_open_capture_intake_fallback(
        context=context,
        manifest=manifest,
        capture_context=capture_context,
    ):
        required["intake_packet"] = context.raw_root / "intake_packet.json"
    return required


def _video_candidates(raw_root: Path, manifest: Mapping[str, Any]) -> List[str]:
    names = [
        "walkthrough.mov",
        "walkthrough.mp4",
        "recording.mov",
        "recording.mp4",
    ]
    out = [name for name in names if (raw_root / name).is_file()]
    video_uri = str(manifest.get("video_uri") or "").strip()
    if not out and video_uri:
        out.append(video_uri)
    return out


def build_capture_preflight_report(capture_root: str | Path) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    manifest = optional_read_json(context.raw_root / "manifest.json") or {}
    intake = optional_read_json(context.raw_root / "intake_packet.json") or {}
    capture_context = optional_read_json(context.raw_root / "capture_context.json") or {}
    required_entries = _required_raw_entries(
        context,
        manifest=manifest,
        capture_context=capture_context,
    )
    preview = preview_capture_bundle(
        bucket=context.bucket,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        gcs_root=context.storage_root,
        raw_prefix_uri=context.raw_prefix_uri,
    )
    video_candidates = _video_candidates(context.raw_root, manifest)

    missing_required = [name for name, path in required_entries.items() if not path.is_file()]
    if not video_candidates:
        missing_required.append("video")

    intake_missing = []
    if not str(intake.get("workflowName") or "").strip():
        intake_missing.append("workflowName")
    if not intake.get("taskSteps"):
        intake_missing.append("taskSteps")
    if not (
        str(intake.get("zone") or "").strip()
        or str(intake.get("owner") or "").strip()
    ):
        intake_missing.append("zone_or_owner")

    descriptor = preview["descriptor"]
    qa_report = preview["qa_report"]
    evidence_tier = str(descriptor.get("evidence_tier") or "pre_screen_video")
    status = "ready_for_materialization" if not missing_required else "blocked"
    if status == "ready_for_materialization" and str(qa_report.get("status")) != "passed":
        status = "pre_screen_only"

    notes = []
    if evidence_tier == "pre_screen_video":
        notes.append("This capture can produce a qualification report, but it will not return ready.")
    if intake_missing:
        notes.append("Intake is incomplete and still requires human completion before decision-grade use.")
    intake_packet_waived = not (context.raw_root / "intake_packet.json").is_file() and _uses_open_capture_intake_fallback(
        context=context,
        manifest=manifest,
        capture_context=capture_context,
    )
    if intake_packet_waived:
        notes.append(
            "Structured intake packet is missing, but open-capture metadata with an accepted task hypothesis is sufficient to run qualification."
        )

    return {
        "schema_version": "v1",
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_modality": descriptor.get("capture_modality"),
        "evidence_tier": evidence_tier,
        "mode_decision": evidence_tier,
        "required_inputs": {
            name: {"path": str(path), "present": path.is_file()}
            for name, path in required_entries.items()
        },
        "intake_packet_waived": intake_packet_waived,
        "video_candidates": video_candidates,
        "intake_missing_fields": intake_missing,
        "qa_preview": qa_report,
        "descriptor_preview": descriptor,
        "notes": notes,
        "missing_required_inputs": missing_required,
        "can_materialize": not missing_required,
        "pre_screen_only": evidence_tier == "pre_screen_video",
        "human_review_required": bool(qa_report.get("escalation_recommendation", {}).get("human_review_required", True)),
        "paths": {
            "capture_root": str(context.capture_root),
            "descriptor_path": str(context.descriptor_path),
            "pipeline_root": str(context.pipeline_root),
        },
        "source_files": {
            "manifest": manifest,
            "intake_packet": intake,
            "capture_context": capture_context,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a local capture folder before running the pipeline")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    try:
        report = build_capture_preflight_report(args.capture_root)
    except Exception as exc:
        print(f"[capture-preflight] FAILED: {exc}")
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "[capture-preflight] "
        f"status={report['status']} "
        f"mode={report['mode_decision']} "
        f"human_review_required={report['human_review_required']}"
    )
    if report["missing_required_inputs"]:
        print(
            "[capture-preflight] missing_required_inputs="
            + ",".join(str(item) for item in report["missing_required_inputs"])
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
