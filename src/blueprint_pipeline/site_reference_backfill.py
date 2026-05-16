"""Offline Site Reference Database v1 backfill utilities.

The backfill command only works from local/staged artifacts. It reports captures
that need a human-reviewed stable site identity instead of fabricating IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .common import read_json, utc_now_iso, write_json
from .retrieval_index_stage import run_retrieval_index_stage

BACKFILL_SCHEMA_VERSION = "site_reference_backfill.v1"
REVIEW_PACKET_SCHEMA_VERSION = "site_reference_backfill_review.v1"


def run_site_reference_backfill(
    *,
    storage_roots: Iterable[str | Path],
    report_path: str | Path | None = None,
    dry_run: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    roots = [Path(root).expanduser().resolve() for root in storage_roots]
    captures = [_evaluate_capture(capture_root) for capture_root in _discover_capture_roots(roots)]
    indexed = 0

    for entry in captures:
        if entry["status"] != "eligible" or dry_run:
            continue
        result = run_retrieval_index_stage(
            capture_root=Path(str(entry["capture_root"])),
            force_rebuild=force_rebuild,
        )
        entry["stage_result"] = result
        if result.get("status") == "completed":
            entry["status"] = "indexed"
            indexed += 1
        else:
            entry["status"] = "skipped"
            entry["blockers"] = [str(result.get("reason") or "retrieval_index_stage_not_completed")]

    summary = _summary(captures)
    if indexed:
        summary["indexed"] = indexed

    output_path = Path(report_path).expanduser().resolve() if report_path else None
    review_packet_path = _review_packet_path(output_path, roots)
    review_packet = _build_review_packet(captures)
    if review_packet["captures"]:
        write_json(review_packet_path, review_packet)

    report = {
        "schema_version": BACKFILL_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "dry_run": bool(dry_run),
        "storage_roots": [str(root) for root in roots],
        "summary": summary,
        "captures": captures,
        "review_packet_path": str(review_packet_path) if review_packet["captures"] else None,
    }
    if output_path:
        write_json(output_path, report)
    return report


def _discover_capture_roots(storage_roots: Iterable[Path]) -> List[Path]:
    roots: set[Path] = set()
    for storage_root in storage_roots:
        if not storage_root.exists():
            continue
        for descriptor in storage_root.rglob("capture_descriptor.json"):
            roots.add(descriptor.parent.resolve())
        for manifest in storage_root.rglob("raw/manifest.json"):
            roots.add(manifest.parent.parent.resolve())
    return sorted(roots, key=lambda path: str(path))


def _evaluate_capture(capture_root: Path) -> Dict[str, Any]:
    descriptor_path = capture_root / "capture_descriptor.json"
    raw_root = capture_root / "raw"
    manifest_path = raw_root / "manifest.json"
    descriptor = _read_optional_json(descriptor_path)
    manifest = _merge_raw_sidecars(_read_optional_json(manifest_path), raw_root)
    site_id = _resolve_site_id(descriptor=descriptor, manifest=manifest)
    scene_id = str(descriptor.get("scene_id") or manifest.get("scene_id") or _path_part(capture_root, "scenes") or "")
    capture_id = str(descriptor.get("capture_id") or manifest.get("capture_id") or capture_root.name)
    candidate = _world_model_candidate(descriptor=descriptor, manifest=manifest)

    blockers: list[str] = []
    status = "eligible"
    if not candidate:
        status = "skipped"
        blockers.append("world_model_candidate=false")
    if not site_id:
        status = "review_required"
        blockers = ["missing_site_id"]
    elif _requires_non_arkit_geometry(descriptor=descriptor, manifest=manifest, capture_root=capture_root):
        status = "geometry_required"
        blockers = ["non_arkit_geometry_missing"]

    geometry_summary_path = capture_root / "pipeline" / "geometry" / "geometry_summary.json"

    return {
        "capture_root": str(capture_root),
        "scene_id": scene_id,
        "capture_id": capture_id,
        "site_id": site_id or None,
        "status": status,
        "blockers": blockers,
        "expected_geometry_summary_path": str(geometry_summary_path) if status == "geometry_required" else None,
        "local_geometry_command": (
            f"python3 scripts/run_geometry_lane.py --capture-root {capture_root} "
            "--provider local_sfm --model local-sfm-offline"
            if status == "geometry_required"
            else None
        ),
        "provider_blocker": _provider_native_geometry_blocker(capture_root) if status == "geometry_required" else None,
        "evidence_paths": {
            "capture_descriptor": str(descriptor_path) if descriptor_path.is_file() else None,
            "raw_manifest": str(manifest_path) if manifest_path.is_file() else None,
            "site_identity_sidecar": str(raw_root / "site_identity.json") if (raw_root / "site_identity.json").is_file() else None,
            "raw_video": _raw_video_path(capture_root, descriptor=descriptor, manifest=manifest),
            "geometry_summary": str(geometry_summary_path) if geometry_summary_path.is_file() else None,
        },
    }


def _merge_raw_sidecars(manifest: Mapping[str, Any], raw_root: Path) -> Dict[str, Any]:
    merged = dict(manifest)
    for key, filename in {
        "site_identity": "site_identity.json",
        "capture_topology": "capture_topology.json",
        "capture_mode": "capture_mode.json",
    }.items():
        if isinstance(merged.get(key), Mapping):
            continue
        payload = _read_optional_json(raw_root / filename)
        if payload:
            merged[key] = payload
    return merged


def _resolve_site_id(*, descriptor: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    candidates: list[Any] = [descriptor.get("site_id"), manifest.get("site_id")]
    for payload in (descriptor, descriptor.get("metadata"), manifest):
        if isinstance(payload, Mapping):
            identity = payload.get("site_identity")
            if isinstance(identity, Mapping):
                candidates.append(identity.get("site_id"))
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _world_model_candidate(*, descriptor: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    quality = descriptor.get("quality") if isinstance(descriptor.get("quality"), Mapping) else {}
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    capture_mode = metadata.get("capture_mode") if isinstance(metadata.get("capture_mode"), Mapping) else {}
    manifest_capture_mode = manifest.get("capture_mode") if isinstance(manifest.get("capture_mode"), Mapping) else {}
    scene_memory = metadata.get("scene_memory_capture") if isinstance(metadata, Mapping) else None
    scene_memory_manifest = manifest.get("scene_memory_capture") if isinstance(manifest.get("scene_memory_capture"), Mapping) else {}
    values = [
        descriptor.get("world_model_candidate"),
        quality.get("world_model_candidate") if isinstance(quality, Mapping) else None,
        scene_memory.get("world_model_candidate") if isinstance(scene_memory, Mapping) else None,
        scene_memory_manifest.get("world_model_candidate") if isinstance(scene_memory_manifest, Mapping) else None,
    ]
    requested_output = str(
        descriptor.get("requested_output")
        or capture_mode.get("requested_output")
        or capture_mode.get("requestedOutput")
        or capture_mode.get("requested_mode")
        or capture_mode.get("requestedMode")
        or manifest.get("requested_output")
        or manifest_capture_mode.get("requested_output")
        or manifest_capture_mode.get("requestedOutput")
        or manifest_capture_mode.get("requested_mode")
        or manifest_capture_mode.get("requestedMode")
        or ""
    ).strip()
    return any(value is True for value in values) or requested_output == "site_world_candidate"


def _requires_non_arkit_geometry(
    *,
    descriptor: Mapping[str, Any],
    manifest: Mapping[str, Any],
    capture_root: Path,
) -> bool:
    if not _is_non_arkit_video_capture(descriptor=descriptor, manifest=manifest):
        return False
    if not _derived_generation_allowed(descriptor=descriptor, manifest=manifest):
        return False
    if not _raw_video_path(capture_root, descriptor=descriptor, manifest=manifest):
        return False
    geometry_summary = _read_optional_json(capture_root / "pipeline" / "geometry" / "geometry_summary.json")
    if not geometry_summary:
        return True
    if bool(geometry_summary.get("geometry_live_ready")) and str(geometry_summary.get("geometry_source") or "") == "video_to_world":
        return False
    if str(geometry_summary.get("geometry_source") or "") == "local_sfm" and bool(
        geometry_summary.get("contract_ready_for_world_model")
    ):
        return False
    return True


def _is_non_arkit_video_capture(*, descriptor: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    media_metadata = metadata.get("media_metadata") if isinstance(metadata.get("media_metadata"), Mapping) else {}
    values = [
        descriptor.get("capture_source"),
        descriptor.get("source_device"),
        descriptor.get("capture_modality"),
        media_metadata.get("source_device"),
        manifest.get("capture_source"),
        manifest.get("source_device"),
        manifest.get("capture_modality"),
    ]
    text = " ".join(str(value or "").strip().lower() for value in values)
    return any(token in text for token in ("meta_glasses", "glasses", "non_arkit_video", "video_only"))


def _derived_generation_allowed(*, descriptor: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    rights = metadata.get("rights_lineage") if isinstance(metadata.get("rights_lineage"), Mapping) else {}
    capture_rights = metadata.get("capture_rights") if isinstance(metadata.get("capture_rights"), Mapping) else {}
    manifest_rights = manifest.get("rights_lineage") if isinstance(manifest.get("rights_lineage"), Mapping) else {}
    manifest_capture_rights = manifest.get("capture_rights") if isinstance(manifest.get("capture_rights"), Mapping) else {}
    values = [
        descriptor.get("derived_generation_allowed"),
        metadata.get("derived_generation_allowed"),
        rights.get("derived_generation_allowed"),
        rights.get("derivedGenerationAllowed"),
        rights.get("derived_scene_generation_allowed"),
        rights.get("derivedSceneGenerationAllowed"),
        capture_rights.get("derived_generation_allowed"),
        capture_rights.get("derivedGenerationAllowed"),
        capture_rights.get("derived_scene_generation_allowed"),
        capture_rights.get("derivedSceneGenerationAllowed"),
        manifest.get("derived_generation_allowed"),
        manifest_rights.get("derived_generation_allowed"),
        manifest_rights.get("derivedGenerationAllowed"),
        manifest_rights.get("derived_scene_generation_allowed"),
        manifest_rights.get("derivedSceneGenerationAllowed"),
        manifest_capture_rights.get("derived_generation_allowed"),
        manifest_capture_rights.get("derivedGenerationAllowed"),
        manifest_capture_rights.get("derived_scene_generation_allowed"),
        manifest_capture_rights.get("derivedSceneGenerationAllowed"),
    ]
    return any(value is True for value in values)


def _raw_video_path(
    capture_root: Path,
    *,
    descriptor: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Optional[str]:
    candidates = [
        descriptor.get("raw_video_uri"),
        manifest.get("raw_video_uri"),
        manifest.get("original_video_uri"),
        capture_root / "raw" / "walkthrough.mov",
        capture_root / "raw" / "walkthrough.mp4",
    ]
    for candidate in candidates:
        if isinstance(candidate, Path):
            if candidate.is_file():
                return str(candidate)
            continue
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _provider_native_geometry_blocker(capture_root: Path) -> Dict[str, Any]:
    return {
        "id": "provider_native_geometry_missing",
        "reason": "video_to_world_runner_not_configured",
        "required_env": ["VIDEO_TO_WORLD_URL", "VIDEO_TO_WORLD_RUNNER_TOKEN"],
        "command": (
            f"VIDEO_TO_WORLD_URL=<provider-url> VIDEO_TO_WORLD_RUNNER_TOKEN=<token> "
            f"python3 scripts/run_geometry_lane.py --capture-root {capture_root} "
            "--provider video_to_world --model video_to_world-default"
        ),
    }


def _summary(captures: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {
        "discovered": 0,
        "eligible": 0,
        "indexed": 0,
        "skipped": 0,
        "review_required": 0,
        "geometry_required": 0,
    }
    for capture in captures:
        counts["discovered"] += 1
        status = str(capture.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _build_review_packet(captures: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for capture in captures:
        if str(capture.get("status") or "") != "review_required":
            continue
        rows.append(
            {
                "scene_id": capture.get("scene_id"),
                "capture_id": capture.get("capture_id"),
                "capture_root": capture.get("capture_root"),
                "blockers": list(capture.get("blockers") or []),
                "evidence_paths": dict(capture.get("evidence_paths") or {}),
                "requested_fields": ["site_identity.site_id", "site_identity.site_id_source"],
                "instruction": "Resolve stable site identity from upstream buyer/site submission/open-capture records; do not fabricate IDs.",
            }
        )
    return {
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "captures": rows,
    }


def _review_packet_path(report_path: Optional[Path], roots: List[Path]) -> Path:
    if report_path is not None:
        return report_path.with_name(report_path.stem + "_review_packet.json")
    base = roots[0] if roots else Path.cwd()
    return base / "site_reference_backfill_review_packet.json"


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _path_part(path: Path, marker: str) -> Optional[str]:
    parts = path.parts
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill local Site Reference Database v1 artifacts.")
    parser.add_argument("storage_roots", nargs="+", type=Path)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Run retrieval indexing for eligible captures.")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    report = run_site_reference_backfill(
        storage_roots=args.storage_roots,
        report_path=args.report_path,
        dry_run=not args.execute,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
