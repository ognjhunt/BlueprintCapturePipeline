"""Import Scaniverse-derived support assets without upgrading capture truth.

The importer stages local Scaniverse exports into a capture root, records hashes
and source metadata, and optionally forwards preflight-supported formats into
the existing CPU scene-asset preflight lane. It does not call Niantic APIs,
download remote assets, run simulators, or mark derived geometry as raw capture
truth.
"""

from __future__ import annotations

import argparse
import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .scene_asset_preflight import SUPPORTED_SCENE_ASSET_SUFFIXES, build_scene_asset_preflight, inspect_scene_asset


SCANIVERSE_IMPORT_SCHEMA_VERSION = "scaniverse_asset_import.v1"
SCANIVERSE_PROOF_BOUNDARY_SCHEMA_VERSION = "scaniverse_asset_import_proof_boundary.v1"

SCANIVERSE_SUPPORTED_EXPORT_SUFFIXES = {
    ".usdz",
    ".ply",
    ".spz",
    ".glb",
    ".gltf",
    ".fbx",
    ".obj",
    ".usd",
    ".usda",
    ".usdc",
}

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "external_scaniverse_derived_support_asset_import",
    "external_provider": "scaniverse",
    "provider_lock_in_required": False,
    "raw_capture_truth": False,
    "raw_bundle_authority_preserved": True,
    "blueprint_capture_stack_replaced": False,
    "repo_local_only": True,
    "niantic_api_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "programmatic_360_upload_api_proven": False,
    "programmatic_asset_generation_api_proven": False,
    "programmatic_usdz_export_download_api_proven": False,
    "simulators_run": False,
    "isaac_sim_execution_proven": False,
    "physics_contact_validated": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "deployment_readiness_proven": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "raw_capture_truth",
        "simulator_execution_completed",
        "isaac_sim_loaded",
        "physics_contact_validated",
        "robot_policy_execution_passed",
        "deployment_ready",
    ],
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _string(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_copy_name(path: Path, seen: set[str]) -> str:
    base = path.name
    if base not in seen:
        seen.add(base)
        return base
    stem = path.stem or "asset"
    suffix = path.suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        index += 1


def _asset_role(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".usdz":
        return "isaac_usd_package"
    if suffix in {".usd", ".usda", ".usdc"}:
        return "openusd_scene_asset"
    if suffix in {".ply", ".spz"}:
        return "gaussian_splat_visual_asset"
    if suffix == ".glb" and "collider" in name:
        return "portable_collider_mesh_candidate"
    if suffix in {".glb", ".gltf", ".fbx", ".obj"}:
        return "mesh_export_asset"
    return "external_export_asset"


def _read_source_manifest(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scaniverse source manifest not found: {manifest_path}")
    payload = read_json_any(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Scaniverse source manifest must be a JSON object: {manifest_path}")
    return dict(payload)


def _missing_recommended_sidecar_fields(payload: Mapping[str, Any]) -> List[str]:
    recommended = (
        "blueprint_assignment_id",
        "blueprint_scene_id",
        "blueprint_capture_id",
        "scaniverse_site_id",
        "scaniverse_scan_id",
        "capture_hardware",
        "source_video_filename",
        "metric_scale_calibrated",
        "export_created_at",
        "export_performed_by",
        "rights_scope",
    )
    return [field for field in recommended if field not in payload]


def _copy_asset(source: Path, destination_dir: Path, seen_names: set[str]) -> Path:
    ensure_dir(destination_dir)
    destination = destination_dir / _safe_copy_name(source, seen_names)
    if source.resolve() == destination.resolve():
        return destination
    shutil.copy2(source, destination)
    return destination


def build_scaniverse_asset_import(
    *,
    capture_root: str | Path,
    assets: Sequence[str | Path],
    source_manifest: str | Path | None = None,
    run_scene_asset_preflight: bool = True,
) -> Dict[str, Any]:
    """Stage local Scaniverse exports as proof-bounded support assets."""

    context = resolve_local_capture_context(capture_root)
    generated_at = utc_now_iso()
    scaniverse_dir = context.pipeline_root / "scaniverse_assets"
    staged_dir = scaniverse_dir / "assets"
    ensure_dir(staged_dir)

    source_payload = _read_source_manifest(source_manifest)
    blockers: List[str] = []
    warnings: List[str] = []
    staged_assets: List[Dict[str, Any]] = []
    preflight_asset_paths: List[Path] = []
    seen_names: set[str] = set()

    if not source_payload:
        blockers.append("missing_blueprint_scaniverse_sidecar_manifest")
    else:
        missing_sidecar_fields = _missing_recommended_sidecar_fields(source_payload)
        if missing_sidecar_fields:
            warnings.append(
                "blueprint_scaniverse_sidecar_recommended_fields_missing:"
                + ",".join(missing_sidecar_fields)
            )

    if not assets:
        blockers.append("missing_scaniverse_asset")

    for raw_asset in assets:
        source_path = Path(raw_asset).expanduser().resolve()
        suffix = source_path.suffix.lower()
        if not source_path.is_file():
            blockers.append(f"missing_scaniverse_asset:{source_path}")
            continue
        if suffix not in SCANIVERSE_SUPPORTED_EXPORT_SUFFIXES:
            blockers.append(f"unsupported_scaniverse_export_suffix:{suffix or 'none'}")
            continue

        staged_path = _copy_asset(source_path, staged_dir, seen_names)
        inspection: Dict[str, Any]
        preflight_supported = suffix in SUPPORTED_SCENE_ASSET_SUFFIXES
        if preflight_supported:
            inspection = inspect_scene_asset(staged_path)
            preflight_asset_paths.append(staged_path)
        else:
            inspection = {
                "asset_type": suffix.lstrip("."),
                "path": str(staged_path),
                "status": "accepted_without_cpu_scene_preflight",
                "confidence": "low",
                "limitations": [
                    "This Scaniverse export format is accepted for provenance/checksum packaging only.",
                    "It is not parsed by the CPU scene-asset preflight lane.",
                ],
            }
            warnings.append(f"scaniverse_export_not_forwarded_to_scene_preflight:{staged_path.name}")

        staged_assets.append(
            {
                "source_path": str(source_path),
                "staged_path": str(staged_path),
                "relative_path": os.path.relpath(staged_path, start=context.capture_root).replace("\\", "/"),
                "filename": staged_path.name,
                "suffix": suffix,
                "asset_role": _asset_role(staged_path),
                "size_bytes": staged_path.stat().st_size,
                "sha256": _sha_file(staged_path),
                "external_derived_support_asset": True,
                "raw_capture_truth": False,
                "cpu_scene_preflight_supported": preflight_supported,
                "inspection": inspection,
            }
        )

    isaac_candidate_assets = [
        asset
        for asset in staged_assets
        if asset["suffix"] in {".usdz", ".usd", ".usda", ".usdc", ".glb", ".gltf", ".obj"}
    ]
    preflight_result: Dict[str, Any] | None = None
    if run_scene_asset_preflight and preflight_asset_paths:
        preflight_result = build_scene_asset_preflight(
            capture_root=context.capture_root,
            scene_assets=preflight_asset_paths,
        )
    elif run_scene_asset_preflight:
        warnings.append("no_scaniverse_assets_supported_by_scene_preflight")

    if not staged_assets and "missing_scaniverse_asset" not in blockers:
        blockers.append("no_scaniverse_assets_imported")

    manifest = {
        "schema_version": SCANIVERSE_IMPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "blocked" if blockers else "ready_for_review",
        "asset_count": len(staged_assets),
        "assets": staged_assets,
        "blueprint_sidecar_manifest": source_payload,
        "source_manifest": source_payload,
        "source_policy": {
            "source": "scaniverse_manual_or_enterprise_export",
            "manual_scaniverse_web_workflow_assumed": True,
            "programmatic_scaniverse_export_api_proven": False,
            "programmatic_360_upload_api_proven": False,
            "programmatic_asset_generation_api_proven": False,
            "programmatic_usdz_export_download_api_proven": False,
            "niantic_api_calls_performed": False,
            "remote_asset_downloads_performed": False,
            "enterprise_api_integration_required_for_automation": True,
            "provider_adapter_required_before_api_use": True,
            "local_files_staged": True,
        },
        "isaac_handoff_candidacy": {
            "candidate": bool(isaac_candidate_assets),
            "status": "candidate_assets_staged_review_required"
            if isaac_candidate_assets
            else "no_isaac_candidate_asset_staged",
            "candidate_asset_count": len(isaac_candidate_assets),
            "candidate_assets": [
                {
                    "filename": asset["filename"],
                    "suffix": asset["suffix"],
                    "asset_role": asset["asset_role"],
                    "relative_path": asset["relative_path"],
                    "sha256": asset["sha256"],
                }
                for asset in isaac_candidate_assets
            ],
            "isaac_sim_execution_proven": False,
            "physics_contact_validated": False,
            "scale_and_up_axis_review_required": True,
            "owner_system_load_trace_required": True,
        },
        "pilot_review_checklist": [
            "pair this Scaniverse import with the same-site Blueprint raw capture",
            "verify asset bounds, scale, up-axis, dependency references, and checksums",
            "inspect visual fidelity and mesh usefulness for placement/collision review",
            "record whether metric scale survived Scaniverse export",
            "capture manual workflow time and failure points",
        ],
        "scene_asset_preflight": {
            "requested": run_scene_asset_preflight,
            "ran": preflight_result is not None,
            "forwarded_asset_count": len(preflight_asset_paths),
            "result": preflight_result,
        },
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    proof_boundary = {
        "schema_version": SCANIVERSE_PROOF_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "complete",
        **CLAIM_BOUNDARY,
        "proof_upgrade_requires": [
            "raw Blueprint capture bundle or explicit source-rights packet",
            "local checksum manifest",
            "scale/unit review",
            "scene-asset preflight for supported formats",
            "owner-system simulator load trace before Isaac/MuJoCo claims",
            "policy/action/contact logs before task-success claims",
        ],
    }
    manifest_path = scaniverse_dir / "scaniverse_import_manifest.json"
    proof_boundary_path = scaniverse_dir / "scaniverse_import_proof_boundary.json"
    write_json(manifest_path, manifest)
    write_json(proof_boundary_path, proof_boundary)

    return {
        "status": manifest["status"],
        "manifest_path": str(manifest_path),
        "proof_boundary_path": str(proof_boundary_path),
        "asset_count": len(staged_assets),
        "scene_asset_preflight_ran": preflight_result is not None,
        "blockers": blockers,
        "warnings": warnings,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage Scaniverse exports as Blueprint support assets without upgrading capture truth"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Local Scaniverse export path; repeat for USDZ/PLY/SPZ/GLB/FBX/etc.",
    )
    parser.add_argument(
        "--source-manifest",
        "--blueprint-sidecar",
        dest="source_manifest",
        help="Blueprint JSON sidecar describing assignment, rights/provenance, and Scaniverse export metadata",
    )
    parser.add_argument(
        "--skip-scene-asset-preflight",
        action="store_true",
        help="Do not forward supported staged assets into scene-asset preflight",
    )
    args = parser.parse_args(argv)
    try:
        result = build_scaniverse_asset_import(
            capture_root=args.capture_root,
            assets=args.asset,
            source_manifest=args.source_manifest,
            run_scene_asset_preflight=not args.skip_scene_asset_preflight,
        )
    except (OSError, ValueError, PipelineError) as exc:
        print(f"[scaniverse-import] FAILED: {exc}")
        return 1
    print(f"[scaniverse-import] manifest={result['manifest_path']}")
    print(f"[scaniverse-import] status={result['status']}")
    return 0 if result["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
