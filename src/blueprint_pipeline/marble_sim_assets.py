"""Local Marble sim-asset handoff artifacts.

This module reads persisted World Labs / Marble manifests and emits deterministic
review packets for Isaac Sim, MuJoCo, and PyBullet. It does not download remote
assets, run simulators, or call the World Labs API.
"""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .common import PipelineError, ensure_dir, read_json_any, write_json
from .local_capture import resolve_local_capture_context

MARBLE_ASSET_SCHEMA_VERSION = "marble_asset_manifest.v1"
MARBLE_VALIDATION_SCHEMA_VERSION = "marble_asset_validation.v1"
MARBLE_BRIDGE_SCHEMA_VERSION = "marble_simready_bridge.v1"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "marble_sim_asset_review_packaging_only",
    "simulator_execution_proven": False,
    "isaac_sim_execution_proven": False,
    "mujoco_execution_proven": False,
    "pybullet_execution_proven": False,
    "robot_readiness_proven": False,
    "physics_contact_validated": False,
    "remote_asset_downloads_performed": False,
    "live_worldlabs_call_performed": False,
    "disallowed_claims": [
        "robot_ready",
        "physics_validated",
        "isaac_sim_loaded",
        "mujoco_loaded",
        "pybullet_loaded",
        "articulated_interaction_ready",
        "policy_execution_passed",
    ],
    "robot_readiness_requires": [
        "simulator load traces",
        "action logs",
        "physics/contact validation logs",
        "robot-team-owned robot assets",
        "accepted simulator or real robot trial evidence",
    ],
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_file(base_dir: Path, target: Path) -> str | None:
    return _relative_to(base_dir, target) if target.is_file() else None


def _deterministic_timestamp(*payloads: Mapping[str, Any]) -> str | None:
    for payload in payloads:
        for key in ("updated_at", "created_at", "completed_at", "generated_at"):
            text = _string(payload.get(key))
            if text:
                return text
    return None


def _collect_urls(value: Any, *, default_key: str) -> Dict[str, str]:
    urls: Dict[str, str] = {}
    if isinstance(value, str):
        text = _string(value)
        if text:
            urls[default_key] = text
        return urls
    if isinstance(value, Mapping):
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = _string(raw_key) or default_key
            if isinstance(raw_value, str):
                text = _string(raw_value)
            elif isinstance(raw_value, Mapping):
                text = _string(
                    raw_value.get("url")
                    or raw_value.get("uri")
                    or raw_value.get("path")
                    or raw_value.get("asset_url")
                )
            else:
                text = ""
            if text:
                urls[key] = text
        return urls
    if isinstance(value, list):
        for index, item in enumerate(value):
            key = f"{default_key}_{index}"
            if isinstance(item, str):
                text = _string(item)
            elif isinstance(item, Mapping):
                text = _string(item.get("url") or item.get("uri") or item.get("path"))
                key = _string(item.get("name") or item.get("quality") or item.get("id")) or key
            else:
                text = ""
            if text:
                urls[key] = text
    return urls


def _merge_url_sources(*sources: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for source in sources:
        for key, value in source.items():
            if value and value not in merged.values():
                merged[key] = value
    return merged


def _is_remote_ref(value: str) -> bool:
    return value.lower().startswith(("http://", "https://", "gs://", "s3://", "omniverse://"))


def _first_value(values: Mapping[str, str], *, remote: bool | None = None) -> str:
    for value in values.values():
        text = _string(value)
        if not text:
            continue
        if remote is True and not _is_remote_ref(text):
            continue
        if remote is False and _is_remote_ref(text):
            continue
        return text
    return ""


def _conversion_manifest_paths(pipeline_dir: Path, marble_dir: Path) -> List[Path]:
    return [
        marble_dir / "conversion_manifest.json",
        marble_dir / "spz_to_ply_conversion_manifest.json",
        pipeline_dir / "marble_ply_conversion_manifest.json",
        pipeline_dir / "worldlabs_ply_conversion_manifest.json",
        pipeline_dir / "worldlabs_export_manifest.json",
    ]


def _conversion_outputs(pipeline_dir: Path, marble_dir: Path) -> Dict[str, Any]:
    manifests: List[Dict[str, Any]] = []
    ply_urls: Dict[str, str] = {}
    usd_urls: Dict[str, str] = {}
    collider_mesh_glb_urls: Dict[str, str] = {}
    high_quality_mesh_glb_urls: Dict[str, str] = {}
    for path in _conversion_manifest_paths(pipeline_dir, marble_dir):
        payload = _read_optional_mapping(path)
        if not payload:
            continue
        manifests.append({"path": str(path.resolve()), "payload": payload})
        ply_urls = _merge_url_sources(
            ply_urls,
            _collect_urls(payload.get("ply_urls"), default_key="conversion_ply"),
            _collect_urls(payload.get("ply_url"), default_key="conversion_ply"),
            _collect_urls(payload.get("output_ply_uri"), default_key="conversion_ply"),
            _collect_urls(payload.get("output_ply_url"), default_key="conversion_ply"),
            _collect_urls(payload.get("output_ply_path"), default_key="conversion_ply"),
        )
        usd_urls = _merge_url_sources(
            usd_urls,
            _collect_urls(payload.get("usd_urls"), default_key="conversion_usd"),
            _collect_urls(payload.get("usd_url"), default_key="conversion_usd"),
            _collect_urls(payload.get("usdz_url"), default_key="conversion_usdz"),
            _collect_urls(payload.get("output_usd_uri"), default_key="conversion_usd"),
            _collect_urls(payload.get("output_usdz_uri"), default_key="conversion_usdz"),
        )
        collider_mesh_glb_urls = _merge_url_sources(
            collider_mesh_glb_urls,
            _collect_urls(payload.get("collider_mesh_url"), default_key="collider_glb"),
            _collect_urls(payload.get("collider_mesh_glb_url"), default_key="collider_glb"),
            _collect_urls(payload.get("collider_glb_url"), default_key="collider_glb"),
            _collect_urls(payload.get("output_collider_mesh_uri"), default_key="collider_glb"),
            _collect_urls(payload.get("output_collider_mesh_url"), default_key="collider_glb"),
            _collect_urls(payload.get("output_collider_mesh_path"), default_key="collider_glb"),
        )
        high_quality_mesh_glb_urls = _merge_url_sources(
            high_quality_mesh_glb_urls,
            _collect_urls(payload.get("high_quality_mesh_urls"), default_key="high_quality_glb"),
            _collect_urls(payload.get("high_quality_mesh_url"), default_key="high_quality_glb"),
            _collect_urls(payload.get("high_quality_mesh_glb_urls"), default_key="high_quality_glb"),
            _collect_urls(payload.get("high_quality_mesh_glb_url"), default_key="high_quality_glb"),
            _collect_urls(payload.get("hq_mesh_url"), default_key="high_quality_glb"),
            _collect_urls(payload.get("hq_mesh_glb_url"), default_key="high_quality_glb"),
            _collect_urls(payload.get("output_high_quality_mesh_uri"), default_key="high_quality_glb"),
            _collect_urls(payload.get("output_high_quality_mesh_url"), default_key="high_quality_glb"),
            _collect_urls(payload.get("output_high_quality_mesh_path"), default_key="high_quality_glb"),
        )
    return {
        "manifests": manifests,
        "ply_urls": ply_urls,
        "usd_urls": usd_urls,
        "collider_mesh_glb_urls": collider_mesh_glb_urls,
        "high_quality_mesh_glb_urls": high_quality_mesh_glb_urls,
        "conversion_completed": bool(
            ply_urls or usd_urls or collider_mesh_glb_urls or high_quality_mesh_glb_urls
        ),
    }


def _normalize_splat_assets(
    *,
    world_manifest: Mapping[str, Any],
    pipeline_dir: Path,
    marble_dir: Path,
) -> Dict[str, Any]:
    assets = _mapping(world_manifest.get("assets"))
    splats = _mapping(assets.get("splats"))
    conversion = _conversion_outputs(pipeline_dir, marble_dir)
    spz_urls = _merge_url_sources(
        _collect_urls(splats.get("spz_urls"), default_key="spz"),
        _collect_urls(splats.get("spz_url"), default_key="spz"),
        _collect_urls(assets.get("spz_urls"), default_key="spz"),
    )
    ply_urls = _merge_url_sources(
        _collect_urls(splats.get("ply_urls"), default_key="ply"),
        _collect_urls(splats.get("ply_url"), default_key="ply"),
        _collect_urls(splats.get("exported_ply_urls"), default_key="ply"),
        _collect_urls(assets.get("ply_urls"), default_key="ply"),
        _mapping(conversion.get("ply_urls")),
    )
    usd_urls = _mapping(conversion.get("usd_urls"))
    return {
        "spz_urls": spz_urls,
        "ply_urls": ply_urls,
        "usd_urls": usd_urls,
        "conversion_manifests": [
            {"path": item["path"]} for item in conversion.get("manifests", []) if item.get("path")
        ],
        "api_direct_ply_supported": False,
        "api_direct_ply_note": "World Labs API docs say direct PLY retrieval is not supported; PLY must come from explicit Marble export or conversion evidence.",
        "preferred_visual_asset_format": "ply"
        if ply_urls
        else "usd"
        if usd_urls
        else "spz"
        if spz_urls
        else "missing",
    }


def _normalize_mesh_assets(
    *,
    world_manifest: Mapping[str, Any],
    pipeline_dir: Path,
    marble_dir: Path,
) -> Dict[str, Any]:
    assets = _mapping(world_manifest.get("assets"))
    mesh = _mapping(assets.get("mesh"))
    conversion = _conversion_outputs(pipeline_dir, marble_dir)
    exported_colliders = _mapping(conversion.get("collider_mesh_glb_urls"))
    collider = _string(
        mesh.get("collider_mesh_url")
        or mesh.get("collider_mesh_glb_url")
        or mesh.get("collider_url")
        or assets.get("collider_mesh_url")
    )
    high_quality = _merge_url_sources(
        _collect_urls(mesh.get("high_quality_mesh_urls"), default_key="high_quality_glb"),
        _collect_urls(mesh.get("high_quality_mesh_url"), default_key="high_quality_glb"),
        _collect_urls(mesh.get("high_quality_mesh_glb_urls"), default_key="high_quality_glb"),
        _collect_urls(mesh.get("high_quality_mesh_glb_url"), default_key="high_quality_glb"),
        _collect_urls(mesh.get("hq_mesh_url"), default_key="high_quality_glb"),
        _collect_urls(mesh.get("hq_mesh_glb_url"), default_key="high_quality_glb"),
        _mapping(conversion.get("high_quality_mesh_glb_urls")),
    )
    if not collider and exported_colliders:
        collider = next(iter(exported_colliders.values()), "")
    local_collider = _first_value(exported_colliders, remote=False)
    remote_collider = collider if _is_remote_ref(collider) else ""
    if local_collider:
        collider = local_collider
    return {
        "collider_mesh_glb_url": collider or None,
        "remote_collider_mesh_glb_url": remote_collider or None,
        "local_collider_mesh_glb_path": local_collider or None,
        "collider_mesh_available": bool(collider),
        "collider_mesh_purpose": "physics_collision_review_input",
        "collider_mesh_source": "materialized_worldlabs_asset"
        if local_collider
        else "world_manifest"
        if collider and not exported_colliders
        else "explicit_marble_export_or_conversion_manifest"
        if collider
        else "missing",
        "exported_collider_mesh_glb_urls": exported_colliders,
        "high_quality_mesh_glb_urls": high_quality,
        "high_quality_mesh_available": bool(high_quality),
    }


def _semantics_metadata(world_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    assets = _mapping(world_manifest.get("assets"))
    splats = _mapping(assets.get("splats"))
    raw = _mapping(
        splats.get("semantics_metadata")
        or assets.get("semantics_metadata")
        or world_manifest.get("semantics_metadata")
    )
    metric_scale_factor = _maybe_float(raw.get("metric_scale_factor"))
    ground_plane_offset = _maybe_float(raw.get("ground_plane_offset"))
    return {
        "raw": raw,
        "metric_scale_factor": metric_scale_factor,
        "ground_plane_offset": ground_plane_offset,
        "metric_scale_available": metric_scale_factor is not None,
        "ground_plane_alignment_available": ground_plane_offset is not None,
        "metric_transform_note": "multiply XYZ and isotropic scale by metric_scale_factor, then subtract ground_plane_offset from Y after scaling",
    }


def _normalize_world(
    *,
    provider_run_manifest: Mapping[str, Any],
    operation_manifest: Mapping[str, Any],
    world_manifest: Mapping[str, Any],
    request_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    operation_error = _mapping(operation_manifest.get("error"))
    operation_status = _string(
        provider_run_manifest.get("operation_terminal_status")
        or operation_manifest.get("status")
        or ("ready" if operation_manifest.get("done") and not operation_error else "")
        or provider_run_manifest.get("status")
    )
    return {
        "world_id": _string(
            world_manifest.get("world_id")
            or provider_run_manifest.get("world_id")
            or operation_manifest.get("world_id")
        )
        or None,
        "operation_id": _string(
            provider_run_manifest.get("worldlabs_operation_id")
            or provider_run_manifest.get("provider_run_id")
            or operation_manifest.get("operation_id")
            or operation_manifest.get("id")
        )
        or None,
        "operation_status": operation_status or None,
        "provider_run_id": _string(provider_run_manifest.get("provider_run_id")) or None,
        "provider_name": _string(provider_run_manifest.get("provider_name")) or "world_labs",
        "model": _string(
            world_manifest.get("model")
            or request_manifest.get("provider_model")
            or _mapping(request_manifest.get("generation_request")).get("model")
            or provider_run_manifest.get("provider_model")
        )
        or None,
        "world_marble_url": _string(
            world_manifest.get("world_marble_url")
            or provider_run_manifest.get("worldlabs_launch_url")
            or provider_run_manifest.get("preview_launch_url")
            or provider_run_manifest.get("launch_url")
        )
        or None,
    }


def _lineage(
    *,
    request_manifest: Mapping[str, Any],
    provider_run_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    input_audit = _mapping(request_manifest.get("input_audit"))
    selected_checksum = _string(
        request_manifest.get("selected_input_checksum_sha256")
        or provider_run_manifest.get("selected_input_checksum_sha256")
        or input_audit.get("output_checksum_sha256")
    )
    source_checksum = _string(
        request_manifest.get("source_input_checksum_sha256")
        or provider_run_manifest.get("source_input_checksum_sha256")
        or input_audit.get("source_checksum_sha256")
    )
    return {
        "selected_video_source_id": _string(request_manifest.get("selected_video_source_id")) or None,
        "selected_video_uri": _string(request_manifest.get("selected_video_uri")) or None,
        "source_manifest_uri": _string(
            request_manifest.get("source_manifest_uri")
            or provider_run_manifest.get("source_manifest_uri")
            or input_audit.get("source_manifest_uri")
        )
        or None,
        "worldlabs_input_audit_uri": _string(
            request_manifest.get("worldlabs_input_audit_uri")
            or provider_run_manifest.get("worldlabs_input_audit_uri")
        )
        or None,
        "privacy_safe_input": bool(
            request_manifest.get("privacy_safe_input")
            or provider_run_manifest.get("privacy_safe_input")
        ),
        "selected_input_checksum_sha256": selected_checksum or None,
        "source_input_checksum_sha256": source_checksum or None,
        "canonical_site_package_uri": _string(
            request_manifest.get("canonical_site_package_uri")
            or provider_run_manifest.get("canonical_site_package_uri")
        )
        or None,
        "provider_adapter_input_uri": _string(
            request_manifest.get("provider_adapter_input_uri")
            or provider_run_manifest.get("provider_adapter_input_uri")
        )
        or None,
        "checksums_available": bool(selected_checksum or source_checksum),
    }


def _validation_payload(asset_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    world = _mapping(asset_manifest.get("world"))
    assets = _mapping(asset_manifest.get("assets"))
    splats = _mapping(assets.get("splats"))
    mesh = _mapping(assets.get("mesh"))
    semantics = _mapping(asset_manifest.get("semantics"))
    lineage = _mapping(asset_manifest.get("request_input_lineage"))
    blockers: List[str] = []
    warnings: List[str] = []

    if not world.get("world_id"):
        blockers.append("missing_world_id")
    if not world.get("world_marble_url"):
        warnings.append("missing_world_marble_url")
    if not splats.get("spz_urls") and not splats.get("ply_urls") and not splats.get("usd_urls"):
        blockers.append("missing_splat_assets")
    if not mesh.get("collider_mesh_glb_url"):
        blockers.append("missing_collider_mesh_glb")
    if semantics.get("metric_scale_factor") is None:
        blockers.append("missing_metric_scale_factor")
    if semantics.get("ground_plane_offset") is None:
        blockers.append("missing_ground_plane_offset")
    if not lineage.get("privacy_safe_input"):
        warnings.append("privacy_safe_input_not_proven_in_request_manifest")
    if not lineage.get("worldlabs_input_audit_uri"):
        warnings.append("missing_worldlabs_input_audit_uri")
    if not lineage.get("checksums_available"):
        warnings.append("missing_input_checksums")

    has_ply_or_usd = bool(splats.get("ply_urls") or splats.get("usd_urls"))
    has_spz = bool(splats.get("spz_urls"))
    needs_conversion = has_spz and not has_ply_or_usd
    if blockers:
        overall_status = "blocked"
    elif needs_conversion:
        overall_status = "review_ready_with_conversion_required"
    elif warnings:
        overall_status = "review_ready_with_warnings"
    else:
        overall_status = "review_ready"

    return {
        "schema_version": MARBLE_VALIDATION_SCHEMA_VERSION,
        "source_time": asset_manifest.get("source_time"),
        "overall_status": overall_status,
        "blockers": blockers,
        "warnings": list(dict.fromkeys(warnings)),
        "physics_collision_review_ready": bool(mesh.get("collider_mesh_glb_url")) and not blockers,
        "metric_alignment_ready": semantics.get("metric_scale_factor") is not None
        and semantics.get("ground_plane_offset") is not None,
        "isaac_visual_conversion_required": needs_conversion,
        "mujoco_direct_load_proven": False,
        "pybullet_direct_load_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _isaac_manifest(asset_manifest: Mapping[str, Any], validation: Mapping[str, Any]) -> Dict[str, Any]:
    splats = _mapping(_mapping(asset_manifest.get("assets")).get("splats"))
    mesh = _mapping(_mapping(asset_manifest.get("assets")).get("mesh"))
    has_ply = bool(splats.get("ply_urls"))
    has_usd = bool(splats.get("usd_urls"))
    needs_conversion: bool | str = False if (has_ply or has_usd) else "spz_to_ply_or_usd"
    return {
        "schema_version": "marble_isaac_sim_review_manifest.v1",
        "status": "blocked"
        if validation.get("overall_status") == "blocked"
        else "asset_review_ready"
        if (has_ply or has_usd)
        else "needs_visual_conversion",
        "review_asset_state": "ply_or_usd_and_collider_available"
        if (has_ply or has_usd)
        else "spz_and_collider_available_conversion_required",
        "visual_assets": {
            "spz_urls": splats.get("spz_urls") or {},
            "ply_urls": splats.get("ply_urls") or {},
            "usd_urls": splats.get("usd_urls") or {},
            "needs_conversion": needs_conversion,
            "conversion_executed": bool(has_usd),
        },
        "collider_mesh_glb_url": mesh.get("collider_mesh_glb_url"),
        "scale_alignment": asset_manifest.get("semantics"),
        "recommended_review_steps": [
            "convert SPZ to PLY/USD or use explicit PLY/USD export when required",
            "import visual asset and GLB collider into Isaac Sim",
            "apply metric scale factor and ground plane offset",
            "capture simulator load trace before claiming execution",
        ],
        "execution_claim": False,
        "load_status": "not_executed",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _framework_manifest(
    *,
    framework: str,
    asset_manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    mesh = _mapping(_mapping(asset_manifest.get("assets")).get("mesh"))
    conversion = {
        "mujoco": "glb_collider_to_mjcf_or_mesh_asset",
        "pybullet": "glb_collider_to_urdf_or_collision_shape",
    }[framework]
    return {
        "schema_version": f"marble_{framework}_review_manifest.v1",
        "status": "blocked"
        if validation.get("overall_status") == "blocked"
        else "collider_conversion_review_input_available",
        "collider_mesh_glb_url": mesh.get("collider_mesh_glb_url"),
        "expected_conversion": conversion,
        "generated_simulator_model": None,
        "generated_model_validated": False,
        "load_trace_uri": None,
        "direct_simulator_success_claim": False,
        "execution_claim": False,
        "load_status": "not_executed",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _sha_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _source_artifacts(
    *,
    marble_dir: Path,
    provider_run_path: Path,
    request_path: Path,
    operation_path: Path,
    world_path: Path,
) -> Dict[str, str | None]:
    return {
        "provider_run_manifest": _relative_if_file(marble_dir, provider_run_path),
        "worldlabs_request_manifest": _relative_if_file(marble_dir, request_path),
        "worldlabs_operation_manifest": _relative_if_file(marble_dir, operation_path),
        "worldlabs_world_manifest": _relative_if_file(marble_dir, world_path),
    }


def build_marble_sim_assets(
    *,
    capture_root: str | Path,
    world_manifest: str | Path | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    marble_dir = pipeline_dir / "marble_sim_assets"
    simulator_dir = marble_dir / "simulators"
    ensure_dir(simulator_dir)

    provider_run_path = pipeline_dir / "provider_run_manifest.json"
    request_path = pipeline_dir / "worldlabs_request_manifest.json"
    operation_path = pipeline_dir / "worldlabs_operation_manifest.json"
    world_path = Path(world_manifest).resolve() if world_manifest else pipeline_dir / "worldlabs_world_manifest.json"
    if world_manifest and not world_path.is_file():
        raise PipelineError(f"World manifest override does not exist: {world_path}")

    provider_run = _read_optional_mapping(provider_run_path)
    request = _read_optional_mapping(request_path)
    operation = _read_optional_mapping(operation_path)
    world = _read_optional_mapping(world_path)
    source_time = _deterministic_timestamp(world, operation, provider_run, request)
    splats = _normalize_splat_assets(
        world_manifest=world,
        pipeline_dir=pipeline_dir,
        marble_dir=marble_dir,
    )
    mesh = _normalize_mesh_assets(
        world_manifest=world,
        pipeline_dir=pipeline_dir,
        marble_dir=marble_dir,
    )
    asset_manifest: Dict[str, Any] = {
        "schema_version": MARBLE_ASSET_SCHEMA_VERSION,
        "artifact_purpose": "local_marble_sim_asset_handoff",
        "source_time": source_time,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "world": _normalize_world(
            provider_run_manifest=provider_run,
            operation_manifest=operation,
            world_manifest=world,
            request_manifest=request,
        ),
        "assets": {
            "splats": splats,
            "mesh": mesh,
        },
        "semantics": _semantics_metadata(world),
        "request_input_lineage": _lineage(
            request_manifest=request,
            provider_run_manifest=provider_run,
        ),
        "source_artifacts": _source_artifacts(
            marble_dir=marble_dir,
            provider_run_path=provider_run_path,
            request_path=request_path,
            operation_path=operation_path,
            world_path=world_path,
        ),
        "download_policy": {
            "remote_asset_downloads_performed": False,
            "remote_asset_downloads_allowed_by_default": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    asset_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": asset_manifest["scene_id"],
            "capture_id": asset_manifest["capture_id"],
            "world": asset_manifest["world"],
            "assets": asset_manifest["assets"],
            "semantics": asset_manifest["semantics"],
            "request_input_lineage": asset_manifest["request_input_lineage"],
        }
    )
    validation = _validation_payload(asset_manifest)
    asset_manifest["status"] = validation["overall_status"]

    isaac = _isaac_manifest(asset_manifest, validation)
    mujoco = _framework_manifest(
        framework="mujoco",
        asset_manifest=asset_manifest,
        validation=validation,
    )
    pybullet = _framework_manifest(
        framework="pybullet",
        asset_manifest=asset_manifest,
        validation=validation,
    )

    asset_path = marble_dir / "marble_asset_manifest.json"
    validation_path = marble_dir / "marble_asset_validation.json"
    bridge_path = marble_dir / "marble_simready_bridge.json"
    isaac_path = simulator_dir / "isaac_sim_review_manifest.json"
    mujoco_path = simulator_dir / "mujoco_review_manifest.json"
    pybullet_path = simulator_dir / "pybullet_review_manifest.json"

    simulator_manifests = {
        "isaac_sim": _relative_to(marble_dir, isaac_path),
        "mujoco": _relative_to(marble_dir, mujoco_path),
        "pybullet": _relative_to(marble_dir, pybullet_path),
    }
    bridge = {
        "schema_version": MARBLE_BRIDGE_SCHEMA_VERSION,
        "status": validation["overall_status"],
        "source_time": source_time,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "world_id": _mapping(asset_manifest.get("world")).get("world_id"),
        "world_marble_url": _mapping(asset_manifest.get("world")).get("world_marble_url"),
        "asset_manifest_path": _relative_to(marble_dir, asset_path),
        "validation_path": _relative_to(marble_dir, validation_path),
        "simulator_review_manifests": simulator_manifests,
        "evaluation_prep_summary": {
            "marble_sim_asset_status": validation["overall_status"],
            "collider_mesh_available": bool(mesh.get("collider_mesh_glb_url")),
            "metric_alignment_ready": bool(validation.get("metric_alignment_ready")),
            "isaac_visual_conversion_required": bool(
                validation.get("isaac_visual_conversion_required")
            ),
            "robot_readiness_proven": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }

    write_json(asset_path, asset_manifest)
    write_json(validation_path, validation)
    write_json(isaac_path, isaac)
    write_json(mujoco_path, mujoco)
    write_json(pybullet_path, pybullet)
    write_json(bridge_path, bridge)

    return {
        "schema_version": "marble_sim_assets_result.v1",
        "capture_root": str(context.capture_root),
        "manifest_path": str(asset_path.resolve()),
        "validation_path": str(validation_path.resolve()),
        "bridge_path": str(bridge_path.resolve()),
        "status": validation["overall_status"],
        "simulator_review_manifests": {
            "isaac_sim": str(isaac_path.resolve()),
            "mujoco": str(mujoco_path.resolve()),
            "pybullet": str(pybullet_path.resolve()),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build local Marble sim-asset review artifacts without live provider or simulator execution"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--world-manifest",
        help="Optional local World Labs world manifest override; defaults to pipeline/worldlabs_world_manifest.json",
    )
    args = parser.parse_args(argv)

    try:
        result = build_marble_sim_assets(
            capture_root=args.capture_root,
            world_manifest=args.world_manifest,
        )
    except (PipelineError, ValueError, OSError) as exc:
        print(f"[marble-sim-assets] FAILED: {exc}")
        return 1

    print(f"[marble-sim-assets] manifest={result['manifest_path']}")
    print(f"[marble-sim-assets] validation={result['validation_path']}")
    print(f"[marble-sim-assets] bridge={result['bridge_path']}")
    print(f"[marble-sim-assets] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
