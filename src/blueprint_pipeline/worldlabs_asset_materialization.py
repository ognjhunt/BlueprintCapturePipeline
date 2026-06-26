"""Materialize World Labs / Marble output assets for local pre-GPU handoff.

This module downloads remote assets that already appear in persisted World Labs
world manifests. It does not call World Labs generation APIs, run simulators, or
upgrade rank-fidelity proof.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
import urllib.error
import urllib.request as _urllib_request
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context

WORLDLABS_ASSET_MATERIALIZATION_SCHEMA_VERSION = "worldlabs_asset_materialization.v1"
WORLDLABS_EXPORT_SCHEMA_VERSION = "worldlabs_export_manifest.v1"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "worldlabs_remote_asset_materialization_for_pre_gpu_handoff",
    "live_worldlabs_generation_call_performed": False,
    "remote_asset_downloads_performed": True,
    "simulators_run": False,
    "simulator_execution_proven": False,
    "owner_gpu_simulator_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "physics_contact_validated": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "simulator_execution_completed",
        "robot_ready",
        "deployment_ready",
        "physics_contact_validated",
        "non_ranking_operational_claim_validated",
    ],
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_url_map(value: Any, *, default_key: str) -> Dict[str, str]:
    if isinstance(value, str):
        text = _string(value)
        return {default_key: text} if text else {}
    if isinstance(value, Mapping):
        out: Dict[str, str] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = _safe_token(_string(raw_key) or default_key)
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
                out[key] = text
        return out
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        out = {}
        for index, item in enumerate(value):
            key = f"{default_key}_{index}"
            if isinstance(item, str):
                text = _string(item)
            elif isinstance(item, Mapping):
                text = _string(item.get("url") or item.get("uri") or item.get("path"))
                key = _safe_token(_string(item.get("name") or item.get("quality") or item.get("id")) or key)
            else:
                text = ""
            if text:
                out[key] = text
        return out
    return {}


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:80].strip("._-")
    return token or "asset"


def _suffix_from_url(url: str, fallback: str) -> str:
    cleaned = url.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(cleaned).suffix.lower()
    return suffix if suffix else fallback


def _world_id(world_manifest: Mapping[str, Any]) -> str | None:
    return _string(world_manifest.get("world_id") or world_manifest.get("id")) or None


def _candidate_assets(
    world_manifest: Mapping[str, Any],
    *,
    include_visual_assets: bool,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    assets = _mapping(world_manifest.get("assets"))
    mesh = _mapping(assets.get("mesh"))
    splats = _mapping(assets.get("splats"))
    candidates: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    collider_url = _string(
        mesh.get("collider_mesh_url")
        or mesh.get("collider_mesh_glb_url")
        or mesh.get("collider_url")
        or assets.get("collider_mesh_url")
    )
    if collider_url:
        candidates.append(
            {
                "kind": "collider_mesh_glb",
                "quality": "collider",
                "url": collider_url,
                "filename": f"worldlabs_collider{_suffix_from_url(collider_url, '.glb')}",
            }
        )

    high_quality_meshes = _collect_url_map(
        mesh.get("high_quality_mesh_glb_urls")
        or mesh.get("high_quality_mesh_urls")
        or mesh.get("high_quality_mesh_url")
        or mesh.get("hq_mesh_url"),
        default_key="high_quality",
    )
    spz_urls = _collect_url_map(
        splats.get("spz_urls") or splats.get("spz_url") or assets.get("spz_urls"),
        default_key="spz",
    )
    ply_urls = _collect_url_map(
        splats.get("ply_urls") or splats.get("ply_url") or assets.get("ply_urls"),
        default_key="ply",
    )
    usd_urls = _collect_url_map(splats.get("usd_urls") or splats.get("usd_url"), default_key="usd")

    for quality, url in high_quality_meshes.items():
        record = {
            "kind": "high_quality_mesh_glb",
            "quality": quality,
            "url": url,
            "filename": f"worldlabs_hq_{quality}{_suffix_from_url(url, '.glb')}",
        }
        (candidates if include_visual_assets else skipped).append(record)
    for kind, url_map, fallback_suffix in (
        ("splat_spz", spz_urls, ".spz"),
        ("splat_ply", ply_urls, ".ply"),
        ("scene_usd", usd_urls, ".usd"),
    ):
        for quality, url in url_map.items():
            record = {
                "kind": kind,
                "quality": quality,
                "url": url,
                "filename": f"worldlabs_{kind}_{quality}{_suffix_from_url(url, fallback_suffix)}",
            }
            (candidates if include_visual_assets else skipped).append(record)

    return candidates, skipped


def _download_remote_asset(url: str, output_path: Path, *, max_bytes: int | None) -> Dict[str, Any]:
    request = _urllib_request.Request(url, headers={"User-Agent": "BlueprintCapturePipeline/1.0"})
    ensure_dir(output_path.parent)
    bytes_written = 0
    try:
        with _urllib_request.urlopen(request, timeout=600) as response, output_path.open("wb") as handle:
            content_length = response.headers.get("Content-Length")
            if max_bytes is not None and content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise RuntimeError("remote_asset_exceeds_max_bytes")
                except ValueError:
                    pass
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if not chunk:  # pragma: no cover - iter(..., b"") consumes the empty sentinel.
                    break
                bytes_written += len(chunk)
                if max_bytes is not None and bytes_written > max_bytes:
                    raise RuntimeError("remote_asset_exceeds_max_bytes")
                handle.write(chunk)
    except (urllib.error.URLError, OSError, RuntimeError):
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    return {
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha_file(output_path),
        "content_type": mimetypes.guess_type(str(output_path))[0],
    }


def _export_manifest(
    *,
    materialization: Mapping[str, Any],
    materialization_path: Path,
    pipeline_dir: Path,
) -> Dict[str, Any]:
    downloads = [
        dict(item)
        for item in materialization.get("downloads") or []
        if isinstance(item, Mapping)
    ]
    collider = next((item for item in downloads if item.get("kind") == "collider_mesh_glb"), {})
    spz = {
        _string(item.get("quality")) or f"spz_{index}": item.get("local_path")
        for index, item in enumerate(downloads)
        if item.get("kind") == "splat_spz"
    }
    ply = {
        _string(item.get("quality")) or f"ply_{index}": item.get("local_path")
        for index, item in enumerate(downloads)
        if item.get("kind") == "splat_ply"
    }
    usd = {
        _string(item.get("quality")) or f"usd_{index}": item.get("local_path")
        for index, item in enumerate(downloads)
        if item.get("kind") == "scene_usd"
    }
    high_quality = {
        _string(item.get("quality")) or f"high_quality_{index}": item.get("local_path")
        for index, item in enumerate(downloads)
        if item.get("kind") == "high_quality_mesh_glb"
    }
    return {
        "schema_version": WORLDLABS_EXPORT_SCHEMA_VERSION,
        "source": "worldlabs_api_asset_materialization",
        "world_id": materialization.get("world_id"),
        "generated_at": materialization.get("generated_at"),
        "asset_materialization_manifest_path": _relative_to(pipeline_dir, materialization_path),
        "remote_asset_downloads_performed": bool(downloads),
        "output_collider_mesh_path": collider.get("local_path"),
        "collider_mesh_glb_url": collider.get("local_path"),
        "remote_collider_mesh_glb_url": collider.get("source_url"),
        "spz_urls": spz,
        "ply_urls": ply,
        "usd_urls": usd,
        "high_quality_mesh_glb_urls": high_quality,
        "download_count": len(downloads),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def materialize_worldlabs_assets(
    *,
    capture_root: str | Path,
    world_manifest: str | Path | None = None,
    include_visual_assets: bool = False,
    max_asset_bytes: int | None = 500_000_000,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    assets_dir = pipeline_dir / "worldlabs_assets"
    ensure_dir(assets_dir)
    world_path = Path(world_manifest).resolve() if world_manifest else pipeline_dir / "worldlabs_world_manifest.json"
    world = _read_optional_mapping(world_path)
    if not world:
        raise PipelineError(f"World Labs world manifest does not exist or is empty: {world_path}")

    generated_at = utc_now_iso()
    candidates, skipped_candidates = _candidate_assets(
        world,
        include_visual_assets=include_visual_assets,
    )
    downloads: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for candidate in candidates:
        url = _string(candidate.get("url"))
        if not url.lower().startswith(("http://", "https://")):
            failures.append(
                {
                    "kind": candidate.get("kind") or "unknown",
                    "source_url": url,
                    "reason": "unsupported_or_missing_remote_url",
                }
            )
            continue
        output_path = assets_dir / _safe_token(candidate.get("filename") or "worldlabs_asset")
        try:
            proof = _download_remote_asset(url, output_path, max_bytes=max_asset_bytes)
        except Exception as exc:
            failures.append(
                {
                    "kind": candidate.get("kind") or "unknown",
                    "quality": candidate.get("quality") or "",
                    "source_url": url,
                    "reason": str(exc) or exc.__class__.__name__,
                }
            )
            continue
        downloads.append(
            {
                "kind": candidate.get("kind"),
                "quality": candidate.get("quality"),
                "source_url": url,
                "local_path": str(output_path.resolve()),
                "relative_path": _relative_to(pipeline_dir, output_path),
                **proof,
            }
        )

    if failures and not downloads:
        status = "blocked"
    elif failures:
        status = "complete_with_download_failures"
    elif downloads:
        status = "complete"
    else:
        status = "blocked_no_materializable_assets"

    manifest_path = assets_dir / "materialized_assets_manifest.json"
    export_path = pipeline_dir / "worldlabs_export_manifest.json"
    manifest: Dict[str, Any] = {
        "schema_version": WORLDLABS_ASSET_MATERIALIZATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "world_id": _world_id(world),
        "status": status,
        "source_world_manifest": str(world_path),
        "include_visual_assets": include_visual_assets,
        "max_asset_bytes": max_asset_bytes,
        "download_count": len(downloads),
        "downloads": downloads,
        "failures": failures,
        "skipped_candidates": [
            {
                "kind": item.get("kind"),
                "quality": item.get("quality"),
                "source_url": item.get("url"),
                "reason": "visual_asset_download_disabled_by_default",
            }
            for item in skipped_candidates
        ],
        "download_policy": {
            "remote_asset_downloads_performed": bool(downloads),
            "visual_asset_downloads_enabled": include_visual_assets,
            "default_scope": "collider_mesh_glb_only",
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(manifest_path, manifest)
    export_manifest = _export_manifest(
        materialization=manifest,
        materialization_path=manifest_path,
        pipeline_dir=pipeline_dir,
    )
    write_json(export_path, export_manifest)
    return {
        "schema_version": "worldlabs_asset_materialization_result.v1",
        "capture_root": str(context.capture_root),
        "manifest_path": str(manifest_path.resolve()),
        "export_manifest_path": str(export_path.resolve()),
        "status": status,
        "download_count": len(downloads),
        "failure_count": len(failures),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download persisted World Labs / Marble output assets for local pre-GPU handoff"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--world-manifest",
        help="Optional worldlabs_world_manifest.json override; defaults to pipeline/worldlabs_world_manifest.json",
    )
    parser.add_argument(
        "--include-visual-assets",
        action="store_true",
        help="Also download SPZ/PLY/USD/high-quality visual assets; default downloads collider GLB only.",
    )
    parser.add_argument(
        "--max-asset-bytes",
        type=int,
        default=500_000_000,
        help="Maximum bytes per downloaded asset; use 0 for no limit.",
    )
    args = parser.parse_args(argv)
    try:
        result = materialize_worldlabs_assets(
            capture_root=args.capture_root,
            world_manifest=args.world_manifest,
            include_visual_assets=bool(args.include_visual_assets),
            max_asset_bytes=None if args.max_asset_bytes == 0 else args.max_asset_bytes,
        )
    except (PipelineError, OSError, ValueError, RuntimeError) as exc:
        print(f"[worldlabs-assets] FAILED: {exc}")
        return 1
    print(f"[worldlabs-assets] manifest={result['manifest_path']}")
    print(f"[worldlabs-assets] export_manifest={result['export_manifest_path']}")
    print(f"[worldlabs-assets] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
