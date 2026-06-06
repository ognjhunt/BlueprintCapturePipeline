from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from blueprint_pipeline.marble_sim_assets import build_marble_sim_assets
from blueprint_pipeline.worldlabs_asset_materialization import materialize_worldlabs_assets


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "pipeline" / "provider_run_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "provider_run_id": "op-1",
            "worldlabs_operation_id": "op-1",
            "status": "ready",
            "world_id": "world-1",
            "worldlabs_launch_url": "https://marble.worldlabs.ai/worlds/world-1",
            "privacy_safe_input": True,
        },
    )
    _write_json(
        capture_root / "pipeline" / "worldlabs_request_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "status": "ready_for_generation",
            "selected_video_uri": "gs://bucket/privacy/final_walkthrough.mov",
            "privacy_safe_input": True,
            "generation_request": {"model": "marble-1.1"},
        },
    )
    _write_json(
        capture_root / "pipeline" / "worldlabs_operation_manifest.json",
        {"operation_id": "op-1", "done": True, "status": "ready"},
    )
    return capture_root


def _world_manifest() -> dict[str, object]:
    return {
        "world_id": "world-1",
        "world_marble_url": "https://marble.worldlabs.ai/worlds/world-1",
        "model": "marble-1.1",
        "updated_at": "2026-06-06T00:00:00Z",
        "assets": {
            "mesh": {"collider_mesh_url": "https://cdn.worldlabs.ai/world-1/collider.glb"},
            "splats": {
                "spz_urls": {"full": "https://cdn.worldlabs.ai/world-1/full.spz"},
                "semantics_metadata": {
                    "metric_scale_factor": 1.0,
                    "ground_plane_offset": 0.5,
                },
            },
        },
    }


def test_materialize_worldlabs_assets_downloads_collider_and_writes_export_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(capture_root / "pipeline" / "worldlabs_world_manifest.json", _world_manifest())

    def _fake_download(url: str, output_path: Path, *, max_bytes: int | None) -> dict[str, object]:
        del max_bytes
        payload = f"downloaded:{url}".encode("utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        return {
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "content_type": "model/gltf-binary",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.worldlabs_asset_materialization._download_remote_asset",
        _fake_download,
    )

    result = materialize_worldlabs_assets(capture_root=capture_root)

    manifest = json.loads(
        (
            capture_root
            / "pipeline"
            / "worldlabs_assets"
            / "materialized_assets_manifest.json"
        ).read_text(encoding="utf-8")
    )
    export_manifest = json.loads(
        (capture_root / "pipeline" / "worldlabs_export_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "complete"
    assert result["download_count"] == 1
    assert manifest["downloads"][0]["kind"] == "collider_mesh_glb"
    assert manifest["downloads"][0]["source_url"].endswith("/collider.glb")
    assert manifest["downloads"][0]["local_path"].endswith("/worldlabs_collider.glb")
    assert manifest["skipped_candidates"][0]["kind"] == "splat_spz"
    assert export_manifest["output_collider_mesh_path"].endswith("/worldlabs_collider.glb")
    assert export_manifest["remote_collider_mesh_glb_url"].endswith("/collider.glb")
    assert export_manifest["claim_boundary"]["robot_readiness_proven"] is False

    handoff = build_marble_sim_assets(capture_root=capture_root)
    marble_asset = json.loads(
        (
            capture_root
            / "pipeline"
            / "marble_sim_assets"
            / "marble_asset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert handoff["status"] == "review_ready_with_conversion_required"
    assert marble_asset["assets"]["mesh"]["collider_mesh_source"] == (
        "materialized_worldlabs_asset"
    )
    assert marble_asset["assets"]["mesh"]["collider_mesh_glb_url"].endswith(
        "/worldlabs_collider.glb"
    )
    assert marble_asset["assets"]["mesh"]["remote_collider_mesh_glb_url"].endswith(
        "/collider.glb"
    )
