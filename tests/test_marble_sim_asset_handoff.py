from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import marble_sim_assets as msa
from blueprint_pipeline.marble_sim_assets import build_marble_sim_assets
from blueprint_pipeline.provider_preview import WorldLabsPreviewProvider, run_preview_provider


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw",
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    pipeline_dir = capture_root / "pipeline"
    _write_json(
        pipeline_dir / "provider_run_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "provider_run_id": "op-1",
            "worldlabs_operation_id": "op-1",
            "operation_terminal_status": "ready",
            "status": "ready",
            "world_id": "world-1",
            "worldlabs_launch_url": "https://marble.worldlabs.ai/worlds/world-1",
            "selected_input_checksum_sha256": "selected-sha",
            "source_input_checksum_sha256": "source-sha",
            "source_manifest_uri": "gs://local-blueprint/pipeline/privacy_processing_manifest.json",
            "worldlabs_input_audit_uri": "gs://local-blueprint/pipeline/worldlabs_input_audit.json",
            "privacy_safe_input": True,
        },
    )
    _write_json(
        pipeline_dir / "worldlabs_request_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "status": "ready_for_generation",
            "selected_video_source_id": "privacy_safe_world_model_input",
            "selected_video_uri": "gs://local-blueprint/privacy/final_walkthrough.mov",
            "source_manifest_uri": "gs://local-blueprint/pipeline/privacy_processing_manifest.json",
            "worldlabs_input_audit_uri": "gs://local-blueprint/pipeline/worldlabs_input_audit.json",
            "selected_input_checksum_sha256": "selected-sha",
            "source_input_checksum_sha256": "source-sha",
            "privacy_safe_input": True,
            "generation_request": {"model": "marble-1.1"},
        },
    )
    _write_json(
        pipeline_dir / "worldlabs_operation_manifest.json",
        {"operation_id": "op-1", "done": True, "status": "ready"},
    )
    return capture_root


def _world_manifest(*, include_collider: bool = True, include_ply: bool = False) -> dict[str, object]:
    mesh: dict[str, object] = {}
    if include_collider:
        mesh["collider_mesh_url"] = "https://cdn.worldlabs.ai/world-1/collider.glb"
    splats: dict[str, object] = {
        "semantics_metadata": {
            "metric_scale_factor": 0.42,
            "ground_plane_offset": 1.25,
        },
        "spz_urls": {"full": "https://cdn.worldlabs.ai/world-1/full.spz"},
    }
    if include_ply:
        splats["ply_urls"] = {"full": "https://cdn.worldlabs.ai/world-1/full.ply"}
    return {
        "display_name": "Fixture Marble world",
        "world_id": "world-1",
        "world_marble_url": "https://marble.worldlabs.ai/worlds/world-1",
        "model": "marble-1.1",
        "updated_at": "2026-06-02T00:00:00Z",
        "assets": {
            "mesh": mesh,
            "splats": splats,
        },
    }


def test_marble_sim_assets_api_world_spz_marks_isaac_conversion_required(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(capture_root / "pipeline" / "worldlabs_world_manifest.json", _world_manifest())

    result = build_marble_sim_assets(capture_root=capture_root)

    marble_root = capture_root / "pipeline" / "marble_sim_assets"
    asset_manifest = json.loads((marble_root / "marble_asset_manifest.json").read_text())
    validation = json.loads((marble_root / "marble_asset_validation.json").read_text())
    isaac = json.loads((marble_root / "simulators" / "isaac_sim_review_manifest.json").read_text())

    assert result["status"] == "review_ready_with_conversion_required"
    assert asset_manifest["assets"]["splats"]["spz_urls"]["full"].endswith("/full.spz")
    assert asset_manifest["assets"]["mesh"]["collider_mesh_glb_url"].endswith("/collider.glb")
    assert asset_manifest["semantics"]["metric_scale_factor"] == 0.42
    assert validation["physics_collision_review_ready"] is True
    assert validation["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert validation["rank_fidelity_result_proven"] is False
    assert isaac["visual_assets"]["needs_conversion"] == "spz_to_ply_or_usd"
    assert isaac["load_status"] == "not_executed"


def test_marble_sim_assets_export_manifest_with_ply_is_isaac_review_ready(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(
        capture_root / "pipeline" / "worldlabs_world_manifest.json",
        _world_manifest(include_ply=True),
    )

    build_marble_sim_assets(capture_root=capture_root)

    marble_root = capture_root / "pipeline" / "marble_sim_assets"
    validation = json.loads((marble_root / "marble_asset_validation.json").read_text())
    isaac = json.loads((marble_root / "simulators" / "isaac_sim_review_manifest.json").read_text())

    assert validation["overall_status"] == "review_ready"
    assert isaac["status"] == "asset_review_ready"
    assert isaac["visual_assets"]["needs_conversion"] is False
    assert isaac["visual_assets"]["ply_urls"]["full"].endswith("/full.ply")
    assert isaac["execution_claim"] is False


def test_marble_sim_assets_explicit_export_manifest_supplies_ply_and_glb(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(
        capture_root / "pipeline" / "worldlabs_world_manifest.json",
        _world_manifest(include_collider=False),
    )
    _write_json(
        capture_root / "pipeline" / "worldlabs_export_manifest.json",
        {
            "schema_version": "worldlabs_export_manifest.v1",
            "source": "marble_web_export",
            "ply_urls": {"full": "https://cdn.worldlabs.ai/world-1/full-export.ply"},
            "collider_mesh_glb_url": "https://cdn.worldlabs.ai/world-1/collider-export.glb",
            "high_quality_mesh_glb_urls": {
                "textured_600k": "https://cdn.worldlabs.ai/world-1/hq-textured.glb",
                "vertex_color_1m": "https://cdn.worldlabs.ai/world-1/hq-vertex.glb",
            },
        },
    )

    result = build_marble_sim_assets(capture_root=capture_root)

    marble_root = capture_root / "pipeline" / "marble_sim_assets"
    asset_manifest = json.loads((marble_root / "marble_asset_manifest.json").read_text())
    validation = json.loads((marble_root / "marble_asset_validation.json").read_text())
    isaac = json.loads((marble_root / "simulators" / "isaac_sim_review_manifest.json").read_text())
    mujoco = json.loads((marble_root / "simulators" / "mujoco_review_manifest.json").read_text())

    assert result["status"] == "review_ready"
    assert validation["overall_status"] == "review_ready"
    assert validation["isaac_visual_conversion_required"] is False
    assert asset_manifest["assets"]["splats"]["ply_urls"]["full"].endswith("/full-export.ply")
    assert asset_manifest["assets"]["mesh"]["collider_mesh_glb_url"].endswith(
        "/collider-export.glb"
    )
    assert asset_manifest["assets"]["mesh"]["high_quality_mesh_available"] is True
    assert asset_manifest["assets"]["mesh"]["high_quality_mesh_glb_urls"][
        "textured_600k"
    ].endswith("/hq-textured.glb")
    assert isaac["status"] == "asset_review_ready"
    assert isaac["visual_assets"]["needs_conversion"] is False
    assert mujoco["collider_mesh_glb_url"].endswith("/collider-export.glb")
    assert mujoco["load_status"] == "not_executed"


def test_marble_sim_assets_missing_collider_blocks_physics_readiness(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(
        capture_root / "pipeline" / "worldlabs_world_manifest.json",
        _world_manifest(include_collider=False),
    )

    result = build_marble_sim_assets(capture_root=capture_root)

    validation = json.loads(
        (capture_root / "pipeline" / "marble_sim_assets" / "marble_asset_validation.json").read_text()
    )
    mujoco = json.loads(
        (
            capture_root
            / "pipeline"
            / "marble_sim_assets"
            / "simulators"
            / "mujoco_review_manifest.json"
        ).read_text()
    )

    assert result["status"] == "blocked"
    assert "missing_collider_mesh_glb" in validation["blockers"]
    assert validation["physics_collision_review_ready"] is False
    assert mujoco["direct_simulator_success_claim"] is False


def test_run_preview_provider_persists_marble_sim_asset_handoff(tmp_path: Path, monkeypatch) -> None:
    capture_root = _build_capture_root(tmp_path)
    pipeline_dir = capture_root / "pipeline"

    def _fake_submit(self, *, descriptor, capture_root, provider_adapter_input=None):  # type: ignore[no-untyped-def]
        del descriptor, capture_root, provider_adapter_input
        return {
            "provider_name": self.provider_name,
            "provider_model": self.provider_model,
            "provider_run_id": "op-1",
            "status": "processing",
            "artifact_uris": {},
            "cost_usd": 0.0,
            "latency_ms": 1,
            "worldlabs_request_manifest": {
                "schema_version": "v1",
                "provider_name": "world_labs",
                "provider_model": "marble-1.1",
                "selected_video_uri": "gs://local-blueprint/privacy/final_walkthrough.mov",
                "source_manifest_uri": "gs://local-blueprint/privacy/manifest.json",
                "worldlabs_input_audit_uri": "gs://local-blueprint/pipeline/worldlabs_input_audit.json",
                "selected_input_checksum_sha256": "selected-sha",
                "privacy_safe_input": True,
            },
            "selected_input_checksum_sha256": "selected-sha",
            "worldlabs_input_audit_uri": "gs://local-blueprint/pipeline/worldlabs_input_audit.json",
            "privacy_safe_input": True,
        }

    def _fake_poll(self, *, run_id):  # type: ignore[no-untyped-def]
        assert run_id == "op-1"
        return {
            "provider_run_id": run_id,
            "status": "ready",
            "world_id": "world-1",
            "launch_url": "https://marble.worldlabs.ai/worlds/world-1",
            "operation_terminal_status": "ready",
            "worldlabs_operation": {"operation_id": run_id, "done": True, "status": "ready"},
            "worldlabs_world": _world_manifest(),
        }

    def _fake_materialize_assets(
        *,
        capture_root,
        world_manifest=None,
        include_visual_assets=False,
        max_asset_bytes=500_000_000,
    ):  # type: ignore[no-untyped-def]
        del world_manifest, include_visual_assets, max_asset_bytes
        materialized_dir = Path(capture_root) / "pipeline" / "worldlabs_assets"
        materialized_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = materialized_dir / "materialized_assets_manifest.json"
        export_path = Path(capture_root) / "pipeline" / "worldlabs_export_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": "worldlabs_asset_materialization.v1",
                "status": "complete",
                "downloads": [],
            },
        )
        _write_json(
            export_path,
            {
                "schema_version": "worldlabs_export_manifest.v1",
                "source": "test_fixture",
            },
        )
        return {
            "schema_version": "worldlabs_asset_materialization_result.v1",
            "status": "complete",
            "manifest_path": str(manifest_path),
            "export_manifest_path": str(export_path),
            "download_count": 0,
            "failure_count": 0,
        }

    monkeypatch.setattr(WorldLabsPreviewProvider, "submit", _fake_submit)
    monkeypatch.setattr(WorldLabsPreviewProvider, "poll", _fake_poll)
    monkeypatch.setattr(
        "blueprint_pipeline.worldlabs_asset_materialization.materialize_worldlabs_assets",
        _fake_materialize_assets,
    )

    result = run_preview_provider(
        provider_name="world_labs",
        descriptor={"capture_id": "capture-1", "raw_prefix_uri": "gs://local-blueprint/raw"},
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
    )

    assert result["marble_sim_asset_handoff"]["status"] == "review_ready_with_conversion_required"
    assert result["worldlabs_asset_materialization"]["status"] == "complete"
    assert result["artifact_uris"]["worldlabs_asset_materialization_manifest_uri"].endswith(
        "/worldlabs_assets/materialized_assets_manifest.json"
    )
    assert result["artifact_uris"]["marble_simready_bridge_uri"].endswith(
        "/marble_sim_assets/marble_simready_bridge.json"
    )
    assert (pipeline_dir / "marble_sim_assets" / "marble_asset_manifest.json").is_file()
    assert (pipeline_dir / "marble_sim_assets" / "simulators" / "pybullet_review_manifest.json").is_file()


def test_marble_helper_normalization_edges(tmp_path: Path) -> None:
    assert msa._string_list(None) == []
    assert msa._string_list("one") == ["one"]
    assert msa._string_list(["one", "one", "", "two"]) == ["one", "two"]
    assert msa._string_list(123) == ["123"]
    assert msa._maybe_float("") is None
    assert msa._maybe_float("not-a-number") is None
    assert msa._deterministic_timestamp({}) is None

    assert msa._collect_urls({"a": {"uri": "gs://bucket/a"}, "b": {"asset_url": "https://b"}, "c": 3}, default_key="asset") == {
        "a": "gs://bucket/a",
        "b": "https://b",
    }
    assert msa._collect_urls(
        ["https://cdn/a", {"name": "full", "path": "/tmp/full.ply"}, 3],
        default_key="ply",
    ) == {"ply_0": "https://cdn/a", "full": "/tmp/full.ply"}
    assert msa._first_value({"empty": "", "remote": "https://cdn/a"}, remote=False) == ""
    assert msa._first_value({"local": "/tmp/a", "remote": "https://cdn/a"}, remote=True) == "https://cdn/a"
    assert msa._first_value({"remote": "https://cdn/a", "local": "/tmp/a"}, remote=False) == "/tmp/a"

    pipeline_dir = tmp_path / "pipeline"
    marble_dir = pipeline_dir / "marble_sim_assets"
    local_collider = tmp_path / "collider.glb"
    local_collider.write_bytes(b"glb")
    _write_json(
        pipeline_dir / "worldlabs_export_manifest.json",
        {"output_collider_mesh_path": str(local_collider)},
    )
    mesh = msa._normalize_mesh_assets(world_manifest={"assets": {"mesh": {}}}, pipeline_dir=pipeline_dir, marble_dir=marble_dir)
    assert mesh["local_collider_mesh_glb_path"] == str(local_collider)
    assert mesh["collider_mesh_glb_url"] == str(local_collider)


def test_marble_validation_warning_and_blocker_edges() -> None:
    blocked = msa._validation_payload({"world": {}, "assets": {}, "semantics": {}, "request_input_lineage": {}})
    assert blocked["overall_status"] == "blocked"
    assert "missing_world_id" in blocked["blockers"]
    assert "missing_splat_assets" in blocked["blockers"]
    assert "missing_collider_mesh_glb" in blocked["blockers"]
    assert "missing_metric_scale_factor" in blocked["blockers"]
    assert "missing_ground_plane_offset" in blocked["blockers"]
    assert "missing_world_marble_url" in blocked["warnings"]
    assert "privacy_safe_input_not_proven_in_request_manifest" in blocked["warnings"]
    assert "missing_worldlabs_input_audit_uri" in blocked["warnings"]
    assert "missing_input_checksums" in blocked["warnings"]

    warning_only = msa._validation_payload(
        {
            "world": {"world_id": "world-1", "world_marble_url": "https://marble"},
            "assets": {
                "splats": {"ply_urls": {"full": "https://cdn/full.ply"}},
                "mesh": {"collider_mesh_glb_url": "https://cdn/collider.glb"},
            },
            "semantics": {"metric_scale_factor": 1.0, "ground_plane_offset": 0.0},
            "request_input_lineage": {},
        }
    )
    assert warning_only["overall_status"] == "review_ready_with_warnings"


def test_marble_manifest_override_and_main_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = _build_capture_root(tmp_path)
    with pytest.raises(msa.PipelineError, match="World manifest override does not exist"):
        build_marble_sim_assets(capture_root=capture_root, world_manifest=tmp_path / "missing.json")

    monkeypatch.setattr(
        msa,
        "build_marble_sim_assets",
        lambda **_: {
            "manifest_path": str(tmp_path / "asset.json"),
            "validation_path": str(tmp_path / "validation.json"),
            "bridge_path": str(tmp_path / "bridge.json"),
            "status": "review_ready",
        },
    )
    assert msa.main(["--capture-root", str(capture_root), "--world-manifest", str(tmp_path / "world.json")]) == 0
    assert "status=review_ready" in capsys.readouterr().out

    def raise_pipeline_error(**_kwargs: object) -> dict[str, object]:
        raise msa.PipelineError("bad world")

    monkeypatch.setattr(msa, "build_marble_sim_assets", raise_pipeline_error)
    assert msa.main(["--capture-root", str(capture_root)]) == 1
    assert "[marble-sim-assets] FAILED: bad world" in capsys.readouterr().out
