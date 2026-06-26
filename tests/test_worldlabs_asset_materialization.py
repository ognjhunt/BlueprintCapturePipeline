from __future__ import annotations

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

from blueprint_pipeline.marble_sim_assets import build_marble_sim_assets
from blueprint_pipeline import worldlabs_asset_materialization as w
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
    assert export_manifest["claim_boundary"]["rank_fidelity_result_proven"] is False

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


def test_worldlabs_asset_materialization_helpers_and_download_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert w._mapping({"a": 1}) == {"a": 1}
    assert w._mapping([]) == {}
    assert w._string(None) == ""
    assert w._read_optional_mapping(tmp_path / "missing.json") == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert w._read_optional_mapping(list_payload) == {}
    nested = tmp_path / "a" / "b.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("bytes", encoding="utf-8")
    assert w._relative_to(tmp_path, nested) == "a/b.txt"
    assert w._sha_file(nested) == sha256(b"bytes").hexdigest()
    assert w._safe_token(" .. bad token !! ") == "bad_token"
    assert w._safe_token("!!!") == "asset"
    assert w._suffix_from_url("https://x/model.glb?sig=1#frag", ".bin") == ".glb"
    assert w._suffix_from_url("https://x/model", ".bin") == ".bin"
    assert w._world_id({"id": "world-id"}) == "world-id"
    assert w._world_id({}) is None

    assert w._collect_url_map("https://x/a.glb", default_key="asset") == {"asset": "https://x/a.glb"}
    assert w._collect_url_map(
        {"High Quality": {"asset_url": "https://x/hq.glb"}, "empty": "", "bad": 3},
        default_key="asset",
    ) == {"High_Quality": "https://x/hq.glb"}
    assert w._collect_url_map(
        [{"name": "full splat", "url": "https://x/full.spz"}, "https://x/fallback.spz", 3],
        default_key="spz",
    ) == {"full_splat": "https://x/full.spz", "spz_1": "https://x/fallback.spz"}
    assert w._collect_url_map(3, default_key="asset") == {}

    candidates, skipped = w._candidate_assets(
        {
            "assets": {
                "mesh": {
                    "collider_mesh_glb_url": "https://x/collider",
                    "high_quality_mesh_glb_urls": {"display": "https://x/display.glb"},
                },
                "splats": {
                    "spz_urls": [{"name": "full", "url": "https://x/full.spz"}],
                    "ply_url": "https://x/full.ply",
                    "usd_url": "https://x/full.usd",
                },
            }
        },
        include_visual_assets=False,
    )
    assert [candidate["kind"] for candidate in candidates] == ["collider_mesh_glb"]
    assert {candidate["kind"] for candidate in skipped} == {
        "high_quality_mesh_glb",
        "splat_spz",
        "splat_ply",
        "scene_usd",
    }
    candidates, skipped = w._candidate_assets(
        {
            "assets": {
                "collider_mesh_url": "https://x/collider.glb",
                "spz_urls": "https://x/full.spz",
                "ply_urls": {"raw": "https://x/raw.ply"},
                "mesh": {"hq_mesh_url": "https://x/hq.glb"},
                "splats": {"usd_urls": {"scene": {"uri": "https://x/scene.usd"}}},
            }
        },
        include_visual_assets=True,
    )
    assert len(candidates) == 5
    assert skipped == []

    class FakeResponse:
        def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None) -> None:
            self.chunks = chunks
            self.headers = headers or {}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return self.chunks.pop(0) if self.chunks else b""

    monkeypatch.setattr(
        w._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse([b"ab", b"cd"], {"Content-Length": "4"}),
    )
    proof = w._download_remote_asset("https://x/a.glb", tmp_path / "downloads" / "a.glb", max_bytes=10)
    assert proof["size_bytes"] == 4
    assert proof["sha256"] == sha256(b"abcd").hexdigest()
    monkeypatch.setattr(
        w._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse([b"ok"], {"Content-Length": "not-a-number"}),
    )
    assert w._download_remote_asset("https://x/invalid-length.glb", tmp_path / "downloads" / "invalid.glb", max_bytes=10)[
        "size_bytes"
    ] == 2

    monkeypatch.setattr(
        w._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse([b"too-large"], {"Content-Length": "99"}),
    )
    too_large = tmp_path / "downloads" / "too-large.glb"
    try:
        w._download_remote_asset("https://x/b.glb", too_large, max_bytes=1)
    except RuntimeError as exc:
        assert str(exc) == "remote_asset_exceeds_max_bytes"
    assert not too_large.exists()
    monkeypatch.setattr(
        w._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse([b"stream-too-large"], {}),
    )
    stream_too_large = tmp_path / "downloads" / "stream-too-large.glb"
    try:
        w._download_remote_asset("https://x/stream.glb", stream_too_large, max_bytes=1)
    except RuntimeError as exc:
        assert str(exc) == "remote_asset_exceeds_max_bytes"
    assert not stream_too_large.exists()
    monkeypatch.setattr(
        w._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("network down")),
    )
    try:
        w._download_remote_asset("https://x/c.glb", tmp_path / "downloads" / "fail.glb", max_bytes=None)
    except OSError as exc:
        assert "network down" in str(exc)


def test_worldlabs_asset_materialization_statuses_and_cli(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    try:
        materialize_worldlabs_assets(capture_root=capture_root)
    except Exception as exc:
        assert "World Labs world manifest" in str(exc)

    _write_json(capture_root / "pipeline" / "worldlabs_world_manifest.json", {"world_id": "world-empty"})
    no_assets = materialize_worldlabs_assets(capture_root=capture_root)
    assert no_assets["status"] == "blocked_no_materializable_assets"

    _write_json(
        capture_root / "pipeline" / "worldlabs_world_manifest.json",
        {
            "id": "world-partial",
            "assets": {
                "mesh": {
                    "collider_mesh_url": "file:///local.glb",
                    "high_quality_mesh_url": "https://x/hq.glb",
                }
            },
        },
    )

    def fake_download(url: str, output_path: Path, *, max_bytes: int | None) -> dict[str, object]:
        del max_bytes
        if url.endswith("fail.glb"):
            raise RuntimeError("download_failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(url.encode("utf-8"))
        return {"size_bytes": output_path.stat().st_size, "sha256": w._sha_file(output_path), "content_type": None}

    monkeypatch.setattr(w, "_download_remote_asset", fake_download)
    partial = materialize_worldlabs_assets(capture_root=capture_root, include_visual_assets=True)
    assert partial["status"] == "complete_with_download_failures"
    manifest = json.loads(
        (capture_root / "pipeline" / "worldlabs_assets" / "materialized_assets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["failures"][0]["reason"] == "unsupported_or_missing_remote_url"
    assert manifest["world_id"] == "world-partial"

    _write_json(
        capture_root / "pipeline" / "worldlabs_world_manifest.json",
        {"world_id": "world-fail", "assets": {"mesh": {"collider_mesh_url": "https://x/fail.glb"}}},
    )
    blocked = materialize_worldlabs_assets(capture_root=capture_root)
    assert blocked["status"] == "blocked"

    _write_json(
        capture_root / "pipeline" / "worldlabs_world_manifest.json",
        {"world_id": "world-cli", "assets": {"mesh": {"collider_mesh_url": "https://x/cli.glb"}}},
    )
    assert w.main(["--capture-root", str(capture_root), "--include-visual-assets", "--max-asset-bytes", "0"]) == 0
    assert "status=complete" in capsys.readouterr().out
    assert w.main(["--capture-root", str(tmp_path / "missing-capture")]) == 1
    assert "FAILED" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["worldlabs_asset_materialization", "--capture-root", str(capture_root)])
    try:
        runpy.run_module("blueprint_pipeline.worldlabs_asset_materialization", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
