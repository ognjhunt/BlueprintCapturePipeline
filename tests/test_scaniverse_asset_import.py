from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.scaniverse_asset_import import build_scaniverse_asset_import, main


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "scenes" / "scene_scaniverse" / "captures" / "capture_001"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v3",
                "scene_id": "scene_scaniverse",
                "capture_id": "capture_001",
                "capture_source": "iphone",
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_ascii_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "2 0 0",
                "2 2 0.25",
                "0 2 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_scaniverse_import_stages_assets_without_raw_truth_upgrade(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    source_dir = tmp_path / "scaniverse_exports"
    ply = source_dir / "warehouse_splat.ply"
    usdz = source_dir / "warehouse_scene.usdz"
    _write_ascii_ply(ply)
    usdz.write_bytes(b"USDZ placeholder for importer contract test")
    source_manifest = source_dir / "scaniverse_source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "scaniverse_site_id": "site_niantic_123",
                "scaniverse_scan_id": "scan_360_456",
                "capture_method": "scaniverse_web_360_upload",
                "blueprint_assignment_id": "assignment_789",
                "blueprint_scene_id": "scene_scaniverse",
                "blueprint_capture_id": "capture_001",
                "capture_hardware": "Insta360 X5",
                "source_video_filename": "warehouse.insv",
                "export_created_at": "2026-07-06T12:00:00Z",
                "export_performed_by": "capturer@example.test",
                "rights_scope": "pilot_review_only",
                "metric_scale_calibrated": False,
            }
        ),
        encoding="utf-8",
    )

    result = build_scaniverse_asset_import(
        capture_root=capture_root,
        assets=[ply, usdz],
        source_manifest=source_manifest,
    )

    assert result["status"] == "ready_for_review"
    assert result["asset_count"] == 2
    assert result["scene_asset_preflight_ran"] is True

    manifest = _read_json(capture_root / "pipeline" / "scaniverse_assets" / "scaniverse_import_manifest.json")
    proof = _read_json(capture_root / "pipeline" / "scaniverse_assets" / "scaniverse_import_proof_boundary.json")
    preflight = _read_json(capture_root / "pipeline" / "simulation_automation" / "scene_asset_preflight.json")

    assert manifest["source_policy"]["programmatic_scaniverse_export_api_proven"] is False
    assert manifest["source_policy"]["programmatic_360_upload_api_proven"] is False
    assert manifest["source_policy"]["programmatic_asset_generation_api_proven"] is False
    assert manifest["source_policy"]["programmatic_usdz_export_download_api_proven"] is False
    assert manifest["source_policy"]["manual_scaniverse_web_workflow_assumed"] is True
    assert manifest["blueprint_sidecar_manifest"]["blueprint_assignment_id"] == "assignment_789"
    assert manifest["source_manifest"]["scaniverse_scan_id"] == "scan_360_456"
    assert {asset["suffix"] for asset in manifest["assets"]} == {".ply", ".usdz"}
    assert all(asset["external_derived_support_asset"] is True for asset in manifest["assets"])
    assert all(asset["raw_capture_truth"] is False for asset in manifest["assets"])
    assert proof["raw_capture_truth"] is False
    assert proof["isaac_sim_execution_proven"] is False
    assert proof["physics_contact_validated"] is False
    assert proof["programmatic_usdz_export_download_api_proven"] is False
    assert proof["blueprint_capture_stack_replaced"] is False

    by_suffix = {asset["suffix"]: asset for asset in manifest["assets"]}
    assert by_suffix[".ply"]["cpu_scene_preflight_supported"] is True
    assert by_suffix[".usdz"]["cpu_scene_preflight_supported"] is False
    assert by_suffix[".usdz"]["inspection"]["status"] == "accepted_without_cpu_scene_preflight"
    assert manifest["isaac_handoff_candidacy"]["candidate"] is True
    assert manifest["isaac_handoff_candidacy"]["isaac_sim_execution_proven"] is False
    assert manifest["isaac_handoff_candidacy"]["physics_contact_validated"] is False
    assert preflight["status"] == "ready_for_episode_setup"
    assert preflight["claim_boundary"]["simulator_execution_proven"] is False


def test_scaniverse_import_blocks_missing_assets_and_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = _build_capture_root(tmp_path)
    missing = tmp_path / "missing.usdz"

    result = build_scaniverse_asset_import(capture_root=capture_root, assets=[missing])

    assert result["status"] == "blocked"
    assert any(item.startswith("missing_scaniverse_asset:") for item in result["blockers"])
    assert "missing_blueprint_scaniverse_sidecar_manifest" in result["blockers"]

    source_dir = tmp_path / "scaniverse_exports"
    ply = source_dir / "scene.ply"
    _write_ascii_ply(ply)
    sidecar = source_dir / "blueprint_scaniverse_sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "blueprint_scene_id": "scene_scaniverse",
                "blueprint_capture_id": "capture_001",
                "scaniverse_scan_id": "scan_cli",
            }
        ),
        encoding="utf-8",
    )
    assert main(["--capture-root", str(capture_root), "--asset", str(ply), "--blueprint-sidecar", str(sidecar)]) == 0
    assert "status=ready_for_review" in capsys.readouterr().out
    assert main(["--capture-root", str(capture_root), "--asset", str(missing)]) == 1
    assert "status=blocked" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scaniverse_asset_import",
            "--capture-root",
            str(capture_root),
            "--asset",
            str(ply),
            "--source-manifest",
            str(sidecar),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("blueprint_pipeline.scaniverse_asset_import", run_name="__main__")
    assert exc.value.code == 0
