from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.site_reference_backfill import main, run_site_reference_backfill


def _capture_root(root: Path, *, capture_id: str = "capture-1") -> Path:
    capture_root = root / "bucket" / "scenes" / "scene-1" / "captures" / capture_id
    capture_root.mkdir(parents=True)
    return capture_root


def test_site_reference_backfill_dry_run_writes_review_packet_for_missing_site_id(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "world_model_candidate": True,
                "quality": {"world_model_candidate": True},
                "metadata": {"site_identity": None},
            }
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "backfill_report.json"
    result = run_site_reference_backfill(storage_roots=[tmp_path], report_path=report_path, dry_run=True)

    assert result["summary"]["review_required"] == 1
    assert report_path.is_file()
    packet_path = Path(result["review_packet_path"])
    assert packet_path.is_file()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["captures"][0]["blockers"] == ["missing_site_id"]
    assert packet["captures"][0]["requested_fields"] == ["site_identity.site_id", "site_identity.site_id_source"]


def test_site_reference_backfill_execute_reuses_retrieval_stage(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, capture_id="capture-eligible")
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-eligible",
                "world_model_candidate": True,
                "quality": {"world_model_candidate": True},
                "metadata": {"site_identity": {"site_id": "site-1", "site_id_source": "fixture"}},
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_stage(*, capture_root: str | Path, force_rebuild: bool = False, embedding_model=None):  # noqa: ANN001
        calls.append(Path(capture_root))
        site_root = tmp_path / "bucket" / "sites" / "site-1" / "reference_memory"
        site_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "site_reference_manifest.json",
            "site_reference_index.jsonl",
            "site_reference_summary_projection.json",
            "retrieval_validation.json",
        ):
            (site_root / name).write_text("{}\n", encoding="utf-8")
        return {"status": "completed", "site_id": "site-1", "site_reference_index": str(site_root / "site_reference_index.jsonl")}

    monkeypatch.setattr("blueprint_pipeline.site_reference_backfill.run_retrieval_index_stage", fake_stage)

    result = run_site_reference_backfill(storage_roots=[tmp_path], dry_run=False)

    assert calls == [capture_root]
    assert result["summary"]["indexed"] == 1
    assert result["captures"][0]["status"] == "indexed"


def test_site_reference_backfill_execute_records_stage_failure(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, capture_id="capture-stage-fails")
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-stage-fails",
                "world_model_candidate": True,
                "site_id": "site-1",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "blueprint_pipeline.site_reference_backfill.run_retrieval_index_stage",
        lambda **_kwargs: {"status": "blocked", "reason": "missing_embeddings"},
    )

    result = run_site_reference_backfill(storage_roots=[tmp_path], dry_run=False)

    assert result["summary"]["skipped"] == 1
    assert result["captures"][0]["blockers"] == ["missing_embeddings"]


def test_site_reference_backfill_reports_geometry_required_for_meta_capture(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, capture_id="capture-meta")
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    (raw_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v3",
                "scene_id": "scene-1",
                "capture_id": "capture-meta",
                "capture_source": "meta_glasses",
                "capture_profile_id": "glasses_pov",
                "video_uri": "walkthrough.mov",
                "site_identity": {"site_id": "site-1", "site_id_source": "fixture"},
                "capture_rights": {
                    "derived_scene_generation_allowed": True,
                    "consent_status": "documented",
                    "permission_document_uri": "gs://local-blueprint/rights/consent-packet.pdf",
                },
                "capture_mode": {
                    "requested_mode": "site_world_candidate",
                    "resolved_mode": "site_world_candidate",
                },
            }
        ),
        encoding="utf-8",
    )
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-meta",
                "capture_source": "glasses",
                "source_device": "meta_glasses",
                "raw_video_uri": "gs://bucket/scenes/scene-1/captures/capture-meta/raw/walkthrough.mov",
                "world_model_candidate": False,
                "quality": {"world_model_candidate": False},
                "metadata": {
                    "site_identity": {"site_id": "site-1", "site_id_source": "fixture"},
                    "capture_mode": {
                        "requested_mode": "site_world_candidate",
                        "resolved_mode": "site_world_candidate",
                    },
                    "capture_rights": {"derived_scene_generation_allowed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_site_reference_backfill(storage_roots=[tmp_path], dry_run=True)

    entry = result["captures"][0]
    assert entry["status"] == "geometry_required"
    assert entry["blockers"] == ["non_arkit_geometry_missing"]
    assert entry["expected_geometry_summary_path"].endswith(
        "pipeline/geometry/geometry_summary.json"
    )
    assert "scripts/run_geometry_lane.py" in entry["local_geometry_command"]
    assert entry["provider_blocker"]["required_env"] == ["VIDEO_TO_WORLD_URL", "VIDEO_TO_WORLD_RUNNER_TOKEN"]


def test_site_reference_backfill_handles_discovery_and_candidate_edges(tmp_path: Path) -> None:
    missing = run_site_reference_backfill(storage_roots=[tmp_path / "missing"], dry_run=True)
    assert missing["summary"]["discovered"] == 0
    assert missing["review_packet_path"] is None

    skipped_root = _capture_root(tmp_path, capture_id="capture-not-candidate")
    (skipped_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": "site-1",
                "scene_id": "scene-1",
                "capture_id": "capture-not-candidate",
                "world_model_candidate": False,
            }
        ),
        encoding="utf-8",
    )
    no_scene_root = tmp_path / "loose" / "captures" / "capture-no-scene"
    no_scene_root.mkdir(parents=True)
    (no_scene_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": "site-1",
                "capture_id": "capture-no-scene",
                "world_model_candidate": True,
            }
        ),
        encoding="utf-8",
    )
    marker_at_end_root = tmp_path / "lonely" / "scenes"
    marker_at_end_root.mkdir(parents=True)
    (marker_at_end_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": "site-1",
                "capture_id": "capture-marker-at-end",
                "world_model_candidate": True,
            }
        ),
        encoding="utf-8",
    )
    scene_from_path_root = tmp_path / "pathcase" / "scenes" / "scene-from-path" / "captures" / "capture-path"
    scene_from_path_root.mkdir(parents=True)
    (scene_from_path_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": "site-1",
                "capture_id": "capture-path",
                "world_model_candidate": True,
            }
        ),
        encoding="utf-8",
    )

    result = run_site_reference_backfill(storage_roots=[tmp_path], dry_run=True)
    by_capture = {entry["capture_id"]: entry for entry in result["captures"]}

    assert by_capture["capture-not-candidate"]["status"] == "skipped"
    assert by_capture["capture-not-candidate"]["blockers"] == ["world_model_candidate=false"]
    assert by_capture["capture-no-scene"]["scene_id"] == ""
    assert by_capture["capture-marker-at-end"]["scene_id"] == ""
    assert by_capture["capture-path"]["scene_id"] == "scene-from-path"


def test_site_reference_backfill_merges_raw_sidecars_and_ignores_invalid_json(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, capture_id="capture-sidecar")
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-sidecar",
                "world_model_candidate": True,
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "manifest.json").write_text("{}", encoding="utf-8")
    (raw_root / "site_identity.json").write_text(
        json.dumps({"site_id": "site-from-sidecar", "site_id_source": "fixture"}),
        encoding="utf-8",
    )
    (raw_root / "capture_mode.json").write_text("{bad-json", encoding="utf-8")

    result = run_site_reference_backfill(storage_roots=[tmp_path], dry_run=True)

    assert result["captures"][0]["status"] == "eligible"
    assert result["captures"][0]["site_id"] == "site-from-sidecar"


def _write_meta_capture(
    capture_root: Path,
    *,
    rights_allowed: bool,
    raw_video: bool,
    geometry_summary: dict[str, object] | None = None,
) -> None:
    capture_root.mkdir(parents=True, exist_ok=True)
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    if raw_video:
        (raw_root / "walkthrough.mov").write_bytes(b"video")
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_id": "site-1",
                "capture_source": "meta_glasses",
                "capture_id": capture_root.name,
                "world_model_candidate": True,
                "metadata": {
                    "capture_rights": {
                        "derived_scene_generation_allowed": rights_allowed,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    if geometry_summary is not None:
        geometry_root = capture_root / "pipeline" / "geometry"
        geometry_root.mkdir(parents=True)
        (geometry_root / "geometry_summary.json").write_text(
            json.dumps(geometry_summary),
            encoding="utf-8",
        )


def test_site_reference_backfill_geometry_gate_accepts_ready_provider_and_local_outputs(
    tmp_path: Path,
) -> None:
    no_rights = _capture_root(tmp_path, capture_id="capture-no-rights")
    _write_meta_capture(no_rights, rights_allowed=False, raw_video=True)
    no_video = _capture_root(tmp_path, capture_id="capture-no-video")
    _write_meta_capture(no_video, rights_allowed=True, raw_video=False)
    provider_ready = _capture_root(tmp_path, capture_id="capture-provider-ready")
    _write_meta_capture(
        provider_ready,
        rights_allowed=True,
        raw_video=True,
        geometry_summary={"geometry_live_ready": True, "geometry_source": "video_to_world"},
    )
    local_ready = _capture_root(tmp_path, capture_id="capture-local-ready")
    _write_meta_capture(
        local_ready,
        rights_allowed=True,
        raw_video=True,
        geometry_summary={
            "geometry_source": "local_sfm",
            "contract_ready_for_world_model": True,
        },
    )
    still_missing = _capture_root(tmp_path, capture_id="capture-still-missing")
    _write_meta_capture(
        still_missing,
        rights_allowed=True,
        raw_video=True,
        geometry_summary={"geometry_source": "preview_only"},
    )

    result = run_site_reference_backfill(storage_roots=[tmp_path], dry_run=True)
    by_capture = {entry["capture_id"]: entry for entry in result["captures"]}

    assert by_capture["capture-no-rights"]["status"] == "eligible"
    assert by_capture["capture-no-video"]["status"] == "eligible"
    assert by_capture["capture-provider-ready"]["status"] == "eligible"
    assert by_capture["capture-local-ready"]["status"] == "eligible"
    assert by_capture["capture-still-missing"]["status"] == "geometry_required"


def test_site_reference_backfill_main_and_module_entrypoint(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    report_path = tmp_path / "report.json"
    assert main([str(tmp_path), "--report-path", str(report_path), "--execute", "--force-rebuild"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == "site_reference_backfill.v1"
    assert report_path.is_file()

    monkeypatch.setattr(sys, "argv", ["site_reference_backfill.py", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("blueprint_pipeline.site_reference_backfill", run_name="__main__")
    assert exc_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "site_reference_backfill.v1"
