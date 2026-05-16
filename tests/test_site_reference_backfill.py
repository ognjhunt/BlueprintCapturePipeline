from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.site_reference_backfill import run_site_reference_backfill


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
