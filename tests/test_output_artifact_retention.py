from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts import manage_output_artifact_retention as retention


def _write(path: Path, content: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _touch_age(path: Path, *, age_days: int, now: float) -> None:
    timestamp = now - age_days * 86400
    os.utime(path, (timestamp, timestamp))
    if path.is_dir():
        for item in path.rglob("*"):
            os.utime(item, (timestamp, timestamp))


def test_output_retention_selects_latest_canonical_and_dry_run_candidates(tmp_path: Path) -> None:
    now = time.time()
    output = tmp_path / "output"
    latest = _write(output / "launch_audit_live_pipeline_setup_20260707.json")
    old = _write(output / "launch_audit_live_pipeline_setup_20260706.json")
    stale_run = output / "provider_reliability_runpod_old"
    _write(stale_run / "bundle.zip", "zip")
    asset_cache = output / "external_assets" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
    _write(asset_cache, "<mujoco />")
    for path, age in ((latest, 1), (old, 2), (stale_run, 45), (asset_cache, 400)):
        _touch_age(path, age_days=age, now=now)

    manifest, candidates = retention.build_manifest(
        output_root=output,
        dry_run=True,
        now=now,
    )

    assert manifest["schema_version"] == retention.SCHEMA_VERSION
    assert manifest["dry_run"] is True
    assert manifest["canonical_artifacts"] == [
        {
            "path": "launch_audit_live_pipeline_setup_20260707.json",
            "kind": "file",
            "size_bytes": 2,
            "mtime_epoch": latest.stat().st_mtime,
            "age_days": 1.0,
            "retention_class": "canonical_launch_evidence",
            "canonical_key": "launch_audit_live_pipeline_setup",
            "canonical": True,
        }
    ]
    assert manifest["superseded_canonical_artifacts"][0]["path"] == (
        "launch_audit_live_pipeline_setup_20260706.json"
    )
    assert [candidate.relative_path for candidate in candidates] == [
        "provider_reliability_runpod_old"
    ]
    assert stale_run.exists()
    assert "external_assets" not in {candidate.relative_path for candidate in candidates}


def test_output_retention_execute_requires_ack_and_deletes_candidates(tmp_path: Path) -> None:
    now = time.time()
    output = tmp_path / "output"
    stale_run = output / "provider_reliability_runpod_old"
    _write(stale_run / "bundle.zip", "zip")
    _touch_age(stale_run, age_days=45, now=now)
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(SystemExit, match="--execute requires"):
        retention.main([
            "--output-root",
            str(output),
            "--manifest-path",
            str(manifest_path),
            "--execute",
        ])

    assert stale_run.exists()

    assert retention.main(
        [
            "--output-root",
            str(output),
            "--manifest-path",
            str(manifest_path),
            "--execute",
            "--acknowledge-delete-output-artifacts",
            retention.EXECUTE_ACK,
        ]
    ) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["prune_actions"] == [
        {"path": "provider_reliability_runpod_old", "status": "deleted"}
    ]
    assert not stale_run.exists()
