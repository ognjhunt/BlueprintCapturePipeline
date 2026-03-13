from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.local_bundle_workflow import detect_bundle_identity


def test_detect_bundle_identity_reads_raw_bundle_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payloads = {
        "manifest.json": {"scene_id": "scene-1", "capture_id": "capture-1"},
        "capture_context.json": {"sceneId": "scene-1"},
        "capture_upload_complete.json": {"captureId": "capture-1"},
    }
    for name, payload in payloads.items():
        (raw_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    identity = detect_bundle_identity(tmp_path)
    assert identity.scene_id == "scene-1"
    assert identity.capture_id == "capture-1"
