from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from blueprint_pipeline.grounded_clip_caption_stage import (
    CAPTION_SCHEMA_VERSION,
    run_grounded_clip_caption_stage,
)


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "bundle"
    frames = bundle / "frames"
    frames.mkdir(parents=True)
    image_paths: list[str] = []
    for index in range(3):
        path = frames / f"{index}.npy"
        np.save(path, np.full((32, 32), index + 1, dtype=np.float32))
        image_paths.append(path.relative_to(bundle).as_posix())
    clips = {
        "clips": [
            {
                "clip_id": "clip-1",
                "task_id": "task-1",
                "object_ids": ["object-1"],
                "frames": [{"image_path": path} for path in image_paths],
            }
        ]
    }
    (bundle / "clips_manifest.json").write_text(json.dumps(clips), encoding="utf-8")
    curation = bundle / "curation.json"
    curation.write_text(
        json.dumps(
            {
                "schema_version": "clip_curation_manifest.v1",
                "accepted_clip_ids": ["clip-1"],
            }
        ),
        encoding="utf-8",
    )
    dedup = bundle / "dedup.json"
    dedup.write_text(
        json.dumps(
            {
                "schema_version": "semantic_dedup_manifest.v2",
                "production_status": "passed",
                "production_accepted_clip_ids": ["clip-1"],
            }
        ),
        encoding="utf-8",
    )
    return bundle, curation, dedup


class _Provider:
    name = "grounded-captioner"
    version = "1.0"
    model_id = "blueprint/grounded-captioner"
    revision = "a" * 40
    production_ready = True

    def __init__(self, *, object_id: str = "object-1", spatial: bool = False) -> None:
        self.object_id = object_id
        self.spatial = spatial
        self.calls = 0

    def caption_clip(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        frame_hash = request["sampled_frames"][0]["sha256"]
        claim: dict[str, Any] = {
            "claim_type": "spatial_relation" if self.spatial else "visible_object",
            "text": "The grounded object is visible.",
            "object_ids": [self.object_id],
            "evidence_frame_sha256": [frame_hash],
        }
        return {
            "schema_version": CAPTION_SCHEMA_VERSION,
            "clip_id": request["clip_id"],
            "caption": "Object one is visible during the recorded task.",
            "task_id": request["task_id"],
            "object_ids": [self.object_id],
            "claims": [claim],
        }


def test_grounded_caption_stage_passes_strict_grounded_response(tmp_path: Path) -> None:
    bundle, curation, dedup = _bundle(tmp_path)
    result = run_grounded_clip_caption_stage(
        bundle_dir=bundle,
        curation_manifest_path=curation,
        dedup_manifest_path=dedup,
        provider=_Provider(),
    )

    assert result["status"] == "passed"
    assert result["accepted_clip_ids"] == ["clip-1"]
    assert len(result["captions"][0]["sampled_frames"]) == 3
    assert result["captions"][0]["provider"]["revision"] == "a" * 40


def test_grounded_caption_stage_rejects_unregistered_object_id(tmp_path: Path) -> None:
    bundle, curation, dedup = _bundle(tmp_path)
    result = run_grounded_clip_caption_stage(
        bundle_dir=bundle,
        curation_manifest_path=curation,
        dedup_manifest_path=dedup,
        provider=_Provider(object_id="invented-object"),
    )

    assert result["status"] == "blocked"
    blockers = result["excluded_clips"][0]["blockers"]
    assert "caption_object_id_not_grounded" in blockers
    assert "caption_claim_object_id_not_grounded" in blockers


def test_grounded_caption_stage_rejects_unsupported_spatial_claim(tmp_path: Path) -> None:
    bundle, curation, dedup = _bundle(tmp_path)
    result = run_grounded_clip_caption_stage(
        bundle_dir=bundle,
        curation_manifest_path=curation,
        dedup_manifest_path=dedup,
        provider=_Provider(spatial=True),
    )

    assert result["status"] == "blocked"
    assert "caption_spatial_claim_missing_geometry_evidence" in result[
        "excluded_clips"
    ][0]["blockers"]


def test_grounded_caption_stage_requires_approved_provider_and_canonical_stages(
    tmp_path: Path,
) -> None:
    bundle, curation, dedup = _bundle(tmp_path)
    result = run_grounded_clip_caption_stage(
        bundle_dir=bundle,
        curation_manifest_path=curation,
        dedup_manifest_path=dedup,
        provider=None,
    )

    assert result["status"] == "blocked"
    assert "caption_provider_not_production_approved" in result["blockers"]


def test_grounded_caption_stage_rejects_frame_escape(tmp_path: Path) -> None:
    bundle, curation, dedup = _bundle(tmp_path)
    manifest_path = bundle / "clips_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.npy"
    np.save(outside, np.ones((32, 32), dtype=np.float32))
    manifest["clips"][0]["frames"][0]["image_path"] = str(outside)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_grounded_clip_caption_stage(
        bundle_dir=bundle,
        curation_manifest_path=curation,
        dedup_manifest_path=dedup,
        provider=_Provider(),
    )

    assert result["status"] == "blocked"
    assert "caption_frame_outside_bundle" in result["excluded_clips"][0]["blockers"]


def test_grounded_caption_stage_retries_invalid_schema_once(tmp_path: Path) -> None:
    bundle, curation, dedup = _bundle(tmp_path)

    class RetryProvider(_Provider):
        def caption_clip(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            if self.calls == 0:
                self.calls += 1
                return {"caption": "missing required fields"}
            return super().caption_clip(request)

    provider = RetryProvider()
    result = run_grounded_clip_caption_stage(
        bundle_dir=bundle,
        curation_manifest_path=curation,
        dedup_manifest_path=dedup,
        provider=provider,
        max_attempts=2,
    )

    assert result["status"] == "passed"
    assert result["captions"][0]["attempts"] == 2
