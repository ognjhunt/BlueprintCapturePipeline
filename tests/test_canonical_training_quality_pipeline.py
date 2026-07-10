from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.action_normalization import (
    SC3_ACTION_ORDER,
    SC3_ACTION_REPRESENTATION,
    SC3_ACTION_UNITS,
)
from blueprint_pipeline.canonical_training_quality_pipeline import (
    run_canonical_training_quality_pipeline,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_pgm(path: Path, offset: int) -> None:
    rows, cols = np.indices((32, 32))
    image = (((rows + cols + offset) % 2) * 160 + 40).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P5\n32 32\n255\n" + image.tobytes(order="C"))


def _configure_signing_key(monkeypatch: Any, tmp_path: Path) -> None:
    key_path = tmp_path / "canonical-signing-key.pem"
    key_path.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    monkeypatch.setenv("BLUEPRINT_PTDP_SIGNING_KEY_FILE", str(key_path))
    monkeypatch.setenv("BLUEPRINT_PTDP_SIGNING_KEY_ID", "test-canonical-key")


class _ProductionEmbeddingProvider:
    name = "dinov3"
    version = "test-1"
    model_id = "test/dinov3"
    revision = "a" * 40
    production_ready = True

    def embed_image(self, gray: np.ndarray) -> np.ndarray:
        image = np.asarray(gray, dtype=np.float64)
        vector = image.reshape(-1)[::16]
        return vector - vector.mean()


class _ProductionCaptionProvider:
    name = "grounded-caption-test"
    version = "test-1"
    model_id = "test/grounded-caption"
    revision = "b" * 40
    production_ready = True

    def caption_clip(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        sampled_hash = request["sampled_frames"][0]["sha256"]
        return {
            "schema_version": "blueprint.grounded_clip_caption.v1",
            "clip_id": request["clip_id"],
            "caption": "Robot observes object one while executing task one.",
            "task_id": request["task_id"],
            "object_ids": ["object-1"],
            "claims": [
                {
                    "claim_type": "visible_object",
                    "text": "Object one is visible.",
                    "object_ids": ["object-1"],
                    "evidence_frame_sha256": [sampled_hash],
                }
            ],
        }


def _seed_bundle(root: Path) -> Path:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    frames: list[dict[str, Any]] = []
    for index in range(70):
        frame_path = root / "frames" / f"frame-{index:03d}.pgm"
        _write_pgm(frame_path, index)
        frames.append(
            {
                "frame_id": f"frame-{index:03d}",
                "image_path": frame_path.relative_to(root).as_posix(),
                "T_world_camera": identity,
            }
        )
    _write_json(
        root / "clips_manifest.json",
        {
            "clip_count": 1,
            "clips": [
                {
                    "clip_id": "clip-1",
                    "attempt_id": "attempt-1",
                    "task_id": "task-1",
                    "object_ids": ["object-1"],
                    "clip_kind": "robot_pov",
                    "frames": frames,
                }
            ],
        },
    )
    action_rows = [
        [
            0.01 * (index + 1),
            0.005 * index,
            -0.004 * index,
            0.01 * index,
            -0.008 * index,
            0.006 * index,
            -0.5 + 0.2 * index,
        ]
        for index in range(5)
    ]
    trace_path = root / "normalized_attempt_trace.json"
    _write_json(
        trace_path,
        {
            "schema_version": "blueprint.normalized_attempt_trace.v1",
            "control_rate_hz": 10.0,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "actions": [
                        {"delta_end_effector_pose_7d": row}
                        for row in action_rows
                    ],
                    "chunk_start_times_sec": [0.0, 0.1, 0.2, 0.3, 0.4],
                    "frame_times_sec": [0.0, 0.1, 0.2, 0.3, 0.4],
                    "control_rate_hz": 10.0,
                }
            ],
        },
    )
    return trace_path


def _action_space() -> dict[str, Any]:
    return {
        "dim": 7,
        "representation": SC3_ACTION_REPRESENTATION,
        "order": list(SC3_ACTION_ORDER),
        "units": list(SC3_ACTION_UNITS),
    }


def test_canonical_quality_pipeline_runs_all_stages_and_signs_chain(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure_signing_key(monkeypatch, tmp_path)
    bundle = tmp_path / "job"
    trace_path = _seed_bundle(bundle)

    manifest = run_canonical_training_quality_pipeline(
        bundle_dir=bundle,
        action_trace_path=trace_path,
        action_space=_action_space(),
        embedding_provider=_ProductionEmbeddingProvider(),
        caption_provider=_ProductionCaptionProvider(),
    )

    assert manifest["status"] == "passed", manifest["blockers"]
    assert manifest["accepted_clip_ids"] == ["clip-1"]
    assert manifest["accepted_attempt_ids"] == ["attempt-1"]
    assert manifest["chain_signature"]["verified_at_write"] is True
    assert list(manifest["stage_artifacts"]) == [
        "clip_curation",
        "semantic_dedup",
        "grounded_clip_caption",
        "action_normalization",
    ]
    assert all(
        artifact["sha256"]
        for artifact in manifest["stage_artifacts"].values()
    )
    assert manifest["claim_boundary"]["rejected_clips_exported"] is False


def test_canonical_quality_pipeline_missing_providers_fails_closed(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _configure_signing_key(monkeypatch, tmp_path)
    bundle = tmp_path / "job"
    trace_path = _seed_bundle(bundle)

    manifest = run_canonical_training_quality_pipeline(
        bundle_dir=bundle,
        action_trace_path=trace_path,
        action_space=_action_space(),
        provider_config={},
    )

    assert manifest["status"] == "blocked"
    assert manifest["accepted_clip_ids"] == []
    assert manifest["accepted_attempt_ids"] == []
    assert manifest["claim_boundary"]["premium_quality_eligible"] is False
    assert manifest["claim_boundary"]["native_training_export_eligible"] is False
    assert any(
        blocker.startswith("canonical_embedding_provider_config_invalid:")
        for blocker in manifest["blockers"]
    )
    assert any(
        blocker.startswith("canonical_caption_provider_config_invalid:")
        for blocker in manifest["blockers"]
    )
