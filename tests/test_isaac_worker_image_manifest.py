from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.isaac_worker_image_manifest import summarize_manifest


def test_split_image_manifest_is_digest_pinned_and_sizes_drive_startup_timeout() -> None:
    result = summarize_manifest(
        image_ref="docker.io/example/isaac:split",
        resolved_digest="sha256:" + "a" * 64,
        raw_manifest={
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {"digest": "sha256:1", "size": 2_400_000_000},
                {"digest": "sha256:2", "size": 2_100_000_000},
                {"digest": "sha256:3", "size": 1_800_000_000},
                {"digest": "sha256:4", "size": 1_700_000_000},
                {"digest": "sha256:5", "size": 1_600_000_000},
            ],
        },
    )

    assert result["status"] == "completed"
    assert result["resolved_digest_ref"] == "docker.io/example/isaac@sha256:" + "a" * 64
    assert result["total_compressed_size_bytes"] == 9_600_000_000
    assert result["largest_layer_size_bytes"] == 2_400_000_000
    assert result["split_layer_layout_suitable"] is True
    assert result["large_image_pull_risk"] is True
    assert result["recommended_startup_no_runtime_timeout_seconds"] == 1320
    assert result["raw_secret_values_recorded"] is False


def test_missing_registry_digest_fails_closed() -> None:
    result = summarize_manifest(
        image_ref="docker.io/example/isaac:split",
        resolved_digest=None,
        raw_manifest={"layers": [{"digest": "sha256:1", "size": 1_000}]},
    )

    assert result["status"] == "blocked"
    assert result["resolved_digest_ref"] is None
    assert result["blockers"] == ["registry_manifest_metadata_incomplete"]


def test_isaac_build_script_keeps_digest_pinned_base_default() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_push_isaac_worker_image.sh"
    ).read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/isaac-sim:6.0.0@sha256:" in script
