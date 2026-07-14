from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from blueprint_pipeline.isaac_worker_image_manifest import (
    derive_required_cuda_version,
    _inspect_local_image,
    _runnable_child_digest,
    inspect_image_manifest,
    summarize_manifest,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


@pytest.mark.parametrize(
    ("config", "expected", "source"),
    [
        (
            {"config": {"Env": ["CUDA_VERSION=12.6.3"]}},
            "12.6",
            "image_config_env:CUDA_VERSION",
        ),
        (
            {
                "config": {
                    "Env": [
                        "CUDA_VERSION=12.4.1",
                        "BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION=12.6",
                    ]
                }
            },
            "12.6",
            "image_config_env:BLUEPRINT_GROOT_OSCAR_REQUIRED_CUDA_VERSION",
        ),
        ({"config": {"Env": ["CUDA_VERSION=unknown"]}}, None, None),
    ],
)
def test_required_cuda_version_is_derived_from_image_config(
    config: dict, expected: str | None, source: str | None
) -> None:
    assert derive_required_cuda_version(config) == (expected, source)


def test_split_image_manifest_is_digest_pinned_and_sizes_drive_startup_timeout() -> None:
    result = summarize_manifest(
        image_ref="docker.io/example/isaac:split",
        resolved_digest="sha256:" + "a" * 64,
        raw_manifest={
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {"digest": _digest("1"), "size": 2_400_000_000},
                {"digest": _digest("2"), "size": 2_100_000_000},
                {"digest": _digest("3"), "size": 1_800_000_000},
                {"digest": _digest("4"), "size": 1_700_000_000},
                {"digest": _digest("5"), "size": 1_600_000_000},
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
        raw_manifest={"layers": [{"digest": _digest("1"), "size": 1_000}]},
    )

    assert result["status"] == "blocked"
    assert result["resolved_digest_ref"] is None
    assert result["blockers"] == ["registry_manifest_metadata_incomplete"]


def test_oci_index_selects_runnable_child_and_ignores_attestation() -> None:
    runnable = "sha256:" + "a" * 64
    attestation = "sha256:" + "b" * 64

    assert (
        _runnable_child_digest(
            {
                "manifests": [
                    {
                        "digest": runnable,
                        "platform": {"os": "linux", "architecture": "amd64"},
                    },
                    {
                        "digest": attestation,
                        "platform": {"os": "unknown", "architecture": "unknown"},
                    },
                ]
            }
        )
        == runnable
    )


def test_registry_history_detects_late_checkpoint_ownership_copyup() -> None:
    result = summarize_manifest(
        image_ref="docker.io/example/isaac:duplicate",
        resolved_digest="sha256:" + "a" * 64,
        raw_manifest={
            "layers": [
                {"digest": _digest("a"), "size": 8_700_000_000},
                {"digest": _digest("b"), "size": 8_720_000_000},
            ]
        },
        image_config={
            "config": {"Env": ["CUDA_VERSION=12.6.3"]},
            "history": [
                {
                    "created_by": (
                        "RUN python -c 'from huggingface_hub import snapshot_download; "
                        "snapshot_download()'"
                    )
                },
                {
                    "created_by": (
                        "RUN groupadd blueprint && useradd blueprint && "
                        "chown -R blueprint:blueprint /workspace /opt/blueprint/ckpts"
                    )
                },
            ]
        },
    )

    assert result["history_layer_count_matches"] is True
    assert result["checkpoint_ownership_copyup_detected"] is True
    assert result["required_cuda_version"] == "12.6"
    assert result["required_cuda_version_source"] == "image_config_env:CUDA_VERSION"
    assert result["checkpoint_ownership_copyup_layers"][0]["size"] == 8_720_000_000


def test_registry_history_accepts_ownership_inside_checkpoint_producing_layer() -> None:
    result = summarize_manifest(
        image_ref="docker.io/example/isaac:fixed",
        resolved_digest="sha256:" + "a" * 64,
        raw_manifest={
            "layers": [{"digest": _digest("a"), "size": 8_700_000_000}]
        },
        image_config={
            "history": [
                {
                    "created_by": (
                        "RUN python -c 'from huggingface_hub import snapshot_download; "
                        "snapshot_download()' && chown -R blueprint:blueprint "
                        "/opt/blueprint/ckpts"
                    )
                }
            ]
        },
    )

    assert result["checkpoint_ownership_copyup_detected"] is False
    assert result["checkpoint_ownership_copyup_layers"] == []


@pytest.mark.parametrize(
    "invalid_layer",
    [
        {},
        {"digest": _digest("b")},
        {"digest": _digest("b"), "size": True},
        {"digest": _digest("b"), "size": 1.5},
        {"digest": _digest("b"), "size": -1},
        {"digest": "sha256:short", "size": 10},
    ],
)
def test_manifest_blocks_when_any_declared_layer_metadata_is_invalid(
    invalid_layer: dict,
) -> None:
    result = summarize_manifest(
        image_ref="docker.io/example/isaac:invalid",
        resolved_digest=_digest("a"),
        raw_manifest={
            "layers": [
                {"digest": _digest("1"), "size": 100},
                invalid_layer,
            ]
        },
    )

    assert result["status"] == "blocked"
    assert "registry_manifest_layer_metadata_invalid" in result["blockers"]
    assert result["layer_count"] == 2
    assert result["total_compressed_size_bytes"] is None
    assert result["largest_layer_size_bytes"] is None


def test_registry_queries_are_pinned_after_single_tag_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    tag = "docker.io/example/isaac:release"
    digest = _digest("a")
    exact_ref = f"docker.io/example/isaac@{digest}"
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if "--raw" in command:
            assert command[-1] == exact_ref
            return Namespace(
                returncode=0,
                stdout=json.dumps(
                    {"layers": [{"digest": _digest("1"), "size": 1_000}]}
                ),
                stderr="",
            )
        if "--format" in command:
            assert command[-1] == exact_ref
            return Namespace(
                returncode=0,
                stdout=json.dumps({"image": {"history": [{"created_by": "RUN ok"}]}}),
                stderr="",
            )
        assert command[-1] == tag
        return Namespace(
            returncode=0,
            stdout=f"Name: {tag}\nDigest: {digest}\nPlatform: linux/amd64\n",
            stderr="",
        )

    monkeypatch.setattr(
        "blueprint_pipeline.isaac_worker_image_manifest.subprocess.run", fake_run
    )

    result = inspect_image_manifest(
        image_ref=tag, output_path=tmp_path / "diagnostic.json"
    )

    assert result["status"] == "completed"
    assert result["resolved_digest_ref"] == exact_ref
    assert result["registry_inspection_ref"] == exact_ref
    assert calls[0][-1] == tag
    assert all(call[-1] == exact_ref for call in calls[1:])


def test_local_inspection_is_explicitly_not_registry_evidence(monkeypatch) -> None:
    payload = [
        {
            "Id": "sha256:" + "a" * 64,
            "Architecture": "amd64",
            "Os": "linux",
            "Size": 1234,
            "RootFS": {"Layers": ["sha256:one", "sha256:two"]},
            "Config": {
                "User": "blueprint:blueprint",
                "Entrypoint": ["worker"],
                "WorkingDir": "/workspace",
            },
        }
    ]
    monkeypatch.setattr(
        "blueprint_pipeline.isaac_worker_image_manifest.subprocess.run",
        lambda *args, **kwargs: Namespace(returncode=0, stdout=json.dumps(payload)),
    )

    result = _inspect_local_image("example:versioned")

    assert result is not None
    assert result["status"] == "completed_local"
    assert result["resolved_digest_ref"] is None
    assert result["total_compressed_size_bytes"] is None
    assert result["local_uncompressed_size_bytes"] == 1234
    assert result["layer_count"] == 2
    assert result["runtime_config"]["user"] == "blueprint:blueprint"


def test_isaac_build_script_keeps_digest_pinned_base_default() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_push_isaac_worker_image.sh"
    ).read_text(encoding="utf-8")
    assert "nvcr.io/nvidia/isaac-sim:6.0.0@sha256:" in script
