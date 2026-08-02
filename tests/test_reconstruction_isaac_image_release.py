from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.reconstruction_isaac_image_release import (
    ReconstructionIsaacImageReleaseError,
    build_reconstruction_isaac_image_release,
    validate_reconstruction_isaac_image_release,
)


SHA = "a" * 40
IMAGE = "registry.example/blueprint/isaac@sha256:" + "b" * 64


def _manifest(**updates):
    value = {
        "schema_version": "isaac_worker_image_manifest_diagnostic.v2",
        "status": "completed",
        "resolved_digest_ref": IMAGE,
        "runnable_platform": "linux/amd64",
        "raw_secret_values_recorded": False,
        "worker_build_identity": {
            "status": "verified",
            "blockers": [],
            "source_commit": SHA,
            "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
            "worker_image_family": "isaac-eval-worker",
            "isaac_sim_major_version": 6,
            "identity_source": "immutable_registry_image_config_environment",
        },
    }
    value.update(updates)
    return value


def test_isaac_image_release_binds_exact_clean_registry_config_and_replays():
    release = build_reconstruction_isaac_image_release(
        image_manifest=_manifest(), expected_source_commit=SHA
    )
    assert release["status"] == "passed"
    assert release["resolved_image_digest"] == IMAGE
    assert release["runtime_smoke_completed"] is False
    assert release["isaac_compatibility_proven"] is False
    assert validate_reconstruction_isaac_image_release(release) == release
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_isaac_image_release.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(release)


@pytest.mark.parametrize(
    ("manifest_update", "expected_commit", "blocker"),
    [
        ({"resolved_digest_ref": "registry.example/isaac:latest"}, SHA, "not_digest_pinned"),
        ({"runnable_platform": "linux/arm64"}, SHA, "platform_invalid"),
        (
            {
                "worker_build_identity": {
                    **_manifest()["worker_build_identity"],
                    "source_dirty_patch_sha256": "f" * 64,
                }
            },
            SHA,
            "dirty_overlay_forbidden",
        ),
        ({}, "c" * 40, "source_commit_mismatch"),
    ],
)
def test_isaac_image_release_fails_closed_on_registry_or_source_drift(
    manifest_update, expected_commit, blocker
):
    release = build_reconstruction_isaac_image_release(
        image_manifest=_manifest(**manifest_update),
        expected_source_commit=expected_commit,
    )
    assert release["status"] == "blocked"
    assert any(blocker in code for code in release["blockers"])
    with pytest.raises(ReconstructionIsaacImageReleaseError):
        validate_reconstruction_isaac_image_release(release)


def test_isaac_image_release_rejects_mutated_recorded_receipt():
    release = build_reconstruction_isaac_image_release(
        image_manifest=_manifest(), expected_source_commit=SHA
    )
    release["provider_startup_proven"] = True
    with pytest.raises(
        ReconstructionIsaacImageReleaseError, match="replay_mismatch"
    ):
        validate_reconstruction_isaac_image_release(release)


def test_isaac_image_release_requires_overlay_layer_lineage() -> None:
    identity = {
        **_manifest()["worker_build_identity"],
        "build_method": "crane_exact_source_overlay_experimental",
        "base_image_digest": "docker.io/example/isaac@sha256:" + "c" * 64,
        "source_manifest_sha256": "sha256:" + "d" * 64,
        "source_layer_digest": "sha256:" + "e" * 64,
        "source_layer_matches_last_registry_layer": True,
    }
    release = build_reconstruction_isaac_image_release(
        image_manifest=_manifest(worker_build_identity=identity),
        expected_source_commit=SHA,
    )
    assert release["status"] == "passed"

    identity["source_layer_matches_last_registry_layer"] = False
    blocked = build_reconstruction_isaac_image_release(
        image_manifest=_manifest(worker_build_identity=identity),
        expected_source_commit=SHA,
    )
    assert blocked["status"] == "blocked"
    assert "reconstruction_isaac_image_source_overlay_lineage_invalid" in blocked["blockers"]
