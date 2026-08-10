from __future__ import annotations

import json

import pytest

from blueprint_pipeline.task_evaluation_artifact_manifest import (
    TaskEvaluationArtifactManifestError,
    build_task_evaluation_artifact_manifest,
)


def test_manifest_hashes_allocator_retained_runtime_and_teardown_bytes(tmp_path) -> None:
    attempt = tmp_path / "attempt_001"
    runtime = attempt / "immutable_execution"
    provider = attempt / "vast_provider_run"
    runtime.mkdir(parents=True)
    provider.mkdir()
    (runtime / "frame.png").write_bytes(b"lossless-frame")
    (runtime / "review.mp4").write_bytes(b"review-video")
    (provider / "vast_provider_adapter_result.json").write_text("{}\n")
    (provider / "vast_teardown_manifest.json").write_text("{}\n")

    manifest = build_task_evaluation_artifact_manifest(
        attempt_root=attempt,
        artifact_roots={
            "provider_runtime_evidence": runtime,
            "allocator_adapter_result": provider / "vast_provider_adapter_result.json",
            "teardown_manifest": provider / "vast_teardown_manifest.json",
        },
        required_roles=(
            "provider_runtime_evidence",
            "allocator_adapter_result",
            "teardown_manifest",
        ),
        binding={"launch_id": "launch-1", "bundle_sha256": "sha256:" + "a" * 64},
    )

    assert manifest["status"] == "completed"
    assert manifest["file_count"] == 4
    assert manifest["raw_secret_values_recorded"] is False
    assert manifest["manifest_digest"].startswith("sha256:")
    assert (attempt / "artifact_manifest.json").is_file()
    assert {
        row["relative_path"] for row in manifest["files"]
    } == {
        "immutable_execution/frame.png",
        "immutable_execution/review.mp4",
        "vast_provider_run/vast_provider_adapter_result.json",
        "vast_provider_run/vast_teardown_manifest.json",
    }


def test_manifest_fails_closed_on_missing_required_role(tmp_path) -> None:
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    present = attempt / "result.json"
    present.write_text("{}\n")

    manifest = build_task_evaluation_artifact_manifest(
        attempt_root=attempt,
        artifact_roots={
            "provider_runtime_evidence": present,
            "teardown_manifest": attempt / "missing.json",
        },
        required_roles=("provider_runtime_evidence", "teardown_manifest"),
        binding={},
    )

    assert manifest["status"] == "blocked"
    assert manifest["blockers"] == [
        "task_evaluation_artifact_role_missing:teardown_manifest"
    ]


def test_manifest_rejects_paths_outside_attempt_root(tmp_path) -> None:
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    with pytest.raises(
        TaskEvaluationArtifactManifestError,
        match="task_evaluation_artifact_path_outside_attempt_root",
    ):
        build_task_evaluation_artifact_manifest(
            attempt_root=attempt,
            artifact_roots={"runtime": outside},
            required_roles=("runtime",),
            binding={},
        )


def test_manifest_is_immutable_for_the_same_attempt(tmp_path) -> None:
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    artifact = attempt / "result.json"
    artifact.write_text("{}\n")
    arguments = {
        "attempt_root": attempt,
        "artifact_roots": {"runtime": artifact},
        "required_roles": ("runtime",),
        "binding": {"run_id": "run-1"},
    }
    first = build_task_evaluation_artifact_manifest(**arguments)
    assert build_task_evaluation_artifact_manifest(**arguments) == first

    artifact.write_text(json.dumps({"changed": True}) + "\n")
    with pytest.raises(
        TaskEvaluationArtifactManifestError,
        match="task_evaluation_artifact_manifest_immutable_conflict",
    ):
        build_task_evaluation_artifact_manifest(**arguments)
