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


def _sealed_index(members: dict) -> dict:
    from blueprint_pipeline.provider_output_member_index import build_member_index, seal_durable_reference
    from tests.provider_output_fixtures import DEFLATED, STORED, Entry, RangeStore, Zeros, build_zip

    archive = build_zip([Entry(name, data, method=STORED if isinstance(data, Zeros) else DEFLATED)
                         for name, data in members.items()])
    index = build_member_index(RangeStore(archive).reader(block_bytes=128 * 1024),
                               maximum_expanded_bytes=64 * 1024**2)
    digest = index["archive"]["sha256"]
    return seal_durable_reference(index, {
        "schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
        "artifact_kind": "policy-canary-provider-output",
        "uri": ("s3://blueprint-artifacts/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/"
                f"policy-canary-provider-output/sha256/{digest.removeprefix('sha256:')}/output.zip"),
        "digest": digest, "size_bytes": index["archive"]["size"], "content_addressed_key": True,
        "remote_identity_verified": True, "full_byte_service_account_readback_passed": True})



def _streamed_attempt(tmp_path, *, local: dict):
    from tests.provider_output_fixtures import Zeros

    members = {"result.json": b'{"status": "completed"}', "media/e0/external.mp4": Zeros(8 * 1024**2),
               "media/e0/frames/external/000000.png": Zeros(1024**2)}
    index = _sealed_index(members)
    attempt = tmp_path / "attempt_001"
    evidence = attempt / "immutable_execution"
    provider = attempt / "vast_provider_run"
    evidence.mkdir(parents=True)
    provider.mkdir()
    for name, data in local.items():
        (evidence / name).parent.mkdir(parents=True, exist_ok=True)
        (evidence / name).write_bytes(data)
    (provider / "vast_provider_adapter_result.json").write_text("{}\n")
    (provider / "vast_teardown_manifest.json").write_text("{}\n")
    arguments = {
        "attempt_root": attempt,
        "artifact_roots": {"provider_runtime_evidence": evidence,
                           "allocator_adapter_result": provider / "vast_provider_adapter_result.json",
                           "teardown_manifest": provider / "vast_teardown_manifest.json"},
        "required_roles": ("provider_runtime_evidence", "allocator_adapter_result", "teardown_manifest"),
        "binding": {"run_id": "run-1"},
        "archive_members": {"provider_runtime_evidence": (index, "immutable_execution")},
    }
    return members, index, arguments


def test_archive_members_are_listed_by_index_digest_without_bytes(tmp_path) -> None:
    members, index, arguments = _streamed_attempt(tmp_path, local={"result.json": b'{"status": "completed"}'})
    rows = {row["path"]: row for row in index["members"]}

    manifest = build_task_evaluation_artifact_manifest(**arguments)

    assert manifest["status"] == "completed" and manifest["blockers"] == []
    by_path = {row["relative_path"]: row for row in manifest["files"]}
    # The materialized member is listed as today, hashed from disk.
    assert by_path["immutable_execution/result.json"] == {
        "relative_path": "immutable_execution/result.json", "roles": ["provider_runtime_evidence"],
        "size_bytes": 23, "sha256": rows["result.json"]["sha256"]}
    # The rest are listed from the index: its digest and size, no bytes read.
    for name in ("media/e0/external.mp4", "media/e0/frames/external/000000.png"):
        assert by_path["immutable_execution/" + name] == {
            "relative_path": "immutable_execution/" + name, "roles": ["provider_runtime_evidence"],
            "size_bytes": rows[name]["size"], "sha256": rows[name]["sha256"], "location": "archive_member"}
        assert not (tmp_path / "attempt_001/immutable_execution" / name).exists()
    assert [row["relative_path"] for row in manifest["files"]] == sorted(by_path)
    assert manifest["file_count"] == 5 and manifest["total_size_bytes"] == 23 + 9 * 1024**2 + 2 * len("{}\n")
    assert build_task_evaluation_artifact_manifest(**arguments) == manifest  # immutable, as before

    # With nothing materialized the role is still present, from the archive.
    _, _, remote_only = _streamed_attempt(tmp_path / "remote_only", local={})
    manifest = build_task_evaluation_artifact_manifest(**remote_only)
    assert manifest["status"] == "completed"
    assert "provider_runtime_evidence" in manifest["observed_roles"]
    assert all(row.get("location") == "archive_member" for row in manifest["files"]
               if row["relative_path"].startswith("immutable_execution/"))


def test_local_member_must_match_its_index_digest(tmp_path) -> None:
    _, index, arguments = _streamed_attempt(tmp_path, local={"result.json": b'{"status": "blocked"}'})
    with pytest.raises(TaskEvaluationArtifactManifestError,
                       match="^task_evaluation_artifact_archive_member_digest_mismatch$"):
        build_task_evaluation_artifact_manifest(**arguments)
    assert not (tmp_path / "attempt_001/artifact_manifest.json").exists()

    _, _, clean = _streamed_attempt(tmp_path / "unsealed", local={})
    unsealed = json.loads(json.dumps(index))
    unsealed["archive"]["durable_reference"] = None
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    unsealed["index_digest"] = canonical_digest(unsealed, digest_field="index_digest")
    for archive_members, code in (
        ({"provider_runtime_evidence": (unsealed, "immutable_execution")},
         "task_evaluation_artifact_archive_member_index_not_durable"),
        ({"provider_runtime_evidence": ({**index, "index_digest": "sha256:" + "0" * 64}, "immutable_execution")},
         "task_evaluation_artifact_archive_member_index_invalid"),
        ({"provider_runtime_evidence": (index, "elsewhere")},
         "task_evaluation_artifact_archive_member_root_mismatch"),
    ):
        with pytest.raises(TaskEvaluationArtifactManifestError, match=f"^{code}$"):
            build_task_evaluation_artifact_manifest(**{**clean, "archive_members": archive_members})
