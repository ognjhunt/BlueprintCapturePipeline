from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.build_release_command_attestation import build_attestation_subject


def test_release_subject_binds_sha_runs_release_and_artifact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "ci" / "result.json"
    artifact.parent.mkdir()
    artifact.write_bytes(b"evidence")
    payload = build_attestation_subject(
        repository="ognjhunt/BlueprintCapturePipeline",
        repository_sha="a" * 40,
        image_digest="sha256:" + "b" * 64,
        release_id="release-17",
        workflow_run_ids={"ci": "1", "full_test": "2", "codeql": "3"},
        evidence_root=tmp_path,
    )
    assert payload["repository_sha"] == "a" * 40
    assert payload["workflow_run_ids"]["full_test"] == "2"
    assert payload["artifact_sha256s"] == {
        "ci/result.json": "sha256:" + hashlib.sha256(b"evidence").hexdigest()
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("repository_sha", "main"), ("image_digest", "sha256:bad")],
)
def test_release_subject_rejects_unpinned_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    (tmp_path / "evidence").write_text("x")
    kwargs = {
        "repository": "ognjhunt/BlueprintCapturePipeline",
        "repository_sha": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "release_id": "release-17",
        "workflow_run_ids": {"ci": "1", "full_test": "2", "codeql": "3"},
        "evidence_root": tmp_path,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        build_attestation_subject(**kwargs)
