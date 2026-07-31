from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import artifact_storage as storage


def test_default_roots_use_explicit_external_environment(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    evidence = tmp_path / "evidence"
    monkeypatch.setenv(storage.ARTIFACT_CACHE_ROOT_ENV, str(cache))
    monkeypatch.setenv(storage.EVIDENCE_ROOT_ENV, str(evidence))

    assert storage.default_artifact_cache_root() == cache.resolve()
    assert storage.default_evidence_root() == evidence.resolve()


def test_repo_output_requires_explicit_legacy_opt_in(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    destination = repo / "output" / "run" / "report.json"
    monkeypatch.delenv(storage.ALLOW_REPO_OUTPUT_ENV, raising=False)

    with pytest.raises(storage.ArtifactStorageError, match="inside repo output"):
        storage.assert_artifact_write_allowed(destination, repo_root=repo)

    monkeypatch.setenv(storage.ALLOW_REPO_OUTPUT_ENV, "1")
    assert storage.assert_artifact_write_allowed(destination, repo_root=repo) == destination.resolve()


def test_large_artifact_requires_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "cache" / "large.bin"
    monkeypatch.delenv(storage.ALLOW_LARGE_ARTIFACTS_ENV, raising=False)

    with pytest.raises(storage.ArtifactStorageError, match="large artifact requires"):
        storage.assert_artifact_write_allowed(
            destination,
            estimated_bytes=storage.LARGE_ARTIFACT_BYTES,
        )

    monkeypatch.setenv(storage.ALLOW_LARGE_ARTIFACTS_ENV, "true")
    assert storage.assert_artifact_write_allowed(
        destination,
        estimated_bytes=storage.LARGE_ARTIFACT_BYTES,
    ) == destination.resolve()


def test_storage_status_has_review_and_hard_stop_states(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "cache"
    sizes = {"value": 0}
    monkeypatch.setattr(storage, "directory_size", lambda _root: sizes["value"])

    assert storage.storage_status(root) == "ok"
    sizes["value"] = storage.REVIEW_THRESHOLD_BYTES
    assert storage.storage_status(root) == "review"
    sizes["value"] = storage.HARD_STOP_BYTES
    assert storage.storage_status(root) == "hard_stop"


def test_cache_run_root_is_namespaced_and_external(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(storage.ARTIFACT_CACHE_ROOT_ENV, str(tmp_path / "cache"))

    path = storage.cache_run_root("sim only / beta gate")

    assert path == (tmp_path / "cache" / "sim-only-beta-gate").resolve()
