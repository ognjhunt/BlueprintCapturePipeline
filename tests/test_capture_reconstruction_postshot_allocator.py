from __future__ import annotations

import subprocess
import json
import hashlib
from pathlib import Path

import pytest

import blueprint_pipeline.capture_reconstruction_postshot_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_paid_allocator_requires_exact_clean_checkout(monkeypatch) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )
    monkeypatch.setattr(allocator.subprocess, "run", lambda *a, **k: next(responses))
    assert allocator._require_exact_clean_checkout("a" * 40) == {
        "source_commit_sha": "a" * 40,
        "checkout_clean": True,
    }


@pytest.mark.parametrize(
    ("head", "status", "message"),
    [
        ("b" * 40, "", "capture_postshot_checkout_commit_mismatch"),
        ("a" * 40, " M src/file.py", "capture_postshot_checkout_not_clean"),
    ],
)
def test_paid_allocator_refuses_commit_or_dirty_checkout(
    monkeypatch, head: str, status: str, message: str
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, head + "\n", ""),
            subprocess.CompletedProcess([], 0, status, ""),
        ]
    )
    monkeypatch.setattr(allocator.subprocess, "run", lambda *a, **k: next(responses))
    with pytest.raises(allocator.CapturePostshotAllocatorError, match=message):
        allocator._require_exact_clean_checkout("a" * 40)


def test_resumed_downstream_request_is_digest_bound(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "schema_version": "capture_reconstruction_downstream_request.v1",
        "capture_id": "capture-1",
        "capture_digest": "sha256:" + "a" * 64,
        "raw_root": str(tmp_path / "raw"),
        "derived_root": str(tmp_path / "derived"),
        "publication": {"publication_digest": "sha256:" + "b" * 64},
    }
    payload["downstream_request_digest"] = canonical_digest(
        payload, digest_field="downstream_request_digest"
    )
    request_path = tmp_path / "downstream.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        allocator,
        "_downstream_dispatcher",
        lambda **kwargs: observed.update(kwargs) or "callback",
    )
    assert allocator.load_postshot_downstream_dispatch(request_path) == "callback"
    assert observed["request"]["capture_id"] == "capture-1"

    payload["raw_root"] = str(tmp_path / "mutated")
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        allocator.CapturePostshotAllocatorError,
        match="capture_postshot_downstream_request_digest_invalid",
    ):
        allocator.load_postshot_downstream_dispatch(request_path)


def _runtime_dependency_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for label, file_env, digest_env, _url_env in allocator._WINDOWS_RUNTIME_DEPENDENCIES:
        path = tmp_path / label
        path.write_bytes((label + "-exact-bytes").encode())
        monkeypatch.setenv(file_env, str(path))
        monkeypatch.setenv(digest_env, hashlib.sha256(path.read_bytes()).hexdigest())


def test_runtime_dependencies_get_fresh_run_local_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _runtime_dependency_environment(monkeypatch, tmp_path)
    staged: list[str] = []

    def fake_stage(**kwargs):
        root = Path(kwargs["job_dir"])
        root.mkdir(parents=True)
        (root / allocator.RUNTIME_DEPENDENCY_URL_FILENAME).write_text(
            "https://example.invalid/fresh-" + root.name,
            encoding="utf-8",
        )
        staged.append(root.name)
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(allocator, "stage_cached_runtime_dependency_object_store", fake_stage)
    environment, roots = allocator._stage_windows_runtime_dependencies(
        root=tmp_path / "runtime", expiration_seconds=5400
    )
    assert staged == [row[0] for row in allocator._WINDOWS_RUNTIME_DEPENDENCIES]
    assert len(roots) == len(allocator._WINDOWS_RUNTIME_DEPENDENCIES)
    for _label, _file_env, digest_env, url_env in allocator._WINDOWS_RUNTIME_DEPENDENCIES:
        assert environment[url_env].startswith("https://example.invalid/fresh-")
        assert environment[digest_env] == allocator._required_env(digest_env)


def test_runtime_dependency_failure_removes_already_issued_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _runtime_dependency_environment(monkeypatch, tmp_path)
    calls = 0
    closed: list[str] = []

    def fake_stage(**kwargs):
        nonlocal calls
        calls += 1
        root = Path(kwargs["job_dir"])
        root.mkdir(parents=True)
        if calls == 2:
            return {"status": "blocked", "blockers": ["remote_identity_mismatch"]}
        (root / allocator.RUNTIME_DEPENDENCY_URL_FILENAME).write_text(
            "https://example.invalid/fresh", encoding="utf-8"
        )
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(allocator, "stage_cached_runtime_dependency_object_store", fake_stage)
    monkeypatch.setattr(
        allocator,
        "close_cached_runtime_dependency_staging",
        lambda root: closed.append(Path(root).name) or {"status": "completed"},
    )
    with pytest.raises(
        allocator.CapturePostshotAllocatorError,
        match="runtime_dependency_staging_blocked",
    ):
        allocator._stage_windows_runtime_dependencies(
            root=tmp_path / "runtime", expiration_seconds=5400
        )
    assert closed == ["nvidia-driver", "postshot-installer"]
