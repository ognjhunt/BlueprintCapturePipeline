from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import blueprint_pipeline.capture_reconstruction_postshot_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest






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
