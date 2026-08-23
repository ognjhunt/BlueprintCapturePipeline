from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pytest

import blueprint_pipeline.capture_reconstruction_postshot_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_paid_allocator_requires_exact_clean_checkout(monkeypatch) -> None:
    responses = iter([
        subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ])
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
    responses = iter([
        subprocess.CompletedProcess([], 0, head + "\n", ""),
        subprocess.CompletedProcess([], 0, status, ""),
    ])
    monkeypatch.setattr(allocator.subprocess, "run", lambda *a, **k: next(responses))
    with pytest.raises(allocator.CapturePostshotAllocatorError, match=message):
        allocator._require_exact_clean_checkout("a" * 40)


def test_resumed_downstream_request_is_digest_bound(
    tmp_path: Path, monkeypatch
) -> None:
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
