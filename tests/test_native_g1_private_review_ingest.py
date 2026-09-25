from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_private_review_ingest as ingest


class _Response:
    status = 201

    def __init__(self, url: str, payload: dict[str, object]) -> None:
        self.url = url
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.payload).encode()


def test_private_review_ingest_signs_exact_body_and_rejects_redirect(monkeypatch) -> None:
    url = ingest._ingest_url("https://tryblueprint.io/api/internal/pipeline/sync")
    assert url == "https://tryblueprint.io/api/internal/pipeline/native-g1-reviews"
    payload = {"run_id": "g1-841757-test", "review": {"review_digest": "sha256:" + "a" * 64}}
    captured = []

    class _Opener:
        def open(self, request, *, timeout):
            captured.append((request, timeout))
            return _Response(url, {"status": "ingested", "run_id": payload["run_id"],
                                   "review_digest": payload["review"]["review_digest"]})

    monkeypatch.setattr(ingest.urllib.request, "build_opener", lambda handler: _Opener())
    result = ingest._post_review(url=url, token="test-secret", payload=payload)
    assert result["status"] == "ingested"
    request, timeout = captured[0]
    assert timeout == 30
    assert json.loads(request.data) == payload
    timestamp = request.get_header("X-blueprint-pipeline-timestamp")
    signature = request.get_header("X-blueprint-pipeline-signature")
    assert signature == "sha256=" + hmac.new(
        b"test-secret", f"{timestamp}.".encode() + request.data, hashlib.sha256,
    ).hexdigest()
    assert "test-secret" not in str(request.header_items())

    class _RedirectingOpener:
        def open(self, request, *, timeout):
            raise urllib.error.HTTPError(request.full_url, 302, "redirect", {}, None)

    monkeypatch.setattr(ingest.urllib.request, "build_opener", lambda handler: _RedirectingOpener())
    with pytest.raises(ValueError, match="g1_review_ingest_http_status_302"):
        ingest._post_review(url=url, token="test-secret", payload=payload)


def test_private_review_ingest_refuses_changed_registry_before_network(
    monkeypatch, tmp_path: Path,
) -> None:
    run = "g1-841757-test"
    root = tmp_path / "results"
    target = root / f"{run}-activation"
    registry = target / "artifacts/result_delivery/artifact_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"run_id": run, "delivery_digest": "sha256:" + "a" * 64,
                                    "registry_digest": "sha256:" + "b" * 64}))
    files = {
        "adapter": {"status": "completed"},
        "bundle": {"implementation_commit": "0" * 40},
        "review": {"review_digest": "sha256:" + "a" * 64},
        "delivery": {"status": "registered_private_development_review", "run_id": run,
                     "review_digest": "sha256:" + "a" * 64, "artifact_count": 12,
                     "run_root": str(target), "registry_digest": "sha256:" + "c" * 64,
                     "public_redistribution_authorized": False},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(ingest, "load_verified_g1_provider_bundle", lambda *_args, **_kw: {})
    monkeypatch.setattr(ingest, "verify_g1_paid_output", lambda *_args, **_kw: {})
    monkeypatch.setattr(ingest, "project_g1_private_review", lambda **_kw: files["review"])
    with pytest.raises(ValueError, match="g1_review_ingest_registry_identity_invalid"):
        ingest._verified_review(
            adapter_result_path=tmp_path / "adapter",
            bundle_receipt_path=tmp_path / "bundle",
            retained_review_path=tmp_path / "review",
            delivery_receipt_path=tmp_path / "delivery",
            result_root=root, run_id=run,
        )


@pytest.mark.parametrize("url", [
    "http://tryblueprint.io", "https://user:pass@tryblueprint.io",
    "https://tryblueprint.io/?token=secret", "https://tryblueprint.io/#fragment",
])
def test_private_review_ingest_requires_plain_https_origin(url: str) -> None:
    with pytest.raises(ValueError, match="g1_review_ingest_webapp_url_invalid"):
        ingest._ingest_url(url)
