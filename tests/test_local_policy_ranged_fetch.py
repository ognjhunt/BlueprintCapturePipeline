"""Large exact downloads come in bounded byte ranges; a failed one names its reason and file."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import local_policy_provisioning as provisioning
from blueprint_pipeline.core.security_controls import SecurityValidationError


def _row(payload: bytes) -> dict:
    return {"url": "https://huggingface.co/org/model/resolve/rev/weights.bin?download=true",
            "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class RangeServer:
    """A fake bounded fetch: honours a byte range with 206, or ignores it with 200."""

    def __init__(self, payload: bytes, *, honour_ranges: bool = True, short: bool = False):
        self.payload, self.honour_ranges, self.short = payload, honour_ranges, short
        self.calls: list[dict] = []

    def __call__(self, url, *, allowed_origins, timeout_seconds, max_bytes, max_redirects, output_path, headers=None):
        self.calls.append({"headers": headers, "max_bytes": max_bytes, "timeout_seconds": timeout_seconds})
        assert allowed_origins == provisioning.ALLOWED_ORIGINS and timeout_seconds <= 600
        if headers and self.honour_ranges:
            first, last = (int(value) for value in headers["Range"].removeprefix("bytes=").split("-"))
            body = self.payload[first:last + 1][:-1 if self.short else None]
            Path(output_path).write_bytes(body)
            return SimpleNamespace(status=206)
        if len(self.payload) > max_bytes:  # the real fetch refuses a body past its bound
            raise SecurityValidationError("remote response exceeds byte limit")
        Path(output_path).write_bytes(self.payload)
        return SimpleNamespace(status=200)


def test_a_file_larger_than_one_range_is_fetched_in_bounded_ranges(tmp_path, monkeypatch):
    """2026-11-05: one bounded request had 600 seconds for a whole multi-gigabyte file."""
    monkeypatch.setattr(provisioning, "RANGE_CHUNK_BYTES", 10)
    payload = bytes(range(256)) * 3 + b"tail"
    server = RangeServer(payload)
    result = provisioning.fetch_exact(_row(payload), tmp_path / "weights.bin", deadline=time.time() + 60,
                                      fetch=server, budget=provisioning.TransferBudget(len(payload)))
    assert result["status"] == "verified" and (tmp_path / "weights.bin").read_bytes() == payload
    assert [call["headers"]["Range"] for call in server.calls[:2]] == ["bytes=0-9", "bytes=10-19"]
    assert all(call["max_bytes"] <= 10 for call in server.calls)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["weights.bin"]


def test_a_small_file_is_one_request_as_before(tmp_path):
    payload = b"small file"
    server = RangeServer(payload)
    provisioning.fetch_exact(_row(payload), tmp_path / "small.bin", deadline=time.time() + 60, fetch=server)
    assert len(server.calls) == 1 and server.calls[0]["headers"] is None


@pytest.mark.parametrize("server_kind, error", [("ignores_ranges", SecurityValidationError),
                                               ("short_range", ValueError)])
def test_a_server_that_ignores_or_shortens_a_range_is_refused(tmp_path, monkeypatch, server_kind, error):
    monkeypatch.setattr(provisioning, "RANGE_CHUNK_BYTES", 10)
    payload = bytes(range(100))
    server = RangeServer(payload, honour_ranges=server_kind != "ignores_ranges", short=server_kind == "short_range")
    with pytest.raises(error) as raised:
        provisioning.fetch_exact(_row(payload), tmp_path / "Weights-00001.safetensors", deadline=time.time() + 60,
                                 fetch=server)
    assert raised.value.local_policy_download == "Weights-00001.safetensors"
    assert list(tmp_path.iterdir()) == []  # no partial or range file is left behind


def test_the_failure_code_names_a_refused_download_without_its_url():
    refusal = SecurityValidationError("remote response exceeded total time limit")
    refusal.local_policy_download = "diffusion_pytorch_model-00001-of-00002.safetensors"
    assert provisioning.provisioning_failure_code("cosmos", refusal) == (
        "cosmos_provisioning_failed:SecurityValidationError:remote_response_exceeded_total_time_limit:"
        "diffusion_pytorch_model-00001-of-00002.safetensors")
    # A provider's own typed code is kept as it is; any other failure names only its class.
    assert provisioning.provisioning_failure_code("cosmos", ValueError("cosmos_checkpoint_file_closure_changed")) == (
        "cosmos_checkpoint_file_closure_changed")
    assert provisioning.provisioning_failure_code(
        "molmoact2", RuntimeError("https://host/path?token=secret failed")) == "molmoact2_provisioning_failed:RuntimeError"


def _http_error(code: int):
    import urllib.error
    return urllib.error.HTTPError("https://github.com/org/repo/releases/download/v1/x.whl", code, "refused", {}, None)


class Flaky:
    """Refuses with the given errors before any body, then serves the payload."""

    def __init__(self, payload: bytes, errors: list[BaseException]):
        self.payload, self.errors, self.calls = payload, list(errors), 0

    def __call__(self, url, *, allowed_origins, timeout_seconds, max_bytes, max_redirects, output_path, headers=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        Path(output_path).write_bytes(self.payload)
        return SimpleNamespace(status=200)


@pytest.mark.parametrize("first", [_http_error(503), _http_error(429), _http_error(403),
                                   SecurityValidationError("DNS resolution failed")])
def test_a_refusal_before_any_body_byte_is_asked_again(tmp_path, monkeypatch, first):
    """2026-11-05: a host's GitHub wheel download failed with an HTTP error status, then answered."""
    monkeypatch.setattr(provisioning.time, "sleep", lambda _seconds: None)
    payload = b"wheel bytes"
    server = Flaky(payload, [first])
    provisioning.fetch_exact(_row(payload), tmp_path / "x.whl", deadline=time.time() + 60, fetch=server)
    assert server.calls == 2 and (tmp_path / "x.whl").read_bytes() == payload


@pytest.mark.parametrize("error", [_http_error(404), SecurityValidationError("remote response exceeded total time limit"),
                                   SecurityValidationError("remote URL origin is not approved")])
def test_a_missing_file_or_a_failure_part_way_through_is_not_retried(tmp_path, monkeypatch, error):
    """A body already moved counts against the transfer budget; it is never fetched twice."""
    monkeypatch.setattr(provisioning.time, "sleep", lambda _seconds: None)
    server = Flaky(b"x", [error])
    with pytest.raises(type(error)):
        provisioning.fetch_exact(_row(b"x"), tmp_path / "x.whl", deadline=time.time() + 60, fetch=server)
    assert server.calls == 1


def test_retries_are_bounded_and_the_code_names_the_status_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(provisioning.time, "sleep", lambda _seconds: None)
    server = Flaky(b"x", [_http_error(503)] * provisioning.ZERO_BYTE_ATTEMPTS)
    name = "flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl"
    with pytest.raises(Exception) as raised:
        provisioning.fetch_exact(_row(b"x"), tmp_path / name, deadline=time.time() + 60, fetch=server)
    assert server.calls == provisioning.ZERO_BYTE_ATTEMPTS
    assert provisioning.provisioning_failure_code("cosmos", raised.value) == (
        "cosmos_provisioning_failed:HTTPError:http_503:flash_attn_3_nv-1.0.3_cu128.torch210-cp39-abi3-linux_x86_64.whl")
