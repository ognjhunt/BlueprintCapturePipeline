import hashlib
import io
import time
import urllib.error

import pytest

from blueprint_pipeline.native_g1_beta_assets import ORDER, selected_asset_inventory, selected_candidates
from blueprint_pipeline import native_g1_beta_download as download
from blueprint_pipeline.native_g1_beta_download import _fetch


def test_dp_selected_inventory_does_not_include_pi_or_tokenizer():
    row = selected_asset_inventory([ORDER[0]])
    assert row["candidate_ids"] == [ORDER[0]]
    assert not row["pi_tokenizer_required"]
    assert {f["candidate_id"] for f in row["checkpoint_files"]} == {ORDER[0]}
    assert row["auxiliary_download_bytes"] == 46830571
    pi = selected_asset_inventory([ORDER[1]])
    assert pi["pi_tokenizer_required"] and not pi["auxiliary_files"]
    assert pi["checkpoint_download_bytes"] > row["checkpoint_download_bytes"]


@pytest.mark.parametrize("ids", [[], [ORDER[1], ORDER[0]], [ORDER[0], ORDER[0]], ["unknown"], ORDER[0]])
def test_selection_must_be_exact_ordered_subset(ids):
    with pytest.raises(ValueError):
        selected_candidates(ids)


class Response(io.BytesIO):
    headers = {}
    def geturl(self):
        return "https://example.test/verified"


def test_streaming_download_checks_bytes_deadline_and_no_retry(tmp_path):
    content = b"abc" * 100
    row = {"relative_path": "model", "url": "https://example.test/model", "size_bytes": len(content),
           "sha256": "sha256:" + hashlib.sha256(content).hexdigest()}
    calls = []
    def opener(*args, **kwargs):
        calls.append(args)
        return Response(content)
    result = _fetch(row, tmp_path / "model", deadline=time.time() + 30, opener=opener)
    assert result["size_bytes"] == len(content) and len(calls) == 1
    for label, data in (("oversize", content + b"x"), ("changed", content[:-1] + b"x"), ("short", content[:-1])):
        with pytest.raises(ValueError):
            _fetch(row, tmp_path / label, deadline=time.time() + 30, opener=lambda *a, **k: Response(data))
        assert not (tmp_path / label).exists()
        assert not (tmp_path / (label + ".partial")).exists()
    with pytest.raises(ValueError, match="deadline_exceeded"):
        _fetch(row, tmp_path / "expired", deadline=time.time() - 1, opener=opener)
    assert len(calls) == 1


class Stalling(io.BytesIO):
    """Serves ``data`` from ``offset`` and stalls (the 45 s read timeout) once ``stall_after`` bytes are read."""

    def __init__(self, data, *, offset=0, total=None, stall_after=None, status=200, content_range=None):
        super().__init__(data[offset:])
        self.stall_after, self.status, self.served = stall_after, status, 0
        self.headers = {"Content-Length": str(len(data) - offset)}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def geturl(self):
        return "https://example.test/verified"

    def read(self, size=-1):
        if self.stall_after is not None and self.served >= self.stall_after:
            raise TimeoutError("The read operation timed out")
        chunk = super().read(size if self.stall_after is None else min(size, self.stall_after - self.served))
        self.served += len(chunk)
        return chunk


def _row(content):
    return {"relative_path": "model", "url": "https://example.test/model", "size_bytes": len(content),
            "sha256": "sha256:" + hashlib.sha256(content).hexdigest()}


def test_a_stalled_transfer_resumes_from_the_byte_it_reached(tmp_path, capsys):
    """2026-11-09, the library job's pi0.5 run: one selected-asset read stalled 27 minutes in, and with no
    resume the run ended with no episode. A resume asks only for the bytes still missing."""
    content = bytes(range(256)) * 40
    requests = []

    def opener(request, timeout):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return Stalling(content, stall_after=1000)
        offset = int(request.get_header("Range")[len("bytes="):-1])
        return Stalling(content, offset=offset, status=206,
                        content_range=f"bytes {offset}-{len(content) - 1}/{len(content)}")

    result = _fetch(_row(content), tmp_path / "model", deadline=time.time() + 30, opener=opener)
    assert result["size_bytes"] == len(content) and (tmp_path / "model").read_bytes() == content
    assert requests == [None, "bytes=1000-"]
    assert capsys.readouterr().out.splitlines() == ["BLUEPRINT_G1_ASSET_RESUME:1:bytes=1000"]
    assert not (tmp_path / "model.partial").exists()


def test_a_stall_before_the_first_byte_asks_again_from_the_start(tmp_path):
    content = b"xyz" * 50
    requests = []

    def opener(request, timeout):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            raise urllib.error.URLError(TimeoutError("timed out"))
        return Stalling(content)

    _fetch(_row(content), tmp_path / "model", deadline=time.time() + 30, opener=opener)
    assert requests == [None, None]


@pytest.mark.parametrize("answer", [
    # The source ignores the Range header and sends the whole file again: refused, never re-downloaded.
    lambda content, offset: Stalling(content),
    # A 206 for other bytes than the missing ones.
    lambda content, offset: Stalling(content, offset=offset - 1, status=206,
                                     content_range=f"bytes {offset - 1}-{len(content) - 1}/{len(content)}"),
])
def test_a_resume_answered_with_other_bytes_ends_the_transfer(tmp_path, answer):
    content = b"q" * 3000
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return Stalling(content, stall_after=1200) if len(calls) == 1 else answer(content, 1200)

    with pytest.raises(ValueError, match="g1_beta_asset_resume_unsupported"):
        _fetch(_row(content), tmp_path / "model", deadline=time.time() + 30, opener=opener)
    assert not (tmp_path / "model").exists() and not (tmp_path / "model.partial").exists()


def test_the_resumes_are_bounded_and_an_http_error_is_never_resumed(tmp_path):
    content = b"r" * 5000
    calls = []

    def stalls(request, timeout):
        calls.append(request.get_header("Range"))
        offset = 0 if len(calls) == 1 else int(request.get_header("Range")[len("bytes="):-1])
        return Stalling(content, offset=offset, stall_after=100, status=206 if offset else 200,
                        content_range=f"bytes {offset}-{len(content) - 1}/{len(content)}" if offset else None)

    with pytest.raises(TimeoutError):
        _fetch(_row(content), tmp_path / "model", deadline=time.time() + 30, opener=stalls)
    assert len(calls) == download.TRANSFER_RESUMES + 1
    assert calls[-1] == f"bytes={100 * download.TRANSFER_RESUMES}-"
    assert not (tmp_path / "model").exists() and not (tmp_path / "model.partial").exists()

    refused = []

    def http_error(request, timeout):
        refused.append(request)
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        _fetch(_row(content), tmp_path / "again", deadline=time.time() + 30, opener=http_error)
    assert len(refused) == 1
