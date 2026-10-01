import os
from urllib.request import Request

import pytest

from blueprint_pipeline.metered_call_evidence import (
    MeteredCallEvidenceError,
    MeteredTransport,
    read_attempts,
    summarize_attempts,
)

URL = "https://console.vast.ai/api/v0/charges/?page=1"


def test_attempt_fsynced_before_dispatch_and_received_separate(tmp_path, monkeypatch):
    synced = []
    original = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), original(fd))[-1])
    root = tmp_path / "events"

    def transport(request, timeout):
        events = read_attempts(root)
        assert len(events) == 1 and events[0]["outcome"] == "unresolved"
        assert synced
        return b"{}"

    wrapped = MeteredTransport(transport, root, run_id="test1")
    assert wrapped(Request(URL), 1) == b"{}"
    event = read_attempts(root)[0]
    assert event["outcome"] == "received" and event["billed_cost"] is None
    assert len(list(root.glob("*/*.json"))) == 2
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.glob("*/*.json"))


@pytest.mark.parametrize(
    "url",
    [
        "https://ce.us-east-1.amazonaws.com",
        "https://sts.us-east-1.amazonaws.com",
        "https://console.vast.ai.evil/api/v0/charges/",
        "http://console.vast.ai/api/v0/charges/",
        "https://console.vast.ai:wrong/api/v0/charges/",
    ],
)
def test_denied_endpoint_locally_visible_zero_dispatch(tmp_path, url):
    calls = []
    wrapped = MeteredTransport(lambda *a: calls.append(a), tmp_path / "events")
    with pytest.raises(MeteredCallEvidenceError):
        wrapped(Request(url), 1)
    assert calls == []
    assert read_attempts(tmp_path / "events")[0]["outcome"] == "denied"


def test_error_crash_unknown_and_redaction(tmp_path):
    root = tmp_path / "events"

    def error(*args):
        raise RuntimeError("Bearer private-key request body")

    wrapped = MeteredTransport(error, root)
    with pytest.raises(RuntimeError):
        wrapped(Request(URL, headers={"Authorization": "Bearer private-key"}), 1)

    def crash(*args):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        MeteredTransport(crash, root)(Request(URL), 1)
    assert sorted(e["outcome"] for e in read_attempts(root)) == ["error", "unresolved"]
    assert summarize_attempts(root)["warnings"] == ["metered_attempt_outcome_requires_attention"]
    assert summarize_attempts(root)["billed_cost"] is None
    assert "private-key" not in "".join(p.read_text() for p in root.glob("*/*.json"))


def test_page_retry_and_run_identities_are_immutable(tmp_path):
    root = tmp_path / "events"
    wrapped = MeteredTransport(lambda *a: b"{}", root, run_id="first")
    for url in [URL, URL, URL.replace("page=1", "page=2")]:
        wrapped(Request(url), 1)
    MeteredTransport(lambda *a: b"{}", root, run_id="second")(Request(URL), 1)
    events = sorted(read_attempts(root), key=lambda e: (e["run_id"], e["attempt_sequence"]))
    assert len({e["attempt_id"] for e in events}) == 4
    assert [e["retry_index"] for e in events] == [0, 1, 0, 0]
    assert events[0]["page_identity"] == events[1]["page_identity"] != events[2]["page_identity"]
    before = {p: p.read_bytes() for p in root.glob("*/*.json")}
    read_attempts(root)
    assert before == {p: p.read_bytes() for p in root.glob("*/*.json")}


def test_unpersistable_or_public_directory_never_dispatches(tmp_path, monkeypatch):
    root = tmp_path / "events"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    calls = []
    wrapped = MeteredTransport(lambda *a: calls.append(a), root)
    with pytest.raises(MeteredCallEvidenceError):
        wrapped(Request(URL), 1)
    assert calls == []
    root.chmod(0o700)
    monkeypatch.setattr(
        "blueprint_pipeline.metered_call_evidence._immutable_write",
        lambda *a: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        wrapped(Request(URL), 1)
    assert calls == []
