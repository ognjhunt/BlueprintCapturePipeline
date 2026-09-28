# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_g1_lifetime_adapter.py
#   src/blueprint_pipeline/control_plane_scratch_lifetime.py
"""Bounded dedicated admission channels cannot promote an unproved worker."""

import os

import pytest

from blueprint_pipeline import control_plane_g1_lifetime_adapter as adapter
from blueprint_pipeline.control_plane_lane_scratch import LaneScratchError
from tests.test_leased_scratch_lifetime import folder, open_use


def test_child_proof_binds_exact_request_output_and_lifetime(tmp_path):
    root, path = folder(tmp_path)
    with open_use(root) as use:
        proof = adapter.worker_proof(use, output=path / "candidate", request_digest="sha256:" + "a" * 64)
        with adapter.adopt_worker_proof(os.dup(use.fd), proof, output=path / "candidate",
                                       request_digest="sha256:" + "a" * 64, now=lambda: 110) as child:
            assert child.identity == use.identity
        use.check()


@pytest.mark.parametrize("fault", ["request", "output", "inode", "owner", "protocol"])
def test_wrong_child_proof_refuses_before_any_output_write(tmp_path, fault):
    root, path = folder(tmp_path)
    with open_use(root) as use:
        proof = adapter.worker_proof(use, output=path / "candidate", request_digest="sha256:" + "a" * 64)
        if fault == "request":
            proof["request_digest"] = "sha256:" + "b" * 64
        elif fault == "output":
            proof["output"] = str(path / "other")
        elif fault == "inode":
            proof["identity"]["inodes"][-1] = (1, 2)
        elif fault == "owner":
            proof["identity"]["owner"] = "foreign"
        else:
            proof["identity"]["consumer_lifetime_contract"] = "wrong"
        with pytest.raises(LaneScratchError, match="lifetime|handshake"):
            adapter.adopt_worker_proof(os.dup(use.fd), proof, output=path / "candidate",
                                       request_digest="sha256:" + "a" * 64, now=lambda: 110)
        assert not (path / "candidate").exists()


@pytest.mark.parametrize("raw", [b"", b"x" * 4097, b"not-json\n", b'{}\nextra'])
def test_channel_refuses_missing_oversized_malformed_or_extra_message(raw):
    read, write = os.pipe()
    try:
        os.write(write, raw)
        os.close(write)
        write = None
        with pytest.raises(LaneScratchError, match="handshake"):
            adapter.read_message(read, timeout=0.01)
    finally:
        os.close(read)
        if write is not None:
            os.close(write)


def test_channel_valid_bounded_message_round_trip():
    read, write = os.pipe()
    try:
        adapter.write_message(write, {"status": "ready"})
        assert adapter.read_message(read, timeout=0.01) == {"status": "ready"}
    finally:
        os.close(read)
        os.close(write)


def test_two_startup_messages_share_one_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(adapter.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(adapter.select, 'select', lambda *args: ([7], [], []))
    monkeypatch.setattr(os, 'read', lambda *args: b'{"status":"ready"}\n')
    assert adapter.read_message(7, _deadline=105) == {'status': 'ready'}
    clock[0] = 106
    monkeypatch.setattr(os, 'read', lambda *args: pytest.fail('read after startup deadline'))
    with pytest.raises(LaneScratchError, match='handshake_timeout'):
        adapter.read_message(7, _deadline=105)


def test_channel_encoding_bound_precedes_encoder(monkeypatch):
    monkeypatch.setattr(adapter.json, "dumps", lambda *args, **kwargs: pytest.fail("encoded oversized proof"))
    with pytest.raises(LaneScratchError, match="handshake"):
        adapter.write_message(-1, {"oversized": "x" * 4097})


@pytest.mark.parametrize("option", ["timeout", "_deadline"])
@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400)], ids=["huge-positive", "huge-negative"])
def test_channel_huge_numeric_options_refuse_before_io(monkeypatch, option, value):
    monkeypatch.setattr(adapter.select, "select", lambda *args: pytest.fail("selected on invalid channel option"))
    monkeypatch.setattr(os, "read", lambda *args: pytest.fail("read on invalid channel option"))
    with pytest.raises(LaneScratchError) as caught:
        adapter.read_message(-1, **{option: value})
    assert caught.value.args == ("lane_scratch_handshake_invalid",)


@pytest.mark.parametrize("text", ["\ud800", "\udfff"], ids=["high-surrogate", "low-surrogate"])
@pytest.mark.parametrize("position", ["key", "value"])
def test_channel_outbound_malformed_unicode_refuses_before_encode_or_write(monkeypatch, text, position):
    monkeypatch.setattr(adapter.json, "dumps", lambda *args, **kwargs: pytest.fail("encoded malformed channel text"))
    monkeypatch.setattr(os, "write", lambda *args: pytest.fail("wrote malformed channel text"))
    value = {text: "valid"} if position == "key" else {"valid": text}
    with pytest.raises(LaneScratchError) as caught:
        adapter.write_message(-1, value)
    assert caught.value.args == ("lane_scratch_handshake_invalid",)


def test_unlocked_inherited_descriptor_establishes_separate_shared_authority(tmp_path):
    from blueprint_pipeline.control_plane_scratch_lifetime import LeasedScratchUse
    root, path = folder(tmp_path)
    use = open_use(root)
    proof = adapter.worker_proof(use, output=path / "candidate", request_digest="sha256:" + "a" * 64)
    use.close()
    unlocked = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    with adapter.adopt_worker_proof(unlocked, proof, output=path / "candidate",
                                   request_digest="sha256:" + "a" * 64, now=lambda: 110):
        with pytest.raises(LaneScratchError, match="consumer_busy"):
            LeasedScratchUse.probe(path, now=lambda: 110)


def test_second_pipe_failure_finalizes_first_owned_pair(tmp_path, monkeypatch):
    from pathlib import Path
    original_pipe, original_stat = os.pipe, os.fstat
    opened = []
    def pipe():
        if opened:
            raise OSError("private pipe creation fault")
        pair = original_pipe()
        opened.extend(pair)
        return pair
    monkeypatch.setattr(adapter, "worker_proof", lambda *args, **kwargs: {})
    monkeypatch.setattr(os, "pipe", pipe)
    with pytest.raises(LaneScratchError, match="handshake"):
        adapter.controlled_worker_run(executable=Path("/fake/python"), request=Path("/request"),
                                      output=Path("/output"), request_digest="digest", use=object(), stdout=None, timeout=10)
    for fd in opened:
        with pytest.raises(OSError):
            original_stat(fd)


def test_popen_failure_and_one_shot_close_still_finalizes_every_pipe(monkeypatch):
    from pathlib import Path
    original_pipe, original_close, original_stat = os.pipe, os.close, os.fstat
    opened, failed = [], []
    def pipe():
        pair = original_pipe()
        opened.extend(pair)
        return pair
    def close(fd):
        if fd in opened and not failed:
            failed.append(fd)
            raise OSError("private definite close failure")
        return original_close(fd)
    monkeypatch.setattr(adapter, "worker_proof", lambda *args, **kwargs: {})
    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private launch fault")))
    with pytest.raises(LaneScratchError, match="handshake"):
        adapter.controlled_worker_run(executable=Path("/fake/python"), request=Path("/request"),
                                      output=Path("/output"), request_digest="digest", use=type("Use", (), {"fd": 999})(), stdout=None, timeout=10)
    for fd in opened:
        with pytest.raises(OSError):
            original_stat(fd)


def test_invalid_proof_closed_descriptor_has_fixed_refusal(tmp_path):
    read, write = os.pipe()
    os.close(read)
    os.close(write)
    with pytest.raises(LaneScratchError, match="handshake|ownership"):
        adapter.adopt_worker_proof(read, {}, output=tmp_path / "candidate", request_digest="digest")


def test_channel_duplicate_keys_are_refused():
    read, write = os.pipe()
    try:
        os.write(write, b'{"status":"wrong","status":"ready"}\n')
        with pytest.raises(LaneScratchError, match="handshake"):
            adapter.read_message(read, timeout=0.01)
    finally:
        os.close(read)
        os.close(write)


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["none", "wrong_ack", "timeout"])
def test_controlled_adapter_preserves_exit_timeout_and_fixed_ack_semantics(tmp_path, monkeypatch, fault):
    import subprocess
    import sys
    from pathlib import Path
    root, path = folder(tmp_path)
    process = []
    class FakeProcess:
        def __init__(self, args, **options):
            self.args, self.killed = args, False
            assert args[:3] == [sys.executable, "-m", "blueprint_pipeline.native_g1_development_worker"]
            self.child_input = os.dup(options["pass_fds"][-2])
            ready_fd = options["pass_fds"][-1]
            adapter.write_message(ready_fd, {"status": "ready", "request_digest": "wrong" if fault == "wrong_ack" else "digest"})
            process.append(self)
        def poll(self):
            return 0 if self.killed else None
        def kill(self):
            self.killed = True
        def wait(self, timeout):
            if fault == "timeout" and not self.killed:
                raise subprocess.TimeoutExpired(self.args, timeout)
            os.close(self.child_input)
            return 7
    monkeypatch.setattr(adapter.subprocess, "Popen", FakeProcess)
    with open_use(root) as use:
        options = dict(executable=Path(sys.executable), request=tmp_path / "request", output=path / "candidate",
                       request_digest="digest", use=use, stdout=None, timeout=0.01)
        if fault == "none":
            assert adapter.controlled_worker_run(**options).returncode == 7
        elif fault == "wrong_ack":
            with pytest.raises(LaneScratchError, match="handshake_invalid"):
                adapter.controlled_worker_run(**options)
            assert process[0].killed
        else:
            with pytest.raises(subprocess.TimeoutExpired):
                adapter.controlled_worker_run(**options)
            assert process[0].killed
        use.check()


def test_channel_invalid_utf8_message_has_fixed_refusal():
    read, write = os.pipe()
    try:
        os.write(write, b'{"status":"\xff"}\n')
        with pytest.raises(LaneScratchError, match="handshake"):
            adapter.read_message(read, timeout=0.01)
    finally:
        os.close(read)
        os.close(write)
