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


def test_channel_encoding_bound_precedes_encoder(monkeypatch):
    monkeypatch.setattr(adapter.json, "dumps", lambda *args, **kwargs: pytest.fail("encoded oversized proof"))
    with pytest.raises(LaneScratchError, match="handshake"):
        adapter.write_message(-1, {"oversized": "x" * 4097})


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
