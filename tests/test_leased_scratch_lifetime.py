# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_scratch_lifetime.py
#   src/blueprint_pipeline/control_plane_lane_scratch.py
#   src/blueprint_pipeline/control_plane_lane_scratch_retention.py
"""Cooperating lifetime authority is inode-bound, optional and close-only."""

import fcntl
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_lane_scratch as leases
from blueprint_pipeline import control_plane_scratch_lifetime as lifetime


def folder(tmp_path, *, enrolled=True):
    root = tmp_path / "lanes"
    root.mkdir()
    path = leases.create_lane_scratch("g1", "pair", root=root, owner="owner", run_ref="run",
                                     reason="fixture", class_intent="evidence", cleanup="owner_review",
                                     ttl_seconds=100, now=lambda: 100,
                                     **({"consumer_lifetime_contract": lifetime.PROTOCOL} if enrolled else {}))
    return root, path


def open_use(root, **kwargs):
    return lifetime.LeasedScratchUse.open(root=root, lane="g1", name="pair", owner="owner",
                                         run_ref="run", now=lambda: 110, **kwargs)


def test_optional_metadata_preserves_legacy_lease_and_strict_reader(tmp_path):
    from blueprint_pipeline.control_plane_lane_scratch_retention import _lease
    root, path = folder(tmp_path, enrolled=False)
    raw = (path / leases.LEASE_FILE).read_bytes()
    assert "consumer_lifetime_contract" not in json.loads(raw)
    assert _lease(raw, "g1", "pair")["cleanup"] == "owner_review"
    with pytest.raises(leases.LaneScratchError, match="participation_unproven"):
        open_use(root)


def test_enrolled_lease_is_still_evidence_and_strictly_readable(tmp_path):
    from blueprint_pipeline.control_plane_lane_scratch_retention import _lease
    _, path = folder(tmp_path)
    lease = _lease((path / leases.LEASE_FILE).read_bytes(), "g1", "pair")
    assert lease["consumer_lifetime_contract"] == lifetime.PROTOCOL
    assert (lease["class_intent"], lease["cleanup"]) == ("evidence", "owner_review")


def test_shared_use_blocks_probe_and_releases_root_for_nested_operations(tmp_path):
    root, path = folder(tmp_path)
    with open_use(root) as use:
        with pytest.raises(leases.LaneScratchError, match="consumer_busy"):
            lifetime.LeasedScratchUse.probe(path, now=lambda: 110)
        assert use.mkdir("diagnostics") == path / "diagnostics"
    with lifetime.LeasedScratchUse.probe(path, now=lambda: 110) as probe:
        assert probe.identity["lease_digest"]
        assert probe.check() is None


@pytest.mark.parametrize("reason", ["missing", "symlink", "busy"])
def test_probe_never_creates_or_waits_for_root_lock(tmp_path, reason):
    root, path = folder(tmp_path)
    lock = root / ".lane-scratch.lock"
    descriptor = None
    if reason == "missing":
        lock.unlink()
    elif reason == "symlink":
        lock.unlink()
        lock.symlink_to(tmp_path / "foreign")
    else:
        descriptor = os.open(lock, os.O_RDONLY)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        with pytest.raises(leases.LaneScratchError, match="root_lock"):
            lifetime.LeasedScratchUse.probe(path, now=lambda: 110)
        if reason == "missing":
            assert not lock.exists()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def test_moved_target_keeps_original_authority_and_detects_visible_substitution(tmp_path):
    root, path = folder(tmp_path)
    with open_use(root) as use:
        moved = path.with_name("moved")
        path.rename(moved)
        path.mkdir()
        fd = os.open(moved, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(leases.LaneScratchError, match="path_changed"):
                use.check()
        finally:
            os.close(fd)


def test_inherited_description_is_close_only(tmp_path, monkeypatch):
    root, path = folder(tmp_path)
    use = open_use(root)
    duplicate = os.dup(use.fd)
    original = fcntl.flock
    def no_unlock(fd, operation):
        assert fd != use.fd or operation != fcntl.LOCK_UN
        return original(fd, operation)
    monkeypatch.setattr(fcntl, "flock", no_unlock)
    use.close()
    try:
        with pytest.raises(leases.LaneScratchError, match="consumer_busy"):
            lifetime.LeasedScratchUse.probe(path, now=lambda: 110)
    finally:
        os.close(duplicate)
    with lifetime.LeasedScratchUse.probe(path, now=lambda: 110):
        pass


@pytest.mark.parametrize("change", [{"owner": "foreign"}, {"run_ref": "foreign"},
                                   {"scene_ref": "run", "run_ref": None}])
def test_exact_identity_is_required_before_admission(tmp_path, change):
    root, _ = folder(tmp_path)
    options = {"root": root, "lane": "g1", "name": "pair", "owner": "owner", "run_ref": "run", "now": lambda: 110}
    options.update(change)
    with pytest.raises(leases.LaneScratchError, match="identity"):
        lifetime.LeasedScratchUse.open(**options)


def test_expiry_does_not_unlock_admitted_use_but_refuses_new_admission(tmp_path):
    root, path = folder(tmp_path)
    with open_use(root) as use:
        use.now = lambda: 201
        assert use.check() is None
        with pytest.raises(leases.LaneScratchError, match="inactive"):
            lifetime.LeasedScratchUse.open(root=root, lane="g1", name="pair", owner="owner", run_ref="run", now=lambda: 201)
        with pytest.raises(leases.LaneScratchError, match="consumer_busy"):
            lifetime.LeasedScratchUse.probe(path, now=lambda: 201)


@pytest.mark.parametrize("value", [None, "unknown.v1", True, []])
def test_unknown_protocol_refused_before_constructor_mutation(tmp_path, value):
    root = tmp_path / "lanes"
    root.mkdir()
    with pytest.raises(leases.LaneScratchError, match="protocol_invalid"):
        leases.create_lane_scratch("g1", "pair", root=root, owner="owner", run_ref="run", reason="fixture",
                                   class_intent="evidence", cleanup="owner_review", ttl_seconds=100,
                                   consumer_lifetime_contract=value)
    assert not (root / "g1").exists()


def test_create_retains_root_coordination_until_target_shared_admission(tmp_path, monkeypatch):
    root = tmp_path / "lanes"
    root.mkdir()
    original = fcntl.flock
    observed = []
    def trace(fd, operation):
        if operation == fcntl.LOCK_SH | fcntl.LOCK_NB:
            other = os.open(root / ".lane-scratch.lock", os.O_RDONLY)
            try:
                with pytest.raises(BlockingIOError):
                    original(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed.append("continuous_root_ownership")
            finally:
                os.close(other)
        return original(fd, operation)
    monkeypatch.setattr(fcntl, "flock", trace)
    with lifetime.LeasedScratchUse.create(root=root, lane="g1", name="pair", owner="owner", run_ref="run",
                                         ttl_seconds=100, now=lambda: 100):
        assert observed == ["continuous_root_ownership"]


def test_borrow_registers_duplicate_before_its_first_fstat_fault(tmp_path, monkeypatch):
    root, path = folder(tmp_path)
    with open_use(root) as use:
        original_dup, original_stat = os.dup, os.fstat
        duplicate = []
        failed = []
        def dup(fd):
            result = original_dup(fd)
            duplicate.append(result)
            return result
        def info(fd):
            if fd in duplicate and not failed:
                failed.append(fd)
                raise OSError("injected identity failure")
            return original_stat(fd)
        monkeypatch.setattr(os, "dup", dup)
        monkeypatch.setattr(os, "fstat", info)
        with pytest.raises(leases.LaneScratchError, match="descriptor_invalid"):
            use.borrow(path / "candidate")
        with pytest.raises(OSError):
            original_stat(duplicate[0])


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["close", "exception", "timeout", "death"])
def test_fake_child_retains_shared_description_after_parent_lifetime_ends(tmp_path, fault):
    root, path = folder(tmp_path)
    ready_read, ready_write = os.pipe()
    gate_read, gate_write = os.pipe()
    child = None
    use = None
    child_script = "import os,sys; fd,ready,gate=map(int,sys.argv[1:]); os.write(ready,b'R'); os.read(gate,1); os.close(fd); os.write(ready,b'C')"
    try:
        if fault == "death":
            script = ("import os,subprocess,sys; from blueprint_pipeline.control_plane_scratch_lifetime import LeasedScratchUse; "
                      "use=LeasedScratchUse.open(root=sys.argv[1],lane='g1',name='pair',owner='owner',run_ref='run',now=lambda:110); "
                      "ready,gate=map(int,sys.argv[2:]); "
                      "p=subprocess.Popen([sys.executable,'-c'," + repr(child_script) + ",str(use.fd),str(ready),str(gate)],pass_fds=(use.fd,ready,gate)); "
                      "os._exit(0)")
            parent = subprocess.Popen([sys.executable, "-c", script, str(root), str(ready_write), str(gate_read)],
                                      pass_fds=(ready_write, gate_read), env={**os.environ, "PYTHONPATH": str(Path(lifetime.__file__).parents[1])})
            assert parent.wait(timeout=5) == 0
        else:
            use = open_use(root)
            child = subprocess.Popen([sys.executable, "-c", child_script, str(use.fd), str(ready_write), str(gate_read)],
                                     pass_fds=(use.fd, ready_write, gate_read))
            if fault == "timeout":
                with pytest.raises(subprocess.TimeoutExpired):
                    child.wait(timeout=0.01)
            if fault == "exception":
                with pytest.raises(RuntimeError, match="fake failure"):
                    with use:
                        raise RuntimeError("fake failure")
            else:
                use.close()
        assert select.select([ready_read], [], [], 5)[0]
        assert os.read(ready_read, 1) == b"R"
        with pytest.raises(leases.LaneScratchError, match="consumer_busy"):
            lifetime.LeasedScratchUse.probe(path, now=lambda: 110)
        os.write(gate_write, b"P")
        assert select.select([ready_read], [], [], 5)[0]
        assert os.read(ready_read, 1) == b"C"
        if child is not None:
            assert child.wait(timeout=5) == 0
        with lifetime.LeasedScratchUse.probe(path, now=lambda: 110):
            pass
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if use is not None:
            use.close()
        for fd in (ready_read, ready_write, gate_read, gate_write):
            os.close(fd)
