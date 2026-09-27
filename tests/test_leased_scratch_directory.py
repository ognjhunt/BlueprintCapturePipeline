"""Only descriptor-backed directory operations are protected by a live lease."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_leased_scratch.py

from __future__ import annotations

import json
import os

import pytest

from blueprint_pipeline import control_plane_lane_scratch as leases


def _fixture(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    folder = leases.create_lane_scratch(
        "arena", "attempt", root=root, owner="owner-a", run_ref="run-a",
        reason="diagnostic", class_intent="evidence", cleanup="owner_review",
        ttl_seconds=100, now=lambda: 1000,
    )
    return root, folder


def _open(root, **overrides):
    from blueprint_pipeline.control_plane_leased_scratch import LeasedScratchDirectory

    options = dict(root=root, lane="arena", name="attempt", owner="owner-a",
                   run_ref="run-a", now=lambda: 1001)
    options.update(overrides)
    return LeasedScratchDirectory.open(**options)


def test_live_capability_creates_only_relative_payload_and_closes(tmp_path):
    root, folder = _fixture(tmp_path)
    with _open(root) as handle:
        assert handle.mkdir("packet/staged", parents=True) == folder / "packet/staged"
        assert handle.mkdir("packet/staged", parents=True, exist_ok=True).is_dir()
        with pytest.raises(FileExistsError):
            handle.mkdir("packet")
        assert handle.path == folder
    with pytest.raises(leases.LaneScratchError, match="capability_closed"):
        handle.mkdir("closed")


@pytest.mark.parametrize("relative", ["../sibling", "/tmp/escape", "a/../b", ".", "", "a//b", "a/./b"])
def test_payload_path_refuses_escape_and_ambiguous_components(tmp_path, relative):
    root, folder = _fixture(tmp_path)
    with _open(root) as handle:
        with pytest.raises(leases.LaneScratchError, match="payload_path_invalid"):
            handle.mkdir(relative, parents=True)
    assert sorted(path.name for path in folder.iterdir()) == [leases.LEASE_FILE]


def test_payload_path_refuses_symlink_without_touching_target(tmp_path):
    root, folder = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (folder / "link").symlink_to(outside, target_is_directory=True)
    with _open(root) as handle:
        with pytest.raises(leases.LaneScratchError):
            handle.mkdir("link/child", parents=True)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("changes", [
    {"owner": "owner-b"}, {"run_ref": "different"}, {"run_ref": None, "scene_ref": "run-a"},
    {"now": lambda: 1100}, {"run_ref": None},
])
def test_open_requires_exact_owner_reference_type_and_active_expiry(tmp_path, changes):
    root, _folder = _fixture(tmp_path)
    with pytest.raises(leases.LaneScratchError):
        _open(root, **changes)


@pytest.mark.parametrize("replacement", ["missing", "malformed", "symlink"])
def test_bad_lease_refuses_open_and_next_operation(tmp_path, replacement):
    root, folder = _fixture(tmp_path)
    with _open(root) as handle:
        path = folder / leases.LEASE_FILE
        original = path.read_bytes()
        path.unlink()
        if replacement == "malformed":
            path.write_text("{}")
        elif replacement == "symlink":
            target = tmp_path / "other-lease"
            target.write_bytes(original)
            path.symlink_to(target)
        with pytest.raises(leases.LaneScratchError):
            handle.mkdir("refused")
        with pytest.raises(leases.LaneScratchError):
            _open(root)
    assert not (folder / "refused").exists()


def test_release_and_expiry_after_open_refuse_payload(tmp_path):
    root, folder = _fixture(tmp_path)
    clock = [1001]
    with _open(root, now=lambda: clock[0]) as handle:
        clock[0] = 1100
        with pytest.raises(leases.LaneScratchError, match="inactive"):
            handle.mkdir("expired")
        clock[0] = 1002
        leases.release_lane_scratch(root=root, lane="arena", name="attempt", owner="owner-a",
                                   expected_digest=handle.lease_digest, now=lambda: 1002)
        with pytest.raises(leases.LaneScratchError):
            handle.mkdir("released")
        with pytest.raises(leases.LaneScratchError):
            handle.refresh()
    assert not (folder / "expired").exists() and not (folder / "released").exists()


def test_renewal_requires_explicit_refresh_with_same_reference(tmp_path):
    root, folder = _fixture(tmp_path)
    with _open(root) as handle:
        renewed = leases.renew_lane_scratch(
            root=root, lane="arena", name="attempt", owner="owner-a",
            expected_digest=handle.lease_digest, ttl_seconds=200, now=lambda: 1001,
        )
        with pytest.raises(leases.LaneScratchError, match="lease_changed"):
            handle.mkdir("stale")
        handle.refresh()
        assert handle.lease_digest == renewed["lease_digest"]
        handle.mkdir("fresh")
        changed = dict(renewed)
        changed.pop("run_ref")
        changed["scene_ref"] = "run-a"
        changed = leases._seal(changed)
        (folder / leases.LEASE_FILE).write_text(json.dumps(changed))
        with pytest.raises(leases.LaneScratchError, match="identity"):
            handle.refresh()
    assert not (folder / "stale").exists()


@pytest.mark.parametrize("level", ["root", "lane", "folder"])
@pytest.mark.parametrize("symlink", [False, True])
def test_replaced_root_lane_or_folder_refuses_before_payload(tmp_path, level, symlink):
    root, folder = _fixture(tmp_path)
    with _open(root) as handle:
        target = {"root": root, "lane": folder.parent, "folder": folder}[level]
        moved = target.with_name(target.name + "-old")
        target.rename(moved)
        if symlink:
            target.symlink_to(moved, target_is_directory=True)
        else:
            target.mkdir()
        with pytest.raises(leases.LaneScratchError):
            handle.mkdir("wrong")
        with pytest.raises(leases.LaneScratchError):
            handle.refresh()
        retained_folder = {"root": moved / "arena/attempt", "lane": moved / "attempt",
                           "folder": moved}[level]
        assert not (retained_folder / "wrong").exists()
        assert not (target / "wrong").exists()
        if not symlink:
            assert list(target.iterdir()) == []


def test_root_ancestor_symlink_is_refused(tmp_path):
    root, _folder = _fixture(tmp_path)
    link = tmp_path / "linked-parent"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(leases.LaneScratchError):
        _open(link / root.name)


@pytest.mark.parametrize("failure", ["owner", "reference", "missing", "malformed", "symlink", "released", "expired"])
def test_all_opened_descriptors_close_on_open_failure(tmp_path, monkeypatch, failure):
    root, folder = _fixture(tmp_path)
    original_open = os.open
    opened = []

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", tracked_open)
    options = {}
    if failure == "owner":
        options["owner"] = "owner-b"
    elif failure == "reference":
        options.update(run_ref=None, scene_ref="run-a")
    elif failure == "expired":
        options["now"] = lambda: 1100
    elif failure == "released":
        lease = json.loads((folder / leases.LEASE_FILE).read_text())
        leases.release_lane_scratch(root=root, lane="arena", name="attempt", owner="owner-a",
                                   expected_digest=lease["lease_digest"], now=lambda: 1001)
    else:
        (folder / leases.LEASE_FILE).unlink()
        if failure == "malformed":
            (folder / leases.LEASE_FILE).write_text("{}")
        elif failure == "symlink":
            (folder / leases.LEASE_FILE).symlink_to(tmp_path / "outside")
    with pytest.raises(leases.LaneScratchError):
        _open(root, **options)
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_all_descriptors_close_after_injected_payload_error(tmp_path, monkeypatch):
    root, folder = _fixture(tmp_path)
    original_open, opened = os.open, []

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", tracked_open)
    with _open(root) as handle:
        monkeypatch.setattr(os, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("injected")))
        with pytest.raises(OSError):
            handle.mkdir("error")
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert not (folder / "error").exists()


def test_create_capability_preserves_atomic_creation_and_no_overwrite(tmp_path):
    from blueprint_pipeline.control_plane_leased_scratch import create_leased_lane_scratch

    root = tmp_path / "lanes"
    root.mkdir()
    options = dict(root=root, owner="owner-a", run_ref="run-a", ttl_seconds=100,
                   reason="diagnostic", class_intent="evidence", cleanup="owner_review",
                   now=lambda: 1000)
    with create_leased_lane_scratch("arena", "attempt", **options) as handle:
        assert (handle.path / leases.LEASE_FILE).is_file()
        handle.mkdir("payload")
    with pytest.raises(leases.LaneScratchError, match="exists"):
        create_leased_lane_scratch("arena", "attempt", **options)


@pytest.mark.parametrize("failed_flag", [leases.fcntl.LOCK_EX, leases.fcntl.LOCK_UN])
def test_coordination_lock_descriptor_closes_when_flock_raises(tmp_path, monkeypatch, failed_flag):
    root, _folder = _fixture(tmp_path)
    opened, original_open, original_flock = [], os.open, leases.fcntl.flock

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def injected_flock(fd, flag):
        if flag == failed_flag:
            raise OSError("injected flock failure")
        return original_flock(fd, flag)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(leases.fcntl, "flock", injected_flock)
    with pytest.raises(leases.LaneScratchError):
        _open(root)
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
