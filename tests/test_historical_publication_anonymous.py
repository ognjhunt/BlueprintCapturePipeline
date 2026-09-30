"""ADP-009D/day28: pure anonymous-acquisition ownership protocol negatives.

All metadata here is explicit syscall simulation, never Linux/native action
proof. Genuine process death under installed rights is tested separately.
"""
# Covers: src/blueprint_pipeline/control_plane_lane_historical_publication.py
import errno
import stat
from types import SimpleNamespace

import pytest

from tests.test_historical_generation_authority import historical_installation  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# ruff: noqa: F811

from blueprint_pipeline import control_plane_lane_historical_publication as publication
from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def protocol(monkeypatch, *, named_changes=None, opened_changes=None):
    trace = []
    parent = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFDIR | 0o700, st_uid=0, st_gid=0)
    values = dict(st_dev=1, st_ino=3, st_mode=stat.S_IFREG | 0o600, st_uid=0, st_gid=0,
                  st_nlink=0, st_size=0, st_mtime_ns=1, st_ctime_ns=1)
    named = SimpleNamespace(**(values | (named_changes or {})))
    opened = SimpleNamespace(**(values | (opened_changes or {})))
    files = _BirthFiles(ReferenceCollectionBudget(monotonic=lambda: 0))
    files.owned[7] = (1, 2, stat.S_IFDIR)
    files.location = lambda fd: trace.append(('parent', fd))
    monkeypatch.setattr(publication.sys, 'platform', 'linux')
    monkeypatch.setattr(publication.os, 'O_TMPFILE', 0o20200000, raising=False)
    def create(path, flags, mode, *, dir_fd):
        trace.append(('open', path, flags, mode, dir_fd))
        return 17
    def observed(path, *, follow_symlinks):
        trace.append(('proc', path, follow_symlinks))
        return named
    def descriptor(fd):
        trace.append(('fstat', fd))
        return parent if fd == 7 else opened
    def slots():
        trace.append(('slots',))
        return {7, 8}
    def view_stat(fd):
        return publication.os.stat('/proc/self/fd/' + str(fd), follow_symlinks=True)
    files._test_view = SimpleNamespace(fd=8, slots=slots, stat=view_stat)
    monkeypatch.setattr(publication.os, 'open', create)
    monkeypatch.setattr(publication.os, 'stat', observed)
    monkeypatch.setattr(publication.os, 'fstat', descriptor)
    return files, trace


def test_direct_creation_is_independently_observed_before_its_first_fstat(monkeypatch):
    files, trace = protocol(monkeypatch)
    fd = publication._create(files, 7, files._test_view)
    assert fd == 17 and files.owned[fd] == (1, 3, stat.S_IFREG)
    assert trace.index(('proc', '/proc/self/fd/17', True)) < trace.index(('fstat', 17))
    creation = next(row for row in trace if row[0] == 'open')
    assert creation[1] == '.' and creation[3:] == (0o600, 7)
    assert creation[2] & publication.os.O_TMPFILE == publication.os.O_TMPFILE
    assert not creation[2] & publication.os.O_EXCL
    assert fd not in files.bindings  # No fabricated regular pathname binding.
    with pytest.raises(ValueError, match='ownership_unproven'):
        files.adopt(fd)


@pytest.mark.parametrize('changes', [dict(st_uid=1), dict(st_gid=1), dict(st_mode=stat.S_IFLNK | 0o600),
    dict(st_mode=stat.S_IFREG | 0o644), dict(st_nlink=1), dict(st_size=1), dict(st_dev=2)])
def test_unsafe_proc_observation_is_never_adopted_or_closed(monkeypatch, changes):
    files, trace = protocol(monkeypatch, named_changes=changes)
    monkeypatch.setattr(publication.os, 'close', lambda fd: pytest.fail('unknown token closed'))
    with pytest.raises(ValueError, match='ownership_unproven'):
        publication._create(files, 7, files._test_view)
    assert files.unresolved == 1 and 17 not in files.owned
    assert ('fstat', 17) not in trace


@pytest.mark.parametrize('changes', [dict(st_ino=9), dict(st_mode=stat.S_IFDIR | 0o600),
                                    dict(st_size=2), dict(st_nlink=1)])
def test_changed_descriptor_cannot_establish_creation_proof(monkeypatch, changes):
    files, _ = protocol(monkeypatch, opened_changes=changes)
    with pytest.raises(ValueError, match='ownership_unproven'):
        publication._create(files, 7, files._test_view)


def test_collision_is_not_adopted_or_closed(monkeypatch):
    files, trace = protocol(monkeypatch)
    files.owned[17] = (8, 9, stat.S_IFREG)
    with pytest.raises(ValueError, match='descriptor_collision'):
        publication._create(files, 7, files._test_view)
    assert files.owned[17] == (8, 9, stat.S_IFREG) and files.unresolved == 1
    assert ('proc', '/proc/self/fd/17', True) not in trace


def test_unsupported_filesystem_has_no_named_publication_fallback(monkeypatch):
    files, trace = protocol(monkeypatch)
    def unavailable(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, 'simulated unsupported anonymous creation')
    monkeypatch.setattr(publication.os, 'open', unavailable)
    monkeypatch.setattr(publication.os, 'link', lambda *a, **kw: pytest.fail('fallback link'))
    with pytest.raises(ValueError, match='anonymous_unavailable'):
        publication._create(files, 7, files._test_view)
    assert set(files.owned) == {7} and files.unresolved == 0
    assert not any(row[0] == 'proc' for row in trace)


def test_mac_is_not_linux_anonymous_creation_proof(monkeypatch):
    files, trace = protocol(monkeypatch)
    monkeypatch.setattr(publication.sys, 'platform', 'darwin')
    with pytest.raises(ValueError, match='anonymous_unavailable'):
        publication._create(files, 7, files._test_view)
    assert trace == []


def test_exhausted_owned_registry_refuses_before_creation(monkeypatch):
    files, trace = protocol(monkeypatch)
    files.owned.update({fd: (1, fd, stat.S_IFREG) for fd in range(100, 203)})
    with pytest.raises(ValueError, match='resource_exhausted'):
        publication._create(files, 7, files._test_view)
    assert not any(row[0] == 'open' for row in trace)


def test_failed_independent_proc_observation_does_not_close_unknown_token(monkeypatch):
    files, _ = protocol(monkeypatch)
    def unknown(*args, **kwargs):
        raise OSError(errno.EIO, 'simulated inaccessible kernel observation')
    monkeypatch.setattr(publication.os, 'stat', unknown)
    monkeypatch.setattr(publication.os, 'close', lambda fd: pytest.fail('unknown token closed'))
    with pytest.raises(ValueError, match='ownership_unproven'):
        publication._create(files, 7, files._test_view)
    assert 17 not in files.owned and files.unresolved == 1



def test_borrowed_unregistered_anonymous_fd_is_never_adopted_written_or_closed(monkeypatch):
    files, trace = protocol(monkeypatch)
    files._test_view.slots = lambda: {7, 8, 17}
    monkeypatch.setattr(publication.os, 'write', lambda *a: pytest.fail('borrowed token written'))
    monkeypatch.setattr(publication.os, 'close', lambda *a: pytest.fail('borrowed token closed'))
    with pytest.raises(ValueError, match='descriptor_collision'):
        publication._create(files, 7, files._test_view)
    assert 17 not in files.owned and files.unresolved == 1
    assert ('proc', '/proc/self/fd/17', True) not in trace





def test_historical_metadata_checkpoint_supports_declared_manifest_reads(historical_installation):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_generation import MAX_MANIFEST_BYTES
    operation = authority._Operation(1030, lambda: 0)
    with authority._session(historical_installation[0], operation) as (files, _, _):
        # One original selection, two publication comparisons and subsequent
        # protected readback fit with conserved acquisition counters.
        needed = 4 * MAX_MANIFEST_BYTES
        assert min(files.raw_cap, files.budget.limits['raw_bytes']) - files.budget.counts['raw_bytes'] >= needed



@pytest.mark.parametrize('kind,name,offset', [('event', 'e-00001.json', -1),
    ('event', 'e-00001.json', 0), ('historical_restore_snapshot', 'restore.snapshot.json', -1),
    ('historical_restore_snapshot', 'restore.snapshot.json', 0)])
def test_complete_read_and_eof_reservation_precedes_any_creation(monkeypatch, kind, name, offset):
    files, _ = protocol(monkeypatch)
    payload = b'{"execution_authorized":false}\n'
    extra = 6 * publication._CAPS['event'] + 80 if kind == 'historical_restore_snapshot' else 0
    files.raw_cap = 3 * len(payload) + extra + offset
    def absent(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr(publication.os, 'stat', absent)
    monkeypatch.setattr(publication, 'CreationProcView', lambda *a: pytest.fail('creation before complete read admission'))
    with pytest.raises(ValueError, match='resource_exhausted'):
        publication._publish(files, 7, name, payload, kind=kind)
    assert files.budget.counts['raw_bytes'] == 0 and set(files.owned) == {7}
