"""Private finite descriptor composition for owner target observations only.

Original identities are proved before adoption and every close. These observed
checks do not provide an atomic fence against arbitrary same-UID threads.
"""
from __future__ import annotations

import errno
import os
import stat

from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch as scratch
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require
from .control_plane_scratch_lifetime import LeasedScratchUse
from .decision_evidence_contracts import canonical_digest


def _typed(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def _matches_tuple(info, expected):
    kind = "directory" if stat.S_ISDIR(info.st_mode) else "regular" if stat.S_ISREG(info.st_mode) else None
    return (info.st_dev, info.st_ino, kind) == (expected["dev"], expected["ino"], expected["type"])


def _lease_metadata(info):
    return dict(dev=info.st_dev, ino=info.st_ino, type="regular" if stat.S_ISREG(info.st_mode) else None,
                mode=stat.S_IMODE(info.st_mode), uid=info.st_uid, gid=info.st_gid,
                nlink=info.st_nlink, size_bytes=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


class _TargetFiles(owners._Files):
    """One actual B, 104 metadata/target slots plus 24 scoped native slots."""
    def __init__(self, budget, *, expected=None):
        super().__init__(budget, raw_cap=2 * 1024 * 1024)
        self.expected = expected
        self.target_owned = set()
        self.target_observations = []
        self.probe_owned, self.probe_groups = {}, []
        self.bindings, self.acquired, self.publication_states = {}, {}, []

    def slot(self):
        self.budget.tick()
        _require(len(self.owned) < 104 and len(self.probe_owned) <= 24
                 and len(self.owned) + len(self.probe_owned) < 128,
                 "owner_target_resource_exhausted")

    def proof(self, fd):
        expected = self.owned.get(fd)
        if expected is None:
            entry = self.probe_owned.get(fd)
            expected = entry[1] if entry is not None else None
        _require(expected is not None, "owner_target_descriptor_ownership_unproven")
        _require(_typed(os.fstat(fd)) == expected, "owner_target_descriptor_changed")
        return expected

    def adopt(self, fd):
        # An arbitrary transferred numeric token has no named original proof.
        raise OwnerTargetVersionError("owner_target_descriptor_ownership_unproven")

    def _adopt_named(self, fd, named, registry, *, group=None):
        _require(type(fd) is int and fd >= 0, "owner_target_descriptor_ownership_unproven")
        if fd in self.owned or fd in self.probe_owned:
            self.unresolved += 1
            raise OwnerTargetVersionError("owner_target_descriptor_collision")
        registry[fd] = None
        try:
            actual = os.fstat(fd)
            _require(_typed(actual) == _typed(named), "owner_target_descriptor_ownership_unproven")
        except (OSError, OwnerTargetVersionError):
            registry.pop(fd, None)
            self.unresolved += 1
            raise OwnerTargetVersionError("owner_target_descriptor_ownership_unproven") from None
        proof = _typed(named)
        registry[fd] = proof
        if group is not None:
            self.probe_owned[fd] = (group, proof)
        return actual

    def open(self, name, flags, *, parent=None, mode=0o600, target=False):
        self.slot()
        if parent is not None:
            self.proof(parent)
        creating = bool(flags & os.O_CREAT)
        _require(not flags & (os.O_TRUNC | os.O_APPEND) and
                 (not creating or flags & os.O_EXCL and parent is not None
                  and isinstance(name, str) and name.startswith(".target-version-") and name.endswith(".tmp")),
                 "owner_target_acquisition_invalid")
        if creating:
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise OwnerTargetVersionError("owner_target_publication_destination_exists")
            named = None
        else:
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) if flags & os.O_DIRECTORY else stat.S_ISREG(named.st_mode),
                     "owner_target_acquisition_invalid")
        self.budget.tick()
        fd = os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=parent)
        if creating:
            # Created birth requires independent named evidence BEFORE first fstat.
            try:
                self.proof(parent)
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                _require(stat.S_ISREG(named.st_mode) and named.st_uid == named.st_gid == 0
                         and named.st_nlink == 1 and not stat.S_IMODE(named.st_mode) & ~0o600,
                         "owner_target_descriptor_ownership_unproven")
            except (OSError, OwnerTargetVersionError):
                self.unresolved += 1
                raise OwnerTargetVersionError("owner_target_descriptor_ownership_unproven") from None
        initial = self._adopt_named(fd, named, self.owned)
        self.bindings[fd] = (parent, name, owners._security(named))
        self.acquired[fd] = initial
        if target:
            self.target_owned.add(fd)
        return fd

    def read_bytes(self, fd, cap):
        pieces, size = [], 0
        while True:
            self.budget.tick()
            self.proof(fd)
            remaining = min(self.raw_cap, self.budget.limits["raw_bytes"]) - self.budget.counts["raw_bytes"]
            amount = min(65536, cap + 1 - size, remaining)
            _require(amount > 0, "owner_target_resource_exhausted")
            self.budget.available("raw_bytes", amount)
            part = os.read(fd, amount)
            self.budget.tick()
            self.proof(fd)
            if not part:
                break
            self.budget.charge("raw_bytes", len(part))
            size += len(part)
            _require(size <= cap, "owner_target_resource_exhausted")
            pieces.append(part)
        self.budget.tick()
        return b"".join(pieces)

    def _close(self, fd, registry, *, probe=False):
        expected = registry.get(fd)
        if expected is None:
            return
        if probe:
            expected = expected[1]
        for _ in range(2):
            try:
                info = os.fstat(fd)
            except OSError as error:
                if error.errno == errno.EBADF:
                    registry.pop(fd, None)
                    self.target_owned.discard(fd)
                    return
                continue
            if _typed(info) != expected:
                registry.pop(fd, None)
                self.target_owned.discard(fd)
                self.unresolved += 1
                return
            try:
                os.close(fd)
            except OSError:
                continue
            registry.pop(fd, None)
            self.target_owned.discard(fd)
            return

    def close(self, fd):
        self._close(fd, self.owned)

    def finish_target(self):
        for fd in reversed(tuple(self.target_owned)):
            self.close(fd)
        _require(not self.target_owned, "owner_target_descriptor_cleanup_failed")
        _require(not self.unresolved, "owner_target_descriptor_ownership_unproven")

    def finish(self):
        # Failed local visibility groups remain here for bounded outer cleanup.
        for group in tuple(self.probe_groups):
            try:
                group.close()
            except (OwnerTargetVersionError, scratch.LaneScratchError):
                pass
        for fd in reversed(tuple(self.owned)):
            self.close(fd)
        _require(not self.unresolved, "owner_target_descriptor_ownership_unproven")
        _require(not self.owned and not self.probe_owned, "owner_target_descriptor_cleanup_failed")

    def final_lease_check(self):
        _require(bool(self.target_observations), "owner_target_version_changed")
        record, raw = self.target_observations[0]
        self._verify_target_record(record)
        os.lseek(record.fd, 0, os.SEEK_SET)
        _require(self.read_bytes(record.fd, scratch.MAX_LEASE_BYTES) == raw, "owner_target_version_changed")
        self._verify_target_record(record)

    def _verify_target_record(self, record):
        self.budget.tick()
        self.proof(record.parent)
        self.proof(record.fd)
        _require(_lease_metadata(os.fstat(record.fd)) == self.expected["lease_file_identity"]
                 == _lease_metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                 "owner_target_version_changed")


def _read_target_lease(files, target_fd):
    expected, budget = files.expected, files.budget
    files.proof(target_fd)
    _require(_matches_tuple(os.fstat(target_fd), expected["folder_identity"]), "owner_target_version_changed")
    named = os.stat(scratch.LEASE_FILE, dir_fd=target_fd, follow_symlinks=False)
    _require(_lease_metadata(named) == expected["lease_file_identity"], "owner_target_version_changed")
    fd = files.open(scratch.LEASE_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, parent=target_fd, target=True)
    record = owners._Acquired(fd, target_fd, scratch.LEASE_FILE, named)
    files._verify_target_record(record)
    raw = files.read_bytes(fd, scratch.MAX_LEASE_BYTES)
    _require(len(raw) == expected["lease_raw_size_bytes"]
             and retained._digest(raw, _work_budget=budget) == expected["lease_raw_sha256"],
             "owner_target_version_changed")
    try:
        lease = retained._document(raw, scratch.MAX_LEASE_BYTES, _work_budget=budget)
    except retained.CensusDecisionError:
        raise OwnerTargetVersionError("owner_target_lease_invalid") from None
    fields = {"schema_version", "lane", "name", "owner", "reason", "class_intent", "cleanup",
              "created_at_epoch", "expires_at_epoch", "released_at_epoch", "size_budget_bytes",
              "consumer_lifetime_contract", "lease_digest", expected["lease"]["reference_kind"]}
    _require(set(lease) in (fields, fields | {"renewed_at_epoch"})
             and lease["schema_version"] == scratch.SCHEMA_VERSION and scratch._lease_fields_valid(lease)
             and all(type(lease[key]) in (int, float) for key in
                     ("created_at_epoch", "expires_at_epoch", "renewed_at_epoch", "released_at_epoch")
                     if key in lease and lease[key] is not None),
             "owner_target_lease_invalid")
    budget.available("output_bytes", budget.measure(lease, cap=scratch.MAX_LEASE_BYTES))
    budget.tick()
    _require(canonical_digest(lease, digest_field="lease_digest") == expected["lease_digest"] == lease["lease_digest"],
             "owner_target_version_changed")
    reference = expected["lease"]["reference_kind"]
    projection = {key: lease[key] for key in expected["lease"]
                  if key not in {"reference_kind", "reference_value", "renewed_at_epoch"}}
    projection.update(reference_kind=reference, reference_value=lease[reference],
                      renewed_at_epoch=lease.get("renewed_at_epoch", lease["created_at_epoch"]))
    _require(projection == expected["lease"] and lease["lane"] == expected["lane"]
             and lease["name"] == expected["name"], "owner_target_version_changed")
    files._verify_target_record(record)
    if files.target_observations:
        _require(files.target_observations[0][1] == raw, "owner_target_version_changed")
    budget.charge("facts")
    _require(len(files.target_observations) < 2, "owner_target_resource_exhausted")
    files.target_observations.append((record, raw))
    return lease


def _safe_probe_type(files):
    """Private scoped dispatch; no native default/API/source is changed."""
    _require(files.expected is not None, "owner_target_expected_invalid")

    class _OwnerTargetProbe(LeasedScratchUse):
        def __init__(self):
            super().__init__()
            _require(len(files.probe_groups) < 2, "owner_target_resource_exhausted")
            files.budget.charge("groups")
            files.probe_groups.append(self)

        def _tick(self):
            files.budget.tick()

        def _capacity(self):
            self._tick()
            _require(not self._closed and len(files.probe_owned) < 24
                     and len(files.owned) <= 104 and len(files.owned) + len(files.probe_owned) < 128,
                     "owner_target_resource_exhausted")

        def _open(self, name, flags, parent=None, *, mode=0o777):
            self._capacity()
            _require(not flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND), "owner_target_acquisition_invalid")
            if parent is not None:
                files.proof(parent)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) if flags & os.O_DIRECTORY else stat.S_ISREG(named.st_mode),
                     "owner_target_acquisition_invalid")
            self._tick()
            fd = os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=parent)
            files._adopt_named(fd, named, self._owned, group=self)
            self._tick()
            return fd

        def _absolute(self, path):
            files.budget.charge("roots")
            return super()._absolute(path)

        def _visible(self):
            current = type(self)()
            try:
                root = current._absolute(self.root)
                lane = current._open(self.lane, scratch._DIR_FLAGS, root)
                folder = current._open(self.name, scratch._DIR_FLAGS, lane)
                for fresh, held, key in ((root, self._root_fd, "root_identity"),
                                         (lane, self._lane_fd, "lane_identity"),
                                         (folder, self.fd, "folder_identity")):
                    self._tick()
                    files.proof(fresh)
                    files.proof(held)
                    _require(_typed(os.fstat(fresh)) == _typed(os.fstat(held))
                             and _matches_tuple(os.fstat(held), files.expected[key]), "owner_target_version_changed")
            finally:
                current.close()

        def _lease(self):
            self._tick()
            return _read_target_lease(files, self.fd)

        def _close_one(self, fd):
            files._close(fd, files.probe_owned, probe=True)
            if fd not in files.probe_owned:
                self._owned.pop(fd, None)

        def _cleanup_status(self):
            _require(not files.unresolved, "owner_target_descriptor_ownership_unproven")

        def close(self):
            self._closed = True
            for fd in reversed(tuple(self._owned)):
                self._close_one(fd)
            if not self._owned and self in files.probe_groups:
                files.probe_groups.remove(self)
            self._cleanup_status()
            _require(not self._owned, "owner_target_descriptor_cleanup_failed")

    return _OwnerTargetProbe
