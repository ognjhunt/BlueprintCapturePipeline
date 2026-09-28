"""Declared experiment phases and one conserved invocation payload allowance.

Native budgets are single-use and unchanged. Payload IO has its own finite clock,
fragment and descriptor accounting; it does not resurrect a metadata budget.
"""
from __future__ import annotations

import math
import os
import stat
import time
from pathlib import Path

from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget

_PHASES = {'manifest': (1, 100000), 'ready': (1, 10000), 'removal_batch': (256, 10000),
           'restore_admission': (1, 10000), 'restore_batch': (256, 10000), 'finalize': (1, 100000)}
_ROLES = frozenset({'issue_hash', 'archive_prehash', 'archive_stream', 'remove_hash', 'restore_read', 'restore_write'})
_QUANTUM, _AGGREGATE = 1024 * 1024, 20 * 1024 * 1024


class _ActionFiles(_BirthFiles):
    def __init__(self, *, monotonic=time.monotonic, now=time.time):
        self.monotonic, self.now = monotonic, now
        self.controller_origin = monotonic()
        self.controller_epoch = now()
        self.last_clock = self.controller_origin
        _require(_epoch(self.controller_origin) and _epoch(self.controller_epoch), 'experiment_work_clock_invalid')
        self.deadline_epoch = self.controller_epoch + 4 * 3600
        self.failure, self.phase_name = None, 'admission'
        self.phase_counts, self.phase_history = {'admission': 1}, []
        self.conserved = dict(raw_bytes=0, output_bytes=0)
        self.payload_mode, self.payload_root, self.payload_fd, self.payload_size = False, None, None, None
        self.work, self.windows = {}, {}
        super().__init__(ReferenceCollectionBudget(monotonic=monotonic, values_limit=10000))

    def check_long(self):
        if self.failure:
            raise OwnerTargetVersionError(self.failure)
        try:
            current, epoch = self.monotonic(), self.now()
            _require(_epoch(current) and _epoch(epoch) and self.last_clock <= current
                     <= self.controller_origin + 4 * 3600 and epoch < self.deadline_epoch,
                     'experiment_work_deadline')
            _require(not self.unresolved, 'experiment_work_descriptor_unproven')
            self.last_clock = current
        except ValueError:
            self.failure = 'experiment_work_refused'
            raise OwnerTargetVersionError(self.failure) from None

    def bind_deadline(self, deadline):
        self.check_long()
        _require(_epoch(deadline), 'experiment_work_deadline')
        self.deadline_epoch = min(self.deadline_epoch, deadline)
        self.check_long()

    def _end(self):
        self.check_long()
        if not self.budget.closed:
            self.budget.tick()
            _require(self.budget.failure is None, 'experiment_work_refused')
            for key in self.conserved:
                _require(self.conserved[key] + self.budget.counts[key] <= _AGGREGATE, 'experiment_work_aggregate_limit')
                self.conserved[key] += self.budget.counts[key]
            self.phase_history.append((self.phase_name, dict(self.budget.counts)))
            self.budget.close()

    def phase(self, name):
        self.check_long()
        _require(name in _PHASES and self.phase_counts.get(name, 0) < _PHASES[name][0], 'experiment_work_phase_invalid')
        self._end()
        self.payload_mode = False
        self.phase_name = name
        self.phase_counts[name] = self.phase_counts.get(name, 0) + 1
        self.budget = ReferenceCollectionBudget(monotonic=self.monotonic, values_limit=_PHASES[name][1])
        self.budget.tick()

    def payload(self, root, fd, *, expected_payload_bytes):
        self.check_long()
        _require(type(expected_payload_bytes) is int and 0 <= expected_payload_bytes <= 128 * 1024**3
                 and isinstance(root, Path) and root.is_absolute() and self.parents.get(root) == fd,
                 'experiment_work_payload_invalid')
        self.location(fd)
        if self.payload_root is not None:
            _require((root, fd, expected_payload_bytes) == (self.payload_root, self.payload_fd, self.payload_size),
                     'experiment_work_payload_changed')
        self._end()
        self.payload_mode, self.payload_root, self.payload_fd, self.payload_size = True, root, fd, expected_payload_bytes

    def reserve_output(self, size):
        self.check_long()
        _require(type(size) is int and size >= 0 and not self.payload_mode
                 and self.conserved['output_bytes'] + self.budget.counts['output_bytes'] + size <= _AGGREGATE,
                 'experiment_work_aggregate_limit')
        self.budget.available('output_bytes', size)

    def location(self, fd, *, cleanup=False):
        if not cleanup:
            self.check_long()
        return super().location(fd, cleanup=cleanup or self.payload_mode)

    def slot(self):
        self.check_long()
        if not self.payload_mode:
            return super().slot()
        _require(len(self.owned) < 104 and len(self.owned) + len(self.probe_owned) < 128,
                 'experiment_work_descriptor_limit')

    def open(self, name, flags, *, parent=None, mode=0o600, target=False):
        if not self.payload_mode:
            return super().open(name, flags, parent=parent, mode=mode, target=target)
        self.slot()
        _require(parent is not None and type(name) is str and name not in ('', '.', '..')
                 and '/' not in name and '\x00' not in name and len(name.encode()) <= 255
                 and not flags & (os.O_TRUNC | os.O_APPEND), 'experiment_work_acquisition_invalid')
        self.location(parent)
        creating = bool(flags & os.O_CREAT)
        if creating:
            _require(flags & os.O_EXCL and name.startswith('.target-version-') and name.endswith('.tmp'),
                     'experiment_work_acquisition_invalid')
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise OwnerTargetVersionError('experiment_work_destination_exists')
            named = None
        else:
            _require(flags & os.O_ACCMODE == os.O_RDONLY, 'experiment_work_acquisition_invalid')
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) if flags & os.O_DIRECTORY else stat.S_ISREG(named.st_mode),
                     'experiment_work_acquisition_invalid')
        self.check_long()
        fd = os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=parent)
        if creating:
            try:
                self.location(parent)
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                _require(stat.S_ISREG(named.st_mode) and named.st_uid == named.st_gid == 0
                         and named.st_nlink == 1 and not stat.S_IMODE(named.st_mode) & ~0o600,
                         'experiment_work_descriptor_unproven')
            except (OSError, ValueError):
                self.unresolved += 1
                raise OwnerTargetVersionError('experiment_work_descriptor_unproven') from None
        initial = self._adopt_named(fd, named, self.owned)
        self.bindings[fd] = (parent, name, owners._security(named))
        self.acquired[fd] = initial
        if target:
            self.target_owned.add(fd)
        return fd

    def parent(self, path, *, protected=False):
        if not self.payload_mode:
            return super().parent(path, protected=protected)
        selected = Path(path)
        _require(not protected and selected.is_absolute() and selected.is_relative_to(self.payload_root)
                 and not any(part in ('.', '..') for part in selected.parts)
                 and len(str(selected).encode()) <= len(str(self.payload_root).encode()) + 2048,
                 'experiment_work_payload_path_invalid')
        fd, prefix = self.payload_fd, self.payload_root
        self.location(fd)
        for component in selected.relative_to(prefix).parts[:-1]:
            prefix = prefix / component
            child = self.parents.get(prefix)
            if child is None:
                child = self.open(component, os.O_RDONLY | os.O_DIRECTORY, parent=fd)
                self.parents[prefix] = child
            self.location(child)
            fd = child
        return fd, selected.name

    def read_bytes(self, fd, cap):
        _require(not self.payload_mode, 'experiment_work_metadata_phase_required')
        before = self.budget.counts['raw_bytes']
        remaining = _AGGREGATE - self.conserved['raw_bytes']
        _require(before < remaining, 'experiment_work_aggregate_limit')
        self.raw_cap = min(2 * _QUANTUM, remaining)
        return super().read_bytes(fd, cap)

    def payload_read(self, fd, amount, *, role):
        self.check_long()
        _require(self.payload_mode and role in _ROLES and type(amount) is int and 0 < amount <= _QUANTUM,
                 'experiment_work_payload_invalid')
        self.location(fd)
        identity = self.proof(fd)
        key = (role, identity)
        position, window, fragments = self.windows.get(key, (0, 0, 0))
        current = position // _QUANTUM
        if current != window:
            window, fragments = current, 0
        maximum = 8 * (math.ceil(self.payload_size / _QUANTUM) + 4096)
        if fragments >= 8 or self.work.get(role, 0) >= maximum:
            self.failure = 'experiment_work_fragment_limit'
            raise OwnerTargetVersionError(self.failure)
        self.work[role] = self.work.get(role, 0) + 1
        self.windows[key] = (position, window, fragments + 1)
        block = os.read(fd, min(amount, _QUANTUM - position % _QUANTUM))
        self.location(fd)
        _require(type(block) is bytes and len(block) <= amount, 'experiment_work_payload_invalid')
        self.windows[key] = (position + len(block), window, fragments + 1)
        return block

    def verify_record(self, record):
        if not self.payload_mode:
            return super().verify_record(record)
        self.location(record.parent)
        self.proof(record.fd)
        _require(owners._metadata(os.fstat(record.fd)) == owners._metadata(record.info)
                 == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                 'experiment_work_record_changed')

    def finish(self):
        super().finish()
        _require(not self.owned and not self.probe_owned, 'experiment_work_cleanup_failed')

    def verify(self):
        if not self.payload_mode:
            return super().verify()
        self.check_long()
        for record in self.records:
            self.location(record.parent)
            self.proof(record.fd)
            _require(owners._metadata(os.fstat(record.fd)) == owners._metadata(record.info)
                     == owners._metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False)),
                     'experiment_work_record_changed')

    def trim_payload(self, *, keep=()):
        self.check_long()
        protected = {self.payload_fd, *keep, *(record.parent for record in self.records)}
        for path, fd in sorted(tuple(self.parents.items()), key=lambda item: len(item[0].parts), reverse=True):
            if self.payload_root is None or not path.is_relative_to(self.payload_root) or fd in protected:
                continue
            if any(binding[0] == fd for child, binding in self.bindings.items() if child in self.owned):
                continue
            self.location(fd)
            self.close(fd)

    def close(self, fd):
        super().close(fd)
        if fd not in self.owned:
            for path, current in tuple(self.parents.items()):
                if current == fd:
                    self.parents.pop(path)
            self.bindings.pop(fd, None)
            self.acquired.pop(fd, None)
