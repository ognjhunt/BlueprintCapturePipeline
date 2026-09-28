"""Declared experiment phases and one conserved invocation payload allowance.

Native budgets are single-use and unchanged. Payload IO has its own finite clock,
fragment and descriptor accounting; it does not resurrect a metadata budget.
"""
from __future__ import annotations

import math
import os
import re
import stat
import time
from pathlib import Path

from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget

_PHASES = {'restore_checkpoint_manifest': (1, 100000), 'restore_checkpoint_compare': (1, 100000), 'restore_checkpoint_record': (1, 10000), 'manifest': (1, 100000), 'ready': (1, 10000), 'removal_batch': (256, 10000), 'recovery_batch': (256, 10000), 'restore_recovery_batch': (257, 10000), 'restore_activation_manifest': (1, 100000),
           'restore_admission': (1, 10000), 'restore_prepare': (1, 100000), 'restore_stage': (1, 100000), 'restore_stage_verify': (1, 100000), 'restore_union_finalize': (1, 100000), 'restore_directories': (256, 10000), 'restore_cleanup': (256, 10000), 'restore_batch': (256, 10000), 'finalize': (1, 100000)}
_ROLES = frozenset({'issue_hash', 'archive_prehash', 'archive_stream', 'archive_digest_hash', 'archive_upload_hash', 'archive_digest_stream', 'archive_upload_stream', 'remove_hash', 'restore_read', 'restore_write', 'restore_validate', 'restore_stage_validate', 'restore_activation_validate', 'restore_checkpoint_validate'})
_QUANTUM, _AGGREGATE = 1024 * 1024, 20 * 1024 * 1024
_BOOT_PATH = Path('/proc/sys/kernel/random/boot_id')
_CONTROLLER_FIELDS = frozenset({'boot_id', 'origin_monotonic', 'deadline_monotonic',
                              'origin_epoch', 'deadline_epoch', 'last_monotonic', 'last_epoch'})


def _controller_boot_id(files):
    raw, record = files.read(_BOOT_PATH, cap=40, protected=True)
    files.verify_record(record)
    _require(re.fullmatch(rb'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\n', raw) is not None,
             'experiment_work_boot_unknown')
    value = raw[:-1].decode('ascii')
    files.records.remove(record)
    files.close(record.fd)
    _require(record.fd not in files.owned and not files.unresolved, 'experiment_work_boot_unknown')
    return value


class _ActionFiles(_BirthFiles):
    def __init__(self, *, monotonic=time.monotonic, now=time.time):
        _require('_action_initialized' not in self.__dict__, 'experiment_work_initialization_reused')
        _require(callable(monotonic) and callable(now), 'experiment_work_clock_invalid')
        origin, epoch = monotonic(), now()
        _require(_epoch(origin) and _epoch(epoch), 'experiment_work_clock_invalid')
        # First valid initialization is permanently consumed even if later
        # allocation fails. Never lose owned descriptors/closed or failed B.
        self._action_initialized = True
        self.monotonic, self.now = monotonic, now
        self.controller_origin = origin
        self.controller_epoch = epoch
        self.last_clock = self.controller_origin
        self.last_epoch = self.controller_epoch
        self.deadline_epoch = self.controller_epoch + 4 * 3600
        self.deadline_monotonic = self.controller_origin + 4 * 3600
        self.boot_id, self.controller_bound = None, False
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
                     <= self.controller_origin + 4 * 3600 and current < self.deadline_monotonic
                     and self.last_epoch <= epoch < self.deadline_epoch,
                     'experiment_work_deadline')
            _require(not self.unresolved, 'experiment_work_descriptor_unproven')
            self.last_clock = current
            self.last_epoch = epoch
        except ValueError:
            self.failure = 'experiment_work_refused'
            raise OwnerTargetVersionError(self.failure) from None

    def controller(self):
        self.check_long()
        if self.boot_id is None:
            self.boot_id = _controller_boot_id(self)
        self.check_long()
        return dict(boot_id=self.boot_id, origin_monotonic=self.controller_origin,
                    deadline_monotonic=self.deadline_monotonic, origin_epoch=self.controller_epoch,
                    deadline_epoch=self.deadline_epoch, last_monotonic=self.last_clock,
                    last_epoch=self.last_epoch)

    def bind_controller(self, value):
        """Restrict a NEW invocation to its authenticated original operation clock.

        Callers select protected durable records under actual authority locks.
        These numeric fields are not independent permission to use a target.
        """
        try:
            _require(not self.controller_bound and isinstance(value, dict)
                     and set(value) == _CONTROLLER_FIELDS
                     and all(_epoch(value[key]) for key in _CONTROLLER_FIELDS - {'boot_id'})
                     and isinstance(value['boot_id'], str), 'experiment_work_clock_invalid')
            current = self.controller()
            _require(value['boot_id'] == current['boot_id']
                     and value['origin_monotonic'] <= value['last_monotonic'] <= current['last_monotonic']
                     < value['deadline_monotonic'] <= value['origin_monotonic'] + 14400
                     and value['origin_epoch'] <= value['last_epoch'] <= current['last_epoch']
                     < value['deadline_epoch'] <= value['origin_epoch'] + 14400,
                     'experiment_work_clock_invalid')
            self.controller_origin = min(self.controller_origin, value['origin_monotonic'])
            self.controller_epoch = min(self.controller_epoch, value['origin_epoch'])
            self.deadline_monotonic = min(self.deadline_monotonic, value['deadline_monotonic'])
            self.deadline_epoch = min(self.deadline_epoch, value['deadline_epoch'])
            self.controller_bound = True
            self.check_long()
        except ValueError:
            self.failure = self.failure or 'experiment_work_clock_invalid'
            raise OwnerTargetVersionError(self.failure) from None

    def bind_deadline(self, deadline):
        self.check_long()
        _require(_epoch(deadline), 'experiment_work_deadline')
        self.deadline_epoch = min(self.deadline_epoch, deadline)
        # Preserve the original owner's remaining duration even if wall time
        # subsequently stalls. A later retry cannot mint another duration.
        self.deadline_monotonic = min(self.deadline_monotonic,
            self.controller_origin + max(0, deadline - self.controller_epoch))
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

    def publication_complete(self, parent, name, before):
        """Release newly created metadata FDs only after immutable readback.

        The native publisher retains its original write token. Its temporary
        pathname has disappeared, so closing proves that original token against
        the final named inode rather than adopting another observed identity.
        """
        self.location(parent)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        for fd in self.owned.keys() - before:
            _require(fd not in self.parents.values() and not any(record.fd == fd for record in self.records),
                     'experiment_work_publication_unproven')
            self.proof(fd)
            _require(owners._metadata(os.fstat(fd)) == owners._metadata(named),
                     'experiment_work_publication_unproven')
            self.close(fd)
            _require(fd not in self.owned, 'experiment_work_cleanup_failed')

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
        if os.lseek(fd, 0, os.SEEK_CUR) != position:
            self.failure = 'experiment_work_payload_cursor_changed'
            raise OwnerTargetVersionError(self.failure)
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

    def removed_directory(self, path, original):
        cached = self.parents.get(path)
        if cached is not None:
            _require(self.proof(cached) == self.proof(original), 'experiment_work_descriptor_unproven')
            self.close(cached)
            _require(cached not in self.owned, 'experiment_work_cleanup_failed')

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
