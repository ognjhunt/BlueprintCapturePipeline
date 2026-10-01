"""Current diagnostic reference observations under one original finite budget.

These observations supply no owner grant. Only the authenticated caller may
select an action; unknown tables or kernel views refuse before its effects.
"""
from __future__ import annotations

import os
import stat
from contextlib import ExitStack

from . import control_plane_lane_owner_consents as owners
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_experiment_work import _AGGREGATE
from .control_plane_lane_historical_processes import _Scan, _mapping_inodes, refuse_historical_process_references
from .control_plane_lane_historical_references import historical_reference_fence
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require
from .control_plane_reference_budget import ReferenceCollectionBudget


def _refusal(error):
    return {
        'historical_generation_process_reference': 'experiment_diagnostic_process_reference',
        'historical_generation_table_reference': 'experiment_diagnostic_queue_reference',
        'historical_generation_release_reference': 'experiment_diagnostic_release_reference',
        'historical_generation_pin_reference': 'experiment_diagnostic_pin_reference',
    }.get(str(error), 'experiment_diagnostic_references_unknown')


class _ReferenceFiles(_BirthFiles):
    def __init__(self, budget, action_files):
        self.action_files = action_files
        self.selected_reads = {}
        self.metadata_bytes = 0
        super().__init__(budget)

    def read_bytes(self, fd, cap):
        # Kernel and protected metadata observations share the original B.
        # Its kernel bytes must not consume this owner's separate two-MiB
        # metadata ceiling; every metadata reread still charges both ceilings.
        pieces, size = [], 0
        while True:
            self.budget.tick()
            self.proof(fd)
            remaining = min(self.raw_cap - self.metadata_bytes,
                            self.budget.limits['raw_bytes'] - self.budget.counts['raw_bytes'])
            amount = min(65536, cap + 1 - size, remaining)
            _require(amount > 0, 'owner_target_resource_exhausted')
            self.budget.available('raw_bytes', amount)
            part = os.read(fd, amount)
            self.metadata_bytes += len(part)
            self.budget.charge('raw_bytes', len(part))
            self.budget.tick()
            self.proof(fd)
            size += len(part)
            _require(size <= cap, 'owner_target_resource_exhausted')
            if not part:
                break
            pieces.append(part)
        self.budget.tick()
        return b''.join(pieces)

    def read(self, path, *, cap, protected=False, mode=None):
        key = os.fspath(path)
        selected = self.selected_reads.get(key)
        if selected is None:
            raw, record = super().read(path, cap=cap, protected=protected, mode=mode)
            self.selected_reads[key] = raw, record
            return raw, record
        original, record = selected
        self.location(record.parent)
        self.verify_record(record)
        if protected:
            owners._protected(os.fstat(record.fd), mode=mode)
        _require(len(original) <= cap, 'experiment_diagnostic_references_unknown')
        os.lseek(record.fd, 0, os.SEEK_SET)
        raw = self.read_bytes(record.fd, cap)
        self.verify_record(record)
        self.budget.charge('entries')
        _require(raw == original, 'experiment_diagnostic_references_unknown')
        return raw, record

    def slot(self):
        super().slot()
        action = self.action_files
        # Reserve sixteen actual scanner-local descriptors. No original owned
        # action/reference descriptors are forgotten or refunded here.
        _require(len(self.owned) + len(self.probe_owned) + len(action.owned)
                 + len(action.probe_owned) < 112, 'experiment_work_descriptor_limit')

    def table_descriptor_check(self, transient_count):
        self.budget.tick()
        action = self.action_files
        _require(len(self.owned) + len(self.probe_owned) + len(action.owned)
                 + len(action.probe_owned) + transient_count <= 128,
                 'experiment_work_descriptor_limit')


class DiagnosticReferences:
    """Hold original installed settings and state/pin locks through the effect."""
    def __init__(self, action_files, config, target, target_fd, manifest, *, issued, held_pins, stage_fd=None):
        self.action_files, self.target, self.target_fd = action_files, target, target_fd
        def clock():
            action_files.check_long()
            return action_files.monotonic()
        # Keep the canonical concrete budget accepted by descriptor owners.
        # The wrapper preserves its original native charge and then immediately
        # accounts the actual same bytes in the original action aggregate.
        self.budget = ReferenceCollectionBudget(monotonic=clock)
        native_charge = self.budget.charge
        def charge(kind, amount=1):
            native_charge(kind, amount)
            if kind == 'raw_bytes':
                current = 0 if action_files.budget.closed else action_files.budget.counts[kind]
                _require(action_files.conserved[kind] + current + amount <= _AGGREGATE,
                         'experiment_work_aggregate_limit')
                action_files.conserved[kind] += amount
        self.budget.charge = charge
        self.files, self.stack = _ReferenceFiles(self.budget, action_files), ExitStack()
        self.original = {tuple(map(int, row[2].split(':')[:2])) for row in manifest['members']}
        self.stage = None
        self.closed = False
        try:
            if stage_fd is not None:
                self.bind_stage(stage_fd)
            self.table_guard = self.stack.enter_context(historical_reference_fence(
                self.files, config, target, observed_at=issued, _held_pins=held_pins))
            self.guard()
        except OwnerTargetVersionError as error:
            self.close()
            if error.code.startswith('experiment_diagnostic_'):
                raise
            raise OwnerTargetVersionError(_refusal(error)) from None
        except (OSError, ValueError) as error:
            self.close()
            raise OwnerTargetVersionError(_refusal(error)) from None

    def guard(self, *, processes=True):
        try:
            self.budget.tick()
            self.table_guard()
            self.files.verify()
            action = self.action_files
            _require(len(self.files.owned) + len(self.files.probe_owned) + len(action.owned)
                     + len(action.probe_owned) <= 112, 'experiment_work_descriptor_limit')
            action.location(self.target_fd)
            info = os.fstat(self.target_fd)
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == 0
                     and stat.S_IMODE(info.st_mode) == 0o700,
                     'experiment_diagnostic_rights_changed')
            current = self.target_fd
            for _ in range(64):
                action.proof(current)
                owners._protected(os.fstat(current), directory=True)
                parent = action.bindings[current][0]
                if parent is None:
                    break
                current = parent
            else:
                raise OwnerTargetVersionError('experiment_work_descriptor_limit')
            identities = self.original | {(info.st_dev, info.st_ino)}
            allowed = {'.lane-scratch.v1.json', '.registered-experiment.v1.json',
                       'disk-capacity-report.v1.json'}
            if self.stage is not None:
                allowed.add(action.bindings[self.stage][1])
            with os.scandir(self.target_fd) as entries:
                names = set()
                for item in entries:
                    self.budget.charge('entries')
                    _require(item.name in allowed and item.name not in names,
                             'experiment_diagnostic_namespace_changed')
                    names.add(item.name)
            _require({'.lane-scratch.v1.json', '.registered-experiment.v1.json'} <= names,
                     'experiment_diagnostic_namespace_changed')
            # Named paths alone are insufficient for inherited/unlinked aliases.
            # Keep original payload identities even after journaled unlink.
            for name in ('.lane-scratch.v1.json', '.registered-experiment.v1.json'):
                value = os.stat(name, dir_fd=self.target_fd, follow_symlinks=False)
                _require(stat.S_ISREG(value.st_mode) and value.st_uid == value.st_gid == 0
                         and stat.S_IMODE(value.st_mode) == 0o600 and value.st_nlink == 1,
                         'experiment_diagnostic_rights_changed')
                identities.add((value.st_dev, value.st_ino))
            if self.stage is not None:
                action.location(self.stage)
                value = os.fstat(self.stage)
                identities.add((value.st_dev, value.st_ino))
                with os.scandir(self.stage) as entries:
                    for item in entries:
                        self.budget.charge('entries')
                        value = os.stat(item.name, dir_fd=self.stage, follow_symlinks=False)
                        _require(stat.S_ISREG(value.st_mode) and value.st_uid == value.st_gid == 0
                                 and stat.S_IMODE(value.st_mode) == 0o600,
                                 'experiment_diagnostic_rights_changed')
                        if item.name != 'disk-capacity-report.v1.json':
                            _require(any(action.bindings[fd][:2] == (self.stage, item.name)
                                         and action.proof(fd)[:2] == (value.st_dev, value.st_ino)
                                         for fd in action.owned), 'experiment_restore_stage_changed')
                        identities.add((value.st_dev, value.st_ino))
            try:
                value = os.stat('disk-capacity-report.v1.json', dir_fd=self.target_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                identities.add((value.st_dev, value.st_ino))
            self.original.update(identities)
            if processes:
                self._own_handles(identities)
                refuse_historical_process_references(dict(target_path=str(self.target),
                    members=[dict(version=identity) for identity in sorted(identities)]),
                    tick=self.budget.tick, budget=self.budget)
            self.budget.tick()
        except OwnerTargetVersionError:
            raise
        except (OSError, ValueError) as error:
            raise OwnerTargetVersionError(_refusal(error)) from None

    def bind_stage(self, fd):
        """Only an original caller-owned, named restore stage is admitted."""
        action = self.action_files
        action.proof(fd)
        action.location(fd)
        _require(action.bindings[fd][0] == self.target_fd and (self.stage is None or self.stage == fd),
                 'experiment_restore_stage_changed')
        value = os.fstat(fd)
        _require(stat.S_ISDIR(value.st_mode) and value.st_uid == value.st_gid == 0
                 and stat.S_IMODE(value.st_mode) == 0o700, 'experiment_restore_stage_changed')
        self.stage = fd

    def removed_stage(self, fd):
        _require(self.stage == fd, 'experiment_restore_stage_changed')
        self.action_files.proof(fd)
        value = os.fstat(fd)
        _require(stat.S_ISDIR(value.st_mode) and value.st_nlink == 0,
                 'experiment_restore_stage_changed')
        self.original.add((value.st_dev, value.st_ino))
        self.stage = None

    def _own_handles(self, identities):
        """The controlled worker exclusion never admits unknown inherited FDs/maps."""
        proc = os.open('/proc/' + str(os.getpid()), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        scan = _Scan(self.budget.tick, self.budget)
        try:
            directory = os.open('fd', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=proc)
            try:
                # Keep the actual enumeration descriptor live while checking
                # entries; closing it first would manufacture a changing view.
                with os.scandir(directory) as entries:
                    for item in entries:
                        self.budget.charge('entries')
                        value = os.stat(item.name, dir_fd=directory)
                        if (value.st_dev, value.st_ino) in identities:
                            fd = int(item.name)
                            owner = (self.action_files if fd in self.action_files.owned or fd in self.action_files.probe_owned else
                                     self.files if fd in self.files.owned or fd in self.files.probe_owned else None)
                            _require(owner is not None, 'experiment_diagnostic_process_reference')
                            owner.proof(fd)
            finally:
                os.close(directory)
            value = os.stat('cwd', dir_fd=proc)
            current = os.readlink('cwd', dir_fd=proc)
            _require((value.st_dev, value.st_ino) not in identities
                     and not (current == str(self.target) or current.startswith(str(self.target) + '/')),
                     'experiment_diagnostic_process_reference')
            _require(not (_mapping_inodes(scan.read(proc, 'maps')) & identities),
                     'experiment_diagnostic_process_reference')
        finally:
            os.close(proc)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.stack.close()
            finally:
                try:
                    self.files.finish()
                finally:
                    self.budget.close()
