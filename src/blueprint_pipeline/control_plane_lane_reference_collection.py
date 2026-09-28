"""Bounded selected lane-reference evidence; never a held fence or GC grant.

Only explicitly selected preparation/activation and SAM locations are joined.
Linux process channels yield positives, never a general absence proof. Historical
configuration, current collector selection, and live producer selection differ.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import control_plane_queue_observation as queues
from .control_plane_queue_auxiliary_observation import AuxiliaryQueueContract, observe_preparation_sam_auxiliaries
from .control_plane_preparation_activation_references import ReferenceFamilyContract, RetainedReferenceRecord, interpret_preparation_activation_references
from .control_plane_reference_budget import ReferenceCollectionBudget, ReferenceCollectionBudgetError
from .control_plane_scratch_lifetime import LeasedScratchUse, LANE_ROOTS
from .control_plane_lane_scratch import LaneScratchError, LEASE_FILE
from .control_plane_storage_pin_observation import observe_storage_pins

PREPARATION_STATES = ('pending', 'processing', 'awaiting_source_preparation', 'awaiting_capacity', 'materialized', 'completed', 'blocked')
ACTIVATION_STATES = ('pending', 'processing', 'prepared', 'blocked')
SAM_STATES = ('pending', 'processing', 'waiting_external', 'completed', 'failed')
PROC_CHANNEL_BYTES = 1024 * 1024
_ENV_PATHS = {
    'BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT': 'preparation_queue_root',
    'BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT': 'preparation_input_root',
    'BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT': 'activation_queue_root',
    'BLUEPRINT_TASK_EVALUATION_LAUNCH_ACTIVATION_ROOT': 'activation_input_root',
    'BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG': 'progression_config',
    'BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT': 'intent_root',
    'BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT': 'source_binding_root',
    'BLUEPRINT_TASK_EVALUATION_OWNER_SOURCE_STORE_ROOT': 'owner_source_store_root',
    'PIPELINE_CAPTURE_INTAKE_STORE_ROOT': 'capture_store_root',
    'BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT': 'pins_root',
}
_CONFIG_PATHS = ('intent_root', 'preparation_queue_root', 'child_queue_root', 'child_execution_root',
                 'launch_queue_root', 'launch_execution_root', 'public_source_binding_root', 'machinery_path',
                 'publication_lock_root', 'service_status_path', 'capture_store_root', 'source_store_root',
                 'activation_queue_root', 'activation_root', 'sam31_profile_registry_root')
_MODULES = {'blueprint_pipeline.task_evaluation_scene_progression': 'progression',
            'blueprint_pipeline.task_evaluation_launch_preparation_worker': 'preparation',
            'blueprint_pipeline.task_evaluation_launch_activation_worker': 'activation',
            'blueprint_pipeline.task_evaluation_sam31_preparation_execution': 'sam'}
_DEFERRED = ('construction_compilation_launch_profile', 'policy_canary_terminal_release',
             'registry_process_enabled_sources', 'auxiliary_consumer_joins', 'mmap_other_namespaces_future_processes',
             'child_consumer_participation_unproven', 'external_input_consumers_unproven',
             'owner_grants_offload_restore_unproven')


class LaneReferenceCollectionError(ValueError):
    """Fixed API refusal without supplied settings, paths or exceptions."""


@dataclass(frozen=True)
class TargetProbe:
    status: str
    target_path: str
    lease_raw_sha256: str | None = None
    lease_raw_size_bytes: int | None = None
    lease_class: str | None = None
    cleanup_policy: str | None = None
    lease_digest: str | None = None
    owner: str | None = None
    reference_kind: str | None = None
    reference_value: str | None = None
    inodes: tuple[tuple[int, int], ...] = ()
    started_monotonic: float | None = None
    ended_monotonic: float | None = None
    held_after_return: bool | None = False


@dataclass(frozen=True)
class ProcessReference:
    pid: int
    start_time: int
    channel: str
    deleted_suffix_observed: bool = False
    scope: str = 'positive_observed_channel_only'


@dataclass(frozen=True)
class DeclaredSource:
    path: str
    role: str
    origin: str
    pid: int | None = None
    start_time: int | None = None


@dataclass(frozen=True)
class ConfigObservation:
    path: str
    origin: str
    raw_sha256: str
    raw_size_bytes: int
    canonical_digest: str
    row_identity: tuple[int, ...]
    pid: int | None = None
    start_time: int | None = None


@dataclass(frozen=True)
class LaneReferenceCollection:
    target_probe: TargetProbe
    source_origin: str
    selected_sources: tuple[DeclaredSource, ...]
    configurations: tuple[ConfigObservation, ...]
    process_references: tuple[ProcessReference, ...]
    kept_reasons: tuple[str, ...]
    complete_selected_observations: bool
    primary: Any = None
    auxiliary: Any = None
    pins: Any = None
    interpreted: Any = None
    fixed_process_channels_complete: bool = False
    mutations: int = 0
    candidate_bytes: int | None = None
    references_clear: bool = False
    consumer_fence_checked: bool = False
    general_reference_inventory_complete: bool = False
    general_process_inventory_complete: bool = False
    execution_authorized: bool = False
    apply_supported: bool = False


def _path(value: Any) -> str:
    try:
        path = queues._path(value)
        if any(len(part.encode('utf-8')) > 255 for part in path.split('/')):
            raise ValueError
        return path
    except (ValueError, TypeError, UnicodeError):
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid') from None


def _roots(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > 16:
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
    result = tuple(_path(value) for value in values)
    if len(set(result)) != len(result):
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
    return result


def _below(path: str, target: str) -> bool:
    return path == target or path.startswith(target + '/')


class _Acquisition(queues._Scan):
    """Existing pure/descriptor mechanics with section11 owned finalization."""
    def __init__(self, *args):
        super().__init__(*args)
        self.owner = LeasedScratchUse()

    def open(self, name, flags, parent=None):
        self.tick()
        fd = self.owner._open(name, flags, parent)
        self.tick()
        return fd

    def close(self, fd):
        self.owner._close_one(fd)
        self.owner._cleanup_status()
        if fd in self.owner._owned:
            self.block('reference_descriptor_cleanup_failed')

    def close_all(self):
        self.owner.close()


class _Collector:
    def __init__(self, target, observed, budget, origin):
        self.target, self.observed, self.budget, self.origin = target, observed, budget, origin
        self.scan = _Acquisition((), observed, time.monotonic, 5, budget)
        self.sources, self.configs, self.processes = [], [], []
        self.config_snapshots, self.proc_snapshots = [], []
        self.reasons = set(_DEFERRED)
        self.process_complete = False
        self.producer_seen = False

    def keep(self, code):
        self.budget.block(code)
        if len(self.reasons) < 32 or code in self.reasons:
            self.reasons.add(code)
        else:
            self.reasons.add('reference_blockers_truncated')

    def add(self, collection, item, *, fact=False):
        if fact:
            self.budget.charge('facts')
        self.budget.retain(item)
        collection.append(item)

    def source(self, path, role, origin, pid=None, start=None):
        path = _path(path)
        self.add(self.sources, DeclaredSource(path, role, origin, pid, start), fact=True)

    def binary(self, fd, name, *, cap=PROC_CHANNEL_BYTES):
        self.budget.tick()
        record = self.scan.open(name, queues._FILE_FLAGS, fd)
        try:
            before = self.scan.call(os.fstat, record)
            queues._require(stat.S_ISREG(before.st_mode), 'reference_channel_unsafe')
            # Proc files often advertise size zero; read a capped stream+sentinel.
            queues._require(before.st_size <= cap, 'reference_record_bytes_limit')
            self.budget.available('raw_bytes', before.st_size)
            raw = bytearray()
            while True:
                chunk = self.scan.call(os.read, record, min(4096, cap + 1 - len(raw)))
                if not chunk:
                    break
                queues._require(len(raw) + len(chunk) <= cap, 'reference_record_bytes_limit')
                self.budget.charge('raw_bytes', len(chunk))
                raw.extend(chunk)
            after = self.scan.call(os.fstat, record)
            queues._require(queues._identity(before) == queues._identity(after), 'reference_channel_changed')
            return bytes(raw), queues._identity(before)
        finally:
            self.scan.close(record)

    def json_value(self, raw):
        text = self.scan.parse(raw)
        self.budget.preflight(text)
        self.budget.tick()
        return json.loads(text)

    def configuration(self, path, origin, pid=None, start=None):
        path = _path(path)
        self.budget.charge('roots')
        parent = self.scan.walk(str(Path(path).parent))
        raw, identity = self.binary(parent[-1][0], Path(path).name, cap=4 * 1024 * 1024)
        text = raw.decode('utf-8')
        self.scan.preflight(text)
        value = self.json_value(raw)
        queues._require(value.get('schema_version') == 'task_evaluation_scene_progression_config.v1', 'reference_config_unsupported')
        seal = value.get('config_digest')
        document = {key: item for key, item in value.items() if key != 'config_digest'}
        self.budget.measure(document)
        digest = hashlib.sha256()
        for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=False).iterencode(document):
            self.budget.tick()
            for offset in range(0, len(chunk), 1024):
                self.budget.tick()
                digest.update(chunk[offset:offset + 1024].encode('utf-8'))
        canonical = 'sha256:' + digest.hexdigest()
        queues._require(seal == canonical, 'reference_config_seal_invalid')
        self.add(self.configs, ConfigObservation(path, origin, 'sha256:' + hashlib.sha256(raw).hexdigest(),
                                              len(raw), canonical, identity, pid, start))
        self.config_snapshots.append((path, parent, identity))
        for key in _CONFIG_PATHS:
            if key in value:
                self.source(value[key], key, origin, pid, start)
        nested = value.get('preparation_worker')
        if nested is not None:
            queues._require(isinstance(nested, dict), 'reference_config_shape_invalid')
            for key in ('input_root', 'construction_queue_root', 'storage_pins_root'):
                if key in nested:
                    self.source(nested[key], 'preparation_worker.' + key, origin, pid, start)
        # Dynamic provider/catalog/source bindings cannot be represented as empty.
        self.keep('configuration_additional_sources_unobserved')

    def start_time(self, raw, pid):
        prefix, close, suffix = raw.rpartition(b')')
        queues._require(bool(close) and prefix.startswith(str(pid).encode() + b' ('), 'reference_process_identity_invalid')
        fields = suffix.split()
        queues._require(len(fields) >= 20 and fields[19].isdigit() and len(fields[19]) <= 20,
                        'reference_process_identity_invalid')
        value = int(fields[19])
        queues._require(value < 2**64, 'reference_process_identity_invalid')
        return value

    def hint(self, pid, start, channel, raw):
        if self.target.encode('utf-8') in raw:
            self.add(self.processes, ProcessReference(pid, start, channel), fact=True)

    def link(self, parent, name, pid, start, channel):
        before = self.scan.call(os.stat, name, dir_fd=parent, follow_symlinks=False)
        queues._require(stat.S_ISLNK(before.st_mode), 'reference_process_link_unsafe')
        value = self.scan.call(os.readlink, name, dir_fd=parent)
        self.budget.measure(value, cap=4096)
        after = self.scan.call(os.stat, name, dir_fd=parent, follow_symlinks=False)
        queues._require(queues._identity(before) == queues._identity(after), 'reference_process_changed')
        deleted = value.endswith(' (deleted)')
        observed = value[:-10] if deleted else value
        if _below(observed, self.target):
            self.add(self.processes, ProcessReference(pid, start, channel, deleted), fact=True)

    def tokens(self, raw):
        count = 1
        for offset in range(0, len(raw), 4096):
            self.budget.tick()
            count += raw[offset:offset + 4096].count(b'\0')
        self.budget.charge('values', count)
        return raw.split(b'\0')

    def producer(self, pid, start, command, environment):
        # Select known argv tokens only; never parse shell wrappers or emit argv.
        argv = self.tokens(command)
        if argv and not argv[-1]:
            argv.pop()
        if (len(argv) < 3 or len(argv[0]) > 4096 or len(argv[2]) > 128 or not re.fullmatch(rb'(?:.*/)?python(?:[0-9.]+)?', argv[0])
                or argv[1] != b'-m'):
            return
        try:
            module = argv[2].decode('utf-8')
        except UnicodeError:
            return
        if module not in _MODULES:
            return
        selected = {}
        for token in self.tokens(environment):
            key, separator, value = token.partition(b'=')
            if not separator:
                continue
            try:
                if len(key) > 128:
                    continue
                key = key.decode('ascii')
                if key in _ENV_PATHS:
                    if key in selected:
                        raise ValueError
                    selected[key] = value.decode('utf-8')
            except (ValueError, UnicodeError):
                raise queues._Blocked('reference_process_configuration_ambiguous') from None
        if module == 'blueprint_pipeline.task_evaluation_scene_progression':
            configs = []
            for index, token in enumerate(argv[3:], 3):
                if token == b'--config':
                    queues._require(index + 1 < len(argv), 'reference_process_configuration_ambiguous')
                    configs.append(argv[index + 1].decode('utf-8'))
                elif token.startswith(b'--config='):
                    configs.append(token[9:].decode('utf-8'))
            if len(configs) > 1:
                raise queues._Blocked('reference_process_configuration_ambiguous')
            config = configs[0] if configs else selected.get('BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG')
            if config is None:
                self.keep('producer_configuration_unproven')
            else:
                self.configuration(config, 'live_selected_producer_config', pid, start)
                self.producer_seen = True
        # CLI root overrides must be represented explicitly or kept ambiguous.
        if any(token.endswith(b'-root') or b'-root=' in token for token in argv[3:]):
            self.keep('reference_process_configuration_ambiguous')
            return
        for key, value in selected.items():
            if key != 'BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG':
                self.source(value, _ENV_PATHS[key], 'live_selected_producer_environment', pid, start)
        self.keep('producer_other_effective_settings_unobserved')

    def proc(self, proc_root):
        self.budget.charge('roots')
        chain = self.scan.walk(proc_root)
        root = chain[-1][0]
        names = self.scan.names(root, 0)
        pid_snapshots = []
        failures = False
        for name in names:
            self.budget.tick()
            if not name.isascii() or not name.isdigit():
                continue
            queues._require(len(name) <= 10, 'reference_process_identity_invalid')
            pid = int(name)
            if pid <= 0:
                continue
            try:
                pid_fd = self.scan.open(name, queues._DIR_FLAGS, root)
                before = self.scan.call(os.fstat, pid_fd)
                stat_raw, _ = self.binary(pid_fd, 'stat', cap=8192)
                start = self.start_time(stat_raw, pid)
                command, _ = self.binary(pid_fd, 'cmdline')
                environment, _ = self.binary(pid_fd, 'environ')
                self.hint(pid, start, 'cmdline_hint', command)
                self.hint(pid, start, 'environ_hint', environment)
                self.link(pid_fd, 'cwd', pid, start, 'cwd_link')
                fd_dir = self.scan.open('fd', queues._DIR_FLAGS, pid_fd)
                fd_before = self.scan.call(os.fstat, fd_dir)
                fd_names = self.scan.names(fd_dir, 0)
                for descriptor in fd_names:
                    self.budget.tick()
                    queues._require(descriptor.isascii() and descriptor.isdigit() and len(descriptor) <= 10,
                                    'reference_process_fd_unknown')
                    self.link(fd_dir, descriptor, pid, start, 'fd_link')
                self.producer(pid, start, command, environment)
                pid_snapshots.append((name, pid, start, pid_fd, before, fd_dir, fd_before, fd_names))
            except (OSError, UnicodeError, LaneReferenceCollectionError):
                failures = True
                self.keep('reference_process_unavailable')
            except queues._Blocked as error:
                if error.code.endswith('limit') or error.code in queues._RESOURCE_CODES:
                    raise
                failures = True
                self.keep(error.code)
        self.proc_snapshots.append((proc_root, chain, names, pid_snapshots))
        self.process_complete = not failures

    def verify(self):
        for path, chain, identity in self.config_snapshots:
            parent = chain[-1][0]
            fd = self.scan.open(Path(path).name, queues._FILE_FLAGS, parent)
            try:
                queues._require(queues._identity(self.scan.call(os.fstat, fd)) == identity, 'reference_config_changed')
                queues._require([identity for _, identity in self.scan.walk(str(Path(path).parent))]
                                == [identity for _, identity in chain], 'reference_config_changed')
            finally:
                self.scan.close(fd)
        for proc_root, chain, names, pids in self.proc_snapshots:
            root = chain[-1][0]
            queues._require(self.scan.names(root, 1) == names, 'reference_process_inventory_changed')
            queues._require([identity for _, identity in self.scan.walk(proc_root)]
                            == [identity for _, identity in chain], 'reference_process_inventory_changed')
            for name, pid, start, pid_fd, before, fd_dir, fd_before, fd_names in pids:
                named = self.scan.call(os.stat, name, dir_fd=root, follow_symlinks=False)
                queues._require(queues._identity(named) == queues._identity(before)
                                and queues._identity(self.scan.call(os.fstat, pid_fd)) == queues._identity(before),
                                'reference_process_changed')
                raw, _ = self.binary(pid_fd, 'stat', cap=8192)
                queues._require(self.start_time(raw, pid) == start, 'reference_process_changed')
                queues._require(self.scan.names(fd_dir, 1) == fd_names
                                and queues._identity(self.scan.call(os.fstat, fd_dir)) == queues._identity(fd_before),
                                'reference_process_changed')
        self.budget.tick()

    def close(self):
        try:
            self.scan.close_all()
        except LaneScratchError as error:
            self.process_complete = False
            self.keep('reference_descriptor_ownership_unproven' if 'ownership_unproven' in str(error)
                      else 'reference_descriptor_cleanup_failed')
        for code in self.scan.blockers:
            self.keep(code)


def collect_lane_references(target_path: str, *, pins_root: str, preparation_roots: Sequence[str] = (),
                            activation_roots: Sequence[str] = (), sam_roots: Sequence[str] = (),
                            proc_root: str = '/proc', supplied_config_path: str | None = None,
                            observed_at_epoch: float, monotonic: Callable[[], float] = time.monotonic,
                            _collector_effective: bool = False,
                            _selected_settings: Sequence[tuple[str, str]] = ()) -> LaneReferenceCollection:
    """Return historical selected evidence; no probe authority survives return."""
    target, pins_root, proc_root = _path(target_path), _path(pins_root), _path(proc_root)
    target_path_object = Path(target)
    if (target_path_object.parent.parent not in LANE_ROOTS or not queues._finite(observed_at_epoch)
            or observed_at_epoch < 0 or not callable(monotonic)):
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
    preparation, activation, sam = _roots(preparation_roots), _roots(activation_roots), _roots(sam_roots)
    all_roots = preparation + activation + sam
    if len(all_roots) > 16 or len(set(all_roots)) != len(all_roots):
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
    if supplied_config_path is not None:
        supplied_config_path = _path(supplied_config_path)
    if not isinstance(_selected_settings, (tuple, list)) or len(_selected_settings) > 16:
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
    for entry in _selected_settings:
        if (not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[0], str)
                or entry[0] not in {*_ENV_PATHS.values(), 'gc_selected_queue_root'}):
            raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
        _path(entry[1])
    # Fixed initial queue/metadata selections cannot alias/contain one another.
    try:
        queues._contracts([queues.QueueRootContract(path, ('selected',)) for path in
                           (*all_roots, pins_root, proc_root)])
    except queues.QueueObservationError:
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid') from None
    origin = 'collector_process_effective_only' if _collector_effective else 'supplied_only'
    budget = ReferenceCollectionBudget(monotonic=monotonic)
    state = _Collector(target, float(observed_at_epoch), budget, origin)
    probe = TargetProbe('kept', target)
    handle = None
    primary = auxiliary = pins = interpreted = None
    try:
        budget.tick()
        for role, values in (('pins_root', (pins_root,)), ('preparation_queue_root', preparation),
                             ('activation_queue_root', activation), ('child_queue_root', sam)):
            for path in values:
                state.source(path, role, origin)
        for role, path in _selected_settings:
            state.source(path, role, origin)
        if not preparation or not activation:
            state.keep('selected_reference_family_unconfigured')
        budget.charge('raw_bytes', 8192)
        try:
            started = budget.last
            handle = LeasedScratchUse.probe(target, now=lambda: observed_at_epoch)
            budget.tick()
            parent = state.scan.walk(target)
            raw, identity = state.binary(parent[-1][0], LEASE_FILE, cap=8192)
            lease = state.json_value(raw)
            reference = 'run_ref' if 'run_ref' in lease else 'scene_ref'
            probe = TargetProbe('observed_exclusive_interval', target, 'sha256:' + hashlib.sha256(raw).hexdigest(),
                                len(raw), lease['class_intent'], lease['cleanup'], lease['lease_digest'], lease['owner'],
                                reference, lease[reference], tuple(handle.identity['inodes']), started)
            budget.retain(probe)
        except LaneScratchError as error:
            code = str(error)
            state.keep(code if code in {'lane_scratch_consumer_busy', 'lane_scratch_consumer_participation_unproven',
                                        'lane_scratch_root_lock_unavailable', 'lane_scratch_root_lock_unsafe',
                                        'lane_scratch_descriptor_ownership_unproven', 'lane_scratch_descriptor_cleanup_failed'}
                       else 'reference_target_identity_unproven')
        except OSError:
            state.keep('reference_target_identity_unproven')
        if supplied_config_path:
            try:
                state.configuration(supplied_config_path, 'supplied_historical_config')
            except (OSError, UnicodeError, LaneReferenceCollectionError):
                state.keep('reference_config_unavailable')
            except queues._Blocked as error:
                if error.code.endswith('limit') or error.code in queues._RESOURCE_CODES:
                    raise
                state.keep(error.code)
        try:
            state.proc(proc_root)
        except (OSError, UnicodeError, LaneReferenceCollectionError):
            state.keep('reference_process_inventory_unavailable')
        except queues._Blocked as error:
            if error.code.endswith('limit') or error.code in queues._RESOURCE_CODES:
                raise
            state.keep(error.code)
        # Finite family roles come only from literal configured paths, never
        # queue basenames, current row fields or a schema-free recursive guess.
        preparation = tuple(sorted(set(preparation) | {row.path for row in state.sources if row.role == 'preparation_queue_root'}))
        activation = tuple(sorted(set(activation) | {row.path for row in state.sources if row.role == 'activation_queue_root'}))
        sam = tuple(sorted(set(sam) | {row.path for row in state.sources if row.role == 'child_queue_root'}))
        if len(preparation + activation + sam) > 16:
            budget.fail('reference_roots_limit')
        contracts = [queues.QueueRootContract(path, states) for roots, states in
                     ((preparation, PREPARATION_STATES), (activation, ACTIVATION_STATES + ('identities', 'results')),
                      (sam, SAM_STATES)) for path in roots]
        if contracts:
            primary = queues.observe_queue_states(contracts, observed_at_epoch=observed_at_epoch, budget=budget)
            for code in primary.blockers:
                state.keep(code)
        aux_contracts = [AuxiliaryQueueContract(family, path) for family, values in
                         (('preparation', preparation), ('sam', sam)) for path in values]
        if aux_contracts:
            auxiliary = observe_preparation_sam_auxiliaries(aux_contracts, observed_at_epoch=observed_at_epoch, budget=budget)
            for code in auxiliary.blockers:
                state.keep(code)
        pins = observe_storage_pins(pins_root, observed_at_epoch=observed_at_epoch, budget=budget)
        for code in pins.blockers:
            state.keep(code)
        retained = []
        family_roots = {path: family for family, values in (('preparation', preparation), ('activation', activation)) for path in values}
        for row in primary.rows if primary else ():
            family = family_roots.get(row.root_path)
            if family is None:
                continue
            role = row.state if row.state in {'identities', 'results'} else 'envelope'
            role = {'identities': 'identity', 'results': 'result'}.get(role, role)
            budget.retain({'family': family, 'queue_root': row.root_path, 'role': role, 'row_path': row.row_path,
                           'raw_bytes': row.raw_text, 'observed_identity': row.row_identity})
            retained.append(RetainedReferenceRecord(family, row.root_path, role, row.row_path,
                                                    row.raw_text.encode('utf-8'), row.row_identity))
        for row in auxiliary.rows if auxiliary else ():
            if row.family != 'preparation' or row.layout_role not in {'identity', 'result', 'result_conflict'}:
                continue
            budget.retain({'family': row.family, 'queue_root': row.root_path, 'role': row.layout_role,
                           'row_path': row.row_path, 'raw_bytes': row.raw_text, 'observed_identity': row.row_identity})
            retained.append(RetainedReferenceRecord(row.family, row.root_path, row.layout_role, row.row_path,
                                                    row.raw_text.encode('utf-8'), row.row_identity))
        reference_contracts = [ReferenceFamilyContract(family, path) for path, family in family_roots.items()]
        if reference_contracts:
            interpreted = interpret_preparation_activation_references(reference_contracts, retained, budget=budget)
            for code in interpreted.blockers:
                state.keep(code)
        state.verify()
        if handle is not None:
            budget.charge('raw_bytes', 8192)
            handle.check()
            budget.tick()
            budget.measure(probe)
            probe = TargetProbe(**{**asdict(probe), 'ended_monotonic': budget.last})
        if not state.producer_seen:
            state.keep('producer_configuration_unproven')
    except (ReferenceCollectionBudgetError, queues._Blocked) as error:
        state.process_complete = False
        state.keep(error.code)
    except (LaneScratchError, ValueError, TypeError, UnicodeError, OSError):
        state.process_complete = False
        state.keep('reference_collection_incomplete')
    finally:
        state.close()
        if handle is not None:
            try:
                handle.close()
            except LaneScratchError:
                state.keep('reference_descriptor_cleanup_failed')
                probe = TargetProbe('kept_cleanup_unproven', target, held_after_return=None)
    # Ordinary bad sources preserve safe positive evidence. Exhausted finalization
    # cannot construct an amplified report; only a fixed empty incomplete packet.
    try:
        if budget.failure is not None:
            return LaneReferenceCollection(TargetProbe('kept', target, held_after_return=probe.held_after_return),
                                           origin, (), (), (), tuple(sorted(state.reasons)), False)
        budget.tick()
        complete = bool(primary and primary.complete and pins and pins.complete
                        and (not aux_contracts or auxiliary and auxiliary.complete) and state.process_complete)
        if probe.status == 'observed_exclusive_interval' and probe.ended_monotonic is None:
            probe = TargetProbe('kept_changed', target)
            complete = False
        result = LaneReferenceCollection(probe, origin, tuple(state.sources), tuple(state.configs), tuple(state.processes),
                                        tuple(sorted(state.reasons)), complete, primary, auxiliary, pins, interpreted,
                                        state.process_complete)
        budget.measure(result)
        budget.tick()
        return result
    except ReferenceCollectionBudgetError as error:
        state.keep(error.code)
        return LaneReferenceCollection(TargetProbe('kept', target, held_after_return=probe.held_after_return),
                                       origin, (), (), (), tuple(sorted(state.reasons)), False)
    finally:
        budget.close()


def collect_gc_lane_references(target_path: str, *, pins_root: str, queue_roots: Sequence[str],
                               observed_at_epoch: float, scene_intent_root: str | None = None,
                               scene_binding_root: str | None = None) -> LaneReferenceCollection:
    """Actual resolved GC arguments plus this process's finite environment only."""
    values = []
    for key, role in _ENV_PATHS.items():
        value = os.environ.get(key)
        if value:
            values.append((role, value))
    if scene_intent_root is not None:
        values.append(('intent_root', scene_intent_root))
    if scene_binding_root is not None:
        values.append(('source_binding_root', scene_binding_root))
    for root in _roots(queue_roots):
        values.append(('gc_selected_queue_root', root))
    # More than16 selected metadata paths is a bounded refusal before copying.
    if len(values) > 16:
        raise LaneReferenceCollectionError('lane_reference_parameters_invalid')
    return collect_lane_references(target_path, pins_root=pins_root, observed_at_epoch=observed_at_epoch,
                                   _collector_effective=True, _selected_settings=tuple(values))
