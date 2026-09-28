"""Bounded selected queue-state evidence for ADP-009D/day-28 disk safety.

This unused read-only seam observes explicit immediate directories. Neither
complete scoped evidence nor empty rows grant general reference or GC authority.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

MAX_ROOTS, MAX_STATES = 16, 16
MAX_ENTRIES, MAX_ROWS = 20_000, 10_000
MAX_ROW_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_VALUES, MAX_DEPTH = 100_000, 64
MAX_BLOCKERS = 32
MAX_PATH_BYTES, MAX_PATH_COMPONENTS = 4096, 64
READ_CHUNK_BYTES = 64 * 1024
PREFLIGHT_CHECK_CHARS = 1024
_STATE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_RESOURCE_CODES = {"queue_deadline_exceeded", "queue_clock_invalid", "queue_result_invalid"}


class QueueObservationError(ValueError):
    """Invalid API parameters, with fixed text only."""


class _Blocked(Exception):
    def __init__(self, code: str):
        self.code = code


@dataclass(frozen=True)
class QueueRootContract:
    root_path: str
    states: tuple[str, ...]


@dataclass(frozen=True)
class ObservedQueueRow:
    root_path: str
    state: str
    row_path: str
    raw_text: str
    raw_sha256: str
    raw_size_bytes: int
    row_identity: tuple[int, ...]


@dataclass(frozen=True)
class ObservedQueueRoot:
    root_path: str
    selected_states: tuple[str, ...]
    root_identity: tuple[int, int] | None
    state_identities: tuple[tuple[str, tuple[int, int]], ...]
    missing_states: tuple[str, ...]
    unobserved_root_entries: tuple[str, ...]


@dataclass(frozen=True)
class QueueStateObservation:
    complete: bool
    observed_at_epoch: float
    roots: tuple[ObservedQueueRoot, ...]
    rows: tuple[ObservedQueueRow, ...]
    blockers: tuple[str, ...]
    scope: str = "selected_primary_queue_states_only"
    mutations: int = 0
    execution_authorized: bool = False
    producer_seals_verified: bool = False
    general_reference_inventory_complete: bool = False
    consumer_fence_checked: bool = False


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise _Blocked(code)


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _path(value: Any) -> str:
    if not isinstance(value, str):
        raise QueueObservationError("queue_parameters_invalid")
    try:
        valid = (len(value) <= MAX_PATH_BYTES and len(value.encode("utf-8")) <= MAX_PATH_BYTES
                 and value.startswith("/") and not value.startswith("//")
                 and not any(ord(c) < 32 or ord(c) == 127 or c in "\\<>*?[]" for c in value))
    except UnicodeError:
        valid = False
    if not valid:
        raise QueueObservationError("queue_parameters_invalid")
    parts = value[1:].split("/") if value != "/" else []
    if len(parts) > MAX_PATH_COMPONENTS or any(p in {"", ".", ".."} for p in parts):
        raise QueueObservationError("queue_parameters_invalid")
    return value


def _contracts(values: Sequence[QueueRootContract]) -> tuple[QueueRootContract, ...]:
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= MAX_ROOTS:
        raise QueueObservationError("queue_parameters_invalid")
    output = []
    for value in values:
        if not isinstance(value, QueueRootContract):
            raise QueueObservationError("queue_parameters_invalid")
        root = _path(value.root_path)
        states = value.states
        if (not isinstance(states, tuple) or not 1 <= len(states) <= MAX_STATES
                or any(not isinstance(s, str) or _STATE.fullmatch(s) is None for s in states)
                or len(set(states)) != len(states)):
            raise QueueObservationError("queue_parameters_invalid")
        for previous in output:
            a, b = root.rstrip("/"), previous.root_path.rstrip("/")
            if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                raise QueueObservationError("queue_parameters_invalid")
        output.append(QueueRootContract(root, tuple(sorted(states))))
    return tuple(sorted(output, key=lambda row: row.root_path))


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _fd_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


class _Scan:
    def __init__(self, contracts: tuple[QueueRootContract, ...], observed: float,
                 clock: Callable[[], float], budget: float):
        self.contracts, self.observed, self.clock, self.budget = contracts, observed, clock, budget
        self.deadline: float | None = None
        self.last_clock: float | None = None
        self.fds: list[int] = []
        self.fd_identities: dict[int, tuple[int, int, int]] = {}
        self.failed_closes: set[int] = set()
        self.blockers: set[str] = set()
        self.roots: list[ObservedQueueRoot] = []
        self.rows: list[ObservedQueueRow] = []
        self.entries = [0, 0]
        self.bytes = self.row_count = self.values = 0
        self.root_inodes: set[tuple[int, int]] = set()
        self.snapshots: list[tuple[Any, ...]] = []
        self.opened_roots: set[str] = set()
        self._row_bytes_limit = MAX_ROW_BYTES

    def block(self, code: str) -> None:
        if code in self.blockers or len(self.blockers) < MAX_BLOCKERS:
            self.blockers.add(code)
        else:
            self.blockers.add("queue_blockers_truncated")

    def tick(self) -> None:
        try:
            current = self.clock()
            if not _finite(current):
                raise ValueError("clock")
            current = float(current)
            if self.last_clock is not None and current < self.last_clock:
                raise ValueError("clock")
            self.last_clock = current
        except Exception:
            raise _Blocked("queue_clock_invalid") from None
        if self.deadline is None:
            self.deadline = current + self.budget
        _require(current < self.deadline, "queue_deadline_exceeded")

    def call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.tick()
        result = operation(*args, **kwargs)
        self.tick()
        return result

    def forget(self, fd: int) -> None:
        self.fds.remove(fd)
        self.fd_identities.pop(fd, None)
        self.failed_closes.discard(fd)

    def close(self, fd: int) -> None:
        # Establish identity before the first close too: acquisition fstat may
        # have failed. After an ambiguous close, never close an unproven reuse.
        try:
            current = os.fstat(fd)
            identity = _fd_identity(current)
        except OSError as error:
            if error.errno == errno.EBADF:
                self.forget(fd)
                return
            identity = None
        previous = self.fd_identities.get(fd)
        if fd in self.failed_closes and (previous is None or identity is None):
            self.block("queue_descriptor_changed")
            return
        if previous is not None and identity is not None and identity != previous:
            self.block("queue_descriptor_changed")
            self.forget(fd)
            return
        if identity is not None:
            self.fd_identities[fd] = identity
        try:
            os.close(fd)
        except OSError as error:
            self.block("queue_descriptor_close_failed")
            if error.errno == errno.EBADF:
                self.forget(fd)
            else:
                self.failed_closes.add(fd)
        else:
            self.forget(fd)

    def open(self, name: str, flags: int, parent: int | None = None) -> int:
        self.tick()
        fd = os.open(name, flags, dir_fd=parent)
        # A genuinely fresh open proves any prior ownership of this numeric
        # slot ended, even if a prior close/fstat left its state ambiguous.
        while fd in self.fds:
            self.forget(fd)
        self.fds.append(fd)  # Ownership precedes every fallible post-open step.
        value = os.fstat(fd)
        self.fd_identities[fd] = _fd_identity(value)
        self.tick()
        return fd

    def walk(self, path: str) -> list[tuple[int, tuple[int, int]]]:
        chain = []
        parent = None
        for name in ["/", *path[1:].split("/")] if path != "/" else ["/"]:
            parent = self.open(name, _DIR_FLAGS, parent)
            metadata = self.call(os.fstat, parent)
            _require(stat.S_ISDIR(metadata.st_mode), "queue_root_unsafe")
            chain.append((parent, (metadata.st_dev, metadata.st_ino)))
        return chain

    def names(self, fd: int, pass_index: int) -> tuple[str, ...]:
        self.tick()
        output = []
        with os.scandir(fd) as iterator:
            self.tick()
            for entry in iterator:
                self.tick()
                self.entries[pass_index] += 1
                _require(self.entries[pass_index] <= MAX_ENTRIES, "queue_entries_limit")
                output.append(entry.name)
            self.tick()
        result = tuple(sorted(output))
        self.tick()
        return result

    def parse(self, raw: bytes) -> str:
        self.tick()

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            output = {}
            for key, value in items:
                self.tick()
                if key in output:
                    raise ValueError("duplicate")
                output[key] = value
            return output

        def number(text: str) -> int | float:
            self.tick()
            result = float(text) if any(c in text for c in ".eE") else int(text)
            if not _finite(result):
                raise ValueError("number")
            return result

        try:
            text = raw.decode("utf-8")
            self.tick()
            self.preflight(text)
            value = json.loads(text, object_pairs_hook=pairs, parse_int=number,
                               parse_float=number, parse_constant=number)
            self.tick()
            _require(isinstance(value, dict), "queue_row_invalid")
            pending = [value]
            while pending:
                self.tick()
                item = pending.pop()
                if isinstance(item, dict):
                    for key, child in item.items():
                        self.tick()
                        key.encode("utf-8")
                        pending.append(child)
                elif isinstance(item, list):
                    for child in item:
                        self.tick()
                        pending.append(child)
                elif isinstance(item, str):
                    item.encode("utf-8")
            self.tick()
            return text
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            raise _Blocked("queue_row_invalid") from None

    def preflight(self, text: str) -> None:
        """Bound containers and lexical values (including keys) before parsing.

        String-aware scanning delegates syntax to JSON's parser; it never
        interprets strings as reference paths. A character is <=4 UTF-8 bytes.
        """
        depth = 0
        quoted = escaped = atom = False
        for offset, char in enumerate(text):
            if offset % PREFLIGHT_CHECK_CHARS == 0:
                self.tick()
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
                atom = False
                self.values += 1
            elif char in "{[":
                depth += 1
                _require(depth <= MAX_DEPTH, "queue_depth_limit")
                atom = False
                self.values += 1
            elif char in "}]":
                depth -= 1
                _require(depth >= 0, "queue_row_invalid")
                atom = False
            elif char in " \t\r\n,:":
                atom = False
            elif not atom:
                atom = True
                self.values += 1
            _require(self.values <= MAX_VALUES, "queue_values_limit")
        self.tick()
        _require(not quoted and depth == 0, "queue_row_invalid")

    def read_row(self, root: str, state: str, directory: int, name: str) -> ObservedQueueRow:
        self.tick()
        _require(name.endswith(".json"), "queue_entry_unknown")
        self.row_count += 1
        _require(self.row_count <= MAX_ROWS, "queue_rows_limit")
        fd = self.open(name, _FILE_FLAGS, directory)
        try:
            before = self.call(os.fstat, fd)
            _require(stat.S_ISREG(before.st_mode), "queue_row_unsafe")
            _require(0 <= before.st_size <= self._row_bytes_limit, "queue_row_bytes_limit")
            _require(before.st_size <= MAX_TOTAL_BYTES - self.bytes, "queue_bytes_limit")
            raw = bytearray()
            while len(raw) <= before.st_size:
                chunk = self.call(os.read, fd, min(READ_CHUNK_BYTES, before.st_size + 1 - len(raw)))
                self.bytes += len(chunk)
                _require(self.bytes <= MAX_TOTAL_BYTES, "queue_bytes_limit")
                if not chunk:
                    break
                raw.extend(chunk)
            after = self.call(os.fstat, fd)
            current = self.call(os.stat, name, dir_fd=directory, follow_symlinks=False)
            _require(len(raw) == before.st_size and stat.S_ISREG(current.st_mode)
                     and _identity(before) == _identity(after) == _identity(current), "queue_row_changed")
            text = self.parse(bytes(raw))
            try:
                row_path = _path(root.rstrip("/") + "/" + state + "/" + name)
            except QueueObservationError:
                raise _Blocked("queue_row_invalid") from None
            return ObservedQueueRow(root, state, row_path, text,
                                    "sha256:" + hashlib.sha256(raw).hexdigest(), len(raw), _identity(before))
        finally:
            self.close(fd)

    def observe_root(self, contract: QueueRootContract) -> None:
        chain = self.walk(contract.root_path)
        root = chain[-1][0]
        root_identity = chain[-1][1]
        self.opened_roots.add(contract.root_path)
        _require(root_identity not in self.root_inodes, "queue_root_alias")
        self.root_inodes.add(root_identity)
        initial = self.call(os.fstat, root)
        root_names = self.names(root, 0)
        directories: dict[str, tuple[int, os.stat_result, tuple[str, ...]]] = {}
        missing = []
        self.snapshots.append((contract, chain, initial, root_names, directories))
        for state in contract.states:
            self.tick()
            if state not in root_names:
                missing.append(state)
                continue
            try:
                directory = self.open(state, _DIR_FLAGS, root)
                before = self.call(os.fstat, directory)
                names = self.names(directory, 0)
                directories[state] = directory, before, names
            except OSError:
                self.block("queue_state_unavailable")
                continue
            for name in names:
                try:
                    self.rows.append(self.read_row(contract.root_path, state, directory, name))
                except OSError:
                    self.block("queue_row_unavailable")
                except _Blocked as error:
                    if error.code.endswith("limit") or error.code in _RESOURCE_CODES:
                        raise
                    self.block(error.code)
        self.roots.append(ObservedQueueRoot(
            contract.root_path, contract.states, root_identity,
            tuple((state, (metadata.st_dev, metadata.st_ino)) for state, (_, metadata, _) in sorted(directories.items())),
            tuple(missing), tuple(name for name in root_names if name not in contract.states)))

    def verify_root(self, snapshot: tuple[Any, ...]) -> None:
        contract, chain, initial, root_names, directories = snapshot
        root = chain[-1][0]
        _require(self.names(root, 1) == root_names and _identity(self.call(os.fstat, root)) == _identity(initial),
                 "queue_directory_changed")
        for state, (directory, before, names) in directories.items():
            self.tick()
            _require(self.names(directory, 1) == names and _identity(self.call(os.fstat, directory)) == _identity(before),
                     "queue_directory_changed")
            named = self.call(os.stat, state, dir_fd=root, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) and _identity(named) == _identity(before), "queue_directory_changed")
        for row in self.rows:
            self.tick()
            if row.root_path != contract.root_path:
                continue
            directory = directories[row.state][0]
            fd = self.open(row.row_path.rsplit("/", 1)[1], _FILE_FLAGS, directory)
            try:
                current = self.call(os.fstat, fd)
                _require(stat.S_ISREG(current.st_mode) and _identity(current) == row.row_identity, "queue_row_changed")
            finally:
                self.close(fd)
        _require([identity for _, identity in self.walk(contract.root_path)] == [identity for _, identity in chain],
                 "queue_root_changed")

    def result(self) -> QueueStateObservation:
        try:
            self.tick()
            observed = {row.root_path: row for row in self.roots}
            roots = tuple(observed.get(contract.root_path, ObservedQueueRoot(
                contract.root_path, contract.states, None, (), (), ())) for contract in self.contracts)
            self.tick()
            rows = tuple(sorted(self.rows, key=lambda row: (row.root_path, row.state, row.row_path)))
            self.tick()
            result = QueueStateObservation(not self.blockers, self.observed, roots, rows, tuple(sorted(self.blockers)))
            document = asdict(result)
            self.tick()
            self.output_size(document)
            self.tick()
            return result
        except _Blocked as error:
            self.block(error.code)
        except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
            self.block("queue_result_invalid")
        return QueueStateObservation(False, self.observed, (), (), tuple(sorted(self.blockers)))

    def output_size(self, document: dict[str, Any]) -> None:
        """Count default-spaced, ensure_ascii=False JSON without allocating it.

        iterencode may allocate a whole escaped16MiB string before yielding;
        private output has fixed shallow structure, so count its scalar bytes.
        """
        size = 0

        def add(value: int) -> None:
            nonlocal size
            size += value
            _require(size <= MAX_OUTPUT_BYTES, "queue_output_limit")

        def string_size(value: str) -> None:
            add(2)
            for offset, char in enumerate(value):
                if offset % PREFLIGHT_CHECK_CHARS == 0:
                    self.tick()
                ordinal = ord(char)
                if char in '\\"\b\f\n\r\t':
                    add(2)
                elif ordinal < 32:
                    add(6)
                elif ordinal < 128:
                    add(1)
                elif ordinal < 2048:
                    add(2)
                elif 0xD800 <= ordinal <= 0xDFFF:
                    raise _Blocked("queue_result_invalid")
                elif ordinal < 65536:
                    add(3)
                else:
                    add(4)
            self.tick()

        def visit(value: Any) -> None:
            self.tick()
            if isinstance(value, str):
                string_size(value)
            elif isinstance(value, dict):
                add(2)
                for index, (key, child) in enumerate(value.items()):
                    self.tick()
                    if index:
                        add(2)
                    string_size(key)
                    add(2)
                    visit(child)
            elif isinstance(value, (tuple, list)):
                add(2)
                for index, child in enumerate(value):
                    self.tick()
                    if index:
                        add(2)
                    visit(child)
            elif value is None:
                add(4)
            elif isinstance(value, bool):
                add(4 if value else 5)
            elif _finite(value):
                add(len(str(value)))
            else:
                raise _Blocked("queue_result_invalid")

        visit(document)
        self.tick()


def observe_queue_states(contracts: Sequence[QueueRootContract], *, observed_at_epoch: float,
                         monotonic: Callable[[], float] = time.monotonic,
                         time_budget_seconds: float = 5.0) -> QueueStateObservation:
    """Observe explicitly selected primary states; never repair or clear references."""
    normalized = _contracts(contracts)
    if not (_finite(observed_at_epoch) and observed_at_epoch >= 0 and _finite(time_budget_seconds)
            and 0 < time_budget_seconds <= 5 and callable(monotonic)):
        raise QueueObservationError("queue_parameters_invalid")
    scan = _Scan(normalized, float(observed_at_epoch), monotonic, float(time_budget_seconds))
    exhausted = False
    try:
        for contract in normalized:
            try:
                scan.observe_root(contract)
            except FileNotFoundError:
                scan.block("queue_root_missing" if contract.root_path not in scan.opened_roots
                           else "queue_inventory_changed")
            except OSError:
                scan.block("queue_inventory_unavailable")
            except _Blocked as error:
                scan.block(error.code)
                if error.code.endswith("limit") or error.code in _RESOURCE_CODES:
                    exhausted = True
                    break
            except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
                scan.block("queue_inventory_invalid")
        if not exhausted:
            # All initial reads precede every final pass, including other roots.
            # This is a stability check, never atomic publication/use exclusion.
            for snapshot in scan.snapshots:
                try:
                    scan.verify_root(snapshot)
                except OSError:
                    scan.block("queue_inventory_changed")
                except _Blocked as error:
                    scan.block(error.code)
                    if error.code.endswith("limit") or error.code in _RESOURCE_CODES:
                        break
    finally:
        for _pass in range(2):
            for fd in tuple(reversed(scan.fds)):
                scan.close(fd)
    return scan.result()
