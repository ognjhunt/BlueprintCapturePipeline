"""Bounded read-only pin observations for ADP-009D/day-28 disk safety.

Completeness covers this ledger observation only. No general reference or
consumer fence, authenticated release, eviction or execution authority follows.
"""
from __future__ import annotations

import fcntl
import errno
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .control_plane_reference_budget import ReferenceCollectionBudget

from .control_plane_storage_pins import PIN_KINDS, SCHEMA_VERSION

MAX_ROWS = MAX_VALUES = 10_000
MAX_ENTRIES = 10_032
MAX_ROW_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_BLOCKERS = 32
MAX_PATH_BYTES, MAX_PATH_COMPONENTS = 4096, 64
READ_CHUNK_BYTES = 64 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
_FIELDS = {"schema_version", "kind", "owner_id", "paths", "depends_on",
           "created_at_epoch", "expires_at_epoch", "released_at_epoch"}
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class StoragePinObservationError(ValueError):
    """Invalid API parameters; fixed text never echoes input."""


class _Blocked(Exception):
    def __init__(self, code: str):
        self.code = code


@dataclass(frozen=True, order=True)
class PinIdentity:
    kind: str
    owner_id: str


@dataclass(frozen=True)
class ObservedStoragePin:
    kind: str
    owner_id: str
    paths: tuple[str, ...]
    depends_on: tuple[PinIdentity, ...]
    created_at_epoch: float
    expires_at_epoch: float
    released_at_epoch: float | None
    status: str
    row_path: str
    raw_sha256: str
    raw_size_bytes: int
    row_identity: tuple[int, ...]


@dataclass(frozen=True)
class StoragePinObservation:
    complete: bool
    observed_at_epoch: float
    root_path: str
    root_identity: tuple[int, int] | None
    rows: tuple[ObservedStoragePin, ...]
    protected_identities: tuple[PinIdentity, ...]
    protected_paths: tuple[str, ...]
    blockers: tuple[str, ...]
    scope: str = "storage_pins_only"
    mutations: int = 0
    execution_authorized: bool = False
    general_reference_inventory_complete: bool = False
    consumer_fence_checked: bool = False


def _require(condition: bool, code: str = "pin_row_invalid") -> None:
    if not condition:
        raise _Blocked(code)


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _path(value: Any) -> str:
    if not isinstance(value, str):
        raise StoragePinObservationError("pin_parameters_invalid")
    try:
        valid = (len(value) <= MAX_PATH_BYTES and len(value.encode("utf-8")) <= MAX_PATH_BYTES
                 and value.startswith("/") and not value.startswith("//")
                 and not any(ord(c) < 32 or ord(c) == 127 or c in "\\<>*" for c in value))
    except UnicodeError:
        valid = False
    if not valid:
        raise StoragePinObservationError("pin_parameters_invalid")
    parts = value[1:].split("/") if value != "/" else []
    if len(parts) > MAX_PATH_COMPONENTS or any(p in {"", ".", ".."} for p in parts):
        raise StoragePinObservationError("pin_parameters_invalid")
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate")
        value[key] = item
    return value


def _number(text: str) -> int | float:
    value = float(text) if any(c in text for c in ".eE") else int(text)
    if not _finite(value):
        raise ValueError("number")
    return value


def _json(raw: bytes, *, budget: ReferenceCollectionBudget | None = None) -> dict[str, Any]:
    if budget is not None:
        from .control_plane_reference_budget import ReferenceCollectionBudgetError
        try:
            budget.preflight(raw.decode("utf-8"))
        except ReferenceCollectionBudgetError as error:
            raise _Blocked(error.code) from None
        except UnicodeError:
            raise _Blocked("pin_row_invalid") from None
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_int=_number, parse_float=_number,
                           parse_constant=lambda text: _number(text))
        if budget is not None:
            try:
                budget.measure(value)
            except ReferenceCollectionBudgetError as error:
                raise _Blocked(error.code) from None
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        _require(isinstance(value, dict) and set(value) == _FIELDS)
        return value
    except (ValueError, UnicodeError, OverflowError, RecursionError, TypeError):
        raise _Blocked("pin_row_invalid") from None


class _Scan:
    def __init__(self, root: str, observed: float, clock: Callable[[], float], budget: float,
                 shared: ReferenceCollectionBudget | None = None):
        self.root, self.observed, self.clock = root, observed, clock
        self.deadline: float | None = None
        self.budget = budget
        self.shared = shared
        self.fds: list[int] = []
        self.fd_identities: dict[int, tuple[int, int, int]] = {}
        self.failed_closes: set[int] = set()
        self.last_clock: float | None = None
        self.rows: list[ObservedStoragePin] = []
        self.blockers: set[str] = set()
        self.root_identity: tuple[int, int] | None = None
        self.bytes = self.values = self.output_bytes = 0
        self.entries = [0, 0]
        self.row_count = 0

    def tick(self) -> None:
        if self.shared is not None:
            from .control_plane_reference_budget import ReferenceCollectionBudgetError
            try:
                self.shared.tick()
                return
            except ReferenceCollectionBudgetError as error:
                raise _Blocked(error.code) from None
        try:
            current = self.clock()
            if not _finite(current):
                raise ValueError("clock")
            current = float(current)
            if self.last_clock is not None and current < self.last_clock:
                raise ValueError("clock")
            self.last_clock = current
        except Exception:
            raise _Blocked("pin_clock_invalid") from None
        if self.deadline is None:
            self.deadline = current + self.budget
        _require(current < self.deadline, "pin_deadline_exceeded")

    def shared_charge(self, kind: str, amount: int = 1) -> None:
        if self.shared is not None:
            from .control_plane_reference_budget import ReferenceCollectionBudgetError
            try:
                self.shared.charge(kind, amount)
            except ReferenceCollectionBudgetError as error:
                raise _Blocked(error.code) from None

    def shared_retain(self, value: Any) -> None:
        if self.shared is not None:
            from .control_plane_reference_budget import ReferenceCollectionBudgetError
            try:
                self.shared.retain(value)
            except ReferenceCollectionBudgetError as error:
                raise _Blocked(error.code) from None

    def block(self, code: str) -> None:
        if self.shared is not None:
            self.shared.block(code)
        if code in self.blockers or len(self.blockers) < MAX_BLOCKERS:
            self.blockers.add(code)
        else:
            self.blockers.add("pin_blockers_truncated")

    def close(self, fd: int) -> None:
        if fd in self.failed_closes:
            # A close error may mean it closed. Never blindly close a reused FD.
            try:
                current = os.fstat(fd)
                same = self.fd_identities.get(fd) == (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode))
            except OSError as error:
                if error.errno == errno.EBADF:
                    self.forget(fd)
                return
            if not same:
                self.block("pin_descriptor_changed")
                self.forget(fd)
                return
        try:
            os.close(fd)
        except OSError as error:
            self.block("pin_descriptor_close_failed")
            if error.errno == errno.EBADF:
                self.forget(fd)
            else:
                self.failed_closes.add(fd)
        else:
            self.forget(fd)

    def forget(self, fd: int) -> None:
        self.fds.remove(fd)
        self.fd_identities.pop(fd, None)
        self.failed_closes.discard(fd)

    def call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.tick()
        value = operation(*args, **kwargs)
        self.tick()
        return value

    def open(self, name: str, flags: int, parent: int | None = None) -> int:
        self.tick()
        fd = os.open(name, flags, dir_fd=parent)
        self.fds.append(fd)  # Own it before the post-syscall deadline check.
        self.tick()
        value = self.call(os.fstat, fd)
        self.fd_identities[fd] = (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))
        self.tick()
        return fd

    def walk(self) -> list[tuple[int, tuple[int, int]]]:
        chain = []
        parent = None
        for name in ["/", *self.root.strip("/").split("/")] if self.root != "/" else ["/"]:
            parent = self.open(name, _DIR_FLAGS, parent)
            observed = self.call(os.fstat, parent)
            _require(stat.S_ISDIR(observed.st_mode), "pin_root_unsafe")
            chain.append((parent, (observed.st_dev, observed.st_ino)))
        return chain

    def names(self, fd: int, pass_index: int) -> tuple[str, ...]:
        self.tick()
        with os.scandir(fd) as iterator:
            self.tick()
            names = []
            for entry in iterator:
                self.tick()
                self.entries[pass_index] += 1
                _require(self.entries[pass_index] <= MAX_ENTRIES, "pin_entries_limit")
                self.shared_charge("entries")
                self.shared_retain(entry.name)
                names.append(entry.name)
            self.tick()
        return tuple(sorted(names))

    def read_row(self, directory: int, kind: str, name: str) -> ObservedStoragePin:
        self.tick()
        self.row_count += 1
        _require(self.row_count <= MAX_ROWS, "pin_rows_limit")
        self.shared_charge("rows")
        _require(name.endswith(".json") and _ID.fullmatch(name[:-5]) is not None, "pin_entry_unknown")
        fd = self.open(name, _FILE_FLAGS, directory)
        try:
            before = self.call(os.fstat, fd)
            _require(stat.S_ISREG(before.st_mode), "pin_row_unavailable")
            _require(before.st_size <= MAX_ROW_BYTES, "pin_row_bytes_limit")
            _require(before.st_size <= MAX_TOTAL_BYTES - self.bytes, "pin_bytes_limit")
            if self.shared is not None:
                from .control_plane_reference_budget import ReferenceCollectionBudgetError
                try:
                    self.shared.available("raw_bytes", before.st_size)
                except ReferenceCollectionBudgetError as error:
                    raise _Blocked(error.code) from None
            raw = bytearray()
            while len(raw) <= before.st_size:
                chunk = self.call(os.read, fd, min(READ_CHUNK_BYTES, before.st_size + 1 - len(raw)))
                self.bytes += len(chunk)
                _require(self.bytes <= MAX_TOTAL_BYTES, "pin_bytes_limit")
                self.shared_charge("raw_bytes", len(chunk))
                if not chunk:
                    break
                raw.extend(chunk)
            after = self.call(os.fstat, fd)
            current = self.call(os.stat, name, dir_fd=directory, follow_symlinks=False)
            _require(len(raw) == before.st_size and _identity(before) == _identity(after) == _identity(current)
                     and stat.S_ISREG(current.st_mode), "pin_row_changed")
            value = _json(bytes(raw)) if self.shared is None else _json(bytes(raw), budget=self.shared)
            row = self.validate(value, kind, name, bytes(raw), _identity(before))
            self.tick()
            return row
        finally:
            self.close(fd)

    def validate(self, value: dict[str, Any], kind: str, name: str,
                 raw: bytes, identity: tuple[int, ...]) -> ObservedStoragePin:
        self.tick()
        _require(value["schema_version"] == SCHEMA_VERSION and value["kind"] == kind
                 and value["owner_id"] == name[:-5])
        created, expires, released = (value[k] for k in ("created_at_epoch", "expires_at_epoch", "released_at_epoch"))
        _require(_finite(created) and _finite(expires) and created >= 0 and expires > created
                 and (released is None or (_finite(released) and created <= released <= self.observed)))
        paths, dependencies = value["paths"], value["depends_on"]
        _require(isinstance(paths, list) and isinstance(dependencies, list))
        self.values += len(paths) + len(dependencies)
        _require(self.values <= MAX_VALUES, "pin_values_limit")
        self.shared_charge("facts", len(paths) + len(dependencies))
        normalized_paths = set()
        for path in paths:
            self.tick()
            try:
                normalized_paths.add(_path(path))
            except StoragePinObservationError:
                raise _Blocked("pin_row_invalid") from None
        normalized_dependencies = set()
        for dependency in dependencies:
            self.tick()
            _require(isinstance(dependency, dict) and set(dependency) == {"kind", "owner_id"}
                     and dependency["kind"] in PIN_KINDS and isinstance(dependency["owner_id"], str)
                     and _ID.fullmatch(dependency["owner_id"]) is not None)
            normalized_dependencies.add(PinIdentity(**dependency))
        status = "released" if released is not None else ("live" if expires > self.observed else "expired_unreleased")
        try:
            row_path = _path(self.root.rstrip("/") + "/" + kind + "/" + name)
        except StoragePinObservationError:
            raise _Blocked("pin_row_invalid") from None
        self.shared_retain({"kind": kind, "owner_id": name[:-5], "paths": tuple(sorted(normalized_paths)),
                            "depends_on": tuple(sorted(normalized_dependencies)), "raw_sha256": "sha256:" + "0" * 64,
                            "raw_size_bytes": len(raw), "row_path": row_path, "row_identity": identity,
                            "created_at_epoch": created, "expires_at_epoch": expires, "released_at_epoch": released, "status": status})
        return ObservedStoragePin(kind, name[:-5], tuple(sorted(normalized_paths)), tuple(sorted(normalized_dependencies)),
                                  float(created), float(expires), None if released is None else float(released), status,
                                  row_path,
                                  "sha256:" + hashlib.sha256(raw).hexdigest(), len(raw), identity)

    def observe(self) -> None:
        chain = self.walk()
        root = chain[-1][0]
        self.root_identity = chain[-1][1]
        try:
            self.call(fcntl.flock, root, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _Blocked("pin_inventory_busy") from None
        directories = [(root, self.call(os.fstat, root), self.names(root, 0))]
        kinds: dict[str, int] = {}
        for kind in directories[0][2]:
            self.tick()
            if kind not in PIN_KINDS:
                self.block("pin_entry_unknown")
                continue
            try:
                fd = self.open(kind, _DIR_FLAGS, root)
                kinds[kind] = fd
                initial = self.call(os.fstat, fd)
                names = self.names(fd, 0)
                directories.append((fd, initial, names))
            except OSError:
                self.block("pin_kind_unavailable")
                continue
            for name in names:
                try:
                    row = self.read_row(fd, kind, name)
                    self.tick()
                    size = len(json.dumps(asdict(row), ensure_ascii=False).encode("utf-8"))
                    self.tick()
                    self.output_bytes += size
                    _require(self.output_bytes <= MAX_OUTPUT_BYTES, "pin_output_limit")
                    self.rows.append(row)
                except OSError:
                    self.block("pin_row_unavailable")
                except _Blocked as error:
                    if error.code.endswith("limit") or error.code in {"pin_deadline_exceeded", "pin_clock_invalid", "reference_deadline_exceeded",
                                                                       "reference_clock_invalid", "reference_budget_closed",
                                                                       "reference_output_invalid"}:
                        raise
                    self.block(error.code)
        for fd, initial, names in directories:
            _require(self.names(fd, 1) == names and _identity(self.call(os.fstat, fd)) == _identity(initial),
                     "pin_directory_changed")
        for row in self.rows:
            fd = self.open(row.owner_id + ".json", _FILE_FLAGS, kinds[row.kind])
            try:
                current = self.call(os.fstat, fd)
                _require(stat.S_ISREG(current.st_mode) and _identity(current) == row.row_identity, "pin_row_changed")
            finally:
                self.close(fd)
        _require([identity for _, identity in self.walk()] == [identity for _, identity in chain], "pin_root_changed")

    def fallback(self, code: str) -> StoragePinObservation:
        """Bounded empty evidence is always unknown, never negative authority."""
        self.block(code)
        return StoragePinObservation(False, self.observed, self.root, self.root_identity,
                                     (), (), (), tuple(sorted(self.blockers)))

    def result(self) -> StoragePinObservation:
        try:
            self.tick()  # Refuse before any accepted-evidence sort/allocation.
            rows = tuple(sorted(self.rows, key=lambda row: (row.kind, row.owner_id)))
            self.tick()
            indexed = {}
            pending = []
            for row in rows:
                self.tick()
                identity = PinIdentity(row.kind, row.owner_id)
                indexed[identity] = row
                if row.released_at_epoch is None:
                    pending.append(identity)
            self.tick()
            for row in rows:
                self.tick()
                for dependency in row.depends_on:
                    self.tick()
                    if dependency not in indexed:
                        self.block("pin_dependency_unavailable")
            protected: set[PinIdentity] = set()
            while pending:
                self.tick()
                identity = pending.pop()
                if identity in protected:
                    continue
                protected.add(identity)
                if identity in indexed:
                    for dependency in indexed[identity].depends_on:
                        self.tick()
                        pending.append(dependency)
            paths = set()
            for identity in protected:
                self.tick()
                if identity in indexed:
                    for path in indexed[identity].paths:
                        self.tick()
                        paths.add(path)
            self.tick()
            identities = tuple(sorted(protected))
            self.tick()
            protected_paths = tuple(sorted(paths))
            self.tick()
            result = StoragePinObservation(not self.blockers, self.observed, self.root, self.root_identity,
                                           rows, identities, protected_paths, tuple(sorted(self.blockers)))
            self.tick()
            if self.shared is not None:
                from .control_plane_reference_budget import ReferenceCollectionBudgetError
                try:
                    self.shared.measure(result)
                except ReferenceCollectionBudgetError as error:
                    raise _Blocked(error.code) from None
            document = asdict(result)
            self.tick()
            size = 0
            for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(document):
                self.tick()
                size += len(chunk.encode("utf-8"))
                _require(size <= MAX_OUTPUT_BYTES, "pin_output_limit")
            self.tick()
            return result
        except _Blocked as error:
            return self.fallback(error.code)
        except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError, OSError):
            return self.fallback("pin_result_invalid")


def observe_storage_pins(pins_root: str, *, observed_at_epoch: float,
                         monotonic: Callable[[], float] = time.monotonic,
                         time_budget_seconds: float = 5.0,
                         budget: ReferenceCollectionBudget | None = None) -> StoragePinObservation:
    """Read one explicit ledger, never repairing it or clearing general references."""
    if budget is not None:
        from .control_plane_reference_budget import bind_budget
        bind_budget(budget, monotonic=monotonic, time_budget_seconds=time_budget_seconds,
                    error=StoragePinObservationError, code="pin_parameters_invalid")
    root = _path(pins_root)
    if not (_finite(observed_at_epoch) and _finite(time_budget_seconds)
            and 0 < time_budget_seconds <= 5 and callable(monotonic)):
        raise StoragePinObservationError("pin_parameters_invalid")
    scan = _Scan(root, float(observed_at_epoch), monotonic, float(time_budget_seconds), budget)
    try:
        scan.shared_charge("roots")
        scan.shared_charge("groups", len(PIN_KINDS))
        scan.observe()
    except FileNotFoundError:
        scan.block("pin_root_missing" if scan.root_identity is None else "pin_inventory_changed")
    except OSError:
        scan.block("pin_inventory_unavailable")
    except _Blocked as error:
        scan.block(error.code)
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        scan.block("pin_inventory_invalid")
    finally:
        # One bounded second cleanup pass also covers a definite close failure
        # occurring for the first time in final cleanup. close verifies identity.
        for _pass in range(2):
            for fd in tuple(reversed(scan.fds)):
                scan.close(fd)
    return scan.result()
