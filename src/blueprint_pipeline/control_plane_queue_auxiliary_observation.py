"""Unused bounded preparation/SAM receipt locations for ADP-009D/day-28.

Layout labels and raw versions prove neither consumer bindings nor authority.
The fixed subset never repairs queues, follows payload references or takes locks.
"""
from __future__ import annotations

import os
import re
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from .control_plane_reference_budget import ReferenceCollectionBudget

from . import control_plane_queue_observation as primary

MAX_DIRECTORIES, MAX_FDS = 512, 768
MAX_AUXILIARY_ROW_BYTES = 4 * 1024 * 1024
_HEX = r"[0-9a-f]{64}"
_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}"
_STEM = _ID + "-" + _HEX
_CHILD = "sam31-" + _HEX
_SEQ = r"[0-9]{6,}"
# Literal source layouts, not caller-selectable traversal policies.
_ROLES = {
    "activation": (
        ("identities", "identity", _ID + r"\.json", None),
        ("results", "result", _STEM + r"\.json", None),
        ("results/conflicts", "result_conflict", _STEM + "-" + _HEX + r"\.json", None),
    ),
    "preparation": (
        ("identities", "identity", _ID + r"\.json", None),
        ("results", "result", _STEM + r"\.json", None),
        ("results/conflicts", "result_conflict", _STEM + "-" + _HEX + r"\.json", None),
        ("source-progress", "source_progress", _SEQ + "-" + _HEX + r"\.json", _STEM),
        ("source-resume-pending", "resume_pending", _HEX + r"\.json", None),
        ("source-resume-blocked", "resume_blocked", _HEX + r"(?:\.failure)?\.json", None),
        ("source-resume-completed", "resume_completed", _HEX + r"\.json", _STEM),
    ),
    "sam": (
        ("results", "result", _CHILD + r"(?:\.conflict-" + _HEX + r")?\.json", None),
        ("started", "started", _CHILD + r"\.json", None),
        ("progress", "progress", _SEQ + r"\.json", _CHILD),
        ("wake-pending", "wake_pending", _CHILD + r"\.json", None),
        ("wake-completed", "wake_completed", _CHILD + r"\.json", None),
    ),
}


class AuxiliaryQueueObservationError(ValueError):
    """Fixed invalid-parameter refusal without supplied text."""


@dataclass(frozen=True)
class AuxiliaryQueueContract:
    family: Literal["preparation", "activation", "sam"]
    root_path: str


@dataclass(frozen=True)
class ObservedAuxiliaryRow:
    family: str
    layout_role: str
    expected_container_role: str
    root_path: str
    relative_directory: str
    row_path: str
    raw_text: str
    raw_sha256: str
    raw_size_bytes: int
    row_identity: tuple[int, ...]


@dataclass(frozen=True)
class ObservedAuxiliaryDirectory:
    relative_path: str
    identity: tuple[int, int] | None
    status: str


@dataclass(frozen=True)
class ObservedAuxiliaryRoot:
    family: str
    root_path: str
    root_identity: tuple[int, int] | None
    attempted_roles: tuple[str, ...]
    role_directories: tuple[ObservedAuxiliaryDirectory, ...]
    grouping_directories: tuple[str, ...]
    unobserved_root_entries: tuple[str, ...]


@dataclass(frozen=True)
class AuxiliaryQueueObservation:
    complete: bool
    observed_at_epoch: float
    roots: tuple[ObservedAuxiliaryRoot, ...]
    rows: tuple[ObservedAuxiliaryRow, ...]
    blockers: tuple[str, ...]
    scope: str = "preparation_sam_auxiliary_layouts_only"
    mutations: int = 0
    execution_authorized: bool = False
    producer_seals_verified: bool = False
    consumer_bindings_verified: bool = False
    history_chain_complete: bool = False
    general_reference_inventory_complete: bool = False
    consumer_fence_checked: bool = False


def _contracts(values: Sequence[AuxiliaryQueueContract]) -> tuple[AuxiliaryQueueContract, ...]:
    try:
        if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= primary.MAX_ROOTS:
            raise ValueError
        roots = []
        for value in values:
            if not isinstance(value, AuxiliaryQueueContract) or value.family not in _ROLES:
                raise ValueError
            roots.append(primary.QueueRootContract(value.root_path, ("auxiliary",)))
        primary._contracts(roots)
        return tuple(sorted(values, key=lambda value: value.root_path))
    except (ValueError, TypeError):
        raise AuxiliaryQueueObservationError("auxiliary_parameters_invalid") from None


def _attempted(family: str) -> tuple[str, ...]:
    roles = {row[1] for row in _ROLES[family]}
    if family == "preparation":
        roles.add("resume_failure")
    elif family in {"activation", "sam"}:
        roles.add("result_conflict")
    return tuple(sorted(roles))


class _AuxScan(primary._Scan):
    def __init__(self, contracts, observed, clock, budget, shared=None):
        super().__init__((), observed, clock, budget, shared)
        self.aux_contracts = contracts
        self._row_bytes_limit = MAX_AUXILIARY_ROW_BYTES
        self.directory_count = 0
        self.aux_rows: list[ObservedAuxiliaryRow] = []
        self.aux_roots: dict[str, ObservedAuxiliaryRoot] = {}
        self.aux_snapshots = []

    def open(self, name: str, flags: int, parent: int | None = None) -> int:
        primary._require(len(self.fds) < MAX_FDS, "queue_fds_limit")
        return super().open(name, flags, parent)

    def location(self, root: str, relative: str) -> str:
        try:
            path = primary._path(root.rstrip("/") + "/" + relative)
            primary._require(all(len(part.encode("utf-8")) <= 255 for part in relative.split("/")),
                             "auxiliary_location_invalid")
            return path
        except primary.QueueObservationError:
            raise primary._Blocked("auxiliary_location_invalid") from None

    def evidence(self, relative, identity, status):
        self.shared_retain({"relative_path": relative, "identity": identity, "status": status})
        return ObservedAuxiliaryDirectory(relative, identity, status)

    def directory(self, contract, relative, parent, name, directories, evidence):
        self.tick()
        self.location(contract.root_path, relative)
        primary._require(self.directory_count < MAX_DIRECTORIES, "queue_directories_limit")
        self.directory_count += 1
        try:
            fd = self.open(name, primary._DIR_FLAGS, parent)
            metadata = self.call(os.fstat, fd)
            names = self.names(fd, 0)
            directories[relative] = (fd, metadata, names, parent, name)
            evidence[relative] = self.evidence(relative, (metadata.st_dev, metadata.st_ino), "observed")
            return fd, names
        except OSError:
            self.block("auxiliary_directory_unavailable")
            evidence[relative] = self.evidence(relative, None, "unavailable")
            return None

    def rows_in(self, contract, relative, fd, names, role, pattern):
        for name in names:
            self.tick()
            # Only the reviewed reserved child is processed by a separate role.
            if contract.family in {"preparation", "activation"} and relative == "results" and name == "conflicts":
                continue
            try:
                self.location(contract.root_path, relative + "/" + name)
                metadata = self.call(os.stat, name, dir_fd=fd, follow_symlinks=False)
                primary._require(stat.S_ISREG(metadata.st_mode) and name.endswith(".json"),
                                 "auxiliary_entry_unknown")
                recognized = re.fullmatch(pattern, name) is not None
                if role in {"source_progress", "progress"}:
                    recognized = recognized and int(name.split("-", 1)[0].split(".", 1)[0]) > 0
                layout_role = role if recognized else "unrecognized_row"
                if recognized and role == "resume_blocked" and name.endswith(".failure.json"):
                    layout_role = "resume_failure"
                if recognized and contract.family == "sam" and role == "result" and ".conflict-" in name:
                    layout_role = "result_conflict"
                if not recognized:
                    self.block("auxiliary_layout_unknown")
                row = self.read_row(contract.root_path, relative, fd, name)
                self.shared_retain({"family": contract.family, "layout_role": layout_role,
                                    "expected_container_role": layout_role if recognized else role,
                                    "relative_directory": relative, "root_path": contract.root_path})
                self.aux_rows.append(ObservedAuxiliaryRow(
                    contract.family, layout_role, layout_role if recognized else role, contract.root_path,
                    relative, row.row_path, row.raw_text, row.raw_sha256, row.raw_size_bytes, row.row_identity))
            except OSError:
                self.block("queue_row_unavailable")
            except primary._Blocked as error:
                if error.code.endswith("limit") or error.code in primary._RESOURCE_CODES:
                    raise
                self.block(error.code)

    def observe_aux_root(self, contract):
        chain = self.walk(contract.root_path)
        root, identity = chain[-1]
        self.opened_roots.add(contract.root_path)
        primary._require(identity not in self.root_inodes, "queue_root_alias")
        self.root_inodes.add(identity)
        primary._require(self.directory_count < MAX_DIRECTORIES, "queue_directories_limit")
        self.directory_count += 1
        initial = self.call(os.fstat, root)
        root_names = self.names(root, 0)
        directories = {"": (root, initial, root_names, None, None)}
        evidence = {}
        groups = []
        self.aux_snapshots.append((contract, chain, directories))
        try:
            for relative, role, pattern, grouping in _ROLES[contract.family]:
                self.tick()
                parent_relative, _, name = relative.rpartition("/")
                parent_record = directories.get(parent_relative)
                if parent_record is None or name not in parent_record[2]:
                    evidence[relative] = self.evidence(relative, None, "missing_unproven")
                    self.block("auxiliary_directory_missing_unproven")
                    continue
                opened = self.directory(contract, relative, parent_record[0], name, directories, evidence)
                if opened is None:
                    continue
                fd, names = opened
                if grouping is None:
                    self.rows_in(contract, relative, fd, names, role, pattern)
                    continue
                for group in names:
                    self.tick()
                    self.shared_charge("groups")
                    try:
                        self.location(contract.root_path, relative + "/" + group)
                        child_relative = relative + "/" + group
                        if re.fullmatch(grouping, group) is None:
                            evidence[child_relative] = self.evidence(child_relative, None, "unknown_layout")
                            raise primary._Blocked("auxiliary_group_unknown")
                        self.shared_retain(child_relative)
                        groups.append(child_relative)
                        child = self.directory(contract, child_relative, fd, group, directories, evidence)
                        if child is not None:
                            self.rows_in(contract, child_relative, child[0], child[1], role, pattern)
                    except primary._Blocked as error:
                        if error.code.endswith("limit") or error.code in primary._RESOURCE_CODES:
                            raise
                        self.block(error.code)
        finally:
            # An exhausted/invalid clock forbids even partial typed conversion.
            # The public finalization fallback returns empty incomplete evidence.
            self.tick()
            top_names = {row[0].split("/", 1)[0] for row in _ROLES[contract.family]}
            evidence_keys = sorted(evidence)
            self.tick()
            observed_directories = []
            for key in evidence_keys:
                self.tick()
                observed_directories.append(evidence[key])
            sorted_groups = sorted(groups)
            self.tick()
            observed_groups = []
            for group in sorted_groups:
                self.tick()
                observed_groups.append(group)
            unobserved_names = []
            for name in root_names:
                self.tick()
                if name not in top_names:
                    unobserved_names.append(name)
            self.tick()
            self.shared_retain({"family": contract.family, "root_path": contract.root_path,
                                "root_identity": identity, "attempted_roles": _attempted(contract.family),
                                "unobserved_root_entries": tuple(unobserved_names)})
            self.aux_roots[contract.root_path] = ObservedAuxiliaryRoot(
                contract.family, contract.root_path, identity, _attempted(contract.family),
                tuple(observed_directories), tuple(observed_groups), tuple(unobserved_names))
            self.tick()

    def verify_aux_root(self, snapshot):
        contract, chain, directories = snapshot
        for relative, (fd, metadata, names, parent, name) in directories.items():
            self.tick()
            primary._require(self.names(fd, 1) == names
                             and primary._identity(self.call(os.fstat, fd)) == primary._identity(metadata),
                             "queue_directory_changed")
            if relative:
                named = self.call(os.stat, name, dir_fd=parent, follow_symlinks=False)
                primary._require(stat.S_ISDIR(named.st_mode) and primary._identity(named) == primary._identity(metadata),
                                 "queue_directory_changed")
        for row in self.aux_rows:
            self.tick()
            if row.root_path != contract.root_path:
                continue
            fd = self.open(row.row_path.rsplit("/", 1)[1], primary._FILE_FLAGS,
                           directories[row.relative_directory][0])
            try:
                current = self.call(os.fstat, fd)
                primary._require(stat.S_ISREG(current.st_mode) and primary._identity(current) == row.row_identity,
                                 "queue_row_changed")
            finally:
                self.close(fd)
        primary._require([identity for _, identity in self.walk(contract.root_path)]
                         == [identity for _, identity in chain], "queue_root_changed")

    def result(self):
        try:
            self.tick()
            roots = []
            for contract in self.aux_contracts:
                self.tick()
                roots.append(self.aux_roots.get(contract.root_path, ObservedAuxiliaryRoot(
                    contract.family, contract.root_path, None, _attempted(contract.family), (), (), ())))
            rows = tuple(sorted(self.aux_rows, key=lambda row: (row.root_path, row.relative_directory, row.row_path)))
            self.tick()
            result = AuxiliaryQueueObservation(not self.blockers, self.observed, tuple(roots), rows,
                                               tuple(sorted(self.blockers)))
            if self.shared is not None:
                from .control_plane_reference_budget import ReferenceCollectionBudgetError
                try:
                    self.shared.measure(result)
                except ReferenceCollectionBudgetError as error:
                    raise primary._Blocked(error.code) from None
            self.output_size(asdict(result))
            self.tick()
            return result
        except primary._Blocked as error:
            self.block(error.code)
        except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
            self.block("queue_result_invalid")
        return AuxiliaryQueueObservation(False, self.observed, (), (), tuple(sorted(self.blockers)))


def observe_preparation_sam_auxiliaries(contracts: Sequence[AuxiliaryQueueContract], *, observed_at_epoch: float,
                                      monotonic: Callable[[], float] = time.monotonic,
                                      time_budget_seconds: float = 5.0,
                                      budget: ReferenceCollectionBudget | None = None) -> AuxiliaryQueueObservation:
    """Observe fixed auxiliary layouts; never validate, repair or clear references."""
    if budget is not None:
        from .control_plane_reference_budget import bind_budget
        bind_budget(budget, monotonic=monotonic, time_budget_seconds=time_budget_seconds,
                    error=AuxiliaryQueueObservationError, code="auxiliary_parameters_invalid")
    normalized = _contracts(contracts)
    if not (primary._finite(observed_at_epoch) and observed_at_epoch >= 0 and primary._finite(time_budget_seconds)
            and 0 < time_budget_seconds <= 5 and callable(monotonic)):
        raise AuxiliaryQueueObservationError("auxiliary_parameters_invalid")
    scan = _AuxScan(normalized, float(observed_at_epoch), monotonic, float(time_budget_seconds), budget)
    exhausted = False
    try:
        scan.shared_charge("roots", len(normalized))
        scan.shared_charge("groups", sum(len(_attempted(row.family)) for row in normalized))
        for contract in normalized:
            try:
                scan.observe_aux_root(contract)
            except FileNotFoundError:
                scan.block("queue_root_missing" if contract.root_path not in scan.opened_roots
                           else "queue_inventory_changed")
            except OSError:
                scan.block("queue_inventory_unavailable")
            except primary._Blocked as error:
                scan.block(error.code)
                if error.code.endswith("limit") or error.code in primary._RESOURCE_CODES:
                    exhausted = True
                    break
        if not exhausted:
            for snapshot in scan.aux_snapshots:
                try:
                    scan.verify_aux_root(snapshot)
                except OSError:
                    scan.block("queue_inventory_changed")
                except primary._Blocked as error:
                    scan.block(error.code)
                    if error.code.endswith("limit") or error.code in primary._RESOURCE_CODES:
                        break
    except primary._Blocked as error:
        scan.block(error.code)
    finally:
        for _pass in range(2):
            for fd in tuple(reversed(scan.fds)):
                scan.close(fd)
    return scan.result()
