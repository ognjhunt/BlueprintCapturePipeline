"""The Quick-10 provider-output delivery mode and its member contract (plan 15, PR C).

Delivery mode. ``BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY`` unset or empty means
auto (the founder's default, 2026-09-30); an explicit ``download`` or
``stream`` means exactly what it says. Auto streams only when both hold, and
otherwise downloads, today's path:

- promotion would accept the dedicated B2 artifact store
  (``artifact_store_configured``: promotion's own private-file reader, client
  construction and bucket identity check, ``verify_dedicated_artifact_store``), and
- the host holds a needed-set measurement that fits
  (``needed_set_measurement_refusal``): a sealed record at ``MEASUREMENT_PATH``
  naming the current contract and selection version, measured on an archive
  of the full ``QUICK10_SHAPE`` with a needed set, of a run the contract
  admits within the hold the session takes (``PolicyCanaryOutputContract.admits``,
  ``forecast_hold_cap_bytes``).

Deploying is therefore never the flip. The switch is the one host command
that measures a retained Quick-10 and seals the record:
``python -m blueprint_pipeline.provider_output_member_view plan --archive <zip>
--contract policy_canary_output_member_contract.v1 --record``. Neither check
raises: whatever cannot be read is a reason to download.

Only the Quick-10 session
(``native_task_arena_vast.run_native_task_arena_policy_canary_session_vast``)
resolves the setting, before its session authority is consumed; any other
value, however close (``Stream``, ``stream ``, ``auto``), refuses there with
``policy_canary_output_delivery_mode_invalid`` and zero provider mutations. An
explicit ``stream`` is the owner's override: it needs no measurement, and
refuses there without the store (review I4). The session records the effective
mode and why (``REASON_MODES``) in every result it returns once resolved, the
lane result included, as ``provider_output_delivery_resolution``. Every other
arena caller keeps the lane's ``download`` default and records nothing.
Readers never read the environment: they learn the mode from the records on
disk (the lane result, a member view descriptor).

Contract ``policy_canary_output_member_contract.v1``. A streamed attempt
materializes under ``immutable_execution/`` every file member whose name ends
in ``.json`` except

- anything under a ``policy-requests`` directory (bulk, as the storage GC
  treats it), and
- the ten per-cell child results
  ``cell_runs/NN/native_task_arena_policy_canary_session_result.v1.json``
  (review I7). The top-level aggregate embeds the twenty episode receipts the
  children repeat; only partial recovery and SSH adoption read the children,
  for runs that did not complete, and they read them through the member view.

Everything else stays in the promoted archive in B2. The version is bound into
the member selection (``build_member_selection(..., selection_version=...)``).

Budget. ``NEEDED_SET_BUDGET_BYTES`` bounds what the contract may materialize.
It is sized from a measurement on a Quick-10-shaped fixture
(``tests/provider_output_fixtures.quick10_production_shape``): the lifecycle
rehearsal's real worker output, whose contract set is 2.29 times its
4,407,038-byte aggregate, scaled to production's measured 190,573,875-byte
aggregate, needs 436,485,098 bytes (416 MiB); the children would add
191,208,330. 640 MiB is 1.5 times that measurement, and with the ingestion
metadata still fits the ``policy_canary_output`` role's 1 GiB declared
footprint. A run whose needed set exceeds it seals blocked with its archive
durable. The measurement is an estimate, which is why auto streams only
once the host has measured one retained Quick-10 and recorded that it fits
(docs/CONTROL_PLANE_STORAGE.md).
"""

from __future__ import annotations

import json
import errno
import fcntl
import grp
import os
import pwd
import stat
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DELIVERY_ENV = "BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY"
DOWNLOAD = "download"
STREAM = "stream"
DELIVERY_MODES = (DOWNLOAD, STREAM)
MODE_INVALID = "policy_canary_output_delivery_mode_invalid"
ARTIFACT_STORE_NOT_CONFIGURED = "policy_canary_output_stream_artifact_store_not_configured"
# Why the mode was chosen: the setting named it, or it was unset or empty (auto)
# and auto streamed, or downloaded for the first condition that failed.
EXPLICIT = "explicit"
AUTO_NEEDED_SET_WITHIN_BUDGET = "auto_needed_set_within_budget"
AUTO_ARTIFACT_STORE_NOT_CONFIGURED = "auto_artifact_store_not_configured"
AUTO_ARTIFACT_STORE_INVALID = "auto_artifact_store_invalid"
AUTO_NEEDED_SET_UNMEASURED = "auto_needed_set_unmeasured"
AUTO_NEEDED_SET_RECORD_INVALID = "auto_needed_set_record_invalid"
AUTO_NEEDED_SET_OVER_BUDGET = "auto_needed_set_over_budget"
# Each reason, and the modes it can explain.
REASON_MODES: dict[str, tuple[str, ...]] = {
    EXPLICIT: DELIVERY_MODES,
    AUTO_NEEDED_SET_WITHIN_BUDGET: (STREAM,),
    AUTO_ARTIFACT_STORE_NOT_CONFIGURED: (DOWNLOAD,),
    AUTO_ARTIFACT_STORE_INVALID: (DOWNLOAD,),
    AUTO_NEEDED_SET_UNMEASURED: (DOWNLOAD,),
    AUTO_NEEDED_SET_RECORD_INVALID: (DOWNLOAD,),
    AUTO_NEEDED_SET_OVER_BUDGET: (DOWNLOAD,),
}
# The result field holding ``OutputDelivery.record()``.
RESOLUTION_FIELD = "provider_output_delivery_resolution"
# The host's needed-set measurement (review: deploying must not be the flip):
# one sealed record at a fixed path under the control plane's state, beside
# its other policy-canary roots, which the dispatcher (``blueprint``) reads.
MEASUREMENT_SCHEMA = "policy_canary_output_needed_set_measurement.v1"
MEASUREMENT_PATH = Path(
    "/var/lib/blueprint/pipeline-control-plane/policy-canary-output/needed-set-measurement.v1.json")
# Redirecting the default for a fixture or a custom caller does not enroll a
# new shared destination. Only this installer-owned state path has that role.
_CANONICAL_MEASUREMENT_PATH = MEASUREMENT_PATH
MEASUREMENT_MAXIMUM_BYTES = 64 * 1024
_MEASUREMENT_FIELDS = frozenset({
    "schema_version", "contract", "selection_version", "needed_set_budget_bytes", "materialized_members",
    "materialized_bytes", "archive", "quick10_shape", "measured_at", "record_digest"})
_MEASURED_ARCHIVE_FIELDS = frozenset({"name", "size_bytes", "sha256", "members"})
CONTRACT_VERSION = "policy_canary_output_member_contract.v1"
# The worker's per-cell child result name (native_task_arena_policy_canary_session.
# PROVIDER_RESULT_FILENAME, kept equal by a test so this module imports nothing heavy).
# The session's aggregate result has the same name, at the archive's root.
CHILD_RESULT_NAME = "native_task_arena_policy_canary_session_result.v1.json"
# The Quick-10 layout a measurement must come from (review: a record of the wrong
# archive would stream every Quick-10 after it): the aggregate result, one child
# result ``cell_runs/NN/<CHILD_RESULT_NAME>`` per cell, and the first cell's
# static startup preflight, which also binds the streamed native inventory
# (``arena_provider_output_streaming.IDENTITY_DOCUMENT``). One cell runs per
# episode of each policy (the session's ``EPISODES_PER_POLICY``, kept equal by a test).
QUICK10_CELL_COUNT = 10
STARTUP_PREFLIGHT_MEMBER = "cell_runs/00/policy_canary_static_startup_preflight.v1.json"
QUICK10_SHAPE = {"aggregate": True, "cell_results": QUICK10_CELL_COUNT, "startup_preflight": True}
# The disk role whose hold a streamed session takes before its run.
OUTPUT_ROLE = "policy_canary_output"
# A streamed output that arrived but was never ingested: the lane's and the dispatcher's media gap.
NOT_INGESTED_GAP = "provider_output_not_ingested"
NEEDED_SET_BUDGET_BYTES = 640 * 1024**2
# Ingestion metadata the hold covers beside the members: a journal row and a
# receipt row per archive member, plus a fixed allowance for the receipt, the
# view descriptor and filesystem overhead.
PER_MEMBER_METADATA_BYTES = 1024
FIXED_METADATA_BYTES = 16 * 1024**2
# The forecast taken before the run (review minor 5), when neither the index nor
# its member count exists: the needed-set budget, plus the metadata of up to
# FORECAST_MEMBER_COUNT members (about three times the production shape's 6,745)
# and an index row each (measured at 484 bytes on that shape).
FORECAST_MEMBER_COUNT = 20_000
FORECAST_INDEX_ROW_BYTES = 1024


class PolicyCanaryOutputDeliveryError(ValueError):
    """A typed refusal; the message is the stable code."""


@dataclass(frozen=True)
class OutputDelivery:
    """The effective delivery mode (one of ``DELIVERY_MODES``) and why (a ``REASON_MODES`` key)."""

    mode: str
    reason: str

    def record(self) -> dict[str, str]:
        """What the session's results record as ``RESOLUTION_FIELD``."""
        return {"mode": self.mode, "reason": self.reason}


def resolve_output_delivery(environ: Mapping[str, str] | None = None, *,
                            measurement_path: str | Path | None = None) -> OutputDelivery:
    """The mode the setting names; unset or empty is auto; any other value is a typed refusal.

    Auto checks the store first, then the needed-set record at
    ``measurement_path`` (default ``MEASUREMENT_PATH``), and downloads for the
    first that fails; neither check raises. An explicit ``stream`` resolves
    without either: the session, not the resolution, refuses it without the store.
    """
    values = os.environ if environ is None else environ
    raw = values.get(DELIVERY_ENV)
    if raw is None or raw == "":
        refusal = _artifact_store_refusal(values) or needed_set_measurement_refusal(measurement_path)
        return OutputDelivery(DOWNLOAD, refusal) if refusal else OutputDelivery(STREAM, AUTO_NEEDED_SET_WITHIN_BUDGET)
    if raw in DELIVERY_MODES:
        return OutputDelivery(raw, EXPLICIT)
    raise PolicyCanaryOutputDeliveryError(MODE_INVALID)


def artifact_store_configured(environ: Mapping[str, str] | None = None) -> bool:
    """Whether promotion would accept the dedicated B2 artifact store (review I4); never raises.

    Configured means exactly what promotion's own client accepts
    (``verify_dedicated_artifact_store``): all five settings name private,
    non-empty UTF-8 files its reader takes, and the bucket is the expected one.
    Anything else would only fail when promotion first reads it, after the
    paid run (review minor 8), so auto, an explicit ``stream`` and the
    session's re-check all use this one check.
    """
    return _artifact_store_refusal(os.environ if environ is None else environ) is None


def _artifact_store_refusal(values: Mapping[str, str]) -> str | None:
    """None when promotion would accept the store, else why auto downloads.

    Never raises: a setting that cannot be read -- below Python 3.13 even one
    under a parent this process cannot traverse -- is a refusal, not an error.
    """
    from .task_evaluation_configured_scene_object_store import (
        _ARTIFACT_STORE_FILE_ENV,
        verify_dedicated_artifact_store,
    )

    if not any(str(values.get(name) or "").strip() for name in _ARTIFACT_STORE_FILE_ENV.values()):
        return AUTO_ARTIFACT_STORE_NOT_CONFIGURED
    try:
        verify_dedicated_artifact_store(values)
    except (OSError, ValueError, RuntimeError):  # the store's typed refusal is a RuntimeError
        return AUTO_ARTIFACT_STORE_INVALID
    return None


def _needed(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or PurePosixPath(path).suffix.lower() != ".json" or "policy-requests" in parts:
        return False
    return not (len(parts) == 3 and parts[0] == "cell_runs" and parts[2] == CHILD_RESULT_NAME)


@dataclass(frozen=True)
class PolicyCanaryOutputContract:
    """Which archive members a streamed Quick-10 keeps on the host, and what they may cost."""

    version: str = CONTRACT_VERSION
    needed_set_budget_bytes: int = NEEDED_SET_BUDGET_BYTES

    @staticmethod
    def needed(path: str) -> bool:
        """Whether the file member at archive path ``path`` is materialized."""
        return _needed(path)

    def paths(self, index: Mapping[str, Any]) -> list[str]:
        return sorted(row["path"] for row in index["members"] if row["kind"] == "file" and self.needed(row["path"]))

    def selection(self, index: Mapping[str, Any]) -> dict:
        """The member selection for ``index``, bound to its digest and to this contract's version."""
        from .provider_output_member_index import build_member_selection

        return build_member_selection(index, self.paths(index), selection_version=self.version)

    def needed_bytes(self, index: Mapping[str, Any]) -> int:
        return sum(row["size"] for row in index["members"] if row["kind"] == "file" and self.needed(row["path"]))

    @staticmethod
    def hold_bytes(*, needed_bytes: int, member_count: int, index_file_bytes: int) -> int:
        """What the ``policy_canary_output`` hold must cover once the index is known."""
        return needed_bytes + PER_MEMBER_METADATA_BYTES * member_count + index_file_bytes + FIXED_METADATA_BYTES

    def forecast_hold_bytes(self) -> int:
        """The hold before the run: the budget's, for up to ``FORECAST_MEMBER_COUNT`` members.

        Every run the contract admits (a needed set within budget, members and
        index rows within the allowance) then only shrinks it in place; a larger
        archive blocks after the run with its archive durable, for the door.
        """
        return self.hold_bytes(needed_bytes=self.needed_set_budget_bytes, member_count=FORECAST_MEMBER_COUNT,
                               index_file_bytes=FORECAST_MEMBER_COUNT * FORECAST_INDEX_ROW_BYTES)

    def admits(self, *, needed_bytes: int, member_count: int, hold_cap_bytes: int | None = None) -> bool:
        """Whether a run of this shape only ever shrinks the hold taken before it.

        Its needed set is within the budget, and its hold -- index rows at the
        forecast's allowance, since only the run's own index would say better
        -- is within ``hold_cap_bytes``: the hold the session takes
        (``forecast_hold_cap_bytes``), by default the uncapped forecast. A run
        it does not admit pays and then seals blocked with its archive durable.
        """
        cap = self.forecast_hold_bytes() if hold_cap_bytes is None else hold_cap_bytes
        return (needed_bytes <= self.needed_set_budget_bytes
                and self.hold_bytes(needed_bytes=needed_bytes, member_count=member_count,
                                    index_file_bytes=member_count * FORECAST_INDEX_ROW_BYTES) <= cap)


POLICY_CANARY_OUTPUT_CONTRACT = PolicyCanaryOutputContract()


def forecast_hold_cap_bytes(contract: PolicyCanaryOutputContract | None = None) -> int:
    """The ``OUTPUT_ROLE`` hold a streamed session takes before its run.

    The contract's forecast, capped at the role's declared footprint or the
    operator's ``BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_POLICY_CANARY_OUTPUT_BYTES``.
    Raises ``ControlPlaneDiskBudgetError`` when that override is not a byte count.
    """
    from .control_plane_disk_ledger import footprint_bytes

    contract = POLICY_CANARY_OUTPUT_CONTRACT if contract is None else contract
    return min(contract.forecast_hold_bytes(), footprint_bytes(OUTPUT_ROLE))


def quick10_shape(member_paths: Iterable[str]) -> dict[str, Any]:
    """Which of the Quick-10 layout's identifying members ``member_paths`` holds (``QUICK10_SHAPE`` when all)."""
    paths = set(member_paths)
    return {"aggregate": CHILD_RESULT_NAME in paths,
            "cell_results": sum(f"cell_runs/{cell:02d}/{CHILD_RESULT_NAME}" in paths
                                for cell in range(QUICK10_CELL_COUNT)),
            "startup_preflight": STARTUP_PREFLIGHT_MEMBER in paths}


def seal_needed_set_measurement(*, contract: str, materialized_members: int, materialized_bytes: int,
                                archive: Mapping[str, Any], quick10_shape: Mapping[str, Any],
                                measured_at: str) -> dict[str, Any]:
    """The sealed record of one retained Quick-10's measured needed set (``MEASUREMENT_SCHEMA``).

    ``contract`` is the rule it was measured under; the record also names the
    current contract's selection version and budget. ``archive`` is the
    measured archive's {name, size_bytes, sha256, members}, and
    ``quick10_shape`` the identifying members it held (``quick10_shape()``).
    ``record_digest`` seals every other field.
    """
    from .decision_evidence_contracts import canonical_digest

    current = POLICY_CANARY_OUTPUT_CONTRACT
    record = {"schema_version": MEASUREMENT_SCHEMA, "contract": contract, "selection_version": current.version,
              "needed_set_budget_bytes": current.needed_set_budget_bytes,
              "materialized_members": materialized_members, "materialized_bytes": materialized_bytes,
              "archive": dict(archive), "quick10_shape": dict(quick10_shape), "measured_at": measured_at,
              "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    return record


def _unsafe_measurement_path() -> OSError:
    return PermissionError(errno.EPERM, "policy_canary_output_measurement_path_unsafe")


def _measurement_identity(info: os.stat_result) -> tuple:
    return info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode


def _measurement_file_identity(info: os.stat_result | None) -> tuple | None:
    if info is None:
        return None
    return (*_measurement_identity(info), info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _measurement_parent(target: Path, *, canonical: bool):
    """Hold no-follow ancestry; only an installed canonical state admits shared reads."""
    state = _CANONICAL_MEASUREMENT_PATH.parent.parent
    owners = {0, os.geteuid()}
    reader = None
    if canonical:
        try:
            reader = pwd.getpwnam("blueprint").pw_uid, grp.getgrnam("blueprint").gr_gid
        except KeyError:
            raise _unsafe_measurement_path() from None
        owners = {0, reader[0]}
        if os.geteuid() not in owners:
            raise _unsafe_measurement_path()
    held: list[tuple[Path, int | None, str, int]] = []

    def verify():
        for directory, parent, name, descriptor in held:
            info = os.fstat(descriptor)
            # Root-owned sticky temporary roots permit private fixture children;
            # every other ancestor must deny unadmitted writers.
            sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in owners
                    or (info.st_mode & 0o022 and not sticky_root)):
                raise _unsafe_measurement_path()
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if _measurement_identity(info) != _measurement_identity(named):
                raise _unsafe_measurement_path()
            if canonical and directory == state and (
                    (info.st_uid, info.st_gid) != reader or stat.S_IMODE(info.st_mode) != 0o750):
                raise _unsafe_measurement_path()
            if canonical and directory.is_relative_to(state) and directory != state:
                readable = (info.st_uid == reader[0] and bool(info.st_mode & 0o100)
                            or info.st_gid == reader[1] and bool(info.st_mode & 0o010)
                            or bool(info.st_mode & 0o001))
                if not readable:
                    raise _unsafe_measurement_path()

    try:
        descriptor = os.open(target.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        held.append((Path(target.anchor), None, target.anchor, descriptor))
        verify()
        directory = Path(target.anchor)
        for component in target.parent.parts[1:]:
            directory /= component
            parent = descriptor
            try:
                descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=parent)
            except FileNotFoundError:
                if canonical and directory in (state, *state.parents):
                    raise _unsafe_measurement_path() from None
                verify()
                os.mkdir(component, 0o755 if canonical else 0o700, dir_fd=parent)
                descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=parent)
                try:
                    if os.fstat(descriptor).st_uid != os.geteuid():
                        raise _unsafe_measurement_path()
                    os.fchmod(descriptor, 0o755 if canonical else 0o700)
                except BaseException:
                    os.close(descriptor)
                    raise
            held.append((directory, parent, component, descriptor))
            verify()
        yield descriptor, owners if canonical else {os.geteuid()}, verify
    finally:
        for _, _, _, descriptor in reversed(held):
            os.close(descriptor)


def _measurement_file(parent: int, name: str, *, mode: int, owners: set[int], descriptor: int | None = None):
    try:
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        if descriptor is None:
            return None
        raise _unsafe_measurement_path() from None
    if (not stat.S_ISREG(named.st_mode) or named.st_uid not in owners
            or named.st_nlink != 1 or stat.S_IMODE(named.st_mode) != mode):
        raise _unsafe_measurement_path()
    if descriptor is not None and _measurement_file_identity(named) != _measurement_file_identity(os.fstat(descriptor)):
        raise _unsafe_measurement_path()
    return named


@contextmanager
def _measurement_write_lock(parent: int, name: str):
    # All writers use this stable inode; failures never unlink another holder's lock.
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    created = False
    try:
        descriptor = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
    except FileExistsError:
        descriptor = os.open(name, flags, dir_fd=parent)
    else:
        created = True
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        _measurement_file(parent, name, mode=0o600, owners={os.geteuid()}, descriptor=descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _measurement_file(parent, name, mode=0o600, owners={os.geteuid()}, descriptor=descriptor)
        yield
    finally:
        os.close(descriptor)


def write_needed_set_measurement(record: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Atomically replace a measurement through held, writer-protected ancestry.

    Arbitrary mappings and custom destinations are private: new directories
    ``0700``, files ``0600``. Their contents are not assumed secret-free.
    Only the exact canonical destination, below verified ``blueprint:blueprint
    0750`` STATE, retains ``0644``/``0755`` for the existing dispatcher reader.
    This scopes readership to that installed role; it does not assert group
    membership or authorize public redistribution. Unsafe incumbents refuse.
    """
    result = Path(MEASUREMENT_PATH if path is None else path)
    target = result if result.is_absolute() else Path.cwd() / result
    if ".." in target.parts or not target.name:
        raise _unsafe_measurement_path()
    canonical = target == _CANONICAL_MEASUREMENT_PATH
    payload = dict(record)
    content = json.dumps(payload, indent=2).encode("utf-8")
    if canonical and (len(content) > MEASUREMENT_MAXIMUM_BYTES or not _sealed_measurement(payload)):
        raise _unsafe_measurement_path()
    mode = 0o644 if canonical else 0o600
    with _measurement_parent(target, canonical=canonical) as (parent, owners, verify):
        with _measurement_write_lock(parent, f".{target.name}.write.lock"):
            verify()
            previous = _measurement_file_identity(_measurement_file(parent, target.name, mode=mode, owners=owners))
            temporary = f".{target.name}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                 0o600, dir_fd=parent)
            try:
                os.fchmod(descriptor, 0o600)
                _measurement_file(parent, temporary, mode=0o600, owners={os.geteuid()}, descriptor=descriptor)
                with os.fdopen(descriptor, "wb", closefd=False) as output:
                    output.write(content)
                    output.flush()
                    os.fsync(descriptor)
                verify()
                _measurement_file(parent, temporary, mode=0o600, owners={os.geteuid()}, descriptor=descriptor)
                os.fchmod(descriptor, mode)
                _measurement_file(parent, temporary, mode=mode, owners={os.geteuid()}, descriptor=descriptor)
                verify()
                if previous != _measurement_file_identity(_measurement_file(parent, target.name, mode=mode, owners=owners)):
                    raise _unsafe_measurement_path()
                if previous is None:
                    # Exclusive publication preserves even a late competing incumbent.
                    os.link(temporary, target.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                    os.unlink(temporary, dir_fd=parent)
                else:
                    # Other permitted writers share the stable lock; unadmitted
                    # identities cannot replace entries in the checked parent.
                    os.replace(temporary, target.name, src_dir_fd=parent, dst_dir_fd=parent)
                _measurement_file(parent, target.name, mode=mode, owners=owners, descriptor=descriptor)
                verify()
                os.fsync(parent)
            finally:
                os.close(descriptor)
                # A failed publication retains its unique private temporary;
                # no stat-to-unlink cleanup can erase a competing replacement.
    return result


def needed_set_measurement_refusal(path: str | Path | None = None) -> str | None:
    """None when the host's recorded needed set lets auto stream, else why auto downloads.

    The record at ``path`` (default ``MEASUREMENT_PATH``) must be a regular,
    non-symlink file of at most ``MEASUREMENT_MAXIMUM_BYTES`` holding exactly
    the ``MEASUREMENT_SCHEMA`` fields, sealed by a ``record_digest`` that
    verifies, naming the current contract and selection version, and measured
    on an archive of the full ``QUICK10_SHAPE`` with a needed set that is not
    empty; else ``auto_needed_set_record_invalid`` (no file at all is
    ``auto_needed_set_unmeasured``). Its run must be one the current contract
    admits within the hold the session can take (``forecast_hold_cap_bytes``),
    else ``auto_needed_set_over_budget``. Never raises.
    """
    target = Path(MEASUREMENT_PATH if path is None else path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
    except FileNotFoundError:
        return AUTO_NEEDED_SET_UNMEASURED
    except (OSError, ValueError):
        return AUTO_NEEDED_SET_RECORD_INVALID
    data = b""
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MEASUREMENT_MAXIMUM_BYTES:
            return AUTO_NEEDED_SET_RECORD_INVALID
        while chunk := os.read(descriptor, MEASUREMENT_MAXIMUM_BYTES + 1 - len(data)):
            data += chunk
            if len(data) > MEASUREMENT_MAXIMUM_BYTES:
                return AUTO_NEEDED_SET_RECORD_INVALID
    except OSError:
        return AUTO_NEEDED_SET_RECORD_INVALID
    finally:
        os.close(descriptor)
    try:
        record = json.loads(data.decode("utf-8"))
        sealed = _sealed_measurement(record)
    except (ValueError, TypeError, RecursionError):  # UnicodeError and JSONDecodeError are ValueErrors
        return AUTO_NEEDED_SET_RECORD_INVALID
    if not sealed:
        return AUTO_NEEDED_SET_RECORD_INVALID
    contract = POLICY_CANARY_OUTPUT_CONTRACT
    try:
        cap = forecast_hold_cap_bytes(contract)
    except RuntimeError:  # an override that is not a byte count: the session could take no hold either
        return AUTO_NEEDED_SET_OVER_BUDGET
    admitted = contract.admits(needed_bytes=record["materialized_bytes"], member_count=record["archive"]["members"],
                               hold_cap_bytes=cap)
    return None if admitted else AUTO_NEEDED_SET_OVER_BUDGET


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _sealed_measurement(record: Any) -> bool:
    """Whether ``record`` is exactly a sealed measurement under the current contract."""
    from .decision_evidence_contracts import canonical_digest

    if not isinstance(record, dict) or set(record) != _MEASUREMENT_FIELDS:
        return False
    archive, shape = record["archive"], record["quick10_shape"]
    return (record["schema_version"] == MEASUREMENT_SCHEMA
            and record["contract"] == CONTRACT_VERSION
            and record["selection_version"] == POLICY_CANARY_OUTPUT_CONTRACT.version
            and all(_count(record[key]) for key in ("needed_set_budget_bytes", "materialized_members",
                                                    "materialized_bytes"))
            and record["materialized_bytes"] > 0
            and isinstance(archive, dict) and set(archive) == _MEASURED_ARCHIVE_FIELDS
            and _count(archive["size_bytes"]) and _count(archive["members"])
            # The measured archive was a Quick-10 (review): every identifying member, typed exactly.
            and isinstance(shape, dict) and shape == QUICK10_SHAPE
            and all(type(shape[key]) is type(value) for key, value in QUICK10_SHAPE.items())
            and record["record_digest"] == canonical_digest(record, digest_field="record_digest"))


__all__ = [
    "ARTIFACT_STORE_NOT_CONFIGURED",
    "AUTO_ARTIFACT_STORE_INVALID",
    "AUTO_ARTIFACT_STORE_NOT_CONFIGURED",
    "AUTO_NEEDED_SET_OVER_BUDGET",
    "AUTO_NEEDED_SET_RECORD_INVALID",
    "AUTO_NEEDED_SET_UNMEASURED",
    "AUTO_NEEDED_SET_WITHIN_BUDGET",
    "CHILD_RESULT_NAME",
    "CONTRACT_VERSION",
    "DELIVERY_ENV",
    "DELIVERY_MODES",
    "DOWNLOAD",
    "EXPLICIT",
    "FORECAST_INDEX_ROW_BYTES",
    "FORECAST_MEMBER_COUNT",
    "MEASUREMENT_MAXIMUM_BYTES",
    "MEASUREMENT_PATH",
    "MEASUREMENT_SCHEMA",
    "MODE_INVALID",
    "NOT_INGESTED_GAP",
    "NEEDED_SET_BUDGET_BYTES",
    "OUTPUT_ROLE",
    "OutputDelivery",
    "POLICY_CANARY_OUTPUT_CONTRACT",
    "PolicyCanaryOutputContract",
    "PolicyCanaryOutputDeliveryError",
    "QUICK10_CELL_COUNT",
    "QUICK10_SHAPE",
    "REASON_MODES",
    "RESOLUTION_FIELD",
    "STARTUP_PREFLIGHT_MEMBER",
    "STREAM",
    "artifact_store_configured",
    "forecast_hold_cap_bytes",
    "needed_set_measurement_refusal",
    "quick10_shape",
    "resolve_output_delivery",
    "seal_needed_set_measurement",
    "write_needed_set_measurement",
]
