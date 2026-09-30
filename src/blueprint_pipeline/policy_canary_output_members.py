"""The Quick-10 provider-output delivery mode and its member contract (plan 15, PR C).

Delivery mode. ``BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY`` unset or empty means
auto (the founder's default, 2026-09-30); an explicit ``download`` or
``stream`` means exactly what it says. Auto streams only when both hold, and
otherwise downloads, today's path:

- promotion would accept the dedicated B2 artifact store
  (``artifact_store_configured``: promotion's own private-file reader and
  bucket identity check, ``verify_dedicated_artifact_store``), and
- the host holds a needed-set measurement that fits
  (``needed_set_measurement_refusal``): a sealed record at ``MEASUREMENT_PATH``
  naming the current contract and selection version, of a run the contract
  admits (``PolicyCanaryOutputContract.admits``).

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
import os
import stat
from collections.abc import Mapping
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
MEASUREMENT_MAXIMUM_BYTES = 64 * 1024
_MEASUREMENT_FIELDS = frozenset({
    "schema_version", "contract", "selection_version", "needed_set_budget_bytes", "materialized_members",
    "materialized_bytes", "archive", "measured_at", "record_digest"})
_MEASURED_ARCHIVE_FIELDS = frozenset({"name", "size_bytes", "sha256", "members"})
CONTRACT_VERSION = "policy_canary_output_member_contract.v1"
# The worker's per-cell child result name (native_task_arena_policy_canary_session.
# PROVIDER_RESULT_FILENAME, kept equal by a test so this module imports nothing heavy).
CHILD_RESULT_NAME = "native_task_arena_policy_canary_session_result.v1.json"
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

    def admits(self, *, needed_bytes: int, member_count: int) -> bool:
        """Whether a run of this shape only ever shrinks the forecast hold.

        Its needed set is within the budget, and its hold -- index rows at the
        forecast's allowance, since only the run's own index would say better
        -- is within ``forecast_hold_bytes``. A run it does not admit pays and
        then seals blocked with its archive durable.
        """
        return (needed_bytes <= self.needed_set_budget_bytes
                and self.hold_bytes(needed_bytes=needed_bytes, member_count=member_count,
                                    index_file_bytes=member_count * FORECAST_INDEX_ROW_BYTES)
                <= self.forecast_hold_bytes())


POLICY_CANARY_OUTPUT_CONTRACT = PolicyCanaryOutputContract()


def seal_needed_set_measurement(*, contract: str, materialized_members: int, materialized_bytes: int,
                                archive: Mapping[str, Any], measured_at: str) -> dict[str, Any]:
    """The sealed record of one retained Quick-10's measured needed set (``MEASUREMENT_SCHEMA``).

    ``contract`` is the rule it was measured under; the record also names the
    current contract's selection version and budget. ``archive`` is the
    measured archive's {name, size_bytes, sha256, members}. ``record_digest``
    seals every other field.
    """
    from .decision_evidence_contracts import canonical_digest

    current = POLICY_CANARY_OUTPUT_CONTRACT
    record = {"schema_version": MEASUREMENT_SCHEMA, "contract": contract, "selection_version": current.version,
              "needed_set_budget_bytes": current.needed_set_budget_bytes,
              "materialized_members": materialized_members, "materialized_bytes": materialized_bytes,
              "archive": dict(archive), "measured_at": measured_at, "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    return record


def write_needed_set_measurement(record: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Replace the record at ``path`` (default ``MEASUREMENT_PATH``) whole; returns the path.

    The file is ``0644`` and a directory this call creates ``0755``, so the
    dispatcher (``blueprint``) reads the record whoever wrote it; it holds no
    secret. Raises ``OSError``.
    """
    from .common import write_json

    target = Path(MEASUREMENT_PATH if path is None else path)
    created = not target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    if created:
        os.chmod(target.parent, 0o755)
    write_json(target, dict(record))
    os.chmod(target, 0o644)
    return target


def needed_set_measurement_refusal(path: str | Path | None = None) -> str | None:
    """None when the host's recorded needed set lets auto stream, else why auto downloads.

    The record at ``path`` (default ``MEASUREMENT_PATH``) must be a regular,
    non-symlink file of at most ``MEASUREMENT_MAXIMUM_BYTES`` holding exactly
    the ``MEASUREMENT_SCHEMA`` fields, sealed by a ``record_digest`` that
    verifies, and naming the current contract and selection version; else
    ``auto_needed_set_record_invalid`` (no file at all is
    ``auto_needed_set_unmeasured``). Its run must be one the current contract
    admits, else ``auto_needed_set_over_budget``. Never raises.
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
    admitted = POLICY_CANARY_OUTPUT_CONTRACT.admits(needed_bytes=record["materialized_bytes"],
                                                    member_count=record["archive"]["members"])
    return None if admitted else AUTO_NEEDED_SET_OVER_BUDGET


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _sealed_measurement(record: Any) -> bool:
    """Whether ``record`` is exactly a sealed measurement under the current contract."""
    from .decision_evidence_contracts import canonical_digest

    if not isinstance(record, dict) or set(record) != _MEASUREMENT_FIELDS:
        return False
    archive = record["archive"]
    return (record["schema_version"] == MEASUREMENT_SCHEMA
            and record["contract"] == CONTRACT_VERSION
            and record["selection_version"] == POLICY_CANARY_OUTPUT_CONTRACT.version
            and all(_count(record[key]) for key in ("needed_set_budget_bytes", "materialized_members",
                                                    "materialized_bytes"))
            and isinstance(archive, dict) and set(archive) == _MEASURED_ARCHIVE_FIELDS
            and _count(archive["size_bytes"]) and _count(archive["members"])
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
    "OutputDelivery",
    "POLICY_CANARY_OUTPUT_CONTRACT",
    "PolicyCanaryOutputContract",
    "PolicyCanaryOutputDeliveryError",
    "REASON_MODES",
    "RESOLUTION_FIELD",
    "STREAM",
    "artifact_store_configured",
    "needed_set_measurement_refusal",
    "resolve_output_delivery",
    "seal_needed_set_measurement",
    "write_needed_set_measurement",
]
