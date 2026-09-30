"""The Quick-10 provider-output delivery mode and its member contract (plan 15, PR C).

Delivery mode. ``BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY`` unset or empty means
auto (the founder's default, 2026-09-30): ``stream`` when the dedicated B2
artifact store is configured (``artifact_store_configured``, the very check
an explicit ``stream`` must pass), else ``download``, today's path. An
explicit ``download`` or ``stream`` means exactly what it says, whatever the
store. Only the Quick-10 session
(``native_task_arena_vast.run_native_task_arena_policy_canary_session_vast``)
resolves it, before its session authority is consumed; any other value,
however close (``Stream``, ``stream ``, ``auto``), refuses there with
``policy_canary_output_delivery_mode_invalid`` and zero provider mutations, and
``stream`` refuses there without the store (review I4). The session records
the effective mode and why (``explicit``, ``auto_artifact_store_configured``
or ``auto_artifact_store_not_configured``) in every result it returns once
resolved, the lane result included, as ``provider_output_delivery_resolution``.
Every other arena caller keeps the lane's ``download`` default and records
nothing. Readers never read the environment: they learn the mode from the
records on disk (the lane result, a member view descriptor).

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
durable. The measurement is an estimate: before a host streams its first
run (with the default, its first Quick-10 once the B2 store is configured)
the owner measures one retained Quick-10 with
``python -m blueprint_pipeline.provider_output_member_view plan --archive <zip>
--contract policy_canary_output_member_contract.v1`` (docs/CONTROL_PLANE_STORAGE.md).
"""

from __future__ import annotations

import os
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
# and the dedicated B2 store was, or was not, configured.
EXPLICIT = "explicit"
AUTO_ARTIFACT_STORE_CONFIGURED = "auto_artifact_store_configured"
AUTO_ARTIFACT_STORE_NOT_CONFIGURED = "auto_artifact_store_not_configured"
DELIVERY_REASONS = (EXPLICIT, AUTO_ARTIFACT_STORE_CONFIGURED, AUTO_ARTIFACT_STORE_NOT_CONFIGURED)
# The result field holding ``OutputDelivery.record()``.
RESOLUTION_FIELD = "provider_output_delivery_resolution"
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
    """The effective delivery mode (one of ``DELIVERY_MODES``) and why (one of ``DELIVERY_REASONS``)."""

    mode: str
    reason: str

    def record(self) -> dict[str, str]:
        """What the session's results record as ``RESOLUTION_FIELD``."""
        return {"mode": self.mode, "reason": self.reason}


def resolve_output_delivery(environ: Mapping[str, str] | None = None) -> OutputDelivery:
    """The mode the setting names; unset or empty is auto; any other value is a typed refusal.

    Auto streams exactly when ``artifact_store_configured`` holds for the same
    environment. An explicit ``stream`` resolves without that check: the
    session, not the resolution, refuses it when the store is missing.
    """
    values = os.environ if environ is None else environ
    raw = values.get(DELIVERY_ENV)
    if raw is None or raw == "":
        if artifact_store_configured(values):
            return OutputDelivery(STREAM, AUTO_ARTIFACT_STORE_CONFIGURED)
        return OutputDelivery(DOWNLOAD, AUTO_ARTIFACT_STORE_NOT_CONFIGURED)
    if raw in DELIVERY_MODES:
        return OutputDelivery(raw, EXPLICIT)
    raise PolicyCanaryOutputDeliveryError(MODE_INVALID)


def artifact_store_configured(environ: Mapping[str, str] | None = None) -> bool:
    """Whether all five dedicated B2 artifact-store settings name readable regular files (review I4).

    A setting that is empty, or names a missing file, a directory or a file
    this process cannot read, would only fail when promotion first reads it,
    after the paid run (review minor 8).
    """
    from .task_evaluation_configured_scene_object_store import _ARTIFACT_STORE_FILE_ENV

    values = os.environ if environ is None else environ
    for name in _ARTIFACT_STORE_FILE_ENV.values():
        setting = str(values.get(name) or "").strip()
        if not setting or not Path(setting).is_file() or not os.access(setting, os.R_OK):
            return False
    return True


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


POLICY_CANARY_OUTPUT_CONTRACT = PolicyCanaryOutputContract()

__all__ = [
    "ARTIFACT_STORE_NOT_CONFIGURED",
    "AUTO_ARTIFACT_STORE_CONFIGURED",
    "AUTO_ARTIFACT_STORE_NOT_CONFIGURED",
    "CHILD_RESULT_NAME",
    "CONTRACT_VERSION",
    "DELIVERY_ENV",
    "DELIVERY_MODES",
    "DELIVERY_REASONS",
    "DOWNLOAD",
    "EXPLICIT",
    "FORECAST_INDEX_ROW_BYTES",
    "FORECAST_MEMBER_COUNT",
    "MODE_INVALID",
    "NOT_INGESTED_GAP",
    "NEEDED_SET_BUDGET_BYTES",
    "OutputDelivery",
    "POLICY_CANARY_OUTPUT_CONTRACT",
    "PolicyCanaryOutputContract",
    "PolicyCanaryOutputDeliveryError",
    "RESOLUTION_FIELD",
    "STREAM",
    "artifact_store_configured",
    "resolve_output_delivery",
]
