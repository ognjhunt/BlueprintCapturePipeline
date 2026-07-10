"""Atomic canary-to-worker startup supervisor (startup reliability P0-1).

Before this module, the parity job could run a startup canary and could
warm-restart a stopped pod, but no single transaction validated a worker and
then either handed the same allocation to the full job or deleted it: a
passing canary left a stopped pod plus an intentionally open teardown record
for the operator to resolve by hand.

This supervisor is provider-neutral and drives one attempt at a time through:

1. inventory and spend admission before each allocation;
2. at most one billable GPU resource at any moment;
3. allocate, record provider machine identity, wait for the attempt-bound
   early marker;
4. the strict driver/Isaac/RTX canary;
5. on failure: upload artifacts, terminate, verify provider ``not_found``,
   close the pending teardown, update the cumulative spend ledger, quarantine
   the machine when the failure is machine-attributable, and retry only while
   caps permit;
6. on pass with ``promote_to_warm_job``: transfer ownership of the same
   allocation and pending-teardown record to the full job — never a second
   cold pull;
7. on pass with ``terminate_after_canary``: delete the allocation and verify
   zero residual inventory;
8. a process exception or SIGTERM executes the same finalizer.

The provider surface and the marker/canary probes are injected callables so
every terminal path is hermetically testable with fake providers.
"""

from __future__ import annotations

import signal
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import ensure_dir, read_json, utc_now_iso, write_json
from .machine_quarantine_registry import (
    KNOWN_FAILURE_CLASSES,
    PHASE_PRE_RUNTIME,
    PHASE_RUNTIME_CANARY,
    QuarantineRefused,
    find_active_quarantine,
    record_machine_quarantine,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    open_pending_teardown,
    provider_state_from_inspect,
)
from .provider_phase_trace import (
    PHASE_ALLOCATION_CREATED,
    PHASE_ALLOCATION_REQUESTED,
    PHASE_DELETE,
    PHASE_EARLY_MARKER,
    PHASE_FINAL_INVENTORY,
    PHASE_IMAGE_PULL_NO_RUNTIME,
    PHASE_ISAAC_START,
    PHASE_MACHINE_IDENTITY_OBSERVED,
    PHASE_PRE_SPEND_INVENTORY,
    PHASE_PROMOTE,
    PHASE_TEARDOWN_VERIFICATION,
    TRACE_FILENAME,
    PhaseTraceRecorder,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)
from .startup_spend_reconciliation import (
    CumulativeSpendLedger,
    SpendCapExceeded,
    build_spend_reconciliation,
)

SCHEMA_VERSION = "isaac_startup_supervisor.v1"
SUPERVISOR_LANE = "isaac_startup_supervisor"
MANIFEST_FILENAME = "startup_supervisor_manifest.json"
ATTEMPTS_FILENAME = "startup_attempts.json"
QUARANTINE_REFERENCES_FILENAME = "provider_machine_quarantine.json"
OWNERSHIP_RECEIPT_FILENAME = "ownership_transfer_receipt.json"
FINAL_INVENTORY_FILENAME = "final_zero_spend_inventory.json"

TERMINAL_MODE_PROMOTE = "promote_to_warm_job"
TERMINAL_MODE_TERMINATE = "terminate_after_canary"
VALID_TERMINAL_MODES = frozenset({TERMINAL_MODE_PROMOTE, TERMINAL_MODE_TERMINATE})

FAILURE_CLASS_NO_CAPACITY = "no_capacity"
FAILURE_CLASS_NO_RUNTIME = "no_runtime"
FAILURE_CLASS_STALE_MARKER = "stale_marker"
FAILURE_CLASS_QUARANTINED_MACHINE = "quarantined_machine_reallocated"


class SupervisorInterrupted(RuntimeError):
    """Raised by the SIGTERM handler so the finalizer runs on the normal path."""


@dataclass(frozen=True)
class StartupSupervisorRequest:
    """Immutable input contract for one supervised startup transaction."""

    run_id: str
    launch_nonce: str
    provider: str
    gpu_types: tuple[str, ...]
    image_ref: str
    image_manifest_checksum: str
    max_attempts: int
    wall_clock_cap_seconds: float
    total_spend_cap_usd: float
    hourly_rate_usd: float
    per_attempt_reserved_seconds: float
    terminal_mode: str
    isaac_version: str = "6.0.0"
    stopped_disk_usd_per_hour: float | None = None
    full_job_request: Mapping[str, Any] | None = None

    def validation_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not str(self.run_id or "").strip():
            blockers.append("startup_supervisor_run_id_missing")
        if not str(self.launch_nonce or "").strip():
            blockers.append("startup_supervisor_launch_nonce_missing")
        if not str(self.provider or "").strip():
            blockers.append("startup_supervisor_provider_missing")
        if not self.gpu_types:
            blockers.append("startup_supervisor_gpu_types_missing")
        if "@sha256:" not in str(self.image_ref or ""):
            blockers.append("startup_supervisor_image_not_digest_pinned")
        if not str(self.image_manifest_checksum or "").strip():
            blockers.append("startup_supervisor_image_manifest_checksum_missing")
        if int(self.max_attempts) < 1:
            blockers.append("startup_supervisor_max_attempts_invalid")
        if float(self.wall_clock_cap_seconds) <= 0:
            blockers.append("startup_supervisor_wall_clock_cap_invalid")
        if float(self.total_spend_cap_usd) <= 0:
            blockers.append("startup_supervisor_total_spend_cap_invalid")
        if float(self.hourly_rate_usd) < 0:
            blockers.append("startup_supervisor_hourly_rate_invalid")
        if self.terminal_mode not in VALID_TERMINAL_MODES:
            blockers.append("startup_supervisor_terminal_mode_invalid")
        return blockers

    @property
    def image_digest(self) -> str:
        image = str(self.image_ref or "")
        return image.split("@", 1)[1] if "@" in image else ""


@dataclass
class _AllocationState:
    """The at-most-one live allocation the finalizer is responsible for."""

    attempt_id: str = ""
    instance_id: str = ""
    machine_id: str = ""
    pending_teardown_path: str = ""
    promoted: bool = False
    teardown_proof: dict[str, Any] | None = None
    quarantine_refs: list[dict[str, Any]] = field(default_factory=list)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _transfer_pending_teardown_ownership(
    record_path: str, *, new_owner_lane: str, receipt_path: str
) -> dict[str, Any]:
    """Hand the still-open teardown obligation to the full job, never closing it."""
    path = Path(record_path)
    record = read_json(path)
    record["owner_lane"] = new_owner_lane
    record["ownership_transferred_at"] = utc_now_iso()
    record["ownership_receipt_path"] = receipt_path
    write_json(path, record)
    return record


def run_startup_supervisor(
    *,
    request: StartupSupervisorRequest,
    provider_client: Any,
    out_dir: str | Path,
    inventory: Callable[[], Mapping[str, Any]],
    wait_for_marker: Callable[..., Mapping[str, Any]],
    run_canary: Callable[..., Mapping[str, Any]],
    upload_failure_artifacts: Callable[..., Mapping[str, Any]] | None = None,
    full_job: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    quarantine_registry_dir: str | Path | None = None,
    teardown_registry_dir: str | Path | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run the atomic canary-to-worker transaction and return the manifest.

    ``provider_client`` needs ``name``, ``allocate(**ctx) -> dict``,
    ``inspect(instance_id) -> dict`` (gpu_render_providers inspect shape) and
    ``terminate(instance_id) -> dict``. ``wait_for_marker``/``run_canary``
    receive ``attempt_id``, ``launch_nonce`` and ``instance_id`` keywords and
    return dicts with at least ``status`` plus optional ``failure_class``,
    ``gpu_name`` and ``driver_version``.
    """
    output_dir = Path(out_dir)
    ensure_dir(output_dir)
    provider_name = str(request.provider or "").strip().lower()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "running",
        "blockers": [],
        "run_id": request.run_id,
        "launch_nonce": request.launch_nonce,
        "provider": provider_name,
        "gpu_types": list(request.gpu_types),
        "image_ref": request.image_ref,
        "image_digest": request.image_digest or None,
        "image_manifest_checksum": request.image_manifest_checksum,
        "terminal_mode": request.terminal_mode,
        "isaac_version": request.isaac_version,
        "caps": {
            "max_attempts": int(request.max_attempts),
            "wall_clock_cap_seconds": float(request.wall_clock_cap_seconds),
            "total_spend_cap_usd": float(request.total_spend_cap_usd),
        },
        "attempts": [],
        "quarantine_references": [],
        "ownership_transfer_receipt": None,
        "final_inventory": None,
        "claim_boundary": (
            "This manifest proves startup-supervisor transaction outcomes only: "
            "canary verdicts, spend admission, and API-confirmed teardown or an "
            "explicit ownership transfer. It proves nothing about kitchen scene "
            "load, placement, policy quality, or task success."
        ),
    }

    def _persist_manifest() -> None:
        manifest["updated_at"] = utc_now_iso()
        write_json(output_dir / MANIFEST_FILENAME, manifest)

    validation = request.validation_blockers()
    if validation:
        manifest["status"] = "blocked"
        manifest["blockers"] = validation
        _persist_manifest()
        return manifest

    trace = PhaseTraceRecorder(
        output_dir / TRACE_FILENAME,
        run_id=request.run_id,
        attempt_id="supervisor",
        launch_nonce=request.launch_nonce,
        provider=provider_name,
    )
    ledger = CumulativeSpendLedger(
        output_dir / "startup_cumulative_spend_ledger.json",
        total_cap_usd=float(request.total_spend_cap_usd),
    )
    started_epoch = clock()
    attempts: list[dict[str, Any]] = []
    state = _AllocationState()

    def _persist_attempts() -> None:
        write_json(
            output_dir / ATTEMPTS_FILENAME,
            {
                "schema_version": "isaac_startup_supervisor_attempts.v1",
                "run_id": request.run_id,
                "launch_nonce": request.launch_nonce,
                "spend_ledger": ledger.snapshot(),
                "attempts": attempts,
            },
        )
        manifest["attempts"] = attempts
        _persist_manifest()

    def _persist_quarantine_refs() -> None:
        write_json(
            output_dir / QUARANTINE_REFERENCES_FILENAME,
            {
                "schema_version": "isaac_startup_supervisor_quarantine_refs.v1",
                "run_id": request.run_id,
                "references": state.quarantine_refs,
            },
        )
        manifest["quarantine_references"] = state.quarantine_refs

    def _teardown_live_allocation(reason: str) -> dict[str, Any] | None:
        """Terminate, API-verify terminal, close the pending teardown record."""
        if not state.instance_id or state.promoted:
            return None
        trace.record(PHASE_DELETE, detail={"reason": reason})
        provider_client.terminate(state.instance_id)
        try:
            verify = provider_state_from_inspect(
                provider_client.inspect(state.instance_id)
            )
        except Exception:
            verify = {"provider_status": "", "api_confirmed": False, "http": 0}
        trace.record(
            PHASE_TEARDOWN_VERIFICATION,
            detail={
                "provider_status": verify["provider_status"] or None,
                "api_confirmed": verify["api_confirmed"],
            },
        )
        proof = build_teardown_proof(
            provider=provider_name,
            allocation_id=state.instance_id,
            terminate_requested=True,
            provider_terminal_status=verify["provider_status"] or None,
            verified_at=utc_now_iso() if verify["api_confirmed"] else None,
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API
            if verify["api_confirmed"]
            else None,
        )
        if state.pending_teardown_path:
            closure = close_pending_teardown(state.pending_teardown_path, proof)
            proof["pending_teardown_status"] = closure.get("status")
        state.teardown_proof = proof
        state.instance_id = ""
        return proof

    def _final_inventory_snapshot() -> dict[str, Any]:
        trace.record(PHASE_FINAL_INVENTORY)
        snapshot = _mapping(inventory())
        record = {
            "schema_version": "isaac_startup_supervisor_final_inventory.v1",
            "generated_at": utc_now_iso(),
            "run_id": request.run_id,
            "live_resource_count": int(snapshot.get("live_resource_count") or 0),
            "resources": list(snapshot.get("resources") or []),
        }
        write_json(output_dir / FINAL_INVENTORY_FILENAME, record)
        manifest["final_inventory"] = record
        if record["live_resource_count"] > 0:
            manifest["blockers"] = sorted(
                {*manifest["blockers"], "residual_billable_resource_present"}
            )
        return record

    def _quarantine_machine(
        failure_class: str,
        *,
        phase: str,
        gpu_name: str | None,
        driver_version: str | None,
        evidence_paths: tuple[str, ...] = (),
    ) -> None:
        if not state.machine_id or not request.image_digest:
            return
        try:
            entry = record_machine_quarantine(
                provider=provider_name,
                machine_id=state.machine_id,
                image_digest=request.image_digest,
                isaac_version=request.isaac_version,
                failure_class=failure_class,
                phase=phase,
                gpu_name=gpu_name,
                driver_version=driver_version,
                evidence_paths=evidence_paths,
                run_id=request.run_id,
                registry_dir=quarantine_registry_dir,
            )
        except QuarantineRefused:
            return
        state.quarantine_refs.append(
            {
                "action": "recorded",
                "machine_id": state.machine_id,
                "failure_class": failure_class,
                "path": entry["path"],
            }
        )
        _persist_quarantine_refs()

    previous_sigterm = None

    def _sigterm_handler(signum, frame):  # pragma: no cover - exercised via raise
        raise SupervisorInterrupted("sigterm")

    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)
    except ValueError:
        previous_sigterm = None  # not the main thread; tests raise directly

    reserved_usd = (
        float(request.hourly_rate_usd)
        * float(request.per_attempt_reserved_seconds)
        / 3600.0
    )
    outcome_status = "blocked"
    try:
        for attempt_number in range(1, int(request.max_attempts) + 1):
            attempt_id = f"{request.run_id}-attempt-{attempt_number:02d}"
            attempt_nonce = f"{request.launch_nonce}-a{attempt_number:02d}"
            attempt_started = clock()
            attempt: dict[str, Any] = {
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "launch_nonce": attempt_nonce,
                "gpu_type": request.gpu_types[
                    (attempt_number - 1) % len(request.gpu_types)
                ],
                "reserved_usd": round(reserved_usd, 6),
                "outcome": None,
                "failure_class": None,
            }
            attempts.append(attempt)

            if clock() - started_epoch >= float(request.wall_clock_cap_seconds):
                attempt["outcome"] = "not_started_wall_clock_cap"
                manifest["blockers"].append(
                    "startup_supervisor_wall_clock_cap_exceeded"
                )
                _persist_attempts()
                break

            trace.record(PHASE_PRE_SPEND_INVENTORY)
            pre_inventory = _mapping(inventory())
            attempt["pre_spend_inventory_live_count"] = int(
                pre_inventory.get("live_resource_count") or 0
            )
            if attempt["pre_spend_inventory_live_count"] > 0:
                attempt["outcome"] = "blocked_existing_billable_resource"
                manifest["blockers"].append(
                    "startup_supervisor_existing_billable_resource_present"
                )
                _persist_attempts()
                break

            try:
                ledger.admit(attempt_id=attempt_id, reserved_usd=reserved_usd)
            except SpendCapExceeded as exc:
                attempt["outcome"] = "blocked_spend_cap"
                attempt["spend_admission"] = exc.admission
                manifest["blockers"].append("startup_supervisor_spend_cap_exhausted")
                _persist_attempts()
                break

            pending = open_pending_teardown(
                provider=provider_name,
                lane=SUPERVISOR_LANE,
                run_id=attempt_id,
                job_dir=output_dir,
                registry_dir=teardown_registry_dir,
            )
            state.attempt_id = attempt_id
            state.pending_teardown_path = pending["path"]
            attempt["pending_teardown_record"] = pending["path"]

            def _settle(outcome: str, failure_class: str | None) -> None:
                elapsed = max(0.0, clock() - attempt_started)
                reconciliation = build_spend_reconciliation(
                    provider=provider_name,
                    hourly_rate_usd=float(request.hourly_rate_usd),
                    reserved_seconds=float(request.per_attempt_reserved_seconds),
                    elapsed_seconds=elapsed,
                    stopped_disk_usd_per_hour=request.stopped_disk_usd_per_hour,
                )
                ledger.settle(
                    attempt_id=attempt_id,
                    elapsed_upper_bound_usd=reconciliation[
                        "elapsed_rate_upper_bound_usd"
                    ],
                    outcome=outcome,
                )
                attempt["outcome"] = outcome
                attempt["failure_class"] = failure_class
                attempt["elapsed_seconds"] = round(elapsed, 3)
                attempt["spend_reconciliation"] = reconciliation
                if state.teardown_proof is not None:
                    attempt["teardown_proof_status"] = state.teardown_proof.get(
                        "status"
                    )
                    attempt["teardown_proof"] = state.teardown_proof
                    state.teardown_proof = None
                _persist_attempts()

            trace.record(PHASE_ALLOCATION_REQUESTED, detail={"attempt": attempt_id})
            allocation = _mapping(
                provider_client.allocate(
                    gpu_type=attempt["gpu_type"],
                    attempt_id=attempt_id,
                    launch_nonce=attempt_nonce,
                )
            )
            if allocation.get("status") != "launched" or not allocation.get(
                "instance_id"
            ):
                cancel_pending_teardown(
                    pending["path"],
                    reason="allocation_failed_no_instance",
                    evidence={"blockers": allocation.get("blockers")},
                )
                state.pending_teardown_path = ""
                failure_class = (
                    FAILURE_CLASS_NO_CAPACITY
                    if allocation.get("capacity_outcome")
                    else "allocation_failed"
                )
                attempt["allocation_blockers"] = list(
                    allocation.get("blockers") or []
                )
                _settle("allocation_failed", failure_class)
                continue

            state.instance_id = str(allocation["instance_id"])
            bind_pending_teardown_instance(pending["path"], state.instance_id)
            attempt["instance_id"] = state.instance_id
            trace.record(PHASE_ALLOCATION_CREATED, allocation_id=state.instance_id)

            snapshot = _mapping(provider_client.inspect(state.instance_id))
            state.machine_id = str(snapshot.get("machineId") or "").strip()
            attempt["machine_id"] = state.machine_id or None
            attempt["machine_snapshot"] = {
                "desiredStatus": snapshot.get("desiredStatus"),
                "runtime_present": snapshot.get("runtime_present"),
                "public_ip_present": snapshot.get("public_ip_present"),
                "costPerHr": snapshot.get("costPerHr"),
            }
            trace.record(
                PHASE_MACHINE_IDENTITY_OBSERVED,
                detail={"machine_id": state.machine_id or None},
            )

            if state.machine_id and request.image_digest:
                match = find_active_quarantine(
                    provider=provider_name,
                    machine_id=state.machine_id,
                    image_digest=request.image_digest,
                    isaac_version=request.isaac_version,
                    registry_dir=quarantine_registry_dir,
                )
            else:
                match = None
                if state.machine_id == "":
                    # Some providers cannot reveal machine identity at create
                    # time; record the limitation instead of pretending the
                    # quarantine registry guarantees a different host.
                    attempt["quarantine_lookup"] = "machine_identity_unavailable"
            if match is not None and snapshot.get("runtime_present") is not True:
                state.quarantine_refs.append(
                    {
                        "action": "matched",
                        "machine_id": state.machine_id,
                        "failure_class": match.get("failure_class"),
                        "path": match.get("path"),
                    }
                )
                _persist_quarantine_refs()
                _teardown_live_allocation("quarantined_machine_reallocated")
                _settle("quarantined_machine", FAILURE_CLASS_QUARANTINED_MACHINE)
                continue

            trace.record(PHASE_IMAGE_PULL_NO_RUNTIME)
            marker = _mapping(
                wait_for_marker(
                    attempt_id=attempt_id,
                    launch_nonce=attempt_nonce,
                    instance_id=state.instance_id,
                )
            )
            trace.heartbeat()
            if marker.get("status") != "marker_verified":
                stale = marker.get("status") == "stale_marker"
                failure_class = (
                    FAILURE_CLASS_STALE_MARKER if stale else FAILURE_CLASS_NO_RUNTIME
                )
                attempt["marker"] = {"status": marker.get("status")}
                if not stale:
                    _quarantine_machine(
                        "container_never_started",
                        phase=PHASE_PRE_RUNTIME,
                        gpu_name=None,
                        driver_version=None,
                    )
                if upload_failure_artifacts is not None:
                    attempt["failure_artifacts"] = _mapping(
                        upload_failure_artifacts(
                            attempt_id=attempt_id, instance_id=state.instance_id
                        )
                    )
                _teardown_live_allocation(failure_class)
                _settle("startup_failed", failure_class)
                continue

            trace.record(PHASE_EARLY_MARKER)
            trace.record(PHASE_ISAAC_START)
            canary = _mapping(
                run_canary(
                    attempt_id=attempt_id,
                    launch_nonce=attempt_nonce,
                    instance_id=state.instance_id,
                )
            )
            attempt["canary"] = {
                "status": canary.get("status"),
                "blockers": list(canary.get("blockers") or []),
                "failure_class": canary.get("failure_class"),
                "gpu_name": canary.get("gpu_name"),
                "driver_version": canary.get("driver_version"),
            }
            if canary.get("status") != "passed":
                failure_class = (
                    str(canary.get("failure_class") or "").strip()
                    or "canary_blocked"
                )
                if failure_class in KNOWN_FAILURE_CLASSES:
                    _quarantine_machine(
                        failure_class,
                        phase=PHASE_RUNTIME_CANARY,
                        gpu_name=canary.get("gpu_name"),
                        driver_version=canary.get("driver_version"),
                        evidence_paths=tuple(canary.get("evidence_paths") or ()),
                    )
                if upload_failure_artifacts is not None:
                    attempt["failure_artifacts"] = _mapping(
                        upload_failure_artifacts(
                            attempt_id=attempt_id, instance_id=state.instance_id
                        )
                    )
                _teardown_live_allocation(failure_class)
                _settle("canary_failed", failure_class)
                continue

            # ---- canary passed: terminal transaction -----------------------
            if request.terminal_mode == TERMINAL_MODE_TERMINATE:
                _teardown_live_allocation("terminate_after_canary")
                _settle("canary_passed_terminated", None)
                _final_inventory_snapshot()
                outcome_status = "passed_and_terminated"
                break

            # promote_to_warm_job: same allocation, no second cold launch.
            trace.record(PHASE_PROMOTE)
            receipt_path = str(output_dir / OWNERSHIP_RECEIPT_FILENAME)
            receipt = {
                "schema_version": "isaac_startup_supervisor_ownership_receipt.v1",
                "generated_at": utc_now_iso(),
                "run_id": request.run_id,
                "attempt_id": attempt_id,
                "launch_nonce": attempt_nonce,
                "provider": provider_name,
                "instance_id": state.instance_id,
                "machine_id": state.machine_id or None,
                "image_digest": request.image_digest,
                "pending_teardown_record": state.pending_teardown_path,
                "new_owner_lane": "full_job",
                "teardown_obligation_transferred": True,
                "revoked": False,
            }
            write_json(Path(receipt_path), receipt)
            _transfer_pending_teardown_ownership(
                state.pending_teardown_path,
                new_owner_lane="full_job",
                receipt_path=receipt_path,
            )
            manifest["ownership_transfer_receipt"] = receipt
            try:
                if full_job is not None:
                    attempt["full_job_result"] = _mapping(
                        full_job(
                            {
                                "instance_id": state.instance_id,
                                "machine_id": state.machine_id or None,
                                "launch_nonce": attempt_nonce,
                                "pending_teardown_record": state.pending_teardown_path,
                                "full_job_request": _mapping(
                                    request.full_job_request
                                ),
                            }
                        )
                    )
            except Exception as exc:
                receipt["revoked"] = True
                receipt["revoked_reason"] = "full_job_raised_during_promotion"
                write_json(Path(receipt_path), receipt)
                manifest["ownership_transfer_receipt"] = receipt
                attempt["promotion_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                _teardown_live_allocation("exception_during_promotion")
                _settle("promotion_failed", "exception_during_promotion")
                manifest["blockers"].append(
                    "startup_supervisor_promotion_failed_terminated"
                )
                _final_inventory_snapshot()
                outcome_status = "promotion_failed"
                break
            state.promoted = True
            _settle("canary_passed_promoted", None)
            outcome_status = "passed_and_promoted"
            break
        else:
            manifest["blockers"].append("startup_supervisor_attempts_exhausted")

        if outcome_status == "blocked" and not state.promoted:
            _final_inventory_snapshot()
        manifest["status"] = outcome_status
    except BaseException as exc:
        manifest["status"] = "aborted"
        manifest["blockers"].append(
            "startup_supervisor_interrupted"
            if isinstance(exc, SupervisorInterrupted)
            else "startup_supervisor_exception"
        )
        manifest["abort_error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        if not isinstance(exc, SupervisorInterrupted):
            raise
    finally:
        # The one finalizer every path shares: an un-promoted live allocation
        # is terminated with API-confirmed proof, never left billing.
        proof = _teardown_live_allocation("finalizer")
        if proof is not None:
            manifest["finalizer_teardown_proof"] = proof
            if attempts:
                attempts[-1]["teardown_proof"] = proof
                attempts[-1]["teardown_proof_status"] = proof.get("status")
            if manifest["status"] in {"aborted", "running"}:
                _final_inventory_snapshot()
        _persist_attempts()
        _persist_quarantine_refs()
        _persist_manifest()
        if previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
            except ValueError:  # pragma: no cover
                pass
    return manifest
