"""Deterministic dependency-graph stage scheduler for pipeline orchestrators.

The customer pipeline is a dependency graph, not one long serial script. This
module is the single reusable scheduler for running independent stages
concurrently while preserving the repository's evidence rules:

- Declared dependency edges are the only ordering authority. Orderings that
  are load-bearing for proof validity (rights review before privacy delivery,
  seal before outcome release, controls before scored policy cells, asset
  qualification before exact contact checks, teardown before paid-resource
  closure) must be expressed as edges so no scheduling policy can drop them.
- Serial default: ``max_concurrency=1`` executes stages one at a time in
  deterministic topological order, which for a graph derived from an existing
  sequential list reproduces that list exactly.
- Paid stages never overlap each other unless the caller passes
  ``paid_concurrency_authorized=True``. This mirrors the paid-resource
  allocator's explicit concurrent-instance authority: serial paid execution is
  the already-authorized default, concurrency is a separately granted
  authority, and this scheduler cannot widen it implicitly.
- Failure is fail-closed and typed: a failed stage marks every transitive
  dependent ``blocked`` with the failed stage named in the reason. Independent
  stages still run and their evidence is retained. There is no automatic
  retry.
- Results are deterministic: execution rows are emitted in declared stage
  order regardless of completion order. Wall-clock fields and completion
  order are observability evidence only and must never enter a digest;
  callers that digest a manifest must use :meth:`StageGraphResult.manifest`
  with ``include_timing=False``.

The scheduler runs callables in-process. It never launches providers, spends
money, or retries on behalf of a stage; stages keep their own spend guards,
watchdogs, and teardown obligations.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso
from .stage_outcome import OutcomeKind, StageOutcome

STAGE_GRAPH_SCHEMA_VERSION = "stage_graph_execution.v1"

PAID_SERIAL_GROUP = "__paid_serial__"

_STAGE_ID_MAX_LENGTH = 256
_REASON_MAX_LENGTH = 500


class StageGraphError(ValueError):
    """Raised when a stage graph is structurally invalid before execution."""


@dataclass(frozen=True)
class StageSpec:
    """One schedulable unit of pipeline work.

    ``run`` receives no arguments and returns the stage artifact (a mapping)
    or ``None``. Inputs must be bound by the caller (closure or functools
    partial) so the dependency edges remain the only inter-stage coupling.
    Stages that mutate shared state must either declare edges that serialize
    those mutations or be left in a serial group.
    """

    stage_id: str
    run: Callable[[], Mapping[str, Any] | None]
    depends_on: tuple[str, ...] = ()
    paid: bool = False
    serial_group: str | None = None


@dataclass(frozen=True)
class StageExecution:
    stage_id: str
    status: str  # completed | failed | blocked
    outcome: StageOutcome
    depends_on: tuple[str, ...]
    paid: bool
    serial_group: str | None
    started_at: str | None
    completed_at: str | None
    duration_seconds: float | None

    def to_row(self, *, include_timing: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {
            "stage_id": self.stage_id,
            "status": self.status,
            "outcome": self.outcome.to_mapping(),
            "depends_on": list(self.depends_on),
            "paid": self.paid,
            "serial_group": self.serial_group,
        }
        if include_timing:
            row["started_at"] = self.started_at
            row["completed_at"] = self.completed_at
            row["duration_seconds"] = self.duration_seconds
        return row


@dataclass(frozen=True)
class StageGraphResult:
    executions: tuple[StageExecution, ...]
    completion_order: tuple[str, ...]
    max_concurrency: int
    paid_concurrency_authorized: bool
    observed_max_overlap: int
    status: str = field(init=False)

    def __post_init__(self) -> None:
        failed = any(execution.status != "completed" for execution in self.executions)
        object.__setattr__(self, "status", "completed_with_failures" if failed else "completed")

    def execution(self, stage_id: str) -> StageExecution:
        for row in self.executions:
            if row.stage_id == stage_id:
                return row
        raise KeyError(stage_id)

    def artifact(self, stage_id: str) -> Mapping[str, Any] | None:
        return self.execution(stage_id).outcome.artifact

    def manifest(self, *, include_timing: bool = True) -> dict[str, Any]:
        """Deterministic execution evidence.

        Rows follow declared stage order, never completion order. With
        ``include_timing=False`` the manifest is byte-stable for identical
        stage outcomes and safe to digest; ``completion_order`` and timing are
        wall-clock observability and are omitted in that mode.
        """

        manifest: dict[str, Any] = {
            "schema_version": STAGE_GRAPH_SCHEMA_VERSION,
            "status": self.status,
            "max_concurrency": self.max_concurrency,
            "paid_concurrency_authorized": self.paid_concurrency_authorized,
            "stages": [row.to_row(include_timing=include_timing) for row in self.executions],
        }
        if include_timing:
            manifest["completion_order"] = list(self.completion_order)
            manifest["observed_max_overlap"] = self.observed_max_overlap
        return manifest


def _bounded_reason(value: str) -> str:
    text = " ".join(str(value).split())
    if len(text) > _REASON_MAX_LENGTH:
        return text[: _REASON_MAX_LENGTH - 3] + "..."
    return text


def _validate_stages(stages: Sequence[StageSpec]) -> list[StageSpec]:
    ordered = list(stages)
    if not ordered:
        raise StageGraphError("stage_graph_empty")
    seen: set[str] = set()
    for stage in ordered:
        stage_id = stage.stage_id
        if (
            not isinstance(stage_id, str)
            or not stage_id.strip()
            or stage_id != stage_id.strip()
            or len(stage_id) > _STAGE_ID_MAX_LENGTH
            or any(character.isspace() for character in stage_id)
        ):
            raise StageGraphError(f"stage_graph_invalid_stage_id:{stage_id!r}")
        if stage_id in seen:
            raise StageGraphError(f"stage_graph_duplicate_stage_id:{stage_id}")
        seen.add(stage_id)
        if not callable(stage.run):
            raise StageGraphError(f"stage_graph_stage_not_callable:{stage_id}")
    known = {stage.stage_id for stage in ordered}
    for stage in ordered:
        for dependency in stage.depends_on:
            if dependency not in known:
                raise StageGraphError(
                    f"stage_graph_unknown_dependency:{stage.stage_id}->{dependency}"
                )
            if dependency == stage.stage_id:
                raise StageGraphError(f"stage_graph_self_dependency:{stage.stage_id}")
    _require_acyclic(ordered)
    return ordered


def _require_acyclic(stages: Sequence[StageSpec]) -> None:
    remaining: dict[str, set[str]] = {
        stage.stage_id: set(stage.depends_on) for stage in stages
    }
    while remaining:
        ready = sorted(
            stage_id for stage_id, deps in remaining.items() if not deps
        )
        if not ready:
            cycle_members = ",".join(sorted(remaining))
            raise StageGraphError(f"stage_graph_cycle_detected:{cycle_members}")
        for stage_id in ready:
            del remaining[stage_id]
        for deps in remaining.values():
            deps.difference_update(ready)


def _effective_serial_group(stage: StageSpec, *, paid_concurrency_authorized: bool) -> str | None:
    if stage.serial_group is not None:
        return stage.serial_group
    if stage.paid and not paid_concurrency_authorized:
        return PAID_SERIAL_GROUP
    return None


def stage_concurrency_from_env(
    variable: str,
    *,
    default: int = 1,
    maximum: int = 8,
) -> int:
    """Read a bounded stage-concurrency knob from the environment.

    Invalid or out-of-range values fail closed to the serial default rather
    than silently widening concurrency.
    """

    raw = str(os.getenv(variable) or "").strip()
    if not raw:
        return max(1, min(int(default), maximum))
    try:
        value = int(raw)
    except ValueError:
        return 1
    if value < 1 or value > maximum:
        return 1
    return value


def run_stage_graph(
    stages: Sequence[StageSpec],
    *,
    max_concurrency: int = 1,
    paid_concurrency_authorized: bool = False,
    cancel_pending_on_failure: bool = False,
) -> StageGraphResult:
    """Execute a validated stage graph with bounded, authorized concurrency.

    ``max_concurrency=1`` runs stages sequentially in deterministic
    topological order (declared order among simultaneously-ready stages).
    Dependents of a failed or blocked stage are blocked with a typed reason
    naming the failed ancestor. With ``cancel_pending_on_failure=True`` no new
    stage starts after the first failure; already-running stages always finish
    so their own teardown obligations complete.
    """

    ordered = _validate_stages(stages)
    if int(max_concurrency) < 1:
        raise StageGraphError(f"stage_graph_invalid_max_concurrency:{max_concurrency}")
    max_workers = int(max_concurrency)

    by_id = {stage.stage_id: stage for stage in ordered}
    declared_order = [stage.stage_id for stage in ordered]
    dependents: dict[str, list[str]] = {stage_id: [] for stage_id in declared_order}
    for stage in ordered:
        for dependency in stage.depends_on:
            dependents[dependency].append(stage.stage_id)

    executions: dict[str, StageExecution] = {}
    completion_order: list[str] = []
    unfinished_dependencies = {
        stage.stage_id: set(stage.depends_on) for stage in ordered
    }
    busy_groups: set[str] = set()
    failure_observed = False
    running = 0
    observed_max_overlap = 0
    lock = threading.Lock()
    completion_lock = threading.Lock()
    completion_sequence = 0
    completion_rank: dict[str, int] = {}

    def _record_terminal(
        stage_id: str, execution: StageExecution, *, executed: bool = True
    ) -> None:
        executions[stage_id] = execution
        if executed:
            completion_order.append(stage_id)
        for dependent in dependents[stage_id]:
            unfinished_dependencies[dependent].discard(stage_id)

    def _block(stage_id: str, reason: str) -> None:
        stage = by_id[stage_id]
        _record_terminal(
            stage_id,
            StageExecution(
                stage_id=stage_id,
                status="blocked",
                outcome=StageOutcome(kind=OutcomeKind.BLOCKED, reason=reason),
                depends_on=stage.depends_on,
                paid=stage.paid,
                serial_group=stage.serial_group,
                started_at=None,
                completed_at=None,
                duration_seconds=None,
            ),
            executed=False,
        )

    def _execute(stage: StageSpec) -> StageExecution:
        nonlocal completion_sequence
        started_at = utc_now_iso()
        started_monotonic = time.monotonic()
        try:
            artifact = stage.run()
        except BaseException as error:  # noqa: BLE001 - typed, retained, fail-closed
            duration = time.monotonic() - started_monotonic
            reason = _bounded_reason(f"{type(error).__name__}: {error}")
            execution = StageExecution(
                stage_id=stage.stage_id,
                status="failed",
                outcome=StageOutcome(kind=OutcomeKind.FAILED, reason=reason),
                depends_on=stage.depends_on,
                paid=stage.paid,
                serial_group=stage.serial_group,
                started_at=started_at,
                completed_at=utc_now_iso(),
                duration_seconds=duration,
            )
        else:
            duration = time.monotonic() - started_monotonic
            mapped = dict(artifact) if isinstance(artifact, Mapping) else {}
            execution = StageExecution(
                stage_id=stage.stage_id,
                status="completed",
                outcome=StageOutcome(kind=OutcomeKind.PRODUCED, artifact=mapped),
                depends_on=stage.depends_on,
                paid=stage.paid,
                serial_group=stage.serial_group,
                started_at=started_at,
                completed_at=utc_now_iso(),
                duration_seconds=duration,
            )
        # A FIRST_COMPLETED wait returns a set. When multiple futures finish
        # before the scheduler wakes, iterating that set invents an arbitrary
        # completion order. Record the worker-observed order before each
        # future becomes done, then use it only for observability evidence.
        with completion_lock:
            completion_rank[stage.stage_id] = completion_sequence
            completion_sequence += 1
        return execution

    def _resolve_blocked() -> None:
        # Repeatedly settle stages whose dependencies can no longer all
        # complete, so transitive dependents fail closed with a typed reason.
        settled = True
        while settled:
            settled = False
            for stage_id in declared_order:
                if stage_id in executions:
                    continue
                failed_ancestors = sorted(
                    dependency
                    for dependency in by_id[stage_id].depends_on
                    if dependency in executions
                    and executions[dependency].status != "completed"
                )
                if failed_ancestors:
                    _block(
                        stage_id,
                        f"blocked_by_dependency_failure:{','.join(failed_ancestors)}",
                    )
                    settled = True

    def _ready_stages() -> list[StageSpec]:
        ready: list[StageSpec] = []
        for stage_id in declared_order:
            if stage_id in executions:
                continue
            if unfinished_dependencies[stage_id]:
                continue
            stage = by_id[stage_id]
            group = _effective_serial_group(
                stage, paid_concurrency_authorized=paid_concurrency_authorized
            )
            if group is not None and group in busy_groups:
                continue
            ready.append(stage)
        return ready

    if max_workers == 1:
        while len(executions) < len(ordered):
            _resolve_blocked()
            if failure_observed and cancel_pending_on_failure:
                for stage_id in declared_order:
                    if stage_id not in executions:
                        _block(stage_id, "cancelled_after_prior_stage_failure")
                break
            ready = _ready_stages()
            if not ready:
                if len(executions) < len(ordered):
                    _resolve_blocked()
                    if len(executions) < len(ordered) and not _ready_stages():
                        # Cycle detection makes this unreachable; fail closed.
                        raise StageGraphError("stage_graph_scheduling_stalled")
                continue
            stage = ready[0]
            execution = _execute(stage)
            if execution.status != "completed":
                failure_observed = True
            _record_terminal(stage.stage_id, execution)
        _resolve_blocked()
        observed_max_overlap = 1
    else:
        futures: dict[Future[StageExecution], str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while True:
                with lock:
                    _resolve_blocked()
                    if failure_observed and cancel_pending_on_failure:
                        for stage_id in declared_order:
                            if stage_id not in executions and not any(
                                stage_id == pending for pending in futures.values()
                            ):
                                _block(stage_id, "cancelled_after_prior_stage_failure")
                    submittable: list[StageSpec] = []
                    if not (failure_observed and cancel_pending_on_failure):
                        in_flight = set(futures.values())
                        for stage in _ready_stages():
                            if stage.stage_id in in_flight:
                                continue
                            if running + len(submittable) >= max_workers:
                                break
                            group = _effective_serial_group(
                                stage,
                                paid_concurrency_authorized=paid_concurrency_authorized,
                            )
                            if group is not None:
                                # Re-check inside the batch: an earlier pick in
                                # this same submission round may already hold
                                # the group token.
                                if group in busy_groups:
                                    continue
                                busy_groups.add(group)
                            submittable.append(stage)
                    for stage in submittable:
                        running += 1
                        observed_max_overlap = max(observed_max_overlap, running)
                        futures[pool.submit(_execute, stage)] = stage.stage_id
                    if not futures:
                        if len(executions) < len(ordered) and not (
                            failure_observed and cancel_pending_on_failure
                        ):
                            _resolve_blocked()
                            if len(executions) < len(ordered) and not _ready_stages():
                                raise StageGraphError("stage_graph_scheduling_stalled")
                            continue
                        break
                done, _pending = wait(set(futures), return_when=FIRST_COMPLETED)
                with lock:
                    for future in sorted(
                        done, key=lambda item: completion_rank[futures[item]]
                    ):
                        stage_id = futures.pop(future)
                        execution = future.result()
                        running -= 1
                        stage = by_id[stage_id]
                        group = _effective_serial_group(
                            stage,
                            paid_concurrency_authorized=paid_concurrency_authorized,
                        )
                        if group is not None:
                            busy_groups.discard(group)
                        if execution.status != "completed":
                            failure_observed = True
                        _record_terminal(stage_id, execution)
        with lock:
            _resolve_blocked()
            if failure_observed and cancel_pending_on_failure:
                for stage_id in declared_order:
                    if stage_id not in executions:
                        _block(stage_id, "cancelled_after_prior_stage_failure")

    ordered_executions = tuple(executions[stage_id] for stage_id in declared_order)
    return StageGraphResult(
        executions=ordered_executions,
        completion_order=tuple(completion_order),
        max_concurrency=max_workers,
        paid_concurrency_authorized=paid_concurrency_authorized,
        observed_max_overlap=observed_max_overlap,
    )


__all__ = [
    "PAID_SERIAL_GROUP",
    "STAGE_GRAPH_SCHEMA_VERSION",
    "StageExecution",
    "StageGraphError",
    "StageGraphResult",
    "StageSpec",
    "run_stage_graph",
    "stage_concurrency_from_env",
]
