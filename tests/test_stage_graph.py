"""Contract tests for the deterministic dependency-graph stage scheduler.

These pin the general-harness parallelization contract: independent stages may
run concurrently, declared edges are the only ordering authority, paid stages
never overlap without explicit authorization, failures block dependents with
typed reasons instead of silently skipping them, and emitted evidence rows are
deterministic regardless of completion order.
"""

from __future__ import annotations

import json
import threading

import pytest

from blueprint_pipeline.core.stage_graph import (
    PAID_SERIAL_GROUP,
    StageGraphError,
    StageSpec,
    run_stage_graph,
    stage_concurrency_from_env,
)

_WAIT_SECONDS = 5.0
_NO_OVERLAP_PROBE_SECONDS = 0.3


def _spec(stage_id: str, run, **kwargs) -> StageSpec:
    return StageSpec(stage_id=stage_id, run=run, **kwargs)


def test_serial_mode_runs_declared_order_and_collects_artifacts() -> None:
    order: list[str] = []

    def _make(stage_id: str):
        def _run():
            order.append(stage_id)
            return {"stage": stage_id}

        return _run

    result = run_stage_graph(
        [
            _spec("alpha", _make("alpha")),
            _spec("beta", _make("beta"), depends_on=("alpha",)),
            _spec("gamma", _make("gamma")),
        ],
        max_concurrency=1,
    )
    assert order == ["alpha", "beta", "gamma"]
    assert result.status == "completed"
    assert result.completion_order == ("alpha", "beta", "gamma")
    assert result.artifact("beta") == {"stage": "beta"}
    assert [row.stage_id for row in result.executions] == ["alpha", "beta", "gamma"]


def test_validation_fails_closed() -> None:
    def _noop():
        return {}

    with pytest.raises(StageGraphError, match="stage_graph_empty"):
        run_stage_graph([])
    with pytest.raises(StageGraphError, match="stage_graph_duplicate_stage_id"):
        run_stage_graph([_spec("dup", _noop), _spec("dup", _noop)])
    with pytest.raises(StageGraphError, match="stage_graph_unknown_dependency"):
        run_stage_graph([_spec("solo", _noop, depends_on=("ghost",))])
    with pytest.raises(StageGraphError, match="stage_graph_self_dependency"):
        run_stage_graph([_spec("self", _noop, depends_on=("self",))])
    with pytest.raises(StageGraphError, match="stage_graph_cycle_detected"):
        run_stage_graph(
            [
                _spec("first", _noop, depends_on=("second",)),
                _spec("second", _noop, depends_on=("first",)),
            ]
        )
    with pytest.raises(StageGraphError, match="stage_graph_invalid_max_concurrency"):
        run_stage_graph([_spec("solo", _noop)], max_concurrency=0)
    with pytest.raises(StageGraphError, match="stage_graph_invalid_stage_id"):
        run_stage_graph([_spec("bad id", _noop)])
    with pytest.raises(StageGraphError, match="stage_graph_stage_not_callable"):
        run_stage_graph([StageSpec(stage_id="solo", run=None)])  # type: ignore[arg-type]


def test_independent_stages_actually_overlap_when_authorized_concurrency() -> None:
    barrier = threading.Barrier(2, timeout=_WAIT_SECONDS)

    def _rendezvous(stage_id: str):
        def _run():
            barrier.wait()
            return {"stage": stage_id}

        return _run

    result = run_stage_graph(
        [
            _spec("left", _rendezvous("left")),
            _spec("right", _rendezvous("right")),
        ],
        max_concurrency=2,
    )
    assert result.status == "completed"
    assert result.observed_max_overlap == 2


def test_dependency_failure_blocks_transitive_dependents_and_keeps_independents() -> None:
    def _boom():
        raise RuntimeError("stage exploded")

    def _ok():
        return {"fine": True}

    result = run_stage_graph(
        [
            _spec("broken", _boom),
            _spec("child", _ok, depends_on=("broken",)),
            _spec("grandchild", _ok, depends_on=("child",)),
            _spec("independent", _ok),
        ],
        max_concurrency=1,
    )
    assert result.status == "completed_with_failures"
    broken = result.execution("broken")
    assert broken.status == "failed"
    assert "RuntimeError: stage exploded" in (broken.outcome.reason or "")
    child = result.execution("child")
    assert child.status == "blocked"
    assert child.outcome.reason == "blocked_by_dependency_failure:broken"
    grandchild = result.execution("grandchild")
    assert grandchild.status == "blocked"
    assert grandchild.outcome.reason == "blocked_by_dependency_failure:child"
    assert result.execution("independent").status == "completed"


def test_cancel_pending_on_failure_blocks_unstarted_independent_stages() -> None:
    def _boom():
        raise RuntimeError("halt")

    def _ok():
        return {}

    result = run_stage_graph(
        [
            _spec("broken", _boom),
            _spec("later", _ok),
        ],
        max_concurrency=1,
        cancel_pending_on_failure=True,
    )
    assert result.execution("broken").status == "failed"
    later = result.execution("later")
    assert later.status == "blocked"
    assert later.outcome.reason == "cancelled_after_prior_stage_failure"


def test_paid_stages_never_overlap_without_explicit_authorization() -> None:
    first_paid_finished = threading.Event()
    second_paid_started = threading.Event()

    def _first_paid():
        overlapped = second_paid_started.wait(_NO_OVERLAP_PROBE_SECONDS)
        first_paid_finished.set()
        return {"overlapped": overlapped}

    def _second_paid():
        second_paid_started.set()
        return {"first_already_finished": first_paid_finished.is_set()}

    result = run_stage_graph(
        [
            _spec("paid-one", _first_paid, paid=True),
            _spec("paid-two", _second_paid, paid=True),
        ],
        max_concurrency=4,
        paid_concurrency_authorized=False,
    )
    assert result.status == "completed"
    assert result.artifact("paid-one") == {"overlapped": False}
    assert result.artifact("paid-two") == {"first_already_finished": True}


def test_paid_stages_overlap_only_with_explicit_authorization() -> None:
    barrier = threading.Barrier(2, timeout=_WAIT_SECONDS)

    def _paid(stage_id: str):
        def _run():
            barrier.wait()
            return {"stage": stage_id}

        return _run

    result = run_stage_graph(
        [
            _spec("paid-one", _paid("paid-one"), paid=True),
            _spec("paid-two", _paid("paid-two"), paid=True),
        ],
        max_concurrency=2,
        paid_concurrency_authorized=True,
    )
    assert result.status == "completed"
    assert result.observed_max_overlap == 2


def test_unpaid_stage_overlaps_paid_stage_without_authorization() -> None:
    barrier = threading.Barrier(2, timeout=_WAIT_SECONDS)

    def _rendezvous(stage_id: str):
        def _run():
            barrier.wait()
            return {"stage": stage_id}

        return _run

    result = run_stage_graph(
        [
            _spec("paid-one", _rendezvous("paid-one"), paid=True),
            _spec("free-one", _rendezvous("free-one")),
        ],
        max_concurrency=2,
        paid_concurrency_authorized=False,
    )
    assert result.status == "completed"
    assert result.observed_max_overlap == 2


def test_custom_serial_group_serializes_members() -> None:
    group_running = threading.Event()

    def _member(stage_id: str):
        def _run():
            assert not group_running.is_set(), "serial group members overlapped"
            group_running.set()
            try:
                return {"stage": stage_id}
            finally:
                group_running.clear()

        return _run

    result = run_stage_graph(
        [
            _spec("member-one", _member("member-one"), serial_group="gpu0"),
            _spec("member-two", _member("member-two"), serial_group="gpu0"),
        ],
        max_concurrency=4,
    )
    assert result.status == "completed"


def test_manifest_rows_follow_declared_order_even_when_completion_reverses() -> None:
    release_first = threading.Event()

    def _slow_first():
        released = release_first.wait(_WAIT_SECONDS)
        return {"released": released}

    def _fast_second():
        release_first.set()
        return {"released": True}

    result = run_stage_graph(
        [
            _spec("slow-first", _slow_first),
            _spec("fast-second", _fast_second),
        ],
        max_concurrency=2,
    )
    assert result.completion_order == ("fast-second", "slow-first")
    manifest = result.manifest(include_timing=False)
    assert [row["stage_id"] for row in manifest["stages"]] == ["slow-first", "fast-second"]
    assert "completion_order" not in manifest
    assert "started_at" not in manifest["stages"][0]

    serial_manifest = run_stage_graph(
        [
            _spec("slow-first", lambda: {"released": True}),
            _spec("fast-second", lambda: {"released": True}),
        ],
        max_concurrency=1,
    ).manifest(include_timing=False)
    concurrent_manifest = dict(manifest)
    concurrent_manifest["max_concurrency"] = serial_manifest["max_concurrency"]
    assert json.dumps(concurrent_manifest, sort_keys=True) == json.dumps(
        serial_manifest, sort_keys=True
    )


def test_diamond_dependencies_gate_the_join_stage() -> None:
    completed: list[str] = []
    lock = threading.Lock()

    def _make(stage_id: str):
        def _run():
            with lock:
                completed.append(stage_id)
            return {"stage": stage_id}

        return _run

    result = run_stage_graph(
        [
            _spec("root", _make("root")),
            _spec("left", _make("left"), depends_on=("root",)),
            _spec("right", _make("right"), depends_on=("root",)),
            _spec("join", _make("join"), depends_on=("left", "right")),
        ],
        max_concurrency=3,
    )
    assert result.status == "completed"
    assert completed[0] == "root"
    assert completed[-1] == "join"
    assert set(completed[1:3]) == {"left", "right"}


def test_stage_returning_none_produces_empty_artifact() -> None:
    result = run_stage_graph([_spec("quiet", lambda: None)])
    assert result.artifact("quiet") == {}
    assert result.execution("quiet").status == "completed"


def test_stage_concurrency_from_env_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    variable = "BLUEPRINT_TEST_STAGE_CONCURRENCY"
    monkeypatch.delenv(variable, raising=False)
    assert stage_concurrency_from_env(variable) == 1
    assert stage_concurrency_from_env(variable, default=3) == 3
    monkeypatch.setenv(variable, "4")
    assert stage_concurrency_from_env(variable) == 4
    monkeypatch.setenv(variable, "not-a-number")
    assert stage_concurrency_from_env(variable) == 1
    monkeypatch.setenv(variable, "0")
    assert stage_concurrency_from_env(variable) == 1
    monkeypatch.setenv(variable, "99")
    assert stage_concurrency_from_env(variable) == 1
    monkeypatch.setenv(variable, "6")
    assert stage_concurrency_from_env(variable, maximum=4) == 1


def test_paid_serial_group_constant_is_reserved_for_the_scheduler() -> None:
    assert PAID_SERIAL_GROUP == "__paid_serial__"
