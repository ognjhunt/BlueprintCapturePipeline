from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _worker():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp009d_articulated_controls_worker.py"
    )
    spec = importlib.util.spec_from_file_location("controls_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _App:
    """A simulator whose close() never returns, as Isaac's can behave."""

    def __init__(self, order: list[str], hard_exit: bool = True):
        self.order = order
        self.hard_exit = hard_exit

    def close(self):
        self.order.append("close")
        if self.hard_exit:
            raise SystemExit(0)


def test_the_result_is_written_before_the_simulator_is_closed(tmp_path) -> None:
    """Isaac's close() can end the process, taking the diagnosis with it.

    That is not hypothetical: a run died on a physics-scene conflict, the
    shutdown swallowed the traceback, and the retained evidence said only
    "process exited without result" - which is the one thing a paid,
    no-retry launch must never come back with.
    """

    module = _worker()
    order: list[str] = []
    output = tmp_path / "result.json"
    original = module._persist
    module._persist = lambda path, value: (order.append("persist"), original(path, value))[1]

    with pytest.raises(SystemExit):
        module._finalize(
            output=output,
            result={"schema_version": "x", "status": "blocked", "blockers": ["boom"]},
            simulation_app=_App(order),
        )

    assert order == ["persist", "close"]
    assert json.loads(output.read_text(encoding="utf-8"))["blockers"] == ["boom"]


def test_a_close_that_fails_does_not_lose_an_already_written_result(tmp_path) -> None:
    module = _worker()
    output = tmp_path / "result.json"

    class _Raising:
        def close(self):
            raise RuntimeError("driver gone")

    module._finalize(
        output=output,
        result={"schema_version": "x", "status": "completed"},
        simulation_app=_Raising(),
    )

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"


def test_the_persisted_result_always_carries_both_digest_fields(tmp_path) -> None:
    """The collector reads _canonical_digest; a result without it is discarded."""

    module = _worker()
    output = tmp_path / "result.json"

    module._finalize(
        output=output,
        result={"schema_version": "x", "status": "completed"},
        simulation_app=_App([], hard_exit=False),
    )

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["result_digest"] == stored["_canonical_digest"]
    assert stored["result_digest"].startswith("sha256:")


def test_the_angle_trace_survives_into_the_result_at_bounded_size() -> None:
    """Summary angles cannot show how the door got there.

    Start, release and settle are three numbers; whether the door crept, slammed
    and bounced, or oscillated through the seal is invisible in them, and that
    is exactly what distinguishes a good positive from a lucky one. The trace
    has to come back - bounded, because a 600-step run at full rate bloats
    every retained receipt for no extra information.
    """

    module = _worker()
    trace = [float(i) * 0.1 for i in range(600)]

    kept = module._downsample(trace, limit=64)

    assert len(kept) <= 64
    assert kept[0] == trace[0]
    assert kept[-1] == trace[-1]


def test_a_short_trace_is_returned_whole() -> None:
    module = _worker()
    trace = [0.0, 1.0, 2.0]

    assert module._downsample(trace, limit=64) == trace


def test_an_empty_trace_does_not_explode() -> None:
    module = _worker()

    assert module._downsample([], limit=64) == []


def test_the_peak_is_never_lost_to_downsampling() -> None:
    """A transient overshoot that decays is the whole story of a bad release."""

    module = _worker()
    trace = [0.0] * 50 + [88.0] + [50.0] * 50

    kept = module._downsample(trace, limit=16)

    assert max(kept) == 88.0
