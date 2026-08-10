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


def _positive(**overrides):
    row = {
        "control_id": "forced_positive",
        "angle_at_release_degrees": 30.79,
        "settled_angle_degrees": 51.39,
        "maximum_angle_degrees": 51.39,
        "drift_after_release_degrees": 20.60,
        "angle_trace_degrees": [0.0, 6.2, 30.8, 45.0, 51.3, 51.39, 51.39, 51.39],
    }
    row.update(overrides)
    return row


def test_reaching_the_window_is_about_where_the_door_gets_to() -> None:
    """The release angle is deliberately below the window, not inside it.

    The schedule lets go early on purpose and lets the door coast in, so
    checking the release angle against the window contradicts the design by
    construction - it can only pass if the coast model is wrong.
    """

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(), window=[45.0, 55.0], hold_tolerance_degrees=0.5
    )

    assert verdict["reaches_success_window"]["passed"] is True
    assert verdict["reaches_success_window"]["maximum_angle_degrees"] == 51.39


def test_a_door_that_never_gets_there_still_fails() -> None:
    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(maximum_angle_degrees=8.73, settled_angle_degrees=8.73),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["reaches_success_window"]["passed"] is False


def test_holding_means_the_door_stopped_not_merely_that_it_ended_up_there() -> None:
    """Coast is not drift, and the endpoint alone cannot tell them apart.

    A door still swinging when the clock runs out lands somewhere by accident;
    one that came to rest is holding. Only the tail of the settle window
    distinguishes them.
    """

    module = _worker()

    still_moving = module._evaluate_positive(
        positive=_positive(
            angle_trace_degrees=[0.0, 20.0, 35.0, 44.0, 47.0, 49.0, 50.5, 51.39]
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )
    at_rest = module._evaluate_positive(
        positive=_positive(), window=[45.0, 55.0], hold_tolerance_degrees=0.5
    )

    assert still_moving["holds_after_release"]["passed"] is False
    assert at_rest["holds_after_release"]["passed"] is True


def test_coming_to_rest_outside_the_window_is_not_holding() -> None:
    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(
            settled_angle_degrees=58.5,
            maximum_angle_degrees=58.5,
            angle_trace_degrees=[0.0, 30.0, 58.5, 58.5, 58.5, 58.5],
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["passed"] is False


def test_a_result_without_a_trace_says_so_rather_than_guessing() -> None:
    """Older runs carry no trace; inferring stillness from an endpoint would lie."""

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(angle_trace_degrees=[]),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["tail_motion_degrees"] is None
    assert verdict["holds_after_release"]["passed"] is False
