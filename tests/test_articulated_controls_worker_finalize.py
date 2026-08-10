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

    Both of these finish at the same angle inside the window. One is still
    travelling when the clock runs out and happens to be passing through; the
    other has come to rest. Only how the motion changes across the settle
    window separates them.
    """

    module = _worker()

    still_moving = module._evaluate_positive(
        positive=_positive(
            settle_trace_degrees=[44.0, 45.5, 47.0, 48.5, 50.0, 51.39]
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )
    at_rest = module._evaluate_positive(
        positive=_positive(
            settle_trace_degrees=[49.0, 50.4, 51.0, 51.2, 51.35, 51.39]
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert still_moving["holds_after_release"]["passed"] is False
    assert at_rest["holds_after_release"]["passed"] is True


def test_coming_to_rest_outside_the_window_is_not_holding() -> None:
    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(
            settled_angle_degrees=58.5,
            maximum_angle_degrees=58.5,
            settle_trace_degrees=[58.5, 58.5, 58.5, 58.5, 58.5, 58.5],
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["passed"] is False


def test_a_result_without_a_trace_says_so_rather_than_guessing() -> None:
    """Older runs carry no trace; inferring stillness from an endpoint would lie."""

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(settle_trace_degrees=[]),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["tail_motion_degrees"] is None
    assert verdict["holds_after_release"]["passed"] is False


def test_holding_is_judged_on_the_settle_window_alone() -> None:
    """A tail taken from the whole episode still contains the coast.

    The door is meant to be moving then - that is the coast doing its job - so
    measuring across it reads deceleration as failure to hold.
    """

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(
            settle_trace_degrees=[49.0, 50.4, 51.0, 51.15, 51.19, 51.20],
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["passed"] is True
    assert verdict["holds_after_release"]["motion_is_decaying"] is True


def test_a_door_swinging_shut_through_the_window_does_not_hold() -> None:
    """Self-closing is the failure mode this readback exists to catch."""

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(
            settle_trace_degrees=[51.0, 49.0, 47.0, 45.5, 43.0, 40.0],
            settled_angle_degrees=40.0,
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["passed"] is False


def test_a_door_that_leaves_the_window_and_returns_does_not_hold() -> None:
    """Ending up inside is not the same as staying inside."""

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(
            settle_trace_degrees=[51.0, 56.5, 57.0, 54.0, 51.5, 51.2],
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["passed"] is False


def test_still_accelerating_away_does_not_hold_however_small_the_motion() -> None:
    """Decaying motion settles; growing motion is on its way out of the window."""

    module = _worker()

    verdict = module._evaluate_positive(
        positive=_positive(
            settle_trace_degrees=[51.0, 51.05, 51.12, 51.25, 51.5, 52.0],
        ),
        window=[45.0, 55.0],
        hold_tolerance_degrees=0.5,
    )

    assert verdict["holds_after_release"]["motion_is_decaying"] is False
    assert verdict["holds_after_release"]["passed"] is False


def _settle(trace, **overrides):
    return _positive(settle_trace_degrees=trace, settled_angle_degrees=trace[-1],
                     maximum_angle_degrees=max(trace), **overrides)


def _hold(trace):
    return _worker()._evaluate_positive(
        positive=_settle(trace), window=[45.0, 55.0], hold_tolerance_degrees=0.5
    )["holds_after_release"]


def test_the_coast_into_the_window_is_not_a_failure_to_stay_in_it() -> None:
    """The settle window opens at release, and release is below the window.

    The schedule lets go early on purpose, so the door spends the first part of
    settling on its way in. Requiring every settle sample to be inside the
    window fails the design rather than the door - which is what the real run
    reported: stayed_inside_window False on a door that came to rest at 51.2.
    """

    assert _hold([30.7, 38.0, 45.5, 49.0, 50.8, 51.15, 51.19, 51.20])["passed"] is True


def test_once_inside_the_door_has_to_stay_inside() -> None:
    """Entering and then swinging back out is the self-closing failure."""

    assert _hold([30.7, 46.0, 50.0, 48.0, 44.0, 40.0])["passed"] is False


def test_overshooting_out_the_far_side_and_returning_is_not_holding() -> None:
    assert _hold([30.7, 47.0, 53.0, 56.5, 54.0, 51.2])["passed"] is False


def test_a_door_that_never_reaches_the_window_never_holds() -> None:
    assert _hold([5.0, 6.5, 7.5, 8.2, 8.6, 8.73])["passed"] is False


def test_entry_is_reported_so_a_failure_says_where_it_went_wrong() -> None:
    verdict = _hold([30.7, 46.0, 50.0, 48.0, 44.0, 40.0])

    assert verdict["entered_window"] is True
    assert verdict["stayed_inside_after_entry"] is False
